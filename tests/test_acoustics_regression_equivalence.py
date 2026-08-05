"""Regression equivalence verification for the acoustics module.

This test file verifies three things:

1. **Bug identification** — the original ``compute_spl_spectrum`` has a Hann
   window length bug: when ``n_fft > T`` the window is created with length
   ``min(n_fft, T) = T`` *before* zero-padding the signal to ``n_fft``, causing
   a shape mismatch (``RuntimeError``) at the element-wise multiply step.

2. **Equivalence in the normal case** — when ``n_fft <= T`` (no padding
   needed), the fixed and original ``compute_spl_spectrum`` produce
   bit-identical SPL spectra, because the window length is ``n_fft`` in both
   paths.

3. **Function-level equivalence** — ``compute_fwh_far_field`` and
   ``extract_surface_pressure`` are byte-for-byte identical between the
   original and modified modules (the extraction commit only touched
   ``compute_spl_spectrum``), so their outputs are trivially equivalent.

The original (pre-fix) module is loaded from ``tests/fixtures/orig_acoustics.py``
via ``importlib`` so that both versions coexist in the same process.
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

import pytest
import torch

# ---------------------------------------------------------------------------
# Load the original (pre-fix) acoustics module from the fixture file
# ---------------------------------------------------------------------------

_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "orig_acoustics.py")


def _load_orig_module():
    """Load the original acoustics.py as an isolated module object."""
    spec = importlib.util.spec_from_file_location("_orig_acoustics", _FIXTURE_PATH)
    assert spec is not None, f"Could not create spec for {_FIXTURE_PATH}"
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so that @dataclass can resolve cls.__module__.
    sys.modules["_orig_acoustics"] = mod
    # The original module only imports `math`, `dataclasses`, and `torch` —
    # no internal tensorlbm imports — so it can be executed standalone.
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_orig = _load_orig_module()

# Import the current (fixed) module normally
from tensorlbm import acoustics as _fixed  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_surface(n_src=4, T=64, dt=1e-4, c0=343.0, pressure_amp=1.0):
    """Build a small FWHSurface with a sinusoidal pressure fluctuation."""
    positions = torch.zeros(n_src, 3)
    positions[:, 0] = torch.linspace(0.0, 0.1, n_src)
    normals = torch.zeros(n_src, 3)
    normals[:, 0] = 1.0
    areas = torch.full((n_src,), 0.01)
    t = torch.arange(T, dtype=torch.float32) * dt
    pressure = pressure_amp * torch.sin(2.0 * math.pi * 500.0 * t)
    pressure = pressure.unsqueeze(0).expand(n_src, T).clone()
    return _fixed.FWHSurface(
        positions=positions, normals=normals, areas=areas,
        pressure=pressure, dt=dt, c0=c0,
    )


# ===========================================================================
# 1. Bug identification: Hann window length mismatch when n_fft > T
# ===========================================================================

class TestSPLWindowBugIdentification:
    """Identify that the original compute_spl_spectrum crashes when n_fft > T."""

    def test_original_crashes_when_n_fft_exceeds_T(self):
        """The original code creates a Hann window of length min(n_fft, T)=T
        *before* zero-padding the signal to n_fft, so the element-wise
        multiply raises a shape-mismatch RuntimeError."""
        T = 64
        p_prime = torch.randn(2, T) * 1e-3
        # n_fft > T triggers the bug: window has length T but sig is padded to n_fft
        with pytest.raises(RuntimeError, match="size of tensor"):
            _orig.compute_spl_spectrum(p_prime, dt=1e-4, n_fft=256)

    def test_fixed_handles_n_fft_exceeds_T(self):
        """The fixed code creates the window *after* padding, with length
        n_fft, so no shape mismatch occurs."""
        T = 64
        p_prime = torch.randn(2, T) * 1e-3
        spl, freqs = _fixed.compute_spl_spectrum(p_prime, dt=1e-4, n_fft=256)
        assert spl.shape == (2, 129)
        assert len(freqs) == 129
        assert torch.isfinite(spl).all()

    def test_original_window_length_is_min_n_fft_T(self):
        """Directly verify the root cause: the original window has length
        min(n_fft, T), not n_fft, when n_fft > T."""
        n_fft, T = 256, 64
        # Reproduce the original window-creation line
        orig_window = torch.hann_window(min(n_fft, T), periodic=False)
        assert orig_window.shape[0] == T  # too short!
        # The fixed version uses n_fft
        fixed_window = torch.hann_window(n_fft, periodic=False)
        assert fixed_window.shape[0] == n_fft


# ===========================================================================
# 2. Equivalence: compute_spl_spectrum in the normal case (n_fft <= T)
# ===========================================================================

class TestSPLSpectrumEquivalence:
    """Verify that the fixed and original SPL spectra are identical when
    n_fft <= T (the normal, non-padding case)."""

    def test_default_n_fft_equal_T(self):
        """n_fft defaults to T — no padding, both paths identical."""
        torch.manual_seed(42)
        T = 128
        p_prime = torch.randn(3, T) * 1e-3
        spl_orig, freqs_orig = _orig.compute_spl_spectrum(p_prime, dt=1e-4)
        spl_fixed, freqs_fixed = _fixed.compute_spl_spectrum(p_prime, dt=1e-4)
        assert torch.allclose(spl_orig, spl_fixed, atol=0.0, rtol=0.0)
        assert freqs_orig == freqs_fixed

    def test_n_fft_less_than_T(self):
        """n_fft < T — truncation only, no padding, both paths identical."""
        torch.manual_seed(99)
        T = 128
        p_prime = torch.randn(2, T) * 1e-3
        spl_orig, freqs_orig = _orig.compute_spl_spectrum(p_prime, dt=1e-4, n_fft=64)
        spl_fixed, freqs_fixed = _fixed.compute_spl_spectrum(p_prime, dt=1e-4, n_fft=64)
        assert torch.allclose(spl_orig, spl_fixed, atol=0.0, rtol=0.0)
        assert freqs_orig == freqs_fixed

    def test_n_fft_equals_T_explicit(self):
        """Explicitly pass n_fft == T — same as default, both identical."""
        torch.manual_seed(7)
        T = 256
        p_prime = torch.randn(1, T) * 1e-4
        spl_orig, _ = _orig.compute_spl_spectrum(p_prime, dt=1e-3, n_fft=T)
        spl_fixed, _ = _fixed.compute_spl_spectrum(p_prime, dt=1e-3, n_fft=T)
        assert torch.allclose(spl_orig, spl_fixed, atol=0.0, rtol=0.0)

    def test_equivalence_with_pure_tone(self):
        """Equivalence holds for a structured (sinusoidal) input too."""
        T = 512
        dt = 1e-4
        t = torch.arange(T, dtype=torch.float32) * dt
        p_prime = (1e-3 * torch.sin(2 * math.pi * 1000.0 * t)).unsqueeze(0)
        spl_orig, freqs_orig = _orig.compute_spl_spectrum(p_prime, dt)
        spl_fixed, freqs_fixed = _fixed.compute_spl_spectrum(p_prime, dt)
        assert torch.allclose(spl_orig, spl_fixed, atol=0.0, rtol=0.0)
        assert freqs_orig == freqs_fixed

    def test_equivalence_with_zero_input(self):
        """Equivalence holds for zero input (eps-clamp floor)."""
        p_prime = torch.zeros(2, 128)
        spl_orig, _ = _orig.compute_spl_spectrum(p_prime, dt=1e-4)
        spl_fixed, _ = _fixed.compute_spl_spectrum(p_prime, dt=1e-4)
        assert torch.allclose(spl_orig, spl_fixed, atol=0.0, rtol=0.0)


# ===========================================================================
# 3. Function-level equivalence: compute_fwh_far_field
# ===========================================================================

class TestFWHFarFieldEquivalence:
    """compute_fwh_far_field is byte-for-byte identical between original and
    fixed modules (the extraction commit only touched compute_spl_spectrum).
    Verify output equivalence directly."""

    def test_single_observer_equivalence(self):
        surface = _make_surface(n_src=4, T=64)
        observers = [_fixed.AcousticObserver(x=0.1, y=0.0, z=0.0)]
        p_orig, t_orig = _orig.compute_fwh_far_field(surface, observers)
        p_fixed, t_fixed = _fixed.compute_fwh_far_field(surface, observers)
        assert torch.allclose(p_orig, p_fixed, atol=0.0, rtol=0.0)
        assert t_orig == t_fixed

    def test_multiple_observers_equivalence(self):
        surface = _make_surface(n_src=8, T=100)
        observers = [
            _fixed.AcousticObserver(x=0.1, y=0.0, z=0.0, label="A"),
            _fixed.AcousticObserver(x=0.0, y=0.1, z=0.0, label="B"),
        ]
        p_orig, _ = _orig.compute_fwh_far_field(surface, observers)
        p_fixed, _ = _fixed.compute_fwh_far_field(surface, observers)
        assert torch.allclose(p_orig, p_fixed, atol=0.0, rtol=0.0)

    def test_causality_preserved(self):
        """Both versions enforce the same causality (no signal before
        propagation delay)."""
        pressure = torch.ones(1, 20)
        surface = _fixed.FWHSurface(
            positions=torch.tensor([[0.0, 0.0, 0.0]]),
            normals=torch.tensor([[1.0, 0.0, 0.0]]),
            areas=torch.tensor([1.0]),
            pressure=pressure, dt=1.0, c0=1.0,
        )
        observers = [_fixed.AcousticObserver(x=10.0, y=0.0)]
        p_orig, _ = _orig.compute_fwh_far_field(surface, observers)
        p_fixed, _ = _fixed.compute_fwh_far_field(surface, observers)
        assert torch.allclose(p_orig, p_fixed, atol=0.0, rtol=0.0)
        # Both have zero signal before delay
        assert torch.count_nonzero(p_fixed[:, :10]).item() == 0


# ===========================================================================
# 4. Function-level equivalence: extract_surface_pressure
# ===========================================================================

class TestExtractSurfacePressureEquivalence:
    """extract_surface_pressure is byte-for-byte identical between original
    and fixed modules. Verify output equivalence directly."""

    def test_2d_equivalence(self):
        T, ny, nx = 32, 10, 10
        torch.manual_seed(123)
        rho_history = torch.ones(T, ny, nx) + 0.01 * torch.randn(T, ny, nx)
        surface_indices = torch.tensor([[2, 3], [2, 4], [3, 3]], dtype=torch.long)
        normals = torch.zeros(3, 3)
        normals[:, 0] = 1.0
        areas = torch.full((3,), 0.01)
        surf_orig = _orig.extract_surface_pressure(
            rho_history, surface_indices, normals, areas,
            dt=1e-4, c0=343.0, physical_dx=0.01,
        )
        surf_fixed = _fixed.extract_surface_pressure(
            rho_history, surface_indices, normals, areas,
            dt=1e-4, c0=343.0, physical_dx=0.01,
        )
        assert torch.allclose(surf_orig.pressure, surf_fixed.pressure, atol=0.0, rtol=0.0)
        assert torch.allclose(surf_orig.positions, surf_fixed.positions, atol=0.0, rtol=0.0)

    def test_3d_equivalence(self):
        T, nz, ny, nx = 16, 6, 6, 6
        torch.manual_seed(456)
        rho_history = torch.ones(T, nz, ny, nx) + 0.01 * torch.randn(T, nz, ny, nx)
        surface_indices = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.long)
        normals = torch.zeros(2, 3)
        normals[:, 0] = 1.0
        areas = torch.full((2,), 0.001)
        surf_orig = _orig.extract_surface_pressure(
            rho_history, surface_indices, normals, areas,
        )
        surf_fixed = _fixed.extract_surface_pressure(
            rho_history, surface_indices, normals, areas,
        )
        assert torch.allclose(surf_orig.pressure, surf_fixed.pressure, atol=0.0, rtol=0.0)
        assert torch.allclose(surf_orig.positions, surf_fixed.positions, atol=0.0, rtol=0.0)

    def test_mean_removal_equivalence(self):
        """Both versions remove the time-mean identically."""
        T, ny, nx = 64, 8, 8
        rho_history = torch.ones(T, ny, nx) * 1.5
        surface_indices = torch.tensor([[2, 3], [4, 5]], dtype=torch.long)
        normals = torch.zeros(2, 3)
        normals[:, 0] = 1.0
        areas = torch.full((2,), 0.01)
        surf_orig = _orig.extract_surface_pressure(
            rho_history, surface_indices, normals, areas,
        )
        surf_fixed = _fixed.extract_surface_pressure(
            rho_history, surface_indices, normals, areas,
        )
        assert torch.allclose(surf_orig.pressure, surf_fixed.pressure, atol=0.0, rtol=0.0)
        mean_orig = surf_orig.pressure.mean(dim=-1)
        assert torch.allclose(mean_orig, torch.zeros_like(mean_orig), atol=1e-6)
