"""Contract tests for the aeroacoustic post-processing functions.

These tests verify operator well-formedness (shape, finiteness, causality,
basic physics identities), NOT acoustic physics accuracy or FWH validation
against experimental data.  They serve as the contract-test evidence recorded
in ``acoustics_capability_contract``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.acoustics import (
    AcousticObserver,
    FWHResult,
    FWHSurface,
    compute_fwh_far_field,
    compute_fwh_result,
    compute_spl_spectrum,
    extract_surface_pressure,
    oaspl,
)

_REF_PRESSURE = 20.0e-6  # 20 µPa


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_surface(
    n_src: int = 4,
    T: int = 64,
    dt: float = 1e-4,
    c0: float = 343.0,
    pressure_amp: float = 1.0,
) -> FWHSurface:
    """Build a small FWHSurface with a sinusoidal pressure fluctuation."""
    positions = torch.zeros(n_src, 3)
    positions[:, 0] = torch.linspace(0.0, 0.1, n_src)
    normals = torch.zeros(n_src, 3)
    normals[:, 0] = 1.0
    areas = torch.full((n_src,), 0.01)
    t = torch.arange(T, dtype=torch.float32) * dt
    pressure = pressure_amp * torch.sin(2.0 * math.pi * 500.0 * t).unsqueeze(0).expand(n_src, T)
    return FWHSurface(positions=positions, normals=normals, areas=areas,
                      pressure=pressure.clone(), dt=dt, c0=c0)


# ---------------------------------------------------------------------------
# compute_fwh_far_field: shape, finite, causality
# ---------------------------------------------------------------------------

class TestFWHFarField:
    def test_output_shape(self) -> None:
        surface = _make_simple_surface(n_src=4, T=64)
        observers = [AcousticObserver(x=1.0, y=0.0, z=0.0)]
        p_prime, time_list = compute_fwh_far_field(surface, observers)
        assert p_prime.shape == (1, 64)
        assert len(time_list) == 64

    def test_output_finite(self) -> None:
        surface = _make_simple_surface(n_src=4, T=64)
        observers = [AcousticObserver(x=1.0, y=0.0, z=0.0)]
        p_prime, _ = compute_fwh_far_field(surface, observers)
        assert torch.isfinite(p_prime).all()

    def test_zero_pressure_gives_zero_output(self) -> None:
        """A zero-pressure surface produces zero far-field pressure."""
        surface = _make_simple_surface(pressure_amp=0.0)
        observers = [AcousticObserver(x=1.0, y=0.0, z=0.0)]
        p_prime, _ = compute_fwh_far_field(surface, observers)
        assert torch.allclose(p_prime, torch.zeros_like(p_prime), atol=1e-30)

    def test_causality_no_signal_before_propagation_delay(self) -> None:
        """Far-field pressure is zero before the retarded-time arrival."""
        dt = 1e-4
        c0 = 343.0
        T = 128
        surface = _make_simple_surface(n_src=1, T=T, dt=dt, c0=c0)
        # Observer at r=1.0 m → delay ≈ 1.0/(343*1e-4) ≈ 29 samples
        observers = [AcousticObserver(x=1.0, y=0.0, z=0.0)]
        p_prime, _ = compute_fwh_far_field(surface, observers)
        delay_samples = int(round(1.0 / (c0 * dt)))
        # Allow a small margin for the near-field term (1/r²) which also
        # respects retarded time.
        assert delay_samples > 1
        assert torch.allclose(
            p_prime[0, :delay_samples], torch.zeros(delay_samples), atol=1e-30
        ), f"Non-zero signal before propagation delay at sample {delay_samples}"

    def test_multiple_observers(self) -> None:
        surface = _make_simple_surface(n_src=4, T=64)
        observers = [
            AcousticObserver(x=1.0, y=0.0, z=0.0, label="A"),
            AcousticObserver(x=0.0, y=1.0, z=0.0, label="B"),
            AcousticObserver(x=0.0, y=0.0, z=1.0, label="C"),
        ]
        p_prime, _ = compute_fwh_far_field(surface, observers)
        assert p_prime.shape == (3, 64)

    def test_invalid_pressure_shape_raises(self) -> None:
        surface = _make_simple_surface(n_src=4, T=64)
        # Corrupt the pressure to have wrong first dimension
        bad_surface = FWHSurface(
            positions=surface.positions,
            normals=surface.normals,
            areas=surface.areas,
            pressure=torch.zeros(3, 64),  # mismatched N
            dt=surface.dt,
            c0=surface.c0,
        )
        observers = [AcousticObserver(x=1.0, y=0.0)]
        with pytest.raises(ValueError, match="pressure must have shape"):
            compute_fwh_far_field(bad_surface, observers)

    def test_too_few_samples_raises(self) -> None:
        surface = _make_simple_surface(n_src=4, T=1)
        observers = [AcousticObserver(x=1.0, y=0.0)]
        with pytest.raises(ValueError, match="at least two pressure samples"):
            compute_fwh_far_field(surface, observers)


# ---------------------------------------------------------------------------
# compute_spl_spectrum: shape, finite, Parseval, DC handling
# ---------------------------------------------------------------------------

class TestSPLSpectrum:
    def test_output_shape(self) -> None:
        T = 128
        dt = 1e-4
        p_prime = torch.randn(2, T)
        spl, freqs = compute_spl_spectrum(p_prime, dt)
        n_freq = T // 2 + 1
        assert spl.shape == (2, n_freq)
        assert len(freqs) == n_freq

    def test_output_finite(self) -> None:
        p_prime = torch.randn(2, 128)
        spl, _ = compute_spl_spectrum(p_prime, 1e-4)
        assert torch.isfinite(spl).all()

    def test_frequency_axis_correct(self) -> None:
        T = 128
        dt = 1e-4
        _, freqs = compute_spl_spectrum(torch.randn(1, T), dt)
        assert freqs[0] == pytest.approx(0.0)
        assert freqs[-1] == pytest.approx(1.0 / (2 * dt))

    def test_zero_pressure_gives_eps_floor_spl(self) -> None:
        """Zero pressure → SPL at eps-clamp floor (finite, not -inf)."""
        p_prime = torch.zeros(1, 128)
        spl, _ = compute_spl_spectrum(p_prime, 1e-4)
        # eps clamp prevents -inf; floor is 10*log10(eps / p_ref²) ≈ 24.7 dB
        assert torch.isfinite(spl).all()
        eps = torch.finfo(torch.float32).eps
        expected_floor = 10.0 * math.log10(eps / (_REF_PRESSURE ** 2))
        assert spl[0, 0].item() == pytest.approx(expected_floor, rel=1e-3)

    def test_nfft_padding(self) -> None:
        """n_fft larger than signal pads with zeros."""
        p_prime = torch.randn(1, 64)
        spl, freqs = compute_spl_spectrum(p_prime, 1e-4, n_fft=256)
        assert spl.shape == (1, 129)
        assert len(freqs) == 129


# ---------------------------------------------------------------------------
# extract_surface_pressure: shape, mean removal, 2D/3D
# ---------------------------------------------------------------------------

class TestExtractSurfacePressure:
    def test_2d_output_shape(self) -> None:
        T, ny, nx = 32, 10, 10
        rho_history = torch.ones(T, ny, nx) * 1.0
        # 4 surface points
        surface_indices = torch.tensor([[2, 3], [2, 4], [3, 3], [3, 4]],
                                       dtype=torch.long)
        surface_normals = torch.zeros(4, 3)
        surface_normals[:, 0] = 1.0
        surface_areas = torch.full((4,), 0.01)
        fwh_surface = extract_surface_pressure(
            rho_history, surface_indices, surface_normals, surface_areas,
            dt=1e-4, c0=343.0, physical_dx=0.01,
        )
        assert fwh_surface.pressure.shape == (4, T)
        assert fwh_surface.positions.shape == (4, 3)

    def test_mean_removal(self) -> None:
        """Extracted pressure fluctuations have zero time-mean per source."""
        T, ny, nx = 64, 8, 8
        rho_history = torch.ones(T, ny, nx) * 1.5  # constant density
        surface_indices = torch.tensor([[2, 3], [4, 5]], dtype=torch.long)
        surface_normals = torch.zeros(2, 3)
        surface_normals[:, 0] = 1.0
        surface_areas = torch.full((2,), 0.01)
        fwh_surface = extract_surface_pressure(
            rho_history, surface_indices, surface_normals, surface_areas,
        )
        mean = fwh_surface.pressure.mean(dim=-1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)

    def test_3d_output_shape(self) -> None:
        T, nz, ny, nx = 16, 6, 6, 6
        rho_history = torch.ones(T, nz, ny, nx)
        surface_indices = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.long)
        surface_normals = torch.zeros(2, 3)
        surface_normals[:, 0] = 1.0
        surface_areas = torch.full((2,), 0.001)
        fwh_surface = extract_surface_pressure(
            rho_history, surface_indices, surface_normals, surface_areas,
            dt=1e-4, c0=343.0, physical_dx=0.01,
        )
        assert fwh_surface.pressure.shape == (2, T)
        assert fwh_surface.positions.shape == (2, 3)

    def test_pressure_proportional_to_density_fluctuation(self) -> None:
        """p' = (ρ - ρ̄) * c_s² where c_s² = 1/3."""
        T, ny, nx = 32, 5, 5
        rho_history = torch.ones(T, ny, nx) * 2.0
        # Spike at one time step only so the time-mean ≠ the spike value
        rho_history[16, 2, 2] = 3.0
        surface_indices = torch.tensor([[2, 2]], dtype=torch.long)
        surface_normals = torch.zeros(1, 3)
        surface_normals[:, 0] = 1.0
        surface_areas = torch.full((1,), 0.01)
        fwh_surface = extract_surface_pressure(
            rho_history, surface_indices, surface_normals, surface_areas,
        )
        # Mean density at this point = (31*2 + 1*3)/32 = 2.03125
        # Fluctuation at spike step = 3 - 2.03125 = 0.96875
        # p' = 0.96875 / 3 ≈ 0.3229
        expected_spike = (3.0 - (31 * 2.0 + 1 * 3.0) / 32) / 3.0
        assert fwh_surface.pressure[0].max().item() == pytest.approx(expected_spike, rel=1e-4)


# ---------------------------------------------------------------------------
# oaspl: shape, finite, zero-pressure floor
# ---------------------------------------------------------------------------

class TestOASPL:
    def test_output_length_matches_observers(self) -> None:
        p_prime = torch.randn(3, 128)
        result = oaspl(p_prime, 1e-4)
        assert len(result) == 3

    def test_output_finite(self) -> None:
        p_prime = torch.randn(2, 64)
        result = oaspl(p_prime, 1e-4)
        assert all(math.isfinite(v) for v in result)

    def test_zero_pressure_gives_eps_floor_oaspl(self) -> None:
        """Zero pressure → OASPL at eps-clamp floor (finite, not -inf)."""
        p_prime = torch.zeros(1, 64)
        result = oaspl(p_prime, 1e-4)
        eps = torch.finfo(torch.float32).eps
        expected_floor = 20.0 * math.log10(eps / _REF_PRESSURE)
        assert math.isfinite(result[0])
        assert result[0] == pytest.approx(expected_floor, rel=1e-3)

    def test_known_tone(self) -> None:
        """A pure sinusoid with known RMS should give the expected OASPL."""
        dt = 1e-4
        T = 1000
        freq = 500.0
        amp = 1.0  # Pa
        t = torch.arange(T, dtype=torch.float32) * dt
        p_prime = (amp * math.sqrt(2.0) * torch.sin(2 * math.pi * freq * t)).unsqueeze(0)
        result = oaspl(p_prime, dt)
        # p_rms = amp (for a sine with amplitude √2*amp, RMS = amp)
        expected = 20.0 * math.log10(amp / _REF_PRESSURE)
        assert result[0] == pytest.approx(expected, abs=0.5)


# ---------------------------------------------------------------------------
# compute_fwh_result: wrapper integration
# ---------------------------------------------------------------------------

class TestFWHResultWrapper:
    def test_result_has_all_fields(self) -> None:
        surface = _make_simple_surface(n_src=4, T=64)
        observers = [AcousticObserver(x=1.0, y=0.0, z=0.0)]
        result = compute_fwh_result(surface, observers)
        assert isinstance(result, FWHResult)
        assert len(result.time) == 64
        assert result.p_prime.shape == (1, 64)
        assert len(result.frequencies) > 0
        assert result.spl.shape[0] == 1
        assert len(result.oaspl) == 1
        assert result.observers == observers

    def test_result_finite(self) -> None:
        surface = _make_simple_surface(n_src=4, T=128)
        observers = [AcousticObserver(x=1.0, y=0.0, z=0.0)]
        result = compute_fwh_result(surface, observers)
        assert torch.isfinite(result.p_prime).all()
        assert torch.isfinite(result.spl).all()
        assert all(math.isfinite(v) for v in result.oaspl)
