"""Tests for tensorlbm.acoustics – FWH aeroacoustic analogy."""
from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_surface(N: int = 8, T: int = 100, freq_hz: float = 500.0, dt: float = 1e-4):
    """Create a synthetic FWHSurface with a sinusoidal pressure signal."""
    from tensorlbm.acoustics import FWHSurface

    t = torch.linspace(0, T * dt, T)
    pressure = torch.zeros(N, T)
    for n in range(N):
        phase = n * 2 * math.pi / N
        pressure[n] = 1e-3 * torch.sin(2 * math.pi * freq_hz * t + phase)

    positions = torch.zeros(N, 3)
    positions[:, 0] = torch.linspace(0, 0.1, N)  # x positions

    normals = torch.zeros(N, 3)
    normals[:, 0] = 1.0  # outward in +x

    areas = torch.full((N,), 0.01)

    return FWHSurface(
        positions=positions,
        normals=normals,
        areas=areas,
        pressure=pressure,
        dt=dt,
        c0=343.0,
    )


# ---------------------------------------------------------------------------
# AcousticObserver
# ---------------------------------------------------------------------------

def test_acoustic_observer_defaults():
    from tensorlbm.acoustics import AcousticObserver

    obs = AcousticObserver(x=10.0, y=0.0)
    assert obs.z == 0.0
    assert obs.label == ""


def test_acoustic_observer_label():
    from tensorlbm.acoustics import AcousticObserver

    obs = AcousticObserver(x=1.0, y=2.0, z=3.0, label="test")
    assert obs.label == "test"


# ---------------------------------------------------------------------------
# FWHSurface
# ---------------------------------------------------------------------------

def test_fwh_surface_creation():
    surf = _make_surface(N=4, T=50)
    assert surf.pressure.shape == (4, 50)
    assert surf.positions.shape == (4, 3)
    assert surf.normals.shape == (4, 3)
    assert surf.areas.shape == (4,)


# ---------------------------------------------------------------------------
# compute_fwh_far_field
# ---------------------------------------------------------------------------

def test_compute_fwh_far_field_shape():
    from tensorlbm.acoustics import AcousticObserver, compute_fwh_far_field

    surf = _make_surface(N=4, T=50)
    observers = [
        AcousticObserver(x=10.0, y=0.0),
        AcousticObserver(x=0.0, y=10.0),
    ]
    p_prime, time_list = compute_fwh_far_field(surf, observers)
    assert p_prime.shape == (2, 50)
    assert len(time_list) == 50


def test_compute_fwh_far_field_time_axis():
    from tensorlbm.acoustics import AcousticObserver, compute_fwh_far_field

    dt = 1e-4
    T = 80
    surf = _make_surface(N=4, T=T, dt=dt)
    observers = [AcousticObserver(x=10.0, y=0.0)]
    _, time_list = compute_fwh_far_field(surf, observers)
    assert len(time_list) == T
    assert abs(time_list[1] - dt) < 1e-10


def test_compute_fwh_far_field_nonzero():
    """Far-field pressure should be non-trivially non-zero for a dipole source."""
    from tensorlbm.acoustics import AcousticObserver, compute_fwh_far_field

    surf = _make_surface(N=8, T=100)
    # Keep propagation delay inside the recorded window.  A 10 m observer at
    # dt=1e-4 s lies 291 samples away and cannot receive a causal signal here.
    observers = [AcousticObserver(x=0.1, y=0.0)]
    p_prime, _ = compute_fwh_far_field(surf, observers)
    # Should have some non-zero values for a sinusoidal source
    assert p_prime.abs().max().item() > 0.0


def test_compute_fwh_far_field_is_causal_for_delayed_source():
    """A nonzero initial source sample must not appear before propagation."""
    from tensorlbm.acoustics import AcousticObserver, FWHSurface, compute_fwh_far_field

    # Observer is ten samples away.  The old negative-index clamp incorrectly
    # filled samples 0..9 with p(t=0) for a nonzero source.
    pressure = torch.ones(1, 20)
    surface = FWHSurface(
        positions=torch.tensor([[0.0, 0.0, 0.0]]),
        normals=torch.tensor([[1.0, 0.0, 0.0]]),
        areas=torch.tensor([1.0]), pressure=pressure, dt=1.0, c0=1.0,
    )
    p_prime, _ = compute_fwh_far_field(surface, [AcousticObserver(x=10.0, y=0.0)])
    assert torch.count_nonzero(p_prime[:, :10]).item() == 0
    assert p_prime[0, 10].abs().item() > 0.0


def test_compute_fwh_far_field_preserves_input_dtype():
    from tensorlbm.acoustics import AcousticObserver, compute_fwh_far_field

    surface = _make_surface(N=4, T=20)
    surface.pressure = surface.pressure.double()
    p_prime, _ = compute_fwh_far_field(surface, [AcousticObserver(x=10.0, y=0.0)])
    assert p_prime.dtype == torch.float64


# ---------------------------------------------------------------------------
# compute_spl_spectrum
# ---------------------------------------------------------------------------

def test_compute_spl_spectrum_shape():
    from tensorlbm.acoustics import compute_spl_spectrum

    T = 128
    dt = 1e-4
    p_prime = torch.randn(2, T) * 1e-3
    spl, freqs = compute_spl_spectrum(p_prime, dt)
    assert spl.shape[0] == 2
    assert len(freqs) == spl.shape[1]


def test_compute_spl_spectrum_frequency_axis():
    from tensorlbm.acoustics import compute_spl_spectrum

    T = 128
    dt = 1e-4
    fs = 1.0 / dt
    p_prime = torch.zeros(1, T)
    p_prime[0, :] = torch.sin(2 * math.pi * 1000 * torch.linspace(0, T * dt, T))
    spl, freqs = compute_spl_spectrum(p_prime, dt)
    # Nyquist should be fs/2
    assert abs(freqs[-1] - fs / 2) < 1.0


def test_spl_values_finite():
    from tensorlbm.acoustics import compute_spl_spectrum

    T = 64
    dt = 1e-4
    p_prime = torch.randn(3, T) * 1e-4
    spl, freqs = compute_spl_spectrum(p_prime, dt)
    assert torch.isfinite(spl).all()


# ---------------------------------------------------------------------------
# OASPL
# ---------------------------------------------------------------------------

def test_oaspl_returns_list():
    from tensorlbm.acoustics import oaspl

    p_prime = torch.randn(2, 100) * 1e-3
    result = oaspl(p_prime, dt=1e-4)
    assert len(result) == 2


def test_oaspl_values_are_floats():
    from tensorlbm.acoustics import oaspl

    p_prime = torch.randn(1, 200) * 2e-3
    result = oaspl(p_prime, dt=1e-4)
    assert isinstance(result[0], float)


def test_oaspl_louder_signal_higher_spl():
    from tensorlbm.acoustics import oaspl

    T = 200
    quiet = torch.randn(1, T) * 1e-4
    loud = torch.randn(1, T) * 1e-2
    spl_quiet = oaspl(quiet, dt=1e-4)[0]
    spl_loud = oaspl(loud, dt=1e-4)[0]
    assert spl_loud > spl_quiet


# ---------------------------------------------------------------------------
# extract_surface_pressure
# ---------------------------------------------------------------------------

def test_extract_surface_pressure_2d():
    from tensorlbm.acoustics import extract_surface_pressure

    T, ny, nx = 20, 16, 16
    rho_history = torch.ones(T, ny, nx)
    # Add a simple fluctuation to one row
    rho_history[:, 5, :] = 1.0 + 0.01 * torch.sin(
        torch.linspace(0, 2 * math.pi, T)
    ).unsqueeze(-1)

    surface_indices = torch.tensor([[5, 4], [5, 8], [5, 12]], dtype=torch.long)
    normals = torch.zeros(3, 3)
    normals[:, 1] = 1.0  # +y normal
    areas = torch.ones(3) * 1.0
    surf = extract_surface_pressure(
        rho_history, surface_indices, normals, areas, dt=1e-4
    )
    assert surf.pressure.shape == (3, T)
    # Mean of pressure fluctuations should be ~0
    assert surf.pressure.mean().abs().item() < 1e-6


# ---------------------------------------------------------------------------
# compute_fwh_result convenience wrapper
# ---------------------------------------------------------------------------

def test_compute_fwh_result_fields():
    from tensorlbm.acoustics import AcousticObserver, FWHResult, compute_fwh_result

    surf = _make_surface(N=4, T=64)
    observers = [AcousticObserver(x=10.0, y=0.0)]
    result = compute_fwh_result(surf, observers)

    assert isinstance(result, FWHResult)
    assert len(result.time) == 64
    assert result.p_prime.shape[0] == 1
    assert len(result.oaspl) == 1
    assert len(result.frequencies) > 0
    assert result.spl.shape[0] == 1
