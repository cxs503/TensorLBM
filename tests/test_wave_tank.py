"""Tests for the phase-field LBM wave-tank benchmark (Stokes-2nd generation).

Three layers:

* analytic theory checks (dispersion, kinematics, second-order consistency)
* gauge / reflection-analysis checks against synthetically generated records
  with *known* incident and reflected amplitudes
* one short end-to-end tank run (guarded by ``WT_SLOW=1`` because a single
  phase-field step is expensive on a shared machine)
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from tensorlbm.wave_tank.gauges import (
    fit_harmonics,
    liquid_fraction,
    phase_speed_from_phase,
    surface_elevation,
    two_probe_split,
)
from tensorlbm.wave_tank.stokes2 import Stokes2Wave, dispersion_solve


# ---------------------------------------------------------------------------
# theory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kh", [0.4, 0.7, 1.5, 3.0, 6.0])
def test_dispersion_inverts_closed_form(kh: float) -> None:
    """dispersion_solve must invert omega^2 = g k tanh(kh) to machine precision."""
    h, g = 12.0, 6.5e-3
    k_true = kh / h
    omega = math.sqrt(g * k_true * math.tanh(kh))
    k = dispersion_solve(omega, g, h)
    assert abs(k - k_true) / k_true < 1e-12


def test_wavenumber_monotone_in_frequency() -> None:
    """The dispersion relation is monotone: finer frequency -> shorter wave."""
    h, g = 24.0, 6.5e-3
    ks = [dispersion_solve(w, g, h) for w in (0.005, 0.01, 0.02, 0.04)]
    assert all(b > a for a, b in zip(ks, ks[1:]))


@pytest.mark.parametrize("kh,eps", [(0.7, 0.02), (1.5, 0.08), (3.0, 0.08)])
def test_stokes2_free_surface_harmonics(kh: float, eps: float) -> None:
    """eta(x=0, t) carries exactly the requested first/second harmonics."""
    w = Stokes2Wave.from_kh(kh, eps, h=12.0 * kh, g=6.5e-3)
    t = np.linspace(0.0, w.period, 4096, endpoint=False)
    eta = np.asarray(w.eta(0.0, t))
    fit = fit_harmonics(t, eta, w.omega, n_harmonics=2)
    assert abs(fit.a1 - w.a) / w.a < 1e-6
    assert abs(fit.a2 - w.eta2) / w.eta2 < 1e-6
    assert abs(fit.mean) < 1e-6 * w.a


def test_second_harmonic_ratio_matches_analytic() -> None:
    """eta2 / a = (ka/4)(3 - tanh^2 kh)/tanh^3 kh  (closed form cross-check)."""
    w = Stokes2Wave.from_kh(1.5, 0.08, h=24.0, g=6.5e-3)
    s = math.tanh(w.kh)
    expected = 0.25 * w.eps * (3.0 - s * s) / s ** 3
    assert abs(w.eta2 / w.a - expected) < 1e-12


@pytest.mark.parametrize("kh", [0.7, 1.5, 3.0])
def test_velocity_bottom_no_penetration(kh: float) -> None:
    """w must vanish at the bed y = -h (first and second order terms)."""
    w = Stokes2Wave.from_kh(kh, 0.08, h=12.0 * kh, g=6.5e-3)
    t = 0.37 * w.period
    for x in (0.0, 0.3 * w.wavelength, 0.7 * w.wavelength):
        _u, wv = w.velocity(x, -w.h, t)
        scale = w.a * w.omega
        assert abs(float(wv)) < 1e-9 * scale


@pytest.mark.parametrize("kh,eps", [(0.7, 0.02), (0.7, 0.08), (1.5, 0.02),
                                     (1.5, 0.08), (3.0, 0.02), (3.0, 0.08)])
def test_kinematic_free_surface_condition(kh: float, eps: float) -> None:
    """w = eta_t + u eta_x at the actual surface must hold to O(a^3).

    The residual is normalised by ``a omega``, so a consistent second-order
    truncation leaves ``O((ka)^2 / sigma^3)``: the ``1/sigma^3`` factor is the
    familiar shallow-water amplification of the third-order Stokes terms (at
    ``kh = 0.7`` it is 4.5x, at ``kh >= 1.5`` it is O(1)).  A first-order error
    instead leaves ``O(ka)``, which this bound rejects by more than an order of
    magnitude for every case tested.
    """
    w = Stokes2Wave.from_kh(kh, eps, h=12.0 * kh, g=6.5e-3)
    tol = 3.5 * w.eps ** 2 / w.sigma ** 3
    for frac in (0.0, 0.2, 0.37, 0.5, 0.75):
        t = frac * w.period
        resid = w.kinematic_residual(x=0.0, t=t)
        assert resid < tol, f"kinematic FSBC residual {resid:.3e} > {tol:.3e} at t/T={frac}"


def test_second_order_potential_vanishes_in_deep_water() -> None:
    """C -> 0 as kh -> inf (classical deep-water second-order result)."""
    shallow = Stokes2Wave.from_kh(0.7, 0.08, h=8.4, g=6.5e-3)
    deep = Stokes2Wave.from_kh(12.0, 0.08, h=144.0, g=6.5e-3)
    assert abs(deep.second_order_potential()) < 1e-2 * abs(shallow.second_order_potential())


def test_eta_mean_setdown_is_negative_and_shallow_limited() -> None:
    """The second-order mean set-down is negative and vanishes in deep water."""
    shallow = Stokes2Wave.from_kh(0.7, 0.08, h=8.4, g=6.5e-3)
    deep = Stokes2Wave.from_kh(8.0, 0.08, h=96.0, g=6.5e-3)
    assert shallow.eta_mean < 0.0
    assert abs(deep.eta_mean) < 1e-3 * abs(shallow.eta_mean)


# ---------------------------------------------------------------------------
# gauges
# ---------------------------------------------------------------------------


def test_liquid_fraction_clips_phase_field() -> None:
    phi = np.array([[-0.1, 0.0, 0.25, 1.0, 1.2]])
    np.testing.assert_allclose(liquid_fraction(phi), [[0.0, 0.0, 0.25, 1.0, 1.0]])


@pytest.mark.parametrize("y0", [20.0, 20.3, 20.5, 20.77])
def test_surface_elevation_is_subcell_accurate(y0: float) -> None:
    """The column integral of the tanh interface recovers its centre."""
    ny, nx, width = 40, 6, 2.0
    y = np.arange(ny).reshape(-1, 1)
    phi = 0.5 + 0.5 * np.tanh(2.0 * (y0 - y) / width)
    phi = np.repeat(phi, nx, axis=1)
    # a cell row spans [j-1/2, j+1/2] and the bottom wall face is at y=1/2,
    # so the column integral equals y0 - 1/2; that constant offset is removed
    eta = surface_elevation(phi, y_start=1, y_stop=ny - 1, y_ref=y0 - 0.5)
    assert np.max(np.abs(eta)) < 0.02


def test_surface_elevation_recovers_wave_amplitude() -> None:
    """A sub-cell (0.4 cell) sinusoidal surface is recovered to <2 %."""
    ny, nx, width = 40, 256, 2.0
    amp, lam = 0.4, 64.0
    x = np.arange(nx)
    y = np.arange(ny).reshape(-1, 1)
    surf = 20.0 + amp * np.cos(2.0 * math.pi * x / lam)[None, :]
    phi = 0.5 + 0.5 * np.tanh(2.0 * (surf - y) / width)
    eta = surface_elevation(phi, y_start=1, y_stop=ny - 1, y_ref=20.0)
    fit = fit_harmonics(x / lam, eta, 2.0 * math.pi)  # one period over x/lambda
    assert abs(fit.a1 - amp) / amp < 0.02


def test_fit_harmonics_recovers_known_signal() -> None:
    omega, T = 0.02, 2 * math.pi / 0.02
    t = np.linspace(0.0, 5 * T, 1001)
    t = t[:-1]
    a1, a2, p1, p2, mean = 1.3, 0.21, 0.7, -1.1, 0.05
    eta = mean + a1 * np.cos(omega * t - p1) + a2 * np.cos(2 * omega * t - p2)
    fit = fit_harmonics(t, eta, omega, n_harmonics=2)
    assert abs(fit.a1 - a1) < 1e-8
    assert abs(fit.a2 - a2) < 1e-8
    assert abs(fit.mean - mean) < 1e-8
    assert fit.residual_rms < 1e-9 * a1


def test_two_probe_split_recovers_known_reflection() -> None:
    """Synthetic incident + reflected field with kr = 0.2 is split exactly."""
    k, x1, dx = 2.0 * math.pi / 64.0, 12.0, 64.0 / 6.0
    x2 = x1 + dx
    a_inc, kr_true, ph = 1.0, 0.2, 0.9
    A, B = a_inc, kr_true * a_inc * np.exp(1j * ph)
    # complex amplitudes of the e^{+i omega t} convention: incident e^{-ikx}
    e1 = A * np.exp(-1j * k * x1) + B * np.exp(1j * k * x1)
    e2 = A * np.exp(-1j * k * x2) + B * np.exp(1j * k * x2)
    split = two_probe_split(e1, e2, x1, x2, k)
    assert abs(split.kr - kr_true) < 1e-10
    assert abs(split.a_incident - a_inc) < 1e-10
    assert abs(split.condition - abs(math.sin(k * dx))) < 1e-12


def test_two_probe_split_rejects_degenerate_spacing() -> None:
    k = 2.0 * math.pi / 64.0
    with pytest.raises(ValueError):
        two_probe_split(1.0 + 0j, 1.0 + 0j, 0.0, math.pi / k, k)


def test_phase_speed_from_phase_and_unwrapping() -> None:
    k, omega, dx = 2.0 * math.pi / 100.0, 0.02, 80.0  # k*dx = 5.03 rad > pi
    e1 = np.exp(-1j * k * 0.0)
    e2 = np.exp(-1j * k * dx)
    c, k_meas = phase_speed_from_phase(e1, e2, dx, omega, k_expected=k)
    assert abs(k_meas - k) / k < 1e-12
    assert abs(c - omega / k) / (omega / k) < 1e-12


# ---------------------------------------------------------------------------
# end-to-end (short) tank run
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get("WT_SLOW") != "1", reason="set WT_SLOW=1 to run the LBM tank")
def test_tank_short_run_is_stable_and_generates_a_wave() -> None:
    """A 2-period mini tank stays finite, conserves mass and makes a wave."""
    import torch

    from tensorlbm.wave_tank.tank import WaveTank, WaveTankConfig

    torch.set_num_threads(int(os.environ.get("WT_THREADS", "4")))
    cfg = WaveTankConfig(
        kh=1.5,
        eps=0.08,
        h=8.0,
        g=6.5e-3,
        nu=0.004,
        air_cells=8,
        lam_in=0.7,
        lam_mid=0.6,
        lam_out=0.7,
        ramp_periods=1.0,
        warm_periods=0.5,
        meas_periods=1.5,
        sample_every=5,
        mobility=0.05,
    )
    tank = WaveTank(cfg)
    res = tank.run(quiet=True)
    eta = res.series["probe_eta"]
    assert np.all(np.isfinite(eta))
    assert np.all(np.isfinite(res.series["mass"]))
    # a wave must actually exist at the gauges
    assert np.std(eta[:, 0]) > 0.05 * tank.wave.a
    # mass must not run away on the mini tank (short window, loose bound)
    drift = abs(res.series["mass"][-1] - tank.mass0) / tank.mass0
    assert drift < 0.02, drift