"""Verification tests for the Goda–Suzuki reflection analysis.

The separation method is tested against synthetic two-wave fields with known
incident/reflected complex amplitudes, so every assertion has an exact
reference.  The tests also pin down the two classical failure modes that a
wave-tank campaign must respect:

* the ``k ds = n pi`` probe-spacing degeneracy (ill conditioning), and
* sensitivity to measurement noise (error amplification ``1/|sin k ds|``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.wave_tank2d.goda import (
    apparent_reflection,
    conditioning,
    fit_harmonics,
    least_squares_reflection,
    phase_speed_from_phasors,
    phasor_from_series,
    recommended_spacing,
    two_point_reflection,
)


def _synthetic(k=0.075, K_R=0.2, phase=0.7, inc_amp=0.27, s=(12.0, 30.0), n=8000,
               tmax=None, noise=0.0, seed=0):
    """Two-wave record ``eta = Re[I e^{i(wt+ks)} + R e^{i(wt-ks)}]`` at two probes."""
    omega = 0.04 * k
    tmax = tmax if tmax is not None else 2 * math.pi / omega
    t = np.linspace(0.0, tmax, n)
    i_amp = inc_amp
    R = K_R * inc_amp * np.exp(1j * phase)
    rng = np.random.default_rng(seed)
    z = []
    for sj in s:
        eta = np.real(i_amp * np.exp(1j * (omega * t + k * sj))
                      + R * np.exp(1j * (omega * t - k * sj)))
        if noise:
            eta = eta + noise * inc_amp * rng.standard_normal(t.size)
        z.append(phasor_from_series(t, eta, omega))
    return z, t, omega, i_amp, R


def test_phasor_recovers_known_sinusoid() -> None:
    t = np.linspace(0.0, 10.0, 500)
    omega = 2.3
    z_true = 0.31 * np.exp(1j * 0.42)
    eta = np.real(z_true * np.exp(1j * omega * t)) + 0.05
    z = phasor_from_series(t, eta, omega)
    assert abs(z - z_true) < 1e-12


def test_two_point_recovers_exact_reflection() -> None:
    for K_R in (0.0, 0.05, 0.2, 0.6, 1.0):
        z, _, _, i_amp, R = _synthetic(K_R=K_R, phase=0.7)
        res = two_point_reflection(z[0], z[1], 0.075, 12.0, 30.0)
        assert res["K_R"] == pytest.approx(K_R, abs=1e-9)
        assert abs(res["I"] - i_amp) < 1e-9
        assert abs(res["R"] - R) < 1e-9


def test_two_point_recovers_reflection_phase() -> None:
    for phase in (0.0, 0.7, -2.1, 3.0):
        z, _, _, _, R = _synthetic(K_R=0.35, phase=phase)
        res = two_point_reflection(z[0], z[1], 0.075, 12.0, 30.0)
        assert res["phase"] == pytest.approx(phase, abs=1e-9)


def test_degenerate_spacing_is_rejected() -> None:
    k, lam = 0.075, 2 * math.pi / 0.075
    z, _, _, _, _ = _synthetic(k=k, s=(10.0, 10.0 + lam))
    with pytest.raises(ValueError, match="degenerate"):
        two_point_reflection(z[0], z[1], k, 10.0, 10.0 + lam)
    assert math.isinf(conditioning(k, lam))


def test_conditioning_grows_near_half_wavelength() -> None:
    k = 0.075
    lam = 2 * math.pi / k
    good = conditioning(k, 0.15 * lam)
    worse = conditioning(k, 0.45 * lam)
    assert 1.0 < good < 2.0   # optimum spacing is ds = lambda/4 (cond = 1)
    assert worse > good
    # recommended spacings all stay in the Goda-Suzuki safe band
    for info in recommended_spacing(k).values():
        assert 0.05 <= info["ds_over_lambda"] <= 0.4501
        assert math.isfinite(info["conditioning"])


def test_noise_amplification_matches_conditioning() -> None:
    """1 % measurement noise must not blow K_R up by more than ~cond*1 %."""
    k = 0.075
    z, _, _, _, _ = _synthetic(k=k, K_R=0.2, s=(20.0, 20.0 + 0.15 * 2 * math.pi / k),
                               noise=0.01)
    res = two_point_reflection(z[0], z[1], k, 20.0, 20.0 + 0.15 * 2 * math.pi / k)
    assert abs(res["K_R"] - 0.2) < 0.05
    assert abs(res["K_R"] - 0.2) < 0.01 * res["conditioning"]


def test_multi_probe_least_squares_beats_degenerate_pair() -> None:
    """A 4-probe fit stays well conditioned where a two-probe pair degenerates."""
    k = 0.075
    lam = 2 * math.pi / k
    s = np.array([10.0, 10.0 + 0.25 * lam, 10.0 + 0.5 * lam, 10.0 + 0.75 * lam])
    z, _, _, i_amp, _r_amp = _synthetic(k=k, K_R=0.3, phase=-1.2, s=tuple(s), noise=0.002)
    fit = least_squares_reflection(z, s, k)
    assert fit["K_R"] == pytest.approx(0.3, abs=0.03)
    assert fit["rms_residual"] < 0.01 * abs(i_amp)
    assert fit["n_probes"] == 4


def test_apparent_reflection_is_consistent_with_standing_pattern() -> None:
    """A pure standing wave (K_R = 1) gives an envelope estimate of 1."""
    assert apparent_reflection(1.0, 0.0) == pytest.approx(1.0)
    assert apparent_reflection(0.5, 0.5) == pytest.approx(0.0)


def test_wavenumber_and_phase_speed_from_phasor_row() -> None:
    k, omega = 0.081, 0.0031
    s = np.linspace(0.0, 80.0, 21)
    z = 0.2 * np.exp(1j * k * s)
    res = phase_speed_from_phasors(z, s, omega)
    assert res["k"] == pytest.approx(k, rel=1e-9)
    assert res["c"] == pytest.approx(omega / k, rel=1e-9)
    assert res["phase_scatter"] < 1e-9


def test_harmonic_fit_separates_second_order_bound_wave() -> None:
    """A Stokes wave record must be split into a1/a2 without leakage."""
    omega, period = 0.003, 2094.0
    t = np.linspace(0.0, 2 * period, 4000)
    a1, a2 = 0.27, 0.004
    eta = a1 * np.cos(omega * t + 0.3) + a2 * np.cos(2 * omega * t - 1.1) + 0.05
    fit = fit_harmonics(t, eta, omega, n_harm=2)
    assert fit["amplitudes"][0] == pytest.approx(a1, rel=1e-6)
    assert fit["amplitudes"][1] == pytest.approx(a2, rel=1e-5)
    assert fit["mean"] == pytest.approx(0.05, rel=1e-6)
    assert fit["rms_residual"] < 1e-9