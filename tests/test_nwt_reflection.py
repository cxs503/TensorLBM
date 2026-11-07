"""Reflection separation (Goda & Suzuki two-point method) on synthetic signals.

The estimators are validated against a *constructed* wave field: an incident
train of known amplitude plus a reflected train of known amplitude and phase.
Recovering the input reflection coefficient to machine accuracy (and to <1 %
under a 1 %-of-signal noise floor) is what makes the NWT's C3 measurement
trustworthy — otherwise a "5 % reflection" number could just be estimator error.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.nwt.reflection import complex_amplitude, least_squares_reflection, two_point_reflection

DT = 1.0
OMEGA = 2.0 * math.pi / 400.0
K = 2.0 * math.pi / 60.0


def synthetic(xs, a_i=1.0, a_r=0.2, phi_r=1.1, n=4000, noise=0.0, seed=0):
    t = torch.arange(n, dtype=torch.float64) * DT
    gen = torch.Generator().manual_seed(seed)
    out = []
    for x in xs:
        eta = a_i * torch.cos(OMEGA * t - K * x) + a_r * torch.cos(OMEGA * t + K * x + phi_r)
        if noise:
            eta = eta + noise * torch.randn(n, dtype=torch.float64, generator=gen)
        out.append(eta)
    return out


def test_complex_amplitude_recovers_sinusoid_with_partial_period():
    n = 1357  # deliberately not an integer number of periods
    t = torch.arange(n, dtype=torch.float64) * DT
    eta = 0.7 * torch.cos(OMEGA * t + 0.3) + 0.15
    Z = complex_amplitude(eta, DT, OMEGA)
    assert abs(Z) == pytest.approx(0.7, rel=1e-9)
    assert math.atan2(Z.imag, Z.real) == pytest.approx(0.3, abs=1e-9)


@pytest.mark.parametrize("dx", [0.05 * 60.0, 0.15 * 60.0, 0.25 * 60.0])
def test_two_point_recovers_known_reflection_coefficient(dx):
    xs = [0.0, dx]
    eta = synthetic(xs, a_i=1.0, a_r=0.2, phi_r=1.1)
    r = two_point_reflection(eta[0], eta[1], xs[0], xs[1], DT, OMEGA, K)
    assert r["KR"] == pytest.approx(0.2, rel=1e-8)
    assert r["a_incident"] == pytest.approx(1.0, rel=1e-8)
    assert r["condition"] > 0.25


def test_two_point_is_robust_to_one_percent_noise():
    xs = [0.0, 0.2 * 60.0]
    eta = synthetic(xs, a_i=1.0, a_r=0.2, noise=0.01, seed=7)
    r = two_point_reflection(eta[0], eta[1], xs[0], xs[1], DT, OMEGA, K)
    assert r["KR"] == pytest.approx(0.2, abs=0.02)


def test_pure_incident_wave_gives_zero_reflection():
    xs = [0.0, 12.0]
    eta = synthetic(xs, a_i=1.0, a_r=0.0)
    r = two_point_reflection(eta[0], eta[1], xs[0], xs[1], DT, OMEGA, K)
    assert r["KR"] < 1e-9


def test_least_squares_matches_two_point_and_is_overdetermined():
    xs = [0.0, 15.0, 30.0, 45.0]
    eta = synthetic(xs, a_i=1.0, a_r=0.35, phi_r=-0.4)
    tp = two_point_reflection(eta[0], eta[1], xs[0], xs[1], DT, OMEGA, K)
    ls = least_squares_reflection(eta, xs, DT, OMEGA, K)
    assert ls["KR"] == pytest.approx(0.35, rel=1e-8)
    assert ls["KR"] == pytest.approx(tp["KR"], rel=1e-6)
    assert ls["max_residual"] < 1e-8


def test_degenerate_spacing_is_rejected():
    xs = [0.0, 0.5 * (2 * math.pi / K)]  # dx = lambda/2 -> sin(k dx) = 0
    eta = synthetic(xs)
    with pytest.raises(ValueError):
        two_point_reflection(eta[0], eta[1], xs[0], xs[1], DT, OMEGA, K)


def test_reflection_phase_is_recovered():
    xs = [0.0, 9.0]
    phi_r = 2.0
    eta = synthetic(xs, a_i=1.0, a_r=0.3, phi_r=phi_r)
    r = two_point_reflection(eta[0], eta[1], xs[0], xs[1], DT, OMEGA, K)
    # reflected train phase at x = 0 is phi_r
    assert r["phase_reflected"] == pytest.approx(phi_r, abs=1e-9)