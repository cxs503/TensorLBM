"""Verification tests for the analytical Stokes second-order reference.

The point of these tests is to certify the *reference itself* before it is used
as the ground truth of the wave-tank campaign:

* the dispersion solver is the exact inverse of ``w^2 = g k tanh(k h)``;
* the deep- and shallow-water limits are recovered;
* the velocity field is solenoidal and irrotational (potential flow), which is
  a PDE-level check that does not rely on any remembered formula;
* the free-surface kinematic boundary condition residual is ``O(eps^2)`` — i.e.
  the field is a genuine second-order solution whose error vanishes with the
  steepness;
* the second-harmonic amplitude is the standard bound-wave value and its time
  mean surface elevation is zero (no spurious set-up).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.wave_tank2d.stokes2 import (
    Stokes2Wave,
    dispersion_k,
    dispersion_omega,
    kinematic_residual,
    phase_speed,
    second_order_amplitude,
    stokes_validity,
)


def _wave(kh=1.5, eps=0.02, h=20.0, g=1.3e-4):
    k = kh / h
    return Stokes2Wave(a=eps / k, k=k, h=h, g=g)


def test_dispersion_solver_round_trip() -> None:
    """``dispersion_k`` inverts ``dispersion_omega`` to machine precision."""
    for kh in (0.4, 0.7, 1.5, 3.0, 8.0):
        h = 20.0
        k = kh / h
        omega = dispersion_omega(k, h)
        k_back = dispersion_k(omega, h)
        assert abs(k_back - k) / k < 1e-12, (kh, k, k_back)


def test_deep_and_shallow_water_limits() -> None:
    """``kh -> inf`` gives ``c = sqrt(g/k)``; ``kh -> 0`` gives ``c = sqrt(g h)``."""
    g, h = 9.81, 10.0
    k_deep = 40.0 / h  # kh = 40
    c_deep = phase_speed(k_deep, h, g)
    assert abs(c_deep - math.sqrt(g / k_deep)) / math.sqrt(g / k_deep) < 1e-6

    k_shallow = 1e-4 / h  # kh = 1e-4
    c_shallow = phase_speed(k_shallow, h, g)
    assert abs(c_shallow - math.sqrt(g * h)) / math.sqrt(g * h) < 1e-3
    # shallow water is non-dispersive: c independent of k
    assert abs(phase_speed(2e-4 / h, h, g) - c_shallow) / c_shallow < 1e-6


def test_bottom_and_kinematic_boundary_conditions() -> None:
    """No flow through the bed and zero free-surface kinematic residual to O(a)."""
    w = _wave(kh=1.5, eps=1e-3)
    _, vz = w.kinematics(np.linspace(0, w.wavelength, 11), -w.h, 0.0)
    assert np.max(np.abs(vz)) < 1e-15

    res = kinematic_residual(w, np.linspace(0, 2 * w.wavelength, 41), 0.3 * w.period)
    # linear solution leaves only the O(eps^2) second-order residual
    assert res["normalised"] < 1e-4


def test_kinematic_residual_scales_as_steepness_squared() -> None:
    """The truncated solution is second order: residual / (a w eps^2) is constant."""
    ratios = []
    for eps in (0.0025, 0.005, 0.01, 0.02):
        w = _wave(kh=1.5, eps=eps)
        res = kinematic_residual(w, np.linspace(0, 2 * w.wavelength, 41), 0.3 * w.period)
        ratios.append(res["normalised"] / eps ** 2)
    ratios = np.array(ratios)
    assert ratios.std() / ratios.mean() < 0.05, ratios
    assert ratios.mean() == pytest.approx(2.65, rel=0.05)


def test_velocity_field_is_potential_flow() -> None:
    """Continuity and irrotationality hold to finite-difference accuracy."""
    w = _wave(kh=1.5, eps=0.02)
    x = np.linspace(0.0, w.wavelength, 13).reshape(1, -1)
    z = np.linspace(-0.9 * w.h, 0.9 * w.h, 7).reshape(-1, 1)
    dx, dz = 1e-6 * w.wavelength, 1e-6 * w.h
    u, v = w.kinematics(x, z, 0.0)
    u_p, _ = w.kinematics(x + dx, z, 0.0)
    u_m, _ = w.kinematics(x - dx, z, 0.0)
    _, v_p = w.kinematics(x, z + dz, 0.0)
    _, v_m = w.kinematics(x, z - dz, 0.0)
    _, v_xp = w.kinematics(x + dx, z, 0.0)
    _, v_xm = w.kinematics(x - dx, z, 0.0)
    u_zp, _ = w.kinematics(x, z + dz, 0.0)
    u_zm, _ = w.kinematics(x, z - dz, 0.0)
    scale = w.a * w.omega * w.k
    div = (u_p - u_m) / (2 * dx) + (v_p - v_m) / (2 * dz)
    curl = (v_xp - v_xm) / (2 * dx) - (u_zp - u_zm) / (2 * dz)
    assert np.max(np.abs(div)) / scale < 1e-6
    assert np.max(np.abs(curl)) / scale < 1e-6
    assert np.max(np.abs(u)) > 0.0 and np.max(np.abs(v)) > 0.0


def test_second_harmonic_amplitude_and_zero_mean() -> None:
    """Bound second harmonic follows ``(ka^2/4)(3-s^2)/s^3`` and has zero mean."""
    w = _wave(kh=3.0, eps=0.08)
    s = math.tanh(w.kh)
    expected = 0.25 * w.k * w.a ** 2 * (3.0 - s * s) / s ** 3
    assert w.a2 == pytest.approx(expected, rel=1e-12)
    assert second_order_amplitude(w.a, w.k, w.h) == pytest.approx(expected, rel=1e-12)

    # amplitude ratio in deep water approaches ka/2
    deep = Stokes2Wave(a=0.5, k=2.0, h=200.0)
    assert deep.a2 / deep.a == pytest.approx(0.5 * deep.eps, rel=0.02)

    # a full period of eta at a fixed station has (numerically) zero mean
    t = np.linspace(0.0, w.period, 4001)
    eta = w.eta(0.0, t)
    assert abs(eta.mean()) < 1e-3 * w.a
    # and the harmonic content matches a1/a2 by FFT
    spec = np.abs(np.fft.rfft(eta)) * 2.0 / len(t)
    dt = t[1] - t[0]
    f1 = int(round((1.0 / w.period) / (1.0 / (len(t) * dt))))
    assert spec[f1] == pytest.approx(w.a, rel=2e-3)
    assert spec[2 * f1] == pytest.approx(w.a2, rel=2e-2)


def test_validity_flags_shallow_water_breakdown() -> None:
    """Shallow-water / steep cases are flagged as outside Stokes-2 validity."""
    ok = stokes_validity(0.02 / (1.5 / 20.0), 1.5 / 20.0, 20.0)
    assert ok["second_order_ok"] is True
    bad = stokes_validity(0.08 / (0.7 / 20.0), 0.7 / 20.0, 1.0)
    assert bad["second_order_ok"] is False
    assert bad["ursell"] > 1.0


def test_summary_reports_consistent_derived_quantities() -> None:
    w = _wave(kh=0.7, eps=0.02)
    s = w.summary()
    assert s["wavelength"] == pytest.approx(2 * math.pi / w.k)
    assert s["period"] == pytest.approx(2 * math.pi / w.omega)
    assert s["phase_speed"] == pytest.approx(w.c)
    assert s["eps"] == pytest.approx(w.eps)
    # closed-form dispersion relation
    assert w.c ** 2 == pytest.approx(w.g / w.k * math.tanh(w.kh), rel=1e-12)