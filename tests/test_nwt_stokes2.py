"""Closed-form second-order Stokes theory: machine-checkable analytic anchors.

These tests do not touch the solver.  They pin the *reference* the numerical
wave tank is judged against, so that a wrong anchor cannot masquerade as a
solver error (or vice versa):

* the dispersion relation and its deep/shallow asymptotes;
* the ``cos 2 theta`` coefficient and its ``kh -> inf`` limit ``k a^2 / 2``;
* the exact free-surface boundary conditions, whose residual must be
  second order in the steepness (``eps^2``) — the sharpest available proof that
  the retained second-order terms and the kinematics are mutually consistent;
* continuity of the second-order velocity field.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.nwt.stokes2 import (
    Stokes2Wave,
    dispersion_k,
    dispersion_omega,
    group_speed,
    phase_speed,
)


def test_dispersion_closed_form_identity():
    g, h = 9.81, 10.0
    for k in (0.02, 0.1, 0.5, 1.0):
        omega = dispersion_omega(k, h, g)
        assert omega**2 == pytest.approx(g * k * math.tanh(k * h), rel=1e-14)
        c = phase_speed(k, h, g)
        assert c**2 == pytest.approx((g / k) * math.tanh(k * h), rel=1e-14)


def test_deep_and_shallow_water_limits():
    g = 9.81
    # deep water: tanh(kh) -> 1
    k, h = 1.0, 1.0e3
    assert phase_speed(k, h, g) == pytest.approx(math.sqrt(g / k), rel=1e-6)
    # shallow water: c -> sqrt(g h)
    k, h = 1.0e-4, 5.0
    assert phase_speed(k, h, g) == pytest.approx(math.sqrt(g * h), rel=2e-6)
    # group speed: n -> 1/2 deep, n -> 1 shallow
    assert group_speed(1.0, 1.0e3, g) / phase_speed(1.0, 1.0e3, g) == pytest.approx(0.5, rel=1e-5)
    assert group_speed(1.0e-4, 5.0, g) / phase_speed(1.0e-4, 5.0, g) == pytest.approx(1.0, rel=1e-4)


def test_wavenumber_inversion_round_trip():
    g, h = 9.81, 12.0
    for k in (0.05, 0.2, 0.9):
        omega = dispersion_omega(k, h, g)
        assert dispersion_k(omega, h, g) == pytest.approx(k, rel=1e-12)


def test_second_harmonic_coefficient_and_deep_water_limit():
    g = 9.81
    h, kh, eps = 20.0, 3.0, 0.05
    k = kh / h
    w = Stokes2Wave.from_steepness(eps, k, h, g)
    s = math.tanh(kh)
    assert w.second_harmonic_amplitude == pytest.approx(k * w.amplitude**2 * (3 - s**2) / s**3 / 4, rel=1e-14)
    # deep water: A2 -> k a^2 / 2
    w_deep = Stokes2Wave.from_steepness(eps, k, 1.0e4, g)
    assert w_deep.second_harmonic_amplitude == pytest.approx(k * w_deep.amplitude**2 / 2.0, rel=1e-6)


def test_surface_boundary_conditions_are_second_order_in_steepness():
    """Kinematic and dynamic surface residuals must scale like ``eps**2``."""
    g, h = 9.81, 10.0
    k = 1.2 / h  # kh = 1.2
    res = {}
    for eps in (0.01, 0.02):
        w = Stokes2Wave.from_steepness(eps, k, h, g)
        res[eps] = w.surface_bc_residual(n=800)
    for key in ("kinematic_rms", "dynamic_rms_harmonic"):
        ratio = res[0.02][key] / res[0.01][key]
        # second order => residual grows by 4 when eps doubles
        assert 3.2 < ratio < 4.8, (key, ratio)
        # and it is small in absolute terms (normalised by a*omega resp. g*a)
        assert res[0.02][key] < 5e-3
    # the *constant* part of the dynamic residual is the second-order Bernoulli
    # constant (first order in eps, balanced by a mean-level term the two-harmonic
    # anchor omits); it must scale like eps and must not be confused with the
    # harmonic residual tested above.
    assert 1.8 < res[0.02]["dynamic_mean"] / res[0.01]["dynamic_mean"] < 2.2
    assert res[0.02]["dynamic_mean"] > 3.0 * res[0.02]["dynamic_rms_harmonic"]


def test_velocity_field_is_divergence_free_and_matches_potential():
    g, h = 9.81, 8.0
    w = Stokes2Wave.from_steepness(0.05, 0.9 / h, h, g)
    x = torch.linspace(0.0, 2.0 * w.wavelength, 400, dtype=torch.float64)
    # divergence-free: du/dx + dw/dz = 0 (4th-order differences)
    dx = float(x[1] - x[0])
    for z in (0.2, 0.5, 0.9):
        zz = z * h
        u, _ = w.kinematics(x, torch.full_like(x, zz), 0.0)
        dudx = (-u[4:] + 8 * u[3:-1] - 8 * u[1:-3] + u[:-4]) / (12 * dx)
        dz = 1e-6 * h
        _, wp = w.kinematics(x, torch.full_like(x, zz + dz), 0.0)
        _, wm = w.kinematics(x, torch.full_like(x, zz - dz), 0.0)
        dwdz = ((wp - wm) / (2 * dz))[2:-2]
        scale = float(u.abs().max()) / (0.1 * h)
        assert float((dudx + dwdz).abs().max()) < 1e-3 * scale


def test_kinematics_reduce_to_linear_airy_for_small_steepness():
    g, h = 9.81, 6.0
    k = 1.0
    eps = 1e-4
    w = Stokes2Wave.from_steepness(eps, k, h, g)
    x = torch.tensor([0.3, 1.7, 2.9], dtype=torch.float64)
    z = torch.full_like(x, 0.75 * h)
    u, ww = w.kinematics(x, z, 0.0)
    theta = w.wavenumber * x
    shkh = math.sinh(k * h)
    u_lin = w.amplitude * w.omega * torch.cosh(k * z) / shkh * torch.cos(theta)
    w_lin = w.amplitude * w.omega * torch.sinh(k * z) / shkh * torch.sin(theta)
    assert torch.allclose(u, u_lin, rtol=1e-6)
    assert torch.allclose(ww, w_lin, rtol=1e-6)


def test_from_period_round_trip():
    g, h = 9.81, 15.0
    period = 8.0
    w = Stokes2Wave.from_period(period, amplitude=0.3, depth=h, gravity=g)
    assert w.period == pytest.approx(period, rel=1e-12)
    assert w.phase_speed == pytest.approx(math.sqrt(g / w.wavenumber * math.tanh(w.wavenumber * h)), rel=1e-12)


def test_as_dict_is_finite():
    w = Stokes2Wave.from_steepness(0.08, 0.1, 20.0, 9.81)
    d = w.as_dict()
    for key, val in d.items():
        assert math.isfinite(val), key
    assert d["steepness_eps"] == pytest.approx(0.08, rel=1e-12)