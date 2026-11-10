"""Verification of the compressibility-error-law / Mach-envelope module (p83w2-n01).

These tests pin the *machinery*: the lattice normalisation of the heat-flux
estimator, the exact moment identities of the scalar lattice, bit-identity of the
hydrodynamic collision with the reference implementation, the Richardson and
power-law fits on synthetic data with a known answer, the envelope algebra, and a
smoke run of the DDF solver on a grid small enough for CI.

The physics results of the campaign (the exponents, the hold-out ratio, the
Churchill--Bernstein anchor) live in ``artifacts/p83w2_n01/verification.json``;
a unit test must not pretend to reproduce a 30-core-hour sweep.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.compressibility_error_law import (
    CylinderGeometry,
    ThermalCylinderSetup,
    churchill_bernstein_mean_nu,
    cylinder_geometry,
    fit_ma_error_law,
    flux_moment_diffusivity,
    mach_envelope,
    mach_floor_from_stiffness,
    nusselt_number,
    predict_holdout_envelope,
    richardson_nu0,
    run_thermal_cylinder,
    thermal_equilibrium,
)
from tensorlbm.weak_compressibility import (
    CS2,
    CX,
    CY,
    collide,
    equilibrium,
    stream_periodic,
)
from tensorlbm.compressibility_error_law import _hydro_step, _thermal_collide


# ---------------------------------------------------------------------------
# External anchor
# ---------------------------------------------------------------------------
def test_churchill_bernstein_reference_values():
    """Nu(Re=100, Pr=0.7) = 5.156 and the correlation grows like sqrt(Re)."""
    assert churchill_bernstein_mean_nu(100.0, 0.7) == pytest.approx(5.1561, abs=5e-4)
    assert churchill_bernstein_mean_nu(40.0, 0.7) == pytest.approx(3.3639, abs=5e-4)
    assert churchill_bernstein_mean_nu(200.0, 0.7) == pytest.approx(7.1882, abs=5e-4)
    nus = [churchill_bernstein_mean_nu(re, 0.7) for re in (10, 40, 100, 200, 1000)]
    assert all(b > a for a, b in zip(nus, nus[1:]))
    with pytest.raises(ValueError):
        churchill_bernstein_mean_nu(0.0)


# ---------------------------------------------------------------------------
# Scalar lattice: moments and flux normalisation
# ---------------------------------------------------------------------------
def test_thermal_equilibrium_moments_exact():
    rng = np.random.default_rng(0)
    T = rng.random((4, 5))
    ux = 0.1 * rng.standard_normal((4, 5))
    uy = 0.1 * rng.standard_normal((4, 5))
    g = thermal_equilibrium(T, ux, uy)
    assert np.allclose(g.sum(axis=0), T)
    assert np.allclose(np.tensordot(CX, g, axes=(0, 0)), T * ux)
    assert np.allclose(np.tensordot(CY, g, axes=(0, 0)), T * uy)


@pytest.mark.parametrize("tau_t", [0.55, 0.6, 1.0])
def test_flux_moment_equals_tau_cs2_grad_t(tau_t):
    """``sum_i c_i g_i^neq = -tau_T c_s^2 grad T`` (the ``1/2`` is *not* there).

    This is the normalisation trap of the module: using the physical diffusivity
    ``alpha = c_s^2 (tau_T - 1/2)`` instead of ``kappa = c_s^2 tau_T`` inflates
    ``Nu`` by ``tau_T/(tau_T - 1/2)`` (13.8x at ``Pr = 0.7, Ma = 0.1, D = 16``).
    The test drives a periodic decaying sine wave, for which ``grad T`` is
    analytic, and checks both the asymptotic moment ratio and the decay rate
    (which must give the *physical* ``alpha``, i.e. the ``1/2`` of
    Chapman--Enskog does live in the decay rate).
    """
    nx, ny, amp = 64, 4, 1e-3
    k = 2 * math.pi / nx
    alpha = CS2 * (tau_t - 0.5)
    x = np.arange(nx)[None, :]
    T = amp * np.sin(k * x * np.ones((ny, nx)))
    ux = np.zeros((ny, nx))
    uy = np.zeros((ny, nx))
    g = thermal_equilibrium(T, ux, uy)

    def advance(g, n):
        for _ in range(n):
            Tloc = g.sum(axis=0)
            g = _thermal_collide(g, Tloc, ux, uy, tau_t, "reg")
            g = stream_periodic(g)
        return g

    # asymptotic non-equilibrium state, then the moment identity
    g = advance(g, 300)
    T = g.sum(axis=0)
    gneq = g - thermal_equilibrium(T, ux, uy)
    jx = np.tensordot(CX, gneq, axes=(0, 0))[1]
    dTdx = np.gradient(T[1])
    node = int(np.argmax(np.abs(dTdx)))
    ratio = jx[node] / (-flux_moment_diffusivity(tau_t) * dTdx[node])
    assert ratio == pytest.approx(1.0, abs=0.02)

    # decay rate between two *developed* states -> the physical diffusivity
    a0 = 0.5 * (T.max() - T.min())
    steps = 400
    g = advance(g, steps)
    T = g.sum(axis=0)
    a1 = 0.5 * (T.max() - T.min())
    alpha_eff = -math.log(a1 / a0) / (k * k * steps)
    assert alpha_eff == pytest.approx(alpha, rel=0.02)


# ---------------------------------------------------------------------------
# Hydrodynamics: same operator as the reference implementation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("collision", ["bgk", "trt", "reg"])
def test_hydro_step_matches_reference_collision(collision):
    rng = np.random.default_rng(7)
    f = 0.2 + 0.01 * rng.standard_normal((9, 12, 17))
    f = np.abs(f)
    tau = 0.58
    fast, rho, ux, uy = _hydro_step(f, tau, collision)
    ref = collide(f, tau, collision=collision)
    assert np.array_equal(fast, ref)
    assert np.allclose(rho, f.sum(axis=0))
    assert np.allclose(ux, np.tensordot(CX, f, axes=(0, 0)) / rho)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def test_cylinder_geometry_effective_diameter_and_links():
    geo = cylinder_geometry(240, 400, 120.0, 100.0, 10.0)
    # halfway bounce-back puts the surface on the link midpoints: within one cell
    assert 9.0 <= geo.radius_eff <= 11.0
    assert geo.diameter_eff == pytest.approx(2 * geo.radius_eff)
    assert geo.blockage == pytest.approx(geo.diameter_eff / (geo.ny - 1.0))
    assert geo.link_weight.sum() == pytest.approx(1.0)
    norm = np.hypot(geo.link_normal_x, geo.link_normal_y)
    assert np.allclose(norm, 1.0)
    assert geo.links_y.size > 2.0 * math.pi * geo.radius_eff * 0.8
    assert geo.perimeter == pytest.approx(2 * math.pi * geo.radius_eff)


def test_tau_floor_mach_is_monotone_in_re_and_grid():
    """The stiffness floor grows with Re and shrinks with the grid."""
    f = mach_floor_from_stiffness
    assert f(200.0, 32.0) > f(100.0, 32.0) > f(40.0, 32.0)
    assert f(200.0, 64.0) < f(200.0, 32.0)
    # tau - 1/2 = 3 nu and nu = Ma c_s D / Re  ->  Ma = (tau-1/2) Re /(3 c_s D)
    assert f(100.0, 30.0, tau_floor=0.505) == pytest.approx(
        0.005 * 100.0 / (3.0 * (1.0 / math.sqrt(3.0)) * 30.0), rel=1e-12
    )


# ---------------------------------------------------------------------------
# Ma -> 0 extrapolation and power-law fit on synthetic data
# ---------------------------------------------------------------------------
def test_richardson_recovers_polynomial_limit():
    nu0_true, c2, c4 = 5.2, -1.7, 3.1
    machs = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    nus = nu0_true + c2 * machs ** 2 + c4 * machs ** 4
    out = richardson_nu0(machs, nus, np.full(machs.size, 1e-4))
    assert out["nu0"] == pytest.approx(nu0_true, abs=1e-6)
    assert out["coefficients"][1] == pytest.approx(c2, rel=1e-5)
    assert out["coefficients"][2] == pytest.approx(c4, rel=1e-4)
    assert out["r2"] == pytest.approx(1.0, abs=1e-12)
    # two-point Richardson is consistent for a pure Ma^2 model
    pure = nu0_true + c2 * machs ** 2
    two = richardson_nu0(machs, pure)
    assert two["nu0_two_point"] == pytest.approx(nu0_true, abs=1e-9)


def test_error_law_fit_recovers_exponent_two():
    rng = np.random.default_rng(3)
    nu0, c_true, n_true = 5.2, 0.9, 2.0
    machs = np.array([0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3])
    sig = np.full(machs.size, 1e-4)  # 1e-4 absolute measurement noise
    nus = nu0 * (1.0 + c_true * machs ** n_true) + sig * rng.standard_normal(machs.size)
    out = fit_ma_error_law(machs, nus, nu0, sig)
    assert out["n"] == pytest.approx(n_true, abs=0.05)
    assert out["c"] == pytest.approx(c_true, rel=0.05)
    assert out["r2_loglog"] > 0.99
    assert out["n_ci95"][0] <= n_true <= out["n_ci95"][1]
    assert out["rms_residual_log_n_fixed"] < 0.05


def test_error_law_fit_flags_non_quadratic_scaling():
    """A cubic law must *not* fit the pinned n = 2 model."""
    machs = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    nus = 5.0 * (1.0 + 0.5 * machs ** 3)
    out = fit_ma_error_law(machs, nus, 5.0)
    assert out["n"] == pytest.approx(3.0, abs=0.1)
    assert out["rms_residual_log_n_fixed"] > 0.2


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------
def _per_re_row(re, c, n, d_eff=32.0, stable=0.5, stderr=1e-4):
    return {
        "re": re,
        "nu0": 5.0,
        "nu0_stderr": stderr,
        "c": c,
        "n": n,
        "n_ci95": [n - 0.1, n + 0.1],
        "r2_loglog": 0.995,
        "ma_stable_max": stable,
        "diameter_eff": d_eff,
    }


def test_mach_envelope_band_is_consistent_with_tolerance():
    per_re = {100.0: _per_re_row(100.0, 1.0, 2.0)}
    env = mach_envelope(per_re, tol=0.03)
    row = env["rows"][0]
    assert env["schema"] == "tensorlbm.mach_envelope/v1"
    assert row["ma_max_error"] == pytest.approx(math.sqrt(0.03 / 1.0))
    assert row["ma_band"][0] == pytest.approx(row["ma_min_stiffness"])
    assert row["ma_band"][1] == pytest.approx(row["ma_max_error"])
    assert row["status"] == "accuracy_limited"
    assert env["ok"] is True


def test_mach_envelope_stability_and_empty_limits():
    # stability below the accuracy bound -> stability-limited
    env = mach_envelope({100.0: _per_re_row(100.0, 1.0, 2.0, stable=0.1)})
    assert env["rows"][0]["status"] == "stability_limited"
    assert env["rows"][0]["ma_band"][1] == pytest.approx(0.1)
    # a coefficient so large that the tolerance is unreachable -> empty band
    env = mach_envelope({200.0: _per_re_row(200.0, 1e6, 2.0)})
    assert env["rows"][0]["status"] == "empty"
    assert env["ok"] is False


def test_holdout_prediction_follows_pregistered_rule():
    """``log C`` linear in ``log Re`` reproduces an exact power law for C(Re)."""
    c_of_re = lambda re: 0.5 * (re / 100.0) ** 0.75  # noqa: E731
    train = [
        {"re": 40.0, "c": c_of_re(40.0), "n": 2.0},
        {"re": 100.0, "c": c_of_re(100.0), "n": 2.0},
    ]
    pred = predict_holdout_envelope(train, 200.0, tol=0.03)
    assert pred["c_pred"] == pytest.approx(c_of_re(200.0), rel=1e-9)
    assert pred["ma_pred"] == pytest.approx(math.sqrt(0.03 / c_of_re(200.0)), rel=1e-9)
    assert pred["n_pred"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Solver smoke test (tiny grid, a few hundred steps)
# ---------------------------------------------------------------------------
def test_thermal_cylinder_smoke():
    setup = ThermalCylinderSetup(
        mach=0.1, re=100.0, diameter=6.0,
        transient_tu=6.0, sample_tu=6.0, max_tu=14.0, sample_every_tu=0.5,
    )
    rec = run_thermal_cylinder(setup)
    assert rec["diverged"] is False
    assert np.isfinite(rec["nu_mean"]) and rec["nu_mean"] > 1.0
    assert 0.5 < rec["cd_mean"] < 5.0
    assert rec["case"]["tau"] > 0.5
    assert len(rec["nu_history"]) >= 8
    assert abs(rec["nu_global_balance"]) > 0.0
    # Nu is a *mean over the surface*: the stagnation point is hotter than the wake
    assert rec["nu_surface_max"] > rec["nu_surface_min"]


def test_nusselt_number_is_positive_and_scales_with_diameter():
    """A larger cylinder at the same Re thins the thermal layer -> higher Nu."""
    small = ThermalCylinderSetup(mach=0.1, re=100.0, diameter=6.0,
                                 transient_tu=6.0, sample_tu=6.0, max_tu=12.0,
                                 sample_every_tu=0.5)
    big = ThermalCylinderSetup(mach=0.1, re=100.0, diameter=12.0,
                               transient_tu=6.0, sample_tu=6.0, max_tu=12.0,
                               sample_every_tu=0.5)
    r_small = run_thermal_cylinder(small)
    r_big = run_thermal_cylinder(big)
    assert 0.0 < r_small["nu_mean"] < r_big["nu_mean"]
