"""pytest coverage for :mod:`tensorlbm.fsi_added_mass` (P49-N02).

Covers the closed-form interface theory, the 1D anchor, both accelerators, the
error-sequence estimator, and the 2D LBM coupled interface (added-mass operator,
explicit divergence vs stabilised convergence, short free decay).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.fsi_added_mass import (
    AitkenRelaxation,
    HeavingDiskPartitioned,
    InterfaceModel1D,
    IQNILSRelaxation,
    PistonPartitioned1D,
    critical_density_ratio,
    estimate_rho_dd,
    identify_added_mass_damping,
    potential_added_mass,
    rho_dd_analytic,
)


# --------------------------------------------------------------------------- #
# closed form
# --------------------------------------------------------------------------- #
def test_rho_dd_closed_form_values():
    assert rho_dd_analytic(1.0, c_a=1.0, omega=1.0) == pytest.approx(0.0, abs=1e-12)
    assert rho_dd_analytic(0.5, c_a=1.0, omega=1.0) == pytest.approx(1.0, abs=1e-12)
    assert rho_dd_analytic(0.1, c_a=1.0, omega=1.0) == pytest.approx(9.0, abs=1e-12)
    # under-relaxation shrinks the factor
    assert rho_dd_analytic(0.1, c_a=1.0, omega=0.1) == pytest.approx(0.0, abs=1e-12)
    # residual form (structure with its own inertia) is the other textbook value
    assert rho_dd_analytic(2.0, c_a=1.0, form="residual") == pytest.approx(0.5, abs=1e-12)


def test_critical_density_ratio():
    # |1 - omega c_a / r| = 1  <=>  r = omega c_a / 2
    assert critical_density_ratio(c_a=1.0, omega=1.0) == pytest.approx(0.5)
    assert critical_density_ratio(c_a=0.9515, omega=1.0) == pytest.approx(0.47575)
    assert critical_density_ratio(c_a=1.0, omega=0.4) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        rho_dd_analytic(0.0)


# --------------------------------------------------------------------------- #
# 1D anchor: analytic vs numerical
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ratio", [0.1, 0.5, 1.0, 2.0])
def test_1d_anchor_matches_analytic(ratio):
    model = InterfaceModel1D(
        mass_s=ratio, added_mass=1.0, stiffness=1.0, displacement=0.01, velocity=0.0, window=1.0
    )
    rec = PistonPartitioned1D(model).sub_iterate(scheme="none", max_iter=60, tol=1e-12)
    analytic = model.rho_dd(1.0)
    if analytic > 1e-6:
        assert rec["rho_dd_numeric"] == pytest.approx(analytic, rel=1e-6)
    # marginal (analytic == 1) and unstable cases must not converge
    if analytic >= 1.0 - 1e-12:
        assert not rec["converged"]


def test_1d_anchor_converges_at_high_density_ratio():
    model = InterfaceModel1D(mass_s=4.0, added_mass=1.0, stiffness=1.0, displacement=0.05)
    rec = PistonPartitioned1D(model).sub_iterate(scheme="none", max_iter=90, tol=1e-8)
    assert rec["converged"] and rec["rho_dd_numeric"] == pytest.approx(0.75, rel=1e-5)
    assert rec["velocity_error"] <= 1e-7


# --------------------------------------------------------------------------- #
# accelerators
# --------------------------------------------------------------------------- #
def test_aitken_stabilises_divergent_ratio():
    model = InterfaceModel1D(mass_s=0.1, added_mass=1.0, stiffness=1.0, displacement=0.01)
    explicit = PistonPartitioned1D(model).sub_iterate(scheme="none", max_iter=40)
    aitken = PistonPartitioned1D(model).sub_iterate(scheme="aitken", max_iter=40)
    assert not explicit["converged"] and explicit["growth_sigma"] > 0.0
    assert aitken["converged"] and aitken["iterations"] <= 6
    assert aitken["velocity_error"] <= 1e-10


def test_iqn_ils_secant_is_exact_for_linear_map():
    # scalar linear map r(x) = -3 (x - x*): IQN-ILS must find x* in <= 3 iterations
    def f(x):
        return -3.0 * (x - 2.0)

    acc = IQNILSRelaxation(reuse=5)
    acc.reset()
    x = 0.0
    iters = 0
    for _ in range(10):
        r = np.array([f(x)])
        dx, _ = acc.step(np.array([x]), r)
        x += float(dx[0])
        iters += 1
        if abs(x - 2.0) < 1e-12:
            break
    assert abs(x - 2.0) < 1e-10
    assert iters <= 3


def test_iqn_ils_history_is_bounded():
    acc = IQNILSRelaxation(reuse=3)
    acc.reset()
    for k in range(10):
        acc.step(np.array([float(k)]), np.array([1.0 / (k + 1.0)]))
    assert len(acc._dR) <= 3 and len(acc._dX) <= 3


def test_aitken_is_robust_to_identical_residuals():
    acc = AitkenRelaxation(omega0=1.0)
    acc.reset()
    dx, om = acc.step(np.array([0.0]), np.array([1.0]))
    assert np.allclose(dx, [1.0]) and om == pytest.approx(1.0)
    dx2, om2 = acc.step(np.array([1.0]), np.array([1.0]))  # zero denominator
    assert np.all(np.isfinite(dx2)) and math.isfinite(om2)


# --------------------------------------------------------------------------- #
# estimator
# --------------------------------------------------------------------------- #
def test_estimate_rho_dd_recovers_geometric_sequence():
    rho = 0.62
    seq = [rho**k for k in range(12)]
    est = estimate_rho_dd(seq)
    assert est["rho_ratio"] == pytest.approx(rho, rel=1e-9)
    assert est["rho_fit"] == pytest.approx(rho, rel=1e-6)
    assert est["r2"] > 0.999
    growing = [1.7**k for k in range(10)]
    est_g = estimate_rho_dd(growing)
    assert est_g["sigma"] > 0.0 and est_g["rho_ratio"] > 1.0


def test_estimate_rho_dd_rejects_empty():
    with pytest.raises(ValueError):
        estimate_rho_dd([])


# --------------------------------------------------------------------------- #
# free-decay identification (synthetic, no LBM)
# --------------------------------------------------------------------------- #
def test_identify_added_mass_damping_synthetic():
    omega, zeta, mass, stiffness = 0.4, 0.05, 2.0, 1.0
    added = stiffness / omega**2 - mass
    t = np.arange(0.0, 200.0, 0.5)
    y = np.exp(-zeta * omega * t) * np.cos(omega * t)
    ident = identify_added_mass_damping(t, y, mass, stiffness)
    assert ident["omega"] == pytest.approx(omega, rel=2e-3)
    assert ident["zeta"] == pytest.approx(zeta, rel=0.15)
    assert ident["added_mass"] == pytest.approx(added, rel=1e-2)


# --------------------------------------------------------------------------- #
# 2D LBM coupled interface
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# potential-flow reference (incompressible, same staircase mask)
# --------------------------------------------------------------------------- #
def test_potential_added_mass_circular_body():
    res = potential_added_mass(48, 48, 6.0)
    # unbounded value for a circular cylinder is exactly rho*pi*R^2
    assert 0.85 < res["c_a"] < 1.20
    assert res["body_area"] == pytest.approx(math.pi * 36.0, rel=0.05)
    # both resolutions stay within 8% of the analytic rho*pi*R^2 (staircase mask)
    big = potential_added_mass(96, 96, 12.0)
    assert abs(big["c_a"] - 1.0) < 0.08
    assert abs(big["body_area"] - math.pi * 144.0) / (math.pi * 144.0) < 0.02


@pytest.fixture(scope="module")
def small_case_factory():
    cache = {}

    def make(ratio):
        if ratio not in cache:
            cache[ratio] = HeavingDiskPartitioned(
                nx=48,
                ny=48,
                radius=6.0,
                density_ratio=ratio,
                tau=0.55,
                window=10,
                spring_c=2.0,
                initial_displacement=1e-3,
            )
        return cache[ratio]

    return make


def test_2d_added_mass_operator_is_physical(small_case_factory):
    case = small_case_factory(1.0)
    res = case.measure_impulse_added_mass()
    assert res["K"] > 0.0
    # discrete operator of a 10-step window on a periodic box: O(1) x rho*V
    assert 0.3 < res["c_a"] < 1.5
    assert res["nonlinearity"] < 0.05


def test_2d_explicit_matches_closed_form_at_ratio_one(small_case_factory):
    case = small_case_factory(1.0)
    c_a = case.measure_impulse_added_mass()["c_a"]
    rec = case.sub_iterate(scheme="none", max_iter=30, tol=1e-6)
    assert rec["converged"]
    assert rec["rho_dd_numeric"] == pytest.approx(rho_dd_analytic(1.0, c_a=c_a), abs=5e-3)


def test_2d_explicit_diverges_for_light_body(small_case_factory):
    case = small_case_factory(0.1)
    rec = case.sub_iterate(scheme="none", max_iter=25, tol=1e-6)
    assert rec["diverged"] and rec["growth_sigma_preclip"] > 0.0
    assert rec["rho_dd_preclip"] > 1.5


def test_2d_stabilised_schemes_converge_for_light_body(small_case_factory):
    for scheme in ("aitken", "iqn-ils"):
        case = small_case_factory(0.1)
        rec = case.sub_iterate(scheme=scheme, max_iter=25, tol=1e-6)
        assert rec["converged"], scheme
        assert rec["iterations"] <= 8


def test_2d_short_free_decay_runs(small_case_factory):
    case = HeavingDiskPartitioned(
        nx=48,
        ny=48,
        radius=4.0,
        density_ratio=1.0,
        tau=0.55,
        window=5,
        spring_c=0.5,
        initial_displacement=0.2,
    )
    res = case.run_time_accurate(n_windows=16, scheme="aitken", tol=1e-8, max_iter=10)
    assert not res["lbm_failed"]
    assert res["n_windows"] == 16
    assert np.all(np.isfinite(res["y"]))
    assert max(abs(np.asarray(res["y"]))) < 0.5
