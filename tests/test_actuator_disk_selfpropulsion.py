"""Tests for the self-propulsion actuator-disk model (P162W1-N05).

The analytic layers (force distribution, open-water anchor, self-propulsion
iteration, momentum theory, adjudication) are tested exactly; the LBM layer is
smoke-tested on a small grid (finite fields, exact uniform-flow fixed point at
``T = 0``, thrust recoverable from the momentum flux).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.actuator_disk_selfpropulsion import (
    AXIAL_SHAPES,
    RADIAL_SHAPES,
    DiskForceSpec,
    LatticeConfig,
    ResistanceModel,
    adjudicate_coefficients,
    axial_profile,
    build_disk_force,
    disk_force_integrals,
    kp505_anchor,
    radial_profile,
    rankine_disk,
    run_actuator_disk_lbm,
    solve_self_propulsion,
    solve_self_propulsion_brent,
)


# ---------------------------------------------------------------------------
# 1. Loading shapes and the force distribution
# ---------------------------------------------------------------------------


def test_radial_shapes_are_bounded_and_vanish_at_tip() -> None:
    eps = np.linspace(0.0, 1.0, 101)
    for kind in RADIAL_SHAPES:
        g = radial_profile(eps, kind)
        assert np.all(g >= 0.0)
        assert np.isclose(g[-1], 0.0) or kind == "uniform"
    assert np.allclose(radial_profile(eps, "uniform"), 1.0)
    # betz: zero at the hub, maximum at 1/sqrt(2) -- minimum-induced-loss shape
    g_betz = radial_profile(eps, "betz")
    assert int(np.argmax(g_betz)) == int(np.argmin(np.abs(eps - 1.0 / math.sqrt(2.0))))
    with pytest.raises(ValueError):
        radial_profile(eps, "not-a-shape")


def test_axial_shapes_have_unit_compact_support() -> None:
    s = np.linspace(-1.5, 1.5, 601)
    box = axial_profile(s, "box")
    cos = axial_profile(s, "cosine")
    assert np.all(box[np.abs(s) <= 1.0] == 1.0)
    assert np.all(box[np.abs(s) > 1.0] == 0.0)
    assert np.isclose(cos[np.abs(s) >= 1.0].max(), 0.0)
    # cosine integrates to exactly the width times 1 (half-cosine area = 1)
    assert math.isclose(float(np.trapezoid(cos, s)), 1.0, rel_tol=1e-6)
    with pytest.raises(ValueError):
        axial_profile(s, "not-a-shape")


@pytest.mark.parametrize("loading", RADIAL_SHAPES)
@pytest.mark.parametrize("axial", AXIAL_SHAPES)
def test_force_distribution_integrals_are_exact(loading: str, axial: str) -> None:
    """The discrete force must integrate to exactly the requested T and Q."""
    spec = DiskForceSpec(
        diameter=16.0, thrust=1.0, torque=0.25, loading=loading, axial=axial,
        center=(20.0, 24.0, 24.0),
    )
    shape = (48, 48, 60)
    fld = build_disk_force(spec, shape)
    integ = disk_force_integrals(fld["fx"], fld["fy"], fld["fz"], spec.center)
    assert integ["thrust"] == pytest.approx(1.0, rel=1e-12)
    assert integ["torque"] == pytest.approx(0.25, rel=1e-12)
    assert fld["n_cells"] > 0


def test_force_points_downstream_and_scales_linearly() -> None:
    spec = DiskForceSpec(diameter=12.0, thrust=0.5, torque=0.1, center=(15.0, 16.0, 16.0))
    fld = build_disk_force(spec, (32, 32, 40))
    assert fld["fx"].min() >= 0.0  # force on the fluid is downstream (+x)
    assert fld["fx"].max() > 0.0
    spec2 = DiskForceSpec(diameter=12.0, thrust=1.0, torque=0.2, center=(15.0, 16.0, 16.0))
    fld2 = build_disk_force(spec2, (32, 32, 40))
    assert np.allclose(fld2["fx"], 2.0 * fld["fx"])
    assert np.allclose(fld2["fy"], 2.0 * fld["fy"])


def test_zero_thrust_gives_zero_force() -> None:
    spec = DiskForceSpec(diameter=12.0, thrust=0.0, torque=0.0, center=(15.0, 16.0, 16.0))
    fld = build_disk_force(spec, (32, 32, 40))
    assert not np.any(fld["fx"])
    assert not np.any(fld["fy"])
    assert not np.any(fld["fz"])
    assert fld["thrust_applied"] == 0.0


def test_invalid_specs_raise() -> None:
    with pytest.raises(ValueError):
        DiskForceSpec(diameter=10.0, thrust=1.0, torque=0.0, hub_ratio=1.2)
    with pytest.raises(ValueError):
        DiskForceSpec(diameter=10.0, thrust=1.0, torque=0.0, loading="bogus")
    with pytest.raises(ValueError):
        DiskForceSpec(diameter=10.0, thrust=1.0, torque=0.0, half_thickness=0.0)


# ---------------------------------------------------------------------------
# 2. Open-water anchor
# ---------------------------------------------------------------------------


def test_anchor_reproduces_published_curve() -> None:
    from tensorlbm.selfpropulsion import kp505_reference_table

    anchor = kp505_anchor(degree=4)
    table = kp505_reference_table()
    j = np.asarray(table["J"], float)
    kt = np.asarray(table["KT"], float)
    kq = np.asarray(table["KQ"], float)
    assert np.max(np.abs(anchor.kt(j) - kt)) < 0.01
    assert np.max(np.abs(anchor.kq(j) - kq)) < 0.002
    assert anchor.fit_rmse[0] < 0.005 and anchor.fit_rmse[1] < 0.001


def test_anchor_is_monotone_and_has_analytic_slope() -> None:
    anchor = kp505_anchor()
    j = np.linspace(0.1, 1.05, 200)
    kt = anchor.kt(j)
    assert np.all(np.diff(kt) < 0.0)  # criterion C5
    # analytic derivative vs central difference of the polynomial
    for jj in (0.3, 0.7, 1.0):
        h = 1e-6
        fd = (float(anchor.kt(jj + h)) - float(anchor.kt(jj - h))) / (2 * h)
        assert math.isclose(anchor.dkt_dj(jj), fd, rel_tol=1e-5)
    # eta0 in a physically plausible band for a propeller
    eta = float(anchor.eta0(0.7))
    assert 0.5 < eta < 0.85


# ---------------------------------------------------------------------------
# 3. Self-propulsion iteration
# ---------------------------------------------------------------------------


def _ship():
    return ResistanceModel(length=230.0, wetted_area=9500.0, form_factor=1.15,
                          rho=1025.0, nu=1.19e-6)


def test_newton_self_propulsion_converges_within_20_iterations() -> None:
    """Criterion C2: |T - R|/R < 1e-3 within 20 iterations."""
    anchor = kp505_anchor()
    res = _ship()
    U = 12.0 * 0.514444
    out = solve_self_propulsion(U, anchor, res, 6.0, rho=1025.0, tol=1e-6)
    assert out["converged"]
    assert out["iterations"] <= 20
    assert out["rel_residual"] < 1e-3
    assert out["iterations_to_1e-3"] is not None and out["iterations_to_1e-3"] <= 20
    # the converged point really balances thrust and resistance
    assert out["thrust"] == pytest.approx(res.resistance(U), rel=1e-6)
    # residual history is monotonically informative (first entry above tol)
    assert out["history"][0]["rel_residual"] > out["history"][-1]["rel_residual"]


def test_newton_matches_brent() -> None:
    anchor = kp505_anchor()
    res = _ship()
    U = 12.0 * 0.514444
    newton = solve_self_propulsion(U, anchor, res, 6.0, rho=1025.0,
                                   wake_fraction=0.25, thrust_deduction=0.15,
                                   mode="effective", tol=1e-10)
    brent = solve_self_propulsion_brent(U, anchor, res, 6.0, rho=1025.0,
                                        wake_fraction=0.25, thrust_deduction=0.15,
                                        mode="effective")
    assert math.isclose(newton["n"], brent["n"], rel_tol=1e-6)
    assert math.isclose(newton["J"], brent["J"], rel_tol=1e-6)
    # thrust deduction mode balances T(1-t) = R
    assert newton["thrust"] * (1.0 - 0.15) == pytest.approx(res.resistance(U), rel=1e-6)


def test_self_propulsion_point_scales_with_resistance() -> None:
    """Doubling the resistance raises revolutions and lowers the advance ratio.

    Because ``T = K_T(J) rho n^2 D^4`` with ``K_T`` increasing as ``J`` falls, the
    revolutions do not grow like ``sqrt(2)`` -- the operating point slides up the
    ``K_T`` curve instead.  Both trends must be present and the shift bounded.
    """
    anchor = kp505_anchor()
    res = _ship()
    U = 6.0
    a = solve_self_propulsion(U, anchor, res, 6.0, rho=1025.0)
    res_hard = ResistanceModel(length=230.0, wetted_area=19000.0, form_factor=1.15,
                               rho=1025.0, nu=1.19e-6)
    b = solve_self_propulsion(U, anchor, res_hard, 6.0, rho=1025.0)
    assert b["n"] > a["n"]
    assert b["J"] < a["J"]
    assert 1.10 < b["n"] / a["n"] < 1.30
    assert a["converged"] and b["converged"]


# ---------------------------------------------------------------------------
# 4. Rankine momentum theory
# ---------------------------------------------------------------------------


def test_rankine_relations_and_energy_identity() -> None:
    rho, U, R = 1.0, 0.05, 10.0
    area = math.pi * R * R
    thrust = 0.8
    rk = rankine_disk(thrust, rho, U, area)
    # C_T = 4 a (1 + a)
    assert math.isclose(rk["CT"], 4.0 * rk["a"] * (1.0 + rk["a"]), rel_tol=1e-12)
    assert math.isclose(rk["u_disk"], U * (1.0 + rk["a"]), rel_tol=1e-12)
    assert math.isclose(rk["u_far"], U * (1.0 + 2.0 * rk["a"]), rel_tol=1e-12)
    # C6 (analytic form): power input equals the far-wake kinetic-energy flux
    ke_flux = 0.5 * rho * area * rk["u_disk"] * (rk["u_far"] ** 2 - U ** 2)
    assert math.isclose(rk["power_input"], ke_flux, rel_tol=1e-12)
    # thrust from the momentum flux equals the imposed thrust
    assert math.isclose(rho * area * rk["u_disk"] * (rk["u_far"] - U), thrust, rel_tol=1e-12)


def test_rankine_zero_thrust_is_freestream() -> None:
    rk = rankine_disk(0.0, 1.0, 0.05, math.pi * 100.0)
    assert rk["a"] == 0.0
    assert rk["u_disk"] == pytest.approx(0.05)
    assert rk["u_far"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# 5. Adjudication
# ---------------------------------------------------------------------------


def test_adjudication_detects_discrepancies() -> None:
    same = adjudicate_coefficients(0.7, {"a": (0.29, 0.047), "b": (0.29, 0.047)})
    assert same["max_kt_error"] == 0.0 and same["max_kq_error"] == 0.0
    diff = adjudicate_coefficients(0.7, {"a": (0.29, 0.047), "b": (0.319, 0.047)})
    assert diff["max_kt_error"] == pytest.approx(0.1, rel=1e-9)
    with pytest.raises(ValueError):
        adjudicate_coefficients(0.7, {})


# ---------------------------------------------------------------------------
# 6. LBM layer (small-grid smoke tests)
# ---------------------------------------------------------------------------


def _small_case(thrust_scale: float, steps: int = 60) -> LatticeConfig:
    D = 6.0
    u0 = 0.05
    j = 0.7
    anchor = kp505_anchor()
    n = u0 / (j * D)
    return LatticeConfig(
        nx=50, ny=20, nz=20, diameter=D, u_inlet=u0,
        tau=0.5 + 3.0 * (u0 * D / 60.0),   # Re_D = 60, nu = u0 D / Re
        thrust=float(anchor.kt(j)) * n * n * D ** 4 * thrust_scale,
        torque=float(anchor.kq(j)) * n * n * D ** 5 * thrust_scale,
        n_steps=steps, sample_steps=max(10, steps // 3), report_every=steps,
        disk_x_over_D=2.5, far_station_over_D=3.0, name="unit-test",
    )


def test_lbm_negative_control_keeps_uniform_flow() -> None:
    """Criterion C4: with T = 0 the flow must remain the uniform freestream."""
    out = run_actuator_disk_lbm(_small_case(0.0, steps=40))
    assert out["force_checks"]["thrust_applied"] == 0.0
    assert out["uniform_dev_max"] < 1e-6
    assert out["u_max_abs"] == pytest.approx(0.05, rel=1e-5)


def test_lbm_actuator_disk_case_is_finite_and_momentum_consistent() -> None:
    out = run_actuator_disk_lbm(_small_case(1.0, steps=80))
    assert np.isfinite(out["u_disk_lbm"])
    assert np.isfinite(out["u_far_lbm"])
    assert np.isfinite(out["energy"]["residual_rel"])
    cfg = out["config"]
    assert 0.85 < out["rho_range"][0] and out["rho_range"][1] < 1.15
    # thrust recovered from the axial momentum flux is of the right magnitude
    # (a coarse short run is only expected to be within a factor ~2)
    assert 0.3 < out["thrust_momentum"] / cfg["thrust"] < 2.5
    # ... and the disk accelerates the flow
    assert out["u_disk_lbm"] > cfg["u_inlet"]
    assert out["u_max_abs"] < 0.4  # lattice Mach number stays low