"""Tests for the stand-off resolvability law, its negative controls and its
mass-flux bookkeeping.

The expensive production run lives in ``scripts/run_resolvability_ladder.sh``;
these tests use deliberately tiny grids so the whole file stays in seconds while
still exercising every judgement the study makes:

* the three-state rule, including the *hard* R <= 2 negative control (a coarse
  grid may never be certified, even when its number happens to look right);
* Richardson order recovery and the cross-triplet stability criterion on
  synthetic data with a known order;
* the blockage-corner mass closure (relative caliber < 1e-6 per step) and the
  naive bookkeeping it replaces;
* equivalence of the two accelerators (torch mirror, y-slab pool) with the
  reference numpy kernel -- they must be identical, not merely close.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tensorlbm.compressible_shock import resolvability as R  # noqa: E402
from tensorlbm.compressible_shock.cases import forward_facing_step_case  # noqa: E402


# --------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def step_small():
    """Small forward-step field, advanced a little so a shock has formed."""
    solver, meta = forward_facing_step_case(nx=60, ny=20, mach=3.0, t_end=0.0)
    solver.advance(0.2)
    return solver, meta


@pytest.fixture(scope="module")
def step_unaligned():
    """Step at a non-lattice position: the corner cell is a degenerate cut cell."""
    solver, meta = forward_facing_step_case(
        nx=72, ny=24, mach=3.0, t_end=0.0, step_x=0.6117, step_h=0.2137)
    solver.advance(0.2)
    return solver, meta


# ------------------------------------------------- 1. three-state rule (A1)
def test_hard_rule_never_certifies_below_R2():
    """R <= 2 is UNRESOLVED even if every other indicator looks perfect."""
    v = R.classify_resolvability(R=1.9, shock_thickness_cells=0.5, gci_fine=1e-4,
                                 rel_err_vs_extrapolated=1e-4)
    assert v["state"] == "UNRESOLVED"
    assert "by rule" in v["reasons"][0]


def test_R_between_two_and_four_is_unresolved():
    v = R.classify_resolvability(R=3.0, shock_thickness_cells=0.5, gci_fine=1e-3)
    assert v["state"] == "UNRESOLVED"
    assert any("fewer than" in s for s in v["reasons"])


def test_certified_case_passes_with_reasons():
    v = R.classify_resolvability(R=9.0, shock_thickness_cells=1.2, gci_fine=0.01,
                                 rel_err_vs_extrapolated=0.004)
    assert v["state"] == "PASS"
    assert len(v["reasons"]) >= 3


def test_thick_shock_blocks_certification():
    v = R.classify_resolvability(R=6.0, shock_thickness_cells=3.5, gci_fine=0.001)
    assert v["state"] == "UNRESOLVED"
    assert any("thickness" in s for s in v["reasons"])


def test_large_gci_blocks_certification():
    v = R.classify_resolvability(R=12.0, shock_thickness_cells=1.0, gci_fine=0.08)
    assert v["state"] == "UNRESOLVED"
    assert any("GCI" in s for s in v["reasons"])


def test_non_finite_input_is_withheld_not_passed():
    v = R.classify_resolvability(R=float("nan"), shock_thickness_cells=1.0, gci_fine=0.0)
    assert v["state"] == "WITHHELD"


# ---------------------------------------------- 2. Richardson / GCI (A2)
def _synthetic(levels, order=2.0, limit=0.1234, const=0.05, r=2.0):
    """f_i = limit + const * h_i**order with h_i = r**(1-i)."""
    hs = [r ** (1 - i) for i in range(len(levels))]
    return {lab: limit + const * h ** order for lab, h in zip(levels, hs)}


def test_richardson_recovers_second_order():
    vals = _synthetic(["a", "b", "c"], order=2.0)
    res = R.richardson_triplet(vals["a"], vals["b"], vals["c"], r=2.0)
    assert res["ok"] and res["monotone"]
    assert abs(res["p"] - 2.0) < 1e-8
    assert abs(res["extrapolated"] - 0.1234) < 1e-10
    # GCI on the fine pair is finite, positive and smaller than on the coarse pair
    assert 0.0 < res["gci_fine"] < res["gci_coarse"]
    assert res["gci_fine"] == pytest.approx(
        R.FS_SAFETY * abs((vals["c"] - vals["b"]) / vals["c"]) / (2.0 ** 2 - 1.0))
    assert res["asymptotic"] is True


def test_richardson_recovers_first_order():
    vals = _synthetic(["a", "b", "c"], order=1.0)
    res = R.richardson_triplet(vals["a"], vals["b"], vals["c"], r=2.0)
    assert abs(res["p"] - 1.0) < 1e-8


def test_richardson_flags_non_monotone_sequence():
    res = R.richardson_triplet(1.0, 1.02, 1.01, r=2.0)
    assert res["monotone"] is False
    assert math.isnan(res["p"])
    assert not res["asymptotic"]


def test_cross_triplet_stable_for_consistent_order():
    vals = _synthetic(["G1", "G2", "G3", "G4"], order=2.0)
    out = R.cross_triplet(vals, ["G1", "G2", "G3", "G4"])
    assert out["ok"] and out["stable"]
    assert out["order_in_band"] is True
    assert out["cross_triplet_spread"] < 0.03


def test_cross_triplet_unstable_when_orders_disagree():
    # coarse triplet converges at order 1, fine triplet at order 3: the
    # extrapolated values cannot agree
    r = 2.0
    hs = [r ** (1 - i) for i in range(4)]
    vals = {"G1": 0.1 + 0.5 * hs[0], "G2": 0.1 + 0.5 * hs[1],
            "G3": 0.1 + 0.02 * hs[2] ** 3, "G4": 0.1 + 0.02 * hs[3] ** 3}
    out = R.cross_triplet(vals, ["G1", "G2", "G3", "G4"])
    assert out["ok"]
    assert out["stable"] is False


# ------------------------------------- 3. blockage-corner mass accounting (A3)
def test_relative_bookkeeping_closes_to_roundoff(step_small):
    solver, _ = step_small
    audit = R.blockage_mass_audit(solver, step_x=0.6, step_h=0.2)
    assert audit["ok"]
    assert audit["residual_relative"] < R.MASS_REL_MAX
    assert audit["meets_A3"] is True


def test_naive_bookkeeping_is_much_worse(step_small):
    solver, _ = step_small
    audit = R.blockage_mass_audit(solver, step_x=0.6, step_h=0.2)
    assert audit["residual_naive"] > 100 * audit["residual_relative"]
    assert audit["residual_naive"] > 1e-2


def test_degenerate_corner_cell_also_closes(step_unaligned):
    solver, _ = step_unaligned
    audit = R.blockage_mass_audit(solver, step_x=0.6117, step_h=0.2137)
    corner = audit["corner_cell"]
    assert 0.0 < corner["open_fraction"] < 1.0        # genuinely a cut cell
    assert audit["residual_relative"] < R.MASS_REL_MAX


def test_corner_cell_may_stay_in_the_measurement_window(step_small):
    """Keeping the corner cell must not move the measured stand-off."""
    solver, _ = step_small
    keep = R.measure_standoff(solver, 0.6, 0.2, 3.0, include_corner=True)
    drop = R.measure_standoff(solver, 0.6, 0.2, 3.0, include_corner=False)
    assert keep["ok"] and drop["ok"]
    assert abs(keep["standoff"] - drop["standoff"]) / keep["standoff"] < 0.01


# ------------------------------------------------------- 4. measurement
def test_measure_standoff_recovers_synthetic_shock_position():
    """Build a 1-D profile with a known jump position and misalign it to the grid."""

    class Fake:
        gamma = 1.4
        dx = 0.01
        dy = 0.01
        xc = np.arange(0.005, 3.0, 0.01)
        yc = np.arange(0.005, 1.0, 0.01)

    x_step, x_shock = 0.6, 0.5023
    rho1, rho2 = 1.4, 1.4 * 3.647        # Mach 3 density ratio
    prof = np.where(Fake.xc >= x_shock, rho2, rho1)
    # smear over ~2 cells so the sub-cell interpolation has to do real work
    from scipy.ndimage import uniform_filter1d

    prof = uniform_filter1d(prof, size=3, mode="nearest")
    Fake.U = np.repeat(np.stack([prof, np.zeros_like(prof), np.zeros_like(prof),
                                 prof], axis=-1)[None, ...], len(Fake.yc), axis=0)
    m = R.measure_standoff(Fake, step_x=x_step, step_h=0.2, mach=3.0)
    assert m["ok"]
    assert abs(m["standoff"] - (x_step - x_shock)) < 5 * Fake.dx
    assert m["R"] == pytest.approx(m["standoff"] / Fake.dx)
    assert 0.5 < m["shock_thickness_cells"] < 6.0


def _tanh_profile_solver(nx, x_shock=0.5023, width=0.02, step_x=0.6, ly=1.0):
    """Fake solver holding a *physically* smeared shock (fixed width, any dx)."""
    dxv = 3.0 / nx

    class Fake:
        gamma = 1.4
        dx = dxv
        dy = dxv
        xc = (np.arange(nx) + 0.5) * dxv
        yc = np.arange(0.5 * dxv, ly, dxv)

    # rho2 from the exact Rankine-Hugoniot ratio at Mach 3 (gamma=1.4): the
    # module normalises the mass deficit by that jump, so the synthetic profile
    # must jump to the same state or the comparison carries a constant bias
    rho1, rho2 = 1.4, 1.4 * 3.857142857142857
    # exact cell averages of a tanh front on a sub-sampled stencil
    xf = (np.arange(400 * nx) + 0.5) * (3.0 / (400 * nx))
    fine = rho1 + (rho2 - rho1) * 0.5 * (1.0 + np.tanh((xf - x_shock) / width))
    k = 400
    prof = fine.reshape(nx, k).mean(axis=1)
    Fake.U = np.repeat(np.stack([prof, np.zeros_like(prof), np.zeros_like(prof),
                                 prof], axis=-1)[None, ...], len(Fake.yc), axis=0)
    return Fake, rho1, rho2


def test_mass_deficit_is_second_order_on_a_fixed_profile():
    """The pre-registered QoI converges at (at least) second order on a *fixed*
    physical shock profile -- the property the Richardson ladder needs."""
    ref_solver, _, _ = _tanh_profile_solver(400 * 16)
    ref = R.measure_standoff(ref_solver, step_x=0.6, step_h=0.2, mach=3.0)
    e_mass, dxs = [], []
    for nx in (150, 300, 600, 1200):
        fake, _, _ = _tanh_profile_solver(nx)
        m = R.measure_standoff(fake, step_x=0.6, step_h=0.2, mach=3.0)
        dxs.append(fake.dx)
        e_mass.append(abs(m["standoff_mass"] - ref["standoff_mass"]))
    p_mass = np.polyfit(np.log(dxs), np.log(np.maximum(e_mass, 1e-16)), 1)[0]
    assert p_mass > 1.8, f"mass-deficit QoI converged at p={p_mass:.2f}"


def test_grid_scale_front_limits_the_iso_crossing_but_not_the_mass_deficit():
    """Why the QoI changed.

    A *shock-capturing* front is ~2 cells wide, so its sub-cell position is not
    carried by the data: the 0.5-level crossing is first order.  The mass
    deficit is an exact functional of the cell averages (conservation), so it
    does not care where inside the front the level falls.  Both statements are
    measured here against high-resolution quadrature of the *same* profile."""
    xq = np.linspace(0.15, 0.6, 400001)

    def exact_of(w):
        return float(np.trapezoid(0.5 * (1.0 + np.tanh((xq - 0.5023) / w)), xq))
    e_mass, e_iso, dxs = [], [], []
    for nx in (150, 300, 600, 1200):
        width = 2.0 * 3.0 / nx
        fake, _, _ = _tanh_profile_solver(nx, width=width)
        m = R.measure_standoff(fake, step_x=0.6, step_h=0.2, mach=3.0)
        exact = exact_of(width)
        dxs.append(fake.dx)
        e_mass.append(abs(m["standoff_mass"] - exact))
        e_iso.append(abs(m["standoff"] - exact))
    p_iso = np.polyfit(np.log(dxs), np.log(e_iso), 1)[0]
    assert 0.8 < p_iso < 1.6, f"iso crossing should be ~first order, got p={p_iso:.2f}"
    assert max(e_mass) < 1e-6, e_mass
    assert max(e_iso) > 100 * max(e_mass)


def test_mass_deficit_agrees_with_iso_on_a_smeared_jump():
    fake, _, _ = _tanh_profile_solver(600)
    m = R.measure_standoff(fake, step_x=0.6, step_h=0.2, mach=3.0)
    assert abs(m["standoff_mass"] - m["standoff"]) / m["standoff"] < 0.1


def test_mass_deficit_includes_the_corner_cell():
    """A3: the measurement window may keep the blockage corner cell."""
    fake, _, _ = _tanh_profile_solver(300)
    keep = R.measure_standoff(fake, step_x=0.6, step_h=0.2, mach=3.0, include_corner=True)
    drop = R.measure_standoff(fake, step_x=0.6, step_h=0.2, mach=3.0, include_corner=False)
    assert abs(keep["standoff_mass"] - drop["standoff_mass"]) / keep["standoff_mass"] < 0.01


# ------------------------------------------------------ 5b. A2 / A4 verdicts
def _trio(p, ext=0.28, spread=0.01, gci=0.01):
    return {"ok": True, "triplets": [{"ok": True, "p": p}, {"ok": True, "p": p}],
            "gci_21": gci, "cross_triplet_spread": spread,
            "extrapolated": [ext, ext * (1 + spread)], "extrapolated_finest": ext}


def test_a2_accepts_second_order_stable_triplets():
    a = R.evaluate_a2(_trio(2.1))
    assert a["pass"] and a["order_ok"] and a["gci_ok"] and a["spread_ok"]


def test_a2_rejects_sub_first_order():
    a = R.evaluate_a2(_trio(0.3))
    assert not a["pass"] and not a["order_ok"]
    assert any("outside" in r for r in a["reasons"])


def test_a2_rejects_large_gci_and_unstable_extrapolation():
    assert not R.evaluate_a2(_trio(2.0, gci=0.09))["pass"]
    assert not R.evaluate_a2(_trio(2.0, spread=0.05))["pass"]


def test_a2_without_triplet_never_passes():
    assert not R.evaluate_a2(None)["pass"]
    assert not R.evaluate_a2({"ok": False})["pass"]


def test_a4_monotone_decreasing_trend_passes_with_withheld_correlation():
    per_mach = {3.0: {"standoff_extrapolated": 0.28},
                2.0: {"standoff_extrapolated": 0.39},
                1.5: {"standoff_extrapolated": 0.46}}
    a = R.mach_scan_summary(per_mach, step_h=0.2)
    # Mach order is ascending after sorting; values must decrease
    assert a["direction_ok"] and a["pass"]
    assert a["correlation_withheld"]        # Billig is 20x below a 2-D step value


def test_a4_rejects_wrong_direction():
    per_mach = {2.0: {"standoff_extrapolated": 0.2},
                3.0: {"standoff_extrapolated": 0.3}}
    a = R.mach_scan_summary(per_mach, step_h=0.2)
    assert not a["direction_ok"] and not a["pass"]


# ---------------------------------------- 5. accelerators must be identical
def test_face_fluxes_reproduce_the_rhs(step_small):
    """Guard for the face_fluxes refactor: the flux divergence *is* the RHS."""
    solver, _ = step_small
    Fx, Fy = solver.face_fluxes()
    rh = -(Fx[:, 1:, :] - Fx[:, :-1, :]) / solver.dx - (Fy[1:, :, :] - Fy[:-1, :, :]) / solver.dy
    if solver.solid is not None:
        rh = np.where(solver.solid[..., None], 0.0, rh)
    np.testing.assert_allclose(rh, solver._rhs(solver.U), rtol=1e-13, atol=1e-12)


def test_torch_mirror_matches_numpy():
    pytest.importorskip("torch")
    from tensorlbm.compressible_shock.fast2d import to_torch

    solver, _ = forward_facing_step_case(nx=48, ny=24, mach=3.0, t_end=0.0)
    mirror = to_torch(solver, threads=2)
    np.testing.assert_allclose(mirror.rhs(mirror.U).numpy(), solver._rhs(solver.U),
                               rtol=0, atol=0)
    dt = solver.cfl / (solver.max_speed() * (1 / solver.dx + 1 / solver.dy))
    for _ in range(3):
        U0 = solver.U.copy()
        U1 = solver._enforce_admissible(U0, U0 + dt * solver._rhs(U0))
        U2 = 0.5 * U0 + 0.5 * (U1 + dt * solver._rhs(U1))
        solver.U = solver._enforce_admissible(U0, U2)
        solver.t += dt
        mirror.step(dt)
    np.testing.assert_allclose(mirror.U.numpy(), solver.U, rtol=0, atol=0)


def test_parallel_pool_matches_serial():
    pytest.importorskip("multiprocessing")
    from tensorlbm.compressible_shock.par2d import ParallelEulerSolver2D, self_check

    if not hasattr(os, "fork"):
        pytest.skip("needs fork-based shared memory")
    assert self_check(nx=48, ny=24, nprocs=3) == 0.0
    solver, _ = forward_facing_step_case(nx=48, ny=24, mach=3.0, t_end=0.0)
    ps = ParallelEulerSolver2D.from_solver(solver, nprocs=3)
    try:
        np.testing.assert_allclose(ps.rhs(), solver._rhs(solver.U), rtol=0, atol=0)
        ps.advance(0.02)
        solver.advance(0.02)
        np.testing.assert_allclose(ps.U, solver.U, rtol=0, atol=0)
    finally:
        ps.close()


# ------------------------------------------------------- 6. budget law
def test_grid_budget_law_recovers_the_imposed_scaling():
    """Synthetic ladders with N_min ~ 1/delta must show the imposed Mach trend."""
    per_mach = {}
    for mach, delta, const in ((2.0, 0.32, 0.02), (3.0, 0.27, 0.03), (5.0, 0.19, 0.05)):
        grids = {}
        for nx, dx in ((150, 0.02), (300, 0.01), (600, 0.005)):
            grids[f"G{nx}"] = {"ok": True, "standoff": delta * (1 + const * (dx / 0.02) ** 2),
                               "dx": dx}
        per_mach[mach] = {"standoff_extrapolated": delta, "grids": grids}
    law = R.grid_budget_law(per_mach)
    assert len(law["rows"]) == 3
    # the error metric is |d_fine - d_coarse|/|d_fine|, so the recovered order is
    # 2 only up to the (small) denominator drift: 1.99 for this synthetic family
    for row in law["rows"]:
        assert abs(row["p_fit"] - 2.0) < 0.05
    assert law["exponent_identifiable"] is True
    assert "exponent_q" in law
    assert law["law"].startswith("N_min ~")


def test_grid_budget_law_refuses_to_invent_an_exponent():
    """One difference per Mach cannot separate the order from the Mach trend."""
    per_mach = {m: {"standoff_extrapolated": d,
                    "grids": {f"G{nx}": {"ok": True, "standoff": d * (1 + 0.02 * (dx / 0.02) ** 2),
                                         "dx": dx} for nx, dx in ((150, 0.02), (300, 0.01))}}
                for m, d in ((2.0, 0.32), (3.0, 0.27), (5.0, 0.19))}
    law = R.grid_budget_law(per_mach, p_ref=1.7)
    assert law["exponent_identifiable"] is False
    assert law["p_pooled"] is None
    for row in law["rows"]:
        assert row["p_fit"] == pytest.approx(1.7)


def test_billig_correlation_is_decreasing_in_mach():
    assert R.billig_standoff(2.0) > R.billig_standoff(3.0) > R.billig_standoff(5.0)
    assert R.billig_standoff(3.0) > 0.0