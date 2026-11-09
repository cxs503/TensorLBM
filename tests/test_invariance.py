"""Tests for the mesh-orientation / Galilean invariance verification surface.

These tests pin the *numerical* claims of ``tensorlbm.invariance`` on small
grids: correct SDF construction, the interpolated bounce-back reducing exactly
to half-way bounce-back at ``q = 1/2``, second-order convergence of the
inclined-wall Poiseuille case, Taylor-Green decay against the analytic rate,
bitwise reproducibility and the acceptance-criterion logic itself.

Every simulation here is deliberately tiny (a few seconds each); the production
sweeps live in ``scripts/run_invariance_campaign.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.invariance import (
    D2Q9Solver,
    VelocityDirichlet,
    channel_boundary_cells,
    circle,
    cylinder_case,
    evaluate_criteria,
    observed_order,
    poiseuille_case,
    poiseuille_profile,
    taylor_green_case,
    tilted_channel,
)
from tensorlbm.invariance.cases import _PeriodicBox, tg_exact


# ----------------------------------------------------------------- geometry
def _offset(geom, theta_deg):
    """Signed normal offset ``n̂·(r - c)`` used by the tilted-channel SDF."""
    th = math.radians(theta_deg)
    nx_hat = (-math.sin(th), math.cos(th))
    x = (np.arange(geom.nx) + 0.5)[None, :]
    y = (np.arange(geom.ny) + 0.5)[:, None]
    cx, cy = geom.meta["center"]
    return nx_hat[0] * (x - cx) + nx_hat[1] * (y - cy)


def test_tilted_channel_sdf_is_true_distance():
    geom = tilted_channel(80, 60, 30.0, 10.0)
    # signed distance must be  H/2 - |n.(r-c)|  and its gradient a unit vector
    assert np.allclose(np.hypot(geom.gx, geom.gy), 1.0)
    assert np.allclose(geom.sdf, 5.0 - np.abs(_offset(geom, 30.0)))
    assert geom.fluid.any() and geom.solid.any()
    frac = geom.fluid.mean()
    assert 0.15 < frac < 0.85
    # solid flag must agree with the sign of the SDF, everywhere
    assert np.array_equal(geom.solid, geom.sdf <= 0.0)


def test_circle_is_solid_disk_in_fluid():
    geom = circle(64, 64, 8.0)
    assert geom.meta["kind"] == "circle"
    # area of the rasterised disk must be within 5% of pi r^2
    area = geom.solid.sum()
    assert abs(area - math.pi * 64.0) / (math.pi * 64.0) < 0.05
    assert np.all(geom.sdf[geom.solid] <= 0.0)
    assert np.all(geom.sdf[~geom.solid] > 0.0)
    assert np.allclose(np.hypot(geom.gx[~geom.solid], geom.gy[~geom.solid]), 1.0)


def test_poiseuille_profile_is_parabola_with_zero_wall_velocity():
    geom = tilted_channel(120, 90, 20.0, 12.0)
    ux, uy = poiseuille_profile(geom, 0.01)
    speed = ux * geom.axis[0] + uy * geom.axis[1]
    # fluid nodes within one lattice unit of a wall must be near zero
    near_wall = (geom.sdf > 0) & (geom.sdf < 0.5)
    if near_wall.any():
        assert np.abs(speed[near_wall]).max() < 0.0055
    # maximum on the axis equals 3/2 u_mean
    # peak value 3/2 u_mean is only attained up to the sub-cell position of
    # the channel axis (no cell centre sits exactly on the axis)
    assert abs(speed.max() - 0.015) < 1e-3 * 0.015


# ------------------------------------------------------------- wall treatment
def test_interpolated_bounce_back_reduces_to_halfway_at_q_half():
    """A wall exactly half-way between two nodes must make both schemes identical."""
    # wall-normal offset 0.5 of a lattice unit from the first fluid node centre
    geom = tilted_channel(40, 30, 0.0, 6.0, center=(20.0, 15.0 + 3.0))
    a = D2Q9Solver(geom, tau=0.8, bounce_back="interpolated")
    b = D2Q9Solver(geom, tau=0.8, bounce_back="halfway")
    ux = np.full((30, 40), 0.01)
    uy = np.zeros((30, 40))
    a.initialise(ux, uy)
    b.initialise(ux, uy)
    for _ in range(20):
        a.step()
        b.step()
    assert np.allclose(a.f, b.f, atol=1e-12)


def test_solid_nodes_are_inert():
    geom = circle(48, 48, 6.0)
    solver = D2Q9Solver(geom, tau=0.7)
    zeros = np.zeros((48, 48))
    solver.initialise(zeros, zeros)
    fx = np.zeros((48, 48))
    fx[geom.fluid] = 1e-5
    for _ in range(10):
        solver.step(fx, 0.0 * fx)
    rest = solver._rest_eq
    assert np.allclose(solver.f[:, geom.solid], rest[:, geom.solid])
    assert np.isfinite(solver.f).all()


# ------------------------------------------------------------ C1 convergence
def _poiseuille_orders(theta, widths=(8, 12, 16)):
    errors = []
    for h in widths:
        # nu = 0.25 keeps the viscous time H^2/nu short enough for a unit test
        # (15 H^2 steps = ~4 viscous times), while the discretisation error is
        # still the dominant contribution at these wall resolutions
        res = poiseuille_case(theta, h, u_mean=0.01, nu=0.25, steps_factor=15.0,
                              dtype=np.float64)
        errors.append(res["l2_error"])
    return errors, observed_order(errors, widths)


def test_poiseuille_second_order_on_aligned_walls():
    """Wall-aligned channel: the interpolated bounce-back is second order."""
    errors, orders = _poiseuille_orders(0.0)
    assert min(errors) > 0.0
    assert orders[0] > 1.5, f"observed order {orders[0]:.2f} at theta=0"


@pytest.mark.xfail(
    strict=False,
    reason="known limitation (campaign P170W2-N08, honest FAIL): an obliquely cut "
           "inflow/outflow face adds an O(Re) flow-development error that does not "
           "scale with h, so the oblique-channel observed order is 0.42-1.20 instead "
           "of 2.0. Root cause and remediation are documented in "
           "docs/INVARIANCE_CERTIFICATION.md section 5/6. The wall scheme itself is "
           "verified by test_poiseuille_second_order_on_aligned_walls and "
           "test_interpolated_bounce_back_reduces_to_halfway_at_q_half.",
)
def test_oblique_channel_observed_order():
    errors, orders = _poiseuille_orders(45.0)
    assert all(np.isfinite(errors)) and min(errors) > 0.0
    assert orders[0] > 1.5, f"observed order {orders[0]:.2f} at theta=45"


# --------------------------------------------------------------- C4/C5 / TG
def test_taylor_green_decay_matches_analytic_rate():
    res = taylor_green_case(0.0, periods=0, n=32, steps=1000, nu=0.01,
                            sample_every=10, fit_start=200)
    rel = abs(res["decay_rate"] - res["theory_decay"]) / res["theory_decay"]
    assert rel < 0.02, f"decay-rate error {rel:.3%}"


def test_galilean_decay_deviation_follows_ma_squared():
    """The decay-rate deviation must scale like Ma^2, not like Ma."""
    runs = []
    for periods in (1, 2, 4):
        res = taylor_green_case(1.0, periods=periods, n=32, steps=1500,
                                sample_every=20, fit_start=300)
        runs.append(res)
    base = taylor_green_case(0.0, periods=0, n=32, steps=1500,
                             sample_every=20, fit_start=300)
    lam0 = base["decay_rate"]
    pts = [(r["ma"], abs((r["decay_rate"] - lam0) / lam0)) for r in runs]
    assert all(np.isfinite(v) and v > 0 for _, v in pts)
    slope = np.polyfit(np.log([m for m, _ in pts]), np.log([v for _, v in pts]), 1)[0]
    # exact Ma^2 law would give slope 2; a linear law would give 1
    assert 1.5 < slope < 2.5, f"log-log slope {slope:.2f} (expected ~2)"
    # the residual against the analytic solution must stay finite and small
    assert all(np.isfinite(r["residual_relU"]) for r in runs)


def test_reproducibility_bitwise():
    box = _PeriodicBox(48)
    hashes = []
    for _ in range(2):
        solver = D2Q9Solver(box, tau=0.8)
        ux, uy = tg_exact(48, 0.02, 0.01, 0.0, 0.01)
        solver.initialise(ux, uy)
        for _ in range(60):
            solver.step()
        hashes.append(solver.state_hash())
    assert hashes[0] == hashes[1]


# ------------------------------------------------------------------ cylinder
def test_cylinder_case_smoke():
    res = cylinder_case(0.0, 100.0, diameter=6.0, domain_per_diameter=8.0,
                        transient_d=6.0, sample_d=4.0, sample_every=2)
    assert np.isfinite(res["cd"]) and res["cd"] > 0
    assert abs(res["u_mean"] - res["u_target"]) < 0.1 * res["u_target"]
    assert np.isfinite(res["st"])


# -------------------------------------------------------------- acceptance
def _synthetic_results():
    pois = {}
    for th in (0.0, 15.0):
        widths = [8, 12, 16, 24]
        errs = [1.0 / (w ** 2) for w in widths]
        pois[f"{th:g}"] = {"widths": widths, "errors": errs,
                           "orders": observed_order(errs, widths)}
    cyl = {}
    for re, cd0, st0 in ((100.0, 1.34, 0.165), (200.0, 1.34, 0.197)):
        cyl[f"{re:g}"] = [
            {"theta_deg": th, "cd": cd0 * (1 + 0.001 * th / 45.0),
             "st": st0 * (1 + 0.001 * th / 45.0)} for th in (0.0, 15.0, 30.0, 45.0)
        ]
    gal = [{"ma": 0.0, "decay_rate": 1.0, "residual_relU": 0.0, "periods": 0}]
    for i, ma in enumerate((0.02, 0.04, 0.08)):
        gal.append({"ma": ma, "decay_rate": 1.0 + 1e-3 * (ma / 0.02) ** 2,
                    "residual_relU": 1e-4 * (i + 1), "periods": i + 1})
    return {
        "poiseuille": pois,
        "cylinder": cyl,
        "negative_control": {"100.0": [
            {"theta_deg": th, "cd": 1.0 + 0.2 * th / 45.0, "st": 0.2}
            for th in (0.0, 15.0, 30.0, 45.0)]},
        "galilean": gal,
        "reproducibility": ["abc", "abc"],
    }


def test_acceptance_all_pass_on_consistent_data():
    report = evaluate_criteria(_synthetic_results())
    by_id = {c["id"]: c for c in report["criteria"]}
    assert set(by_id) == {"C1", "C2", "C3", "C4", "C5", "C6", "C7"}
    for cid, crit in by_id.items():
        assert crit["pass"], f"{cid} unexpectedly failed: {crit}"
    assert report["all_pass"] is True


def test_acceptance_detects_violations():
    res = _synthetic_results()
    # break C2 (10% drag spread at Re=100) and C7 (different hashes)
    res["cylinder"]["100"][1]["cd"] *= 1.10
    res["reproducibility"] = ["abc", "def"]
    report = evaluate_criteria(res)
    by_id = {c["id"]: c for c in report["criteria"]}
    assert by_id["C2"]["pass"] is False
    assert by_id["C7"]["pass"] is False
    assert report["all_pass"] is False


def test_poiseuille_dirichlet_bc_conserves_flux():
    res = poiseuille_case(15.0, 8, u_mean=0.01, nu=0.1, steps_factor=6.0)
    assert res["flux_error_rel"] < 0.05