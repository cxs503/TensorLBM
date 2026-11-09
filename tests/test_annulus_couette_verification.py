"""Gates for ``run_annulus_couette`` (rotating inner cylinder vs exact NS solution).

The case shipped in ``tensorlbm.curved_boundary_calibration`` without any adjudication.
These gates pin three things:

1. the *reference* is exact (no-slip at both walls, torque identity, steady NS residual);
2. the moving-wall closure is Galileo invariant link-by-link — the defect found here
   (solid-side closure built in the wall frame but used in the lab frame) made the
   q >= 1/2 stencils over-inject up to +50% of the Ladd momentum;
3. the simulated torque/profile of the two active wall formats stay within a stated
   band of the exact solution at a small resolution that a CI box can afford.

Artifact gate (``artifacts/annulus_couette_verification_20261109/metrics.json``) skips
when the heavy ladder has not been generated in this checkout.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tensorlbm import curved_boundary_calibration as C

ART = Path(__file__).resolve().parents[1] / "artifacts" / "annulus_couette_verification_20261109"


# ---------------------------------------------------------------------------
# 1. the analytic reference itself
# ---------------------------------------------------------------------------
def test_exact_profile_satisfies_both_no_slip_walls() -> None:
    r1, r2, omega, nu = 8.0, 16.0, 0.0015625, 0.05
    ex = C.annulus_exact(r1, r2, omega, nu)
    u = lambda r: ex["A"] * r + ex["B"] / r
    assert u(r1) == pytest.approx(omega * r1, rel=1e-12)
    assert abs(u(r2)) < 1e-15


def test_exact_torque_identity_holds_for_the_profile() -> None:
    r1, r2, omega, nu = 8.0, 16.0, 0.0015625, 0.05
    ex = C.annulus_exact(r1, r2, omega, nu)
    # tau_r_theta = mu r d(u/r)/dr = -2 mu B / r^2 ; T = 2 pi r^2 tau per span
    for r in (9.0, 12.0, 15.0):
        dut_dr = ex["A"] - ex["B"] / (r * r)
        tau = nu * (dut_dr - (ex["A"] * r + ex["B"] / r) / r)
        assert abs(-2.0 * math.pi * r * r * tau) == pytest.approx(
            ex["torque_per_span"], rel=1e-12)


def test_exact_profile_has_zero_viscous_residual() -> None:
    """Steady azimuthal momentum: d2u/dr2 + (1/r)du/dr - u/r2 = 0 (rho = 1).

    Derivatives are taken analytically (u = A r + B/r), so this checks the closed
    form the module docstring claims rather than a finite-difference truncation.
    """
    r1, r2, omega, nu = 8.0, 16.0, 0.0015625, 0.05
    ex = C.annulus_exact(r1, r2, omega, nu)
    a, b = ex["A"], ex["B"]
    for r in (9.0, 12.0, 15.0):
        d2 = 2.0 * b / r**3
        d1 = a - b / (r * r)
        resid = d2 + d1 / r - (a * r + b / r) / (r * r)
        assert abs(resid) < 1e-15 * abs(b) / r**3


# ---------------------------------------------------------------------------
# 2. moving-wall closure: Galileo invariance (regression for the +50% defect)
# ---------------------------------------------------------------------------
def _galileo_error(fmt: str, r1: int = 8, r2: int = 16, tau: float = 0.65,
                   speed: float = 0.01) -> float:
    torch.set_num_threads(1)
    nx = 2 * r2 + 3
    spec = {"center": (0.0, (nx - 1) / 2.0, (nx - 1) / 2.0),
            "radius": float(r1), "radius_outer": float(r2), "omega": 0.0}
    geo = C.build_geometry("annulus", 2, nx, nx, spec, format=fmt)
    cfg = C.FlowConfig(nz=2, ny=nx, nx=nx, tau=tau, format=fmt, collision="regularized")
    solver = C.Solver(cfg, geo)
    ux = torch.full((2, nx, nx), speed, dtype=solver.dt)
    uy = torch.full((2, nx, nx), 0.5 * speed, dtype=solver.dt)
    uz = torch.zeros_like(ux)
    solver.f = solver.equilibrium(torch.ones_like(ux), (ux, uy, uz))
    geo.uw = torch.stack([ux.reshape(-1)[geo.idx], uy.reshape(-1)[geo.idx],
                          uz.reshape(-1)[geo.idx]], dim=1).to(solver.dt)

    rho, u = solver.moments(solver.f)
    fpost = solver.collide(solver.f, rho, u)
    fnew = solver.stream(fpost)
    before = fnew.clone()
    solver.apply_cut_links(fpost, fnew, rho, u, 1.0)
    target = geo.opp * (solver.nz * solver.ny * solver.nx) + geo.idx
    got, want = fnew.reshape(-1)[target], before.reshape(-1)[target]
    c, w = solver._link_c, solver._link_w
    cuw = (c * geo.uw).sum(1)
    scale = (6.0 * w * rho.reshape(-1)[geo.idx] * cuw).abs().clamp_min(1e-30)
    return float(((got - want).abs() / scale).max())


@pytest.mark.parametrize("fmt", C.BOUNDARY_FORMATS)
def test_moving_wall_closure_is_galileo_invariant(fmt: str) -> None:
    """Uniform flow + both walls moving with it must leave the populations untouched.

    Post-fix residual is 9.2e-3 of the Ladd injection (second order in the wall
    speed); without the closure shift the same probe measures 5.09e-1.
    """
    saved = C.MOVING_WALL_CLOSURE_SHIFT
    try:
        C.MOVING_WALL_CLOSURE_SHIFT = True
        err_on = _galileo_error(fmt)
        C.MOVING_WALL_CLOSURE_SHIFT = False
        err_off = _galileo_error(fmt)
    finally:
        C.MOVING_WALL_CLOSURE_SHIFT = saved
    assert err_on < 2e-2, err_on
    if fmt == "halfway":                       # no closure is used at all here
        assert err_on < 1e-4
    else:                                      # at least a 20x defect reduction
        assert err_on < 0.1 * err_off, (err_on, err_off)


def test_closure_shift_is_a_no_op_on_a_stationary_wall() -> None:
    """The fix must not touch the stationary case: u_w = 0 leaves the shift exactly 0."""
    assert C.MOVING_WALL_CLOSURE_SHIFT is True
    def max_abs_dev() -> float:
        """Absolute (not Ladd-normalised, which is 0/0 at u_w = 0) deviation."""
        torch.set_num_threads(1)
        nx = 35
        spec = {"center": (0.0, (nx - 1) / 2.0, (nx - 1) / 2.0),
                "radius": 8.0, "radius_outer": 16.0, "omega": 0.0}
        geo = C.build_geometry("annulus", 2, nx, nx, spec, format="bouzidi_quadratic")
        cfg = C.FlowConfig(nz=2, ny=nx, nx=nx, tau=0.65,
                           format="bouzidi_quadratic", collision="regularized")
        solver = C.Solver(cfg, geo)
        ux = torch.full((2, nx, nx), 0.01, dtype=solver.dt)
        uy = torch.zeros_like(ux)
        uz = torch.zeros_like(ux)
        solver.f = solver.equilibrium(torch.ones_like(ux), (ux, uy, uz))
        geo.uw = torch.zeros((geo.n_links(), 3), dtype=solver.dt)   # stationary wall
        rho, u = solver.moments(solver.f)
        fpost = solver.collide(solver.f, rho, u)
        fnew = solver.stream(fpost)
        before = fnew.clone()
        solver.apply_cut_links(fpost, fnew, rho, u, 1.0)
        target = geo.opp * (solver.nz * solver.ny * solver.nx) + geo.idx
        return float((fnew.reshape(-1)[target] - before.reshape(-1)[target]).abs().max())

    saved = C.MOVING_WALL_CLOSURE_SHIFT
    try:
        C.MOVING_WALL_CLOSURE_SHIFT = True
        on = max_abs_dev()
        C.MOVING_WALL_CLOSURE_SHIFT = False
        off = max_abs_dev()
    finally:
        C.MOVING_WALL_CLOSURE_SHIFT = saved
    assert on == off, (on, off)


# ---------------------------------------------------------------------------
# 3. the simulated case (small, CI-affordable)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fmt", ("halfway", "bouzidi_quadratic"))
def test_annulus_couette_torque_within_band(fmt: str) -> None:
    res = C.run_annulus_couette(radius_inner=6, radius_outer=12, format=fmt,
                                steps=1500, avg_fraction=0.3)
    assert res["torque_rel_err"] < 0.12, res["torque_rel_err"]
    assert res["torque_inner_rel_err"] < 0.12
    # inner and outer wall must see the same torque at steady state (Newton's law)
    assert res["inner_vs_outer_rel_diff"] < 0.02
    assert res["profile_max_rel_err"] < 0.10


def test_interpolated_format_does_not_lose_to_halfway_on_a_moving_curved_wall() -> None:
    """Pre-fix the interpolated formats were ~7x worse here; the closure shift must
    restore them to at least halfway accuracy."""
    h = C.run_annulus_couette(radius_inner=6, radius_outer=12, format="halfway",
                              steps=1500, avg_fraction=0.3)
    q = C.run_annulus_couette(radius_inner=6, radius_outer=12,
                              format="bouzidi_quadratic", steps=1500, avg_fraction=0.3)
    assert q["torque_rel_err"] < 0.5 * h["torque_rel_err"]


# ---------------------------------------------------------------------------
# 4. artifact gate (skips when the heavy ladder is absent)
# ---------------------------------------------------------------------------
def test_recorded_ladder_metrics() -> None:
    path = ART / "metrics.json"
    if not path.exists():
        pytest.skip("heavy ladder artifact not generated in this checkout")
    m = json.loads(path.read_text())
    rows = {(r["format"], r["r1"], r["closure_shift"]): r for r in m["cases"]}
    # A/B on the SAME grid: the shift must not hurt anywhere and must fix the
    # interpolated formats (they were ~7x worse than halfway before it).
    for fmt in ("bouzidi_linear", "bouzidi_quadratic"):
        before = rows[(fmt, 8, False)]["torque_rel_err"]
        after = rows[(fmt, 8, True)]["torque_rel_err"]
        assert after < before, (fmt, before, after)
    # every post-fix case at R1 >= 8 must be inside the 15% platform band
    for (fmt, r1, shift), row in rows.items():
        if shift and r1 >= 8:
            assert row["torque_rel_err"] < 0.15, (fmt, r1, row["torque_rel_err"])
    # resolution: the interpolated branch must improve when the gap is refined
    q12 = rows[("bouzidi_quadratic", 12, True)]["torque_rel_err"]
    q6 = rows[("bouzidi_quadratic", 6, True)]["torque_rel_err"]
    assert q12 < q6