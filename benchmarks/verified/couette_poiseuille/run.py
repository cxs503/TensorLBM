#!/usr/bin/env python3
"""Couette-Poiseuille mixed-driven channel flow (D2Q9 BGK) -- analytic validation.

Physics
-------
Flow between two parallel plates where (a) a constant body force (equivalently a
constant pressure gradient) drives a Poiseuille parabola and (b) the top plate
slides with velocity ``u_wall`` and drags a Couette shear.  The steady
incompressible solution of

    nu * u''(y) + g = 0,     u(0) = 0,     u(H) = u_wall

superposes the two classical profiles:

    u(y)  = g/(2*nu) * y * (H - y) + u_wall * y / H
    y_v   = H/2 + nu*u_wall/(g*H)          (velocity extremum; off-centre)
    ubar  = g*H^2/(12*nu) + u_wall/2       (cross-section mean)

Because the peak is *off-centre*, the case exercises the Couette/Poiseuille
coupling and not just one profile shape.  ``g = 0`` degenerates to pure Couette
(linear), ``u_wall = 0`` to pure Poiseuille (centred parabola).

Numerics -- library primitives only, no correction factors, extrap: none
-----------------------------------------------------------------------
* D2Q9 + BGK single relaxation: ``tensorlbm.d2q9.equilibrium`` /
  ``macroscopic``, ``tensorlbm.solver.stream``.
* Body force via the second-order Guo (2002) scheme: half-force velocity shift
  ``u* = u + F/(2*rho)`` in the equilibrium plus the discrete source
  ``S_i = (1 - 1/(2*tau)) * w_i * [3*(c_ix - ux) + 9*(c_i.u)*c_ix] * F_x``.
* Walls: pre-streaming half-way bounce-back through two ghost rows, so the
  no-slip surfaces sit exactly at ``y = 0.5`` and ``y = ny - 1.5`` and the
  effective height is ``H = ny - 2``.  The top ghost row also carries the Ladd
  momentum transfer ``-2*w_i*rho*(c_i.u_wall)/cs2`` for the sliding plate.
* Streamwise periodicity (no inlet/outlet closure error), so the comparison is
  a pure wall-normal 1-D validation.

Every reported number comes from the simulation; the analytic profile is
evaluated at the *same* half-way wall positions.  The script exits non-zero if a
tutorial gate (see ``TOL_*`` below) is violated.

Usage
-----
    python run.py single out.json [--ny 34] [--tau 0.8] [--u-wall 0.02]
        [--accel G] [--steps 12000] [--checkpoint 1000]
    python run.py scan out_dir          # ny convergence + pure-Couette control
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

from tensorlbm.d2q9 import OPPOSITE, C, W, equilibrium, macroscopic
from tensorlbm.solver import stream

# D2Q9 direction groups (see tensorlbm.d2q9.C)
_Y_UP = (2, 5, 6)  # c_y = +1
_Y_DOWN = (4, 7, 8)  # c_y = -1

# Tutorial gates.  A case that misses one of these is a documented FAIL, not a
# number quietly pasted into a table.
TOL_PROFILE_L2 = 0.02  # L2 relative error of the whole wall-normal profile
TOL_PEAK_POS = 0.05  # relative error of the extremum location y_v/H
TOL_WALL_UC = 0.03  # relative error of u at the moving-wall node
TOL_UBAR = 0.02  # relative error of the cross-section mean velocity
# Fluid mass drift is float32 round-off: ~5e-9 per step, i.e. ~6e-5 over 12000
# steps.  The gate is therefore 0.1 % (a physics-level mass leak would be >>).
TOL_MASS = 1e-3
# Profile change over the last checkpoint, normalised by the peak velocity.
TOL_STEADY = 1e-3


def _c_dev(device: torch.device) -> torch.Tensor:
    return C.to(device=device, dtype=torch.float32)


def _w_dev(device: torch.device) -> torch.Tensor:
    return W.to(device=device, dtype=torch.float32)


def make_state(ny: int, nx: int, device: torch.device) -> torch.Tensor:
    """Rest state: rho = 1, u = 0 (kinematic normalisation: nu = (tau-0.5)/3)."""
    rho = torch.ones((ny, nx), device=device)
    zero = torch.zeros_like(rho)
    return equilibrium(rho, zero, zero)


def guo_source(
    ux: torch.Tensor,
    accel: float,
    tau: float,
    device: torch.device,
) -> torch.Tensor:
    """Guo (2002) discrete force source for a unit-density, x-only body force."""
    c = _c_dev(device)
    w = _w_dev(device).view(9, 1, 1)
    cx = c[:, 0].view(9, 1, 1)
    ux_b = ux.unsqueeze(0)
    cdotu = cx * ux_b  # uy == 0 for this wall-normal family of flows
    return (1.0 - 1.0 / (2.0 * tau)) * w * (3.0 * (cx - ux_b) + 9.0 * cdotu * cx) * accel


def apply_bounce_back(
    f: torch.Tensor,
    rho: torch.Tensor,
    u_wall: float,
    device: torch.device,
) -> None:
    """Pre-streaming half-way bounce-back on the two ghost rows (in place).

    ``f`` must already be post-collision.  Ghost row 0 receives the reflection
    of the populations that fluid row 1 is about to push into the bottom wall
    (no-slip); ghost row ``ny-1`` does the same for the sliding top wall, with
    the Ladd momentum transfer added.
    """
    c = _c_dev(device)
    w = _w_dev(device)
    cs2 = 1.0 / 3.0
    ny = f.shape[1]

    f[:, 0, :] = 0.0
    f[:, ny - 1, :] = 0.0

    for i in _Y_DOWN:  # bottom wall, u_w = 0
        f[OPPOSITE[i], 0, :] = f[i, 1, :]

    for i in _Y_UP:  # top wall, u_w = (u_wall, 0)
        j = OPPOSITE[i]
        cu = float(c[i, 0]) * u_wall
        f[j, ny - 1, :] = f[i, ny - 2, :] - 2.0 * float(w[i]) * rho[ny - 2, :] * cu / cs2


def run_case(
    ny: int = 34,
    nx: int = 8,
    tau: float = 0.8,
    u_wall: float = 0.02,
    accel: float | None = None,
    steps: int = 12000,
    checkpoint: int = 1000,
    device: str = "cpu",
    return_field: bool = False,
) -> dict:
    """Run one grid, return metrics, profiles and gate verdicts.

    ``accel=None`` selects the *self-similar* drive ``g = 4*nu*u_wall/H^2``,
    which pins the analytic extremum at ``y_v/H = 0.75`` for every grid, so a
    grid-refinement study really refines one problem.  Pass ``accel=0.0`` for
    the pure-Couette control or an explicit value for a fixed pressure gradient.

    ``return_field=True`` also returns the final ``u_x`` field (list of rows) so
    the demo script can plot it without re-running the solver.
    """
    if ny < 6:
        raise ValueError("need ny >= 6 (two ghost rows plus fluid)")
    if nx < 1:
        raise ValueError("need nx >= 1")
    if tau <= 0.5:
        raise ValueError("tau <= 0.5 gives a non-positive viscosity")

    dev = torch.device(device)
    torch.manual_seed(0)
    nu = (tau - 0.5) / 3.0
    H = float(ny - 2)  # wall-to-wall distance implied by the half-way bounce-back
    if accel is None:
        # self-similar drive: y_v/H = 1/2 + nu*u_wall/(g*H^2) = 3/4 for all ny
        accel = 4.0 * nu * u_wall / (H * H)
    fluid = slice(1, ny - 1)  # ghost rows are buffers, not fluid mass

    f = make_state(ny, nx, dev)
    mass0 = float(f[:, fluid, :].sum())
    rho, ux, uy = macroscopic(f)

    steady = float("inf")
    u_prev: torch.Tensor | None = None
    t0 = time.perf_counter()
    for step in range(steps):
        rho, ux, uy = macroscopic(f)
        ux_star = ux + 0.5 * accel / rho  # Guo half-force velocity shift
        feq = equilibrium(rho, ux_star, uy)
        f = f + (feq - f) / tau + guo_source(ux_star, accel, tau, dev)
        apply_bounce_back(f, rho, u_wall, dev)
        f = stream(f)
        if checkpoint > 0 and (step + 1) % checkpoint == 0:
            u_now = ux[fluid, :].mean(dim=1).to(torch.float64)
            if u_prev is not None:
                steady = float((u_now - u_prev).abs().max())
            u_prev = u_now
    walltime = time.perf_counter() - t0

    # --- post-processing -----------------------------------------------------
    rho, ux, uy = macroscopic(f)
    mass1 = float(f[:, fluid, :].sum())
    mass_drift = abs(mass1 - mass0) / mass0

    rows = torch.arange(1, ny - 1, device=dev)
    y_star = rows.to(torch.float64) - 0.5  # walls at y = 0.5 and y = ny - 1.5
    u_num = ux[fluid, :].mean(dim=1).to(torch.float64)

    g = accel
    u_exact = (g / (2.0 * nu)) * y_star * (H - y_star) + u_wall * y_star / H

    denom = float(torch.linalg.vector_norm(u_exact))
    profile_l2 = (
        float(torch.linalg.vector_norm(u_num - u_exact)) / denom if denom > 0.0 else 0.0
    )
    profile_max_abs = float((u_num - u_exact).abs().max())
    u_scale = max(float(u_exact.abs().max()), 1e-12)
    profile_max_rel = profile_max_abs / u_scale

    # Extremum: discrete peak row vs analytic vertex (clipped to the channel).
    y_peak = float(y_star[int(torch.argmax(u_num))])
    if g > 0.0:
        y_v_exact = H / 2.0 + nu * u_wall / (g * H)
    else:
        y_v_exact = H  # pure Couette: maximum at the sliding plate
    y_v_exact = min(max(y_v_exact, 0.0), H)
    peak_pos_err = abs(y_peak - y_v_exact) / H

    u_wall_node = float(u_num[-1])
    u_wall_node_exact = float(u_exact[-1])
    moving_wall_uc_err = abs(u_wall_node - u_wall_node_exact) / max(abs(u_wall_node_exact), 1e-12)

    u_bottom_node = float(u_num[0])
    bottom_node_exact = float(u_exact[0])
    bottom_node_rel_err = abs(u_bottom_node - bottom_node_exact) / u_scale

    ubar = float(u_num.mean())
    ubar_exact = g * H * H / (12.0 * nu) + 0.5 * u_wall
    ubar_err = abs(ubar - ubar_exact) / max(abs(ubar_exact), 1e-12)

    steady_rel = steady / u_scale if math.isfinite(steady) else float("inf")
    finite = bool(torch.isfinite(f).all())

    checks = {
        "profile_l2": {"value": profile_l2, "tol": TOL_PROFILE_L2, "ok": profile_l2 <= TOL_PROFILE_L2},
        "peak_position": {"value": peak_pos_err, "tol": TOL_PEAK_POS, "ok": peak_pos_err <= TOL_PEAK_POS},
        "moving_wall_uc": {
            "value": moving_wall_uc_err,
            "tol": TOL_WALL_UC,
            "ok": moving_wall_uc_err <= TOL_WALL_UC,
        },
        "mean_velocity": {"value": ubar_err, "tol": TOL_UBAR, "ok": ubar_err <= TOL_UBAR},
        "mass_drift": {"value": mass_drift, "tol": TOL_MASS, "ok": mass_drift <= TOL_MASS},
        "steady_state": {"value": steady_rel, "tol": TOL_STEADY, "ok": steady_rel <= TOL_STEADY},
        "finite": {"value": finite, "tol": True, "ok": finite},
    }

    return {
        "case": "couette_poiseuille",
        "params": {
            "ny": ny,
            "nx": nx,
            "tau": tau,
            "nu": nu,
            "u_wall": u_wall,
            "accel": accel,
            "steps": steps,
            "checkpoint": checkpoint,
            "H": H,
            "Re": ubar_exact * H / nu,
            "Ma": float(u_num.max()) / math.sqrt(1.0 / 3.0),
        },
        "metrics": {
            "profile_l2": profile_l2,
            "profile_max_abs": profile_max_abs,
            "profile_max_rel": profile_max_rel,
            "peak_y_star": y_peak,
            "peak_y_star_exact": y_v_exact,
            "peak_pos_rel_err": peak_pos_err,
            "u_wall_node": u_wall_node,
            "u_wall_node_exact": u_wall_node_exact,
            "moving_wall_uc_rel_err": moving_wall_uc_err,
            "u_bottom_node": u_bottom_node,
            "u_bottom_node_exact": bottom_node_exact,
            "bottom_node_rel_err": bottom_node_rel_err,
            "ubar": ubar,
            "ubar_exact": ubar_exact,
            "ubar_rel_err": ubar_err,
            "mass_drift": mass_drift,
            "steady_change_rel": steady_rel,
            "walltime_s": walltime,
        },
        "profile": {
            "y_star": [float(v) for v in y_star],
            "u_num": [float(v) for v in u_num],
            "u_exact": [float(v) for v in u_exact],
        },
        **(
            {"field_ux": [[float(v) for v in row] for row in ux.cpu()]}
            if return_field
            else {}
        ),
        "checks": checks,
        "all_ok": all(bool(c["ok"]) for c in checks.values()),
    }


def _print_summary(res: dict) -> None:
    p, m = res["params"], res["metrics"]
    print(
        f"ny={p['ny']} nx={p['nx']} tau={p['tau']} nu={p['nu']:.4f} u_wall={p['u_wall']} "
        f"g={p['accel']:.6g} steps={p['steps']} Re={p['Re']:.1f} Ma={p['Ma']:.4f}"
    )
    print(f"profile L2 rel err    = {m['profile_l2']:.6f}  (tol {TOL_PROFILE_L2})")
    print(f"profile max |du|      = {m['profile_max_abs']:.3e}  ({m['profile_max_rel']:.4%} of peak)")
    print(f"peak y*/H rel err     = {m['peak_pos_rel_err']:.6f}  (tol {TOL_PEAK_POS})")
    print(f"moving-wall u rel err = {m['moving_wall_uc_rel_err']:.6f}  (tol {TOL_WALL_UC})")
    print(f"mean velocity rel err = {m['ubar_rel_err']:.6f}  (tol {TOL_UBAR})")
    print(f"fluid mass drift      = {m['mass_drift']:.3e}  (tol {TOL_MASS})")
    print(f"steady change / peak  = {m['steady_change_rel']:.3e}  (tol {TOL_STEADY})")
    print(f"walltime              = {m['walltime_s']:.2f} s")
    print(f"all_ok                = {res['all_ok']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--ny", type=int, default=34)
        p.add_argument("--nx", type=int, default=8)
        p.add_argument("--tau", type=float, default=0.8)
        p.add_argument("--u-wall", type=float, default=0.02)
        p.add_argument(
            "--accel",
            type=float,
            default=None,
            help="body force; default self-similar 4*nu*u_wall/H^2 (y_v/H = 0.75)",
        )
        p.add_argument("--steps", type=int, default=12000)
        p.add_argument("--checkpoint", type=int, default=1000)
        p.add_argument("--device", default="cpu")

    p_single = sub.add_parser("single", help="run one grid and write a JSON result")
    p_single.add_argument("out", type=Path)
    add_common(p_single)

    p_scan = sub.add_parser("scan", help="ny convergence + pure-Couette control")
    p_scan.add_argument("out", type=Path)
    add_common(p_scan)

    args = parser.parse_args(argv)
    kw = dict(
        nx=args.nx,
        tau=args.tau,
        u_wall=args.u_wall,
        accel=args.accel,
        steps=args.steps,
        checkpoint=args.checkpoint,
        device=args.device,
    )

    if args.cmd == "single":
        res = run_case(ny=args.ny, **kw)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2))
        _print_summary(res)
        return 0 if res["all_ok"] else 1

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    for ny in (18, 34, 50):
        report[f"mixed_ny{ny}"] = run_case(ny=ny, **kw)
    report["couette_control"] = run_case(
        ny=args.ny, **{k: v for k, v in kw.items() if k != "accel"}, accel=0.0
    )
    (out / "scan.json").write_text(json.dumps(report, indent=2))
    for name, res in report.items():
        print(f"{name:18s} L2={res['metrics']['profile_l2']:.6f}  all_ok={res['all_ok']}")
    return 0 if all(r["all_ok"] for r in report.values()) else 1


if __name__ == "__main__":
    sys.exit(main())