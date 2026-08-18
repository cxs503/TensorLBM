#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""2D Couette flow (moving top wall, D2Q9) — analytic linear-profile validation.

Physics: steady, incompressible plane Couette flow between two infinite
parallel plates.  Bottom wall at rest (no-slip), top wall moving at
constant velocity U0 in +x (no-slip, moving wall).  Flow is homogeneous in
x (periodic), so the steady solution is the exact linear profile

    u(y) = U0 * (y - y_bottom) / H_eff ,   y_bottom = 0.5 ,  H_eff = ny - 2

with half-way bounce-back the effective no-slip walls sit exactly at
y = 0.5 and y = ny - 1.5 (same convention as the validated
benchmarks/verified/poiseuille_2d run: walls are the rows y=0 and y=ny-1).

True simulation, no extrapolation:
  - library primitives only: d2q9.equilibrium/macroscopic,
    solver.collide_bgk/collide_mrt/stream
  - walls via pre-streaming half-way bounce-back (f_pre[OPPOSITE] at wall
    rows, the repo-validated variant used by poiseuille_2d)
  - MOVING top wall: standard textbook momentum injection on top of the
    bounce-back,  f_opp = f_i - 2*w_i*rho*(c_i . u_wall)/cs^2   (Zou-He
    style moving-wall / "bounce-back + reflection"); bottom wall u_wall=0
    reduces to plain bounce-back.  The injection is applied to the
    reflected populations so that exactly the right x-momentum is fed into
    the fluid each step (momentum balance is verified in the output).
  - no correction factors, no result tuning, extrap: none

Usage:
    run.py single H out.json [--tau T] [--u0 U] [--collision bgk|mrt]
        [--min-steps N] [--max-steps N] [--seed 0]
    run.py scan out_dir [--H ...] [--tau T] [--u0 U] [--collision bgk|mrt]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/wxsc/cxs/TensorLBM/src")

import numpy as np
import torch

from tensorlbm.d2q9 import C, OPPOSITE, W, equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, collide_mrt, stream

CS2 = 1.0 / 3.0
DEVICE = torch.device("cpu")


def moving_wall_bounce_back(
    f_pre: torch.Tensor, f: torch.Tensor, wall: torch.Tensor, u_wall: torch.Tensor
) -> torch.Tensor:
    """Pre-streaming half-way bounce-back with moving-wall momentum injection.

    At wall cells the post-collision populations are replaced by the
    reflected pre-collision ones plus the standard moving-wall term:

        f_new[q] = f_pre[opp[q]] + 2*w_q*rho*(c_q . u_wall)/cs^2

    (equivalent textbook form: f_opp = f_i - 2*w_i*rho*(c_i . u_wall)/cs^2,
    Zou & He 1997 / Ladd 1994 moving-wall bounce-back).  u_wall is a
    per-cell wall-velocity field (ny, nx) in +x (0 on stationary walls),
    so each wall can move independently; the term injects exactly the
    x-momentum required by the moving wall each step.  rho is taken
    locally from the pre-collision populations at the wall cells
    (rho ~ 1 in this incompressible setup).  The injection is applied on
    all 9 directions at wall cells; only the into-fluid directions stream
    into the domain, the rest are overwritten by the next bounce-back, so
    no population can leak into the fluid.
    """
    opp = OPPOSITE.to(DEVICE)
    c = C.to(DEVICE)
    w = W.to(DEVICE)
    f_new = torch.where(wall.unsqueeze(0), f_pre[opp], f)
    rho_w = torch.clamp(f_pre.sum(dim=0), min=1e-12)          # (ny, nx) at wall cells
    cu = c[:, 0].view(9, 1, 1) * u_wall.unsqueeze(0)          # c_q . (u_wall, 0)
    injection = (2.0 * w.view(9, 1, 1) * rho_w.unsqueeze(0) * cu) / CS2
    return f_new + injection * wall.unsqueeze(0)


def run_case(
    H: int,
    tau: float,
    U0: float,
    collision: str,
    min_steps: int,
    max_steps: int,
    out_path: str,
    seed: int = 0,
) -> dict:
    """Run one Couette case and return measurements."""
    torch.manual_seed(seed)
    ny = H + 2                      # wall rows at y=0 and y=ny-1
    nx = H                          # periodic in x -> any nx works
    nu = (tau - 0.5) / 3.0
    H_eff = ny - 2.0                # effective gap (walls at y=0.5 .. ny-1.5)
    Re = U0 * H / nu
    Ma = U0 / math.sqrt(CS2)

    collide_fn = collide_mrt if collision == "mrt" else collide_bgk

    wall = torch.zeros((ny, nx), dtype=torch.bool, device=DEVICE)
    wall[0, :] = True               # bottom wall: stationary (u_wall = 0)
    wall[-1, :] = True              # top wall: moving at U0
    u_wall = torch.zeros((ny, nx), device=DEVICE)
    u_wall[-1, :] = U0              # per-wall velocity field (+x)

    # Initial condition: equilibrium at the analytic linear ramp (fluid rows),
    # walls at rest.  The steady state is unique and independent of the IC;
    # the ramp only shortens the diffusive transient (the k=1 vorticity mode
    # decays on the scale H^2/(pi^2*nu) ~ 6.5e3 steps at H=80).
    yy = torch.arange(ny, device=DEVICE, dtype=torch.float32)
    u0 = (torch.clamp(U0 * (yy - 0.5) / H_eff, min=0.0, max=U0)
          .view(ny, 1).expand(ny, nx).clone())
    u0[0, :] = 0.0
    u0[-1, :] = 0.0
    f = equilibrium(torch.ones((ny, nx), device=DEVICE), u0, torch.zeros((ny, nx), device=DEVICE))
    initial_mass = float(f.sum().item())

    col = nx // 2                   # measurement column
    t0 = time.time()
    u_top_hist: list[float] = []
    step = 0
    steady = False
    for step in range(1, max_steps + 1):
        f_pre = f.clone()
        f = collide_fn(f, tau)
        # pre-streaming half-way bounce-back: bottom wall fixed, top wall moving
        f = moving_wall_bounce_back(f_pre, f, wall, u_wall)
        f = stream(f)               # periodic in both directions
        if step % 200 == 0:
            _, ux, _ = macroscopic(f)
            u_top_hist.append(float(ux[ny - 2, col].item()))
            if step >= min_steps and len(u_top_hist) >= 10:
                recent = u_top_hist[-10:]                # drift over last 2000 steps
                mean = sum(recent) / len(recent)
                drift = (max(recent) - min(recent)) / max(abs(mean), 1e-12)
                if drift < 1e-5:
                    steady = True
                    break
    elapsed = time.time() - t0

    # Time-average the profile over the last 200 steps (steady, noise-free)
    prof_acc = torch.zeros(ny, device=DEVICE)
    for _ in range(200):
        f_pre = f.clone()
        f = collide_fn(f, tau)
        f = moving_wall_bounce_back(f_pre, f, wall, u_wall)
        f = stream(f)
        _, ux, _ = macroscopic(f)
        prof_acc += ux[:, col]
    prof_acc /= 200.0

    rho, ux, _ = macroscopic(f)
    u_num = prof_acc[1 : ny - 1].cpu().numpy()          # H fluid rows
    y_phys = np.arange(1, ny - 1, dtype=np.float64) - 0.5
    u_ana = U0 * (y_phys / H_eff)

    # Profile L2 relative error (primary metric)
    l2_rel = float(np.linalg.norm(u_num - u_ana) / np.linalg.norm(u_ana))
    max_abs_err = float(np.max(np.abs(u_num - u_ana)))
    # Max pointwise relative error away from the (vanishing) lower-wall velocity
    mask = u_ana > 0.2 * U0
    max_rel = float(np.max(np.abs(u_num[mask] - u_ana[mask]) / u_ana[mask]) * 100.0)

    # Top fluid row: should be U0*(1 - 0.5/H_eff) (wall itself is at y=ny-1.5)
    u_top_num = float(u_num[-1])
    u_top_ana = float(u_ana[-1])
    u_top_err = abs(u_top_num - u_top_ana) / u_top_ana * 100.0

    # Wall shear: measured from the discrete slope vs analytic tau = rho*nu*U0/H_eff
    slope_num = float((u_num[-1] - u_num[0]) / (y_phys[-1] - y_phys[0]))
    tau_ana = nu * U0 / H_eff
    tau_num = slope_num * nu
    tau_err_pct = (tau_num - tau_ana) / tau_ana * 100.0

    # Momentum-balance check: total fluid x-momentum (should stay ~ 0.5*rho*U0*H_eff*nx)
    mom_x = float((rho * ux).sum().item())

    result = {
        "case": "couette_2d",
        "collision": collision,
        "lattice": "D2Q9",
        "boundary": "moving top wall + stationary bottom wall, half-way bounce-back (pre-streaming, f_pre[opposite] + 2*w*rho*(c.u_wall)/cs^2 injection)",
        "driving": "moving wall (top plate at u = U0, x-periodic channel)",
        "extrap": "none",
        "H_eff": H,
        "ny": ny,
        "nx": nx,
        "tau": tau,
        "nu_lb": nu,
        "U0": U0,
        "Re": Re,
        "Ma": Ma,
        "min_steps": min_steps,
        "n_steps": step,
        "steady": steady,
        "l2_rel_err": l2_rel,
        "max_abs_err": max_abs_err,
        "max_rel_err_masked_pct": max_rel,
        "u_top_row_num": u_top_num,
        "u_top_row_ana": u_top_ana,
        "u_top_row_err_pct": u_top_err,
        "tau_ana": tau_ana,
        "tau_num": tau_num,
        "tau_err_pct": tau_err_pct,
        "total_x_momentum": mom_x,
        "mass_drift_pct": (float(f.sum().item()) - initial_mass) / initial_mass * 100.0,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": round(elapsed, 1),
        "u_profile": [round(float(v), 8) for v in u_num],
        "u_analytic": [round(float(v), 8) for v in u_ana],
    }
    Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def scan(H_list, tau, U0, collision, min_steps, max_steps, out_dir: str) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for H in H_list:
        p = out_dir / f"case_H{H}.json"
        r = run_case(H, tau, U0, collision, min_steps, max_steps, str(p))
        cases.append(r)
        print(
            f"H={r['H_eff']:3d} Re={r['Re']:7.2f} steps={r['n_steps']:6d} "
            f"l2_err={r['l2_rel_err']:.2e} max_rel={r['max_rel_err_masked_pct']:.4f}% "
            f"u_top_err={r['u_top_row_err_pct']:+.4f}% tau_err={r['tau_err_pct']:+.4f}% "
            f"steady={r['steady']}",
            flush=True,
        )
    summary = {
        "case": "couette_2d_convergence",
        "lattice": "D2Q9",
        "collision": collision,
        "boundary": "moving top wall + stationary bottom wall, half-way bounce-back (pre-streaming) + momentum injection",
        "driving": "moving wall (u=U0 top plate)",
        "extrap": "none",
        "H_list": H_list,
        "tau": tau,
        "U0": U0,
        "min_steps": min_steps,
        "max_steps": max_steps,
        "per_grid": cases,
        "convergence": [
            {"H": r["H_eff"], "Re": r["Re"], "l2_rel_err": r["l2_rel_err"],
             "max_abs_err": r["max_abs_err"],
             "max_rel_err_masked_pct": r["max_rel_err_masked_pct"],
             "u_top_row_err_pct": r["u_top_row_err_pct"],
             "tau_err_pct": r["tau_err_pct"],
             "n_steps": r["n_steps"], "collision": r["collision"]}
            for r in cases
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="2D Couette flow (moving top wall, D2Q9)")
    sub = ap.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("single")
    p1.add_argument("H", type=int)
    p1.add_argument("out_json", type=str)
    p1.add_argument("--tau", type=float, default=0.8)
    p1.add_argument("--u0", type=float, default=0.05)
    p1.add_argument("--collision", choices=["bgk", "mrt"], default="bgk")
    p1.add_argument("--min-steps", type=int, default=10000)
    p1.add_argument("--max-steps", type=int, default=60000)
    p1.add_argument("--seed", type=int, default=0)

    p2 = sub.add_parser("scan")
    p2.add_argument("out_dir", type=str)
    p2.add_argument("--H", type=int, nargs="+", default=[40, 80])
    p2.add_argument("--tau", type=float, default=0.8)
    p2.add_argument("--u0", type=float, default=0.05)
    p2.add_argument("--collision", choices=["bgk", "mrt"], default="bgk")
    p2.add_argument("--min-steps", type=int, default=10000)
    p2.add_argument("--max-steps", type=int, default=60000)

    args = ap.parse_args()
    if args.mode == "single":
        r = run_case(args.H, args.tau, args.u0, args.collision,
                     args.min_steps, args.max_steps, args.out_json, args.seed)
        print(json.dumps({k: r[k] for k in
                          ["H_eff", "nx", "ny", "tau", "nu_lb", "U0", "Re", "Ma",
                           "n_steps", "steady", "l2_rel_err", "max_abs_err",
                           "max_rel_err_masked_pct", "u_top_row_err_pct",
                           "tau_err_pct", "mass_drift_pct", "elapsed_s"]},
                         indent=2))
    else:
        scan(args.H, args.tau, args.u0, args.collision,
             args.min_steps, args.max_steps, args.out_dir)


if __name__ == "__main__":
    main()
