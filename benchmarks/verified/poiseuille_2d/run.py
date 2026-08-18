#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""B13: 2D Poiseuille flow (pressure-difference driven, D2Q9) — analytic validation.

Physics: fully-developed 2D laminar channel flow between parallel plates,
driven by a pressure difference imposed with Zou-He pressure boundary
conditions (rho_in at x=0, rho_out at x=nx-1), top/bottom walls via
half-way bounce-back (pre-streaming variant, consistent with the repo's
validated 3D internal-flow benchmarks, see mem_vs_pf_worker.py /
friction_test3_poiseuille series).

Analytic solution (steady, fully developed, no-slip walls):
    u(y) = dp/(2*nu*L) * (y^2 - H*y) = 4*u_max*(y/H)*(1 - y/H)
    u_max = dp*H^2/(8*nu*L),   dp = (rho_in - rho_out)*cs2,  cs2 = 1/3
    nu = (tau - 0.5)/3,        L = nx (channel length)
    H = ny - 2 (effective wall-to-wall height; pre-streaming half-way BB
    places the no-slip walls exactly at y = 0.5 and y = ny - 1.5)

True simulation, no extrapolation:
  - library primitives only: d2q9.equilibrium/macroscopic,
    solver.collide_bgk/collide_mrt/stream,
    boundaries.zou_he_outlet_pressure
  - standard textbook Zou-He pressure INLET reconstruction (~10 lines,
    mirror of the outlet; Zou & He 1997, Phys. Fluids 9, 1591)
  - pre-streaming half-way bounce-back (f_pre[OPPOSITE] at wall rows)
  - no correction factors, no result tuning, extrap: none

Usage:
    run.py single H out.json [--tau T] [--umax U] [--collision bgk|mrt]
        [--min-steps N] [--max-steps N] [--seed 0]
    run.py scan out_dir [--max-steps N] [--include-re80]
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

from tensorlbm.d2q9 import OPPOSITE, equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, collide_mrt, stream
from tensorlbm.boundaries import zou_he_outlet_pressure

CS2 = 1.0 / 3.0
DEVICE = torch.device("cpu")


def zou_he_inlet_pressure(f: torch.Tensor, rho_in: float) -> torch.Tensor:
    """Zou-He pressure (density) inlet at the left column (x=0).

    Prescribes rho = rho_in and uy = 0; recovers ux and the unknown
    in-flowing populations f1, f5, f8 from the known ones (mirror image of
    the pressure outlet; standard, Zou & He 1997).
    """
    f0, f2, f3, f4, f6, f7 = f[0, :, 0], f[2, :, 0], f[3, :, 0], f[4, :, 0], f[6, :, 0], f[7, :, 0]
    rho = torch.full_like(f0, rho_in)
    ux = 1.0 - (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / rho
    f_new = f.clone()
    f_new[1, :, 0] = f3 + (2.0 / 3.0) * rho * ux
    f_new[5, :, 0] = f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * ux
    f_new[8, :, 0] = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * ux
    return f_new


def run_case(
    H: int,
    tau: float,
    u_max_target: float,
    collision: str,
    min_steps: int,
    max_steps: int,
    out_path: str,
    seed: int = 0,
) -> dict:
    """Run one pressure-driven 2D Poiseuille case and return measurements."""
    torch.manual_seed(seed)
    ny = H + 2                      # wall rows at y=0 and y=ny-1
    nx = 3 * H                      # L = 3H
    nu = (tau - 0.5) / 3.0
    # u_max = dp*H^2/(8*nu*L), dp = d_rho*cs2, L = nx = 3H  =>  d_rho = 24*nu*u_max/(cs2*H)
    delta_rho = 24.0 * nu * u_max_target / (CS2 * H)
    rho_in = 1.0 + delta_rho / 2.0
    rho_out = 1.0 - delta_rho / 2.0
    u_max_ana = delta_rho * CS2 * H * H / (8.0 * nu * nx)
    u_mean = (2.0 / 3.0) * u_max_ana
    Re = u_mean * H / nu
    Ma = u_max_ana / math.sqrt(CS2)

    collide_fn = collide_mrt if collision == "mrt" else collide_bgk

    # Walls: top/bottom rows; pre-streaming half-way bounce-back
    wall = torch.zeros((ny, nx), dtype=torch.bool, device=DEVICE)
    wall[0, :] = True
    wall[-1, :] = True

    # Initial condition: rest + linear density ramp rho_in -> rho_out
    xx = torch.arange(nx, device=DEVICE, dtype=torch.float32)
    rho0 = (rho_out + (rho_in - rho_out) * (1.0 - xx / (nx - 1))).view(1, nx).expand(ny, nx)
    f = equilibrium(rho0, torch.zeros((ny, nx), device=DEVICE), torch.zeros((ny, nx), device=DEVICE))
    initial_mass = float(f.sum().item())

    col = nx // 2                   # measurement column (mid-channel)
    t0 = time.time()
    umax_hist: list[float] = []
    step = 0
    steady = False
    for step in range(1, max_steps + 1):
        f_pre = f.clone()
        f = collide_fn(f, tau)
        # pre-streaming half-way bounce-back at wall rows (repo-validated variant)
        f = torch.where(wall.unsqueeze(0), f_pre[OPPOSITE.to(DEVICE)], f)
        f = stream(f)               # periodic gather; boundary columns overwritten below
        f = zou_he_inlet_pressure(f, rho_in)
        f = zou_he_outlet_pressure(f, rho_out)
        if step % 200 == 0:
            _, ux, _ = macroscopic(f)
            umax_hist.append(float(ux[:, col].max().item()))
            if step >= min_steps and len(umax_hist) >= 10:
                recent = umax_hist[-10:]                # drift over last 2000 steps
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
        f = torch.where(wall.unsqueeze(0), f_pre[OPPOSITE.to(DEVICE)], f)
        f = stream(f)
        f = zou_he_inlet_pressure(f, rho_in)
        f = zou_he_outlet_pressure(f, rho_out)
        _, ux, _ = macroscopic(f)
        prof_acc += ux[:, col]
    prof_acc /= 200.0

    rho, ux, _ = macroscopic(f)
    u_num = prof_acc[1 : ny - 1].cpu().numpy()          # H fluid rows
    y_phys = np.arange(1, ny - 1, dtype=np.float64) - 0.5
    u_ana = 4.0 * u_max_ana * (y_phys / H) * (1.0 - y_phys / H)

    # Profile L2 relative error (primary metric)
    l2_rel = float(np.linalg.norm(u_num - u_ana) / np.linalg.norm(u_ana))

    # Max-velocity relative error (compare at the row that carries the peak)
    imax = int(np.argmax(u_num))
    u_max_num = float(u_num[imax])
    u_max_ana_at_row = 4.0 * u_max_ana * (y_phys[imax] / H) * (1.0 - y_phys[imax] / H)
    u_max_err = abs(u_max_num - u_max_ana_at_row) / u_max_ana_at_row * 100.0

    # Max pointwise relative error in the central region (|u| > 20% of u_max)
    mask = u_ana > 0.2 * u_max_ana
    max_rel = float(np.max(np.abs(u_num[mask] - u_ana[mask]) / u_ana[mask]) * 100.0)

    # Pressure-difference diagnostics: measured rho along x at mid height
    rho_x = rho[ny // 2, :].cpu().numpy().astype(np.float64)
    slope_meas = (rho_x[-1] - rho_x[0]) / (nx - 1)
    slope_nom = (rho_out - rho_in) / (nx - 1)
    lin = rho_x[0] + slope_meas * np.arange(nx)
    max_lin_dev = float(np.max(np.abs(rho_x - lin)))

    result = {
        "case": "B13_poiseuille_2d",
        "collision": collision,
        "lattice": "D2Q9",
        "boundary": "zou_he_pressure_inlet/outlet + half-way bounce-back (pre-streaming)",
        "driving": "pressure_difference (rho_in > rho_out, dp=(rho_in-rho_out)*cs2)",
        "H_eff": H,
        "ny": ny,
        "nx": nx,
        "L_over_H": float(nx / H),
        "tau": tau,
        "nu_lb": nu,
        "rho_in": rho_in,
        "rho_out": rho_out,
        "delta_rho": delta_rho,
        "u_max_ana": u_max_ana,
        "u_mean_ana": u_mean,
        "Re": Re,
        "Ma": Ma,
        "min_steps": min_steps,
        "n_steps": step,
        "steady": steady,
        "u_max_num": u_max_num,
        "u_max_ana_at_peak_row": u_max_ana_at_row,
        "u_max_err_pct": u_max_err,
        "l2_rel_err": l2_rel,
        "max_rel_err_central_pct": max_rel,
        "rho_slope_meas": slope_meas,
        "rho_slope_nominal": slope_nom,
        "rho_slope_ratio": slope_meas / slope_nom if slope_nom != 0 else float("nan"),
        "max_rho_lin_dev": max_lin_dev,
        "mass_drift_pct": (float(f.sum().item()) - initial_mass) / initial_mass * 100.0,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": round(elapsed, 1),
        "u_profile": [round(float(v), 8) for v in u_num],
        "u_analytic": [round(float(v), 8) for v in u_ana],
    }
    Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def scan(H_list, tau, u_max, collision, min_steps, max_steps, out_dir: str,
         include_re80: bool = False) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for H in H_list:
        p = out_dir / f"case_H{H}.json"
        r = run_case(H, tau, u_max, collision, min_steps, max_steps, str(p))
        cases.append(r)
        print(
            f"H={r['H_eff']:3d} Re={r['Re']:7.2f} steps={r['n_steps']:6d} "
            f"l2_err={r['l2_rel_err']:.5f} u_max_err={r['u_max_err_pct']:+.4f}% "
            f"max_rel={r['max_rel_err_central_pct']:.4f}% steady={r['steady']}",
            flush=True,
        )
    if include_re80:
        p = out_dir / "case_Re80_H60_mrt.json"
        r = run_case(60, 0.56, 0.04, "mrt", min_steps, max_steps, str(p))
        cases.append(r)
        print(
            f"H={r['H_eff']:3d} Re={r['Re']:7.2f} (MRT) steps={r['n_steps']:6d} "
            f"l2_err={r['l2_rel_err']:.5f} u_max_err={r['u_max_err_pct']:+.4f}% "
            f"max_rel={r['max_rel_err_central_pct']:.4f}% steady={r['steady']}",
            flush=True,
        )
    summary = {
        "case": "B13_poiseuille_2d_convergence",
        "lattice": "D2Q9",
        "collision": collision,
        "boundary": "zou_he_pressure_inlet/outlet + half-way bounce-back (pre-streaming)",
        "driving": "pressure_difference",
        "extrap": "none",
        "H_list": H_list,
        "tau": tau,
        "u_max_target": u_max,
        "min_steps": min_steps,
        "max_steps": max_steps,
        "per_grid": cases,
        "convergence": [
            {"H": r["H_eff"], "Re": r["Re"], "l2_rel_err": r["l2_rel_err"],
             "u_max_err_pct": r["u_max_err_pct"],
             "max_rel_err_central_pct": r["max_rel_err_central_pct"],
             "n_steps": r["n_steps"], "collision": r["collision"]}
            for r in cases
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="B13 2D Poiseuille (pressure-driven, D2Q9)")
    sub = ap.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("single")
    p1.add_argument("H", type=int)
    p1.add_argument("out_json", type=str)
    p1.add_argument("--tau", type=float, default=0.8)
    p1.add_argument("--umax", type=float, default=0.04)
    p1.add_argument("--collision", choices=["bgk", "mrt"], default="bgk")
    p1.add_argument("--min-steps", type=int, default=10000)
    p1.add_argument("--max-steps", type=int, default=60000)
    p1.add_argument("--seed", type=int, default=0)

    p2 = sub.add_parser("scan")
    p2.add_argument("out_dir", type=str)
    p2.add_argument("--H", type=int, nargs="+", default=[20, 40, 60])
    p2.add_argument("--tau", type=float, default=0.8)
    p2.add_argument("--umax", type=float, default=0.04)
    p2.add_argument("--collision", choices=["bgk", "mrt"], default="bgk")
    p2.add_argument("--min-steps", type=int, default=10000)
    p2.add_argument("--max-steps", type=int, default=60000)
    p2.add_argument("--include-re80", action="store_true")

    args = ap.parse_args()
    if args.mode == "single":
        r = run_case(args.H, args.tau, args.umax, args.collision,
                     args.min_steps, args.max_steps, args.out_json, args.seed)
        print(json.dumps({k: r[k] for k in
                          ["H_eff", "nx", "ny", "tau", "nu_lb", "delta_rho", "u_max_ana", "Re",
                           "Ma", "n_steps", "steady", "u_max_err_pct", "l2_rel_err",
                           "max_rel_err_central_pct", "rho_slope_ratio", "mass_drift_pct",
                           "elapsed_s"]},
                         indent=2))
    else:
        scan(args.H, args.tau, args.umax, args.collision,
             args.min_steps, args.max_steps, args.out_dir, args.include_re80)


if __name__ == "__main__":
    main()
