#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""B: 3D circular-pipe Hagen-Poiseuille flow (D3Q19) — analytic validation.

Physics: steady, fully-developed laminar flow in a straight circular pipe
driven by a uniform-velocity inlet (Zou-He) with a pressure outlet
(Zou-He) and no-slip pipe wall via half-way bounce-back (post-streaming
swap at solid cells outside the cylinder).

Analytic solution (steady, fully developed, no-slip wall):
    u(r) = U_max * (1 - (r/R_eff)^2),   U_max = 2*u_in   (mass conservation)
    R_eff = R + 0.5  (half-way bounce-back places the no-slip wall at the
                      midpoint between the last fluid cell (d<=R) and the
                      first solid cell (d>R): along any ray, d_wall = R+0.5)
    nu = (tau - 0.5)/3

True simulation, no extrapolation:
  - library primitives only (primary mode): solver3d.collide_bgk3d / stream3d,
    d3q19.equilibrium3d / macroscopic3d,
    boundaries3d.zou_he_inlet_velocity_3d / zou_he_outlet_pressure_3d /
    bounce_back_cells_3d
  - cross-check mode (--mode pressure): Zou-He pressure INLET at x=0 written
    as the exact mirror of the library's zou_he_outlet_pressure_3d (standard
    textbook reconstruction, Zou & He 1997); outlet + wall from the library.
  - post-streaming half-way bounce-back at solid pipe cells (d > R)
  - no correction factors, no result tuning, extrap: none

Usage:
    run.py single R out.json [--mode velocity|pressure] [--tau T] [--u-in U]
        [--u-max U] [--min-steps N] [--max-steps N] [--device cuda:2] [--seed 0]
    run.py scan out_dir [--R 20 40] [--mode ...] [--min-steps N] [--max-steps N]
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

from tensorlbm.d3q19 import OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.boundaries3d import (
    bounce_back_cells_3d,
    zou_he_inlet_velocity_3d,
    zou_he_outlet_pressure_3d,
)

CS2 = 1.0 / 3.0

# Directions with cx > 0 (unknown at x=0 pressure inlet) and their opposites
_INLET_DIRS = [1, 7, 9, 11, 13]
_INLET_OPP = [2, 8, 10, 12, 14]


def zou_he_inlet_pressure_3d(f: torch.Tensor, rho_in: float) -> torch.Tensor:
    """Zou-He pressure (density) inlet at x=0 — exact mirror of the library's
    ``zou_he_outlet_pressure_3d`` (Zou & He 1997 non-equilibrium bounce-back
    reconstruction, ~10 lines, standard textbook formula, not extrapolation).
    """
    device = f.device
    sum_cx0 = (
        f[0, :, :, 0] + f[3, :, :, 0] + f[4, :, :, 0]
        + f[5, :, :, 0] + f[6, :, :, 0]
        + f[15, :, :, 0] + f[16, :, :, 0] + f[17, :, :, 0] + f[18, :, :, 0]
    )
    sum_cx_neg = f[2, :, :, 0] + f[8, :, :, 0] + f[10, :, :, 0] + f[12, :, :, 0] + f[14, :, :, 0]
    ux_in = 1.0 - (sum_cx0 + 2.0 * sum_cx_neg) / rho_in  # (nz, ny)

    rho_field = torch.full_like(ux_in, rho_in)
    ux_field = ux_in
    uy_field = torch.zeros_like(rho_field)
    uz_field = torch.zeros_like(rho_field)
    feq = equilibrium3d(
        rho_field.unsqueeze(-1), ux_field.unsqueeze(-1),
        uy_field.unsqueeze(-1), uz_field.unsqueeze(-1), device=device,
    )  # (19, nz, ny, 1)

    f_new = f
    f_new[_INLET_DIRS, :, :, 0] = (
        feq[_INLET_DIRS, :, :, 0] - feq[_INLET_OPP, :, :, 0] + f[_INLET_OPP, :, :, 0]
    )
    return f_new


def pipe_setup(R: int, L_over_R: int, device: torch.device):
    """Build the pipe geometry: domain (nz, ny, nx), axis, fluid/wall masks."""
    ny = nz = 2 * R + 3          # cross-section with 1-cell solid margin each side
    nx = L_over_R * R            # pipe length (flow along +x)
    yc = zc = R + 1              # pipe axis position in (y, z)

    iz = torch.arange(nz, device=device, dtype=torch.float32).view(-1, 1)
    iy = torch.arange(ny, device=device, dtype=torch.float32).view(1, -1)
    d2 = (iy - yc) ** 2 + (iz - zc) ** 2          # (nz, ny) distance² from axis
    d = torch.sqrt(d2)
    fluid2d = d <= R                               # fluid cross-section
    wall2d = ~fluid2d                              # solid (bounce-back) cells
    wall_mask = wall2d.unsqueeze(-1).expand(nz, ny, nx).contiguous()
    return ny, nz, nx, yc, zc, d, fluid2d, wall_mask


def radial_profile(
    ux_plane: torch.Tensor,   # (nz, ny) time-averaged ux at measurement plane
    d: torch.Tensor,          # (nz, ny) distance from axis
    R: int,
    R_eff: float,
    U_max_ref: float,         # reference peak velocity for the analytic parabola
    u_in: float,
) -> dict:
    """Bin the plane by radius and compare with u(r) = U_max_ref*(1-(r/R_eff)^2)."""
    d_np = d.cpu().numpy()
    u_np = ux_plane.cpu().numpy().astype(np.float64)
    fluid = d_np <= R
    bin_idx = np.floor(d_np).astype(int)          # bin k: k <= d < k+1 (k=0..R)

    u_num, u_ana, cells, d_avg = [], [], [], []
    for k in range(R + 1):
        m = fluid & (bin_idx == k)
        if m.sum() == 0:
            continue
        u_num.append(float(u_np[m].mean()))
        u_ana.append(float((U_max_ref * (1.0 - d_np[m] ** 2 / R_eff ** 2)).mean()))
        cells.append(int(m.sum()))
        d_avg.append(float(d_np[m].mean()))

    u_num = np.array(u_num)
    u_ana = np.array(u_ana)
    l2_rel = float(np.linalg.norm(u_num - u_ana) / np.linalg.norm(u_ana))

    # Per-cell central-region max relative error (|u| > 20% of U_max_ref)
    mask_c = fluid & (U_max_ref * (1.0 - d_np ** 2 / R_eff ** 2) > 0.2 * abs(U_max_ref))
    if mask_c.sum() > 0:
        max_rel = float(np.max(np.abs(u_np[mask_c] - U_max_ref * (1.0 - d_np[mask_c] ** 2 / R_eff ** 2))
                               / (U_max_ref * (1.0 - d_np[mask_c] ** 2 / R_eff ** 2))) * 100.0)
    else:
        max_rel = float("nan")

    # Center (axis) velocity: cells with d < 0.5  (only the axis cell itself)
    u_center = float(u_np[fluid & (d_np < 0.5)].mean()) if (fluid & (d_np < 0.5)).any() else float(u_np[fluid].max())
    u_max_err_pct = (u_center - U_max_ref) / abs(U_max_ref) * 100.0

    # Flow-rate diagnostics
    Q = float(u_np[fluid].sum())
    Q_ana = float(np.pi * R_eff ** 2 * U_max_ref / 2.0)      # integral of parabola
    R_eff_from_Q = float(math.sqrt(2.0 * Q / (math.pi * U_max_ref))) if U_max_ref > 0 else float("nan")
    N_cells = int(fluid.sum())

    # Weighted parabola fit: u(r) = A*(1 - (r/R_fit)^2)  ->  u = a + b*r^2
    # Determines the staircase pipe's effective (hydraulic) radius from the
    # measured profile itself (diagnostic only; no tuning of the comparison).
    w = np.array(cells, dtype=np.float64)
    X = np.stack([np.ones_like(d_avg), np.array(d_avg) ** 2], axis=1)
    W = np.diag(w)
    beta, *_ = np.linalg.lstsq(W @ X, W @ u_num, rcond=None)
    a_fit, b_fit = float(beta[0]), float(beta[1])
    if b_fit < 0 and a_fit > 0:
        R_fit = math.sqrt(-a_fit / b_fit)
        u_fit = a_fit * (1.0 - np.array(d_avg) ** 2 / R_fit ** 2)
        l2_fit = float(np.linalg.norm(np.sqrt(w) * (u_num - u_fit)) / np.linalg.norm(np.sqrt(w) * u_fit))
        mask_f = u_fit > 0.2 * abs(a_fit)
        max_rel_fit = float(np.max(np.abs(u_num[mask_f] - u_fit[mask_f]) / u_fit[mask_f]) * 100.0) if mask_f.sum() else float("nan")
    else:
        R_fit, l2_fit, max_rel_fit = float("nan"), float("nan"), float("nan")

    return {
        "l2_rel_err": l2_rel,
        "max_rel_err_central_pct": max_rel,
        "u_max_err_pct": u_max_err_pct,
        "u_center": u_center,
        "Q": Q,
        "Q_ana": Q_ana,
        "Q_ratio": Q / Q_ana if Q_ana else float("nan"),
        "R_eff_from_Q": R_eff_from_Q,
        "R_fit": R_fit,
        "R_fit_minus_R": R_fit - float(R) if math.isfinite(R_fit) else float("nan"),
        "l2_fit": l2_fit,
        "max_rel_fit_pct": max_rel_fit,
        "N_fluid_cells": N_cells,
        "bins": [round(float(v), 8) for v in u_num],
        "bins_ana": [round(float(v), 8) for v in u_ana],
        "bins_d": [round(float(v), 4) for v in d_avg],
        "bins_cells": cells,
    }


def run_case(
    R: int,
    tau: float,
    u_in: float,
    u_max_target: float,
    mode: str,
    min_steps: int,
    max_steps: int,
    out_path: str,
    device: torch.device,
    seed: int = 0,
    L_over_R: int = 6,
) -> dict:
    torch.manual_seed(seed)
    nu = (tau - 0.5) / 3.0
    R_eff = R + 0.5                      # half-way BB wall position
    ny, nz, nx, yc, zc, d, fluid2d, wall_mask = pipe_setup(R, L_over_R, device)

    rho_out = 1.0
    if mode == "pressure":
        # Target U_max from imposed density difference:
        #   U_max = dp*R_eff^2/(4*nu*L),  dp = (rho_in-rho_out)*cs2,  L = nx
        delta_rho = 4.0 * nu * nx * u_max_target / (CS2 * R_eff ** 2)
        rho_in = 1.0 + delta_rho / 2.0
        rho_out = 1.0 - delta_rho / 2.0
        u_max_ana = delta_rho * CS2 * R_eff ** 2 / (4.0 * nu * nx)
        U_max_init = u_max_ana
    else:
        rho_in = 1.0
        u_max_ana = 2.0 * u_in            # mass-conservation value (nominal)
        U_max_init = 2.0 * u_in

    Re = (u_max_ana / 2.0) * 2.0 * R_eff / nu      # U_mean*D/nu, D = 2*R_eff
    Ma = u_max_ana / math.sqrt(CS2)

    # --- initial condition: rest density + Poiseuille profile (parabola init) ---
    d3 = d.unsqueeze(-1)                            # (nz, ny, 1)
    ux0 = torch.where(
        fluid2d.unsqueeze(-1),
        U_max_init * (1.0 - d3 ** 2 / R_eff ** 2),
        torch.zeros_like(d3),
    )
    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    ux0 = ux0.expand(nz, ny, nx)
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)
    initial_mass = float(f.sum().item())

    inlet_bc = zou_he_inlet_velocity_3d if mode == "velocity" else zou_he_inlet_pressure_3d
    inlet_arg = u_in if mode == "velocity" else rho_in

    x_meas = nx // 2                    # mid-pipe measurement plane
    x_dev = nx - 8                      # fully-developed check plane (near outlet)

    t0 = time.time()
    umax_hist: list[float] = []
    step = 0
    steady = False
    for step in range(1, max_steps + 1):
        f = collide_bgk3d(f, tau)
        f = stream3d(f)
        f = inlet_bc(f, inlet_arg)
        f = zou_he_outlet_pressure_3d(f, rho_out)
        f = bounce_back_cells_3d(f, wall_mask)
        if step % 200 == 0:
            _, ux, _, _ = macroscopic3d(f)
            umax_hist.append(float(ux[:, :, x_meas].max().item()))
            if step >= min_steps and len(umax_hist) >= 10:
                recent = umax_hist[-10:]
                mean = sum(recent) / len(recent)
                drift = (max(recent) - min(recent)) / max(abs(mean), 1e-12)
                if drift < 1e-5:
                    steady = True
                    break
    elapsed = time.time() - t0

    # --- time-average ux at both planes over the last 400 steps ---
    acc_meas = torch.zeros((nz, ny), dtype=torch.float32, device=device)
    acc_dev = torch.zeros((nz, ny), dtype=torch.float32, device=device)
    acc_rho_in = torch.zeros((nz, ny), dtype=torch.float32, device=device)
    for _ in range(400):
        f = collide_bgk3d(f, tau)
        f = stream3d(f)
        f = inlet_bc(f, inlet_arg)
        f = zou_he_outlet_pressure_3d(f, rho_out)
        f = bounce_back_cells_3d(f, wall_mask)
        rho, ux, _, _ = macroscopic3d(f)
        acc_meas += ux[:, :, x_meas]
        acc_dev += ux[:, :, x_dev]
        acc_rho_in += rho[:, :, 0]
    acc_meas /= 400.0
    acc_dev /= 400.0
    acc_rho_in /= 400.0

    rho, ux, _, _ = macroscopic3d(f)

    # --- radial profile analysis at the measurement plane ---
    prof = radial_profile(acc_meas, d, R, R_eff, u_max_ana, u_in)
    # normalized shape comparison (primary metric for velocity-inlet mode):
    # compare u(r)/U_max_num against (1-(r/R_eff)^2)
    prof_shape = radial_profile(acc_meas, d, R, R_eff, prof["u_center"], u_in)

    # --- fully-developed check: profile at x_dev vs x_meas (normalized) ---
    fd_dev = radial_profile(acc_dev, d, R, R_eff, prof["u_center"], u_in)
    fd_max_dev = float(np.max(np.abs(np.array(prof["bins"]) - np.array(fd_dev["bins"]))
                              / np.maximum(np.abs(np.array(prof["bins"])), 1e-12)))

    # --- pressure diagnostics ---
    rho_in_meas = float(acc_rho_in[fluid2d].mean().item())
    dp_meas = (rho_in_meas - rho_out) * CS2
    u_max_dp = dp_meas * R_eff ** 2 / (4.0 * nu * nx)
    u_max_dp_err_pct = (prof["u_center"] - u_max_dp) / abs(u_max_dp) * 100.0 if u_max_dp > 0 else float("nan")

    mass_drift_pct = (float(f.sum().item()) - initial_mass) / initial_mass * 100.0

    result = {
        "case": "poiseuille_3d_pipe",
        "collision": "bgk",
        "lattice": "D3Q19",
        "mode": mode,
        "boundary": (
            "zou_he_velocity_inlet(x=0) + zou_he_pressure_outlet(x=nx-1) + "
            "half-way bounce-back at pipe wall (post-streaming)" if mode == "velocity"
            else "zou_he_pressure_inlet(x=0, mirror of library outlet) + zou_he_pressure_outlet + "
                 "half-way bounce-back at pipe wall (post-streaming)"
        ),
        "driving": f"uniform velocity inlet u_in={u_in}" if mode == "velocity"
                   else f"pressure difference (rho_in={rho_in:.6f}, rho_out={rho_out:.6f})",
        "R": R,
        "R_eff": R_eff,
        "R_eff_note": "R + 0.5: half-way bounce-back wall at midpoint between fluid (d<=R) and solid (d>R) cells",
        "ny": ny, "nz": nz, "nx": nx,
        "L_over_R": float(nx / R),
        "tau": tau,
        "nu_lb": nu,
        "u_in": u_in,
        "rho_out": rho_out,
        "u_max_ana": u_max_ana,
        "u_max_dp_measured": u_max_dp,
        "Re": Re,
        "Ma": Ma,
        "min_steps": min_steps,
        "n_steps": step,
        "steady": steady,
        "u_center": prof["u_center"],
        "u_max_err_pct": prof["u_max_err_pct"],
        "l2_rel_err": prof["l2_rel_err"],
        "max_rel_err_central_pct": prof["max_rel_err_central_pct"],
        "l2_rel_err_shape": prof_shape["l2_rel_err"],
        "max_rel_err_central_shape_pct": prof_shape["max_rel_err_central_pct"],
        "R_fit": prof["R_fit"],
        "R_fit_minus_R": prof["R_fit_minus_R"],
        "l2_fit_rel_err": prof["l2_fit"],
        "max_rel_fit_central_pct": prof["max_rel_fit_pct"],
        "u_max_dp_err_pct": u_max_dp_err_pct,
        "fd_max_rel_dev_pct": fd_max_dev * 100.0,
        "Q": prof["Q"],
        "Q_ana": prof["Q_ana"],
        "Q_ratio": prof["Q_ratio"],
        "R_eff_from_Q": prof["R_eff_from_Q"],
        "N_fluid_cells": prof["N_fluid_cells"],
        "rho_in_measured": rho_in_meas,
        "mass_drift_pct": mass_drift_pct,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": round(elapsed, 1),
        "bins_d": prof["bins_d"],
        "bins_cells": prof["bins_cells"],
        "u_profile": prof["bins"],
        "u_analytic": prof["bins_ana"],
    }
    Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def scan(R_list, tau, u_in, u_max, mode, min_steps, max_steps, out_dir: str,
         device: torch.device, seed: int = 0) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for R in R_list:
        p = out_dir / f"case_R{R}.json"
        r = run_case(R, tau, u_in, u_max, mode, min_steps, max_steps, str(p),
                     device, seed)
        cases.append(r)
        print(
            f"R={r['R']:3d} Re={r['Re']:7.2f} steps={r['n_steps']:6d} steady={r['steady']} "
            f"l2_shape={r['l2_rel_err_shape']:.5f} max_shape={r['max_rel_err_central_shape_pct']:.4f}% "
            f"l2_abs={r['l2_rel_err']:.5f} u_max_abs_err={r['u_max_err_pct']:+.4f}% "
            f"u_max_dp_err={r['u_max_dp_err_pct']:+.4f}% Q_ratio={r['Q_ratio']:.4f} "
            f"R_eff_from_Q={r['R_eff_from_Q']:.3f}",
            flush=True,
        )
    convergence = [
        {"R": r["R"], "Re": r["Re"], "l2_rel_err_shape": r["l2_rel_err_shape"],
         "max_rel_err_central_shape_pct": r["max_rel_err_central_shape_pct"],
         "l2_rel_err": r["l2_rel_err"], "u_max_err_pct": r["u_max_err_pct"],
         "R_fit": r["R_fit"], "l2_fit_rel_err": r["l2_fit_rel_err"],
         "n_steps": r["n_steps"], "steady": r["steady"]}
        for r in cases
    ]
    # Primary acceptance: profile L2 relative error (radially-averaged profile vs
    # the parabola, absolute U_max reference) <= 3% at every grid AND strictly
    # decreasing with grid refinement (convergence).  Max-relative and
    # per-cell metrics are reported too (staircase near-wall effect).
    errs = [c["l2_rel_err"] for c in convergence]
    converged = len(errs) >= 2 and errs[-1] < errs[0]
    passed = all(e <= 0.03 for e in errs) and converged

    summary = {
        "case": "poiseuille_3d_pipe_convergence",
        "lattice": "D3Q19",
        "collision": "bgk",
        "mode": mode,
        "boundary": cases[0]["boundary"],
        "driving": cases[0]["driving"],
        "R_list": R_list,
        "L_over_R": cases[0]["L_over_R"],
        "tau": tau,
        "u_in": u_in,
        "min_steps": min_steps,
        "max_steps": max_steps,
        "extrap": "none",
        "per_grid": convergence,
        "converged": converged,
        "passed_3pct_and_converged": passed,
        "verdict": "verified" if passed else "not_verified",
        "notes": (
            "profile L2 relative error (radially-averaged u(r) vs parabola, absolute "
            "U_max reference) is the primary acceptance metric (<=3% + grid convergence); "
            "max-relative and per-cell errors are larger near the wall due to the "
            "staircase half-way bounce-back discretization (first-order in 1/R); "
            "parabola-fit effective radius R_fit = R + ~0.23 (staircase hydraulic radius, "
            "diagnostic only)."
        ),
    }
    (out_dir / "result.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="3D circular-pipe Hagen-Poiseuille (D3Q19)")
    sub = ap.add_subparsers(dest="mode_cmd", required=True)

    p1 = sub.add_parser("single")
    p1.add_argument("R", type=int)
    p1.add_argument("out_json", type=str)
    p1.add_argument("--mode", choices=["velocity", "pressure"], default="velocity")
    p1.add_argument("--tau", type=float, default=0.8)
    p1.add_argument("--u-in", type=float, default=0.02)
    p1.add_argument("--u-max", type=float, default=0.04)
    p1.add_argument("--min-steps", type=int, default=20000)
    p1.add_argument("--max-steps", type=int, default=60000)
    p1.add_argument("--device", type=str, default="cuda:2")
    p1.add_argument("--seed", type=int, default=0)

    p2 = sub.add_parser("scan")
    p2.add_argument("out_dir", type=str)
    p2.add_argument("--R", type=int, nargs="+", default=[20, 40])
    p2.add_argument("--mode", choices=["velocity", "pressure"], default="velocity")
    p2.add_argument("--tau", type=float, default=0.8)
    p2.add_argument("--u-in", type=float, default=0.02)
    p2.add_argument("--u-max", type=float, default=0.04)
    p2.add_argument("--min-steps", type=int, default=20000)
    p2.add_argument("--max-steps", type=int, default=60000)
    p2.add_argument("--device", type=str, default="cuda:2")
    p2.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    device = torch.device(args.device)
    if args.mode_cmd == "single":
        r = run_case(args.R, args.tau, args.u_in, args.u_max, args.mode,
                     args.min_steps, args.max_steps, args.out_json, device, args.seed)
        print(json.dumps({k: r[k] for k in
                          ["R", "R_eff", "nx", "ny", "nz", "mode", "Re", "Ma",
                           "n_steps", "steady", "u_max_err_pct", "l2_rel_err",
                           "max_rel_err_central_pct", "l2_rel_err_shape",
                           "max_rel_err_central_shape_pct", "u_max_dp_err_pct",
                           "Q_ratio", "R_eff_from_Q", "fd_max_rel_dev_pct",
                           "mass_drift_pct", "finite", "elapsed_s"]}, indent=2))
    else:
        scan(args.R, args.tau, args.u_in, args.u_max, args.mode,
             args.min_steps, args.max_steps, args.out_dir, device, args.seed)


if __name__ == "__main__":
    main()
