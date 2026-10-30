#!/usr/bin/env python3
"""SDAA first-physics case: 2D Couette flow with analytical comparison.

Runs a small D2Q9 LBM Couette flow on the SDAA device in **pure torch eager
mode** (no triton, no torch.compile).  The steady-state analytical solution
is the linear profile  u(y) = U0 * y / H_eff, valid for any Re.

This script does NOT modify any existing business code; it only imports
library primitives (tensorlbm.d2q9, tensorlbm.solver) and runs them on
the 'sdaa' device.

Usage:
    PYTHONPATH=src python scripts/sdaa_first_case.py [--H 62] [--steps 500]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tensorlbm.d2q9 import C, OPPOSITE, W, equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream

CS2 = 1.0 / 3.0


def moving_wall_bounce_back(
    f_pre: torch.Tensor,
    f: torch.Tensor,
    wall: torch.Tensor,
    u_wall: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Pre-streaming half-way bounce-back with moving-wall momentum injection.

    Same physics as benchmarks/verified/couette_2d/run.py:
      f_new[q] = f_pre[opp[q]] + 2*w_q*rho*(c_q . u_wall)/cs^2
    """
    opp = OPPOSITE.to(device)
    c = C.to(device)
    w = W.to(device)
    f_new = torch.where(wall.unsqueeze(0), f_pre[opp], f)
    rho_w = torch.clamp(f_pre.sum(dim=0), min=1e-12)
    cu = c[:, 0].view(9, 1, 1) * u_wall.unsqueeze(0)
    injection = (2.0 * w.view(9, 1, 1) * rho_w.unsqueeze(0) * cu) / CS2
    return f_new + injection * wall.unsqueeze(0)


def run_couette_sdaa(
    H: int,
    tau: float,
    U0: float,
    n_steps: int,
    n_avg: int,
    out_path: str,
) -> dict:
    """Run one Couette case on SDAA and return measurements."""
    device = torch.device("sdaa")
    torch.manual_seed(0)

    ny = H + 2  # wall rows at y=0 and y=ny-1
    nx = H      # periodic in x
    nu = (tau - 0.5) / 3.0
    H_eff = ny - 2.0  # effective gap (walls at y=0.5 .. ny-1.5)
    Re = U0 * H / nu
    Ma = U0 / math.sqrt(CS2)

    # Wall masks
    wall = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall[0, :] = True    # bottom wall: stationary
    wall[-1, :] = True   # top wall: moving at U0
    u_wall = torch.zeros((ny, nx), device=device)
    u_wall[-1, :] = U0

    # Initialize from analytic linear ramp (fluid rows) — fast convergence
    yy = torch.arange(ny, device=device, dtype=torch.float32)
    u0 = torch.clamp(U0 * (yy - 0.5) / H_eff, min=0.0, max=U0)
    u0 = u0.view(ny, 1).expand(ny, nx).clone()
    u0[0, :] = 0.0
    u0[-1, :] = 0.0
    f = equilibrium(
        torch.ones((ny, nx), device=device),
        u0,
        torch.zeros((ny, nx), device=device),
    )
    initial_mass = float(f.sum().item())

    col = nx // 2  # measurement column

    # ---- Time stepping (pure torch eager, no compile/triton) ----
    t0 = time.time()
    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_bgk(f, tau)
        f = moving_wall_bounce_back(f_pre, f, wall, u_wall, device)
        f = stream(f)
    step_elapsed = time.time() - t0

    # Time-average profile over n_avg additional steps
    t_avg0 = time.time()
    prof_acc = torch.zeros(ny, device=device)
    for _ in range(n_avg):
        f_pre = f.clone()
        f = collide_bgk(f, tau)
        f = moving_wall_bounce_back(f_pre, f, wall, u_wall, device)
        f = stream(f)
        _, ux, _ = macroscopic(f)
        prof_acc += ux[:, col]
    prof_acc /= n_avg
    avg_elapsed = time.time() - t_avg0

    total_elapsed = time.time() - t0

    # Extract numerical profile (fluid rows only)
    u_num = prof_acc[1 : ny - 1].cpu().numpy()
    y_phys = np.arange(1, ny - 1, dtype=np.float64) - 0.5
    u_ana = U0 * (y_phys / H_eff)

    # Error metrics
    l2_rel = float(np.linalg.norm(u_num - u_ana) / np.linalg.norm(u_ana))
    max_abs_err = float(np.max(np.abs(u_num - u_ana)))
    mask = u_ana > 0.2 * U0
    max_rel = float(np.max(np.abs(u_num[mask] - u_ana[mask]) / u_ana[mask]) * 100.0)

    u_top_num = float(u_num[-1])
    u_top_ana = float(u_ana[-1])
    u_top_err = abs(u_top_num - u_top_ana) / u_top_ana * 100.0

    # Wall shear
    slope_num = float((u_num[-1] - u_num[0]) / (y_phys[-1] - y_phys[0]))
    tau_ana = nu * U0 / H_eff
    tau_num = slope_num * nu
    tau_err_pct = (tau_num - tau_ana) / tau_ana * 100.0

    # Device evidence
    device_name = torch.sdaa.get_device_name(0)

    result = {
        "case": "couette_2d_sdaa",
        "lattice": "D2Q9",
        "collision": "bgk",
        "boundary": "moving top wall + stationary bottom wall, half-way bounce-back (pre-streaming)",
        "driving": "moving wall (top plate at u=U0, x-periodic channel)",
        "extrap": "none",
        "compile_mode": "eager (pure torch, no triton, no torch.compile)",
        "device": str(device),
        "device_name": device_name,
        "H_eff": H,
        "ny": ny,
        "nx": nx,
        "grid": f"{ny}x{nx}",
        "tau": tau,
        "nu_lb": nu,
        "U0": U0,
        "Re": Re,
        "Ma": Ma,
        "n_steps": n_steps,
        "n_avg_steps": n_avg,
        "l2_rel_err": l2_rel,
        "max_abs_err": max_abs_err,
        "max_rel_err_masked_pct": max_rel,
        "u_top_row_num": u_top_num,
        "u_top_row_ana": u_top_ana,
        "u_top_row_err_pct": u_top_err,
        "tau_ana": tau_ana,
        "tau_num": tau_num,
        "tau_err_pct": tau_err_pct,
        "mass_drift_pct": (float(f.sum().item()) - initial_mass) / initial_mass * 100.0,
        "finite": bool(torch.isfinite(f).all().item()),
        "step_elapsed_s": round(step_elapsed, 2),
        "avg_elapsed_s": round(avg_elapsed, 2),
        "total_elapsed_s": round(total_elapsed, 2),
        "u_profile_num": [round(float(v), 8) for v in u_num],
        "u_profile_ana": [round(float(v), 8) for v in u_ana],
    }

    Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="SDAA first-physics: 2D Couette flow")
    ap.add_argument("--H", type=int, default=62, help="channel height (fluid rows)")
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--u0", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=500, help="main stepping steps")
    ap.add_argument("--avg", type=int, default=200, help="time-averaging steps")
    ap.add_argument("--out", type=str, default="artifacts/sdaa_couette_result.json")
    args = ap.parse_args()

    # Ensure output directory exists
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"=== SDAA Couette 2D: H={args.H}, tau={args.tau}, U0={args.u0}, "
          f"steps={args.steps}, avg={args.avg} ===", flush=True)

    result = run_couette_sdaa(
        H=args.H,
        tau=args.tau,
        U0=args.u0,
        n_steps=args.steps,
        n_avg=args.avg,
        out_path=args.out,
    )

    print(f"\n--- Device Evidence ---", flush=True)
    print(f"torch.sdaa.get_device_name(0) = {result['device_name']}", flush=True)
    print(f"device = {result['device']}", flush=True)

    print(f"\n--- Physics Results ---", flush=True)
    print(f"Grid: {result['grid']}  (ny={result['ny']}, nx={result['nx']}, H_eff={result['H_eff']})", flush=True)
    print(f"Re={result['Re']:.2f}  Ma={result['Ma']:.4f}  nu={result['nu_lb']:.4f}", flush=True)
    print(f"Steps: {result['n_steps']} + {result['n_avg_steps']} avg", flush=True)
    print(f"L2 relative error:     {result['l2_rel_err']:.6e}", flush=True)
    print(f"Max rel error (masked): {result['max_rel_err_masked_pct']:.4f}%", flush=True)
    print(f"u_top error:            {result['u_top_row_err_pct']:+.4f}%", flush=True)
    print(f"Wall shear error:       {result['tau_err_pct']:+.4f}%", flush=True)
    print(f"Finite: {result['finite']}  Mass drift: {result['mass_drift_pct']:.2e}%", flush=True)

    print(f"\n--- Timing ---", flush=True)
    print(f"Step phase:  {result['step_elapsed_s']}s", flush=True)
    print(f"Avg phase:   {result['avg_elapsed_s']}s", flush=True)
    print(f"Total wall:  {result['total_elapsed_s']}s", flush=True)

    print(f"\nOutput written to: {args.out}", flush=True)


if __name__ == "__main__":
    main()
