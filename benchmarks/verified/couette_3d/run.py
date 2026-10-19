#!/usr/bin/env python3
"""Couette 3D gate runner v2 (preregistered driver, W11-D case 1).

Lineage: worktree benchmarks/pending/couette_3d/run.py (skeleton) with
THREE disclosed, evidence-backed changes -- everything else verbatim:

  1. INIT FIX: the skeleton's ``u0[0,:,:]=0; u0[-1,:,:]=0`` zeroes the
     z=0/z=nz-1 FLUID planes (2D wall-row indexing lifted verbatim from
     couette_2d).  Diagnosis (diag_arms.json, diag_dtype.json):
     buggy init reproduces the pilot anti-convergence (H16 l2 6.7e-5,
     grows ~x10 per H doubling; spanwise std 1e-3), fixed init gives
     z/x-exactly-uniform fields (spanwise std 0.0) and l2 at the fp32
     accumulation floor.  v2 initializes the clamped analytic ramp on
     ALL cells (wall rows take the clamped 0/U0 values, as in 2D).
  2. CUBE DOMAIN: nx=nz=H (skeleton used 8).  Equivalence at H16 measured
     (l2 1.046e-5 both, arm B vs B16); cube removes lateral extent as a
     hidden parameter and makes the ladder meaningfully 3D.
  3. dtype parameter (fp32 primary GPU / fp64 supplementary CPU ladder,
     monotonicity-exactness evidence tier) + equal run length
     min=20000/max=24000 for every tier (fair cross-tier comparison).

Frozen gates live in ../prereg.md (md5 recorded there); this driver only
measures and writes JSON.  No verdicts inside.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tensorlbm.d3q19 import OPPOSITE, W_EXACT64, C, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d

CS2 = 1.0 / 3.0
DEVICE = None  # set in main


def moving_wall_bounce_back3d(f_pre, f, wall, u_wall):
    """VERBATIM from pending/couette_3d/run.py (md5 in provenance block)."""
    opp = OPPOSITE.to(DEVICE)
    c = C.to(DEVICE)
    w = W_EXACT64.to(device=DEVICE, dtype=f.dtype)
    f_new = torch.where(wall.unsqueeze(0), f_pre[opp], f)
    rho_w = torch.clamp(f_pre.sum(dim=0), min=1e-12)
    cu = c[:, 0].view(19, 1, 1, 1) * u_wall.unsqueeze(0)
    injection = (2.0 * w.view(19, 1, 1, 1) * rho_w.unsqueeze(0) * cu) / CS2
    return f_new + injection * wall.unsqueeze(0)


def run_case(
    H: int,
    tau: float,
    U0: float,
    min_steps: int,
    max_steps: int,
    out_path: str,
    dtype: torch.dtype,
    seed: int = 0,
) -> dict:
    """One 3D Couette tier; estimator VERBATIM from the skeleton."""
    torch.manual_seed(seed)
    ny = H + 2
    nx = H  # CHANGE 2: cube (skeleton: 8)
    nz = H
    nu = (tau - 0.5) / 3.0
    H_eff = ny - 2.0
    Re = U0 * H / nu
    Ma = U0 / math.sqrt(CS2)

    wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=DEVICE)
    wall[:, 0, :] = True
    wall[:, -1, :] = True
    u_wall = torch.zeros((nz, ny, nx), dtype=dtype, device=DEVICE)
    u_wall[:, -1, :] = U0

    # CHANGE 1: ramp on all cells; no plane zeroing (skeleton bug removed)
    yy = torch.arange(ny, device=DEVICE, dtype=dtype)
    u0 = (
        torch.clamp(U0 * (yy - 0.5) / H_eff, min=0.0, max=U0)
        .view(1, ny, 1)
        .expand(nz, ny, nx)
        .clone()
    )
    f = equilibrium3d(
        torch.ones((nz, ny, nx), dtype=dtype, device=DEVICE),
        u0,
        torch.zeros((nz, ny, nx), dtype=dtype, device=DEVICE),
        torch.zeros((nz, ny, nx), dtype=dtype, device=DEVICE),
    )
    initial_mass = float(f.sum().item())

    def _step(f):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau)
        f = moving_wall_bounce_back3d(f_pre, f, wall, u_wall)
        return stream3d(f)

    z0, x0 = nz // 2, nx // 2
    t0 = time.time()
    u_top_hist: list[float] = []
    step = 0
    steady = False
    for step in range(1, max_steps + 1):
        f = _step(f)
        if step % 200 == 0:
            _, ux, _, _ = macroscopic3d(f)
            u_top_hist.append(float(ux[z0, ny - 2, x0].item()))
            if step >= min_steps and len(u_top_hist) >= 10:
                recent = u_top_hist[-10:]
                mean = sum(recent) / len(recent)
                drift = (max(recent) - min(recent)) / max(abs(mean), 1e-12)
                if drift < 1e-5:
                    steady = True
                    break
    elapsed = time.time() - t0

    prof_acc = torch.zeros(ny, dtype=dtype, device=DEVICE)
    for _ in range(200):
        f = _step(f)
        _, ux, _, _ = macroscopic3d(f)
        prof_acc += ux[z0, :, x0]
    prof_acc /= 200.0

    rho, ux, uy, uz = macroscopic3d(f)
    u_num = prof_acc[1 : ny - 1].cpu().numpy().astype(np.float64)
    y_phys = np.arange(1, ny - 1, dtype=np.float64) - 0.5
    u_ana = U0 * (y_phys / H_eff)

    l2_rel = float(np.linalg.norm(u_num - u_ana) / np.linalg.norm(u_ana))
    max_abs_err = float(np.max(np.abs(u_num - u_ana)))
    mask = u_ana > 0.2 * U0
    max_rel = float(np.max(np.abs(u_num[mask] - u_ana[mask]) / u_ana[mask]) * 100.0)
    u_top_num = float(u_num[-1])
    u_top_ana = float(u_ana[-1])
    u_top_err = abs(u_top_num - u_top_ana) / u_top_ana * 100.0
    slope_num = float((u_num[-1] - u_num[0]) / (y_phys[-1] - y_phys[0]))
    tau_ana = nu * U0 / H_eff
    tau_num = slope_num * nu
    tau_err_pct = (tau_num - tau_ana) / tau_ana * 100.0
    uz_max = float(torch.abs(uz).max().item())
    ux_spanwise_std = float(ux.std(dim=0).mean().item())
    mom_x = float((rho * ux).sum().item())

    result = {
        "case": "couette_3d",
        "driver": "run_v2.py",
        "changes_vs_skeleton": [
            "init: clamped analytic ramp on all cells (skeleton z-plane zeroing bug removed; diag_arms.json)",
            "domain: cube nx=nz=H (skeleton 8; equivalence arm B vs B16)",
            "dtype parameter + equal run length min/max (20000/24000) all tiers",
        ],
        "collision": "bgk",
        "lattice": "D3Q19",
        "boundary": "moving top wall + stationary bottom wall, half-way bounce-back (pre-streaming, f_pre[opp] + 2*w*rho*(c.u_wall)/cs^2 injection)",
        "driving": "moving wall (top plate at u = U0, x/z-periodic channel)",
        "extrap": "none",
        "dtype": str(dtype).replace("torch.", ""),
        "H_eff": H,
        "ny": ny,
        "nx": nx,
        "nz": nz,
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
        "uz_max": uz_max,
        "ux_spanwise_std": ux_spanwise_std,
        "total_x_momentum": mom_x,
        "mass_drift_pct": (float(f.sum().item()) - initial_mass) / initial_mass * 100.0,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": round(elapsed, 1),
        "u_profile": [round(float(v), 8) for v in u_num],
        "u_analytic": [round(float(v), 8) for v in u_ana],
    }
    Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    global DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp32", "fp64"], default="fp32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--H", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--U0", type=float, default=0.05)
    ap.add_argument("--min-steps", type=int, default=20000)
    ap.add_argument("--max-steps", type=int, default=24000)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    DEVICE = torch.device(args.device)
    dt = torch.float32 if args.dtype == "fp32" else torch.float64
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for H in args.H:
        p = out / f"case_H{H}_{args.dtype}.json"
        r = run_case(H, args.tau, args.U0, args.min_steps, args.max_steps, str(p), dt)
        print(
            f"H={r['H_eff']:3d} {r['dtype']:5s} steps={r['n_steps']:6d} l2={r['l2_rel_err']:.3e} "
            f"max_rel={r['max_rel_err_masked_pct']:.5f}% u_top={r['u_top_row_err_pct']:+.5f}% "
            f"tau={r['tau_err_pct']:+.4f}% mass={r['mass_drift_pct']:+.5f}% "
            f"uz={r['uz_max']:.1e} span={r['ux_spanwise_std']:.1e} steady={r['steady']} "
            f"t={r['elapsed_s']:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
