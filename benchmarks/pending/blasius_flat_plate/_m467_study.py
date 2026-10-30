#!/usr/bin/env python3
"""m467: Systematic study of Blasius flat-plate C_f error reduction.

Strategy:
1. Grid refinement: test finer grids to see if error continues to decrease
2. Domain height: test larger domain heights to reduce edge acceleration
3. Wall shear stress computation: test alternative methods
4. Collision model: compare BGK vs TRT

Key insight from m449: The error is ~27% for the best case (plate400_y1600 TRT).
The attribution analysis shows:
- Edge acceleration: ~21% (physical, due to finite domain height)
- Edge normalized residual: ~14% (numerical, inherent to LBM)

This script tests:
- Finer grids (plate800, plate1600)
- Larger domain heights (y3200, y6400)
- Alternative C_f computation methods
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import tensorlbm
from tensorlbm.boundaries import zou_he_outlet_pressure
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, collide_trt, stream

SPEC = torch.tensor([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=torch.int64)

# Extended grid definitions
GRIDS = {
    # Original grids
    "plate200": dict(nx=240, ny=1400, le=20, plate_len=200, probe=200),
    "plate400": dict(nx=440, ny=1400, le=20, plate_len=400, probe=200),
    "plate400_y1600": dict(nx=440, ny=1600, le=20, plate_len=400, probe=200),
    # m467: Finer grids
    "plate400_y3200": dict(nx=440, ny=3200, le=20, plate_len=400, probe=200),
    "plate400_y6400": dict(nx=440, ny=6400, le=20, plate_len=400, probe=200),
    "plate800_y1400": dict(nx=840, ny=1400, le=20, plate_len=800, probe=420),
    "plate800_y2800": dict(nx=840, ny=2800, le=20, plate_len=800, probe=420),
    "plate1600_y1400": dict(nx=1640, ny=1400, le=20, plate_len=1600, probe=820),
    "plate1600_y2800": dict(nx=1640, ny=2800, le=20, plate_len=1600, probe=820),
}

CF_COEFF = 2.0 * 0.3320573362151963


def blasius_table(eta_max: float = 10.0, h: float = 0.005):
    """Solve Blasius ODE using RK4 + bisection."""
    def rhs(s):
        f, fp, fpp = s
        return np.array([fp, fpp, -0.5 * f * fpp])

    def rk4(s, dt):
        k1 = rhs(s)
        k2 = rhs(s + 0.5 * dt * k1)
        k3 = rhs(s + 0.5 * dt * k2)
        k4 = rhs(s + dt * k3)
        return s + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    n = int(round(eta_max / h))

    def shoot(fpp0):
        s = np.array([0.0, 0.0, fpp0])
        for _ in range(n):
            s = rk4(s, h)
        return s[1]

    lo, hi = 0.30, 0.37
    flo = shoot(lo) - 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        fm = shoot(mid) - 1.0
        if flo * fm <= 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
    fpp0 = 0.5 * (lo + hi)

    etas = np.zeros(n + 1)
    fprimes = np.zeros(n + 1)
    s = np.array([0.0, 0.0, fpp0])
    for i in range(1, n + 1):
        s = rk4(s, h)
        etas[i] = i * h
        fprimes[i] = s[1]
    return etas, fprimes, fpp0


def plate_hwbb_mid(f: torch.Tensor, py: int, mask: torch.Tensor) -> torch.Tensor:
    """Half-way bounce-back for mid-domain thin plate."""
    f = f.clone()
    # upper surface
    f[2, py, :] = torch.where(mask, f[4, py + 1, :], f[2, py, :])
    f[5, py, :-1] = torch.where(mask[:-1], f[7, py + 1, 1:], f[5, py, :-1])
    f[6, py, 1:] = torch.where(mask[1:], f[8, py + 1, :-1], f[6, py, 1:])
    # lower surface
    f[4, py, :] = torch.where(mask, f[2, py - 1, :], f[4, py, :])
    f[7, py, 1:] = torch.where(mask[1:], f[5, py - 1, :-1], f[7, py, 1:])
    f[8, py, :-1] = torch.where(mask[:-1], f[6, py - 1, 1:], f[8, py, :-1])
    return f


def run_case(grid_name, collision, U, nu, n_steps, device_str="cpu", avg_steps=200, n_threads=32):
    """Run a single case and return diagnostics."""
    g = GRIDS[grid_name]
    nx, ny = g["nx"], g["ny"]
    le, plate_len, probe = g["le"], g["plate_len"], g["probe"]
    plate_end = le + plate_len
    py = ny // 2
    tau = 3.0 * nu + 0.5
    device = torch.device(device_str)
    if device_str == "cpu":
        torch.set_num_threads(n_threads)
    torch.manual_seed(0)

    lam_trt = 3.0 / 16.0
    tau_minus = 0.5 + lam_trt / (tau - 0.5)

    plate = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    plate[py, le:plate_end] = True
    plate_row = plate[py]

    rho0 = torch.ones((ny, nx), device=device)
    ux0 = torch.full((ny, nx), U, device=device)
    uy0 = torch.zeros((ny, nx), device=device)
    f = equilibrium(rho0, ux0, uy0, device=device)
    initial_mass = float(f.sum().item())

    spec = SPEC.to(device)
    feq_in = equilibrium(
        torch.ones((ny, 1), device=device),
        torch.full((ny, 1), U, device=device),
        torch.zeros((ny, 1), device=device),
    )[:, :, 0].contiguous()

    def apply_collision(f_):
        if collision == "bgk":
            return collide_bgk(f_, tau)
        return collide_trt(f_, tau, lam_trt)

    def step_fn(f_):
        f_ = apply_collision(f_)
        f_ = plate_hwbb_mid(f_, py, plate_row)
        f_ = f_.clone()
        f_[:, 0, :] = f_[:, 1, :][spec]
        f_[:, -1, :] = f_[:, -2, :][spec]
        f_ = stream(f_)
        f_ = f_.clone()
        f_[:, :, 0] = feq_in
        f_ = zou_he_outlet_pressure(f_, 1.0)
        return f_

    t0 = time.time()
    for step in range(1, n_steps + 1):
        f = step_fn(f)
    elapsed = time.time() - t0

    # Averaging
    st_t = torch.tensor([probe], dtype=torch.int64, device=device)
    prof_acc = torch.zeros((1, ny), device=device)
    n_avg = 0
    for _ in range(avg_steps):
        f = step_fn(f)
        _, ux, _ = macroscopic(f)
        prof_acc += ux[:, st_t].T
        n_avg += 1
    prof = (prof_acc / n_avg).cpu().numpy()[0]

    rho, ux_f, uy_f = macroscopic(f)
    mass_drift_pct = (float(f.sum().item()) - initial_mass) / initial_mass * 100.0
    finite = bool(torch.isfinite(f).all().item())

    x_eff = probe - le
    Rex = U * x_eff / nu
    scale = math.sqrt(nu * x_eff / U)
    etas, fprimes, fpp0 = blasius_table()

    # Extract velocity profile
    y_w = py + 0.5
    y_hi = np.arange(py + 1, ny, dtype=np.float64) - y_w
    eta_hi = y_hi / scale
    fp_ref_hi = np.interp(eta_hi, etas, fprimes)
    u_ref_hi = U * fp_ref_hi
    u_hi = prof[py + 1:]

    # Edge velocity
    u_edge = float(ux_f[ny - 2, probe].item())
    u_edge_over_U = u_edge / U

    # C_f computation methods
    # Method 1: 3-point one-sided FD (standard)
    s_dist = np.array([0.5, 1.5, 2.5])
    A = np.vstack([np.ones(3), s_dist, s_dist**2]).T
    w_der = np.linalg.solve(A.T, np.array([0.0, 1.0, 0.0]))
    u123 = np.array([prof[py + 1], prof[py + 2], prof[py + 3]])
    dudy_fd = float(w_der @ u123)
    Cf_fd = 2.0 * nu * dudy_fd / (U * U)

    # Method 2: 2-point one-sided FD (first point only)
    dudy_2pt = prof[py + 1] / 0.5
    Cf_2pt = 2.0 * nu * dudy_2pt / (U * U)

    # Method 3: Corrected for edge velocity (use u_edge instead of U)
    Cf_fd_ue = 2.0 * nu * dudy_fd / (u_edge * u_edge)

    # Method 4: Fit to Blasius profile (least squares)
    # Find best fit scaling factor
    m_fit = (eta_hi > 0.1) & (eta_hi < 4.0) & (fp_ref_hi > 0.01) & (fp_ref_hi < 0.99)
    if m_fit.sum() > 10:
        u_fit = u_hi[m_fit]
        uref_fit = u_ref_hi[m_fit]
        # Linear fit: u_sim = A * u_ref
        A_fit = np.dot(u_fit, uref_fit) / np.dot(uref_fit, uref_fit)
        # C_f from fitted gradient
        dudy_fit = A_fit * U * fpp0 / scale
        Cf_fit = 2.0 * nu * dudy_fit / (U * U)
    else:
        Cf_fit = float('nan')
        A_fit = float('nan')

    # Reference C_f
    Cf_ref = CF_COEFF / math.sqrt(Rex)

    # Profile L2 error
    m = (eta_hi > 0.05) & (eta_hi < 5.0) & (fp_ref_hi > 0.02) & (fp_ref_hi < 0.995)
    rel_err = np.abs(u_hi - u_ref_hi) / np.maximum(u_ref_hi, 1e-12)
    l2_rel = float(np.linalg.norm(rel_err[m]) / np.linalg.norm(np.ones(m.sum()))) if m.any() else float('nan')

    return {
        "grid": grid_name,
        "collision": collision,
        "nx": nx,
        "ny": ny,
        "n_steps": n_steps,
        "elapsed_s": elapsed,
        "Rex": Rex,
        "x_eff": x_eff,
        "scale": scale,
        "u_edge_over_U": u_edge_over_U,
        "Cf_ref": Cf_ref,
        "Cf_fd": Cf_fd,
        "Cf_fd_err_pct": (Cf_fd - Cf_ref) / Cf_ref * 100,
        "Cf_2pt": Cf_2pt,
        "Cf_2pt_err_pct": (Cf_2pt - Cf_ref) / Cf_ref * 100,
        "Cf_fd_ue": Cf_fd_ue,
        "Cf_fd_ue_err_pct": (Cf_fd_ue - Cf_ref) / Cf_ref * 100,
        "Cf_fit": Cf_fit,
        "Cf_fit_err_pct": (Cf_fit - Cf_ref) / Cf_ref * 100,
        "A_fit": A_fit,
        "l2_rel_err_profile": l2_rel,
        "mass_drift_pct": mass_drift_pct,
        "finite": finite,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=str, default="plate400_y1600,plate400_y3200,plate400_y6400")
    ap.add_argument("--collision", type=str, default="trt")
    ap.add_argument("--U", type=float, default=0.05)
    ap.add_argument("--nu", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="results/m467_study.json")
    args = ap.parse_args()

    grids = [g.strip() for g in args.grids.split(",")]
    results = []

    for grid in grids:
        print(f"\n=== Running {grid} with {args.collision} ===")
        try:
            res = run_case(grid, args.collision, args.U, args.nu, args.steps, args.device)
            results.append(res)
            print(f"  C_f ref: {res['Cf_ref']:.6f}")
            print(f"  C_f FD:  {res['Cf_fd']:.6f} (err: {res['Cf_fd_err_pct']:+.2f}%)")
            print(f"  C_f 2pt: {res['Cf_2pt']:.6f} (err: {res['Cf_2pt_err_pct']:+.2f}%)")
            print(f"  C_f ue:  {res['Cf_fd_ue']:.6f} (err: {res['Cf_fd_ue_err_pct']:+.2f}%)")
            print(f"  C_f fit: {res['Cf_fit']:.6f} (err: {res['Cf_fit_err_pct']:+.2f}%)")
            print(f"  u_e/U:   {res['u_edge_over_U']:.5f}")
            print(f"  L2 err:  {res['l2_rel_err_profile']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"study": "m467_blasius_grid_refinement", "runs": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
