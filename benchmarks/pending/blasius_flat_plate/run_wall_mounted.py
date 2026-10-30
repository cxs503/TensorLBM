#!/usr/bin/env python
"""Blasius flat-plate boundary layer — wall-mounted plate configuration.

Fix for m410/m446 failure: The previous mid-domain plate setup with mirror BCs
created a channel flow with boundary layers growing on both sides, causing
displacement-induced edge acceleration (u_edge/U > 1) and 27% error in C_f.

New configuration:
- Plate mounted on BOTTOM wall (y=0)
- Bottom wall: bounce-back (no-slip) for plate region
- Top boundary: symmetry (mirror) far from plate (ny=800, plate at y=0)
- Inlet: uniform velocity profile (equilibrium at U)
- Outlet: Zou-He pressure outlet
- Boundary layer grows on ONE side only (top of plate)
- Top boundary far enough that it doesn't affect BL development

This eliminates the channel blockage effect and should give C_f error < 3%.

Reference: Blasius ODE self-solved (RK4 shooting, h=0.005), f''(0)=0.3320573362,
C_f_ref = 0.6641146724/sqrt(Re_x).
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

# Mirror (specular) direction map for symmetry BC: cy -> -cy.
SPEC = torch.tensor([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=torch.int64)

# Wall-mounted plate grids: plate starts at leading edge (x=le), grows upward
GRIDS = {
    # Small grid for quick testing
    "wall_small": dict(nx=240, ny=400, le=20, plate_len=200, probe=100),
    # Medium grid: probe at x_eff=100, Re_x=500
    "wall_medium": dict(nx=440, ny=600, le=20, plate_len=400, probe=120),
    # Large grid: probe at x_eff=200, Re_x=1000
    "wall_large": dict(nx=640, ny=800, le=20, plate_len=600, probe=220),
    # Extra large: probe at x_eff=300, Re_x=1500
    "wall_xlarge": dict(nx=840, ny=1000, le=20, plate_len=800, probe=320),
}

CF_COEFF = 2.0 * 0.3320573362151963  # 0.6641146724303926


def blasius_table(eta_max: float = 10.0, h: float = 0.005):
    """Solve f''' + 0.5*f*f'' = 0, f(0)=f'(0)=0, f'(inf)=1 (RK4 + bisection)."""

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


def plate_hwbb_wall(f: torch.Tensor, plate_mask: torch.Tensor) -> torch.Tensor:
    """Half-way bounce-back for wall-mounted plate at y=0.
    
    Plate is at the bottom wall (y=0). Fluid is above (y>=1).
    Wall is at y=-0.5 (half-way between y=0 solid and y=1 fluid).
    
    For the plate region, we reflect the distributions that would stream
    into the wall back into the fluid with opposite vertical velocity.
    """
    f = f.clone()
    # Upper surface of plate (fluid at y=1, wall at y=0.5)
    # Distributions streaming downward (toward wall): 4, 7, 8
    # After bounce-back, they become: 2, 5, 6 (streaming upward)
    # With half-way BB, the reflection happens at the wall location
    
    # f[4, 0, :] (streaming down) -> f[2, 1, :] (streaming up)
    # f[7, 0, :] (streaming down-left) -> f[5, 1, :] (streaming up-right)
    # f[8, 0, :] (streaming down-right) -> f[6, 1, :] (streaming up-left)
    
    # But we need to apply this AFTER collision and BEFORE streaming
    # Actually, the standard approach is:
    # 1. Collision
    # 2. Bounce-back at wall (reflect distributions)
    # 3. Streaming
    
    # For a wall at y=0 (solid) with fluid at y>=1:
    # After collision, the distributions at y=1 that would stream into y=0
    # are bounced back. These are directions 4 (0,-1), 7 (-1,-1), 8 (1,-1).
    
    # The bounce-back rule: f_i(x_wall+0.5*e_i) = f_i_opposite(x_wall+0.5*e_i)
    # For half-way BB, the wall is at y=0.5 (between y=0 solid and y=1 fluid)
    
    # At y=1 (first fluid row), after collision:
    # f[4, 1, x] would stream to y=0 (solid) -> bounce back to f[2, 1, x]
    # f[7, 1, x] would stream to (x-1, 0) -> bounce back to f[5, 1, x+1]
    # f[8, 1, x] would stream to (x+1, 0) -> bounce back to f[6, 1, x-1]
    
    # Apply bounce-back for plate region
    f[2, 1, :] = torch.where(plate_mask, f[4, 1, :], f[2, 1, :])
    f[5, 1, :-1] = torch.where(plate_mask[:-1], f[7, 1, 1:], f[5, 1, :-1])
    f[6, 1, 1:] = torch.where(plate_mask[1:], f[8, 1, :-1], f[6, 1, 1:])
    
    return f


def run_case(
    grid_name, collision, U, nu, n_steps, out_path, device_str="cpu", avg_steps=200, n_threads=32
):
    g = GRIDS[grid_name]
    nx, ny = g["nx"], g["ny"]
    le, plate_len, probe = g["le"], g["plate_len"], g["probe"]
    plate_end = le + plate_len
    device = torch.device(device_str)
    if device_str == "cpu":
        torch.set_num_threads(n_threads)
    torch.manual_seed(0)

    tau = 3.0 * nu + 0.5
    lam_trt = 3.0 / 16.0
    tau_minus = 0.5 + lam_trt / (tau - 0.5)

    # Plate mask: bottom wall (y=0) from x=le to x=plate_end
    plate = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    plate[0, le:plate_end] = True
    plate_row = plate[0]  # (nx,)

    # Initial condition: uniform flow
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
        # 1. Collision
        f_ = apply_collision(f_)
        
        # 2. Bounce-back at bottom wall (plate region)
        f_ = plate_hwbb_wall(f_, plate_row)
        
        # 3. Streaming
        f_ = stream(f_)
        
        # 4. Boundary conditions
        # Inlet (x=0): uniform velocity (equilibrium)
        f_[:, :, 0] = feq_in
        
        # Outlet (x=nx-1): Zou-He pressure outlet
        f_ = zou_he_outlet_pressure(f_, 1.0)
        
        # Top boundary (y=ny-1): symmetry (mirror)
        # Reflect distributions with positive y-velocity
        f_[2, -1, :] = f_[4, -2, :]
        f_[5, -1, :-1] = f_[7, -2, 1:]
        f_[6, -1, 1:] = f_[8, -2, :-1]
        
        # Bottom boundary (y=0): no-slip wall (bounce-back) for non-plate region
        # For x < le and x >= plate_end, apply bounce-back
        non_plate_mask = ~plate_row
        f_[2, 0, :] = torch.where(non_plate_mask, f_[4, 0, :], f_[2, 0, :])
        f_[5, 0, :-1] = torch.where(non_plate_mask[:-1], f_[7, 0, 1:], f_[5, 0, :-1])
        f_[6, 0, 1:] = torch.where(non_plate_mask[1:], f_[8, 0, :-1], f_[6, 0, 1:])
        
        return f_

    t0 = time.time()
    for step in range(1, n_steps + 1):
        f = step_fn(f)
    elapsed = time.time() - t0

    # Averaging window
    x_eff = probe - le
    Rex = U * x_eff / nu
    scale = math.sqrt(nu * x_eff / U)
    etas, fprimes, fpp0 = blasius_table()

    # Collect profiles at probe location
    prof_acc = torch.zeros(ny, device=device)
    n_avg = 0
    for _ in range(avg_steps):
        f = step_fn(f)
        _, ux, _ = macroscopic(f)
        prof_acc += ux[:, probe]
        n_avg += 1
    prof = (prof_acc / n_avg).cpu().numpy()  # (ny,)

    rho, ux_f, uy_f = macroscopic(f)
    mass_drift_pct = (float(f.sum().item()) - initial_mass) / initial_mass * 100.0
    finite = bool(torch.isfinite(f).all().item())

    # Extract velocity profile above the plate
    y_w = 0.5  # wall location (half-way between y=0 solid and y=1 fluid)
    y_hi = np.arange(1, ny, dtype=np.float64) - y_w
    eta_hi = y_hi / scale
    fp_ref_hi = np.interp(eta_hi, etas, fprimes)
    u_ref_hi = U * fp_ref_hi
    u_hi = prof[1:]  # fluid rows y=1 to y=ny-1

    # Profile comparison
    m = (eta_hi > 0.05) & (eta_hi < 5.0) & (fp_ref_hi > 0.02) & (fp_ref_hi < 0.995)
    rel_err = np.abs(u_hi - u_ref_hi) / np.maximum(u_ref_hi, 1e-12)
    l2_rel = (
        float(np.linalg.norm(rel_err[m]) / np.linalg.norm(np.ones(m.sum())))
        if m.any()
        else float("nan")
    )

    # Edge velocity (at top of domain)
    u_edge_probe = float(ux_f[ny - 2, probe].item())

    # C_f measurement: 3-point one-sided FD on first 3 fluid rows
    # Correct weights: (-2, 3, -1) for s = (0.5, 1.5, 2.5)
    s_dist = np.array([0.5, 1.5, 2.5])
    A = np.vstack([np.ones(3), s_dist, s_dist**2]).T
    w_der = np.linalg.solve(A.T, np.array([0.0, 1.0, 0.0]))
    
    u123 = np.array([prof[1], prof[2], prof[3]])
    dudy = float(w_der @ u123)
    Cf_fd = 2.0 * nu * dudy / (U * U)
    Cf_ref = CF_COEFF / math.sqrt(Rex)
    cf_err_pct = (Cf_fd - Cf_ref) / Cf_ref * 100.0

    result = {
        "grid": grid_name,
        "nx": nx,
        "ny": ny,
        "plate_len": plate_len,
        "probe_x_eff": x_eff,
        "Re_x": Rex,
        "collision": collision,
        "tau": tau,
        "n_steps": n_steps,
        "cf_sim": Cf_fd,
        "cf_ref": Cf_ref,
        "cf_err_pct": cf_err_pct,
        "u_edge_over_U": u_edge_probe / U,
        "l2_rel_err_profile": l2_rel,
        "mass_drift_pct": mass_drift_pct,
        "finite": finite,
        "elapsed_sec": elapsed,
    }

    # Save result
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Grid: {grid_name}, Re_x={Rex:.0f}, C_f={Cf_fd:.6f}, C_f_ref={Cf_ref:.6f}, err={cf_err_pct:+.2f}%")
    print(f"  u_edge/U={u_edge_probe/U:.5f}, L2_err={l2_rel:.4f}, mass_drift={mass_drift_pct:.4f}%")
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_path", type=str)
    parser.add_argument("--grid", type=str, default="wall_medium")
    parser.add_argument("--collision", type=str, default="trt", choices=["bgk", "trt"])
    parser.add_argument("--U", type=float, default=0.05)
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--avg-steps", type=int, default=200)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()

    run_case(
        args.grid,
        args.collision,
        args.U,
        args.nu,
        args.steps,
        args.out_path,
        args.device,
        args.avg_steps,
        args.threads,
    )


if __name__ == "__main__":
    main()
