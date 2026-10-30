#!/usr/bin/env python
"""Quick test of wall-mounted Blasius flat plate — optimized for CPU.

Simplified version with minimal overhead for fast testing.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from tensorlbm.boundaries import zou_he_outlet_pressure
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream

SPEC = torch.tensor([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=torch.int64)
CF_COEFF = 2.0 * 0.3320573362151963


def blasius_table(eta_max: float = 10.0, h: float = 0.005):
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


def run_quick_test():
    """Quick test with small grid and few steps."""
    # Small grid: nx=120, ny=100, plate from x=10 to x=60, probe at x=35
    nx, ny = 120, 100
    le, plate_len = 10, 50
    plate_end = le + plate_len
    probe = 35
    U, nu = 0.05, 0.01
    tau = 3.0 * nu + 0.5
    n_steps = 5000
    
    device = torch.device('cpu')
    torch.set_num_threads(16)
    
    # Plate at bottom wall (y=0)
    plate_mask = torch.zeros(nx, dtype=torch.bool, device=device)
    plate_mask[le:plate_end] = True
    non_plate_mask = ~plate_mask
    
    # Initial condition
    rho0 = torch.ones((ny, nx), device=device)
    ux0 = torch.full((ny, nx), U, device=device)
    uy0 = torch.zeros((ny, nx), device=device)
    f = equilibrium(rho0, ux0, uy0, device=device)
    
    feq_in = equilibrium(
        torch.ones((ny, 1), device=device),
        torch.full((ny, 1), U, device=device),
        torch.zeros((ny, 1), device=device),
    )[:, :, 0].contiguous()
    
    spec = SPEC.to(device)
    
    t0 = time.time()
    for step in range(1, n_steps + 1):
        # Collision
        f = collide_bgk(f, tau)
        
        # Bounce-back at bottom wall (plate region)
        f[2, 1, :] = torch.where(plate_mask, f[4, 1, :], f[2, 1, :])
        f[5, 1, :-1] = torch.where(plate_mask[:-1], f[7, 1, 1:], f[5, 1, :-1])
        f[6, 1, 1:] = torch.where(plate_mask[1:], f[8, 1, :-1], f[6, 1, 1:])
        
        # Bounce-back at bottom wall (non-plate region)
        f[2, 0, :] = torch.where(non_plate_mask, f[4, 0, :], f[2, 0, :])
        f[5, 0, :-1] = torch.where(non_plate_mask[:-1], f[7, 0, 1:], f[5, 0, :-1])
        f[6, 0, 1:] = torch.where(non_plate_mask[1:], f[8, 0, :-1], f[6, 0, 1:])
        
        # Streaming
        f = stream(f)
        
        # Inlet
        f[:, :, 0] = feq_in
        
        # Outlet
        f = zou_he_outlet_pressure(f, 1.0)
        
        # Top boundary (symmetry)
        f[2, -1, :] = f[4, -2, :]
        f[5, -1, :-1] = f[7, -2, 1:]
        f[6, -1, 1:] = f[8, -2, :-1]
    
    elapsed = time.time() - t0
    print(f"Simulation time: {elapsed:.2f}s for {n_steps} steps")
    
    # Averaging
    prof_acc = torch.zeros(ny, device=device)
    n_avg = 50
    for _ in range(n_avg):
        # Same step as above
        f = collide_bgk(f, tau)
        f[2, 1, :] = torch.where(plate_mask, f[4, 1, :], f[2, 1, :])
        f[5, 1, :-1] = torch.where(plate_mask[:-1], f[7, 1, 1:], f[5, 1, :-1])
        f[6, 1, 1:] = torch.where(plate_mask[1:], f[8, 1, :-1], f[6, 1, 1:])
        f[2, 0, :] = torch.where(non_plate_mask, f[4, 0, :], f[2, 0, :])
        f[5, 0, :-1] = torch.where(non_plate_mask[:-1], f[7, 0, 1:], f[5, 0, :-1])
        f[6, 0, 1:] = torch.where(non_plate_mask[1:], f[8, 0, :-1], f[6, 0, 1:])
        f = stream(f)
        f[:, :, 0] = feq_in
        f = zou_he_outlet_pressure(f, 1.0)
        f[2, -1, :] = f[4, -2, :]
        f[5, -1, :-1] = f[7, -2, 1:]
        f[6, -1, 1:] = f[8, -2, :-1]
        
        _, ux, _ = macroscopic(f)
        prof_acc += ux[:, probe]
    
    prof = (prof_acc / n_avg).cpu().numpy()
    
    # Analysis
    x_eff = probe - le
    Rex = U * x_eff / nu
    scale = math.sqrt(nu * x_eff / U)
    etas, fprimes, fpp0 = blasius_table()
    
    y_w = 0.5
    y_hi = np.arange(1, ny, dtype=np.float64) - y_w
    eta_hi = y_hi / scale
    fp_ref_hi = np.interp(eta_hi, etas, fprimes)
    u_ref_hi = U * fp_ref_hi
    u_hi = prof[1:]
    
    # C_f measurement
    s_dist = np.array([0.5, 1.5, 2.5])
    A = np.vstack([np.ones(3), s_dist, s_dist**2]).T
    w_der = np.linalg.solve(A.T, np.array([0.0, 1.0, 0.0]))
    
    u123 = np.array([prof[1], prof[2], prof[3]])
    dudy = float(w_der @ u123)
    Cf_fd = 2.0 * nu * dudy / (U * U)
    Cf_ref = CF_COEFF / math.sqrt(Rex)
    cf_err_pct = (Cf_fd - Cf_ref) / Cf_ref * 100.0
    
    u_edge = float(prof[-2])
    
    print(f"\nResults:")
    print(f"  Grid: {nx}x{ny}, plate_len={plate_len}, probe_x_eff={x_eff}")
    print(f"  Re_x = {Rex:.0f}")
    print(f"  C_f_sim = {Cf_fd:.6f}")
    print(f"  C_f_ref = {Cf_ref:.6f}")
    print(f"  C_f error = {cf_err_pct:+.2f}%")
    print(f"  u_edge/U = {u_edge/U:.5f}")
    
    # Profile comparison
    m = (eta_hi > 0.05) & (eta_hi < 5.0) & (fp_ref_hi > 0.02) & (fp_ref_hi < 0.995)
    rel_err = np.abs(u_hi - u_ref_hi) / np.maximum(u_ref_hi, 1e-12)
    l2_rel = float(np.linalg.norm(rel_err[m]) / np.linalg.norm(np.ones(m.sum()))) if m.any() else float('nan')
    print(f"  L2 profile error = {l2_rel:.4f}")
    
    return {
        "cf_sim": Cf_fd,
        "cf_ref": Cf_ref,
        "cf_err_pct": cf_err_pct,
        "u_edge_over_U": u_edge / U,
        "l2_rel_err": l2_rel,
        "Re_x": Rex,
    }


if __name__ == "__main__":
    result = run_quick_test()
    
    # Save result
    out_path = Path(__file__).parent / "results" / "R_wall_quick_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to {out_path}")
