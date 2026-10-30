#!/usr/bin/env python
"""Quick test of wall-mounted Blasius plate — minimal grid for CPU."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from tensorlbm.boundaries import zou_he_outlet_pressure
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream

torch.set_num_threads(32)
SPEC = torch.tensor([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=torch.int64)


def blasius_tab(eta_max=14.0, h=0.002):
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    def rhs(e, s):
        return [s[1], s[2], -0.5 * s[0] * s[2]]

    def shoot(a):
        return (
            solve_ivp(rhs, [0, 14], [0, 0, a], rtol=1e-11, atol=1e-13, max_step=0.01).y[1, -1] - 1.0
        )

    a = brentq(shoot, 0.3, 0.37, xtol=1e-13)
    n = int(round(eta_max / h))
    s = np.zeros((3, n + 1))
    s[:, 0] = [0, 0, a]
    etas = np.arange(n + 1) * h
    for i in range(n):
        e = etas[i]
        k1 = np.array(rhs(e, s[:, i]))
        k2 = np.array(rhs(e + h / 2, s[:, i] + h / 2 * k1))
        k3 = np.array(rhs(e + h / 2, s[:, i] + h / 2 * k2))
        k4 = np.array(rhs(e + h, s[:, i] + h * k3))
        s[:, i + 1] = s[:, i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return etas, s[1], s[0], a


def run_quick():
    """Quick test with minimal grid."""
    # Minimal grid for quick CPU test
    nx, ny = 220, 150
    le, plate_len = 20, 180
    probe = 110  # x_eff = 90
    U, nu = 0.05, 0.01
    tau = 3 * nu + 0.5
    n_steps = 8000
    
    dev = torch.device('cpu')
    plate_end = le + plate_len
    
    # Wall-mounted plate at y=0
    solid = torch.zeros((ny, nx), dtype=torch.bool, device=dev)
    solid[0, le:plate_end] = True
    mask = solid[0]
    
    f = equilibrium(
        torch.ones((ny, nx), device=dev),
        torch.full((ny, nx), U, device=dev),
        torch.zeros((ny, nx), device=dev),
        device=dev,
    )
    m0 = float(f.sum().item())
    spec = SPEC.to(dev)
    feq_in = equilibrium(
        torch.ones((ny, 1), device=dev),
        torch.full((ny, 1), U, device=dev),
        torch.zeros((ny, 1), device=dev),
    )[:, :, 0].contiguous()
    
    def step(f_):
        f_ = collide_bgk(f_, tau)
        f_ = f_.clone()
        # Bottom wall bounce-back
        f_[2, 0, :] = torch.where(mask, f_[4, 1, :], f_[2, 0, :])
        f_[5, 0, :-1] = torch.where(mask[:-1], f_[7, 1, 1:], f_[5, 0, :-1])
        f_[6, 0, 1:] = torch.where(mask[1:], f_[8, 1, :-1], f_[6, 0, 1:])
        # Top boundary: symmetry
        f_[:, -1, :] = f_[:, -2, :][spec]
        f_ = stream(f_)
        f_ = f_.clone()
        # Inlet
        f_[:, :, 0] = feq_in
        # Outlet: Zou-He pressure
        f_ = zou_he_outlet_pressure(f_, 1.0)
        return f_
    
    print(f"Running wall-mounted plate: {nx}x{ny}, {n_steps} steps...")
    import time
    t0 = time.time()
    for i in range(1, n_steps + 1):
        f = step(f)
        if i % 2000 == 0:
            _, ux, _ = macroscopic(f)
            print(f"  step {i}: umax={float(ux.max()):.5f}", flush=True)
    elapsed = time.time() - t0
    print(f"Simulation time: {elapsed:.2f}s")
    
    # Average profiles
    acc = torch.zeros(ny, device=dev)
    navg = 100
    for _ in range(navg):
        f = step(f)
        _, ux, _ = macroscopic(f)
        acc += ux[:, probe]
    acc /= navg
    
    _, ux_f, _ = macroscopic(f)
    mass_drift = (float(f.sum().item()) - m0) / m0 * 100.0
    
    # Analysis
    etas, fprime, ffun, fpp0 = blasius_tab()
    
    x_eff = probe - le
    Rex = U * x_eff / nu
    scale = math.sqrt(nu * x_eff / U)
    
    y_w = 0.5
    rows = np.arange(1, ny)
    y = rows - y_w
    eta = y / scale
    prof = acc.cpu().numpy()[rows]
    ue = float(np.mean(acc.cpu().numpy()[-20:]))
    fp_ref = np.interp(eta, etas, fprime)
    u_ref = U * fp_ref
    
    # Profile comparison
    m = (eta > 0.05) & (eta < 5.0)
    rel = np.abs(prof - u_ref) / np.maximum(u_ref, 1e-12)
    l2 = float(np.linalg.norm(rel[m]) / np.linalg.norm(np.ones(m.sum())))
    
    # C_f measurement
    def q_deriv(u3):
        return float(
            np.linalg.solve(
                np.vstack([np.ones(3), [0.5, 1.5, 2.5], [0.25, 2.25, 6.25]]).T, [0, 1, 0]
            )
            @ u3
        )
    
    Cf_ref = 0.6641146724 / math.sqrt(Rex)
    Cf1 = 2 * nu * (prof[0] / 0.5) / U**2
    Cf2 = 2 * nu * q_deriv(prof[0:3]) / U**2
    
    print(f"\nResults:")
    print(f"  Grid: {nx}x{ny}, plate_len={plate_len}, probe_x_eff={x_eff}")
    print(f"  Re_x = {Rex:.0f}")
    print(f"  C_f_ref = {Cf_ref:.6f}")
    print(f"  C_f_1st = {Cf1:.6f}, err = {(Cf1 - Cf_ref) / Cf_ref * 100:+.2f}%")
    print(f"  C_f_2nd = {Cf2:.6f}, err = {(Cf2 - Cf_ref) / Cf_ref * 100:+.2f}%")
    print(f"  u_edge/U = {ue / U:.5f}")
    print(f"  L2 profile error = {l2:.4f}")
    print(f"  mass_drift = {mass_drift:.4f}%")
    
    return {
        "grid": f"{nx}x{ny}",
        "nx": nx,
        "ny": ny,
        "plate_len": plate_len,
        "probe_x_eff": x_eff,
        "Re_x": Rex,
        "cf_ref": Cf_ref,
        "cf_1st": Cf1,
        "cf_err_1st_pct": (Cf1 - Cf_ref) / Cf_ref * 100,
        "cf_2nd": Cf2,
        "cf_err_2nd_pct": (Cf2 - Cf_ref) / Cf_ref * 100,
        "u_edge_over_U": ue / U,
        "l2_rel_err": l2,
        "mass_drift_pct": mass_drift,
        "elapsed_sec": elapsed,
    }


if __name__ == "__main__":
    result = run_quick()
    
    # Save result
    out_path = Path(__file__).parent / "results" / "R_wall_quick_cpu.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to {out_path}")
