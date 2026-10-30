#!/usr/bin/env python3
"""m467: Focused study on Blasius C_f error with profile fitting method.

Key insight: Standard FD methods give ~27% error. Profile fitting gives ~37% 
with only 1000 steps (not converged). Let's test with converged flows.
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
from tensorlbm.solver import collide_bgk, collide_trt, stream

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


def plate_hwbb_mid(f: torch.Tensor, py: int, mask: torch.Tensor) -> torch.Tensor:
    f = f.clone()
    f[2, py, :] = torch.where(mask, f[4, py + 1, :], f[2, py, :])
    f[5, py, :-1] = torch.where(mask[:-1], f[7, py + 1, 1:], f[5, py, :-1])
    f[6, py, 1:] = torch.where(mask[1:], f[8, py + 1, :-1], f[6, py, 1:])
    f[4, py, :] = torch.where(mask, f[2, py - 1, :], f[4, py, :])
    f[7, py, 1:] = torch.where(mask[1:], f[5, py - 1, :-1], f[7, py, 1:])
    f[8, py, :-1] = torch.where(mask[:-1], f[6, py - 1, 1:], f[8, py, :-1])
    return f


def run_quick(grid_name, collision, U, nu, n_steps, device_str="cpu"):
    """Quick run with minimal diagnostics."""
    grids = {
        "plate200": dict(nx=240, ny=1400, le=20, plate_len=200, probe=200),
        "plate400": dict(nx=440, ny=1400, le=20, plate_len=400, probe=200),
        "plate400_y1600": dict(nx=440, ny=1600, le=20, plate_len=400, probe=200),
    }
    g = grids[grid_name]
    nx, ny = g["nx"], g["ny"]
    le, plate_len, probe = g["le"], g["plate_len"], g["probe"]
    plate_end = le + plate_len
    py = ny // 2
    tau = 3.0 * nu + 0.5
    device = torch.device(device_str)
    torch.set_num_threads(32)
    torch.manual_seed(0)

    lam_trt = 3.0 / 16.0

    plate = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    plate[py, le:plate_end] = True
    plate_row = plate[py]

    rho0 = torch.ones((ny, nx), device=device)
    ux0 = torch.full((ny, nx), U, device=device)
    uy0 = torch.zeros((ny, nx), device=device)
    f = equilibrium(rho0, ux0, uy0, device=device)

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

    # Average over 100 steps
    prof_acc = torch.zeros(ny, device=device)
    for _ in range(100):
        f = step_fn(f)
        _, ux, _ = macroscopic(f)
        prof_acc += ux[:, probe]
    prof = (prof_acc / 100).cpu().numpy()

    x_eff = probe - le
    Rex = U * x_eff / nu
    scale = math.sqrt(nu * x_eff / U)
    etas, fprimes, fpp0 = blasius_table()

    y_w = py + 0.5
    y_hi = np.arange(py + 1, ny, dtype=np.float64) - y_w
    eta_hi = y_hi / scale
    fp_ref_hi = np.interp(eta_hi, etas, fprimes)
    u_ref_hi = U * fp_ref_hi
    u_hi = prof[py + 1:]

    # FD method
    s_dist = np.array([0.5, 1.5, 2.5])
    A = np.vstack([np.ones(3), s_dist, s_dist**2]).T
    w_der = np.linalg.solve(A.T, np.array([0.0, 1.0, 0.0]))
    u123 = np.array([prof[py + 1], prof[py + 2], prof[py + 3]])
    dudy_fd = float(w_der @ u123)
    Cf_fd = 2.0 * nu * dudy_fd / (U * U)

    # Fit method
    m_fit = (eta_hi > 0.1) & (eta_hi < 4.0) & (fp_ref_hi > 0.01) & (fp_ref_hi < 0.99)
    if m_fit.sum() > 10:
        u_fit = u_hi[m_fit]
        uref_fit = u_ref_hi[m_fit]
        A_fit = float(np.dot(u_fit, uref_fit) / np.dot(uref_fit, uref_fit))
        dudy_fit = A_fit * U * fpp0 / scale
        Cf_fit = 2.0 * nu * dudy_fit / (U * U)
    else:
        Cf_fit = float('nan')
        A_fit = float('nan')

    Cf_ref = CF_COEFF / math.sqrt(Rex)

    return {
        "grid": grid_name,
        "collision": collision,
        "n_steps": n_steps,
        "elapsed_s": float(elapsed),
        "Rex": float(Rex),
        "Cf_ref": float(Cf_ref),
        "Cf_fd": float(Cf_fd),
        "Cf_fd_err_pct": float((Cf_fd - Cf_ref) / Cf_ref * 100),
        "Cf_fit": float(Cf_fit),
        "Cf_fit_err_pct": float((Cf_fit - Cf_ref) / Cf_ref * 100),
        "A_fit": float(A_fit),
    }


# Test with existing best case
print("Testing baseline (plate400_y1600, TRT, 30000 steps)...")
res = run_quick("plate400_y1600", "trt", 0.05, 0.01, 30000, "cpu")
print(f"  C_f ref: {res['Cf_ref']:.6f}")
print(f"  C_f FD:  {res['Cf_fd']:.6f} (err: {res['Cf_fd_err_pct']:+.2f}%)")
print(f"  C_f fit: {res['Cf_fit']:.6f} (err: {res['Cf_fit_err_pct']:+.2f}%)")
print(f"  A_fit:   {res['A_fit']:.5f}")

# Save
with open("results/m467_quick_test.json", "w") as f:
    json.dump(res, f, indent=2)
print("\nSaved to results/m467_quick_test.json")
