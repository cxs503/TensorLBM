#!/usr/bin/env python
"""Blasius flat plate — tall domain to minimize blockage effect.

Instead of changing to wall-mounted plate (which has BC issues), keep the
mid-domain plate but use a MUCH taller domain to minimize the channel
blockage effect.

Key insight: The blockage ratio = 2*δ*/ny. If we make ny very large,
the blockage becomes negligible and u_edge/U → 1.

For Re_x=900, δ* ≈ 10.32, so we need ny >> 2*10.32 = 20.64.
Let's try ny=5000 (blockage ratio = 20.64/5000 = 0.4%).
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
    """Fluid-side-reflection half-way BB for the thin plate row y=py."""
    f = f.clone()
    # upper surface
    f[2, py, :] = torch.where(mask, f[4, py + 1, :], f[2, py, :])
    f[5, py, :-1] = torch.where(mask[:-1], f[7, py + 1, 1:], f[5, py, :-1])
    f[6, py, 1:] = torch.where(mask[1:], f[8, py + 1, :-1], f[6, py, 1:])
    # lower surface
    f[4, py, :] = torch.where(mask, f[2, py - 1, :], f[4, py, :])
    f[7, py, 1:] = torch.where(mask[1:], f[5, py - 1, :-1], f[7, py, 1:])
    f[8, py, :-1] = torch.where(mask[:-1], f[6, py - 1, 1:], f[8, py, 1:])
    return f


def run_tall_domain():
    """Test with very tall domain to minimize blockage."""
    # Tall domain: nx=440, ny=5000, plate_len=400, probe at x_eff=180
    nx, ny = 440, 5000
    le, plate_len = 20, 400
    plate_end = le + plate_len
    probe = 200  # x_eff = 180
    py = ny // 2
    U, nu = 0.05, 0.01
    tau = 3.0 * nu + 0.5
    lam_trt = 3.0 / 16.0
    n_steps = 30000
    
    device = torch.device('cpu')
    torch.set_num_threads(32)
    
    # Plate at mid-domain
    plate = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    plate[py, le:plate_end] = True
    plate_row = plate[py]
    
    # Initial condition
    rho0 = torch.ones((ny, nx), device=device)
    ux0 = torch.full((ny, nx), U, device=device)
    uy0 = torch.zeros((ny, nx), device=device)
    f = equilibrium(rho0, ux0, uy0, device=device)
    initial_mass = float(f.sum().item())
    
    feq_in = equilibrium(
        torch.ones((ny, 1), device=device),
        torch.full((ny, 1), U, device=device),
        torch.zeros((ny, 1), device=device),
    )[:, :, 0].contiguous()
    
    spec = SPEC.to(device)
    
    def step_fn(f_):
        f_ = collide_bgk(f_, tau)
        f_ = plate_hwbb_mid(f_, py, plate_row)
        f_ = f_.clone()
        # Mirror BCs at top and bottom
        f_[:, 0, :] = f_[:, 1, :][spec]
        f_[:, -1, :] = f_[:, -2, :][spec]
        f_ = stream(f_)
        f_ = f_.clone()
        f_[:, :, 0] = feq_in
        f_ = zou_he_outlet_pressure(f_, 1.0)
        return f_
    
    print(f"Running tall domain test: {nx}x{ny}, {n_steps} steps...")
    t0 = time.time()
    for step in range(1, n_steps + 1):
        f = step_fn(f)
        if step % 5000 == 0:
            print(f"  Step {step}/{n_steps}")
    elapsed = time.time() - t0
    print(f"Simulation time: {elapsed:.2f}s")
    
    # Averaging
    prof_acc = torch.zeros(ny, device=device)
    n_avg = 200
    for _ in range(n_avg):
        f = step_fn(f)
        _, ux, _ = macroscopic(f)
        prof_acc += ux[:, probe]
    
    prof = (prof_acc / n_avg).cpu().numpy()
    
    rho, ux_f, uy_f = macroscopic(f)
    mass_drift_pct = (float(f.sum().item()) - initial_mass) / initial_mass * 100.0
    
    # Analysis
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
    
    # C_f measurement
    s_dist = np.array([0.5, 1.5, 2.5])
    A = np.vstack([np.ones(3), s_dist, s_dist**2]).T
    w_der = np.linalg.solve(A.T, np.array([0.0, 1.0, 0.0]))
    
    u123 = np.array([prof[py + 1], prof[py + 2], prof[py + 3]])
    dudy = float(w_der @ u123)
    Cf_fd = 2.0 * nu * dudy / (U * U)
    Cf_ref = CF_COEFF / math.sqrt(Rex)
    cf_err_pct = (Cf_fd - Cf_ref) / Cf_ref * 100.0
    
    u_edge = float(ux_f[ny - 2, probe].item())
    
    print(f"\nResults:")
    print(f"  Grid: {nx}x{ny}, plate_len={plate_len}, probe_x_eff={x_eff}")
    print(f"  Re_x = {Rex:.0f}")
    print(f"  C_f_sim = {Cf_fd:.6f}")
    print(f"  C_f_ref = {Cf_ref:.6f}")
    print(f"  C_f error = {cf_err_pct:+.2f}%")
    print(f"  u_edge/U = {u_edge/U:.5f}")
    print(f"  mass_drift = {mass_drift_pct:.4f}%")
    
    # Profile comparison
    m = (eta_hi > 0.05) & (eta_hi < 5.0) & (fp_ref_hi > 0.02) & (fp_ref_hi < 0.995)
    rel_err = np.abs(u_hi - u_ref_hi) / np.maximum(u_ref_hi, 1e-12)
    l2_rel = float(np.linalg.norm(rel_err[m]) / np.linalg.norm(np.ones(m.sum()))) if m.any() else float('nan')
    print(f"  L2 profile error = {l2_rel:.4f}")
    
    return {
        "grid": f"{nx}x{ny}",
        "nx": nx,
        "ny": ny,
        "plate_len": plate_len,
        "probe_x_eff": x_eff,
        "Re_x": Rex,
        "cf_sim": Cf_fd,
        "cf_ref": Cf_ref,
        "cf_err_pct": cf_err_pct,
        "u_edge_over_U": u_edge / U,
        "l2_rel_err": l2_rel,
        "mass_drift_pct": mass_drift_pct,
        "elapsed_sec": elapsed,
    }


if __name__ == "__main__":
    result = run_tall_domain()
    
    # Save result
    out_path = Path(__file__).parent / "results" / "R_tall_domain_5000.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to {out_path}")
