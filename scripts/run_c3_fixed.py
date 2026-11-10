#!/usr/bin/env python3
"""P96W3-N05: Fixed Taylor-Aris verification (no pulse wrapping)."""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from tensorlbm.scalar import (
    fit_diffusivity,
    poiseuille_velocity_profile,
    scalar_bounce_back,
    scalar_collide_bgk,
    scalar_macroscopic,
    scalar_stream,
    second_moment_x,
    init_scalar_line,
    taylor_aris_deff,
    tau_from_diffusivity,
)

DEVICE = torch.device("cpu")
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "p96w3_n05"


def run_c3_fixed(Pe, D, ny, nx, n_steps):
    """Run Taylor-Aris with fixed flow, ensuring no pulse wrapping."""
    half_height = (ny - 1) / 2.0
    H = float(ny - 1)
    U_mean = Pe * D / H
    u_max = 1.5 * U_mean
    tau_s = tau_from_diffusivity(D)
    
    # Check if pulse will wrap
    x_start = nx // 6
    x_end = x_start + U_mean * n_steps
    if x_end > nx * 0.8:
        print(f"  WARNING: Pulse will wrap! x_end={x_end:.0f} > nx*0.8={nx*0.8:.0f}")
        return None, None, None, None
    
    y = torch.arange(ny, device=DEVICE, dtype=torch.float32)
    ux = poiseuille_velocity_profile(y, ny, u_max).unsqueeze(1).expand(ny, nx)
    uy = torch.zeros(ny, nx, device=DEVICE)
    
    g = torch.zeros(5, ny, nx, device=DEVICE, dtype=torch.float32)
    g = init_scalar_line(g, ux, uy, x_start, 2.0, 1.0)
    
    solid_mask = torch.zeros(ny, nx, dtype=torch.bool, device=DEVICE)
    solid_mask[0, :] = True
    solid_mask[-1, :] = True
    
    history = []
    for step in range(n_steps):
        c = scalar_macroscopic(g)
        g = scalar_collide_bgk(g, c, ux, uy, tau_s)
        g = scalar_stream(g)
        g = scalar_bounce_back(g, solid_mask)
        if (step + 1) % 20 == 0:
            c_now = scalar_macroscopic(g)
            mom = second_moment_x(c_now)
            history.append({"time": float(step + 1), **mom})
    
    return history, U_mean, u_max, half_height


def main():
    print("C3: Taylor-Aris (fixed, no wrapping)")
    results = []
    D = 0.05
    
    # Use longer domains to avoid wrapping
    configs = [
        (10, 9, 256, 1500),   # Pe=10, longer domain
        (30, 13, 512, 2000),  # Pe=30, even longer
    ]
    
    for Pe, ny, nx, n_steps in configs:
        t0 = time.time()
        history, U_mean, u_max, half_height = run_c3_fixed(Pe, D, ny, nx, n_steps)
        elapsed = time.time() - t0
        
        if history is None:
            print(f"  Pe={Pe}: SKIPPED (wrapping)")
            continue
        
        times = [h["time"] for h in history]
        variances = [h["var_x"] for h in history]
        t_transition = half_height ** 2 / D
        fit = fit_diffusivity(times, variances, t_min=t_transition * 0.7)
        
        D_eff_meas = fit["D_eff"]
        D_eff_anal = taylor_aris_deff(D, U_mean, half_height)
        err = abs(D_eff_meas - D_eff_anal) / D_eff_anal
        
        results.append({
            "Pe": Pe,
            "D_eff_measured": D_eff_meas,
            "D_eff_analytical": D_eff_anal,
            "rel_error": err,
            "pass": err <= 0.05,
            "elapsed_s": round(elapsed, 1),
        })
        print(f"  Pe={Pe}: D_eff={D_eff_meas:.6f}, anal={D_eff_anal:.6f}, err={err:.4%} [{'PASS' if err<=0.05 else 'FAIL'}] ({elapsed:.1f}s)")
    
    (ARTIFACTS_DIR / "c3_results_fixed.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
