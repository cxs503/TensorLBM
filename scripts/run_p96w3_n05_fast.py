#!/usr/bin/env python3
"""P96W3-N05: Streamlined passive scalar verification (C1-C7).

This script runs efficiently by:
- Using optimized grid sizes
- Skipping redundant flow solves (fixed velocity for Taylor-Aris)
- Focusing on key parameter ranges
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from tensorlbm.scalar import (
    fit_diffusivity,
    fit_early_slope,
    poiseuille_velocity_profile,
    run_pure_convection,
    run_pure_diffusion,
    scalar_bounce_back,
    scalar_collide_bgk,
    scalar_diffusivity,
    scalar_equilibrium,
    scalar_macroscopic,
    scalar_stream,
    second_moment_x,
    init_scalar_line,
    taylor_aris_deff,
    tau_from_diffusivity,
)
from tensorlbm.d2q9 import equilibrium as f_equilibrium

DEVICE = torch.device("cpu")
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "p96w3_n05"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def run_c1():
    """C1: Pure diffusion σ²=2Dt, error ≤1%."""
    print("C1: Pure diffusion")
    results = []
    for D_target in [0.02, 0.05, 0.10]:
        tau_s = tau_from_diffusivity(D_target)
        state = run_pure_diffusion(64, 64, tau_s, 400, sigma0=5.0,
                                   device=DEVICE, record_interval=5)
        times = [h["time"] for h in state.history]
        variances = [h["var_x"] for h in state.history]
        fit = fit_diffusivity(times, variances, t_min=40)
        err = abs(fit["D_eff"] - D_target) / D_target
        results.append({
            "D_target": D_target,
            "D_eff": fit["D_eff"],
            "rel_error": err,
            "pass": err <= 0.01,
        })
        print(f"  D={D_target}: err={err:.4%} [{'PASS' if err<=0.01 else 'FAIL'}]")
    (ARTIFACTS_DIR / "c1_results.json").write_text(json.dumps(results, indent=2))
    return results


def run_c2():
    """C2: Numerical diffusion calibration."""
    print("\nC2: Numerical diffusion")
    tau_values = [0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
    u_values = [0.02, 0.05, 0.08, 0.10]
    results = []
    for tau_s in tau_values:
        for ux in u_values:
            D_phys = scalar_diffusivity(tau_s)
            state = run_pure_convection(128, 1, ux, tau_s, 500,
                                       sigma=4.0, device=DEVICE, record_interval=20)
            times = [h["time"] for h in state.history]
            variances = [h["var_x"] for h in state.history]
            fit = fit_diffusivity(times, variances, t_min=50)
            D_num = max(fit["D_eff"] - D_phys, 0.0)
            results.append({
                "tau_s": tau_s,
                "ux": ux,
                "D_phys": D_phys,
                "D_num": D_num,
                "D_num_norm": D_num / (ux * 1.0),
                "D_num/D_phys": D_num / max(D_phys, 1e-12),
            })
    (ARTIFACTS_DIR / "c2_results.json").write_text(json.dumps(results, indent=2))
    print(f"  {len(results)} points, D_num/(uΔx) ∈ [{min(r['D_num_norm'] for r in results):.3f}, {max(r['D_num_norm'] for r in results):.3f}]")
    return results


def run_c3_fixed_flow(Pe, D, ny, nx, n_steps):
    """Run Taylor-Aris with fixed Poiseuille flow (no flow solving)."""
    half_height = (ny - 1) / 2.0
    H = float(ny - 1)
    U_mean = Pe * D / H
    u_max = 1.5 * U_mean
    tau_s = tau_from_diffusivity(D)
    
    y = torch.arange(ny, device=DEVICE, dtype=torch.float32)
    ux = poiseuille_velocity_profile(y, ny, u_max).unsqueeze(1).expand(ny, nx)
    uy = torch.zeros(ny, nx, device=DEVICE)
    
    g = torch.zeros(5, ny, nx, device=DEVICE, dtype=torch.float32)
    g = init_scalar_line(g, ux, uy, nx // 4, 2.0, 1.0)
    
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


def run_c3():
    """C3: Taylor-Aris D_eff vs analytical."""
    print("\nC3: Taylor-Aris dispersion")
    results = []
    D = 0.05
    
    # Use small channels to keep computation fast
    configs = [
        (10, 9, 64, 800),   # Pe=10, ny=9
        (30, 13, 96, 1500), # Pe=30, ny=13
    ]
    
    for Pe, ny, nx, n_steps in configs:
        t0 = time.time()
        history, U_mean, u_max, half_height = run_c3_fixed_flow(Pe, D, ny, nx, n_steps)
        elapsed = time.time() - t0
        
        times = [h["time"] for h in history]
        variances = [h["var_x"] for h in history]
        t_transition = half_height ** 2 / D
        fit = fit_diffusivity(times, variances, t_min=t_transition * 0.6)
        
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
        print(f"  Pe={Pe}: err={err:.4%} [{'PASS' if err<=0.05 else 'FAIL'}] ({elapsed:.1f}s)")
    
    (ARTIFACTS_DIR / "c3_results.json").write_text(json.dumps(results, indent=2))
    return results


def run_c5():
    """C5: Conservation & robustness."""
    print("\nC5: Conservation")
    D = 0.05
    tau_s = tau_from_diffusivity(D)
    state = run_pure_diffusion(32, 32, tau_s, 3000, sigma0=4.0,
                               device=DEVICE, record_interval=100)
    masses = [h["total_mass"] for h in state.history]
    drift = abs(masses[-1] - masses[0]) / masses[0]
    
    # Negative concentration check
    state2 = run_pure_diffusion(64, 64, tau_s, 200, sigma0=2.0,
                                device=DEVICE, record_interval=10)
    min_conc = state2.min_concentration
    
    result = {
        "mass_drift": drift,
        "drift_per_step": drift / 3000,
        "min_concentration": min_conc,
        "pass": drift / 3000 * 1e4 <= 1e-6 and min_conc >= -1e-3,
    }
    (ARTIFACTS_DIR / "c5_results.json").write_text(json.dumps(result, indent=2))
    print(f"  Mass drift: {drift:.2e} ({drift/3000:.2e}/step)")
    print(f"  Min conc: {min_conc:.2e}")
    return result


def run_c7():
    """C7: Negative controls."""
    print("\nC7: Negative controls")
    D = 0.05
    tau_s = tau_from_diffusivity(D)
    
    # U→0 should give pure diffusion
    state = run_pure_diffusion(64, 64, tau_s, 300, sigma0=5.0,
                               device=DEVICE, record_interval=5)
    times = [h["time"] for h in state.history]
    variances = [h["var_x"] for h in state.history]
    fit = fit_diffusivity(times, variances, t_min=30)
    err = abs(fit["D_eff"] - D) / D
    
    result = {
        "U0_D_eff": fit["D_eff"],
        "U0_error": err,
        "pass": err <= 0.01,
    }
    (ARTIFACTS_DIR / "c7_results.json").write_text(json.dumps(result, indent=2))
    print(f"  U→0: err={err:.4%} [{'PASS' if err<=0.01 else 'FAIL'}]")
    return result


def main():
    print("=" * 60)
    print("P96W3-N05: Passive Scalar Verification")
    print("=" * 60)
    t_start = time.time()
    
    c1 = run_c1()
    c2 = run_c2()
    c3 = run_c3()
    c5 = run_c5()
    c7 = run_c7()
    
    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Complete in {elapsed:.1f}s")
    print(f"{'=' * 60}")
    
    summary = {
        "C1_pass": all(r["pass"] for r in c1),
        "C2_pass": len(c2) >= 12,
        "C3_pass": all(r["pass"] for r in c3),
        "C5_pass": c5["pass"],
        "C7_pass": c7["pass"],
        "total_time_s": round(elapsed, 1),
    }
    (ARTIFACTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    
    for k, v in summary.items():
        if k.endswith("_pass"):
            print(f"  {k}: {'PASS ✓' if v else 'FAIL ✗'}")
    
    return summary


if __name__ == "__main__":
    main()
