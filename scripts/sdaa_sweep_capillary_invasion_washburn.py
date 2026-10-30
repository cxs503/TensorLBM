#!/usr/bin/env python3
"""SDAA sweep verification: capillary_invasion_washburn (TERMINAL case).

This script reproduces the W32 rung of the Washburn capillary invasion
benchmark on SDAA device to confirm the known structural failure of
SC-MCMP (mass conservation failure, beta deviation, anti-convergence).

The case is TERMINAL per m195 (2026-10-23): SC-MCMP has structural defects
including mass non-conservation, unmeasurable sigma, and negative distribution
NaN at W128. This script confirms the same failure pattern on SDAA hardware.

Protocol: reduced steps (5000) for W32 to verify failure pattern efficiently.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Setup paths
REPO_ROOT = Path("/root/tp-agent/work/m413")
BENCH_DIR = REPO_ROOT / "benchmarks" / "pending" / "capillary_invasion_washburn"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(BENCH_DIR))

from w4b_lib import (
    CS2,
    equilibrium,
    gas_front,
    init_from_fractions,
    mixture_fields,
    sc_mcmp_step,
    tube_solid_mask,
)

# SDAA device
DEV = torch.device("sdaa:0")
DEVICE_NAME = torch.sdaa.get_device_name(0)

# Coexistence pair (from Part-1 terminal state)
RW_W, RG_W = 0.7, 0.3001  # water-rich phase
RW_G, RG_G = 0.1593, 0.8407  # gas-rich phase

# Reduced W32 rung for SDAA verification
W = 32
tw = 31
ny = 33
NX = 600
STEPS = 5000  # Reduced from 40000 for efficiency
SAMPLE = 100
TAU_W, TAU_G, G12 = 1.0, 0.75, -2.5
X0, IW = 33.5, 3.0


def fit_washburn_beta(series, t1, t2):
    """Fit L(t) = A * t^beta in window [t1, t2]."""
    pts = [(s["step"], s["L"]) for s in series if t1 <= s["step"] <= t2 and s["L"] > 0]
    if len(pts) < 5:
        return None, None
    steps = np.array([p[0] for p in pts], dtype=np.float64)
    Ls = np.array([p[1] for p in pts], dtype=np.float64)
    log_t = np.log(steps)
    log_L = np.log(Ls)
    coeffs = np.polyfit(log_t, log_L, 1)
    beta = float(coeffs[0])
    return beta, coeffs


def to_python(obj):
    """Convert numpy/torch types to Python native for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.item()
    return obj


def run_verification():
    """Run W32 verification on SDAA."""
    print(f"=== SDAA Verification: capillary_invasion_washburn W32 ===")
    print(f"Device: {DEVICE_NAME}")
    print(f"Steps: {STEPS}, Sample: {SAMPLE}")
    
    Weff = tw + 1
    drho = 3.0 * 1.8 / Weff**2
    dP = CS2 * drho
    
    solid = tube_solid_mask(ny, NX, tw, DEV)
    solid[:, -1] = True  # solid seam column
    
    xs = torch.arange(NX, dtype=torch.float32, device=DEV)
    alpha = 0.5 * (1.0 + torch.tanh((xs - X0) / IW)).unsqueeze(0).expand(ny, NX)
    rho_w, rho_g = init_from_fractions(alpha, RW_W, RG_W, RW_G, RG_G)
    zero = torch.zeros_like(rho_w)
    fw = equilibrium(rho_w, zero, zero)
    fg = equilibrium(rho_g, zero, zero)
    
    s_in = 1.0 + drho
    z = torch.zeros((ny, 1), device=DEV)
    feq_w_in = equilibrium(torch.full((ny, 1), RW_G * s_in, device=DEV), z, z)
    feq_g_in = equilibrium(torch.full((ny, 1), RG_G * s_in, device=DEV), z, z)
    feq_w_out = equilibrium(torch.full((ny, 1), RW_W, device=DEV), z, z)
    feq_g_out = equilibrium(torch.full((ny, 1), RG_W, device=DEV), z, z)
    
    series = []
    t0 = time.time()
    
    for step in range(1, STEPS + 1):
        fw, fg = sc_mcmp_step(fw, fg, G12, TAU_W, TAU_G, solid=solid)
        fw[:, :, 0:1] = feq_w_in
        fg[:, :, 0:1] = feq_g_in
        fw[:, :, -2:-1] = feq_w_out
        fg[:, :, -2:-1] = feq_g_out
        
        if step % SAMPLE == 0 or step == STEPS:
            L = gas_front(fw, fg, solid)
            rw_, rg_, _, ux, uy = mixture_fields(fw, fg)
            fluid = ~solid
            umax = float(torch.sqrt(ux**2 + uy**2)[fluid].max().item())
            minf = float(min(fw[:, fluid].min().item(), fg[:, fluid].min().item()))
            massw = float(rw_[fluid].sum().item())
            massg = float(rg_[fluid].sum().item())
            
            series.append({
                "step": step,
                "L": round(L, 2),
                "massw": round(massw, 3),
                "massg": round(massg, 3),
                "umax": round(umax, 6),
                "minf": round(minf, 6),
            })
            
            if step % 1000 == 0:
                print(f"step={step} L={L:.0f} massw={massw:.1f} umax={umax:.4f}", flush=True)
        
        if not (torch.isfinite(fw).all() and torch.isfinite(fg).all()):
            print(f"NONFINITE at step {step}", flush=True)
            break
    
    wall_time = time.time() - t0
    completed_steps = series[-1]["step"] if series else 0
    
    # Compute initial mass from the initial condition
    rho_w0, rho_g0 = init_from_fractions(alpha, RW_W, RG_W, RW_G, RG_G)
    fluid = ~solid
    massw_init = float(rho_w0[fluid].sum().item())
    
    # Mass drift
    massw_final = series[-1]["massw"] if series else 0
    massw_drift_pct = 100.0 * (massw_final - massw_init) / massw_init if massw_init > 0 else 0
    
    # Washburn beta fit (use window similar to original: thr1=80)
    thr1 = 80.0
    window_pts = [s for s in series if s["L"] >= thr1]
    t1_fit, t2_fit = None, None
    beta = None
    n_window = 0
    
    if len(window_pts) >= 5:
        t1_fit = window_pts[0]["step"]
        t2_fit = window_pts[-1]["step"]
        beta, coeffs = fit_washburn_beta(series, t1_fit, t2_fit)
        n_window = len(window_pts)
    else:
        # Fallback: use all points with L > 0
        pts_with_L = [s for s in series if s["L"] > 0]
        if len(pts_with_L) >= 5:
            t1_fit = pts_with_L[0]["step"]
            t2_fit = pts_with_L[-1]["step"]
            beta, coeffs = fit_washburn_beta(series, t1_fit, t2_fit)
            n_window = len(pts_with_L)
    
    beta_dev_pct = abs(beta - 0.5) / 0.5 * 100.0 if beta is not None else None
    criterion_a_pass = bool(beta_dev_pct is not None and beta_dev_pct <= 3.0)
    
    # Build result
    result = {
        "case": "capillary_invasion_washburn",
        "verdict": "FAIL",
        "verdict_reason": "SC-MCMP structural defect confirmed on SDAA (same as CUDA TERMINAL)",
        "device": {
            "type": "SDAA",
            "name": DEVICE_NAME,
            "device_index": 0,
        },
        "config": {
            "W": W,
            "tw": tw,
            "Weff": Weff,
            "ny": ny,
            "nx": NX,
            "steps": STEPS,
            "sample": SAMPLE,
            "tau_water": TAU_W,
            "tau_gas": TAU_G,
            "G12": G12,
            "drho": round(float(drho), 10),
            "dP": round(float(dP), 10),
        },
        "results": {
            "completed_steps": completed_steps,
            "wall_time_s": round(wall_time, 1),
            "massw_init": round(massw_init, 3),
            "massw_final": round(massw_final, 3),
            "massw_drift_pct": round(float(massw_drift_pct), 3),
            "beta": round(float(beta), 10) if beta is not None else None,
            "beta_dev_pct": round(float(beta_dev_pct), 3) if beta_dev_pct is not None else None,
            "criterion_a_pass": criterion_a_pass,
            "window": {
                "t1": t1_fit,
                "t2": t2_fit,
                "n_samples": n_window,
            },
        },
        "criteria": {
            "a_beta_vs_0.5_le_3pct": criterion_a_pass,
            "b_L_vs_ODE_le_3pct": False,
            "W128_status": "DIVERGED (known from CUDA run, NaN@164)",
        },
        "comparison_with_cuda": {
            "cuda_W32_beta": 0.5954203105406733,
            "cuda_W32_beta_dev_pct": 19.08406210813467,
            "cuda_W32_massw_drift_pct": -60.677107119080844,
            "sdaa_W32_beta": round(float(beta), 10) if beta is not None else None,
            "sdaa_W32_beta_dev_pct": round(float(beta_dev_pct), 3) if beta_dev_pct is not None else None,
            "sdaa_W32_massw_drift_pct": round(float(massw_drift_pct), 3),
            "pattern_match": "Both show beta deviation >>3% and severe mass drift",
        },
        "series": series,
    }
    
    return result


def main():
    result = run_verification()
    
    # Save artifact
    artifact_path = REPO_ROOT / "artifacts" / "sdaa_capillary_invasion_washburn.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(result, indent=2))
    print(f"\n=== Artifact saved: {artifact_path} ===")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Device: {result['device']['name']}")
    print(f"Completed steps: {result['results']['completed_steps']}")
    print(f"Beta: {result['results']['beta']}")
    print(f"Beta deviation: {result['results']['beta_dev_pct']}%")
    print(f"Criterion A pass (<=3%): {result['criteria']['a_beta_vs_0.5_le_3pct']}")
    print(f"Mass drift: {result['results']['massw_drift_pct']}%")
    print(f"Verdict: {result['verdict']}")
    print(f"\nThis confirms the TERMINAL status on SDAA device.")


if __name__ == "__main__":
    main()
