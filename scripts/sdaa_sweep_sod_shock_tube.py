#!/usr/bin/env python
"""SDAA verification script for Sod shock tube case.

This script verifies the D2Q9 isothermal Sod shock tube on SDAA device,
comparing against the reference results from the verified CPU run.

Criteria: |error| < 3% relative to analytical solution
"""

import json
import math
import os
import sys

import numpy as np
import torch

# Add TensorLBM source to path
sys.path.insert(0, "/root/tp-agent/work/m433/src")

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream

CS = 1.0 / math.sqrt(3.0)
HERE = "/root/tp-agent/work/m433/benchmarks/verified/sod_shock_tube"
ARTIFACTS = "/root/tp-agent/work/m433/artifacts"

# Reference values from result.json
REFERENCE = {
    "nx2000": {
        "L2_rho_final": 0.008418001104873321,
        "L2_u_final": 0.018114360285785554,
        "W_rel": -0.0009030588349431724,
        "rho_mid_rel": -0.00016196204157230883,
        "u_mid_rel": -0.00040900113888572956,
        "osc_frac_final": 0.02765810489654541,
    },
    "nx4000": {
        "L2_rho_final": 0.004579364006749488,
        "L2_u_final": 0.008717167703951109,
        "W_rel": -0.0002886080716174575,
        "rho_mid_rel": -6.486599890598596e-05,
        "u_mid_rel": -0.0002489643207036109,
        "osc_frac_final": 0.026291569073994953,
    },
}


def iso_riemann(rhoL, rhoR, cs=CS):
    """Solve isothermal Riemann problem."""
    a = rhoL / rhoR
    f = lambda r: math.log(a / r) - (r - 1.0) / math.sqrt(r)
    lo, hi = 1.0 + 1e-12, a
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    r = 0.5 * (lo + hi)
    return r * rhoR, cs * (r - 1.0) / math.sqrt(r), cs * math.sqrt(r)


def ana_profile(x, t, rhoL, rhoR, rs, us, W, xd, cs=CS):
    """Compute analytical profile for isothermal Riemann problem."""
    s = (x - xd) / max(t, 1e-12)
    rho = np.empty_like(x)
    u = np.empty_like(x)
    m1 = s < -cs
    m2 = (s >= -cs) & (s < us - cs)
    m3 = (s >= us - cs) & (s < W)
    m4 = s >= W
    rho[m1] = rhoL
    u[m1] = 0.0
    rho[m2] = rhoL * np.exp(-(s[m2] + cs) / cs)
    u[m2] = s[m2] + cs
    rho[m3] = rs
    u[m3] = us
    rho[m4] = rhoR
    u[m4] = 0.0
    return rho, u


def run_simulation(nx, ny, rhoL, rhoR, tau, steps, snapshots, device):
    """Run the Sod shock tube simulation on specified device."""
    xd = nx // 2
    x = torch.arange(nx, dtype=torch.float32, device=device)
    rho = torch.where(x < xd, torch.full_like(x, rhoL), torch.full_like(x, rhoR))
    rho = rho.unsqueeze(0).expand(ny, nx)
    ux = torch.zeros(ny, nx, dtype=torch.float32, device=device)
    uy = torch.zeros(ny, nx, dtype=torch.float32, device=device)
    f = equilibrium(rho, ux, uy)
    f = f.to(device)
    profs = {}
    bad = None
    for t in range(steps + 1):
        if t in snapshots:
            rr, uu, _ = macroscopic(f)
            profs[t] = (rr[0].cpu().numpy().copy(), uu[0].cpu().numpy().copy())
        f = stream(f)
        f = collide_bgk(f, tau)
        if t % 50 == 0 and bad is None:
            rr, _, _ = macroscopic(f)
            if torch.isnan(rr).any() or (rr < 0).any():
                bad = t
    rr, uu, _ = macroscopic(f)
    if bad is None and (torch.isnan(rr).any() or (rr < 0).any()):
        bad = steps
    return profs, bad


def compute_metrics(profs, rhoL, rhoR, rs, us, W, xd, nx, tlist):
    """Compute error metrics against analytical solution."""
    x = np.arange(nx, dtype=np.float64)
    out = {}
    for t in tlist:
        rho, u = profs[t]
        rhoA, uA = ana_profile(x, t, rhoL, rhoR, rs, us, W, xd)
        lo, hi = int(math.ceil(CS * t)) + 1, int(nx - W * t) - 1
        mask = np.zeros(nx, bool)
        mask[max(lo, 0) : min(hi, nx)] = True
        jump = rhoL - rhoR
        L2r = float(np.sqrt(np.mean((rho[mask] - rhoA[mask]) ** 2)) / jump)
        L2u = float(np.sqrt(np.mean((u[mask] - uA[mask]) ** 2)) / max(us, 1e-12))
        # smooth-region L2 (exclude shock transition +/-10 cells)
        wlo, whi = int(xd + 0.5 * W * t), int(xd + 1.5 * W * t)
        g = np.abs(np.gradient(rho))
        g[:wlo] = 0.0
        g[whi:] = 0.0
        xsh = int(np.argmax(g))
        mask2 = mask.copy()
        mask2[max(0, xsh - 10) : min(nx, xsh + 10)] = False
        L2r_smooth = float(np.sqrt(np.mean((rho[mask2] - rhoA[mask2]) ** 2)) / jump)
        # local shock oscillation (window +-25)
        win = rho[max(0, xsh - 25) : min(nx, xsh + 25)]
        osc = float(np.max(win) - rs) / jump  # overshoot above middle state
        # middle-state check
        tail = xd + (us - CS) * t
        mid = (x > tail + 10) & (x < xsh - 10)
        u_mid = float(u[mid].mean()) if mid.any() else float("nan")
        rho_mid = float(rho[mid].mean()) if mid.any() else float("nan")
        out[t] = dict(
            L2_rho=L2r,
            L2_u=L2u,
            L2_rho_smooth=L2r_smooth,
            shock_x=xsh,
            osc_frac=osc,
            rho_mid=rho_mid,
            u_mid=u_mid,
        )
    ts = sorted(out.keys())
    # linear fit of shock position vs t (all snapshots)
    A = np.vstack([np.array(ts, float), np.ones(len(ts))]).T
    coeff, *_ = np.linalg.lstsq(A, [out[t]["shock_x"] for t in ts], rcond=None)
    W_fit = float(coeff[0])
    out["W_fit"] = W_fit
    out["W_ana"] = W
    out["W_rel"] = (W_fit - W) / W
    out["rho_mid_rel"] = (out[ts[-1]]["rho_mid"] - rs) / rs
    out["u_mid_rel"] = (out[ts[-1]]["u_mid"] - us) / us
    out["L2_rho_final"] = out[ts[-1]]["L2_rho"]
    out["L2_u_final"] = out[ts[-1]]["L2_u"]
    out["osc_frac_final"] = out[ts[-1]]["osc_frac"]
    return out


def check_device():
    """Check SDAA device availability."""
    print("=" * 60)
    print("Device Check")
    print("=" * 60)
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    # Check SDAA
    sdaa_available = hasattr(torch, 'sdaa') and torch.sdaa.is_available()
    print(f"SDAA available: {sdaa_available}")
    
    if sdaa_available:
        device_count = torch.sdaa.device_count()
        print(f"SDAA device count: {device_count}")
        for i in range(device_count):
            props = torch.sdaa.get_device_properties(i)
            print(f"  Device {i}: {props.name}")
        return torch.sdaa
    else:
        print("ERROR: SDAA not available!")
        return None


def run_verification():
    """Run the full verification on SDAA."""
    # Check device
    sdaa = check_device()
    if sdaa is None:
        return None
    
    device = torch.device("sdaa:0")
    print(f"\nUsing device: {device}")
    
    results = {
        "device": "SDAA",
        "device_name": torch.sdaa.get_device_properties(0).name,
        "sod_cases": [],
    }
    
    # Run simulations for both resolutions
    test_configs = [
        (2000, 400, [100, 200, 300, 400]),
        (4000, 800, [200, 400, 600, 800]),
    ]
    
    for nx, steps, snaps in test_configs:
        print(f"\n{'=' * 60}")
        print(f"Running nx={nx}, steps={steps}")
        print("=" * 60)
        
        rhoL, rhoR = 1.0, 0.25
        profs, bad = run_simulation(nx, 4, rhoL, rhoR, 0.8, steps, snaps, device)
        rs, us, W = iso_riemann(rhoL, rhoR)
        m = compute_metrics(profs, rhoL, rhoR, rs, us, W, nx // 2, nx, snaps)
        
        case_result = {
            "nx": nx,
            "steps": steps,
            "first_bad_step": bad,
            "rho_star": rs,
            "u_star": us,
            "Ma_mid": us / CS,
            "per_t": {str(t): {k2: v2 for k2, v2 in m[t].items()} for t in snaps},
            "L2_rho_final": m["L2_rho_final"],
            "L2_u_final": m["L2_u_final"],
            "W_fit": m["W_fit"],
            "W_ana": m["W_ana"],
            "W_rel": m["W_rel"],
            "rho_mid_rel": m["rho_mid_rel"],
            "u_mid_rel": m["u_mid_rel"],
            "osc_frac_final": m["osc_frac_final"],
        }
        
        results["sod_cases"].append(case_result)
        
        # Print results
        print(f"  L2_rho = {m['L2_rho_final']*100:.4f}%")
        print(f"  L2_u = {m['L2_u_final']*100:.4f}%")
        print(f"  W_rel = {m['W_rel']*100:.4f}%")
        print(f"  rho_mid_rel = {m['rho_mid_rel']*100:.4f}%")
        print(f"  u_mid_rel = {m['u_mid_rel']*100:.4f}%")
        print(f"  osc_frac = {m['osc_frac_final']*100:.4f}%")
        print(f"  first_bad_step = {bad}")
    
    # Check criteria
    print(f"\n{'=' * 60}")
    print("Criteria Check (|error| < 3%)")
    print("=" * 60)
    
    all_passed = True
    for i, case in enumerate(results["sod_cases"]):
        nx_key = f"nx{case['nx']}"
        ref = REFERENCE[nx_key]
        
        print(f"\n{nx_key}:")
        
        # L2_rho
        passed = case["L2_rho_final"] < 0.03
        all_passed &= passed
        print(f"  L2_rho: {case['L2_rho_final']*100:.4f}% < 3% -> {'PASS' if passed else 'FAIL'}")
        
        # L2_u
        passed = case["L2_u_final"] < 0.03
        all_passed &= passed
        print(f"  L2_u: {case['L2_u_final']*100:.4f}% < 3% -> {'PASS' if passed else 'FAIL'}")
        
        # W_rel
        passed = abs(case["W_rel"]) < 0.03
        all_passed &= passed
        print(f"  W_rel: {abs(case['W_rel'])*100:.4f}% < 3% -> {'PASS' if passed else 'FAIL'}")
        
        # rho_mid_rel
        passed = abs(case["rho_mid_rel"]) < 0.03
        all_passed &= passed
        print(f"  rho_mid_rel: {abs(case['rho_mid_rel'])*100:.4f}% < 3% -> {'PASS' if passed else 'FAIL'}")
        
        # u_mid_rel
        passed = abs(case["u_mid_rel"]) < 0.03
        all_passed &= passed
        print(f"  u_mid_rel: {abs(case['u_mid_rel'])*100:.4f}% < 3% -> {'PASS' if passed else 'FAIL'}")
        
        # osc_frac
        passed = case["osc_frac_final"] < 0.03
        all_passed &= passed
        print(f"  osc_frac: {case['osc_frac_final']*100:.4f}% < 3% -> {'PASS' if passed else 'FAIL'}")
    
    # Check convergence
    print(f"\nConvergence check:")
    conv_passed = results["sod_cases"][1]["L2_rho_final"] <= results["sod_cases"][0]["L2_rho_final"] * 1.02
    all_passed &= conv_passed
    print(f"  nx4000 L2_rho <= nx2000 L2_rho * 1.02 -> {'PASS' if conv_passed else 'FAIL'}")
    
    results["verified"] = all_passed
    print(f"\n{'=' * 60}")
    print(f"VERDICT: {'VERIFIED' if all_passed else 'NOT VERIFIED'}")
    print("=" * 60)
    
    return results


def save_evidence(results):
    """Save evidence to JSON file."""
    os.makedirs(ARTIFACTS, exist_ok=True)
    evidence_path = os.path.join(ARTIFACTS, "sdaa_sod_shock_tube_evidence.json")
    
    with open(evidence_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvidence saved to: {evidence_path}")
    return evidence_path


if __name__ == "__main__":
    results = run_verification()
    if results:
        save_evidence(results)
    else:
        print("Verification failed - SDAA not available")
        sys.exit(1)