#!/usr/bin/env python3
"""Rayleigh-Benard / Natural Convection Cavity at Ra=1e5 — de Vahl Davis 1983 benchmark.

This script runs the differentially heated square cavity benchmark at Ra=1e5, Pr=0.71
using TensorLBM's thermal solver (D2Q9+D2Q5, BGK, Guo buoyancy).

Reference: de Vahl Davis (1983) "Natural convection of air in a square cavity:
a bench mark numerical solution", Int. J. Numer. Methods Fluids, 3, 249-264.
Table 1, page 257: Ra=10^5, Pr=0.71
  - Nu_mean = 4.519
  - u_max = 34.72 (on vertical centerline, diffusion-scaled)
  - v_max = 68.60 (on horizontal centerline, diffusion-scaled)

Note: The task anchor says "CASE-RAYLEIGH-BENARD-RA1E5" but the reference is
de Vahl Davis 1983 which is for a differentially heated square cavity (side-heated).
This is the standard natural convection benchmark at Ra=1e5.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "src")
sys.path.insert(0, "benchmarks/verified/thermal_cavity")

import driver  # noqa: E402

# de Vahl Davis 1983, Table 1, page 257: Ra=10^5, Pr=0.71
REF_DVD_1E5 = {
    "nu": 4.519,
    "u_max": 34.72,
    "v_max": 68.60,
    "nu_max": None,  # Not in Table 1 for Ra=1e5
    "nu_min": None,
    "nu_mid": None,
    "u_max_y": 0.813,  # Approximate location
    "v_max_x": 0.066,  # Approximate location
}

# Also include Ra=1e4 for comparison (already verified)
REF_DVD_1E4 = {
    "nu": 2.243,
    "u_max": 16.178,
    "v_max": 19.617,
}


def run_case(ra: float, nx: int, device: torch.device, max_steps: int, dtype=torch.float32):
    """Run a single case and return observables."""
    tag = f"ra{int(math.log10(ra))}e{int(math.log10(ra))}_n{nx}" if ra >= 1 else f"ra{ra}_n{nx}"
    tag = f"ra1e{int(round(math.log10(ra)))}_n{nx}"
    
    print(f"[run] {tag}: starting, max_steps={max_steps}", flush=True)
    
    f, g, wall, p = driver.make_fields(nx, ra, 0.71, 0.6, device, dtype=dtype)
    tau_T, g_beta = p["tau_T"], p["g_beta"]
    
    t0 = time.time()
    prev_obs = None
    steady_step = None
    hist = []
    
    for step in range(1, max_steps + 1):
        f, g = driver.advance(f, g, wall, 0.6, tau_T, g_beta)
        
        if step % 5000 == 0 or step == max_steps:
            obs = driver.observables(f, g, p)
            hist.append({"step": step, **{k: obs[k] for k in ["nu", "u_max_nd", "v_max_nd"]}})
            
            # Check for steady state
            if prev_obs is not None and step > 10000:
                drift = max(
                    abs(obs["nu"] - prev_obs["nu"]),
                    abs(obs["u_max_nd"] - prev_obs["u_max_nd"]) / max(abs(prev_obs["u_max_nd"]), 1e-10),
                )
                if drift < 1e-4 and steady_step is None:
                    steady_step = step
                    print(f"[run] {tag}: steady at step={step}, drift={drift:.2e}", flush=True)
            
            if step % 20000 == 0 or step == max_steps:
                elapsed = time.time() - t0
                print(
                    f"[run] {tag}: step={step} Nu={obs['nu']:.4f} "
                    f"u_max={obs['u_max_nd']:.4f} v_max={obs['v_max_nd']:.4f} "
                    f"({elapsed:.0f}s)",
                    flush=True,
                )
            prev_obs = obs
    
    # Final observables
    obs = driver.observables(f, g, p)
    elapsed = time.time() - t0
    
    result = {
        "tag": tag,
        "ra": ra,
        "pr": 0.71,
        "nx": nx,
        "ny": nx,
        "tau": 0.6,
        "tau_T": tau_T,
        "params": p,
        "steps_run": step,
        "steady_step": steady_step,
        "elapsed_s": round(elapsed, 1),
        "obs": obs,
        "history": hist,
    }
    
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ra", type=float, default=1e5)
    ap.add_argument("--nx", nargs="+", type=int, default=[64, 128, 256])
    ap.add_argument("--max-steps", type=int, default=500000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--out", default="result_ra1e5.json")
    args = ap.parse_args()
    
    device = torch.device(args.device)
    torch.set_num_threads(min(32, torch.get_num_threads()))
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    
    print(f"[main] Ra={args.ra}, grids={args.nx}, device={device}, dtype={dtype}", flush=True)
    
    # Reference values
    if abs(args.ra - 1e5) < 1e3:
        ref = REF_DVD_1E5
    elif abs(args.ra - 1e4) < 100:
        ref = REF_DVD_1E4
    else:
        ref = None
    
    results = {
        "case": "natural_convection_cavity_ra1e5",
        "title": "de Vahl Davis 1983 natural convection square cavity at Ra=1e5",
        "reference": "de Vahl Davis (1983) Int. J. Numer. Methods Fluids 3:249-264, Table 1 p.257",
        "ref_values": ref,
        "lattice": "D2Q9+D2Q5",
        "collision": "BGK + Guo buoyancy",
        "pr": 0.71,
        "tau": 0.6,
        "grids": {},
    }
    
    for nx in args.nx:
        res = run_case(args.ra, nx, device, args.max_steps, dtype)
        
        # Compute errors vs reference
        if ref is not None:
            nu_err = 100.0 * abs(res["obs"]["nu"] - ref["nu"]) / ref["nu"]
            u_err = 100.0 * abs(res["obs"]["u_max_nd"] - ref["u_max"]) / ref["u_max"]
            v_err = 100.0 * abs(res["obs"]["v_max_nd"] - ref["v_max"]) / ref["v_max"]
            res["err_pct"] = {
                "nu": round(nu_err, 3),
                "u_max": round(u_err, 3),
                "v_max": round(v_err, 3),
            }
            res["pass"] = nu_err < 8.0 and u_err < 8.0 and v_err < 8.0
        else:
            res["err_pct"] = None
            res["pass"] = None
        
        results["grids"][str(nx)] = res
    
    # Convergence analysis
    if len(args.nx) >= 2:
        nus = [results["grids"][str(nx)]["obs"]["nu"] for nx in args.nx]
        results["convergence"] = {
            "nu_values": nus,
            "nu_change_pct": [
                round(100.0 * abs(nus[i+1] - nus[i]) / nus[i], 3) if i < len(nus)-1 else None
                for i in range(len(nus))
            ],
        }
    
    # Save results
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[main] Results saved to {args.out}", flush=True)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    if ref:
        print(f"Reference (de Vahl Davis 1983, Ra={args.ra:.0e}):")
        print(f"  Nu = {ref['nu']}, u_max = {ref['u_max']}, v_max = {ref['v_max']}")
    print("\nResults:")
    for nx in args.nx:
        res = results["grids"][str(nx)]
        obs = res["obs"]
        print(f"  N={nx}: Nu={obs['nu']:.4f}, u_max={obs['u_max_nd']:.4f}, v_max={obs['v_max_nd']:.4f}")
        if res.get("err_pct"):
            ep = res["err_pct"]
            status = "✓" if res["pass"] else "✗"
            print(f"    Errors: Nu={ep['nu']:.2f}%, u_max={ep['u_max']:.2f}%, v_max={ep['v_max']:.2f}% {status}")
    
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
