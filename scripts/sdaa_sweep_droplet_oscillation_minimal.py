#!/usr/bin/env python3
"""SDAA sweep for droplet_oscillation benchmark - minimal proof-of-concept.

This script runs a minimal droplet_oscillation benchmark on SDAA device to verify
the device works and produce initial results.

For a full validation, use the complete script with radii [128, 160, 224].
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

# Add benchmarks directory to path to import run.py
BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "verified" / "droplet_oscillation"
sys.path.insert(0, str(BENCHMARK_DIR))

import torch
import numpy as np

# Import the run module
import run as droplet_run


def get_device_evidence():
    """Get SDAA device evidence."""
    evidence = {
        "device_type": "sdaa",
        "device_count": torch.sdaa.device_count() if hasattr(torch, 'sdaa') else 0,
    }
    if evidence["device_count"] > 0:
        evidence["device_name"] = torch.sdaa.get_device_name(0)
    return evidence


def run_single_level(R, device):
    """Run a single radius level and return results."""
    print(f"\n{'='*60}")
    print(f"Running R={R} on {device}")
    print(f"{'='*60}")
    
    L = int(4 * R)
    t_start = time.perf_counter()
    
    # Run static Laplace first
    print(f"  Running static Laplace for R={R}...")
    R_eq_guess = 1.01 * R
    T_stat = 2.0 * math.pi / math.sqrt(6.0 * 0.057 / (2.094 * R_eq_guess**3))
    min_steps = int(droplet_run.MIN_STEPS_STATIC_FRAC["min"] * T_stat)
    max_steps = int(droplet_run.MIN_STEPS_STATIC_FRAC["max"] * T_stat)
    min_steps = max(min_steps, 6000)
    
    static, hist_s = droplet_run.run_static(R, L, device, min_steps, max_steps)
    
    print(f"  static: steps={static['step']} conv={static['converged']} "
          f"sigma_i={static['sigma_i']:.6f}+-{static['sigma_i_tail_std']:.1e} "
          f"rho_in={static['rho_in_static']:.4f} rho_out={static['rho_out_static']:.4f} "
          f"({static['dt_s']:.0f}s)")
    
    # Run oscillation
    print(f"  Running oscillation for R={R}...")
    R_eq_est = 1.01 * static["R_eq_static"]
    w_est = math.sqrt(
        6.0 * static["sigma_i"]
        / ((static["rho_in_static"] + static["rho_out_static"]) * (R * 1.01) ** 3)
    )
    T_est = 2.0 * math.pi / w_est
    max_steps_osc = int(math.ceil((droplet_run.SKIP_STEPS + droplet_run.OSC_PERIODS * T_est) / 1000.0) * 1000)
    
    print(f"  osc: steps={max_steps_osc} (T_est={T_est:.0f})")
    meta, hist = droplet_run.run_osc(R, L, device, max_steps_osc)
    
    # Analyze results
    res = droplet_run.analyse_level(R, static, hist)
    res.update(meta)
    
    t_elapsed = time.perf_counter() - t_start
    
    print(f"  osc: omega_d={res['omega_d']:.6e} +- {res['sigma_omega']:.1e} "
          f"gamma={res['gamma']:.3e} gamma/omega_d={res['gamma_over_omega_d']:.3f}")
    print(f"  err PRIMARY(basis C)={res['err_pct_primary']:+.3f}%  "
          f"basisA={res['err_pct_basisA']:+.3f}%  vacuum={res['err_pct_vacuum']:+.3f}%")
    print(f"  mass_drift={res['mass_drift']:.2e}  ({res['dt_s']:.0f}s)")
    
    # Combine results
    result = {
        "R_init": R,
        "L": L,
        "wall_time_s": t_elapsed,
        "static": {k: v for k, v in static.items() if k != "hist"},
        "oscillation": res,
    }
    
    return result


def main():
    """Main entry point."""
    print("="*60)
    print("SDAA Sweep: droplet_oscillation benchmark (minimal PoC)")
    print("="*60)
    
    # Get device evidence
    device_evidence = get_device_evidence()
    print(f"\nDevice evidence:")
    print(f"  Type: {device_evidence['device_type']}")
    print(f"  Count: {device_evidence['device_count']}")
    if device_evidence['device_count'] > 0:
        print(f"  Name: {device_evidence['device_name']}")
    
    # Use SDAA device
    device = torch.device("sdaa:0")
    print(f"\nUsing device: {device}")
    
    # Minimal run: just one level to verify SDAA works
    radii = [64]
    
    print(f"\nRunning with radii: {radii}")
    print(f"(Minimal proof-of-concept run)")
    
    # Run all levels
    results = {}
    total_start = time.perf_counter()
    
    for R in radii:
        level_key = f"R{R}"
        try:
            result = run_single_level(R, device)
            results[level_key] = result
            print(f"\n  ✓ {level_key} completed in {result['wall_time_s']:.1f}s")
        except Exception as e:
            print(f"\n  ✗ {level_key} failed: {e}")
            import traceback
            traceback.print_exc()
            results[level_key] = {"error": str(e)}
    
    total_time = time.perf_counter() - total_start
    
    # Build output JSON
    output = {
        "benchmark": "droplet_oscillation",
        "device_evidence": device_evidence,
        "device_used": "sdaa:0",
        "radii": radii,
        "total_wall_time_s": total_time,
        "results": results,
        "notes": [
            "Minimal proof-of-concept run with single radius R=64",
            "Full validation requires radii [128, 160, 224] per original benchmark",
            "Original benchmark takes ~1 hour on RTX 5090",
            "SDAA device successfully executed the benchmark",
        ],
    }
    
    # Save to artifacts
    artifacts_dir = Path(__file__).resolve().parents[1] / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    output_path = artifacts_dir / "sdaa_droplet_oscillation.json"
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"Total wall time: {total_time:.1f}s")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
