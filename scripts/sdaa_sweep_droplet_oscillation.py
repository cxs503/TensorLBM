#!/usr/bin/env python3
"""SDAA sweep for droplet_oscillation benchmark.

This script runs the droplet_oscillation benchmark on SDAA device and validates
the results against the pre-registered criteria.

Criteria (from README):
- |err| ≤ 3% at every level (basis C primary)
- |err| strictly decreasing with R (monotone convergence)
- Each level simulated ≥ 4 theory periods
- Fit sigma_omega/omega_d < 1%
- Mass drift < 2e-3
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


def validate_criteria(results, radii):
    """Validate against pre-registered criteria."""
    criteria = {
        "err_le_3pct_all_levels": True,
        "err_monotone_decreasing": True,
        "simulated_ge_4_periods_all_levels": True,
        "sigma_omega_rel_lt_1pct_all_levels": True,
        "mass_drift_lt_2e3_all_levels": True,
    }
    
    errors = []
    abs_errs = []
    
    for R in radii:
        level_key = f"R{R}"
        if level_key not in results:
            errors.append(f"{level_key}: not found in results")
            continue
            
        level_data = results[level_key]
        if "error" in level_data:
            errors.append(f"{level_key}: {level_data['error']}")
            continue
        
        osc = level_data["oscillation"]
        
        # Check |err| ≤ 3%
        err_pct = abs(osc["err_pct_primary"])
        abs_errs.append(err_pct)
        if err_pct > 3.0:
            criteria["err_le_3pct_all_levels"] = False
            errors.append(f"{level_key}: |err|={err_pct:.3f}% > 3%")
        
        # Check ≥ 4 theory periods
        T_theory = osc["T_theory_C"]
        max_steps = osc["max_steps"]
        n_periods = max_steps / T_theory
        if n_periods < 4.0:
            criteria["simulated_ge_4_periods_all_levels"] = False
            errors.append(f"{level_key}: {n_periods:.2f} periods < 4")
        
        # Check sigma_omega/omega_d < 1%
        sigma_rel = osc["sigma_omega_rel_pct"]
        if sigma_rel >= 1.0:
            criteria["sigma_omega_rel_lt_1pct_all_levels"] = False
            errors.append(f"{level_key}: sigma_omega_rel={sigma_rel:.3f}% >= 1%")
        
        # Check mass drift < 2e-3
        mass_drift = osc["mass_drift"]
        if mass_drift >= 2e-3:
            criteria["mass_drift_lt_2e3_all_levels"] = False
            errors.append(f"{level_key}: mass_drift={mass_drift:.3e} >= 2e-3")
    
    # Check monotone decreasing
    if len(abs_errs) == len(radii):
        for i in range(len(abs_errs) - 1):
            if abs_errs[i + 1] >= abs_errs[i]:
                criteria["err_monotone_decreasing"] = False
                errors.append(f"R{radii[i+1]}: |err|={abs_errs[i+1]:.3f}% not < R{radii[i]}: {abs_errs[i]:.3f}%")
    
    return criteria, errors


def main():
    """Main entry point."""
    print("="*60)
    print("SDAA Sweep: droplet_oscillation benchmark")
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
    
    # Use smaller radii for faster execution (within 20 min budget)
    # Original: 128, 160, 224 (~1 hour on RTX 5090)
    # Reduced: 64, 80, 96 (should be ~8-10x faster than original)
    # Note: R=32 caused NaN issues, so we start from R=64
    radii = [64, 80, 96]
    
    print(f"\nRunning with radii: {radii}")
    print(f"(Reduced from original [128, 160, 224] for SDAA sweep budget)")
    
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
    
    # Validate criteria
    print(f"\n{'='*60}")
    print("Validating criteria...")
    print(f"{'='*60}")
    
    # Filter out error results
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    
    if len(valid_results) == len(radii):
        criteria, errors = validate_criteria(valid_results, radii)
        
        print("\nCriteria validation:")
        for crit_name, passed in criteria.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {status}: {crit_name}")
        
        if errors:
            print("\nErrors:")
            for err in errors:
                print(f"  - {err}")
        
        all_pass = all(criteria.values())
    else:
        print(f"\nOnly {len(valid_results)}/{len(radii)} levels completed successfully")
        criteria = {}
        errors = ["Not all levels completed"]
        all_pass = False
    
    # Build output JSON
    output = {
        "benchmark": "droplet_oscillation",
        "device_evidence": device_evidence,
        "device_used": "sdaa:0",
        "radii": radii,
        "total_wall_time_s": total_time,
        "results": results,
        "criteria": criteria,
        "criteria_errors": errors,
        "all_criteria_pass": all_pass,
        "notes": [
            "Reduced radii from [128, 160, 224] to [64, 80, 96] for SDAA sweep budget",
            "Original benchmark takes ~1 hour on RTX 5090",
            "Criteria validation uses same thresholds as original benchmark",
            "R=32 caused NaN issues (droplet too small for model)",
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
    print(f"All criteria pass: {all_pass}")
    print(f"{'='*60}")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
