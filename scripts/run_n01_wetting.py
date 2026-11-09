#!/usr/bin/env python
"""N01 wetting dynamics: cluster driver for full parameter sweep + convergence + negative controls."""

import json
import math
import sys
import time
from pathlib import Path

# Add src to path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.wetting_n01 import (
    run_parameter_sweep,
    run_grid_convergence,
    run_negative_controls,
    generate_field_plots,
    generate_comparison_curves,
    generate_report,
    THETA_TARGETS,
    RADIUS_MM,
)

def main():
    quick = "--quick" in sys.argv
    device = "cpu"
    
    print(f"Starting N01 wetting dynamics computation (quick={quick})...")
    t0 = time.time()
    
    # 1. Parameter sweep
    print("Running parameter sweep...")
    relax_steps = 500 if quick else 4000
    sweep = run_parameter_sweep(
        relax_steps=relax_steps,
        device=device,
        quick=quick,
    )
    print(f"  Sweep complete: {len(sweep.cases)} cases in {time.time()-t0:.1f}s")
    
    # 2. Grid convergence (for 90° case)
    print("Running grid convergence...")
    conv_relax = 500 if quick else 3000
    conv = run_grid_convergence(
        theta_target_deg=90.0,
        R_mm=1.0,
        relax_steps=conv_relax,
        device=device,
    )
    sweep.convergence.append(conv)
    print(f"  Convergence: p={conv.convergence_order:.3f}, passes={conv.passes}")
    
    # 3. Negative controls
    print("Running negative controls...")
    nc_relax = 200 if quick else 2000
    ncs = run_negative_controls(device=device, relax_steps=nc_relax)
    sweep.negative_controls = ncs
    for nc in ncs:
        print(f"  {nc.test_name}: {'PASS' if nc.passed else 'FAIL'}")
    
    # 4. Generate field plots (for a representative case)
    print("Generating field plots...")
    fig_dir = REPO / "figures"
    plots = generate_field_plots(
        theta_target_deg=90.0,
        R_mm=1.0,
        relax_steps=conv_relax,
        device=device,
        output_dir=fig_dir,
    )
    print(f"  Generated {len(plots)} plots")
    
    # Also generate plots for hydrophilic and hydrophobic cases
    for theta in [30.0, 150.0]:
        plots.update(generate_field_plots(
            theta_target_deg=theta,
            R_mm=1.0,
            relax_steps=conv_relax,
            device=device,
            output_dir=fig_dir,
        ))
    print(f"  Total plots: {len(plots)}")
    
    # 5. Generate comparison curves
    print("Generating comparison curves...")
    curves = generate_comparison_curves(sweep, output_dir=fig_dir)
    print(f"  Generated {len(curves)} comparison curves")
    
    # 6. Generate report
    print("Generating report...")
    report_path = generate_report(sweep, output_dir=REPO / "docs")
    print(f"  Report: {report_path}")
    
    # 7. Save results JSON
    artifacts_dir = REPO / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_path = artifacts_dir / "wetting_n01_results.json"
    with open(results_path, 'w') as f:
        json.dump(sweep.to_dict(), f, indent=2, default=str)
    print(f"  Results: {results_path}")
    
    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")
    
    # Summary
    n_cases = len(sweep.cases)
    theta_errors = [c.theta_error_deg for c in sweep.cases if not math.isnan(c.theta_error_deg)]
    max_te = max(theta_errors) if theta_errors else float('nan')
    n_pass_te = sum(1 for e in theta_errors if e < 5.0)
    
    nc_pass = sum(1 for nc in ncs if nc.passed)
    
    print(f"\n=== SUMMARY ===")
    print(f"Cases: {n_cases}")
    print(f"Max theta error: {max_te:.2f}°")
    print(f"Theta < 5° pass: {n_pass_te}/{len(theta_errors)}")
    print(f"Convergence order: {conv.convergence_order:.3f} (passes={conv.passes})")
    print(f"Negative controls: {nc_pass}/{len(ncs)}")
    print(f"Figures: {len(plots) + len(curves)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
