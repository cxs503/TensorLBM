#!/usr/bin/env python3
"""
Natural Convection Square Cavity Validation
============================================
De Vahl Davis benchmark (1983) for natural convection in a differentially heated square cavity.

Configuration:
- Left wall: T_hot = 1.0
- Right wall: T_cold = 0.0
- Top/bottom walls: adiabatic, no-slip
- Gravity: downward (-y direction)
- Boussinesq approximation

Test cases:
1. Ra = 1e4, Pr = 0.71 → Reference Nu_avg = 2.194
2. Ra = 1e5, Pr = 0.71 → Reference Nu_avg = 4.446

Criterion: |Nu_computed - Nu_reference| / Nu_reference ≤ 5%

Method: Double-distribution LBM (D2Q9 for flow, D2Q5 for temperature)
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from tensorlbm.thermal import simulate_natural_convection


def run_case(ra: float, nx: int, steps: int, label: str) -> dict:
    """Run a single natural convection case."""
    print(f"\n{'='*60}")
    print(f"Running {label}: Ra={ra:.0e}, Pr=0.71, nx={nx}, steps={steps}")
    print(f"{'='*60}")
    
    t0 = time.time()
    result = simulate_natural_convection(
        nx=nx,
        ra=ra,
        pr=0.71,
        tau=0.6,
        t_hot=1.0,
        t_cold=0.0,
        steps=steps,
        device='cpu',
        seed_t='linear',
        report_every=max(1000, steps // 10),
    )
    elapsed = time.time() - t0
    
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Nu_grad2: {result['nu_grad2']:.4f}")
    print(f"Nu_grad1: {result['nu_grad1']:.4f}")
    print(f"Nu_halfway: {result['nu_halfway']:.4f}")
    print(f"Residuals: u={result['last_resid_u']:.2e}, T={result['last_resid_T']:.2e}")
    print(f"u_max: {result['u_max']:.4f}")
    print(f"T range: [{result['T_min']:.4f}, {result['T_max']:.4f}]")
    
    return result


def plot_temperature_field(result: dict, ra: float, save_path: Path):
    """Generate temperature field contour plot."""
    # Re-run to get the final temperature field
    # Actually, we need to modify simulate_natural_convection to return T field
    # For now, we'll create a placeholder plot structure
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract Nu history
    steps = [h['step'] for h in result['nu_history']]
    nu_vals = [h['nu'] for h in result['nu_history']]
    
    # Reference value
    nu_ref = 2.194 if ra == 1e4 else 4.446
    
    ax.plot(steps, nu_vals, 'b-', linewidth=2, label='Computed Nu_avg')
    ax.axhline(nu_ref, color='r', linestyle='--', linewidth=2, label=f'Reference (de Vahl Davis): {nu_ref:.3f}')
    ax.axhline(nu_ref * 1.05, color='g', linestyle=':', linewidth=1, label='+5%')
    ax.axhline(nu_ref * 0.95, color='g', linestyle=':', linewidth=1, label='-5%')
    
    ax.set_xlabel('Iteration Step', fontsize=12)
    ax.set_ylabel('Average Nusselt Number', fontsize=12)
    ax.set_title(f'Natural Convection Square Cavity\nRa = {ra:.0e}, Pr = 0.71', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max(steps)])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved convergence plot: {save_path}")
    plt.close()


def plot_nu_convergence(results: dict, save_path: Path):
    """Plot Nu convergence for both cases."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (ra, result) in enumerate(results.items()):
        ax = axes[idx]
        steps = [h['step'] for h in result['nu_history']]
        nu_vals = [h['nu'] for h in result['nu_history']]
        
        nu_ref = 2.194 if ra == 1e4 else 4.446
        
        ax.plot(steps, nu_vals, 'b-', linewidth=2, label='Computed Nu_avg')
        ax.axhline(nu_ref, color='r', linestyle='--', linewidth=2, label=f'Reference: {nu_ref:.3f}')
        ax.axhline(nu_ref * 1.05, color='g', linestyle=':', linewidth=1, alpha=0.7, label='±5% criterion')
        ax.axhline(nu_ref * 0.95, color='g', linestyle=':', linewidth=1, alpha=0.7)
        
        ax.set_xlabel('Iteration Step', fontsize=11)
        ax.set_ylabel('Average Nusselt Number', fontsize=11)
        ax.set_title(f'Ra = {ra:.0e}', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, max(steps)])
    
    plt.suptitle('Natural Convection Square Cavity - Nusselt Number Convergence\n(de Vahl Davis Benchmark)', 
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot: {save_path}")
    plt.close()


def main():
    """Main validation routine."""
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("Natural Convection Square Cavity Validation")
    print("De Vahl Davis Benchmark (1983)")
    print("="*60)
    
    # Configuration
    # Use moderate grid size for CPU feasibility
    # nx=41 gives reasonable accuracy while completing in reasonable time
    nx = 41
    steps_ra4 = 8000  # Ra=1e4 converges faster
    steps_ra5 = 12000  # Ra=1e5 needs more steps
    
    # Run cases
    results = {}
    
    # Case 1: Ra = 1e4
    results[1e4] = run_case(ra=1e4, nx=nx, steps=steps_ra4, label="Case 1")
    
    # Case 2: Ra = 1e5
    results[1e5] = run_case(ra=1e5, nx=nx, steps=steps_ra5, label="Case 2")
    
    # Generate plots
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)
    
    # Individual convergence plots
    for ra, result in results.items():
        plot_temperature_field(result, ra, output_dir / f"nu_convergence_ra{int(ra)}.png")
    
    # Combined comparison plot
    plot_nu_convergence(results, output_dir / "nu_convergence_comparison.png")
    
    # Compute errors and generate report
    print("\n" + "="*60)
    print("Validation Results")
    print("="*60)
    
    report = {
        "benchmark": "de Vahl Davis (1983) - Natural Convection Square Cavity",
        "method": "Double-distribution LBM (D2Q9 + D2Q5)",
        "grid_size": nx,
        "prandtl": 0.71,
        "cases": {},
    }
    
    all_pass = True
    for ra, result in results.items():
        nu_ref = 2.194 if ra == 1e4 else 4.446
        nu_computed = result['nu_grad2']
        error_abs = abs(nu_computed - nu_ref)
        error_rel = error_abs / nu_ref * 100
        passed = error_rel <= 5.0
        
        if not passed:
            all_pass = False
        
        report["cases"][f"Ra{int(ra)}"] = {
            "rayleigh": ra,
            "prandtl": 0.71,
            "grid_size": nx,
            "steps": result['steps'],
            "nu_reference": nu_ref,
            "nu_computed_grad2": nu_computed,
            "nu_computed_grad1": result['nu_grad1'],
            "nu_computed_halfway": result['nu_halfway'],
            "error_absolute": error_abs,
            "error_relative_percent": error_rel,
            "criterion_5percent": passed,
            "residual_u": result['last_resid_u'],
            "residual_T": result['last_resid_T'],
            "u_max": result['u_max'],
            "T_min": result['T_min'],
            "T_max": result['T_max'],
            "elapsed_s": result['elapsed_s'],
        }
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\nRa = {ra:.0e}:")
        print(f"  Reference Nu:    {nu_ref:.3f}")
        print(f"  Computed Nu:     {nu_computed:.4f}")
        print(f"  Absolute error:  {error_abs:.4f}")
        print(f"  Relative error:  {error_rel:.2f}%")
        print(f"  Criterion (≤5%): {status}")
        print(f"  Residuals:       u={result['last_resid_u']:.2e}, T={result['last_resid_T']:.2e}")
    
    report["overall_pass"] = all_pass
    report["conclusion"] = (
        "All cases passed the 5% criterion" if all_pass 
        else "Some cases did not meet the 5% criterion - see details above"
    )
    
    # Save report
    report_path = output_dir / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report: {report_path}")
    
    # Save markdown summary
    md_path = output_dir / "VALIDATION_SUMMARY.md"
    with open(md_path, 'w') as f:
        f.write("# Natural Convection Square Cavity Validation\n\n")
        f.write("## Benchmark\n\n")
        f.write("**Reference:** de Vahl Davis (1983) - Natural convection in a differentially heated square cavity\n\n")
        f.write("## Configuration\n\n")
        f.write("- Geometry: Square cavity (L × L)\n")
        f.write("- Left wall: T_hot = 1.0 (isothermal)\n")
        f.write("- Right wall: T_cold = 0.0 (isothermal)\n")
        f.write("- Top/bottom walls: Adiabatic, no-slip\n")
        f.write("- Gravity: Downward (-y direction)\n")
        f.write("- Boussinesq approximation\n")
        f.write("- Pr = 0.71 (air)\n\n")
        f.write("## Method\n\n")
        f.write("Double-distribution Lattice Boltzmann Method:\n")
        f.write("- D2Q9 for flow field (velocity, pressure)\n")
        f.write("- D2Q5 for temperature field\n")
        f.write("- BGK collision with Guo force for buoyancy\n")
        f.write("- Half-way bounce-back for no-slip walls\n")
        f.write("- Non-equilibrium extrapolation for isothermal walls\n\n")
        f.write("## Results\n\n")
        f.write("| Case | Ra | Grid | Steps | Nu_ref | Nu_comp | Error | Pass? |\n")
        f.write("|------|-----|------|-------|--------|---------|-------|-------|\n")
        
        for ra, data in report["cases"].items():
            status = "✓" if data["criterion_5percent"] else "✗"
            f.write(f"| {ra} | {data['rayleigh']:.0e} | {data['grid_size']}×{data['grid_size']} | "
                   f"{data['steps']} | {data['nu_reference']:.3f} | {data['nu_computed_grad2']:.4f} | "
                   f"{data['error_relative_percent']:.2f}% | {status} |\n")
        
        f.write(f"\n## Conclusion\n\n{report['conclusion']}\n\n")
        
        if not all_pass:
            f.write("### Notes\n\n")
            f.write("The computed Nusselt numbers show deviation from the reference values. ")
            f.write("Possible reasons:\n")
            f.write("- Grid resolution (41×41) may be insufficient for high accuracy\n")
            f.write("- Insufficient iterations to reach fully converged steady state\n")
            f.write("- LBM discretization errors at moderate resolution\n")
            f.write("- Temperature boundary condition implementation (non-equilibrium extrapolation)\n\n")
            f.write("For production validation, finer grids (≥128×128) and longer runs are recommended.\n")
    
    print(f"Saved summary: {md_path}")
    
    print("\n" + "="*60)
    print("Validation complete!")
    print("="*60)
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
