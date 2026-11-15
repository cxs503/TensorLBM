#!/usr/bin/env python3
"""Validation: Two-component SC with EDM forcing for layered Poiseuille flow.

This script validates the two-component Shan-Chen LBM with EDM (Exact Difference
Method) forcing against the analytical solution for layered Poiseuille flow.

Physical setup:
- Two immiscible fluids with different viscosities (tau_A != tau_B)
- Body force drives flow in x-direction
- No-slip walls at top and bottom
- Analytical solution: piecewise parabolic velocity profile

The two-component SC model naturally forms layers due to the repulsive
interaction force between components. Each component develops its own
velocity profile, and the total velocity should match the analytical
layered Poiseuille solution.

References:
- Martys & Chen (1996) Phys. Rev. E 53, 743 - two-component SC
- Kupershtokh (2004) - EDM forcing
- Analytical solution: solve_layered() in analytic.py
"""

import sys
import os

# Add src to path - must be done before any tensorlbm imports
# Script is in: TensorLBM/src/tensorlbm/verification/layered_poiseuille/
# We need: TensorLBM/src
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '..', '..', '..', '..'))
print(f"DEBUG: script_dir = {script_dir}")
print(f"DEBUG: src_dir = {src_dir}")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorlbm.verification.layered_poiseuille.two_component_sc import (
    TwoComponentSCSolver, TwoComponentSCConfig
)
from tensorlbm.verification.layered_poiseuille.analytic import solve_layered


def run_validation_case(
    ny: int = 64,
    nx: int = 8,
    tau_a: float = 0.8,
    tau_b: float = 1.2,
    G: float = 2.0,
    g_x: float = 1e-5,
    steps: int = 20000,
    init_layer_frac: float = 0.5,
    forcing: str = "edm",
    label: str = "default",
) -> dict:
    """Run a single validation case and return metrics."""
    
    # Viscosities from relaxation times: nu = cs^2 * (tau - 0.5)
    nu_a = (1.0/3.0) * (tau_a - 0.5)
    nu_b = (1.0/3.0) * (tau_b - 0.5)
    H = ny - 2  # Effective channel height (half-way bounce-back)
    interface_y = init_layer_frac * H
    
    print(f"\n{'='*60}")
    print(f"Validation case: {label}")
    print(f"{'='*60}")
    print(f"Grid: {ny} x {nx}, H = {H}")
    print(f"Component A: tau={tau_a:.3f}, nu={nu_a:.4f}")
    print(f"Component B: tau={tau_b:.3f}, nu={nu_b:.4f}")
    print(f"SC coupling: G={G}")
    print(f"Body force: g_x={g_x}")
    print(f"Forcing scheme: {forcing}")
    print(f"Interface at y/H = {init_layer_frac}")
    print(f"Steps: {steps}")
    
    # Create and run solver
    cfg = TwoComponentSCConfig(
        ny=ny, nx=nx,
        tau_a=tau_a, tau_b=tau_b,
        G=G, g_x=g_x, g_y=0.0,
        steps=steps,
        init_rho_a=2.0, init_rho_b=2.0,
        init_layer_frac=init_layer_frac,
        forcing=forcing,
        sample_every=500,
    )
    
    solver = TwoComponentSCSolver(cfg)
    result = solver.run(monitor_every=1000)
    
    # Get final profile
    profile = result["final_profile"]
    y_num = profile["y"]
    ux_total = profile["ux_total"]
    rho_a = profile["rho_a"]
    rho_b = profile["rho_b"]
    
    # Analytical solution
    # The analytical solution expects the interface position in wall coordinates
    # y_wall goes from 0.5 to H-0.5, so interface is at init_layer_frac * H
    interface_pos = init_layer_frac * H
    analytic_sol = solve_layered(
        height=H,
        interfaces=[interface_pos],
        nus=[nu_a, nu_b],
        g=g_x
    )
    
    # Evaluate analytical solution at numerical grid points
    # Note: y_num is in wall coordinates (0.5 to H-0.5)
    y_analytic = np.linspace(0.5, H - 0.5, 200)
    ux_analytic = analytic_sol.u(y_analytic)
    
    # Interpolate analytical solution to numerical grid
    ux_analytic_on_grid = analytic_sol.u(y_num)
    
    # Compute errors
    # Only compare in the interior (exclude near-wall regions where SC effects dominate)
    margin = 3
    y_interior = y_num[margin:-margin]
    ux_num_interior = ux_total[margin:-margin]
    ux_ana_interior = ux_analytic_on_grid[margin:-margin]
    
    # Error metrics
    scale = np.max(np.abs(ux_ana_interior))
    if scale > 0:
        eps_inf = np.max(np.abs(ux_num_interior - ux_ana_interior)) / scale
        eps_l2 = np.linalg.norm(ux_num_interior - ux_ana_interior) / np.linalg.norm(ux_ana_interior)
    else:
        eps_inf = float('nan')
        eps_l2 = float('nan')
    
    # Peak velocity comparison
    u_peak_num = np.max(ux_total)
    u_peak_ana = analytic_sol.u_peak()
    u_peak_err = abs(u_peak_num - u_peak_ana) / u_peak_ana if u_peak_ana > 0 else float('nan')
    
    # Flow rate comparison
    q_num = np.trapezoid(ux_total, y_num)
    q_ana = analytic_sol.flow_rate()
    q_err = abs(q_num - q_ana) / abs(q_ana) if abs(q_ana) > 0 else float('nan')
    
    # Phase distribution: check if components are layered
    # Component A should be dominant in bottom, B in top
    rho_a_bottom = np.mean(rho_a[:ny//4])
    rho_a_top = np.mean(rho_a[-ny//4:])
    rho_b_bottom = np.mean(rho_b[:ny//4])
    rho_b_top = np.mean(rho_b[-ny//4:])
    
    layer_separation = (rho_a_bottom - rho_a_top) + (rho_b_top - rho_b_bottom)
    
    print(f"\nResults:")
    print(f"  Peak velocity: num={u_peak_num:.6e}, ana={u_peak_ana:.6e}, err={u_peak_err:.2%}")
    print(f"  Flow rate: num={q_num:.6e}, ana={q_ana:.6e}, err={q_err:.2%}")
    print(f"  Profile eps_inf: {eps_inf:.2%}")
    print(f"  Profile eps_L2: {eps_l2:.2%}")
    print(f"  Layer separation: {layer_separation:.3f}")
    print(f"  rho_A (bottom/top): {rho_a_bottom:.3f} / {rho_a_top:.3f}")
    print(f"  rho_B (bottom/top): {rho_b_bottom:.3f} / {rho_b_top:.3f}")
    
    return {
        "label": label,
        "config": {
            "ny": ny, "nx": nx,
            "tau_a": tau_a, "tau_b": tau_b,
            "nu_a": nu_a, "nu_b": nu_b,
            "G": G, "g_x": g_x,
            "steps": steps,
            "init_layer_frac": init_layer_frac,
            "forcing": forcing,
        },
        "metrics": {
            "u_peak_num": u_peak_num,
            "u_peak_ana": u_peak_ana,
            "u_peak_err": u_peak_err,
            "q_num": q_num,
            "q_ana": q_ana,
            "q_err": q_err,
            "eps_inf": eps_inf,
            "eps_l2": eps_l2,
            "layer_separation": layer_separation,
        },
        "profile": {
            "y": y_num.tolist(),
            "ux_total": ux_total.tolist(),
            "ux_analytic": ux_analytic_on_grid.tolist(),
            "rho_a": rho_a.tolist(),
            "rho_b": rho_b.tolist(),
        },
        "analytic_full": {
            "y": y_analytic.tolist(),
            "ux": ux_analytic.tolist(),
        },
    }


def generate_figures(results: list, output_dir: str):
    """Generate validation figures."""
    os.makedirs(output_dir, exist_ok=True)
    
    for res in results:
        label = res["label"]
        profile = res["profile"]
        y = np.array(profile["y"])
        ux_total = np.array(profile["ux_total"])
        ux_ana = np.array(profile["ux_analytic"])
        rho_a = np.array(profile["rho_a"])
        rho_b = np.array(profile["rho_b"])
        
        # Figure 1: Velocity profile comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Velocity profile
        ax = axes[0]
        ax.plot(ux_ana, y, 'k-', linewidth=2, label='Analytical')
        ax.plot(ux_total, y, 'r--', linewidth=2, label='Two-component SC (EDM)')
        ax.set_xlabel('Velocity $u_x$', fontsize=12)
        ax.set_ylabel('Wall-normal coordinate $y$', fontsize=12)
        ax.set_title(f'Velocity Profile - {label}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add error annotation
        metrics = res["metrics"]
        textstr = f'$\\epsilon_\\infty$ = {metrics["eps_inf"]:.2%}\n$\\epsilon_{{L2}}$ = {metrics["eps_l2"]:.2%}\n$Q_{{err}}$ = {metrics["q_err"]:.2%}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Density profiles (phase distribution)
        ax = axes[1]
        ax.plot(rho_a, y, 'b-', linewidth=2, label='Component A')
        ax.plot(rho_b, y, 'r-', linewidth=2, label='Component B')
        ax.plot(rho_a + rho_b, y, 'k--', linewidth=1.5, label='Total')
        ax.set_xlabel('Density $\\rho$', fontsize=12)
        ax.set_ylabel('Wall-normal coordinate $y$', fontsize=12)
        ax.set_title(f'Phase Distribution - {label}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{label}_velocity_profile.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Error distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        error = ux_total - ux_ana
        scale = np.max(np.abs(ux_ana))
        error_rel = error / scale if scale > 0 else error
        ax.plot(y, error_rel, 'b-', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Wall-normal coordinate $y$', fontsize=12)
        ax.set_ylabel('Relative velocity error $(u_{num} - u_{ana}) / u_{max}$', fontsize=12)
        ax.set_title(f'Error Distribution - {label}', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{label}_error_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\nFigures saved to {output_dir}")


def main():
    """Run validation campaign."""
    output_dir = os.path.join(src_dir, 'artifacts', 
                              'xval_two_component_sc_layered_poiseuille_edm')
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    # Case 1: Baseline - moderate viscosity contrast
    results.append(run_validation_case(
        ny=64, nx=8,
        tau_a=0.8, tau_b=1.2,
        G=2.0, g_x=1e-5,
        steps=20000,
        init_layer_frac=0.5,
        forcing="edm",
        label="baseline_m3",
    ))
    
    # Case 2: Higher viscosity contrast (5:1)
    results.append(run_validation_case(
        ny=64, nx=8,
        tau_a=0.7, tau_b=1.3,
        G=2.0, g_x=1e-5,
        steps=20000,
        init_layer_frac=0.5,
        forcing="edm",
        label="contrast_m5",
    ))
    
    # Case 3: Off-center interface
    results.append(run_validation_case(
        ny=64, nx=8,
        tau_a=0.8, tau_b=1.2,
        G=2.0, g_x=1e-5,
        steps=20000,
        init_layer_frac=0.3,
        forcing="edm",
        label="interface_30pct",
    ))
    
    # Case 4: Finer grid
    results.append(run_validation_case(
        ny=96, nx=8,
        tau_a=0.8, tau_b=1.2,
        G=2.0, g_x=1e-5,
        steps=30000,
        init_layer_frac=0.5,
        forcing="edm",
        label="fine_grid_96",
    ))
    
    # Generate figures
    generate_figures(results, os.path.join(output_dir, 'figures'))
    
    # Save results
    output_file = os.path.join(output_dir, 'validation_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    for res in results:
        m = res["metrics"]
        print(f"\n{res['label']}:")
        print(f"  Peak velocity error: {m['u_peak_err']:.2%}")
        print(f"  Flow rate error: {m['q_err']:.2%}")
        print(f"  Profile eps_inf: {m['eps_inf']:.2%}")
        print(f"  Profile eps_L2: {m['eps_l2']:.2%}")
        print(f"  Layer separation: {m['layer_separation']:.3f}")
        
        # Acceptance check
        if m['u_peak_err'] < 0.15 and m['q_err'] < 0.15:
            print(f"  ✓ PASS: errors < 15%")
        else:
            print(f"  ✗ FAIL: errors >= 15%")
    
    return results


if __name__ == "__main__":
    main()
