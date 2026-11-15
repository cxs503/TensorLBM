#!/usr/bin/env python3
"""Minimal validation run for two-component SC layered Poiseuille.

This script runs a quick validation case to demonstrate the concept.
For production validation, use run_two_component_sc_validation.py with more steps.
"""

import sys
import os
sys.path.insert(0, '/srv/tensor/agent/work/m21558/TensorLBM/src')

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


def run_quick_validation():
    """Run a quick validation case."""
    print("="*60)
    print("Two-Component SC Layered Poiseuille - Quick Validation")
    print("="*60)
    
    # Configuration
    ny, nx = 32, 4
    tau_a, tau_b = 0.8, 1.2
    nu_a = (1.0/3.0) * (tau_a - 0.5)
    nu_b = (1.0/3.0) * (tau_b - 0.5)
    H = ny - 2
    G = 2.0
    g_x = 1e-4  # Stronger force for faster convergence
    steps = 2000
    
    print(f"\nConfiguration:")
    print(f"  Grid: {ny} × {nx}, H = {H}")
    print(f"  Component A: τ={tau_a}, ν={nu_a:.4f}")
    print(f"  Component B: τ={tau_b}, ν={nu_b:.4f}")
    print(f"  SC coupling: G={G}")
    print(f"  Body force: g_x={g_x}")
    print(f"  Steps: {steps}")
    
    # Create solver
    cfg = TwoComponentSCConfig(
        ny=ny, nx=nx,
        tau_a=tau_a, tau_b=tau_b,
        G=G, g_x=g_x, g_y=0.0,
        steps=steps,
        init_rho_a=2.0, init_rho_b=2.0,
        init_layer_frac=0.5,
        forcing="edm",
        sample_every=200,
    )
    
    solver = TwoComponentSCSolver(cfg)
    
    # Run simulation
    print(f"\nRunning simulation...")
    for step in range(steps):
        solver.step()
        if step % 500 == 0:
            macro = solver.get_macroscopic()
            ux_max = macro["ux_total"].max().item()
            print(f"  Step {step}: ux_max = {ux_max:.6e}")
    
    # Get final profile
    profile = solver.get_x_averaged_profile()
    y_num = profile["y"]
    ux_total = profile["ux_total"]
    rho_a = profile["rho_a"]
    rho_b = profile["rho_b"]
    
    # Analytical solution
    interface_pos = 0.5 * H
    analytic_sol = solve_layered(
        height=H,
        interfaces=[interface_pos],
        nus=[nu_a, nu_b],
        g=g_x
    )
    
    ux_analytic = analytic_sol.u(y_num)
    
    # Compute errors (exclude near-wall regions)
    margin = 2
    y_interior = y_num[margin:-margin]
    ux_num_interior = ux_total[margin:-margin]
    ux_ana_interior = ux_analytic[margin:-margin]
    
    scale = np.max(np.abs(ux_ana_interior))
    if scale > 0:
        eps_inf = np.max(np.abs(ux_num_interior - ux_ana_interior)) / scale
        eps_l2 = np.linalg.norm(ux_num_interior - ux_ana_interior) / np.linalg.norm(ux_ana_interior)
    else:
        eps_inf = float('nan')
        eps_l2 = float('nan')
    
    u_peak_num = np.max(ux_total)
    u_peak_ana = analytic_sol.u_peak()
    u_peak_err = abs(u_peak_num - u_peak_ana) / u_peak_ana if u_peak_ana > 0 else float('nan')
    
    print(f"\nResults:")
    print(f"  Peak velocity: num={u_peak_num:.6e}, ana={u_peak_ana:.6e}")
    print(f"  Peak velocity error: {u_peak_err:.2%}")
    print(f"  Profile eps_inf: {eps_inf:.2%}")
    print(f"  Profile eps_L2: {eps_l2:.2%}")
    
    # Generate figure
    output_dir = '/srv/tensor/agent/work/m21558/TensorLBM/artifacts/xval_two_component_sc_layered_poiseuille_edm'
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Velocity profile
    ax = axes[0]
    ax.plot(ux_analytic, y_num, 'k-', linewidth=2, label='Analytical')
    ax.plot(ux_total, y_num, 'r--', linewidth=2, label='Two-component SC (EDM)')
    ax.set_xlabel('Velocity $u_x$', fontsize=12)
    ax.set_ylabel('Wall-normal coordinate $y$', fontsize=12)
    ax.set_title('Velocity Profile Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    textstr = f'$\\epsilon_\\infty$ = {eps_inf:.2%}\n$\\epsilon_{{L2}}$ = {eps_l2:.2%}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Density profiles
    ax = axes[1]
    ax.plot(rho_a, y_num, 'b-', linewidth=2, label='Component A')
    ax.plot(rho_b, y_num, 'r-', linewidth=2, label='Component B')
    ax.plot(rho_a + rho_b, y_num, 'k--', linewidth=1.5, label='Total')
    ax.set_xlabel('Density $\\rho$', fontsize=12)
    ax.set_ylabel('Wall-normal coordinate $y$', fontsize=12)
    ax.set_title('Phase Distribution', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'quick_validation.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nFigure saved to: {fig_path}")
    
    # Save results
    results = {
        "config": {
            "ny": ny, "nx": nx,
            "tau_a": tau_a, "tau_b": tau_b,
            "nu_a": nu_a, "nu_b": nu_b,
            "G": G, "g_x": g_x,
            "steps": steps,
        },
        "metrics": {
            "u_peak_num": float(u_peak_num),
            "u_peak_ana": float(u_peak_ana),
            "u_peak_err": float(u_peak_err),
            "eps_inf": float(eps_inf),
            "eps_l2": float(eps_l2),
        },
        "profile": {
            "y": y_num.tolist(),
            "ux_total": ux_total.tolist(),
            "ux_analytic": ux_analytic.tolist(),
            "rho_a": rho_a.tolist(),
            "rho_b": rho_b.tolist(),
        },
    }
    
    results_path = os.path.join(output_dir, 'quick_validation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    run_quick_validation()
