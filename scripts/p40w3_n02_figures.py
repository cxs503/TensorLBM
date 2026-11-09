#!/usr/bin/env python3
"""Generate figures for P40W3-N02 cavitation validation report."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np

matplotlib.use('Agg')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorlbm.cavitation_airfoil import (
    CavitationFlowConfig,
    MixtureCavitationSolver,
    reference_data,
)

torch.set_num_threads(1)


def load_results():
    """Load validation results."""
    results_file = Path(__file__).parent.parent / "artifacts" / "p40w3_n02" / "validation_results.json"
    with open(results_file) as f:
        return json.load(f)


def plot_lcav_vs_sigma(results):
    """Plot cavity length vs cavitation number."""
    ref = results["metadata"]["reference_data"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    sheet_data = [r for r in results["sheet_cavitation"] if r["status"] == "success"]
    sigmas = [r["config"]["sigma"] for r in sheet_data]
    l_cavs = [r["l_cav"] for r in sheet_data]
    
    # Reference data
    sigma_ref = ref["sheet"]["sigma"]
    l_cav_brennen = ref["sheet"]["l_cav_exp"]
    l_cav_merkle = ref["sheet"]["l_cav_merkle"]
    
    # Plot
    ax.plot(sigmas, l_cavs, 'bo-', linewidth=2, markersize=10, label='This work (isobaric cutoff)')
    ax.plot(sigma_ref, l_cav_brennen, 'rs-', linewidth=2, markersize=10, label='Brennen (1994) experiment')
    ax.plot(sigma_ref, l_cav_merkle, 'g^-', linewidth=2, markersize=10, label='Merkle et al. (1998) numerical')
    
    ax.set_xlabel('Cavitation number σ', fontsize=14)
    ax.set_ylabel('Cavity length L_cav / c', fontsize=14)
    ax.set_title('Sheet Cavitation: Cavity Length vs Cavitation Number\nNACA0015, α=5°, Re=10⁵', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 1.1)
    ax.set_ylim(0, 0.8)
    
    plt.tight_layout()
    fig_path = Path(__file__).parent.parent / "figures" / "p40w3_n02_lcav_vs_sigma.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close()


def plot_delta_vs_sigma(results):
    """Plot cavity thickness vs cavitation number."""
    ref = results["metadata"]["reference_data"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    sheet_data = [r for r in results["sheet_cavitation"] if r["status"] == "success"]
    sigmas = [r["config"]["sigma"] for r in sheet_data]
    deltas = [r["delta_cav"] for r in sheet_data]
    
    # Reference data
    sigma_ref = ref["sheet"]["sigma"]
    delta_brennen = ref["sheet"]["delta_cav_exp"]
    
    # Plot
    ax.plot(sigmas, deltas, 'bo-', linewidth=2, markersize=10, label='This work (isobaric cutoff)')
    ax.plot(sigma_ref, delta_brennen, 'rs-', linewidth=2, markersize=10, label='Brennen (1994) experiment')
    
    ax.set_xlabel('Cavitation number σ', fontsize=14)
    ax.set_ylabel('Maximum cavity thickness δ_cav / c', fontsize=14)
    ax.set_title('Sheet Cavitation: Maximum Thickness vs Cavitation Number\nNACA0015, α=5°, Re=10⁵', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 1.1)
    ax.set_ylim(0, 0.08)
    
    plt.tight_layout()
    fig_path = Path(__file__).parent.parent / "figures" / "p40w3_n02_delta_vs_sigma.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close()


def plot_grid_convergence(results):
    """Plot grid convergence study."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract grid convergence data
    grid_data = [r for r in results["grid_convergence"] if r["status"] == "success"]
    cpcs = [r["config"]["cells_per_chord"] for r in grid_data]
    l_cavs = [r["l_cav"] for r in grid_data]
    deltas = [r["delta_cav"] for r in grid_data]
    
    # Plot L_cav
    ax1.plot(cpcs, l_cavs, 'bo-', linewidth=2, markersize=10)
    ax1.set_xlabel('Cells per chord', fontsize=14)
    ax1.set_ylabel('Cavity length L_cav / c', fontsize=14)
    ax1.set_title('Grid Convergence: Cavity Length\nNACA0015, σ=0.8, α=5°', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Plot delta
    ax2.plot(cpcs, deltas, 'ro-', linewidth=2, markersize=10)
    ax2.set_xlabel('Cells per chord', fontsize=14)
    ax2.set_ylabel('Maximum thickness δ_cav / c', fontsize=14)
    ax2.set_title('Grid Convergence: Cavity Thickness\nNACA0015, σ=0.8, α=5°', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    fig_path = Path(__file__).parent.parent / "figures" / "p40w3_n02_grid_convergence.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close()


def plot_pressure_field():
    """Plot pressure field with cavity overlay for one case."""
    # Run a single case to get the field
    cfg = CavitationFlowConfig(
        airfoil='NACA0015',
        alpha_deg=5.0,
        sigma=0.6,
        cells_per_chord=25,
        domain_x_chords=4.0,
        domain_y_chords=2.0,
        le_x_chords=1.0,
        le_y_chords=1.0,
        n_steps=300,
        cfl=0.2,
        c_dest=0.0,
        c_prod=0.0,
        pressure_anchor=True,
    )
    
    solver = MixtureCavitationSolver(cfg)
    solver.u = solver.u.masked_fill(solver.damp_u > 1.5, 0.0)
    solver.v = solver.v.masked_fill(solver.damp_v > 1.5, 0.0)
    
    for _ in range(300):
        solver.step()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))
    
    h = cfg.h
    nx, ny = cfg.nx, cfg.ny
    xs = (torch.arange(nx) + 0.5) * h
    ys = (torch.arange(ny) + 0.5) * h
    X, Y = torch.meshgrid(xs, ys, indexing='ij')
    
    # Pressure field
    p = solver.p.numpy()
    p_v = cfg.p_v
    
    # Cavity indicator (isobaric cutoff)
    alpha_ind = torch.sigmoid((p_v - solver.p.double()) / (0.05 * abs(p_v))).numpy()
    
    # Plot pressure contours
    levels = np.linspace(p.min(), min(0, p.max()), 30)
    cs = ax.contourf(X.numpy(), Y.numpy(), p, levels=levels, cmap='viridis', alpha=0.7)
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label('Pressure coefficient p / (ρU²)', fontsize=12)
    
    # Overlay cavity contour
    ax.contour(X.numpy(), Y.numpy(), alpha_ind, levels=[0.5], colors='red', linewidths=2)
    
    # Mark airfoil
    ax.fill(X.numpy()[solver.solid.numpy()], Y.numpy()[solver.solid.numpy()], color='black', alpha=0.5)
    
    ax.set_xlabel('x / c', fontsize=14)
    ax.set_ylabel('y / c', fontsize=14)
    ax.set_title('Pressure Field with Cavity Overlay (red contour)\nNACA0015, σ=0.6, α=5°, cpc=25', fontsize=14)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    fig_path = Path(__file__).parent.parent / "figures" / "p40w3_n02_pressure_field.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close()


def plot_cloud_cavitation_history(results):
    """Plot time history of cavity length for cloud cavitation cases."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    for i, (ax, case_data) in enumerate(zip(axes, results["cloud_cavitation"])):
        if case_data["status"] != "success":
            continue
        
        sigma = case_data["config"]["sigma"]
        time_series = case_data.get("time_series", [])
        lcav_series = case_data.get("lcav_series", [])
        
        if len(time_series) > 0:
            ax.plot(time_series, lcav_series, 'b-', linewidth=1.5)
            ax.set_xlabel('Time t·U/c', fontsize=12)
            ax.set_ylabel('L_cav / c', fontsize=12)
            ax.set_title(f'Cloud Cavitation Time History: Clark-Y, σ={sigma}, α=6°', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.text(0.02, 0.95, f'No self-sustained shedding observed\n(Amplitude = {case_data.get("shedding_amplitude", 0):.4f})',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    fig_path = Path(__file__).parent.parent / "figures" / "p40w3_n02_cloud_history.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close()


def main():
    print("Generating P40W3-N02 validation figures...")
    print("=" * 80)
    
    results = load_results()
    
    plot_lcav_vs_sigma(results)
    plot_delta_vs_sigma(results)
    plot_grid_convergence(results)
    plot_pressure_field()
    plot_cloud_cavitation_history(results)
    
    print("=" * 80)
    print("All figures generated successfully!")


if __name__ == "__main__":
    main()
