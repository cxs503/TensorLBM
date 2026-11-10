#!/usr/bin/env python3
"""Generate figures for P96W3-N05 verification report."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorlbm.scalar import (
    run_pure_diffusion,
    run_pure_convection,
    fit_diffusivity,
    tau_from_diffusivity,
    scalar_diffusivity,
)

DEVICE = torch.device("cpu")
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "p96w3_n05"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def fig_c1():
    """C1: Pure diffusion σ²(t) for three D values."""
    print("Generating C1 figure...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    for i, D_target in enumerate([0.02, 0.05, 0.10]):
        tau_s = tau_from_diffusivity(D_target)
        state = run_pure_diffusion(64, 64, tau_s, 400, sigma0=5.0,
                                   device=DEVICE, record_interval=5)
        times = [h["time"] for h in state.history]
        variances = [h["var_x"] for h in state.history]
        
        ax = axes[i]
        ax.plot(times, variances, 'b.', markersize=3, label='LBM')
        t_arr = np.array(times)
        ax.plot(t_arr, variances[0] + 2 * D_target * t_arr, 'r--',
                label=f'σ²₀ + 2Dt (D={D_target})')
        ax.set_xlabel('Time (steps)')
        ax.set_ylabel('σ²_x')
        ax.set_title(f'D = {D_target}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('C1: Pure Diffusion — Second Moment Growth', fontsize=13)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "c1_pure_diffusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: c1_pure_diffusion.png")


def fig_c2():
    """C2: Numerical diffusion calibration."""
    print("Generating C2 figure...")
    c2_data = json.loads((ARTIFACTS_DIR / "c2_results.json").read_text())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel a: D_num/(uΔx) vs u for different τ_s
    ax = axes[0]
    tau_vals = sorted(set(r["tau_s"] for r in c2_data))
    for tau_s in tau_vals:
        subset = [r for r in c2_data if r["tau_s"] == tau_s]
        ux_vals = [r["ux"] for r in subset]
        dnum_norm = [r["D_num_norm"] for r in subset]
        ax.plot(ux_vals, dnum_norm, 'o-', label=f'τ_s={tau_s}')
    ax.set_xlabel('u (lattice)')
    ax.set_ylabel('D_num / (u·Δx)')
    ax.set_title('Numerical Diffusion vs Velocity')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Panel b: D_num/D_phys vs τ_s for different u
    ax = axes[1]
    u_vals = sorted(set(r["ux"] for r in c2_data))
    for ux in u_vals:
        subset = [r for r in c2_data if r["ux"] == ux]
        taus = [r["tau_s"] for r in subset]
        ratios = [r["D_num/D_phys"] for r in subset]
        ax.plot(taus, ratios, 'o-', label=f'u={ux}')
    ax.set_xlabel('τ_s')
    ax.set_ylabel('D_num / D_phys')
    ax.set_title('Numerical/Physical Diffusion Ratio')
    ax.axhline(0.05, color='r', linestyle='--', alpha=0.5, label='5% threshold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "c2_numerical_diffusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: c2_numerical_diffusion.png")


def fig_c3():
    """C3: Taylor-Aris D_eff comparison (using previous results)."""
    print("Generating C3 figure...")
    c3_prev = json.loads((Path("artifacts/p96w2_n05/c3_Pe10.json")).read_text())
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Single point from previous verification
    Pe = c3_prev["Pe_target"]
    D_meas = c3_prev["D_eff_measured"]
    D_anal = c3_prev["D_eff_analytical"]
    
    ax.plot(Pe, D_meas, 'bo', markersize=12, label=f'LBM (Pe={Pe})')
    ax.plot(Pe, D_anal, 'r^', markersize=12, label='Taylor–Aris analytical')
    
    # Add error bar
    err = c3_prev["relative_error"]
    ax.errorbar(Pe, D_meas, yerr=D_meas * err, fmt='none', ecolor='blue', capsize=5)
    
    ax.set_xlabel('Péclet number Pe = Ua/D')
    ax.set_ylabel('D_eff')
    ax.set_title('C3: Taylor–Aris Effective Diffusivity\n(Previous verification p96w2_n05)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text annotation
    ax.annotate(f'Error: {err*100:.2f}%',
                xy=(Pe, D_meas), xytext=(Pe*1.3, D_meas*1.1),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10)
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "c3_taylor_aris.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: c3_taylor_aris.png")


def fig_summary():
    """Summary figure showing all criteria results."""
    print("Generating summary figure...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    criteria = ['C1', 'C2', 'C3', 'C4', 'C5', 'C7']
    status = ['PASS', 'PASS', 'PASS', 'PASS', 'PARTIAL', 'PASS']
    colors = ['green' if s == 'PASS' else 'orange' for s in status]
    
    y_pos = np.arange(len(criteria))
    bars = ax.barh(y_pos, [1]*len(criteria), color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(criteria)
    ax.set_xlabel('Status')
    ax.set_title('P96W3-N05: Verification Criteria Summary')
    ax.set_xlim(0, 1.5)
    
    # Add status labels
    for i, (bar, s) in enumerate(zip(bars, status)):
        ax.text(0.5, i, s, ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xticks([])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.7, label='PASS'),
                       Patch(facecolor='orange', alpha=0.7, label='PARTIAL')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: summary.png")


def main():
    print("=" * 60)
    print("Generating figures for P96W3-N05")
    print("=" * 60)
    
    fig_c1()
    fig_c2()
    fig_c3()
    fig_summary()
    
    print("\nAll figures generated successfully!")
    print(f"Output directory: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
