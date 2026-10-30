#!/usr/bin/env python3
"""Generate diagnostic plots for the permeability SDAA verification report."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main():
    # Load scan_summary.json
    summary_path = HERE / "scan_summary.json"
    with open(summary_path) as f:
        scan_summary = json.load(f)

    out_dir = HERE / "sdaa_plots"
    out_dir.mkdir(exist_ok=True)

    # Group by phi
    by_phi = {}
    for r in scan_summary:
        by_phi.setdefault(r['phi'], []).append(r)

    # Plot 1: Error convergence vs d
    print("Generating error convergence plot...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for phi, recs in sorted(by_phi.items()):
        ds = [r['d'] for r in recs]
        errs = [r['err_pct'] for r in recs]
        ax = axes[0]
        ax.plot(ds, errs, 'o-', label=f'φ={phi}', markersize=8, linewidth=2)
        ax.set_xlabel('Grid size d', fontsize=12)
        ax.set_ylabel('Error vs k_ref (%)', fontsize=12)
        ax.set_title('Permeability Error Convergence', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=3.0, color='r', linestyle='--', alpha=0.7, linewidth=2, label='3% threshold')

    # Plot 2: k_sim vs k_ref (parity plot)
    ax = axes[1]
    all_k_ref = [r['k_ref'] for r in scan_summary]
    all_k_sim = [r['k_sim'] for r in scan_summary]
    for phi, recs in sorted(by_phi.items()):
        krefs = [r['k_ref'] for r in recs]
        ksims = [r['k_sim'] for r in recs]
        ax.scatter(krefs, ksims, label=f'φ={phi}', s=100, alpha=0.7)
    lims = [min(all_k_ref + all_k_sim) * 0.5, max(all_k_ref + all_k_sim) * 1.5]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='1:1 line')
    ax.set_xlabel('k_ref (Sangani & Acrivos 1982)', fontsize=12)
    ax.set_ylabel('k_sim (LBM)', fontsize=12)
    ax.set_title('Parity: k_sim vs k_ref', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "error_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'error_convergence.png'}")

    # Plot 3: Error vs phi_actual (showing the effect of actual solid fraction)
    print("Generating phi_actual sensitivity plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for phi, recs in sorted(by_phi.items()):
        phi_acts = [r['phi_actual'] for r in recs]
        errs = [r['err_pct'] for r in recs]
        ds = [r['d'] for r in recs]
        scatter = ax.scatter(phi_acts, errs, c=ds, cmap='viridis', s=100, alpha=0.7, 
                            label=f'φ_nominal={phi}')
        for i, d in enumerate(ds):
            ax.annotate(f'd={d}', (phi_acts[i], errs[i]), textcoords="offset points", 
                       xytext=(5, 5), fontsize=9)
    
    ax.set_xlabel('φ_actual (actual solid fraction)', fontsize=12)
    ax.set_ylabel('Error vs k_ref (%)', fontsize=12)
    ax.set_title('Error Sensitivity to Actual Solid Fraction', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, label='Grid size d')
    plt.tight_layout()
    fig.savefig(out_dir / "phi_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir / 'phi_sensitivity.png'}")

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
