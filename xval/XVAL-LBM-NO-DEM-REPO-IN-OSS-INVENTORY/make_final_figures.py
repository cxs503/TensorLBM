#!/usr/bin/env python3
"""Generate figures for the final XVAL-LBM-NO-DEM-REPO-IN-OSS-INVENTORY report.

This script generates publication-quality figures showing:
1. Restitution coefficient convergence (before/after fix)
2. Contact duration and peak overlap convergence
3. Settling velocity validation
4. LBM fluid-side Cd(Re) comparison
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
})

HERE = Path(__file__).resolve().parent
RESULTS = json.loads((HERE / "results.json").read_text())
ROUND3 = json.loads((HERE / "round3_lbm_cd.json").read_text())

def fig1_restitution_convergence():
    """Figure 1: Restitution coefficient convergence before/after fix."""
    contact = RESULTS["arms"]["contact"]
    
    # Extract data (clip=True is the fixed version)
    steps = [r["steps_per_contact"] for r in contact if r["no_tension_clip"]]
    e_sim = [r["e_sim"] for r in contact if r["no_tension_clip"]]
    e_analytic = contact[0]["e_analytic"]
    
    # Round 2 data (before fix) - from REPORT.md
    e_sim_old = [0.579295, 0.575345, 0.573489]
    
    fig, ax = plt.subplots()
    
    ax.axhline(e_analytic, color='k', linestyle='--', linewidth=2, label=f'Analytical: {e_analytic:.6f}')
    ax.plot(steps, e_sim_old, 'ro-', markersize=8, linewidth=2, label='Before fix (max(0,F) clip)')
    ax.plot(steps, e_sim, 'bs-', markersize=8, linewidth=2, label='After fix (overlap>0 check)')
    
    ax.set_xlabel('Steps per contact time ($t_c/\\Delta t$)')
    ax.set_ylabel('Restitution coefficient $e$')
    ax.set_title('Contact Restitution Coefficient Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Add error annotations
    for i, (s, e_new, e_old) in enumerate(zip(steps, e_sim, e_sim_old)):
        err_new = (e_new - e_analytic) / e_analytic * 100
        err_old = (e_old - e_analytic) / e_analytic * 100
        ax.annotate(f'{err_new:+.2f}%', (s, e_new), textcoords="offset points",
                   xytext=(10, 5), fontsize=9, color='blue')
        if i == 0:  # Only annotate first point for old data
            ax.annotate(f'{err_old:+.2f}%', (s, e_old), textcoords="offset points",
                       xytext=(10, -15), fontsize=9, color='red')
    
    plt.tight_layout()
    plt.savefig(HERE / "fig1_restitution_convergence.png", dpi=150, bbox_inches='tight')
    print(f"Saved fig1_restitution_convergence.png")
    plt.close()


def fig2_contact_duration_overlap():
    """Figure 2: Contact duration and peak overlap convergence."""
    contact = RESULTS["arms"]["contact"]
    
    steps = [r["steps_per_contact"] for r in contact if r["no_tension_clip"]]
    t_c_sim = [r["t_c_sim_s"] for r in contact if r["no_tension_clip"]]
    t_c_analytic = contact[0]["t_c_analytic_s"]
    delta_max_sim = [r["delta_max_sim_m"] for r in contact if r["no_tension_clip"]]
    delta_max_analytic = contact[0]["delta_max_analytic_m"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Contact duration
    ax1.axhline(t_c_analytic, color='k', linestyle='--', linewidth=2, 
                label=f'Analytical: {t_c_analytic*1e6:.2f} μs')
    ax1.plot(steps, [t*1e6 for t in t_c_sim], 'bs-', markersize=8, linewidth=2)
    ax1.set_xlabel('Steps per contact time')
    ax1.set_ylabel('Contact duration $t_c$ (μs)')
    ax1.set_title('(a) Contact Duration')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Peak overlap
    ax2.axhline(delta_max_analytic*1e6, color='k', linestyle='--', linewidth=2,
                label=f'Analytical: {delta_max_analytic*1e6:.3f} μm')
    ax2.plot(steps, [d*1e6 for d in delta_max_sim], 'bs-', markersize=8, linewidth=2)
    ax2.set_xlabel('Steps per contact time')
    ax2.set_ylabel('Peak overlap $\\delta_{max}$ (μm)')
    ax2.set_title('(b) Peak Overlap')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(HERE / "fig2_contact_duration_overlap.png", dpi=150, bbox_inches='tight')
    print(f"Saved fig2_contact_duration_overlap.png")
    plt.close()


def fig3_settling_velocity():
    """Figure 3: Settling velocity validation (Stokes and Schiller-Naumann)."""
    stokes = RESULTS["arms"]["stokes_settling"]
    sn = RESULTS["arms"]["schiller_naumann_settling"]
    
    fig, ax = plt.subplots()
    
    # Stokes
    D_stokes = [s["diameter_m"] for s in stokes]
    u_stokes_sim = [s["u_terminal_sim"] for s in stokes]
    u_stokes_ref = [s["u_terminal_reference"] for s in stokes]
    
    ax.plot(D_stokes, u_stokes_ref, 'k-', linewidth=2, label='Stokes (analytical)')
    ax.plot(D_stokes, u_stokes_sim, 'ko', markersize=8, label='Stokes (sim)')
    
    # Schiller-Naumann
    D_sn = [s["diameter_m"] for s in sn]
    u_sn_sim = [s["u_terminal_sim"] for s in sn]
    u_sn_ref = [s["u_terminal_reference"] for s in sn]
    
    ax.plot(D_sn, u_sn_ref, 'r-', linewidth=2, label='Schiller-Naumann vs Clift-Gauvin')
    ax.plot(D_sn, u_sn_sim, 'rs', markersize=8, label='SN (sim)')
    
    ax.set_xlabel('Particle diameter $D$ (m)')
    ax.set_ylabel('Terminal velocity $u_t$ (m/s)')
    ax.set_title('Settling Velocity Validation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(HERE / "fig3_settling_velocity.png", dpi=150, bbox_inches='tight')
    print(f"Saved fig3_settling_velocity.png")
    plt.close()


def fig4_lbm_cd_re():
    """Figure 4: LBM fluid-side Cd(Re) comparison with Clift-Gauvin."""
    rows = ROUND3["rows"]
    
    # Extract unique Re values
    Re_vals = sorted(set(r["Re"] for r in rows))
    
    # Get Clift-Gauvin reference
    cd_cg = [next(r["cd_clift_gauvin"] for r in rows if r["Re"] == Re) for Re in Re_vals]
    
    # Get raw LBM (lambda=1/8)
    cd_raw = []
    for Re in Re_vals:
        r = next((r for r in rows if r["Re"] == Re and r["variant"] == "raw_run_lambda"), None)
        cd_raw.append(r["cd_lbm"] if r else None)
    
    # Get resolution extrapolated
    cd_res = []
    for Re in Re_vals:
        r = next((r for r in rows if r["Re"] == Re and r["variant"] == "resolution_extrapolated"), None)
        cd_res.append(r["cd_lbm"] if r and r["cd_lbm"] is not None else None)
    
    fig, ax = plt.subplots()
    
    ax.loglog(Re_vals, cd_cg, 'k-', linewidth=2, label='Clift-Gauvin (reference)')
    ax.loglog(Re_vals, cd_raw, 'ro-', markersize=8, linewidth=2, label='LBM raw (λ=1/8)')
    
    # Filter None values for resolution extrapolated
    Re_res = [Re for Re, cd in zip(Re_vals, cd_res) if cd is not None]
    cd_res_filtered = [cd for cd in cd_res if cd is not None]
    if cd_res_filtered:
        ax.loglog(Re_res, cd_res_filtered, 'bs-', markersize=8, linewidth=2, 
                 label='LBM 1/D→0 extrapolation')
    
    ax.set_xlabel('Reynolds number $Re$')
    ax.set_ylabel('Drag coefficient $C_D$')
    ax.set_title('LBM Fluid-Side Drag Coefficient vs Clift-Gauvin')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(HERE / "fig4_lbm_cd_re.png", dpi=150, bbox_inches='tight')
    print(f"Saved fig4_lbm_cd_re.png")
    plt.close()


def fig5_error_summary():
    """Figure 5: Error summary bar chart."""
    contact = RESULTS["arms"]["contact"]
    
    # Contact arm errors (fixed version)
    steps = [r["steps_per_contact"] for r in contact if r["no_tension_clip"]]
    e_err = [r["e_rel_err"] * 100 for r in contact if r["no_tension_clip"]]
    t_c_err = [r["t_c_rel_err"] * 100 for r in contact if r["no_tension_clip"]]
    delta_err = [r["delta_max_rel_err"] * 100 for r in contact if r["no_tension_clip"]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(steps))
    width = 0.25
    
    bars1 = ax.bar(x - width, e_err, width, label='Restitution $e$', color='blue', alpha=0.7)
    bars2 = ax.bar(x, t_c_err, width, label='Contact duration $t_c$', color='green', alpha=0.7)
    bars3 = ax.bar(x + width, delta_err, width, label='Peak overlap $\\delta_{max}$', color='orange', alpha=0.7)
    
    ax.axhline(15, color='r', linestyle='--', linewidth=2, label='15% threshold (owner standard)')
    ax.axhline(2, color='k', linestyle=':', linewidth=2, label='2% threshold (in-house)')
    ax.axhline(-2, color='k', linestyle=':', linewidth=2)
    
    ax.set_xlabel('Steps per contact time')
    ax.set_ylabel('Relative error (%)')
    ax.set_title('Contact Arm Validation: Error Summary')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}' for s in steps])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(HERE / "fig5_error_summary.png", dpi=150, bbox_inches='tight')
    print(f"Saved fig5_error_summary.png")
    plt.close()


if __name__ == "__main__":
    fig1_restitution_convergence()
    fig2_contact_duration_overlap()
    fig3_settling_velocity()
    fig4_lbm_cd_re()
    fig5_error_summary()
    print("\nAll figures generated successfully!")
