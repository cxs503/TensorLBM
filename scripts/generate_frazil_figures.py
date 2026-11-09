#!/usr/bin/env python3
"""Generate figures from saved campaign results JSON."""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

def main():
    json_path = ROOT / "artifacts" / "frazil_campaign_p54w2_n08_results.json"
    with open(json_path) as f:
        results = json.load(f)
    
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: mu_eff and n vs c_v
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    cv = np.array(results["a_dilute"]["cv_sweep"])
    mu_r = np.array(results["a_dilute"]["mu_r_sweep"])
    n_vals = np.array(results["a_dilute"]["n_sweep"])
    
    ax1.plot(cv, mu_r, 'b-', linewidth=2, label='Krieger-Dougherty $\\mu_r(\\phi)$')
    ax1.axhline(y=1.0, color='r', linestyle='--', label='$\\mu_w$ reference')
    ax1.set_xlabel('Ice volume fraction $c_v$', fontsize=12)
    ax1.set_ylabel('Relative viscosity $\\mu_r$', fontsize=12)
    ax1.set_title('(a) Effective viscosity vs concentration', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 0.4)
    
    ax2.plot(cv, n_vals, 'g-', linewidth=2, label='Flow index $n(\\phi)$')
    ax2.axhline(y=1.0, color='r', linestyle='--', label='Newtonian ($n=1$)')
    ax2.set_xlabel('Ice volume fraction $c_v$', fontsize=12)
    ax2.set_ylabel('Power-law index $n$', fontsize=12)
    ax2.set_title('Flow index vs concentration', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 0.4)
    
    plt.tight_layout()
    fig.savefig(fig_dir / "frazil_campaign_mu_eff_n_vs_cv.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / 'frazil_campaign_mu_eff_n_vs_cv.png'}")

    # Figure 2: Poiseuille profiles
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, n_key in enumerate(["n0.5", "n0.7", "n1.0"]):
        ax = axes[idx]
        data = results["b_poiseuille"][n_key]
        y = np.array(data["y_phys"])
        u_exact = np.array(data["u_exact"])
        u_num = np.array(data["u_num"])
        err = data["L2_error"]
        
        ax.plot(u_exact, y, 'b-', linewidth=2, label='Analytical')
        ax.plot(u_num, y, 'r--', linewidth=1.5, label=f'LBM (err={err:.2%})')
        ax.set_xlabel('Velocity $u$', fontsize=11)
        ax.set_ylabel('Wall-normal $y$', fontsize=11)
        n_val = n_key[1:]
        ax.set_title(f'$n={n_val}$ (L2 err={err:.2%})', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('(b) Power-law Poiseuille profiles vs analytical solution', fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(fig_dir / "frazil_campaign_poiseuille_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / 'frazil_campaign_poiseuille_profiles.png'}")

    # Figure 3: Settling velocity vs concentration
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    cv_s = np.array(results["c_settling"]["cv_sweep"])
    v_model = np.array(results["c_settling"]["v_rz_model"])
    v_exact = np.array(results["c_settling"]["v_rz_exact"])
    v_s0 = results["c_settling"]["v_s0"]
    
    ax1.plot(cv_s, v_model / v_s0, 'b-', linewidth=2, label='Richardson-Zaki model')
    ax1.plot(cv_s, v_exact / v_s0, 'k--', linewidth=1, label='Exact RZ formula')
    ax1.set_xlabel('Volume fraction $c_v$', fontsize=12)
    ax1.set_ylabel('$v_s / v_{s0}$', fontsize=12)
    ax1.set_title('(c) Hindered settling velocity attenuation', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # LBM settling column summary
    v_meas = results["c_settling"]["lbm_interface_velocity"]
    v_exp = results["c_settling"]["lbm_expected_velocity"]
    rel_err = results["c_settling"]["lbm_relative_error"]
    
    times = results["c_settling"].get("times", [])
    heights = results["c_settling"].get("heights", [])
    
    if len(times) > 0 and len(heights) > 0:
        times_arr = np.array(times)
        heights_arr = np.array(heights)
        ax2.plot(times_arr, heights_arr, 'bo-', markersize=3, label=f'LBM interface (err={rel_err:.2%})')
        h0 = heights_arr[0]
        ax2.plot(times_arr, h0 - v_exp * times_arr, 'r--', linewidth=1.5, label=f'Expected (v={v_exp:.4f})')
    else:
        ax2.text(0.5, 0.6, 
                f'LBM settling column\n$v_{{meas}}$ = {v_meas:.5f}\n$v_{{exp}}$ = {v_exp:.5f}\nrel. err = {rel_err:.2%}', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.set_xlabel('Time [steps]', fontsize=12)
    ax2.set_ylabel('Interface height', fontsize=12)
    ax2.set_title('LBM settling column: Kynch front', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(fig_dir / "frazil_campaign_settling_velocity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / 'frazil_campaign_settling_velocity.png'}")

    # Figure 4: Krieger-Dougherty viscosity curve
    fig, ax = plt.subplots(figsize=(8, 5))
    
    phi = np.array(results["d_kd"]["phi_sweep"])
    mu_r_kd = np.array(results["d_kd"]["mu_r_model"])
    
    ax.semilogy(phi, mu_r_kd, 'b-', linewidth=2, label='Krieger-Dougherty')
    ax.axvline(x=0.6, color='r', linestyle='--', label='$\\phi_{max}=0.6$')
    ax.set_xlabel('Volume fraction $\\phi$', fontsize=12)
    ax.set_ylabel('Relative viscosity $\\mu_r$', fontsize=12)
    ax.set_title('(d) Krieger-Dougherty effective viscosity', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    fig.savefig(fig_dir / "frazil_campaign_kd_viscosity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_dir / 'frazil_campaign_kd_viscosity.png'}")

    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
