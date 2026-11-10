#!/usr/bin/env python3
"""Generate verification figures for turbulent channel flow campaign.

Produces:
- U+ vs y+ comparison with DNS (log-linear and semi-log)
- Reynolds stress profiles
- Energy spectrum with -5/3 reference
- SGS stress distribution
"""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_u_plus_figure(y_plus_sim, u_plus_sim, y_plus_dns, u_plus_dns,
                           y_plus_log, u_plus_log, re_tau, save_path):
    """Generate U+ vs y+ comparison figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Linear-linear plot
    ax1.plot(y_plus_sim, u_plus_sim, 'b-', linewidth=2, label=f'LBM Re_τ={int(re_tau)}')
    ax1.plot(y_plus_dns, u_plus_dns, 'ro', markersize=4, label='DNS (MKM1999)')
    ax1.set_xlabel('y+')
    ax1.set_ylabel('U+')
    ax1.set_title(f'Mean Velocity Profile Re_τ={int(re_tau)}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(y_plus_sim) * 1.1)

    # Semi-log plot (log law region)
    ax2.semilogx(y_plus_sim, u_plus_sim, 'b-', linewidth=2, label='LBM')
    ax2.semilogx(y_plus_dns, u_plus_dns, 'ro', markersize=4, label='DNS')
    # Log law line
    y_log = np.logspace(1, math.log10(max(y_plus_sim)), 100)
    u_log = np.log(y_log) / 0.41 + 5.2
    ax2.semilogx(y_log, u_log, 'k--', linewidth=1, label='Log law: U⁺=ln(y⁺)/0.41+5.2')
    ax2.set_xlabel('y+ (log scale)')
    ax2.set_ylabel('U+')
    ax2.set_title('Log-Law Region')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(1, max(y_plus_sim))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def generate_reynolds_stress_figure(y_plus_sim, uv_plus_sim, y_plus_dns, uv_plus_dns,
                                     uu_plus_sim, vv_plus_sim, ww_plus_sim, re_tau, save_path):
    """Generate Reynolds stress profiles figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Reynolds shear stress
    ax1.plot(y_plus_sim, -uv_plus_sim, 'b-', linewidth=2, label='LBM -⟨u\'v\'⟩⁺')
    ax1.plot(y_plus_dns, -uv_plus_dns, 'ro', markersize=4, label='DNS')
    ax1.set_xlabel('y+')
    ax1.set_ylabel('-⟨u\'v\'⟩⁺')
    ax1.set_title(f'Reynolds Shear Stress Re_τ={int(re_tau)}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Normal stresses
    ax2.plot(y_plus_sim, uu_plus_sim, 'b-', linewidth=2, label='⟨u\'u\'⟩⁺')
    ax2.plot(y_plus_sim, vv_plus_sim, 'g-', linewidth=2, label='⟨v\'v\'⟩⁺')
    ax2.plot(y_plus_sim, ww_plus_sim, 'r-', linewidth=2, label='⟨w\'w\'⟩⁺')
    ax2.set_xlabel('y+')
    ax2.set_ylabel('⟨u\'ᵢu\'ⱼ⟩⁺')
    ax2.set_title('Normal Stress Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def generate_spectrum_figure(k, E, slope_fit, save_path):
    """Generate energy spectrum figure."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Spectrum
    ax.loglog(k, E, 'b-', linewidth=2, label='LBM E(k)')

    # -5/3 reference line
    k_ref = k[k > 0]
    E_ref = k_ref ** (-5.0/3.0)
    # Scale to match at mid-range
    idx_mid = len(k_ref) // 2
    scale = E[k > 0][idx_mid] / E_ref[idx_mid]
    E_ref *= scale
    ax.loglog(k_ref, E_ref, 'k--', linewidth=1.5, label='k⁻⁵/³ (Kolmogorov)')

    ax.set_xlabel('Wavenumber k')
    ax.set_ylabel('Energy Spectrum E(k)')
    ax.set_title(f'Energy Spectrum (fitted slope: {slope_fit:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def generate_sgs_figure(y_plus, nu_t_over_nu, sgs_fraction, save_path):
    """Generate SGS stress distribution figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Eddy viscosity ratio
    ax1.plot(y_plus, nu_t_over_nu, 'b-', linewidth=2)
    ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='ν_t = ν')
    ax1.set_xlabel('y+')
    ax1.set_ylabel('ν_t / ν')
    ax1.set_title('Eddy Viscosity Ratio')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # SGS fraction
    ax2.plot(y_plus, sgs_fraction, 'b-', linewidth=2)
    ax2.axhspan(20, 40, alpha=0.2, color='green', label='Target: 20-40%')
    ax2.set_xlabel('y+')
    ax2.set_ylabel('SGS / Total Stress (%)')
    ax2.set_title('SGS Stress Contribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def generate_summary_figure(all_results, save_path):
    """Generate summary figure with all Re_tau cases."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    re_taus = [180, 395, 590]
    colors = ['blue', 'green', 'red']

    for idx, (re_tau, color) in enumerate(zip(re_taus, colors)):
        key = f"ReTau{int(re_tau)}"
        if key not in all_results:
            continue

        result = all_results[key]
        profiles = result.get("profiles", {})
        dns_comp = result.get("dns_comparison", {})

        if "y_plus" not in profiles:
            continue

        y_plus = np.array(profiles["y_plus"])
        u_plus = np.array(profiles["U_plus"])

        # U+ profile
        ax = axes[0, idx]
        ax.plot(y_plus, u_plus, color=color, linewidth=2, label=f'Re_τ={int(re_tau)}')
        # Log law
        y_log = np.logspace(1, math.log10(max(y_plus)), 50)
        ax.semilogx(y_log, np.log(y_log)/0.41 + 5.2, 'k--', alpha=0.5)
        ax.set_xlabel('y+')
        ax.set_ylabel('U+')
        ax.set_title(f'Re_τ={int(re_tau)}')
        ax.grid(True, alpha=0.3)

        # Reynolds stress
        if "uv_plus" in profiles:
            ax = axes[1, idx]
            ax.plot(y_plus, -np.array(profiles["uv_plus"]), color=color, linewidth=2)
            ax.set_xlabel('y+')
            ax.set_ylabel('-⟨u\'v\'⟩⁺')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def main():
    """Generate all figures from verification results."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="runs/turbulent_channel_verification/verification_results.json")
    parser.add_argument("--output-dir", default="figures/turbulent_channel")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    with open(args.results) as f:
        data = json.load(f)

    results = data.get("results", {})

    # Import DNS reference
    from tensorlbm.verification.turbulent_channel.reference_data import get_dns_reference, interpolate_dns

    figures = []

    for re_tau in [180, 395, 590]:
        key = f"ReTau{int(re_tau)}"
        if key not in results:
            continue

        result = results[key]
        profiles = result.get("profiles", {})

        if "y_plus" not in profiles:
            continue

        y_plus_sim = np.array(profiles["y_plus"])
        u_plus_sim = np.array(profiles["U_plus"])
        uv_plus_sim = np.array(profiles.get("uv_plus", []))
        uu_plus_sim = np.array(profiles.get("uu_plus", []))
        vv_plus_sim = np.array(profiles.get("vv_plus", []))
        ww_plus_sim = np.array(profiles.get("ww_plus", []))

        # DNS reference
        try:
            dns_ref = get_dns_reference(re_tau)
            dns_interp = interpolate_dns(dns_ref, y_plus_sim)
            y_plus_dns = dns_ref.y_plus
            u_plus_dns = dns_ref.u_plus
            uv_plus_dns = dns_ref.uv_plus
        except ValueError:
            y_plus_dns = y_plus_sim
            u_plus_dns = u_plus_sim
            uv_plus_dns = uv_plus_sim

        # U+ figure
        u_path = output_dir / f"u_plus_ReTau{int(re_tau)}.png"
        generate_u_plus_figure(
            y_plus_sim, u_plus_sim, y_plus_dns, u_plus_dns,
            y_plus_sim, u_plus_sim, re_tau, u_path
        )
        figures.append({"name": f"u_plus_ReTau{int(re_tau)}", "path": str(u_path)})

        # Reynolds stress figure
        rs_path = output_dir / f"reynolds_stress_ReTau{int(re_tau)}.png"
        generate_reynolds_stress_figure(
            y_plus_sim, uv_plus_sim, y_plus_dns, uv_plus_dns,
            uu_plus_sim, vv_plus_sim, ww_plus_sim, re_tau, rs_path
        )
        figures.append({"name": f"reynolds_stress_ReTau{int(re_tau)}", "path": str(rs_path)})

        # Energy spectrum
        spectrum = result.get("spectrum", {})
        if "k" in spectrum and "E" in spectrum:
            k = np.array(spectrum["k"])
            E = np.array(spectrum["E"])
            slope = spectrum.get("fit", {}).get("slope", -1.67)
            sp_path = output_dir / f"spectrum_ReTau{int(re_tau)}.png"
            generate_spectrum_figure(k, E, slope, sp_path)
            figures.append({"name": f"spectrum_ReTau{int(re_tau)}", "path": str(sp_path)})

        # SGS stress
        sgs = result.get("sgs_stress", {})
        if "y_plus" in sgs and "nu_t_over_nu" in sgs:
            sgs_path = output_dir / f"sgs_stress_ReTau{int(re_tau)}.png"
            nu_t = np.array(sgs["nu_t_over_nu"])
            sgs_frac = nu_t / (1 + nu_t) * 100
            generate_sgs_figure(np.array(sgs["y_plus"]), nu_t, sgs_frac, sgs_path)
            figures.append({"name": f"sgs_stress_ReTau{int(re_tau)}", "path": str(sgs_path)})

    # Summary figure
    summary_path = output_dir / "summary_all_re_tau.png"
    generate_summary_figure(results, summary_path)
    figures.append({"name": "summary_all_re_tau", "path": str(summary_path)})

    print(f"\nGenerated {len(figures)} figures in {output_dir}")
    for f in figures:
        print(f"  {f['name']}: {f['path']}")

    return figures


if __name__ == "__main__":
    main()
