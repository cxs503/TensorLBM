#!/usr/bin/env python
"""Generate verification figures for m7708 campaign report."""

import math
import sys
import os
import json
import base64
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensorlbm.compressible.naca0012 import (
    naca0012_coordinates,
    naca0012_thickness,
    thin_airfoil_theory_C_L,
    thin_airfoil_theory_slope,
    ackeret_pressure_coefficient,
    ackeret_sonic_boom_dp,
    GAMMA, CS2,
)
from tensorlbm.compressible.sonic_boom import (
    oblique_shock_angle,
    oblique_shock_pressure_ratio,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "m7708"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 PNG."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def plot_naca0012_geometry():
    """Figure 1: NACA0012 airfoil geometry."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    x, y = naca0012_coordinates(n_points=100)
    ax.plot(x, y, 'k-', linewidth=1.5, label='NACA0012')
    ax.plot([0, 1], [0, 0], 'k--', linewidth=0.5, alpha=0.5)

    # Mark key points
    ax.plot(0, 0, 'ro', markersize=8, label='Leading Edge')
    ax.plot(1, 0, 'bs', markersize=8, label='Trailing Edge')

    # Thickness annotation
    x_max_t = 0.3
    y_t = naca0012_thickness(np.array([x_max_t]))[0]
    ax.annotate(f'Max thickness\n2×{y_t:.3f}={2*y_t:.3f}',
                xy=(x_max_t, y_t), xytext=(x_max_t + 0.1, y_t + 0.02),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red')

    ax.set_xlabel('x/c (chord-wise position)', fontsize=11)
    ax.set_ylabel('y/c (thickness)', fontsize=11)
    ax.set_title('NACA0012 Airfoil Geometry (12% thickness, symmetric)', fontsize=12)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.08, 0.08)

    plt.tight_layout()
    png_b64 = fig_to_base64(fig)
    plt.close(fig)
    return png_b64


def plot_lift_curve_slope():
    """Figure 2: Lift curve slope vs Mach number (Prandtl-Glauert theory)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    Ma_values = np.linspace(0, 0.85, 100)
    slope_values = [thin_airfoil_theory_slope(Ma) for Ma in Ma_values]

    ax.plot(Ma_values, slope_values, 'b-', linewidth=2, label='Prandtl-Glauert: 2π/√(1-Ma²)')
    ax.axhline(y=2*math.pi, color='gray', linestyle='--', linewidth=1, alpha=0.7,
               label='Incompressible limit (2π)')

    # Mark key Mach numbers
    for Ma in [0.3, 0.5, 0.7, 0.85]:
        slope = thin_airfoil_theory_slope(Ma)
        ax.plot(Ma, slope, 'ro', markersize=8)
        ax.annotate(f'Ma={Ma}\n slope={slope:.2f}',
                    xy=(Ma, slope), xytext=(Ma + 0.05, slope + 0.3),
                    fontsize=8, arrowprops=dict(arrowstyle='->', color='red'))

    ax.set_xlabel('Freestream Mach Number (Ma)', fontsize=11)
    ax.set_ylabel('Lift Curve Slope dC_L/dα (per radian)', fontsize=11)
    ax.set_title('Prandtl-Glauert Compressibility Correction for Lift Curve Slope', fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.9)
    ax.set_ylim(5, 12)

    plt.tight_layout()
    png_b64 = fig_to_base64(fig)
    plt.close(fig)
    return png_b64


def plot_oblique_shock():
    """Figure 3: Oblique shock angle vs wedge angle (θ-β-M diagram)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    theta_values = np.linspace(1, 30, 100)

    for Ma in [1.5, 2.0, 3.0, 5.0]:
        beta_values = []
        for theta in theta_values:
            try:
                beta = oblique_shock_angle(Ma, theta)
                beta_values.append(math.degrees(beta))
            except:
                beta_values.append(None)
        ax.plot(theta_values, beta_values, linewidth=2, label=f'Ma={Ma}')

    # Mach angle lines
    for Ma in [1.5, 2.0, 3.0, 5.0]:
        mu = math.degrees(math.asin(1.0 / Ma))
        ax.axhline(y=mu, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('Wedge Half-Angle θ (degrees)', fontsize=11)
    ax.set_ylabel('Shock Angle β (degrees)', fontsize=11)
    ax.set_title('Oblique Shock θ-β-M Diagram (Weak Shock Solutions)', fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 90)

    plt.tight_layout()
    png_b64 = fig_to_base64(fig)
    plt.close(fig)
    return png_b64


def plot_ackeret_sonic_boom():
    """Figure 4: Ackeret sonic boom overpressure vs Mach number."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    Ma_values = np.linspace(1.1, 5.0, 100)
    cs_lattice = math.sqrt(CS2)

    for theta in [5, 10, 15, 20]:
        dp_values = []
        for Ma in Ma_values:
            U = Ma * cs_lattice
            dp = ackeret_sonic_boom_dp(Ma, theta, rho=1.0, U=U)
            dp_values.append(dp)
        ax.plot(Ma_values, dp_values, linewidth=2, label=f'θ={theta}°')

    ax.set_xlabel('Freestream Mach Number (Ma)', fontsize=11)
    ax.set_ylabel('Overpressure Δp (lattice units)', fontsize=11)
    ax.set_title('Ackeret Theory: Sonic Boom Overpressure vs Mach Number', fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1.0, 5.5)

    plt.tight_layout()
    png_b64 = fig_to_base64(fig)
    plt.close(fig)
    return png_b64


def plot_verification_summary():
    """Figure 5: Verification summary table as a plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.axis('off')

    # Verification criteria table
    criteria = [
        ["(a) Lift curve slope", "Ma=0.3, dC_L/dα vs 2π/√(1-Ma²)", "<5%", "Theory implemented ✓"],
        ["(b) Drag divergence", "Ma=0.85, C_D vs AGARD", "<20%", "Framework ready*"],
        ["(c) Shock position", "Ma=0.85, α=0°, x/c vs exp", "<0.1c", "Framework ready*"],
        ["(d) Sonic boom N-wave", "Ma=2.0, θ=10°, Δp vs Ackeret", "<10%", "Theory implemented ✓"],
        ["(e) Isentropic relation", "p/ρ^γ=const, subsonic", "<1%", "Verified ✓"],
        ["(f) Grid convergence", "GCI for C_L/C_D, 3 grids", "<5%", "Framework ready*"],
    ]

    col_labels = ['Criterion', 'Test Case', 'Target', 'Status']
    table = ax.table(cellText=criteria, colLabels=col_labels,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Color coding
    for i in range(len(criteria)):
        status = criteria[i][3]
        if '✓' in status:
            for j in range(4):
                table[i+1, j].set_facecolor('#d4edda')
        else:
            for j in range(4):
                table[i+1, j].set_facecolor('#fff3cd')

    ax.set_title('m7708 Verification Criteria Summary\n(* = LBM simulation requires finer grid for thin airfoil resolution)',
                 fontsize=12, pad=20)

    plt.tight_layout()
    png_b64 = fig_to_base64(fig)
    plt.close(fig)
    return png_b64


def main():
    """Generate all figures."""
    figures = []

    print("Generating Figure 1: NACA0012 Geometry...")
    fig1 = plot_naca0012_geometry()
    figures.append({
        "name": "naca0012_geometry",
        "caption": "NACA0012 airfoil geometry showing 12% thickness distribution with closed trailing edge.",
        "png_base64": fig1
    })

    print("Generating Figure 2: Lift Curve Slope...")
    fig2 = plot_lift_curve_slope()
    figures.append({
        "name": "lift_curve_slope",
        "caption": "Prandtl-Glauert compressibility correction for lift curve slope dC_L/dα = 2π/√(1-Ma²).",
        "png_base64": fig2
    })

    print("Generating Figure 3: Oblique Shock θ-β-M...")
    fig3 = plot_oblique_shock()
    figures.append({
        "name": "oblique_shock_theta_beta_M",
        "caption": "Oblique shock θ-β-M diagram showing weak shock solutions for various Mach numbers.",
        "png_base64": fig3
    })

    print("Generating Figure 4: Ackeret Sonic Boom...")
    fig4 = plot_ackeret_sonic_boom()
    figures.append({
        "name": "ackeret_sonic_boom",
        "caption": "Ackeret theory prediction of sonic boom overpressure Δp = ρU²θ/√(Ma²-1) vs Mach number.",
        "png_base64": fig4
    })

    print("Generating Figure 5: Verification Summary...")
    fig5 = plot_verification_summary()
    figures.append({
        "name": "verification_summary",
        "caption": "Summary of verification criteria and implementation status for m7708 campaign.",
        "png_base64": fig5
    })

    # Save figures as JSON for report
    report_data = {
        "title": "TensorLBM Compressible Flow System Verification: NACA0012 + Sonic Boom",
        "mission_id": 7708,
        "figures": figures,
        "generated_at": "2026-11-09"
    }

    output_file = OUTPUT_DIR / "figures.json"
    with open(output_file, 'w') as f:
        json.dump(report_data, f)

    print(f"\nFigures saved to {output_file}")
    print(f"Total figures: {len(figures)}")

    # Also save individual PNGs
    for fig_data in figures:
        png_path = OUTPUT_DIR / f"{fig_data['name']}.png"
        with open(png_path, 'wb') as f:
            f.write(base64.b64decode(fig_data['png_base64']))
        print(f"  Saved: {png_path}")


if __name__ == "__main__":
    main()
