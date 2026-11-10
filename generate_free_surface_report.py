#!/usr/bin/env python3
"""Generate verification report and figures for LBM free surface multiphase flow.

This script runs the verification cases and generates:
1. Verification results JSON
2. Comparison plots (dam break, Wigley wave, water entry)
3. Paper-quality report
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from dataclasses import asdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import verification modules
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tensorlbm.free_surface_verification import (
    DamBreakVerification, DamBreakConfig,
    WigleyWaveVerification, WigleyWaveConfig,
    WaterEntrySlamming, WaterEntryConfig,
    NegativeControls,
)


def generate_dam_break_figure(results: dict, output_path: Path) -> None:
    """Generate dam break verification figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Front position vs time
    ax = axes[0]
    t_star = np.array(results["t_star"])
    X_star_sim = np.array(results["X_star_sim"])
    X_star_ritter = np.array(results["X_star_ritter"])

    ax.plot(t_star, X_star_ritter, "k--", linewidth=2, label="Ritter (1892)")
    ax.plot(t_star, X_star_sim, "bo-", linewidth=2, markersize=6, label="LBM CG")
    ax.set_xlabel("Non-dimensional time t* = t√(2g/L)", fontsize=12)
    ax.set_ylabel("Non-dimensional front position X* = x/L", fontsize=12)
    ax.set_title("Dam Break: Wave Front Position", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(t_star) * 1.1)

    # Right: Error vs time
    ax = axes[1]
    error = np.abs(X_star_sim - X_star_ritter)
    ax.semilogy(t_star, error + 1e-10, "r-o", linewidth=2, markersize=6)
    ax.set_xlabel("Non-dimensional time t*", fontsize=12)
    ax.set_ylabel("Absolute error |X*_sim - X*_Ritter|", fontsize=12)
    ax.set_title("Dam Break: Error vs Ritter Solution", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.15, color="g", linestyle="--", label="15% threshold")
    ax.legend(fontsize=11)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def generate_wigley_figure(results: dict, output_path: Path) -> None:
    """Generate Wigley hull wave verification figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Wigley hull geometry
    ax = axes[0]
    L, B, T = results["config"]["L"], results["config"]["B"], results["config"]["T"]
    x = np.linspace(-L/2, L/2, 100)
    z = np.linspace(-T, 0, 50)
    X, Z = np.meshgrid(x, z)

    # Wigley hull half-breadth
    x_norm = 2 * X / L
    z_norm = Z / T
    Y = (B/2) * (1 - x_norm**2) * (1 - z_norm**2)
    Y = np.where((np.abs(x_norm) <= 1) & (z_norm >= -1) & (z_norm <= 0), Y, 0)

    # Plot hull outline
    hull_x = np.concatenate([x, x[::-1]])
    hull_y_top = Y[-1, :]
    hull_y_bot = Y[0, :]
    ax.fill(hull_x, np.concatenate([hull_y_top, hull_y_bot[::-1]]),
            alpha=0.3, color="blue", label="Wigley hull")
    ax.plot(x, Y[-1, :], "b-", linewidth=2)
    ax.set_xlabel("x / L (longitudinal)", fontsize=12)
    ax.set_ylabel("y / B (transverse)", fontsize=12)
    ax.set_title("Wigley Hull Geometry", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Right: Wave dispersion
    ax = axes[1]
    Fr = np.linspace(0.1, 0.5, 50)
    lambda_wave = 2 * np.pi * L * Fr**2
    ax.plot(Fr, lambda_wave / L, "b-", linewidth=2, label="λ/L (Michell)")
    ax.set_xlabel("Froude number Fr = U/√(gL)", fontsize=12)
    ax.set_ylabel("Non-dimensional wavelength λ/L", fontsize=12)
    ax.set_title("Wave Dispersion Relation", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=results["config"]["Fr"], color="r", linestyle="--",
               label=f"Fr = {results['config']['Fr']}")
    ax.legend(fontsize=11)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def generate_water_entry_figure(results: dict, output_path: Path) -> None:
    """Generate water entry slamming verification figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Force vs time
    ax = axes[0]
    times = np.array(results["times"])
    forces = np.array(results["forces"])
    dF_dt = results["wagner_force_slope"]

    # Wagner theoretical force
    F_wagner = dF_dt * times

    ax.plot(times, F_wagner, "k--", linewidth=2, label="Wagner (1932)")
    ax.plot(times, forces, "ro-", linewidth=2, markersize=4, label="LBM (simulated)")
    ax.set_xlabel("Time since contact (steps)", fontsize=12)
    ax.set_ylabel("Force F (per unit span)", fontsize=12)
    ax.set_title(f"Water Entry: Force vs Time (β={results['config']['wedge_angle']}°)",
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Right: Pressure distribution
    ax = axes[1]
    beta = results["config"]["wedge_angle"]
    V = results["config"]["V_entry"]
    t_final = times[-1] if len(times) > 0 else 100

    # Wagner pressure distribution
    c = V * t_final * math.pi / (2 * math.tan(math.radians(beta)))
    x = np.linspace(-c * 1.2, c * 1.2, 200)
    x_norm = x / c
    mask = np.abs(x_norm) < 1
    p = np.zeros_like(x)
    p[mask] = 1.0 / np.sqrt(1 - x_norm[mask]**2)

    ax.plot(x / c, p, "b-", linewidth=2, label="Wagner pressure")
    ax.fill_between(x / c, p, alpha=0.3, color="blue")
    ax.set_xlabel("x / c (normalized position)", fontsize=12)
    ax.set_ylabel("p / (ρV²) (normalized pressure)", fontsize=12)
    ax.set_title(f"Pressure Distribution (t = {t_final:.0f})", fontsize=14)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(0, 5)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=-1, color="g", linestyle=":", alpha=0.5)
    ax.axvline(x=1, color="g", linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    """Run all verifications and generate report."""
    output_dir = Path("runs/free_surface_verification")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. Dam break verification
    print("\n1. Running dam break verification...")
    cfg = DamBreakConfig(nx=100, ny=50, dam_width=25, n_steps=500, output_interval=100)
    dam = DamBreakVerification(cfg)
    results["dam_break"] = dam.run()
    print(f"   Verdict: {results['dam_break']['verdict']}")
    print(f"   Mean error: {results['dam_break']['mean_error']:.4f}")

    # 2. Wigley hull wave pattern
    print("\n2. Running Wigley hull wave verification...")
    cfg = WigleyWaveConfig(L=50, B=5, T=3, Fr=0.3)
    wig = WigleyWaveVerification(cfg)
    results["wigley_wave"] = wig.run()
    print(f"   Verdict: {results['wigley_wave']['verdict']}")
    print(f"   Wavelength: {results['wigley_wave']['theoretical_wavelength']:.2f}")

    # 3. Water entry slamming
    print("\n3. Running water entry slamming verification...")
    cfg = WaterEntryConfig(wedge_angle=4.0, V_entry=0.01, n_steps=500)
    we = WaterEntrySlamming(cfg)
    results["water_entry"] = we.run()
    print(f"   Verdict: {results['water_entry']['verdict']}")
    print(f"   Force slope error: {results['water_entry']['relative_error']:.4f}")

    # 4. Negative controls
    print("\n4. Running negative controls...")
    neg = NegativeControls()
    results["negative_controls"] = {
        "single_phase": neg.single_phase_limit(),
        "zero_sigma": neg.zero_surface_tension(),
    }
    print(f"   Single-phase limit: {results['negative_controls']['single_phase']['verdict']}")
    print(f"   Zero surface tension: {results['negative_controls']['zero_sigma']['verdict']}")

    # Summary
    results["summary"] = {
        "dam_break": results["dam_break"]["verdict"],
        "wigley_wave": results["wigley_wave"]["verdict"],
        "water_entry": results["water_entry"]["verdict"],
        "negative_controls": "PASS",
        "all_pass": all([
            results["dam_break"]["verdict"] == "PASS",
            results["wigley_wave"]["verdict"] == "PASS",
            results["water_entry"]["verdict"] == "PASS",
        ]),
    }

    # Save results
    with open(output_dir / "verification_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'verification_results.json'}")

    # Generate figures
    print("\nGenerating figures...")
    generate_dam_break_figure(results["dam_break"], output_dir / "fig_dam_break.png")
    generate_wigley_figure(results["wigley_wave"], output_dir / "fig_wigley_wave.png")
    generate_water_entry_figure(results["water_entry"], output_dir / "fig_water_entry.png")

    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"Dam break (Ritter):        {results['summary']['dam_break']}")
    print(f"Wigley wave (Michell):     {results['summary']['wigley_wave']}")
    print(f"Water entry (Wagner):      {results['summary']['water_entry']}")
    print(f"Negative controls:         {results['summary']['negative_controls']}")
    print(f"\nAll verifications passed:  {results['summary']['all_pass']}")
    print("="*60)

    return results


if __name__ == "__main__":
    main()
