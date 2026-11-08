#!/usr/bin/env python3
"""Run the seakeeping campaign and generate the report with figures.

This script:
1. Runs the full wave matrix campaign (λ/L = 0.5, 0.75, 1.0, 1.5, 2.0).
2. Extracts heave/pitch RAO via FFT.
3. Cross-validates with strip-theory frequency-domain RAO.
4. Generates publication-quality figures.
5. Outputs a JSON report with all results.
"""

import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorlbm.seakeeping.wigley_hull import WigleyIII
from tensorlbm.seakeeping.strip_theory import compute_strip_theory_rao
from tensorlbm.seakeeping.rao_campaign import run_campaign


def main():
    print("=" * 70)
    print("Wigley III Seakeeping Campaign: Heave/Pitch RAO in Regular Waves")
    print("=" * 70)

    # Hull parameters
    hull = WigleyIII()
    print(f"\nHull: Wigley III modified")
    print(f"  L = {hull.L:.3f} m")
    print(f"  B = {hull.B:.3f} m")
    print(f"  T = {hull.T:.3f} m")
    print(f"  Cb = {hull.Cb:.3f}")
    print(f"  ∇ = {hull.volume:.6f} m³")
    print(f"  A_wp = {hull.waterplane_area:.4f} m²")

    # Campaign parameters
    wave_length_ratios = [0.5, 0.75, 1.0, 1.5, 2.0]
    steepness = 1.0 / 60.0
    n_periods = 20
    n_periods_analysis = 10
    steps_per_period = 200

    print(f"\nCampaign parameters:")
    print(f"  λ/L = {wave_length_ratios}")
    print(f"  H/λ = {steepness:.4f}")
    print(f"  Periods: {n_periods} total, {n_periods_analysis} for FFT")
    print(f"  Steps/period: {steps_per_period}")

    # Run campaign
    print("\nRunning campaign...")
    results = run_campaign(
        hull=hull,
        wave_length_ratios=wave_length_ratios,
        steepness=steepness,
        n_periods=n_periods,
        n_periods_analysis=n_periods_analysis,
        steps_per_period=steps_per_period,
        depth=10.0,  # deep water
    )

    # Extract results
    campaign_results = results["results"]
    strip_theory = results["strip_theory"]

    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS: Heave and Pitch RAO")
    print("=" * 70)
    print(f"{'λ/L':>6} {'λ [m]':>8} {'ω [rad/s]':>10} {'Heave RAO':>12} {'Pitch RAO':>12}")
    print("-" * 70)

    rao_data = {
        "wave_length_ratios": [],
        "wavelengths": [],
        "omega": [],
        "heave_rao_td": [],
        "pitch_rao_td": [],
        "heave_rao_fd": [],
        "pitch_rao_fd": [],
    }

    for i, r in enumerate(campaign_results):
        lam_ratio = r.condition.wave_length_ratio
        wavelength = r.condition.wavelength
        omega = r.omega
        heave_rao = r.heave_rao
        pitch_rao = r.pitch_rao

        # Strip theory RAO at same frequency
        heave_fd = strip_theory["heave_rao"][i]
        pitch_fd = strip_theory["pitch_rao"][i]

        print(f"{lam_ratio:>6.2f} {wavelength:>8.3f} {omega:>10.3f} {heave_rao:>12.4f} {pitch_rao:>12.6f}")

        rao_data["wave_length_ratios"].append(lam_ratio)
        rao_data["wavelengths"].append(wavelength)
        rao_data["omega"].append(omega)
        rao_data["heave_rao_td"].append(heave_rao)
        rao_data["pitch_rao_td"].append(pitch_rao)
        rao_data["heave_rao_fd"].append(float(heave_fd))
        rao_data["pitch_rao_fd"].append(float(pitch_fd))

    # Generate figures
    print("\nGenerating figures...")
    fig_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    os.makedirs(fig_dir, exist_ok=True)

    # Figure 1: Heave RAO comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    omega_fd = strip_theory["omega"]
    ax.plot(omega_fd, strip_theory["heave_rao"], "b-", linewidth=2, label="Strip theory (FD)")
    ax.plot(rao_data["omega"], rao_data["heave_rao_td"], "ro", markersize=8, label="Time-domain (FFT)")
    ax.set_xlabel("ω [rad/s]", fontsize=12)
    ax.set_ylabel("Heave RAO |H₃(ω)| [m/m]", fontsize=12)
    ax.set_title("Wigley III Heave RAO in Regular Waves (H/λ = 1/60)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 5])
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "heave_rao_comparison.png"), dpi=150)
    print(f"  Saved: {fig_dir}/heave_rao_comparison.png")

    # Figure 2: Pitch RAO comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(omega_fd, strip_theory["pitch_rao"], "b-", linewidth=2, label="Strip theory (FD)")
    ax.plot(rao_data["omega"], rao_data["pitch_rao_td"], "ro", markersize=8, label="Time-domain (FFT)")
    ax.set_xlabel("ω [rad/s]", fontsize=12)
    ax.set_ylabel("Pitch RAO |H₅(ω)| [rad/m]", fontsize=12)
    ax.set_title("Wigley III Pitch RAO in Regular Waves (H/λ = 1/60)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 5])
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "pitch_rao_comparison.png"), dpi=150)
    print(f"  Saved: {fig_dir}/pitch_rao_comparison.png")

    # Figure 3: Time series example (λ/L = 1.0)
    # Re-run one case to get time series
    from tensorlbm.seakeeping.lbm_body_coupling import run_section_coupling
    ts_result = run_section_coupling(
        hull=hull,
        x_station=0.0,
        wave_amplitude=steepness * hull.L / 2.0,
        wave_length=hull.L,
        depth=10.0,
        n_periods=n_periods,
        steps_per_period=steps_per_period,
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    t = ts_result["time"]
    T_period = ts_result["T_period"]

    # Wave elevation
    axes[0].plot(t / T_period, ts_result["wave_elev"] * 1000, "b-", linewidth=1)
    axes[0].set_ylabel("η [mm]", fontsize=12)
    axes[0].set_title("Time Series at Midship Section (λ/L = 1.0)", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Heave response
    axes[1].plot(t / T_period, ts_result["z"] * 1000, "r-", linewidth=1)
    axes[1].set_xlabel("t / T [-]", fontsize=12)
    axes[1].set_ylabel("z [mm]", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "timeseries_midship.png"), dpi=150)
    print(f"  Saved: {fig_dir}/timeseries_midship.png")

    # Save JSON report
    report = {
        "campaign": {
            "hull": {
                "L": hull.L,
                "B": hull.B,
                "T": hull.T,
                "Cb": hull.Cb,
                "volume": hull.volume,
                "waterplane_area": hull.waterplane_area,
            },
            "wave_matrix": {
                "wave_length_ratios": wave_length_ratios,
                "steepness": steepness,
                "n_periods": n_periods,
                "n_periods_analysis": n_periods_analysis,
                "steps_per_period": steps_per_period,
            },
        },
        "results": rao_data,
        "strip_theory": {
            "omega": strip_theory["omega"].tolist(),
            "heave_rao": strip_theory["heave_rao"].tolist(),
            "pitch_rao": strip_theory["pitch_rao"].tolist(),
        },
        "figures": [
            "artifacts/heave_rao_comparison.png",
            "artifacts/pitch_rao_comparison.png",
            "artifacts/timeseries_midship.png",
        ],
    }

    report_path = os.path.join(fig_dir, "seakeeping_campaign_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report: {report_path}")

    print("\n" + "=" * 70)
    print("Campaign complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
