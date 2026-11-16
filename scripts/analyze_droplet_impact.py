#!/usr/bin/env python
"""Analyze droplet impact results and generate report figures.

Reads history.csv and run_metadata.json from a droplet impact run,
generates comparison figures, and outputs analysis summary.
"""

import json
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_run(run_dir: Path):
    """Load history.csv and run_metadata.json from a run directory."""
    history_path = run_dir / "history.csv"
    metadata_path = run_dir / "run_metadata.json"

    with open(history_path, "r") as f:
        reader = csv.DictReader(f)
        history = [row for row in reader]

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return history, metadata


def generate_comparison_figure(runs: list[tuple[str, Path]], output_path: Path):
    """Generate comparison figure for multiple runs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    for label, run_dir in runs:
        history, metadata = load_run(run_dir)
        steps = [int(row["step"]) for row in history]
        mass_drift = [float(row["relative_phase_mass_drift"]) for row in history]
        centroid_y = [float(row["liquid_centroid_y"]) for row in history]
        max_speed = [float(row["max_speed"]) for row in history]
        ke = [float(row["kinetic_energy_proxy"]) for row in history]

        axes[0, 0].plot(steps, mass_drift, marker="o", markersize=3, label=label)
        axes[0, 1].plot(steps, centroid_y, marker="o", markersize=3, label=label)
        axes[1, 0].plot(steps, max_speed, marker="o", markersize=3, label=label)
        axes[1, 1].plot(steps, ke, marker="o", markersize=3, label=label)

    axes[0, 0].set(title="Relative phase-mass drift", xlabel="step", ylabel="relative drift")
    axes[0, 0].legend()
    axes[0, 0].set_yscale("log")

    axes[0, 1].set(title="Liquid mass-weighted centroid", xlabel="step", ylabel="y [cells]")
    axes[0, 1].legend()

    axes[1, 0].set(title="Maximum speed", xlabel="step", ylabel="|u| [lu/step]")
    axes[1, 0].legend()

    axes[1, 1].set(title="Kinetic-energy proxy", xlabel="step", ylabel="sum(ρ|u|²/2)")
    axes[1, 1].legend()

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Comparison figure saved to: {output_path}")


def generate_analysis_summary(runs: list[tuple[str, Path]]) -> dict:
    """Generate analysis summary for multiple runs."""
    summary = {}
    for label, run_dir in runs:
        history, metadata = load_run(run_dir)
        final = history[-1]
        summary[label] = {
            "n_steps": int(final["step"]),
            "final_mass_drift": float(final["relative_phase_mass_drift"]),
            "final_centroid_y": float(final["liquid_centroid_y"]),
            "final_max_speed": float(final["max_speed"]),
            "final_ke": float(final["kinetic_energy_proxy"]),
            "phi_min": float(final["phi_min"]),
            "phi_max": float(final["phi_max"]),
            "density_ratio": metadata["derived"]["density_ratio_actual"],
            "elapsed_seconds": metadata["derived"]["elapsed_seconds_including_plotting"],
        }
    return summary


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_droplet_impact.py <run_dir1> [run_dir2] ...")
        sys.exit(1)

    runs = []
    for path in sys.argv[1:]:
        p = Path(path)
        runs.append((p.name, p))

    output_dir = Path("docs/m23013")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate comparison figure
    generate_comparison_figure(runs, output_dir / "comparison_curves.png")

    # Generate analysis summary
    summary = generate_analysis_summary(runs)
    with open(output_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nAnalysis Summary:")
    for label, data in summary.items():
        print(f"\n{label}:")
        for key, val in data.items():
            print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
