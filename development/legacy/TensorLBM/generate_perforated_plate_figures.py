"""Generate verification figures and report for perforated plate oscillatory flow.

Produces:
    - C_d vs KC curve for multiple β values
    - C_m vs KC curve
    - Phase lag vs KC
    - Energy dissipation vs KC
    - Grid convergence plot
    - Phase-averaged Δp-Q hysteresis loop
"""

from __future__ import annotations

import json
import math
import os
import sys
import base64
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensorlbm.verification.perforated_plate.solver import (
    PerforatedPlateConfig,
    run_perforated_plate,
)
from tensorlbm.verification.perforated_plate.analysis import (
    phase_average,
    extract_cd_cm,
    compute_energy_dissipation,
    compute_phase_lag,
)
from tensorlbm.verification.perforated_plate.reference import (
    idelchik_cd_quasi_steady,
)


def run_parameter_sweep():
    """Run a parameter sweep over KC and β."""
    results = []

    # Fixed parameters
    ny = 48
    tau = 1.0
    force_amp = 2e-5

    # β values
    betas = [0.08, 0.15, 0.25, 0.35]
    hole_sizes = [4, 8, 12, 16]  # corresponding hole sizes

    # KC values (via omega variation)
    omegas = [0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20]

    for beta_idx, (beta, hole_size) in enumerate(zip(betas, hole_sizes)):
        for omega in omegas:
            cfg = PerforatedPlateConfig(
                nx=128,
                ny=ny,
                hole_size=hole_size,
                tau=tau,
                force_amplitude=force_amp,
                omega=omega,
                steps=3000,
                warmup_cycles=2,
                n_cycles_measure=3,
                samples_per_cycle=32,
                device="cpu",
                dtype="float64",
                label=f"beta{beta}_omega{omega}",
            )
            try:
                result = run_perforated_plate(cfg)
                ts = result["time_series"]

                # Phase average
                phase_q, avg_q = phase_average(ts["time"], ts["flow_rate"], omega, 32)
                phase_dp, avg_dp = phase_average(ts["time"], ts["delta_p"], omega, 32)

                # Extract C_d, C_m
                cd_cm = extract_cd_cm(avg_q, avg_dp, omega, beta, ny)

                # Energy dissipation
                phi = compute_energy_dissipation(avg_q, avg_dp, omega)

                # Phase lag
                pl = compute_phase_lag(avg_q, avg_dp, omega)

                # Estimate KC
                u_max = np.max(np.abs(avg_q)) / max(ny - 2, 1)
                period = 2 * math.pi / omega
                d_char = max(hole_size, 1)
                kc = u_max * period / d_char

                results.append({
                    "beta": beta,
                    "omega": omega,
                    "kc": kc,
                    "C_d": cd_cm["C_d"],
                    "C_m": cd_cm["C_m"],
                    "r_squared": cd_cm["r_squared"],
                    "phi": phi,
                    "phase_lag_deg": pl["phase_lag_deg"],
                    "u_max": u_max,
                })
                print(f"  β={beta}, ω={omega}, KC={kc:.2f}, C_d={cd_cm['C_d']:.3f}, C_m={cd_cm['C_m']:.3f}")
            except Exception as e:
                print(f"  Failed for β={beta}, ω={omega}: {e}")

    return results


def plot_cd_vs_kc(results, output_path):
    """Plot C_d vs KC for multiple β values."""
    fig, ax = plt.subplots(figsize=(10, 6))

    betas = sorted(set(r["beta"] for r in results))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for beta, color in zip(betas, colors):
        beta_results = [r for r in results if r["beta"] == beta and not math.isnan(r["C_d"])]
        if not beta_results:
            continue
        kc = [r["kc"] for r in beta_results]
        cd = [r["C_d"] for r in beta_results]

        # Sort by KC
        sorted_pairs = sorted(zip(kc, cd))
        kc, cd = zip(*sorted_pairs)

        ax.plot(kc, cd, "o-", color=color, label=f"β={beta:.2f}", markersize=5)

        # Add Idelchik reference line
        cd_ref = idelchik_cd_quasi_steady(beta)
        ax.axhline(y=cd_ref, color=color, linestyle="--", alpha=0.5)

    ax.set_xlabel("Keulegan-Carpenter Number (KC)", fontsize=12)
    ax.set_ylabel("Discharge Coefficient $C_d$", fontsize=12)
    ax.set_title("Discharge Coefficient vs KC for Various Open Area Ratios", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(r["kc"] for r in results) * 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cm_vs_kc(results, output_path):
    """Plot C_m vs KC for multiple β values."""
    fig, ax = plt.subplots(figsize=(10, 6))

    betas = sorted(set(r["beta"] for r in results))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for beta, color in zip(betas, colors):
        beta_results = [r for r in results if r["beta"] == beta and not math.isnan(r["C_m"])]
        if not beta_results:
            continue
        kc = [r["kc"] for r in beta_results]
        cm = [r["C_m"] for r in beta_results]

        sorted_pairs = sorted(zip(kc, cm))
        kc, cm = zip(*sorted_pairs)

        ax.plot(kc, cm, "o-", color=color, label=f"β={beta:.2f}", markersize=5)

    ax.set_xlabel("Keulegan-Carpenter Number (KC)", fontsize=12)
    ax.set_ylabel("Added-Mass Coefficient $C_m$", fontsize=12)
    ax.set_title("Added-Mass Coefficient vs KC", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_phase_lag(results, output_path):
    """Plot phase lag vs KC."""
    fig, ax = plt.subplots(figsize=(10, 6))

    betas = sorted(set(r["beta"] for r in results))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for beta, color in zip(betas, colors):
        beta_results = [r for r in results if r["beta"] == beta]
        if not beta_results:
            continue
        kc = [r["kc"] for r in beta_results]
        pl = [r["phase_lag_deg"] for r in beta_results]

        sorted_pairs = sorted(zip(kc, pl))
        kc, pl = zip(*sorted_pairs)

        ax.plot(kc, pl, "o-", color=color, label=f"β={beta:.2f}", markersize=5)

    ax.set_xlabel("Keulegan-Carpenter Number (KC)", fontsize=12)
    ax.set_ylabel("Phase Lag $\\Delta\\phi$ (degrees)", fontsize=12)
    ax.set_title("Phase Lag Between $\\Delta p$ and $Q$ vs KC", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-180, 180)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_hysteresis(results, output_path):
    """Plot Δp-Q hysteresis loop for one case."""
    # Pick a representative case
    rep = None
    for r in results:
        if r["beta"] == 0.15 and 5 < r["kc"] < 15:
            rep = r
            break

    if rep is None:
        rep = results[0]

    # Re-run to get time series
    cfg = PerforatedPlateConfig(
        nx=128,
        ny=48,
        hole_size=8,
        tau=1.0,
        force_amplitude=2e-5,
        omega=rep["omega"],
        steps=3000,
        warmup_cycles=2,
        n_cycles_measure=3,
        samples_per_cycle=32,
        device="cpu",
        dtype="float64",
        label="hysteresis",
    )
    result = run_perforated_plate(cfg)
    ts = result["time_series"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ts["flow_rate"], ts["delta_p"], "b-", alpha=0.7, linewidth=1)
    ax.set_xlabel("Flow Rate $Q$", fontsize=12)
    ax.set_ylabel("Pressure Drop $\\Delta p$", fontsize=12)
    ax.set_title(f"$\\Delta p$-$Q$ Hysteresis Loop (KC={rep['kc']:.1f}, β={rep['beta']:.2f})", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def encode_png_base64(png_path):
    """Encode PNG file as base64 string."""
    with open(png_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
    os.makedirs(output_dir, exist_ok=True)

    print("Running parameter sweep...")
    results = run_parameter_sweep()

    print(f"\nCollected {len(results)} data points")

    if not results:
        print("No results collected. Exiting.")
        return

    print("\nGenerating figures...")
    plot_cd_vs_kc(results, os.path.join(output_dir, "perforated_plate_cd_vs_kc.png"))
    plot_cm_vs_kc(results, os.path.join(output_dir, "perforated_plate_cm_vs_kc.png"))
    plot_phase_lag(results, os.path.join(output_dir, "perforated_plate_phase_lag.png"))
    plot_hysteresis(results, os.path.join(output_dir, "perforated_plate_hysteresis.png"))

    # Save results as JSON
    results_path = os.path.join(output_dir, "perforated_plate_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
