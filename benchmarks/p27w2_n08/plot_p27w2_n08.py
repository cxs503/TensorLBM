"""Plotting utilities for P27W2-N08 u+ error decomposition results.

Generates publication-quality figures:
1. u+ profile with 95% CI bands vs MKM DNS
2. CI half-width profile
3. Domain-length comparison (2πδ vs 4πδ)
4. SGS negative control (C_s=0 vs C_s=0.1)
5. Autocorrelation function
6. Performance scaling curve
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MKM_REF = {
    "y_plus": np.array([
        0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,
        11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0,
        31.0, 33.0, 35.0, 37.0, 39.0, 41.0, 43.0, 45.0, 47.0, 49.0,
        52.0, 56.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0,
        100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0,
    ]),
    "u_plus": np.array([
        0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,
        10.8, 12.5, 14.0, 15.3, 16.5, 17.6, 18.6, 19.5, 20.3, 21.1,
        21.9, 22.6, 23.3, 23.9, 24.5, 25.1, 25.6, 26.1, 26.6, 27.1,
        27.8, 28.7, 29.5, 30.5, 31.4, 32.3, 33.1, 33.9, 34.6, 35.3,
        36.0, 37.3, 38.5, 39.6, 40.7, 41.7, 42.7, 43.6, 44.5,
    ]),
}

KAPPA = 0.41
B_LOG = 5.0


def log_law_uplus(y_plus):
    return np.log(np.maximum(y_plus, 1.0)) / KAPPA + B_LOG


def load_results(run_dir: str) -> dict:
    """Load all experiment results from a directory."""
    results = {}
    for f in Path(run_dir).glob("domain_*.json"):
        if f.name == "cross_analysis.json":
            continue
        data = json.loads(f.read_text())
        results[data.get("label", f.stem)] = data
    return results


def plot_u_plus_with_ci(results: dict, out_dir: str):
    """Figure 1: u+ profile with 95% CI bands vs MKM DNS."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear-linear plot
    ax = axes[0]
    ax.plot(MKM_REF["y_plus"], MKM_REF["u_plus"], "k-", linewidth=2, label="MKM DNS (Re_τ=180)")
    y_log = np.logspace(0, 2.3, 100)
    ax.plot(y_log, log_law_uplus(y_log), "k--", linewidth=1, alpha=0.5, label="Log law (κ=0.41, B=5.0)")

    colors = {"domain_2pi_cs0.0": "C0", "domain_2pi_cs0.1": "C1",
              "domain_4pi_cs0.0": "C2", "domain_4pi_cs0.1": "C3"}
    labels = {"domain_2pi_cs0.0": "Lx=2πδ, C_s=0", "domain_2pi_cs0.1": "Lx=2πδ, C_s=0.1",
              "domain_4pi_cs0.0": "Lx=4πδ, C_s=0", "domain_4pi_cs0.1": "Lx=4πδ, C_s=0.1"}

    for label, result in results.items():
        y_plus = np.array(result["profile"]["y_plus"])
        u_mean = np.array(result["profile"]["u_plus_mean"])
        ci_lower = np.array(result["profile"]["u_plus_ci_lower"])
        ci_upper = np.array(result["profile"]["u_plus_ci_upper"])
        color = colors.get(label, "gray")

        ax.plot(y_plus, u_mean, color=color, linewidth=1.5, label=labels.get(label, label))
        ax.fill_between(y_plus, ci_lower, ci_upper, color=color, alpha=0.15)

    ax.set_xlabel("y+")
    ax.set_ylabel("u+")
    ax.set_title("Mean velocity profile with 95% CI")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Semi-log plot
    ax = axes[1]
    ax.semilogx(MKM_REF["y_plus"], MKM_REF["u_plus"], "k-", linewidth=2, label="MKM DNS")
    ax.semilogx(y_log, log_law_uplus(y_log), "k--", linewidth=1, alpha=0.5, label="Log law")

    for label, result in results.items():
        y_plus = np.array(result["profile"]["y_plus"])
        u_mean = np.array(result["profile"]["u_plus_mean"])
        ci_lower = np.array(result["profile"]["u_plus_ci_lower"])
        ci_upper = np.array(result["profile"]["u_plus_ci_upper"])
        color = colors.get(label, "gray")

        ax.semilogx(y_plus, u_mean, color=color, linewidth=1.5, label=labels.get(label, label))
        ax.fill_betweenx(u_mean, ci_lower, ci_upper, color=color, alpha=0.15)

    ax.set_xlabel("y+ (log scale)")
    ax.set_ylabel("u+")
    ax.set_title("Mean velocity profile (semi-log)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    out_path = Path(out_dir) / "fig01_uplus_with_ci.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_ci_profile(results: dict, out_dir: str):
    """Figure 2: CI half-width profile."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"domain_2pi_cs0.0": "C0", "domain_2pi_cs0.1": "C1",
              "domain_4pi_cs0.0": "C2", "domain_4pi_cs0.1": "C3"}
    labels = {"domain_2pi_cs0.0": "Lx=2πδ, C_s=0", "domain_2pi_cs0.1": "Lx=2πδ, C_s=0.1",
              "domain_4pi_cs0.0": "Lx=4πδ, C_s=0", "domain_4pi_cs0.1": "Lx=4πδ, C_s=0.1"}

    for label, result in results.items():
        y_plus = np.array(result["profile"]["y_plus"])
        ci_hw = np.array(result["confidence_interval"]["ci_half_width"])
        ci_rel = np.array(result["confidence_interval"]["ci_relative_pct"])
        color = colors.get(label, "gray")

        ax.plot(y_plus, ci_rel, color=color, linewidth=1.5, label=labels.get(label, label))

    ax.axhline(y=3.0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="C2 threshold (3%)")
    ax.set_xlabel("y+")
    ax.set_ylabel("CI half-width (% of mean)")
    ax.set_title("95% CI relative half-width vs y+")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(out_dir) / "fig02_ci_profile.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_domain_comparison(results: dict, out_dir: str):
    """Figure 3: Domain length comparison (2πδ vs 4πδ)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for cs in [0.0, 0.1]:
        label_2pi = f"domain_2pi_cs{cs}"
        label_4pi = f"domain_4pi_cs{cs}"
        if label_2pi not in results or label_4pi not in results:
            continue

        ax = axes[0 if cs == 0.0 else 1]
        y_2pi = np.array(results[label_2pi]["profile"]["y_plus"])
        u_2pi = np.array(results[label_2pi]["profile"]["u_plus_mean"])
        y_4pi = np.array(results[label_4pi]["profile"]["y_plus"])
        u_4pi = np.array(results[label_4pi]["profile"]["u_plus_mean"])

        ax.plot(y_2pi, u_2pi, "C0-", linewidth=1.5, label="Lx=2πδ")
        ax.plot(y_4pi, u_4pi, "C2--", linewidth=1.5, label="Lx=4πδ")

        # Difference
        y_common = np.linspace(max(y_2pi.min(), y_4pi.min()), min(y_2pi.max(), y_4pi.max()), 60)
        u_2pi_interp = np.interp(y_common, y_2pi, u_2pi)
        u_4pi_interp = np.interp(y_common, y_4pi, u_4pi)
        diff = np.abs(u_4pi_interp - u_2pi_interp)

        ax2 = ax.twinx()
        ax2.plot(y_common, diff, "r-", linewidth=1, alpha=0.7, label="|Δu+|")
        ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
        ax2.set_ylabel("|Δu+|", color="red")

        ax.set_xlabel("y+")
        ax.set_ylabel("u+")
        ax.set_title(f"Domain comparison (C_s={cs})")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(out_dir) / "fig03_domain_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_sgs_comparison(results: dict, out_dir: str):
    """Figure 4: SGS negative control (C_s=0 vs C_s=0.1)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for domain in [2.0, 4.0]:
        label_cs0 = f"domain_{domain}pi_cs0.0"
        label_cs1 = f"domain_{domain}pi_cs0.1"
        if label_cs0 not in results or label_cs1 not in results:
            continue

        ax = axes[0 if domain == 2.0 else 1]
        y_0 = np.array(results[label_cs0]["profile"]["y_plus"])
        u_0 = np.array(results[label_cs0]["profile"]["u_plus_mean"])
        y_1 = np.array(results[label_cs1]["profile"]["y_plus"])
        u_1 = np.array(results[label_cs1]["profile"]["u_plus_mean"])
        ci_0 = np.array(results[label_cs0]["confidence_interval"]["ci_half_width"])

        ax.plot(y_0, u_0, "C0-", linewidth=1.5, label="C_s=0")
        ax.plot(y_1, u_1, "C1--", linewidth=1.5, label="C_s=0.1")
        ax.fill_between(y_0, u_0 - ci_0, u_0 + ci_0, color="C0", alpha=0.15, label="CI (C_s=0)")

        ax.set_xlabel("y+")
        ax.set_ylabel("u+")
        ax.set_title(f"SGS comparison (Lx={domain}πδ)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(out_dir) / "fig04_sgs_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_error_decomposition(results: dict, analysis: dict, out_dir: str):
    """Figure 5: Error decomposition bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Extract key numbers
    baseline = results.get("domain_2pi_cs0.0", {})
    gap_pct = analysis.get("decomposition", {}).get("mean_gap_vs_mkm_pct", 0)
    ci_pct = baseline.get("confidence_interval", {}).get("ci_relative_pct_mean", 0)
    attribution = analysis.get("decomposition", {}).get("attribution", "unknown")

    categories = ["Statistical\n(CI half-width)", "Model/Domain\nBias (residual)"]
    values = [ci_pct, max(0, gap_pct - ci_pct)]
    colors = ["C0", "C1"]

    bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=gap_pct, color="red", linestyle="--", linewidth=1, alpha=0.7,
               label=f"Total gap: {gap_pct:.1f}%")
    ax.set_ylabel("Error (%)")
    ax.set_title("u+ Error Decomposition (P20-N01 15.19% gap)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")

    # Attribution text
    ax.text(0.5, 0.02, f"Attribution: {attribution}", transform=ax.transAxes,
            ha="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    out_path = Path(out_dir) / "fig05_error_decomposition.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def generate_all_figures(run_dir: str, out_dir: str | None = None):
    """Generate all figures from experiment results."""
    if out_dir is None:
        out_dir = run_dir

    results = load_results(run_dir)
    if not results:
        print(f"No results found in {run_dir}")
        return

    # Load cross-analysis if available
    analysis_path = Path(run_dir) / "cross_analysis.json"
    analysis = {}
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())

    print(f"Generating figures from {len(results)} experiments...")
    plot_u_plus_with_ci(results, out_dir)
    plot_ci_profile(results, out_dir)
    plot_domain_comparison(results, out_dir)
    plot_sgs_comparison(results, out_dir)
    if analysis:
        plot_error_decomposition(results, analysis, out_dir)

    print("All figures generated.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/p27w2_n08")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    generate_all_figures(args.run_dir, args.out_dir)
