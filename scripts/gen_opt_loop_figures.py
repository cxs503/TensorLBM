#!/usr/bin/env python
"""Generate figures for the GAP-PP-4-opt-loop report.

Produces:
1. Convergence curves (loss vs iteration) for each track
2. Tau/Re recovery bar chart with error bars across seeds
3. Field comparison (analytic vs recovered) at final time
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARTIFACT = Path("artifacts/opt_loop_tau_recovery.json")
OUTDIR = Path("figures")


def load_data() -> dict:
    with open(ARTIFACT) as f:
        return json.load(f)


def fig_convergence_curves(data: dict) -> Path:
    """Loss vs iteration for each track (first seed)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    track_order = ["analytic16", "analytic32", "twin16",
                   "analytic16_fixed", "twin16_noise0.2", "twin16_noise1.0"]
    track_labels = {
        "analytic16": "Analytic n=16 (free A)",
        "analytic32": "Analytic n=32 (free A)",
        "twin16": "Twin n=16 (free A)",
        "analytic16_fixed": "Analytic n=16 (fixed A) [control]",
        "twin16_noise0.2": "Twin n=16 noise=0.2%",
        "twin16_noise1.0": "Twin n=16 noise=1.0%",
    }

    for idx, track_name in enumerate(track_order):
        track = next(t for t in data["tracks"] if t["track"] == track_name)
        ax = axes[idx]
        # Use first seed's history
        run = track["runs"][0]
        history = run.get("history", [])
        if not history:
            ax.text(0.5, 0.5, "no history", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(track_labels.get(track_name, track_name))
            continue

        iters = [h["iter"] for h in history]
        losses = [h["loss"] for h in history]
        phases = [h.get("phase", "") for h in history]

        # Color by phase
        adam_idx = [i for i, p in enumerate(phases) if p == "adam"]
        lbfgs_idx = [i for i, p in enumerate(phases) if "lbfgs" in p]

        if adam_idx:
            ax.plot([iters[i] for i in adam_idx], [losses[i] for i in adam_idx],
                    "b-o", markersize=3, label="Adam")
        if lbfgs_idx:
            ax.plot([iters[i] for i in lbfgs_idx], [losses[i] for i in lbfgs_idx],
                    "r-s", markersize=3, label="L-BFGS")

        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title(track_labels.get(track_name, track_name), fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Convergence Curves: Loss vs Iteration (seed=0)", fontsize=12, y=1.01)
    fig.tight_layout()
    out = OUTDIR / "opt_loop_convergence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_recovery_barchart(data: dict) -> Path:
    """Recovered tau and Re with error bars across 3 seeds."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    track_order = ["analytic16", "analytic32", "twin16",
                   "analytic16_fixed", "twin16_noise0.2", "twin16_noise1.0"]
    track_labels = {
        "analytic16": "Analytic\nn=16",
        "analytic32": "Analytic\nn=32",
        "twin16": "Twin\nn=16",
        "analytic16_fixed": "Fixed A\nn=16 [ctrl]",
        "twin16_noise0.2": "Twin+noise\n0.2%",
        "twin16_noise1.0": "Twin+noise\n1.0%",
    }

    tau_true = 0.55
    x = np.arange(len(track_order))
    width = 0.6

    tau_hats = []
    tau_stds = []
    re_hats = []
    re_stds = []

    for track_name in track_order:
        track = next(t for t in data["tracks"] if t["track"] == track_name)
        taus = [r["tau_hat"] for r in track["runs"]]
        res = [r["re_hat"] for r in track["runs"]]
        tau_hats.append(np.mean(taus))
        tau_stds.append(np.std(taus))
        re_hats.append(np.mean(res))
        re_stds.append(np.std(res))

    colors = ["#2196F3", "#4CAF50", "#9C27B0", "#FF5722", "#FFC107", "#00BCD4"]

    bars1 = ax1.bar(x, tau_hats, width, yerr=tau_stds, capsize=4,
                    color=colors, edgecolor="white", linewidth=0.5)
    ax1.axhline(tau_true, color="red", linestyle="--", linewidth=1.5, label=f"True τ={tau_true}")
    ax1.set_xticks(x)
    ax1.set_xticklabels([track_labels[t] for t in track_order], fontsize=8)
    ax1.set_ylabel("Recovered τ")
    ax1.set_title("τ Recovery (3 seeds)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    # Add error percentage labels
    for i, (h, s) in enumerate(zip(tau_hats, tau_stds)):
        err_pct = 100 * (h - tau_true) / tau_true
        ax1.text(i, h + s + 0.001, f"{err_pct:+.2f}%", ha="center", fontsize=7,
                 color="red" if abs(err_pct) > 3 else "green")

    # Re recovery
    re_true = 0.05 * 16 / ((0.55 - 0.5) / 3)  # u0 * L / nu
    bars2 = ax2.bar(x, re_hats, width, yerr=re_stds, capsize=4,
                    color=colors, edgecolor="white", linewidth=0.5)
    ax2.axhline(re_true, color="red", linestyle="--", linewidth=1.5, label=f"True Re={re_true:.1f}")
    ax2.set_xticks(x)
    ax2.set_xticklabels([track_labels[t] for t in track_order], fontsize=8)
    ax2.set_ylabel("Recovered Re")
    ax2.set_title("Re Recovery (3 seeds)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Taylor-Green τ/Re Recovery Across 3 Seeds", fontsize=12, y=1.01)
    fig.tight_layout()
    out = OUTDIR / "opt_loop_recovery.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_error_comparison(data: dict) -> Path:
    """Relative error comparison across tracks and seeds."""
    fig, ax = plt.subplots(figsize=(10, 5))

    track_order = ["analytic16", "analytic32", "twin16",
                   "analytic16_fixed", "twin16_noise0.2", "twin16_noise1.0"]
    track_labels = {
        "analytic16": "Analytic n=16",
        "analytic32": "Analytic n=32",
        "twin16": "Twin n=16",
        "analytic16_fixed": "Fixed A [control]",
        "twin16_noise0.2": "Twin+0.2%",
        "twin16_noise1.0": "Twin+1.0%",
    }

    x = np.arange(len(track_order))
    width = 0.15

    for seed_idx in range(3):
        errs = []
        for track_name in track_order:
            track = next(t for t in data["tracks"] if t["track"] == track_name)
            errs.append(track["runs"][seed_idx]["rel_err_pct"])

        offset = (seed_idx - 1) * width
        ax.bar(x + offset, errs, width, label=f"Seed {seed_idx}",
               alpha=0.8, edgecolor="white")

    ax.axhline(3, color="red", linestyle="--", linewidth=1, label="+3% threshold")
    ax.axhline(-3, color="red", linestyle="--", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([track_labels[t] for t in track_order], fontsize=9)
    ax.set_ylabel("Relative Error (%)")
    ax.set_title("τ Recovery Error: All Tracks × 3 Seeds")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = OUTDIR / "opt_loop_errors.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    data = load_data()

    p1 = fig_convergence_curves(data)
    print(f"Wrote {p1}")

    p2 = fig_recovery_barchart(data)
    print(f"Wrote {p2}")

    p3 = fig_error_comparison(data)
    print(f"Wrote {p3}")


if __name__ == "__main__":
    main()
