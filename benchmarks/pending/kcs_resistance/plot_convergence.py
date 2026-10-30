#!/usr/bin/env python3
"""Generate convergence plot for KCS resistance benchmark."""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(result_path: Path, out_dir: Path):
    with open(result_path) as f:
        r = json.load(f)

    cf_ref = r["Cf_ref_Blasius"]
    ss = r["steady_state"]

    # Read force log from the run directory
    # The force log is stored in the engine, not in result.json
    # We need to reconstruct from the convergence data

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("KCS Resistance Benchmark — Convergence Analysis (m460)\n"
                 f"Re=1000, L={r['L_cells']} cells/Lpp, double body, MRT, SDAA eager",
                 fontsize=13, fontweight="bold")

    # --- Panel 1: Cd_total convergence sequence ---
    ax = axes[0, 0]
    conv = r["convergence_windows"]
    fracs = [0.25, 0.5, 0.75, 1.0]
    labels = [f"{int(f*100)}%" for f in fracs]
    vals = [conv.get(l, float("nan")) for l in labels]
    ax.plot(labels, vals, "bo-", markersize=8, linewidth=2, label="Cd_total (time-avg window)")
    ax.axhline(cf_ref, color="green", linestyle="--", linewidth=1.5, label=f"Blasius Cf = {cf_ref:.4f}")
    ax.set_xlabel("Log fraction")
    ax.set_ylabel("Cd_total")
    ax.set_title("Cd_total Convergence Sequence")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Three-window drift ---
    ax = axes[0, 1]
    wins = ss["window_Cd_totals"]
    win_labels = [f"W{i+1}" for i in range(len(wins))]
    ax.bar(win_labels, wins, color=["#2196F3", "#4CAF50", "#FF9800"][:len(wins)], edgecolor="black")
    ax.axhline(cf_ref, color="green", linestyle="--", linewidth=1.5, label=f"Blasius Cf = {cf_ref:.4f}")
    ax.set_ylabel("Cd_total")
    ax.set_title(f"Three-Window Convergence\nMax drift = {ss['max_drift_pct']:.4f}% "
                 f"({'CONVERGED' if ss['converged'] else 'NOT CONVERGED'})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(wins):
        ax.text(i, v + 0.0005, f"{v:.6f}", ha="center", fontsize=9)

    # --- Panel 3: Component breakdown ---
    ax = axes[1, 0]
    components = ["Cd_friction\n(Cf)", "Cd_pressure\n(Cp)", "Cd_total\n(Cf+Cp)"]
    values = [ss["Cd_friction"], ss["Cd_pressure"], ss["Cd_total"]]
    colors = ["#2196F3", "#FF5722", "#9C27B0"]
    bars = ax.bar(components, values, color=colors, edgecolor="black")
    ax.axhline(cf_ref, color="green", linestyle="--", linewidth=1.5, label=f"Blasius Cf = {cf_ref:.4f}")
    ax.set_ylabel("Cd")
    ax.set_title("Drag Component Breakdown (steady-state avg)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.001, f"{v:.6f}", ha="center", fontsize=9)

    # --- Panel 4: Error summary ---
    ax = axes[1, 1]
    err_cf = ss["err_pct_Cf_vs_Blasius"]
    err_cp = ss["err_pct_Cp_vs_Blasius"]
    err_tot = ss["err_pct_Cd_total_vs_Blasius"]
    labels_err = ["Cf vs Blasius", "Cp/Cf_ref", "Cd_total vs Blasius"]
    vals_err = [err_cf, err_cp, err_tot]
    colors_err = ["#4CAF50" if abs(e) < 3 else "#F44336" for e in vals_err]
    bars = ax.barh(labels_err, vals_err, color=colors_err, edgecolor="black")
    ax.axvline(3, color="orange", linestyle="--", linewidth=1, label="±3% threshold")
    ax.axvline(-3, color="orange", linestyle="--", linewidth=1)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Error (%)")
    ax.set_title("Error vs Blasius Cf_ref")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    for bar, v in zip(bars, vals_err):
        x_pos = v + 1 if v >= 0 else v - 1
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f"{v:+.2f}%",
                va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_path = out_dir / "kcs_convergence_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {out_path}")
    return out_path


if __name__ == "__main__":
    result_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("result.json")
    out_dir = result_path.parent
    plot_convergence(result_path, out_dir)
