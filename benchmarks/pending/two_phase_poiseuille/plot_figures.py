#!/usr/bin/env python
"""Plot figures for two_phase_poiseuille terminal verdict.

Generates:
  figures/curve_convergence.png — u_max convergence history (H64, H128)
  figures/curve_error_profile.png — velocity profile comparison with analytical
  figures/field_velocity_profile.png — side-by-side sim vs analytical profiles

All data from out_part1.json and result.json (zero hand-copied).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASE_DIR = Path(__file__).parent
DATA = CASE_DIR / "out_part1.json"
RESULTS = CASE_DIR / "result.json"
FIG_DIR = CASE_DIR / "figures"

def load_data():
    with open(DATA) as f:
        return json.load(f)

def plot_convergence_curve(d, save_path):
    """Plot u_max convergence history for H64 and H128."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, color in [("H64", "blue"), ("H128", "orange")]:
        conv = d[key]["convergence"]
        steps = [c["step"] for c in conv]
        umax = [c["u_max"] for c in conv]
        ax.plot(steps, umax, "o-", color=color, label=f"{key} (ny={d[key]['ny']})")
    ax.set_xlabel("Step")
    ax.set_ylabel("u_max (lattice units)")
    ax.set_title("Two-Phase Poiseuille: u_max Convergence History")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")

def plot_error_profile(d, save_path):
    """Plot relative error vs position across channel for both grids."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, color, marker in [("H64", "blue", "o"), ("H128", "orange", "s")]:
        y_sim = d[key]["u_profile_sim"]
        y_ana = d[key]["u_profile_analytical"]
        rel_err = np.array(y_sim) - np.array(y_ana)
        # Normalized by max analytical for meaningful comparison
        max_ana = max(abs(float(v)) for v in y_ana)
        if max_ana > 0:
            rel_err_pct = [float(e) / max_ana * 100 for e in rel_err]
        else:
            rel_err_pct = [0.0] * len(rel_err)
        ny = d[key]["ny"]
        y_pos = np.linspace(0, ny - 1, ny)
        ax.plot(y_pos, rel_err_pct, marker=marker, markersize=3,
                color=color, label=f"{key} rel_err_pct (max={d[key]['center_err_mask_simmax']['max_rel_err_pct']:.1f}%)",
                alpha=0.7)
    ax.axhline(y=3.0, color="red", linestyle="--", alpha=0.5, label="3% threshold")
    ax.axhline(y=-3.0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("y position (lattice units)")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Two-Phase Poiseuille: Velocity Error Profile vs Channel Position")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")

def plot_velocity_profile(d, save_path):
    """Plot sim vs analytical velocity profiles side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, (key, color) in enumerate([("H64", "blue"), ("H128", "orange")]):
        ax = axes[idx]
        ny = d[key]["ny"]
        y_pos = np.linspace(0, ny - 1, ny)
        u_sim = [float(v) for v in d[key]["u_profile_sim"]]
        u_ana = [float(v) for v in d[key]["u_profile_analytical"]]
        ax.plot(u_sim, y_pos, color=color, label="Simulation", linewidth=2)
        ax.plot(u_ana, y_pos, "k--", label="Analytical", linewidth=1.5)
        ax.set_xlabel("u (lattice units)")
        ax.set_ylabel("y position")
        ax.set_title(f"{key} (ny={ny}): Sim vs Analytical\nu_max ratio = {d[key]['u_max_ratio']:.4f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Two-Phase Poiseuille: Velocity Profiles", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")

def plot_convergence_error(r, save_path):
    """Plot max_rel_err comparison between H64 and H128 showing anti-convergence."""
    fig, ax = plt.subplots(figsize=(6, 4))
    grids = ["H64", "H128"]
    errs = [r["part1"]["H64"]["max_rel_err"] * 100,
            r["part1"]["H128"]["max_rel_err"] * 100]
    bars = ax.bar(grids, errs, color=["blue", "orange"], edgecolor="black")
    ax.axhline(y=3.0, color="red", linestyle="--", label="3% threshold", linewidth=1.5)
    for bar, val in zip(bars, errs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.2f}%", ha="center", fontsize=11)
    ax.set_ylabel("Max relative error (%)")
    ax.set_title("Two-Phase Poiseuille SC-MCMP: Error Anti-Convergence\n(26.82% → 33.17%)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    d = load_data()
    with open(RESULTS) as f:
        r = json.load(f)

    # Figure 1: u_max convergence history
    plot_convergence_curve(d, FIG_DIR / "curve_convergence.png")

    # Figure 2: Error profile across channel
    plot_error_profile(d, FIG_DIR / "curve_error_profile.png")

    # Figure 3: Sim vs analytical velocity profiles
    plot_velocity_profile(d, FIG_DIR / "field_velocity_profile.png")

    # Figure 4: Anti-convergence bar chart
    plot_convergence_error(r, FIG_DIR / "curve_anti_convergence.png")

if __name__ == "__main__":
    main()
