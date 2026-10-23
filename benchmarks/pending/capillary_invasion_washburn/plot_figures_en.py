"""Plot figures for capillary_invasion_washburn terminal verdict report.

Generates:
  fig_beta_convergence.png   — curve: Washburn beta vs 0.5 for W32/W64
  fig_mass_drift.png         — curve: mass conservation drift history
  fig_front_Lt.png           — curve: invasion front L(t) + Washburn fit
  fig_phase_field.png        — field: phase field phi(y,x) snapshot (W32, W64)
  fig_convergence_comparison.png — curve: b_max_rel_diff across grid levels

All data read from result.json + out_part2_W*.json + out_part2_W*_final.npz.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASE_DIR = Path(__file__).resolve().parent
FIG_DIR = CASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
})


def load_series(W):
    """Load time series from out_part2_W{W}.json."""
    p = CASE_DIR / f"out_part2_W{W}.json"
    d = json.load(open(p))
    s = d["series"]
    steps = np.array([x["step"] for x in s])
    L = np.array([x["L"] for x in s])
    massw = np.array([x["massw"] for x in s])
    massg = np.array([x["massg"] for x in s])
    return steps, L, massw, massg


def plot_beta_convergence():
    """Figure 1 (curve): Washburn exponent beta vs theoretical 0.5."""
    r = json.load(open(CASE_DIR / "result.json"))
    W32 = r["part2"]["W32"]
    W64 = r["part2"]["W64"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["W32", "W64"],
                  [W32["beta"], W64["beta"]],
                  color=["#d62728", "#ff7f0e"],
                  alpha=0.8, width=0.5)
    ax.axhline(0.5, color="k", ls="--", lw=1.5, label="Theory beta=0.5")
    ax.set_ylabel("Washburn exponent beta")
    ax.set_title("Capillary Invasion: Washburn exponent beta (theory 0.5)")
    ax.set_ylim(0.3, 0.7)
    ax.legend()

    for bar, w in zip(bars, ["W32", "W64"]):
        val = bar.get_height()
        dev = W32["beta_criterion_dev_pct"] if w == "W32" else W64["beta_criterion_dev_pct"]
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f"beta={val:.3f}\n(dev {dev:.1f}%)",
                ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Tube width W (lattice units)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_beta_convergence.png")
    plt.close(fig)
    print("  fig_beta_convergence.png OK")


def plot_mass_drift():
    """Figure 2 (curve): mass conservation drift history for W32 and W64."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for W, color, label in [(32, "#d62728", "W32"), (64, "#ff7f0e", "W64")]:
        steps, L, massw, massg = load_series(W)
        massw0 = massw[0]
        massg0 = massg[0]
        drift_w = 100.0 * (massw - massw0) / massw0
        drift_g = 100.0 * (massg - massg0) / massg0

        axes[0].plot(steps, drift_w, color=color, lw=1.2, label=label)
        axes[1].plot(steps, drift_g, color=color, lw=1.2, label=label)

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Water mass drift (%)")
    axes[0].set_title("Water phase mass conservation")
    axes[0].axhline(0, color="k", ls="--", lw=0.8)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Gas mass drift (%)")
    axes[1].set_title("Gas phase mass conservation")
    axes[1].axhline(0, color="k", ls="--", lw=0.8)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("Mass conservation drift (SC-MCMP structural failure: -60.7% / -67.0%)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_mass_drift.png")
    plt.close(fig)
    print("  fig_mass_drift.png OK")


def plot_front_Lt():
    """Figure 3 (curve): invasion front L(t) vs Washburn fit."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    r = json.load(open(CASE_DIR / "result.json"))

    for W, color, label in [(32, "#d62728", "W32"), (64, "#ff7f0e", "W64")]:
        steps, L, _, _ = load_series(W)
        ax.plot(steps, L, color=color, lw=1.0, alpha=0.8, label=f"Sim L(t) W={W}")

        # Washburn fit: L = A * t^beta
        beta = r["part2"][f"W{W}"]["beta"]
        win = r["part2"][f"W{W}"]["window"]
        t1, t2 = win["t1"], win["t2"]
        mask = (steps >= t1) & (steps <= t2)
        if mask.any():
            t_win = steps[mask]
            L_win = L[mask]
            log_t = np.log(t_win)
            log_L = np.log(L_win)
            coeffs = np.polyfit(log_t, log_L, 1)
            beta_fit, log_A = coeffs[0], coeffs[1]
            A_fit = np.exp(log_A)
            t_full = np.linspace(t1, steps[-1], 200)
            L_fit = A_fit * t_full**beta_fit
            ax.plot(t_full, L_fit, color=color, ls="--", lw=1.5,
                    label=f"Fit L={A_fit:.1f}*t^{beta_fit:.3f}")

    ax.set_xlabel("Step t")
    ax.set_ylabel("Invasion front position L(t)")
    ax.set_title("Capillary invasion front L(t): simulation vs Washburn fit")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_front_Lt.png")
    plt.close(fig)
    print("  fig_front_Lt.png OK")


def plot_phase_field():
    """Figure 4 (field): phase field snapshots for W32 and W64."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for i, W in enumerate([32, 64]):
        d = np.load(CASE_DIR / f"out_part2_W{W}_final.npz")
        phi = d["phi"]  # (ny, nx)
        solid = d["solid"]

        phi_plot = np.where(solid, np.nan, phi)
        im = axes[i].imshow(phi_plot, aspect="auto", origin="lower",
                            cmap="RdBu_r", vmin=-1, vmax=1)
        axes[i].set_title(f"W={W} phase field phi (final state)")
        axes[i].set_xlabel("x (lattice units)")
        axes[i].set_ylabel("y (lattice units)")
        plt.colorbar(im, ax=axes[i], label="phi=(rho_w-rho_g)/(rho_w+rho_g)")

    fig.suptitle("Phase field snapshots (red=water, blue=gas): SC-MCMP interface agitation visible", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_phase_field.png")
    plt.close(fig)
    print("  fig_phase_field.png OK")


def plot_convergence_comparison():
    """Figure 5 (curve): b_max_rel_diff convergence across grid levels."""
    r = json.load(open(CASE_DIR / "result.json"))
    W32 = r["part2"]["W32"]
    W64 = r["part2"]["W64"]

    fig, ax = plt.subplots(figsize=(6, 4))
    grids = ["W32", "W64"]
    vals = [W32["b_max_rel_diff"], W64["b_max_rel_diff"]]
    colors = ["#d62728" if v > 0.03 else "#2ca02c" for v in vals]

    bars = ax.bar(grids, vals, color=colors, alpha=0.8, width=0.5)
    ax.axhline(0.03, color="k", ls="--", lw=1.5, label="3% threshold")
    ax.set_ylabel("b_max_rel_diff (max rel. deviation L(t) vs ODE)")
    ax.set_title("L(t) absolute criterion: anti-convergence (6.894 -> 9.492)")
    ax.legend()

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Grid width")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_convergence_comparison.png")
    plt.close(fig)
    print("  fig_convergence_comparison.png OK")


if __name__ == "__main__":
    print("Generating capillary_invasion_washburn terminal verdict figures...")
    plot_beta_convergence()
    plot_mass_drift()
    plot_front_Lt()
    plot_phase_field()
    plot_convergence_comparison()
    print("All figures generated successfully.")
