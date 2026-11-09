#!/usr/bin/env python3
"""Generate validation plots for IN99-N03."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RUNS_DIR = Path(__file__).parent / "runs"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_profiles(label):
    """Load profile data for a case."""
    with open(RUNS_DIR / f"{label}_profiles.json") as f:
        return json.load(f)


def plot_u_plus(ax, profiles, label, dns_label="DNS"):
    """Plot mean velocity profile."""
    yp = np.array(profiles["y_plus"])
    up = np.array(profiles["u_plus"])
    dns_yp = np.array(profiles["dns_y_plus"])
    dns_up = np.array(profiles["dns_u_plus"])

    ax.semilogx(dns_yp, dns_up, "k-", linewidth=2, label=dns_label)
    ax.semilogx(yp, up, "ro", markersize=4, label=f"LBM ({label})")

    # Log law reference
    y_log = np.logspace(1, 3, 100)
    ax.semilogx(y_log, np.log(y_log) / 0.41 + 5.2, "k--", linewidth=1, alpha=0.5, label="Log law")
    ax.semilogx(y_log, y_log, "k:", linewidth=1, alpha=0.5, label="u+=y+")

    ax.set_xlabel("y+")
    ax.set_ylabel("U+")
    ax.set_xlim(1, 1000)
    ax.set_ylim(0, 25)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_reynolds_stresses(ax, profiles, label):
    """Plot Reynolds stress profiles."""
    yp = np.array(profiles["y_plus"])
    uu = np.array(profiles["uu_plus"])
    vv = np.array(profiles["vv_plus"])
    ww = np.array(profiles["ww_plus"])
    uv = -np.array(profiles["uv_plus"])

    dns_yp = np.array(profiles["dns_y_plus"])
    dns_uu = np.array(profiles["dns_uu_plus"])
    dns_vv = np.array(profiles["dns_vv_plus"])
    dns_ww = np.array(profiles["dns_ww_plus"])
    dns_uv = -np.array(profiles["dns_uv_plus"])

    ax.plot(dns_yp, dns_uu, "k-", linewidth=1.5, label="DNS <uu>")
    ax.plot(dns_yp, dns_vv, "k--", linewidth=1.5, label="DNS <vv>")
    ax.plot(dns_yp, dns_ww, "k-.", linewidth=1.5, label="DNS <ww>")
    ax.plot(dns_yp, dns_uv, "k:", linewidth=1.5, label="DNS <-uv>")

    ax.plot(yp, uu, "ro", markersize=3, label="LBM <uu>")
    ax.plot(yp, vv, "bs", markersize=3, label="LBM <vv>")
    ax.plot(yp, ww, "g^", markersize=3, label="LBM <ww>")
    ax.plot(yp, uv, "md", markersize=3, label="LBM <-uv>")

    ax.set_xlabel("y+")
    ax.set_ylabel("Stress+")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 5)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)


def main():
    labels = ["retau180", "retau395", "retau590"]
    re_taus = [180, 395, 590]

    for label, re_tau in zip(labels, re_taus):
        prof_path = RUNS_DIR / f"{label}_profiles.json"
        if not prof_path.exists():
            print(f"Skipping {label}: no profiles found")
            continue

        profiles = load_profiles(label)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        plot_u_plus(ax1, profiles, label, f"DNS Re_τ={re_tau}")
        ax1.set_title(f"Mean Velocity Profile - Re_τ={re_tau}")

        plot_reynolds_stresses(ax2, profiles, label)
        ax2.set_title(f"Reynolds Stresses - Re_τ={re_tau}")

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"{label}_profiles.png", dpi=150)
        plt.close(fig)
        print(f"Saved {FIGURES_DIR / f'{label}_profiles.png'}")

    # Combined plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, (label, re_tau) in enumerate(zip(labels, re_taus)):
        prof_path = RUNS_DIR / f"{label}_profiles.json"
        if not prof_path.exists():
            continue
        profiles = load_profiles(label)
        plot_u_plus(axes[0, i], profiles, label, f"DNS Re_τ={re_tau}")
        axes[0, i].set_title(f"U+ Re_τ={re_tau}")
        plot_reynolds_stresses(axes[1, i], profiles, label)
        axes[1, i].set_title(f"Stresses Re_τ={re_tau}")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "combined_profiles.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIGURES_DIR / 'combined_profiles.png'}")


if __name__ == "__main__":
    main()
