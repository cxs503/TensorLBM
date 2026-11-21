#!/usr/bin/env python3
"""Generate verification figures for P96W3-N06 — fast version."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.wave_wake_michell import (
    G, RHO,
    WIGLEY_III, WIGLEY_STANDARD,
    ParabolicHull,
    caustic_angle_deg,
    catamaran_interference,
    hump_hollow,
    michell_resistance,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

SERIES60_CB06 = ParabolicHull(
    length=6.096, beam=0.938, draft=0.342,
    vertical="power", alpha=9.0, name="series60-cb0.60",
)


def fig01_cw_curves():
    """C_w(Fn) curves for all three hulls."""
    fns = np.arange(0.20, 0.50, 0.01)

    fig, ax = plt.subplots(figsize=(10, 6))

    for hull, label, color in [
        (WIGLEY_STANDARD, "Wigley Standard (Cb=4/9)", "#1f77b4"),
        (WIGLEY_III, "Wigley III (Cb=0.555)", "#ff7f0e"),
        (SERIES60_CB06, "Series 60 (Cb=0.60)", "#2ca02c"),
    ]:
        cw_vals = []
        for fn in fns:
            rec = michell_resistance(hull, fn, n_x=121, n_z=41, n_theta=400)
            cw_vals.append(rec["cw"])
        ax.plot(fns, cw_vals, label=label, color=color, linewidth=2)

        # Mark the 3 Fn points
        for fn in [0.20, 0.30, 0.40]:
            rec = michell_resistance(hull, fn, n_x=161, n_z=61, n_theta=600)
            ax.plot(fn, rec["cw"], "o", color=color, markersize=8)

    ax.set_xlabel("Froude number Fn", fontsize=12)
    ax.set_ylabel("Wave resistance coefficient $C_W$", fontsize=12)
    ax.set_title("Michell Thin-Ship Wave Resistance Coefficient $C_W$(Fn)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.20, 0.50)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig01_cw_curves.png", dpi=150)
    plt.close()
    print("  Saved fig01_cw_curves.png")


def fig03_catamaran_interference():
    """Catamaran interference factor vs separation."""
    s_vals = np.linspace(0.01, 0.5, 50)

    fig, ax = plt.subplots(figsize=(10, 6))

    for mode, label, color in [
        ("transverse", "Transverse separation", "#1f77b4"),
        ("longitudinal", "Longitudinal (tandem) separation", "#ff7f0e"),
    ]:
        factors = []
        for s in s_vals:
            rec = catamaran_interference(
                WIGLEY_STANDARD, 0.30, (s,), mode=mode,
                n_x=121, n_z=41, n_theta=400,
            )
            factors.append(rec["rows"][0]["interference_factor"])
        ax.plot(s_vals, factors, label=label, color=color, linewidth=2)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No interference")
    ax.axhline(y=2.0, color="red", linestyle="--", alpha=0.5, label="Full constructive")

    ax.set_xlabel("Separation ratio S/L", fontsize=12)
    ax.set_ylabel("Interference factor", fontsize=12)
    ax.set_title("Catamaran Wave Interference Factor at Fn=0.3", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig03_catamaran_interference.png", dpi=150)
    plt.close()
    print("  Saved fig03_catamaran_interference.png")


def fig04_grid_convergence():
    """Grid convergence for Wigley standard at Fn=0.3."""
    levels = [
        (81, 31, 300, "Coarse"),
        (161, 61, 600, "Medium"),
        (321, 121, 1200, "Fine"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    cw_vals = []
    labels = []
    for nx, nz, nt, label in levels:
        rec = michell_resistance(WIGLEY_STANDARD, 0.30, n_x=nx, n_z=nz, n_theta=nt)
        cw_vals.append(rec["cw"])
        labels.append(label)

    ax1.bar(labels, cw_vals, color=["#ff7f0e", "#2ca02c", "#1f77b4"])
    ax1.set_ylabel("$C_W$", fontsize=12)
    ax1.set_title("Grid Convergence — Wigley Standard at Fn=0.3", fontsize=12)
    for i, v in enumerate(cw_vals):
        ax1.text(i, v, f"{v:.6e}", ha="center", va="bottom", fontsize=9)

    changes = [
        abs(cw_vals[1] - cw_vals[0]) / cw_vals[0] * 100,
        abs(cw_vals[2] - cw_vals[1]) / cw_vals[1] * 100,
    ]
    ax2.bar(["Coarse→Medium", "Medium→Fine"], changes, color=["#ff7f0e", "#2ca02c"])
    ax2.axhline(y=5.0, color="red", linestyle="--", label="5% threshold")
    ax2.set_ylabel("Relative change [%]", fontsize=12)
    ax2.set_title("Relative Change Between Grid Levels", fontsize=12)
    ax2.legend()
    for i, v in enumerate(changes):
        ax2.text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig04_grid_convergence.png", dpi=150)
    plt.close()
    print("  Saved fig04_grid_convergence.png")


def fig05_mass_conservation():
    """Mass conservation from LBM 2D dispersion runs."""
    lbm_dir = Path(__file__).resolve().parents[1] / "benchmarks" / "wave_wake_michell" / "lbm_runs"
    json_files = sorted(lbm_dir.glob("lbm_disp_lam*.json"))

    fig, ax = plt.subplots(figsize=(8, 5))

    lambdas = []
    drifts = []
    for f in json_files:
        data = json.loads(f.read_text())
        lambdas.append(data["lambda_over_dx"])
        drifts.append(abs(data["mass_relative_drift"]) * 100)

    ax.bar([f"λ/Δx={lam}" for lam in lambdas], drifts, color="#2ca02c")
    ax.axhline(y=1.0, color="red", linestyle="--", label="1% threshold")
    ax.set_ylabel("Mass relative drift [%]", fontsize=12)
    ax.set_title("Mass Conservation — LBM 2D Free Surface (1000 steps)", fontsize=12)
    ax.legend()
    for i, v in enumerate(drifts):
        ax.text(i, v, f"{v:.3f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig05_mass_conservation.png", dpi=150)
    plt.close()
    print("  Saved fig05_mass_conservation.png")


def fig06_hump_hollow():
    """Hump/hollow structure of C_w(Fn)."""
    fns = np.arange(0.20, 0.50, 0.01)
    cw_vals = []
    for fn in fns:
        rec = michell_resistance(WIGLEY_STANDARD, fn, n_x=121, n_z=41, n_theta=400)
        cw_vals.append(rec["cw"])
    cw_vals = np.array(cw_vals)

    hh = hump_hollow(fns, cw_vals, min_prominence_rel=1e-3)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fns, cw_vals, "b-", linewidth=2, label="$C_W$(Fn)")

    for h in hh["humps"]:
        ax.plot(h["fn"], h["cw"], "r^", markersize=12, label="Hump" if h == hh["humps"][0] else "")
        ax.annotate(f"Fn={h['fn']:.3f}", (h["fn"], h["cw"]),
                    textcoords="offset points", xytext=(10, 10), fontsize=9)

    for h in hh["hollows"]:
        ax.plot(h["fn"], h["cw"], "gv", markersize=12, label="Hollow" if h == hh["hollows"][0] else "")
        ax.annotate(f"Fn={h['fn']:.3f}", (h["fn"], h["cw"]),
                    textcoords="offset points", xytext=(10, -15), fontsize=9)

    ax.set_xlabel("Froude number Fn", fontsize=12)
    ax.set_ylabel("$C_W$", fontsize=12)
    ax.set_title("Hump/Hollow Structure of $C_W$(Fn) — Wigley Standard", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig06_hump_hollow.png", dpi=150)
    plt.close()
    print("  Saved fig06_hump_hollow.png")


if __name__ == "__main__":
    print("Generating verification figures...")
    fig01_cw_curves()
    fig03_catamaran_interference()
    fig04_grid_convergence()
    fig05_mass_conservation()
    fig06_hump_hollow()
    print("All figures generated.")
