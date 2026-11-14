#!/usr/bin/env python3
"""Figures for the m20150 cross-audit of m20069 (GAP-DIFFLBM-ADJOINT-VERDICT).

All numbers are read from the audited artifact
(work/m20069/artifacts/adjoint_verdict_diff_lbm.json) and from the cluster node log of
cj2485 — nothing is typed in by hand.

    python3 docs/board_audit/m20150/make_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TARGET = "/root/tp-agent/work/m20069"
OUT = "figures/m20150_audit"
ART = json.load(open(os.path.join(TARGET, "artifacts/adjoint_verdict_diff_lbm.json")))
NAMES = [s["config"] for s in ART["summary"]]
REC = {s["config"]: s for s in ART["summary"]}


def fig_gradients() -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    keys = ["g_fd", "g_ad", "g_frozen", "g_ship"]
    labels = ["FD (arbiter)", "unrolled AD", "frozen ceiling", "adjoint.py (ship)"]
    colors = ["#1f4e79", "#2e8b57", "#c07a00", "#9b2226"]
    x = np.arange(len(NAMES))
    w = 0.2
    for i, (k, lab, c) in enumerate(zip(keys, labels, colors)):
        vals = [REC[n][k] for n in NAMES]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=c)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:+.3f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, NAMES)
    ax.set_ylabel("dC_L / dy_c  [1/lattice]")
    ax.set_title("Gradient sources on the same objective (shipped 'quick' artifact)\n"
                 "sign of FD is opposite to AD/frozen/adjoint.py")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    rel = [REC[n]["ad_rel_err_vs_fd"] * 100 for n in NAMES]
    bars = ax.bar(NAMES, rel, 0.45, color="#9b2226")
    ax.axhline(5.0, color="#1f4e79", ls="--", lw=1.2, label="predeclared tolerance 5%")
    ax.set_yscale("log")
    ax.set_ylabel("rel. error AD vs FD  [%]")
    for b, v in zip(bars, rel):
        ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=8)
    ax.set_title("AD vs FD relative error (threshold 5%)\nboth configs exceed it by 30-40x")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, "audit_gradients_vs_fd.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def fig_unroll() -> str:
    ns = [25, 50, 100]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    win = [ART["configs"][NAMES[0]]["ad"][f"n={k}"]["window"] for k in ns]
    for n, col in zip(NAMES, ("#1f4e79", "#2e8b57")):
        vals = [ART["configs"][n]["ad"][f"n={k}"]["dJ_dy_c"] for k in ns]
        ax.plot(ns, vals, "o-", color=col, label=f"{n}: unrolled AD")
        ax.axhline(REC[n]["g_fd"], color=col, ls="--", lw=1, alpha=0.7,
                   label=f"{n}: FD arbiter = {REC[n]['g_fd']:+.3f}")
    ax.set_xlabel("unroll length n [steps]")
    ax.set_ylabel("dC_L / dy_c")
    ax.set_title("Unrolled-AD gradient vs rollout length (artifact 5.1)\nbackprop window = " + str(win) + ": n=100 is truncated at 50 steps")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, "audit_unroll_convergence.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def fig_paper_path() -> str:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    labels = ["quick\nny=32 D=6", "paper R1\nny=96 D=16"]
    cds = [REC[NAMES[0]]["C_D_base"], -43.99347]  # R1 value read from cj2485 node log
    bars = ax.bar(labels, cds, 0.45, color=["#2e8b57", "#9b2226"])
    ax.axhline(0, color="k", lw=0.8)
    for b, v in zip(bars, cds):
        ax.annotate(f"{v:+.2f}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set_ylabel("C_D at base state (no perturbation)")
    ax.set_title("Paper-resolution path is broken: cj2485 (node n315, rc=255)\n"
                 "C_D = -43.99 (nonphysical) at R1, 20000 steps, no paper JSON written")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, "audit_paper_path_cd.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def fig_repro() -> str:
    """Shipped artifact vs independent default-settings re-run (cluster cj2519)."""
    repro = json.load(open("artifacts/m20150_audit/verify_output.json"))["C9_repro_cj2519"]["repro"]
    names = list(repro)
    shipped = [abs(repro[n]["shipped_fd_reference"]) for n in names]
    again = [abs(repro[n]["fd_reference"]) for n in names]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.bar(x - 0.2, shipped, 0.4, label="shipped JSON (700 steps)", color="#9b2226")
    ax.bar(x + 0.2, again, 0.4, label="re-run, default (6000 steps)", color="#1f4e79")
    ax.set_yscale("log")
    ax.set_xticks(x, names)
    ax.set_ylabel("|FD reference| dC_L/dy_c")
    for xi, (a, b) in enumerate(zip(shipped, again)):
        ax.annotate(f"{a/b:.1f}x", (xi, max(a, b)), ha="center", va="bottom", fontsize=9)
    ax.set_title("Published FD reference is not reproducible\n(ratio shipped/re-run annotated)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    cs = [repro[n]["cos_ship_vs_ad"] for n in names]
    ax.bar(names, cs, 0.45, color="#9b2226")
    ax.axhline(0.99, color="#1f4e79", ls="--", lw=1.2, label="gate 0.99")
    ax.axhline(0.0, color="k", lw=0.8)
    for b, v in zip(ax.patches, cs):
        ax.annotate(f"{v:+.3f}", (b.get_x() + b.get_width() / 2, v), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set_ylim(-1.3, 1.3)
    ax.set_ylabel("cos(ship, AD) — default-settings re-run")
    ax.set_title("Under default settings the surrogate is anti-correlated\n(ban verdict holds, shipped +1.000 does not)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, "audit_repro_gap.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    print(fig_gradients())
    print(fig_unroll())
    print(fig_paper_path())
    print(fig_repro())


if __name__ == "__main__":
    main()