#!/usr/bin/env python3
"""Figures for XVAL-LBM-SHARED-MEMORY-STREAMING-WARP-SHIFT (reads results.json)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())


def fig_decay() -> Path:
    c = R["decay_curve"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.semilogy(c["t"], c["analytic"], "k-", lw=2, label="analytic $E_0e^{-2\\nu k^2t}$")
    ax.semilogy(c["t"], c["warp_shift"], "o", ms=3.5, label="warp-shift, wraps closed (A2)")
    ax.semilogy(
        c["t"], c["oss_verbatim"], "s", ms=3.5, mfc="none", label="upstream edge rule (A1)"
    )
    ax.set_xlabel("lattice steps $t$")
    ax.set_ylabel("kinetic energy $E(t)$")
    ax.set_title(
        f"Taylor-Green decay, L={c['L']}, $u_0$={c['u0']}, $\\tau$={c['tau']}"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    p = HERE / "fig_tg_decay.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_k3() -> Path:
    rows = R["K3"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for u0 in sorted({r["u0"] for r in rows}):
        sel = [r for r in rows if r["u0"] == u0]
        sel.sort(key=lambda r: r["L"])
        ax.plot(
            [r["L"] for r in sel],
            [100 * r["rel_err"] for r in sel],
            "o-",
            label=f"$u_0$={u0} (Ma={sel[0]['Ma']:.3f})",
        )
    ax.axhline(15.0, color="r", ls="--", lw=1.2, label="owner band 15 %")
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 128, 256])
    ax.set_xticklabels(["64", "128", "256"])
    ax.set_xlabel("grid resolution $L$ (lattice sites)")
    ax.set_ylabel("kinetic-energy decay-rate error  [%]")
    ax.set_title("K3: warp-shift streaming vs analytic decay rate ($\\tau$=0.8, 800 steps)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    p = HERE / "fig_k3_error.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_timing() -> Path:
    rows = R["K5_timing"]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    labels, vals, colors = [], [], []
    for L in sorted({r["L"] for r in rows}):
        sel = [r for r in rows if r["L"] == L]
        for r in sel:
            tag = "pull (repo ref)" if r["backend"] == "pull" else f"warp tile {r['tile']}"
            labels.append(f"L={L}\n{tag}")
            vals.append(r["mlups"])
            colors.append("#4c72b0" if r["backend"] == "pull" else "#dd8452")
    x = range(len(vals))
    ax.bar(x, vals, color=colors)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=6.5, rotation=90)
    ax.set_ylabel("MLUPS (M site-updates / s)")
    ax.set_title("K5: streaming throughput, single CPU thread, float64")
    ax.grid(axis="y", alpha=0.3)
    p = HERE / "fig_timing.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_operator() -> Path:
    """Schematic of the upstream staging rule (no numbers, mechanism only)."""
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    w = 8
    for i in range(w):
        ax.add_patch(plt.Rectangle((i, 1.0), 0.9, 0.7, fc="#dbe6f2", ec="k"))
        ax.add_patch(plt.Rectangle((i, 0.0), 0.9, 0.7, fc="#f6e2cf", ec="k"))
    ax.annotate("", xy=(0.45, 1.0), xytext=(1.45, 1.0),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#3b6ea5"))
    ax.annotate("", xy=(w - 0.55, 1.0), xytext=(w - 1.55, 1.0),
                arrowprops=dict(arrowstyle="<-", lw=1.4, color="#a5533b"))
    ax.annotate("", xy=(0.95, 0.2), xytext=(0.95, 1.0),
                arrowprops=dict(arrowstyle="-", lw=1.0, ls=":", color="k"))
    ax.text(w / 2 - 0.4, 1.85, "row tile staged in shared memory ($w$ = warp size)",
            fontsize=8, ha="center")
    ax.text(w / 2 - 0.4, -0.45,
            "one-lane shift: E/NE/SE -> lane+1, W/NW/SW -> lane-1; the edge lane keeps the\n"
            "departing population (modular wrap) and the cross-tile repair copies it in global memory",
            fontsize=7, ha="center")
    ax.set_xlim(-0.3, w + 0.2)
    ax.set_ylim(-0.8, 2.1)
    ax.axis("off")
    p = HERE / "fig_operator.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


if __name__ == "__main__":
    for p in (fig_operator(), fig_decay(), fig_k3(), fig_timing()):
        print(p.name, p.stat().st_size)