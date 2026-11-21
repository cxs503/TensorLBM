#!/usr/bin/env python3
"""K3-MOMCONV figures: crater depth, conversion budget, axial momentum.

Every number is read from the arm artefacts in ``runs/k3_*/result.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARMS = ("ctrl", "inherit", "redistribute")
COLORS = {"ctrl": "#333333", "inherit": "#1f77b4", "redistribute": "#d62728"}
EXP_DEPTH_D = 1.857434
EXP_TIME_MS = 10.394
ANCHOR_DEPTH_D = 1.4352204748
ANCHOR_TIME_MS = 11.090571428571428


def load(tree: Path, arm: str) -> dict:
    return json.loads((tree / "runs" / f"k3_{arm}" / "result.json").read_text())


def series(result: dict, key: str, source: str = "history"):
    rows = result.get(source) or []
    if source == "budget_diag":
        rows = (result.get("budget_diag") or {}).get("rows") or []
    out = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        out.append((row["time_ms"], float(value)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", default=".")
    args = parser.parse_args()
    tree = Path(args.tree).resolve()
    out_dir = tree / "docs" / "audits" / "k3_momconv_20261121" / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {arm: load(tree, arm) for arm in ARMS}

    # ---- fig 1: crater depth vs time -------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for arm in ARMS:
        xs = [(t, d) for t, d in series(data[arm], "depth_over_D") if d is not None]
        ax.plot([p[0] for p in xs], [p[1] for p in xs], color=COLORS[arm], lw=1.6, label=arm)
    ax.scatter([EXP_TIME_MS], [EXP_DEPTH_D], marker="*", s=180, color="black",
               zorder=5, label="experiment 1.857434 D @ 10.394 ms")
    ax.scatter([ANCHOR_TIME_MS], [ANCHOR_DEPTH_D], marker="o", s=55, facecolors="none",
               edgecolors="black", zorder=5, label="anchor (ctrl) 1.435220 D @ 11.091 ms")
    ax.axhline(EXP_DEPTH_D, ls=":", lw=1.0, color="black")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("crater depth / D")
    ax.set_title("K3-MOMCONV: crater depth (Santini case-3, cells=8)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_depth.png", dpi=150)
    plt.close(fig)

    # ---- fig 2: conversion channel budget --------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    width = 0.26
    labels = []
    loss, gain, net = [], [], []
    for arm in ARMS:
        rows = (data[arm].get("budget_diag") or {}).get("rows") or []
        if not rows:
            continue
        t_hi = max((r["time_ms"] for r in rows), default=0.0)
        peak = [r for r in rows if r["time_ms"] == t_hi][0]
        e_scale = rows[0]["ke_liq"] or 1.0
        window = [r for r in rows if r["time_ms"] <= t_hi]
        labels.append(arm)
        loss.append(sum(r["conv_loss"] for r in window) / e_scale)
        gain.append(sum(r["conv_gain"] for r in window) / e_scale)
        net.append((sum(r["conv_loss"] for r in window) - sum(r["conv_gain"] for r in window)) / e_scale)
    idx = range(len(labels))
    ax.bar([i - width for i in idx], loss, width, label="conv_loss", color="#d62728", alpha=0.85)
    ax.bar(list(idx), gain, width, label="conv_gain", color="#2ca02c", alpha=0.85)
    ax.bar([i + width for i in idx], net, width, label="net conversion loss", color="#9467bd", alpha=0.85)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(list(idx))
    ax.set_xticklabels(labels)
    ax.set_ylabel("share of E_scale (contact KE)")
    ax.set_title("K3-MOMCONV: mass-exchange conversion channel (cumulative to window end)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_conversion_budget.png", dpi=150)
    plt.close(fig)

    # ---- fig 3: axial liquid-column kinetic energy ------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for arm in ARMS:
        rows = (data[arm].get("budget_diag") or {}).get("rows") or []
        if not rows:
            continue
        e_scale = rows[0]["ke_liq"] or 1.0
        xs = [r["time_ms"] for r in rows]
        ys = [r["ke_axial"] / e_scale for r in rows]
        ax.plot(xs, ys, color=COLORS[arm], lw=1.6, label=arm)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("KE_axial / E_scale")
    ax.set_title("K3-MOMCONV: axial (liquid column) kinetic energy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_ke_axial.png", dpi=150)
    plt.close(fig)

    # ---- fig 4: liquid momentum (z) ---------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for arm in ARMS:
        rows = (data[arm].get("budget_diag") or {}).get("rows") or []
        if not rows:
            continue
        xs = [r["time_ms"] for r in rows]
        ys = [r["p_liq_z"] for r in rows]
        ax.plot(xs, ys, color=COLORS[arm], lw=1.6, label=arm)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("liquid momentum p_liq_z (lattice)")
    ax.set_title("K3-MOMCONV: liquid momentum (z component)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_momentum.png", dpi=150)
    plt.close(fig)

    # ---- fig 5: mid-plane fill field (crater cross-section) ---------------
    planes = {}
    for arm in ARMS:
        path = tree / "runs" / f"k3_{arm}_planes" / "planes.npz"
        if path.exists():
            planes[arm] = np.load(path, allow_pickle=True)
    if planes:
        target = EXP_TIME_MS
        fig, axes = plt.subplots(1, len(planes) + 1, figsize=(4.0 * (len(planes) + 1), 4.0))
        for ax, (arm, data) in zip(axes, planes.items()):
            times = data["time_ms"]
            idx = int(np.argmin(np.abs(times - target)))
            fill = data["fill"][idx]
            x = data["x_over_D"]
            z = data["z_over_D"]
            im = ax.contourf(x, z, fill, levels=np.linspace(0.0, 1.0, 21), cmap="Blues", vmin=0.0, vmax=1.0)
            ax.contour(x, z, fill, levels=[0.5], colors="k", linewidths=1.0)
            ax.set_title(f"{arm}  t = {times[idx]:.3f} ms")
            ax.set_xlabel("x / D")
            ax.set_ylabel("z / D")
            ax.invert_yaxis()
            fig.colorbar(im, ax=ax, fraction=0.046)
        # crater surface profile overlay
        ax = axes[-1]
        for arm, data in planes.items():
            times = data["time_ms"]
            idx = int(np.argmin(np.abs(times - target)))
            fill = data["fill"][idx]
            x = data["x_over_D"]
            z = data["z_over_D"]
            surface = []
            for j in range(fill.shape[1]):
                column = fill[:, j]
                wet = np.nonzero(column > 0.5)[0]
                surface.append(z[wet[0]] if wet.size else np.nan)
            ax.plot(x, surface, marker="o", ms=2.5, lw=1.2, label=arm)
        ax.axhline(0.0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("x / D")
        ax.set_ylabel("crater surface z / D")
        ax.set_title("crater surface at the experimental peak time")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fig5_crater_field.png", dpi=150)
        plt.close(fig)

    print("figures written to", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
