#!/usr/bin/env python3
"""Figures for the Re=100 cylinder LBM/FVM cross-validation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REF_CD = 1.35
REF_BAND = (1.32, 1.37)


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fig_history(lbm: dict, fvm: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # LBM: t* = steps * u_lat / D ; FVM: t* = t * U / D (U = 1, D = 1)
    lbm_t = np.arange(1, len(lbm["drag_lift_history"]["drag"]) + 1) * lbm["numerics"]["u_in_lattice"] / lbm["mesh"]["diameter_lattice_units"]
    fvm_t = np.arange(1, len(fvm["drag_lift_history"]["drag"]) + 1) * fvm["numerics"]["time_step"] / fvm["mesh"]["diameter"]
    for ax, key, label in ((axes[0], "drag", "$C_d$"), (axes[1], "lift", "$C_l$")):
        ax.plot(lbm_t, lbm["drag_lift_history"][key], lw=0.8, color="#1f77b4",
                label=f"TensorLBM D2Q9 ({lbm['mesh']['diameter_lattice_units']} cells/D)")
        ax.plot(fvm_t, fvm["drag_lift_history"][key], lw=0.9, color="#d62728",
                label=f"TensorFVM SIMPLE ({fvm['mesh']['nx']}x{fvm['mesh']['ny']} O-grid)")
        ax.set_xlabel("convective time  $tU/D$")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].axhline(REF_CD, ls="--", color="k", lw=0.8, label="Henderson 1995")
    axes[0].axhspan(*REF_BAND, color="grey", alpha=0.15)
    axes[0].set_title("Drag coefficient history (shaded = literature band)")
    axes[1].set_title("Lift coefficient history (shedding phase differs)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_bar(lbm: dict, fvm: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    names = ["TensorLBM\nD2Q9", "TensorFVM\nSIMPLE", "Henderson 1995"]
    vals = [lbm["cd_mean"], fvm["cd_mean"], REF_CD]
    colors = ["#1f77b4", "#d62728", "#555555"]
    bars = ax.bar(names, vals, color=colors, width=0.55)
    ax.axhspan(*REF_BAND, color="grey", alpha=0.18, zorder=0)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.4f}",
                ha="center", fontsize=9)
    ax.set_ylim(min(vals) - 0.08, max(vals) + 0.08)
    ax.set_ylabel("cycle-averaged $C_d$")
    gap = 100 * abs(vals[0] - vals[1]) / (0.5 * (vals[0] + vals[1]))
    ax.set_title(f"Re=100 cylinder drag: cross-solver gap {gap:.2f}% (tolerance 5%)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_grid(lbm: dict, fvm: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for solver, color, marker in ((lbm, "#1f77b4", "o"), (fvm, "#d62728", "s")):
        levels = solver["grid_levels"]
        cells = [lv["cells"] for lv in levels]
        cds = [lv["cd_mean"] for lv in levels]
        ax.plot(cells, cds, marker=marker, color=color, label=solver["solver"].split("(")[0].strip())
    ax.axhspan(*REF_BAND, color="grey", alpha=0.18, zorder=0, label="literature band")
    ax.axhline(REF_CD, ls="--", color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("cells in the domain")
    ax.set_ylabel("cycle-averaged $C_d$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title("Grid level sensitivity of the drag coefficient")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lbm", type=Path, required=True)
    ap.add_argument("--fvm", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("docs/assets"))
    args = ap.parse_args(argv)
    lbm, fvm = _load(args.lbm), _load(args.fvm)
    args.outdir.mkdir(parents=True, exist_ok=True)
    fig_history(lbm, fvm, args.outdir / "cylinder_cross_validation_history.png")
    fig_bar(lbm, fvm, args.outdir / "cylinder_cross_validation_cd.png")
    fig_grid(lbm, fvm, args.outdir / "cylinder_cross_validation_grid.png")
    print("figures written to", args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
