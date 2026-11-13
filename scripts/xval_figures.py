"""Generate figures for the XVAL report."""

import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from tensorlbm.units.anchors import ANCHORS, TOLERANCES, build_case, evaluate
from tensorlbm.xval.xval_benchmark import (
    XVAL_CASES, _make_bridge, _make_lbt_converter,
    _compare_derived_quantities, _anchor_xval, _grid_convergence_study,
)


def fig_derived_quantities_comparison(out_dir: Path) -> str:
    """Bar chart of max relative difference per derived quantity across all cases."""
    quantities = ["dx", "dt", "nu_lb", "tau", "Ma", "Re"]
    max_diffs = {q: 0.0 for q in quantities}

    for name, case in XVAL_CASES.items():
        bridge = _make_bridge(case)
        lbt = _make_lbt_converter(case, bridge)
        derived = _compare_derived_quantities(bridge, lbt, case)
        for q in quantities:
            max_diffs[q] = max(max_diffs[q], derived[q]["rel_diff"])

    fig, ax = plt.subplots(figsize=(8, 4))
    vals = [max_diffs[q] for q in quantities]
    colors = ["#2ca02c" if v < 1e-12 else "#d62728" for v in vals]
    bars = ax.bar(quantities, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=0.15, color="red", linestyle="--", linewidth=1.5, label="15% threshold")
    ax.axhline(y=1e-12, color="blue", linestyle=":", linewidth=1.0, label="Machine precision")
    ax.set_yscale("log")
    ax.set_ylabel("Max relative difference")
    ax.set_title("TensorLBM LatticeBridge vs 2b-t/LB-t Converter\nDerived Quantities Comparison")
    ax.legend()
    ax.set_ylim(bottom=1e-18)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.5,
                f"{val:.1e}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = out_dir / "fig_derived_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def fig_anchor_xval(out_dir: Path) -> str:
    """Bar chart comparing anchor simulation errors vs converter differences."""
    anchor_names = list(ANCHORS)
    sim_errors = []
    conv_diffs = []
    for name in anchor_names:
        result = _anchor_xval(name, grid=32)
        sim_errors.append(result["anchor_sim_error"])
        conv_diffs.append(result.get("converter_rel_diff",
                                     result.get("rate_converter_rel_diff", 0)))

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(anchor_names))
    width = 0.35
    bars1 = ax.bar(x - width / 2, sim_errors, width, label="Simulation error",
                   color="#1f77b4", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, [max(d, 1e-18) for d in conv_diffs], width,
                   label="Converter diff", color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax.axhline(y=0.15, color="red", linestyle="--", linewidth=1.5, label="15% threshold")
    ax.set_yscale("log")
    ax.set_ylabel("Relative error")
    ax.set_title("Anchor Cross-Validation: Simulation Error vs Converter Difference")
    ax.set_xticks(x)
    ax.set_xticklabels(anchor_names)
    ax.legend()
    ax.set_ylim(bottom=1e-18)

    fig.tight_layout()
    path = out_dir / "fig_anchor_xval.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def fig_grid_convergence(out_dir: Path) -> str:
    """Line plot of anchor errors vs grid resolution."""
    grids = (16, 32, 48, 64, 96)
    fig, ax = plt.subplots(figsize=(8, 5))

    for anchor_name in ANCHORS:
        errors = []
        for g in grids:
            case = build_case(anchor_name, 0, g)
            res = evaluate(case)
            errors.append(res["worst_rel_error"])
        ax.loglog(grids, errors, "o-", label=anchor_name)

    ax.set_xlabel("Grid resolution (cells)")
    ax.set_ylabel("Worst relative error")
    ax.set_title("Anchor Grid Convergence Study")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    path = out_dir / "fig_grid_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


if __name__ == "__main__":
    out_dir = Path("artifacts/xval_lbt_converter")
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = fig_derived_quantities_comparison(out_dir)
    p2 = fig_anchor_xval(out_dir)
    p3 = fig_grid_convergence(out_dir)
    print(f"Figures written: {p1}, {p2}, {p3}")
