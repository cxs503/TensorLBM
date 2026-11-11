"""Generate figures for sphere drag benchmark.

Produces:
1. C_D vs Re (log-log) with Schiller–Naumann curve and LBM data points
2. Grid convergence plot (C_D vs D for each Re)
3. Velocity magnitude contour at Re=100 (mid-z slice)

Usage::
    PYTHONPATH=src python benchmarks/verified/sphere_drag_bfl/generate_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def schiller_naumann_cd(re_vals: list[float]) -> list[float]:
    return [24.0 / r * (1.0 + 0.15 * r ** 0.687) + 0.42 / (1.0 + 42500.0 / r ** 1.16)
            for r in re_vals]


def load_results(path: str | Path = "benchmarks/verified/sphere_drag_bfl/results.json") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def plot_cd_vs_re(results: dict, output_dir: Path) -> Path:
    """C_D vs Re log-log plot with Schiller–Naumann reference."""
    re_fine = np.logspace(-1, 3.5, 200)
    cd_sn = schiller_naumann_cd(re_fine.tolist())

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.loglog(re_fine, cd_sn, "k-", linewidth=1.5, label="Schiller–Naumann (1933)")

    # Plot LBM data points
    summary = results.get("summary", [])
    for diam in [20, 40]:
        re_vals = [s["re"] for s in summary if s["diameter"] == diam]
        cd_vals = [s["cd"] for s in summary if s["diameter"] == diam]
        if re_vals:
            ax.loglog(re_vals, cd_vals, "o", markersize=6,
                      label=f"TensorLBM D3Q19 (D={diam})")

    ax.set_xlabel("Reynolds number Re", fontsize=12)
    ax.set_ylabel("Drag coefficient $C_D$", fontsize=12)
    ax.set_title("Sphere Drag Coefficient vs Reynolds Number", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    out = output_dir / "cd_vs_re.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_grid_convergence(results: dict, output_dir: Path) -> Path:
    """Grid convergence: C_D vs diameter for each Re."""
    grid_conv = results.get("grid_convergence", {})
    if not grid_conv:
        # Create placeholder
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.text(0.5, 0.5, "Grid convergence data pending\n(run full study first)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        out = output_dir / "grid_convergence.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    re_vals = sorted(grid_conv.keys(), key=lambda x: float(x.replace("Re", "")))
    x = np.arange(len(re_vals))
    width = 0.35

    cd_d20 = [grid_conv[r]["cd_D20"] for r in re_vals]
    cd_d40 = [grid_conv[r]["cd_D40"] for r in re_vals]

    ax.bar(x - width/2, cd_d20, width, label="D=20", color="steelblue")
    ax.bar(x + width/2, cd_d40, width, label="D=40", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(re_vals, fontsize=9)
    ax.set_xlabel("Reynolds number", fontsize=11)
    ax.set_ylabel("$C_D$", fontsize=11)
    ax.set_title("Grid Convergence: Sphere Drag Coefficient", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out = output_dir / "grid_convergence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_flow_field(results: dict, output_dir: Path) -> Path | None:
    """Velocity magnitude contour at Re=100 (mid-z slice)."""
    # Look for flow field data
    flow_files = list(output_dir.glob("flow_field_*.npz"))
    if not flow_files:
        return None

    # Use the first available flow field
    data = np.load(flow_files[0])
    speed = data["speed_mid_z"]
    solid = data["solid_mid_z"]

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    im = ax.imshow(speed, origin="lower", cmap="viridis", aspect="auto")
    ax.contour(solid, levels=[0.5], colors="white", linewidths=1.0)
    ax.set_xlabel("x (lattice units)", fontsize=11)
    ax.set_ylabel("y (lattice units)", fontsize=11)
    ax.set_title("Velocity Magnitude – Mid-z Slice (Re=100)", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, label="|u| (lattice units)")

    out = output_dir / "flow_field_Re100.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    output_dir = Path("benchmarks/verified/sphere_drag_bfl")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results()

    if results:
        p1 = plot_cd_vs_re(results, output_dir)
        print(f"Saved: {p1}")

        p2 = plot_grid_convergence(results, output_dir)
        print(f"Saved: {p2}")

        p3 = plot_flow_field(results, output_dir)
        if p3:
            print(f"Saved: {p3}")
    else:
        print("No results.json found; creating placeholder figures")
        plot_cd_vs_re({}, output_dir)
        plot_grid_convergence({}, output_dir)


if __name__ == "__main__":
    main()
