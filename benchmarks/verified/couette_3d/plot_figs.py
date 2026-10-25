#!/usr/bin/env python3
"""Generate figures for couette_3d benchmark (paper profile).

Reads existing result artifacts and generates:
1. field_midplane_speed.png: midplane u(y) velocity field visualization
2. curve_profile_vs_analytical.png: u(y) profile vs analytical (fp64 four tiers)
3. curve_l2_convergence.png: l2 error convergence curve

All figures generated from existing data artifacts (zero hand-copying).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Data source paths
DATA_DIR = Path("/nfs/wangxi/runs/bm_widen_w11_20261006/d_siblings/couette3d/gate/fp64")
RESULT_JSON = Path(__file__).parent / "result.json"


# Load data
def load_tier_data(H: int) -> dict:
    """Load fp64 tier data from artifact."""
    path = DATA_DIR / f"case_H{H}_fp64.json"
    with open(path) as f:
        return json.load(f)


def load_result() -> dict:
    """Load result.json summary."""
    with open(RESULT_JSON) as f:
        return json.load(f)


# Figure 1: Midplane velocity field visualization
def plot_field_midplane():
    """Generate midplane u(y) velocity field (H=32 tier)."""
    data = load_tier_data(32)
    H = data["H_eff"]
    ny = data["ny"]
    nx = data["nx"]
    nz = data["nz"]

    # Reconstruct full 3D field including walls
    # u_profile contains fluid cells only (ny-2 elements)
    u_fluid = np.array(data["u_profile"])

    # Full profile: wall (u=0) + fluid + wall (u=U0)
    u_full = np.zeros(ny)
    u_full[1:-1] = u_fluid
    u_full[-1] = data["U0"]  # Top wall velocity

    # Create 2D midplane (x-y at z=nz//2)
    # Since flow is x/z uniform, midplane is just the profile repeated
    u_midplane = np.tile(u_full[:, np.newaxis], (1, nx))

    # y coordinates
    y = np.arange(ny)
    x = np.arange(nx)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot velocity magnitude as colormap
    im = ax.pcolormesh(X, Y, u_midplane, cmap="viridis", shading="auto")
    cbar = plt.colorbar(im, ax=ax, label="u_x (lattice units)")

    # Mark walls
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2, label="Bottom wall (u=0)")
    ax.axhline(y=ny - 1, color="blue", linestyle="--", linewidth=2, label="Top wall (u=U₀)")

    ax.set_xlabel("x (lattice units)", fontsize=12)
    ax.set_ylabel("y (lattice units)", fontsize=12)
    ax.set_title(f"Couette 3D: Midplane Velocity Field (H={H}, z={nz // 2})", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_aspect("equal")

    plt.tight_layout()
    out_path = Path(__file__).parent / "figures" / "field_midplane_speed.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_path} ({out_path.stat().st_size} bytes)")


# Figure 2: u(y) profile vs analytical (all fp64 tiers)
def plot_profile_vs_analytical():
    """Plot u(y) profiles for all fp64 tiers vs analytical solution."""
    result = load_result()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    tiers = [16, 32, 64, 128]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (H, color) in enumerate(zip(tiers, colors)):
        ax = axes[idx]
        data = load_tier_data(H)

        u_num = np.array(data["u_profile"])
        u_ana = np.array(data["u_analytic"])
        ny = data["ny"]
        H_eff = data["H_eff"]

        # u_profile and u_analytic are fluid cells only (H_eff elements)
        # y for fluid cells: 1 to ny-2
        y_fluid = np.arange(1, ny - 1)

        # Normalize y to [0, 1]
        y_norm = (y_fluid - 0.5) / H_eff

        # Plot numerical vs analytical
        ax.plot(u_ana, y_norm, "k--", linewidth=2, label="Analytical", alpha=0.7)
        ax.plot(u_num, y_norm, "-", color=color, linewidth=2, label=f"H={H} (fp64)")

        ax.set_xlabel("u_x (lattice units)", fontsize=11)
        ax.set_ylabel("y / H_eff", fontsize=11)
        ax.set_title(f"H={H}: l₂={data['l2_rel_err']:.2e}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.055)

    fig.suptitle("Couette 3D: Velocity Profile vs Analytical Solution (fp64)", fontsize=14, y=0.995)
    plt.tight_layout()
    out_path = Path(__file__).parent / "figures" / "curve_profile_vs_analytical.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_path} ({out_path.stat().st_size} bytes)")


# Figure 3: l2 error convergence curve
def plot_l2_convergence():
    """Plot l2_rel_err vs grid size H."""
    result = load_result()

    tiers = [16, 32, 64, 128]
    l2_errors = []

    for H in tiers:
        data = load_tier_data(H)
        l2_errors.append(data["l2_rel_err"])

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.semilogy(
        tiers, l2_errors, "o-", color="#2ca02c", linewidth=2, markersize=10, label="fp64 l₂ error"
    )

    # Add threshold lines
    ax.axhline(y=0.03, color="red", linestyle="--", linewidth=1.5, label="3% threshold", alpha=0.7)
    ax.axhline(
        y=1e-7,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label="1e-7 (G2b threshold)",
        alpha=0.7,
    )

    # Annotate points
    for H, l2 in zip(tiers, l2_errors):
        ax.annotate(
            f"{l2:.1e}",
            xy=(H, l2),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

    ax.set_xlabel("Grid size H", fontsize=12)
    ax.set_ylabel("l₂ relative error", fontsize=12)
    ax.set_title("Couette 3D: l₂ Error Convergence (fp64)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(10, 140)

    plt.tight_layout()
    out_path = Path(__file__).parent / "figures" / "curve_l2_convergence.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Generated: {out_path} ({out_path.stat().st_size} bytes)")


def main():
    """Generate all figures."""
    print("Generating couette_3d figures...")
    plot_field_midplane()
    plot_profile_vs_analytical()
    plot_l2_convergence()
    print("All figures generated successfully.")


if __name__ == "__main__":
    main()
