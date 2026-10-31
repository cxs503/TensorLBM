"""Validation test: D2Q9 plane Couette flow vs analytical linear profile.

Acceptance criteria (machine判据):
    1. Linear profile L2 relative error <= 2%
    2. Top wall no-slip error |u(ny-1) - U| / U <= 3%
    3. Bottom wall no-slip error |u(0)| / U <= 3%

The analytical solution for plane Couette flow is:
    u_x(y) = U * y / H,   u_y = 0

where U is the top wall velocity and H is the channel height.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from couette_flow import CouetteConfig, run_couette

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Acceptance thresholds
# ---------------------------------------------------------------------------
L2_ERROR_THRESHOLD = 0.02       # 2%
NO_SLIP_THRESHOLD = 0.03        # 3%


def run_validation(output_dir: Path) -> dict:
    """Run Couette validation and return results + pass/fail."""
    # Grid tuned for speed + accuracy; 16×16 with 5000 steps converges well
    cfg = CouetteConfig(
        nx=16,
        ny=16,
        u_top=0.05,
        nu=0.02,
        n_steps=5000,
        n_warmup=2000,
        device="cpu",
    )

    results = run_couette(cfg)

    # Compute pass/fail
    l2_pass = results["l2_error"] <= L2_ERROR_THRESHOLD
    top_pass = results["no_slip_top"] / cfg.u_top <= NO_SLIP_THRESHOLD
    bottom_pass = results["no_slip_bottom"] / cfg.u_top <= NO_SLIP_THRESHOLD
    all_pass = l2_pass and top_pass and bottom_pass

    results["l2_pass"] = l2_pass
    results["top_pass"] = top_pass
    results["bottom_pass"] = bottom_pass
    results["all_pass"] = all_pass

    return results


def plot_velocity_profile(results: dict, output_dir: Path) -> Path:
    """Plot computed vs analytical velocity profile."""
    y = results["y_coords"].numpy()
    ux = results["ux_profile"].numpy()
    ua = results["u_analytical"].numpy()
    uy = results["uy_profile"].numpy()
    u_top = results["u_top"]
    ny = results["ny"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: ux profile comparison
    ax = axes[0]
    ax.plot(ua, y / (ny - 1), "k--", linewidth=2, label="Analytical: $u_x = U y / H$")
    ax.plot(ux, y / (ny - 1), "ro", markersize=4, label="LBM D2Q9", zorder=5)
    ax.set_xlabel(r"$u_x$ (lattice units)", fontsize=12)
    ax.set_ylabel(r"$y / H$", fontsize=12)
    ax.set_title("Plane Couette Flow — Velocity Profile", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, u_top * 1.1)
    ax.set_ylim(-0.02, 1.02)

    # Annotate L2 error
    l2_pct = results["l2_error"] * 100
    ax.text(
        0.95, 0.05,
        f"L2 error = {l2_pct:.3f}%",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    # Right: uy profile (should be ~0)
    ax = axes[1]
    ax.plot(uy, y / (ny - 1), "b-", linewidth=1.5, label="$u_y$ (LBM)")
    ax.axhline(0, color="k", linestyle="--", linewidth=1, label="Analytical: $u_y = 0$")
    ax.set_xlabel(r"$u_y$ (lattice units)", fontsize=12)
    ax.set_ylabel(r"$y / H$", fontsize=12)
    ax.set_title("Transverse Velocity (should be zero)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"D2Q9 Couette Flow  |  Re={results['re']:.0f}  |  "
        f"τ={results['tau']:.3f}  |  {results['nx']}×{results['ny']}  |  "
        f"Steps={results['n_steps']}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    out_path = output_dir / "couette_velocity_profile.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_error_distribution(results: dict, output_dir: Path) -> Path:
    """Plot error distribution across the channel."""
    y = results["y_coords"].numpy()
    ux = results["ux_profile"].numpy()
    ua = results["u_analytical"].numpy()
    u_top = results["u_top"]
    ny = results["ny"]

    error = ux - ua
    # Avoid division by zero at y=0
    rel_error = np.where(np.abs(ua) > 1e-10, error / ua, error / u_top)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: absolute error
    ax = axes[0]
    ax.plot(error * 1000, y / (ny - 1), "b-", linewidth=1.5, marker="o", markersize=3)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"Absolute error $u_x^{LBM} - u_x^{exact}$ (×10⁻³)", fontsize=11)
    ax.set_ylabel(r"$y / H$", fontsize=12)
    ax.set_title("Absolute Error Distribution", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Right: relative error
    ax = axes[1]
    ax.plot(rel_error * 100, y / (ny - 1), "r-", linewidth=1.5, marker="o", markersize=3)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax.axhline(2, color="orange", linestyle=":", linewidth=1, label="2% threshold")
    ax.axhline(-2, color="orange", linestyle=":", linewidth=1)
    ax.set_xlabel(r"Relative error (%)", fontsize=11)
    ax.set_ylabel(r"$y / H$", fontsize=12)
    ax.set_title("Relative Error Distribution", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"D2Q9 Couette Flow — Error Analysis  |  Re={results['re']:.0f}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    out_path = output_dir / "couette_error_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def encode_png_base64(png_path: Path) -> str:
    """Read PNG file and return base64 string."""
    with open(png_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def main():
    output_dir = Path(__file__).resolve().parent
    print("=" * 60)
    print("  D2Q9 Plane Couette Flow — Validation")
    print("=" * 60)

    # Run simulation
    results = run_validation(output_dir)

    # Save JSON results
    from couette_flow import save_results
    save_results(results, output_dir)

    # Generate figures
    fig1 = plot_velocity_profile(results, output_dir)
    fig2 = plot_error_distribution(results, output_dir)

    # Print results
    print(f"\n  Grid:          {results['nx']} × {results['ny']}")
    print(f"  Re:            {results['re']:.1f}")
    print(f"  tau:           {results['tau']:.4f}")
    print(f"  Steps:         {results['n_steps']}")
    print(f"  U_top:         {results['u_top']}")
    print(f"\n  --- Acceptance Criteria ---")
    print(f"  L2 error:      {results['l2_error']*100:.4f}%  (threshold: {L2_ERROR_THRESHOLD*100:.1f}%)  {'PASS' if results['l2_pass'] else 'FAIL'}")
    print(f"  No-slip top:   {results['no_slip_top']*100:.4f}% of U  (threshold: {NO_SLIP_THRESHOLD*100:.1f}%)  {'PASS' if results['top_pass'] else 'FAIL'}")
    print(f"  No-slip bot:   {results['no_slip_bottom']*100:.4f}% of U  (threshold: {NO_SLIP_THRESHOLD*100:.1f}%)  {'PASS' if results['bottom_pass'] else 'FAIL'}")
    print(f"\n  Overall:       {'PASS' if results['all_pass'] else 'FAIL'}")
    print(f"\n  Figures:       {fig1.name}")
    print(f"                 {fig2.name}")

    # Write summary for report
    summary = {
        "l2_error_pct": round(results["l2_error"] * 100, 4),
        "no_slip_top_pct": round(results["no_slip_top"] / results["u_top"] * 100, 4),
        "no_slip_bottom_pct": round(results["no_slip_bottom"] / results["u_top"] * 100, 4),
        "l2_pass": results["l2_pass"],
        "top_pass": results["top_pass"],
        "bottom_pass": results["bottom_pass"],
        "all_pass": results["all_pass"],
        "re": results["re"],
        "tau": results["tau"],
        "nx": results["nx"],
        "ny": results["ny"],
        "n_steps": results["n_steps"],
        "u_top": results["u_top"],
        "nu": results["nu"],
        "figure1": str(fig1),
        "figure2": str(fig2),
    }
    with open(output_dir / "validation_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    return 0 if results["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
