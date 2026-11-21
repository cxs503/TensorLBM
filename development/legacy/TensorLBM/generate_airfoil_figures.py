#!/usr/bin/env python3
"""Generate figures for NACA 0012 airfoil benchmark report."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
OUTDIR = REPO / "TensorLBM" / "artifacts" / "airfoil"

# Load results
results_path = OUTDIR / "results.json"
data = json.loads(results_path.read_text())
sweep = data["sweep"]
cases = sweep["cases"]
analysis = sweep["analysis"]

# Reference data
from tensorlbm.airfoil_benchmark import (
    ABBOTT_VON_DOENHOFF_NACA0012,
    THIN_AIRFOIL_CL_SLOPE,
    thin_airfoil_cl,
    abbott_von_denhoff_cl,
    abbott_von_denhoff_cd,
)

# ============================================================================
# Figure 1: C_L vs alpha
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 6))

# LBM results
alphas_lbm = [c["alpha_deg"] for c in cases]
cls_lbm = [c["cl"] for c in cases]
ax.plot(alphas_lbm, cls_lbm, "ro-", label="TensorLBM (Re=200, C=16 lu)", markersize=8, linewidth=2)

# Thin airfoil theory
alpha_th = np.linspace(-4, 12, 100)
cl_th = [thin_airfoil_cl(a) for a in alpha_th]
ax.plot(alpha_th, cl_th, "k--", label="Thin airfoil theory ($2\\pi\\alpha$)", linewidth=1.5)

# Abbott-von Doenhoff (high Re)
alphas_avd = sorted(ABBOTT_VON_DOENHOFF_NACA0012.keys())
cls_avd = [abbott_von_denhoff_cl(a) for a in alphas_avd]
ax.plot(alphas_avd, cls_avd, "bs-", label="Abbott–von Doenhoff (Re~3-6×10⁶)", markersize=6, linewidth=1.5)

ax.set_xlabel("Angle of attack $\\alpha$ [deg]", fontsize=12)
ax.set_ylabel("Lift coefficient $C_L$", fontsize=12)
ax.set_title("NACA 0012: Lift coefficient vs angle of attack", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 10)
ax.set_ylim(-0.2, 1.2)

# Add text box with slope info
textstr = (
    f"LBM slope: {analysis['cl_slope_per_rad']:.2f} /rad\n"
    f"Theory: {THIN_AIRFOIL_CL_SLOPE:.2f} /rad (2π)\n"
    f"Error: {analysis['cl_slope_err_pct']:.0f}%"
)
props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", bbox=props)

plt.tight_layout()
fig.savefig(OUTDIR / "fig_cl_vs_alpha.png", dpi=150)
plt.close()
print(f"Saved {OUTDIR / 'fig_cl_vs_alpha.png'}")

# ============================================================================
# Figure 2: Airfoil geometry
# ============================================================================

from tensorlbm.airfoil_benchmark import naca0012_half_thickness, build_airfoil_mask

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Airfoil shape
xi = np.linspace(0, 1, 200)
xi_t = np.tensor(xi) if hasattr(np, 'tensor') else None
import torch
xi_torch = torch.linspace(0, 1, 200)
yt = naca0012_half_thickness(xi_torch, t=0.12).numpy()

ax1.plot(xi, yt, "b-", linewidth=2, label="Upper surface")
ax1.plot(xi, -yt, "b-", linewidth=2, label="Lower surface")
ax1.fill(np.concatenate([xi, xi[::-1]]), np.concatenate([yt, -yt[::-1]]), alpha=0.3, color="cyan")
ax1.set_xlabel("x/c", fontsize=12)
ax1.set_ylabel("y/c", fontsize=12)
ax1.set_title("NACA 0012 Geometry", fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_aspect("equal")

# Right: Grid with airfoil mask
mask = build_airfoil_mask(120, 60, 30.0, alpha_deg=4.0)
ax2.imshow(mask.float().numpy(), cmap="gray_r", origin="lower", extent=[0, 120, 0, 60])
ax2.set_xlabel("x [lu]", fontsize=12)
ax2.set_ylabel("y [lu]", fontsize=12)
ax2.set_title("Cartesian Grid with Airfoil Mask (α=4°)", fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUTDIR / "fig_geometry.png", dpi=150)
plt.close()
print(f"Saved {OUTDIR / 'fig_geometry.png'}")

# ============================================================================
# Figure 3: C_D polar (with caveats)
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 6))

# LBM results (note: C_D is not physical at this resolution)
cls_lbm = [c["cl"] for c in cases]
cds_lbm = [c["cd"] for c in cases]
ax.plot(cls_lbm, cds_lbm, "ro-", label="TensorLBM (Re=200, C=16 lu)", markersize=8, linewidth=2)

# Abbott-von Doenhoff (high Re)
cl_range = np.linspace(-0.4, 1.2, 100)
cd_avd = [abbott_von_denhoff_cd(math.degrees(math.asin(max(-1, min(1, cl / 1.2)))) if abs(cl) < 1.2 else 0) 
          for cl in cl_range]
# Better: use alpha as parameter
alphas_polar = np.linspace(-4, 14, 100)
cl_avd = [abbott_von_denhoff_cl(a) for a in alphas_polar]
cd_avd = [abbott_von_denhoff_cd(a) for a in alphas_polar]
ax.plot(cl_avd, cd_avd, "bs-", label="Abbott–von Doenhoff (Re~3-6×10⁶)", markersize=4, linewidth=1.5)

ax.set_xlabel("Lift coefficient $C_L$", fontsize=12)
ax.set_ylabel("Drag coefficient $C_D$", fontsize=12)
ax.set_title("NACA 0012: Drag polar", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.5, 0.1)

# Add caveat text
textstr = (
    "⚠ LBM C_D not physical at this resolution\n"
    "(staircase boundary errors, negative values)\n"
    "Validated C_D requires C > 128 lu and Re > 1000"
)
props = dict(boxstyle="round", facecolor="lightyellow", alpha=0.9)
ax.text(0.5, 0.95, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="center", bbox=props)

plt.tight_layout()
fig.savefig(OUTDIR / "fig_cd_polar.png", dpi=150)
plt.close()
print(f"Saved {OUTDIR / 'fig_cd_polar.png'}")

print("\nAll figures generated successfully.")
