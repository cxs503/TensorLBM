#!/usr/bin/env python3
"""Generate figures for XVAL-LBM-VOXEL-IMGPROC-POROSITY-REPORT."""

from __future__ import annotations

import json
import os
import base64
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "results.json")) as f:
    R = json.load(f)

figures = []

# ---- Figure 1: A1 Random 2D porosity accuracy ----
fig, ax = plt.subplots(figsize=(7, 4.5))
cases = R["A1_random_2d"]["cases"]
# Group by phi_target (sigma=0 cases only)
phi_targets = sorted(set(c["phi_target"] for c in cases if "sigma" not in c))
for phi_t in phi_targets:
    errs = [c["rel_error"] * 100 for c in cases if c["phi_target"] == phi_t and "sigma" not in c]
    ax.bar([f"φ={phi_t}"], [np.mean(errs)], yerr=[np.std(errs)] if len(errs) > 1 else None,
           capsize=5, alpha=0.7, label=f"φ_target={phi_t}")
ax.axhline(y=15, color='r', linestyle='--', linewidth=1.5, label='15% threshold')
ax.set_ylabel("Relative error (%)")
ax.set_title("A1: Random 2-D mask porosity accuracy")
ax.legend(fontsize=8)
ax.set_yscale('log')
ax.set_ylim(1e-4, 20)
fig.tight_layout()
buf = BytesIO()
fig.savefig(buf, format='png', dpi=150)
plt.close(fig)
figures.append({"name": "fig_a1_random_2d", "caption": "Random 2-D mask porosity accuracy: relative error vs target porosity. All cases well below 15% threshold.", "png_base64": base64.b64encode(buf.getvalue()).decode()})

# ---- Figure 2: A3 Analytical geometry porosity ----
fig, ax = plt.subplots(figsize=(7, 4.5))
cases_a3 = R["A3_analytical"]["cases"]
labels = [c["geometry"] for c in cases_a3]
phi_ref = [c["phi_analytical"] for c in cases_a3]
phi_got = [c["phi_computed"] for c in cases_a3]
x_pos = np.arange(len(labels))
width = 0.35
ax.bar(x_pos - width/2, phi_ref, width, label='Analytical', alpha=0.8, color='steelblue')
ax.bar(x_pos + width/2, phi_got, width, label='Computed', alpha=0.8, color='coral')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
ax.set_ylabel("Porosity φ")
ax.set_title("A3: Analytical geometry porosity — computed vs analytical")
ax.legend()
fig.tight_layout()
buf = BytesIO()
fig.savefig(buf, format='png', dpi=150)
plt.close(fig)
figures.append({"name": "fig_a3_analytical", "caption": "Analytical geometry porosity: computed (voxelized) vs analytical values for sphere, cylinder, and ellipsoid. Errors are below 0.05%.", "png_base64": base64.b64encode(buf.getvalue()).decode()})

# ---- Figure 3: A4 Convergence ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Sphere convergence
cases_sph = R["A4_convergence"]["cases"]
hs = [c["h"] for c in cases_sph]
errs_sph = [c["rel_error"] for c in cases_sph]
axes[0].loglog(hs, errs_sph, 'o-', label='Sphere data')
conv_sph = R["A4_convergence"]["convergence"]
p_sph = conv_sph["order_p"]
hs_fit = np.logspace(np.log10(min(hs)), np.log10(max(hs)), 50)
errs_fit = hs_fit ** p_sph * np.exp(np.polyfit(np.log(hs), np.log(errs_sph), 1)[1])
axes[0].loglog(hs_fit, errs_fit, '--', label=f'fit p={p_sph:.2f}')
axes[0].axhline(y=0.15, color='r', linestyle=':', label='15% threshold')
axes[0].set_xlabel("Cell size h = 1/n")
axes[0].set_ylabel("Relative error")
axes[0].set_title("A4: Sphere porosity convergence")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Cylinder convergence
cases_cyl = R["A4_convergence"]["cylinder_cases"]
errs_cyl = [c["rel_error"] for c in cases_cyl]
axes[1].loglog(hs, errs_cyl, 's-', label='Cylinder data')
conv_cyl = R["A4_convergence"]["convergence_cylinder"]
p_cyl = conv_cyl["order_p"]
errs_fit_cyl = hs_fit ** p_cyl * np.exp(np.polyfit(np.log(hs), np.log(errs_cyl), 1)[1])
axes[1].loglog(hs_fit, errs_fit_cyl, '--', label=f'fit p={p_cyl:.2f}')
axes[1].axhline(y=0.15, color='r', linestyle=':', label='15% threshold')
axes[1].set_xlabel("Cell size h = 1/n")
axes[1].set_ylabel("Relative error")
axes[1].set_title("A4: Cylinder porosity convergence")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
buf = BytesIO()
fig.savefig(buf, format='png', dpi=150)
plt.close(fig)
figures.append({"name": "fig_a4_convergence", "caption": "Porosity convergence: sphere (p≈1.84) and cylinder (p≈1.63) both show >1st order convergence. Errors well below 15% threshold at all resolutions.", "png_base64": base64.b64encode(buf.getvalue()).decode()})

# ---- Figure 4: A5 Sigma sensitivity ----
fig, ax = plt.subplots(figsize=(7, 4.5))
cases_a5 = R["A5_sigma_sensitivity"]["cases"]
sigmas = sorted(set(c["sigma"] for c in cases_a5))
mean_errs = []
max_errs = []
for sig in sigmas:
    errs = [c["rel_error"] * 100 for c in cases_a5 if c["sigma"] == sig]
    mean_errs.append(np.mean(errs))
    max_errs.append(np.max(errs))
ax.bar(range(len(sigmas)), mean_errs, alpha=0.7, label='Mean error')
ax.plot(range(len(sigmas)), max_errs, 'ro-', markersize=4, label='Max error')
ax.axhline(y=15, color='r', linestyle='--', linewidth=1.5, label='15% threshold')
ax.set_xticks(range(len(sigmas)))
ax.set_xticklabels([f"σ={s}" for s in sigmas], fontsize=8)
ax.set_ylabel("Relative error (%)")
ax.set_title("A5: Sigma sensitivity — porosity accuracy vs correlation length")
ax.set_yscale('log')
ax.set_ylim(1e-5, 20)
ax.legend(fontsize=8)
fig.tight_layout()
buf = BytesIO()
fig.savefig(buf, format='png', dpi=150)
plt.close(fig)
figures.append({"name": "fig_a5_sigma", "caption": "Sigma sensitivity: porosity accuracy remains excellent across all Gaussian correlation lengths. The quantile-based thresholding ensures exact porosity regardless of sigma.", "png_base64": base64.b64encode(buf.getvalue()).decode()})

# Save figures data
with open(os.path.join(HERE, "figures.json"), "w") as f:
    json.dump(figures, f)

print(f"Generated {len(figures)} figures")
for fig in figures:
    print(f"  {fig['name']}: {len(fig['png_base64'])} bytes base64")
