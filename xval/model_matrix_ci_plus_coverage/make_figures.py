#!/usr/bin/env python3
"""Generate figures for XVAL model-matrix-ci-plus-coverage report."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ART = Path(__file__).resolve().parent / "artifacts"
FIGS = Path(__file__).resolve().parent / "figures"
FIGS.mkdir(exist_ok=True)

# Load results
result = json.loads((ART / "result.json").read_text())

# Figure 1: Model Matrix Coverage Comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

comp = result["tclb_comparison"]["comparison"]

# Lattice coverage
labels = ["D2Q9", "D3Q19", "D3Q27"]
tclb_lat = [1, 1, 1]
tlbm_lat = [1, 1, 0]
x = np.arange(len(labels))
width = 0.35
axes[0].bar(x - width/2, tclb_lat, width, label="TCLB", color="#2196F3")
axes[0].bar(x + width/2, tlbm_lat, width, label="TensorLBM", color="#FF9800")
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels)
axes[0].set_ylabel("Available")
axes[0].set_title("Lattice Types")
axes[0].legend()
axes[0].set_ylim(0, 1.2)

# Collision coverage
labels = ["BGK", "MRT", "Cascaded", "Entropic", "RLBM", "Smag-MRT"]
tclb_col = [1, 1, 1, 1, 0, 0]
tlbm_col = [1, 1, 0, 0, 1, 1]
x = np.arange(len(labels))
axes[1].bar(x - width/2, tclb_col, width, label="TCLB", color="#2196F3")
axes[1].bar(x + width/2, tlbm_col, width, label="TensorLBM", color="#FF9800")
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, rotation=45, ha="right")
axes[1].set_ylabel("Available")
axes[1].set_title("Collision Operators")
axes[1].legend()
axes[1].set_ylim(0, 1.2)

# Physics coverage
labels = ["single_phase", "multiphase_pf", "multiphase_sc", "thermal", "turbulence_les", "acoustic", "adjoint"]
tclb_phy = [1, 1, 1, 1, 1, 0, 0]
tlbm_phy = [1, 1, 1, 0, 1, 1, 1]
x = np.arange(len(labels))
axes[2].bar(x - width/2, tclb_phy, width, label="TCLB", color="#2196F3")
axes[2].bar(x + width/2, tlbm_phy, width, label="TensorLBM", color="#FF9800")
axes[2].set_xticks(x)
axes[2].set_xticklabels(labels, rotation=45, ha="right")
axes[2].set_ylabel("Available")
axes[2].set_title("Physics Models")
axes[2].legend()
axes[2].set_ylim(0, 1.2)

plt.tight_layout()
fig.savefig(FIGS / "fig1_model_matrix_comparison.png", dpi=150)
plt.close()

# Figure 2: Coverage Radar
fig, ax = plt.subplots(subplot_kw=dict(projection="polar"), figsize=(8, 8))

categories = ["Lattice", "Collision", "Physics", "Boundary", "CI Practice"]
tclb_vals = [1.0, 0.75, 0.8, 1.0, 1.0]
tlbm_vals = [0.67, 0.5, 0.83, 0.83, 0.5]

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
tclb_vals += tclb_vals[:1]
tlbm_vals += tlbm_vals[:1]
angles += angles[:1]

ax.plot(angles, tclb_vals, "o-", linewidth=2, label="TCLB", color="#2196F3")
ax.fill(angles, tclb_vals, alpha=0.15, color="#2196F3")
ax.plot(angles, tlbm_vals, "o-", linewidth=2, label="TensorLBM", color="#FF9800")
ax.fill(angles, tlbm_vals, alpha=0.15, color="#FF9800")
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0, 1.1)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
ax.set_title("Model Matrix Coverage Comparison", pad=20)

plt.tight_layout()
fig.savefig(FIGS / "fig2_coverage_radar.png", dpi=150)
plt.close()

# Figure 3: Test Results Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))

test_names = [r["name"] for r in result["results"]]
errors = [r.get("relative_error_pct", r.get("u_error_pct", 0)) for r in result["results"]]
passed = [r["pass"] for r in result["results"]]

colors = ["#4CAF50" if p else "#F44336" for p in passed]
bars = ax.bar(test_names, errors, color=colors, edgecolor="black", linewidth=0.5)
ax.axhline(y=15, color="red", linestyle="--", linewidth=2, label="15% criterion")
ax.set_ylabel("Relative Error (%)")
ax.set_title("Test Results vs 15% Criterion")
ax.legend()
ax.set_ylim(0, max(errors) * 1.2 if errors else 20)

for bar, err in zip(bars, errors):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{err:.1f}%", ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
fig.savefig(FIGS / "fig3_test_results.png", dpi=150)
plt.close()

print("Figures saved to", FIGS)
