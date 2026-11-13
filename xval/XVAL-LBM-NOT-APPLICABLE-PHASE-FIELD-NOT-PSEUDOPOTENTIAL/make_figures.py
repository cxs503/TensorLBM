#!/usr/bin/env python3
"""Generate comparison figure for XVAL: Phase-field vs Pseudopotential."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
import os

out_dir = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Figure 1: Structural comparison diagram
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: SCMP schematic
ax = axes[0]
ax.set_title("Shan-Chen Pseudopotential (SCMP)\nTensorLBM multiphase", fontsize=12, fontweight="bold")

# Draw single distribution
ax.text(0.5, 0.85, "f (single distribution)", ha="center", va="center",
        fontsize=14, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

# Arrow to collision
ax.annotate("", xy=(0.5, 0.65), xytext=(0.5, 0.75),
            arrowprops=dict(arrowstyle="->", lw=2, color="navy"))
ax.text(0.5, 0.70, "BGK + Guo forcing", ha="center", va="center", fontsize=10, color="navy")

# Arrow to streaming
ax.annotate("", xy=(0.5, 0.45), xytext=(0.5, 0.55),
            arrowprops=dict(arrowstyle="->", lw=2, color="navy"))
ax.text(0.5, 0.50, "stream + bounce-back", ha="center", va="center", fontsize=10, color="navy")

# Force box
ax.text(0.5, 0.30, "F = G·cs²·ψ·Σwᵢψ(x+cᵢ)cᵢ\nψ(ρ) = ρ₀(1−e⁻ᵖ/ᵖ⁰)", ha="center", va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

ax.annotate("", xy=(0.5, 0.38), xytext=(0.5, 0.42),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="orange", linestyle="--"))

# Output
ax.text(0.5, 0.15, "ρ, u (macroscopic)\nσ emergent from G", ha="center", va="center",
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
ax.annotate("", xy=(0.5, 0.22), xytext=(0.5, 0.28),
            arrowprops=dict(arrowstyle="->", lw=2, color="navy"))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Right: Phase-field schematic
ax = axes[1]
ax.set_title("Fakhari Phase-Field (CudaPhaseField)\nAminZar69/CudaPhaseField", fontsize=12, fontweight="bold")

# Two distributions
ax.text(0.3, 0.85, "h (phase field C)", ha="center", va="center",
        fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
ax.text(0.7, 0.85, "g (hydrodynamics)", ha="center", va="center",
        fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightsalmon", alpha=0.8))

# Collision boxes
ax.text(0.3, 0.65, "AC collision\nw_c = 1/τ_h\n+ anti-diffusion eF", ha="center", va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
ax.text(0.7, 0.65, "NS collision\ns₈ = 1/τ_g\n+ CSF force", ha="center", va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

ax.annotate("", xy=(0.3, 0.75), xytext=(0.3, 0.78),
            arrowprops=dict(arrowstyle="->", lw=2, color="darkred"))
ax.annotate("", xy=(0.7, 0.75), xytext=(0.7, 0.78),
            arrowprops=dict(arrowstyle="->", lw=2, color="darkred"))

# Chemical potential
ax.text(0.5, 0.48, "μ = 4βC(C−1)(C−0.5) − κ∇²C\nF = μ∇C + Δp·∇C", ha="center", va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

ax.annotate("", xy=(0.3, 0.55), xytext=(0.45, 0.52),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="orange", linestyle="--"))
ax.annotate("", xy=(0.7, 0.55), xytext=(0.55, 0.52),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="orange", linestyle="--"))

# Streaming
ax.text(0.3, 0.35, "stream h", ha="center", va="center", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcoral", alpha=0.6))
ax.text(0.7, 0.35, "stream g", ha="center", va="center", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightsalmon", alpha=0.6))

ax.annotate("", xy=(0.3, 0.42), xytext=(0.3, 0.55),
            arrowprops=dict(arrowstyle="->", lw=2, color="darkred"))
ax.annotate("", xy=(0.7, 0.42), xytext=(0.7, 0.55),
            arrowprops=dict(arrowstyle="->", lw=2, color="darkred"))

# Output
ax.text(0.5, 0.15, "C (phase field), u (velocity)\nσ explicit input", ha="center", va="center",
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
ax.annotate("", xy=(0.5, 0.22), xytext=(0.5, 0.28),
            arrowprops=dict(arrowstyle="->", lw=2, color="darkred"))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

plt.suptitle("Structural Comparison: Pseudopotential vs Phase-Field LBM\n(NOT APPLICABLE for direct cross-validation)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig_path = os.path.join(out_dir, "figures", "structural_comparison.png")
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {fig_path}")

# ---------------------------------------------------------------------------
# Figure 2: Parameter space comparison
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

categories = ["Distributions", "Surface Tension", "Density Ratio",
              "Gradient Order", "Anti-diffusion", "Force Type"]
scmp_vals = [1, 0, 0, 1, 0, 0]  # binary-ish encoding
pf_vals = [2, 1, 1, 4, 1, 1]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, scmp_vals, width, label="SCMP (Pseudopotential)", color="lightblue", edgecolor="navy")
bars2 = ax.bar(x + width/2, pf_vals, width, label="Phase-Field (Fakhari)", color="lightcoral", edgecolor="darkred")

ax.set_ylabel("Value / Count")
ax.set_title("Key Parameter Differences", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=15, ha="right")
ax.legend()

# Add annotations
for i, (s, p) in enumerate(zip(scmp_vals, pf_vals)):
    if s != p:
        ax.annotate("≠", xy=(i, max(s, p) + 0.1), ha="center", fontsize=14, fontweight="bold", color="red")

plt.tight_layout()
fig_path2 = os.path.join(out_dir, "figures", "parameter_comparison.png")
fig.savefig(fig_path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {fig_path2}")

# ---------------------------------------------------------------------------
# Figure 3: Verified SCMP Laplace data (from repo)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))

# Data from benchmarks/verified/laplace_cac_phasefield/result.json
radii = [20.0, 32.0, 48.0]
sigma_rec = [0.009994, 0.010000, 0.010044]
sigma_target = 0.01

ax.plot(radii, [sigma_target]*3, "k--", label="σ_target = 0.01", linewidth=2)
ax.scatter(radii, sigma_rec, color="navy", s=80, zorder=5, label="SCMP recovered σ")
for r, s in zip(radii, sigma_rec):
    err = abs(s - sigma_target) / sigma_target * 100
    ax.annotate(f"{err:.2f}%", xy=(r, s), xytext=(r, s + 0.0003),
                ha="center", fontsize=9, color="navy")

ax.set_xlabel("Droplet Radius R (lattice units)")
ax.set_ylabel("Recovered Surface Tension σ")
ax.set_title("SCMP Laplace Test: σ Recovery vs Radius\n(TensorLBM verified data)", fontsize=11, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path3 = os.path.join(out_dir, "figures", "scmp_laplace_verified.png")
fig.savefig(fig_path3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {fig_path3}")

print("All figures generated.")
