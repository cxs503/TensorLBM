#!/usr/bin/env python3
"""m372: Generate convergence plot and final report for suboff_re1000."""
import json, math, os, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = _REPO_ROOT / "artifacts"

# ============================================================
# Data: verified + diagnostic results
# ============================================================
cf_ref = 1.328 / math.sqrt(1000.0)  # 0.041995

# Verified results from README
verified = {
    48: {"Cd_tot": 0.040664, "err_pct": -3.17, "steps": 20000, "R_cells": 2.8,
         "status": "verified_coarse", "source": "README verified 2026-09-29"},
    80: {"Cd_tot": 0.041513, "err_pct": -1.15, "steps": 12000, "R_cells": 4.7,
         "status": "verified_pass", "source": "README verified 2026-09-29"},
}

# Diagnostic results from m372 (not fully converged, but show trend)
diagnostic = {
    16: {"Cd_tot": 0.750875, "err_pct": 1688.01, "steps": 2000, "R_cells": 0.9,
         "ms_step": 46.20, "status": "not_converged", "source": "m372 diagnostic"},
    24: {"Cd_tot": 1.909892, "err_pct": 4447.90, "steps": 500, "R_cells": 1.4,
         "ms_step": 71.55, "status": "not_converged", "source": "m372 diagnostic"},
    32: {"Cd_tot": 1.961200, "err_pct": 4570.07, "steps": 300, "R_cells": 1.9,
         "ms_step": 110.82, "status": "not_converged", "source": "m372 diagnostic"},
    48: {"Cd_tot": 2.751941, "err_pct": 6453.01, "steps": 200, "R_cells": 2.8,
         "ms_step": 268.03, "status": "not_converged", "source": "m372 diagnostic"},
}

# m319 mission result
m319 = {"L": 24, "Cd_tot": 0.061495, "err_pct": 46.43, "steps": "~200",
        "status": "m319_partial_fail", "source": "mission m319"}

# ============================================================
# Figure 1: Cd(L) convergence ladder
# ============================================================
L_vals = sorted(set(list(verified.keys()) + list(diagnostic.keys())))

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot verified results (solid markers)
L_ver = sorted(verified.keys())
Cd_ver = [verified[L]["Cd_tot"] for L in L_ver]
ax1.plot(L_ver, Cd_ver, "go-", markersize=10, linewidth=2, label="Verified (converged)")

# Plot diagnostic results (open markers, not converged)
L_diag = sorted(diagnostic.keys())
Cd_diag = [diagnostic[L]["Cd_tot"] for L in L_diag]
ax1.plot(L_diag, Cd_diag, "r^--", markersize=8, linewidth=1.5, label="Diagnostic (not converged)")

# m319 point
ax1.scatter([m319["L"]], [m319["Cd_tot"]], c="orange", s=150, marker="*", zorder=5,
            label=f"m319 (Cd={m319['Cd_tot']:.5f})")

# Reference line
ax1.axhline(y=cf_ref, color="blue", linestyle="-.", linewidth=1.5, alpha=0.7,
            label=f"Blasius Cf_ref = {cf_ref:.6f}")

# ±3% band
ax1.axhspan(cf_ref * 0.97, cf_ref * 1.03, alpha=0.1, color="green", label="±3% acceptance band")

ax1.set_xlabel("Resolution L (cells per hull length)", fontsize=12)
ax1.set_ylabel("Cd_total (wetted-area normalized)", fontsize=12)
ax1.set_title("SUBOFF Re=1000: Cd(L) Convergence Ladder\n(m372 diagnostic + verified results)", fontsize=13)
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(L_vals)
ax1.set_xticklabels([f"L={L}" for L in L_vals])

# Add annotations
for L in L_ver:
    r = verified[L]
    ax1.annotate(f"{r['Cd_tot']:.4f}\n({r['err_pct']:+.1f}%)",
                 (L, r["Cd_tot"]), textcoords="offset points",
                 xytext=(0, 12), ha="center", fontsize=8, color="green")

for L in L_diag:
    r = diagnostic[L]
    ax1.annotate(f"{r['Cd_tot']:.2f}\n({r['err_pct']:+.0f}%)",
                 (L, r["Cd_tot"]), textcoords="offset points",
                 xytext=(0, -15), ha="center", fontsize=7, color="red")

plt.tight_layout()
fig_path = ARTIFACTS / "m372_cd_convergence.png"
fig.savefig(fig_path, dpi=150)
plt.close()
print(f"Figure saved: {fig_path}")

# ============================================================
# Figure 2: Geometry resolution vs error
# ============================================================
fig2, ax2 = plt.subplots(figsize=(10, 5))

L_all = L_vals
R_cells_all = []
err_all = []
colors_all = []
for L in L_all:
    if L in verified:
        R_cells_all.append(verified[L]["R_cells"])
        err_all.append(verified[L]["err_pct"])
        colors_all.append("green")
    elif L in diagnostic:
        R_cells_all.append(diagnostic[L]["R_cells"])
        err_all.append(diagnostic[L]["err_pct"])
        colors_all.append("red")

# Use log scale for error
ax2.bar([f"L={L}" for L in L_all], err_all, color=colors_all, alpha=0.7)
ax2.axhline(y=3, color="green", linestyle="--", alpha=0.5, label="+3% threshold")
ax2.axhline(y=-3, color="green", linestyle="--", alpha=0.5, label="-3% threshold")
ax2.axhline(y=0, color="black", linewidth=0.5)
ax2.set_xlabel("Resolution L", fontsize=11)
ax2.set_ylabel("Error vs Blasius (%)", fontsize=11)
ax2.set_title("SUBOFF Re=1000: Error vs Resolution\n(log scale, diagnostic runs not converged)", fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")
ax2.set_yscale("symlog", linthresh=1)

# Add R_cells annotation
for i, L in enumerate(L_all):
    if L in verified:
        rc = verified[L]["R_cells"]
    else:
        rc = diagnostic[L]["R_cells"]
    ax2.annotate(f"R={rc:.1f}", (i, err_all[i]), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=8)

plt.tight_layout()
fig2_path = ARTIFACTS / "m372_error_vs_resolution.png"
fig2.savefig(fig2_path, dpi=150)
plt.close()
print(f"Figure saved: {fig2_path}")

# ============================================================
# Save comprehensive JSON artifact
# ============================================================
import base64

def png_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

artifact = {
    "mission": "m372",
    "case": "suboff_re1000_resolution_sweep",
    "device": "sdaa:0",
    "torch_version": "2.7.1",
    "torch_sdaa_version": "3.1.1a3+gite3971b9",
    "tcaicard_devices": 32,
    "cf_ref": cf_ref,
    "cf_ref_formula": "1.328/sqrt(Re)",
    "Re": 1000,
    "normalization": "wetted area S=pi*D*L",
    "geometry": {
        "suboff_length_m": 4.356,
        "suboff_radius_m": 0.254,
        "L_D_ratio": 8.57,
    },
    "verified_results": {
        str(L): {**v, "L": L} for L, v in verified.items()
    },
    "diagnostic_results": {
        str(L): {**d, "L": L} for L, d in diagnostic.items()
    },
    "m319_reference": m319,
    "geometry_analysis": {
        f"L={L}": {
            "R_cells": verified.get(L, diagnostic.get(L, {})).get("R_cells", "N/A"),
            "verdict": "PASS" if L == 80 else ("coarse" if L == 48 else "SEVERELY under-resolved"),
        }
        for L in L_all
    },
    "compile_routing": {
        "mode_requested": "default",
        "mode_effective_on_sdaa": "compiled (simple step) / eager_fallback (full LBM chain)",
        "note": "teco_inductor (Torch-SDAA 3.1.1a3) cannot compile full LBM step chain; "
                "_CompileWithEagerFallback catches InductorError and falls back to eager. "
                "This is expected and documented in compile_route.py.",
    },
    "conclusion": {
        "verdict": "HONEST_FAIL",
        "reason": "L<=32 cannot achieve <3% accuracy due to severe geometry under-resolution "
                  "(hull radius < 2 cells). L=48 achieves -3.17% (slightly over threshold). "
                  "L=80 achieves -1.15% (PASS). The +46% error from m319 (Cd=0.061495) was "
                  "from an under-resolved (L=24) and under-stepped (~200 steps) run that "
                  "had not converged.",
        "minimum_L_for_3pct": 80,
        "estimated_time_L80_sdaa": "3.4 hours (12000 steps, 13.7M cells)",
    },
    "figures": [
        {
            "name": "cd_convergence",
            "caption": "Cd(L) convergence ladder: verified results (green) vs diagnostic (red). "
                       "m319 result (orange star) at L=24. Blasius reference (blue dashed). "
                       "Green band = ±3% acceptance.",
            "png_base64": png_to_base64(fig_path),
        },
        {
            "name": "error_vs_resolution",
            "caption": "Error vs resolution (symlog scale). R_cells annotation shows hull radius "
                       "in grid cells. L<=32 has R<2 cells, causing severe geometry error.",
            "png_base64": png_to_base64(fig2_path),
        },
    ],
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}

out_path = ARTIFACTS / "m372_suboff_re1000_diagnostic.json"
out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
print(f"\nArtifact saved: {out_path}")
print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")
