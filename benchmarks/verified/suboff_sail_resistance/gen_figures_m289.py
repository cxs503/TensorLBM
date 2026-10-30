#!/usr/bin/env python3
"""Generate figures for the SDAA diagnostic report."""

import json, math, base64, io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("artifacts/sdaa_suboff_sail_resistance_m289.json") as f:
    data = json.load(f)

tiers = data["tiers"]
CF_REF = data["cf_ref_blasius"]
DEVICE_LIMIT = 14.94  # GiB

# ---- Figure 1: Memory scaling + OOM boundary + extrapolation ----
fig, ax = plt.subplots(figsize=(10, 6))

completed_L = [16, 24, 32, 48, 56, 64]
completed_peak = [tiers[str(L)]["mem_peak_run_gib"] for L in completed_L]
completed_cells = [tiers[str(L)]["total_cells"] for L in completed_L]

oom_L = [66, 68, 72, 80]
oom_tried = [tiers[str(L)].get("oom_tried_gib", 0) for L in oom_L]

# Plot completed tiers
ax.plot(completed_L, completed_peak, "go-", markersize=8, linewidth=2, label="SDAA COMPLETED (peak memory)")

# Plot OOM tiers
ax.plot(oom_L, oom_tried, "rx", markersize=12, markeredgewidth=2, label="SDAA OOM (tried to allocate)")

# Device limit line
ax.axhline(y=DEVICE_LIMIT, color="r", linestyle="--", linewidth=1.5, label=f"SDAA limit ({DEVICE_LIMIT} GiB)")

# Extrapolation line
ma = data["memory_analysis"]
slope = ma["peak_memory_fit"]["slope_gib_per_cell"]
intercept = ma["peak_memory_fit"]["intercept_gib"]
all_L = list(range(16, 130, 8))
all_cells = []
for L in all_L:
    R_lb = 0.254 / 4.356 * L
    ny = int(2 * R_lb + 2 * L)
    nx = 6 * L
    all_cells.append(nx * ny * ny)
all_peak = [slope * c + intercept for c in all_cells]
ax.plot(all_L, all_peak, "b--", alpha=0.5, linewidth=1, label=f"Linear fit ({ma['peak_memory_fit']['bytes_per_cell']:.0f} B/cell, R²={ma['peak_memory_fit']['r_squared']:.4f})")

# Mark L=96 (gate-passing tier on 32 GiB)
L96_peak = ma["extrapolation"]["L96"]["peak_extrapolated_gib"]
ax.plot(96, L96_peak, "b^", markersize=12, markeredgewidth=2, label=f"L=96 extrapolated ({L96_peak:.1f} GiB)")

# Mark existing 32 GiB results
ax.annotate(f"L=96 on 32GiB GPU\nerr=1.56% (PASS 3% gate)", xy=(96, L96_peak),
            xytext=(70, 28), fontsize=8, arrowprops=dict(arrowstyle="->", color="blue"),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="blue"))

ax.set_xlabel("Resolution L (lattice units)", fontsize=12)
ax.set_ylabel("Peak Memory (GiB)", fontsize=12)
ax.set_title("SDAA Memory Scaling — suboff_sail_resistance (D3Q19 MRT, Re=1000)", fontsize=13)
ax.legend(loc="upper left", fontsize=9)
ax.set_xlim(10, 135)
ax.set_ylim(0, 35)
ax.grid(True, alpha=0.3)

# Add text box with key findings
textstr = (
    f"Device: /dev/tcaicard16 ({DEVICE_LIMIT} GiB)\n"
    f"OOM boundary: L=64→66\n"
    f"OOM cause: ~640 B/cell single alloc\n"
    f"empty_cache() does NOT help\n"
    f"L=96 needs ~{L96_peak:.0f} GiB → impossible"
)
props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=8,
        verticalalignment="bottom", horizontalalignment="right", bbox=props)

plt.tight_layout()
buf = io.BytesIO()
plt.savefig(buf, format="png", dpi=150)
plt.close()
fig1_b64 = base64.b64encode(buf.getvalue()).decode()

# ---- Figure 2: Cd vs L with reference and gate ----
fig, ax = plt.subplots(figsize=(10, 6))

# SDAA results (not converged)
sdAA_L = completed_L
sdAA_Cd = [tiers[str(L)]["cd_results"]["Cd_total"] for L in completed_L]
sdAA_steps = [tiers[str(L)]["steps_completed"] for L in completed_L]

ax.plot(sdAA_L, sdAA_Cd, "rs-", markersize=8, linewidth=2,
        label="SDAA (unconverged, 100-500 steps)")

# 32 GiB GPU results (converged)
gpu_L = [96, 112, 128]
gpu_Cd = [0.042649, 0.044485, 0.045862]
gpu_err = [1.56, 5.93, 9.21]
ax.plot(gpu_L, gpu_Cd, "bs-", markersize=8, linewidth=2,
        label="32 GiB GPU (converged, 14400-19200 steps)")

# Blasius reference
ax.axhline(y=CF_REF, color="g", linestyle="-", linewidth=2, label=f"Blasius Cf_ref = {CF_REF:.5f}")

# 3% gate bounds
gate_lo = CF_REF * 0.97
gate_hi = CF_REF * 1.03
ax.axhspan(gate_lo, gate_hi, alpha=0.15, color="green", label="3% gate")

# Annotate SDAA points with step counts
for L, Cd, steps in zip(sdAA_L, sdAA_Cd, sdAA_steps):
    ax.annotate(f"{steps}st", xy=(L, Cd), xytext=(0, 10), textcoords="offset points",
                fontsize=7, ha="center")

# Annotate GPU points with error
for L, Cd, err in zip(gpu_L, gpu_Cd, gpu_err):
    ax.annotate(f"err={err:.1f}%", xy=(L, Cd), xytext=(0, -15), textcoords="offset points",
                fontsize=8, ha="center")

ax.set_xlabel("Resolution L (lattice units)", fontsize=12)
ax.set_ylabel("Cd_total (wetted-area normalized)", fontsize=12)
ax.set_title("Cd vs Resolution — suboff_sail_resistance (Re=1000, Blasius ref)", fontsize=13)
ax.legend(loc="upper left", fontsize=9)
ax.set_xlim(10, 135)
ax.grid(True, alpha=0.3)

textstr2 = (
    "SDAA points are NOT converged\n"
    "(need 150×L steps; only 100-500 run)\n"
    "L=96 on 32GiB GPU: err=1.56% PASS\n"
    "SDAA cannot run L≥66 → cannot verify gate"
)
ax.text(0.98, 0.95, textstr2, transform=ax.transAxes, fontsize=8,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
buf = io.BytesIO()
plt.savefig(buf, format="png", dpi=150)
plt.close()
fig2_b64 = base64.b64encode(buf.getvalue()).decode()

# ---- Figure 3: Memory per cell + OOM allocation analysis ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: peak memory per cell
bpc = [tiers[str(L)]["bytes_per_cell_peak"] for L in completed_L]
ax1.bar(range(len(completed_L)), bpc, color="green", alpha=0.7)
ax1.set_xticks(range(len(completed_L)))
ax1.set_xticklabels([f"L={L}" for L in completed_L])
ax1.set_ylabel("Bytes per cell (peak)", fontsize=11)
ax1.set_title("Peak Memory per Cell (completed tiers)", fontsize=12)
ax1.axhline(y=1054, color="r", linestyle="--", label=f"Fit: {1054} B/cell")
ax1.legend()
ax1.grid(True, alpha=0.3, axis="y")

# Right: OOM analysis
oom_data = []
for L in oom_L:
    t = tiers[str(L)]
    oom_data.append({
        "L": L,
        "tried": t.get("oom_tried_gib", 0),
        "free": t.get("oom_free_gib", 0),
        "reserved": t.get("oom_reserved_gib", 0),
    })

x = np.arange(len(oom_data))
width = 0.25
ax2.bar(x - width, [d["tried"] for d in oom_data], width, label="Tried to allocate", color="red", alpha=0.7)
ax2.bar(x, [d["free"] for d in oom_data], width, label="Free at OOM", color="orange", alpha=0.7)
ax2.bar(x + width, [d["reserved"] for d in oom_data], width, label="Reserved by SDAA", color="purple", alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels([f"L={d['L']}" for d in oom_data])
ax2.set_ylabel("Memory (GiB)", fontsize=11)
ax2.set_title("OOM Analysis — SDAA Allocator Reservation", fontsize=12)
ax2.axhline(y=DEVICE_LIMIT, color="r", linestyle="--", linewidth=1, label=f"Total ({DEVICE_LIMIT} GiB)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
buf = io.BytesIO()
plt.savefig(buf, format="png", dpi=150)
plt.close()
fig3_b64 = base64.b64encode(buf.getvalue()).decode()

# Save figures to files too
for name, b64 in [("fig1_memory_scaling", fig1_b64), ("fig2_cd_vs_L", fig2_b64), ("fig3_oom_analysis", fig3_b64)]:
    with open(f"artifacts/{name}_m289.png", "wb") as f:
        f.write(base64.b64decode(b64))
    print(f"Saved artifacts/{name}_m289.png")

# Save figure data for report
fig_data = {
    "fig1_memory_scaling": fig1_b64,
    "fig2_cd_vs_L": fig2_b64,
    "fig3_oom_analysis": fig3_b64,
}
with open("/tmp/fig_data.json", "w") as f:
    json.dump(fig_data, f)
print("Figures generated successfully")
