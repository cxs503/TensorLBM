#!/usr/bin/env python3
"""Generate XVAL comparison figures for the block-AMR stream correction report."""

import json
import os
import base64
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("artifacts/xval_block_amr_stream_correction")
OUT.mkdir(parents=True, exist_ok=True)

# Load ladder data
with open("artifacts/amr_reflux_ladder/ladder.json") as f:
    ladder = json.load(f)

# ---------------------------------------------------------------------------
# Figure 1: 1-D conservation comparison (reflux on vs off)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Drift comparison
series = ladder.get("series", {})
for label, color, ls in [("reflux_on", "C0", "-"), ("reflux_off", "C1", "--")]:
    if label in series:
        drift = series[label].get("drift_rel", [])
        steps = list(range(1, len(drift) + 1))
        axes[0].semilogy(steps, [max(d, 1e-18) for d in drift], ls, color=color, lw=1.5, label=label)

axes[0].axhline(1e-10, color="k", ls="--", lw=0.8, label="1e-10 criterion")
axes[0].set_xlabel("coarse step")
axes[0].set_ylabel("relative mass drift")
axes[0].legend(fontsize=8)
axes[0].set_title("1-D advection: composite mass conservation")

# Interface mismatch
for label, color, ls in [("reflux_on", "C0", "-"), ("reflux_off", "C1", "--")]:
    if label in series:
        mismatch = series[label].get("mismatch", [])
        steps = list(range(1, len(mismatch) + 1))
        axes[1].semilogy(steps, [max(abs(m), 1e-18) for m in mismatch], ls, color=color, lw=1.5, label=label)

axes[1].axhline(1e-10, color="k", ls="--", lw=0.8, label="1e-10 criterion")
axes[1].set_xlabel("coarse step")
axes[1].set_ylabel("|interface mismatch|")
axes[1].legend(fontsize=8)
axes[1].set_title("coarse/fine interface flux imbalance")

fig.tight_layout()
fig.savefig(OUT / "fig1_conservation.png", dpi=140)
plt.close(fig)
print(f"Saved {OUT / 'fig1_conservation.png'}")

# ---------------------------------------------------------------------------
# Figure 2: Approach comparison diagram
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis("off")

# TensorLBM box
tlbm_box = plt.Rectangle((0.05, 0.55), 0.4, 0.35, fill=True, facecolor="C0", alpha=0.15, edgecolor="C0", lw=2)
ax.add_patch(tlbm_box)
ax.text(0.25, 0.85, "TensorLBM AMR Reflux", ha="center", va="center", fontsize=12, fontweight="bold", color="C0")
tlbm_items = [
    "• Filippova-Hänel rescaling",
    "• Berger-Colella reflux (flux ledger)",
    "• All 9 directions (D2Q9)",
    "• Time interpolation (t^n → t^{n+1})",
    "• Explicit conservation metrics",
    "• Machine-precision mass drift (3.7e-16)",
]
for i, item in enumerate(tlbm_items):
    ax.text(0.07, 0.80 - i * 0.045, item, fontsize=9, va="top", family="monospace")

# waLBerla box
wb_box = plt.Rectangle((0.55, 0.55), 0.4, 0.35, fill=True, facecolor="C1", alpha=0.15, edgecolor="C1", lw=2)
ax.add_patch(wb_box)
ax.text(0.75, 0.85, "waLBerla EqualLevelBorderStreamCorrection", ha="center", va="center", fontsize=12, fontweight="bold", color="C1")
wb_items = [
    "• Population save/restore at borders",
    "• No explicit flux ledger",
    "• Edge/corner stencils only",
    "• Linear explosion (spatial)",
    "• Implicit conservation",
    "• MPI parallel production-ready",
]
for i, item in enumerate(wb_items):
    ax.text(0.57, 0.80 - i * 0.045, item, fontsize=9, va="top", family="monospace")

# Arrow
ax.annotate("", xy=(0.52, 0.72), xytext=(0.48, 0.72),
            arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5))
ax.text(0.5, 0.75, "same problem\ndifferent philosophy", ha="center", va="center", fontsize=8, color="gray")

# Key metrics box
metrics_box = plt.Rectangle((0.05, 0.05), 0.9, 0.4, fill=True, facecolor="white", edgecolor="gray", lw=1, alpha=0.9)
ax.add_patch(metrics_box)
ax.text(0.5, 0.42, "Key Metrics Comparison", ha="center", va="center", fontsize=11, fontweight="bold")
metrics = [
    ("1-D mass drift (reflux on)", "3.691e-16", "N/A (different problem class)"),
    ("1-D mass drift (reflux off)", "1.429e-02", "N/A (always enabled)"),
    ("2-D interface mismatch", "2.055e-05", "Not measured"),
    ("2-D Ghia max deviation", "0.5269 (52.69%)", "Not reported"),
    ("Conservation verification", "Explicit (ledger)", "Implicit (population correctness)"),
    ("Parallel support", "Single-core (benchmark)", "MPI production"),
]
for i, (aspect, tlbm, wb) in enumerate(metrics):
    y = 0.37 - i * 0.055
    ax.text(0.07, y, aspect, fontsize=8, va="top", fontweight="bold")
    ax.text(0.35, y, f"TensorLBM: {tlbm}", fontsize=7.5, va="top", color="C0", family="monospace")
    ax.text(0.60, y, f"waLBerla: {wb}", fontsize=7.5, va="top", color="C1", family="monospace")

fig.tight_layout()
fig.savefig(OUT / "fig2_approach_comparison.png", dpi=140)
plt.close(fig)
print(f"Saved {OUT / 'fig2_approach_comparison.png'}")

# ---------------------------------------------------------------------------
# Encode figures as base64 for report embedding
# ---------------------------------------------------------------------------
figures = []
for name in ["fig1_conservation.png", "fig2_approach_comparison.png"]:
    png_path = OUT / name
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    figures.append({"name": name, "png_base64": b64})

print(f"Encoded {len(figures)} figures")

# Save figure metadata
fig_meta = {
    "figures": [{"name": f["name"], "caption": {
        "fig1_conservation.png": "1-D advection: mass conservation and interface mismatch (reflux on vs off)",
        "fig2_approach_comparison.png": "Side-by-side comparison of TensorLBM and waLBerla approaches",
    }[f["name"]]} for f in figures]
}
(OUT / "figures.json").write_text(json.dumps(fig_meta, indent=1))
print("Done")
