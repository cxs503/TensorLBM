#!/usr/bin/env python3
"""Generate diagnostic figure for laplace_cac_phasefield SDAA m292."""
import json
import base64
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Load result.json
with open("benchmarks/verified/laplace_cac_phasefield/result.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: sigma_rec per arm vs target
ax = axes[0]
arms = data["arms"]
arm_names = []
sigma_recs = []
errors = []
for name in ["b1_R20", "b1_R32", "b1_R48", "b3_rho10", "b3_rho100"]:
    a = arms[name]
    arm_names.append(name)
    sigma_recs.append(a["sigma_rec_window_mean"])
    errors.append(a["err_pct"])

x = np.arange(len(arm_names))
colors = ["#2196F3", "#2196F3", "#2196F3", "#FF9800", "#FF9800"]
bars = ax.bar(x, sigma_recs, color=colors, edgecolor="white", width=0.6, label="σ_rec")
ax.axhline(y=0.01, color="red", linestyle="--", linewidth=1.5, label="σ_target = 0.01")
ax.set_xticks(x)
ax.set_xticklabels(arm_names, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("σ_rec", fontsize=11)
ax.set_title("Recovered Surface Tension per Arm", fontsize=12)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
for i, (bar, err) in enumerate(zip(bars, errors)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5e-6,
            f"{err:.3f}%", ha="center", va="bottom", fontsize=8, color="darkgreen")

# Right: error breakdown
ax2 = axes[1]
threshold = 1.5
ax2.bar(x, errors, color=colors, edgecolor="white", width=0.6)
ax2.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold}%")
ax2.set_xticks(x)
ax2.set_xticklabels(arm_names, rotation=30, ha="right", fontsize=9)
ax2.set_ylabel("|σ_rec/σ_target − 1| (%)", fontsize=11)
ax2.set_title("Per-Arm Relative Error", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)
ax2.set_ylim(0, max(errors) * 1.5)

plt.tight_layout()
buf = io.BytesIO()
plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
buf.seek(0)
png_base64 = base64.b64encode(buf.read()).decode("utf-8")
print(png_base64[:80] + "...")

# Save to file
with open("artifacts/laplace_cac_sdaa_diag.png", "wb") as f:
    f.write(buf.read())
print("Saved artifacts/laplace_cac_sdaa_diag.png")
