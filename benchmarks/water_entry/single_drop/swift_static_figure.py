"""Generate the PF5-SWFE report figure: Swift model stability vs density ratio."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

# Load the static droplet results
results = json.loads(
    (ROOT / "benchmarks" / "water_entry" / "single_drop" / "swift_static_results.json").read_text()
)

ratios = [r["ratio"] for r in results]
stable = [r["stable"] for r in results]
nan_steps = [r["nan_step"] for r in results]
umax = [r["umax"] for r in results]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel 1: stability
ax = axes[0]
colors = ["#2ca02c" if s else "#d62728" for s in stable]
ax.bar(range(len(ratios)), [1 if s else 0 for s in stable], color=colors, alpha=0.7, width=0.6)
ax.set_xticks(range(len(ratios)))
ax.set_xticklabels([f"{r:.0f}" for r in ratios])
ax.set_xlabel("Density ratio $\\rho_l / \\rho_g$")
ax.set_ylabel("Stable (300 steps)")
ax.set_title("Swift free-energy model: static-droplet stability")
ax.set_ylim(-0.1, 1.1)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5)
for i, (s, ns) in enumerate(zip(stable, nan_steps)):
    if not s:
        ax.text(i, 0.05, f"NaN@{ns}", ha="center", fontsize=9, color="darkred")
    else:
        ax.text(i, 0.05, "OK", ha="center", fontsize=9, color="darkgreen")

# Panel 2: umax (log scale)
ax = axes[1]
ax.semilogy(ratios, umax, "o-", color="#d62728", markersize=8)
ax.set_xlabel("Density ratio $\\rho_l / \\rho_g$")
ax.set_ylabel("$u_{max}$ (lattice units, log)")
ax.set_title("Peak spurious current (diverges on instability)")
ax.set_xscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.set_xlim(8, 1200)

plt.tight_layout()
out = ROOT / "artifacts" / "pf5_swfe_stability.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
