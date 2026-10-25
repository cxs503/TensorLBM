#!/usr/bin/env python3
"""Plot energy decay history E(t) for collision kernels benchmark.

Reads energy_history_*.csv files directly (zero hand-copying).
Shows E(t)/E0 decay + LSQ fit for KBC (PASS) vs cumulant (D3Q27, FAIL).
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent

# Select representative kernels: KBC (PASS) and cumulant (FAIL representative)
# Pick N=96 (middle tier) as the showcase
kernels_showcase = {
    "kbc_d3q19": {"label": "KBC D3Q19 [PASS]", "color": "#d62728", "N": 96},
    "cumulant_d3q27": {"label": "cumulant D3Q27 [FAIL]", "color": "#1f77b4", "N": 96},
    "mrt27": {"label": "mrt D3Q27 [FAIL]", "color": "#2ca02c", "N": 96},
    "trt27": {"label": "trt D3Q27 [FAIL]", "color": "#ff7f0e", "N": 96},
}

# Read result.json for gamma_theory and gamma_sim
import json

with open(HERE / "result.json") as f:
    rdata = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: E(t)/E0 decay (log scale)
ax = axes[0]
for kname, info in kernels_showcase.items():
    csv_path = HERE / f"energy_history_{kname}_N{info['N']}.csv"
    steps, energies = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(float(row["step"]))
            energies.append(float(row["energy"]))
    steps = np.array(steps)
    energies = np.array(energies)
    E0 = energies[0]

    ax.semilogy(steps, energies / E0, color=info["color"], linewidth=2, label=info["label"])

    # Overlay LSQ fit line
    kernel_data = rdata["kernels"][kname]
    case = [c for c in kernel_data["cases"] if c["n"] == info["N"]][0]
    gamma_sim = case["gamma_e_sim"]
    fit_line = np.exp(-gamma_sim * steps)
    ax.semilogy(steps, fit_line, color=info["color"], linestyle="--", alpha=0.5, linewidth=1)

ax.set_xlabel("Time step t", fontsize=11)
ax.set_ylabel("E(t) / E₀", fontsize=11)
ax.set_title("Kinetic Energy Decay (N=96, Re=24)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3, linestyle=":")
ax.set_ylim(1e-5, 2)

# Right panel: -d(ln E)/dt instantaneous decay rate
ax2 = axes[1]
for kname, info in kernels_showcase.items():
    csv_path = HERE / f"energy_history_{kname}_N{info['N']}.csv"
    steps, energies = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(float(row["step"]))
            energies.append(float(row["energy"]))
    steps = np.array(steps)
    energies = np.array(energies)

    # Compute instantaneous -d(ln E)/dt via central differences
    lnE = np.log(energies)
    gamma_inst = -np.gradient(lnE, steps)

    # Skip first point (noisy)
    ax2.plot(
        steps[1:],
        gamma_inst[1:],
        color=info["color"],
        linewidth=1.5,
        alpha=0.8,
        label=info["label"],
    )

    # Horizontal line: gamma_sim (LSQ fit)
    kernel_data = rdata["kernels"][kname]
    case = [c for c in kernel_data["cases"] if c["n"] == info["N"]][0]
    gamma_sim = case["gamma_e_sim"]
    gamma_theory = case["gamma_e_theory"]
    ax2.axhline(gamma_sim, color=info["color"], linestyle="--", alpha=0.4, linewidth=1)

# Add theory annotation
case_kbc = [c for c in rdata["kernels"]["kbc_d3q19"]["cases"] if c["n"] == 96][0]
ax2.text(
    0.98,
    0.95,
    f"γ_theory = {case_kbc['gamma_e_theory']:.5f}\n" + "(analytical 6νk²)",
    transform=ax2.transAxes,
    fontsize=9,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

ax2.set_xlabel("Time step t", fontsize=11)
ax2.set_ylabel("Instantaneous γ_E = -d(ln E)/dt", fontsize=11)
ax2.set_title("Instantaneous Decay Rate (N=96)", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9, loc="lower right")
ax2.grid(True, alpha=0.3, linestyle=":")

plt.tight_layout()

output_path = HERE / "figures" / "curve_energy_decay_history.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved: {output_path} ({output_path.stat().st_size} bytes)")
plt.close()
