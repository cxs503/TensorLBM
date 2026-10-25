#!/usr/bin/env python3
"""Plot figures for dam_break_sc terminal verdict (m163).

Generates:
- fig_convergence_history.png: X(T) convergence for a=40/80 baseline
- fig_error_vs_T.png: err% vs T for baseline and Guo sweep
- fig_phase_diagram.png: parameter space (G, tau) pass/fail map
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Load data
result = json.loads(Path("result.json").read_text())
ref = result["reference_digitisation_lethe"]
T_ref = np.array(ref["T"])
Z_ref = np.array(ref["Z"])

# Baseline results
a40 = result["official_rerun_correct_convention"]["a40"]
a80 = result["official_rerun_correct_convention"]["a80"]

# Checkpoints
T_check = [1.0, 2.0, 2.96]
X_check = [1.326, 2.360, 3.670]

# ============================================================
# Figure 1: Convergence history X(T)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

# Reference curve
ax.plot(T_ref, Z_ref, "k-", lw=2, label="Reference (Lethe digitisation)")

# Checkpoints
ax.plot(T_check, X_check, "ko", ms=8, label="Checkpoints (T=1,2,2.96)")

# Simulation: reconstruct from checkpoints
# a=40
X_a40 = [a40[f"X_T{t}"] if t != 2.96 else a40["X_T2p96"] for t in [1, 2, 2.96]]
ax.plot(T_check, X_a40, "r--s", ms=6, label="Sim a=40 (baseline)")

# a=80
X_a80 = [a80[f"X_T{t}"] if t != 2.96 else a80["X_T2p96"] for t in [1, 2, 2.96]]
ax.plot(T_check, X_a80, "b--^", ms=6, label="Sim a=80 (baseline)")

ax.set_xlabel("Dimensionless time $T = t\\sqrt{2g/a}$", fontsize=12)
ax.set_ylabel("Front position $X = x_{front}/a$", fontsize=12)
ax.set_title("Dam Break: Front Position vs Time", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 3.2])
ax.set_ylim([0.8, 4.0])

plt.tight_layout()
plt.savefig("figures/fig_convergence_history.png", dpi=150)
print("Saved figures/fig_convergence_history.png")
plt.close()

# ============================================================
# Figure 2: Error vs T
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

# a=40 errors
err_a40 = [a40["err_pct_T1"], a40["err_pct_T2"], a40["err_pct_T2p96"]]
ax.plot(T_check, err_a40, "r-o", ms=8, lw=2, label="a=40 (baseline)")

# a=80 errors
err_a80 = [a80["err_pct_T1"], a80["err_pct_T2"], a80["err_pct_T2p96"]]
ax.plot(T_check, err_a80, "b-s", ms=8, lw=2, label="a=80 (baseline)")

# Guo sweep (from README table - hardcoded from documented results)
# guo G=5.0 tau=0.8
err_guo_50 = [+5.66, -7.75, -17.04]
ax.plot(T_check, err_guo_50, "g--^", ms=6, label="a=80 Guo G=5.0 τ=0.8")

# 3% threshold
ax.axhline(y=3, color="k", linestyle=":", lw=1, alpha=0.5, label="3% threshold")
ax.axhline(y=-3, color="k", linestyle=":", lw=1, alpha=0.5)

ax.set_xlabel("Dimensionless time $T = t\\sqrt{2g/a}$", fontsize=12)
ax.set_ylabel("Relative error (%)", fontsize=12)
ax.set_title("Dam Break: Error vs Time", fontsize=14)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([0.8, 3.2])

plt.tight_layout()
plt.savefig("figures/fig_error_vs_T.png", dpi=150)
print("Saved figures/fig_error_vs_T.png")
plt.close()

# ============================================================
# Figure 3: Parameter space diagram
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

# Guo sweep results (from result.json)
guo_arms = [
    ("G=5.0 τ=0.8", 5.0, 0.8, err_guo_50),
    ("G=4.6 τ=0.8", 4.6, 0.8, [31.11, 16.61, 1.01]),
    ("vel_shift G=5.0 τ=0.8", 5.0, 0.8, [4.71, -7.75, -16.70]),
]

# Plot T=1 error vs G
G_vals = [5.0, 4.6, 5.0]
err_T1 = [err_guo_50[0], 31.11, 4.71]
err_T296 = [err_guo_50[2], 1.01, -16.70]

ax.plot([5.0, 4.6], [err_guo_50[0], 31.11], "r-o", ms=8, lw=2, label="T=1 err% (Guo)")
ax.plot([5.0, 4.6], [err_guo_50[2], 1.01], "b-s", ms=8, lw=2, label="T=2.96 err% (Guo)")

# 3% threshold
ax.axhline(y=3, color="k", linestyle=":", lw=1, alpha=0.5, label="3% threshold")
ax.axhline(y=-3, color="k", linestyle=":", lw=1, alpha=0.5)

ax.set_xlabel("SC coupling $G$", fontsize=12)
ax.set_ylabel("Relative error (%)", fontsize=12)
ax.set_title("Dam Break: Error vs G (showing opposite dependencies)", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([4.4, 5.2])

# Annotation
ax.annotate("T=1: lower G → worse", xy=(4.7, 20), fontsize=10, color="r")
ax.annotate("T=2.96: lower G → better", xy=(4.7, -5), fontsize=10, color="b")

plt.tight_layout()
plt.savefig("figures/fig_phase_diagram.png", dpi=150)
print("Saved figures/fig_phase_diagram.png")
plt.close()

print("\nAll figures generated successfully.")
