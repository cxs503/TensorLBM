#!/usr/bin/env python
"""Generate figures for m410 Blasius report."""
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load baseline data
with open("benchmarks/pending/blasius_flat_plate/results/R6p_plate400_y1600_trt.json") as f:
    r6p = json.load(f)

# Blasius reference
CF_COEFF = 0.6641146724303926

# Figure 1: Cf error vs x_eff
stations = r6p["stations"]
x_eff = [s["x_eff"] for s in stations]
cf_err = [s["Cf_fd_err_pct"] for s in stations]
ue_local = [s["u_edge_local_over_U"] for s in stations]

fig, ax1 = plt.subplots(figsize=(10, 6))
color1 = 'tab:red'
ax1.set_xlabel('x_eff (cells)')
ax1.set_ylabel('Cf error (%)', color=color1)
ax1.plot(x_eff, cf_err, 'ro-', label='Cf error (%)', linewidth=2)
ax1.axhline(y=3, color='green', linestyle='--', alpha=0.7, label='3% threshold')
ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_title('Blasius Flat Plate: Cf Error & Edge Velocity vs x_eff\n(plate400_y1600, TRT λ=3/16, 30k+200 steps)')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.set_ylabel('u_edge_local / U', color=color2)
ax2.plot(x_eff, ue_local, 'bs-', label='u_edge/U', linewidth=2)
ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.7)
ax2.axhline(y=1.02, color='orange', linestyle=':', alpha=0.5, label='±2% band')
ax2.axhline(y=0.98, color='orange', linestyle=':', alpha=0.5)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.legend(loc='upper right')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('benchmarks/pending/blasius_flat_plate/fig1_cf_error_vs_xeff.png', dpi=150)
plt.close()

# Figure 2: Velocity profile comparison at x_eff=180
etas = np.array([s["eta"] for s in r6p["eta_tab_comparison"]])
u_sim = np.array([s["u_sim_over_U"] for s in r6p["eta_tab_comparison"]])
u_ref = np.array([s["fprime_ref"] for s in r6p["eta_tab_comparison"]])

# Full profile from subsampled data
prof_data = r6p["profile_subsampled"]
eta_full = np.array([p["eta"] for p in prof_data])
u_sim_full = np.array([p["u_sim"] for p in prof_data]) / r6p["U"]
u_ref_full = np.array([p["u_blasius"] for p in prof_data]) / r6p["U"]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(eta_full, u_sim_full, 'b-', label='LBM simulation', linewidth=2)
ax.plot(eta_full, u_ref_full, 'r--', label='Blasius reference', linewidth=2)
ax.plot(etas, u_sim, 'bo', markersize=8, label='η-tab points')
ax.plot(etas, u_ref, 'rs', markersize=8)
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('η = y / √(νx/U)')
ax.set_ylabel('u / U')
ax.set_title(f'Blasius Velocity Profile at x_eff=180 (Re_x=900)\nplate400_y1600, TRT λ=3/16')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('benchmarks/pending/blasius_flat_plate/fig2_velocity_profile.png', dpi=150)
plt.close()

# Figure 3: Convergence trend (from validation_report.json)
with open("benchmarks/pending/blasius_flat_plate/validation_report.json") as f:
    vr = json.load(f)

grids = vr["grids_tested"]
plate400_err = 27.38

re_vals = [900, 2000, 4000]
err_vals = [plate400_err, grids[0]["cf_err_pct"], grids[1]["cf_err_pct"]]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(re_vals, err_vals, 'ro-', linewidth=2, markersize=10)
ax.axhline(y=3, color='green', linestyle='--', alpha=0.7, label='3% threshold')
ax.set_xlabel('Re_x')
ax.set_ylabel('Cf error (%)')
ax.set_title('Cf Error vs Re_x: Anti-convergence Trend\n(Mid-plate, mirror BCs, TRT)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
plt.tight_layout()
plt.savefig('benchmarks/pending/blasius_flat_plate/fig3_error_vs_rex.png', dpi=150)
plt.close()

# Figure 4: δ*/H ratio analysis
x_eff_vals = [30, 60, 90, 120, 150, 180, 240, 300, 340, 380]
U, nu = 0.05, 0.01
delta_star_vals = [1.7208 * xe / math.sqrt(U * xe / nu) for xe in x_eff_vals]
H = 800
delta_star_over_H = [ds / H for ds in delta_star_vals]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x_eff_vals, delta_star_over_H, 'go-', linewidth=2, markersize=8)
ax.axhline(y=0.01, color='orange', linestyle='--', alpha=0.5, label='δ*/H = 0.01')
ax.set_xlabel('x_eff (cells)')
ax.set_ylabel('δ* / H')
ax.set_title('Displacement Thickness Ratio vs x_eff\n(plate400_y1600: H=800 cells)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('benchmarks/pending/blasius_flat_plate/fig4_dstar_over_H.png', dpi=150)
plt.close()

print("Figures generated successfully")
