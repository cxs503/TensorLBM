#!/usr/bin/env python3
"""Generate figures for backward_step benchmark report."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from pathlib import Path

# ── Figure 1: SDAA convergence trajectory ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Grid m=8, u=0.15
steps_m8 = [1000, 2000, 3000, 4000, 5000]
x1_m8 = [0.7832, 0.9270, 1.4998, 2.2830, 2.8076]

# Grid m=12, u=0.12
steps_m12 = [1000, 2000, 3000, 4000, 5000]
x1_m12 = [0.7039, 0.7810, 1.0210, 1.5831, 1.9705]

# Grid m=16, u=0.06
steps_m16 = [1000, 2000, 3000, 4000]
x1_m16 = [0.5463, 0.7968, 0.9746, 0.3817]

ax = axes[0]
ax.plot(steps_m8, x1_m8, 'o-', label='m=8, u=0.15', color='tab:blue')
ax.plot(steps_m12, x1_m12, 's-', label='m=12, u=0.12', color='tab:orange')
ax.plot(steps_m16, x1_m16, '^-', label='m=16, u=0.06', color='tab:green')
ax.axhline(y=6.10, color='red', linestyle='--', linewidth=2, label='Reference X1/H=6.10')
ax.set_xlabel('LBM Steps')
ax.set_ylabel('X1 / H')
ax.set_title('SDAA Short-Run Convergence Trajectory\n(Gartling 1990 BFS, Re=800, ER=2)')
ax.legend(fontsize=9)
ax.set_ylim(0, 7)
ax.grid(True, alpha=0.3)

# ── Figure 2: Pre-existing tau-matched ladder results (Armaly Re=100) ──
ax = axes[1]
h_i_vals = [51, 82, 123]
xr_psi_vals = [2.6800, 2.7177, 2.7534]
erturk_ref = 2.878
armaly_ref = 3.05

ax.plot(h_i_vals, xr_psi_vals, 'o-', color='tab:blue', label='TensorLBM (τ-matched)', markersize=8)
ax.axhline(y=erturk_ref, color='red', linestyle='--', linewidth=2, label=f'Erturk 2008: {erturk_ref}')
ax.axhline(y=armaly_ref, color='darkred', linestyle=':', linewidth=2, label=f'Armaly 1983 exp: {armaly_ref}')
ax.fill_between([45, 130], 2.792, 2.965, alpha=0.15, color='red', label='Erturk acceptance band')

# Extrapolated continuum limit
ax.plot(200, 2.845, '*', color='tab:blue', markersize=12, label=f'Extrapolated X∞=2.845±0.04')

ax.set_xlabel('Inlet half-height h_i (cells)')
ax.set_ylabel('X_r (step-height units)')
ax.set_title('Pre-existing τ-Matched Ladder Results\n(Armaly 1983 BFS, Re=100, ER=1.9423)')
ax.legend(fontsize=8)
ax.set_xlim(40, 210)
ax.set_ylim(2.5, 3.2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/bfs_fig1.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Figure 3: Re-sweep diagnostic ──
fig, ax = plt.subplots(figsize=(8, 5))
re_vals = [100, 150, 200]
xr_vals = [2.7534, 3.7854, 4.8047]
erturk_vals = [2.878, 3.889, 4.900]
err_vals = [-4.33, -2.66, -1.94]

ax.plot(re_vals, xr_vals, 'o-', color='tab:blue', label='TensorLBM (L3 grid, τ=0.602)', markersize=8)
ax.plot(re_vals, erturk_vals, 's--', color='red', label='Erturk 2008', markersize=8)

for i, (re, err) in enumerate(zip(re_vals, err_vals)):
    ax.annotate(f'{err}%', (re, xr_vals[i]), textcoords="offset points", xytext=(10, -15), fontsize=9, color='tab:blue')

ax.set_xlabel('Reynolds Number')
ax.set_ylabel('X_r (step-height units)')
ax.set_title('Re-Sweep Diagnostic: Same Grid/τ, Varying Re\n(L3 grid h_i=123, τ=0.602; absolute deviation ~constant ≈ -0.1)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/bfs_fig2.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Encode figures as base64 ──
figs = []
for name, path in [('bfs_convergence_trajectory', '/tmp/bfs_fig1.png'), ('bfs_re_sweep', '/tmp/bfs_fig2.png')]:
    with open(path, 'rb') as fp:
        png_b64 = base64.b64encode(fp.read()).decode('ascii')
    figs.append({'name': name, 'png_base64': png_b64})

# Save figures info
with open('/tmp/bfs_figures.json', 'w') as fp:
    json.dump(figs, fp)

print(f"Generated {len(figs)} figures")
for f in figs:
    print(f"  {f['name']}: {len(f['png_base64'])} chars base64")
