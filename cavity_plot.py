#!/usr/bin/env python
"""Generate cavity Re=100 verification plots from saved simulation data."""
import json, base64, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Our results (from earlier runs)
results_32 = {
    "nx": 32, "re": 100.0, "u_lid": 0.06, "tau": 0.5576, "steps": 11000,
    "u_mid": -0.2111082673072815, "u_bot": -0.025952625088393688,
    "v_mid": 0.06321149505674839, "vortex": [0.6338, 0.755],
    "rmse_u": 0.013767307721246277, "rmse_v": 0.015210128524288985,
    "max_abs_dev_pct": 2.247926354467865
}
results_48 = {
    "nx": 48, "re": 100.0, "u_lid": 0.06, "tau": 0.5864, "steps": 11000,
    "u_mid": -0.2143961489200592, "u_bot": -0.03259865113068372,
    "v_mid": 0.05380906164646146, "vortex": [0.6117, 0.7466],
    "rmse_u": 0.009227728577815791, "rmse_v": 0.010109316252020413,
    "max_abs_dev_pct": 1.899782122612001
}

# Baseline (from result.json)
baseline_128 = {"nx": 128, "max_abs_dev_pct": 0.7513717701435135, "u_mid": -0.21084705740213394, "vortex": [0.6139, 0.7402]}
baseline_192 = {"nx": 192, "max_abs_dev_pct": 0.7263405613303187, "u_mid": -0.21028464287519455, "vortex": [0.6141, 0.7388]}

# Plot 1: Convergence of max_abs_dev vs grid
fig, ax = plt.subplots(figsize=(6, 4))
nx_vals = [32, 48, 128, 192]
dev_vals = [results_32["max_abs_dev_pct"], results_48["max_abs_dev_pct"], baseline_128["max_abs_dev_pct"], baseline_192["max_abs_dev_pct"]]
ax.semilogy(nx_vals, dev_vals, 'bo-', markersize=8)
ax.axhline(y=3.0, color='r', linestyle='--', alpha=0.7, label='3% threshold')
ax.set_xlabel('Grid size (nx)')
ax.set_ylabel('Max absolute deviation (%)')
ax.set_title('Cavity Re=100: Grid convergence (Ghia 1982)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cavity_convergence.png', dpi=150, bbox_inches='tight')
print('Saved cavity_convergence.png')

# Plot 2: u_mid and vortex position comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
nx_all = [32, 48, 128, 192]
u_mid_vals = [results_32["u_mid"], results_48["u_mid"], baseline_128["u_mid"], baseline_192["u_mid"]]
ax.plot(nx_all, u_mid_vals, 'bo-', markersize=8, label='LBM u(0.5,0.5)')
ax.axhline(y=-0.20581, color='r', linestyle='--', alpha=0.7, label='Ghia: -0.20581')
ax.set_xlabel('Grid size (nx)')
ax.set_ylabel('u(0.5, 0.5) / U_lid')
ax.set_title('Centerline velocity convergence')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
vx_vals = [results_32["vortex"][0], results_48["vortex"][0], baseline_128["vortex"][0], baseline_192["vortex"][0]]
vy_vals = [results_32["vortex"][1], results_48["vortex"][1], baseline_128["vortex"][1], baseline_192["vortex"][1]]
ax.plot(nx_all, vx_vals, 'bo-', markersize=8, label='x_vortex')
ax.plot(nx_all, vy_vals, 'rs-', markersize=8, label='y_vortex')
ax.axhline(y=0.6172, color='b', linestyle='--', alpha=0.5)
ax.axhline(y=0.7344, color='r', linestyle='--', alpha=0.5)
ax.set_xlabel('Grid size (nx)')
ax.set_ylabel('Vortex center position')
ax.set_title('Vortex center convergence (Ghia: 0.6172, 0.7344)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cavity_vortex_convergence.png', dpi=150, bbox_inches='tight')
print('Saved cavity_vortex_convergence.png')

# Encode images as base64 for report
def encode_png(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')

figures = [
    {"name": "convergence", "caption": "Grid convergence: max absolute deviation vs grid size (nx=32,48,128,192). All cases below 3% threshold.", "png_base64": encode_png('cavity_convergence.png')},
    {"name": "vortex_convergence", "caption": "Vortex center position and centerline velocity convergence with grid refinement.", "png_base64": encode_png('cavity_vortex_convergence.png')}
]

# Save figures data
with open('cavity_figures.json', 'w') as f:
    json.dump(figures, f)
print('Saved cavity_figures.json')
