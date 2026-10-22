#!/usr/bin/env python3
"""Generate figures for cylinder_re40_st case."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
results = {}
for re in [20, 40]:
    for d in [40, 80]:
        f = f"/nfs/wangxi/runs/bm_close/cylinder_re40_st/w11a2/formal_w11a2/re{re}_D{d}.json"
        with open(f) as fp:
            results[(re, d)] = json.load(fp)

# Create figures directory
fig_dir = Path("benchmarks/pending/cylinder_re40_st/figures")
fig_dir.mkdir(exist_ok=True)

# Figure 1: Convergence history for Re=20
fig, ax = plt.subplots(figsize=(8, 5))
for d in [40, 80]:
    trace_file = f"/nfs/wangxi/runs/bm_close/cylinder_re40_st/w11a2/formal_w11a2/trace_re20_D{d}.npz"
    data = np.load(trace_file)
    t = data['t']
    cd = data['cd']
    ax.plot(t, cd, label=f'D={d}', linewidth=1.5)

ax.axhline(y=2.0001, color='r', linestyle='--', label='Reference (Fornberg 1980)')
ax.set_xlabel('Simulation Steps', fontsize=12)
ax.set_ylabel('Drag Coefficient Cd', fontsize=12)
ax.set_title('Re=20: Convergence History', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / 're20_convergence.png', dpi=150)
plt.close()

# Figure 2: Convergence history for Re=40
fig, ax = plt.subplots(figsize=(8, 5))
for d in [40, 80]:
    trace_file = f"/nfs/wangxi/runs/bm_close/cylinder_re40_st/w11a2/formal_w11a2/trace_re40_D{d}.npz"
    data = np.load(trace_file)
    t = data['t']
    cd = data['cd']
    ax.plot(t, cd, label=f'D={d}', linewidth=1.5)

ax.axhline(y=1.498, color='r', linestyle='--', label='Reference (Fornberg 1980)')
ax.set_xlabel('Simulation Steps', fontsize=12)
ax.set_ylabel('Drag Coefficient Cd', fontsize=12)
ax.set_title('Re=40: Convergence History', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / 're40_convergence.png', dpi=150)
plt.close()

# Figure 3: Error vs grid resolution
fig, ax = plt.subplots(figsize=(8, 5))
re20_errs = [results[(20, 40)]['err_pct'], results[(20, 80)]['err_pct']]
re40_errs = [results[(40, 40)]['err_pct'], results[(40, 80)]['err_pct']]
d_vals = [40, 80]

ax.plot(d_vals, re20_errs, 'o-', label='Re=20', linewidth=2, markersize=8)
ax.plot(d_vals, re40_errs, 's-', label='Re=40', linewidth=2, markersize=8)
ax.axhline(y=3.0, color='r', linestyle='--', label='3% threshold', alpha=0.7)
ax.set_xlabel('Grid Resolution D', fontsize=12)
ax.set_ylabel('Relative Error (%)', fontsize=12)
ax.set_title('Grid Convergence: Error vs Resolution', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / 'error_convergence.png', dpi=150)
plt.close()

print("Figures generated successfully:")
print(f"  - {fig_dir / 're20_convergence.png'}")
print(f"  - {fig_dir / 're40_convergence.png'}")
print(f"  - {fig_dir / 'error_convergence.png'}")
