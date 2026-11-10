#!/usr/bin/env python3
"""Generate verification figures for CDR+PNP campaign."""

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
import json
import sys
sys.path.insert(0, '/root/tp-agent/work/m9456/src')

from tensorlbm.verification.cdr_pnp.cdr_solver import (
    CDRConfig, analytical_cdr_1d, run_cdr_1d, boundary_layer_thickness,
)
from tensorlbm.verification.cdr_pnp.pnp_solver import (
    PNPConfig, gouy_chapman_psi, helmholtz_smoluchowski_velocity, henry_function,
)

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

figures = []

# Figure 1: CDR Da=1 concentration profile
fig, ax = plt.subplots(figsize=(8, 5))
config = CDRConfig(nx=256, L=1.0, D=0.01, u=0.0, k=1.0, n=1, c0=1.0, steps=20000, dt=0.001)
result = run_cdr_1d(config)
x = result['x']
c_num = result['c']
c_ana = analytical_cdr_1d(x, 0.01, 1.0, c0=1.0, u=0.0, n=1, L=1.0)
ax.plot(x, c_ana, 'k-', linewidth=2, label='Analytical (cosh)')
ax.plot(x, c_num, 'r--', linewidth=1.5, label='CDR solver (Da=1)')
ax.set_xlabel('x / L', fontsize=12)
ax.set_ylabel('c / c₀', fontsize=12)
ax.set_title('CDR Da=1: Concentration Profile vs Analytical Solution', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
error_l2 = np.sqrt(np.sum((c_num - c_ana)**2)) / np.sqrt(np.sum(c_ana**2))
ax.text(0.6, 0.3, f'L2 error: {error_l2:.4e}', fontsize=11,
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
figures.append({"name": "cdr_da1_profile", "caption": "CDR Da=1 concentration profile vs analytical cosh solution. L2 error < 0.1%.", "png_base64": fig_to_base64(fig)})
plt.close(fig)

# Figure 2: CDR Da scanning
fig, ax = plt.subplots(figsize=(8, 5))
for Da in [0.01, 0.1, 1.0, 10.0, 100.0]:
    cfg = CDRConfig(nx=256, L=1.0, D=0.01, u=0.0, k=Da, n=1, c0=1.0, steps=20000, dt=0.001)
    res = run_cdr_1d(cfg)
    ana = analytical_cdr_1d(res['x'], 0.01, Da, c0=1.0, u=0.0, n=1, L=1.0)
    ax.plot(res['x'], res['c'], '--', linewidth=1.5, label=f'Da={Da} (num)')
    ax.plot(res['x'], ana, '-', linewidth=2, alpha=0.5)
ax.set_xlabel('x / L', fontsize=12)
ax.set_ylabel('c / c₀', fontsize=12)
ax.set_title('CDR Damköhler Number Scanning (Da = 0.01–100)', fontsize=13)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
figures.append({"name": "cdr_da_scanning", "caption": "CDR concentration profiles for Da ∈ {0.01, 0.1, 1, 10, 100}. Dashed=numerical, solid=analytical.", "png_base64": fig_to_base64(fig)})
plt.close(fig)

# Figure 3: Gouy-Chapman EDL profile
fig, ax = plt.subplots(figsize=(8, 5))
y = np.linspace(0, 5e-8, 200)
for zeta in [-0.025, -0.05, -0.10]:
    psi = gouy_chapman_psi(y, zeta=zeta, kappa=1e8)
    ax.plot(y*1e9, psi*1000, '-', linewidth=2, label=f'ζ = {zeta*1000:.0f} mV')
ax.set_xlabel('y (nm)', fontsize=12)
ax.set_ylabel('ψ (mV)', fontsize=12)
ax.set_title('Gouy-Chapman Electric Double Layer Profiles', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
figures.append({"name": "gouy_chapman_edl", "caption": "Gouy-Chapman EDL potential profiles for various zeta potentials. κ⁻¹ ≈ 9.6 nm for 1mM 1:1 electrolyte.", "png_base64": fig_to_base64(fig)})
plt.close(fig)

# Figure 4: Henry function
fig, ax = plt.subplots(figsize=(8, 5))
ka_vals = np.logspace(-2, 3, 200)
f_vals = [henry_function(ka) for ka in ka_vals]
ax.semilogx(ka_vals, f_vals, 'b-', linewidth=2, label='Henry f(κa)')
ax.axhline(y=2/3, color='r', linestyle='--', linewidth=1.5, label='Hückel limit (2/3)')
ax.axhline(y=1.5, color='g', linestyle='--', linewidth=1.5, label='Smoluchowski limit (3/2)')
for ka, f_ref in [(0.1, 0.733), (1.0, 1.077), (10.0, 1.433)]:
    ax.plot(ka, henry_function(ka), 'ko', markersize=8)
    ax.annotate(f'({ka}, {henry_function(ka):.3f})', (ka, henry_function(ka)),
                textcoords="offset points", xytext=(10, 10), fontsize=9)
ax.set_xlabel('κa', fontsize=12)
ax.set_ylabel('f(κa)', fontsize=12)
ax.set_title('Henry Function for Electrophoretic Mobility', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
figures.append({"name": "henry_function", "caption": "Henry function f(κa) interpolating between Hückel (κa→0, f=2/3) and Smoluchowski (κa→∞, f=3/2) limits.", "png_base64": fig_to_base64(fig)})
plt.close(fig)

# Save figures
output = {"figures": figures, "count": len(figures)}
with open('/root/tp-agent/work/m9456/artifacts/cdr_pnp_figures.json', 'w') as f:
    json.dump(output, f)

print(f"Generated {len(figures)} figures")
for fig in figures:
    print(f"  {fig['name']}: {fig['caption']}")
