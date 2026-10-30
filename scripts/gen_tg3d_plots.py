#!/usr/bin/env python
"""Generate plots for taylor_green_3d_decay SDAA sweep report."""
import json
import base64
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = "/root/tp-agent/work/m427/benchmarks/verified/taylor_green_3d"
OUT = "/root/tp-agent/work/m427/benchmarks/pending/taylor_green_3d_decay"

def load_history(n):
    data = np.loadtxt(f"{HERE}/energy_history_N{n}.csv", delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]  # step, energy

def encode_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

# ── Plot 1: Energy decay (E/E0 vs step) ──────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 5))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
labels = ['N=32', 'N=48', 'N=64']
ns = [32, 48, 64]
steps_list = [500, 1000, 1000]

for i, (n, steps) in enumerate(zip(ns, steps_list)):
    t, E = load_history(n)
    E0 = E[0]
    # Theoretical decay
    U0 = 0.05
    Re = 24.0
    nu = U0 * n / Re
    k = 2 * np.pi / n
    gamma_E = 6 * nu * k**2
    t_theory = np.linspace(0, steps, 200)
    E_theory = E0 * np.exp(-gamma_E * t_theory)
    
    ax1.plot(t, E/E0, 'o-', color=colors[i], label=f'{labels[i]} (sim)', markersize=3, alpha=0.8)
    ax1.plot(t_theory, E_theory/E0, '--', color=colors[i], label=f'{labels[i]} (theory)', alpha=0.5)

ax1.set_xlabel('Time step')
ax1.set_ylabel('E / E₀')
ax1.set_title('3D Taylor-Green Vortex Decay: Energy vs Time (SDAA CPU)')
ax1.legend(fontsize=8, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

png1 = encode_png(fig1)
plt.close(fig1)

# ── Plot 2: ln(E) vs step with linear fit ─────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))

for i, (n, steps) in enumerate(zip(ns, steps_list)):
    t, E = load_history(n)
    E0 = E[0]
    lnE = np.log(E[1:] / E0)  # skip t=0
    t_fit = t[1:]
    
    # Linear fit
    a, b = np.polyfit(t_fit, lnE, 1)
    gamma_sim = -a
    
    ax2.plot(t_fit, lnE, 'o', color=colors[i], label=f'{labels[i]} (sim)', markersize=3, alpha=0.7)
    ax2.plot(t_fit, a * t_fit + b, '--', color=colors[i], 
             label=f'{labels[i]}: γ={gamma_sim:.4e}', alpha=0.6)

ax2.set_xlabel('Time step')
ax2.set_ylabel('ln(E / E₀)')
ax2.set_title('3D Taylor-Green Vortex Decay: ln(E) vs Time (Linear Fit)')
ax2.legend(fontsize=7, loc='lower left')
ax2.grid(True, alpha=0.3)

png2 = encode_png(fig2)
plt.close(fig2)

# ── Plot 3: Error convergence ─────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(6, 4))

# SDAA sweep results
sdaa_errs = [1.2576, 0.5816, 0.5755]

ax3.plot(ns, sdaa_errs, 'o-', color='#1f77b4', label='SDAA CPU sweep', markersize=8, linewidth=2)
ax3.axhline(y=3.0, color='red', linestyle='--', alpha=0.5, label='3% threshold')

# Annotate
for n, err in zip(ns, sdaa_errs):
    ax3.annotate(f'+{err:.2f}%', (n, err), textcoords="offset points", 
                 xytext=(0, 10), ha='center', fontsize=9)

ax3.set_xlabel('Grid size N')
ax3.set_ylabel('Error in γ_E (%)')
ax3.set_title('Convergence: γ_E Error vs Grid Size')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(ns)

png3 = encode_png(fig3)
plt.close(fig3)

# ── Output JSON with base64 images ────────────────────────────────
report_figs = [
    {
        "name": "energy_decay",
        "caption": "Energy decay E/E₀ vs time step for N=32/48/64. Solid: simulation; dashed: theoretical exponential E(t)=E₀·exp(-6νk²t).",
        "png_base64": png1
    },
    {
        "name": "lnE_linear_fit",
        "caption": "ln(E/E₀) vs time step with linear fit. Slope gives γ_E_sim; compare with γ_E_theory=6νk².",
        "png_base64": png2
    },
    {
        "name": "error_convergence",
        "caption": "γ_E error (%) vs grid size N. All grids within 3% threshold (red dashed). Monotone convergence: 1.26% → 0.58% → 0.58%.",
        "png_base64": png3
    }
]

with open(f"{OUT}/report_figures.json", "w") as f:
    json.dump(report_figs, f)

print("Figures generated and saved to report_figures.json")
print(f"  energy_decay: {len(png1)} bytes base64")
print(f"  lnE_linear_fit: {len(png2)} bytes base64")
print(f"  error_convergence: {len(png3)} bytes base64")
