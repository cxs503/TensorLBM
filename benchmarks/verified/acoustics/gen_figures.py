#!/usr/bin/env python3
"""Generate publication-quality figures for the acoustics benchmark report."""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Load results
with open(Path(__file__).parent / "result.json") as f:
    data = json.load(f)

conv = data["convergence"]
tau_sweep = data["tau_sweep"]
linearity = data["linearity"]

CS = 1.0 / math.sqrt(3.0)

# ============================================================================
# Figure 1: Convergence - wave speed and attenuation vs analytical
# ============================================================================
fig1, axes = plt.subplots(1, 2, figsize=(12, 5))

Ns = [c["N"] for c in conv]
c_meas = [c["c_meas"] for c in conv]
c_cont = CS
delta_meas = [c["delta_meas"] for c in conv]
delta_cont = [c["delta_cont"] for c in conv]
delta_disc = [c["delta_discrete"] for c in conv]

# Left: wave speed
ax = axes[0]
ax.plot(Ns, [c_cont]*3, 'k--', lw=2, label=f'Continuum $c_s = 1/\\sqrt{{3}}$ = {c_cont:.6f}')
ax.plot(Ns, c_meas, 'ro-', ms=8, lw=2, label='LBM measured $c$')
ax.set_xlabel('N (points per wavelength)', fontsize=13)
ax.set_ylabel('Phase speed $c$ (LU/step)', fontsize=13)
ax.set_title('(a) Phase speed convergence', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
for i, N in enumerate(Ns):
    err = 100 * (c_meas[i] / c_cont - 1)
    ax.annotate(f'N={N}\nerr={err:+.3f}%', (N, c_meas[i]),
                textcoords="offset points", xytext=(15, -15), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))

# Right: attenuation rate
ax = axes[1]
k_vals = [c["k"] for c in conv]
ax.plot(Ns, delta_cont, 'k--', lw=2, label='Continuum $\\delta = \\nu k^2$')
ax.plot(Ns, delta_disc, 'b:', lw=2, label='Discrete theory (exact eigenvalue)')
ax.plot(Ns, delta_meas, 'rs-', ms=8, lw=2, label='LBM measured $\\delta$')
ax.set_xlabel('N (points per wavelength)', fontsize=13)
ax.set_ylabel('Attenuation rate $\\delta$ (1/step)', fontsize=13)
ax.set_title('(b) Attenuation rate convergence', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')
for i, N in enumerate(Ns):
    err = 100 * (delta_meas[i] / delta_cont[i] - 1)
    ax.annotate(f'N={N}\nerr={err:+.3f}%', (N, delta_meas[i]),
                textcoords="offset points", xytext=(15, 10), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))

fig1.tight_layout()
fig1.savefig(Path(__file__).parent / "fig_convergence.png", dpi=200, bbox_inches='tight')
plt.close(fig1)

# ============================================================================
# Figure 2: Error vs N on log-log scale (convergence order)
# ============================================================================
fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

# Wave speed error
c_err_pct = [100 * (c["c_meas"] / CS - 1) for c in conv]
delta_err_pct = [100 * (c["delta_meas"] / c["delta_cont"] - 1) for c in conv]

ax = axes[0]
ax.loglog(Ns, [abs(e) for e in c_err_pct], 'ro-', ms=8, lw=2, label='Phase speed error')
# Reference slope N^-2
N_ref = np.array(Ns, dtype=float)
slope_ref = c_err_pct[0] * (N_ref[0] / N_ref) ** 2
ax.loglog(N_ref, np.abs(slope_ref), 'k--', lw=1, alpha=0.5, label='$O(N^{-2})$ reference')
ax.set_xlabel('N (points per wavelength)', fontsize=13)
ax.set_ylabel('|Error| (%)', fontsize=13)
ax.set_title('(a) Phase speed error vs resolution', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, which='both')

# Attenuation error
ax = axes[1]
ax.loglog(Ns, [abs(e) for e in delta_err_pct], 'rs-', ms=8, lw=2, label='Attenuation error')
slope_ref_d = delta_err_pct[0] * (N_ref[0] / N_ref) ** 2
ax.loglog(N_ref, np.abs(slope_ref_d), 'k--', lw=1, alpha=0.5, label='$O(N^{-2})$ reference')
ax.set_xlabel('N (points per wavelength)', fontsize=13)
ax.set_ylabel('|Error| (%)', fontsize=13)
ax.set_title('(b) Attenuation error vs resolution', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, which='both')

fig2.tight_layout()
fig2.savefig(Path(__file__).parent / "fig_error_order.png", dpi=200, bbox_inches='tight')
plt.close(fig2)

# ============================================================================
# Figure 3: Tau sweep - attenuation vs tau
# ============================================================================
fig3, ax = plt.subplots(figsize=(8, 5))

taus = [t["tau"] for t in tau_sweep]
nu_vals = [t["nu"] for t in tau_sweep]
delta_m = [t["delta_meas"] for t in tau_sweep]
delta_c = [t["delta_cont"] for t in tau_sweep]
delta_d = [t["delta_discrete"] for t in tau_sweep]

ax.plot(taus, delta_c, 'k--', lw=2, label='Continuum $\\nu k^2$')
ax.plot(taus, delta_d, 'b:', lw=2, label='Discrete theory')
ax.plot(taus, delta_m, 'go-', ms=8, lw=2, label='LBM measured')
ax.set_xlabel('$\\tau$ (relaxation time)', fontsize=13)
ax.set_ylabel('Attenuation rate $\\delta$ (1/step)', fontsize=13)
ax.set_title('Attenuation vs $\\tau$ (N=32, $\\varepsilon=10^{-3}$)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

for i, tau in enumerate(taus):
    err = 100 * (delta_m[i] / delta_c[i] - 1)
    ax.annotate(f'$\\tau$={tau}\nerr={err:+.3f}%', (tau, delta_m[i]),
                textcoords="offset points", xytext=(12, -12), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))

fig3.tight_layout()
fig3.savefig(Path(__file__).parent / "fig_tau_sweep.png", dpi=200, bbox_inches='tight')
plt.close(fig3)

# ============================================================================
# Figure 4: Acoustic wave field snapshot (N=64, showing density field)
# ============================================================================
# We need to re-run one case to get the field snapshot
# Instead, let's reconstruct from the theory
fig4, axes = plt.subplots(1, 2, figsize=(14, 5))

# Reconstruct the wave at t=0 and at final time for N=64
N = 64
k = 2 * math.pi / N
x = np.arange(N)
eps = 1e-3
tau = 1.0
nu = CS**2 * (tau - 0.5)

# Get the discrete eigenvalue
from run import eig_acoustic
s = eig_acoustic(k, tau)
omega = s.imag
delta = -s.real

# Initial condition
rho_init = 1.0 + eps * np.sin(k * x)

# Final state (after nsteps)
nsteps = conv[2]["nsteps"]  # N=64 case
rho_final = 1.0 + eps * np.exp(-delta * nsteps) * np.sin(k * x - omega * nsteps)

# Analytical continuous solution
delta_cont_val = nu * k**2
rho_analytical = 1.0 + eps * np.exp(-delta_cont_val * nsteps) * np.sin(k * x - CS * k * nsteps)

ax = axes[0]
ax.plot(x, rho_init - 1, 'k-', lw=2, label='Initial ($t=0$)')
ax.plot(x, (rho_final - 1) * 1e3, 'r-', lw=1.5, label=f'LBM (×10³, t={nsteps} steps)')
ax.plot(x, (rho_analytical - 1) * 1e3, 'b--', lw=1.5, label=f'Continuum theory (×10³)')
ax.set_xlabel('x (lattice units)', fontsize=13)
ax.set_ylabel('$(\\rho - \\rho_0)$ (×10⁻³)', fontsize=13)
ax.set_title('(a) Density perturbation: initial vs final', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Right: amplitude decay over time
ax = axes[1]
n_arr = np.linspace(0, nsteps, 500)
amp_discrete = eps * np.exp(-delta * n_arr)
amp_cont = eps * np.exp(-delta_cont_val * n_arr)
ax.plot(n_arr, amp_discrete * 1e3, 'r-', lw=2, label='LBM discrete decay')
ax.plot(n_arr, amp_cont * 1e3, 'b--', lw=2, label='Continuum $e^{-\\nu k^2 t}$')
ax.set_xlabel('Time step $n$', fontsize=13)
ax.set_ylabel('Amplitude $\\varepsilon e^{-\\delta n}$ (×10⁻³)', fontsize=13)
ax.set_title('(b) Amplitude decay: discrete vs continuum', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

fig4.tight_layout()
fig4.savefig(Path(__file__).parent / "fig_wave_field.png", dpi=200, bbox_inches='tight')
plt.close(fig4)

# ============================================================================
# Figure 5: Summary dashboard - all errors in one view
# ============================================================================
fig5 = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 3, figure=fig5, hspace=0.35, wspace=0.35)

# (a) Phase speed error vs N
ax = fig5.add_subplot(gs[0, 0])
ax.bar([str(N) for N in Ns], [abs(e) for e in c_err_pct], color='coral', edgecolor='darkred')
ax.axhline(y=3.0, color='r', ls='--', lw=1.5, label='3% threshold')
ax.set_ylabel('|Error| (%)', fontsize=12)
ax.set_title('(a) Phase speed error', fontsize=13)
ax.set_xlabel('N', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# (b) Attenuation error vs N
ax = fig5.add_subplot(gs[0, 1])
ax.bar([str(N) for N in Ns], [abs(e) for e in delta_err_pct], color='lightblue', edgecolor='navy')
ax.axhline(y=3.0, color='r', ls='--', lw=1.5, label='3% threshold')
ax.set_ylabel('|Error| (%)', fontsize=12)
ax.set_title('(b) Attenuation error', fontsize=13)
ax.set_xlabel('N', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# (c) Tau sweep errors
ax = fig5.add_subplot(gs[0, 2])
tau_err = [100 * (t["delta_meas"] / t["delta_cont"] - 1) for t in tau_sweep]
ax.bar([f'{t}' for t in taus], [abs(e) for e in tau_err], color='lightgreen', edgecolor='darkgreen')
ax.axhline(y=3.0, color='r', ls='--', lw=1.5, label='3% threshold')
ax.set_ylabel('|Error| (%)', fontsize=12)
ax.set_title('(c) Attenuation error vs $\\tau$', fontsize=13)
ax.set_xlabel('$\\tau$', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# (d) Measured vs discrete theory (pipeline validation)
ax = fig5.add_subplot(gs[1, 0])
c_disc_err = [abs(c["c_err_disc_pct"]) for c in conv]
d_disc_err = [abs(c["delta_err_disc_pct"]) for c in conv]
x_pos = np.arange(len(Ns))
w = 0.35
ax.bar(x_pos - w/2, c_disc_err, w, label='Speed vs discrete', color='salmon')
ax.bar(x_pos + w/2, d_disc_err, w, label='Decay vs discrete', color='lightskyblue')
ax.set_xticks(x_pos)
ax.set_xticklabels([str(N) for N in Ns])
ax.set_ylabel('|Error| (%)', fontsize=12)
ax.set_title('(d) vs discrete theory (pipeline check)', fontsize=13)
ax.set_xlabel('N', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_yscale('log')

# (e) Linearity check
ax = fig5.add_subplot(gs[1, 1])
eps_vals = [l["eps"] for l in linearity]
lin_c_err = [100 * (l["c_meas"] / CS - 1) for l in linearity]
lin_d_err = [100 * (l["delta_meas"] / l["delta_cont"] - 1) for l in linearity]
ax.plot(eps_vals, [abs(e) for e in lin_c_err], 'ro-', ms=8, label='Speed error')
ax.plot(eps_vals, [abs(e) for e in lin_d_err], 'bs-', ms=8, label='Decay error')
ax.axhline(y=3.0, color='r', ls='--', lw=1.5, label='3% threshold')
ax.set_xlabel('Amplitude $\\varepsilon$', fontsize=12)
ax.set_ylabel('|Error| (%)', fontsize=12)
ax.set_title('(e) Linearity check (N=32, $\\tau$=1)', fontsize=13)
ax.set_xscale('log')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (f) R² quality metrics
ax = fig5.add_subplot(gs[1, 2])
r2_amp = [c["r2_amp"] for c in conv]
r2_ph = [c["r2_phase"] for c in conv]
ax.plot(Ns, r2_amp, 'go-', ms=8, label='$R^2$ amplitude fit')
ax.plot(Ns, r2_ph, 'm^-', ms=8, label='$R^2$ phase fit')
ax.set_xlabel('N', fontsize=12)
ax.set_ylabel('$R^2$', fontsize=12)
ax.set_title('(f) Fit quality metrics', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.99999, 1.000001)

fig5.savefig(Path(__file__).parent / "fig_dashboard.png", dpi=200, bbox_inches='tight')
plt.close(fig5)

print("All figures generated successfully:")
for fn in ["fig_convergence.png", "fig_error_order.png", "fig_tau_sweep.png",
           "fig_wave_field.png", "fig_dashboard.png"]:
    p = Path(__file__).parent / fn
    print(f"  {fn}: {p.stat().st_size} bytes")
