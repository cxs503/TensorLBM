#!/usr/bin/env python3
"""Generate figures for capillary_invasion_washburn SDAA verification report."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path("/root/tp-agent/work/m413")
ARTIFACT_PATH = REPO_ROOT / "artifacts" / "sdaa_capillary_invasion_washburn.json"
FIGURES_DIR = REPO_ROOT / "artifacts" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load artifact
with open(ARTIFACT_PATH) as f:
    data = json.load(f)

series = data["series"]
steps = [s["step"] for s in series]
L_vals = [s["L"] for s in series]
massw_vals = [s["massw"] for s in series]
massg_vals = [s["massg"] for s in series]

# Figure 1: Washburn exponent beta comparison
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(steps, L_vals, 'o-', label='SDAA W32', markersize=4, linewidth=1.5)

# Fit line
beta_sdaa = data["results"]["beta"]
if beta_sdaa is not None:
    # Use window from artifact
    t1 = data["results"]["window"]["t1"]
    t2 = data["results"]["window"]["t2"]
    fit_pts = [(s["step"], s["L"]) for s in series if t1 <= s["step"] <= t2 and s["L"] > 0]
    if fit_pts:
        fit_steps = np.array([p[0] for p in fit_pts])
        fit_L = np.array([p[1] for p in fit_pts])
        # L = A * t^beta => log(L) = log(A) + beta*log(t)
        log_t = np.log(fit_steps)
        log_L = np.log(fit_L)
        coeffs = np.polyfit(log_t, log_L, 1)
        A = np.exp(coeffs[1])
        t_fit = np.linspace(fit_steps.min(), fit_steps.max(), 100)
        L_fit = A * t_fit**beta_sdaa
        ax.plot(t_fit, L_fit, '--', label=f'Fit: β={beta_sdaa:.3f}', linewidth=2)

# Theoretical Washburn (beta=0.5)
# Use same scale as fit
if beta_sdaa is not None and fit_pts:
    # Normalize to match at t1
    t_ref = fit_steps[0]
    L_ref = fit_L[0]
    A_theory = L_ref / (t_ref**0.5)
    L_theory = A_theory * t_fit**0.5
    ax.plot(t_fit, L_theory, ':', label='Theory: β=0.5', linewidth=2, color='green')

ax.set_xlabel('Time step', fontsize=12)
ax.set_ylabel('Front position L(t)', fontsize=12)
ax.set_title('Washburn Invasion Front: SDAA vs Theory', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig1_path = FIGURES_DIR / "fig_washburn_beta_sdaa.png"
plt.savefig(fig1_path, dpi=150)
plt.close()
print(f"Saved: {fig1_path}")

# Figure 2: Mass conservation drift
fig, ax = plt.subplots(figsize=(8, 5))
massw_init = data["results"]["massw_init"]
massw_drift = [(m - massw_init) / massw_init * 100 for m in massw_vals]
massg_init = series[0]["massg"]
massg_drift = [(m - massg_init) / massg_init * 100 for m in massg_vals]

ax.plot(steps, massw_drift, 'o-', label='Water phase', markersize=4, linewidth=1.5, color='blue')
ax.plot(steps, massg_drift, 's-', label='Gas phase', markersize=4, linewidth=1.5, color='red')
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Time step', fontsize=12)
ax.set_ylabel('Mass drift (%)', fontsize=12)
ax.set_title('Mass Conservation: SDAA W32', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig2_path = FIGURES_DIR / "fig_mass_drift_sdaa.png"
plt.savefig(fig2_path, dpi=150)
plt.close()
print(f"Saved: {fig2_path}")

# Figure 3: Comparison with CUDA results
fig, ax = plt.subplots(figsize=(8, 5))
cuda_beta = data["comparison_with_cuda"]["cuda_W32_beta"]
sdaa_beta = data["comparison_with_cuda"]["sdaa_W32_beta"]
cuda_dev = data["comparison_with_cuda"]["cuda_W32_beta_dev_pct"]
sdaa_dev = data["comparison_with_cuda"]["sdaa_W32_beta_dev_pct"]

x_pos = np.array([1, 2])
width = 0.35
bars1 = ax.bar(x_pos - width/2, [cuda_beta, sdaa_beta], width, label='β value', color='steelblue')
bars2 = ax.bar(x_pos + width/2, [cuda_dev, sdaa_dev], width, label='Deviation from 0.5 (%)', color='coral')

ax.set_ylabel('Value / Deviation (%)', fontsize=12)
ax.set_title('Washburn Exponent: CUDA vs SDAA', fontsize=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(['CUDA (40k steps)', 'SDAA (5k steps)'], fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
fig3_path = FIGURES_DIR / "fig_cuda_vs_sdaa_comparison.png"
plt.savefig(fig3_path, dpi=150)
plt.close()
print(f"Saved: {fig3_path}")

print("\nAll figures generated successfully.")
