#!/usr/bin/env python
"""Generate analysis plots and report for Womersley benchmark."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
with open('scan_out/result.json') as f:
    data = json.load(f)

# Load individual case files for detailed profiles
cases = {}
for fname in Path('scan_out').glob('case_H*.json'):
    with open(fname) as f:
        case = json.load(f)
        key = (case['H'], case['alpha'])
        cases[key] = case

# Create figure 1: Convergence plot
fig1, ax1 = plt.subplots(figsize=(8, 6))
H_values = sorted(set(entry['H'] for entry in data['per_grid']))
for alpha in [4, 8]:
    errors = []
    for H in H_values:
        entry = next(e for e in data['per_grid'] if e['H'] == H and e['alpha'] == alpha)
        errors.append(entry['complex_rel_max_pct'])
    ax1.plot(H_values, errors, 'o-', label=f'α={alpha}', linewidth=2, markersize=8)

ax1.axhline(y=3.0, color='r', linestyle='--', label='3% threshold', linewidth=2)
ax1.set_xlabel('Channel height H (lattice units)', fontsize=12)
ax1.set_ylabel('Complex amplitude error (%)', fontsize=12)
ax1.set_title('Womersley Benchmark: Grid Convergence', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.set_yscale('log')
fig1.tight_layout()
fig1.savefig('convergence_plot.png', dpi=150)
print('✓ Saved convergence_plot.png')

# Create figure 2: Velocity profiles comparison (α=8, H=59)
fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (H, alpha) in enumerate([(59, 8), (119, 8)]):
    ax = axes[idx]
    case = cases[(H, alpha)]
    y = np.array(case['y_profile'])
    u_num = np.array(case['u_num_final_phase'])
    
    # Compute analytical solution at the final phase
    y_phys = y
    H_eff = case['H_eff']
    a0 = case['a0']
    omega = case['omega']
    lam = np.sqrt(1j) * alpha
    yp = (y_phys - H_eff / 2.0) / (H_eff / 2.0)
    U_ana = (a0 / (1j * omega)) * (1.0 - np.cosh(lam * yp) / np.cosh(lam))
    
    # Get the phase of the final snapshot
    final_phase = case['phase_table'][-1]['phase_deg']
    u_ref = np.real(U_ana * np.exp(1j * np.radians(final_phase)))
    
    ax.plot(y, u_ref, 'k-', label='Analytical', linewidth=2)
    ax.plot(y, u_num, 'r--', label='TensorLBM', linewidth=2)
    ax.set_xlabel('y (lattice units)', fontsize=11)
    ax.set_ylabel('u (lattice units/time)', fontsize=11)
    ax.set_title(f'H={H}, α={alpha}, phase={final_phase:.1f}°', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

fig2.suptitle('Velocity Profile Comparison (Final Phase)', fontsize=14, y=1.02)
fig2.tight_layout()
fig2.savefig('profile_comparison.png', dpi=150, bbox_inches='tight')
print('✓ Saved profile_comparison.png')

# Create figure 3: Phasor comparison (α=8, H=59)
fig3, ax3 = plt.subplots(figsize=(8, 8))
case = cases[(59, 8)]
phasor_re = np.array(case['phasor_re'])
phasor_im = np.array(case['phasor_im'])
phasor_ana_re = np.array(case['phasor_ana_re'])
phasor_ana_im = np.array(case['phasor_ana_im'])

ax3.plot(phasor_ana_re, phasor_ana_im, 'k-', label='Analytical', linewidth=2)
ax3.plot(phasor_re, phasor_im, 'r--', label='TensorLBM', linewidth=2)
ax3.set_xlabel('Re(U) (lattice units/time)', fontsize=12)
ax3.set_ylabel('Im(U) (lattice units/time)', fontsize=12)
ax3.set_title('Complex Amplitude (Phasor) Comparison\nH=59, α=8', fontsize=14)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal')
fig3.tight_layout()
fig3.savefig('phasor_comparison.png', dpi=150)
print('✓ Saved phasor_comparison.png')

# Create figure 4: Error distribution across the channel
fig4, ax4 = plt.subplots(figsize=(10, 6))
case = cases[(59, 8)]
y = np.array(case['y_profile'])
phasor_num = np.array(case['phasor_re']) + 1j * np.array(case['phasor_im'])
phasor_ana = np.array(case['phasor_ana_re']) + 1j * np.array(case['phasor_ana_im'])
rel_error = np.abs(phasor_num / phasor_ana - 1.0) * 100

ax4.semilogy(y, rel_error, 'b-o', linewidth=2, markersize=6)
ax4.axhline(y=3.0, color='r', linestyle='--', label='3% threshold', linewidth=2)
ax4.set_xlabel('y (lattice units)', fontsize=12)
ax4.set_ylabel('Relative error (%)', fontsize=12)
ax4.set_title('Per-Row Complex Amplitude Error Distribution\nH=59, α=8', fontsize=14)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)
fig4.tight_layout()
fig4.savefig('error_distribution.png', dpi=150)
print('✓ Saved error_distribution.png')

print('\n✓ All plots generated successfully!')
