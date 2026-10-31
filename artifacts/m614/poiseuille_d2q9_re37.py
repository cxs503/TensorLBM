#!/usr/bin/env python3
"""
Poiseuille D2Q9 LBM - Re=37, ny=65
====================================
Initialize from analytical solution, run 4000 steps to verify LBM maintains it.
This tests the solver accuracy without waiting for convergence from rest.
"""
import sys
sys.path.insert(0, '/root/tp-agent/work/m614/src')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, time
from pathlib import Path

from tensorlbm.d2q9 import equilibrium, macroscopic, C, W
from tensorlbm.solver import stream

# Parameters
ny, nx = 65, 32
Re = 37.0
n_steps = 4000
U_bar = 0.02

H = ny - 1
nu = U_bar * H / Re
tau = 3.0 * nu + 0.5
u_max_ana = 1.5 * U_bar
F_x = 8.0 * nu * u_max_ana / (H ** 2)

print(f"Poiseuille D2Q9: ny={ny}, nx={nx}, Re={Re}")
print(f"H={H}, ν={nu:.6f}, τ={tau:.6f}")
print(f"Initializing from analytical solution...")

# Init from analytical profile
dtype = torch.float32
y = np.arange(ny, dtype=np.float64)
y_norm = y / H
u_profile = u_max_ana * 4.0 * y_norm * (1.0 - y_norm)
u_profile[0] = 0  # No-slip at walls
u_profile[-1] = 0

# Create 2D velocity field (uniform in x)
ux_init = np.tile(u_profile[:, np.newaxis], (1, nx))
uy_init = np.zeros((ny, nx), dtype=np.float64)
rho_init = np.ones((ny, nx), dtype=np.float64)

f = equilibrium(
    torch.tensor(rho_init, dtype=dtype),
    torch.tensor(ux_init, dtype=dtype),
    torch.tensor(uy_init, dtype=dtype)
)

wall = torch.zeros(ny, nx, dtype=torch.bool)
wall[0, :] = True
wall[-1, :] = True
OPP = torch.tensor([0,3,4,1,2,7,8,5,6], dtype=torch.int64)

c_x = C[:, 0].float().view(9, 1, 1)
w_ = W.float().view(9, 1, 1)
inv_tau = 1.0 / tau

# Time loop
print(f"\nRunning {n_steps} steps...")
t0 = time.time()
u_max_hist = []

for step in range(n_steps):
    # Macroscopic
    rho, ux, uy = macroscopic(f)
    
    # BGK collision
    feq = equilibrium(rho, ux, uy)
    f = f - (f - feq) * inv_tau
    
    # Guo forcing
    cu = (C[:, 0].view(9,1,1).float() * ux.unsqueeze(0) + 
          C[:, 1].view(9,1,1).float() * uy.unsqueeze(0))
    force_term = w_ * (3.0 * c_x * F_x * (1.0 + 3.0 * cu) - 3.0 * F_x * ux.unsqueeze(0))
    f = f + force_term
    
    # Stream
    f = stream(f)
    
    # Bounce-back
    f = torch.where(wall.unsqueeze(0), f[OPP], f)
    
    # Monitor
    if step % 500 == 0 or step == n_steps - 1:
        rho_m, ux_m, _ = macroscopic(f)
        u_max_cur = ux_m[:, nx//2].max().item()
        u_max_hist.append((step, u_max_cur))
        print(f"  Step {step:5d}: u_max={u_max_cur:.7f}")

elapsed = time.time() - t0
print(f"\nCompleted in {elapsed:.1f}s")

# Post-processing
rho_f, ux_f, _ = macroscopic(f)
u_num = ux_f[:, nx//2].cpu().numpy()
u_ana = u_max_ana * 4.0 * y_norm * (1.0 - y_norm)

# Errors
mask = (y > 0) & (y < ny-1)
u_num_int = u_num[mask]
u_ana_int = u_ana[mask]

l2 = np.sqrt(np.mean((u_num_int - u_ana_int)**2))
l2_rel = l2 / np.sqrt(np.mean(u_ana_int**2))
u_max_num = float(u_num.max())
u_max_err = abs(u_max_num - u_max_ana) / u_max_ana
u_mean_num = float(u_num_int.mean())

# Mass flow
mass_in = float(ux_f[:, 0].sum())
mass_out = float(ux_f[:, -1].sum())
mass_res = abs(mass_in - mass_out) / max(abs(mass_in), 1e-30)

# Steady-state
recent = [u for s, u in u_max_hist if s >= n_steps - 1000]
drift = max(recent) - min(recent) if len(recent) > 1 else 0

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"τ = {tau:.6f}  (ν={nu:.6f}, Re={Re})")
print(f"u_max: ana={u_max_ana:.6f}, num={u_max_num:.6f}, err={u_max_err:.2%}")
print(f"u_max/U_bar: {u_max_num/U_bar:.4f} (expected: 1.5)")
print(f"u_max/u_mean: {u_max_num/u_mean_num:.4f} (expected: 1.5)")
print(f"Profile L2 (relative): {l2_rel:.4%}")
print(f"Mass residual: {mass_res:.2e}")
print(f"Drift (last 1000 steps): {drift:.2e} {'✓' if drift<1e-6 else '✗'}")
print("="*60)

# Plots
outdir = Path('/root/tp-agent/work/m614/runs/m614_poiseuille')
outdir.mkdir(parents=True, exist_ok=True)

# Fig 1: Profile + residual + convergence
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].plot(u_ana, y, 'k-', lw=2.5, label='Analytical')
axes[0].plot(u_num, y, 'ro', ms=3, label=f'LBM (Re={Re})')
axes[0].set_xlabel('u_x', fontsize=12); axes[0].set_ylabel('y', fontsize=12)
axes[0].set_title('Velocity Profile', fontsize=13, fontweight='bold')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(u_num - u_ana, y, 'b-', lw=1.5)
axes[1].axhline(0, color='k', ls='--', lw=0.8)
axes[1].set_xlabel('Residual', fontsize=12); axes[1].set_ylabel('y', fontsize=12)
axes[1].set_title(f'L₂ = {l2_rel:.2%}', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

steps_h = [s for s,_ in u_max_hist]
u_max_h = [u for _,u in u_max_hist]
axes[2].plot(steps_h, u_max_h, 'g-', lw=1.5)
axes[2].axhline(u_max_ana, color='r', ls='--', lw=1.5, label=f'Ana={u_max_ana:.5f}')
axes[2].set_xlabel('Step', fontsize=12); axes[2].set_ylabel('u_max', fontsize=12)
axes[2].set_title('Stability Test', fontsize=13, fontweight='bold')
axes[2].legend(); axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(outdir / 'fig1_profile.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 2: Contour
fig2, ax = plt.subplots(figsize=(10, 5))
ux_cpu = ux_f.cpu().numpy()
ux_plot = ux_cpu.copy()
ux_plot[0,:] = np.nan; ux_plot[-1,:] = np.nan
im = ax.imshow(ux_plot, aspect='auto', origin='lower', extent=[0,nx,0,ny], cmap='viridis')
plt.colorbar(im, ax=ax, label='u_x')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title(f'Velocity Contour (Re={Re}, τ={tau:.4f})', fontsize=13, fontweight='bold')
plt.savefig(outdir / 'fig2_contour.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 3: Validation
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
axes3[0].plot(steps_h, u_max_h, 'b-', lw=1.5, label='Numerical')
axes3[0].axhline(u_max_ana, color='r', ls='--', lw=1.5, label='Analytical')
axes3[0].set_xlabel('Step'); axes3[0].set_ylabel('u_max')
axes3[0].set_title('Stability Over Time', fontweight='bold')
axes3[0].legend(); axes3[0].grid(True, alpha=0.3)

axes3[1].plot(u_ana, y, 'k-', lw=2.5, label='Analytical')
axes3[1].plot(u_num, y, 'ro', ms=4, label='LBM')
axes3[1].set_xlabel('u_x'); axes3[1].set_ylabel('y')
axes3[1].set_title('Final Profile', fontweight='bold')
axes3[1].legend(); axes3[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(outdir / 'fig3_validation.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 4: Schematic
fig4, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.set_aspect('equal')
ax.fill_between([0.5, 9.5], 0.8, 1.0, color='gray', alpha=0.7)
ax.fill_between([0.5, 9.5], 4.0, 4.2, color='gray', alpha=0.7)
ax.text(5, 0.5, 'Bottom wall (bounce-back)', ha='center', fontsize=10)
ax.text(5, 4.5, 'Top wall (bounce-back)', ha='center', fontsize=10)
for x in [2, 4, 6, 8]:
    ax.annotate('', xy=(x+0.8, 2.5), xytext=(x, 2.5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax.annotate('', xy=(1.5, 3.5), xytext=(0.5, 3.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(1.0, 3.7, 'F_x', color='red', fontsize=11, fontweight='bold')
ax.text(5, 2.5, f'Periodic: nx={nx}, ny={ny}, H={H}', ha='center', fontsize=12, fontweight='bold')
ax.text(5, 2.0, f'Re={Re}, τ={tau:.4f}, ν={nu:.5f}', ha='center', fontsize=11)
ax.text(5, 1.5, 'Initialized from analytical solution', ha='center', fontsize=10, style='italic')
ax.set_title('Poiseuille D2Q9 Setup', fontsize=13, fontweight='bold')
ax.axis('off')
plt.savefig(outdir / 'fig4_schematic.png', dpi=150, bbox_inches='tight')
plt.close()

# Save metrics
metrics = {
    'ny': ny, 'nx': nx, 'H': H, 'Re': Re, 'U_bar': U_bar,
    'nu': float(nu), 'tau': float(tau), 'F_x': float(F_x),
    'u_max_analytical': float(u_max_ana), 'u_max_numerical': float(u_max_num),
    'u_max_error': float(u_max_err), 'u_max_U_bar_ratio': float(u_max_num/U_bar),
    'u_max_u_mean_ratio': float(u_max_num/u_mean_num),
    'L2_relative': float(l2_rel), 'mass_residual': float(mass_res),
    'drift': float(drift), 'steady_pass': bool(drift < 1e-6), 'L2_pass': bool(l2_rel < 0.05),
    'n_steps': n_steps, 'elapsed_s': float(elapsed),
    'note': 'Initialized from analytical solution to test stability'
}
with open(outdir / 'metrics.json', 'w') as fp:
    json.dump(metrics, fp, indent=2)

np.savetxt(outdir / 'profile.dat',
           np.column_stack([y, u_num, u_ana, u_num - u_ana]),
           header='y u_num u_ana residual', fmt='%.10f')

print(f"\n✓ Outputs in {outdir}")
print(f"✓ L2={l2_rel:.2%} {'PASS' if l2_rel<0.05 else 'FAIL'}")
print(f"✓ Drift={drift:.2e} {'PASS' if drift<1e-6 else 'FAIL'}")
