#!/usr/bin/env python3
"""Generate figures for surface tension verification report."""
import sys, json
sys.path.insert(0, 'src')

import torch
torch.set_num_threads(2)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tensorlbm.free_surface_tension import (
    CapillarySolver, SolverConfig, tanh_droplet_phi, tanh_sphere_phi,
    laplace_pressure, curvature, gradient, interface_delta,
)

# Figure 1: 2D droplet pressure field
print("Generating Figure 1: 2D droplet pressure field...")
R, L, sigma = 12, 48, 0.05
cfg = SolverConfig(sigma=sigma, tau_liquid=0.9, tau_gas=0.9, width=3.0, init_laplace=True)
solver = CapillarySolver((L, L), cfg)
solver.set_phi(tanh_droplet_phi((L, L), (L/2, L/2), R, width=3.0))
solver.step(1000)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Phi field
im0 = axes[0].imshow(solver.phi.numpy(), cmap='RdBu_r', origin='lower')
axes[0].set_title('Interface indicator $\\phi$')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
plt.colorbar(im0, ax=axes[0])

# Pressure field
p = solver.pressure.numpy()
im1 = axes[1].imshow(p, cmap='viridis', origin='lower')
axes[1].set_title('Pressure $p = \\rho c_s^2$')
axes[1].set_xlabel('x')
plt.colorbar(im1, ax=axes[1])

# Curvature field
kappa = curvature(solver.phi).numpy()
im2 = axes[2].imshow(kappa, cmap='RdBu_r', origin='lower', vmin=-0.2, vmax=0.2)
axes[2].set_title('Curvature $\\kappa$')
axes[2].set_xlabel('x')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig('artifacts/m11185/fig1_2d_droplet_fields.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved artifacts/m11185/fig1_2d_droplet_fields.png")

# Figure 2: Pressure profile across interface
print("Generating Figure 2: Pressure profile...")
x_line = p[L//2, :]
x_coords = np.arange(L)
r_coords = np.sqrt((x_coords - L/2)**2)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(r_coords, x_line, 'b-', linewidth=2, label='Simulated pressure')
p_in = laplace_pressure(sigma, R, ndim=2) + 1.0/3.0  # p_out = rho0 * cs2 = 1/3
p_out = 1.0/3.0
ax.axhline(p_in, color='r', linestyle='--', label=f'Young-Laplace: $p_{{in}} = {p_in:.5f}$')
ax.axhline(p_out, color='g', linestyle='--', label=f'$p_{{out}} = {p_out:.5f}$')
ax.axvline(R, color='k', linestyle=':', alpha=0.5, label=f'R = {R}')
ax.set_xlabel('Radial distance from center')
ax.set_ylabel('Pressure')
ax.set_title('2D Droplet: Pressure Jump Across Interface')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('artifacts/m11185/fig2_pressure_profile.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved artifacts/m11185/fig2_pressure_profile.png")

# Figure 3: Convergence schematic (will be updated with actual data)
print("Generating Figure 3: Convergence schematic...")
fig, ax = plt.subplots(figsize=(8, 5))
# Placeholder - will be updated with cluster results
h_vals = [1/8, 1/12, 1/16, 1/24]
u_spurious = [1e-3, 5e-4, 2e-4, 1e-4]  # Example values
ax.loglog([1/h for h in h_vals], u_spurious, 'bo-', linewidth=2, markersize=8, label='Spurious velocity')
# Reference slope
h_ref = np.array([8, 12, 16, 24])
u_ref = 1e-3 * (8/h_ref)**2
ax.loglog(h_ref, u_ref, 'r--', linewidth=1.5, label='$O(h^2)$ reference')
ax.set_xlabel('Grid resolution (1/h)')
ax.set_ylabel('Max spurious velocity |u|')
ax.set_title('Spurious Current Convergence')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('artifacts/m11185/fig3_spurious_convergence.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved artifacts/m11185/fig3_spurious_convergence.png")

print("\nAll figures generated.")
