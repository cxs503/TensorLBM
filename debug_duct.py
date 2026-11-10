"""Debug square duct analytical solution."""
import torch
import sys
sys.path.insert(0, '/root/tp-agent/work/m8122/src')

from tensorlbm.yield_stress_analytical_3d import bingham_square_duct_profile, newtonian_square_duct_profile

# Test Newtonian case first
N = 8
ax = 1e-5
mu = 0.02

u_newton, y, z = newtonian_square_duct_profile(N, ax, mu)
print(f"Newtonian square duct N={N}")
print(f"u shape: {u_newton.shape}")
print(f"u max: {u_newton.max():.6e}")
print(f"u min: {u_newton.min():.6e}")
print(f"u mean: {u_newton.mean():.6e}")
print(f"y: {y}")
print(f"z: {z}")
print(f"u[:, N//2]: {u_newton[:, N//2]}")

# Expected: parabolic profile with max at center
# For 2D channel: u_max = ax * H^2 / (8 * mu) = 1e-5 * 64 / 0.16 = 4e-5
# For square duct: u_max ≈ 0.42 * u_max_2d ≈ 1.68e-5
print(f"\nExpected u_max (2D channel): {ax * N**2 / (8 * mu):.6e}")
print(f"Expected u_max (square duct ~0.42x): {0.42 * ax * N**2 / (8 * mu):.6e}")

# Test Bingham case
tau_y = 2e-6
m = 100.0
u_bingham, y, z = bingham_square_duct_profile(N, ax, tau_y, mu)
print(f"\nBingham square duct N={N}, tau_y={tau_y}")
print(f"u max: {u_bingham.max():.6e}")
print(f"u min: {u_bingham.min():.6e}")
print(f"u[:, N//2]: {u_bingham[:, N//2]}")
