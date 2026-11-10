"""Check 2D channel flow to isolate the issue."""
import torch
import sys
sys.path.insert(0, '/root/tp-agent/work/m8122/src')

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, OPPOSITE
from tensorlbm.solver3d import stream3d
from tensorlbm.yield_stress_lbm_3d import collide_bingham_bgk_forced_3d

# Simple 2D channel (periodic in x and z, walls in y)
N = 8  # 8 fluid cells in y
ny = N + 2  # walls at y=0 and y=N+1
nx = 8
nz = 8

# Create wall mask (only y-walls)
wall = torch.zeros((nz, ny, nx), dtype=torch.bool)
wall[:, 0, :] = True
wall[:, -1, :] = True

# Parameters
ax = 1e-5
mu_0 = 0.02
tau_y = 0.0  # Newtonian
m = 100.0

# Initialize
rho = torch.ones((nz, ny, nx), dtype=torch.float64)
ux = torch.zeros((nz, ny, nx), dtype=torch.float64)
uy = torch.zeros((nz, ny, nx), dtype=torch.float64)
uz = torch.zeros((nz, ny, nx), dtype=torch.float64)
f = equilibrium3d(rho, ux, uy, uz)

# Run simulation
for step in range(5000):
    f = collide_bingham_bgk_forced_3d(f, tau_y, mu_0, m, ax)
    f = stream3d(f)
    # Bounce-back
    opp = OPPOSITE
    f = torch.where(wall.unsqueeze(0), f[opp], f)

rho, ux, uy, uz = macroscopic3d(f)

# Extract profile at mid-z, mid-x
z_mid = nz // 2
x_mid = nx // 2
u_sim = ux[z_mid, 1:-1, x_mid]

print(f"2D channel (periodic in x and z)")
print(f"N={N}, ax={ax}, mu={mu_0}")
print(f"u_sim: {u_sim}")
print(f"u_sim max: {u_sim.max():.6e}")

# Expected: parabolic profile u(y) = ax/(2*mu) * y * (H-y)
# where H = N, y is distance from bottom wall
# Cell centers at y = 0.5, 1.5, ..., N-0.5 (distance from bottom wall at y=0)
# u_max at y = N/2: u_max = ax * (N/2)^2 / (2*mu) = ax * N^2 / (8*mu)
u_max_expected = ax * N**2 / (8 * mu_0)
print(f"Expected u_max (2D channel): {u_max_expected:.6e}")

# Analytical profile
y_phys = torch.arange(1, N+1, dtype=torch.float64) - 0.5  # 0.5, 1.5, ..., N-0.5
H = float(N)
u_analytic = ax / (2 * mu_0) * y_phys * (H - y_phys)
print(f"u_analytic: {u_analytic}")
print(f"u_analytic max: {u_analytic.max():.6e}")
