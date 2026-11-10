"""Check 2D channel flow - quick test."""
import torch
import sys
sys.path.insert(0, '/root/tp-agent/work/m8122/src')

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, OPPOSITE
from tensorlbm.solver3d import stream3d
from tensorlbm.yield_stress_lbm_3d import collide_bingham_bgk_forced_3d

# Simple 2D channel
N = 8
ny = N + 2
nx = 4
nz = 4

wall = torch.zeros((nz, ny, nx), dtype=torch.bool)
wall[:, 0, :] = True
wall[:, -1, :] = True

ax = 1e-5
mu_0 = 0.02
tau_y = 0.0
m = 100.0

rho = torch.ones((nz, ny, nx), dtype=torch.float64)
ux = torch.zeros((nz, ny, nx), dtype=torch.float64)
uy = torch.zeros((nz, ny, nx), dtype=torch.float64)
uz = torch.zeros((nz, ny, nx), dtype=torch.float64)
f = equilibrium3d(rho, ux, uy, uz)

print("Running 2D channel simulation...")
for step in range(2000):
    f = collide_bingham_bgk_forced_3d(f, tau_y, mu_0, m, ax)
    f = stream3d(f)
    f = torch.where(wall.unsqueeze(0), f[OPPOSITE], f)
    
    if step % 500 == 0:
        rho, ux, uy, uz = macroscopic3d(f)
        print(f"Step {step}: ux max = {ux.max():.6e}")

rho, ux, uy, uz = macroscopic3d(f)
z_mid = nz // 2
x_mid = nx // 2
u_sim = ux[z_mid, 1:-1, x_mid]

print(f"\n2D channel N={N}, ax={ax}, mu={mu_0}")
print(f"u_sim: {u_sim}")
print(f"u_sim max: {u_sim.max():.6e}")

u_max_expected = ax * N**2 / (8 * mu_0)
print(f"Expected u_max: {u_max_expected:.6e}")

y_phys = torch.arange(1, N+1, dtype=torch.float64) - 0.5
H = float(N)
u_analytic = ax / (2 * mu_0) * y_phys * (H - y_phys)
print(f"u_analytic: {u_analytic}")
print(f"Ratio sim/analytic: {u_sim.max() / u_analytic.max():.4f}")
