"""Debug square duct simulation vs analytical."""
import torch
import sys
sys.path.insert(0, '/root/tp-agent/work/m8122/src')

from tensorlbm.yield_stress_channel_3d import run_bingham_square_duct_3d
from tensorlbm.yield_stress_analytical_3d import bingham_square_duct_profile

N = 8
ax = 1e-5
tau_y = 2e-6
mu_0 = 0.02
m = 100.0

print(f"Running simulation N={N}, ax={ax}, tau_y={tau_y}, mu_0={mu_0}, m={m}")
result = run_bingham_square_duct_3d(
    N, ax, tau_y, mu_0, m,
    max_iter=10000, tol=1e-6,
    dtype=torch.float64,
)

print(f"Converged: {result['converged']}")
print(f"Iterations: {result['iterations']}")
print(f"L2 error: {result['l2_error']:.6f}")
print(f"Linf error: {result['linf_error']:.6e}")
print(f"Linf rel: {result['linf_rel']:.6f}")

u_sim = result['u_sim']
u_analytic = result['analytic_u'][:, N//2]

print(f"\nu_sim shape: {u_sim.shape}")
print(f"u_analytic shape: {u_analytic.shape}")
print(f"u_sim: {u_sim}")
print(f"u_analytic: {u_analytic}")
print(f"u_sim max: {u_sim.max():.6e}")
print(f"u_analytic max: {u_analytic.max():.6e}")
print(f"Difference: {(u_sim - u_analytic).abs().max():.6e}")
