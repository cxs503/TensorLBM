"""Measure per-step cost of the verified cylinder Re=100 stepper on CPU (short probe)."""
import os
import sys
import time

import torch

sys.path.insert(0, "src")

from tensorlbm.boundaries import (  # noqa: E402
    compute_obstacle_forces,
    cylinder_mask,
    far_field_bc_2d,
    make_sponge_strength,
)
from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.solver import collide_mrt, stream  # noqa: E402

nthreads = int(sys.argv[1]) if len(sys.argv) > 1 else 32
nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
torch.set_num_threads(nthreads)
D = int(os.environ.get("D","48"))
ny = int(float(os.environ.get("HD","40"))*D); nx = int(float(os.environ.get("LD","40"))*D)
u_in = 0.05
nu = u_in * D / 100.0
tau = 0.5 + 3.0 * nu
mask = cylinder_mask(nx, ny, float(os.environ.get("UP","10")) * D, ny / 2.0, D / 2.0, torch.device("cpu"))
sig = make_sponge_strength(ny, nx, int(nx - 10 * D), int(10 * D), power=2.0, device=torch.device("cpu"))
tau_field = tau * (1.0 + 10.0 * sig)
rho0 = torch.ones((ny, nx))
f = equilibrium(rho0, torch.full_like(rho0, u_in), torch.zeros_like(rho0))
print("grid", ny, nx, "tau", round(tau, 4), "solid", int(mask.sum()))

t0 = time.time()
for _ in range(nsteps):
    f = collide_mrt(f, tau, tau_field=tau_field)
    f = stream(f)
    _fx, _fy = compute_obstacle_forces(f, mask)
    f = far_field_bc_2d(f, u_in, mask)
dt = (time.time() - t0) / nsteps
print(f"threads={nthreads} per-step {dt*1000:.1f} ms -> 60000 steps = {dt*60000/60:.1f} min")