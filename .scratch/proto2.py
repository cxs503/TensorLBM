import math

import torch

torch.set_num_threads(1)
from tensorlbm.passive_scalar import (
    collide_scalar_bgk,
    equilibrium_scalar,
    macroscopic_scalar,
    stream_scalar,
)

ny, nx = 16, 128
D = 0.1
tau = 3.0 * D + 0.5
z = torch.zeros(ny, nx, dtype=torch.float64)
k = 2 * math.pi / nx
x = torch.arange(nx, dtype=torch.float64)
phi0 = 1.0 + 1e-3 * torch.cos(k * x).view(1, nx) * torch.ones(ny, 1)
g = equilibrium_scalar(phi0, z, z)
print("sum phi0 =", float(macroscopic_scalar(g).mean()))
nsteps = 2000
for s in range(nsteps):
    c = macroscopic_scalar(g)
    g = collide_scalar_bgk(g, c, z, z, tau)
    g = stream_scalar(g)
c = macroscopic_scalar(g)
cm = c.mean(dim=0)
re = float((cm * torch.cos(k * x)).sum())
im = float((cm * torch.sin(k * x)).sum())
amp = math.hypot(re, im)
a0 = float((phi0[0] * torch.cos(k * x)).sum())
rate = -math.log(amp / a0) / nsteps
print(f"periodic 1D diffusion: rate={rate:.6e}  D k^2={D*k**2:.6e}  ratio={rate/(D*k*k):.4f}")
print("total scalar drift:", float(macroscopic_scalar(g).mean() - 1.0))

# advection direction check: uniform u, single mode moving
u = 0.01
ux = torch.full((ny, nx), u, dtype=torch.float64)
phi0 = 1.0 + 1e-3 * torch.cos(k * x).view(1, nx) * torch.ones(ny, 1)
g = equilibrium_scalar(phi0, ux, z)
nsteps = 3000
for s in range(nsteps):
    c = macroscopic_scalar(g)
    g = collide_scalar_bgk(g, c, ux, z, 1.0)  # tau=1 -> D=1/6? use large tau to minimize diffusion
    g = stream_scalar(g)
c = macroscopic_scalar(g)
cm = c.mean(dim=0)
re = float((cm * torch.cos(k * x)).sum())
im = float((cm * torch.sin(k * x)).sum())
phase = math.atan2(im, re)
print(f"advection phase (expect {-u*k*nsteps:.4f} for +x motion): {phase:.4f}")