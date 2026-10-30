#!/usr/bin/env python3
import sys, time, math
sys.path.insert(0, '/root/tp-agent/work/m217/src')
sys.path.insert(0, '/root/tp-agent/work/m217/benchmarks')
import torch
import numpy as np
from tensorlbm.boundaries import zou_he_inlet_velocity
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream

device = 'sdaa'
dev = torch.device(device)
dt = torch.float32
ny, nx = 128, 384
re_val, u0_val, xmax = 40.0, 0.03, 3.0
lam = re_val/2.0 - math.sqrt(re_val**2/4.0 + 4.0*math.pi**2)
nu = u0_val * ny / re_val
tau = 0.5 + 3.0 * nu

y = torch.arange(ny, device=dev, dtype=dt)
x = torch.arange(nx, device=dev, dtype=dt)
yy, xx = torch.meshgrid(y, x, indexing='ij')
xp = xx / nx * xmax
yp = yy / ny
e = torch.exp(lam * xp)
u_ana = u0_val * (1.0 - e * torch.cos(2.0*math.pi*yp))
v_ana = u0_val * (lam / (2.0*math.pi)) * e * torch.sin(2.0*math.pi*yp)

f = equilibrium(torch.ones((ny, nx), device=dev, dtype=dt), u_ana, v_ana)
u_in = u_ana[:, 0].contiguous()
v_in = v_ana[:, 0].contiguous()

t0 = time.time()
for step in range(1, 20001):
    f = collide_bgk(f, tau)
    f = stream(f)
    f[:, :, -1] = f[:, :, -2]
    f = zou_he_inlet_velocity(f, u_in, v_in)
    if step % 5000 == 0:
        print(f'step {step}: {time.time()-t0:.1f}s', flush=True)

_, u_num, v_num = macroscopic(f)
np.save('artifacts/kovasznay_2d_sdaa_v2/u_num_ny128.npy', u_num.cpu().numpy())
np.save('artifacts/kovasznay_2d_sdaa_v2/v_num_ny128.npy', v_num.cpu().numpy())
np.save('artifacts/kovasznay_2d_sdaa_v2/u_ana_ny128.npy', u_ana.cpu().numpy())
np.save('artifacts/kovasznay_2d_sdaa_v2/v_ana_ny128.npy', v_ana.cpu().numpy())
print(f'Fields saved. Total time: {time.time()-t0:.1f}s')
