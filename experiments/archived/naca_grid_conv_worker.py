#!/usr/bin/env python3
'''NACA 0012 Re=1000 grid convergence — parallel on SDAA 0-4'''
import sys, math, time
import torch

sys.path.insert(0, 'src')
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.drag_pressure import get_near_wall_2d, SurfaceMesh, drag_pressure_integration, drag_friction_integration

def build_naca(chord, nx, ny, x_le, y_c, device):
    solid = torch.zeros(4, ny, nx, dtype=torch.bool, device=device)
    for k in range(4):
        for i in range(nx):
            xc = (i - x_le) / chord
            if 0 <= xc <= 1:
                yt = 0.6 * (0.2969*math.sqrt(xc) - 0.1260*xc - 0.3516*xc**2 + 0.2843*xc**3 - 0.1015*xc**4)
                j_lo = max(0, int(y_c - yt*chord))
                j_hi = min(ny-1, int(y_c + yt*chord))
                solid[k, j_lo:j_hi+1, i] = True
    return solid

chord = int(sys.argv[1])
did = int(sys.argv[2])
nx = int(chord * 6)
ny = int(chord * 2)
nz = 4
u_in = 0.05
re = 1000
nu = u_in * chord / re
tau = 3 * nu + 0.5
ref = 0.05
x_le = int(nx * 0.25)
y_c = ny // 2

torch.sdaa.set_device(did)
d = torch.device(f'sdaa:{did}')

solid = build_naca(chord, nx, ny, x_le, y_c, d)
near = get_near_wall_2d(solid)
mesh = SurfaceMesh.from_gradient(solid, near)
sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
dpS = 0.5 * u_in**2 * chord * nz

r0 = torch.ones(nz, ny, nx, device=d)
u0 = torch.full((nz, ny, nx), u_in, device=d)
u0[solid] = 0
f = equilibrium3d(r0, u0, torch.zeros_like(u0), torch.zeros_like(u0))
im = float(r0.sum().item())

t0 = time.time()
for step in range(1, 10001):
    f_pre = f.clone()
    f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=0.05)
    for q in range(19):
        f[q] = torch.where(sm[q], f_pre[q], f[q])
    f = bounce_back_cells_3d(f, solid)
    f = stream3d(f)
    f = far_field_bc_3d(f, u_in)
    if step % 200 == 0:
        f = correct_mass3d(f, im)
    if step % 2000 == 0:
        print(f'  chord{chord} step={step} ({time.time()-t0:.0f}s)', flush=True)

fx_p, _, _ = drag_pressure_integration(f, mesh, dpS)
fx_f, _, _ = drag_friction_integration(f, mesh, dpS, nu)
tot = fx_p + fx_f
err = abs(tot - ref) / ref * 100
print(f'DONE chord{chord} Cd_p={fx_p:.4f} Cd_f={fx_f:.4f} Cd_tot={tot:.4f} err={err:.1f}% ({time.time()-t0:.0f}s)', flush=True)
