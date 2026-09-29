#!/usr/bin/env python3
"""NACA 0012 chord400 — 4-card parallel (domain decomposition in x).

Splits the x-domain across 4 SDAA cards (same P2P group).
After streaming, exchanges 1 halo cell between adjacent cards.
Each card computes its part of the drag.
"""
import sys, math, time
import torch

sys.path.insert(0, 'src')
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C, W
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.drag_pressure import get_near_wall_2d, SurfaceMesh, drag_pressure_integration, drag_friction_integration

def build_naca_strip(chord, nx_local, x_offset, ny, y_c, device):
    """Build NACA solid mask for a local x-strip."""
    solid = torch.zeros(4, ny, nx_local, dtype=torch.bool, device=device)
    for k in range(4):
        for i_local in range(nx_local):
            i_global = x_offset + i_local
            xc = (i_global - int(chord * 6 * 0.25)) / chord
            if 0 <= xc <= 1:
                yt = 0.6 * (0.2969*math.sqrt(xc) - 0.1260*xc - 0.3516*xc**2 + 0.2843*xc**3 - 0.1015*xc**4)
                j_lo = max(0, int(y_c - yt*chord))
                j_hi = min(ny-1, int(y_c + yt*chord))
                solid[k, j_lo:j_hi+1, i_local] = True
    return solid

chord = 400
nx_total = int(chord * 6)  # 2400
ny = int(chord * 2)        # 800
nz = 4
n_cards = 4
cards = [4, 5, 6, 7]  # Same P2P group

u_in = 0.05
re = 1000
nu = u_in * chord / re
tau = 3 * nu + 0.5
ref = 0.05
x_le = int(nx_total * 0.25)
y_c = ny // 2

# Domain decomposition in x
nx_local = nx_total // n_cards  # 600 per card
remainder = nx_total % n_cards

print(f'NACA chord={chord} nx_total={nx_total} ny={ny} 4-card parallel', flush=True)
print(f'  nx_local={nx_local} per card, tau={tau:.4f}', flush=True)

# Initialize each card
devices = []
f_list = []
solid_list = []
near_list = []
mesh_list = []
sm_list = []
im_list = []

for rank in range(n_cards):
    did = cards[rank]
    torch.sdaa.set_device(did)
    d = torch.device(f'sdaa:{did}')
    devices.append(d)

    x_start = rank * nx_local
    x_end = x_start + nx_local
    if rank == n_cards - 1:
        x_end = nx_total  # last card gets remainder

    nx_l = x_end - x_start
    solid = build_naca_strip(chord, nx_l, x_start, ny, y_c, d)
    near = get_near_wall_2d(solid)
    mesh = SurfaceMesh.from_gradient(solid, near)
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx_l)

    r0 = torch.ones(nz, ny, nx_l, device=d)
    u0 = torch.full((nz, ny, nx_l), u_in, device=d)
    u0[solid] = 0
    f = equilibrium3d(r0, u0, torch.zeros_like(u0), torch.zeros_like(u0))
    im = float(r0.sum().item())

    f_list.append(f)
    solid_list.append(solid)
    near_list.append(near)
    mesh_list.append(mesh)
    sm_list.append(sm)
    im_list.append(im)
    print(f'  rank={rank} SDAA:{did} x=[{x_start}:{x_end}] solid={solid.sum().item()}', flush=True)

# Halo exchange function
def halo_exchange(f_list, devices):
    """Exchange 1 halo cell in x-direction between adjacent cards."""
    for rank in range(n_cards - 1):
        # rank sends right edge to rank+1's left
        # rank+1 sends left edge to rank's right
        # f shape: (19, nz, ny, nx_local)
        right_edge = f_list[rank][:, :, :, -1:].to(devices[rank+1])
        left_edge = f_list[rank+1][:, :, :, :1].to(devices[rank])
        f_list[rank+1][:, :, :, :1] = right_edge
        f_list[rank][:, :, :, -1:] = left_edge

t0 = time.time()
for step in range(1, 10001):
    for rank in range(n_cards):
        torch.sdaa.set_device(cards[rank])
        f_pre = f_list[rank].clone()
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=0.05)
        sm = sm_list[rank]
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid_list[rank])
        f = stream3d(f)
        f_list[rank] = f

    # Halo exchange after streaming
    halo_exchange(f_list, devices)

    # BC on boundary cards
    # rank 0: inlet (x=0)
    # rank n-1: outlet (x=-1)
    torch.sdaa.set_device(cards[0])
    rho1 = torch.ones(nz, ny, nx_local, device=devices[0])
    feq = equilibrium3d(rho1, torch.full_like(rho1, u_in), torch.zeros_like(rho1), torch.zeros_like(rho1))
    f_list[0][:, :, :, 0] = feq[:, :, :, 0]

    torch.sdaa.set_device(cards[-1])
    f_list[-1][:, :, :, -1] = f_list[-1][:, :, :, -2]

    # far_field_bc on y boundaries for all cards
    for rank in range(n_cards):
        torch.sdaa.set_device(cards[rank])
        f = f_list[rank]
        nz_l, ny_l, nx_l = f.shape[1:]
        rho1 = torch.ones(nz_l, ny_l, nx_l, device=devices[rank])
        feq = equilibrium3d(rho1, torch.full_like(rho1, u_in), torch.zeros_like(rho1), torch.zeros_like(rho1))
        f[:, :, 0, :] = feq[:, :, 0, :]
        f[:, :, -1, :] = feq[:, :, -1, :]
        f_list[rank] = f

    # correct_mass every 200 steps
    if step % 200 == 0:
        for rank in range(n_cards):
            torch.sdaa.set_device(cards[rank])
            f_list[rank] = correct_mass3d(f_list[rank], im_list[rank])

    if step % 2000 == 0:
        print(f'  step={step} ({time.time()-t0:.0f}s)', flush=True)

# Compute drag on each card, sum
total_p = 0
total_f = 0
dpS = 0.5 * u_in**2 * chord * nz
for rank in range(n_cards):
    torch.sdaa.set_device(cards[rank])
    f = f_list[rank]
    mesh = mesh_list[rank]
    fx_p, _, _ = drag_pressure_integration(f, mesh, dpS)
    fx_f, _, _ = drag_friction_integration(f, mesh, dpS, nu)
    total_p += fx_p
    total_f += fx_f

tot = total_p + total_f
err = abs(tot - ref) / ref * 100
print(f'DONE chord{chord} Cd_p={total_p:.4f} Cd_f={total_f:.4f} Cd_tot={tot:.4f} err={err:.1f}% ({time.time()-t0:.0f}s)', flush=True)
