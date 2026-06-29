"""Cylinder Cd with far-field BC (no blockage) — vs channel-wall baseline.

Compares the 2D cylinder Re=100 drag with channel walls (blockage ~0.24)
versus far-field lateral BC.  The far-field removes blockage and should
bring Cd closer to the Williamson reference (~1.38).

    PYTHONPATH=src python examples/cylinder_farfield_bench.py
"""
from __future__ import annotations

import math

import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.boundaries import far_field_bc_2d, apply_simple_channel_boundaries, make_channel_wall_mask
from tensorlbm.solver import collide_bgk, stream, correct_mass

REF_CD = 1.38   # Williamson (1988) Re=100
REF_ST = 0.166


def cylinder_mask(ny, nx, cx, cy, radius, device):
    yy, xx = torch.meshgrid(torch.arange(ny, device=device, dtype=torch.float32),
                            torch.arange(nx, device=device, dtype=torch.float32), indexing="ij")
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def run(nx=320, ny=200, radius=12, u_in=0.08, re=100, n_steps=4000, far_field=True, device="cpu"):
    nu = u_in * 2 * radius / re
    tau = 3 * nu + 0.5
    cx, cy = nx * 0.25, ny * 0.5
    solid = cylinder_mask(ny, nx, cx, cy, radius, device)
    rho0 = torch.ones(ny, nx, device=device)
    ux0 = torch.full((ny, nx), u_in, device=device); ux0[solid] = 0
    f = equilibrium(rho0, ux0, torch.zeros_like(ux0)).to(torch.float64)
    initial_mass = float(rho0.sum().item())
    dyn_p = 0.5 * u_in ** 2 * 2 * radius   # frontal area (2D: diameter)
    from tensorlbm.d2q9 import C as C2D
    c_dev = C2D.to(device).float()
    cd_samples = []
    cl_prev = 0; cl_list = []; t_shed = []

    for step in range(1, n_steps + 1):
        f = collide_bgk(f, tau)
        f = stream(f)
        if far_field:
            f = far_field_bc_2d(f, u_in, solid)
        else:
            wall = make_channel_wall_mask(ny, nx, solid, device)
            f = apply_simple_channel_boundaries(f, u_in, wall, solid)
        if step % 200 == 0:
            f = correct_mass(f, initial_mass)
        if step > 1500:
            rho, ux, uy = macroscopic(f)
            fx = 2 * (c_dev[:, 0].view(9,1,1) * f * solid.unsqueeze(0)).sum().item()
            fy = 2 * (c_dev[:, 1].view(9,1,1) * f * solid.unsqueeze(0)).sum().item()
            cd = -fx / dyn_p; cl = -fy / dyn_p
            cd_samples.append(cd)
            # detect zero-crossing of Cl for St
            if cl_prev * cl < 0:
                t_shed.append(step)
            cl_prev = cl; cl_list.append(cl)
    cd_mean = sum(cd_samples) / len(cd_samples)
    if len(t_shed) > 2:
        periods = [t_shed[i+1]-t_shed[i] for i in range(len(t_shed)-1)]
        T = sum(periods)/len(periods)
        st = 1.0 / T   # St = f·D/U, f=1/T, D=2r, U=u_in → St = 2r/(T·u_in)
        st *= 2 * radius / u_in
    else:
        st = float('nan')
    mode = "far-field" if far_field else "channel"
    print(f"  {mode:>12}: Cd={cd_mean:.3f} (ref {REF_CD}, err {abs(cd_mean-REF_CD)/REF_CD*100:.1f}%)  "
          f"St={st:.4f} (ref {REF_ST}, err {abs(st-REF_ST)/REF_ST*100:.1f}%)")


if __name__ == "__main__":
    device = "cuda"
    print("Cylinder Re=100: channel-wall (blockage 0.12) vs far-field (no blockage)")
    run(far_field=False, device=device)
    run(far_field=True, device=device)
