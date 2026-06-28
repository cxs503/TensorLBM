"""IBM (BFL) smooth sphere in the real-DG-LBM — Cd vs Schiller-Naumann/Clift.

The staircased boolean mask is replaced by Bouzidi-Firdaouss-Lallemand
interpolated bounce-back (smooth wall, fractional distance q from
:func:`compute_q_sphere`).  The DG band sits as a fluid-refinement shell
*offset* from the sphere (it does not enclose the obstacle), so the BFL
boundary cells stay in the exterior LBM and the two methods compose cleanly.

    PYTHONPATH=src python examples/dg_lbm_sphere_ibm.py
"""
from __future__ import annotations

import math

import torch

from tensorlbm.d3q19 import C as C3D, OPPOSITE as OPP3D, W as W3D, equilibrium3d, macroscopic3d
from tensorlbm.dg_advection import get_ops
from tensorlbm.dg_band import build_band_topology, hybrid_step
from tensorlbm.boundaries3d import make_channel_wall_mask_3d, sphere_mask
from tensorlbm.interpolated_bc import bouzidi_bounce_back_3d, compute_q_sphere
from tensorlbm.solver3d import correct_mass3d


def cd_ref(re):
    return 24.0 / re * (1 + 0.15 * re ** 0.687) + 0.42 / (1 + 4.25e4 / re ** 1.16)


def run(nz=48, ny=48, nx=96, radius=8.0, band_offset=3, band_thickness=4,
        u_in=0.1, tau_lbm=0.8, n_steps=1500, device="cuda"):
    nu = (tau_lbm - 0.5) / 3.0
    re = u_in * 2 * radius / nu
    cx, cy, cz = nx * 0.25, ny * 0.5, nz * 0.5
    solid = sphere_mask(nx, ny, nz, cx, cy, cz, radius, device=device)

    # DG band: fluid shell at distance [r+offset, r+offset+thickness] from centre
    # (a near-wall refinement ring, NOT enclosing the sphere).
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
    band = (dist >= radius + band_offset) & (dist < radius + band_offset + band_thickness) & ~solid
    topo = build_band_topology(band, solid_mask=solid, periodic=False).to(device)
    ops = get_ops(degree=1, dx=1.0, dtype=torch.float32, device=device)

    # BFL boundary data (smooth sphere)
    fluid_bc, q_field = compute_q_sphere(nx, ny, nz, cx, cy, cz, radius, device=device)
    wall_mask = make_channel_wall_mask_3d(nz, ny, nx, solid, device=device)

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device); ux0[solid] = 0.0
    f_lbm = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    cb = topo.band_coords
    f_dg = f_lbm[:, cb[:, 0], cb[:, 1], cb[:, 2]].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2, 2).contiguous()

    C_d = C3D.to(torch.float32).to(device)
    W_d = W3D.to(torch.float32).to(device)
    opp = OPP3D.to(device)
    c_dev = C3D.to(device).float()
    initial_mass = float(f_lbm.sum().item())
    dyn_p = 0.5 * u_in ** 2 * math.pi * radius ** 2
    fx_all: list[float] = []

    for step in range(1, n_steps + 1):
        f_lbm[:, solid] = equilibrium3d(torch.ones_like(rho0), torch.zeros_like(rho0),
                                        torch.zeros_like(rho0), torch.zeros_like(rho0), device=device)[:, solid]
        f_lbm, f_dg = hybrid_step(f_lbm, f_dg, C_d, W_d, ops, topo, tau_lbm=tau_lbm,
                                  dt=1.0, n_substeps=10, opposite=opp)
        # BFL smooth-wall bounce-back at the sphere (exterior LBM boundary cells).
        f_pre = f_lbm.clone()
        for d in range(1, 19):
            if bool(fluid_bc[d].any()):
                f_lbm = bouzidi_bounce_back_3d(f_lbm, f_pre, fluid_bc[d], q_field[d], d)
        # inlet/outlet/walls (no obstacle bounce-back — BFL handles the sphere)
        from tensorlbm.boundaries3d import apply_zou_he_channel_boundaries_3d
        f_lbm = apply_zou_he_channel_boundaries_3d(f_lbm, u_in=u_in, wall_mask=wall_mask,
                                                   obstacle_mask=torch.zeros_like(solid))
        if step % 200 == 0:
            f_lbm = correct_mass3d(f_lbm, initial_mass)
        # BFL momentum-injection force on the solid
        fx_b = 0.0
        for d in range(1, 19):
            if bool(fluid_bc[d].any()):
                bm = fluid_bc[d]
                delta = f_lbm[d][bm] - f_pre[d][bm]
                fx_b -= float((delta * c_dev[d, 0]).sum().item())
        if step > 300:
            fx_all.append(fx_b)
        if step % 300 == 0 or step == n_steps:
            cd = sum(fx_all[-300:]) / max(min(len(fx_all), 300), 1) / dyn_p
            print(f"step {step:4d}: Cd={cd:.4f}  (ref {cd_ref(re):.4f})")

    cd = sum(fx_all) / max(len(fx_all), 1) / dyn_p
    print(f"\nRe={re:.1f}  Cd={cd:.4f}  ref={cd_ref(re):.4f}  err={abs(cd-cd_ref(re))/cd_ref(re)*100:.1f}%")


if __name__ == "__main__":
    run()
