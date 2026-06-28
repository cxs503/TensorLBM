"""Grid-refinement study: does Cd converge to the reference?

Runs the IBM (BFL) sphere with and without the DG band at two resolutions,
keeping Re approximately fixed.  If Cd -> ref as the grid refines, the error
is resolution-limited; if the band-inflated Cd persists, the band is the issue.

    PYTHONPATH=src python examples/dg_sphere_refinement.py
"""
from __future__ import annotations

import math

import torch

from tensorlbm.d3q19 import C as C3D, OPPOSITE as OPP3D, W as W3D, equilibrium3d
from tensorlbm.dg_advection import get_ops
from tensorlbm.dg_band import build_band_topology, hybrid_step
from tensorlbm.boundaries3d import make_channel_wall_mask_3d, sphere_mask, apply_zou_he_channel_boundaries_3d
from tensorlbm.interpolated_bc import bouzidi_bounce_back_3d, compute_q_sphere
from tensorlbm.solver3d import correct_mass3d


def cd_ref(re):
    return 24.0 / re * (1 + 0.15 * re ** 0.687) + 0.42 / (1 + 4.25e4 / re ** 1.16)


def run(radius, use_band, tau_lbm=0.8, n_steps=1000, device="cuda"):
    nu = (tau_lbm - 0.5) / 3.0
    u_in = 0.8 / radius                 # keeps Re = u*2r/nu ≈ 2r*0.8/r/0.1 = 16
    re = u_in * 2 * radius / nu
    nx, ny, nz = int(12 * radius), int(8 * radius), int(8 * radius)
    cx, cy, cz = nx * 0.25, ny * 0.5, nz * 0.5
    solid = sphere_mask(nx, ny, nz, cx, cy, cz, radius, device=device)
    fluid_bc, q_field = compute_q_sphere(nx, ny, nz, cx, cy, cz, radius, device=device)
    wall_mask = make_channel_wall_mask_3d(nz, ny, nx, solid, device=device)
    c_dev = C3D.to(device).float()

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device); ux0[solid] = 0.0
    f_lbm = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    f_eq_solid = equilibrium3d(torch.ones_like(rho0), torch.zeros_like(rho0), torch.zeros_like(rho0), torch.zeros_like(rho0), device=device)
    initial_mass = float(f_lbm.sum().item())
    dyn_p = 0.5 * u_in ** 2 * math.pi * radius ** 2

    band_topo = None; f_dg = None; ops = None
    if use_band:
        zz, yy, xx = torch.meshgrid(torch.arange(nz, device=device, dtype=torch.float32),
                                    torch.arange(ny, device=device, dtype=torch.float32),
                                    torch.arange(nx, device=device, dtype=torch.float32), indexing="ij")
        dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
        band = (dist >= radius + 3) & (dist < radius + 7) & ~solid
        band_topo = build_band_topology(band, solid_mask=solid, periodic=False).to(device)
        ops = get_ops(degree=1, dx=1.0, dtype=torch.float32, device=device)
        cb = band_topo.band_coords
        f_dg = f_lbm[:, cb[:, 0], cb[:, 1], cb[:, 2]].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2, 2).contiguous()
    C_d = C3D.to(torch.float32).to(device); W_d = W3D.to(torch.float32).to(device); opp = OPP3D.to(device)

    fx_all = []
    for step in range(1, n_steps + 1):
        f_lbm[:, solid] = f_eq_solid[:, solid]
        if use_band:
            f_lbm, f_dg = hybrid_step(f_lbm, f_dg, C_d, W_d, ops, band_topo, tau_lbm=tau_lbm,
                                      dt=1.0, n_substeps=10, opposite=opp)
        else:
            from tensorlbm.solver3d import collide_bgk3d, stream3d
            f_lbm = collide_bgk3d(f_lbm, tau_lbm); f_lbm = stream3d(f_lbm)
        f_pre = f_lbm.clone()
        for d in range(1, 19):
            if bool(fluid_bc[d].any()):
                f_lbm = bouzidi_bounce_back_3d(f_lbm, f_pre, fluid_bc[d], q_field[d], d)
        f_lbm = apply_zou_he_channel_boundaries_3d(f_lbm, u_in=u_in, wall_mask=wall_mask, obstacle_mask=torch.zeros_like(solid))
        if step % 200 == 0:
            f_lbm = correct_mass3d(f_lbm, initial_mass)
        fx_b = 0.0
        for d in range(1, 19):
            if bool(fluid_bc[d].any()):
                delta = f_lbm[d][fluid_bc[d]] - f_pre[d][fluid_bc[d]]
                fx_b -= float((delta * c_dev[d, 0]).sum().item())
        if step > 300:
            fx_all.append(fx_b)
    cd = sum(fx_all) / max(len(fx_all), 1) / dyn_p
    return re, cd


if __name__ == "__main__":
    print(f"{'radius':>7} {'mode':>10} {'Re':>6} {'Cd':>8} {'Cd_ref':>8} {'err%':>7}")
    for r in (8.0, 12.0):
        for use_band, label in ((False, "BFL-only"), (True, "BFL+DGband")):
            re, cd = run(r, use_band)
            ref = cd_ref(re)
            print(f"{r:>7.0f} {label:>10} {re:>6.1f} {cd:>8.3f} {ref:>8.3f} {abs(cd-ref)/ref*100:>7.1f}")
