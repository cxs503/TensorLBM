"""Fair head-to-head: standard LBM vs DG-LBM, same far-field BC + grid.

Both use BFL (smooth sphere) + far-field BC + Smagorinsky-MRT exterior.  The
DG-LBM adds a near-wall DG band (hybrid_step).  If DG-LBM has an advantage it
shows at COARSE grids (higher-order ⇒ faster Cd convergence).  At fine grids
both should converge to the reference.

    PYTHONPATH=src python examples/dg_vs_standard_sphere.py
"""
from __future__ import annotations

import math

import torch

from tensorlbm.d3q19 import C as C3D, OPPOSITE as OPP3D, W as W3D, equilibrium3d, macroscopic3d
from tensorlbm.dg_advection import get_ops
from tensorlbm.dg_band import build_band_topology, hybrid_step
from tensorlbm.boundaries3d import make_channel_wall_mask_3d, sphere_mask, far_field_bc_3d
from tensorlbm.interpolated_bc import bouzidi_bounce_back_3d, compute_q_sphere
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d


def cd_ref(re):
    return 24.0 / re * (1 + 0.15 * re ** 0.687) + 0.42 / (1 + 4.25e4 / re ** 1.16)


def run(radius, use_dg, tau_lbm=0.78, u_in=0.1, cs=0.1, n_steps=2000, warmup=800, device="cuda"):
    # Fix tau_lbm so τ_dg = τ_lbm − 0.5 stays in the DG-stable range; Re follows.
    tau = tau_lbm
    nu = (tau_lbm - 0.5) / 3.0
    re = u_in * 2 * radius / nu
    nx, ny, nz = int(10 * radius), int(6 * radius), int(6 * radius)
    cx, cy, cz = nx * 0.3, ny / 2.0, nz / 2.0
    solid = sphere_mask(nx, ny, nz, cx, cy, cz, radius, device=device)
    fluid_bc, q_field = compute_q_sphere(nx, ny, nz, cx, cy, cz, radius, device=device)
    c_dev = C3D.to(device).float()

    rho0 = torch.ones(nz, ny, nx, device=device)
    f_eq_solid = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0), torch.zeros_like(rho0), device=device)
    f = equilibrium3d(rho0, torch.full_like(rho0, u_in), torch.zeros_like(rho0), torch.zeros_like(rho0), device=device)
    initial_mass = float(f.sum().item())
    dyn_p = 0.5 * u_in ** 2 * math.pi * radius ** 2

    band_topo = None; f_dg = None; ops = None
    if use_dg:
        zz, yy, xx = torch.meshgrid(torch.arange(nz, device=device, dtype=torch.float32),
                                    torch.arange(ny, device=device, dtype=torch.float32),
                                    torch.arange(nx, device=device, dtype=torch.float32), indexing="ij")
        dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
        band = (dist >= radius + 2) & (dist < radius + 6) & ~solid
        band_topo = build_band_topology(band, solid_mask=solid, periodic=False).to(device)
        ops = get_ops(degree=1, dx=1.0, dtype=torch.float32, device=device)
        cb = band_topo.band_coords
        f_dg = f[:, cb[:, 0], cb[:, 1], cb[:, 2]].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2, 2).contiguous()
    C_d = C3D.to(torch.float32).to(device); W_d = W3D.to(torch.float32).to(device); opp = OPP3D.to(device)
    collide = lambda f, t: collide_smagorinsky_mrt3d(f, t, cs)

    fx_all = []
    for step in range(1, n_steps + 1):
        f[:, solid] = f_eq_solid[:, solid]
        if use_dg:
            f, f_dg = hybrid_step(f, f_dg, C_d, W_d, ops, band_topo, tau_lbm=tau,
                                  dt=1.0, n_substeps=10, opposite=opp, collide_fn=collide)
        else:
            f = collide_smagorinsky_mrt3d(f, tau, cs); f = stream3d(f)
        f = far_field_bc_3d(f, u_in=u_in)                       # far-field (no blockage)
        f_pre = f.clone()
        for d in range(1, 19):
            if bool(fluid_bc[d].any()):
                f = bouzidi_bounce_back_3d(f, f_pre, fluid_bc[d], q_field[d], d)
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)
        if step > warmup:
            fx_b = 0.0
            for d in range(1, 19):
                if bool(fluid_bc[d].any()):
                    delta = f[d][fluid_bc[d]] - f_pre[d][fluid_bc[d]]
                    fx_b -= float((delta * c_dev[d, 0]).sum().item())
            fx_all.append(fx_b)
    return sum(fx_all) / max(len(fx_all), 1) / dyn_p


if __name__ == "__main__":
    print("Sphere, far-field BC, tau_lbm=0.78 (tau_dg=0.28, DG-stable), Re varies with radius\n")
    print(f"{'radius':>7} {'Re':>5} {'std Cd':>8} {'err%':>6}  {'DG Cd':>8} {'err%':>6}  {'DG adv':>7}")
    for r in (8.0, 12.0, 16.0):
        cd_std = run(r, use_dg=False)
        cd_dg = run(r, use_dg=True)
        nu = (0.78 - 0.5) / 3.0; re = 0.1 * 2 * r / nu
        ref = cd_ref(re)
        e_std = abs(cd_std-ref)/ref*100; e_dg = abs(cd_dg-ref)/ref*100
        adv = (e_std - e_dg) / e_std * 100 if e_std > 1e-9 else float('nan')
        print(f"{r:>7.0f} {re:>5.0f} {cd_std:>8.3f} {e_std:>6.1f}  {cd_dg:>8.3f} {e_dg:>6.1f}  {adv:>+7.1f}%")
