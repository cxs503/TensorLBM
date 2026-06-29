"""Validate the DG-solid-interface force against the canonical sphere Cd.

The drag coefficient of a sphere is one of the best-characterised benchmarks:
  Re=20 -> Cd~1.35,  Re=50 -> Cd~0.95,  Re=100 -> Cd~1.09  (clift-gauvin).

This runs the real-DG hybrid sphere with the DG-interface momentum-exchange
force and compares Cd to the correlation.  If the force is correct, Cd lands
near the correlation; a systematic factor reveals a force bug.

    PYTHONPATH=src python examples/dg_lbm_sphere_force_validate.py
"""
from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F

from tensorlbm.d3q19 import C as C3D, OPPOSITE as OPP3D, W as W3D, equilibrium3d, macroscopic3d
from tensorlbm.dg_advection import get_ops
from tensorlbm.dg_band import build_band_topology, compute_dg_solid_force, hybrid_step
from tensorlbm.boundaries3d import apply_simple_channel_boundaries_3d, make_channel_wall_mask_3d, sphere_mask
from tensorlbm.dg_band import project_band_to_lbm
from tensorlbm.obstacles import compute_obstacle_forces_3d
from tensorlbm.solver3d import correct_mass3d


def dilate3d(mask, k):
    s = mask.float().unsqueeze(0).unsqueeze(0)
    return (F.max_pool3d(s, 2 * k + 1, 1, k).squeeze(0).squeeze(0) > 0.5) & ~mask


def sphere_cd_correlation(re):
    # Clift-Gauvin (laminar): Cd = 24/Re*(1+0.15*Re^0.687) + 0.42/(1+4.25e4/Re^1.16)
    return 24.0 / re * (1 + 0.15 * re ** 0.687) + 0.42 / (1 + 4.25e4 / re ** 1.16)


def run(nz=48, ny=48, nx=96, radius=8.0, band_thickness=3, u_in=0.1, tau_lbm=0.8,
        n_steps=600, device="cuda"):
    nu = (tau_lbm - 0.5) / 3.0
    re = u_in * 2 * radius / nu
    cx, cy, cz = nx * 0.25, ny * 0.5, nz * 0.5
    solid = sphere_mask(nx, ny, nz, cx, cy, cz, radius, device=device)
    band = dilate3d(solid, band_thickness)
    topo = build_band_topology(band, solid_mask=solid, periodic=False).to(device)
    ops = get_ops(degree=1, dx=1.0, dtype=torch.float32, device=device)

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device); ux0[solid] = 0.0
    f_lbm = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    cb = topo.band_coords
    f_dg = f_lbm[:, cb[:, 0], cb[:, 1], cb[:, 2]].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2, 2).contiguous()
    wall_mask = make_channel_wall_mask_3d(nz, ny, nx, solid, device=device)
    C_d, W_d, opp = C3D.to(torch.float32).to(device), W3D.to(torch.float32).to(device), OPP3D.to(device)
    initial_mass = float(f_lbm.sum().item())

    drag = 0.0
    for step in range(1, n_steps + 1):
        f_lbm, f_dg = hybrid_step(f_lbm, f_dg, C_d, W_d, ops, topo, tau_lbm=tau_lbm,
                                  dt=1.0, n_substeps=10, opposite=opp)
        f_lbm = apply_simple_channel_boundaries_3d(f_lbm, u_in, wall_mask, solid)
        if step % 50 == 0:
            f_lbm = correct_mass3d(f_lbm, initial_mass)
        if step % 100 == 0 or step == n_steps:
            fvec = compute_dg_solid_force(f_dg, topo, C_d, ops)
            drag_dg = float(fvec[0].item())
            # P0-projection force: project band into f_lbm, standard momentum exchange on obstacle.
            f_proj = project_band_to_lbm(f_lbm, f_dg, topo)
            fxp, _, _ = compute_obstacle_forces_3d(f_proj, solid)
            drag_p0 = float(fxp.item())
            print(f"step {step:3d}: drag_DG={drag_dg:.4f}  drag_P0={drag_p0:.4f}")
    A = math.pi * radius ** 2
    cd_dg = abs(drag_dg) / (0.5 * 1.0 * u_in ** 2 * A)
    cd_p0 = abs(drag_p0) / (0.5 * 1.0 * u_in ** 2 * A)
    cd_ref = sphere_cd_correlation(re)
    print(f"\nRe = {re:.1f}  Cd_ref(Clift) = {cd_ref:.4f}")
    print(f"  DG-interface force : Cd = {cd_dg:.4f}  err {abs(cd_dg-cd_ref)/cd_ref*100:.1f}%")
    print(f"  P0-projection force: Cd = {cd_p0:.4f}  err {abs(cd_p0-cd_ref)/cd_ref*100:.1f}%")


if __name__ == "__main__":
    run()
