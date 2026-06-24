"""End-to-end 3-D hybrid DG-LBM sphere-flow validation.

3-D analogue of ``dg_lbm_cylinder_hybrid``: a DG band encloses the sphere
(inner faces bounce-back, outer faces couple to the exterior D3Q19 LBM), the
exterior handles the channel.  Proves the real DG-LBM drives a 3-D obstacle
flow end-to-end — the direct precursor of the SUBOFF hull case.

CPU smoke (tiny grid, few steps):

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python examples/dg_lbm_sphere_hybrid.py 30

A real Cd run needs a larger grid on GPU.
"""
from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F

from tensorlbm.d3q19 import C as C3D, OPPOSITE as OPP3D, W as W3D, equilibrium3d, macroscopic3d
from tensorlbm.dg_advection import equilibrium_dg, get_ops
from tensorlbm.dg_band import build_band_topology, hybrid_step
from tensorlbm.boundaries3d import (
    apply_simple_channel_boundaries_3d,
    make_channel_wall_mask_3d,
    sphere_mask,
)
from tensorlbm.obstacles import compute_obstacle_forces_3d
from tensorlbm.solver3d import correct_mass3d


def dilate3d(mask: torch.Tensor, k: int) -> torch.Tensor:
    s = mask.float().unsqueeze(0).unsqueeze(0)
    d = F.max_pool3d(s, kernel_size=2 * k + 1, stride=1, padding=k)
    return (d.squeeze(0).squeeze(0) > 0.5) & ~mask


def run(nz=32, ny=32, nx=64, radius=5.0, band_thickness=3, u_in=0.1, tau_lbm=0.9,
        n_steps=30, device="cpu", dtype=torch.float64):
    torch.manual_seed(0)
    cx, cy, cz = nx * 0.25, ny * 0.5, nz * 0.5
    solid = sphere_mask(nx, ny, nz, cx, cy, cz, radius, device=device)
    band = dilate3d(solid, band_thickness)
    topo = build_band_topology(band, solid_mask=solid, periodic=False)
    ops = get_ops(degree=1, dx=1.0, dtype=dtype)

    rho0 = torch.ones(nz, ny, nx, dtype=dtype, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, dtype=dtype, device=device)
    ux0[solid] = 0.0
    f_lbm = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0)).to(dtype)
    cb = topo.band_coords                                   # (n_band, 3) in (z,y,x)
    f_dg = f_lbm[:, cb[:, 0], cb[:, 1], cb[:, 2]]            # (Q, n_band)
    f_dg = f_dg.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2, 2).contiguous()

    wall_mask = make_channel_wall_mask_3d(nz, ny, nx, solid, device=device)
    initial_mass = float(f_lbm.sum().item())
    C = C3D.to(dtype)
    W = W3D.to(dtype)
    opp = OPP3D.to(device)

    for step in range(1, n_steps + 1):
        f_lbm, f_dg = hybrid_step(
            f_lbm, f_dg, C, W, ops, topo, tau_lbm=tau_lbm,
            dt=1.0, n_substeps=10, opposite=opp,       # 3D ⇒ ≥10 sub-steps
        )
        f_lbm = apply_simple_channel_boundaries_3d(f_lbm, u_in, wall_mask, solid)
        if step % 10 == 0:
            f_lbm = correct_mass3d(f_lbm, initial_mass)
        if step % 10 == 0 or step == n_steps:
            rho, ux, uy, uz = macroscopic3d(f_lbm)
            ux[solid] = 0.0
            ms = float(torch.sqrt(ux * ux + uy * uy + uz * uz).max().item())
            fx, fy, fz = compute_obstacle_forces_3d(f_lbm, solid)
            print(f"step {step:3d}: max|u|={ms:.4f}  drag={float(fx):.4f}")
            if not math.isfinite(ms) or ms > 5.0:
                print("  -> UNSTABLE"); return False
    return True


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    ok = run(n_steps=n)
    print("STABLE" if ok else "FAILED")
