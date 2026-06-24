"""End-to-end 2-D hybrid DG-LBM cylinder-flow validation.

Exercises the full real-DG-LBM pipeline on an obstacle: a DG band encloses the
cylinder (its inner faces bounce-back off the solid; its outer faces couple to
the exterior LBM), while the exterior handles the channel (inlet/outlet/walls)
via the standard LBM machinery.  This is the 2-D analogue of the SUBOFF use
case and the gate that the validated numerical core actually drives a flow
without blowing up.

Run (CPU smoke):

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python examples/dg_lbm_cylinder_hybrid.py
"""
from __future__ import annotations

import math
import sys

import torch

from tensorlbm import (
    C as _C2D_unused,  # noqa: F401  (ensure package import side-effects)
)
from tensorlbm.d2q9 import C as C2D, OPPOSITE as OPP2D, W as W2D, equilibrium as eq2d, macroscopic as mac2d
from tensorlbm.dg_advection import equilibrium_dg, get_ops
from tensorlbm.dg_band import build_band_topology, hybrid_step
from tensorlbm.boundaries import apply_simple_channel_boundaries, compute_obstacle_forces, make_channel_wall_mask
from tensorlbm.solver import correct_mass


def cylinder_mask(ny: int, nx: int, cx: float, cy: float, r: float, device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def dilate(mask: torch.Tensor, k: int) -> torch.Tensor:
    import torch.nn.functional as F
    s = mask.float().unsqueeze(0).unsqueeze(0)
    d = F.max_pool2d(s, kernel_size=2 * k + 1, stride=1, padding=k)
    return (d.squeeze(0).squeeze(0) > 0.5) & ~mask


def run(ny=64, nx=128, r=8.0, band_thickness=4, u_in=0.1, tau_lbm=0.9,
        n_steps=400, device="cpu", dtype=torch.float64):
    torch.manual_seed(0)
    cx, cy = nx * 0.25, ny * 0.5
    solid = cylinder_mask(ny, nx, cx, cy, r, device)
    band = dilate(solid, band_thickness)
    topo = build_band_topology(band, solid_mask=solid, periodic=False)
    ops = get_ops(degree=1, dx=1.0, dtype=dtype)

    # Initial field: uniform inlet flow, zero inside the solid.
    rho0 = torch.ones(ny, nx, dtype=dtype, device=device)
    ux0 = torch.full((ny, nx), u_in, dtype=dtype, device=device)
    ux0[solid] = 0.0
    f_lbm = eq2d(rho0, ux0, torch.zeros_like(ux0)).to(dtype)
    # Seed band DOFs (P0) from the band-cell LBM values.
    cb = topo.band_coords
    f_dg = f_lbm[:, cb[:, 0], cb[:, 1]].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2).contiguous()

    wall_mask = make_channel_wall_mask(ny, nx, solid, device=device)
    n_cell = ny * nx
    initial_mass = float(f_lbm.sum().item())
    C = C2D.to(dtype)
    W = W2D.to(dtype)
    opp = OPP2D.to(device)          # keep int64 (do NOT cast to dtype)

    max_speed_prev = 0.0
    for step in range(1, n_steps + 1):
        f_lbm, f_dg = hybrid_step(
            f_lbm, f_dg, C, W, ops, topo, tau_lbm=tau_lbm,
            dt=1.0, n_substeps=6, opposite=opp,
        )
        # Channel BCs on the exterior (inlet/outlet/walls); obstacle is inside the band.
        f_lbm = apply_simple_channel_boundaries(f_lbm, u_in, wall_mask, solid)
        if step % 50 == 0:
            f_lbm = correct_mass(f_lbm, initial_mass)
        if step % 100 == 0 or step == n_steps:
            rho, ux, uy = mac2d(f_lbm)
            ux[solid] = 0.0
            ms = float(torch.sqrt(ux * ux + uy * uy).max().item())
            fx, fy = compute_obstacle_forces(f_lbm, solid)
            print(f"step {step:4d}: max|u|={ms:.4f}  drag={float(fx):.4f}  lift={float(fy):.4f}")
            if not math.isfinite(ms) or ms > 5.0:
                print("  -> UNSTABLE, aborting")
                return False
            max_speed_prev = ms
    return True


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    ok = run(n_steps=n)
    print("STABLE" if ok else "FAILED")
