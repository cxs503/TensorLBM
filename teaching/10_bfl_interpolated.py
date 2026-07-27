#!/usr/bin/env python
"""Teaching Example 10: BFL interpolated bounce-back + friction.

Demonstrates the Bouzidi-Firdaouss-Lallemand (BFL) interpolated
bounce-back for curved surfaces where the wall doesn't align with
the grid.

BFL formula (wall at fractional distance q from fluid cell):
    q < 0.5:  f_wall = 2q·f_opp + (1-2q)·f_prev
    q ≥ 0.5:  f_wall = f_opp/(2q) + (2q-1)/(2q)·f_prev_opp

The friction drag uses the BFL-corrected formula:
    τ = ν · u_1 / q  (instead of τ = 2ν·u_1 for q=0.5)

Usage:
    PYTHONPATH=src python teaching/10_bfl_interpolated.py [--device sdaa:13]
"""
from __future__ import annotations

import argparse
import math

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C, OPPOSITE
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.drag_pressure import (
    SurfaceMesh, get_near_wall_3d,
    drag_pressure_integration, drag_friction_integration,
)
from tensorlbm.momentum_exchange import momentum_exchange_standard, momentum_exchange_bfl


def compute_q_wall_sphere(solid, cx, cy, cz, R, device):
    """Compute fractional wall distance q for sphere geometry.

    For each near-wall fluid cell, q = distance to wall surface / 1.0
    (in lattice units, where the wall is at the sphere surface).
    """
    nz, ny, nx = solid.shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    dist_to_center = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
    # q = fractional distance from fluid cell to wall
    # For a cell at distance r from center, wall is at R
    # q = |R - r| (distance from cell to wall surface)
    q = (R - dist_to_center).abs()
    q = q.clamp(min=0.01, max=0.99)  # clamp to valid range
    return q


def run(
    nx=48, ny=40, nz=40, R=6.0, u_in=0.05, tau=0.6,
    n_steps=1500, warmup=500, device="sdaa:13",
):
    dev = torch.device(device)
    cx, cy, cz = nx // 4, ny // 2, nz // 2
    D = 2 * R
    nu = (tau - 0.5) / 3.0
    Re = u_in * D / nu
    dpS = 0.5 * u_in ** 2 * math.pi * R ** 2

    print(f"=== BFL Interpolated BB + Friction: Re={Re:.0f} ===")
    print(f"Grid: {nx}×{ny}×{nz}, R={R}, D={D}")
    print(f"Device: {device}, steps={n_steps}")
    print()

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev, dtype=torch.float32),
        torch.arange(ny, device=dev, dtype=torch.float32),
        torch.arange(nx, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    solid = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= R ** 2
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_sphere(solid, near, cx, cy, cz, R)

    # Compute q_wall for BFL
    q_wall = compute_q_wall_sphere(solid, cx, cy, cz, R, dev)
    print(f"q_wall range: [{float(q_wall[near].min()):.3f}, {float(q_wall[near].max()):.3f}]")
    print(f"q_wall mean:  {float(q_wall[near].mean()):.3f}")
    print()

    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=dev)
    f[:, solid] = 0

    cd_std_hist = []
    cd_bfl_hist = []
    cd_pf_std_hist = []
    cd_pf_bfl_hist = []

    print(f"{'Step':>6s}  {'Cd_MEM':>8s}  {'Cd_BFL':>8s}  {'Cd_f_std':>8s}  {'Cd_f_bfl':>8s}")
    print("-" * 50)

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)
        f = stream3d(f)
        f = far_field_bc_3d(f, u_in=u_in, obstacle_mask=solid)

        if step > warmup and step % 20 == 0:
            # Standard MEM (q=0.5 implicit)
            fx_std, _, _ = momentum_exchange_standard(f, solid, near)
            cd_std = fx_std / dpS

            # BFL MEM (weighted by q)
            fx_bfl, _, _ = momentum_exchange_bfl(f, solid, near, q_wall)
            cd_bfl = fx_bfl / dpS

            # Friction: standard (q=0.5) vs BFL (q-weighted)
            _, _, cd_f_std = drag_friction_integration(f, mesh, dpS, nu, formula='standard')
            _, _, cd_f_bfl = drag_friction_integration(f, mesh, dpS, nu, q_wall=q_wall, formula='bfl')

            cd_std_hist.append(cd_std)
            cd_bfl_hist.append(cd_bfl)
            cd_pf_std_hist.append(cd_f_std)
            cd_pf_bfl_hist.append(cd_f_bfl)

        if step % 500 == 0 or step == n_steps:
            n = len(cd_std_hist)
            if n > 0:
                print(f" {step:5d}  {cd_std_hist[-1]:8.4f}  {cd_bfl_hist[-1]:8.4f}  "
                      f"{cd_pf_std_hist[-1]:8.4f}  {cd_pf_bfl_hist[-1]:8.4f}")

    n = len(cd_std_hist)
    if n > 0:
        cd_std = sum(cd_std_hist) / n
        cd_bfl = sum(cd_bfl_hist) / n
        cf_std = sum(cd_pf_std_hist) / n
        cf_bfl = sum(cd_pf_bfl_hist) / n

        print()
        print("=== Final Comparison ===")
        print(f"  Cd_MEM (standard, q=0.5): {cd_std:.4f}")
        print(f"  Cd_MEM (BFL, q-weighted): {cd_bfl:.4f}")
        print(f"  Cf (standard, q=0.5):     {cf_std:.4f}")
        print(f"  Cf (BFL, q-weighted):      {cf_bfl:.4f}")
        print(f"  BFL/standard ratio (MEM):  {cd_bfl/cd_std:.3f}" if abs(cd_std) > 1e-10 else "")
        return {"cd_std": cd_std, "cd_bfl": cd_bfl, "cf_std": cf_std, "cf_bfl": cf_bfl, "Re": Re}
    return {"Re": Re}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BFL interpolated BB")
    parser.add_argument("--device", default="sdaa:13")
    parser.add_argument("--n-steps", type=int, default=1500)
    args = parser.parse_args()
    run(n_steps=args.n_steps, device=args.device)
