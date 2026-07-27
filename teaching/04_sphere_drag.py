#!/usr/bin/env python
"""Teaching Example 04: Sphere drag at Re=100.

Simulates 3D sphere flow and measures drag coefficient Cd.
Target: Cd ≈ 1.09 (experimental, Re=100).

Uses corrected BB + BGK.  MEM and pressure+friction drag compared.

Usage:
    PYTHONPATH=src python teaching/04_sphere_drag.py [--device sdaa:12]
"""
from __future__ import annotations

import argparse
import math

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.drag_pressure import (
    SurfaceMesh, get_near_wall_3d,
    drag_pressure_integration, drag_friction_integration,
)
from tensorlbm.momentum_exchange import momentum_exchange_standard


def run(
    nx: int = 64, ny: int = 48, nz: int = 48,
    R: float = 6.0, u_in: float = 0.05, tau: float = 0.6,
    n_steps: int = 3000, warmup: int = 1000,
    device: str = "sdaa:12",
):
    dev = torch.device(device)
    cx, cy, cz = nx // 4, ny // 2, nz // 2
    D = 2 * R
    nu = (tau - 0.5) / 3.0
    Re = u_in * D / nu
    dpS = 0.5 * u_in ** 2 * math.pi * R ** 2

    print(f"=== Sphere Drag: Re={Re:.0f} ===")
    print(f"Grid: {nx}×{ny}×{nz}, D={D}, u_in={u_in}, tau={tau}")
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

    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0),
                      device=dev)
    f[:, solid] = 0

    cd_mem_hist = []
    cd_pf_hist = []

    print(f"{'Step':>6s}  {'Cd_MEM':>8s}  {'Cd_PF':>8s}  {'max|u|':>8s}")
    print("-" * 40)

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)
        f = stream3d(f)
        f = far_field_bc_3d(f, u_in=u_in, obstacle_mask=solid)

        if step > warmup and step % 20 == 0:
            fx_me, _, _ = momentum_exchange_standard(f, solid, near)
            cd_me = fx_me / dpS
            cd_p, _, _ = drag_pressure_integration(f, mesh, dpS, solid=solid)
            cd_f, _, _ = drag_friction_integration(f, mesh, dpS, nu)
            cd_pf = cd_p + cd_f
            cd_mem_hist.append(cd_me)
            cd_pf_hist.append(cd_pf)

        if step % 500 == 0 or step == n_steps:
            _, ux, uy, uz = macroscopic3d(f)
            ms = float(torch.sqrt(ux**2 + uy**2 + uz**2).max().item())
            n = len(cd_mem_hist)
            if n > 0:
                print(f" {step:5d}  {cd_mem_hist[-1]:8.4f}  "
                      f"{cd_pf_hist[-1]:8.4f}  {ms:8.4f}")
            else:
                print(f" {step:5d}  {'---':>8s}  {'---':>8s}  {ms:8.4f}")

    n = len(cd_mem_hist)
    if n > 0:
        cd_mem_mean = sum(cd_mem_hist) / n
        cd_pf_mean = sum(cd_pf_hist) / n
        print()
        print("=== Final Statistics ===")
        print(f"  Cd_MEM (mean):  {cd_mem_mean:.4f}  (target ~1.09)")
        print(f"  Cd_PF  (mean):  {cd_pf_mean:.4f}")
        return {"cd_mem": cd_mem_mean, "cd_pf": cd_pf_mean, "Re": Re}
    return {"Re": Re}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sphere drag Re=100")
    parser.add_argument("--device", default="sdaa:12")
    parser.add_argument("--n-steps", type=int, default=3000)
    args = parser.parse_args()
    run(n_steps=args.n_steps, device=args.device)
