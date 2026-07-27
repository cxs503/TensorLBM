#!/usr/bin/env python
"""Teaching Example 05: SUBOFF bare hull drag at Re=1000.

Simulates SUBOFF bare hull axisymmetric body and measures drag coefficient.
Target: Cd ≈ 0.042 (experimental, within 0.5% with 4-level refinement + Cumulant).

Uses corrected BB + MRT+Smagorinsky.  MEM and pressure+friction compared.

Usage:
    PYTHONPATH=src python teaching/05_suboff_best.py [--device sdaa:13]
"""
from __future__ import annotations

import argparse
import math

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import stream3d
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.drag_pressure import (
    SurfaceMesh, get_near_wall_3d,
    drag_pressure_integration, drag_friction_integration,
)
from tensorlbm.momentum_exchange import momentum_exchange_standard


def run(
    nx: int = 80, ny: int = 40, nz: int = 40,
    u_in: float = 0.05, tau: float = 0.55, cs: float = 0.05,
    n_steps: int = 2000, warmup: int = 500,
    device: str = "sdaa:13",
):
    dev = torch.device(device)
    from tensorlbm.suboff_cad import build_suboff_mask
    from tensorlbm.suboff_resistance import _voxel_wetted_area

    hull_length = nx * 0.6
    cx, cy, cz = nx * 0.35, ny / 2.0, nz / 2.0
    nu = (tau - 0.5) / 3.0
    Re = u_in * hull_length / nu

    print(f"=== SUBOFF Bare Hull: Re={Re:.0f} ===")
    print(f"Grid: {nx}×{ny}×{nz}, L={hull_length:.0f}, u_in={u_in}, tau={tau}")
    print(f"Device: {device}, steps={n_steps}")

    solid, _stats = build_suboff_mask(
        hull_type="bare_hull", nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz, length=hull_length, device=dev,
    )
    near = get_near_wall_3d(solid)
    S_wet = _voxel_wetted_area(solid, 1.0)
    dpS = 0.5 * u_in ** 2 * S_wet
    print(f"Wetted area: {S_wet:.0f}, dpS={dpS:.6f}")
    print()

    # Use gradient-based normals for SUBOFF
    mesh = SurfaceMesh.from_gradient(solid, near)

    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0),
                      device=dev)
    f[:, solid] = 0

    cd_mem_hist = []
    cd_pf_hist = []

    print(f"{'Step':>6s}  {'Cd_MEM':>10s}  {'Cd_PF':>10s}  {'max|u|':>8s}")
    print("-" * 45)

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs)
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)
        f = stream3d(f)
        f = far_field_bc_3d(f, u_in=u_in, obstacle_mask=solid)

        if step > warmup and step % 50 == 0:
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
                print(f" {step:5d}  {cd_mem_hist[-1]:10.6f}  "
                      f"{cd_pf_hist[-1]:10.6f}  {ms:8.4f}")
            else:
                print(f" {step:5d}  {'---':>10s}  {'---':>10s}  {ms:8.4f}")

    n = len(cd_mem_hist)
    if n > 0:
        cd_mem_mean = sum(cd_mem_hist) / n
        cd_pf_mean = sum(cd_pf_hist) / n
        print()
        print("=== Final Statistics ===")
        print(f"  Cd_MEM (mean):  {cd_mem_mean:.6f}  (target ~0.042)")
        print(f"  Cd_PF  (mean):  {cd_pf_mean:.6f}")
        err = abs(cd_pf_mean - 0.042) / 0.042 * 100
        print(f"  PF error vs target: {err:.1f}%")
        return {"cd_mem": cd_mem_mean, "cd_pf": cd_pf_mean, "Re": Re}
    return {"Re": Re}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SUBOFF drag Re=1000")
    parser.add_argument("--device", default="sdaa:13")
    parser.add_argument("--n-steps", type=int, default=2000)
    args = parser.parse_args()
    run(n_steps=args.n_steps, device=args.device)
