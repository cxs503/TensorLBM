#!/usr/bin/env python
"""Teaching Example 03: Cylinder drag — Cd, Cl, St at Re=200.

Simulates 2D extruded cylinder flow at Re=200 and measures:
- Drag coefficient Cd (target: ~1.30)
- Lift coefficient Cl (oscillating, amplitude ~0.5)
- Strouhal number St (target: ~0.20)

Uses corrected BB + BGK collision.  MEM and pressure+friction
drag are both computed for comparison.

Usage:
    PYTHONPATH=src python teaching/03_cylinder_drag.py [--device sdaa:10]

Expected output:
    Cd ≈ 1.0-1.3 (coarse grid, converges to 1.33 with refinement)
    St ≈ 0.18-0.20
"""
from __future__ import annotations

import argparse
import math
import time

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
    nx: int = 128, ny: int = 64, nz: int = 4,
    R: float = 8.0, u_in: float = 0.05, tau: float = 0.55,
    n_steps: int = 5000, warmup: int = 2000,
    device: str = "sdaa:10",
):
    dev = torch.device(device)
    cx, cy = nx // 4, ny // 2
    D = 2 * R
    nu = (tau - 0.5) / 3.0
    Re = u_in * D / nu
    dpS = 0.5 * u_in ** 2 * D * nz

    print(f"=== Cylinder Drag: Re={Re:.0f} ===")
    print(f"Grid: {nx}×{ny}×{nz}, D={D}, u_in={u_in}, tau={tau}")
    print(f"Device: {device}, steps={n_steps}")
    print()

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev, dtype=torch.float32),
        torch.arange(ny, device=dev, dtype=torch.float32),
        torch.arange(nx, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    solid = (xx - cx) ** 2 + (yy - cy) ** 2 <= R ** 2
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_cylinder(solid, near, cx, cy, R, axis='z')

    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0),
                      device=dev)
    f[:, solid] = 0

    cd_mem_hist = []
    cd_pf_hist = []
    cl_hist = []

    print(f"{'Step':>6s}  {'Cd_MEM':>8s}  {'Cd_PF':>8s}  {'Cl':>8s}  {'max|u|':>8s}")
    print("-" * 50)

    for step in range(1, n_steps + 1):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)
        f = stream3d(f)
        f = far_field_bc_3d(f, u_in=u_in, obstacle_mask=solid)

        if step > warmup and step % 10 == 0:
            # MEM drag
            fx_me, fy_me, _ = momentum_exchange_standard(f, solid, near)
            cd_me = fx_me / dpS
            cl_me = fy_me / dpS

            # Pressure+friction
            cd_p, _, _ = drag_pressure_integration(f, mesh, dpS, solid=solid)
            cd_f, _, _ = drag_friction_integration(f, mesh, dpS, nu)
            cd_pf = cd_p + cd_f

            cd_mem_hist.append(cd_me)
            cd_pf_hist.append(cd_pf)
            cl_hist.append(cl_me)

        if step % 500 == 0 or step == n_steps:
            _, ux, uy, _ = macroscopic3d(f)
            ms = float(torch.sqrt(ux**2 + uy**2).max().item())
            n = len(cd_mem_hist)
            if n > 0:
                print(f" {step:5d}  {cd_mem_hist[-1]:8.4f}  "
                      f"{cd_pf_hist[-1]:8.4f}  {cl_hist[-1]:8.4f}  {ms:8.4f}")
            else:
                print(f" {step:5d}  {'---':>8s}  {'---':>8s}  {'---':>8s}  {ms:8.4f}")

    # --- Statistics ---
    n = len(cd_mem_hist)
    if n > 10:
        cd_mem_mean = sum(cd_mem_hist) / n
        cd_pf_mean = sum(cd_pf_hist) / n
        cl_mean = sum(cl_hist) / n
        cl_amp = max(abs(max(cl_hist)), abs(min(cl_hist)))

        # Strouhal: St = f*D/U, estimate from Cl oscillation period
        # Simple peak detection
        st = 0.0
        if len(cl_hist) > 20:
            # Count zero crossings (upward)
            crossings = 0
            for i in range(1, len(cl_hist)):
                if cl_hist[i-1] < 0 and cl_hist[i] >= 0:
                    crossings += 1
            if crossings > 1:
                period_steps = len(cl_hist) / crossings
                freq = 1.0 / period_steps  # cycles per step
                st = freq * D / u_in

        print()
        print("=== Final Statistics ===")
        print(f"  Cd_MEM (mean):     {cd_mem_mean:.4f}  (target ~1.33)")
        print(f"  Cd_PF  (mean):     {cd_pf_mean:.4f}")
        print(f"  Cl (mean):         {cl_mean:.6f}")
        print(f"  Cl (amplitude):    {cl_amp:.4f}")
        print(f"  Strouhal:          {st:.4f}  (target ~0.20)")

        return {
            "cd_mem": cd_mem_mean, "cd_pf": cd_pf_mean,
            "cl_amp": cl_amp, "st": st, "Re": Re,
        }
    return {"Re": Re}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cylinder drag Re=200")
    parser.add_argument("--device", default="sdaa:10")
    parser.add_argument("--n-steps", type=int, default=5000)
    args = parser.parse_args()
    run(n_steps=args.n_steps, device=args.device)
