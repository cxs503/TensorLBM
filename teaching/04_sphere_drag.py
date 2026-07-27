#!/usr/bin/env python
"""Teaching Example 04: Sphere drag at Re=100.

Common interface pipeline:
  solid → get_near_wall_3d → SurfaceMesh.from_sphere
  → lbm_step_correct (BGK + far_field_bc_3d)
  → drag_pressure_integration + drag_friction_integration

Usage:
  PYTHONPATH=src python teaching/04_sphere_drag.py [device_id]
"""
from __future__ import annotations
import sys, math
sys.path.insert(0, 'src')
import torch
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, correct_mass3d
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.drag_pressure import (
    get_near_wall_3d, SurfaceMesh,
    drag_pressure_integration, drag_friction_integration,
)


def run(device_id=0, nx=64, ny=48, nz=48, R=6.0, u_in=0.05, tau=0.6,
        n_steps=3000, warmup=1000):
    dev = torch.device(f'sdaa:{device_id}')
    torch.sdaa.set_device(dev)
    cx, cy, cz = nx // 4, ny // 2, nz // 2
    D = 2 * R
    nu = (tau - 0.5) / 3.0
    Re = u_in * D / nu
    dpS = 0.5 * u_in ** 2 * math.pi * R ** 2
    Cd_ref = 1.09

    print(f"=== Sphere Drag: Re={Re:.0f} (SDAA:{device_id}) ===")
    print(f"Grid: {nx}×{ny}×{nz}, D={D}, u_in={u_in}, tau={tau}")
    print(f"Steps: {n_steps}, warmup: {warmup}")
    print()

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev, dtype=torch.float32),
        torch.arange(ny, device=dev, dtype=torch.float32),
        torch.arange(nx, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    solid = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= R ** 2

    # --- Common interface: get_near_wall_3d → SurfaceMesh.from_sphere ---
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_sphere(solid, near, cx, cy, cz, R)

    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0),
                      device=dev)
    f[:, solid] = 0
    initial_mass = float(rho0.sum().item())

    cd_p_hist, cd_f_hist = [], []

    print(f"{'Step':>6s}  {'Cd_p':>8s}  {'Cd_f':>8s}  {'Cd':>8s}")
    print("-" * 40)

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_bgk3d, tau, solid, u_in, far_field_bc_3d,
            correct_mass_fn=correct_mass3d, target_mass=initial_mass,
            step=step, mass_interval=200,
        )

        if not torch.isfinite(f).all():
            print(f"DIVERGED at step {step}")
            break

        if step > warmup:
            cdp_x, _, _ = drag_pressure_integration(f, mesh, dpS, solid=solid)
            cdf_x, _, _ = drag_friction_integration(f, mesh, dpS, nu)
            cd_p_hist.append(cdp_x)
            cd_f_hist.append(cdf_x)

        if step % 500 == 0 or step == n_steps:
            n = len(cd_p_hist)
            if n > 0:
                cd_p = sum(cd_p_hist[-min(100,n):]) / min(100,n)
                cd_f = sum(cd_f_hist[-min(100,n):]) / min(100,n)
                print(f" {step:5d}  {cd_p:8.4f}  {cd_f:8.4f}  {cd_p+cd_f:8.4f}")
            else:
                _, ux, _, _ = macroscopic3d(f)
                ms = float(ux.abs().max().item())
                print(f" {step:5d}  (max|u|={ms:.4f})")

    n = len(cd_p_hist)
    if n > 0:
        n_avg = max(1, n // 2)
        cd_p_mean = sum(cd_p_hist[-n_avg:]) / n_avg
        cd_f_mean = sum(cd_f_hist[-n_avg:]) / n_avg
        cd_mean = cd_p_mean + cd_f_mean
        err = abs(cd_mean - Cd_ref) / Cd_ref * 100
        print(f"\n=== FINAL ===")
        print(f"  Cd_p = {cd_p_mean:.4f}")
        print(f"  Cd_f = {cd_f_mean:.4f}")
        print(f"  Cd   = {cd_mean:.4f}  (ref={Cd_ref}, err={err:.1f}%)")
        passed = err < 5.0
        print(f"\n{'PASS' if passed else 'FAIL'}: Cd err={err:.1f}% (target < 5%)")
        return passed
    return False


if __name__ == "__main__":
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run(dev)
