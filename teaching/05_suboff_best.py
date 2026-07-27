#!/usr/bin/env python
"""Teaching Example 05: SUBOFF bare hull drag at Re=1000.

Common interface pipeline:
  solid → get_near_wall_3d → SurfaceMesh.from_suboff
  → lbm_step_correct (MRT+Smagorinsky + far_field_bc_3d)
  → drag_pressure_integration + drag_friction_integration

Usage:
  PYTHONPATH=src python teaching/05_suboff_best.py [device_id]
"""
from __future__ import annotations
import sys, math
sys.path.insert(0, 'src')
import torch
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.drag_pressure import (
    get_near_wall_3d, SurfaceMesh,
    drag_pressure_integration, drag_friction_integration,
)
from tensorlbm.suboff_cad import build_suboff_mask
from tensorlbm.suboff_resistance import _voxel_wetted_area


def run(device_id=0, nx=80, ny=40, nz=40, u_in=0.05, tau=0.55, cs=0.05,
        n_steps=2000, warmup=500):
    dev = torch.device(f'sdaa:{device_id}')
    torch.sdaa.set_device(dev)

    hull_length = nx * 0.6
    cx, cy, cz = nx * 0.35, ny / 2.0, nz / 2.0
    nu = (tau - 0.5) / 3.0
    Re = u_in * hull_length / nu
    Cd_ref = 0.042

    print(f"=== SUBOFF Bare Hull: Re={Re:.0f} (SDAA:{device_id}) ===")
    print(f"Grid: {nx}×{ny}×{nz}, L={hull_length:.0f}, u_in={u_in}, tau={tau}")
    print(f"Steps: {n_steps}, warmup: {warmup}")
    print()

    solid, stats = build_suboff_mask(
        hull_type="bare_hull", nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz, length=hull_length, device=dev,
    )
    hull_radius = stats['radius']

    # --- Common interface: get_near_wall_3d → SurfaceMesh.from_suboff ---
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_suboff(solid, near, cx, cy, cz, hull_length, hull_radius)
    S_wet = _voxel_wetted_area(solid, 1.0)
    dpS = 0.5 * u_in ** 2 * S_wet
    print(f"Wetted area: {S_wet:.0f}, dpS={dpS:.6f}")
    print(f"Solid cells: {int(solid.sum().item())}")
    print()

    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0),
                      device=dev)
    f[:, solid] = 0
    initial_mass = float(rho0.sum().item())

    cd_p_hist, cd_f_hist = [], []

    print(f"{'Step':>6s}  {'Cd_p':>10s}  {'Cd_f':>10s}  {'Cd':>10s}")
    print("-" * 45)

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_smagorinsky_mrt3d, tau, solid, u_in, far_field_bc_3d,
            correct_mass_fn=correct_mass3d, target_mass=initial_mass,
            step=step, mass_interval=200, C_s=cs,
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
                print(f" {step:5d}  {cd_p:10.6f}  {cd_f:10.6f}  {cd_p+cd_f:10.6f}")
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
        print(f"  Cd_p = {cd_p_mean:.6f}")
        print(f"  Cd_f = {cd_f_mean:.6f}")
        print(f"  Cd   = {cd_mean:.6f}  (ref={Cd_ref}, err={err:.1f}%)")
        passed = err < 6.0
        print(f"\n{'PASS' if passed else 'FAIL'}: Cd err={err:.1f}% (target < 6%)")
        return passed
    return False


if __name__ == "__main__":
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run(dev)
