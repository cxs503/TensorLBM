#!/usr/bin/env python
"""Teaching Example 04: Sphere drag at Re=100.

Common interface pipeline:
  solid → get_near_wall_3d → SurfaceMesh.from_sphere
  → lbm_step_correct (MRT+Smagorinsky + far_field_bc_3d)
  → drag_pressure_integration(extrap='none') + drag_friction_integration

Bug 37 fix: ``f[:, solid] = 0`` breaks bounce-back because it zeros the
equilibrium populations at solid cells, destroying the BB reflection.
The correct approach is ``ux0[solid] = 0.0`` — set velocity to zero at
solid cells but keep rho=1 equilibrium (NoDynamics handles the rest).

Usage:
  PYTHONPATH=src python teaching/04_sphere_drag.py [device_id]
"""
from __future__ import annotations
import sys, math, time
from functools import partial
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


def run(device_id=17, D=40.0, u_in=0.08, Re=100.0, cs=0.05,
        n_steps=3000, warmup=600):
    dev = torch.device(f'sdaa:{device_id}')
    torch.sdaa.set_device(dev)

    R = D / 2.0
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    nx, ny, nz = 180, 180, 180
    cx, cy, cz = nx * 0.25, ny * 0.5, nz * 0.5
    A_frontal = math.pi * R ** 2
    dpS = 0.5 * u_in ** 2 * A_frontal
    Cd_ref = 1.09                     # Clift, Re=100

    # 3D sphere: all lateral faces far-field (no periodicity)
    bc_config = {
        'far_field_faces': ['y-', 'y+', 'z-', 'z+'],
        'periodic_faces': [],
    }
    bc_fn = partial(far_field_bc_3d, bc_config=bc_config)

    print(f"=== Sphere Drag: Re={Re:.0f} (SDAA:{device_id}) ===")
    print(f"Grid: {nx}×{ny}×{nz}, D={D}, u_in={u_in}, tau={tau:.4f}, Cs={cs}")
    print(f"Steps: {n_steps}, warmup: {warmup}")
    print(f"Bug 37 fix: ux0[solid]=0 (not f[:,solid]=0)")
    print()

    t0 = time.time()
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev, dtype=torch.float32),
        torch.arange(ny, device=dev, dtype=torch.float32),
        torch.arange(nx, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    solid = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 < R ** 2

    # --- Common interface: get_near_wall_3d → SurfaceMesh.from_sphere ---
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_sphere(solid, near, cx, cy, cz, R)
    n_solid = int(solid.sum().item())
    n_near = int(near.sum().item())
    print(f"solid={n_solid} near={n_near} mesh=from_sphere "
          f"({time.time()-t0:.1f}s)")

    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    # Bug 37 fix: zero velocity at solid, keep rho=1 equilibrium
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0),
                      device=dev)
    initial_mass = float(rho0.sum().item())
    print(f"init done mass={initial_mass} ({time.time()-t0:.1f}s)")
    print()

    cd_p_hist, cd_f_hist = [], []

    print(f"{'Step':>6s}  {'Cd_p':>8s}  {'Cd_f':>8s}  {'Cd':>8s}")
    print("-" * 40)

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_smagorinsky_mrt3d, tau, solid, u_in, bc_fn,
            correct_mass_fn=correct_mass3d, target_mass=initial_mass,
            step=step, mass_interval=200, C_s=cs,
        )

        if not torch.isfinite(f).all():
            print(f"DIVERGED at step {step}")
            break

        cdp_x, _, _ = drag_pressure_integration(f, mesh, dpS, extrap='none',
                                                solid=solid)
        cdf_x, _, _ = drag_friction_integration(f, mesh, dpS, nu)
        cd_tot = cdp_x + cdf_x

        if step > warmup:
            cd_p_hist.append(cdp_x)
            cd_f_hist.append(cdf_x)

        if step % 500 == 0 or step == n_steps:
            n = len(cd_p_hist)
            if n > 0:
                n_avg = min(100, n)
                cd_p = sum(cd_p_hist[-n_avg:]) / n_avg
                cd_f = sum(cd_f_hist[-n_avg:]) / n_avg
                print(f" {step:5d}  {cd_p:8.4f}  {cd_f:8.4f}  "
                      f"{cd_p+cd_f:8.4f}  ({time.time()-t0:.0f}s)")
            else:
                _, ux, _, _ = macroscopic3d(f)
                ms = float(ux.abs().max().item())
                print(f" {step:5d}  (max|u|={ms:.4f})")

    elapsed = time.time() - t0
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
        print(f"  time = {elapsed:.0f}s")
        passed = err < 10.0
        print(f"\n{'PASS' if passed else 'FAIL'}: Cd err={err:.1f}% "
              f"(target < 10%)")
        return passed
    return False


if __name__ == "__main__":
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 17
    run(dev)
