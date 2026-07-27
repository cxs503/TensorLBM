#!/usr/bin/env python
"""Teaching Example 05: SUBOFF bare hull drag at Re=1000.

Common interface pipeline:
  solid → get_near_wall_3d → SurfaceMesh.from_suboff
  → lbm_step_correct (MRT+Smagorinsky + far_field_bc_3d)
  → drag_pressure_integration(extrap='none') + drag_friction_integration

Bug 37 fix: ``f[:, solid] = 0`` breaks bounce-back because it zeros the
equilibrium populations at solid cells, destroying the BB reflection.
The correct approach is ``ux0[solid] = 0.0`` — set velocity to zero at
solid cells but keep rho=1 equilibrium (NoDynamics handles the rest).

Usage:
  PYTHONPATH=src python teaching/05_suboff_best.py [device_id]
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
from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig


def run(device_id=18, L=80.0, u_in=0.06, Re=1000.0, cs=0.05,
        n_steps=5000, warmup=1000):
    dev = torch.device(f'sdaa:{device_id}')
    torch.sdaa.set_device(dev)

    config = SuboffConfig()
    radius = config.r_over_l * L
    D = 2.0 * radius
    nu = u_in * L / Re
    tau = 3.0 * nu + 0.5
    nx, ny, nz = 200, 80, 80
    cx, cy, cz = nx * 0.30, ny * 0.5, nz * 0.5
    # Wetted surface area (analytical cylinder approximation)
    S_wet = math.pi * D * L
    dpS = 0.5 * u_in ** 2 * S_wet
    Cd_ref = 1.328 / math.sqrt(Re)     # Blasius Cf, Re=1000 → 0.042

    # 3D hull: all lateral faces far-field (no periodicity)
    bc_config = {
        'far_field_faces': ['y-', 'y+', 'z-', 'z+'],
        'periodic_faces': [],
    }
    bc_fn = partial(far_field_bc_3d, bc_config=bc_config)

    print(f"=== SUBOFF Bare Hull: Re={Re:.0f} (SDAA:{device_id}) ===")
    print(f"Grid: {nx}×{ny}×{nz}, L={L}, D={D:.3f}, u_in={u_in}, "
          f"tau={tau:.4f}, Cs={cs}")
    print(f"Steps: {n_steps}, warmup: {warmup}")
    print(f"Bug 37 fix: ux0[solid]=0 (not f[:,solid]=0)")
    print()

    t0 = time.time()
    solid, stats = build_suboff_mask(
        hull_type="bare_hull", nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz, length=L, radius=radius,
        config=config, device=str(dev),
    )

    # --- Common interface: get_near_wall_3d → SurfaceMesh.from_suboff ---
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_suboff(solid, near, cx, cy, cz, L, radius, config)
    n_solid = int(solid.sum().item())
    n_near = int(near.sum().item())
    print(f"solid={n_solid} near={n_near} mesh=from_suboff "
          f"({time.time()-t0:.1f}s)")
    print(f"Wetted area: {S_wet:.0f}, dpS={dpS:.6f}")
    print()

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

    print(f"{'Step':>6s}  {'Cd_p':>10s}  {'Cd_f':>10s}  {'Cd':>10s}")
    print("-" * 45)

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
                print(f" {step:5d}  {cd_p:10.6f}  {cd_f:10.6f}  "
                      f"{cd_p+cd_f:10.6f}  ({time.time()-t0:.0f}s)")
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
        print(f"  Cd_p = {cd_p_mean:.6f}")
        print(f"  Cd_f = {cd_f_mean:.6f}")
        print(f"  Cd   = {cd_mean:.6f}  (ref={Cd_ref:.6f}, err={err:.1f}%)")
        print(f"  time = {elapsed:.0f}s")
        passed = err < 6.0
        print(f"\n{'PASS' if passed else 'FAIL'}: Cd err={err:.1f}% "
              f"(target < 6%)")
        return passed
    return False


if __name__ == "__main__":
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    run(dev)
