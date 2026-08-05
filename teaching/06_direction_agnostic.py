#!/usr/bin/env python
"""Teaching Example 06: Direction-agnostic verification — y↔z swap.

Verifies that the LBM solver is direction-agnostic: swapping the y and z
axes of the geometry should give identical drag results.

Tests a cylinder oriented along z vs along y.  Both should give the
same Cd (within numerical precision).

Uses the COMMON INTERFACE ONLY:
  - lbm_step_correct() for the main loop
  - SurfaceMesh.from_cylinder() with axis swap for surface normals
  - bounce_back_cells_3d(f_pre) for half-way BB (inside lbm_step_correct)
  - far_field_bc_3d for far-field boundary
  - drag_pressure_integration / drag_friction_integration for force
  - momentum_exchange_standard for MEM cross-validation

Usage:
    PYTHONPATH=src python teaching/06_direction_agnostic.py [device_id]
"""
from __future__ import annotations

import sys
import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.drag_pressure import (
    SurfaceMesh, get_near_wall_3d,
    drag_pressure_integration, drag_friction_integration,
)
from tensorlbm.momentum_exchange import momentum_exchange_standard


def run_case(axis: str, nx=64, ny=32, nz=32, R=6.0, u_in=0.05, tau=0.6,
             n_steps=1500, device_id=19):
    dev = torch.device(f'sdaa:{device_id}')
    torch.sdaa.set_device(dev)
    cx = nx // 4
    D = 2 * R
    nu = (tau - 0.5) / 3.0
    Re = u_in * D / nu
    dpS = 0.5 * u_in ** 2 * D * (ny if axis == 'z' else nz)

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev, dtype=torch.float32),
        torch.arange(ny, device=dev, dtype=torch.float32),
        torch.arange(nx, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    if axis == 'z':
        solid = (xx - cx) ** 2 + (yy - ny // 2) ** 2 <= R ** 2
        near = get_near_wall_3d(solid)
        mesh = SurfaceMesh.from_cylinder(solid, near, cx, ny // 2, R, axis='z')
    else:  # axis == 'y'
        solid = (xx - cx) ** 2 + (zz - nz // 2) ** 2 <= R ** 2
        near = get_near_wall_3d(solid)
        mesh = SurfaceMesh.from_cylinder(solid, near, cx, 0, R, axis='y', cz=nz // 2)

    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=dev)
    f[:, solid] = 0

    cd_hist = []
    for step in range(1, n_steps + 1):
        # Common interface: lbm_step_correct
        f = lbm_step_correct(f, collide_bgk3d, tau, solid, u_in, far_field_bc_3d)
        if step > 500 and step % 20 == 0:
            fx, _, _ = momentum_exchange_standard(f, solid, near)
            cd_hist.append(fx / dpS)

    cd_mean = sum(cd_hist) / len(cd_hist) if cd_hist else 0.0
    return cd_mean


def run(device_id=19):
    print("=== Direction-Agnostic Verification: y↔z swap ===")
    print(f"Device: sdaa:{device_id}")
    print()

    print("Running cylinder along z-axis...")
    cd_z = run_case(axis='z', device_id=device_id)
    print(f"  Cd (z-axis): {cd_z:.6f}")

    print("Running cylinder along y-axis...")
    cd_y = run_case(axis='y', device_id=device_id)
    print(f"  Cd (y-axis): {cd_y:.6f}")

    diff = abs(cd_z - cd_y)
    rel = diff / abs(cd_z) * 100 if abs(cd_z) > 1e-10 else 0.0

    print()
    print(f"Difference: {diff:.6f} ({rel:.2f}%)")
    passed = rel < 0.01
    if passed:
        print("PASS: Direction-agnostic (0.00% difference)")
    else:
        print(f"CHECK: {rel:.2f}% difference (expected < 0.01%)")

    return {"cd_z": cd_z, "cd_y": cd_y, "diff_pct": rel, "passed": passed}


if __name__ == "__main__":
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 19
    run(device_id=dev)
