#!/usr/bin/env python
"""Teaching Example 06: Direction-agnostic verification — y↔z swap.

Verifies that the LBM solver is direction-agnostic: swapping the y and z
axes of the geometry should give identical drag results.

Tests a cylinder oriented along z vs along y.  Both should give the
same Cd (within numerical precision).

Usage:
    PYTHONPATH=src python teaching/06_direction_agnostic.py [--device sdaa:11]
"""
from __future__ import annotations

import argparse
import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.drag_pressure import SurfaceMesh, get_near_wall_3d, drag_pressure_integration, drag_friction_integration
from tensorlbm.momentum_exchange import momentum_exchange_standard


def run_case(axis: str, nx=64, ny=32, nz=32, R=6.0, u_in=0.05, tau=0.6,
             n_steps=1500, device="sdaa:11"):
    dev = torch.device(device)
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
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        f = bounce_back_cells_3d(f, solid, f_pre=f_pre)
        f = stream3d(f)
        f = far_field_bc_3d(f, u_in=u_in, obstacle_mask=solid)
        if step > 500 and step % 20 == 0:
            fx, _, _ = momentum_exchange_standard(f, solid, near)
            cd_hist.append(fx / dpS)

    cd_mean = sum(cd_hist) / len(cd_hist) if cd_hist else 0.0
    return cd_mean


def run(device="sdaa:11"):
    print("=== Direction-Agnostic Verification: y↔z swap ===")
    print(f"Device: {device}")
    print()

    print("Running cylinder along z-axis...")
    cd_z = run_case(axis='z', device=device)
    print(f"  Cd (z-axis): {cd_z:.6f}")

    print("Running cylinder along y-axis...")
    cd_y = run_case(axis='y', device=device)
    print(f"  Cd (y-axis): {cd_y:.6f}")

    diff = abs(cd_z - cd_y)
    rel = diff / abs(cd_z) * 100 if abs(cd_z) > 1e-10 else 0.0

    print()
    print(f"Difference: {diff:.6f} ({rel:.2f}%)")
    if rel < 0.01:
        print("PASS: Direction-agnostic (0.00% difference)")
    else:
        print(f"CHECK: {rel:.2f}% difference (expected < 0.01%)")

    return {"cd_z": cd_z, "cd_y": cd_y, "diff_pct": rel}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direction-agnostic verification")
    parser.add_argument("--device", default="sdaa:11")
    args = parser.parse_args()
    run(device=args.device)
