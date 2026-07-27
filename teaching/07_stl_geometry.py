#!/usr/bin/env python
"""Teaching Example 07: STL geometry → voxelize → normals → drag.

Demonstrates the full STL workflow using the COMMON INTERFACE ONLY:
1. make_sphere_stl() → generate STL mesh (vertices, faces)
2. write_stl() → write to temp file
3. read_stl() → read back (vertices, faces, face_normals)
4. voxelize_stl() → boolean solid mask
5. SurfaceMesh_from_stl() → surface normals from STL
6. lbm_step_correct() → main loop
7. drag_pressure_integration / drag_friction_integration → force

Usage:
    PYTHONPATH=src python teaching/07_stl_geometry.py [device_id]
"""
from __future__ import annotations

import sys
import math
import tempfile

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
from tensorlbm.stl_geometry import (
    make_sphere_stl, write_stl, read_stl,
    voxelize_stl, SurfaceMesh_from_stl,
)


def run(
    nx=48, ny=40, nz=40, R=6.0, u_in=0.05, tau=0.6,
    n_steps=2000, warmup=800, device_id=19,
):
    dev = torch.device(f'sdaa:{device_id}')
    torch.sdaa.set_device(dev)
    cx, cy, cz = nx // 4, ny // 2, nz // 2
    D = 2 * R
    nu = (tau - 0.5) / 3.0
    Re = u_in * D / nu
    dpS = 0.5 * u_in ** 2 * math.pi * R ** 2
    Cd_ref = 1.09

    print(f"=== STL Geometry → Voxelize → Drag: Re={Re:.0f} (SDAA:{device_id}) ===")
    print(f"Grid: {nx}x{ny}x{nz}, R={R}, D={D}")
    print(f"Steps: {n_steps}, warmup: {warmup}")
    print()

    # 1. Generate STL mesh using common interface make_sphere_stl
    vertices, faces = make_sphere_stl((cx, cy, cz), R, n_lat=16, n_lon=32)
    print(f"STL mesh: {len(vertices)} vertices, {len(faces)} faces")

    # 2. Write STL to temp file, then read it back
    stl_path = tempfile.mktemp(suffix='.stl')
    write_stl(stl_path, vertices, faces, binary=True)
    vertices_r, faces_r, face_normals = read_stl(stl_path)
    print(f"read_stl: {len(vertices_r)} vertices, {len(faces_r)} faces, "
          f"{len(face_normals)} normals")

    # 3. Voxelize STL into boolean solid mask
    solid = voxelize_stl(vertices_r, faces_r, (nx, ny, nz),
                         origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    n_solid = int(solid.sum().item())
    print(f"Voxelize: {n_solid} solid cells")

    # 4. Near-wall mask and STL-derived normals
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"Near-wall cells: {n_near}")

    # Common interface: SurfaceMesh_from_stl for STL-derived normals
    mesh = SurfaceMesh_from_stl(solid, near, vertices_r, faces_r, face_normals,
                                origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    print(f"SurfaceMesh: normals from STL")
    print()

    # 5. LBM simulation using lbm_step_correct
    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=dev)
    f[:, solid] = 0

    cd_mem_hist = []
    cd_pf_hist = []

    print(f"{'Step':>6s}  {'Cd_MEM':>8s}  {'Cd_PF':>8s}  {'max|u|':>8s}")
    print("-" * 40)

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(f, collide_bgk3d, tau, solid, u_in, far_field_bc_3d)

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
            fluid = (~solid).float()
            ms = float(torch.sqrt((ux*fluid)**2 + (uy*fluid)**2 + (uz*fluid)**2).max().item())
            n = len(cd_mem_hist)
            if n > 0:
                print(f" {step:5d}  {cd_mem_hist[-1]:8.4f}  {cd_pf_hist[-1]:8.4f}  {ms:8.4f}")
            else:
                print(f" {step:5d}  {'---':>8s}  {'---':>8s}  {ms:8.4f}")

    import os; os.unlink(stl_path)

    n = len(cd_mem_hist)
    if n > 0:
        cd_mem = sum(cd_mem_hist) / n
        cd_pf = sum(cd_pf_hist) / n
        cd_err = abs(cd_pf - Cd_ref) / Cd_ref * 100
        print()
        print("=== Final ===")
        print(f"  Cd_MEM: {cd_mem:.4f}  (ref {Cd_ref})")
        print(f"  Cd_PF:  {cd_pf:.4f}  (err {cd_err:.1f}%)")
        passed = cd_err < 30.0
        print(f"  PASS (Cd<30%): {passed}")
        return {"cd_mem": cd_mem, "cd_pf": cd_pf, "cd_err_pct": cd_err,
                "Re": Re, "passed": passed}
    return {"Re": Re, "passed": False}


if __name__ == "__main__":
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 19
    run(device_id=dev)
