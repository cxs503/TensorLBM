#!/usr/bin/env python
"""Teaching Example 07: STL geometry → voxelize → normals → drag.

Demonstrates the full STL workflow:
1. Load STL mesh (vertices + faces + normals)
2. Voxelize into a boolean solid mask
3. Compute near-wall mask and surface normals from STL
4. Run LBM simulation and compute drag

Uses a simple sphere STL for demonstration (no external STL file needed).

Usage:
    PYTHONPATH=src python teaching/07_stl_geometry.py [--device sdaa:11]
"""
from __future__ import annotations

import argparse
import math

import torch
import numpy as np

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.drag_pressure import SurfaceMesh, get_near_wall_3d, drag_pressure_integration, drag_friction_integration
from tensorlbm.momentum_exchange import momentum_exchange_standard


def make_sphere_stl(R=6.0, n_seg=16):
    """Generate a sphere STL mesh (vertices, faces, normals)."""
    # UV sphere
    vertices = []
    faces = []
    normals = []

    for i in range(n_seg + 1):
        theta = math.pi * i / n_seg
        for j in range(n_seg):
            phi = 2 * math.pi * j / n_seg
            x = R * math.sin(theta) * math.cos(phi)
            y = R * math.sin(theta) * math.sin(phi)
            z = R * math.cos(theta)
            vertices.append([x, y, z])

    for i in range(n_seg):
        for j in range(n_seg):
            i1 = i * n_seg + j
            i2 = i * n_seg + (j + 1) % n_seg
            i3 = (i + 1) * n_seg + j
            i4 = (i + 1) * n_seg + (j + 1) % n_seg
            if i > 0:
                faces.append([i1, i3, i2])
                v = np.array(vertices[i1])
                normals.append(v / np.linalg.norm(v))
            if i < n_seg - 1:
                faces.append([i2, i3, i4])
                v = np.array(vertices[i2])
                normals.append(v / np.linalg.norm(v))

    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int64), np.array(normals, dtype=np.float32)


def voxelize_sphere(vertices, faces, nx, ny, nz, cx, cy, cz, device):
    """Voxelize STL mesh into boolean mask (sphere approximation)."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # For sphere: distance from center
    R = float(np.max(np.linalg.norm(vertices, axis=1)))
    return (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= R ** 2


def run(
    nx=48, ny=40, nz=40, R=6.0, u_in=0.05, tau=0.6,
    n_steps=1500, warmup=500, device="sdaa:11",
):
    dev = torch.device(device)
    cx, cy, cz = nx // 4, ny // 2, nz // 2
    D = 2 * R
    nu = (tau - 0.5) / 3.0
    Re = u_in * D / nu
    dpS = 0.5 * u_in ** 2 * math.pi * R ** 2

    print(f"=== STL Geometry → Voxelize → Drag: Re={Re:.0f} ===")
    print(f"Grid: {nx}×{ny}×{nz}, R={R}, D={D}")
    print(f"Device: {device}, steps={n_steps}")
    print()

    # 1. Generate STL mesh
    vertices, faces, face_normals = make_sphere_stl(R=R)
    print(f"STL mesh: {len(vertices)} vertices, {len(faces)} faces")

    # 2. Voxelize
    solid = voxelize_sphere(vertices, faces, nx, ny, nz, cx, cy, cz, dev)
    n_solid = int(solid.sum().item())
    print(f"Voxelize: {n_solid} solid cells")

    # 3. Near-wall mask and normals
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"Near-wall cells: {n_near}")

    # Use analytical sphere normals (in practice, from STL)
    mesh = SurfaceMesh.from_sphere(solid, near, cx, cy, cz, R)
    print(f"SurfaceMesh: normals computed from sphere geometry")
    print()

    # 4. LBM simulation
    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=dev)
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
                print(f" {step:5d}  {cd_mem_hist[-1]:8.4f}  {cd_pf_hist[-1]:8.4f}  {ms:8.4f}")
            else:
                print(f" {step:5d}  {'---':>8s}  {'---':>8s}  {ms:8.4f}")

    n = len(cd_mem_hist)
    if n > 0:
        cd_mem = sum(cd_mem_hist) / n
        cd_pf = sum(cd_pf_hist) / n
        print()
        print("=== Final ===")
        print(f"  Cd_MEM: {cd_mem:.4f}  (target ~1.09 for Re=100)")
        print(f"  Cd_PF:  {cd_pf:.4f}")
        return {"cd_mem": cd_mem, "cd_pf": cd_pf, "Re": Re}
    return {"Re": Re}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STL geometry → drag")
    parser.add_argument("--device", default="sdaa:11")
    parser.add_argument("--n-steps", type=int, default=1500)
    args = parser.parse_args()
    run(n_steps=args.n_steps, device=args.device)
