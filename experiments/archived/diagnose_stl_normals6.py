#!/usr/bin/env python3
"""Investigate z-component of normals for KVLCC2."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from tensorlbm.stl_geometry import read_stl, voxelize_stl, mirror_stl, SurfaceMesh_from_stl
from tensorlbm.drag_pressure import get_near_wall_3d, SurfaceMesh

STL_DIR = Path(
    "/root/ship-performance-platform-incoming/ship-performance-platform/"
    "backend/data/geometry/ships"
)


def setup_grid(stl_path, nx=200, ny=80, nz=80):
    vertices, faces, face_normals = read_stl(stl_path)
    vertices_full, faces_full, normals_full = mirror_stl(
        vertices, faces, face_normals, axis=1
    )

    x_min, x_max = vertices_full[:, 0].min(), vertices_full[:, 0].max()
    y_min, y_max = vertices_full[:, 1].min(), vertices_full[:, 1].max()
    z_min, z_max = vertices_full[:, 2].min(), vertices_full[:, 2].max()
    L_stl = x_max - x_min
    L_lattice = int(nx * 0.6)
    spacing = L_stl / L_lattice

    hull_center_x = (x_min + x_max) / 2.0
    origin_x = hull_center_x - (nx * 0.35) * spacing
    hull_center_y = (y_min + y_max) / 2.0
    origin_y = hull_center_y - (ny * 0.5) * spacing
    origin_z = 0.0 - (nz * 0.5) * spacing

    origin = (origin_x, origin_y, origin_z)
    sp = (spacing, spacing, spacing)

    solid = voxelize_stl(vertices_full, faces_full, (nx, ny, nz), origin, sp)
    near = get_near_wall_3d(solid)

    return solid, near, vertices_full, faces_full, normals_full, origin, sp


def main():
    stl_path = STL_DIR / "KVLCC2_Hull.stl"
    print(f"Ship: KVLCC2")

    solid, near, vertices, faces, face_normals, origin, spacing = setup_grid(stl_path)
    nz, ny, nx = solid.shape

    # Build STL mesh
    mesh_stl = SurfaceMesh_from_stl(
        solid, near, vertices, faces, face_normals.astype(np.float32),
        origin, spacing, dA_method="stl_area",
    )

    # Build gradient mesh
    mesh_grad = SurfaceMesh.from_gradient(solid, near)

    near_idx = near.nonzero(as_tuple=False)
    iz = near_idx[:, 0].numpy()
    iy = near_idx[:, 1].numpy()
    ix = near_idx[:, 2].numpy()

    # Compare STL vs gradient normals
    stl_nx = mesh_stl.nx_n[iz, iy, ix].numpy()
    stl_ny = mesh_stl.ny_n[iz, iy, ix].numpy()
    stl_nz = mesh_stl.nz_n[iz, iy, ix].numpy()
    stl_dA = mesh_stl.dA[iz, iy, ix].numpy()

    grad_nx = mesh_grad.nx_n[iz, iy, ix].numpy()
    grad_ny = mesh_grad.ny_n[iz, iy, ix].numpy()
    grad_nz = mesh_grad.nz_n[iz, iy, ix].numpy()

    print(f"\n  STL normals: nx mean={stl_nx.mean():.4f} ny mean={stl_ny.mean():.4f} nz mean={stl_nz.mean():.4f}")
    print(f"  Grad normals: nx mean={grad_nx.mean():.4f} ny mean={grad_ny.mean():.4f} nz mean={grad_nz.mean():.4f}")

    print(f"\n  STL |nx| mean={np.abs(stl_nx).mean():.4f} |ny| mean={np.abs(stl_ny).mean():.4f} |nz| mean={np.abs(stl_nz).mean():.4f}")
    print(f"  Grad |nx| mean={np.abs(grad_nx).mean():.4f} |ny| mean={np.abs(grad_ny).mean():.4f} |nz| mean={np.abs(grad_nz).mean():.4f}")

    # Closure
    print(f"\n  STL closure: x={np.sum(stl_nx*stl_dA):.1f} y={np.sum(stl_ny*stl_dA):.1f} z={np.sum(stl_nz*stl_dA):.1f}")
    print(f"  Grad closure: x={np.sum(grad_nx):.1f} y={np.sum(grad_ny):.1f} z={np.sum(grad_nz):.1f}")

    # Check z-distribution: how many normals have |nz| > 0.5?
    stl_nz_dominant = np.abs(stl_nz) > 0.5
    grad_nz_dominant = np.abs(grad_nz) > 0.5
    print(f"\n  |nz|>0.5: STL={stl_nz_dominant.sum()} Grad={grad_nz_dominant.sum()}")

    # Check: for cells where STL has large |nz| but gradient has small |nz|
    stl_z_large_grad_z_small = (np.abs(stl_nz) > 0.5) & (np.abs(grad_nz) < 0.1)
    print(f"  STL|nz|>0.5 & Grad|nz|<0.1: {stl_z_large_grad_z_small.sum()}")

    # Check: what's the z-distribution of near-wall cells?
    # z=0 is the waterline (origin_z = -nz/2 * spacing)
    # Cells below z=0 are underwater (hull bottom), above are above water
    cell_z = origin[2] + (iz + 0.5) * spacing[2]
    underwater = cell_z < 0
    above_water = cell_z >= 0
    print(f"\n  Underwater cells: {underwater.sum()} Above water: {above_water.sum()}")

    # For underwater cells, check normal z-component
    if underwater.any():
        print(f"  Underwater STL nz: mean={stl_nz[underwater].mean():.4f} |nz| mean={np.abs(stl_nz[underwater]).mean():.4f}")
        print(f"  Underwater Grad nz: mean={grad_nz[underwater].mean():.4f} |nz| mean={np.abs(grad_nz[underwater]).mean():.4f}")

    # The key question: what's the pressure drag contribution from z-normals?
    # F_x = -sum(p * nx * dA) — this is what we want
    # But if the normals have a large z-component, the pressure force is
    # mostly in the z-direction, not x. This doesn't directly affect F_x
    # unless the pressure field is correlated with the z-normal.

    # Actually, the issue might be simpler: the STL normals at the bow/stern
    # have very small |nx| (because the hull is slender), so the pressure
    # drag (which depends on nx) is small. The gradient normals, being
    # staircase normals, have larger |nx| at the bow/stern because the
    # staircase steps create x-facing surfaces.

    # Check |nx| at bow/stern
    solid_coords = np.argwhere(solid.numpy()).astype(np.float64)
    solid_center = solid_coords.mean(axis=0)
    hull_center_x_lattice = solid_center[2]
    bow_mask = ix < hull_center_x_lattice - 5
    stern_mask = ix > hull_center_x_lattice + 5

    print(f"\n  Bow cells |nx|: STL={np.abs(stl_nx[bow_mask]).mean():.4f} Grad={np.abs(grad_nx[bow_mask]).mean():.4f}")
    print(f"  Stern cells |nx|: STL={np.abs(stl_nx[stern_mask]).mean():.4f} Grad={np.abs(grad_nx[stern_mask]).mean():.4f}")

    # Check: what fraction of bow/stern cells have |nx| > 0.1?
    print(f"  Bow |nx|>0.1: STL={((np.abs(stl_nx[bow_mask]) > 0.1).sum())/bow_mask.sum():.3f} Grad={((np.abs(grad_nx[bow_mask]) > 0.1).sum())/bow_mask.sum():.3f}")
    print(f"  Stern |nx|>0.1: STL={((np.abs(stl_nx[stern_mask]) > 0.1).sum())/stern_mask.sum():.3f} Grad={((np.abs(grad_nx[stern_mask]) > 0.1).sum())/stern_mask.sum():.3f}")

    # The real issue: for the gradient method, the bow/stern cells have
    # staircase normals with significant |nx|, creating pressure drag.
    # For the STL method, the bow/stern normals have very small |nx|
    # (the hull is slender), so the pressure drag is small.
    # But the STL method gives NEGATIVE Cd_p, which means the pressure
    # drag is actually in the wrong direction.

    # Let's check: what's the sign of nx at the bow for STL vs gradient?
    print(f"\n  Bow nx sign: STL mean={stl_nx[bow_mask].mean():.4f} Grad mean={grad_nx[bow_mask].mean():.4f}")
    print(f"  Stern nx sign: STL mean={stl_nx[stern_mask].mean():.4f} Grad mean={grad_nx[stern_mask].mean():.4f}")

    # For positive pressure drag:
    # Bow: high pressure (stagnation) * nx (should be negative) → positive contribution to -sum(p*nx*dA)
    # Stern: low pressure * nx (should be positive) → positive contribution to -sum(p*nx*dA)
    # So we need bow nx < 0 and stern nx > 0

    # Check: how many bow cells have nx > 0 (WRONG)?
    print(f"\n  Bow nx>0 (WRONG): STL={(stl_nx[bow_mask] > 0).sum()}/{bow_mask.sum()} Grad={(grad_nx[bow_mask] > 0).sum()}/{bow_mask.sum()}")
    print(f"  Stern nx<0 (WRONG): STL={(stl_nx[stern_mask] < 0).sum()}/{stern_mask.sum()} Grad={(grad_nx[stern_mask] < 0).sum()}/{stern_mask.sum()}")


if __name__ == "__main__":
    main()
