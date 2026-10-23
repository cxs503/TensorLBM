#!/usr/bin/env python3
"""Check the actual normals produced by SurfaceMesh_from_stl."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from tensorlbm.stl_geometry import read_stl, voxelize_stl, mirror_stl, SurfaceMesh_from_stl
from tensorlbm.drag_pressure import get_near_wall_3d

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

    # Build the mesh using current SurfaceMesh_from_stl
    mesh = SurfaceMesh_from_stl(
        solid, near, vertices, faces, face_normals.astype(np.float32),
        origin, spacing, dA_method="stl_area",
    )

    # Extract normals at near-wall cells
    near_idx = near.nonzero(as_tuple=False)
    iz = near_idx[:, 0].numpy()
    iy = near_idx[:, 1].numpy()
    ix = near_idx[:, 2].numpy()

    nx_n = mesh.nx_n[iz, iy, ix].numpy()
    ny_n = mesh.ny_n[iz, iy, ix].numpy()
    nz_n = mesh.nz_n[iz, iy, ix].numpy()
    mesh_normals = np.stack([nx_n, ny_n, nz_n], axis=1)

    # Cell positions
    px = origin[0] + (ix + 0.5) * spacing[0]
    py = origin[1] + (iy + 0.5) * spacing[1]
    pz = origin[2] + (iz + 0.5) * spacing[2]
    cell_pos = np.stack([px, py, pz], axis=1)

    # Triangle centroids
    tri_verts = vertices[faces]
    centroids = tri_verts.mean(axis=1)

    # Nearest triangle
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(centroids)
        _, tri_idx = tree.query(cell_pos, k=1)
    except ImportError:
        diffs = cell_pos[:, None, :] - centroids[None, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        tri_idx = np.argmin(dists, axis=1)

    # Check: do the final normals point toward the cell (outward)?
    nearest_centroids = centroids[tri_idx]
    tri_to_cell = cell_pos - nearest_centroids
    tc_norm = np.linalg.norm(tri_to_cell, axis=1, keepdims=True)
    tc_dir = tri_to_cell / np.where(tc_norm > 1e-10, tc_norm, 1.0)
    dot_tc = (mesh_normals * tc_dir).sum(axis=1)

    n_outward = (dot_tc > 0).sum()
    n_inward = (dot_tc < 0).sum()
    print(f"\n  Final mesh normals vs triangle→cell direction:")
    print(f"    outward (dot>0): {n_outward} ({100*n_outward/len(dot_tc):.1f}%)")
    print(f"    inward (dot<0): {n_inward} ({100*n_inward/len(dot_tc):.1f}%)")

    # Check x-component: for pressure drag, the x-component of the normal matters most
    # At the bow, outward normal should point -x (upstream)
    # At the stern, outward normal should point +x (downstream)
    # The net pressure drag should be positive (force in +x direction = drag)

    # Check: what's the sum of nx_n * dA over all near-wall cells?
    # This is the "closure error" — for a closed surface, sum(n*dA) should be ~0
    dA_vals = mesh.dA[iz, iy, ix].numpy()
    sum_nx_dA = (mesh_normals[:, 0] * dA_vals).sum()
    sum_ny_dA = (mesh_normals[:, 1] * dA_vals).sum()
    sum_nz_dA = (mesh_normals[:, 2] * dA_vals).sum()
    print(f"\n  Closure check (sum n*dA):")
    print(f"    x: {sum_nx_dA:.2f}")
    print(f"    y: {sum_ny_dA:.2f}")
    print(f"    z: {sum_nz_dA:.2f}")

    # Check: for cells at the bow (x < hull_center), what's the normal x-component?
    solid_coords = np.argwhere(solid.numpy()).astype(np.float64)
    solid_center = solid_coords.mean(axis=0)
    hull_center_x_lattice = solid_center[2]  # ix coordinate

    bow_mask = ix < hull_center_x_lattice - 5
    stern_mask = ix > hull_center_x_lattice + 5

    print(f"\n  Bow cells (ix < {hull_center_x_lattice - 5}): {bow_mask.sum()}")
    if bow_mask.any():
        bow_nx = mesh_normals[bow_mask, 0]
        print(f"    nx: mean={bow_nx.mean():.3f} (should be <0 for outward at bow)")
        print(f"    nx>0: {(bow_nx > 0).sum()}, nx<0: {(bow_nx < 0).sum()}")

    print(f"\n  Stern cells (ix > {hull_center_x_lattice + 5}): {stern_mask.sum()}")
    if stern_mask.any():
        stern_nx = mesh_normals[stern_mask, 0]
        print(f"    nx: mean={stern_nx.mean():.3f} (should be >0 for outward at stern)")
        print(f"    nx>0: {(stern_nx > 0).sum()}, nx<0: {(stern_nx < 0).sum()}")

    # Check: what fraction of bow cells have nx > 0 (WRONG - should be < 0)?
    if bow_mask.any():
        bow_wrong = (mesh_normals[bow_mask, 0] > 0).sum()
        print(f"\n  Bow cells with nx>0 (WRONG): {bow_wrong}/{bow_mask.sum()} ({100*bow_wrong/bow_mask.sum():.1f}%)")

    if stern_mask.any():
        stern_wrong = (mesh_normals[stern_mask, 0] < 0).sum()
        print(f"  Stern cells with nx<0 (WRONG): {stern_wrong}/{stern_mask.sum()} ({100*stern_wrong/stern_mask.sum():.1f}%)")


if __name__ == "__main__":
    main()
