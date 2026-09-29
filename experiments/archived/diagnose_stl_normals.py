#!/usr/bin/env python3
"""Diagnose STL normal orientation for ship hulls.

Checks what fraction of STL face normals point outward (away from solid)
using different methods:
1. Gradient-based (current method)
2. Centroid-based
3. Ray casting (proposed fix)
4. Direct STL face normal (proposed simpler fix)
"""
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


def check_normal_orientation(solid, near, vertices, faces, face_normals, origin, spacing):
    """Check how many normals point outward using different methods."""
    nz, ny, nx = solid.shape
    solid_cpu = solid.cpu() if solid.device.type != "cpu" else solid
    near_cpu = near.cpu() if near.device.type != "cpu" else near

    near_idx = near_cpu.nonzero(as_tuple=False)
    n_near = near_idx.shape[0]
    print(f"  n_near = {n_near}")

    # Cell positions
    iz = near_idx[:, 0].numpy().astype(np.float64)
    iy = near_idx[:, 1].numpy().astype(np.float64)
    ix = near_idx[:, 2].numpy().astype(np.float64)
    px = origin[0] + (ix + 0.5) * spacing[0]
    py = origin[1] + (iy + 0.5) * spacing[1]
    pz = origin[2] + (iz + 0.5) * spacing[2]
    cell_pos = np.stack([px, py, pz], axis=1)

    # Triangle centroids
    tri_verts = vertices[faces]
    centroids = tri_verts.mean(axis=1)

    # Nearest triangle normals
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(centroids)
        _, tri_idx = tree.query(cell_pos, k=1)
    except ImportError:
        diffs = cell_pos[:, None, :] - centroids[None, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        tri_idx = np.argmin(dists, axis=1)

    normals = face_normals[tri_idx].astype(np.float64).copy()

    # Method 1: Gradient-based
    s = solid_cpu.float()
    gx = torch.zeros_like(s)
    gy = torch.zeros_like(s)
    gz = torch.zeros_like(s)
    gx[:, :, 1:-1] = (s[:, :, 2:] - s[:, :, :-2]) / 2
    gy[:, 1:-1, :] = (s[:, 2:, :] - s[:, :-2, :]) / 2
    gz[1:-1, :, :] = (s[2:, :, :] - s[:-2, :, :]) / 2
    near_f = near_cpu.float()
    gx = -gx * near_f
    gy = -gy * near_f
    gz = -gz * near_f
    norm = torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2).clamp(min=1e-10)
    grad_nx = (gx / norm)[near_idx[:, 0], near_idx[:, 1], near_idx[:, 2]].numpy()
    grad_ny = (gy / norm)[near_idx[:, 0], near_idx[:, 1], near_idx[:, 2]].numpy()
    grad_nz = (gz / norm)[near_idx[:, 0], near_idx[:, 1], near_idx[:, 2]].numpy()
    grad_normals = np.stack([grad_nx, grad_ny, grad_nz], axis=1)

    dot_grad = (normals * grad_normals).sum(axis=1)
    grad_outward = (dot_grad > 0).sum()
    grad_weak = (np.linalg.norm(grad_normals, axis=1) < 1e-6).sum()
    print(f"  Method 1 (gradient): {grad_outward}/{n_near} ({100*grad_outward/n_near:.1f}%) outward")
    print(f"    (dot<0: {(dot_grad < 0).sum()}, dot~0: {(np.abs(dot_grad) < 0.1).sum()}, weak_grad: {grad_weak})")
    print(f"    dot stats: mean={dot_grad.mean():.3f} std={dot_grad.std():.3f}")

    # Method 2: Centroid-based
    solid_coords = np.argwhere(solid_cpu.numpy()).astype(np.float64)
    solid_center = solid_coords.mean(axis=0)
    cell_to_center = cell_pos - solid_center
    ct_norm = np.linalg.norm(cell_to_center, axis=1, keepdims=True)
    ct_dir = cell_to_center / np.where(ct_norm > 1e-10, ct_norm, 1.0)
    dot_ct = (normals * ct_dir).sum(axis=1)
    ct_outward = (dot_ct > 0).sum()
    print(f"  Method 2 (centroid): {ct_outward}/{n_near} ({100*ct_outward/n_near:.1f}%) outward")
    print(f"    dot stats: mean={dot_ct.mean():.3f} std={dot_ct.std():.3f}")

    # Method 3: Ray casting
    # For each near-wall cell, cast a ray in +x direction and count intersections
    # with STL triangles. Odd = inside, even = outside.
    triangles = vertices[faces].astype(np.float64)
    # Möller–Trumbore ray-triangle intersection
    # Ray origin = cell_pos, direction = (1, 0, 0)
    ray_dir = np.array([1.0, 0.0, 0.0])

    # Vectorized ray casting
    # For each cell, we need to check all triangles
    # This is O(n_near * n_tri) which could be large, so we use a bounding box filter
    tri_min = triangles.min(axis=1)  # (n_tri, 3)
    tri_max = triangles.max(axis=1)  # (n_tri, 3)

    inside_count = 0
    outside_count = 0
    batch_size = 500  # Process in batches to avoid memory issues

    for batch_start in range(0, n_near, batch_size):
        batch_end = min(batch_start + batch_size, n_near)
        batch_pos = cell_pos[batch_start:batch_end]  # (B, 3)
        B = batch_pos.shape[0]

        # For each cell in batch, check all triangles
        # Bounding box filter: triangle must overlap [cell_x, +inf) in x
        # and overlap in y, z
        # cell_x = batch_pos[:, 0]  # (B,)
        # Only triangles with tri_min[:, 0] > cell_x can be hit (ray goes +x)
        # Actually, triangle is hit if tri_min_x <= cell_x is NOT required;
        # the ray starts at cell_x and goes +x, so triangle must have
        # some vertex with x > cell_x (tri_max_x > cell_x)
        # AND the triangle's y,z range must contain the cell's y,z

        hit_counts = np.zeros(B, dtype=np.int32)

        for ti in range(len(triangles)):
            tri = triangles[ti]  # (3, 3)
            # Quick bounding box check
            tri_xmin, tri_xmax = tri[:, 0].min(), tri[:, 0].max()
            tri_ymin, tri_ymax = tri[:, 1].min(), tri[:, 1].max()
            tri_zmin, tri_zmax = tri[:, 2].min(), tri[:, 2].max()

            # Ray goes +x from cell. Triangle must have x > cell_x (at least partially)
            # and cell y,z must be within triangle's y,z bounding box
            mask = (tri_xmax > batch_pos[:, 0]) & \
                   (batch_pos[:, 1] >= tri_ymin - 1e-10) & (batch_pos[:, 1] <= tri_ymax + 1e-10) & \
                   (batch_pos[:, 2] >= tri_zmin - 1e-10) & (batch_pos[:, 2] <= tri_zmax + 1e-10)

            if not mask.any():
                continue

            # Möller–Trumbore for the masked cells
            v0 = tri[0]  # (3,)
            v1 = tri[1]
            v2 = tri[2]
            edge1 = v1 - v0
            edge2 = v2 - v0
            h = np.cross(ray_dir, edge2)  # (3,)
            a = np.dot(edge1, h)

            if abs(a) < 1e-12:
                continue

            f_val = 1.0 / a
            s_vec = batch_pos[mask] - v0  # (M, 3)
            u = f_val * np.dot(s_vec, h)  # (M,)

            if (u < -1e-10).any() or (u > 1 + 1e-10).any():
                # Some may be out of range
                pass

            q = np.cross(s_vec, edge1)  # (M, 3)
            v = f_val * np.dot(q, ray_dir)  # (M,)

            t = f_val * np.dot(q, edge2)  # (M,)

            # Hit if u >= 0, v >= 0, u+v <= 1, t > 0
            hit = (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10) & (t > 1e-10)
            hit_counts[mask] += hit

        for i in range(B):
            if hit_counts[i] % 2 == 1:
                inside_count += 1
            else:
                outside_count += 1

    print(f"  Method 3 (ray cast +x): inside={inside_count} outside={outside_count}")
    print(f"    inside%: {100*inside_count/n_near:.1f}%")

    # For inside cells: normal should point outward (away from solid) = -ray_dir component
    # For outside cells: normal should point inward (toward solid)
    # "Outward" means pointing away from the solid surface
    # If cell is inside (fluid inside solid? No, near-wall cells are fluid)
    # Wait - near-wall cells are FLUID cells adjacent to solid.
    # If the ray from a fluid cell hits an odd number of triangles, the cell is INSIDE the solid.
    # If even, the cell is OUTSIDE the solid.
    # For a near-wall fluid cell that is OUTSIDE the solid (normal case):
    #   the outward normal should point AWAY from solid = toward the fluid = toward the cell
    # For a near-wall fluid cell that is INSIDE the solid (shouldn't happen for external flow):
    #   the outward normal should point toward the solid interior

    # Actually, for external flow around a ship:
    # near-wall cells are fluid cells just outside the hull
    # The "outward" normal (from solid to fluid) should point from the hull surface toward the fluid
    # = away from the solid center

    # The ray casting tells us if the cell is inside or outside the STL surface
    # For cells outside: the outward normal points from solid→fluid = from surface→cell
    # The STL face normal (if correctly oriented) points outward from the solid surface
    # So for outside cells, the STL normal should already point toward the cell (outward)

    # Let's check: does the STL face normal point toward or away from the cell?
    # Direction from nearest triangle centroid to cell
    nearest_centroids = centroids[tri_idx]
    tri_to_cell = cell_pos - nearest_centroids
    tc_norm = np.linalg.norm(tri_to_cell, axis=1, keepdims=True)
    tc_dir = tri_to_cell / np.where(tc_norm > 1e-10, tc_norm, 1.0)
    dot_tc = (normals * tc_dir).sum(axis=1)
    tc_outward = (dot_tc > 0).sum()
    print(f"  Method 4 (tri→cell dir): {tc_outward}/{n_near} ({100*tc_outward/n_near:.1f}%) outward")
    print(f"    dot stats: mean={dot_tc.mean():.3f} std={dot_tc.std():.3f}")

    # Method 5: Direct STL face normal (no flip)
    # Check if the stored STL normals are already outward
    # We can verify by checking against the gradient for cells where gradient is strong
    strong_grad = np.linalg.norm(grad_normals, axis=1) > 0.1
    if strong_grad.any():
        dot_strong = (normals[strong_grad] * grad_normals[strong_grad]).sum(axis=1)
        outward_strong = (dot_strong > 0).sum()
        print(f"  Method 5 (STL direct, strong-grad cells only): {outward_strong}/{strong_grad.sum()} ({100*outward_strong/strong_grad.sum():.1f}%) outward")

    return {
        "n_near": n_near,
        "grad_outward_pct": 100 * grad_outward / n_near,
        "ct_outward_pct": 100 * ct_outward / n_near,
        "tc_outward_pct": 100 * tc_outward / n_near,
        "inside_count": inside_count,
        "outside_count": outside_count,
    }


def main():
    ships = [
        ("KVLCC2", STL_DIR / "KVLCC2_Hull.stl"),
        ("DTMB5415", STL_DIR / "DTMB5415_Hull.stl"),
    ]

    for name, stl_path in ships:
        print(f"\n{'='*60}")
        print(f"Ship: {name}  STL: {stl_path}")
        print(f"{'='*60}")

        solid, near, vertices, faces, normals, origin, spacing = setup_grid(stl_path)
        print(f"  solid={int(solid.sum())} near={int(near.sum())} faces={len(faces)}")
        print(f"  spacing={spacing[0]:.4f}")

        results = check_normal_orientation(
            solid, near, vertices, faces, normals, origin, spacing
        )


if __name__ == "__main__":
    main()
