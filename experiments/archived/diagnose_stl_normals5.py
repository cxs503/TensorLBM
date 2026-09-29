#!/usr/bin/env python3
"""Test closest-point-on-triangle approach for normal orientation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from tensorlbm.stl_geometry import read_stl, voxelize_stl, mirror_stl
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


def closest_point_on_triangle(p, a, b, c):
    """Closest point on triangle ABC to point p.
    
    Uses the algorithm from Real-Time Collision Detection by Christer Ericson.
    All inputs are (N, 3) arrays, p is (N, 3), a/b/c are (N, 3).
    Returns (N, 3) closest points.
    """
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = np.sum(ab * ap, axis=1)
    d2 = np.sum(ac * ap, axis=1)
    
    # Vertex region A
    mask = (d1 <= 0) & (d2 <= 0)
    
    bp = p - b
    d3 = np.sum(ab * bp, axis=1)
    d4 = np.sum(ac * bp, axis=1)
    
    # Vertex region B
    mask_b = (d3 >= 0) & (d4 <= d3)
    mask = mask | mask_b
    
    # Edge AB
    vc = d1 * d4 - d3 * d2
    mask_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~mask
    t_ab = np.where(d1 - d3 > 0, d1 / (d1 - d3 + 1e-30), 0)
    cp_ab = a + t_ab[:, None] * ab
    
    # Edge AC
    d5 = np.sum(ac * bp, axis=1)
    d6 = np.sum(ac * (p - c), axis=1)
    vb = d5 * d2 - d1 * d6
    mask_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & ~mask
    t_ac = np.where(d2 - d6 > 0, d2 / (d2 - d6 + 1e-30), 0)
    cp_ac = a + t_ac[:, None] * ac
    
    # Edge BC
    va = d3 * d6 - d5 * d4
    mask_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0) & ~mask
    t_bc = np.where(d4 - d3 - d5 + d6 > 0, (d4 - d3) / (d4 - d3 - d5 + d6 + 1e-30), 0)
    cp_bc = b + t_bc[:, None] * (c - b)
    
    # Inside
    denom = 1.0 / (va + vb + vc + 1e-30)
    v_inside = vb * denom
    w_inside = vc * denom
    cp_inside = a + v_inside[:, None] * ab + w_inside[:, None] * ac
    
    # Combine
    cp = np.where(mask[:, None], a, p)  # default to vertex A or B
    cp = np.where(mask_ab[:, None], cp_ab, cp)
    cp = np.where(mask_ac[:, None], cp_ac, cp)
    cp = np.where(mask_bc[:, None], cp_bc, cp)
    
    # Inside the triangle
    inside_mask = ~mask & ~mask_ab & ~mask_ac & ~mask_bc
    cp = np.where(inside_mask[:, None], cp_inside, cp)
    
    return cp


def check_normals(solid, near, vertices, faces, face_normals, origin, spacing, method_name, orient_fn):
    nz, ny, nx = solid.shape
    near_idx = near.nonzero(as_tuple=False)
    n_near = near_idx.shape[0]

    iz = near_idx[:, 0].numpy().astype(np.float64)
    iy = near_idx[:, 1].numpy().astype(np.float64)
    ix = near_idx[:, 2].numpy().astype(np.float64)
    px = origin[0] + (ix + 0.5) * spacing[0]
    py = origin[1] + (iy + 0.5) * spacing[1]
    pz = origin[2] + (iz + 0.5) * spacing[2]
    cell_pos = np.stack([px, py, pz], axis=1)

    tri_verts = vertices[faces]
    centroids = tri_verts.mean(axis=1)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(centroids)
        _, tri_idx = tree.query(cell_pos, k=1)
    except ImportError:
        diffs = cell_pos[:, None, :] - centroids[None, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        tri_idx = np.argmin(dists, axis=1)

    normals = face_normals[tri_idx].astype(np.float64).copy()
    normals = orient_fn(normals, cell_pos, centroids, tri_idx, vertices, faces, solid, near, near_idx)

    # Check: do normals point toward the cell (outward)?
    nearest_centroids = centroids[tri_idx]
    tri_to_cell = cell_pos - nearest_centroids
    tc_norm = np.linalg.norm(tri_to_cell, axis=1, keepdims=True)
    tc_dir = tri_to_cell / np.where(tc_norm > 1e-10, tc_norm, 1.0)
    dot_tc = (normals * tc_dir).sum(axis=1)
    n_outward = (dot_tc > 0).sum()

    # Closure check
    sum_nx = normals[:, 0].sum()
    sum_ny = normals[:, 1].sum()
    sum_nz = normals[:, 2].sum()

    # Bow/stern check
    solid_coords = np.argwhere(solid.numpy()).astype(np.float64)
    solid_center = solid_coords.mean(axis=0)
    hull_center_x_lattice = solid_center[2]

    bow_mask = ix < hull_center_x_lattice - 5
    stern_mask = ix > hull_center_x_lattice + 5

    bow_wrong = (normals[bow_mask, 0] > 0).sum() if bow_mask.any() else 0
    stern_wrong = (normals[stern_mask, 0] < 0).sum() if stern_mask.any() else 0

    # Also check: for bow cells, what's the mean nx?
    bow_nx_mean = normals[bow_mask, 0].mean() if bow_mask.any() else 0
    stern_nx_mean = normals[stern_mask, 0].mean() if stern_mask.any() else 0

    print(f"\n  {method_name}:")
    print(f"    outward: {n_outward}/{n_near} ({100*n_outward/n_near:.1f}%)")
    print(f"    closure: x={sum_nx:.1f} y={sum_ny:.1f} z={sum_nz:.1f}")
    print(f"    bow: wrong={bow_wrong}/{bow_mask.sum()} ({100*bow_wrong/bow_mask.sum():.1f}%) mean_nx={bow_nx_mean:.3f}")
    print(f"    stern: wrong={stern_wrong}/{stern_mask.sum()} ({100*stern_wrong/stern_mask.sum():.1f}%) mean_nx={stern_nx_mean:.3f}")

    return normals


def orient_gradient(normals, cell_pos, centroids, tri_idx, vertices, faces, solid, near, near_idx):
    """Current method: gradient-based flip."""
    s = solid.float()
    gx = torch.zeros_like(s)
    gy = torch.zeros_like(s)
    gz = torch.zeros_like(s)
    gx[:, :, 1:-1] = (s[:, :, 2:] - s[:, :, :-2]) / 2
    gy[:, 1:-1, :] = (s[:, 2:, :] - s[:, :-2, :]) / 2
    gz[1:-1, :, :] = (s[2:, :, :] - s[:-2, :, :]) / 2
    near_f = near.float()
    gx = -gx * near_f
    gy = -gy * near_f
    gz = -gz * near_f
    norm = torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2).clamp(min=1e-10)
    grad_nx = (gx / norm)[near_idx[:, 0], near_idx[:, 1], near_idx[:, 2]].numpy()
    grad_ny = (gy / norm)[near_idx[:, 0], near_idx[:, 1], near_idx[:, 2]].numpy()
    grad_nz = (gz / norm)[near_idx[:, 0], near_idx[:, 1], near_idx[:, 2]].numpy()
    grad_normals = np.stack([grad_nx, grad_ny, grad_nz], axis=1)

    dot = (normals * grad_normals).sum(axis=1)
    flip_mask = dot < 0
    normals[flip_mask] = -normals[flip_mask]

    grad_norm_arr = np.linalg.norm(grad_normals, axis=1)
    weak_grad = grad_norm_arr < 1e-6
    solid_coords = np.argwhere(solid.numpy()).astype(np.float64)
    if len(solid_coords) > 0 and weak_grad.any():
        solid_center = solid_coords.mean(axis=0)
        wg_idx = np.where(weak_grad)[0]
        cell_pos_wg = cell_pos[wg_idx]
        cell_to_center = cell_pos_wg - solid_center
        ct_norm = np.linalg.norm(cell_to_center, axis=1, keepdims=True)
        ct_dir = cell_to_center / np.where(ct_norm > 1e-10, ct_norm, 1.0)
        dot_ct = (normals[wg_idx] * ct_dir).sum(axis=1)
        inward = dot_ct < -0.3
        inward_idx = wg_idx[inward]
        normals[inward_idx] = -normals[inward_idx]

    zero_grad = grad_norm_arr < 1e-6
    if zero_grad.any():
        fallback_dir = cell_pos[zero_grad] - centroids[tri_idx[zero_grad]]
        fb_norm = np.linalg.norm(fallback_dir, axis=1, keepdims=True)
        fallback_dir = fallback_dir / np.where(fb_norm > 1e-10, fb_norm, 1.0)
        dot_fb = (normals[zero_grad] * fallback_dir).sum(axis=1)
        flip_fb = dot_fb < 0
        zg_idx = np.where(zero_grad)[0]
        flip_zg = zg_idx[flip_fb]
        normals[flip_zg] = -normals[flip_zg]

    return normals


def orient_tri_to_cell(normals, cell_pos, centroids, tri_idx, vertices, faces, solid, near, near_idx):
    """Proposed: flip normal to point toward cell (from nearest triangle centroid)."""
    nearest_centroids = centroids[tri_idx]
    tri_to_cell = cell_pos - nearest_centroids
    tc_norm = np.linalg.norm(tri_to_cell, axis=1, keepdims=True)
    tc_dir = tri_to_cell / np.where(tc_norm > 1e-10, tc_norm, 1.0)
    dot_tc = (normals * tc_dir).sum(axis=1)
    flip_mask = dot_tc < 0
    normals[flip_mask] = -normals[flip_mask]
    return normals


def orient_closest_point(normals, cell_pos, centroids, tri_idx, vertices, faces, solid, near, near_idx):
    """Proposed: flip normal to point toward cell (from closest point on triangle)."""
    # Get triangle vertices for each cell's nearest triangle
    tri_verts = vertices[faces]  # (n_tri, 3, 3)
    nearest_tris = tri_verts[tri_idx]  # (n_near, 3, 3)
    a = nearest_tris[:, 0, :]  # (n_near, 3)
    b = nearest_tris[:, 1, :]
    c = nearest_tris[:, 2, :]

    # Find closest point on each triangle to the cell
    cp = closest_point_on_triangle(cell_pos, a, b, c)  # (n_near, 3)

    # Direction from closest point to cell
    cp_to_cell = cell_pos - cp
    cp_norm = np.linalg.norm(cp_to_cell, axis=1, keepdims=True)
    cp_dir = cp_to_cell / np.where(cp_norm > 1e-10, cp_norm, 1.0)

    dot_cp = (normals * cp_dir).sum(axis=1)
    flip_mask = dot_cp < 0
    normals[flip_mask] = -normals[flip_mask]
    return normals


def main():
    for ship_name, stl_file in [("KVLCC2", "KVLCC2_Hull.stl"), ("DTMB5415", "DTMB5415_Hull.stl")]:
        stl_path = STL_DIR / stl_file
        print(f"\n{'='*60}")
        print(f"Ship: {ship_name}")
        print(f"{'='*60}")

        solid, near, vertices, faces, face_normals, origin, spacing = setup_grid(stl_path)
        print(f"  solid={int(solid.sum())} near={int(near.sum())}")

        # Method 1: Current gradient-based
        check_normals(solid, near, vertices, faces, face_normals, origin, spacing,
                      "Gradient (current)", orient_gradient)

        # Method 2: Triangle→cell direction (centroid)
        check_normals(solid, near, vertices, faces, face_normals, origin, spacing,
                      "Tri→cell (centroid)", orient_tri_to_cell)

        # Method 3: Closest point on triangle
        check_normals(solid, near, vertices, faces, face_normals, origin, spacing,
                      "Closest point", orient_closest_point)


if __name__ == "__main__":
    main()
