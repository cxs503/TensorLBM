#!/usr/bin/env python3
"""Test the triangle→cell direction approach for normal orientation."""
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
    normals = orient_fn(normals, cell_pos, centroids, tri_idx, solid, near, near_idx)

    # Check: do normals point toward the cell (outward)?
    nearest_centroids = centroids[tri_idx]
    tri_to_cell = cell_pos - nearest_centroids
    tc_norm = np.linalg.norm(tri_to_cell, axis=1, keepdims=True)
    tc_dir = tri_to_cell / np.where(tc_norm > 1e-10, tc_norm, 1.0)
    dot_tc = (normals * tc_dir).sum(axis=1)
    n_outward = (dot_tc > 0).sum()

    # Closure check
    # Use dA=1 for simplicity
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

    print(f"\n  {method_name}:")
    print(f"    outward: {n_outward}/{n_near} ({100*n_outward/n_near:.1f}%)")
    print(f"    closure: x={sum_nx:.1f} y={sum_ny:.1f} z={sum_nz:.1f}")
    print(f"    bow wrong: {bow_wrong}/{bow_mask.sum()} ({100*bow_wrong/bow_mask.sum():.1f}%)")
    print(f"    stern wrong: {stern_wrong}/{stern_mask.sum()} ({100*stern_wrong/stern_mask.sum():.1f}%)")

    return normals


def orient_gradient(normals, cell_pos, centroids, tri_idx, solid, near, near_idx):
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

    # Centroid fallback for weak gradient
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

    # Zero-gradient fallback
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


def orient_tri_to_cell(normals, cell_pos, centroids, tri_idx, solid, near, near_idx):
    """Proposed method: flip normal to point toward cell (away from solid)."""
    nearest_centroids = centroids[tri_idx]
    tri_to_cell = cell_pos - nearest_centroids
    tc_norm = np.linalg.norm(tri_to_cell, axis=1, keepdims=True)
    tc_dir = tri_to_cell / np.where(tc_norm > 1e-10, tc_norm, 1.0)
    dot_tc = (normals * tc_dir).sum(axis=1)
    flip_mask = dot_tc < 0
    normals[flip_mask] = -normals[flip_mask]
    return normals


def orient_ray_cast(normals, cell_pos, centroids, tri_idx, solid, near, near_idx):
    """Proposed method: ray casting to determine inside/outside, then orient."""
    # Cast ray in +x direction from each cell
    # If odd intersections → inside solid → normal should point toward cell (outward from solid)
    # If even → outside → normal should point toward solid (inward)
    # But wait: near-wall cells are FLUID cells. For external flow:
    #   - outside the solid → normal should point away from solid = toward cell
    #   - inside the solid (shouldn't happen for external flow)
    # Actually, for a closed solid, near-wall fluid cells are always OUTSIDE.
    # So we should always orient normals to point toward the cell.
    # But ray casting can tell us if the cell is truly outside.

    triangles = solid  # placeholder
    # Actually, let's just use the tri_to_cell approach since ray casting
    # confirms cells are outside
    return orient_tri_to_cell(normals, cell_pos, centroids, tri_idx, solid, near, near_idx)


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

        # Method 2: Triangle→cell direction
        check_normals(solid, near, vertices, faces, face_normals, origin, spacing,
                      "Tri→cell (proposed)", orient_tri_to_cell)


if __name__ == "__main__":
    main()
