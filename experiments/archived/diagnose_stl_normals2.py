#!/usr/bin/env python3
"""Deeper diagnosis: check if gradient normals are correct."""
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


def main():
    stl_path = STL_DIR / "KVLCC2_Hull.stl"
    print(f"Ship: KVLCC2  STL: {stl_path}")

    solid, near, vertices, faces, face_normals, origin, spacing = setup_grid(stl_path)
    nz, ny, nx = solid.shape
    print(f"  solid={int(solid.sum())} near={int(near.sum())} faces={len(faces)}")
    print(f"  spacing={spacing[0]:.4f}")

    # Compute gradient normals
    s = solid.float()
    gx = torch.zeros_like(s)
    gy = torch.zeros_like(s)
    gz = torch.zeros_like(s)
    gx[:, :, 1:-1] = (s[:, :, 2:] - s[:, :, :-2]) / 2
    gy[:, 1:-1, :] = (s[:, 2:, :] - s[:, :-2, :]) / 2
    gz[1:-1, :, :] = (s[2:, :, :] - s[:-2, :, :]) / 2
    # Outward = -gradient (from solid to fluid)
    near_f = near.float()
    gx_out = -gx * near_f
    gy_out = -gy * near_f
    gz_out = -gz * near_f
    norm = torch.sqrt(gx_out ** 2 + gy_out ** 2 + gz_out ** 2).clamp(min=1e-10)
    grad_nx = gx_out / norm
    grad_ny = gy_out / norm
    grad_nz = gz_out / norm

    # Check gradient direction at near-wall cells
    near_idx = near.nonzero(as_tuple=False)
    iz = near_idx[:, 0].numpy()
    iy = near_idx[:, 1].numpy()
    ix = near_idx[:, 2].numpy()

    # For each near-wall cell, check if the gradient points away from solid
    # A near-wall cell is a fluid cell adjacent to at least one solid cell.
    # The gradient should point from solid→fluid = away from the nearest solid cell.
    # Let's verify: for each near-wall cell, find the nearest solid neighbor
    # and check if gradient points away from it.

    solid_np = solid.numpy()
    n_checked = 0
    n_correct = 0
    n_wrong = 0
    n_ambiguous = 0

    # Sample some cells
    sample_indices = np.random.choice(len(near_idx), min(1000, len(near_idx)), replace=False)

    for idx in sample_indices:
        z, y, x = iz[idx], iy[idx], ix[idx]
        gx_val = grad_nx[z, y, x].item()
        gy_val = grad_ny[z, y, x].item()
        gz_val = grad_nz[z, y, x].item()

        # Find solid neighbors
        solid_neighbors = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    zz, yy, xx = z + dz, y + dy, x + dx
                    if 0 <= zz < nz and 0 <= yy < ny and 0 <= xx < nx:
                        if solid_np[zz, yy, xx]:
                            solid_neighbors.append((dz, dy, dx))

        if len(solid_neighbors) == 0:
            n_ambiguous += 1
            continue

        # The gradient should point AWAY from solid neighbors
        # For each solid neighbor at (dz, dy, dx), the outward direction is (-dz, -dy, -dx)
        # Check if gradient is aligned with the average outward direction
        avg_out = np.array([-np.mean([s[0] for s in solid_neighbors]),
                             -np.mean([s[1] for s in solid_neighbors]),
                             -np.mean([s[2] for s in solid_neighbors])])
        avg_out_norm = np.linalg.norm(avg_out)
        if avg_out_norm < 1e-10:
            n_ambiguous += 1
            continue
        avg_out_dir = avg_out / avg_out_norm

        grad_vec = np.array([gx_val, gy_val, gz_val])
        grad_norm = np.linalg.norm(grad_vec)
        if grad_norm < 1e-6:
            n_ambiguous += 1
            continue
        grad_dir = grad_vec / grad_norm

        dot = np.dot(grad_dir, avg_out_dir)
        if dot > 0:
            n_correct += 1
        else:
            n_wrong += 1
        n_checked += 1

    print(f"\n  Gradient direction check (sampled {len(sample_indices)} cells):")
    print(f"    correct (points away from solid): {n_correct}/{n_checked} ({100*n_correct/n_checked:.1f}%)")
    print(f"    wrong (points toward solid): {n_wrong}/{n_checked} ({100*n_wrong/n_checked:.1f}%)")
    print(f"    ambiguous: {n_ambiguous}")

    # Now check: what does the gradient look like for cells at the bow/stern vs sides?
    # Bow/stern: x-dominant normal. Sides: y-dominant normal.
    # For the gradient to work, it needs to correctly identify the outward direction.

    # Check gradient x-component distribution
    gx_vals = grad_nx[near_idx[:, 0], near_idx[:, 1], near_idx[:, 2]].numpy()
    gy_vals = grad_ny[near_idx[:, 0], near_idx[:, 1], near_idx[:, 2]].numpy()
    gz_vals = grad_nz[near_idx[:, 0], near_idx[:, 1], near_idx[:, 2]].numpy()

    print(f"\n  Gradient component stats at near-wall cells:")
    print(f"    gx: mean={gx_vals.mean():.3f} std={gx_vals.std():.3f} "
          f"frac|gx|>0.1: {(np.abs(gx_vals) > 0.1).mean():.3f}")
    print(f"    gy: mean={gy_vals.mean():.3f} std={gy_vals.std():.3f} "
          f"frac|gy|>0.1: {(np.abs(gy_vals) > 0.1).mean():.3f}")
    print(f"    gz: mean={gz_vals.mean():.3f} std={gz_vals.std():.3f} "
          f"frac|gz|>0.1: {(np.abs(gz_vals) > 0.1).mean():.3f}")

    # Check: for cells where |gx| is dominant (bow/stern), is the gradient correct?
    abs_gx = np.abs(gx_vals)
    abs_gy = np.abs(gy_vals)
    abs_gz = np.abs(gz_vals)
    x_dominant = (abs_gx > abs_gy) & (abs_gx > abs_gz) & (abs_gx > 0.1)
    y_dominant = (abs_gy > abs_gx) & (abs_gy > abs_gz) & (abs_gy > 0.1)
    z_dominant = (abs_gz > abs_gx) & (abs_gz > abs_gy) & (abs_gz > 0.1)

    print(f"\n  Dominant gradient direction:")
    print(f"    x-dominant: {x_dominant.sum()} cells")
    print(f"    y-dominant: {y_dominant.sum()} cells")
    print(f"    z-dominant: {z_dominant.sum()} cells")
    print(f"    none/weak: {(~x_dominant & ~y_dominant & ~z_dominant).sum()} cells")

    # For x-dominant cells (bow/stern), check if gradient points in +x or -x
    if x_dominant.any():
        x_dom_gx = gx_vals[x_dominant]
        print(f"\n  x-dominant cells (bow/stern):")
        print(f"    gx>0 (pointing +x, toward stern/wake): {(x_dom_gx > 0).sum()}")
        print(f"    gx<0 (pointing -x, toward bow/inflow): {(x_dom_gx < 0).sum()}")

    # Now check the STL face normals
    # Cell positions
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

    stl_normals = face_normals[tri_idx].astype(np.float64)

    print(f"\n  STL face normal stats at near-wall cells:")
    print(f"    nx: mean={stl_normals[:, 0].mean():.3f} std={stl_normals[:, 0].std():.3f}")
    print(f"    ny: mean={stl_normals[:, 1].mean():.3f} std={stl_normals[:, 1].std():.3f}")
    print(f"    nz: mean={stl_normals[:, 2].mean():.3f} std={stl_normals[:, 2].std():.3f}")

    # Check: do STL normals point toward or away from the solid center?
    solid_coords = np.argwhere(solid.numpy()).astype(np.float64)
    solid_center = solid_coords.mean(axis=0)
    cell_to_center = cell_pos - solid_center
    ct_norm = np.linalg.norm(cell_to_center, axis=1, keepdims=True)
    ct_dir = cell_to_center / np.where(ct_norm > 1e-10, ct_norm, 1.0)
    dot_ct = (stl_normals * ct_dir).sum(axis=1)
    print(f"\n  STL normal vs centroid direction:")
    print(f"    dot>0 (outward from center): {(dot_ct > 0).sum()} ({100*(dot_ct > 0).mean():.1f}%)")
    print(f"    dot<0 (inward toward center): {(dot_ct < 0).sum()} ({100*(dot_ct < 0).mean():.1f}%)")

    # Check: do STL normals point toward or away from the cell (from nearest triangle)?
    nearest_centroids = centroids[tri_idx]
    tri_to_cell = cell_pos - nearest_centroids
    tc_norm = np.linalg.norm(tri_to_cell, axis=1, keepdims=True)
    tc_dir = tri_to_cell / np.where(tc_norm > 1e-10, tc_norm, 1.0)
    dot_tc = (stl_normals * tc_dir).sum(axis=1)
    print(f"\n  STL normal vs triangle→cell direction:")
    print(f"    dot>0 (normal points toward cell = outward): {(dot_tc > 0).sum()} ({100*(dot_tc > 0).mean():.1f}%)")
    print(f"    dot<0 (normal points away from cell = inward): {(dot_tc < 0).sum()} ({100*(dot_tc < 0).mean():.1f}%)")

    # Key question: after the gradient flip, what fraction of normals point outward?
    # The flip logic: flip if dot(stl_normal, grad_normal) < 0
    grad_normals = np.stack([gx_vals, gy_vals, gz_vals], axis=1)
    grad_norms = np.linalg.norm(grad_normals, axis=1, keepdims=True)
    grad_normals_unit = grad_normals / np.where(grad_norms > 1e-10, grad_norms, 1.0)

    dot_grad = (stl_normals * grad_normals_unit).sum(axis=1)
    flip_mask = dot_grad < 0
    flipped_normals = stl_normals.copy()
    flipped_normals[flip_mask] = -flipped_normals[flip_mask]

    # After flip, check against triangle→cell direction
    dot_tc_after = (flipped_normals * tc_dir).sum(axis=1)
    print(f"\n  AFTER gradient flip:")
    print(f"    dot_tc>0 (outward): {(dot_tc_after > 0).sum()} ({100*(dot_tc_after > 0).mean():.1f}%)")
    print(f"    dot_tc<0 (inward): {(dot_tc_after < 0).sum()} ({100*(dot_tc_after < 0).mean():.1f}%)")

    # After flip, check against centroid direction
    dot_ct_after = (flipped_normals * ct_dir).sum(axis=1)
    print(f"    dot_ct>0 (outward from center): {(dot_ct_after > 0).sum()} ({100*(dot_ct_after > 0).mean():.1f}%)")
    print(f"    dot_ct<0 (inward toward center): {(dot_ct_after < 0).sum()} ({100*(dot_ct_after < 0).mean():.1f}%)")

    # What if we just flip ALL STL normals (since they're mostly inward)?
    all_flipped = -stl_normals
    dot_tc_all = (all_flipped * tc_dir).sum(axis=1)
    print(f"\n  If we flip ALL STL normals:")
    print(f"    dot_tc>0 (outward): {(dot_tc_all > 0).sum()} ({100*(dot_tc_all > 0).mean():.1f}%)")

    # What if we use triangle→cell direction to orient normals?
    # Flip if normal points away from cell (dot_tc < 0)
    flip_tc = dot_tc < 0
    tc_flipped = stl_normals.copy()
    tc_flipped[flip_tc] = -tc_flipped[flip_tc]
    # Check against gradient
    dot_grad_tc = (tc_flipped * grad_normals_unit).sum(axis=1)
    print(f"\n  If we orient using triangle→cell direction:")
    print(f"    dot_grad>0 (agrees with gradient): {(dot_grad_tc > 0).sum()} ({100*(dot_grad_tc > 0).mean():.1f}%)")
    print(f"    dot_grad<0 (disagrees with gradient): {(dot_grad_tc < 0).sum()} ({100*(dot_grad_tc < 0).mean():.1f}%)")


if __name__ == "__main__":
    main()
