#!/usr/bin/env python3
"""Build the genuine DTMB 5415 underwater bare-hull STL for the benchmark.

Provenance
----------
The mesh is generated from the **zenodo / CNR-INSEAN official DTMB 5415
baseline point cloud** (2250 surface points, dataset:
``/tmp/dr5415/{DTMB5415-database.mat,baseline_cloud.npy}``; Serani et al.
2022/2025) via the sibling project's verified generator
``ship-performance-platform/backend/api/zenodo_5415.generate_official_5415_mesh``
(100 % official-point match, watertight after weld + fill).

This script:
  1. calls that generator,
  2. clips the surface at the design waterline ``z = T = 6.15 m`` (full-scale
     Lpp = 142 m, B = 19.06 m, T = 6.15 m, Cb = 0.507; baseline z = 0, the
     sonar-dome bulb reaches z ~ -3.28 m),
  3. writes a binary STL ``geometry/dtmb5415_underwater.stl``,
  4. voxelizes it with the repo's own ``tensorlbm.stl_geometry.voxelize_stl``
     and reports the numerical block coefficient + wetted-cell count.

The output STL is committed so ``run.py`` does not depend on the sibling
project at run time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_GEOM = _HERE / "geometry"
_PLATFORM_BACKEND = Path(
    "/root/ship-performance-platform-incoming/ship-performance-platform/backend"
)

# Full-scale principal dimensions (standard_models_detailed.py, Stern 1996)
LPP = 142.0  # m, length between perpendiculars
BEAM = 19.06  # m
DRAFT = 6.15  # m, design waterline above baseline (z = 0)
Z_WL = DRAFT  # waterline plane


def main() -> None:
    sys.path.insert(0, str(_PLATFORM_BACKEND))
    sys.path.insert(0, str(Path("/root/TensorLBM_feat2/src")))
    from api.zenodo_5415 import generate_official_5415_mesh  # type: ignore

    from tensorlbm.stl_geometry import voxelize_stl, write_stl

    mesh = generate_official_5415_mesh()
    verts = np.asarray(mesh["vertices"], dtype=np.float64)
    faces = np.asarray(mesh["indices"], dtype=np.int64).reshape(-1, 3)
    print(f"official mesh: {verts.shape[0]} verts, {faces.shape[0]} tris, "
          f"match={mesh['match_pct']}%")

    # Keep triangles whose centroid is at/below the waterline.
    cz = verts[faces, 2].mean(axis=1)
    keep = cz <= Z_WL
    faces_u = faces[keep]
    print(f"underwater clip: kept {faces_u.shape[0]}/{faces.shape[0]} tris "
          f"(centroid z <= {Z_WL})")

    _GEOM.mkdir(parents=True, exist_ok=True)
    stl_path = _GEOM / "dtmb5415_underwater.stl"
    write_stl(stl_path, verts, faces_u, binary=True)
    print(f"wrote {stl_path} ({stl_path.stat().st_size/1e6:.2f} MB)")

    # ---- voxel sanity: numerical Cb of the underwater body --------------
    L_cells = 250
    dx = LPP / L_cells
    x_min = float(verts[:, 0].min()) - 2 * dx
    x_max = float(verts[:, 0].max()) + 2 * dx
    y_half = BEAM / 2 + 2 * dx
    z_min = float(verts[:, 2].min()) - 2 * dx
    nx = int(np.ceil((x_max - x_min) / dx))
    ny = int(np.ceil(2 * y_half / dx))
    nz = int(np.ceil((Z_WL - z_min) / dx))
    solid = voxelize_stl(
        verts, faces_u, (nx, ny, nz),
        origin=(x_min, -y_half, z_min), spacing=(dx, dx, dx),
    )
    n_solid = int(solid.sum().item())
    volume = n_solid * dx**3
    cb_num = volume / (LPP * BEAM * DRAFT)
    print(f"grid {nx}x{ny}x{nz} dx={dx:.3f} m  solid={n_solid}")
    print(f"underwater volume = {volume:.1f} m^3  ->  Cb_num = {cb_num:.4f}")
    print(f"(reference Cb = 0.507, wetted surface ~2972 m^2)")


if __name__ == "__main__":
    main()