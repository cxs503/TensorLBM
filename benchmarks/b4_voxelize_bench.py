"""B4-P2a voxelisation benchmark: STL mesh -> solid mask wall-clock cost.

Meshes (generated in-script, no datasets): a subdiv-4 icosphere
(5120 triangles) and a SUBOFF-like prolate stretch of it (L/D = 6),
each voxelised on the B4 canonical grid via
``tensorlbm.voxelize.place_on_grid`` + ``mask_from_stl``.

One JSON line is printed per (mesh, grid) config; numbers are recorded
in ``docs/voxelize_stl_20260824.md``.

Usage::

    PYTHONPATH=src python benchmarks/b4_voxelize_bench.py
"""

from __future__ import annotations

import json
import time

import numpy as np

from tensorlbm.voxelize import is_watertight, mask_from_stl, place_on_grid

GRIDS: tuple[tuple[int, int, int], ...] = ((32, 32, 64), (64, 64, 128))


def icosphere(subdiv: int) -> np.ndarray:
    """Unit icosphere: 20-face icosahedron + midpoint subdivision."""
    t = (1.0 + 5.0**0.5) / 2.0
    verts = [
        (-1, t, 0),
        (1, t, 0),
        (-1, -t, 0),
        (1, -t, 0),
        (0, -1, t),
        (0, 1, t),
        (0, -1, -t),
        (0, 1, -t),
        (t, 0, -1),
        (t, 0, 1),
        (-t, 0, -1),
        (-t, 0, 1),
    ]
    verts = [v / np.linalg.norm(v) for v in verts]
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    vlist: list[list[float]] = [list(map(float, v)) for v in verts]
    for _ in range(subdiv):
        cache: dict[tuple[int, int], int] = {}
        new_faces: list[tuple[int, int, int]] = []

        def midpoint(i: int, j: int) -> int:
            key = (i, j) if i < j else (j, i)
            if key not in cache:
                m = (np.asarray(vlist[i]) + np.asarray(vlist[j])) / 2.0
                vlist.append(list(m / np.linalg.norm(m)))
                cache[key] = len(vlist) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])
        faces = new_faces
    pts = np.asarray(vlist, dtype=np.float64)
    return pts[np.asarray(faces, dtype=np.int64)]


def main() -> None:
    meshes = {
        "icosphere_subdiv4": icosphere(4),
        "prolate_suboff_like_L6": icosphere(4) * np.array([3.0, 1.0, 1.0]),
    }
    for name, tris in meshes.items():
        t0 = time.perf_counter()
        watertight = is_watertight(tris)
        t_wt = time.perf_counter() - t0
        for shape in GRIDS:
            t0 = time.perf_counter()
            placement = place_on_grid(tris, shape)
            t_place = time.perf_counter() - t0
            t0 = time.perf_counter()
            mask = mask_from_stl(placement.tris, shape, origin=placement.origin)
            t_mask = time.perf_counter() - t0
            print(
                json.dumps(
                    {
                        "mesh": name,
                        "tris": int(tris.shape[0]),
                        "shape": list(shape),
                        "watertight": bool(watertight),
                        "watertight_s": round(t_wt, 4),
                        "place_s": round(t_place, 4),
                        "mask_s": round(t_mask, 4),
                        "solid_cells": int(mask.sum()),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
