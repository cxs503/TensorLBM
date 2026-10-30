#!/usr/bin/env python3
"""Close the DTMB 5415 waterline so the underwater STL is a *watertight* body.

Why
---
``geometry/dtmb5415_underwater.stl`` is produced by ``_build_geometry.py`` by
clipping the official zenodo/CNR-INSEAN surface at the design waterline
``z = T = 6.15 m``.  The clip keeps a *triangle-centroid* test, which leaves the
body **open** at the waterline plane.  Inside/outside voxelisation then has no
closed surface to work with, and ``tensorlbm.stl_geometry.voxelize_stl``
resolves the ambiguity by filling a solid pillar *below* the waterline: at the
benchmark resolution (L = 48 cells over Lpp) the engine's own solid mask has

    Cb_voxel = 18.0   (reference 0.507, i.e. 35x the true displacement)
    A_stair / A_smooth = 16.0

both of which are geometry, not physics.  Normalising the integrated wall
forces by the *smooth* wetted area then inflates Cd by ~15x, which is exactly
the observed ratio ``Cd_friction / Cf_Blasius = 0.526 / 0.042 = 12.5``.

What this script does
---------------------
1. reads the underwater STL,
2. welds vertices (bit-identical duplicates) and finds the open boundary edges
   (each referenced by exactly one triangle),
3. assembles them into closed loops (the waterline curve),
4. caps each loop with a fan of triangles at the waterline plane,
5. writes ``geometry/dtmb5415_underwater_closed.stl``,
6. verifies closure: every edge of the written mesh must be shared by exactly
   two triangles, and the voxelised body must recover Cb ~ 0.507 with a
   staircase-to-smooth wetted-area ratio close to 1.

Usage
-----
  python _close_waterline.py [--in ...] [--out ...] [--verify-L 24 32 48]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

LPP = 142.0
BEAM = 19.06
DRAFT = 6.15
CB_REF = 0.507
DISPLACEMENT_M3 = CB_REF * LPP * BEAM * DRAFT  # 8439 m^3


def _welded(verts: np.ndarray, faces: np.ndarray, tol: float = 1e-6):
    """Merge coincident vertices; return (verts, faces) with compact indexing."""
    key = np.round(np.asarray(verts, dtype=np.float64) / tol).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return np.asarray(verts)[first], inverse.reshape(-1)[faces].reshape(-1, 3)


def boundary_edges(faces: np.ndarray) -> np.ndarray:
    """Edges referenced by exactly one triangle, as an (n, 2) array."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    return uniq[counts == 1]


def ordered_loops(edges: np.ndarray) -> list[list[int]]:
    """Assemble boundary edges into closed vertex loops."""
    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    seen: set[int] = set()
    loops: list[list[int]] = []
    for start in adj:
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        prev, cur = None, start
        while True:
            nxt = [n for n in adj[cur] if n != prev and n not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            loop.append(cur)
            seen.add(cur)
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def orient_caps(verts_cap: np.ndarray, faces_cap: np.ndarray,
                body_verts: np.ndarray, cap_faces: np.ndarray) -> np.ndarray:
    """Orient cap triangles so their normal points +z (out of the body)."""
    allv = np.vstack([body_verts, verts_cap])
    a = allv[cap_faces[:, 0]]
    b = allv[cap_faces[:, 1]]
    c = allv[cap_faces[:, 2]]
    nz = np.cross(b - a, c - a)[:, 2]
    flip = nz < 0
    out = cap_faces.copy()
    out[flip] = out[flip][:, [0, 2, 1]]
    return out


def is_closed(faces: np.ndarray) -> bool:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool((counts == 2).all())


def wetted_area_m2(verts: np.ndarray, faces: np.ndarray) -> float:
    a = verts[faces[:, 1]] - verts[faces[:, 0]]
    b = verts[faces[:, 2]] - verts[faces[:, 0]]
    return float(0.5 * np.linalg.norm(np.cross(a, b), axis=1).sum())


def voxel_report(verts: np.ndarray, faces: np.ndarray, l_cells: int) -> dict:
    """Cb and staircase/smooth wetted-area ratio of the voxel body."""
    from tensorlbm.stl_geometry import voxelize_stl

    dx = LPP / l_cells
    x0 = float(verts[:, 0].min()) - 2 * dx
    x1 = float(verts[:, 0].max()) + 2 * dx
    yh = float(np.abs(verts[:, 1]).max()) + 2 * dx
    z0 = float(verts[:, 2].min()) - 2 * dx
    nx = int(np.ceil((x1 - x0) / dx))
    ny = int(np.ceil(2 * yh / dx))
    nz = int(np.ceil((DRAFT - z0) / dx))
    solid = voxelize_stl(verts, faces, (nx, ny, nz),
                         origin=(x0, -yh, z0), spacing=(dx, dx, dx))
    s = solid.cpu().numpy().astype(np.int8)
    n_solid = int(s.sum())
    vol = n_solid * dx**3
    walls = (np.abs(np.diff(s, axis=2)).sum()
             + np.abs(np.diff(s, axis=1)).sum()
             + np.abs(np.diff(s, axis=0)).sum())
    s_smooth_lb = wetted_area_m2(verts, faces) / dx**2
    return {
        "L_cells": l_cells,
        "dx_m": dx,
        "draft_over_dx": DRAFT / dx,
        "beam_over_dx": BEAM / dx,
        "grid": [nx, ny, nz],
        "solid_cells": n_solid,
        "volume_m3": vol,
        "cb_voxel": vol / (LPP * BEAM * DRAFT),
        "cb_reference": CB_REF,
        "cb_error_pct": (vol / DISPLACEMENT_M3 - 1.0) * 100.0,
        "stair_wall_faces": int(walls),
        "smooth_wetted_lb": s_smooth_lb,
        "stair_over_smooth": float(walls / s_smooth_lb) if s_smooth_lb else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src",
                    default=str(_HERE / "geometry" / "dtmb5415_underwater.stl"))
    ap.add_argument("--out", dest="dst",
                    default=str(_HERE / "geometry" / "dtmb5415_underwater_closed.stl"))
    ap.add_argument("--verify-L", type=int, nargs="*", default=[24, 32, 48])
    args = ap.parse_args()

    from tensorlbm.stl_geometry import read_stl, write_stl

    verts, faces, _ = read_stl(args.src)
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    print(f"open STL: {len(verts)} verts, {len(faces)} tris, closed={is_closed(faces)}")

    wv, wf = _welded(verts, faces)
    be = boundary_edges(wf)
    loops = ordered_loops(be)
    print(f"welded: {len(wv)} verts, {len(wf)} tris, boundary edges={len(be)}, loops={[len(l) for l in loops]}")
    if not loops:
        print("nothing to cap (already closed?)")
    # cap each loop with a fan about its centroid (one new vertex per loop, so
    # the appended-vertex indices stay consistent with the face indices)
    extra, cap_faces = [], []
    for lp in loops:
        pts = wv[lp]
        c = pts.mean(axis=0)
        c[2] = float(np.median(pts[:, 2]))
        ci = len(wv) + len(extra)
        extra.append(c)
        for k in range(len(lp)):
            cap_faces.append([lp[k], lp[(k + 1) % len(lp)], ci])
    cap_faces = np.asarray(cap_faces, dtype=np.int64).reshape(-1, 3)
    extra = np.asarray(extra, dtype=np.float64).reshape(-1, 3)
    cap_faces = orient_caps(extra, cap_faces, wv, cap_faces)
    all_v = np.vstack([wv, extra])
    all_f = np.vstack([wf, cap_faces])
    print(f"with cap: {len(all_v)} verts, {len(all_f)} tris, closed={is_closed(all_f)}")

    write_stl(args.dst, all_v, all_f, binary=True)
    print(f"wrote {args.dst} ({Path(args.dst).stat().st_size / 1e6:.2f} MB)")

    # Sidecar so that consumers (run.py) can subtract the cap exactly: the cap
    # triangles are the last ``len(cap_faces)`` faces of the written file.
    sidecar = Path(str(args.dst) + ".capinfo.json")
    sidecar.write_text(json.dumps({
        "source_stl": str(args.src),
        "closed": is_closed(all_f),
        "waterline_z_m": DRAFT,
        "n_body_triangles": int(len(wf)),
        "cap_triangle_count": int(len(cap_faces)),
        "cap_vertex_count": int(len(extra)),
        "cap_area_m2": float(wetted_area_m2(all_v, all_f[-len(cap_faces):]))
        if len(cap_faces) else 0.0,
        "wetted_area_m2": wetted_area_m2(wv, wf),
    }, indent=2))
    print(f"wrote {sidecar}")
    print(f"wetted area (smooth) = {wetted_area_m2(all_v, all_f):.1f} m^2 "
          f"(reference ~2972-3022 m^2)")

    rows = [voxel_report(all_v, all_f, L) for L in args.verify_L]
    for r in rows:
        print(f"  L={r['L_cells']:3d} T/dx={r['draft_over_dx']:5.2f} solid={r['solid_cells']:6d} "
              f"Cb={r['cb_voxel']:.3f} ({r['cb_error_pct']:+.1f}%) "
              f"A_stair/A_smooth={r['stair_over_smooth']:.2f}")
    out = Path(args.dst).with_suffix(".verify.json")
    out.write_text(json.dumps({"file": str(args.dst), "rows": rows,
                              "closed": is_closed(all_f),
                              "wetted_area_m2": wetted_area_m2(all_v, all_f)}, indent=2))
    print(f"verify -> {out}")


if __name__ == "__main__":
    main()