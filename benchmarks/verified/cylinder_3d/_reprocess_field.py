#!/usr/bin/env python3
"""Turn a saved converged cylinder_3d field into a three-caliber case JSON.

The Ladd MEM sum is recomputed on the SAME converged field with three masks
(surface / all / interior) using exactly the formula in ``run.py``
(``obstacles.compute_obstacle_forces_3d`` = ``2·Σ c_ix f_i`` on the
post-stream field).  This lets the two-grid acceptance numbers be recovered
from runs whose live code predated the surface caliber.

    python _reprocess_field.py <field.pt> <D_cells> <base.json> <out.json> \
        [--lateral L] [--u-in 0.08] [--nz N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
for _p in (_REPO / "src", _REPO / "benchmarks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

from tensorlbm.d3q19 import C  # noqa: E402

CD_REF = 1.50


def mem_x(f, mask, dpS):
    c = C.to(f.device).float()
    cx = c[:, 0].view(19, 1, 1, 1)
    return float((2.0 * (cx * (f * mask.unsqueeze(0))).sum()).item()) / dpS


def surface_mask(solid):
    fluid = ~solid
    surf = torch.zeros_like(solid)
    for dim in (0, 1, 2):
        surf |= solid & torch.roll(fluid, 1, dim)
        surf |= solid & torch.roll(fluid, -1, dim)
    return surf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("field", type=Path)
    ap.add_argument("D", type=int)
    ap.add_argument("base", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--nz", type=int, default=1)
    ap.add_argument("--u-in", type=float, default=0.08)
    a = ap.parse_args()

    f = torch.load(a.field, map_location="cpu").float()
    _, nz, ny, nx = f.shape
    D = a.D
    R = D / 2.0
    cx, cy = nx // 2, ny // 2
    dpS = 0.5 * a.u_in**2 * (D * nz)

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    solid = (xx - cx) ** 2 + (yy - cy) ** 2 <= R**2
    surf = surface_mask(solid)
    interior = solid & ~surf

    cd_all = mem_x(f, solid, dpS)
    cd_surf = mem_x(f, surf, dpS)
    cd_int = mem_x(f, interior, dpS)

    r = json.loads(a.base.read_text())
    r["n_solid_cells"] = int(solid.sum())
    r["n_surface_cells"] = int(surf.sum())
    r["n_interior_cells"] = int(interior.sum())
    r["cd_mem_surface"] = cd_surf
    r["err_surf_pct"] = (cd_surf - CD_REF) / CD_REF * 100.0
    r["cd_mem_all"] = cd_all
    r["err_mem_all_pct"] = (cd_all - CD_REF) / CD_REF * 100.0
    r["cd_mem_interior"] = cd_int
    r["cd_mem"] = cd_all
    r["err_mem_pct"] = r["err_mem_all_pct"]
    r["cd"] = cd_surf
    r["err_pct"] = r["err_surf_pct"]
    r["caliber_note"] = (
        "three-caliber decomposition of the converged final field "
        f"({a.field.name}); formula identical to run.py surface_mask + "
        "obstacles.compute_obstacle_forces_3d"
    )
    a.out.write_text(json.dumps(r, indent=2))
    print(
        f"D={D} nz={nz} nx={nx} solid={int(solid.sum())} surface={int(surf.sum())} "
        f"interior={int(interior.sum())}"
    )
    print(
        f"  Cd_mem_surface={cd_surf:.4f} ({(cd_surf - CD_REF) / CD_REF * 100:+.2f}%)  "
        f"Cd_mem_all={cd_all:.4f} ({(cd_all - CD_REF) / CD_REF * 100:+.2f}%)  "
        f"Cd_mem_interior={cd_int:.4f}"
    )
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
