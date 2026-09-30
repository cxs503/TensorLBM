#!/usr/bin/env python3
"""Assemble ``result.json`` for the cylinder_3d Re=40 MEM benchmark.

Reads the per-grid ``case_D*.json`` emitted by ``run.py single`` and writes a
single ``result.json`` with the acceptance verdict, in exactly the schema
``run.py verify`` produces (so the two paths are interchangeable).

Acceptance caliber: **surface-only** Ladd MEM (``cd_mem_surface``).
All-solid (``cd_mem_all``) and interior (``cd_mem_interior``) are carried as
diagnostics.  See ``docs/mem_surface_caliber_finding.md``.

    python _assemble_result.py <out_dir> <case_D20.json> <case_D40.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CD_REF = 1.50
CD_CLUSTER = (1.48, 1.52)


def main() -> int:
    out_dir = Path(sys.argv[1])
    cases = [Path(p) for p in sys.argv[2:]]
    per_grid = [json.loads(p.read_text()) for p in cases]
    per_grid.sort(key=lambda r: r["D_cells"])

    cd_surf = [r["cd_mem_surface"] for r in per_grid]
    cd_all = [r["cd_mem_all"] for r in per_grid]
    cd_int = [r["cd_mem_interior"] for r in per_grid]
    cd = cd_surf
    errs = [r.get("err_surf_pct", (r["cd_mem_surface"] - CD_REF) / CD_REF * 100.0)
            for r in per_grid]
    span = abs(cd[-1] - cd[0]) / CD_REF * 100.0 if len(cd) > 1 else float("nan")
    span_all = abs(cd_all[-1] - cd_all[0]) / CD_REF * 100.0 if len(cd_all) > 1 else float("nan")
    ok = all(abs(e) <= 3.0 for e in errs)
    conv = span <= 3.0
    res = {
        "case": "cylinder_3d_re40",
        "description": "3D extruded circular cylinder, infinite span (z periodic), Re=40 steady",
        "lattice": "D3Q19", "collision": "bgk",
        "force_method": "ladd_momentum_exchange_3d (post-stream, pre-bounce-back); "
                        "primary caliber = SURFACE cells only (wall-adjacent)",
        "reference": per_grid[0]["ref_cd"],
        "ref_cd": per_grid[0]["ref_cd"], "ref_cluster": per_grid[0]["ref_cluster"],
        "ref_sources": per_grid[0]["ref_sources"],
        "grids": per_grid,
        "cd_mem_surface_by_grid": cd_surf,
        "cd_mem_all_by_grid": cd_all,
        "cd_mem_interior_by_grid": cd_int,
        "cd_mem_by_grid": cd_surf,
        "cd_pressure_friction_by_grid": [r["cd_pf_standard"] for r in per_grid],
        "convergence": {
            "caliber": "surface_only",
            "cd": cd, "cd_span_pct": span,
            "cd_within_3pct": ok, "grid_span_within_3pct": conv,
            "cd_all_span_pct": span_all,
        },
        "verified": bool(ok and conv),
        "verdict": "verified" if (ok and conv) else "not_verified",
        "compile_mode_effective": per_grid[0].get("compile_mode_effective") if per_grid else None,
    }
    (out_dir / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())