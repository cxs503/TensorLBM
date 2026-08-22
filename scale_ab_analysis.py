#!/usr/bin/env python
"""B1-5 A/B analysis: C_D per (sail_scale, fin_scale, Re) point.

C_D = 2 F_x_tail / (rho u_in^2 A_proj): force from the drag-history tail
mean (last 25% of samples, DragSurveySpec interval 25), projected area
from the point's own solid mask (so the normalisation tracks the grown
appendages).  Raw tail forces are also printed — with a FIXED reference
area (the scale-1 mask) — to separate "bigger object" from "different
flow".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import h5py  # noqa: E402

from tensorlbm.ai.drag_surrogate import load_exact_cd_per_point  # noqa: E402
from tensorlbm.drag_survey import projected_area  # noqa: E402

OUT = Path("/nfs/wangxi/datasets/b15_ab_20260823")


def main() -> None:
    cd = load_exact_cd_per_point(OUT, OUT)
    rows = []
    for pid, cd_val in sorted(cd.items()):
        status = json.loads((OUT / "points" / pid / "status.json").read_text())
        p = status["params"]
        hist = json.loads((OUT / "points" / pid / "drag_history.json").read_text())["samples"]
        fx = np.asarray([s["force_x"] for s in hist], dtype=np.float64)
        tail = float(fx[int(len(fx) * 0.75) :].mean())
        with h5py.File(OUT / "points" / pid / "fields.h5", "r") as f:
            last = sorted(f.keys())[-1]
            u_in = float(f[last].attrs["u_in"])
            a_proj = projected_area(np.asarray(f[last]["solid_mask"]))
        rows.append(
            {
                "pid": pid,
                "sail": float(p["sail_scale"]),
                "fin": float(p["fin_scale"]),
                "re": float(p["re"]),
                "fx_tail": tail,
                "u_in": u_in,
                "a_proj": a_proj,
                "cd": cd_val,
            }
        )

    a_ref = None
    for r in rows:
        if (r["sail"], r["fin"]) == (1.0, 1.0):
            a_ref = r["a_proj"]
            break
    # fixed-reference-area drag coefficient for the "bigger object" check
    print(
        f"{'pid':<6} {'(sail,fin)':<10} {'Re':>5} {'F_x_tail':>10} {'A_proj':>7} {'C_D':>8} "
        f"{'C_D@A1':>8}"
    )
    for r in sorted(rows, key=lambda r: (r["re"], r["sail"])):
        cd_fixed = 2.0 * r["fx_tail"] / (r["u_in"] ** 2 * a_ref)
        print(
            f"{r['pid']:<6} ({r['sail']:.0f},{r['fin']:.0f})     {r['re']:>5.0f} "
            f"{r['fx_tail']:>10.4f} {r['a_proj']:>7d} {r['cd']:>8.4f} {cd_fixed:>8.4f}"
        )
    print()
    for re_val in sorted({r["re"] for r in rows}):
        sel = sorted([r for r in rows if r["re"] == re_val], key=lambda r: r["sail"])
        cds = [r["cd"] for r in sel]
        if len(cds) >= 2:
            print(
                f"Re={re_val:.0f}: C_D(s) = {['%.4f' % c for c in cds]}  "
                f"spread (max-min)/min = {100 * (max(cds) - min(cds)) / min(cds):.1f}%"
            )


if __name__ == "__main__":
    main()
