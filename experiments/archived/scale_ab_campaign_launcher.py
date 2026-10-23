#!/usr/bin/env python
"""B1-5: appendage-scale A/B — does the geometry axis separate C_D?

B1-4 (docs/hull_geometry_campaign_20260823.md) found the three-
configuration axis degenerate: the sail contributes only 0.7% of hull
solid cells at n128, so voxelisation cannot tell the hulls apart.
``SuboffConfig.sail_scale`` / ``fin_scale`` (this branch) multiply each
appendage's own dimensions about its DARPA anchors and open the axis:
at n128 the full configuration's appendage share grows 1.6% -> 11.1% ->
40.4% for scales 1/2/3.

This A/B runs the FULL configuration at scale combos (1,1)/(2,2)/(3,3)
x Re {200, 600} — 6 points, 2500 steps, cumulant, u_in = 0.1 — through
the same production chain as B1-4 (ScanExecutor + DragSurveySpec
margin 4 / interval 25), so the drag tails are directly comparable.
C_D is computed post-hoc as 2 F_x_tail / (rho u_in^2 A_proj) from the
drag_history tail mean (scale_ab_analysis.py).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.scan_drag import DragSurveySpec  # noqa: E402
from tensorlbm.scan_runner import (  # noqa: E402
    ScanExecutor,
    ScanPlan,
    ScanPoint,
    ScanVariable,
    git_code_sha,
)

SCALES = ((1.0, 1.0), (2.0, 2.0), (3.0, 3.0))
RES = (200.0, 600.0)
STEPS, SNAP = 2500, 500
FIXED = {
    "resolution": 128,
    "collision": "cumulant",
    "u_in": 0.10,
    "hull_type": "full",
}
OUT = Path("/nfs/wangxi/datasets/b15_ab_20260823")


def build_plan(smoke: bool) -> ScanPlan:
    scan_id = "scan-suboff-scale-ab"
    if smoke:
        rows = [(1.0, 1.0, 200.0)]
        steps, snap = 120, 60
    else:
        rows = [(s, f, re) for s, f in SCALES for re in RES]
        steps, snap = STEPS, SNAP

    points = tuple(
        ScanPoint(
            index=i,
            point_id=f"p{i:04d}",
            run_id=f"{scan_id}-p{i:04d}",
            params={"sail_scale": s, "fin_scale": f, "re": re},
        )
        for i, (s, f, re) in enumerate(rows)
    )
    return ScanPlan(
        scan_id=scan_id,
        case="suboff_n128",
        variables=(
            ScanVariable(name="re", low=min(RES), high=max(RES)),
            ScanVariable(name="sail_scale", low=1.0, high=3.0),
            ScanVariable(name="fin_scale", low=1.0, high=3.0),
        ),
        method="grid",
        n_points=len(points),
        seed=20260823,
        steps=steps,
        snapshot_every=snap,
        code_sha=git_code_sha(REPO),
        fixed_params=FIXED,
        drag_survey=DragSurveySpec(margin=4, interval=25),
        points=points,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def main() -> None:
    smoke = "--smoke" in sys.argv
    out = Path("/nfs/wangxi/datasets/b15_ab_smoke") if smoke else OUT
    out.mkdir(parents=True, exist_ok=True)
    plan = build_plan(smoke)
    print(f"{len(plan.points)} points, steps={plan.steps}, out={out}")
    ex = ScanExecutor(
        plan,
        out,
        gpus=(1,) if smoke else (1, 2, 3, 4, 5, 6),
        serial_device="cuda:1",
        checkpoint_every=250,
    )
    try:
        ex.run(resume=True)
    finally:
        print("done:", out)


if __name__ == "__main__":
    main()
