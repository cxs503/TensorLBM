#!/usr/bin/env python
"""B1-4: hull-geometry x Re campaign — the no-closed-form-prior experiment.

B1-3 established the surrogate beats power-law priors on (Re, u_in) at
N=40 — but a power law in Re still *exists* there. Hull geometry is the
axis where it does not: three real DARPA SUBOFF configurations
(bare_hull / with_sail / full) have different appendage drag, and no
2-parameter law spans them. If the field->C_D surrogate generalises
across hulls, this is its value case.

42 points = 3 hull types x 14 log-Re LHS each (re in [50, 800],
u_in = 0.1 fixed, resolution 128, cumulant). tau = 0.5 + 23.04/re in
[0.529, 0.960] — inside the B1 envelope by construction. hull_type flows
as a categorical scan param (string); the scan_runner metadata/plan
serialisation casts numerics only (this branch's small fix).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/nfs/wangxi/worktrees/hull_scan")
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.scan_drag import DragSurveySpec  # noqa: E402
from tensorlbm.scan_runner import (  # noqa: E402
    ScanExecutor,
    ScanPlan,
    ScanPoint,
    ScanVariable,
    git_code_sha,
)

HULLS = ("bare_hull", "with_sail", "full")
N_PER_HULL = 14
SEED = 20260823
RE_LOW, RE_HIGH = 50.0, 800.0
FIXED = {"resolution": 128, "collision": "cumulant", "u_in": 0.10}


def tau_of(re: float) -> float:
    return 0.5 + 23.04 / re


def lhs_re(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = (rng.permutation(n) + rng.random(n)) / n
    return 10.0 ** (np.log10(RE_LOW) + s * (np.log10(RE_HIGH) - np.log10(RE_LOW)))


def build_plan(smoke: bool) -> ScanPlan:
    scan_id = "scan-suboff-hull-re"
    if smoke:
        rows = [
            ("full", 800.0),  # worst tau + most appendages
            ("full", 50.0),
            ("bare_hull", 800.0),
            ("bare_hull", 50.0),
        ]
        steps, snap = 300, 150
    else:
        rows = []
        for h_i, hull in enumerate(HULLS):
            for re in lhs_re(N_PER_HULL, SEED + h_i):
                rows.append((hull, float(re)))
        steps, snap = 4000, 500

    points = tuple(
        ScanPoint(
            index=i,
            point_id=f"p{i:04d}",
            run_id=f"{scan_id}-p{i:04d}",
            params={"hull_type": hull, "re": re},
        )
        for i, (hull, re) in enumerate(rows)
    )
    plan = ScanPlan(
        scan_id=scan_id,
        case="suboff_n128",
        variables=(ScanVariable(name="re", low=RE_LOW, high=RE_HIGH),),
        method="lhs",
        n_points=len(points),
        seed=SEED,
        steps=steps,
        snapshot_every=snap,
        code_sha=git_code_sha(REPO),
        fixed_params=FIXED,
        drag_survey=DragSurveySpec(margin=4, interval=25),
        points=points,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    taus = [tau_of(p.params["re"]) for p in plan.points]
    counts = {h: sum(1 for p in plan.points if p.params["hull_type"] == h) for h in HULLS}
    print(f"hulls: {counts}  tau range [{min(taus):.4f}, {max(taus):.4f}]")
    expected_counts = (
        {"full": 2, "bare_hull": 2, "with_sail": 0} if smoke else {h: N_PER_HULL for h in HULLS}
    )
    assert counts == expected_counts, counts
    assert min(taus) >= tau_of(RE_HIGH) - 1e-9
    assert len({(p.params["hull_type"], p.params["re"]) for p in plan.points}) == len(points)
    return plan


def main() -> None:
    smoke = "--smoke" in sys.argv
    out = Path(
        "/nfs/wangxi/datasets/scan_suboff_hull_re_smoke"
        if smoke
        else "/nfs/wangxi/datasets/scan_suboff_hull_re_20260823"
    )
    out.mkdir(parents=True, exist_ok=True)
    plan = build_plan(smoke)
    ex = ScanExecutor(plan, out, gpus=tuple(range(8)), serial_device="cuda:0", checkpoint_every=250)
    try:
        ex.run(resume=True)
    finally:
        ex.close()
    print("done ->", out, "points:", plan.n_points)


if __name__ == "__main__":
    main()
