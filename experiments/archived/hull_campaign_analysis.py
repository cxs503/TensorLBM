#!/usr/bin/env python
"""B1-4 analysis: does the field->C_D surrogate generalise across hull geometry?

Two evaluations:
1. campaign split (mixed hulls): FNO(normalized) vs global power law
   (2-param) vs per-hull power law (2 params x 3 hulls = 6, the strong
   parametric baseline);
2. leave-one-hull-out: train on two hulls, test on the held-out one —
   where per-hull laws have no data at all. This is the surrogate's
   value case: no closed-form prior spans the geometry axis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/nfs/wangxi/worktrees/ml_fno")
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.ai.drag_surrogate import (  # noqa: E402
    DragTrainConfig,
    FNODragArch,
    PlaneSampleSpec,
    build_drag_split,
    load_exact_cd_per_point,
    predict_cd,
    regression_metrics,
    train_drag_surrogate,
)

HULLS = ("bare_hull", "with_sail", "full")
NEW = Path("/nfs/wangxi/datasets/scan_suboff_hull_re_20260823")
SPEC = PlaneSampleSpec(steps=(500,), velocity_scale=True)
CFG_KW = dict(
    epochs=600, batch_size=32, lr=1e-3, weight_decay=1e-4, patience=80, seed=0, device="cuda"
)


def hull_of(fields_dir: Path, pid: str) -> str:
    import h5py

    with h5py.File(fields_dir / "points" / pid / "fields.h5") as f:
        key = sorted(f.keys())[0]
        return str(f[key].attrs["hull_type"])


def power_fit(split):
    coef = np.polyfit(np.log10(split.re), np.log10(split.cd), 1)
    return coef  # [slope, intercept]


def power_pred(coef, split):
    b, a = coef
    return 10.0 ** (a + b * np.log10(split.re))


def report(name, m):
    print(f"    {name:<26} MAPE {m['mape']:6.2f}%  MAE {m['mae']:.3f}  R2 {m['r2']:.4f}")


def main() -> None:
    split_points = json.loads((NEW / "dataset.json").read_text())["split_points"]
    print("splits:", {k: len(v) for k, v in split_points.items()})
    cd_by_point = load_exact_cd_per_point(NEW, NEW)
    hull_by_point = {pid: hull_of(NEW, pid) for pid in sorted(cd_by_point)}
    by_hull = {h: [p for p, hh in hull_by_point.items() if hh == h] for h in HULLS}
    print("points per hull:", {h: len(v) for h, v in by_hull.items()})

    sp = {
        name: build_drag_split(NEW, point_ids=pids, spec=SPEC, cd_by_point=cd_by_point)
        for name, pids in split_points.items()
    }

    # -- 1. campaign split --------------------------------------------------
    print("\n[1] campaign split (mixed hulls)")
    result = train_drag_surrogate(sp["train"], sp["val"], FNODragArch(), DragTrainConfig(**CFG_KW))
    print(f"    best_epoch {result.best_epoch}")
    report(
        "fno[norm]",
        regression_metrics(
            sp["test"].cd, predict_cd(result.model, sp["test"], result.norm, device="cuda")
        ),
    )

    gcoef = power_fit(sp["train"])
    report("power2-global", regression_metrics(sp["test"].cd, power_pred(gcoef, sp["test"])))

    # per-hull power law (6 params): fit on train points of each hull
    train_pids = split_points["train"]
    per_hull = {}
    for h in HULLS:
        hp = [p for p in train_pids if hull_by_point[p] == h]
        if len(hp) >= 2:
            hs = build_drag_split(NEW, point_ids=hp, spec=SPEC, cd_by_point=cd_by_point)
            per_hull[h] = power_fit(hs)
    pred = []
    for i, pid in enumerate(sp["test"].point_id):
        pred.append(power_pred(per_hull[hull_by_point[pid]], sp["test"])[i])
    report("power-per-hull", regression_metrics(sp["test"].cd, np.asarray(pred)))

    # mean C_D per hull as difficulty reference
    hull_means = {h: float(np.mean([cd_by_point[p] for p in by_hull[h]])) for h in HULLS}
    print("    C_D mean per hull:", {h: f"{v:.2f}" for h, v in hull_means.items()})

    # -- 2. leave-one-hull-out ----------------------------------------------
    print("\n[2] leave-one-hull-out (train on two hulls, test the third)")
    for held in HULLS:
        train_pids = [p for h in HULLS if h != held for p in by_hull[h]]
        rng = np.random.default_rng(7)
        rng.shuffle(train_pids := np.asarray(train_pids))
        tp = list(train_pids[:-3])
        vp = list(train_pids[-3:])
        train = build_drag_split(NEW, point_ids=tp, spec=SPEC, cd_by_point=cd_by_point)
        val = build_drag_split(NEW, point_ids=vp, spec=SPEC, cd_by_point=cd_by_point)
        test = build_drag_split(NEW, point_ids=by_hull[held], spec=SPEC, cd_by_point=cd_by_point)
        r = train_drag_surrogate(train, val, FNODragArch(), DragTrainConfig(**CFG_KW))
        mf = regression_metrics(test.cd, predict_cd(r.model, test, r.norm, device="cuda"))
        # global power law fitted on the two training hulls
        g = power_fit(train)
        mp = regression_metrics(test.cd, power_pred(g, test))
        # naive transfer: per-hull law of nearest training hull (bare->sail->full chain)
        donor = {"bare_hull": "with_sail", "with_sail": "bare_hull", "full": "with_sail"}[held]
        dh = build_drag_split(
            NEW, point_ids=[p for p in by_hull[donor]], spec=SPEC, cd_by_point=cd_by_point
        )
        md = regression_metrics(test.cd, power_pred(power_fit(dh), test))
        print(f"  held-out {held:<10} (n={len(test)}):")
        report("fno", mf)
        report("power2(2-hull fit)", mp)
        report(f"donor law ({donor})", md)


if __name__ == "__main__":
    main()
