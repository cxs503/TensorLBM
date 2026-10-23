#!/usr/bin/env python
"""B1-3 statistical tightening: 5-fold x 3-seed on the LHS40 campaign.

v1.2's caveat: the FNO-vs-power3 win margin (1.15 vs 1.56 % test MAPE)
was one 28/6/6 split, one seed. This script puts error bars on it:
sorted-stratified 5-fold (sort by C_D, deal round-robin), test = fold,
val = next fold, train = rest; FNO with seeds {0,1,2} vs power2/power3
fit per fold. Reports mean +- std of test MAPE across folds (FNO also
across seeds).
"""

from __future__ import annotations

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

NEW = Path("/nfs/wangxi/datasets/scan_suboff_re_uin_lhs40_20260822")
K = 5
SEEDS = (0, 1, 2)


def main() -> None:
    cd_by_point = load_exact_cd_per_point(NEW, NEW)
    pids = sorted(cd_by_point)
    # stratify by C_D: sort, deal round-robin into K folds
    pids_by_cd = sorted(pids, key=lambda p: cd_by_point[p])
    folds = [[pids_by_cd[i] for i in range(len(pids_by_cd)) if i % K == f] for f in range(K)]
    print(f"{len(pids)} points -> {K} folds of {[len(f) for f in folds]}")

    spec = PlaneSampleSpec(steps=(500,), velocity_scale=True)
    splits = {
        f"f{i}": build_drag_split(NEW, point_ids=fold, spec=spec, cd_by_point=cd_by_point)
        for i, fold in enumerate(folds)
    }

    def feats(split, n_params):
        cols = [np.ones(len(split)), np.log10(split.re)]
        if n_params >= 3:
            cols.append(np.log10(split.u_in))
        return np.stack(cols, axis=1)

    fno_mapes, p2_mapes, p3_mapes = [], [], []
    for i in range(K):
        test_i, val_i = i, (i + 1) % K
        train_pids = [p for j in range(K) if j not in (test_i, val_i) for p in folds[j]]
        train = build_drag_split(NEW, point_ids=train_pids, spec=spec, cd_by_point=cd_by_point)
        val, test = splits[f"f{val_i}"], splits[f"f{test_i}"]

        for n_params, sink in ((2, p2_mapes), (3, p3_mapes)):
            coef, *_ = np.linalg.lstsq(feats(train, n_params), np.log10(train.cd), rcond=None)
            m = regression_metrics(test.cd, 10.0 ** (feats(test, n_params) @ coef))
            sink.append(m["mape"])

        fold_mapes = []
        for seed in SEEDS:
            cfg = DragTrainConfig(
                epochs=600,
                batch_size=32,
                lr=1e-3,
                weight_decay=1e-4,
                patience=80,
                seed=seed,
                device="cuda",
            )
            result = train_drag_surrogate(train, val, FNODragArch(), cfg)
            m = regression_metrics(
                test.cd, predict_cd(result.model, test, result.norm, device="cuda")
            )
            fold_mapes.append(m["mape"])
        fno_mapes.append(float(np.mean(fold_mapes)))
        print(
            f"fold {i}: fno {np.mean(fold_mapes):.2f}±{np.std(fold_mapes):.2f} "
            f"(seeds {['%.2f' % m for m in fold_mapes]})  "
            f"p2 {p2_mapes[-1]:.2f}  p3 {p3_mapes[-1]:.2f}"
        )

    def report(name, xs):
        a = np.asarray(xs)
        print(f"{name:<12} test MAPE {a.mean():.2f}% ± {a.std(ddof=1):.2f}%  (n={len(a)})")

    print()
    report("fno(norm)", fno_mapes)
    report("power2", p2_mapes)
    report("power3", p3_mapes)
    wins = int(np.sum(np.asarray(fno_mapes) < np.asarray(p3_mapes)))
    print(f"\nfno beats power3 on {wins}/{K} folds")


if __name__ == "__main__":
    main()
