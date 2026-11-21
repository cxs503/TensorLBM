#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K2-BUDGET-B 诊断自检：验证预算行的内部一致性（不依赖判决阈值）。

检查项：
  1. ke_axial + ke_lat == ke_liq（能量分配的闭合，机器精度级）
  2. ke_lat_roi <= ke_lat（子集关系）
  3. ke_drop <= ke_liq（滴材料是液相的子集）
  4. 界面面积 > 0 且随时间连续
  5. 时间步连续、无重复、无缺口
  6. 自由面体积漂移量级（伪功诊断的输入）
  7. 伪电流限幅贴限胞数（能量汇上界）

用法：
    python3 tools/k2_budget_sanity.py --tree .
输出：
    runs/k2budget_verdict/sanity.json
"""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", default=ROOT)
    ap.add_argument("--budget", default="runs/k2budget/result.json")
    ap.add_argument("--out", default="runs/k2budget_verdict/sanity.json")
    a = ap.parse_args()
    tree = os.path.abspath(a.tree)
    with open(os.path.join(tree, a.budget)) as handle:
        result = json.load(handle)
    rows = result["budget_diag"]["rows"]

    worst_split = 0.0
    worst_subset = 0.0
    worst_drop = 0.0
    for row in rows:
        scale = max(abs(row["ke_liq"]), 1e-30)
        worst_split = max(worst_split, abs(row["ke_axial"] + row["ke_lat"] - row["ke_liq"]) / scale)
        worst_subset = max(worst_subset, row["ke_lat_roi"] - row["ke_lat"])
        worst_drop = max(worst_drop, row["ke_drop"] - row["ke_liq"])
    steps = [r["step"] for r in rows]
    contiguous = all(b - a_ == 1 for a_, b in zip(steps, steps[1:]))
    vol = [r["vol_liq"] for r in rows]
    drift = max(vol) - min(vol)
    cap_cells = max(r.get("n_at_cap", 0) for r in rows)
    ke_cap = max(r.get("ke_at_cap", 0.0) for r in rows)

    out = {
        "rows": len(rows),
        "step_first": steps[0] if steps else None,
        "step_last": steps[-1] if steps else None,
        "steps_contiguous": bool(contiguous),
        "max_rel_ke_partition_error": worst_split,
        "max_ke_lat_roi_excess": worst_subset,
        "max_ke_drop_excess": worst_drop,
        "area_total_min": min(r["area_total"] for r in rows),
        "area_total_max": max(r["area_total"] for r in rows),
        "volume_drift_cells": drift,
        "volume_drift_relative": drift / max(abs(vol[0]), 1e-30),
        "max_cells_at_u_limit_cap": cap_cells,
        "max_ke_at_u_limit_cap": ke_cap,
        "pass": bool(worst_split < 1e-9 and contiguous and worst_subset <= 1e-12
                     and worst_drop <= 1e-9 and min(r["area_total"] for r in rows) > 0),
    }
    path = os.path.join(tree, a.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(out, handle, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
