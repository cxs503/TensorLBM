#!/usr/bin/env python3
"""P187-W1-N08 分析流水线：窗均值 -> 双参数偏差定律拟合 -> C1..C6 判据 -> 图 -> 报告数据。

输入：``runs/p187_w1_n08/hd<HD>_D<D>.json``（每个 = 一个 (H/D, 网格级) 算例的 Nu(t)）。
输出：
* ``artifacts/p187_w1_n08/criteria.json``：判据数字 + 溯源（每个数字带来源算例/窗）
* ``artifacts/p187_w1_n08/window_table.json``：完整窗均值表（零手抄）
* ``artifacts/p187_w1_n08/figures/*.png``：图
* ``runs/p187_w1_n08/summary.json``：图表数据与诊断（Re/S t/一致性）

用法：
    python scripts/analyze_p187_w1_n08.py --metric balance
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.confined_cylinder_law import (  # noqa: E402
    MISSION_WINDOWS,
    STATIONARY_WINDOWS,
    churchill_bernstein,
    evaluate_criteria,
    fit_multiplicative_law,
    interaction_residuals,
    window_means,
)

RUNS = ROOT / "runs" / "p187_w1_n08"
ART = ROOT / "artifacts" / "p187_w1_n08"
FIG = ART / "figures"

METRIC_KEY = {"balance": "nu_global", "surface": "nu_local"}


def load_runs(min_t: float = 1e9, allow_partial: bool = False) -> dict[int, dict[float, dict]]:
    """按网格级（D_eff）分组：{D_eff: {H/D: run}}"""
    out: dict[int, dict[float, dict]] = {}
    for p in sorted(RUNS.glob("hd*_D*.json")):
        with open(p) as fh:
            r = json.load(fh)
        if r.get("status") != "done":
            if not allow_partial or float(r.get("t_reached", 0.0)) < min_t:
                continue
        d = int(round(r["geometry"]["D_eff"]))
        out.setdefault(d, {})[round(float(r["geometry"]["h_over_d"]), 4)] = r
    return out


def build_rows(runs: dict[float, dict], windows, key: str) -> list[dict]:
    rows = []
    for hd, r in sorted(runs.items()):
        wm = window_means(r["times"], r[key], windows)
        for w, nu in wm.items():
            if not np.isfinite(nu):
                continue
            rows.append({
                "h_over_d": float(hd), "tau": float(w[1] - w[0]),
                "window": [float(w[0]), float(w[1])], "nu": float(nu),
                "source": r["label"], "grid_D_eff": float(r["geometry"]["D_eff"]),
                "ny": r["geometry"]["ny"], "nx": r["geometry"]["nx"],
            })
    return rows


def strouhal(run: dict, t0: float = 40.0, t1: float = 105.0) -> float:
    """由 Nu(t)（或焓平衡序列）的谱峰求 St = f D/U。"""
    t = np.asarray(run["times"], float)
    s = np.asarray(run[METRIC_KEY["balance"]], float)
    m = (t >= t0) & (t < t1)
    if m.sum() < 32:
        return float("nan")
    ts, ss = t[m], s[m] - s[m].mean()
    dt = float(np.median(np.diff(ts)))
    f = np.fft.rfftfreq(len(ss), d=dt)
    amp = np.abs(np.fft.rfft(ss))
    amp[0] = 0.0
    return float(f[np.argmax(amp)])


def recirculation_length(run: dict) -> float:
    """由 JSON 里没有的场量无法算；返回 nan（场量图单独脚本产出）。"""
    return float("nan")


def richardson(coarse: float, fine: float, ratio: float, order: float = 1.0) -> float:
    """两级网格 Richardson 外推（假定误差 ~ h^order）。"""
    r = ratio ** order
    return (r * fine - coarse) / (r - 1.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metric", default="balance", choices=sorted(METRIC_KEY))
    ap.add_argument("--coarse-d", type=int, default=11)
    ap.add_argument("--fine-d", type=int, default=17)
    ap.add_argument("--min-t", type=float, default=60.0, help="允许部分算例时的最短时长 [D/U]")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args(argv)

    key = METRIC_KEY[args.metric]
    runs = load_runs(args.min_t, args.allow_partial)
    levels = sorted(runs)
    ART.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    if args.fine_d not in runs or args.coarse_d not in runs:
        raise SystemExit(f"缺少网格级：有 {levels}，需要 {args.coarse_d} 与 {args.fine_d}")

    fine, coarse = runs[args.fine_d], runs[args.coarse_d]
    hds = sorted(fine)
    if sorted(coarse) != hds:
        print(f"[warn] 粗网格 H/D 集合与细网格不同：{sorted(coarse)} vs {hds}")

    rows_primary = build_rows(fine, STATIONARY_WINDOWS, key)
    rows_mission = build_rows(fine, MISSION_WINDOWS, key)
    rows_coarse = build_rows(coarse, MISSION_WINDOWS, key)
    rows_coarse_stat = build_rows(coarse, STATIONARY_WINDOWS, key)

    nu_cb = churchill_bernstein(float(next(iter(fine.values()))["config"]["re"]),
                               float(next(iter(fine.values()))["config"]["pr"]))
    try:
        crit = evaluate_criteria(rows_primary, rows_mission, rows_coarse, nu_cb)
    except Exception as exc:  # 数据不足（扫掠未跑完）时也要留痕
        crit = {"error": f"{type(exc).__name__}: {exc}", "all_pass": False,
                "C1": {"pass": None}, "C2": {"pass": None}, "C3": {"pass": None},
                "C4": {"pass": None}, "C5": {"pass": None},
                "note": "扫掠未跑完 -> 判据无法全部程序化评估（honest partial）"}

    # ---------- 网格收敛（最宽域） ----------
    hd_max = max(hds)
    nu_f = float(np.mean([r["nu"] for r in rows_primary if r["h_over_d"] == hd_max]))
    nu_c = float(np.mean([r["nu"] for r in rows_coarse_stat if r["h_over_d"] == hd_max]))
    ratio = args.fine_d / args.coarse_d
    grid = {
        "h_over_d": hd_max,
        "nu_coarse": nu_c, "grid_coarse": args.coarse_d,
        "nu_fine": nu_f, "grid_fine": args.fine_d,
        "ratio": ratio,
        "richardson_order1": richardson(nu_c, nu_f, ratio, 1.0),
        "richardson_order2": richardson(nu_c, nu_f, ratio, 2.0),
        "dev_pct_fine": abs(nu_f - nu_cb) / nu_cb * 100.0,
        "dev_pct_rich1": abs(richardson(nu_c, nu_f, ratio, 1.0) - nu_cb) / nu_cb * 100.0,
        "dev_pct_rich2": abs(richardson(nu_c, nu_f, ratio, 2.0) - nu_cb) / nu_cb * 100.0,
    }

    # ---------- 物理保真诊断：St、Re_meas、表面/平衡一致性 ----------
    diag = {}
    for hd, r in sorted(fine.items()):
        d = r["diag"]
        re_nom = float(r["config"]["re"])
        u_meas = float(np.mean([x["mass_in"] for x in d[-20:]])) / r["geometry"]["H_eff"]
        diag[f"hd{hd:g}"] = {
            "st": strouhal(r),
            "re_nominal": re_nom,
            "u_mean_measured": u_meas,
            "re_measured": u_meas * r["geometry"]["D_eff"] / r["lattice"]["nu"],
            "nu_balance_tail": r["final"]["nu_balance_mean_tail"],
            "nu_surface_tail": r["final"]["nu_local_mean_tail"],
            "surface_over_balance": r["final"]["nu_local_mean_tail"] / r["final"]["nu_balance_mean_tail"],
            "u_max": r["final"]["u_max"],
            "elapsed_s": r["elapsed_s"], "node_steps": r["node_steps"],
        }

    summary = {
        "metric": args.metric,
        "metric_key": key,
        "nu_churchill_bernstein": nu_cb,
        "grid_levels": {str(k): sorted(v) for k, v in runs.items()},
        "rows_window_counts": {"primary": len(rows_primary), "mission": len(rows_mission),
                               "coarse": len(rows_coarse)},
        "grid_convergence": grid,
        "physics_diagnostics": diag,
        "criteria": crit,
    }
    with open(ART / "window_table.json", "w") as fh:
        json.dump({"primary": rows_primary, "mission": rows_mission,
                   "coarse_mission": rows_coarse, "coarse_primary": rows_coarse_stat},
                  fh, indent=1)
    with open(ART / "criteria.json", "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)

    print(json.dumps({
        "nu_cb": round(nu_cb, 4),
        "C1_pass": crit["C1"].get("pass"), "C1_res_max_pct": crit["C1"].get("law_max_abs_residual_pct"),
        "dAIC": crit["C1"].get("delta_aic_mul_minus_add"),
        "C2_pass": crit["C2"].get("pass"), "nu_inf": crit["C2"].get("nu_inf_extrapolated"),
        "C2_dev_pct": crit["C2"].get("deviation_pct"),
        "C2_baseline_dev_pct": (crit["C2"].get("baseline_widest_domain") or {}).get("deviation_pct"),
        "C3_pass": crit["C3"].get("pass"), "C3_obs": crit["C3"].get("observed_diff_pct"),
        "C3_pred": crit["C3"].get("predicted_diff_pct"),
        "C4_pass": crit["C4"].get("pass"), "C4_drift": crit["C4"].get("drift_pct"),
        "C5_pass": crit["C5"].get("pass"),
        "C5_p": [ (crit["C5"].get("hd_constant") or {}).get("p_value"), (crit["C5"].get("tau_constant") or {}).get("p_value")],
        "grid": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in grid.items()},
        "a_m_beta_fine": crit["C4"].get("fine"),
        "a_m_beta_coarse": crit["C4"].get("coarse"),
        "all_pass": crit["all_pass"],
    }, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())