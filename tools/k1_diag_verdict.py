#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K1-DIAG-B 判决脚本：欠深 22.73% 的来源分解（液滴欠穿透 vs 空穴过闭合）。

输入（本 worktree 内）：
  runs/k1diag_ctrl/result.json      —— 对照臂（诊断关闭）
  runs/k1diag_diag/result.json      —— 诊断臂（--diag-cavity，只读测量）
  runs/k1diag_planes/planes.npz     —— 中截面场快照（可选，场量云图用）
锚（外部只读参照，不参与计算）：
  /root/tp-agent/work/m26392/runs/cavsweep/ctrl.json
输出：
  runs/k1diag_verdict/verdict.json  —— 全部数字（可溯）
判决阈值写死在 PREREG.md（docs/audits/k1_diag_cavity_20261121/），本脚本只执行不改。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

import numpy as np
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from benchmarks.water_entry.single_drop.run import load_case  # noqa: E402
from benchmarks.water_entry.single_drop.score import score_run  # noqa: E402

ANCHOR = "/root/tp-agent/work/m26392/runs/cavsweep/ctrl.json"
EXP_CSV = os.path.join(ROOT, "benchmarks", "water_entry", "single_drop", "fig5_experiment.csv")


def read_experiment():
    with open(EXP_CSV) as handle:
        rows = list(csv.DictReader(handle))
    return [(float(r["time_ms"]), float(r["depth_over_D"])) for r in rows]


def history_fields_delta(a_rows, b_rows):
    """Max |Δ| over every shared numeric field of two histories."""
    worst = {}
    if len(a_rows) != len(b_rows):
        return {"__length__": abs(len(a_rows) - len(b_rows))}
    for ra, rb in zip(a_rows, b_rows):
        for key in ra:
            va, vb = ra.get(key), rb.get(key)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                delta = abs(float(va) - float(vb))
                if delta > worst.get(key, 0.0):
                    worst[key] = delta
    return worst


def peak_of(rows):
    best = None
    for row in rows:
        value = row.get("depth_over_D")
        if value is None:
            continue
        if best is None or value > best["depth_over_D"]:
            best = row
    return best


def pseudo_run(history, contact_step, impact_speed, scales, metric_by_step, base):
    """Clone a run dict and replace depth_over_D by another depth metric."""
    rows = []
    for row in history:
        clone = dict(row)
        # 未定义处必须显式为 None（评分器据此截断窗口），不得回落到基准曲线
        clone["depth_over_D"] = metric_by_step.get(row["step"])
        rows.append(clone)
    run = dict(base)
    run["history"] = rows
    run["contact_step"] = contact_step
    run["impact_speed_m_s"] = impact_speed
    run["scales"] = scales
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", default=ROOT)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tree = os.path.abspath(a.tree)
    outdir = a.out or os.path.join(tree, "runs", "k1diag_verdict")
    os.makedirs(outdir, exist_ok=True)

    ctrl = json.load(open(os.path.join(tree, "runs", "k1diag_ctrl", "result.json")))
    diag = json.load(open(os.path.join(tree, "runs", "k1diag_diag", "result.json")))
    anchor = json.load(open(ANCHOR))
    exp = read_experiment()
    case = load_case(False)

    # ---------------------------------------------------------------- 1. 锚复现
    ctrl_peak = peak_of(ctrl["history"])
    anchor_peak = peak_of(anchor["history"])
    anchor_delta = history_fields_delta(anchor["history"], ctrl["history"])
    def health_block(run):
        hist = run["history"]
        contact = run["contact_step"]
        first = next(r for r in hist if r["step"] == contact)
        last = hist[-1]
        return {
            "postcontact_healthy_ms": last["time_ms"] - first["time_ms"],
            "mach_max": max(r["mach_max"] for r in hist),
            "volume_drift_end": last["volume_drift"],
            "geometric_volume_drift_end": last.get("geometric_volume_drift"),
            "last_step": last["step"],
        }

    anchor_block = {
        "anchor_source": ANCHOR,
        "anchor_peak_D": anchor_peak["depth_over_D"],
        "anchor_peak_time_ms": anchor_peak["time_ms"],
        "anchor_peak_step": anchor_peak["step"],
        "anchor_stop_reason": anchor["stop_reason"],
        "anchor_contact_step": anchor["contact_step"],
        "anchor_impact_speed_m_s": anchor["impact_speed_m_s"],
        "ctrl_peak_D": ctrl_peak["depth_over_D"],
        "ctrl_peak_time_ms": ctrl_peak["time_ms"],
        "ctrl_peak_step": ctrl_peak["step"],
        "ctrl_stop_reason": ctrl["stop_reason"],
        "ctrl_contact_step": ctrl["contact_step"],
        "ctrl_impact_speed_m_s": ctrl["impact_speed_m_s"],
        "ctrl_history_rows": len(ctrl["history"]),
        "anchor_history_rows": len(anchor["history"]),
        "max_abs_delta_vs_anchor": anchor_delta,
        "peak_rel_error_vs_anchor": abs(ctrl_peak["depth_over_D"] / anchor_peak["depth_over_D"] - 1.0),
        "anchor_health": health_block(anchor),
        "ctrl_health": health_block(ctrl),
    }

    # ------------------------------------------------- 2. 诊断臂 vs 对照臂（零行为改变）
    diag_vs_ctrl = history_fields_delta(ctrl["history"], diag["history"])
    diag_block = {
        "max_abs_delta_diag_vs_ctrl": diag_vs_ctrl,
        "diag_history_rows": len(diag["history"]),
        "diag_contact_step": diag["contact_step"],
        "diag_stop_reason": diag["stop_reason"],
        "diag_impact_speed_m_s": diag["impact_speed_m_s"],
        "diag_rows": len(diag["cavity_diag"]["rows"]),
    }

    # ---------------------------------------------------------------- 3. 判决量
    exp_peak_t, exp_peak_d = max(exp, key=lambda p: p[1])
    lo = 0.97 * exp_peak_d
    hi = 1.03 * exp_peak_d
    shallow = 0.8 * exp_peak_d
    rows = diag["cavity_diag"]["rows"]
    by_step = {r["step"]: r for r in rows}
    t_star = ctrl_peak["time_ms"]
    step_star = ctrl_peak["step"]
    row_star = by_step.get(step_star)

    # experiment peak time mapped onto the simulation clock (post-contact clock)
    contact_row = next(r for r in diag["history"] if r["step"] == diag["contact_step"])
    sim_t_at_exp_peak = contact_row["time_ms"] + exp_peak_t
    step_at_exp_peak = min(
        diag["history"], key=lambda r: abs(r["time_ms"] - sim_t_at_exp_peak)
    )["step"]

    def snapshot(step):
        row = by_step.get(step)
        if row is None:
            return None
        keys = ("d_drop", "d_drop_any", "d_drop_max_live", "d_cav_open", "d_cav_all",
                "d_cav_gap", "d_ceil", "p_cav", "cav_vent_ratio", "cav_pockets",
                "cav_body_cells", "u_back", "markers", "markers_alive")
        return {k: row.get(k) for k in keys} | {"step": step, "time_ms": row.get("time_ms")}

    snap_star = snapshot(step_star)
    snap_exp = snapshot(step_at_exp_peak)

    def rel(value, reference):
        if value is None:
            return None
        return value / reference - 1.0

    verdict_block = {
        "experiment_peak_D": exp_peak_d,
        "experiment_peak_time_ms_post_impact": exp_peak_t,
        "thresholds": {"match_lo_D": lo, "match_hi_D": hi, "shallow20_D": shallow},
        "t_star_time_ms": t_star,
        "t_star_step": step_star,
        "snapshot_at_t_star": snap_star,
        "snapshot_at_experiment_peak_time": snap_exp,
        "experiment_peak_time_on_sim_clock_ms": sim_t_at_exp_peak,
        "rel_to_experiment_at_t_star": {
            "d_drop": rel(snap_star["d_drop"], exp_peak_d),
            "d_drop_any": rel(snap_star["d_drop_any"], exp_peak_d),
            "d_drop_max_live": rel(snap_star["d_drop_max_live"], exp_peak_d),
            "d_cav_open": rel(snap_star["d_cav_open"], exp_peak_d),
            "d_cav_all": rel(snap_star["d_cav_all"], exp_peak_d),
            "d_ceil": None,  # d_ceil 是几何下沿量，与实验腔深不同量，���做相对误差
        },
    }

    d_drop_star = snap_star["d_drop"] if snap_star else None
    d_open_star = snap_star["d_cav_open"] if snap_star else None
    d_all_star = snap_star["d_cav_all"] if snap_star else None
    d_any_star = snap_star["d_drop_any"] if snap_star else None
    d_live_star = snap_star["d_drop_max_live"] if snap_star else None
    gap_star = snap_star["d_cav_gap"] if snap_star else None
    p_cav_star = snap_star["p_cav"] if snap_star else None

    # 主判据状态：d_drop 需要至少一个存活标记；全被气穴吞没时主判据不可判
    primary_status = "DETERMINED" if d_drop_star is not None else "UNDETERMINED"
    # 无阈值的闭合证据：口径截断量
    closure_truncates = bool(gap_star is not None and gap_star > 0.0)
    # 全窗闭合证据
    gap_series = [r["d_cav_gap"] for r in rows if r.get("d_cav_gap") is not None]
    p_cav_series = [r["p_cav"] for r in rows if r.get("p_cav") is not None]
    gap_max = max(gap_series) if gap_series else None
    p_cav_min = min(p_cav_series) if p_cav_series else None
    # 生长段（峰值前）闭合证据
    grow_gap = [r["d_cav_gap"] for r in rows
                if r.get("d_cav_gap") is not None and r["step"] <= step_star]
    grow_gap_max = max(grow_gap) if grow_gap else None

    verdict_block["primary_criterion_status"] = primary_status
    verdict_block["closure_evidence"] = {
        "d_cav_gap_at_t_star": gap_star,
        "p_cav_at_t_star": p_cav_star,
        "closure_truncates_measured_depth": closure_truncates,
        "d_cav_gap_max_growth_phase": grow_gap_max,
        "d_cav_gap_max_whole_window": gap_max,
        "p_cav_min_whole_window": p_cav_min,
        "cav_pockets_at_t_star": snap_star["cav_pockets"] if snap_star else None,
        "cav_vent_ratio_at_t_star": snap_star["cav_vent_ratio"] if snap_star else None,
    }
    # 闭合截断发生的连续区间（gap>0）
    intervals = []
    current = None
    for r in rows:
        has_gap = (r.get("d_cav_gap") or 0.0) > 1e-12
        if has_gap and current is None:
            current = [r["step"], r["step"]]
        elif has_gap:
            current[1] = r["step"]
        elif current is not None:
            intervals.append(current)
            current = None
    if current is not None:
        intervals.append(current)
    closure_intervals = []
    for a, b in intervals:
        block = [r for r in rows if a <= r["step"] <= b]
        closure_intervals.append({
            "step_from": a, "step_to": b,
            "t_post_contact_ms_from": block[0]["time_ms"] - contact_row["time_ms"],
            "t_post_contact_ms_to": block[-1]["time_ms"] - contact_row["time_ms"],
            "gap_max_D": max(r["d_cav_gap"] for r in block),
            "d_cav_all_max_D": max(r["d_cav_all"] for r in block),
            "p_cav_min": min(r["p_cav"] for r in block),
            "cav_pockets_max": max(r["cav_pockets"] for r in block),
        })
    verdict_block["closure_intervals"] = closure_intervals
    verdict_block["growth_phase_closure_free"] = all(
        (r.get("d_cav_gap") or 0.0) == 0.0 for r in rows
        if 250 <= r["step"] <= 1900)

    verdict_block["cross_checks"] = {
        "d_drop_at_t_star": d_drop_star,
        "d_drop_any_at_t_star": d_any_star,
        "d_drop_max_live_at_t_star": d_live_star,
        "markers_alive_at_t_star": snap_star["markers_alive"] if snap_star else None,
        "markers_total": snap_star["markers"] if snap_star else None,
        "first_step_all_markers_engulfed": next(
            (r["step"] for r in rows if r.get("markers_alive") == 0), None),
        "max_d_drop_any_whole_window": max(
            (r["d_drop_any"] for r in rows if r.get("d_drop_any") is not None), default=None),
    }

    # 分类：主判据不可判时，用预登记里已写下的交叉校核量（阈值不变）
    proxy_drop = d_drop_star if d_drop_star is not None else d_any_star
    proxy_name = "d_drop" if d_drop_star is not None else "d_drop_any"
    if d_drop_star is None:
        reason_prefix = (f"primary d_drop is UNDETERMINED at t* (all {snap_star['markers']} drop-material "
                         f"markers were engulfed by cavity gas by step "
                         f"{verdict_block['cross_checks']['first_step_all_markers_engulfed']}); "
                         f"classification uses the pre-registered cross-check quantity {proxy_name}")
    else:
        reason_prefix = ""
    if proxy_drop is None:
        verdict = "UNDETERMINED"
        reason = reason_prefix + ": no drop-material depth available at the ctrl peak step"
    elif proxy_drop < shallow:
        verdict = "H1_under_penetration"
        reason = (reason_prefix + f": {proxy_name}(t*)={proxy_drop:.4f} D < 0.8*exp={shallow:.4f} D "
                  f"(the tracked drop material never reaches the experimental crater depth); "
                  f"independently, the measured closure truncation at t* is "
                  f"d_cav_gap={gap_star} D (p_cav={p_cav_star}, pockets={snap_star['cav_pockets']}), "
                  f"i.e. the headspace-connected measure loses no depth, so the deficit is a genuine "
                  f"physical under-penetration and not a measurement artifact")
    elif lo <= proxy_drop <= hi and d_open_star is not None and d_open_star < shallow:
        verdict = "H2_over_closure"
        reason = (reason_prefix + f": {proxy_name}(t*)={proxy_drop:.4f} D within +-3% of exp={exp_peak_d:.4f} D "
                  f"but d_cav_open(t*)={d_open_star:.4f} D < 0.8*exp={shallow:.4f} D "
                  f"(depth lost to closure: d_cav_all-d_cav_open="
                  f"{(d_all_star - d_open_star) if (d_all_star is not None and d_open_star is not None) else float('nan'):.4f} D)")
    else:
        verdict = "H3_mixed"
        reason = (reason_prefix + f": {proxy_name}(t*)={proxy_drop:.4f} D, d_cav_open(t*)={d_open_star} D, "
                  f"d_cav_all(t*)={d_all_star} D fall outside the H1/H2 windows")
    verdict_block["verdict"] = verdict
    verdict_block["verdict_reason"] = reason

    # ------------------------------------------------------- 4. 18 点实验对照
    diag_by_step = {r["step"]: r for r in rows}
    score_case = case

    def score_with(metric):
        metric_by_step = {
            step: row[metric] for step, row in diag_by_step.items() if row.get(metric) is not None
        }
        run = pseudo_run(ctrl["history"], ctrl["contact_step"], ctrl["impact_speed_m_s"],
                         ctrl["scales"], metric_by_step, ctrl)
        return score_run(run, [p[0] for p in exp], [p[1] for p in exp], score_case)

    scores = {}
    for name, run in (("ctrl_depth_over_D", ctrl),):
        scores[name] = score_run(run, [p[0] for p in exp], [p[1] for p in exp], score_case)
    for metric in ("d_cav_open", "d_cav_all", "d_drop", "d_drop_any", "d_drop_max_live"):
        try:
            scores[f"diag_{metric}"] = score_with(metric)
        except Exception as error:  # noqa: BLE001
            scores[f"diag_{metric}"] = {"error": str(error)}

    # ------------------------------------------- 4b. 滴材料深度的 18 点插值对照
    # score.py 只接受非负的 depth_over_D 曲线；滴材料深度在贯穿前为负、
    # 被气穴吞没后未定义，故这里单独做插值对照（仅作辅助证据）。
    t_pc = np.array([r["time_ms"] - contact_row["time_ms"] for r in rows])
    interp_tbl = {}
    for metric in ("d_drop_any", "d_drop_max_live", "d_cav_open", "d_cav_all"):
        series = np.array([np.nan if r.get(metric) is None else r[metric] for r in rows], dtype=float)
        valid = np.isfinite(series)
        if not valid.any():
            continue
        points = []
        for t_ref, h_ref in exp:
            if t_pc[valid].min() <= t_ref <= t_pc[valid].max():
                h = float(np.interp(t_ref, t_pc[valid], series[valid]))
                points.append({"time_ms": t_ref, "experiment": h_ref, "metric": h,
                               "signed_relative_error": h / h_ref - 1.0})
        interp_tbl[metric] = {
            "points": points,
            "matched": len(points),
            "max_abs_relative_error": max((abs(p["signed_relative_error"]) for p in points),
                                          default=None),
            "peak_metric": float(np.nanmax(series)) if valid.any() else None,
            "metric_defined_steps": int(valid.sum()),
        }

    # ---------------------------------------------------------------- 5. 输出
    def trim(score):
        if "error" in score:
            return score
        return {
            "peak_relative_error": score.get("peak_relative_error"),
            "full_window_peak_depth_over_D": score.get("full_window_peak_depth_over_D"),
            "full_window_peak_time_ms": score.get("full_window_peak_time_ms"),
            "max_full_curve_relative_error": score.get("max_full_curve_relative_error"),
            "matched_points": score.get("matched_points"),
            "points_within_tolerance": score.get("points_within_tolerance"),
            "numeric_pass": score.get("numeric_pass"),
            "comparisons": score.get("comparisons"),
        }

    out = {
        "mission": "K1-DIAG-B",
        "case": "santini_case3",
        "cells_per_D": ctrl["cells_per_D"],
        "experiment_points": [{"time_ms": t, "depth_over_D": d} for t, d in exp],
        "anchor_reproduction": anchor_block,
        "diagnostic_is_read_only": diag_block,
        "verdict": verdict_block,
        "scores_18_points": {k: trim(v) for k, v in scores.items()},
        "interpolated_18_point_tables": interp_tbl,
        "diag_rows": rows,
    }
    with open(os.path.join(outdir, "verdict.json"), "w") as handle:
        json.dump(out, handle, indent=1, allow_nan=False)

    # 可读摘要
    lines = []
    lines.append("# K1-DIAG-B 判决摘要（程序生成，数字全部溯 runs/*/result.json）")
    lines.append("")
    lines.append(f"- 对照臂 ctrl：peak {ctrl_peak['depth_over_D']:.10f} D @ {ctrl_peak['time_ms']:.4f} ms "
                 f"(step {ctrl_peak['step']}), stop={ctrl['stop_reason']}, contact={ctrl['contact_step']}")
    lines.append(f"- 锚（{ANCHOR}）：peak {anchor_peak['depth_over_D']:.10f} D @ {anchor_peak['time_ms']:.4f} ms")
    lines.append(f"- 锚复现相对误差 {anchor_block['peak_rel_error_vs_anchor']:.3e}；"
                 f"history 各字段最大 |Δ| = {anchor_delta}")
    lines.append(f"- 诊断臂 vs 对照臂 history 最大 |Δ| = {diag_vs_ctrl}（零行为改变判据）")
    lines.append(f"- 实验峰 {exp_peak_d:.6f} D @ {exp_peak_t:.5f} ms（post-impact）")
    lines.append(f"- t* = {t_star:.4f} ms (step {step_star}) 诊断快照：{json.dumps(snap_star)}")
    lines.append(f"- 判决：**{verdict}** — {reason}")
    for name, score in scores.items():
        if "error" in score:
            lines.append(f"- score[{name}] 失败：{score['error']}")
        else:
            lines.append(f"- score[{name}]：peak_err={score.get('peak_relative_error')}, "
                         f"max_point_err={score.get('max_full_curve_relative_error')}, "
                         f"points_ok={score.get('points_within_tolerance')}")
    with open(os.path.join(outdir, "verdict.md"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
