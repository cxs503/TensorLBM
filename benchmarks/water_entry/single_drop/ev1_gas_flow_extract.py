#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DE-X-EV1-GAS_FLOW-R2 · 程序化提取与判决（零手抄）。

用法：python3 benchmarks/water_entry/single_drop/ev1_gas_flow_extract.py --tree .
输入：runs/EV1-GAS_FLOW-CTRL/result.json、runs/EV1-GAS_FLOW/result.json、
      benchmarks/water_entry/single_drop/fig5_experiment.csv
输出：runs/EV1-GAS_FLOW/verdict.json + 终端判决表（数字全部来自输入文件）。
判据：docs/audits/ev1_gas_flow_20261121/PREREG.md（跑前锁定）。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

# 战役锚（CAVSWEEP ctrl，m26392/runs/cavsweep/ctrl.json；m26887 已位级复现）
ANCHOR = {
    "peak_depth_D": 1.4352204748231814,
    "peak_time_ms": 11.090571428571428,
    "stop_reason": "window_complete",
    "t_end_ms": 24.55248214285714,
    "contact_step": 45,
    "mach_max": 0.25980762113533173,
    "final_volume_drift": 4.8058856125798854e-06,
    "impact_speed_m_s": 1.4091844081433575,
}
# 实验基准（Santini case-3，fig5_experiment.csv 峰点）
EXP_PEAK_DEPTH_D = 1.857434
EXP_PEAK_TIME_PC_MS = 10.39446
# 预登记判据（TRACK_EV_W1 处方段）
PREDICT_DEPTH_PCT = 2.0
FALSIFY_DEPTH_PCT = 3.0
FALSIFY_PHASE_MS = 0.5


def load_arm(path: Path) -> dict:
    d = json.loads(path.read_text())
    h = d["history"]
    dt_ms = h[1]["time_ms"] - h[0]["time_ms"]
    pk = max(h, key=lambda r: r.get("depth_over_D", -1.0))
    contact_ms = (d.get("contact_step") or 0) * dt_ms
    return {
        "stop_reason": d["stop_reason"],
        "n_hist": len(h),
        "t_end_ms": h[-1]["time_ms"],
        "peak_depth_D": pk["depth_over_D"],
        "peak_step": pk["step"],
        "peak_time_ms": pk["time_ms"],
        "peak_time_pc_ms": pk["time_ms"] - contact_ms,
        "mach_max": max(r["mach_max"] for r in h),
        "final_volume_drift": h[-1]["volume_drift"],
        "contact_step": d.get("contact_step"),
        "impact_speed_m_s": d.get("impact_speed_m_s"),
        "dt_ms": dt_ms,
        "history": h,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=".")
    a = ap.parse_args()
    tree = Path(a.tree).resolve()

    ctrl = load_arm(tree / "runs" / "EV1-GAS_FLOW-CTRL" / "result.json")
    arm = load_arm(tree / "runs" / "EV1-GAS_FLOW" / "result.json")

    # 实验 18 点
    exp = []
    with open(tree / "benchmarks" / "water_entry" / "single_drop" / "fig5_experiment.csv") as fh:
        for row in csv.DictReader(fh):
            exp.append((float(row["time_ms"]), float(row["depth_over_D"])))
    exp_peak = max(exp, key=lambda p: p[1])

    # ① CTRL 位级锚门
    anchor_gate = {
        "peak_depth_D": ctrl["peak_depth_D"] == ANCHOR["peak_depth_D"],
        "peak_time_ms": ctrl["peak_time_ms"] == ANCHOR["peak_time_ms"],
        "stop_reason": ctrl["stop_reason"] == ANCHOR["stop_reason"],
        "t_end_ms": ctrl["t_end_ms"] == ANCHOR["t_end_ms"],
        "contact_step": ctrl["contact_step"] == ANCHOR["contact_step"],
        "mach_max": ctrl["mach_max"] == ANCHOR["mach_max"],
        "final_volume_drift": ctrl["final_volume_drift"] == ANCHOR["final_volume_drift"],
        "impact_speed_m_s": ctrl["impact_speed_m_s"] == ANCHOR["impact_speed_m_s"],
    }
    anchor_ok = all(anchor_gate.values())

    # ② 双臂对照（唯一变量 gas_flow）
    d_depth_pct = (arm["peak_depth_D"] - ctrl["peak_depth_D"]) / ctrl["peak_depth_D"] * 100.0
    d_time_ms = arm["peak_time_pc_ms"] - ctrl["peak_time_pc_ms"]
    ad, at = abs(d_depth_pct), abs(d_time_ms)
    if not anchor_ok:
        verdict = "WITHHELD（基础设施 FAIL：ctrl 未逐位复现战役锚，不出物理判决）"
    elif arm["stop_reason"] != "window_complete":
        verdict = ("INFEASIBLE（主臂未跑满窗：gas_flow 满强度势流初值在战役健康门下数值不可行，"
                   "满窗峰深/相位对照不可执行，预登记判据无法套用）")
    elif ad <= PREDICT_DEPTH_PCT and at <= FALSIFY_PHASE_MS:
        verdict = "PREDICTED（预测成立：gas_flow 效应低于可辨阈）"
    elif ad > FALSIFY_DEPTH_PCT or at > FALSIFY_PHASE_MS:
        verdict = "FALSIFIED（预测证伪：gas_flow 有实质效应）"
    else:
        verdict = "INTERMEDIATE（中间区：预测未成立、未证伪）"

    # ③ vs 实验
    err = lambda d: (d - EXP_PEAK_DEPTH_D) / EXP_PEAK_DEPTH_D * 100.0

    out = {
        "anchor_gate": anchor_gate,
        "anchor_ok": anchor_ok,
        "ctrl": {k: v for k, v in ctrl.items() if k != "history"},
        "arm": {k: v for k, v in arm.items() if k != "history"},
        "delta_vs_ctrl": {"depth_pct": d_depth_pct, "time_ms": d_time_ms},
        "verdict": verdict,
        "vs_experiment": {
            "ctrl_peak_depth_err_pct": err(ctrl["peak_depth_D"]),
            "arm_peak_depth_err_pct": err(arm["peak_depth_D"]),
            "ctrl_peak_time_pc_ms": ctrl["peak_time_pc_ms"],
            "arm_peak_time_pc_ms": arm["peak_time_pc_ms"],
            "exp_peak": {"depth_D": EXP_PEAK_DEPTH_D, "time_pc_ms": EXP_PEAK_TIME_PC_MS,
                         "csv_peak": list(exp_peak)},
        },
    }
    outdir = tree / "runs" / "EV1-GAS_FLOW"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "verdict.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    def f(v, n=6):
        return f"{v:.{n}f}" if isinstance(v, float) else str(v)

    print("=== 位级锚门（CTRL vs 战役锚）===")
    for k, ok in anchor_gate.items():
        print(f"  {k}: {'PASS' if ok else 'FAIL'}")
    print(f"anchor_ok = {anchor_ok}")
    print("=== 三元组（程序化提取）===")
    for name, r in (("CTRL", ctrl), ("GAS_FLOW", arm)):
        print(f"  {name}: peak {f(r['peak_depth_D'])}D @ {f(r['peak_time_ms'])}ms(abs)/"
              f"{f(r['peak_time_pc_ms'])}ms(pc) stop={r['stop_reason']} t_end={f(r['t_end_ms'])}ms "
              f"mach_max={f(r['mach_max'])} drift_end={r['final_volume_drift']:.6e} "
              f"contact={r['contact_step']} impact={f(r['impact_speed_m_s'])}m/s")
    print(f"=== Δ(arm-ctrl): depth {d_depth_pct:+.4f}%  time {d_time_ms:+.4f}ms(pc) ===")
    print(f"=== vs 实验 {EXP_PEAK_DEPTH_D}D@{EXP_PEAK_TIME_PC_MS}ms(pc): "
          f"ctrl {err(ctrl['peak_depth_D']):+.3f}%  arm {err(arm['peak_depth_D']):+.3f}% ===")
    print(f"=== 判决: {verdict} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
