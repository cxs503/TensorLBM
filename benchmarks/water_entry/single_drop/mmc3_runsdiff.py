#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MMC3 全臂差分对照提取器（诊断官 m26746）

零手抄：每个数字直接读 runs/mmc3_diff/**/result.json（或 score 工件）计算。
口径声明（防混用）：
  - peak_depth / peak_time_abs_ms  = 原始 history 全窗 argmax（绝对时轴，t=0 为初始化）
  - peak_time_pc_ms                = peak_time_abs_ms - contact_time_ms（接触后时轴，
                                      与 score.py / 实验 10.394ms 同口径）
  - window_abs_ms                  = history 末行 time_ms
  - postcontact_healthy_ms         = window_abs - contact_time（score.py 口径）
  - 死因 = health_limit 时按末行 mach_max>0.3 / volume_drift>1e-4 归因
输出：runs/mmc3_diff/MMC3-RUNSDIFF.json（机读）+ 打印 markdown 表
"""

import json
import os
import sys

ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "runs", "mmc3_diff"
)
EXP_PEAK_D = 1.857434  # Santini case-3 实验峰（REF-WEB-01 / 各卡一致）
EXP_PEAK_PC_MS = 10.394  # 实验峰时刻（接触后时轴，score.py 对比口径）
CTRL_DEPTH = 1.4352204748231814  # 正典基线峰深（ctrl 全窗, 精确值=cavsweep/ctrl.json argmax）

ARMS = [
    # (arm, 系列, 变量, 镜像路径, 原始出处)
    (
        "ctrl",
        "CAVSWEEP",
        "base(UL.15)",
        "cavsweep/ctrl.json",
        "m26392/runs/cavsweep/ctrl.json (node099, 16/16 齐)",
    ),
    ("ul006", "CAVSWEEP", "UL 0.06", "cavsweep/ul006.json", "m26392/runs/cavsweep/ul006.json"),
    ("ul010", "CAVSWEEP", "UL 0.10", "cavsweep/ul010.json", "m26392/runs/cavsweep/ul010.json"),
    ("ul020", "CAVSWEEP", "UL 0.20", "cavsweep/ul020.json", "m26392/runs/cavsweep/ul020.json"),
    ("ul024", "CAVSWEEP", "UL 0.24", "cavsweep/ul024.json", "m26392/runs/cavsweep/ul024.json"),
    ("rs135", "CAVSWEEP", "RS 1.35", "cavsweep/rs135.json", "m26392/runs/cavsweep/rs135.json"),
    ("rs175", "CAVSWEEP", "RS 1.75", "cavsweep/rs175.json", "m26392/runs/cavsweep/rs175.json"),
    ("gap12", "CAVSWEEP", "gap 0.12", "cavsweep/gap12.json", "m26392/runs/cavsweep/gap12.json"),
    ("pd20", "CAVSWEEP", "PD 2.0", "cavsweep/pd20.json", "m26392/runs/cavsweep/pd20.json"),
    (
        "caponly",
        "CAVSWEEP",
        "SMe off",
        "cavsweep/caponly.json",
        "m26392/runs/cavsweep/caponly.json",
    ),
    (
        "simonly",
        "CAVSWEEP",
        "cap off",
        "cavsweep/simonly.json",
        "m26392/runs/cavsweep/simonly.json",
    ),
    ("nearest", "CAVSWEEP", "NS on", "cavsweep/nearest.json", "m26392/runs/cavsweep/nearest.json"),
    ("spd005", "CAVSWEEP", "spd 0.05", "cavsweep/spd005.json", "m26392/runs/cavsweep/spd005.json"),
    ("spd007", "CAVSWEEP", "spd 0.07", "cavsweep/spd007.json", "m26392/runs/cavsweep/spd007.json"),
    (
        "ul012pd20",
        "CAVSWEEP",
        "UL.12+PD2.0",
        "cavsweep/ul012pd20.json",
        "m26392/runs/cavsweep/ul012pd20.json",
    ),
    (
        "rs135ul020",
        "CAVSWEEP",
        "UL.20(重复)",
        "cavsweep/rs135ul020.json",
        "m26392/runs/cavsweep/rs135ul020.json",
    ),
    (
        "kernb_ctrl",
        "KERN-B",
        "base(无旗标)",
        "kernb/ctrl.json",
        "m26685/runs/kernb/ctrl.json (cj5694 仿真段 rc 完整; cj 账 rc=1=包装层, 见卡)",
    ),
    (
        "kernb_B1",
        "KERN-B",
        "cav-deficit 0.995",
        "kernb/b1_0995.json",
        "m26685/runs/kernb/b1_0995.json (cj5695 同上)",
    ),
    (
        "KERN-A-R2",
        "KERN-A-R2",
        "ABB Option3 7+/9-",
        "kern_a_r2/result.json",
        "m26693/runs/KERN-A-R2/result.json (SDAA card0, run.log 首行 arm/card/commit)",
    ),
    (
        "abbs_A",
        "ABBS",
        "ABB=u_pc(放大)",
        "abbs/armA_fix.json",
        "cj5684 fetch tar → repo/runs/abbs/armA_fix.json (rc=0)",
    ),
    (
        "abbs_B",
        "ABBS",
        "ABB 零注入",
        "abbs/armB_zero.json",
        "cj5685 fetch tar → repo/runs/abbs/armB_zero.json (rc=0)",
    ),
    (
        "abbs_C",
        "ABBS",
        "ABB=u_liquid",
        "abbs/armC_liquid.json",
        "cj5687 fetch tar → repo/runs/abbs/armC_liquid.json (rc=0)",
    ),
    (
        "conv12_cal8",
        "CONV12",
        "cells 8 修复臂",
        "conv12/cal8_fixed.json",
        "m26363/runs/t4_r2/cal8_fixed.json (cj5691 rc=0; 与 m26372 复现逐位同)",
    ),
    (
        "conv12_cal12",
        "CONV12",
        "cells 12 修复臂",
        "conv12/case3_cal12_fixed.json",
        "m26372/docs/audits/conv12_20261120/case3_cal12_fixed.json (cj5692 rc=0)",
    ),
]


def extract(path):
    d = json.load(open(path))
    h = d["history"]
    last = h[-1]
    peak = max(h, key=lambda r: r.get("depth_over_D") or 0.0)
    cs = d.get("contact_step")
    ct = next((r["time_ms"] for r in h if r["step"] == cs), None)
    vdrift = max((r.get("volume_drift") or 0) for r in h)
    gdrift = max((r.get("geometric_volume_drift") or 0) for r in h)
    mach = max((r.get("mach_max") or 0) for r in h)
    stop = d.get("stop_reason")
    death = None
    if stop == "health_limit":
        lr = last
        if (lr.get("mach_max") or 0) > 0.3:
            death = "Ma 门 (mach_max>0.3)"
        elif (lr.get("volume_drift") or 0) > 1e-4:
            death = "drift 门 (volume_drift>1e-4)"
        else:
            death = "health 门(其他)"
    return {
        "stop_reason": stop,
        "window_abs_ms": last["time_ms"],
        "contact_step": cs,
        "contact_abs_ms": ct,
        "postcontact_healthy_ms": (last["time_ms"] - ct) if ct is not None else None,
        "peak_depth_D": peak["depth_over_D"],
        "peak_time_abs_ms": peak["time_ms"],
        "peak_time_pc_ms": (peak["time_ms"] - ct) if ct is not None else None,
        "peak_step": peak["step"],
        "max_volume_drift": vdrift,
        "max_geometric_drift": gdrift,
        "max_mach": mach,
        "death_cause": death,
        "impact_speed_m_s": d.get("impact_speed_m_s"),
        "cells_per_D": d.get("cells_per_D"),
    }


def main():
    rows = []
    for arm, series, var, rel, origin in ARMS:
        p = os.path.join(ROOT, rel)
        e = extract(p)
        e.update(
            {
                "arm": arm,
                "series": series,
                "variable": var,
                "source_mirror": "runs/mmc3_diff/" + rel,
                "source_origin": origin,
            }
        )
        e["peak_err_pct"] = (e["peak_depth_D"] / EXP_PEAK_D - 1) * 100
        e["delta_vs_ctrl_D"] = e["peak_depth_D"] - CTRL_DEPTH
        e["direction_vs_ctrl"] = (
            "identical"
            if abs(e["delta_vs_ctrl_D"]) < 1e-6
            else ("deeper" if e["delta_vs_ctrl_D"] > 0 else "shallower")
        )
        exp_peak_abs = EXP_PEAK_PC_MS + (e["contact_abs_ms"] or 0.0)
        e["died_before_exp_peak"] = (
            e["stop_reason"] == "health_limit" and e["window_abs_ms"] < exp_peak_abs
        )
        rows.append(e)
    out = {
        "card": "MMC3-RUNSDIFF",
        "task": "NEW:MMC3-全臂差分诊断对照卡",
        "baseline": {
            "canonical": "ctrl 1.4352D@11.091ms(abs)/10.543ms(pc) 满窗 24.552/24.005ms",
            "experiment": {
                "peak_D": EXP_PEAK_D,
                "peak_time_pc_ms": EXP_PEAK_PC_MS,
                "source": "Santini case-3 (REF-WEB-01); score_pair.json reference_observed_peak",
            },
        },
        "time_origin_note": "abs=绝对时轴(t0=初始化); pc=接触后时轴(score.py/实验同口径); 差=contact 0.547ms",
        "arms": rows,
    }
    outp = os.path.join(ROOT, "MMC3-RUNSDIFF.json")
    json.dump(out, open(outp, "w"), ensure_ascii=False, indent=1)
    print("wrote", outp, "arms=", len(rows))
    # markdown 表
    print(
        "| 臂 | 系列 | 变量 | stop | 窗长abs/pc ms | 峰深D | 峰时abs/pc ms | 误差% | ΔD vs ctrl | 死因 |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for e in rows:
        pc = "%.3f" % e["peak_time_pc_ms"] if e["peak_time_pc_ms"] is not None else "—"
        print(
            "| %s | %s | %s | %s | %.3f/%.3f | %.4f | %.3f/%s | %.2f | %+.4f | %s |"
            % (
                e["arm"],
                e["series"],
                e["variable"],
                e["stop_reason"],
                e["window_abs_ms"],
                e["postcontact_healthy_ms"] or 0,
                e["peak_depth_D"],
                e["peak_time_abs_ms"],
                pc,
                e["peak_err_pct"],
                e["delta_vs_ctrl_D"],
                e["death_cause"] or "—",
            )
        )


if __name__ == "__main__":
    main()
