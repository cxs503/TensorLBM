#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TENS1-CSF-SMOOTH-R2 满窗双臂定谳提取器（m26894）

零手抄：每个数字直接读 runs/ctrl/result.json 与 runs/csf_smooth/result.json 计算。
口径（与 mmc3_runsdiff.py / score.py 一致）：
  - peak_depth_D / peak_time_abs_ms = 原始 history 全窗 argmax（绝对时轴）
  - peak_time_pc_ms                = peak_time_abs_ms - contact_time_ms（接触后时轴，
                                      与实验 10.394ms 同口径）
  - window_abs_ms                  = history 末行 time_ms
  - postcontact_healthy_ms         = window_abs_ms - contact_time_ms
预登记（R1 卡跑前写死，本单沿用）：
  - HIT    = csf_smooth 峰深误差 <15% 且 mach_max 降 20%+
  - 证伪   = 峰深误差 >20% 或 mach_max 不降
  - ctrl 锚 = 1.435220D@11.0906ms(abs)/24.005ms(pc)/mach 0.2598（位级）
输出：runs/tens1_csf_r2/TENS1-CSF-R2.json（机读）+ 打印 markdown 表
"""
import json
import os
import re
import sys

ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXP_PEAK_D = 1.857434          # Santini case-3 实验峰（REF-WEB-01）
EXP_PEAK_PC_MS = 10.394        # 实验峰时刻（pc 口径）
CTRL_DEPTH_EXACT = 1.4352204748231814   # 正典基线峰深（cavsweep/ctrl.json argmax）
CTRL_PEAK_ABS_MS = 11.0906     # 锚：峰时 abs
CTRL_PC_MS = 24.005            # 锚：满窗 pc
CTRL_MACH = 0.2598             # 锚：mach_max


def extract(path):
    d = json.load(open(path))
    h = d["history"]
    last = h[-1]
    peak = max(h, key=lambda r: (r.get("depth_over_D") or 0.0))
    cs = d.get("contact_step")
    ct = next((r["time_ms"] for r in h if r["step"] == cs), None)
    vdrift = max((r.get("volume_drift") or 0) for r in h)
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
        "death_cause": death,
        "window_abs_ms": last["time_ms"],
        "contact_step": cs,
        "contact_abs_ms": ct,
        "postcontact_healthy_ms": (last["time_ms"] - ct) if ct is not None else None,
        "peak_depth_D": peak["depth_over_D"],
        "peak_time_abs_ms": peak["time_ms"],
        "peak_time_pc_ms": (peak["time_ms"] - ct) if ct is not None else None,
        "peak_step": peak["step"],
        "mach_max": mach,
        "max_volume_drift": vdrift,
        "n_history": len(h),
        "sigma_lattice": d.get("scales", {}).get("sigma"),
        "u_limit": d.get("u_limit"),
        "momconv": d.get("momconv"),
    }


def row_diff(a, b):
    """两臂 history 逐行逐字段最大绝对差（位级一致性检验）。"""
    ha, hb = a["history"], b["history"]
    if len(ha) != len(hb):
        return {"n_rows_a": len(ha), "n_rows_b": len(hb), "max_abs_diff": None,
                "bit_identical": False}
    keys = set()
    for r in ha:
        keys.update(r.keys())
    worst, where = 0.0, None
    for i, (ra, rb) in enumerate(zip(ha, hb)):
        for k in sorted(keys):
            va, vb = ra.get(k), rb.get(k)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)) \
                    and not isinstance(va, bool) and not isinstance(vb, bool):
                diff = abs(va - vb)
                if diff > worst:
                    worst, where = diff, (i, k)
    return {"n_rows": len(ha), "max_abs_diff": worst, "worst_at": where,
            "bit_identical": worst == 0.0}


def main():
    ca = os.path.join(ROOT, "runs", "ctrl", "result.json")
    cb = os.path.join(ROOT, "runs", "csf_smooth", "result.json")
    A, B = json.load(open(ca)), json.load(open(cb))
    ea, eb = extract(ca), extract(cb)
    ea["arm"] = "ctrl"
    eb["arm"] = "csf_smooth"
    for e in (ea, eb):
        e["peak_err_pct"] = (e["peak_depth_D"] / EXP_PEAK_D - 1.0) * 100.0
    diff = row_diff(A, B)
    mach_drop_pct = (1.0 - eb["mach_max"] / ea["mach_max"]) * 100.0 \
        if ea["mach_max"] else None

    # 预登记判定
    hit = (abs(eb["peak_err_pct"]) < 15.0) and (mach_drop_pct is not None and mach_drop_pct >= 20.0)
    falsified = (abs(eb["peak_err_pct"]) > 20.0) or (mach_drop_pct is None or mach_drop_pct <= 0.0)
    anchor = {
        "peak_depth_D": ea["peak_depth_D"],
        "peak_depth_bit_match": ea["peak_depth_D"] == CTRL_DEPTH_EXACT,
        "peak_time_abs_ms": ea["peak_time_abs_ms"],
        "peak_time_abs_match": abs(ea["peak_time_abs_ms"] - CTRL_PEAK_ABS_MS) < 5e-4,
        "postcontact_healthy_ms": ea["postcontact_healthy_ms"],
        "postcontact_match": ea["postcontact_healthy_ms"] is not None
        and abs(ea["postcontact_healthy_ms"] - CTRL_PC_MS) < 5e-4,
        "mach_max": ea["mach_max"],
        "mach_match": abs(ea["mach_max"] - CTRL_MACH) < 5e-5,
    }
    trace = {}
    for arm in ("ctrl", "csf_smooth"):
        t = {"card": None, "container": None, "commit": None, "runlog_head": None}
        try:
            t["runlog_head"] = open(os.path.join(ROOT, "runs", arm, "run.log")).readline().strip()
            m = re.search(r"card=(\S+)", t["runlog_head"])
            if m:
                t["card"] = m.group(1)
        except OSError:
            pass
        try:
            st = json.load(open(os.path.join(ROOT, "runs", arm, "_sdaa_state.json")))
            t["container"] = st.get("container")
            t["commit"] = st.get("commit")
        except OSError:
            pass
        trace[arm] = t
    J = {
        "mission": "TENS1-CSF-SMOOTH-R2",
        "trace": trace,
        "arms": [ea, eb],
        "cross_arm_history_diff": diff,
        "mach_drop_pct_csf_vs_ctrl": mach_drop_pct,
        "ctrl_anchor_check": anchor,
        "prereg": {
            "hit": hit,
            "falsified": falsified,
            "hit_rule": "csf_smooth 峰深误差<15% 且 mach_max 降 20%+",
            "falsify_rule": "峰深误差>20% 或 mach_max 不降",
        },
        "sources": {"ctrl": ca, "csf_smooth": cb},
    }
    out = os.path.join(ROOT, "runs", "tens1_csf_r2")
    os.makedirs(out, exist_ok=True)
    json.dump(J, open(os.path.join(out, "TENS1-CSF-R2.json"), "w"), indent=1)

    L = ["| 臂 | stop | 窗长 abs/pc ms | 峰深 D | 峰时 abs/pc ms | 误差% | mach_max | Δmach vs ctrl |",
         "|---|---|---|---|---|---|---|---|"]
    for e in (ea, eb):
        pc = "%.3f" % e["peak_time_pc_ms"] if e["peak_time_pc_ms"] is not None else "—"
        dm = "—" if e["arm"] == "ctrl" else ("%+.2f%%" % mach_drop_pct)
        L.append("| %s | %s | %.3f/%.3f | %.6f | %.3f/%s | %+.2f | %.4f | %s |" % (
            e["arm"], e["stop_reason"], e["window_abs_ms"],
            e["postcontact_healthy_ms"] or 0.0, e["peak_depth_D"],
            e["peak_time_abs_ms"], pc, e["peak_err_pct"], e["mach_max"], dm))
    print("\n".join(L))
    print("\n[anchor] ctrl: depth=%.6fD(bit=%s) t_abs=%.4fms pc=%.4fms mach=%.4f" % (
        ea["peak_depth_D"], anchor["peak_depth_bit_match"],
        ea["peak_time_abs_ms"], ea["postcontact_healthy_ms"], ea["mach_max"]))
    print("[diff] rows=%s max_abs_diff=%s bit_identical=%s" % (
        diff.get("n_rows"), diff.get("max_abs_diff"), diff["bit_identical"]))
    print("[verdict] HIT=%s FALSIFIED=%s (mach_drop=%.4f%%, csf_err=%.2f%%)" % (
        hit, falsified, mach_drop_pct if mach_drop_pct is not None else float("nan"),
        eb["peak_err_pct"]))
    print("wrote", os.path.join(out, "TENS1-CSF-R2.json"))


if __name__ == "__main__":
    main()
