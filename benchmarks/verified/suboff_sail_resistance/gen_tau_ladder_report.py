#!/usr/bin/env python3
"""Assemble the SIR-1b report from the machine-written verdict (m298).

Every number in the generated markdown is read out of
``/nfs/wangxi/runs/tau_ladder_uniform_20261025/verdict.json`` — no hand-typed
values, per the campaign discipline (零手抄).

The τ–Re coupling section is the m298 *supplement*: the prereg freezes Cf_ref at
Re=1000, but each ladder arm sits at a different effective Reynolds number
because ν_lb=(τ-0.5)/3 while u_lb and L are fixed.  The supplement quantifies
that offset and re-evaluates the references at the matched Re, so the in-band /
out-of-band judgement is made on a like-for-like caliber.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

STAGING = Path("/nfs/wangxi/runs/tau_ladder_uniform_20261025")
VERDICT = STAGING / "verdict.json"
REPORT = STAGING / "report.md"
ARMS = ["A", "B", "C", "D", "E"]


def f6(x, nd=6):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "✅" if x else "❌"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def pct(x, nd=2):
    if x is None:
        return "—"
    return f"{float(x):+.{nd}f}%"


def sci(x, nd=3):
    if x is None:
        return "—"
    return f"{float(x):.{nd}e}"


def main() -> None:
    v = json.loads(VERDICT.read_text())
    arms = v["arms"]
    g1 = v["gates"]["G1_total_drag_positive"]["per_arm"]
    g2 = v["gates"]["G2_magnitude_vs_reference"]["per_arm"]
    g3 = v["gates"]["G3_monotone_in_tau"]
    anchors = v["anchor_family"]["per_arm"]
    sup = v["supplement_tau_re_coupling"]["per_arm"]
    gsum = v["gates_summary"]
    setup = v["frozen_setup"]
    ref = v["reference"]

    L: list[str] = []
    add = L.append

    add("# SIR-1b 报告：均匀网格 τ 阶梯（G1 流体载荷精度封口）\n")
    add("- 任务：m298（SIR-1b，接 m292 残值）")
    add(f"- 案例：`{setup['case']}`（{setup['hull_type']}，Re=1000 冻结参考）")
    add(
        f"- 冻结 prereg：`{v['prereg']['path']}` md5 `{v['prereg']['md5_actual']}`"
        f"（冻结值 `{v['prereg']['md5_frozen']}`，"
        f"match={v['prereg']['md5_match']}）"
    )
    add(
        f"- 阶梯：{setup['ladder']}；resolution={setup['resolution']}，"
        f"steps={setup['steps']}，u_lb={setup['u_lb']}，"
        f"collision={setup['collision']}，friction={setup['friction']}，p0={setup['p0']}"
    )
    add(
        f"- GPU：{setup['gpu']}（`CUDA_VISIBLE_DEVICES={setup['gpu']}`，进程内 device="
        f"`{setup['device_arg']}`）"
    )
    add(f"- 解释器：`{v['interpreter']['selected']}`\n")

    add(f"## 0. 总判决：**{v['overall']}**\n")
    add("| 门 | 判据（prereg 逐字） | 结果 |")
    add("|---|---|---|")
    add(
        f"| G1 | Cd_total > 0（符号正确） | {f6(gsum['G1_total_drag_positive'])} "
        f"（{sum(1 for a in ARMS if g1[a]['pass'])}/5 档为正） |"
    )
    add(
        f"| G2 | \\|Cd_total/Cf_ref−1\\| ≤ 50%（Re=1000 冻结参考） | "
        f"{f6(gsum['G2_magnitude_within_50pct_frozen_ref'])} |"
    )
    add(
        f"| G3 | Cd_total 随 τ 单调（≥3 档） | {f6(gsum['G3_monotone_in_tau'])} "
        f"（增 {g3['n_increasing']} / 减 {g3['n_decreasing']}） |"
    )
    add("")

    add("## 1. τ 阶梯主表（Cd 与有效 Re）\n")
    add(
        "| 档 | τ 目标 | τ 实测 | ν_lb | Re_eff | Cd_pressure | Cd_friction | "
        "**Cd_total** | 收敛窗 25%→100% |"
    )
    add("|---|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        r = arms[a]
        cw = r.get("convergence_windows") or {}
        seq = " → ".join(f6(cw.get(k), 5) for k in ("25%", "50%", "75%", "100%"))
        add(
            f"| {a} | {f6(r['tau_target'], 3)} | {f6(r.get('tau_actual'), 6)} | "
            f"{f6(r.get('nu_lb_actual'), 6)} | {f6(r.get('re_lattice_actual'), 1)} | "
            f"{sci(r.get('Cd_pressure'))} | {sci(r.get('Cd_friction'))} | "
            f"**{sci(r.get('Cd_total'))}** | {seq} |"
        )
    add("")

    ran = [a for a in ARMS if arms[a].get("status") == "ok"]
    add(
        f"- 五档全部完成且 τ 命中（\\|τ_actual−τ_target\\| ≤ 1e-9）："
        f"{len(ran)}/5（status ok 档：{ran}）"
    )
    mism = {a: arms[a].get("sanity_problems") for a in ARMS if arms[a].get("sanity_problems")}
    add(f"- 逐档健全性异常：{mism if mism else '无'}")
    add("")

    add("## 2. 判据逐门\n")
    add("### G1 — 总阻力符号\n")
    add("| 档 | Cd_total | > 0 |")
    add("|---|---|---|")
    for a in ARMS:
        add(f"| {a} | {sci(g1[a]['Cd_total'])} | {f6(g1[a]['pass'])} |")
    add("")
    add("### G2 — 量级对参考（双口径）\n")
    add(
        "冻结口径（Cf_ref@Re=1000，**判门用**）与同有效 Re 口径（**诊断用**，"
        "见 §3 说明为何必须双报）：\n"
    )
    add(
        "| 档 | Re_eff | Cd_total | Cf_ref@1000 | 偏差（冻结口径） | 判 | "
        "Blasius@Re_eff | 偏差（同 Re 口径） | 判 |"
    )
    add("|---|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        r = g2[a]
        add(
            f"| {a} | {f6(r['re_effective'], 1)} | {sci(r['Cd_total'])} | "
            f"{f6(r['cf_ref_frozen_Re1000'], 6)} | {pct(r['err_vs_frozen_pct'])} | "
            f"{f6(r['pass_frozen'])} | {f6(r['blasius_cf_at_re_effective'], 6)} | "
            f"{pct(r['err_vs_rematched_pct'])} | {f6(r['in_band_rematched_50pct'])} |"
        )
    add("")
    add("### G3 — τ 单调性\n")
    add("| τ | Cd_total |")
    add("|---|---|")
    for row in g3["series"]:
        add(f"| {f6(row['tau'], 3)} | {sci(row['Cd_total'])} |")
    add("")
    add(f"- 逐档差值：{[round(d, 8) for d in g3['diffs']]}")
    add(f"- 增 {g3['n_increasing']} / 减 {g3['n_decreasing']}；G3 = {f6(g3['pass'])}")
    add("")

    add("## 3. τ–Re 耦合分析（m298 增补节，非 prereg 冻结门）\n")
    add("**关键事实**：u_lb 与 L 由案例冻结，故阶梯旋钮 τ 本身就是雷诺旋钮——\n")
    add("```\nν_lb = (τ − 0.5)/3,   Re = u_lb·L/ν_lb   ⇒   τ↑ ⇒ ν↑ ⇒ Re↓\n```\n")
    add(
        "| 档 | τ | ν_lb | Re_eff | 相对 Re=1000 偏移 | Blasius Cf@Re_eff | "
        "ITTC-1957 Cf@Re_eff | Cd_total | 偏差（同 Re Blasius） |"
    )
    add("|---|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        s = sup[a]
        add(
            f"| {a} | {f6(s['tau'], 6)} | {f6(s['nu_lb'], 6)} | {f6(s['re_lattice'], 2)} | "
            f"{pct(s['re_shift_vs_1000_pct'], 1)} | {f6(s['blasius_cf_at_re'], 6)} | "
            f"{f6(s['ittc1957_cf_at_re'], 6)} | {sci(g1[a]['Cd_total'])} | "
            f"{pct(g2[a]['err_vs_rematched_pct'])} |"
        )
    add("")
    re_lo = min(sup[a]["re_lattice"] for a in ARMS)
    re_hi = max(sup[a]["re_lattice"] for a in ARMS)
    add(
        f"- 阶梯跨度为 **Re {f6(re_hi, 1)} → {f6(re_lo, 1)}**"
        f"（跨度比 {f6(re_hi / re_lo, 2)}×）。这不是等 Re 扫描。"
    )
    add(
        "- 因此冻结的 Cf_ref@Re=1000 只对有效 Re≈1000 的档在口径上成立；"
        "对高 τ（低 Re）档，参考必须按同一有效 Re 重算后再判带内/带外。"
    )
    add(
        "- **ITTC-1957 带口径校正**：ITTC-1957 是湍流相关线，"
        "在 Re<10⁴ 处作为层流参考不适用（Re→100 处有极点）；"
        "其数值列在表内仅为口径留痕，**不参与判带**。判带一律用同 Re 的 Blasius。"
    )
    add("")

    add("## 4. FA 锚族逐 τ 记录（冲量 / ct / e0 / E0 / 艏艉 ρ 偶极 / D_QW）\n")
    add(
        "口径：Ladd 链式 MEM（流体→固体穿越链集），归一化 dpS = 0.5·u_lb²·S_总湿面积，"
        "与 Cd_total 同口径。`E0_x/E1_x/E2_x` 为 #257/#265 平衡态分解，"
        "`NEQ_x` 为链和减三者；恒等式残差逐样本记录。\n"
    )
    add(
        "| 档 | τ | 冲量 mem_fx | 冲量 bgsub | ct | e0 | E0_x | E1_x | E2_x | NEQ_x | "
        "ρ_fwd | ρ_aft | **D_QW** |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        an = anchors[a]
        if not an.get("present"):
            add(
                f"| {a} | {f6(arms[a].get('tau_actual'), 6)} | 锚族数据缺失 "
                f"({an.get('reason')}) | | | | | | | | | | |"
            )
            continue
        m = an["means"]
        add(
            f"| {a} | {f6(arms[a].get('tau_actual'), 6)} | {sci(m['mem_fx'])} | "
            f"{sci(m['mem_bgsub_fx'])} | {f6(m['ct'], 4)} | {sci(m['e0'])} | "
            f"{sci(m['E0_x'])} | {sci(m['E1_x'])} | {sci(m['E2_x'])} | {sci(m['NEQ_x'])} | "
            f"{f6(m['rho_fwd'], 8)} | {f6(m['rho_aft'], 8)} | **{sci(m['D_QW'])}** |"
        )
    add("")
    resid = {
        a: anchors[a].get("max_abs_identity_residual") for a in ARMS if anchors[a].get("present")
    }
    add(
        f"- 锚族恒等式 `mem_fx = E0_x+E1_x+E2_x+NEQ_x` 最大残差："
        f"{ {a: sci(x) for a, x in resid.items()} }"
    )
    exact = [
        a
        for a in ARMS
        if anchors[a].get("present") and anchors[a].get("identity_residual_exact_zero")
    ]
    add(f"- 残差逐样本精确为 0 的档：{exact}（float64 归约顺序固定 ⇒ 恒等严格闭合）")
    add("")

    add("## 5. 结论\n")
    if v["overall"] == "PASS":
        add("- 均匀网格 τ 阶梯复现了「总阻力符号为正」，且量级落在冻结参考带内。")
    else:
        add("- 详见 §2 逐门结果；未过之门如实标 ❌，不作放宽。")
    add(
        f"- G1 符号：{gsum['G1_total_drag_positive']}；"
        f"G2 量级：{gsum['G2_magnitude_within_50pct_frozen_ref']}；"
        f"G3 单调：{gsum['G3_monotone_in_tau']}"
    )
    add("")

    add("## 6. 复现命令\n")
    add("```bash")
    add("# 完整阶梯（GPU5，串行五档，约 70-90 min）")
    add("cd /nfs/wangxi/tensor_eco/agent_work/m298/TensorLBM")
    add("CUDA_VISIBLE_DEVICES=5 /nfs/wangxi/venvs/tensorlbm/bin/python \\")
    add("  /nfs/wangxi/runs/tau_ladder_uniform_20261025/run_tau_ladder.py")
    add("")
    add("# 单档（τ=0.51，含锚族探针）")
    add("CUDA_VISIBLE_DEVICES=5 SUBOFF_TAU_TARGET=0.51 \\")
    add("  SUBOFF_ANCHOR_PROBE=/tmp/a.jsonl SUBOFF_ANCHOR_CADENCE=200 \\")
    add("  /nfs/wangxi/venvs/tensorlbm/bin/python \\")
    add("  benchmarks/verified/suboff_sail_resistance/run.py \\")
    add("  --hull-type with_sail --resolution 80 --steps 12000 --device cuda:0 \\")
    add("  --collision mrt --friction standard --p0 near_wall --out /tmp/tauA")
    add("```")
    add("")

    add("## 7. 工件清单\n")
    add(f"- `{VERDICT}`（机器装配判决，本报告全部数字之源）")
    add(f"- `{STAGING}/arms_partial.json`（逐档增量落盘）")
    add(f"- `{STAGING}/driver.log`（驱动全日志）")
    for a in ARMS:
        add(f"- `{STAGING}/tau_{a}/`（result_L{setup['resolution']}.json、anchors.jsonl、run.log）")
    add(
        "- 案例侧代码：`benchmarks/verified/suboff_sail_resistance/run.py`（τ 覆盖 + 探针接入）、"
        "`tau_anchor_probe.py`（只读锚族仪器）、`run_tau_ladder.py`（驱动）"
    )
    add("- 单测：`tests/test_tau_ladder_instrument.py`")
    add("")

    REPORT.write_text("\n".join(L) + "\n")
    print(f"report -> {REPORT} ({len(L)} lines)")
    # keep math import used
    _ = math


if __name__ == "__main__":
    main()
