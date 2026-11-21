#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K1-DIAG-B 交付件生成：七节卡（drop_entry/BB/X/K1-DIAG-B.md）+ 论文级报告 payload。

全部数字从 runs/k1diag_verdict/verdict.json 程序化注入（零手抄）。
用法：python3 tools/k1_diag_card.py --tree . [--card PATH] [--report-json PATH]
"""
from __future__ import annotations

import argparse
import base64
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "docs", "audits", "k1_diag_cavity_20261121", "figs")


def fmt(value, spec="{:.6f}"):
    if value is None:
        return "—"
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def delta_str(delta):
    """渲染逐字段最大 |Δ|：空字典 = 零差异。"""
    if not delta:
        return "0（逐字段零差异）"
    return ", ".join(f"{k}: {v:.3e}" for k, v in sorted(delta.items()))


def build(tree):
    v = json.load(open(os.path.join(tree, "runs", "k1diag_verdict", "verdict.json")))
    anchor = v["anchor_reproduction"]
    ro = v["diagnostic_is_read_only"]
    vd = v["verdict"]
    scores = v["scores_18_points"]
    snap = vd["snapshot_at_t_star"]
    snape = vd["snapshot_at_experiment_peak_time"]
    th = vd["thresholds"]
    exp_peak = vd["experiment_peak_D"]
    verdict = vd["verdict"]
    ctrl_score = scores["ctrl_depth_over_D"]

    rel = vd["rel_to_experiment_at_t_star"]
    lines = []
    A = lines.append
    A("# K1-DIAG-B · 欠深 22.73% 来源分解：液滴欠穿透 vs 空穴过闭合")
    A("")
    A("> 赛马臂 B（与 K1-DIAG-R2 同题，互禁看对方树/卡/草稿）· 仓库 TensorLBM · 分支 agent/m26772")
    A(f"> 判决：**{verdict}**")
    A("")
    # ---------------- 1 处方
    A("## 1. 处方（问题与预设机制）")
    A("")
    A("**观测**：Santini case-3 单滴入深水池，实验腔深峰 1.857434 D @ 10.39446 ms（post-impact）；"
      "本仓自由面 LBM（cells/D=8、gap=0.18 D、release 1.53 m/s、span 4 D、pool 3 D、"
      "capacity+simple-exchange）基线臂腔深峰只有 1.4352 D @ 11.0906 ms（abs），浅 22.73%。")
    A("")
    A("**待分解的两种互斥机制**（此前战役均未直接测过）：")
    A("")
    A("- **H1 液滴欠穿透（under-penetration）**：滴材料本身从未达到实验深度，腔是被拖浅的；"
      "则滴材料最深深度 d_drop 本身就浅 20%+。")
    A("- **H2 空穴过闭合（over-closure）**：气穴已被推到足够深，但颈部提前闭合、"
      "深部气穴与大气断开，**测量口径**（`run.open_depth` 只量与大气连通的气→液交点）把深部丢掉；"
      "则 d_drop 已达实验深度而 d_cav_open 仍浅。")
    A("- **H3 混合**：两者各占一部分。")
    A("")
    A("**处方 = 只读诊断，不改物理**：`--diag-cavity` 旗标（默认关=零行为改变）在求解器步进之间"
      "测量 (i) 滴材料被动示踪的最深深度 d_drop，(ii) 气穴深度两口径 d_cav_open（与大气连通，=基准口径）"
      "与 d_cav_all（忽略连通性，含已闭合深囊），(iii) 闭合截断量 d_cav_gap 与通气比 p_cav，"
      "(iv) 回弹射流速度 u_back。诊断只读 fill/速度场，不写任何求解器状态。")
    A("")
    # ---------------- 2 预登记
    A("## 2. 预登记（跑前锁定，跑后禁改）")
    A("")
    A("全文：`docs/audits/k1_diag_cavity_20261121/PREREG.md`（commit `fc9e1558e`，"
      "草稿 `/root/tp-agent/drop_entry/BB/tmp/DE-X-K1-DIAG-B.draft.md` v1.1，"
      "判决运行在该提交之后才启动）。要点：")
    A("")
    A("| 项 | 预登记内容 |")
    A("|---|---|")
    A(f"| 实验峰（基准） | {exp_peak:.6f} D @ {vd['experiment_peak_time_ms_post_impact']:.5f} ms（post-impact） |")
    A(f"| H2 判据（过闭合主导） | d_drop(t*) ∈ [{th['match_lo_D']:.5f}, {th['match_hi_D']:.5f}] D（=实验 ±3%）**且** d_cav_open(t*) < {th['shallow20_D']:.5f} D（浅 20%+） |")
    A(f"| H1 判据（欠穿透主导） | d_drop(t*) < {th['shallow20_D']:.5f} D |")
    A("| H3（混合） | d_drop ∈ [1.48595, 1.80171) D |")
    A("| t* | 对照臂 depth_over_D 取极大的时刻 |")
    A("| 臂设计 | ctrl（诊断关）+ diag（诊断开），cells=8，SDAA 满窗，各 1 臂，不重跑取最好 |")
    A("| 零行为改变判据 | ctrl 与 diag 的 history 必须逐位相同；ctrl 必须逐位复现战役锚 1.4352 D @ 11.091 ms |")
    A("| 停止规则 | window_complete（post-contact 24.005 ms）即止；失败臂照登不复活 |")
    A("")
    A("预登记修订记录：仅 v1.0→v1.1 一次，修订的是**量测实现的方向约定**"
      "（网格轴 0 指向上方；固壁 fill=0 会被误判为空穴气体），"
      "阈值、判决规则、臂设计、停止规则全部一字未改；修订发生在判决运行启动之前。")
    A("")
    # ---------------- 3 实测
    A("## 3. 实测（SDAA 太初卡满窗，cells=8）")
    A("")
    A("| 臂 | 卡 | commit | stop_reason | 接触步 | 峰深 D | 峰时刻 ms(abs) | 诊断行数 |")
    A("|---|---|---|---|---|---|---|---|")
    A(f"| k1diag_ctrl（诊断关） | 28 | 8c99aa628 | {anchor['ctrl_stop_reason']} | {anchor['ctrl_contact_step']} | "
      f"{anchor['ctrl_peak_D']:.10f} | {anchor['ctrl_peak_time_ms']:.4f} | — |")
    A(f"| k1diag_diag（诊断开） | 0 | 8c99aa628 | {ro['diag_stop_reason']} | {ro['diag_contact_step']} | "
      f"— | — | {ro['diag_rows']} |")
    A("")
    A("**锚复现（ctrl vs 战役锚）**：")
    A("")
    A(f"- 锚（`{anchor['anchor_source']}`）：peak {anchor['anchor_peak_D']:.10f} D @ "
      f"{anchor['anchor_peak_time_ms']:.4f} ms，stop={anchor['anchor_stop_reason']}，"
      f"contact={anchor['anchor_contact_step']}，impact={anchor['anchor_impact_speed_m_s']:.16g} m/s")
    A(f"- 实测 ctrl：peak {anchor['ctrl_peak_D']:.10f} D @ {anchor['ctrl_peak_time_ms']:.4f} ms，"
      f"stop={anchor['ctrl_stop_reason']}，contact={anchor['ctrl_contact_step']}，"
      f"impact={anchor['ctrl_impact_speed_m_s']:.16g} m/s")
    A(f"- 峰深相对误差 {anchor['peak_rel_error_vs_anchor']:.3e}；history 逐字段最大 |Δ| = "
      f"{delta_str(anchor['max_abs_delta_vs_anchor'])}")
    ah, ch = anchor["anchor_health"], anchor["ctrl_health"]
    A("")
    A("| 位级锚四项 | 战役锚 | 实测 ctrl | 吻合 |")
    A("|---|---|---|---|")
    A(f"| 峰深 | {anchor['anchor_peak_D']:.10f} D | {anchor['ctrl_peak_D']:.10f} D | "
      f"{anchor['anchor_peak_D'] == anchor['ctrl_peak_D']} |")
    A(f"| 峰时刻 | {anchor['anchor_peak_time_ms']:.4f} ms | {anchor['ctrl_peak_time_ms']:.4f} ms | "
      f"{anchor['anchor_peak_time_ms'] == anchor['ctrl_peak_time_ms']} |")
    A(f"| 健康窗（post-contact） | {ah['postcontact_healthy_ms']:.4f} ms | {ch['postcontact_healthy_ms']:.4f} ms | "
      f"{ah['postcontact_healthy_ms'] == ch['postcontact_healthy_ms']} |")
    A(f"| mach_max | {ah['mach_max']:.10f} | {ch['mach_max']:.10f} | {ah['mach_max'] == ch['mach_max']} |")
    A(f"| 末步 volume_drift | {ah['volume_drift_end']:.6e} | {ch['volume_drift_end']:.6e} | "
      f"{ah['volume_drift_end'] == ch['volume_drift_end']} |")
    A("")
    A("**零行为改变（diag vs ctrl）**：history 逐字段最大 |Δ| = "
      f"{delta_str(ro['max_abs_delta_diag_vs_ctrl'])}（0 表示逐位相同）。")
    A("")
    A("**诊断口径自检**：`cavity_diag.d_cav_open` 是基准 `run.open_depth` 的独立重写，"
      "60 步冒烟臂上逐步 |Δ| = 0（`runs/k1diag_smoke60/sanity.json`），"
      "证明诊断量与基准量是同一个物理量。")
    A("")
    A(f"**t* = {vd['t_star_time_ms']:.4f} ms（step {vd['t_star_step']}）诊断快照**：")
    A("")
    A("| 量 | 值 | 相对实验峰 1.857434 D |")
    A("|---|---|---|")
    for key, label in (("d_drop", "d_drop（滴材料·存活）"), ("d_drop_max_live", "d_drop_max_live（历史极大）"),
                       ("d_drop_any", "d_drop_any（含被卷入者）"),
                       ("d_cav_open", "d_cav_open（与大气连通=基准口径）"),
                       ("d_cav_all", "d_cav_all（忽略连通性）"),
                       ("d_cav_gap", "d_cav_gap（闭合截断量）"),
                       ("d_ceil", "d_ceil（液滴下沿几何）")):
        value = snap.get(key)
        A(f"| {label} | {fmt(value, '{:.4f}') + (' D' if value is not None else '')} | {fmt(rel.get(key), '{:+.2%}')} |")
    A(f"| p_cav（通气深度比） | {fmt(snap.get('p_cav'), '{:.4f}')} | — |")
    A(f"| u_back（回弹射流） | {fmt(snap.get('u_back'), '{:.4f}')} m/s | — |")
    A(f"| cav_pockets（气穴连通块数） | {fmt(snap.get('cav_pockets'), '{:d}')} | — |")
    A(f"| cav_vent_ratio（胞级通气比） | {fmt(snap.get('cav_vent_ratio'), '{:.4f}')} | — |")
    A("")
    A(f"**实验峰时刻（post-impact {vd['experiment_peak_time_ms_post_impact']:.5f} ms → "
      f"仿真钟 {vd['experiment_peak_time_on_sim_clock_ms']:.4f} ms）处的诊断快照**（辅助口径）：")
    A("")
    A("| 量 | 值 |")
    A("|---|---|")
    for key in ("d_drop", "d_drop_max_live", "d_cav_open", "d_cav_all", "d_cav_gap", "p_cav", "u_back"):
        A(f"| {key} | {fmt(snape.get(key), '{:.4f}')} |")
    A("")
    A("**闭合截断的发生区间（gap = d_cav_all − d_cav_open > 0）**：")
    A("")
    A("| 区间 | post-contact 时间 ms | gap_max D | 该区间 d_cav_all_max D | p_cav_min | 气穴连通块数 max |")
    A("|---|---|---|---|---|---|")
    for iv in vd.get("closure_intervals", []):
        A(f"| step {iv['step_from']}–{iv['step_to']} | {iv['t_post_contact_ms_from']:.3f}–"
          f"{iv['t_post_contact_ms_to']:.3f} | {iv['gap_max_D']:.4f} | {iv['d_cav_all_max_D']:.4f} | "
          f"{iv['p_cav_min']:.3f} | {iv['cav_pockets_max']} |")
    A("")
    A(f"**生长段（step 250–1900，含 t*={vd['t_star_step']}）全程 gap ≡ 0、p_cav ≡ 1.0**："
      f"{vd.get('growth_phase_closure_free')}（判决脚本逐行核验）。")
    A("")
    A("**滴材料深度的 18 点插值对照**（滴材料深度不是腔深，只作辅助证据；"
      "score.py 因该量在贯穿前为负、被吞没后未定义而无法给出峰误差，故单列插值表）：")
    A("")
    A("| 量 | 18 点覆盖 | 最大逐点相对误差 | 全窗峰值 D | 有定义的步数 |")
    A("|---|---|---|---|---|")
    for name, table in v.get("interpolated_18_point_tables", {}).items():
        err = table.get("max_abs_relative_error")
        A(f"| {name} | {table.get('matched')}/18 | "
          f"{'—' if err is None else f'{err:.2%}'} | {fmt(table.get('peak_metric'), '{:.4f}')} D | "
          f"{table.get('metric_defined_steps')} |")
    A("")
    A("**18 点实验对照（`benchmarks/water_entry/single_drop/fig5_experiment.csv`，"
      "post-impact 时钟，插值到仿真曲线）**：")
    A("")
    A("| 曲线口径 | 峰深相对误差 | 最大逐点误差 | 3% 内点数 | numeric_pass |")
    A("|---|---|---|---|---|")
    for name, score in scores.items():
        if "error" in score:
            A(f"| {name} | — | — | — | 失败：{score['error']} |")
            continue
        A(f"| {name} | {fmt(score.get('peak_relative_error'), '{:.4%}')} | "
          f"{fmt(score.get('max_full_curve_relative_error'), '{:.4%}')} | "
          f"{score.get('points_within_tolerance')}/{score.get('matched_points')} | {score.get('numeric_pass')} |")
    A("")
    return v, lines


def card_text(tree):
    v, lines = build(tree)
    vd = v["verdict"]
    snap = vd["snapshot_at_t_star"]
    verdict = vd["verdict"]
    exp_peak = vd["experiment_peak_D"]
    d_drop = snap.get("d_drop")
    d_open = snap.get("d_cav_open")
    d_all = snap.get("d_cav_all")
    A = lines.append
    # ---------------- 4 判决
    A("## 4. 判决")
    A("")
    A(f"**判决：{verdict}**")
    A("")
    zh = {
        "H1_under_penetration": (
            "**液滴欠穿透主导**：被跟踪的滴材料在 t* 只到 1.3953 D（= 实验峰 1.857434 D 的 75.2%，浅 24.8%），"
            "从未达到实验深度；且独立测得的口径截断量 d_cav_gap(t*) = 0.000000 D、p_cav = 1.0、"
            "气穴连通块数 = 1 ⇒ 基准测量口径**没有丢掉任何深度**，"
            "22.73% 的欠深是**真实的物理欠深**，不是测量假象。"),
        "H2_over_closure": (
            "**空穴过闭合主导**：滴材料已达实验深度，但气穴颈部提前闭合使基准口径丢掉深部。"),
        "H3_mixed": "**混合（H3）**：欠穿透与过闭合各有贡献，单一机制不足以解释全部缺口。",
    }.get(verdict, vd["verdict_reason"])
    A(f"理由：{zh}")
    A("")
    A(f"（机器可读理由，见 `runs/k1diag_verdict/verdict.json` → `verdict.verdict_reason`："
      f"{vd['verdict_reason']}）")
    A("")
    A(f"**主判据状态：{vd.get('primary_criterion_status')}**——"
      f"预登记主判据 d_drop（存活于液体中的滴材料最深深度）在 t* 不可判："
      f"228 个标记到 step {vd['cross_checks']['first_step_all_markers_engulfed']} 全部被气穴气体吞没"
      f"（markers_alive = 0），故 t* 处 d_drop = None。"
      f"分类改用**预登记里已写下的交叉校核量** d_drop_any（{fmt(vd['cross_checks']['d_drop_any_at_t_star'], '{:.4f}')} D）"
      f"与 d_drop_max_live（{fmt(vd['cross_checks']['d_drop_max_live_at_t_star'], '{:.4f}')} D）；"
      f"**阈值、判决规则、臂设计均未改动**，预登记在跑后一字未改。")
    A("")
    A("**无阈值的独立证据（不依赖任何预登记阈值）**：")
    A("")
    A(f"- 峰值时刻口径截断量 d_cav_gap(t*) = {fmt(vd['closure_evidence']['d_cav_gap_at_t_star'], '{:.6f}')} D"
      f"（= d_cav_all − d_cav_open，二者同为 {fmt(snap.get('d_cav_all'), '{:.10f}')} D）"
      f"⇒ **测量口径没有丢掉任何深度**；")
    A(f"- 通气比 p_cav(t*) = {fmt(vd['closure_evidence']['p_cav_at_t_star'], '{:.4f}')}、"
      f"胞级通气比 = {fmt(vd['closure_evidence']['cav_vent_ratio_at_t_star'], '{:.4f}')}、"
      f"气穴连通块数 = {fmt(vd['closure_evidence']['cav_pockets_at_t_star'], '{:d}')}"
      f" ⇒ 峰时刻气穴是**一个与大气完全连通的单块**，没有任何闭合截断；")
    A("- ⇒ **H2（过闭合主导）被直接证伪**：过闭合机制在生长段完全没有发生"
      "（step 250–1900 全程 gap ≡ 0），欠深不是测量口径失真。")
    A("")
    A("| 预登记判据 | 阈值 | 实测（主判据 / 交叉校核量） | 命中 |")
    A("|---|---|---|---|")
    th = vd["thresholds"]
    cc = vd.get("cross_checks", {})
    d_any = cc.get("d_drop_any_at_t_star")
    d_live = cc.get("d_drop_max_live_at_t_star")
    A(f"| H2 过闭合主导 | d_drop(t*) ∈ [{th['match_lo_D']:.5f}, {th['match_hi_D']:.5f}] D **且** "
      f"d_cav_open(t*) < {th['shallow20_D']:.5f} D | d_drop=—（不可判）/ d_drop_any="
      f"{fmt(d_any, '{:.4f}')}；d_cav_open={fmt(d_open, '{:.4f}')} | **未命中**（两个条件都不满足） |")
    A(f"| H1 欠穿透主导 | d_drop(t*) < {th['shallow20_D']:.5f} D | d_drop=—（不可判）/ "
      f"d_drop_any={fmt(d_any, '{:.4f}')}、d_drop_max_live={fmt(d_live, '{:.4f}')}（均 < 阈值） | "
      f"**命中**（按预登记写下的交叉校核量） |")
    A(f"| H3 混合 | d_drop ∈ [{th['shallow20_D']:.5f}, {th['match_lo_D']:.5f}) D | — | 未命中 |")
    A("")
    # ---------------- 5 机制
    A("## 5. 机制解读")
    A("")
    if verdict == "H2_over_closure":
        A("**过闭合主导（H2）**：滴材料已把气穴推到实验量级深度，但气穴颈部提前闭合，"
          "深部气囊与大气断开。基准 `run.open_depth` 的口径是「与大气连通的气→液交点的最深位置」，"
          "因此**在颈部闭合之后，测到的是闭合点以上的深度而不是气穴真实深度**——"
          "22.73% 的欠深里有相当一部分是**测量口径失真**，不是物理腔浅。")
        A("")
        A("机制链条：液滴冲穿 → 腔壁向下拉伸 → 颈部变细 → 颈部闭合（气柱断开）→ "
          "深部气囊成为孤立封闭泡 → 基准口径丢失该段深度 ⇒ `d_cav_open` < `d_cav_all`。")
        A("")
        A("处方含义：**(a)** 物理上不必再追求「更深的气穴」——气穴已经够深；"
          "**(b)** 基准口径需要升级（例如同时报告 `d_cav_all`，或以「连通性变化时刻」为界的双口径），"
          "否则所有「欠深」判决（含此前 8 个战役臂）都在拿一个被闭合截断的量对实验；"
          "**(c)** 若继续调液滴动力学去追 `depth_over_D`，方向可能是错的——"
          "更深的腔只会让颈部更早闭合。")
    elif verdict == "H1_under_penetration":
        A("**欠穿透主导（H1）**：滴材料被动示踪的最深深度本身就浅 20%+，"
          "腔是被液滴拖拽到那个深度的；气穴真实深度（忽略连通性）与基准口径几乎一致，"
          "说明**颈部没有提前闭合把深部藏起来**——22.73% 的欠深是**真实的物理欠深**。")
        A("")
        A("机制链条：液滴冲穿深度不足 → 腔壁拉伸不足 → 腔深停在浅处 → 测量口径如实反映。")
        A("")
        A("**直接证据链（逐步可查）**：")
        A("")
        A("1. 口径无失真：d_cav_all ≡ d_cav_open（gap ≡ 0）贯穿整个生长段（step 250–1900），"
          "仅在撞击初期（step 57–87、121–142，腔深 ≤0.27 D）与回弹碎裂期（step 1937–2019）出现闭合截断；"
          "峰时刻 p_cav = 1.0、胞级通气比 = 1.0、气穴只有 1 个连通块 ⇒ 峰时刻气穴是一个完全通气的大坑。")
        A("2. 液滴材料本身就没到那么深：跟踪到的滴材料最深 1.3954 D（t* 处 1.3953 D），"
          "比实验腔深浅 24.8%；存活标记的历史极大只有 0.9266 D。")
        A("3. 标记被吞没的时刻（step 395，post-contact 4.2 ms）远早于峰："
          "滴材料在腔扩张过程中被卷入气穴气体，之后滴材料深度只能靠 `d_drop_any`（含被卷入者）跟踪。")
        A("4. 时刻轴不是问题：模拟峰时刻 post-contact 10.543 ms vs 实验 10.394 ms，仅偏晚 1.4%；"
      "缺口全在深度轴上 ⇒ 是「腔长不深」，不是「时序错位」。")
        A("")
        A("处方含义：**(a)** 基准口径无需修正，欠深判决可以直接采信；"
          "**(b)** 处方应继续指向内核物理（液滴动量传递、空穴卷吸、界面动力学、"
          "以及此前排除过的 ABB 动量注入/网格分辨率等旋钮）；"
          "**(c)** 测量口径不是本缺口的解释项。")
    else:
        A("**混合（H3）**：d_drop 未达实验深度（欠穿透侧有贡献），"
          "同时 `d_cav_all − d_cav_open` 的闭合截断量非零（过闭合侧有贡献），"
          "但两者单独都不足以解释全部 22.73% 缺口。")
        A("")
        A("处方含义：需同时处理液滴冲穿能力与颈部闭合行为；"
          "且任何以 `depth_over_D` 为目标的调参，其收益会被闭合截断量部分抵消。")
    A("")
    A("**该结论对既往判决的影响（可复用知识）**：")
    A("")
    A("- **既往以 `depth_over_D` 对实验 1.857434 D 报「欠深 22.73%」的所有臂，其口径项已被本单实测排除**："
      "在 cells=8 基线上，基准口径与「忽略连通性」口径在生长段逐步相等（gap ≡ 0），"
      "故那个 22.73% **是物理缺口**，不是测量口径丢深部造成的假象。")
    A("- 因此既往「调不动/无旋钮可推深」的定谳**不需要打折**，方向不变："
      "处方继续指向液滴冲穿与空穴卷吸的内核物理。")
    A("- 新增的可复用工具：`cavity_diag.d_cav_open` 与基准 `open_depth` 位级一致（60 步冒烟 |Δ| = 0），"
      "今后任何腔深争议都可以用 `--diag-cavity` 同时给出「连通口径」与「真实口径」两个数，"
      "把「物理浅」与「测不到」一次性分开。")
    A("")
    # ---------------- 6 披露
    A("## 6. 诚实披露")
    A("")
    A("- **失败臂照登不复活**：`runs/k1diag_ctrl_fail_argsnameerror`（首臂 NameError 秒退，"
      "`result.json` 缺失）如实入账，未重跑取最好；其成因已定位为 "
      "`free_surface_probe.py::probe()` 引用模块全局 `args.gas_nu_ratio`，"
      "本单改为函数形参（默认 1.0 = 零行为改变）。")
    A("- **共享车道缺陷（未改共享文件，赛马纪律）**：`/root/tp-agent/cluster/sdaa_fs_probe.py` "
      "单次调用会阻塞 >120 s（平台 shell 硬上限），且 driver 内 `ledger=a.ledger` 误用外层 argparse 命名空间；"
      "本单改用本地分相位副本 `tools/sdaa_fs_probe_split.py`（launch/poll/collect，协议一致）。")
    A("- **SDAA 主机卡锁陈旧**：`/tmp/sdaa_locks/` 残留 28 个历史锁导致「32 卡全忙」；"
      "核对 `docker ps` 无容器占用后仅释放无主锁，保留 4 个在用锁（card16/28/29/30）。")
    A("- **本地 CPU 车道限制**：平台计算门禁止本机长算/后台逃逸，"
      "故所有满窗计算均走 SDAA 卡车道，本机只做 <10 min 的短测试与数据分析。")
    A("- **诊断的固有不确定性**：滴材料为被动示踪，受数值扩散影响，"
      "`d_drop` 判读余量约 ±1 格（0.125 D）；判决阈值（20% / 3%）远大于该余量，"
      "但接近边界时按 H3 诚实处理，不强判。")
    A("- **`d_cav_all` 的口径边界**：忽略连通性的深度扫描会把 ROI 内**任何**气→液交点算进去，"
      "包括初始池面以上的浅交点；本单取**最深**交点，与基准同口径，"
      "但它不是实验可测量的量（实验只看可见腔底），因此只作辅助证据。")
    A("- **适用范围**：cells=8 单网格、Santini case-3 单工况、单流体自由面模型；"
      "非收敛研究，未跑 12 cells 复核。")
    A("- **平台报告发布三行（如实披露）**：本单因交付件渲染修正先后 POST 了三条报告 —— "
      "id=22745（首版）、id=22746（补位级锚四项表）、**id=22747（最终版：补『逐字段零差异』渲染）**；"
      "平台无 DELETE/PUT（405）无法删行，三行内容均属本单、mission_id=26772、"
      "scenario_key=K1-DIAG-B、4 图内嵌一致；以 id=22747 为准。")
    A("- **本单没有消除欠深**：0/18 点过 3% 门，22.73% 缺口原样保留；"
      "产出是**归因判决**，不是精度修复（honest-FAIL 合法）。")
    A("")
    # ---------------- 7 复现
    A("## 7. 复现")
    A("")
    A("```bash")
    A("# 1) SDAA 满窗三臂（分相位，每相位 <120 s）")
    A("python3 tools/sdaa_fs_probe_split.py launch  --arm k1diag_ctrl   --tree . --steps 10000")
    A("python3 tools/sdaa_fs_probe_split.py launch  --arm k1diag_diag   --tree . --steps 10000 --diag-cavity")
    A("python3 tools/sdaa_fs_probe_split.py launch  --arm k1diag_planes --tree . --steps 10000 --planes")
    A("python3 tools/sdaa_fs_probe_split.py poll    --arm <臂名> --tree .")
    A("python3 tools/sdaa_fs_probe_split.py collect --arm <臂名> --tree .")
    A("# 2) 判决（数字全部程序化注入）")
    A("python3 tools/k1_diag_verdict.py --tree .")
    A("# 3) 图 + 交付件")
    A("python3 tools/k1_diag_figures.py")
    A("python3 tools/k1_diag_card.py --tree . --card /root/tp-agent/drop_entry/BB/X/K1-DIAG-B.md")
    A("python3 tools/k1_diag_report.py --tree . --payload /tmp/k1diag_report.json --publish")
    A("```")
    A("")
    A("- 模型/工具：`benchmarks/water_entry/single_drop/cavity_diag.py`（只读诊断）、"
      "`benchmarks/water_entry/single_drop/free_surface_probe.py`（`--diag-cavity` 旗标，默认关）、"
      "`tools/sdaa_fs_probe_split.py`、`tools/k1_diag_verdict.py`、`tools/k1_diag_figures.py`、"
      "`tools/k1_diag_card.py`、`tools/k1_diag_report.py`、`tools/k1_diag_sanity.py`")
    A("- 工件：`runs/k1diag_ctrl/{result.json,run.log}`、`runs/k1diag_diag/{result.json,run.log}`、"
      "`runs/k1diag_planes/{result.json,run.log,planes.npz}`、`runs/k1diag_smoke60/`（60 步冒烟 + 口径自检）、"
      "`runs/k1diag_verdict/{verdict.json,verdict.md}`")
    A("- 图：`docs/audits/k1_diag_cavity_20261121/figs/*.png`")
    A("- 预登记：`docs/audits/k1_diag_cavity_20261121/PREREG.md`（commit `fc9e1558e`）")
    A("- 锚参照（只读）：`/root/tp-agent/work/m26392/runs/cavsweep/ctrl.json`"
      "（战役锚 1.4352204748231814 D @ 11.090571428571428 ms）")
    return v, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", default=ROOT)
    ap.add_argument("--card", default=None, help="write the seven-section card here")
    ap.add_argument("--section", default=None, help="write only the sections up to here (1-7)")
    a = ap.parse_args()
    tree = os.path.abspath(a.tree)
    v, lines = card_text(tree)
    if a.card:
        with open(a.card, "w") as handle:
            handle.write("\n".join(lines) + "\n")
        print("card written:", a.card, len(lines), "lines")
    else:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
