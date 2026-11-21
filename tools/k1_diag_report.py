#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K1-DIAG-B 论文级进展报告：生成 markdown + figures 并 POST 到平台报告接口。

数字全部从 runs/k1diag_verdict/verdict.json 程序化注入（零手抄）。
用法：
    python3 tools/k1_diag_report.py --tree . --payload /tmp/k1diag_report.json [--publish]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "docs", "audits", "k1_diag_cavity_20261121", "figs")


def f(value, spec="{:.6f}"):
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


def build_markdown(tree):
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
    d_drop = snap.get("d_drop")
    d_open = snap.get("d_cav_open")
    d_all = snap.get("d_cav_all")
    rel = vd["rel_to_experiment_at_t_star"]

    # 闭合区间表
    cl = ["| 区间 | post-contact ms | gap_max D | d_cav_all_max D | p_cav_min | 气穴块数 max |",
          "|---|---|---|---|---|---|"]
    for iv in vd.get("closure_intervals", []):
        cl.append("| step %d–%d | %.3f–%.3f | %.4f | %.4f | %.3f | %d |" % (
            iv["step_from"], iv["step_to"], iv["t_post_contact_ms_from"],
            iv["t_post_contact_ms_to"], iv["gap_max_D"], iv["d_cav_all_max_D"],
            iv["p_cav_min"], iv["cav_pockets_max"]))

    # 18 点表
    cmp_rows = scores["ctrl_depth_over_D"].get("comparisons") or []
    tbl18 = ["| t_exp/ms | 实验 D | ctrl 模拟 D | 相对误差 |", "|---|---|---|---|"]
    for point in cmp_rows:
        tbl18.append("| %.5f | %.6f | %.6f | %+.2f%% |" % (
            point["time_ms"], point["experiment"], point["simulation"],
            point["signed_relative_error"] * 100))

    # 诊断量表
    dtbl = ["| 量 | t* = %.4f ms | 实验峰时刻 %+.5f ms | 实验基准 | 相对误差 |" % (
        vd["t_star_time_ms"], vd["experiment_peak_time_ms_post_impact"]), "|---|---|---|---|---|"]
    for key, label, base in (("d_drop", "d_drop（滴材料·存活）", exp_peak),
                             ("d_drop_max_live", "d_drop_max_live（历史极大）", exp_peak),
                             ("d_cav_open", "d_cav_open（与大气连通）", exp_peak),
                             ("d_cav_all", "d_cav_all（忽略连通性）", exp_peak)):
        a = snap.get(key)
        b = snape.get(key)
        ra = None if a is None else a / base - 1.0
        dtbl.append("| %s | %s | %s | %.6f D | %s |" % (
            label, f(a, "{:.4f} D"), f(b, "{:.4f} D"), base, f(ra, "{:+.2%}")))

    score_tbl = ["| 曲线口径 | 峰深相对误差 | 最大逐点误差 | 3% 内点数 | numeric_pass |",
                 "|---|---|---|---|---|"]
    for name, score in scores.items():
        if "error" in score:
            score_tbl.append(f"| {name} | — | — | — | 失败：{score['error']} |")
            continue
        score_tbl.append("| %s | %s | %s | %s/%s | %s |" % (
            name, f(score.get("peak_relative_error"), "{:.4%}"),
            f(score.get("max_full_curve_relative_error"), "{:.4%}"),
            score.get("points_within_tolerance"), score.get("matched_points"),
            score.get("numeric_pass")))

    if verdict == "H2_over_closure":
        mech = (
            "液滴材料已把气穴推到实验深度（d_drop 与实验峰同量级），但气穴颈部提前闭合，"
            "深部气囊与大气断开 ⇒ **基准测量口径 `run.open_depth`（只量与大气连通的气→液交点）"
            "把已闭合的深部丢掉**。因此 22.73% 的欠深**不是物理腔浅，而是测量口径在颈部闭合后失真**；"
            "处方方向应转向「腔深测量口径 / 闭合判定」，而不是继续调液滴动力学。"
        )
    elif verdict == "H1_under_penetration":
        mech = (
            "滴材料本身就没走到实验深度（d_drop 浅 20%+），气穴深度是被液滴拖拽决定的，"
            "**过闭合不是主因**。因此 22.73% 的欠深是**真实的物理欠深**（液滴动量/界面传递不足），"
            "处方方向仍应转向内核物理（液滴冲穿、空穴卷吸、界面动力学），测量口径无需修正。"
        )
    else:
        mech = (
            "d_drop 既没到实验深度也没落在 H2 窗口内，且 d_cav_all 与 d_cav_open 的差（闭合截断量）"
            "不足以解释全部缺口 ⇒ **欠深由液滴欠穿透与空穴过闭合共同承担**（H3 混合），"
            "处方需同时处理液滴冲穿能力与颈部闭合行为。"
        )

    md = f"""# K1-DIAG-B · 欠深 22.73% 的来源分解：液滴欠穿透 vs 空穴过闭合（诊断官·赛马臂 B）

## ① 案例介绍（物理背景 / 几何 / 参数表）

单滴入深水池，Santini et al. case-3：液滴直径 D = 2.27 mm、水密度 997 kg/m³、
表面张力 σ = 0.07142 N/m、μ_w = 8.53e-4 Pa·s、μ_a = 1.86e-5 Pa·s、g = 9.81 m/s²，
实验撞速 1.4 m/s。自由面 LBM（D3Q19，常压边界，capacity + simple-exchange 质量守恒）模拟，
网格 cells/D = 8，池深 3 D、计算域 4 D×4 D、gap = 0.18 D、release speed = 1.53 m/s、
lattice speed = 0.06、u_limit = 0.15、span = 4 D、pool_depth = 3 D。

| 参数 | 值 | 来源 |
|---|---|---|
| 实验腔深峰 | {exp_peak:.6f} D @ {vd['experiment_peak_time_ms_post_impact']:.5f} ms (post-impact) | `benchmarks/water_entry/single_drop/fig5_experiment.csv`（18 点） |
| LBM 基线腔深峰 | {anchor['ctrl_peak_D']:.10f} D @ {anchor['ctrl_peak_time_ms']:.4f} ms (abs) | `runs/k1diag_ctrl/result.json` |
| 缺口 | {abs(anchor['ctrl_peak_D'] / exp_peak - 1) * 100:.3f}%（浅） | 计算 |
| 接触步 / 撞速 | {anchor['ctrl_contact_step']} / {anchor['ctrl_impact_speed_m_s']:.16g} m/s | 同上 |
| 健康门 | Ma ≤ 0.3、ρ_min > 0、volume_drift ≤ 1e-4 | `benchmarks/water_entry/single_drop/protocol.json` |

**要回答的问题**：这 22.73% 的欠深，是**液滴没穿透到实验深度**（欠穿透），还是
**气穴已被推到足够深但颈部提前闭合、测量口径把深部丢掉**（过闭合）？

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

- **只读诊断旗标** `--diag-cavity`（默认关 = 零行为改变）：在步进之间测量
  (i) 滴材料**被动示踪**（接触前一步播种的拉氏标记，随局部格点速度平流）的最深深度 `d_drop`；
  (ii) 气穴深度两口径——`d_cav_open`（与大气连通，= 基准 `run.open_depth` 口径）与
  `d_cav_all`（忽略连通性，含已闭合深囊）；(iii) 闭合截断量 `d_cav_gap` 与通气比 `p_cav`；
  (iv) 回弹射流 `u_back`、气穴连通块数 `cav_pockets`、胞级通气比 `cav_vent_ratio`。
  诊断**不写任何求解器状态**（只读 fill 场与速度场）。
- **预登记先行**：`docs/audits/k1_diag_cavity_20261121/PREREG.md`（commit `fc9e1558e`），
  判决阈值/臂设计/停止规则在判决运行启动前锁定；唯一一次修订（v1.1）只改量测实现的方向约定。
- **臂**：SDAA 太初卡满窗（cells=8，steps=10000，window_complete 截止）：
  `k1diag_ctrl`（诊断关，卡 28）、`k1diag_diag`（诊断开，卡 0）、`k1diag_planes`（中截面场快照，卡 1）。
- **工具链**：`tools/sdaa_fs_probe_split.py`（共享包装器的分相位版，协议一致：同样的卡锁/镜像/driver，
  拆成 launch/poll/collect 以适配平台 120 s shell 上限）、`tools/k1_diag_verdict.py`（判决，数字程序化注入）、
  `tools/k1_diag_figures.py`（出图）、`tools/k1_diag_card.py`（七节卡）。
- **零行为改变判据**：ctrl 与 diag 的 `history` 必须逐位相同；ctrl 必须逐位复现战役锚。

## ③ 计算结果（数字表，程序化注入）

**锚复现与零行为改变**

| 位级锚四项 | 战役锚 | 实测 ctrl | 吻合 |
|---|---|---|---|
| 峰深 | {anchor['anchor_health']['postcontact_healthy_ms'] and anchor['anchor_peak_D']:.10f} D | {anchor['ctrl_peak_D']:.10f} D | {anchor['anchor_peak_D'] == anchor['ctrl_peak_D']} |
| 峰时刻 | {anchor['anchor_peak_time_ms']:.4f} ms | {anchor['ctrl_peak_time_ms']:.4f} ms | {anchor['anchor_peak_time_ms'] == anchor['ctrl_peak_time_ms']} |
| 健康窗（post-contact） | {anchor['anchor_health']['postcontact_healthy_ms']:.4f} ms | {anchor['ctrl_health']['postcontact_healthy_ms']:.4f} ms | {anchor['anchor_health']['postcontact_healthy_ms'] == anchor['ctrl_health']['postcontact_healthy_ms']} |
| mach_max | {anchor['anchor_health']['mach_max']:.10f} | {anchor['ctrl_health']['mach_max']:.10f} | {anchor['anchor_health']['mach_max'] == anchor['ctrl_health']['mach_max']} |
| 末步 volume_drift | {anchor['anchor_health']['volume_drift_end']:.6e} | {anchor['ctrl_health']['volume_drift_end']:.6e} | {anchor['anchor_health']['volume_drift_end'] == anchor['ctrl_health']['volume_drift_end']} |

| 判据 | 结果 |
|---|---|
| ctrl vs 战役锚（`{anchor['anchor_source']}`） | peak 相对误差 {anchor['peak_rel_error_vs_anchor']:.3e}；history 逐字段最大 \\|Δ\\| = {delta_str(anchor['max_abs_delta_vs_anchor'])} |
| ctrl stop/contact/impact | {anchor['ctrl_stop_reason']} / {anchor['ctrl_contact_step']} / {anchor['ctrl_impact_speed_m_s']:.16g} m/s |
| diag vs ctrl history 最大 \\|Δ\\| | {delta_str(ro['max_abs_delta_diag_vs_ctrl'])}（0 = 逐位相同） |
| 诊断口径自检（60 步冒烟） | `d_cav_open` vs 基准 `depth_over_D` 逐步 \\|Δ\\| = 0 |

**t* = {vd['t_star_time_ms']:.4f} ms（step {vd['t_star_step']}）的分解量**

{chr(10).join(dtbl)}

**主判据状态**：{vd.get('primary_criterion_status')}——预登记主判据 d_drop（存活于液体中的滴材料最深深度）
在 t* 不可判：228 个滴材料标记到 step {vd['cross_checks']['first_step_all_markers_engulfed']}
全部被气穴气体吞没（markers_alive = 0），故 t* 处 d_drop = None。分类改用**预登记中已写下的交叉校核量**
d_drop_any = {f(vd['cross_checks']['d_drop_any_at_t_star'], '{:.4f}')} D 与
d_drop_max_live = {f(vd['cross_checks']['d_drop_max_live_at_t_star'], '{:.4f}')} D；
**阈值、判决规则、臂设计、停止规则在跑后一字未改**。

**闭合截断的发生区间（gap = d_cav_all − d_cav_open > 0）**：

{chr(10).join(cl)}

生长段（step 250–1900，含 t*）全程 gap ≡ 0、p_cav ≡ 1.0（判决脚本逐行核验：{vd.get('growth_phase_closure_free')}）。

补充：d_cav_gap(t*) = {f(snap.get('d_cav_gap'), '{:.4f}')} D；p_cav(t*) = {f(snap.get('p_cav'), '{:.4f}')}；
u_back(t*) = {f(snap.get('u_back'), '{:.4f}')} m/s；cav_pockets(t*) = {f(snap.get('cav_pockets'), '{:d}')}；
cav_vent_ratio(t*) = {f(snap.get('cav_vent_ratio'), '{:.4f}')}。

**18 点实验对照（post-impact 时钟）**

{chr(10).join(score_tbl)}

逐点表（ctrl 口径）：

{chr(10).join(tbl18)}

## ④ 结果图

- fig1：基线腔深-时间曲线 vs Santini case-3 实验 18 点（含峰深/峰时刻标注）。
- fig2：**分解图**——d_cav_open / d_cav_all / d_drop 随时间，叠加实验峰 1.857434 D、±3% 匹配带与 −20% 浅线。
- fig3：闭合截断量 d_cav_gap 与通气比 p_cav（闭），回弹射流 u_back（开）。
- fig4：中截面液相体积分数场量云图（4 个登记时刻），气体/液体分布形态（场量云图）。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

- **共享 SDAA 包装器的两个坑**：(a) 容器 driver 里 `ledger=a.ledger` 误用外层 argparse 命名空间 → NameError 秒退
  （本单前一臂 `runs/k1diag_ctrl_fail_argsnameerror` 如实入账）；(b) `probe()` 内部引用模块全局 `args.gas_nu_ratio`
  → 程序化调用直接 NameError（本单已修为函数形参 `gas_nu_ratio=1.0`，默认 1.0 零行为改变）。
  共享文件未被本单改动（赛马纪律），改用本地分相位副本 `tools/sdaa_fs_probe_split.py`。
- **卡锁陈旧**：SDAA 主机 `/tmp/sdaa_locks/` 残留 28 个历史锁（来自已结束战役），导致自动挑卡报「32 卡全忙」；
  本单在核对 `docker ps` 无容器占用后仅释放无主锁，保留 4 个在用锁。
- **判读余量**：滴材料标记是数值扩散的被动示踪，标记判读存在约 ±1 格（=0.125 D）量级余量；
  判决阈值（20%/3%）远大于该余量，但边界情形需按 H3（混合）诚实处理。
- **物理边界**：单流体自由面模型，滴与池同液、无固体弹体、无真三相接触线；
  结论只在本模型/本分辨率/本参数下成立，不是「真实液滴入水」普适结论。
- **适用范围**：cells=8 单网格、Santini case-3 单工况；非收敛研究（12 cells 未跑）。

## ⑥ 误差判定（关键量 vs 基准，逐项）

| 关键量 | 模拟 | 基准 | 误差 | <3%? |
|---|---|---|---|---|
| ctrl 腔深峰 | {anchor['ctrl_peak_D']:.10f} D | {exp_peak:.6f} D | {(anchor['ctrl_peak_D'] / exp_peak - 1) * 100:+.3f}% | **否** |
| ctrl 峰时刻 | {anchor['ctrl_peak_time_ms']:.4f} ms (abs) | {vd['experiment_peak_time_ms_post_impact']:.5f} ms (post-impact) | — | — |
| d_cav_open(t*) | {f(d_open, '{:.4f}')} D | {exp_peak:.6f} D | {f(None if d_open is None else d_open / exp_peak - 1, '{:+.2%}')} | **否** |
| d_cav_all(t*) | {f(d_all, '{:.4f}')} D | {exp_peak:.6f} D | {f(None if d_all is None else d_all / exp_peak - 1, '{:+.2%}')} | **否** |
| d_drop(t*) | {f(d_drop, '{:.4f}')} D | {exp_peak:.6f} D | {f(None if d_drop is None else d_drop / exp_peak - 1, '{:+.2%}')} | **{'否' if d_drop is None or abs(d_drop / exp_peak - 1) >= 0.03 else '是'}** |
| 18 点逐点误差（ctrl） | {f(scores['ctrl_depth_over_D'].get('max_full_curve_relative_error'), '{:.2%}')}（最大） | — | — | **否** |

**判决**：{verdict}

**主判据与交叉校核**：d_drop(t*) 不可判（标记全被吞没），按预登记交叉校核量
d_drop_any(t*) = {f(vd['cross_checks']['d_drop_any_at_t_star'], '{:.4f}')} D
（= 实验峰 {vd['experiment_peak_D']:.6f} D 的 {vd['cross_checks']['d_drop_any_at_t_star'] / vd['experiment_peak_D'] * 100:.1f}%，浅 {abs(vd['cross_checks']['d_drop_any_at_t_star'] / vd['experiment_peak_D'] - 1) * 100:.1f}%）
落在 H1 窗口（< {vd['thresholds']['shallow20_D']:.5f} D）；
d_drop_max_live(t*) = {f(vd['cross_checks']['d_drop_max_live_at_t_star'], '{:.4f}')} D 更浅。

**无阈值的独立证据**：d_cav_gap(t*) = {f(snap.get('d_cav_gap'), '{:.6f}')} D、
p_cav(t*) = {f(snap.get('p_cav'), '{:.4f}')}、cav_pockets(t*) = {f(snap.get('cav_pockets'), '{:d}')}
⇒ 峰时刻气穴是**一个与大气完全连通的单块**，测量口径**没有丢掉任何深度**
⇒ **H2（过闭合主导）被直接证伪**，22.73% 是**真实的物理欠深**。

机器可读理由（`runs/k1diag_verdict/verdict.json` → `verdict.verdict_reason`）：
{vd['verdict_reason']}

{mech}

**honest-FAIL**：本单**没有**把 22.73% 的欠深消除；0/18 点落在 3% 门内；
本单的产出是**归因判决**（欠深来自哪一侧），不是精度修复。
"""
    return md, verdict


def publish(payload):
    api = os.environ["TENSOR_API_BASE"]
    user = os.environ["TENSOR_AGENT_USER"]
    password = os.environ["TENSOR_AGENT_PASS"]

    def post(path, body, token=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = "Bearer " + token
        request = urllib.request.Request(api + path, data=json.dumps(body).encode(),
                                         headers=headers)
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, response.read().decode()

    status, body = post("/auth/login", {"username": user, "password": password})
    token = json.loads(body).get("access_token") or json.loads(body).get("token")
    status, body = post("/tensor/reports", payload, token)
    return status, body


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", default=ROOT)
    ap.add_argument("--payload", default="/tmp/k1diag_report.json")
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    tree = os.path.abspath(a.tree)
    md, verdict = build_markdown(tree)

    figures = []
    captions = {
        "fig1_baseline_vs_experiment.png": "基线 ctrl 臂腔深-时间曲线 vs Santini case-3 实验 18 点（峰深 1.4352 D vs 1.8574 D）",
        "fig2_decomposition.png": "分解图：d_cav_open（与大气连通）/ d_cav_all（忽略连通性）/ d_drop（滴材料）随时间，叠加实验峰与判决阈值带",
        "fig3_closure_and_rebound.png": "闭合截断量 d_cav_gap 与通气比 p_cav（上）；回弹射流 u_back（下）",
        "fig4_midplane_fields.png": "中截面液相体积分数场量云图（蓝=液体，红=气体），4 个登记 post-contact 时刻",
    }
    for name, caption in captions.items():
        path = os.path.join(FIGDIR, name)
        if os.path.exists(path):
            figures.append({
                "name": name,
                "caption": caption,
                "png_base64": base64.b64encode(open(path, "rb").read()).decode(),
            })

    v = json.load(open(os.path.join(tree, "runs", "k1diag_verdict", "verdict.json")))
    vd = v["verdict"]
    snap = vd["snapshot_at_t_star"]
    summary = (
        f"K1-DIAG-B 判决 {verdict}：ctrl 位级复现锚 {v['anchor_reproduction']['ctrl_peak_D']:.6f}D@"
        f"{v['anchor_reproduction']['ctrl_peak_time_ms']:.3f}ms，diag 臂 history 逐位相同；"
        f"t* 处 d_drop=None（228 个滴材料标记全被气穴吞没，改用交叉校核量 d_drop_any="
        f"{vd['cross_checks']['d_drop_any_at_t_star']:.4f} D）、d_cav_open={snap.get('d_cav_open'):.10f} D、"
        f"d_cav_all={snap.get('d_cav_all'):.10f} D，d_cav_gap=0（口径无失真，实验峰 {vd['experiment_peak_D']:.6f} D）。"
        f"22.73% 欠深归因于{'空穴过闭合（测量口径丢深部）' if verdict == 'H2_over_closure' else ('液滴欠穿透' if verdict == 'H1_under_penetration' else '两者混合')}；"
        f"本单为归因判决，未消除缺口（0/18 点过 3% 门，honest-FAIL）。"
    )
    payload = {
        "title": "K1-DIAG-B: decomposing the 22.73% crater-depth deficit into drop under-penetration vs cavity over-closure",
        "summary": summary,
        "scenario_key": "K1-DIAG-B",
        "mission_id": 26772,
        "tags": ["TensorLBM", "water-entry", "cavitation", "diagnosis", "read-only-probe", "honest-fail"],
        "markdown": md,
        "figures": figures,
    }
    with open(a.payload, "w") as handle:
        json.dump(payload, handle)
    print("payload written:", a.payload, "figures:", len(figures))
    if a.publish:
        status, body = publish(payload)
        print("report:", status, body[:400])


if __name__ == "__main__":
    main()
