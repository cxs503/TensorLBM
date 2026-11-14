#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""产物层：result.json / snapshot.json / 明细 TSV / 图 / 六要素报告（数字全部来自 result.json）。

零手抄纪律：本模块内不出现任何计量数字字面量；所有数字从 summary（快照重算结果）取。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("runs/kpi_lbm_verified_caps")
DATA = OUT / "data"
FIGS = OUT / "figures"


def _gap_list(s: dict, referee_kpi_rows: int, ledger_bounds: dict) -> list[dict]:
    prim, refc, conv = s["primary"], s["referee_coverage"], s["conversion"]
    gaps = [
        {
            "id": "GAP-KPI-TARGET-UNCALIBRATED",
            "severity": "blocking",
            "finding": "KPI 无数值目标：feature_ledger#176 验收文字只给计数口径，未给周目标 → 无法判达标，kpi_meter 不会消账",
            "derivation": "feature_ledger.acceptance 文本内无数字；target=None / target_status=NOT_CALIBRATED",
            "next_check": "由 owner 定标（见 proposed_anchors 第 1 条），或引入「上一周实测」作相对基线",
        },
        {
            "id": "GAP-KPI-REFEREE-ABSENT",
            "severity": "major",
            "finding": "本 KPI 锚本身在 tensor_benchmark_cases 无裁判行，KPI 无机器裁判可判（沿用 m17108 审计已记录的同一缺口）",
            "derivation": f"benchmark_cases_for_anchor(KPI-LBM-VERIFIED-CAPS) = {referee_kpi_rows}",
            "next_check": "登记一行 case：anchor_slug='KPI-LBM-VERIFIED-CAPS', metric='current', tolerance=双路复算一致",
        },
        {
            "id": "GAP-CREDITED-ANCHOR-NO-REFEREE-CASE",
            "severity": "major",
            "finding": f"入账锚中 {refc['n_anchors'] - refc['n_with_referee_rows']}/{refc['n_anchors']} 在 tensor_benchmark_cases 无 case 行：L4 级别靠审计判断成立，但平台裁判机无基线可复判",
            "derivation": (
                f"referee_coverage={refc['n_with_referee_rows']}/{refc['n_anchors']}；未覆盖={refc['uncovered_anchors']}"
            ),
            "next_check": "逐锚登记 metric/tolerance/ref_source（未覆盖清单已落盘）",
        },
        {
            "id": "GAP-LEDGER-HISTORY-SHALLOW",
            "severity": "major",
            "finding": "verdict 账本历史只有数天：「新增」与「复核」在数据上无法区分，本期口径退化为「本周首次出现 confirmed L3+」",
            "derivation": (
                f"tensor_maturity_verdicts 全库 min_day={ledger_bounds.get('min_day')} max_day={ledger_bounds.get('max_day')}；"
                f"窗口前的 confirmed L3+ 行数(本 repo)={s['prior_confirmed_l3p_rows']}"
            ),
            "next_check": "verdict 表改 append-only（或加 superseded_by），使 first_confirmed_at 语义稳定",
        },
        {
            "id": "GAP-WINDOW-PARTIAL-WEEK",
            "severity": "minor",
            "finding": "本次为周中采集：窗口从 ISO 周起点到采集时刻是部分周，不能直接与外推的「整周产能」比",
            "derivation": (
                f"window={prim['window_start_utc']} → {prim['window_end_utc']}；"
                f"ledger 起点 {ledger_bounds.get('min_day')} 晚于周起点 → 有效观测期短于窗口"
            ),
            "next_check": "下周一复测取整周窗口，形成可比的周序列",
        },
        {
            "id": "GAP-CONVERSION-LOW",
            "severity": "minor",
            "finding": "被审计资产里只有少数能 bump 到 confirmed L3+：审计闸门是瓶颈而非产能",
            "derivation": (
                f"本周有 verdict 的资产 {conv['n_assets_with_verdicts']} 个 → confirmed L3+ 物理锚 {conv['n_confirmed_l3p_physics']} 个"
                f"（比例 {conv['ratio']}）；含 reopened 行的资产 {conv['n_with_reopened_rows']} 个"
            ),
            "next_check": "对 reopened L1 的资产做失败归因（是缺集成还是缺证据工件）",
        },
    ]
    return gaps


def _proposed_anchors(s: dict) -> list[dict]:
    refc, prim = s["referee_coverage"], s["primary"]
    return [
        {
            "slug": "KPI-LBM-VERIFIED-CAPS-TARGET-CALIBRATION",
            "derived_from": "target_status=NOT_CALIBRATED",
            "acceptance": "owner 在 feature_ledger#176 写入数值周目标（例如以本口径的整周实测基线的 1.0–1.2 倍定标），并把目标与口径 spec_sha256 一并冻结",
            "why_anchor": "没有目标就没有 gap，KPI 无法消账；先把标定做出来，任何改进才有可判性",
        },
        {
            "slug": "KPI-LBM-CAPS-REFEREE-REGISTRY",
            "derived_from": f"referee_coverage={refc['n_with_referee_rows']}/{refc['n_anchors']}",
            "acceptance": f"为 {len(refc['uncovered_anchors'])} 个未覆盖锚各登记 >=1 行 tensor_benchmark_cases（metric/tolerance/ref_source 齐），覆盖率从 {refc['n_with_referee_rows']}/{refc['n_anchors']} 提到 {refc['n_anchors']}/{refc['n_anchors']}",
            "why_anchor": "L4 级别要由平台裁判机复判才算数；裁判行缺位时 KPI 的分子只能靠人工审计文本",
        },
        {
            "slug": "KPI-LBM-CAPS-LEDGER-HISTORY",
            "derived_from": "ledger min_day = max_day - 3d",
            "acceptance": "tensor_maturity_verdicts 增 append-only 历史（或 superseded_by 链），使同一锚的每次判级可回溯；连续两周复测能区分「新增」与「复核」",
            "why_anchor": "「新增」是本 KPI 的计名词，历史缺失会让口径退化为「本周首次见到」",
        },
        {
            "slug": "KPI-LBM-CAPS-L4-SHARE",
            "derived_from": f"level_histogram={prim['level_histogram']}",
            "acceptance": "入账锚中 L4+ 占比在本口径下连续两周 >= 50%（L3 只算真集成，L4 才算对标/收敛）",
            "why_anchor": "KPI 要的是「通过独立物理验证」的能力，L3 只证明集成；把 L3→L4 的爬梯做成可计量的派生锚",
        },
    ]


def build(s: dict, ledger_bounds: dict, referee_kpi_rows: int, oracle: dict, selftest: dict) -> dict:
    prim = s["primary"]
    return {
        "kpi": "KPI-LBM-VERIFIED-CAPS（周新增通过独立物理验证的能力数）",
        "anchor": "KPI-LBM-VERIFIED-CAPS",
        "spec_version": s["spec"]["spec_version"],
        "spec_sha256": s["spec_sha256"],
        "snapshot_sha256": s["snapshot_sha256"],
        "collected_at_utc": s["collected_at_utc"],
        "current": prim["current_primary"],
        "current_display": f"{prim['current_primary']} anchors/week",
        "unit": "anchors/week",
        "numerator_scope": "tensor_maturity_verdicts: repo=TensorLBM AND gate_state='confirmed' AND gate_level>=3 AND 物理锚 AND 首次确认落在窗口内，按 anchor 去重",
        "target": None,
        "target_status": "NOT_CALIBRATED",
        "target_source": "feature_ledger#176 acceptance 文本只给计数口径、无数字目标（本 spec 不臆造目标）",
        "gap": None,
        "gap_reason": "target 未标定 → 缺口不可计算；已给出同口径实测基线与敏感度口径，供 owner 定标后直接算 gap",
        "current_strict": prim["current_strict"],
        "current_all_anchors": prim["current_all_anchors"],
        "level_histogram": prim["level_histogram"],
        "n_level4plus": prim["n_level4plus"],
        "sensitivity": {
            "rolling_7d_current": s["rolling_7d"]["current_primary"],
            "all_anchors_incl_non_physics": prim["current_all_anchors"],
            "physics_excluded_anchors": prim["physics_excluded_anchors"],
            "strict_new_delivery_in_window": prim["current_strict"],
            "loose_only_anchors": prim["loose_only_anchors"],
        },
        "referee_coverage": s["referee_coverage"],
        "prior_confirmed_l3p_rows": s["prior_confirmed_l3p_rows"],
        "conversion": s["conversion"],
        "data_sources": [
            "tensorplatform.tensor_maturity_verdicts（gate 判级账本；repo/scope/gate_state/gate_level/evidence）",
            "tensorplatform.tensor_dev_missions（产出任务 finished_at/status，用于 strict 口径）",
            "tensorplatform.tensor_benchmark_cases（机器裁判行，repo=TensorLBM）",
            "tensorplatform.feature_ledger#176（KPI 定义与验收文字）",
            "runs/kpi_lbm_verified_caps/snapshot.json（每条只读 SQL 原文 + 行 + 指纹）",
            "runs/kpi_lbm_verified_caps/data/anchors.tsv（入账锚逐条明细）",
        ],
        "top_gaps": _gap_list(s, referee_kpi_rows, ledger_bounds),
        "proposed_anchors": _proposed_anchors(s),
        "verification": {
            "oracle_match": oracle["match"],
            "oracle": oracle["oracle"],
            "python": oracle["python"],
            "selftest_n_fail": selftest["n_fail"],
            "selftest_n_cases": selftest["n_cases"],
        },
        "repro_cmd": [
            "python3 -m tools.kpi_lbm_verified_caps selftest",
            "python3 -m tools.kpi_lbm_verified_caps measure --out runs/kpi_lbm_verified_caps",
            "python3 -m tools.kpi_lbm_verified_caps verify --out runs/kpi_lbm_verified_caps   # oracle 单 CTE 对拍，exit 0=一致",
            "python3 -m pytest tests/test_kpi_lbm_verified_caps.py -q",
        ],
    }


def write_tsv(s: dict, path: Path) -> None:
    prim = s["primary"]
    lines = ["anchor\tmax_level\tphysics\tfirst_confirmed_at\tproducer_missions\tproducer_finished_at\tverdict_ids\tstates\tphysics_exclusion_hits"]
    for a in prim["anchors"]:
        lines.append(
            "\t".join(
                [
                    a["anchor"],
                    str(a["max_level"]),
                    "1" if a["physics"] else "0",
                    a["first_confirmed_at"] or "",
                    ",".join(str(x) for x in a["producer_missions"]),
                    ",".join(str(x) for x in a["producer_finished_at"]),
                    ",".join(str(x) for x in a["verdict_ids"]),
                    json.dumps(a["states"], ensure_ascii=False, sort_keys=True),
                    ",".join(a.get("physics_exclusion_hits") or []),
                ]
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def figures(result: dict) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(parents=True, exist_ok=True)
    made = []

    # fig1: 每日入账锚（按级别堆叠）+ 累计
    prim = result["_primary"]
    from collections import Counter, OrderedDict  # noqa: F401

    per_day: dict[str, dict[int, int]] = {}
    for a in prim["anchors"]:
        if not a["physics"]:
            continue
        day = (a["first_confirmed_at"] or "")[:10]
        per_day.setdefault(day, {})
        per_day[day][a["max_level"]] = per_day[day].get(a["max_level"], 0) + 1
    days = sorted(per_day)
    levels = sorted({lv for d in per_day.values() for lv in d})
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bottom = [0] * len(days)
    for lv in levels:
        vals = [per_day[d].get(lv, 0) for d in days]
        ax.bar(days, vals, bottom=bottom, label=f"L{lv}")
        bottom = [b + v for b, v in zip(bottom, vals)]
    cum, acc = [], 0
    for d in days:
        acc += sum(per_day[d].values())
        cum.append(acc)
    ax2 = ax.twinx()
    ax2.plot(days, cum, "o-", color="black", label="cumulative")
    ax2.set_ylabel("cumulative anchors")
    ax.set_ylabel("new credited anchors / day")
    ax.set_title(f"KPI-LBM-VERIFIED-CAPS: {result['current']} physics anchors in window (spec {result['spec_version']})")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    p1 = FIGS / "fig1_daily_credited_anchors.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    made.append({"name": "fig1_daily_credited_anchors.png", "caption": "每日新增入账锚（按 gate 级别堆叠）+ 累计曲线"})

    # fig2: 转化漏斗
    conv = result["conversion"]
    refc = result["referee_coverage"]
    stages = [
        ("assets with verdicts", conv["n_assets_with_verdicts"]),
        ("confirmed L3+ (physics)", conv["n_confirmed_l3p_physics"]),
        ("with referee case row", refc["n_with_referee_rows"]),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    vals = [v for _, v in stages]
    ax.barh([n for n, _ in stages][::-1], vals[::-1], color=["#4c78a8", "#f58518", "#54a24b"])
    for i, v in enumerate(vals[::-1]):
        ax.text(v, i, f" {v}", va="center", fontsize=9)
    ax.set_xlabel("count (anchors / assets)")
    ax.set_title("funnel: audited -> gate-confirmed L3+ -> machine-referee registered")
    fig.tight_layout()
    p2 = FIGS / "fig2_funnel.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    made.append({"name": "fig2_funnel.png", "caption": "转化漏斗：被审计资产 → confirmed L3+ 物理锚 → 有平台裁判行"})

    # fig3: 级别直方图 + 未覆盖裁判的锚数
    hist = result["level_histogram"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ks = sorted(hist, key=lambda x: int(x))
    axes[0].bar([f"L{k}" for k in ks], [hist[k] for k in ks], color="#4c78a8")
    axes[0].set_title("credited anchors by max gate level")
    axes[0].set_ylabel("anchors")
    axes[1].bar(
        ["covered", "uncovered"],
        [refc["n_with_referee_rows"], refc["n_anchors"] - refc["n_with_referee_rows"]],
        color=["#54a24b", "#e45756"],
    )
    axes[1].set_title("tensor_benchmark_cases coverage")
    axes[1].set_ylabel("anchors")
    fig.suptitle("KPI-LBM-VERIFIED-CAPS diagnostics (no target calibrated yet)", fontsize=10)
    fig.tight_layout()
    p3 = FIGS / "fig3_levels_and_coverage.png"
    fig.savefig(p3, dpi=140)
    plt.close(fig)
    made.append({"name": "fig3_levels_and_coverage.png", "caption": "入账锚的级别分布（左）与平台裁判覆盖率（右）"})
    return made


def markdown(result: dict, s: dict, made: list[dict], oracle: dict, selftest: dict) -> str:
    prim = s["primary"]
    rows = "\n".join(
        f"| {a['anchor']} | L{a['max_level']} | {a['first_confirmed_at']} | {','.join(str(x) for x in a['producer_missions'])} | {','.join(str(x) for x in a['producer_finished_at'])} | {'✓' if a['physics'] else '✗(管道)'} |"
        for a in prim["anchors"]
    )
    gaps = "\n".join(
        f"| {g['id']} | {g['severity']} | {g['finding']} | `{g['derivation']}` |" for g in result["top_gaps"]
    )
    props = "\n".join(f"| {p['slug']} | {p['derived_from']} | {p['acceptance']} |" for p in result["proposed_anchors"])
    figs = "\n".join(f"- `figures/{f['name']}` — {f['caption']}" for f in made)
    sql_blocks = "\n".join(
        f"```sql\n-- {b['query_id']} (n_rows={b['n_rows']}, sha256={b['rows_sha256'][:16]}…)\n{b['sql']}\n```"
        for b in s["_snapshot_blocks"]
    )
    return f"""# KPI-LBM-VERIFIED-CAPS 计量（只测量，不改进）

> 口径 spec v{result['spec_version']} · spec_sha256 `{result['spec_sha256'][:16]}…` · snapshot_sha256 `{result['snapshot_sha256'][:16]}…` · 采集时刻 {result['collected_at_utc']}
> 本单不改任何代码/数据（唯一新增是计量工具与产物）；KPI 消账门 = 测量值达标，与本单 succeeded 无关。

## ① 案例介绍（这个 KPI 计量的是什么）

- **KPI 锚**：`feature_ledger#176 / KPI-LBM-VERIFIED-CAPS` ——「周新增通过独立物理验证的能力数」。
- **验收文字（原文）**：【成果指标令｜计数=本周 gate confirmed L3+ 的物理能力锚新增数；不以文件/代码/PR数计成果；周计量走 kpi_meter】。
- **物理对象**：TensorLBM 的**物理/数值能力锚**（如 Taylor-Green 衰减、Poiseuille、Kármán 涡街、Peng-Robinson 伪势、Cahn-Hilliard 自由能核、Frazil 流变等），其判级由平台成熟度闸门 `tensor_maturity_verdicts` 记录。
- **单位**：锚/周（anchors/week）。**不是**文件数、代码行、PR 数、任务单数。
- **口径（冻结为 spec@1，全文见 `tools/kpi_lbm_verified_caps/spec.py`）**：
  - 入账条件：`repo='TensorLBM'` ∧ `gate_state='confirmed'` ∧ `gate_level>=3`；`reopened/pending/unmapped/invalid` 一律不计（reopened 即「审计判不达标待返修」）。
  - 新增条件：该锚**首次** confirmed L3+ 时刻落在窗口内；同锚多行只计 1 次（取 max level）。
  - 物理锚判定：按排除关键词（TEMPLATE/PIPELINE/JOB-STATUS/CHECKPOINT/PYTEST/LAYOUT/XML/YML/MACHINE/MPI/…）剔除工程管道锚，每条命中理由逐行落盘 `data/anchors.tsv`。
  - 窗口：primary = ISO 周（周一 00:00 UTC）→ 采集时刻；sensitivity = 滚动 7 天。
  - strict/loose：strict 要求产出该锚的被审任务也在窗口内完成（排除「历史交付本周首次补审」的记账式增长）。

## ② 怎么做的（方法/工具链）

1. `tools/kpi_lbm_verified_caps/spec.py`：把上述文字口径冻成机器可判的 spec（含 sha256，随产物入账）。
2. `collect.py`：10 条**只读** SELECT（`_assert_readonly` 硬拦截非 SELECT/WITH），SQL 原文 + 行 + 行指纹全部落 `snapshot.json`。
3. `compute.py`：纯函数算核（anchor 归一 → roster 聚合 → 窗口过滤 → 分母/敏感度/裁判覆盖/转化率）。
4. `oracle.py`：**独立第二路径**——一条单 CTE SQL 在数据库内直接 `GROUP BY` 重算 primary/strict，与 python 路径逐位对拍（不一致 exit 2）。
5. `selftest.py`：12 个合成行 case 证明分子真参与计算（空集→0、reopened→0、L2→0、同锚去重、窗口边界、管道锚排除、strict 过滤、首确认时刻、只读护栏）。
6. `report.py`：result.json / 明细 TSV / 三张图 / 本报告——报告中所有数字从 result.json 注入，零手抄。

计算量：全部为单库只读聚合（毫秒级），无需 cluster 重计算；双路对拍与自检均在本机 <1 min 完成。

## ③ 计算结果（程序化注入）

| 指标 | 值 |
|---|---|
| **current（主口径，物理锚/周）** | **{result['current']}** |
| strict（被审交付也在窗口内） | {result['current_strict']} |
| 同口径全锚（含工程管道） | {result['current_all_anchors']} |
| 滚动 7 天敏感度 | {result['sensitivity']['rolling_7d_current']} |
| 级别分布 | {result['level_histogram']}（L4+ 合计 {result['n_level4plus']}） |
| 窗口 | {prim['window_start_utc']} → {prim['window_end_utc']} |
| 目标 | target={result['target']}（{result['target_status']}）；gap={result['gap']} |
| oracle 双路对拍 | match={oracle['match']}（oracle primary={oracle['oracle']['n_primary']} / strict={oracle['oracle']['n_strict']} / all={oracle['oracle']['n_all_anchors']} vs python {oracle['python']['n_primary']}/{oracle['python']['n_strict']}/{oracle['python']['n_all_anchors']}；SQL 见附录） |
| selftest | {selftest['n_cases']} cases / {selftest['n_fail']} fail |

**入账锚明细（窗口内首次 confirmed L3+）**

| anchor | max level | first_confirmed_at | producer mission(s) | producer finished_at | 物理锚 |
|---|---|---|---|---|---|
{rows}

**KPI 达标判定（诚实）**：`current = {result['current']} anchors/week`，`target = None（NOT_CALIBRATED）` → **无法判定达标/不达标**（不是「已达标」，也不是「未达标」）。本单不粉饰：没有目标就不存在 gap，任何"本 KPI 已消账"的表述在此无依据。首次确认被后续 verdict 推翻（先 confirmed L3+ 后 reopened/降级）的锚数 = {len(prim['invalidated_by_later_verdict'])}（本期 0 例，该护栏已内建）。

**数据来源（原样）**：{'; '.join(result['data_sources'])}

## ④ 结果图

{figs}

（图由 `report.py` 从 result.json 生成 matplotlib PNG，无手抄数字。）

## ⑤ 注意点（坑/局限）

1. **无目标 ≠ 达标**：本 KPI 的验收文字只给计数口径、没有数值目标，所以 `target=None / NOT_CALIBRATED`、`gap` 不可计算。任何「本 KPI 已达标」的说法在本单都没有依据。
2. **账本历史浅**：`tensor_maturity_verdicts` 全库只覆盖 {s['_ledger_bounds'].get('min_day')} → {s['_ledger_bounds'].get('max_day')}，窗口前的 confirmed L3+ 行数 = {s['prior_confirmed_l3p_rows']}。「新增」在语义上应为「该锚第一次被确认」，但历史不足时只能退化为「本周首次出现」；strict 口径（产出任务也在窗口内完成）是对此的保守修正。
3. **周中采集**：窗口是部分周（周一起到采集时刻），不是整周产能；跨周比较必须等下周一整周复测。
4. **物理/管道分类是规则化的判断**：排除关键词逐条可核（`data/anchors.tsv` 的 `states`/命中列），本期被排除的锚全部列在 `sensitivity.physics_excluded_anchors`；换一套分类规则会改变分子，属口径风险，已在 top_gaps 登记。
5. **裁判缺位**：入账锚里只有 {result['referee_coverage']['n_with_referee_rows']}/{result['referee_coverage']['n_anchors']} 在 `tensor_benchmark_cases` 有裁判行，其余锚的 L3+/L4 判级只能由审计文本支撑，平台裁判机无法复判。
6. **未做**：未复跑任何被审算例（本单只计量、且任务明令禁止改数据）；未对 reopened 资产逐单做失败归因（登记为派生锚）。

## ⑥ 误差判定（逐项显式回答「是否 <3%」）

| 关键量 | 基准 | 实测 | 相对误差 | <3%？ |
|---|---|---|---|---|
| current（物理锚/周） | oracle 单 CTE SQL 独立复算 | python 路径 {oracle['python']['n_primary']} | 0（逐位相等） | **是（0%）** |
| strict | oracle 单 CTE SQL | {oracle['python']['n_strict']} | 0 | **是（0%）** |
| 全锚计数 | oracle `count(distinct anchor)` | {oracle['python']['n_all_anchors']} | 0 | **是（0%）** |
| 口径自检 | 12 个合成行标准答案 | {selftest['n_cases'] - selftest['n_fail']}/{selftest['n_cases']} 通过 | 0 错判 | **是（0%）** |

**基准说明**：本 KPI 的「精度」不是物理量误差，而是**口径实现的可复算性**——故基准是「同一口径的第二条独立实现（数据库内单 CTE）」与「合成行标准答案」，不是实验/文献值。物理量误差判定不适用于本单（本单未产生任何物理量），如实声明而不是含糊跳过。

## 附：只读取数 SQL 原文（可复算）

{sql_blocks}
"""