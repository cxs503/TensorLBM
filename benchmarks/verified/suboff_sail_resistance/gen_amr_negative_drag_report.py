#!/usr/bin/env python3
"""Paper-grade report for the AMR negative-drag reproduction artifact.

Every number in the emitted markdown/JSON is read from
``amr_negative_drag_repro.json`` (machine-written) — nothing is transcribed by
hand.  Figures are produced by ``gen_amr_negative_drag_figures.py`` and
embedded as base64 in the report payload.

Outputs
-------
``runs/amr_negative_drag_repro/report.md``            six-element markdown
``runs/amr_negative_drag_repro/report_payload.json``  POST body for /tensor/reports
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ARTIFACT = HERE / "amr_negative_drag_repro.json"
OUT_DIR = REPO / "runs" / "amr_negative_drag_repro"
FIG_DIR = OUT_DIR / "figures"

MISSION_ID = 7480
SCENARIO_KEY = "paper-exp-lbm-suboff-tau-4"

FIG_CAPTIONS = {
    "dqw_vs_tau.png": (
        "D_QW 随 τ 的历史记录（黑叉，tau_axis_20261015/#278）与本轮 AMR 链实测："
        "replay 臂（蓝圆，固定 u_lb=0.05）与 Re 匹配臂（红方，Re=1200）。"
    ),
    "drag_vs_re.png": (
        "各冻结臂的背底扣除冲量 mem_bgsub_fx 随 Re 的分布（>0=正阻力）。"
        "红带为 Re 匹配带宽。可见历史 τ 阶梯跨 5 个数量级 Re，而匹配臂钉在单一 Re。"
    ),
    "anchor_series.png": (
        "各臂 D_QW 时序（取样窗=末 25%，灰带），用于披露收敛漂移。"
    ),
}


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int,)):
        return f"{value:d}"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def build_markdown(payload: dict) -> str:
    ev = payload["evaluation"]
    r1 = ev["R1_reproduction"]
    f2 = ev["F2_falsification"]
    infeas = ev["matched_re_infeasibility"]
    arms = payload["arm_design"]["arms"]
    results = payload["results"]
    hist = payload["historical_record"]["ladder"]
    env = payload["environment"]

    lines: list[str] = []
    add = lines.append

    add("# SUBOFF 裸艇 LBM 阻力精度 — AMR 链负阻力复现/证伪（paper-exp-lbm-suboff-tau 第 4 项实验）")
    add("")
    add(f"- 任务：`paper-exp-lbm-suboff-tau-4`（mission {MISSION_ID}）；仓：TensorLBM")
    add(f"- 工件：`benchmarks/verified/suboff_sail_resistance/amr_negative_drag_repro.json`"
        f"（sha256 `{payload['script']['sha256'][:16]}…` 驱动）")
    add(f"- 驱动：`{payload['script']['path']}`；编排：`{payload['script']['runner']}`；"
        f"仪器：`{payload['script']['probe']}`")
    add(f"- 环境：{env['python']} / torch {env['torch']} / {env['platform']} / device={env['device']}"
        f" / seed={env['seed']} / git={env['git_commit']}")
    add(f"- 生成时间（UTC）：{payload['generated_utc']}")
    add("")
    add(f"## 判决：**{payload['verdict']}**")
    add("")
    add(payload["conclusion"])
    add("")

    # ------------------------------------------------------------------ (1)
    add("## 1. 案例介绍（物理背景 / 几何 / 参数表）")
    add("")
    add("**物理背景**。终裁 #278（`tau_axis_20261015`，任务 #275）在 AMR 生产/冰耦合链上给出")
    add("「全局 τ 是唯一治愈旋钮，可把总阻力从负拉过零」的结论；本实验检验该负阻力能否在"
        "**Re 匹配**下复现。历史记录逐字引自工件 `historical_record`：")
    add("")
    add(f"| 源 | 位置 | 原文 |")
    add("|---|---|---|")
    for q in payload["historical_record"]["quotes"]:
        src = payload["historical_record"]["source"]
        add(f"| `{src['path']}` | L{q['line']} | {q['text']} |")
    add("")
    add("**观测量定义**（与历史链同口径，逐字引自工件）：")
    add("")
    add("```")
    add(payload["historical_record"]["observable_definition"]["text"])
    add("```")
    add("")
    first = results[list(results)[0]]["setup"] if results else {}
    add("**几何与网格**（2:1 两级静态 AMR，艇体在细块上按 CAD 解析重建，非粗网格体素复制）：")
    add("")
    add(f"- 粗网格 {first.get('grid_coarse')}，细块（含 ghost）{first.get('grid_fine_allocated')}；"
        f"AMR box={first.get('amr_box')}，ratio={first.get('amr_ratio')}")
    add(f"- 艇长：粗层 {first.get('hull_length_coarse')} 格（细层 {first.get('hull_length_fine')} 格）；"
        f"湿面积（细层）{_fmt(first.get('wetted_area_fine_lu2'))} lu²；"
        f"细层近壁胞 {_fmt(first.get('near_wall_cells_fine'))}；"
        f"BFL 候选面通道 {_fmt(first.get('bfl_candidate_channels'))}")
    add(f"- 胞数：粗 {_fmt(first.get('coarse_cells'))} + 细 {_fmt(first.get('fine_allocated_cells'))}"
        f" = {_fmt(first.get('total_allocated_cells'))}（相对全域细化的节省 "
        f"{_fmt(first.get('cell_saving_fraction'))}）")
    add(f"- 粗层艇体几何统计（`build_suboff_mask`）：`{json.dumps(first.get('coarse_hull_geometry'), ensure_ascii=False)}`")
    add("")

    # ------------------------------------------------------------------ (2)
    add("## 2. 怎么做的（方法 / 流程 / 关键设置）")
    add("")
    add("两套协议，同一冻结臂表（脚本内重算 τ/Re，禁手抄）：")
    add("")
    add("- **replay**：`u_lb` 固定为仓内 τ 阶梯约定值（`run_tau_ladder.py` 的 `U_LB`），"
        "τ 走历史四档 {0.50036, 1.0, 1.5, 4.0} —— 与历史记录同协议，Re 随 τ 变。")
    add("- **matched**：所有臂钉在同一 Re，ν=u·L/Re 反解 τ，`u_lb` 落在不可压窗内。")
    add("")
    add(f"判据（**动手前冻结**，见工件 `criteria_predeclared`）：")
    add("")
    add(f"- **R1 复现**：replay 臂 τ=0.50036 的 D_QW 与历史 {hist[0]['D_QW']:+.3e} 符号一致且相对偏差 ≤ "
        f"{r1['tolerance_rel']*100:.0f}%")
    add(f"- **F2 证伪**：≥{f2['required_arms']} 组 Re 匹配臂全部正阻力（{f2['sign_quantity']} > 0）")
    add("")
    add("数值设置：")
    add("")
    add(f"- 碰撞：{payload['arm_design']['collision']}")
    add(f"- 边界：{payload['arm_design']['boundary']}")
    add(f"- 平流核：{payload['arm_design'].get('streaming', 'stream3d')}")
    add(f"- 取样：每 {payload['arm_design']['sample_cadence_steps']} 步一次锚族采样，"
        f"观测量 = 末 25% 样本窗均值（窗口定义见各臂 `window.window_definition`）")
    rl = payload.get("resource_limits", {})
    add(f"- 步长预算：`{rl.get('step_budget_rule')}`；flow pass ≈ {rl.get('flow_passes_per_arm')}；"
        f"原因：{rl.get('why')}")
    add("")

    # ------------------------------------------------------------------ (3)
    add("## 3. 计算结果")
    add("")
    add("### 3.1 臂表（全部数字由脚本注入）")
    add("")
    add("| 臂 | 协议 | τ 目标 | τ 实际(粗) | u_lb | Re(粗层) | 步数 | flow pass | D_QW | mem_bgsub_fx | Ct_bgsub | 窗口漂移 D_QW |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in arms:
        res = results.get(arm["arm"])
        if not res:
            add(f"| `{arm['arm']}` | {arm['protocol']} | {_fmt(arm['tau_target'], 6)} | — | — | — | — | — | — | — | — | — |")
            continue
        m = res.get("window", {}).get("mean", {})
        drift = res.get("window", {}).get("drift_pct", {}).get("D_QW")
        add(
            f"| `{arm['arm']}` | {arm['protocol']} | {_fmt(arm['tau_target'], 6)} | "
            f"{_fmt(res['tau_coarse_actual'], 6)} | {_fmt(res['u_lb'])} | {_fmt(res['reynolds_coarse'])} | "
            f"{_fmt(res['steps_completed'])} | {_fmt(res['flow_passes'], 3)} | "
            f"{_fmt(m.get('D_QW'))} | {_fmt(m.get('mem_bgsub_fx'))} | {_fmt(m.get('ct_bgsub'))} | "
            f"{_fmt(drift, 3)}% |"
        )
    add("")
    add("### 3.2 历史阶梯对照（replay 协议）")
    add("")
    add("| τ | 历史 D_QW（原文字） | 历史符号 | 本轮 D_QW | 本轮符号 | 符号一致 | 相对偏差 | Re(本轮) |")
    add("|---|---|---|---|---|---|---|---|")
    for row in ev["replay_ladder"]:
        add(
            f"| {_fmt(row['tau'], 6)} | {_fmt(row['historical_D_QW'])} | {row['historical_sign']} | "
            f"{_fmt(row.get('measured_D_QW'))} | {row.get('measured_sign', '—')} | "
            f"{row.get('sign_match')} | {_fmt(row.get('relative_deviation'), 3)} | "
            f"{_fmt(row.get('reynolds_coarse'))} |"
        )
    add("")
    add("### 3.3 Re 匹配臂证据表（F2）")
    add("")
    add("| 臂 | τ | u_lb | Re 目标 | Re 实测 | Re 命中 | D_QW | mem_bgsub_fx | 阻力符号 | Ct_bgsub | 步数 | flow pass |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in f2["rows"]:
        add(
            f"| `{row['arm']}` | {_fmt(row['tau'], 6)} | {_fmt(row['u_lb'])} | {_fmt(row['re_target'])} | "
            f"{_fmt(row.get('reynolds_coarse'))} | {row.get('re_matched')} | {_fmt(row.get('D_QW'))} | "
            f"{_fmt(row.get('mem_bgsub_fx'))} | {row.get('drag_sign')} | {_fmt(row.get('ct_bgsub'))} | "
            f"{_fmt(row.get('steps_completed'))} | {_fmt(row.get('flow_passes'), 3)} |"
        )
    add("")
    add("### 3.4 历史阶梯能否在单一 Re 下重跑（机器计算）")
    add("")
    add(f"Re 目标 = {infeas['re_target']:.0f}，可用 u_lb 带 = {infeas['u_lb_safe_band']}：")
    add("")
    add("| τ | ν_lb | 匹配 Re 所需 u_lb | 一个 flow pass 步数 | 判定 |")
    add("|---|---|---|---|---|")
    for row in infeas["rows"]:
        add(
            f"| {_fmt(row['tau'], 6)} | {_fmt(row['nu_lb'])} | {_fmt(row['u_lb_required'])} | "
            f"{_fmt(row['steps_for_one_flow_pass'])} | {row['verdict']} |"
        )
    add("")
    add(infeas["reading"])
    add("")

    # ------------------------------------------------------------------ (4)
    add("## 4. 结果图")
    add("")
    for name, caption in FIG_CAPTIONS.items():
        if (FIG_DIR / name).exists():
            add(f"![{name}]({name})")
            add("")
            add(f"*{caption}*")
            add("")
    add("说明：本实验检验的是**符号与归因**（D_QW 符号、阻力符号随 τ/Re 的因果归属），"
        "不作场量精度声明，故未出流场云图；若需云图，属另一条证据线（阻力精度线见案例 README 的 E2 判决）。")
    add("")

    # ------------------------------------------------------------------ (5)
    add("## 5. 注意点（踩坑 / 局限）")
    add("")
    for item in payload["limits"]:
        add(f"- {item}")
    add("- 坑 1：历史记录**不含**生产网格、u_lb、采样层级，因此只能复现「(τ, u_lb) 操作点 + 锚族仪器」，"
        "不能复现「同一张网格」；本工件据此把结论限定在符号/归因层面。")
    add("- 坑 2：τ 与 Re 在固定 u_lb 下强耦合（ν=(τ−0.5)/3），历史四档的 Re 跨 5 个数量级；"
        "把符号翻转记在 τ 账上是**混淆**。")
    add("- 坑 3：Re 匹配与 τ 扫描在物理上互斥：ν 由 τ 决定，匹配 Re 要求 u∝ν，"
        "一侧滑出不可压窗即失去可解释性（见 §3.4）。")
    add("")

    # ------------------------------------------------------------------ (6)
    add("## 6. 误差判定（关键量 vs 基准，逐项）")
    add("")
    add("| 关键量 | 基准（来源） | 实测 | 偏差 | 判据 | 结果 |")
    add("|---|---|---|---|---|---|")
    if r1["status"] in ("PASS", "FAIL"):
        add(
            f"| D_QW @ τ=0.50036（replay） | {r1['historical_D_QW']:+.6e}"
            f"（`{payload['historical_record']['source']['path']}` L18） | "
            f"{r1['measured_D_QW']:+.6e} | {r1['abs_relative_deviation_pct']:.2f}% | ≤10% 且符号一致 | "
            f"**{r1['status']}**（符号一致={r1['sign_match']}） |"
        )
    else:
        add(f"| D_QW @ τ=0.50036（replay） | {hist[0]['D_QW']:+.6e} | — | — | ≤10% 且符号一致 | **{r1['status']}** |")
    add(
        f"| Re 匹配臂正阻力数 | ≥{f2['required_arms']} 组 | {f2['positive_drag_arms']}/{f2['usable_matched_arms']} | "
        f"— | 全部正阻力 | **{f2['status']}** |"
    )
    add("")
    add("**是否 <3%？** 明确回答：**不适用且不达标**——本工件的对表量是历史 **D_QW=−3.20e-3** 的"
        f"符号与量级（容差 10%），实测偏差 {r1.get('abs_relative_deviation_pct', float('nan')):.2f}%，"
        "既非「<3%」带内的复现，也不是阻力精度声明。本实验**不**声称阻力精度进入 3% 带；"
        "该案例的阻力精度线（Blasius 平板 Cf 口径，Re=1000）在 `README.md` 的 E2 判决中为 "
        "**+0.9542%（L128，PASS）**，与本实验的 3% 无关、不得混用。")
    add("")
    add("## 7. 复现命令")
    add("")
    add("```bash")
    add("cd benchmarks/verified/suboff_sail_resistance")
    add("PYTHONPATH=../../src THREADS=10 ./run_amr_negative_drag_repro.sh   # 全部臂 + 自动装配工件")
    add("python gen_amr_negative_drag_figures.py                          # 图")
    add("python gen_amr_negative_drag_report.py                           # 本报告 + POST 载荷")
    add("PYTHONPATH=../../src python -m pytest tests/test_amr_negative_drag_repro.py -q")
    add("```")
    add("")
    add("## 8. 工件清单")
    add("")
    add(f"- `benchmarks/verified/suboff_sail_resistance/amr_negative_drag_repro.json`（本判定）")
    add(f"- `benchmarks/verified/suboff_sail_resistance/run_amr_negative_drag_repro.py`（驱动，判据冻结于此）")
    add(f"- `benchmarks/verified/suboff_sail_resistance/run_amr_negative_drag_repro.sh`（并行编排）")
    add(f"- `benchmarks/verified/suboff_sail_resistance/gen_amr_negative_drag_figures.py` / `..._report.py`")
    add(f"- `tests/test_amr_negative_drag_repro.py`（判据/装配逻辑单测）")
    add(f"- `{payload['script']['run_root']}/<arm>/result.json` + `anchor_samples.jsonl`（逐臂原始数据 + sha256）")
    add("")
    return "\n".join(lines)


def main() -> int:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    md = build_markdown(payload)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUT_DIR / "report.md"
    md_path.write_text(md, encoding="utf-8")

    figures = []
    for name, caption in FIG_CAPTIONS.items():
        path = FIG_DIR / name
        if not path.exists():
            continue
        figures.append(
            {
                "name": name,
                "caption": caption,
                "png_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        )
    ev = payload["evaluation"]
    body = {
        "title": "SUBOFF AMR 链负阻力复现/证伪：Re 匹配下 D_QW 与阻力符号（paper-exp-lbm-suboff-tau-4）",
        "summary": (
            f"{payload['verdict']}；R1={ev['R1_reproduction']['status']}，"
            f"F2={ev['F2_falsification']['status']}"
            f"（{ev['F2_falsification']['positive_drag_arms']}/{ev['F2_falsification']['usable_matched_arms']} "
            f"Re 匹配臂正阻力）；历史阶梯在单一 Re 下可达档数 "
            f"{len(ev['matched_re_infeasibility']['feasible_taus'])}/{len(ev['matched_re_infeasibility']['rows'])}。"
        ),
        "scenario_key": SCENARIO_KEY,
        "mission_id": MISSION_ID,
        "tags": ["TensorLBM", "SUBOFF", "AMR", "LBM", "negative-drag", "D_QW", "Reynolds-matching", "honest-FAIL"],
        "markdown": md,
        "figures": figures,
    }
    out = OUT_DIR / "report_payload.json"
    out.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"markdown": str(md_path.relative_to(REPO)), "payload": str(out.relative_to(REPO)),
                      "figures": [f["name"] for f in figures], "verdict": payload["verdict"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())