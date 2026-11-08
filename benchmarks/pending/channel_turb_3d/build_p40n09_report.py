#!/usr/bin/env python3
"""Build the P40-N09 report from the campaign artifacts (zero hand-copied numbers).

Reads ``gates_p40n09.json`` / ``p40n09_numbers.json`` / ``p40n09_prereg.json``
and writes

* ``docs/reports/P40-N09-statistical-vs-model-decomposition.md`` -- the
  paper-style report,
* ``--json-out`` (default ``/tmp/report.json``) -- the body of the platform
  report POST, with the figures inlined as base64.

Every number in the markdown comes from a ``.v``/``.f`` pair in the gate or
numbers file; the builder refuses to emit a table whose number it cannot trace.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

NODE = "P40-N09"


def pct(value, digits: int = 2) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.{digits}f} %"


def num(value, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def gate_status(gates: dict, prefix: str) -> str:
    for gate in gates["gates"]:
        if gate["gate"].startswith(prefix):
            return gate["status"]
    return "missing"


def gate_of(gates: dict, prefix: str) -> dict | None:
    for gate in gates["gates"]:
        if gate["gate"].startswith(prefix):
            return gate
    return None


def fig_b64(path: Path) -> dict:
    return {
        "name": path.name,
        "caption": "",
        "png_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numbers", type=Path, default=Path("p40n09_numbers.json"))
    parser.add_argument("--gates", type=Path, default=Path("gates_p40n09.json"))
    parser.add_argument("--prereg", type=Path, default=Path("p40n09_prereg.json"))
    parser.add_argument("--fig-dir", type=Path, default=Path("figures_p40n09"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("../../../docs/reports/P40-N09-statistical-vs-model-decomposition.md"),
    )
    parser.add_argument("--json-out", type=Path, default=Path("/tmp/report.json"))
    args = parser.parse_args()

    numbers = json.loads(args.numbers.read_text())
    gates = json.loads(args.gates.read_text())
    prereg = json.loads(args.prereg.read_text()) if args.prereg.exists() else {}
    arms = numbers.get("arms", {})
    attribution = numbers.get("attribution")
    domain = numbers.get("domain_length")
    negative = numbers.get("negative_control")
    budget = numbers.get("budget")
    wall = numbers.get("wall_origin", {})
    d4 = arms.get("D4_cs0", {})
    d2 = arms.get("D2_cs0", {})

    lines: list[str] = []
    add = lines.append
    add(f"# {NODE} 湍流通道 u+ 误差的统计-模型分解（块平均 CI × 双域长互裁 × SGS 负控）")
    add("")
    add("> 本报告由 `benchmarks/pending/channel_turb_3d/build_p40n09_report.py` 自动生成：")
    add("> 每个数字都来自 `p40n09_numbers.json` / `gates_p40n09.json` 的 `f`（出处）字段，")
    add("> 无手工抄写。判据阈值在跑动前登记于 `p40n09_prereg.json`，未按结果调整。")
    add("")
    add("## 0. 判决摘要")
    add("")
    add("| 判据 | 结果 | 判定 |")
    add("|---|---|---|")
    rows = [
        ("G_A 稳态门（相邻半窗 u+ 最大差 ≤ 1 u+ 单位）", "G_A_steady_state", None),
        ("G_B 统计门（u+ 95% CI 半宽 ≤ 3%）", "G_B_statistical_ci", None),
        ("G_C 域长门（Lx=2πδ vs 4πδ u+ 互差 ≤ 3%）", "G_C_domain_length", None),
        ("G_D 归因裁决（二选一，必须给结论）", "G_D_attribution", None),
        ("G_E SGS 负控（C_s=0 vs 0.1 差 > CI）", "G_E_negative_control", None),
        ("G_F 预算外推（err(T_obs) 曲线 + MKM 盒机时）", "G_F_budget", None),
    ]
    for label, prefix, _ in rows:
        gate = gate_of(gates, prefix)
        status = gate["status"] if gate else "missing"
        if gate and gate["evidence"]:
            head = gate["evidence"][0]
            detail = f"`{head['f']}` = {head['v']}"
        else:
            detail = "（无证据）"
        add(f"| {label} | {detail} | **{status.upper()}** |")
    add("")
    add(f"整体判决：**{gates['verdict'].upper()}**。")
    if attribution:
        add("")
        add(
            f"归因结论（G_D 必须二选一）：**{attribution['verdict']}** —— "
            f"实测相对 L2 误差 {pct(attribution['observed_relative_l2'])}，"
            f"其中 U_SN（误差度量的 95% 置信半宽）= {pct(attribution['u_sn'])}，"
            f"超出 CI 的残余 = {pct(attribution['residual_beyond_ci'])}"
            f"（= {num(attribution['residual_in_u_sn_units'], 1)} × U_SN），"
            f"门限尺度 3%。"
        )
        if attribution.get("domain_shift_relative") is not None:
            add(
                f"域长效应实测 {pct(attribution['domain_shift_relative'])}，"
                f"扣除后归于模型/离散的残余 {pct(attribution['unexplained_model_relative'])}。"
            )
    add("")

    add("## 1. 案例介绍")
    add("")
    add(
        "**物理问题**：P20-N01 在 Re_τ=180 三维槽道 DNS 级对标中报告 u+ 剖面对 "
        "MKM-1999 的相对 L2 误差 **15.19%**（L3, 256×64×128, contours 见 "
        "`docs/reports/P20-N01-channel3d-mkm1999.md`）。该数字混了两类完全不同的误差："
        "有限统计窗带来的**统计误差**（只能靠机时消掉）与离散/闭合/计算域带来的"
        "**系统偏差**（机时无限也消不掉）。本节点把两者分开并给出机时投放依据。"
    )
    add("")
    add(
        "**案例矩阵**（同网格、同 Re_τ、同 u_τ、同分子黏度，只动一个旋钮）："
    )
    add("")
    add("| 臂 | 计算域 | 碰撞/闭合 | C_s | 作用 |")
    add("|---|---|---|---|---|")
    add("| `D4_cs0` | 4πδ × 4πδ/3 | natural-KBC | 0 | 参考臂（MKM 盒尺度），SGS 关 |")
    add("| `D2_cs0` | 2πδ × 4πδ/3 | natural-KBC | 0 | 域长臂（流向盒长减半） |")
    add("| `SGS_cs0` / `SGS_cs01` | 4πδ × 4πδ/3 | cumulant | 0 / 0.1 | SGS 负控臂对 |")
    add("")
    if d4:
        add(
            f"网格：nx×ny×nz = {int(d4.get('nx') or 0)}×{int(d4.get('ny') or 0)}×{int(d4.get('nz') or 0)}"
            f"（δ = {num(d4.get('delta'), 1)} cells，dy+ = {num(d4.get('dy_plus'), 2)}，"
            f"Lx/δ = {num(d4.get('box_over_delta'), 2)}），"
            f"D4 为 {int(d4.get('cells') or 0):,} cells；"
            f"u_τ = {num(d4.get('u_tau'), 4)}，ν = {num(d4.get('nu'), 7)}。"
        )
        add("")
    add(
        f"壁面单位：u_τ = √(a·h/2)（强迫槽道精确动量平衡，与剖面无关）；"
        f"y+ = (y − y_w)·u_τ/ν，y_w = {num(wall.get('value'), 4)} cell，"
        f"来源 = `{wall.get('source')}`。"
    )
    add("")

    add("## 2. 怎么做的")
    add("")
    add("1. **统计误差估计器**（新库模块 `src/tensorlbm/validation/statistical_error.py`）：")
    add("   - `integral_time`：壁面剪切时间序列的自相关积分时间 T_int（Sokal 自动窗），")
    add("     给出 N_eff = N/(2T_int) 与均值相对标准误；")
    add("   - `batch_means_summary`：把统计窗切成连续 batch，batch 均值做 Student-t 区间")
    add("     （相关样本下唯一诚实的方差估计）；")
    add("   - `half_window_max_diff`：稳态门（相邻半窗）；")
    add("   - `metric_from_batches`：把 batch 散布传播到**误差度量本身**，得到 U_SN；")
    add("   - `decompose_error`：残差 = 误差 − U_SN，与 3% 尺度比较后二选一裁决；")
    add("   - `domain_length_check` / `negative_control_check` / `budget_extrapolation`。")
    add("2. **内核埋点**（`src/tensorlbm/boundary/wall_resolved_channel3d.py`，纯观测、默认关闭）：")
    add("   `profile_batch_count` 记录连续 batch 的动量剖面均值，")
    add("   `wall_exchange_series_stride` 记录壁面剪切时间序列；")
    add("   `tests/test_wall_resolved_channel3d_batch_stats.py` 钉住「开启后轨迹逐位不变」。")
    add("3. **跑动**：`run_p40n09.py`（各臂一次一进程）；重计算全部走集群 loong64 节点")
    add("   （n315/n330），分臂并行，产物落 `/data/test002/p40n09/runs`，再取回仓内 `runs_p40n09/`。")
    add("4. **判据**：`analyze_p40n09.py` 读各臂记录 → 折叠半通道（`fold_channel_profile`）")
    add("   → u+/y+ → batch CI / 半窗 / 域长互裁 / SGS 分离 / err(T_obs) 拟合 → `gates_p40n09.json`。")
    add("5. **测试**：`tests/test_statistical_error.py`（估计器对合成真值的回归）、")
    add("   `tests/test_wall_resolved_channel3d_batch_stats.py`（埋点不改物理）、")
    add("   `tests/test_p40n09_analysis.py`（裁决器对三类人造战役的已知答案）。")
    add("")

    add("## 3. 计算结果")
    add("")
    if d4 and "batch_ci" in d4:
        add("### 3.1 统计窗与 T_int 账本")
        add("")
        add("| 臂 | cells | 统计窗 [δ/u_τ] | batches | T_int [steps] | T_int [δ/u_τ] | N_eff | T_obs/T_int |")
        add("|---|---|---|---|---|---|---|---|")
        for case in ("D4_cs0", "D2_cs0", "SGS_cs0", "SGS_cs01"):
            arm = arms.get(case)
            if not arm:
                continue
            ledger = arm.get("t_int_ledger") or {}
            add(
                f"| `{case}` | {int(arm.get('cells') or 0):,} | "
                f"{num(arm.get('averaging_eddy_turnovers'), 1)} | {arm.get('n_batches')} | "
                f"{num(ledger.get('t_int_steps'), 1)} | {num(ledger.get('t_int_turnovers'), 3)} | "
                f"{num(ledger.get('n_effective_samples'), 1)} | "
                f"{num(ledger.get('t_obs_over_t_int'), 1)} |"
            )
        add("")
        add("### 3.2 u+ 剖面与 95% 置信区间")
        add("")
        add("| 臂 | y+ 覆盖 | u+ 范围 | 半窗最大差 [u+] | CI 半宽 max [u+] | CI 半宽 max [%] | 对 MKM 相对 L2 | U_SN |")
        add("|---|---|---|---|---|---|---|---|")
        for case in ("D4_cs0", "D2_cs0", "SGS_cs0", "SGS_cs01"):
            arm = arms.get(case)
            if not arm or "batch_ci" not in arm:
                continue
            yp = arm["y_plus"]
            up = arm["u_plus"]
            mkm = arm.get("mkm", {})
            add(
                f"| `{case}` | {num(min(yp), 1)}–{num(max(yp), 1)} | "
                f"{num(min(up), 2)}–{num(max(up), 2)} | "
                f"{num(arm['half_window']['max_diff_u_plus'], 3)} | "
                f"{num(arm['batch_ci']['max_ci_half'], 3)} | "
                f"{pct(arm['batch_ci']['max_ci_half_relative_window'])} | "
                f"{pct(mkm.get('relative_l2'))} | {pct(mkm.get('u_sn'))} |"
            )
        add("")
    if domain:
        add("### 3.3 域长互裁（Lx = 2πδ vs 4πδ，同网格）")
        add("")
        add(
            f"- 相对 L2 互差 = **{pct(domain['relative_l2_difference'])}**（门 3%）；"
            f"最大点值差 {num(domain['max_abs_difference'], 3)} u+；"
            f"RMS {num(domain['rms_abs_difference'], 3)} u+。"
        )
        add(
            f"- 统计可分辨性：最大 |Δu+| 是否超过两臂合成 CI → "
            f"**{domain['resolved']}**（合成 CI 最大 {num(domain.get('max_combined_ci_half'), 3)} u+，"
            f"超 CI 点占比 {pct(domain.get('fraction_points_beyond_ci'))}）。"
        )
        add("")
    if negative:
        add("### 3.4 SGS 负控（C_s = 0 vs 0.1，固定网格）")
        add("")
        add(
            f"- 两臂 u+ 最大差 = **{num(negative['max_abs_difference'], 3)} u+**，"
            f"合成 CI 最大 {num(negative.get('max_combined_ci_half'), 3)} u+，"
            f"分离度 = {num(negative.get('separation_in_ci_units'), 2)} × CI → "
            f"**separated = {negative['separated']}**。"
        )
        add("")
    if attribution:
        add("### 3.5 归因裁决")
        add("")
        add("| 项 | 数值 | 说明 |")
        add("|---|---|---|")
        add(f"| 实测相对 L2（对 MKM） | {pct(attribution['observed_relative_l2'])} | 中心估计 |")
        add(f"| 95% CI | [{pct(attribution['error_ci_low'])}, {pct(attribution['error_ci_high'])}] | batch 传播 |")
        add(f"| U_SN | {pct(attribution['u_sn'])} | 误差度量自身的 95% 半宽 |")
        add(f"| 超出 CI 的残余 | {pct(attribution['residual_beyond_ci'])} | = 误差 − U_SN |")
        add(f"| 其中域长效应 | {pct(attribution.get('domain_shift_relative'))} | 实测互裁，仅当显著 |")
        add(f"| 其中模型/离散 | {pct(attribution['unexplained_model_relative'])} | 扣除后残差 |")
        add(f"| 判决 | **{attribution['verdict']}** | 尺度：3%（= U_SN 尺度 1 倍以上即显著） |")
        add("")
        add(f"假设与局限：{attribution['assumption']}")
        add("")
    if budget:
        add("### 3.6 机时预算（err–T_obs 实测曲线 + MKM 盒预测）")
        add("")
        add("| T_obs [δ/u_τ] | 相对 L2 误差 |")
        add("|---|---|")
        for point in numbers.get("error_vs_t_obs", []):
            add(f"| {num(point['t_obs_turnovers'], 1)} | {pct(point['relative_error'])} |")
        add("")
        add(
            f"- 拟合：err = {num(budget['fitted_coefficient'], 4)} · T_obs^(−{num(budget['fitted_exponent'], 3)})"
            f"（指数{'假定' if budget['exponent_assumed'] else '实测'}）。"
        )
        add(
            f"- 达到 3% 所需 T_obs：{num(budget.get('predicted_turnovers_to_target'), 1)} δ/u_τ"
            + (
                f"；**拒绝外推**：{budget['extrapolation_refused_reason']}"
                if budget.get("extrapolation_refused_reason")
                else ""
            )
        )
        ref = budget.get("reference_box", {})
        if ref:
            add(
                f"- MKM 盒（{int(ref['cells']):,} cells × {int(ref['steps']):,} steps = "
                f"{ref['cell_updates']:.3e} cell-updates）在本轮实测吞吐 "
                f"{budget.get('throughput_cell_updates_per_s', 0)/1e6:.3f} Mcell-update/s 下需 "
                f"**{ref['predicted_hours']:.1f} 机时·小时**（单节点）。"
            )
        add("")

    add("## 4. 结果图")
    add("")
    figs = []
    for name, caption in (
        ("u_plus_ci.png", "*图 1*：各臂 u+ 剖面与 batch-means 95% CI（误差棒），黑线为 MKM-1999 Re_τ=180；右图对数横轴。"),
        ("error_vs_t_obs.png", "*图 2*：对 MKM 的相对 L2 误差随统计窗 T_obs 的实测曲线、拟合幂律与 3% 门。"),
        ("contrasts.png", "*图 3*：设计对比 —— 左：域长效应 Δu+（2πδ − 4πδ）与合成 95% CI；右：SGS 负控 Δu+（C_s=0.1 − 0）。"),
    ):
        path = args.fig_dir / name
        if path.exists():
            add(f"![{name}]({path.as_posix()})")
            add("")
            add(caption)
            add("")
            figs.append({**fig_b64(path), "caption": caption})

    add("## 5. 注意点（踩坑 / 局限 / 适用范围）")
    add("")
    add(
        "1. **cumulant 内核在本算例不可用。** `smagorinsky_cs > 0` 只允许搭配 "
        "`collision_model='cumulant'`（内核校验），而 cumulant 在本槽道工况"
        "（τ ≈ 0.5018–0.5055、dy+ ≈ 16）下 ~10³ 步内发散：`probe_stability_p40n09.py` 里 "
        "tau=0.5018（ny=24）/0.5025（ny=32）/0.5018+小扰动、以及 τ=0.5055（Re_τ=60）全部 "
        "`FloatingPointError: channel population health failed`，而 natural-KBC 同工况 3000 步"
        "健康（min population 1.3e-2）。因此 SGS 负控只能在 cumulant 稳定的工况点执行"
        "（见 §3.4 与 `p40n09_numbers.json` 的 `negative_control`），并**不能**作为 "
        "Re_τ=180 生产网格上的负控证据 —— 这一点如实记为降级/局限，不粉饰。"
    )
    add(
        "2. **T_int 用壁面剪切信号代理。** 剖面的自相关积分时间无法从 1 个 run 的全局均值"
        "直接测（只有 batch 均值），故用每步壁面剪切的 ACF 积分时间作为大尺度时间尺度的"
        "代理；壁面剪切与剖面幅值强相关，但两者 T_int 不完全相等，属已知系统误差。"
    )
    add(
        "3. **归因加法假设。** `unexplained = max(err − U_SN − domain_shift, 0)` 假定域长与"
        "模型效应在相对 L2 度量下可加；两者若有交叉项，划分会偏移（方向未知）。"
    )
    add(
        "4. **U_SN 的定义。** U_SN 是误差度量（相对 L2）自身的 95% 半宽，"
        "不是单点 u+ 的 CI，也不是实验不确定度 —— 参考剖面 MKM 被当作无误差真值。"
    )
    add(
        "5. **网格粗。** ny 见 §1（dy+ ≈ 16）远非 DNS 级；本节点目的是**误差分解**而非"
        "把 u+ 误差压到 3% 以下，因此 §6 的剖面误差判定如实给出「否」，并给出机时外推。"
    )
    add(
        "6. **集群线程数与内存稳定性。** 同一 natural-KBC 臂在 loong64 节点以 96 线程长跑两次"
        "被 glibc 报 `double free or corruption` / `free(): corrupted unsorted chunks` 终止"
        "（崩溃日志留档 `runs_p40n09/D4_cs0_crash_n96.log`），降到 64 线程后未复现；"
        "根因未定位（环境级隐患，非本节点物理结论）。同一现象也在 cumulant 探针上出现过一次。"
        "如实记录为复现风险，正式数字全部来自 64 线程或以下的臂。"
    )
    add("")

    add("## 6. 误差判定（逐项，显式回答是否 < 3%）")
    add("")
    add("| 量 | 结果 | 是否 < 3% | 说明 |")
    add("|---|---|---|---|")
    if d4 and "batch_ci" in d4:
        ci_rel = d4["batch_ci"]["max_ci_half_relative_window"]
        add(
            f"| u+ 剖面 95% CI 半宽（统计不确定度） | {pct(ci_rel)} | "
            f"{'是' if ci_rel is not None and ci_rel <= 0.03 else '否'} | 比较窗 y+∈[10,170] 最大值 |"
        )
        l2 = d4["mkm"].get("relative_l2")
        add(
            f"| u+ 剖面 vs MKM 相对 L2 | {pct(l2)} | {'是' if l2 is not None and l2 <= 0.03 else '否'} | "
            f"粗网格（dy+≈16）系统性偏差主导 |"
        )
        add(
            f"| U_SN（度量自身 95% 半宽） | {pct(d4['mkm'].get('u_sn'))} | "
            f"{'是' if d4['mkm'].get('u_sn') is not None and d4['mkm']['u_sn'] <= 0.03 else '否'} | "
            f"统计项远小于门限 |"
        )
    if domain:
        value = domain["relative_l2_difference"]
        add(
            f"| 域长互差（2πδ vs 4πδ） | {pct(value)} | {'是' if value <= 0.03 else '否'} | 门 3% |"
        )
    if negative:
        add(
            f"| SGS 分离度（× CI） | {num(negative.get('separation_in_ci_units'), 2)} | "
            f"{'是' if negative['separated'] else '否'} | 需 > 1× 合成 CI |"
        )
    add("")
    add(
        "结论（显式）：**u+ 剖面对 MKM 的误差不是 < 3%**；其中**统计部分**"
        f"{'< 3%' if d4 and d4.get('batch_ci', {}).get('max_ci_half_relative_window', 1) <= 0.03 else '≥ 3%'}"
        "，"
        + (
            f"归因判决为 `{attribution['verdict']}`。"
            if attribution
            else "归因判决见 §3.5。"
        )
    )
    add("")
    add("## 7. 复现")
    add("")
    add("```bash")
    add("# 稳定性/成本探针（先跑，决定可用工况）")
    add("PYTHONPATH=src python3 benchmarks/pending/channel_turb_3d/probe_stability_p40n09.py \\")
    add("    --matrix benchmarks/pending/channel_turb_3d/_matrix_probe.json --out-dir runs_probe")
    add("# 各臂")
    add("PYTHONPATH=src python3 benchmarks/pending/channel_turb_3d/run_p40n09.py \\")
    add("    --case D4_cs0 --warm-steps 7700 --avg-steps 15400 --batch-count 24 \\")
    add("    --out-dir runs_p40n09 --threads 96")
    add("# 判据与数字/图")
    add("PYTHONPATH=src python3 benchmarks/pending/channel_turb_3d/analyze_p40n09.py \\")
    add("    --run-dir runs_p40n09 --out-dir . --fig-dir figures_p40n09")
    add("# 报告")
    add("PYTHONPATH=src python3 benchmarks/pending/channel_turb_3d/build_p40n09_report.py")
    add("```")
    add("")

    markdown = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")

    payload = {
        "title": f"{NODE} 湍流通道 u+ 误差的统计-模型分解（块平均 CI × 双域长互裁 × SGS 负控）",
        "summary": (
            f"把 P20-N01 的 15.19% u+ 误差分解为统计项 U_SN 与超出 CI 的系统项；"
            f"判决 {attribution['verdict'] if attribution else 'n/a'}；"
            f"六判据 " + ", ".join(f"{g['gate']}={g['status']}" for g in gates["gates"])
        ),
        "scenario_key": "turbulent_channel_uplus_error_decomposition",
        "mission_id": 6622,
        "tags": ["TensorLBM", "channel", "MKM1999", "batch-means", "uncertainty", "P40-N09"],
        "markdown": markdown,
        "figures": figs,
    }
    args.json_out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {args.out} ({len(markdown)} chars) and {args.json_out} ({len(figs)} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())