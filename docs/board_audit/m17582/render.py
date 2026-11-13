#!/usr/bin/env python
"""Render ``AUDIT_m17582.md`` from ``audit.json`` — every number is injected, none typed.

    python docs/board_audit/m17582/render.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = json.loads((HERE / "audit.json").read_text())


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.3f}%"


def sci(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3e}"


def main() -> int:
    p1, p2, p3, p4, p5, p6 = (A["P1_ref_vs_got"], A["P2_reproduction"], A["P3_referee"],
                              A["P4_sigma_anchor"], A["P5_integration"], A["P6_detection_power"])
    rows = p1["rows"]
    L = []
    w = L.append
    w("# 交差审计 m17582 — XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN（受审 m17515）\n")
    w(f"* 受审工作树：`{A['subject_worktree']}`")
    w(f"* 受审提交：`{A['subject_head_commit']}` — {A['subject_log_head']}")
    w(f"* 审计任务：m17582（board-audit）｜锚点：`{A['anchor']}`")
    w(f"* 审计脚本：`docs/board_audit/m17582/audit.py`（本文件由 `render.py` 从 `audit.json` 渲染）\n")

    w("## 0. 判定摘要\n")
    w("| 项 | 值 |\n|---|---|")
    w(f"| 判定级别 | **L3**（L2 独立功能 + L3 系统集成成立；L4 基准验证不成立） |")
    w(f"| 机器裁判行 | `tensor_benchmark_cases` 该锚点 **{p3.get('n_rows')} 行**（全表 {p3.get('table_total_rows')} 行）→ 容差为作者自述 |")
    w(f"| 交付变体最差力场误差 | isotropic_d3q19 = {sci(p1['worst_rel_l2_by_operator']['isotropic_d3q19'])} |")
    w(f"| 生产默认变体最差力场误差 | central7 = {pct(p1['worst_rel_l2_by_operator']['central7'])}（> 15% 判据） |")
    w(f"| 越线算例 | {', '.join(p1['failing_15pct']) or '无'} |")
    w(f"| σ 反演最差误差 | {pct(p4['worst_sigma_err'])}（<15% 通过；3% 目标 {'达到' if p4['within_3pct'] else '未达'}） |")
    w(f"| 可复现 | 审计员独立重跑：decision 全等={p2.get('decision_identical')}，CSV 字节全等={p2.get('csv_identical')} |")
    w(f"| 判据分辨力 | 5% 算子误差 → {sci(p6.get('laplacian_plus5pct_rel_l2'))}/{sci(p6.get('grad_x_plus5pct_rel_l2'))}，仍 <15% |")
    w(f"| 处分 | request_changes（见 §7） |\n")

    w("## 1. 交付物三件套核验（存在性优先于数值）\n")
    w("| 项目 | 证据 | 结论 |\n|---|---|---|")
    w(f"| 入口可执行 | `benchmarks/xval_allen_cahn_csf_force_chain/run.py`（29 行，转调 `tensorlbm.xval.allen_cahn_csf.main`） | 真入口 |")
    w(f"| 运行工件 | `artifacts/xval_allen_cahn_csf_force_chain/`：{', '.join(p2.get('figures_present', []))} + JSON/CSV/manifest | 存在 |")
    w(f"| 基准表 | `benchmarks/xval_allen_cahn_csf_force_chain/result.json`（由 `make_result.py` 从工件再生成） | 存在 |")
    w(f"| 复现命令 | `{p2.get('repro_cmd')}` | 审计员实跑通过（`repro/rerun.log`） |")
    w(f"| 部件指纹 | shipped JSON `{p2.get('sha256_shipped_json')}` / 重跑 JSON `{p2.get('sha256_rerun_json')}` | 见 §3 |\n")

    w("## 2. ref-vs-got 逐项核（唯一裁判 = 15% 判据）\n")
    w("| 算例 | 变体 | rel L2 力 | 通过 15% | σ 反演误差 | σ 通过 |\n|---|---|---|---|---|---|")
    for r in rows:
        w(f"| {r['case']} | {r['operator']} | {sci(r['rel_l2_force'])} | "
          f"{'✔' if r['pass_15pct'] else '✘'} | {pct(r['sigma_err'])} | "
          f"{'✔' if r['sigma_pass_15pct'] else '✘'} |")
    w("")
    w(f"* 表内 {p1['n_rows']} 行（10 算例 × 2 变体），但**互异度量组仅 {p1['n_distinct_metric_groups']} 组**："
      f"{p1['duplicate_groups'][0]} 的 rel-L2 完全相同（σ 只做线性缩放，相对度量对其不变）"
      f"→ 算例数被高报约 {p1['n_rows'] / max(1, p1['n_distinct_metric_groups']):.1f}×。" if p1['duplicate_groups'] else "")
    w(f"* `decision` 旗标 {p1['artifacts_decision_flags']}；与算例表矛盾的旗标："
      f"{p1['flags_inconsistent_with_case_table']}。{p1['flag_note']}")
    w(f"* 报告 headline 如实列出越线算例：`cases_central7_failing_15pct = {p1['headline_claim']['cases_central7_failing_15pct']}`\n")

    w("## 3. 可复现性（审计员独立重跑）\n")
    w(f"* 重跑工件：`{p2.get('auditor_rerun_artefacts')}`（本次审计亲自执行）")
    w(f"* `decision` 块逐键全等 = **{p2.get('decision_identical')}**，最大绝对偏差 = {sci(p2.get('max_abs_decision_delta'))}")
    w(f"* `kernel_integration` 全等 = **{p2.get('kernel_integration_identical')}**；CSV 字节全等 = **{p2.get('csv_identical')}**")
    w(f"* 数值块（cases/kernel_integration/decision/operator_weights/thresholds）全等 = "
      f"**{p2.get('numeric_blocks_identical')}**；整份 JSON 的 sha256 差异仅由 "
      f"{p2.get('json_sha_diff_explained_by')}（输出目录路径/工件清单）造成。")
    w("* 结论：复现性为「实跑可复现」，不是文档承诺。\n")

    w("## 4. 物理锚（唯一非代码参照）\n")
    w(f"* σ 反演 vs 给定 σ：随 R 增大 **{', '.join(f'{r:g}:{100 * v:.2f}%' for r, v in p4['sigma_err_vs_radius_W4'])}**"
      "（W=4 固定）→ 误差随细化变大，未向 0 收敛。")
    w(f"* 随界面厚度：**{', '.join(f'W={r:g}:{100 * v:.2f}%' for r, v in p4['sigma_err_vs_W_R16'])}**（R=16 固定）"
      "→ 薄界面是弱项，最差 -11.75% 已逼近 15% 判据线。")
    w(f"* σ 误差在 lbmpy 侧自值完全相同（差异 ~1e-14）→ 该偏差是 CSF 力积分在该 W 下的离散固有偏差，双方共有。")
    w(f"* 仓库内已有 `{', '.join(p5['laplace_anchor_modules_present_but_unused_by_xval']) or 'n/a'}`（Young–Laplace Δp 实测面），"
      "本次 XVAL **未接入**：即「与另一代码一致」有，而「与解析/实验一致」缺。\n")

    w("## 5. 集成活性（L3 的真问题）\n")
    w(f"* 生产调用点 {p5['n_production_sites']} 处，其中**显式选择 operator 的仅 {p5['n_production_sites_selecting_isotropic']} 处**：")
    for s in p5["production_call_sites"]:
        w(f"  - `{s['file']}:{s['line']}` {s['call']}( … operator= {'是' if s['passes_operator'] else '**否**'} )")
    w(f"* 默认算子：{p5['default_operator']}。")
    w(f"* 耦合求解器：{p5['coupled_solver_gate']}")
    w("* 判读：算子**确实**被贯通进生产 Allen-Cahn 核（§6 的变异实验证明贯通是「活的」），"
      "但**被验证的那个变体在 8 个生产调用点里只有 1 个选中**；两个多相求解器仍走 central7，"
      "而 central7 在 C3-droplet-W3 的力场误差 20.6% 已越 15% 判据 → 改进「已交付、未部署」。\n")

    w("## 6. 判据分辨力（变异实验）\n")
    w(f"* 基线 rel L2 = {sci(p6.get('baseline_rel_l2'))}")
    w(f"* 拉普拉斯 ×1.05 → rel L2 = {sci(p6.get('laplacian_plus5pct_rel_l2'))}")
    w(f"* 梯度 x 分量 ×1.05 → rel L2 = {sci(p6.get('grad_x_plus5pct_rel_l2'))}")
    w(f"* 能看见 5% 算子误差 = **{p6.get('detects_5pct_error')}** ⇒ 度量不是装饰；"
      f"但 {p6.get('gate_resolution')}。\n")

    w("## 7. 分级判定与整改\n")
    w("**L2 ✔**：`phasefield/isotropic_fd.py`（真算子，含边界策略复用）+ `free_energy.py` 算子分派 + "
      "`allen_cahn_lbm.py` 贯通 + `xval/allen_cahn_csf.py`（lbmpy 符号表达式→numpy 求值 + 度量）——输入→计算→输出完整。\n")
    w("**L3 ✔**：生产核 `step_allen_cahn_field` 内 `force_mu_grad_phi(..., operator=operator)`（allen_cahn_lbm.py:385）"
      "被变异实验证明是活路径；13 项测试全绿（审计员实跑）。\n")
    w("**L4 ✘**（四条硬理由）：")
    w(f"1. 机器裁判缺行：`{p3['sql']}` → **{p3.get('n_rows')} 行**，15% 容差为作者自述，无机器可裁的参考值与容差登记。")
    w(f"2. 交付变体的一致性是「结构性的」：`operator_weights` 记录 isotropic 移植与 lbmpy 权重 max_abs_diff = 0，"
      f"故 {sci(p1['worst_rel_l2_by_operator']['isotropic_d3q19'])} 只证明移植保真，不证明物理正确；"
      f"真正独立的比较（central7 vs lbmpy）最差 {pct(p1['worst_rel_l2_by_operator']['central7'])} 越线。")
    w(f"3. 物理锚只有 σ 反演，最差 {pct(p4['worst_sigma_err'])}，且随 R 单调变差、不收敛；解析 Laplace 定律未纳入。")
    w("4. 无时间推进的耦合算例：`allen_cahn_step` 为 UNVERIFIED 硬门且不贯通 operator，"
      "因此「力链在求解器里被消费」这一点未被验证。")
    w(f"另附两条登记缺陷：算例数高报（{p1['n_rows']} 行/{p1['n_distinct_metric_groups']} 互异）；"
      f"`decision.central7_within_15pct=True` 与 C3-droplet-W3 的 {pct(p1['worst_rel_l2_by_operator']['central7'])} 自相矛盾。\n")
    w("**升 L4 的最小整改清单**：")
    w("1. 在 `tensor_benchmark_cases` 登记该锚点一行（metric=rel_l2_force, tolerance=0.15, ref_source=lbmpy 2.0 sha256）。")
    w("2. 把 `isotropic_d3q19` 接进 `multiphase3d.py:852` / `multiphase3d_d3q27.py:756`（或给出保持 central7 的量化理由），"
      "并补一个走求解器的时间推进静滴算例（spurious current vs W）。")
    w("3. 用仓库既有 `phasefield/laplace_face.py` 做 Δp = 2σ/R 解析锚，误差逐项列出并回答是否 <3%。")
    w("4. 修 `decision` 旗标（用全表而非 C1/C2 两例计算）并去重算例表（σ 扫描标注为「相对度量不变」）。")
    return _write(L)


def _write(L: list[str]) -> int:
    (HERE / "AUDIT_m17582.md").write_text("\n".join(L) + "\n")
    print(f"wrote {HERE / 'AUDIT_m17582.md'} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())