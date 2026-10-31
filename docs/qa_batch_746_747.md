# QA 批次质检报告：Report 746, 747

> **质检席**：qwen3.6-plus（cross-family，作者族未知）
> **批次**：746, 747（对应 mission_id: 767, 774）
> **日期**：2026-10-31
> **判定汇总**：746=PASS，747=PASS

---

## 一、逐篇判定表

| Report ID | Mission ID | 标题 | 判定 | 理由摘要 |
|-----------|-----------|------|------|---------|
| 746 | 767 | 冰级船型多目标优化框架成熟化 · 返工重发报告（取代 report id=700） | **PASS** | 五节齐全；3/3 数字溯源命中；图件豁免理由充分；误差判定逐项显式回答<3%；无超证据断言 |
| 747 | 774 | 返工回执：report 700 → 重发为 report 746 | **PASS** | 回执性质报告，五节齐全；3/3 数字溯源命中；引用 report 746 的图件与误差判定；无超证据断言 |

---

## 二、逐篇详细质检

### Report 746（mission_id=767）

#### ① 结构五节检查

| 节 | 要求 | 实际 | 状态 |
|----|------|------|------|
| 目的 | 明确报告目标 | §0 缺陷清单逐条闭合，§1 案例介绍 | ✓ |
| 方法 | 方法/流程/工具链 | §1.2 物理模型（Lindqvist/Holtrop-Mennen/GM），§2 工具链（build_report_m774.py） | ✓ |
| 关键结果（含数字表） | 数字表+关键发现 | §3 含 7 个子节，多个数字表（目标区间、代表点、约束、收敛） | ✓ |
| 审计 | 独立验证/一致性检查 | §3.2b 前沿非支配性审计，§3.5 约束独立重算，§3.6 工件一致性，§8 误差判定 | ✓ |
| 局限 | 适用范围/未做事项 | §6 注意点（踩坑/局限/适用域），§7 后续工作 | ✓ |

**五节齐全，不缺节。**

#### ② 数字溯源抽查（3 个关键数字）

| # | 报告声称 | 工件路径 | 工件实测 | 命中？ |
|---|---------|---------|---------|--------|
| 1 | R_ice ∈ [36.94, 266.01] kN | `runs/ice_class/pareto_front.csv` | min=36.94, max=266.01 | ✓ |
| 2 | R_ow ∈ [88.90, 671.08] kN | `runs/ice_class/pareto_front.csv` | min=88.90, max=671.08 | ✓ |
| 3 | GM ∈ [1.2679, 8.6316] m | `runs/ice_class/pareto_front.csv` | min=1.2679, max=8.6316 | ✓ |

**3/3 命中。** 验证命令：
```python
import csv
with open('/root/tp-agent/work/m767/runs/ice_class/pareto_front.csv') as f:
    rows = list(csv.DictReader(f))
    # R_ice: min=36.94, max=266.01
    # R_ow: min=88.90, max=671.08
    # GM: min=1.2679, max=8.6316
```

#### ③ honest-FAIL 保留

原报告（id=700）非 FAIL 结论，本报告为返工重发（rework/reissue），不涉及 FAIL 粉饰问题。**不适用。**

#### ④ 图件检查

| 图件类别 | 要求 | 实际 | 状态 |
|---------|------|------|------|
| 场量云图 | 必须有或豁免理由 | §5.1 豁免：本任务为零维/经验公式优化，无空间场可着色；理由三条充分 | ✓ |
| 比较曲线 | 必须有或豁免理由 | §5.2 说明：无实验/模型试验基线，不画"模型 vs 实测"曲线以免充数；内部一致性用表格即可 | ✓ |
| 实际图件 | 5 张 | pareto_3d, pareto_2d_projections, design_space_parallel, convergence_curve, objective_ranges_check | ✓ |

**注**：`objective_ranges_check.png` 实际存在于 m774 的 `artifacts/ice_class/` 而非 m767 的同路径下。因本报告系 m774 返工生成但发布在 m767 名下，属跨目录引用，不影响数字可验证性，记录为轻微备注。

#### ⑤ 误差判定节

§8 误差判定表逐项列出：

| 关键量 | 基准 | 是否<3%？ | 理由 |
|--------|------|----------|------|
| R_ice | 无实验基准 | **否** | 文献散布 15–25%，无基准可对 |
| R_ow | 无实验基准 | **否** | 文献散布 5–10%，且 i_E 系数为标定值 |
| GM | 无完整静水力基准 | **否（对基准）/ 是（对自洽）** | 壁侧近似 vs 实船；工件-模型自洽达机器精度 |
| 工件-模型自洽（全精度） | 冻结 objectives.py 重算 | **是** | 最大偏差 0.0e+00 |
| 工件-模型自洽（CSV 打印精度） | 同上 | **是** | 最大偏差 ~1e-3（纯舍入） |

**逐项显式回答「是否<3%」，无基准项声明原因，格式合规。**

#### ⑥ 结论不超证据

- 报告明确标注多随机种子研究为 WITHHELD（§6.2），未声称"多种子通过"
- 平台集成未打通如实记录（§6.2）
- 前沿退化解（L41 被 L33 支配）如实披露（§3.2b）
- **无超出数据支撑的断言。**

#### 缺陷清单（MINOR 级）

| # | 缺陷 | 严重度 | 说明 |
|---|------|--------|------|
| M1 | `objective_ranges_check.png` 路径跨 mission | MINOR | 图件在 m774/artifacts/ice_class/ 而非 m767/ 下同路径；数字可验证，不影响结论 |

---

### Report 747（mission_id=774）

#### ① 结构五节检查

| 节 | 要求 | 实际 | 状态 |
|----|------|------|------|
| 目的 | 明确报告目标 | §1 缺陷闭合情况（逐条） | ✓ |
| 方法 | 方法/流程/工具链 | §2 怎么做的（build_report_m774.py 工具链） | ✓ |
| 关键结果（含数字表） | 数字表+关键发现 | §3 验收判据（HV/可行性/折中比/wall time） | ✓ |
| 审计 | 独立验证/一致性检查 | §1 缺陷闭合表（7 条逐项），引用 report 746 的完整审计 | ✓ |
| 局限 | 适用范围/未做事项 | §4 注意点与局限（单种子/无基准/平台不可达/退化解） | ✓ |

**五节齐全，不缺节。** 作为回执报告，正文引用 report 746 避免重复，属合理设计。

#### ② 数字溯源抽查（3 个关键数字）

| # | 报告声称 | 工件路径 | 工件实测 | 命中？ |
|---|---------|---------|---------|--------|
| 1 | R_ice ∈ [36.94, 266.01] kN | `runs/ice_class/pareto_front.csv` | min=36.94, max=266.01 | ✓ |
| 2 | R_ow ∈ [88.90, 671.08] kN | `runs/ice_class/pareto_front.csv` | min=88.90, max=671.08 | ✓ |
| 3 | GM ∈ [1.2679, 8.6316] m | `runs/ice_class/pareto_front.csv` | min=1.2679, max=8.6316 | ✓ |

**3/3 命中。** m774 的 runs/ice_class/ 与 m767 同源同内容。

#### ③ honest-FAIL 保留

原报告非 FAIL，回执不涉及 FAIL 粉饰。**不适用。**

#### ④ 图件检查

- 回执附 1 张工件范围核对图（`objective_ranges_check.png`，存在于 m774/artifacts/ice_class/）
- 其余 4 张图引用 report 746
- 作为回执报告，引用主报告图件合理 ✓

#### ⑤ 误差判定节

- 回执本身不重复 report 746 的 §8，但 §4 明确声明"无实验/模型试验基准 → R_ice、R_ow 的「<3%」判定显式回答「否」"
- 作为回执，引用主报告的误差判定节可接受 ✓

#### ⑥ 结论不超证据

- 单种子、无基准、平台不可达、退化解均如实披露
- **无超出数据支撑的断言。**

#### 缺陷清单

**无缺陷。**

---

## 三、附加验证

### pytest 护栏验证

m774 的 `tests/test_ice_class_report_consistency.py`（7 项测试）全部通过：

```
tests/test_ice_class_report_consistency.py::test_objective_ranges_match_artifact PASSED
tests/test_ice_class_report_consistency.py::test_representative_designs_match_artifact PASSED
tests/test_ice_class_report_consistency.py::test_single_seed_claim PASSED
tests/test_ice_class_report_consistency.py::test_error_section_answers_under_three_percent PASSED
tests/test_ice_class_report_consistency.py::test_figure_exemption_documented PASSED
tests/test_ice_class_report_consistency.py::test_metrics_json_consistent_with_csv PASSED
tests/test_ice_class_report_consistency.py::test_front_audit_disclosed PASSED
========================= 7 passed, 1 warning in 0.04s =========================
```

### metrics_m774.json 一致性

`/root/tp-agent/work/m774/docs/ice_class/metrics_m774.json` 中的关键数字与 CSV 一致：
- R_ice_lo=36.94, R_ice_hi=266.01
- R_ow_lo=88.9, R_ow_hi=671.08
- GM_lo=1.2679, GM_hi=8.6316
- convergence.max_hv_change_rate_last_5=0.000534 < 0.01
- constraints.all_pass=true
- tradeoff_ratio=0.3197 > 0.10

---

## 四、处置决定

| Report | 判定 | 处置 |
|--------|------|------|
| 746 | PASS | 记录 MINOR 缺陷 M1，不派返工 |
| 747 | PASS | 无缺陷，不派返工 |

**本批无 RESUBMIT，不派返工单。**

---

## 五、质检元数据

- 质检 agent：qwen3.6-plus
- 评审独立性：cross-family（作者族未知 vs 质检席）
- 验证方式：CSV 数字复核 + pytest 7/7 通过 + metrics.json 一致性检查
- 工件路径：`/root/tp-agent/work/m767/runs/ice_class/pareto_front.csv`，`/root/tp-agent/work/m774/docs/ice_class/metrics_m774.json`
