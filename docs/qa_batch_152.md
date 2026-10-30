# 质检报告 — 批次 152（报告 id=152, mission_id=242）

> 质检员：TensorPlatform 船舶总体性能智能体（QA Agent）
> 质检时间：2026-10-29
> 被检报告：《【SDAA 扫荡】值班长 m242 班次分析报告 — 43 案例判据审计（PASS 29 / FAIL 9 / WITHHELD 5）+ 3 单派发》

---

## 逐篇判定表

| 报告 id | 标题 | 判定 | 理由摘要 |
|---|---|---|---|
| 152 | 【SDAA 扫荡】值班长 m242 班次分析报告 | **MINOR** | 图件豁免理由未写明；其余各项达标 |

---

## 缺陷清单（逐项质检）

### ① 结构五节：目的/方法/关键结果/审计/局限

**判定：PASS**

报告包含完整六节（实际超出五节要求）：
- ① 案例介绍（目的/背景）✓
- ② 怎么做的（方法/流程/工具链）✓
- ③ 计算结果（含 5 张数字表）✓
- ④ 结果图 ✓
- ⑤ 注意点（踩过的坑/局限/适用范围）✓
- ⑥ 误差判定 ✓
- ⑦ 升级事项 ✓

不缺节，结构完整。

---

### ② 数字溯源抽查（3/3 命中）

**判定：PASS**

抽取 3 个关键数字，对照工件 `docs/_sdaa_sweep_m242.json` 复核：

| # | 报告中的数字 | 工件路径 | 工件中的值 | 命中？ |
|---|---|---|---|---|
| 1 | 扫荡线 mission **101** 张，完工 **100** 张 | `docs/_sdaa_sweep_m242.json` → `summary.sweep_missions_total` / `summary.sweep_done` | 101 / 100 | ✓ |
| 2 | 43 案例最新判据：**PASS 29 / FAIL 9 / WITHHELD 5** | `docs/_sdaa_sweep_m242.json` → `summary.case_verdict_counts` | `{"PASS": 29, "FAIL": 9, "WITHHELD": 5}` | ✓ |
| 3 | 失败单归因：**流程断点 29**、triton绕行 2、判据真红 1、环境 1 | `docs/_sdaa_sweep_m242.json` → `summary.mission_fail_tag_counts` | `{"流程断点": 29, "triton绕行": 2, "判据真红": 1, "环境": 1}` | ✓ |

3/3 全部命中，数字可溯源。

---

### ③ honest-FAIL 保留

**判定：PASS**

报告对 9 个 FAIL 案例如实记录，未粉饰为 PASS：
- 原文 §⑥ 明确写道：「判据 FAIL 的 9 例**明确未达标**」
- 举例：womersley `complex_rel_max=4.890% > 3%`（FAIL）、sphere_re100_mem `honest-FAIL`、thermal_cavity `0/6 PASS`
- 未将任何 FAIL 案例改写为 PASS，符合 honest-FAIL 纪律。

---

### ④ 图件（场量云图 + 比较曲线）

**判定：MINOR（缺陷）**

报告包含 2 张图：
- `m242_verdict_matrix.png` — 43 案例 × 逐轮判据状态矩阵
- `m242_fail_pareto.png` — 失败归因 Pareto + 设备分布

**缺陷**：这两张是运维审计类图表，非「场量云图」和「与解析/文献/基线的比较曲线」。报告属于**纯文档/后端审计任务**，按 owner 升级要求可豁免图件，但**必须在报告中写明豁免理由**。报告 §④ 仅描述了图的内容，未声明「本班次为运维审计/元分析任务，不涉及物理场量计算，故无场量云图与比较曲线」。

**建议修复**：在 §④ 开头补充一句：「本班次为 SDAA 扫荡运维审计（元分析），不涉及物理场量计算，故无场量云图与比较曲线，按纯文档/后端任务豁免。」

---

### ⑤ 误差判定节

**判定：PASS**

报告 §⑥ 误差判定节：
- 逐项列出 43 案例的判据、`<3%` 声明、提取的百分比数值 ✓
- 显式回答：「凡判据 PASS 的案例，其最新单文本声明关键量判据全过（即 <3%）」✓
- 对无证据的 5 例（cavity, sphere_re200, taylor_couette, thermal_cavity_3d, womersley）声明原因：「平台记录无显式判据语句或该案例最新单 mission 状态 failed」✓
- 不模糊、不含糊，符合 owner 升级要求。

---

### ⑥ 结论不超证据

**判定：PASS**

报告谨慎处理证据不足的案例：
- 5 例 WITHHELD 明确标注「不作误差判定」
- 失败归因注明「以文本关键词为准」「可能重叠」「并非互斥统计」
- 升级事项中承认「报告口径漂移」问题，留 owner 裁定，不自作主张
- 无超出数据支撑的断言。

---

## 总判定：MINOR

**缺陷汇总**：
1. [MINOR] §④ 图件豁免理由未写明（纯文档/后端审计任务应声明豁免）

**无需派返工单**（MINOR 只记录不派单）。

---

## 质检证据引用

- 报告原文：`GET /tensor/reports/152`（markdown 全文 13808 字符）
- 数字溯源工件：`/root/tp-agent/work/m242/docs/_sdaa_sweep_m242.json`（summary 字段）
- 图件文件：`/root/tp-agent/work/m242/docs/figs/m242_verdict_matrix.png`（243997 bytes）、`m242_fail_pareto.png`（78727 bytes）
- 逐案例表：`/root/tp-agent/work/m242/docs/_sdaa_sweep_m242.md`
