# QA Batch Review — Reports 34, 35, 36

> **QA Agent**: TensorPlatform 船舶总体性能智能体  
> **Batch**: 34, 35, 36  
> **Mission ID**: 20 (all three)  
> **Repo**: TensorLBM  
> **Date**: 2026-10-29  
> **Reviewer**: QA Agent (owner 常设指令)

---

## 逐篇判定表

| Report ID | Title | Verdict | 判定 | 核心理由 |
|---|---|---|---|---|
| 34 | Mission 20 Backfill: SDAA Sweep cylinder_3d (Re=40) — honest-FAIL | honest-FAIL | **RESUBMIT** | ①三篇报告完全重复（同一内容发3次）；②零图件（场量云图+比较曲线均缺失，无豁免说明） |
| 35 | Mission 20 Backfill: SDAA Sweep cylinder_3d (Re=40) — honest-FAIL | honest-FAIL | **RESUBMIT** | 同报告34（完全相同的副本） |
| 36 | Mission 20 Backfill: SDAA Sweep cylinder_3d (Re=40) — honest-FAIL | honest-FAIL | **RESUBMIT** | 同报告34（完全相同的副本） |

---

## 缺陷清单（逐条）

### 缺陷 D1：三篇报告完全重复（RESUBMIT）

**证据**：
- 报告34、35、36的 `markdown` 字段逐字节相同（长度均为 10472 字符）。
- 三篇报告的 `title`、`summary`、`tags`、`figures` 完全一致。
- 创建时间分别为 `2026-10-29T15:03:31`、`15:03:38`、`15:03:45`，间隔仅7秒，明显为批量重复提交。

**要求**：三篇报告应各有独立内容（不同场景/不同工况/不同分析角度），或合并为单篇。当前重复提交浪费审阅资源。

### 缺陷 D2：零图件——场量云图与比较曲线均缺失（RESUBMIT）

**证据**：
- 三篇报告的 `figures` 字段均为空数组 `[]`。
- 报告正文无任何图片引用（无 `![...]` 标记）。
- Owner 升级要求：场量云图 + 比较曲线（与解析/文献/基线）两类齐备；缺任一类须有明确理由（纯文档/后端任务可豁免但必须写明）。
- 本报告为 CFD 计算报告，非纯文档/后端任务，不适用豁免。
- 报告未提供任何缺失图件的理由说明。

**应补图件**：
1. **场量云图**：D=20 和 D=40 网格的速度场/压力场云图（至少各1张）
2. **比较曲线**：Cd 收敛曲线（报告已有表格数据，应绘制为曲线图）与文献参考值（Cd=1.50）的水平对比线

### 缺陷 D3：结论超出数据支撑（MINOR）

**证据**：
- 报告第5.3节 Recommendations 第1条："Full-convergence rerun... PASS is expected based on the convergence trajectory."
- 该断言基于外推（convergence trajectory 外推至30,000步），而非实际运行数据。
- 虽然报告在 Limitations 中已说明 step-budget 问题，但 "PASS is expected" 仍属预测性断言，未以数据支撑。

**建议**：改为 "A full 30,000-step run is recommended to determine whether PASS can be achieved"，避免预设结论。

---

## 质检细则逐项核对

### ① 结构五节检查

| 节 | 报告34 | 报告35 | 报告36 |
|---|---|---|---|
| 1. Purpose | ✅ 有 | ✅ 有 | ✅ 有 |
| 2. Method | ✅ 有 | ✅ 有 | ✅ 有 |
| 3. Key Results（含数字表） | ✅ 有（3.1-3.5节，含多张表格） | ✅ 有 | ✅ 有 |
| 4. Audit | ✅ 有（4.1-4.4节） | ✅ 有 | ✅ 有 |
| 5. Limitations | ✅ 有（5.1-5.3节） | ✅ 有 | ✅ 有 |

**结论**：五节结构完整，不缺节。

### ② 数字溯源抽查（3个关键数字）

抽查数字与工件比对：

| # | 报告中的数字 | 工件路径 | 工件中的值 | 命中？ |
|---|---|---|---|---|
| 1 | Cd_mem_surface (D=20) = **1.5993** | `artifacts/sdaa_cylinder_3d.json` → `grid_results[0].cd_mem_surface` | 1.5993144363164902 | ✅ 命中（四舍五入一致） |
| 2 | Cd_mem_surface (D=40) = **1.7437** | `artifacts/sdaa_cylinder_3d.json` → `grid_results[1].cd_mem_surface` | 1.7436875205021352 | ✅ 命中（四舍五入一致） |
| 3 | Grid span = **9.62%** | `artifacts/sdaa_cylinder_3d.json` → `grid_span_pct` | 9.624872279043 | ✅ 命中（四舍五入一致） |

**辅助验证**：`docs/sdaa_sweep/cylinder_3d.md` 中同样记录 1.5993 / 1.7437 / 9.62%，与报告和工件一致。

**结论**：3/3 命中，数字溯源通过。

### ③ honest-FAIL 保留

- 原报告结论为 **FAIL**（0/3 gates passed）。
- 报告明确标注 "honest-FAIL"，未粉饰为 PASS。
- **结论**：✅ 通过。

### ④ 图件检查（owner 升级要求）

- 场量云图：❌ 缺失
- 比较曲线：❌ 缺失（收敛历史以表格呈现，无曲线图）
- 豁免说明：❌ 无
- **结论**：❌ 不通过 → RESUBMIT

### ⑤ 误差判定节（owner 升级要求）

| 关键量 | 基准值 | 报告值 | 误差 | 是否<3% | 显式回答？ |
|---|---|---|---|---|---|
| Cd (D=20) | 1.50 | 1.5993 | +6.62% | 否 | ✅ Gate G1 明确标 FAIL |
| Cd (D=40) | 1.50 | 1.7437 | +16.25% | 否 | ✅ Gate G2 明确标 FAIL |
| Grid span | ≤3% | 9.62% | — | 否 | ✅ Gate G3 明确标 FAIL |

**结论**：✅ 误差逐项列出且显式回答是否<3%，通过。

### ⑥ 结论不超证据

- 报告核心结论（FAIL due to step-budget）有数据支撑。
- 但 "PASS is expected" 为预测性断言，未以实际数据支撑（见缺陷 D3）。
- **结论**：⚠️ 轻微超出，记为 MINOR。

---

## 处置建议

| Report ID | 判定 | 处置 |
|---|---|---|
| 34 | RESUBMIT | 派返工单：重写报告（去重、补图件、修正预测性断言） |
| 35 | RESUBMIT | 派返工单：同上（或合并到34） |
| 36 | RESUBMIT | 派返工单：同上（或合并到34） |

**返工单要求**：
1. 三篇报告应各有独立内容，或合并为单篇综合报告
2. 补充场量云图（D=20 和 D=40 速度场/压力场）
3. 补充 Cd 收敛曲线图（与 Cd_ref=1.50 水平线对比）
4. 若无法生成图件，须在报告中明确写明豁免理由
5. 将 "PASS is expected" 改为建议性表述，避免预设结论
6. 仅重写报告文件与报告 POST，禁改业务代码

---

## QA 审计记录

- **审阅时间**：2026-10-29
- **审阅依据**：owner 常设指令（结构五节/数字溯源/honest-FAIL/图件/误差判定/结论不超证据）
- **工件路径**：
  - `/root/tp-agent/work/m20/artifacts/sdaa_cylinder_3d.json`
  - `/root/tp-agent/work/m20/docs/sdaa_sweep/cylinder_3d.md`
- **API 调用**：GET /tensor/reports/{34,35,36}
