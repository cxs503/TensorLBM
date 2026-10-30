# QA 批次质检报告：Reports 166, 169

> **质检员**: TensorPlatform QA Agent (m264)  
> **日期**: 2026-10-30  
> **批次**: 166, 169（对应 mission_id 61, 63）  
> **仓库**: TensorLBM  

---

## 逐篇判定表

| 报告 ID | Mission ID | 标题 | 判定 | 理由摘要 |
|---------|-----------|------|------|---------|
| 166 | 61 | stokes_sphere_dg SDAA 扫荡验证 (honest-FAIL / PARTIAL_PASS) | **MINOR** | 结构缺独立"审计"节；图件仅1条API记录且无png_base64内嵌 |
| 169 | 63 | SUBOFF Sail Resistance SDAA 验证案例扫荡（honest-FAIL） | **PASS** | 结构完整，数字可溯源，honest-FAIL保留，图件缺失有明确理由，误差判定显式 |

---

## 报告 166 详细质检

### ① 结构五节检查

| 要求节 | 实际对应 | 状态 |
|--------|---------|------|
| 目的 | §1 案例介绍 | ✅ |
| 方法 | §2 方法/流程 | ✅ |
| 关键结果（含数字表） | §3 计算结果（3.1/3.2/3.3 三表） | ✅ |
| 审计 | 无独立节，内容散见于 §5 注意点 + §6 误差判定 | ⚠️ 缺独立节 |
| 局限 | §5 注意点 + §6.3 无基准声明 | ✅ |

**判定**: 缺独立"审计"节 → 降级因素

### ② 数字溯源抽查（3/3）

| # | 报告数字 | 工件路径 | 工件值 | 命中 |
|---|---------|---------|--------|------|
| 1 | K_sim = 1.77863 | `m61/artifacts/sdaa_stokes_sphere_dg.json` → `tiers[0].verdict.K_sim` | 1.7786343929191741 | ✅ |
| 2 | err = −2.882% | 同上 → `tiers[0].verdict.err_pct` | −2.882038279184651 | ✅ |
| 3 | 质量漂移 = 1.33×10⁻⁷ | 同上 → `tiers[0].verdict.S0_max_mass_drift` | 1.3328911363519325e-07 | ✅ |

**判定**: 3/3 命中 ✅

### ③ honest-FAIL 保留

- 原结论: honest-FAIL / PARTIAL_PASS
- 报告结论: honest-FAIL（PARTIAL_PASS）——覆盖度不足，无法与原案 12/12 VERIFIED 等价
- **判定**: 保留 ✅

### ④ 图件检查

- 报告正文 §4 称"见内嵌 figures（4 面板）"，描述 (a)(b)(c)(d) 四图
- API 返回 figures 数组仅 1 条: `sdaa_stokes_sphere_dg_sweep`，无 `png_base64` 字段
- 实际无内嵌 PNG 图像，仅文字描述
- 本任务为 SDAA 后端验证扫荡，非纯文档任务，图件缺失需说明
- **判定**: ⚠️ 图件未实际内嵌，正文描述与实际不符 → MINOR

### ⑤ 误差判定节

- §6 有独立"误差判定"节
- §6.1 关键量 vs 基准表：逐项列出 K_sim、SDAA vs NVIDIA、质量漂移、双估计器
- 每项显式回答"是否 < 3%"：✅ 是 / ✅ 是 / ✅ 是 / ✅ 是
- §6.3 无基准声明：R-ladder 收敛性门声明 NOT_EVALUATED，说明原因（预算约束）
- **判定**: ✅ 完整

### ⑥ 结论不超证据

- 报告明确声明 PARTIAL_PASS，不声称完整 VERIFIED
- 不可比性声明（§5.5）明确本结果不能替代原案
- **判定**: ✅ 未超证据

### 缺陷清单（Report 166）

| # | 缺陷 | 严重度 | 证据 |
|---|------|--------|------|
| D1 | 缺少独立"审计"节 | MINOR | 报告结构为 §1-§6，无"审计"标题节 |
| D2 | 图件未内嵌 PNG | MINOR | API figures 仅1条且无 png_base64；正文称"4面板"但无实际图像 |

---

## 报告 169 详细质检

### ① 结构五节检查

| 要求节 | 实际对应 | 状态 |
|--------|---------|------|
| 目的 | §1 案例介绍 | ✅ |
| 方法 | §2 方法/流程/关键设置 | ✅ |
| 关键结果（含数字表） | §3 计算结果（3.1.1/3.1.2/3.1.3/3.2/3.3 多表） | ✅ |
| 审计 | §5 注意点（踩过的坑/局限性/适用范围）+ §8 产物清单 | ✅ |
| 局限 | §5.2 局限性 + §6.3 无基准声明 | ✅ |

**判定**: 结构完整 ✅

### ② 数字溯源抽查（3/3）

| # | 报告数字 | 工件路径 | 工件值 | 命中 |
|---|---------|---------|--------|------|
| 1 | L96 OOM: "Tried to allocate 17179869182.15 GiB" | `m63/artifacts/sdaa_suboff_sail_resistance.json` → `tiers.96.blocked_reason` | 完全匹配 | ✅ |
| 2 | B1 Cumulant 误差 0.5% | `m63/artifacts/suboff_best_rans_summary.json` → `benchmarks.bench1_cumulant_re1000.results.error_pct` | 0.5 | ✅ |
| 3 | L68 Cd_total = 0.08312 | `m63/artifacts/sdaa_suboff_sail_resistance.json` → `supplementary_reduced_grid.run.cd_total_window` | 0.0831243664268973 | ✅ |

**判定**: 3/3 命中 ✅

### ③ honest-FAIL 保留

- 原结论: honest-FAIL
- 报告结论: honest-FAIL（显式声明，未降判据，无 CPU 冒充）
- **判定**: 保留 ✅

### ④ 图件检查

- §4 明确声明无图件，给出 4 条理由：
  1. 冻结阶梯三档全部 OOM，未产生流场快照
  2. 补充档 L68 仅跑 406 步（需 10,200 步），流场未收敛
  3. 辅助基准产物为 JSON 时间序列，非场量数据
  4. 任务纪律"不改业务代码"，无法追加场量输出
- 属于"纯文档/后端任务"豁免范畴，理由充分
- **判定**: ✅ 豁免有效

### ⑤ 误差判定节

- §6 有独立"误差判定"节
- §6.1 主案例：显式回答"不满足 <3%"，原因：SDAA 卡显存不足
- §6.2 辅助基准：B1 (0.5%) ✅ <3%；B2 (20.5%) ❌ >3%；B3 (43.2%) ❌ >3%
- §6.3 无基准声明：L68 补充档误差"不作为判据"，KBC 诊断无定量基准
- **判定**: ✅ 完整且显式

### ⑥ 结论不超证据

- 报告明确 honest-FAIL，归因于硬件容量（14.94 GiB vs 需 ≥32 GiB）
- 辅助基准 Cumulant 0.5% 仅证明算法精度足够，不用于主案例判门
- 建议部分明确标注"不在本任务范围"
- **判定**: ✅ 未超证据

### 缺陷清单（Report 169）

| # | 缺陷 | 严重度 | 证据 |
|---|------|--------|------|
| — | 无 | — | 各项检查均通过 |

---

## 质检汇总

| 报告 | 判定 | 缺陷数 | 是否派返工单 |
|------|------|--------|-------------|
| 166 | MINOR | 2 (D1, D2) | 否（MINOR 只记录） |
| 169 | PASS | 0 | 否 |

**本批无需派返工单。**

---

## 质检证据索引

| 证据项 | 路径/引用 |
|--------|----------|
| Report 166 原文 | API GET /tensor/reports/166 |
| Report 169 原文 | API GET /tensor/reports/169 |
| m61 工件 | `/root/tp-agent/work/m61/artifacts/sdaa_stokes_sphere_dg.json` |
| m63 工件 | `/root/tp-agent/work/m63/artifacts/sdaa_suboff_sail_resistance.json` |
| m63 辅助基准 | `/root/tp-agent/work/m63/artifacts/suboff_best_rans_summary.json` |
| m63 KBC 诊断 | `/root/tp-agent/work/m63/artifacts/kbc_diagnostic_16cube_20steps.json` |
