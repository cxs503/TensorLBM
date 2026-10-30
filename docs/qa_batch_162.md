# 质检报告 · 批次 162（报告 id=162，mission_id=60）

**质检员**: TensorPlatform QA Agent  
**质检时间**: 2026-10-30  
**报告标题**: Mission 60 补报：Stokes 第二问题 SDAA 验证案例扫荡  
**报告场景**: report-backfill  
**原单状态**: succeeded / PASS  

---

## 逐篇判定表

| 报告 ID | 标题 | 判定 | 理由摘要 |
|---------|------|------|---------|
| 162 | Mission 60 补报：Stokes 第二问题 SDAA 验证案例扫荡 | **RESUBMIT** | 图件缺失（无 png_base64 内嵌，文件不存在）；其余项合格 |

---

## 缺陷清单（报告 162）

### 🔴 缺陷 1：图件缺失（RESUBMIT 级）

**规则依据**: owner 升级要求——场量云图+比较曲线两类齐备；缺任一类须有明确理由。

**事实**:
- 报告 figures 字段声明 3 张图（fig1_convergence.png, fig2_profile.png, fig3_sdaa_vs_cuda.png），但均**无 png_base64 内嵌**，仅有 name 和 caption。
- 在 m60 工作目录中搜索：`find /root/tp-agent/work/m60/ -name "fig1_convergence.png" -o -name "fig2_profile.png" -o -name "fig3_sdaa_vs_cuda.png"` → **零结果**。
- 报告 markdown 中以 `![Grid Convergence](fig1_convergence.png)` 等 markdown 语法引用图片，但图片文件不存在。
- 未提供任何图件缺失的理由说明（非纯文档/后端任务豁免情形）。

**影响**: 无法验证报告声称的"网格收敛曲线"、"速度衰减剖面"、"SDAA vs CUDA 全配置对比"三类可视化结果。

**修复要求**: 
1. 用 matplotlib 生成 3 张 PNG 图，以 base64 内嵌到 figures 字段（`png_base64` 键）。
2. 或在报告中明确声明图件缺失理由（如"纯计算任务，图件另行生成"），但按 owner 要求，验证类报告必须包含比较曲线。

### 🟡 缺陷 2：结论引用不存在的图件（MINOR 级）

**事实**: 报告结论部分引用了不存在的图件作为证据支撑（"左图：复振幅误差随网格高度 H 的变化"等），但图文件不存在。

**影响**: 结论的证据链不完整。

---

## 合格项核查

### ① 结构五节 ✅ PASS

| 节 | 要求 | 报告对应 | 判定 |
|----|------|---------|------|
| 目的 | 案例介绍/物理背景 | Section 1 案例介绍 | ✅ |
| 方法 | 方法/流程/关键设置 | Section 2 方法 | ✅ |
| 关键结果 | 含数字表 | Section 3 计算结果（3 张表） | ✅ |
| 审计 | 判据比对/注意点 | Section 5 注意点 | ✅ |
| 局限 | 局限性/适用范围 | Section 5 局限性/适用范围 | ✅ |

### ② 数字溯源抽查 ✅ PASS（3/3 命中）

| # | 报告中的数字 | 溯源文件 | 工件中的值 | 判定 |
|---|------------|---------|-----------|------|
| 1 | SDAA H=20,kH=6 幅值误差 **1.010%** | `artifacts/sdaa_stokes_second_problem.json` | `amp_rel_max_pct: 1.009910956405946` | ✅ 命中（四舍五入一致） |
| 2 | CUDA H=50,kH=6 复振幅误差 **0.303%** | `benchmarks/verified/stokes_second_problem/scan_out/result.json` | `complex_rel_max_pct: 0.3028331527700151` | ✅ 命中（四舍五入一致） |
| 3 | 总耗时 **179.3 秒** | `artifacts/sdaa_stokes_second_problem.json` | `wall_clock_s: 179.3` | ✅ 精确命中 |

### ③ honest-FAIL 保留 ✅ PASS

- 原单 mission 60 结论: **succeeded / PASS**
- 报告结论: **PASS**
- 无粉饰行为。

### ⑤ 误差判定节 ✅ PASS

- Section 6 完整列出关键量 vs 基准误差（幅值/相位/复振幅/网格收敛）。
- 显式回答「是否<3%」：四项均为 **✓ 是**。
- 有基准（解析解 + CUDA 基准），无不声明原因的情况。

### ⑥ 结论不超证据 ⚠️ MINOR

- 结论 PASS 与数据一致，未超出数据支撑。
- 但结论中引用的图件证据不存在（见缺陷 1）。

---

## 处置建议

| 报告 ID | 判定 | 处置 |
|---------|------|------|
| 162 | **RESUBMIT** | 派返工单：重写报告，补全图件（matplotlib 生成 PNG → base64 内嵌），重新 POST 报告 |

---

## 返工单要求（如派发）

**goal**: 【返工·报告 162】缺陷清单：
1. 图件缺失：3 张图（fig1_convergence.png, fig2_profile.png, fig3_sdaa_vs_cuda.png）无 png_base64 内嵌且文件不存在。要求用 matplotlib 生成实际图件，以 base64 内嵌到 figures 字段。
2. 结论引用不存在的图件：修复证据链。

**约束**: 只准重写报告文件与报告 POST，禁改业务代码。原报告 id=162，原 mission_id=60。
