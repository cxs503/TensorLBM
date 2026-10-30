# QA 批次质检报告：报告 27, 28, 29

> **质检批次**: batch 27,28,29 | **对应工作单**: mission 15, 18, 17
> **质检员**: TensorPlatform 船舶总体性能智能体 | **日期**: 2026-10-29

---

## 判定汇总表

| 报告 ID | mission_id | 标题 | 判定 | 理由摘要 |
|---------|-----------|------|------|---------|
| 27 | 15 | SDAA 验证案例扫荡 — cavity_3d_full | **PASS** | 结构完整，3/3 数字命中，honest-FAIL 保留，结论不超证据 |
| 28 | 18 | SDAA Sweep couette_3d + SUBOFF/RANS/KBC 汇总 | **PASS** | 结构完整，3/3 数字命中，honest-FAIL 保留，结论不超证据 |
| 29 | 17 | SDAA 验证案例扫荡 — couette_2d（honest-FAIL） | **PASS** | 结构完整，3/3 数字命中，honest-FAIL 保留，结论不超证据 |

---

## 逐篇质检详情

### 报告 27（mission 15）

**标题**: Mission 15 补报：SDAA 验证案例扫荡 — cavity_3d_full

#### ① 结构五节检查
| 节 | 状态 | 说明 |
|---|------|------|
| 目的 | ✅ | §1 目的，4 项目标清晰 |
| 方法 | ✅ | §2 方法，含案例配置表、判据口径、四门验证体系 |
| 关键结果（含数字表） | ✅ | §3 关键结果，含 3 张数字表（cavity/SUBOFF/KBC） |
| 审计 | ✅ | §4.1 审计总结表，含证据等级 |
| 局限 | ✅ | §4.2 局限与诚实 FAIL 声明，§4.3 未覆盖项 |

#### ② 数字溯源抽查（3/3 命中）
| # | 报告数字 | 工件路径 | 工件值 | 判定 |
|---|---------|---------|--------|------|
| 1 | cavity_3d_full 96³ 误差 **0.260%** | `artifacts/sdaa_cavity_3d_full.json` | `"u_min_err_pct_vs_3d_ref": 0.26` | ✅ 命中 |
| 2 | SUBOFF Cumulant Cd **0.04179**，误差 **0.50%** | `artifacts/suboff_bench1_cumulant_sdaa20.json` | `"Cd_total": 0.041785936` | ✅ 命中 |
| 3 | RANS k-ε Re=1e5 误差 **20.5%** | `artifacts/suboff_bench2_rans_re1e5_sdaa21.json` | `"error_pct": 20.529998930713997` | ✅ 命中 |

#### ③ honest-FAIL 检查
- 原单结论含 honest-FAIL（RANS k-ε 20.5%/43.2% 超门限，KBC Cd 符号错误）
- 报告如实标记 ❌ FAIL，未粉饰 ✅

#### ④ 图检查
- 报告无图，但含 3 张详细数字表。未写明无图理由 → **MINOR 备注**

#### ⑤ 结论超证据检查
- 结论与数据一致：cavity 四门全过、SUBOFF Cumulant 0.50%、RANS FAIL、KBC FAIL
- 未发现超出数据支撑的断言 ✅

**判定: PASS**（MINOR 备注：无图且未声明无图理由）

---

### 报告 28（mission 18）

**标题**: Mission 18 补报: SDAA Sweep couette_3d + SUBOFF/RANS/KBC 汇总

#### ① 结构五节检查
| 节 | 状态 | 说明 |
|---|------|------|
| 目的 | ✅ | §1 目的，3 项目标 |
| 方法 | ✅ | §2 方法，含 Couette/SUBOFF/KBC 配置 |
| 关键结果（含数字表） | ✅ | §3 关键结果，含 3 张数字表 |
| 审计 | ✅ | §4.1 证据等级表 |
| 局限 | ✅ | §4.2 局限与后续，§4.3 Honest-FAIL 声明 |

#### ② 数字溯源抽查（3/3 命中）
| # | 报告数字 | 工件路径 | 工件值 | 判定 |
|---|---------|---------|--------|------|
| 1 | couette_3d fp64 L2 误差 **1.62e-14** | `artifacts/sdaa_couette_3d.json` | `"l2_rel_err": 1.6165870021629923e-14` | ✅ 命中 |
| 2 | SUBOFF Cumulant 误差 **0.50%** | `artifacts/suboff_bench1_cumulant_sdaa20.json` | `"Cd_total": 0.041785936`, Cf_ref=0.041995 → 误差≈0.50% | ✅ 命中 |
| 3 | KBC 最终 Cd **22.0622** | `artifacts/kbc_diagnostic_16cube_20steps.json` | `"kbc_final_Cd": 22.06220750870905` | ✅ 命中 |

#### ③ honest-FAIL 检查
- 报告明确声明 "SUBOFF RANS 基准（Bench2/3）和 KBC 诊断均为 FAIL"
- 未粉饰，honest-FAIL 保留 ✅

#### ④ 图检查
- 报告无图，未写明无图理由 → **MINOR 备注**

#### ⑤ 结论超证据检查
- 结论与数据一致，未发现超出数据支撑的断言 ✅

**判定: PASS**（MINOR 备注：无图且未声明无图理由）

---

### 报告 29（mission 17）

**标题**: Mission 17 补报：SDAA 验证案例扫荡 — couette_2d（honest-FAIL）

#### ① 结构五节检查
| 节 | 状态 | 说明 |
|---|------|------|
| 目的 | ✅ | §1 目的，3 项检验目标 |
| 方法 | ✅ | §2 方法，含物理/配置/判据/运行方式 |
| 关键结果（含数字表） | ✅ | §3 关键结果，含 4 张数字表（设备/数值/CPU对照/判据逐项） |
| 审计 | ✅ | §4 审计与分析，含精度评估/收敛性分析/Triton绕行/质量守恒 |
| 局限 | ✅ | §5 局限与后续，含本单局限与后续建议 |

#### ② 数字溯源抽查（3/3 命中）
| # | 报告数字 | 工件路径 | 工件值 | 判定 |
|---|---------|---------|--------|------|
| 1 | couette_2d H=40 max_rel_err **0.0202%** | `artifacts/sdaa_couette_2d.json` | `"max_rel_err_masked_pct": 0.020232588745828447` | ✅ 命中 |
| 2 | couette_2d H=80 max_rel_err **0.0209%** | `artifacts/sdaa_couette_2d.json` | `"max_rel_err_masked_pct": 0.02089242865568622` | ✅ 命中 |
| 3 | CPU 基线 H=40 **0.0198%** | `benchmarks/verified/couette_2d/result.json` | `"max_rel_err_pct": 0.0198` | ✅ 命中 |

#### ③ honest-FAIL 检查
- 报告标题即标注 "honest-FAIL"
- 结论明确 "overall_pass = false（收敛判据未通过）"
- 未粉饰，honest-FAIL 保留 ✅

#### ④ 图检查
- 报告无图，未写明无图理由 → **MINOR 备注**

#### ⑤ 结论超证据检查
- 结论与数据一致：物理精度达标但收敛判据失败
- 根因分析合理（浮点噪声/步数不足/max_rel_err 对局部极值敏感）
- 未发现超出数据支撑的断言 ✅

**判定: PASS**（MINOR 备注：无图且未声明无图理由）

---

## 缺陷清单汇总

| 报告 ID | 缺陷类型 | 严重度 | 描述 |
|---------|---------|--------|------|
| 27 | 无图未声明 | MINOR | 报告无图，未写明无图理由 |
| 28 | 无图未声明 | MINOR | 报告无图，未写明无图理由 |
| 29 | 无图未声明 | MINOR | 报告无图，未写明无图理由 |

**无 RESUBMIT 篇目** — 三篇均为 PASS（仅 MINOR 备注）。

---

## 处置决定

- **报告 27**: PASS — 不派返工单
- **报告 28**: PASS — 不派返工单
- **报告 29**: PASS — 不派返工单

本批次无需派返工单。

---

*质检完成时间: 2026-10-29*
*质检依据: 报告原文 + 工件 JSON + CPU 基线 result.json*
