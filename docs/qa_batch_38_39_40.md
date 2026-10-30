# 质检报告：批次 38, 39, 40

> **质检员**: TensorPlatform QA Agent (m128)  
> **日期**: 2026-10-29  
> **批次**: 38, 39, 40  
> **对应工作单**: mission_id 22, 22, 24  

---

## 逐篇判定表

| 报告 ID | mission_id | 标题 | 判定 | 主要缺陷 |
|---|---|---|---|---|
| 38 | 22 | Mission 22 补报: SDAA Sweep — floating_body_free_decay (honest-FAIL) | **MINOR** | 图件缺豁免声明 |
| 39 | 22 | Mission 22 补报: SDAA Sweep — floating_body_free_decay (honest-FAIL) | **MINOR** | 图件缺豁免声明（同38） |
| 40 | 24 | Mission 24 补报：SDAA 验证扫荡 — Laplace CAC 相场两相 LBM 静态液滴基准 | **MINOR** | 缺审计节 |

---

## 报告 38 详细质检

### ① 结构五节检查

| 节 | 要求 | 实际 | 判定 |
|---|---|---|---|
| 目的 | Section 1 案例介绍 | ✅ 有 | PASS |
| 方法 | Section 2 方法：SDAA 迁移策略 | ✅ 有 | PASS |
| 关键结果 | Section 3 关键结果（含数字表） | ✅ 有，含 SUBOFF 三臂数据表 | PASS |
| 审计 | Section 4 关联工件概览 | ✅ 有，列出 artifact 关键值 | PASS |
| 局限 | Section 5 注意点与局限 | ✅ 有，4 小节 | PASS |

**结构判定**: ✅ 五节齐备

### ② 数字溯源抽查（3 个关键数字）

| # | 报告中的数字 | 工件路径 | 工件中的值 | 命中? |
|---|---|---|---|---|
| 1 | B1 Cf_num = 0.04179 | `m22/artifacts/suboff_bench1_cumulant_sdaa20.json` → `Cf_numerical` | 0.04178593600446055 | ✅ 四舍五入一致 |
| 2 | B2 误差 = 20.5% | `m22/artifacts/suboff_bench2_rans_re1e5_sdaa21.json` → `error_pct` | 20.529998930713997 | ✅ 四舍五入一致 |
| 3 | B3 误差 = 43.2% | `m22/artifacts/suboff_bench3_rans_re1e4_sdaa22.json` → `error_pct` | 43.20732307745836 | ✅ 四舍五入一致 |

**数字判定**: ✅ 3/3 命中

### ③ honest-FAIL 保留

原单结论为 FAIL（业务代码 bug 导致 SDAA 无法执行），报告结论为 **honest-FAIL**，未粉饰。  
**判定**: ✅ PASS

### ④ 图件检查

- 场量云图: ❌ 缺失
- 比较曲线: ❌ 缺失
- 豁免理由: 报告未显式声明"因案例运行失败无法产出图件，故豁免图件要求"。虽然 Section 3.1 显示"案例运行 ❌ FAIL"和"未能产出数值"，但未在图件相关位置明确引用 owner 升级要求的豁免条款。

**判定**: ⚠️ MINOR — 缺图件豁免显式声明

### ⑤ 误差判定节

Section 6 "误差判定" 存在，逐项列出关键量误差并显式回答是否 <3%：
- floating_body ω_n (SDAA): 无法判定（业务代码 bug）
- SUBOFF B1: 0.5% ✅ 是
- SUBOFF B2: 20.5% ❌ 否
- SUBOFF B3: 43.2% ❌ 否

**判定**: ✅ PASS

### ⑥ 结论不超证据

报告结论严格限定在数据支撑范围内：
- 明确指出 FAIL 归因于业务代码 bug，非算法问题
- SUBOFF 结果如实标注 ⚠️ 并解释原因（缺壁面函数、过渡 Re 模型失效）
- 未对未验证的物理判据做出断言

**判定**: ✅ PASS

### 报告 38 缺陷清单

| # | 缺陷类型 | 描述 | 证据引用 |
|---|---|---|---|
| 1 | 图件豁免声明缺失 | 报告未显式声明因案例运行失败而豁免图件要求 | 原文 Section 3.1 显示"案例运行 ❌ FAIL"，但无图件豁免声明 |

---

## 报告 39 详细质检

报告 39 与报告 38 **内容完全相同**（均为 mission 22 补报，标题、正文、结构一致）。

### ① 结构五节检查: ✅ 五节齐备（同 38）
### ② 数字溯源抽查: ✅ 3/3 命中（同 38）
### ③ honest-FAIL 保留: ✅ PASS（同 38）
### ④ 图件检查: ⚠️ MINOR — 缺图件豁免显式声明（同 38）
### ⑤ 误差判定节: ✅ PASS（同 38）
### ⑥ 结论不超证据: ✅ PASS（同 38）

### 报告 39 缺陷清单

| # | 缺陷类型 | 描述 | 证据引用 |
|---|---|---|---|
| 1 | 图件豁免声明缺失 | 同报告 38 | 同上 |

---

## 报告 40 详细质检

### ① 结构五节检查

| 节 | 要求 | 实际 | 判定 |
|---|---|---|---|
| 目的 | Section 1 案例介绍 | ✅ 有 | PASS |
| 方法 | Section 2 方法 | ✅ 有 | PASS |
| 关键结果 | Section 3 关键结果（含数字表） | ✅ 有，含 SDAA 单臂结果表、NVIDIA 对比表、吞吐表、门控汇总表 | PASS |
| 审计 | — | ❌ 无独立审计节。附录"工件清单"仅列文件名，未审计内容 | **FAIL** |
| 局限 | Section 5 注意点 | ✅ 有，5 条注意点 | PASS |

**结构判定**: ⚠️ 缺审计节 → 降级

### ② 数字溯源抽查（3 个关键数字）

| # | 报告中的数字 | 工件路径 | 工件中的值 | 命中? |
|---|---|---|---|---|
| 1 | σ_rec = 0.00999433 | `m24/artifacts/sdaa_laplace_cac_phasefield.json` → `arms.b3_rho100.sigma_rec_window_mean` | 0.00999432666508884 | ✅ 四舍五入一致 |
| 2 | 误差 = 0.057% | 同上 → `err_pct` | 0.05673334911160746 | ✅ 四舍五入一致 |
| 3 | u_max = 6.79×10⁻⁵ | 同上 → `u_max_window` | 6.787214269826273e-05 | ✅ 四舍五入一致 |

**数字判定**: ✅ 3/3 命中

### ③ honest-FAIL 保留

原单结论为 FAIL（臂覆盖不足 1/5），报告结论为 **honest-FAIL**，未粉饰。  
**判定**: ✅ PASS

### ④ 图件检查

- 场量云图: ❌ 缺失
- 比较曲线: ❌ 缺失
- 豁免理由: ✅ Section 4 "结果图" 显式声明："本补报无 npz/png 工件。理由：SDAA 扫荡脚本仅输出 JSON 时间序列（21 个采样点），未保存场数据（φ/u/p 的 npz）。如需可视化，可在后续全量运行中增加 `--save-fields` 选项导出 VTK/npz，再用 matplotlib 绘制 σ_rec 收敛曲线与液滴截面云图。"

**判定**: ✅ PASS（有明确豁免理由）

### ⑤ 误差判定节

Section 6 "误差判定" 存在，显式回答关键量 σ_rec 误差是否 <3%：
- σ_rec 相对误差 = 0.057%，远小于 3% 阈值 ✅
- 声明了基准来源（Young–Laplace 解析关系 + NVIDIA 侧验证）

**判定**: ✅ PASS

### ⑥ 结论不超证据

报告结论严格限定：
- 明确指出 FAIL 原因是臂覆盖不足（1/5），非物理判据失败
- 网格缩减、步数缩减效应均在注意点中声明
- Route-B 标定常数跨设备迁移的结论有数据支撑（σ_rec 误差 0.057%）

**判定**: ✅ PASS

### 报告 40 缺陷清单

| # | 缺陷类型 | 描述 | 证据引用 |
|---|---|---|---|
| 1 | 审计节缺失 | 报告无独立审计节；附录"工件清单"仅列文件名，未对工件内容进行审计（如校验 JSON 字段完整性、数值一致性等） | 原文 Section 4 为"结果图"，Section 5 为"注意点"，Section 6 为"误差判定"，附录仅列文件清单 |

---

## 汇总

| 报告 ID | 判定 | 缺陷数 | 需返工? |
|---|---|---|---|
| 38 | MINOR | 1 | 否 |
| 39 | MINOR | 1 | 否 |
| 40 | MINOR | 1 | 否 |

**本批无需派返工单**（均为 MINOR，仅记录不派单）。

> **注**：报告 38 与报告 39 内容完全相同（diff 验证零差异），均为 mission 22 的补报。建议平台侧核查是否存在重复提交。

---

## 质检证据索引

| 证据项 | 路径/引用 |
|---|---|
| 报告 38 原文 | API GET /tensor/reports/38 |
| 报告 39 原文 | API GET /tensor/reports/39 |
| 报告 40 原文 | API GET /tensor/reports/40 |
| m22 SUBOFF B1 工件 | `/root/tp-agent/work/m22/artifacts/suboff_bench1_cumulant_sdaa20.json` |
| m22 SUBOFF B2 工件 | `/root/tp-agent/work/m22/artifacts/suboff_bench2_rans_re1e5_sdaa21.json` |
| m22 SUBOFF B3 工件 | `/root/tp-agent/work/m22/artifacts/suboff_bench3_rans_re1e4_sdaa22.json` |
| m22 floating_body 工件 | `/root/tp-agent/work/m22/artifacts/sdaa_floating_body_free_decay.json` |
| m24 laplace 工件 | `/root/tp-agent/work/m24/artifacts/sdaa_laplace_cac_phasefield.json` |
| m24 NVIDIA 参考 | `/root/tp-agent/work/m24/benchmarks/verified/laplace_cac_phasefield/result.json` |

---

*质检完成: 2026-10-29, mission 128*
