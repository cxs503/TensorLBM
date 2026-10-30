# Mission 36 补报：议事会 session-001 架构视角提案

> **mission_id**: 36 · **repo**: TensorLBM · **状态**: honest-FAIL（提案已交付，本报告为补发）
> **日期**: 2026-10-29（补报） · **提案原文**: `docs/council/session-001/r1-proposal-strategy.md`

---

## 1. 目的

SDAA 平台（node099，32 卡满编）已完成 2D Couette 跨卡逐位复现（m10）与基础能力普查（m11），owner 给出大方向但具体安排归议事会。本任务以**架构视角**独立提出 2–4 个候选任务方向，含 mission 粒度拆解、32 卡三阶段分配表、风险依赖与放弃条件，为 session-001 终审提供决策依据。

核心问题：
1. 如何从已验证的 2D 基准沿阶梯路线推进到 3D 船型阻力预报？
2. 如何建立跨库（LBM↔FVM↔BEM）耦合框架为多物理场打基础？
3. 如何清偿技术债（Triton 消除 + 跨卡框架固化）以解锁全速路径？

---

## 2. 方法

### 2.1 提案框架

采用 **wedge S0→S4 / SI S0→S3** 阶梯范式，每个方向按阶段递进，每阶段配机器判据与放弃条件：

| # | 方向 | 范式 | 峰值卡位 | 预算量级 |
|---|------|------|---------|---------|
| A | 船舶阻力物理战役楔形 | wedge S0→S4 | 12 | ~10 人日 |
| B | 跨库耦合阶梯 LBM↔FVM↔BEM | SI S0→S3 | 8 | ~8 人日 |
| C | 技术债清偿：Triton 消除 + 跨卡框架固化 | 内部重构 | 4 | ~5 人日 |

### 2.2 数据基础

提案以仓库已有 benchmark 工件为数字基础（零手抄，全部程序化提取）：

- **SUBOFF benchmark 系列**（`artifacts/suboff_bench{1,2,3}_*.json`, `suboff_best_rans_summary.json`）
- **KBC 诊断**（`artifacts/kbc_diagnostic_16cube_20steps.json`）
- **已有文档**：`docs/FORCE_METHODS_SURVEY.md`, `docs/L1_BLOCK_SHARDING_DESIGN.md`, `docs/triton_distributed_notes.md` 等 80+ 篇

### 2.3 32 卡三阶段分配

| 阶段 | 时间窗 | A 阻力战役 | B 耦合阶梯 | C 技术债 | 扫荡线 | 值班 | 合计 |
|------|--------|-----------|-----------|---------|--------|------|------|
| P1 | day 1-2 | 8 (S0-S1) | 4 (S0) | 4 (M1-M2) | 10 | 2 | 28 |
| P2 | day 3-5 | 12 (S2-S3) | 4 (S1) | 4 (M3-M4) | 8 | 2 | 30 |
| P3 | day 6-10 | 12 (S4) | 8 (S2-S3) | 0 | 8 | 2 | 30 |

---

## 3. 关键结果

### 3.1 已有 benchmark 数字（提案引用的证据基线）

以下数字全部从 `artifacts/*.json` 程序化提取，零手抄：

| Benchmark | 碰撞模型 | 网格 | Re | Cd_total | Cf_ref | 误差% | 耗时(s) | 状态 |
|-----------|---------|------|-----|----------|--------|-------|---------|------|
| B1: 4L+Cumulant | Cumulant D3Q19 | 320×120×120 | 1,000 | 0.04179 | 0.04200 (Blasius) | **0.50** | 1983 | ✅ PASS |
| B2: RANS k-ε | RANS k-epsilon (MRT) | 200×80×80 | 100,000 | 0.00662 | 0.00833 (ITT-57) | **20.5** | 1038 | ⚠ >15% |
| B3: RANS k-ε | RANS k-epsilon (MRT) | 200×80×80 | 10,000 | 0.01065 | 0.01875 (ITT-57) | **43.2** | 1028 | ❌ FAIL |

**关键发现**（提取自 `suboff_best_rans_summary.json → key_findings`）：
1. Cumulant D3Q19 on 4L domain achieves **0.5% error** at Re=1000 — best SUBOFF result, 4.6× better than MRT+Smag (2.3%)
2. RANS k-epsilon at Re=1e5: **20.5% error** — slightly better than Smagorinsky (23.9%), worse than Vreman (16.4%)
3. RANS k-epsilon at Re=1e4: **43.2% error** — much worse than Smagorinsky (20.6%), model over-generates eddy viscosity at transitional Re
4. RANS k-epsilon significantly under-predicts friction drag at both high Re — model needs wall functions
5. All 3 simulations stable (no divergence), 5000 steps completed

### 3.2 KBC 诊断数字（16³ cube, 20 steps, Re=100）

| 指标 | BGK | KBC | 参考 |
|------|-----|-----|------|
| 终态 Cd | −3.034 | 22.062 | 1.092 |
| Cd 比值 KBC/BGK | — | −7.27 | — |
| H-定理违反格点数 | N/A | 105 cells (max excess=5.96e-08) | 0 |
| h_norm 峰值 | N/A | 1.019e-02 | — |
| gamma 范围 | N/A | [−0.953, −0.953] (std=0) | — |

**诊断结论**：KBC 的 H-mode 未被有效松弛（h_norm 仅衰减 30%），Cd 严重偏离参考值，表明 KBC 碰撞在低分辨率短步数下尚未收敛到物理态。此诊断支撑方向 A 中 S1+ 阶段需优先验证 KBC/A-regularization 在 SDAA 上的行为。

### 3.3 提案产出工件清单

| 文件 | 内容 | 行数 |
|------|------|------|
| `docs/council/session-001/r1-proposal-strategy.md` | 完整提案（3 方向 + mission 拆解 + 卡位表 + 风险） | 266 |
| `artifacts/suboff_bench1_cumulant_sdaa20.json` | B1 benchmark 原始数据 | — |
| `artifacts/suboff_bench2_rans_re1e5_sdaa21.json` | B2 benchmark 原始数据 | — |
| `artifacts/suboff_bench3_rans_re1e4_sdaa22.json` | B3 benchmark 原始数据 | — |
| `artifacts/suboff_best_rans_summary.json` | 三 benchmark 汇总 + key_findings | — |
| `artifacts/kbc_diagnostic_16cube_20steps.json` | KBC vs BGK 诊断 | — |

---

## 4. 审计

### 4.1 提案完整性检查

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 2-4 个候选方向 | ✅ | 3 个（A/B/C），符合 2-4 要求 |
| Mission 粒度拆解 | ✅ | A: 5 missions (S0-S4), B: 4 missions (S0-S3), C: 4 missions (M1-M4) |
| 机器判据 | ✅ | 每个 mission 有量化判据（误差%/L2/步数） |
| 32 卡分配表 | ✅ | 三阶段分配，峰值 ≤30 卡，留 2 卡弹性 |
| 风险与依赖 | ✅ | 每方向 2-3 条风险 + 缓解措施 |
| 放弃条件 | ✅ | 每方向 2-3 条明确放弃条件 |
| 数字来源标注 | ✅ | 全部引用 00_context.md 或 artifacts/*.json |
| 不确定项标注 | ✅ | 5 条 [待验] 项诚实标注 |

### 4.2 数字溯源审计

| 数字 | 来源文件 | 提取方式 | 可复现 |
|------|---------|---------|--------|
| B1 Cd_total=0.04179, error=0.50% | `suboff_bench1_cumulant_sdaa20.json` | `json.load()['Cd_total']` | ✅ |
| B2 Cd_total=0.00662, error=20.5% | `suboff_bench2_rans_re1e5_sdaa21.json` | `json.load()['Cd_total']` | ✅ |
| B3 Cd_total=0.01065, error=43.2% | `suboff_bench3_rans_re1e4_sdaa22.json` | `json.load()['Cd_total']` | ✅ |
| KBC Cd=22.062, BGK Cd=−3.034 | `kbc_diagnostic_16cube_20steps.json` | `json.load()['kbc_final_Cd']` | ✅ |
| H-定理违反 105 cells | 同上 | `json.load()['findings'][0]` | ✅ |

### 4.3 与 00_context.md 一致性

提案声明"数字引用 00_context.md 已核实记录"——由于 read_file 路径越界无法直接比对 00_context.md 原文，但所有 benchmark 数字均从 artifacts/*.json 程序化提取，与提案正文一致。

---

## 5. 局限与诚实标注

### 5.1 本提案的局限

1. **预算量级为人日估算**：基于 m10（2D Couette 1 天）和 m11（普查 1 天）经验外推，3D 和耦合复杂度更高，实际可能超期 30-50%。
2. **5 条 [待验] 项未实际验证**：TensorFVM/BEM SDAA 兼容性、STL 体素化 SDAA 路径、自由表面模块 SDAA 可跑性、TensorPhysics coupling driver 骨架——均为文档推断，需实际 mission 验证。
3. **KBC 诊断仅为 20 步短跑**：16³ cube Re=100 仅 20 步，Cd 未收敛（BGK=−3.03, KBC=22.06 vs ref=1.09），不能代表长时间物理态，仅用于诊断碰撞算子行为。
4. **RANS k-epsilon 在高 Re 误差 >15%**：B2 (20.5%) 和 B3 (43.2%) 均超精度门限，表明当前 RANS 模型需 wall function 修正才能用于生产。

### 5.2 补报说明

本单（m36/m92）为**补报单**——原 mission 36 已成功交付（提案文件 + 工件均已写入），但缺论文级进展报告。本报告基于已有工件程序化提取数字，未修改任何业务代码。

### 5.3 honest-FAIL 声明

原单交付状态为 **honest-FAIL**：提案已完成且质量达标，但部分 [待验] 项未实际执行验证实验。这是提案阶段的固有局限——提案的价值在于路线规划与风险预判，而非执行。

---

*报告生成时间: 2026-10-29 · 数字提取: python3 json.load 程序化 · 零手抄*
