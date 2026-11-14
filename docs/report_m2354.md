# Mission 2354 补报：DG/FEM/SPH/DEM 四族验证欠账扫描（honest-FAIL meta-report）

**Mission ID**: 2354  
**补报单**: mission 2354-backfill  
**原始仓库**: TensorPapers（原单声明）  
**补报仓库**: TensorLBM（本工作树）  
**日期**: 2026-11-13  
**性质**: honest-FAIL · meta-report · 工件缺失声明

---

## 1. 案例介绍（物理背景 / 原单目标）

### 1.1 原单背景

Mission 2354 属于 W7 波二战役，目标是**补 DG/FEM/SPH/DEM 四族「验证欠账」**。原单声明指出：

> W7 波一（MISSION 2305，报告 #2301）明确留下「验证欠账」清单，本单选择逐条打开四族（DG、FEM、SPH、DEM）的验证状态，扫描提案池、报告局限节、guide_tracker 缺口，定位哪些能力停留在 MOCK/STUB 级别，哪些已有 REAL 证据但缺机器可证伪门。

### 1.2 原单工作范围

原单声明的工作内容包括：
1. 扫描 `/root/tp-agent/proposals/` 提案池
2. GET `/api/v1/tensor/reports?limit=20` 的「局限/待办」节
3. `guide_tracker` 缺口分析
4. 定位 DG/FEM/SPH/DEM 四族的验证状态

### 1.3 四族物理背景

| 族 | 物理覆盖 | 在 TensorLBM 中的角色 |
|---|---|---|
| **DG**（间断 Galerkin） | 高阶对流–扩散、可压缩冲击、混合 DG-LBM 贴体带 | `src/tensorlbm/dg_verification/`、`dg_advection.py`、`dg_lbm.py` |
| **FEM**（有限元） | 结构振动、稳性、冰区弯曲 | 本仓以 LBM 为主干，FEM 仅作为耦合接口（`tensordem_coupling_contract.md`） |
| **SPH**（光滑粒子流体） | 自由面大变形、溃坝、液舱晃荡 | 本仓无原生 SPH 求解器，仅通过 `THREE_WAY_LBM_DEM_FSI.md` 提及 LBM-DEM-FSI 三方耦合 |
| **DEM**（离散元） | 颗粒–流体耦合、冰–螺旋桨碰撞、球形沉降 | `tensordem_coupling_p1.md`、`propeller_ice_lbm_dem.md`、`sphere_settling_gcs_20261107.md` |

---

## 2. 怎么做的（原单方法 / 补报方法）

### 2.1 原单方法（据声明重构）

原单声明的方法论是**只读扫描 + 状态审计**：
- 不修改业务代码
- 通过读取提案池、报告、guide_tracker 定位欠账
- 输出应为「四族验证状态矩阵」+「改进路线图」

### 2.2 补报方法（本单实际执行）

本补报单的执行路径：
1. **登录平台**：POST `/auth/login` 获取 token（成功）
2. **扫描工件目录**：`find . -name "*2354*"` → **零匹配**
3. **扫描 docs/**：检索 DG/FEM/SPH/DEM 相关文档
   - DG：找到 `IN65_N04_dg_frontier.md`、`IN70_N01_dg_convection_diffusion.md`（已有独立验证面）
   - FEM：找到 `FE_VERIFICATION_STATUS_M14723.md`（自由能两相模型审计，非结构 FEM）
   - SPH：无原生 SPH 文档，仅 `THREE_WAY_LBM_DEM_FSI.md` 提及
   - DEM：找到 `tensordem_coupling_contract.md`、`propeller_ice_lbm_dem.md`、`sphere_settling_gcs_20261107.md`
4. **结论**：原单 mission 2354 **未产出任何工件**（无 JSON、无 NPZ、无 PNG、无 VERDICT）

### 2.3 工具链

- 文件系统扫描：`find`、`ls`、`grep`
- 文档阅读：`head`、`cat`
- 报告撰写：Markdown + 表格
- 平台交互：curl POST（本补报单）

---

## 3. 计算结果（数字表 · 程序化注入 · 零手抄）

### 3.1 原单工件清单（据本单扫描）

| 工件类型 | 预期路径 | 实际状态 | 备注 |
|---|---|---|---|
| JSON 结果 | `artifacts/IN2354/*.json` | **未留档** | `find` 零匹配 |
| NPZ 场量 | `artifacts/IN2354/fields_*.npz` | **未留档** | 无 |
| PNG 图表 | `artifacts/IN2354/figures/*.png` | **未留档** | 无 |
| VERDICT | `artifacts/IN2354/VERDICT.json` | **未留档** | 无 |
| 报告 Markdown | `docs/report_m2354.md` | **未留档** | 本补报单即为此 |

### 3.2 四族验证状态（据本仓现有文档推断）

| 族 | 验证面状态 | 机器门 | 证据等级 | 备注 |
|---|---|---|---|---|
| **DG** | ✅ 已建（IN70-N01、IN65-N04） | `tests/test_dg_verification.py`（38 tests） | REAL | 旋转锥/制造解/Sod/方波/质量账本均有 |
| **FEM** | ⚠️ 部分（自由能两相模型审计） | 无独立 FEM 结构验证门 | WITHHELD | `FE_VERIFICATION_STATUS_M14723.md` 为自由能模型，非结构 FEM |
| **SPH** | ❌ 无原生求解器 | N/A | NOT_SUPPORTED | 本仓以 LBM 为主干，SPH 仅作为耦合概念提及 |
| **DEM** | ⚠️ 耦合接口已定义 | 无独立 DEM 验证门 | WITHHELD | `tensordem_coupling_contract.md` 定义接口，但无独立验证 |

### 3.3 关键量误差判定

**原单未产出任何数值结果，故无误差可判。**

| 关键量 | 基准值 | 计算值 | 误差 | 是否 <3% | 备注 |
|---|---|---|---|---|---|
| （无） | — | — | — | — | 原单未执行计算 |

---

## 4. 结果图

**本补报单无可内嵌图表。**

理由：
1. 原单 mission 2354 未产出任何工件（JSON/NPZ/PNG 均缺失）
2. 本补报单为 meta-report，性质为「工件缺失声明」，不执行新计算
3. 四族验证状态为定性判断（REAL/WITHHELD/NOT_SUPPORTED），无需定量图表

若需定量图表，应：
- 重跑 IN70-N01（DG 验证）→ 生成收敛曲线
- 重跑 IN65-N04（DG frontier）→ 生成场量云图
- 为 FEM/DEM 建立独立验证面 → 生成对比图

---

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

### 5.1 原单 FAIL 原因分析

据本单扫描与推断，原单 mission 2354 可能 FAIL 的原因：
1. **范围过大**：同时扫描四族（DG/FEM/SPH/DEM）+ 提案池 + 报告 + guide_tracker，工作量超出单 agent 单次会话容量
2. **工件未落盘**：扫描结果可能仅存在于 agent 内存/对话历史，未写入 `artifacts/` 或 `docs/`
3. **无机器门**：原单声明的方法论是「只读扫描」，未设计可重复执行的验证脚本

### 5.2 本补报单局限

1. **无法还原原单扫描结果**：原单未留档，本补报单只能基于现有文档推断四族状态
2. **未执行新计算**：本补报单不重跑验证，仅声明工件缺失
3. **FEM/SPH 状态不确定**：本仓以 LBM 为主干，FEM/SPH 的独立验证面可能存在于其他仓库（TensorFEM、TensorSPH），本单未跨仓扫描

### 5.3 适用范围

本 meta-report 适用于：
- 追溯 mission 2354 的交付状态
- 定位四族验证欠账的当前状态
- 为后续补账单提供基线

不适用于：
- 作为四族验证的最终结论
- 替代 IN70-N01/IN65-N04 等已有验证面的报告

---

## 6. 误差判定（显式回答）

### 6.1 关键量 vs 基准误差

**无。** 原单未产出任何数值结果，故无误差可判。

### 6.2 是否 <3%？

**不适用。** 无计算值，无基准值，无法判定。

### 6.3 无基准声明原因

1. 原单 mission 2354 的方法论是「只读扫描」，不执行计算
2. 原单未产出任何工件（JSON/NPZ/PNG 均缺失）
3. 本补报单为 meta-report，不执行新计算

---

## 7. 结论与改进路线

### 7.1 结论

Mission 2354 为 **honest-FAIL**：
- 原单未产出任何工件
- 四族验证欠账扫描结果未留档
- 本补报单为 meta-report，声明工件缺失，不粉饰结果

### 7.2 改进路线

若需补账，建议：
1. **DG 族**：已有 IN70-N01/IN65-N04，无需补账
2. **FEM 族**：需在 TensorFEM 仓库建立独立验证面（板弯曲、梁振动）
3. **SPH 族**：需在 TensorSPH 仓库建立验证面（溃坝、液舱晃荡），或明确声明本生态不支持 SPH
4. **DEM 族**：需在 TensorDEM 仓库建立独立验证面（颗粒沉降、堆积角），当前仅有耦合接口

### 7.3 本补报单交付物

- ✅ `docs/report_m2354.md`（本文件）
- ✅ 平台报告 POST（scenario_key='report-backfill', mission_id=2354）
- ✅ finish 调用（summary 一句话报 POST 返回的 report id）

---

**报告结束。**
