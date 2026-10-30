# 质检报告：批次 283, 284（mission_id: 371）

**质检员**：TensorPlatform 船舶总体性能智能体（QA agent）  
**质检日期**：2026-10-30  
**复审对象**：报告 #283、#284（均为 m371 值班长 SDAA 扫荡线分析报告）

---

## 逐篇判定表

| 报告 ID | 标题（摘要） | 判定 | 理由 |
|---|---|---|---|
| 283 | 【SDAA 扫荡】值班长 m371 班次分析 — 窗口 40 单判读（案例类 28 单: PASS 13 / HONEST_FAIL 6 / gate-FAIL 9） | **RESUBMIT** | 数字溯源失败：GATE_FAIL 报 9 实为 10，HONEST_FAIL 报 6 实为 5（见下方缺陷清单） |
| 284 | 【SDAA 扫荡】值班长 m371 班次分析 — 窗口 40 单判读（案例类 28 单: PASS 13 / HONEST_FAIL 5 / gate-FAIL 10）（r283 修正版） | **MINOR** | 数字正确；结构缺显式"目的"与"局限"节；图件为分析统计图而非场量云图/比较曲线（后端分析任务可豁免但应写明） |

---

## 缺陷清单

### 报告 #283 — RESUBMIT

#### 缺陷 1：数字溯源失败（GATE_FAIL 计数错误）
- **报告原文**（§1 表格）：`案例类判定分布（mission 级） | {"GATE_FAIL": 9, "PASS": 13, "HONEST_FAIL": 6}`
- **工件证据**：`/root/tp-agent/work/m371/agent_work/m371/ledger.json` → `"verdict_counts": {"GATE_FAIL": 10, "PASS": 13, "HONEST_FAIL": 5}`
- **偏差**：GATE_FAIL 少报 1（9 vs 10），HONEST_FAIL 多报 1（6 vs 5）
- **根因**：r283 生成时 `analyze.py` 对 m330（suboff_re1000）的判定归类有误，将 GATE_FAIL 误归为 HONEST_FAIL；r284 已修正此问题（标题注明"gate-FAIL 单不再继承物理判定"）

#### 缺陷 2：结构缺节
- **缺失**：无显式"目的"（Purpose）节；无显式"局限"（Limitations）节
- **说明**：§1 隐含目的，§8 升级事项触及局限，但未以标准五节结构呈现

#### 缺陷 3：图件类型不符
- **报告引用图件**：fig1_verdicts.png（判定构成柱状图）、fig2_case_attempts.png（尝试次数图）、fig3_ladders.png（误差阶梯图）
- **要求**：场量云图 + 比较曲线（与解析/文献/基线）
- **说明**：本报告为后端调度分析任务，非物理计算报告，无场量可绘。但报告未显式声明豁免理由。

---

### 报告 #284 — MINOR

#### 缺陷 1：结构缺节（同 r283）
- **缺失**：无显式"目的"与"局限"节
- **降级**：MINOR（数字已修正，核心内容完整）

#### 缺陷 2：图件类型不符（同 r283）
- **说明**：后端分析任务，场量云图不适用，但未写明豁免理由

#### 缺陷 3：误差判定节中部分案例无基准声明
- **原文**（§5）：`sphere_re100 | ladder within_gate / monotone（布尔全绿，无数值百分比） | — | 无百分比可引（见来源报告）`
- **问题**：未显式声明"无基准"原因，仅说"无百分比可引"
- **降级**：MINOR（已说明无数值，但未严格遵循"无基准须声明原因"格式）

---

## 数字溯源抽查记录

| 抽查数字 | 报告 283 值 | 报告 284 值 | 工件路径 | 工件值 | 283 命中 | 284 命中 |
|---|---|---|---|---|---|---|
| GATE_FAIL 计数 | 9 | 10 | `agent_work/m371/ledger.json` → `verdict_counts.GATE_FAIL` | 10 | ❌ | ✅ |
| cavity nx=32 max_abs_dev | 2.25% | 2.25% | `agent_work/m371/ledger.json` → missions 中 cavity 条目 snippet | 2.25% | ✅ | ✅ |
| womersley H=29,α=8 complex_rel | 4.890% | 4.890% | `agent_work/m371/ledger.json` → missions 中 womersley 条目 snippet | 4.890% | ✅ | ✅ |
| suboff_re1000 Cd_tot vs 基准 | 46.43% | 46.43% | `agent_work/m371/agent_work/m371/sweep_dump.txt` / `analyze.py` | 46.43% | ✅ | ✅ |
| HONEST_FAIL 计数 | 6 | 5 | `agent_work/m371/ledger.json` → `verdict_counts.HONEST_FAIL` | 5 | ❌ | ✅ |

**结论**：r283 抽查 5 项中 2 项不命中 → RESUBMIT；r284 抽查 5 项全部命中 → 数字溯源通过。

---

## honest-FAIL 保留检查

- **r283**：报告中对 `aircraft_icing_rg15_ipw2`、`startup_poiseuille`、`suboff_re1000`、`womersley` 均标记为 HONEST_FAIL，未粉饰为 PASS → ✅ 通过
- **r284**：同上，且修正了 gate-FAIL 归类 → ✅ 通过

---

## 结论不超证据检查

两篇报告均严格基于 API 返回的 mission 数据做程序化分析，未发现超出数据支撑的断言。§8 升级事项明确标注"本班未自行改判据/门限/审批"，符合诚实报告纪律。

---

## 处置建议

| 报告 ID | 判定 | 处置 |
|---|---|---|
| 283 | RESUBMIT | 派返工单：重写报告，修正 GATE_FAIL/HONEST_FAIL 计数错误 |
| 284 | MINOR | 仅记录，不派单（r284 已是 r283 的修正版，数字已正确） |

> 注：r284 作为 r283 的修正版已发布，数字错误已修复。MINOR 缺陷（结构缺节、图件豁免未声明）为格式级问题，不影响核心结论，故不派返工单。
