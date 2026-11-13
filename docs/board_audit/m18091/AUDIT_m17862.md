# AUDIT m17862 — XVAL-LBM-IBM-EPSILON-GRAM-IMPLICIT-CORRECTION (交差核验)

审计任务 m18091 · 2026-11-13 · 对象 `TensorLBM` mission 17862 · 分支 `agent/m17862` commit `5dd45499`
裁判顺序：①三件套取证 → ②基准判据（`tensor_benchmark_cases`）逐项 → ③L0–L5 判级 → ④落表

**结论：level = 2（独立功能实现）。** 实现真实、数值可复现（独立重跑 max rel diff = 0），
移植语义经 OSS 源码行级复核无误；但**无单元测试、无集成调用方、无平台基准判据**，
物理臂 Cd 属瞬态量且文献带不可追溯 ⇒ L3/L4 均不成立。报告有两处**虚标**必须记录（下方 F1/F2）。

---

## ① 三件套取证

| 件 | 指针 | 状态 |
|---|---|---|
| impl | `src/tensorlbm/ibm_epsilon_gram.py`（11280 B，`git ls-files` 已跟踪；δ 核 L56-68，support L76-130，ds L138-153，Gram `compute_epsilon_gram` L161-193，spread L256-280，interp L283-305；向量化版 L201-248） | ✅ 真实、已入库 |
| entry / repro_cmd | `python3 xval/XVAL-LBM-IBM-EPSILON-GRAM-IMPLICIT-CORRECTION/run_xval.py`（219 行，sys.path 自挂 src，一键写 results.json） | ✅ 可执行（见下"重跑"） |
| run_artifacts | 同目录 `results.json`(4735 B)、`physics_A1.json`/`physics_A2.json`、`fig1_roundtrip.png`(910×585)、`fig2_bias_epsilon.png`(1170×520)、`fig3_cd.png`(780×546) — 全部 `git ls-files` 已跟踪 | ✅ 存在且为合法 PNG |
| tests | `grep -rn epsilon_gram tests/` → **0 命中**；仅有驱动内置 `selftest_vectorisation`（向量化 vs 循环 4.4e-16） | ⚠️ 无 pytest |
| integration callers | `grep -rn epsilon_gram src/ tests/ examples/ scripts/ benchmarks/` 除模块自身 → **0 命中**；未进 `src/tensorlbm/__init__.py`；未进 `ibm_capability_contract.py`；`ibm.py` / `ibm_force_transfer.py` 主路径未改 | ❌ 无集成 |
| external anchor | joconnor22/LIFE `2af2428`（本机 `/tmp/life_latest` 存在，`git log -1` 即 2af2428） | ✅ 见 ③ |

**独立重跑（审计方自跑，不用任务方 selftest）**：把 `run_xval.py` 复制到 `/tmp/repro_m17862/xval/`，
`PYTHONPATH=/root/tp-agent/work/m17862/src python3 run_xval.py` 全流程跑通（8 往返臂 + 2×700 步物理臂），
与本仓 `results.json` 逐叶比对 107 个数值：**max rel diff = 0.0**（完全复现）。

## ② 基准判据（唯一裁判）

```
SELECT case_name,metric,tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='XVAL-LBM-IBM-EPSILON-GRAM-IMPLICIT-CORRECTION';  --> 0 rows
（放宽 ILIKE '%IBM%' / '%EPSILON%' 亦 0 rows）
```
⇒ **平台侧无机器判据**（`benchmark.platform_rows=0`）。审计退而用报告自声明的预注册判据
（`predeclared.json`：identity 1e-6、owner 15%）+ 其自报 got 逐项核：

| 关键量 | ref | got | 误差 | <15% | <3% | 审计判定 |
|---|---|---|---|---|---|---|
| 往返 L2 N16 const A1 | 0（精确目标） | 1.52e-16 | 1.5e-16 | ✅ | ✅ | 复现一致 |
| 往返 L2 N16 radial A1 | 0 | 1.59e-16 | 1.6e-16 | ✅ | ✅ | 复现一致 |
| 往返 L2 N32 const A1 | 0 | 1.56e-16 | 1.6e-16 | ✅ | ✅ | 复现一致 |
| 往返 L2 N32 radial A1 | 0 | **2.19e-3** | 2.2e-3 | ✅ | ✅ | 复现一致；但**超预注册 1e-6 三个量级** |
| 往返 L2 N32 const/radial A2 | 0 | 4.60e-1 / 4.61e-1 | 46% | ❌ | ❌ | 复现一致（对照组） |
| Cd A1 (Re=100) | 自报带 1.5–1.9 | 1.6156 | 对带中值 −4.96% | 带内 ✅ | ❌ | 见 F2：带不可追溯 + 量为瞬态 ⇒ **WITHHELD** |
| Cd A2 | 同上 | 1.6303 | −4.1% | 带内 | ❌ | 同上 |

报告 ③ 表 20 个数值格 vs `results.json` 程序化比对：**20/20 一致**（零手抄成立）。

## ③ 外部锚点独立复核（LIFE 源码行级）

`/tmp/life_latest` = `2af2428`，逐行核报告引用的 6 条规则，**全部命中**：
`inc/Utils.h:220-231` 3 点 δ 分支（`(1+√(1-3r²))/3` / `(5-3r-√(-3(1-r)²+1))/6` / 0）；
`src/IBMNode.cpp:73-74` `F = force*epsilon*ds*1.0*diracVal`；
`src/Objects.cpp:265-317` `A[i][j] = Σ_s δ_i(s)·δ(|x_j-x_i(s)|)·δ(|y_j-y_i(s)|) × nodej->ds`、`b=1`、LAPACK、
`t==0 || flex==eFlexible` 重算节奏、`UNI_EPSILON` 分支。
移植实现 `ibm_epsilon_gram.py:178-193` 与之逐项同构。
审计另做 **δ 核 11 点独立重实现对照**（`artifacts/board_audit/m18091/verify_transplant.py`）：
max abs diff = **0.0**，1D 5 点骨架分割单位性 = 1.0（offset 0/0.25/0.5/0.75）⇒ 报告"11 点逐点一致"这句话**现在有工件了**
（原 mission result 里此检查无持久化工件）。

## ④ 判级（L0–L5）

| 级 | 判据 | 本单 | 依据 |
|---|---|---|---|
| L0 文档 | 报告 | 超 | — |
| L1 数学/单元验证 | 性质被验证 | 超 | 往返恒等 + δ 核 11 点 + 分割单位性 |
| **L2 独立功能实现** | 真代码 输入→计算→输出 | **✅ 达到** | 模块 305 行真算法（Gram 求解 torch.linalg.solve），驱动可跑，结果独立复现 max rel diff 0 |
| L3 系统集成 | 接入主求解器/平台 | ❌ | 0 调用方、未导出、未进 capability contract、`ibm.py`/`ibm_force_transfer.py` 未用 |
| L4 基准验证 | 完整算例+误差分析+可复现 | ❌ | 平台判据 0 行；物理臂 Cd 为瞬态量、文献带不可追溯（F2） |
| L5 科研工程 | — | ❌ | — |

## ⑤ 发现（按严重度）

**F1（material，claim 完整性）——"H1 成立"与自身预注册判据矛盾。**
`predeclared.json` 明写：`H1 holds iff A1 round-trip L2 < 1e-6 AND A1 total-force conservation rel < 1e-6`，
且 falsifier = `A1 roundtrip >= 1e-6 OR A1 conservation >= 15%`。
实测（`results.json`，已独立复现）：A1 总力守恒 = 2.52e-2（N16 const）、8.52e-1（N32 const）**⇒ falsifier 触发 ⇒ 按预注册 H1 被否定**；
且 N32 radial 往返 2.19e-3 > 1e-6，往返半句在该配置也不成立。
REPORT.md ⑥ 结论句 "**H1 成立**" 属虚标。数据本身在 ③/⑤ 已如实披露（非造假），但结论必须改为
"H1 部分成立：往返恒等在 N=16 与 N=32 常值场达机器精度；守恒半句被证伪"。
审计侧给出了守恒失败的**代数解释**：常值力场下沉积总力 / 目标总力 **≡ ε_mean**
（verify_transplant.json：N16 1.0251661790966031 / N32 1.8521660021352169，与 1+2.5% / 1+85% 逐位吻合）——
即 +85% 不是数值误差而是修正的固有代数代价。

**F2（material，基准效度）——物理臂 Cd 不构成基准验证。**
(1) 域是**周期盒 + 均匀初速 u=0.1、无进出口/无驱动力**，marker 环持续抽走动量：审计探针见 ⟨ux⟩ 单调衰减
（0.0987→0.0945 by step 400，仍在降），故"500–700 步时均 Cd"是**瞬态均值**，不是稳态阻力；
(2) 同一重复算中逐 50 步采样的瞬时 Cd 在 **0.39 ↔ 4.50** 间大幅摆动（非 Re=100 St≈0.165 的物理脱落周期），
200 步窗口均值对窗口位置/相位敏感；
(3) 文献带 `1.5–1.9`（"Norberg 1993; Williamson 1996"）在仓内无可追溯来源，而 2D 无界 Re=100 的通行值为
**Cd≈1.32–1.40**（Braza 1986；Norberg 1987/2003），本算例几何（D/H=12.5% 堵塞）修正后通常约 1.4–1.5。
按 5% 报告值和 1.5 下限比仍"带内"，但**该带本身虚高**，故 Cd 外部锚点判 **WITHHELD**（不判 PASS 也不判 FAIL）。
建议返修：改成带进出口的槽道 + 动量平衡读 Cd，或直接对 Cd=1.34（无界）做堵塞修正后作单点锚。

**F3（minor，工件/流程）**
- `results.json` 内 `physics.reference`（带/引用/n_avg）**不是 `run_xval.py` 的产物**：审计重跑只产出 A1/A2 两臂，107 数值叶子其它全同，
  该块系跑后人工追加 ⇒ "程序化注入零手抄"对这块不成立（数值未受影响）。
- 三张 PNG **无生成脚本**（`grep fig1_roundtrip --include=*.py` → 0）：图不可重生成，仅可核对与 results.json 一致。
- `predeclared.json` 物理臂写 `nu=0.08 -> tau=0.53`，实跑 `tau=0.524`（ν=0.008，Re=100）：预注册文本与执行配置不一致（Re 标签本身对）。
- `__all__` 只列 6 名，漏 `build_spread_matrix_2d`/`spread_force_2d_vec`/`interpolate_force_2d_vec`（驱动实际用 8 名）。
- "往返成为恒等"表述偏松：ε 解 `Aε=1` 只保证**常值力场**精确（审计实测 N32 单点脉冲往返 max err 0.112）；
  N32 radial 2.2e-3 正源于此，报告 ③ 数值无误、措辞应限定作用域。
- 模块未并入 main（`m18091` @ `fc86f508` 内无该文件），属正常合流延迟，不记缺陷。

## ⑥ 未做 / 局限

- 未运行 LIFE 二进制（C++/LAPACK/Boost + 完整 FSI 工况），移植为**语义级**对照（审计只做源码行级 + 数学性质复核）。
- 仅 2D；`UNI_EPSILON` 未移植（报告已声明）。
- 逐 700 步 Cd 轨迹的完整集群复算已递交 `cj1784`（排队中），审计现存瞬时证据来自本地 50 步采样探针
  （`artifacts/board_audit/m18091/probe_cylinder_stepwise.py`，中途 terminate 于高负载 load≈474）；F2 的结论不依赖该复算：
  ⟨ux⟩ 衰减与瞬时 Cd 摆幅在已采到的 0–400 步窗口内即可判定。
- 未复核 PNG 像素内容与 caption 的逐点对应（只核了尺寸/合法性与数值一致性）。