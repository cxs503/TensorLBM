# MP-DAMBREAK S0 现状审计（mission m206）

> 线手册：`/nfs/wangxi/tensor_eco/agent_work/mp_dambreak/README.md`
> 审计日期：2026-10-23；worktree：`agent/m206` @ main（基于 9fa6050e）
> 纪律：所有数字当场用工具直读工件抄录，标注 file:line / commit / 路径。无凭空数字。

---

## 1. benchmarks/pending 溃坝案例审计

### 1.1 当前 worktree 状态

当前 worktree（`agent/m206` @ 9fa6050e）的 `benchmarks/pending/` 目录下**不存在** `dam_break_sc` 和 `dam_break_3d_mm` 目录（`find benchmarks/pending -type d` 输出无溃坝条目）。两个案例的终裁工件仅存在于 git 历史的特定 commit 中，尚未合入 main：

| 案例 | 终裁 commit | merge PR | main 是否包含 |
|------|------------|----------|--------------|
| dam_break_sc | `d6588f1e` (m163 TERMINAL) | PR #391 | ❌ 未合入 |
| dam_break_3d_mm | `2d00e392` (m91 TERMINAL) | — | ❌ 未合入 |

> **注意**：`benchmarks/bench_dam_break.py`、`benchmarks/bench_dam_break_3d.py`、`benchmarks/bench_dam_break_quant.py` 在 main 中存在（已用 `ls -la benchmarks/bench_dam_break*.py` 确认），但这些是 **benchmark suite 驱动脚本**，不是案例工件目录。它们引用的参考值为旧口径（`X*(t*) ≈ 1 + t*`，即 T=t·√(g/a) 缺 √2 的错误版本），已在 m163 中被撤回。

### 1.2 dam_break_sc（m163 终裁，TERMINAL）

**工件路径**（git commit `d6588f1e`）：
- `benchmarks/pending/dam_break_sc/README.md` — 案例文档，状态行 `⏸️ TERMINAL 终裁（2026-10-22 m163 闭环）`
- `benchmarks/pending/dam_break_sc/gates.json` — 六门三元组（4/6 PASS）
- `benchmarks/pending/dam_break_sc/result.json` — Lethe 14 点数字化参考表 + 真跑结果
- `benchmarks/pending/dam_break_sc/terminal_verdict_m163.md` — 终裁详细报告
- `benchmarks/pending/dam_break_sc/terminal_verdict_report.md` — 终裁总结
- `report_dam_break_sc.md` — 平台报告
- `benchmarks/pending/dam_break_sc/figures/` — 3 张 PNG

**四门判决**（`git show d6588f1e:benchmarks/pending/dam_break_sc/gates.json`，直读）：

| 门 | 判决 | 关键数字 |
|----|------|---------|
| G1 共性模块入口 | ✅ PASS | `tensorlbm.multiphase.collide_sc_single_component` / `psi_exp` / `solver.stream` / `boundaries.bounce_back_cells`（run.py:42-44） |
| G2 直接观测量 ≤3% | ❌ FAIL | a=80: T=1 +2.49%（唯一过门）, T=2 -9.723%, T=2.96 -19.086%（result.json `official_rerun_correct_convention.a80`） |
| G3 ≥2 档单调收敛 | ❌ FAIL | T=1 档间差 7.84% > 3%（`two_grid_convergence_a40_vs_a80.baseline.T1.rel_err_pct`） |
| G4 参考先锁定 | ✅ PASS | Lethe 数字化表 14 点冻结于 2026-10-22 |
| G5 README | ✅ PASS | 15073B, 6 小节 |
| G6 图件 | ✅ PASS | 3 张曲线图 |

**终裁根因**（README line 3, terminal_verdict_m163.md §4）：
1. **本构/密度比缺口（主因）**：SC94 psi_exp 给出密度比 12.3 vs M&M 水/空气 ~800；ρ_v~0.16 的"气体"对前沿施加显著拖曳。
2. **反向 G 依赖（结构性障碍）**：err(T=1) 与 err(T=2.96) 对 G 单调依赖方向相反——降 G 加速晚期前沿但炸早期。G5.0→G4.6: T=1 +5.66%→+31.11%, T=2.96 -17.04%→+1.01%。
3. **压力张量格式不稳定**：Ramshaw-Phathanapirom scheme="pressure_tensor"，a=80 NaN@step 460。
4. **质量守恒已解决但非瓶颈**：Guo 力质量守恒修正后 sum S=2.7e-09、漂移 1.7e-05，但 Guo ≈ velocity_shift，"界面伪力"叙事不成立。

**参考锚点**（result.json `reference_digitisation_lethe`，直读）：
```
T: [0.00, 0.41, 0.84, 1.19, 1.43, 1.63, 1.82, 1.97, 2.20, 2.32, 2.50, 2.64, 2.82, 2.96]
Z: [1.00, 1.11, 1.23, 1.44, 1.67, 1.89, 2.11, 2.33, 2.56, 2.78, 3.00, 3.22, 3.44, 3.67]
检查点: T=1.0 → 1.326, T=2.0 → 2.360, T=2.96 → 3.670（线性内插）
```

### 1.3 dam_break_3d_mm（m91 终裁，TERMINAL）

**工件路径**（git commit `2d00e392`）：
- `benchmarks/pending/dam_break_3d_mm/README.md`
- `benchmarks/pending/dam_break_3d_mm/gates.json` — 四门判决
- `benchmarks/pending/dam_break_3d_mm/gates_verified.json`
- `benchmarks/pending/dam_break_3d_mm/validation_report.json` — 关键数+根因链
- `benchmarks/pending/dam_break_3d_mm/engine_design.md` — 引擎重设计提案
- `report_dam_break_3d_mm.md` — 平台报告

**mp_w10 worktree 镜像**（`/nfs/wangxi/worktrees/mp_w10/benchmarks/pending/dam_break_3d_mm/`）：该目录在 mp_w10 @ 1eae1c3b 中存在，含 README.md / result.json / run.py，是 m91 终裁前的原始工件。

**四门判决**（`git show 2d00e392:benchmarks/pending/dam_break_3d_mm/gates.json`，直读）：

| 门 | 判决 | 关键数字 |
|----|------|---------|
| G1 共性模块入口 | ✅ PASS | `tensorlbm.free_surface_lbm`（D3Q19, `free_surface_step`）原版 |
| G2 直接观测量 | ❌ FAIL | 12/12 检查点反收敛，质量门 -49.46%/-62.19% |
| G3 单调收敛 | ❌ FAIL | 细化全面恶化：T*=2: -2.248%→+7.529%（a40→a80） |
| G4 参考锁定 | ✅ PASS | M&M 1952 + FrankenSim n²=2，2026-10-01 controller final ruling |

**引擎级五连根因**（validation_report.json `root_causes`，直读）：
1. RC#1：重力双重施力（g_eff=1.5g@τ=1，库缺陷#2已登记）
2. RC#2：无滑移边界层掐死前锋
3. RC#3：双向账本劈裂（halo 晋升造质量+气/界面边界漏记）
4. RC#4：LIQ→IFACE 回转缺失（H 钉死在 1.0×）
5. **RC#5（结构性核心）**：前锋推进与账本闭合是同一个开关——当前引擎无任何臂能同时闭合账本并保住活前锋

**关键数**（validation_report.json `key_numbers`，直读）：
- mass_drift: a40 -49.4601%, a80 -62.1866%
- Z_err T*=2: a40 -2.248%, a80 +7.529%
- Z_err T*=4.5: a40 +4.728%, a80 +20.678%
- max|Z_err|: a40 6.942%, a80 20.678%
- inventory_drift: a40 +279.73%, a80 +339.19%

**mp_w10 原始 README**（`/nfs/wangxi/worktrees/mp_w10/benchmarks/pending/dam_break_3d_mm/README.md`）补充信息：
- 模型 = 库 Körner 型自由表面模块 `tensorlbm.free_surface_lbm`（D3Q19 `free_surface_step`，原版未修改）
- rho_gas=1.0（1:1 稳定化，非物理，已披露）
- 卡尺修正复测 a=16: X(T=1) err +81.82%, 质量漂移 -28.54%, 界面胞膨胀 17.2×
- 修复实验（去掉非保守的 interface↔interface 质量交换）后：interface 稳定，但 rho_gas=0.1 时质量漂移 +85%，rho_gas=1.0 时波前慢 15-44%

### 1.4 mp_w10 worktree dam_break_sc 工件

`/nfs/wangxi/worktrees/mp_w10/benchmarks/pending/dam_break_sc/` 下仅有 `run.py`（无 README/gates/result）。

**run.py 关键信息**（`cat` 直读）：
- 模型 = 2D SCMP 伪势（psi=exp(-ρ)，G_lib=+5.0 → G_eff=-5.0，τ=1.0）
- 共存密度：rho_l=1.957, rho_v=0.1596，密度比 12.3
- **参考值（旧口径）**：`MM = {1.0: 1.5, 2.0: 2.7}`（run.py:76）——这是 m163 撤回的错误参考表（把 M&M 的 T=3 值 2.7 映到 T=2）
- 时间轴（旧口径）：`T_raw = t*sqrt(g/a)`（缺 √2，run.py 注释 line 25）——已被 m163 修正为 `T_MM = t*sqrt(2*g/a)`

---

## 2. Wave-10 残值审计（mp_w10 @ 1eae1c3b，只读）

### 2.1 mp_w10 worktree 基线确认

```
cd /nfs/wangxi/worktrees/mp_w10 && git rev-parse HEAD
→ 1eae1c3b037f7637065142bbe6ec531deabbaf30
```

线手册明确"Wave-10 多相专项 M3（自由面溃坝）/M4（高密度比伪势）四轨未收官"。实际残值分布在两个位置：
- `/nfs/wangxi/worktrees/mp_w10/` — 源码树（只读，不被修改）
- `/nfs/wangxi/runs/bm_mp_20261001/` — 计算产物

### 2.2 M3 自由面溃坝轨（Wave-10）

**产物路径**：`/nfs/wangxi/runs/bm_mp_20261001/m3_freesurface/`

**核心工件**：`root_cause_chain.json`（`cat` 直读）

- **裁决**：`FAIL — formal ladder FAIL (12/12 checkpoints NON-MONO, anti-convergent), five root causes quantified, A-prime 3D probe FAILS ledger gate`
- **内核版本**：
  - kernel_2d: `src_patched/tensorlbm/free_surface_lbm_2d.py`, base = c3ac6216, git blob 689fac9a
  - kernel_3d: `src_patched/tensorlbm/free_surface_lbm.py`, verbatim c3ac6216 (1663 lines)
  - worktree: `/nfs/wangxi/worktrees/mp_w10 @ 1eae1c3b`（只读，从未修改）

**正式阶梯数据**（root_cause_chain.json `formal_fail_ladder`，直读）：

| 指标 | a=40 | a=80 | 趋势 |
|------|------|------|------|
| mass_drift_pct | -49.46 | -62.19 | 细化恶化 |
| inventory_drift_pct | +279.73 | +339.19 | 细化恶化 |
| iface 计数 (first/max/final) | 119/976/976 | 239/2051/2006 | 细化恶化 |
| Z_err T*=1 | +6.942% | +11.882% | 细化恶化 |
| Z_err T*=2 | -2.248% | +7.529% | 反号 |
| Z_err T*=4.5 | +4.728% | +20.678% | 细化恶化 |
| H_err τ=1 | +23.962% | +23.962% | 不变 |
| H_err τ=2 | +100.0% | +100.0% | 不变 |

**反收敛结论**（root_cause_chain.json `formal_fail_ladder.anti_convergence`）：
> refinement a40→a80 makes |Z err| WORSE at all 9 Z checkpoints (e.g. T*=2: -2.248 → +7.529; T*=4.5: +4.728 → +20.678) and mass drift worsens -49.46 → -62.19 with iface inflation 976 → 2006; **the defect scales WITH resolution**

**五根因**（root_cause_chain.json `root_causes`，直读）：
1. RC#1: gravity double-forcing (g_eff=1.5g@τ=1) — velocity shift (u+τg) PLUS Guo term (1-0.5/τ)·w·cu/cs² → g_eff=(2-0.5/τ)·g=1.5g@τ=1。修复：opt-in `TL_FS_GRAVITY_SCHEME=guo`，g_eff/g 从 1.522→1.043
2. RC#2: no-slip boundary layer chokes the front — 壁邻胞前锋被 bounce-back 阻尼
3. RC#3: 双向账本劈裂 — halo 晋升造质量+气/界面边界漏记
4. RC#4: LIQ→IFACE 回转缺失 — H 钉死 1.0×
5. RC#5: 前锋推进与账本闭合是同一个开关 — 结构性核心

### 2.3 M4 高密度比伪势轨（Wave-10）

**产物路径**：`/nfs/wangxi/runs/bm_mp_20261001/m4_pseudopotential/`

**核心工件**：`CLOSURE.md` + `prereg_v2.md`（冻结版）

**四档判决**（CLOSURE.md，直读）：

| 档 | 判决 | 关键数 |
|----|------|--------|
| (a) Maxwell 共存曲线 | **FAIL（披露型）** | CS 480×128: ρ_v err -1.1384%@0.95(过), -3.5536%@0.90, -7.7671%@0.85(超3%门); 达标温度=1<2 |
| (b) 高密度比 Laplace σ 自测 | **PASS（定案）** | PR@0.55: σ=0.32023/0.32649/0.32942 (L=128/192/256); 残差 0.045-0.100%; sim ratio 400-557; σ 全自测零外部输入 |
| (c) 伪电流 | 披露归档（无门） | u_max 2.0e-2@~5 → 2.57e-1@442(CS)/2.04e-1@480(PR), 高于良调文献锚 1e-3-1e-2 上限 5-25× |
| (d) 液滴振荡 | **FAIL（仅 tier1 精度）** | err(Lamb 主门) +2.02%/+1.06% 过3%门、单调过; tier1 σ_ω/ω=1.70%≥1% 冻结线 |

**库债清单**（CLOSURE.md §库债，直读）：
1. `multiphase.py` docstring "G<0 for phase separation" 在修正符号制式下陈旧（本库 G>0=吸引）
2. 库现役默认演示参数（sc+psi_exp G=-4 膜 96×32）26 步内爆
3. `psi_carnahan_starling`/`psi_peng_robinson` 符号反（ψ²=2(p_EOS-ρcs²)/cs² ⇒ p_eff=max 单调），参数硬编码非 YS2006 约定

**暂存修改**（`src_patched/tensorlbm/multiphase.py`）：
- 修正符号 ψ_EOS 两个（YS2006 逐字参数）
- `collide_sc_single_component` opt-in forcing 参数（"sc"默认位同旧实现；"edm"=精确差分；"guo"=完整 Guo-2002 半移位）
- 电池认证 PASS 6/6（sc/edm 对旧档逐位、guo 有限域逐位）
- diff 纯增量+一个 opt-in 参数，无既有行为改动

**关键物理数**（prereg_v2.md §2，直读）：
- CS EOS: p=ρRT(1+η+η²-η³)/(1-η)³-aρ², η=bρ/4, a=1, b=4, R=1
  - T_c=0.0943287031, ρ_c=0.1304438842, p_c=0.0044168134, Z_c=0.359
- PR EOS: p=ρRT/(1-bρ)-aρ²/(1+2bρ-b²ρ²), a=2/49, b=2/21, R=1
  - T_c=0.0729190372, ρ_c=2.6573041565, p_c=0.0595645555, Z_c=0.307
- CS 与 YS2006 发表值逐位一致（外部锚）

---

## 3. 自由面封闭现状审计

### 3.1 库内自由面/receiver 相关模块

**当前 worktree（agent/m206 @ main）的自由面模块清单**（`find src/tensorlbm -name 'free_surface*'`，直读）：

| 文件 | 行数 | 用途 |
|------|------|------|
| `free_surface_lbm.py` | 1265 | D3Q19 三维自由面主模块（生产路径 `free_surface_step`） |
| `free_surface_lbm_2d.py` | 138 | D2Q9 二维自由面模块 |
| `free_surface_common.py` | 899 | 自由面共性函数 |
| `free_surface_application.py` | — | 自由面应用层 |
| `free_surface_abb_inventory.py` | — | ABB 库存审计 |
| `free_surface_closure_experiment.py` | — | 封闭实验（自声明 DIAGNOSTIC_NOT_PHYSICAL_CLOSURE） |
| `free_surface_i_to_g_exact_ledger.py` | — | I→G 精确账本 |
| `free_surface_i_to_g_failed_residual_audit.py` | — | I→G 残差审计 |
| `free_surface_i_to_g_failure_policy.py` | — | I→G 失败策略 |
| `free_surface_ownership_ledger.py` | — | 所有权账本 |
| `free_surface_population_policy_evidence.py` | — | 群体策略证据 |
| `free_surface_production_evidence.py` | — | 生产证据观察器 |
| `free_surface_runtime_evidence_observer.py` | — | 运行时证据观察器 |
| `free_surface_topology_mutation_replay_contract.py` | — | 拓扑变异重放契约 |
| `free_surface_topology_transaction.py` | — | 拓扑事务 |
| `free_surface_population_transfer_plan.py` | — | 群体转移计划 |
| `free_surface_transaction_contract.py` | — | 事务契约 |
| `hull_free_surface.py` / `hull_free_surface_v2.py` | — | 船体自由面 |
| `full_wet.py` | — | 全湿 |

**关键源码行**：

`free_surface_lbm_2d.py`（`cat` 直读）：
- `init_flags_from_fill_2d` (line 42-50): fill≥1→LIQUID, 0<fill<1→INTERFACE, fill=0→GAS。**界面包络出生时 fill=0**。
- `free_surface_step_2d` (line 71-138):
  - line 703-724: Guo gravity force `(1.0 - 0.5/tau) * w_dev * cu_force / cs2`（3D 版本）
  - line 110: `to_iface = gas_mask & (fill > 0.01) & (~solid_mask)` — 阈值 0.01
  - line 120: `to_gas = ((flags == INTERFACE) | (flags == LIQUID)) & (fill <= 0.01)` — 阈值 0.01
  - line 131-138: gas→interface 晋升时 `fill[to_i] = 0.01`

`free_surface_lbm.py`（`grep` + `sed` 直读）：
- line 703: `# Guo gravity force (only when gravity is non-zero)`
- line 713-714: `Fx = rho_liquid * gx * ng; Fy = rho_liquid * gy * ng` — 重力以 rho_liquid 施加于 non_gas
- line 724: `f = f + (1.0 - 0.5 / tau) * w_dev * cu_force / cs2` — 标准 Guo 因子

**collide_sc_single_component**（`src/tensorlbm/multiphase.py:373`，`sed` 直读）：
- 参数：`f, G=-4.0, tau=1.0, psi_fn=psi_exp, gx=0.0, gy=0.0, solid_mask=None`
- **无 forcing 参数**（当前 main 版本）；M4 暂存版有 opt-in forcing（sc/edm/guo）
- line 396: `feq = equilibrium(rho, ux + tau * Fx / rho_s, uy + tau * Fy / rho_s)` — velocity_shift 方案

### 3.2 PR #229 与 Route D/A/B 线结论

**PR #229** = `a3bf9038 Merge pull request #229 from cxs503/docs/fs-route-d-evidence`

对应的决策文档：`/nfs/wangxi/worktrees/mp_w10/docs/free_surface_receiver_gate_redesign.md`

该文档记录了自由面 receiver gate 的三条路线，**全部执行完毕且全部失败**：

#### Route D（surgical revert: drop G, keep H1+H2）

- **执行日期**：2026-08-23
- **内容**：解除接收者门控 G（`fill > 1e-3`），保留 H1（to_i 仅邻接 LIQUID）+ H2（to_liq 重初始化）
- **测试**：3 臂，harness `runs/fs_route_d_20260823/run_arm.py`
- **结论**（line 199-200）：**Route D fails acceptance §4.1/§4.3**——静态柱质量漂移 +35~82%（与门控存在时相同的泵链）
- **verbatim**（line 199）：`Verdict: Route D fails acceptance §4.1/§4.3. Per §3's own fall-through, the redesign goes to Route A`

#### Route A（structurally correct: move block to to_gas）

- **执行日期**：2026-08-23/24
- **内容**：保留接收者开放，修复 `to_gas` 转换处的质量破坏——启用 opt-in `enable_i_to_g_ownership_closure`（step 5a, `build_i_to_g_ownership_transaction`），对空化界面胞做严格局部封闭
- **测试**：4 臂，harness `runs/fs_route_a_20260823/run_arm.py`
- **关键发现**（line 253）：**Route A fails acceptance §4 on all five criteria — it cannot take a single timestep**
  - 272 donors / 0 legal receivers / 0 donors with legal neighbours
  - WITHHELD_NO_POPULATION_TRANSFER（observer fail-closed 结果）
  - 拓扑转换无 f 群体转移（`i_to_g_population_owner_status="WITHHELD_NO_POPULATION_TRANSFER"`）
- **verbatim**（line 253）：`Verdict: Route A fails acceptance §4 on all five criteria — it cannot take a single timestep.`

#### Route B（initial-condition semantic fix: envelope born at epsilon > 0）

- **执行日期**：2026-08-24
- **内容**：`init_flags_from_fill(..., envelope_fill_epsilon=0.0)` 中设 eps>0，使界面包络出生时 fill>0 而非 fill=0
- **测试**：22 点 epsilon 扫描（0 到 0.05），`probe_step1_boundary.json`
- **关键发现**（line 326）：
  - eps ∈ [0, 2e-3]: WITHHELD (272 donors, 0 receivers)
  - eps ∈ [5e-3, 2.35e-2]: WITHHELD (256→112 donors, 16→160 receivers)
  - eps ≥ 2.4e-2: "ok" but closure never invoked (0 donors)
  - 生存边界 = 0.01 (to_gas line) + 1.375e-2 (max per-cell exchange debit) = 0.02375
  - 中间带 receivers 与 donors 空间反相关 → all-or-nothing policy 仍 withhold 全部候选
- **结论**（line 326）：`Verdict: Route B fails acceptance §4 on all five criteria and comes off the fallback list.`
- **终极结论**（line 326 末）：`With D, A and B all executed and dead, the only live line is the §6.2 closure redesign (receiver set with recv_new promotion / de-synchronised conversion wave / fill-approx-0 exchange guard / negative-donor policy / partial-commit answer to all-or-nothing).`

### 3.3 自由面生产证据（mp_w10 R1）

文档：`/nfs/wangxi/worktrees/mp_w10/docs/free_surface_production_evidence_r1.md`（`cat` 直读）

- 生产路径 = `free_surface_lbm.free_surface_step`
- `build_i_to_g_ownership_transaction` 发出独立质量 debit/credit links + receiver masks + exact same-dtype residual — **明确不是 f 群体转移**
- 拓扑代码将转换供体 populations 归零，文档声明 **no f transfer**
- R1 结论：`WITHHELD_NO_POPULATION_TRANSFER` — fail-closed 结果，不是"永远不可能有证据"的断言

### 3.4 FS2D 路线（dam_break_sc REPORT_FS2D_route.md）

文档：`benchmarks/pending/dam_break_sc/REPORT_FS2D_route.md`（git commit `b7465583`，`git show` 直读）

**四层根因结构**（§0，直读）：
1. 力方案双重计力（库缺陷，已修 commit 634203af）：g_eff=1.5g@τ=1 → Guo 方案精确 1.0g
2. 壁面镜面反射结构性缺陷（实现缺陷，已修 commit 634203af）：specular 源不可得 → 修复为流体侧镜面反射
3. refilling 方案（本轮定位）：默认 EQ 重填 vs 论文 AVG 重填（Lallemand & Luo eq. 28）
4. 湍流模型假设（本轮排除）：Smagorinsky CS=0.1 对 T=1 前沿完全免疫

**§11 waLBerla 外部对照**（直读）：
- 代码：waLBerla commit `01a28162`，Schwarzmeier & Rüde 2022 (AIP Advances 12, 035317)
- 7 组运行（全部 EXIT=0）
- **§11.2 关键结论**："Lethe 表"= Martin & Moyce 1952 **物理实验数据**（Lethe 官方脚本 `examples/multiphysics/dam-break/dam-break-2d.py` 硬编码 `x_exp`/`y_exp` 14 点与本战役 `reference_digitisation_lethe` 逐位相同，注释写明 "Experimental data from Martin & Moyce 1952"）
- **§11.3**：论文自己的代码在同一冻结门禁下 +5.9~+13.1 点全面不过
- **§11.5**：VTK 双口径交叉验证，测量约定差 ≤1.4 个百分点，门禁裁决不变

**waLBerla 原始数据**（`/nfs/wangxi/scratch/dbs_eos/walberla/runs/`）：
- 7 个 `breaking-dam.txt` 文件，4 列（step, T, Z_front, H_residual）
- 路径：`W40P_EQ/breaking-dam.txt` 等
- 注意：这是 waLBerla 模拟输出，不是 M&M 原文实验数据

---

## 4. 审计总结

### 4.1 当前状态一览

| 组件 | 状态 | 证据级 |
|------|------|--------|
| dam_break_sc (SCMP 伪势) | ⏸️ TERMINAL (m163) | gates.json 4/6 PASS, G2/G3 FAIL |
| dam_break_3d_mm (自由面 3D) | ⏸️ TERMINAL (m91) | gates.json 2/4 PASS, G2/G3 FAIL, 引擎级五连根因 |
| Wave-10 M3 (自由面 2D/3D) | FAIL | root_cause_chain.json 12/12 反收敛 |
| Wave-10 M4 (高密度比伪势) | 部分通过 | CLOSURE.md: (b) PASS, (a)(d) FAIL, (c) 披露 |
| Route D (自由面 receiver) | 死 | §7: 静态柱 +35~82% |
| Route A (自由面 to_gas) | 死 | §8: 无法执行单步 |
| Route B (自由面 envelope eps) | 死 | §9: 全 eps 域 WITHHELD 或不触发 |
| FS2D 路线 (2D 自由面) | 未达标但最接近 | §0: 最佳 +3.34/-1.62/-3.13，超 0.34/0/0.13 点 |
| waLBerla 外部对照 | 论文原码也不过 | §11: +5.9~+13.1 点全面 FAIL |

### 4.2 关键发现

1. **SCMP 伪势路线（dam_break_sc）已判 TERMINAL**：密度比 12.3 vs 物理 ~800 + 反向 G 依赖，非参数调优可解。但**参考锚点已冻结**（Lethe 14 点表 = M&M 1952 物理实验数据，§11.2 钉死）。

2. **3D 自由面路线（dam_break_3d_mm）已判 TERMINAL**：前锋推进与账本闭合是同一个开关（RC#5），12/12 检查点反收敛，缺陷随分辨率放大。需引擎级重设计。

3. **2D 自由面路线（FS2D）是唯一有希望的方向**：REPORT_FS2D_route.md 的最佳努力 +3.34/-1.62/-3.13，仅超门限 0.34/0/0.13 点。且 waLBerla 论文原码也不过（+5.9~+13.1 点），我们的实现比论文发表实现近 3~8 点。

4. **Route D/A/B 全部执行死**：自由面 receiver gate 的三条路线已穷尽，唯一活线是 §6.2 closure redesign（recv_new promotion / de-synchronised conversion wave / fill-approx-0 exchange guard / negative-donor policy / partial-commit）。

5. **M4 高密度比伪势 Phase-0 已完成**：Laplace σ 自测 PASS（PR@0.55 ratio 400-557），但未做过溃坝工况。暂存修改（修正符号 ψ_EOS + opt-in forcing）纯增量无行为改动，电池 PASS 6/6。

6. **参考锚点已可冻结**：Lethe 14 点表 = M&M 1952 物理实验数据（Lethe 官方脚本硬编码逐位相同），3 个检查点 T=1/2/2.96 → 1.326/2.360/3.670 已线性内插确定。
