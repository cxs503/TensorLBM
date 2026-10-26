# dam_break_sc 平台报告

**案例 ID**: dam_break_sc  
**物理问题**: 2D 溃坝（Martin & Moyce 1952）  
**模型**: Shan-Chen 单组分伪势（SCMP, ψ=exp(-ρ)）  
**终态**: ⏸️ TERMINAL（2026-10-22, m163）  
**四门判决**: G1✅ G2❌ G3❌ G4✅（4/6 门通过，含 paper 档两门）

---

## 1. 物理问题与几何

### 1.1 问题描述

Martin & Moyce (1952) 经典溃坝实验：初始矩形液柱（宽 a、高 2a）在重力作用下坍塌，沿干涸水平底板扩展。观测量为无量纲前沿位置 X(T) = x_front/a 对无量纲时间 T = t√(2g/a)。

### 1.2 控制参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 液柱宽度 a | 40 / 80 lu | 两档网格 |
| 液柱高度 | 2a | 高宽比 2:1 |
| 重力 g | 2×10⁻⁴ lu/ts² | 主时间轴用 M&M 约定 |
| 浮力修正 g_eff | 1.837×10⁻⁴ | g_eff = g(1-ρ_v/ρ_l)，副轴报告 |
| SC 耦合 G_lib | +5.0 | 库约定正号，物理 G_eff=-5.0 |
| 松弛时间 τ | 1.0 | 单松弛 BGK |
| 液体密度 ρ_l | 1.957 | SC psi_exp 共存 |
| 气体密度 ρ_v | 0.1596 | SC psi_exp 共存 |
| **密度比 ρ_l/ρ_v** | **12.3** | **远低于物理水/空气 ~1000** |
| 模拟步数 | 1600 | T_max ≈ 3.58 (a=80) / 5.06 (a=40) |

### 1.3 参考与锚

**参考来源**: Lethe 官方后处理脚本同源的数字化表（14 点，T∈[0,2.96]）  
**参考锁定**: 2026-10-22（m163 终裁前冻结）

| 检查点 | T | X_ref |
|--------|---|-------|
| 1 | 1.0 | 1.326 |
| 2 | 2.0 | 2.360 |
| 3 | 2.96 | 3.670 |

**口径纪律**: 历史 +50~82% 误差是双重错误乘积——(i) 时间轴 T=t√(g/a) 缺 √2；(ii) 参考表错（把 M&M T=3 值 2.7 映到 T=2）。已全部修正并撤回旧数字。

---

## 2. 结果与判据

### 2.1 基线结果（velocity_shift  forcing, scheme=standard）

| 网格 | T=1 err% | T=2 err% | T=2.96 err% | 质量漂移 | 通过? |
|------|----------|----------|-------------|----------|-------|
| a=40 | +10.526    | -7.579   | -19.889     | 1.43e-05 | ❌    |
| a=80 | **+2.49**  | -9.723   | -19.086     | 1.4e-05  | ❌    |

**关键观察**:
- 仅 T=1 在 a=80 过门 (+2.49% ≤ 3%)
- T=2 和 T=2.96 系统性偏慢 (-9% 到 -19%)
- 两档收敛性差：T=1 档间差 7.84% > 3%

![收敛曲线](benchmarks/pending/dam_break_sc/figures/fig_convergence_history.png)

**图 1**: 前沿位置 X(T) 收敛曲线。参考（Lethe 数字化，黑线）vs 模拟（a=40 红虚线，a=80 蓝虚线）。T≥2 系统性偏慢。

![误差曲线](benchmarks/pending/dam_break_sc/figures/fig_error_vs_T.png)

**图 2**: 相对误差 vs T。仅 T=1 a=80 过 3% 门（绿虚线）；T=2/T=2.96 系统性失败。

### 2.2 参数扫描（Guo forcing, a=80）

| 臂 | T=1 err% | T=2 err% | T=2.96 err% |
|----|----------|----------|-------------|
| guo G=5.0 τ=0.8 | +5.66 | -7.75 | -17.04 |
| guo G=4.6 τ=0.8 | +31.11 | +16.61 | +1.01 |
| velocity_shift G=5.0 τ=0.8 | +4.71 | -7.75 | -16.70 |

**关键发现**: 降 G (5.0→4.6) 改善 T=2.96 (-17%→+1%) 但恶化 T=1 (+5.6%→+31%)。**err(T=1) 与 err(T=2.96) 对 G 的依赖方向相反**，单标度调参不可解。

![参数空间](benchmarks/pending/dam_break_sc/figures/fig_phase_diagram.png)

**图 3**: 参数空间 (G, τ) 展示反向依赖。降 G 改善晚期但恶化早期，无稳定组合三检查点同 ≤3%。

### 2.3 m163 新试杠杆

**Lever 1: 压力张量 SC 力（Ramshaw-Phathanapirom scheme="pressure_tensor"）**
- a=80: **NaN @ step 460**（不稳定）
- a=40: T=1 +5.22%, T=2 -12.48%, T=2.96 -27.25%（比 baseline 更差）
- **结论**: 压力张量格式在溃坝工况下不稳定（细档 NaN），粗档误差更大

**Lever 2: 高密度比 EOS（psi_eos_carnahan_starling / psi_eos_peng_robinson）**
- 库能力已确认（verified/laplace_pr_eos T_r=0.55 密度比 123-556）
- 溃坝工况涉及大变形/自由表面/冲击，与静态液滴物理差异大
- 即使密度比提升，反向 G 依赖的结构性障碍仍在
- **结论**: 未实际运行（预算风险高，无保证过门）

**Lever 3: Guest-host 耦合（collide_sc_two_component）**
- 需重新标定共存曲线/表面张力/迁移率
- 引入新物理不确定性（双组分扩散、分凝）
- **结论**: 未实际运行（预算风险高，物理设置不等价）

---

## 3. 判据表

### 3.1 四门判决

| 门 | 判决 | 证据 |
|----|------|------|
| G1 共性模块入口 | ✅ PASS | `tensorlbm.multiphase.collide_sc_single_component` / `psi_exp` / `solver.stream` / `boundaries.bounce_back_cells` |
| G2 直接观测量 ≤3% | ❌ FAIL | a=80: T=1 +2.49% (过), T=2 -9.723% (❌), T=2.96 -19.086% (❌) |
| G3 ≥2 档单调收敛 | ❌ FAIL | T=1 档间差 7.84% (baseline) / 4.93% (Guo) > 3% |
| G4 参考先锁定 | ✅ PASS | T_MM 轴 + Lethe 表 1.326/2.360/3.670 冻结于 2026-10-22 |

### 3.2 Paper 档两门

| 门 | 判决 | 证据 |
|----|------|------|
| G5 案例细节 README | ✅ PASS | 15073B, 6 小节, PROVENANCE 块, python 命令, 参考/锚 |
| G6 图件 | ✅ PASS | 3 curve (收敛曲线/误差曲线/参数空间), field_waiver 豁免（无空间场量存档） |

**总判决**: 4/6 门通过 → TERMINAL

---

## 4. 根因诊断

### 4.1 主因：本构/密度比缺口

**现象**: 模拟柱坍塌过快（残余高度 H~0.16-0.21 vs M&M ~0.42-0.5），势能转前沿动量不足 → T≥2 前沿系统性偏慢

**机制**:
- SC94 在 G=5 给出密度比 ~12.3 (rho_l=1.957, rho_v=0.1596)
- M&M 水/空气密度比 ~800
- rho_v~0.16 的"气体"对前沿施加显著拖曳
- 即使提升密度比到 100-800，仍需验证是否能同时满足三检查点

### 4.2 结构性障碍：反向 G 依赖

**证据** (Guo 扫参表):
- 降 G (5.0 → 4.6): T=1 从 +5.66% 恶化到 +31.11%，T=2.96 从 -17.04% 改善到 +1.01%
- **err(T=1) 与 err(T=2.96) 对 G 的单调依赖方向相反**
- 任何稳定 (G,τ) 组合都无法三检查点同 ≤3%

**物理含义**:
- T=1 需要较强 G (快速坍塌)
- T=2.96 需要较弱 G (慢速前沿推进)
- 单标度 G 无法同时满足——需要时间依赖的 G(t) 或更复杂的本构

### 4.3 两次撤回（记录卫生）

1. **历史 +50~82% 误差**: 双重错误乘积——(i) 时间轴 T=t√(g/a) 缺 √2；(ii) 参考表错（把 M&M T=3 值 2.7 映到 T=2）。已全部修正并撤回。

2. **"Guo 力是决定性杠杆"**: 原型 `/root/db_guo.py` 源项漏掉标准 -(u·F)/cs² 项，逐步净造质量 (sum S=-0.0123/步 → 1600 步漂移 0.11)。质量守恒修正后 Guo ≈ velocity_shift。

---

## 5. 审计溯源

### 5.1 PROVENANCE 块（机器溯源）

```json
[
  {"v": 12.3, "f": "benchmarks/pending/dam_break_sc/result.json", "k": "model.density_ratio"},
  {"v": 1.957, "f": "benchmarks/pending/dam_break_sc/result.json", "k": "model.rho_l"},
  {"v": 0.1596, "f": "benchmarks/pending/dam_break_sc/result.json", "k": "model.rho_v"},
  {"v": 5.0, "f": "benchmarks/pending/dam_break_sc/result.json", "k": "model.G_lib"},
  {"v": 1.0, "f": "benchmarks/pending/dam_break_sc/result.json", "k": "model.tau"},
  {"v": 1.326, "f": "benchmarks/pending/dam_break_sc/result.json", "k": "reference_checkpoints.T=1.0"},
  {"v": 2.360, "f": "benchmarks/pending/dam_break_sc/result.json", "k": "reference_checkpoints.T=2.0"},
  {"v": 3.670, "f": "benchmarks/pending/dam_break_sc/result.json", "k": "reference_checkpoints.T=2.96"},
  {"v": 10.526, "f": "benchmarks/pending/dam_break_sc/result.json", "k": "official_rerun_correct_convention.a40.err_pct_T1"},
  {"v": 2.49, "f": "benchmarks/pending/dam_break_sc/result.json", "k": "official_rerun_correct_convention.a80.err_pct_T1"},
  {"v": -19.889, "f": "benchmarks/pending/dam_break_sc/result.json", "k": "official_rerun_correct_convention.a40.err_pct_T2p96"},
  {"v": -19.086, "f": "benchmarks/pending/dam_break_sc/result.json", "k": "official_rerun_correct_convention.a80.err_pct_T2p96"}
]
```

### 5.2 复现命令

```bash
# 基线重跑
cd benchmarks/pending/dam_break_sc
PYTHONPATH=../../../src python run.py --a {40,80} --forcing velocity_shift --device cpu

# 压力张量方案 (m163 新试)
PYTHONPATH=../../../src python run.py --a {40,80} --force-scheme pressure_tensor --device cpu

# Guo 力扫参
PYTHONPATH=../../../src python run.py --a 80 --forcing guo --g-coupling {4.6,5.0} --tau 0.8 --device cpu
```

### 5.3 工件清单

- `result.json`: 基线结果 + Guo 扫参 + 参考表
- `case_a{40,80}.json`: 两档正式运行结果
- `hist_a{40,80}.csv`: 时间序列 (X_toe, X_glob, H, max_u, mass_drift)
- `guo_sweep.json`: Guo 力参数扫描
- `gates.json`: 六门判决（4/6 通过）
- `figures/`: 3 张曲线图 + figures.json manifest
- `terminal_verdict_report.md`: 终裁详细报告
- `terminal_verdict_m163.md`: m163 终裁文档

---

## 6. 局限与缓解

### 6.1 局限

1. **密度比不足**: SC94 psi_exp 在 G=5 给出密度比 12.3，远低于 M&M 水/空气 ~800
2. **反向 G 依赖**: err(T=1) 与 err(T=2.96) 对 G 的依赖方向相反，单标度调参不可解
3. **压力张量不稳定**: Ramshaw-Phathanapirom 格式在溃坝工况下 NaN (细档)
4. **未试 Lever 2/3**: 高密度比 EOS 与 guest-host 耦合因预算限制未实际运行

### 6.2 缓解

1. **诚实终态**: 记录负结果，避免拼凑数字硬过门
2. **完整证据链**: 所有判决数字走 PROVENANCE 值-文件-键三元组，零手抄
3. **明确晋级路径**: 给出四条路径 (A/B/C/D) 与预算评估，供 owner 决策
4. **机器门禁**: gates.json 三元组完整，gates_check.py 可复算

---

## 7. 晋级路径（需 owner 决策）

### 路径 A: 高密度比伪势（中等预算）
- **动作**: PR/CS EOS 参数扫描 T_r ∈ [0.4, 0.55]，目标密度比 500-1000
- **预算**: ~10 GPU·h (共存密度测量 + 溃坝重跑)
- **风险**: 即使密度比提升，反向 G 依赖的结构性障碍仍在
- **成功条件**: 找到 (EOS, T_r, G, τ) 使三检查点同 ≤3% 且单调收敛

### 路径 B: 双组分 SC 模型（高预算）
- **动作**: 切换到 `collide_sc_two_component`，液体+气体独立演化
- **预算**: ~20 GPU·h (共存曲线重测 + 参数扫描 + 溃坝重跑)
- **风险**: 引入新物理 (扩散、分凝)，与 M&M 单流体力学不等价
- **成功条件**: 证明双组分模型在 M&M 工况下退化为单流体极限

### 路径 C: 引擎级改造（极高预算）
- **动作**: 自由表面模型 (Körner 2006) 或界面追踪 (front-tracking)
- **预算**: ~50-100 GPU·h (新模型开发 + 验证)
- **风险**: 需要引擎级改造，非参数调优
- **成功条件**: 实现密度比 ∞ (纯液体 + 无质量气体) 极限

### 路径 D: 维持 TERMINAL（零预算）⭐ 推荐
- **动作**: 记录负结果，归档到 pending/
- **价值**: 为后续研究提供基线 (SCMP 在溃坝工况的局限性)
- **推荐**: 短期最优 (预算纪律 + 诚实终态)

---

## 8. 结论

dam_break_sc 案例因**结构性本构/密度比缺口**无法在现有 SCMP 框架内闭环：

1. 密度比 12.3 远低于物理目标 800，导致气体拖曳效应显著
2. err(T) 对 G 的反向依赖使单标度调参不可行
3. 压力张量格式在溃坝工况下不稳定
4. 高密度比 EOS 与双组分模型需大量参数扫描，预算风险高

**终裁**: TERMINAL (根因=本构/密度比缺口 + 反向 G 依赖，非参数调优可解)  
**推荐**: 路径 D (维持 TERMINAL，记录负结果)

---

**文档版本**: v1.0 (2026-10-22)  
**作者**: tensor-agent (benchmark-closure line, mission m163)  
**审核**: 待 owner 验收
