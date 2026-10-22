# dam_break_sc 终裁文档：TERMINAL

**案例**: 2D 溃坝 dam_break_sc (Shan-Chen 单组分伪势, Martin & Moyce 1952)  
**终裁**: TERMINAL (结构性本构/密度比缺口，非参数调优可解)  
**日期**: 2026-10-22  
**Mission**: m163  

---

## 1. 物理问题与参考

**问题**: Martin & Moyce (1952) 2D 溃坝实验——液柱坍塌前沿 X=x_front/a 对无量纲时间 T=t·√(2g/a)  
**参考**: Lethe 官方后处理脚本同源的数字化表（14 点，T∈[0,2.96]）  
**检查点**: T=1.0 → X=1.326, T=2.0 → X=2.360, T=2.96 → X=3.670 (线性内插)  
**模型**: 2D SCMP 伪势 (psi=exp(-ρ), G_lib=+5.0 → G_eff=-5.0, τ=1.0)  
**共存密度**: rho_l=1.957, rho_v=0.1596, 密度比=12.3  
**重力**: g=2e-4 (浮力修正 g_eff=1.837e-4)  

---

## 2. 基线结果 (result.json official_rerun_correct_convention)

| 网格 | T=1 err% | T=2 err% | T=2.96 err% | 质量漂移 | 通过? |
|------|----------|----------|-------------|----------|-------|
| a=40 | +10.526    | -7.579   | -19.889     | 1.43e-05 | ❌    |
| a=80 | +2.49      | -9.723   | -19.086     | 1.4e-05  | ❌    |

**关键观察**:
- 仅 T=1 在 a=80 过门 (+2.49% ≤ 3%)
- T=2 和 T=2.96 系统性偏慢 (-9% 到 -19%)
- 两档收敛性差：T=1 档间差 7.84% (baseline) / 4.93% (Guo)

---

## 3. 已尝试杠杆与失败证据

### 3.1 Guo 力扫参 (a=80, 质量守恒源项)

| 臂 | T=1 err% | T=2 err% | T=2.96 err% |
|----|----------|----------|-------------|
| guo G=5.0 τ=0.8 | +5.66 | -7.75 | -17.04 |
| guo G=4.6 τ=0.8 | +31.11 | +16.61 | +1.01 |
| velocity_shift G=5.0 τ=0.8 | +4.71 | -7.75 | -16.70 |

**结论**: Guo 力与 velocity_shift 实质等价（质量守恒修正后 sum S=2.7e-09），"界面伪力"叙事不成立。

### 3.2 参数空间全扫 (README 记录)

- G ∈ [4.6, 5.2], τ ∈ [0.7, 1.0], g ∈ [1e-4, 3e-4]
- g=3e-4 时 T=2/2.96 恶化到 +35.87/+69.62%
- **无任何稳定 (G,τ) 三检查点同 ≤3%**

### 3.3 压力张量 SC 力 (Lever 1, m163 新试)

**配置**: scheme="pressure_tensor" (Ramshaw-Phathanapirom), forcing="velocity_shift", G=5.0, tau=1.0

**结果**:
- a=80: **NaN @ step 460** (不稳定)
- a=40: T=1 +5.22%, T=2 -12.48%, T=2.96 -27.25% (比 baseline 更差)

**结论**: 压力张量格式在溃坝工况下不稳定（细档 NaN），粗档误差更大。

### 3.4 高密度比 EOS (Lever 2, 评估)

**库能力**: `psi_eos_carnahan_starling` / `psi_eos_peng_robinson` 已落库  
**参考案例**: `verified/laplace_pr_eos` 在 T_r=0.55 达到密度比 123 (Maxwell) / 400-556 (模拟)  
**目标**: 水/空气密度比 ~800

**评估**:
- 需要 T_r ≈ 0.4-0.5 达到密度比 ~500-1000
- 但溃坝涉及大变形、自由表面、冲击，与静态液滴 Laplace 基准物理差异大
- 即使密度比提升，**结构性障碍仍在**：err(T=1) 与 err(T=2.96) 对 G 的依赖方向相反
- 预算评估：参数扫描 ~10 GPU·h，无保证过门

### 3.5 Guest-Host 耦合 (Lever 3, 未试)

**库能力**: `sc_two_component_force` / `collide_sc_two_component` 已落库  
**物理**: 双组分 SC 模型，液体+气体独立演化，cross-coupling G_12  
**评估**:
- 需要重新标定共存曲线、表面张力、迁移率
- 物理设置与单组分 SCMP 不等价（双组分有扩散、分凝）
- 预算评估：~20 GPU·h，引入新物理不确定性

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

### 4.3 两次撤回 (记录卫生)

1. **历史 +50~82% 误差**: 双重错误乘积——(i) 时间轴 T=t·√(g/a) 缺 √2；(ii) 参考表错（把 M&M T=3 值 2.7 映到 T=2）。已全部修正并撤回。

2. **"Guo 力是决定性杠杆"**: 原型 `/root/db_guo.py` 源项漏掉标准 -(u·F)/cs² 项，逐步净造质量 (sum S=-0.0123/步 → 1600 步漂移 0.11)。质量守恒修正后 Guo ≈ velocity_shift。

---

## 5. 四门判决

| 门 | 判决 | 证据 |
|----|------|------|
| G1 共性模块入口 | ✅ PASS | `tensorlbm.multiphase.collide_sc_single_component` / `psi_exp` / `solver.stream` / `boundaries.bounce_back_cells` |
| G2 直接观测量 ≤3% | ❌ FAIL | a=80: T=1 +2.49% (过), T=2 -9.723% (❌), T=2.96 -19.086% (❌) |
| G3 ≥2 档单调收敛 | ❌ FAIL | T=1 档间差 7.84% (baseline) / 4.93% (Guo) > 3% |
| G4 参考先锁定 | ✅ PASS | T_MM 轴 + Lethe 表 1.326/2.360/3.670 冻结于 2026-09-29 |

**总判决**: ❌ FAIL → TERMINAL

---

## 6. 晋级路径 (需 owner 决策)

### 路径 A: 高密度比伪势 (中等预算)
- **动作**: PR/CS EOS 参数扫描 T_r ∈ [0.4, 0.55]，目标密度比 500-1000
- **预算**: ~10 GPU·h (共存密度测量 + 溃坝重跑)
- **风险**: 即使密度比提升，反向 G 依赖的结构性障碍仍在
- **成功条件**: 找到 (EOS, T_r, G, τ) 使三检查点同 ≤3% 且单调收敛

### 路径 B: 双组分 SC 模型 (高预算)
- **动作**: 切换到 `collide_sc_two_component`，液体+气体独立演化
- **预算**: ~20 GPU·h (共存曲线重测 + 参数扫描 + 溃坝重跑)
- **风险**: 引入新物理 (扩散、分凝)，与 M&M 单流体力学不等价
- **成功条件**: 证明双组分模型在 M&M 工况下退化为单流体极限

### 路径 C: 引擎级改造 (极高预算)
- **动作**: 自由表面模型 (Körner 2006) 或界面追踪 (front-tracking)
- **预算**: ~50-100 GPU·h (新模型开发 + 验证)
- **风险**: 需要引擎级改造，非参数调优
- **成功条件**: 实现密度比 ∞ (纯液体 + 无质量气体) 极限

### 路径 D: 维持 TERMINAL (零预算)
- **动作**: 记录负结果，归档到 pending/
- **价值**: 为后续研究提供基线 (SCMP 在溃坝工况的局限性)
- **推荐**: 短期最优 (预算纪律 + 诚实终态)

---

## 7. 审计溯源

### 7.1 PROVENANCE 块 (机器溯源)

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

### 7.2 复现命令

```bash
# 基线重跑
cd benchmarks/pending/dam_break_sc
PYTHONPATH=../../../src python run.py --a {40,80} --forcing velocity_shift --device cpu

# 压力张量方案 (m163 新试)
PYTHONPATH=../../../src python run.py --a {40,80} --force-scheme pressure_tensor --device cpu

# Guo 力扫参
PYTHONPATH=../../../src python run.py --a 80 --forcing guo --g-coupling {4.6,5.0} --tau 0.8 --device cpu
```

### 7.3 工件清单

- `result.json`: 基线结果 + Guo 扫参 + 参考表
- `case_a{40,80}.json`: 两档正式运行结果
- `hist_a{40,80}.csv`: 时间序列 (X_toe, X_glob, H, max_u, mass_drift)
- `guo_sweep.json`: Guo 力参数扫描
- `force_form_sweep.json`: 压力张量方案扫描 (m163)

---

## 8. 局限与缓解

### 8.1 局限

1. **密度比不足**: SC94 psi_exp 在 G=5 给出密度比 12.3，远低于 M&M 水/空气 ~800
2. **反向 G 依赖**: err(T=1) 与 err(T=2.96) 对 G 的依赖方向相反，单标度调参不可解
3. **压力张量不稳定**: Ramshaw-Phathanapirom 格式在溃坝工况下 NaN (细档)
4. **未试 Lever 2/3**: 高密度比 EOS 与 guest-host 耦合因预算限制未实际运行

### 8.2 缓解

1. **诚实终态**: 记录负结果，避免拼凑数字硬过门
2. **完整证据链**: 所有判决数字走 PROVENANCE 值-文件-键三元组，零手抄
3. **明确晋级路径**: 给出四条路径 (A/B/C/D) 与预算评估，供 owner 决策
4. **机器门禁**: gates.json 三元组完整，gates_check.py 可复算

---

## 9. 结论

dam_break_sc 案例因**结构性本构/密度比缺口**无法在现有 SCMP 框架内闭环：

1. 密度比 12.3 远低于物理目标 800，导致气体拖曳效应显著
2. err(T) 对 G 的反向依赖使单标度调参不可行
3. 压力张量格式在溃坝工况下不稳定
4. 高密度比 EOS 与双组分模型需大量参数扫描，预算风险高

**终裁**: TERMINAL (根因=本构/密度比缺口 + 反向 G 依赖，非参数调优可解)  
**推荐**: 路径 D (维持 TERMINAL，记录负结果) 或路径 A (高密度比伪势，~10 GPU·h，有风险)

---

**文档版本**: v1.0 (2026-10-22)  
**作者**: tensor-agent (benchmark-closure line, mission m163)  
**审核**: 待 owner 验收
