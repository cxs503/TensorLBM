# DIT Turbulence 平台报告

**案例**: 衰减各向同性湍流 (Decaying Isotropic Turbulence)  
**Mission**: m167  
**状态**: ⏸️ TERMINAL（结构性 LES 模型缺口）  
**日期**: 2026-10-22

---

## 1. 物理问题

衰减各向同性湍流 (DIT) 是湍流研究的经典基准问题：

- **初始条件**: 3D 周期域，随机无散速度场，目标能谱 E(k) ∝ k⁴·exp(-2(k/m₀)²)
- **控制参数**: u_rms=0.04, τ=0.501 (分子 ν=10⁻³), Re_L≈65
- **物理过程**: 无外力自由衰减，能量从大尺度向小尺度级联
- **验证判据**:
  - 惯性区能谱斜率 vs Kolmogorov -5/3（误差 ≤3%）
  - 衰减指数 n ∈ [1.2, 1.5]（Saffman/Batchelor 带）
  - ≥2 档网格单调收敛

## 2. 计算设置

**共性模块入口**:
- `tensorlbm.solver3d.stream3d` (D3Q19 流场演化)
- `tensorlbm.turbulence.collide_smagorinsky_bgk3d` (固定 C_s=0.12 Smagorinsky LES)
- `tensorlbm.d3q19.equilibrium3d` + `macroscopic3d`

**网格配置**:
| 网格 | 分辨率 | 谱斜率误差 | 衰减指数 n | 状态 |
|------|--------|-----------|-----------|------|
| N=128 | 128³ | +1.69% | 1.449 | ✅ 通过 |
| N=256 | 256³ | +17.05% | 1.693 | ❌ 失败 |

## 3. 收敛性诊断

![收敛性诊断](benchmarks/pending/dit_turbulence/figures/dit_convergence_diagnosis.png)

**图 1**: DIT 收敛性诊断。(a) 谱斜率误差随网格变化：N=128 达标 (+1.69%)，N=256 超差 (+17.05%)；(b) 衰减指数随网格变化：N=128 在物理带内 (1.449)，N=256 超带 (1.693)；(c) 能谱对比：N=256 在惯性区明显偏平（欠耗散）。

**关键发现**:
- N=128 数值耗散恰好补偿 SGS 耗散不足，偶然达标
- N=256 数值耗散降低，SGS 耗散不足暴露，能谱偏平
- 档间不收敛是系统性问题，非统计噪声

## 4. 晋级路径评估

### 4.1 Dynamic Smagorinsky (Germano-Lilly)

**库入口**: `tensorlbm.turbulence.collide_dynamic_smagorinsky_bgk3d`

**评估结果**:
| 参数 | 稳定性 | 能谱斜率 | 衰减指数 | 判定 |
|------|--------|----------|----------|------|
| τ=0.501 | ❌ NaN@609步 | — | — | 不稳定 |
| τ=0.51 | ✅ 稳定 | -7.71 (-362%) | 1.73 | 灾难性失败 |

**根因诊断**:
1. **稳定性缺口**: 库实现在 τ=0.501（近零分子黏性）时失稳
2. **实现缺陷**: C_s² 采用全局平均（非标准 Germano-Lilly 局部动态过程）
3. **clamp 限制**: cs2 被 clamp 到 [0, 0.1]，抑制动态适应性

**结论**: ❌ 不可行（需重构库实现）

### 4.2 系综平均

**评估**: 系综平均（多 IC/多 seed）不能解决系统性偏差。当前问题是 LES 模型参数化的结构性缺陷，非统计噪声。

**结论**: ❌ 不适用

### 4.3 更长运行

**评估**: N=256 在 15000 步已充分衰减，延长运行不改变谱形状。欠耗散是结构性问题。

**结论**: ❌ 不适用

## 5. 四门判决

| 门 | 标准 | 结果 | 证据 |
|----|------|------|------|
| G1 共性模块入口 | src/tensorlbm 库路径 | ✅ PASS | `solver3d.stream3d` + `turbulence.collide_smagorinsky_bgk3d` |
| G2 直接观测量 ≤3% | slope err, n decay | ❌ FAIL | N=128: +1.69% (OK); N=256: +17.05% (FAIL) |
| G3 ≥2 档单调收敛 | 档间差 ≤3% | ❌ FAIL | slope 差 0.256 (15.4%), n 差 0.244 (16.8%) |
| G4 参考先锁定 | prereg append-only | ✅ PASS | 参考 Kolmogorov -5/3 先锁定 |

**综合判定**: FAIL → TERMINAL

## 6. 根因证据链

```
固定 C_s Smagorinsky
    ↓
分辨率↑ → 数值耗散↓ → SGS 耗散相对不足
    ↓
N=64:  过耗散 → 谱过陡（-1.646, -2.0%）
N=128: 数值耗散补偿 → 短暂穿越 -5/3（-1.639, +1.7%）
N=256: 欠耗散 → 谱太平坦（-1.383, +17.05%）
    ↓
档间不收敛（系统性，非统计）
    ↓
C_s 微调无效（N=256 对 C_s∈[0.12,0.13] 不敏感）
    ↓
Dynamic Smagorinsky 库实现缺陷（全局平均 + clamp 限制）
    ↓
TERMINAL：结构性 LES 模型缺口
```

## 7. 未来晋级路径

1. **重构 Dynamic Smagorinsky** (~10 GPU·h)
   - 实现标准 Germano-Lilly 局部动态过程（C_s²(x,t) 场，非全局标量）
   - 移除 cs2 clamp 限制（或放宽到 [-0.01, 0.1]，允许 backscatter）
   - 添加拉格朗日平均（Meneveau et al. 1996）稳定化

2. **实现其他 LES 模型** (~15 GPU·h)
   - WALE: C_w=0.325（已扫过，slope=-0.336，欠耗散死路）
   - Vreman: C_V=0.025（需评估）

3. **引擎级改造** (~50+ GPU·h)
   - 高阶格式（WENO/TVD）减少数值耗散
   - 自适应网格细化（AMR）解析惯性区
   - 隐式 LES（无显式 SGS，依赖数值耗散）

4. **维持 TERMINAL** (0 GPU·h，推荐)
   - 记录负结果：固定 C_s Smagorinsky 在 DIT 工况下跨网格不收敛
   - 待引擎级改造后重新评估

## 8. 审计溯源

**PROVENANCE 块**:
```json
[
  {"v": -1.6385711187348206, "f": "benchmarks/pending/dit_turbulence/result.json", "k": "verdict.slope_per_grid.128"},
  {"v": -1.3825001626134472, "f": "benchmarks/pending/dit_turbulence/result.json", "k": "verdict.slope_per_grid.256"},
  {"v": 1.6857, "f": "benchmarks/pending/dit_turbulence/result.json", "k": "verdict.slope_err_pct_per_grid.128"},
  {"v": 17.05, "f": "benchmarks/pending/dit_turbulence/result.json", "k": "verdict.slope_err_pct_per_grid.256"},
  {"v": 1.4491344383709526, "f": "benchmarks/pending/dit_turbulence/result.json", "k": "verdict.n_decay_per_grid.128"},
  {"v": 1.6932376806677203, "f": "benchmarks/pending/dit_turbulence/result.json", "k": "verdict.n_decay_per_grid.256"},
  {"v": -7.706362, "f": "benchmarks/pending/dit_turbulence/result_dsmag.json", "k": "verdict.slope_per_grid.128"},
  {"v": 1.7318, "f": "benchmarks/pending/dit_turbulence/result_dsmag.json", "k": "verdict.n_decay_per_grid.128"}
]
```

**复现命令**:
```bash
# 固定 C_s 基准（已有）
python run.py --n 128 --steps 15000 --cs 0.12
python run.py --n 256 --steps 15000 --cs 0.12

# Dynamic Smagorinsky 评估
CUDA_VISIBLE_DEVICES=5 python run_formal_dsmag.py --n 128 --steps 15000 --tau 0.51
```

## 9. 局限与缓解

**局限**:
1. 仅评估了固定 C_s 与 DSMag（库实现），未评估 WALE/Vreman
2. DSMag 评估受限于库实现缺陷（全局平均 + 稳定性问题）
3. 未尝试更高 C_s（≥0.18）或更高分子黏性（τ≥0.55）

**缓解**:
1. 终裁记录完整证据链，供后续引擎级改造参考
2. 库缺口 G-DIT-1/2（3D FFT 谱/无散场）已登记
3. DSMag 实现问题已定位，可独立修复

---

**结论**: DIT 案例 TERMINAL——固定 C_s Smagorinsky 在跨网格收敛性上存在结构性缺口，库内 DSMag 实现不足以解决。晋级需引擎级改造（高阶格式/AMR/隐式 LES）或 DSMag 库实现修复。
