# DIT Turbulence 终裁报告：TERMINAL

**案例**：衰减各向同性湍流（DIT）  
**判定**：TERMINAL（结构性 LES 模型缺口）  
**日期**：2026-10-22  
**Mission**：m167

---

## 1. 物理问题

3D 周期域衰减各向同性湍流（Decaying Isotropic Turbulence, DIT）：
- **初始场**：随机无散速度场，目标能谱 E(k) ∝ k⁴·exp(−2(k/k₀)²)
- **控制参数**：u_rms=0.04, τ=0.501（分子 ν=10⁻³）, Re_L≈65
- **亚格子模型**：Smagorinsky LES（固定 C_s=0.12）
- **双判据**：
  1. 惯性区能谱斜率 vs Kolmogorov -5/3（≤3%）
  2. 衰减指数 n ∈ [1.2, 1.5]（Saffman/Batchelor 带）

## 2. 晋级路径评估（三菜单）

### 2.1 动态 Smagorinsky（Germano-Lilly）

**库入口**：`tensorlbm.turbulence.collide_dynamic_smagorinsky_bgk3d`

**评估结果**：

| 参数 | 稳定性 | 能谱斜率 | 衰减指数 | 判定 |
|------|--------|----------|----------|------|
| τ=0.501 | ❌ NaN@609步 | — | — | 不稳定 |
| τ=0.51 | ✅ 稳定 | -7.71（-362%） | 1.73 | 灾难性失败 |

**根因诊断**：
1. **稳定性缺口**：库实现在 τ=0.501（近零分子黏性）时失稳——DSMag 需要额外分子耗散缓冲
2. **实现缺陷**：C_s² 采用全局平均（Germano et al. 1991 的简化版），非标准 Lilly 局部动态过程
   - 标准 Germano-Lilly：C_s²(x,t) = ⟨L_ij M_ij⟩/⟨M_kl M_kl⟩（局部或平面平均）
   - 库实现：cs2 = mean(L·M) / mean(M²)（全局标量，每步重算）
3. **clamp 限制**：cs2 被 clamp 到 [0, 0.1]，抑制了动态适应性

**结论**：库实现不足以解决固定 C_s 的跨网格收敛问题。

### 2.2 系综平均

**评估**：系综平均（多 IC/多 seed）不能解决系统性偏差。

- 现有结果（seed=42）已显示 N=128 与 N=256 的**系统性差异**（斜率差 0.256）
- 这不是统计噪声，而是 LES 模型参数化的结构性问题
- 系综平均会降低方差，但不会改变均值偏差

**结论**：不适用。

### 2.3 更长运行

**评估**：延长运行时间（15000→30000 步）不会改善谱形状。

- N=256 在 15000 步时已充分衰减（E(t) ∝ t^-n 拟合窗口 [0.15T, 0.6T]）
- 谱斜率问题源于**欠耗散**（数值耗散↓ + 固定 SGS 耗散 → 惯性区能量堆积）
- 延长运行只会让衰减继续，不会改变谱的渐近形状

**结论**：不适用。

## 3. 四门判决

| 门 | 标准 | 结果 | 证据 |
|----|------|------|------|
| G1 共性模块入口 | src/tensorlbm 库路径 | ✅ PASS | `solver3d.stream3d` + `turbulence.collide_smagorinsky_bgk3d` |
| G2 直接观测量 ≤3% | slope err, n decay | ❌ FAIL | N=128: +1.69% (OK); N=256: +17.05% (FAIL) |
| G3 ≥2 档单调收敛 | 档间差 ≤3% | ❌ FAIL | slope 差 0.256 (15.4%), n 差 0.244 (16.8%) |
| G4 参考先锁定 | prereg append-only | ✅ PASS | 参考 Kolmogorov -5/3 先锁定 |

**综合判定**：FAIL → TERMINAL

## 4. 根因证据链

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

## 5. 晋级路径（需引擎级改造）

1. **修复 DSMag 库实现**（~5 GPU·h）
   - 实现标准 Germano-Lilly 局部动态过程（C_s²(x,t) 场，非全局标量）
   - 移除 cs2 clamp 限制（或放宽到 [-0.01, 0.1]，允许 backscatter）
   - 添加拉格朗日平均（Meneveau et al. 1996）稳定化

2. **实现 WALE/Vreman 模型**（~10 GPU·h）
   - WALE：C_w=0.325（已扫过，slope=-0.336，欠耗散死路）
   - Vreman：C_V=0.025（需评估）

3. **引擎级改造**（~20-50 GPU·h）
   - 高阶格式（WENO/TVD）减少数值耗散
   - 自适应网格细化（AMR）解析惯性区
   - 隐式 LES（无显式 SGS，依赖数值耗散）

4. **维持 TERMINAL**（0 GPU·h，推荐）
   - 记录负结果：固定 C_s Smagorinsky 在 DIT 工况下跨网格不收敛
   - 待引擎级改造后重新评估

## 6. 审计溯源

**PROVENANCE 块**：
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

**复现命令**：
```bash
# 固定 C_s 基准（已有）
python run.py --n 128 --steps 15000 --cs 0.12
python run.py --n 256 --steps 15000 --cs 0.12

# Dynamic Smagorinsky 评估
CUDA_VISIBLE_DEVICES=5 python run_formal_dsmag.py --n 128 --steps 15000 --tau 0.51
```

## 7. 局限与缓解

**局限**：
1. 仅评估了固定 C_s 与 DSMag（库实现），未评估 WALE/Vreman
2. DSMag 评估受限于库实现缺陷（全局平均 + 稳定性问题）
3. 未尝试更高 C_s（≥0.18）或更高分子黏性（τ≥0.55）

**缓解**：
1. 终裁记录完整证据链，供后续引擎级改造参考
2. 库缺口 G-DIT-1/2（3D FFT 谱/无散场）已登记
3. DSMag 实现问题已定位，可独立修复

---

**结论**：DIT 案例 TERMINAL——固定 C_s Smagorinsky 在跨网格收敛性上存在结构性缺口，库内 DSMag 实现不足以解决。晋级需引擎级改造（高阶格式/AMR/隐式 LES）或 DSMag 库实现修复。
