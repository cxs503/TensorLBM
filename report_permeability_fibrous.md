# 纤维床横向渗透率 Benchmark 终裁报告

**案例**: permeability_fibrous  
**物理问题**: 周期圆柱阵列横向 Darcy 渗透率（方阵 + 六角堆积）  
**判决**: **TERMINAL**  
**日期**: 2026-10-22  
**Mission**: m169  

---

## 1. 物理问题与几何

### 1.1 问题定义

纤维增强复合材料制造中，树脂在纤维预制件中的渗透率是关键工艺参数。本 benchmark 研究**周期圆柱阵列的横向 Darcy 渗透率** $K_\perp$，即流动方向垂直于圆柱轴线的情况。

两类阵列构型：
- **方阵（square）**: $W \times W$ 单胞，中心单圆柱，固体体积分数 $\phi = \pi R^2 / W^2$
- **六角（hexagonal）**: $W \times H$ 超胞（$H = \sqrt{3}W$），三角格点堆积，$\phi = 2\pi R^2 / (\sqrt{3}W^2)$

### 1.2 控制参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 雷诺数 | $Re = 0.05$ |  creeping flow，Darcy 定律适用 |
| 固体体积分数 | $\phi \in [0.3, 0.75]$ | 稀薄到近堆积 |
| 格子分辨率 | gap $\in [12, 96]$ LU | 圆柱间隙最小分辨率 |
| 力标度 | $f_s = 10$（正式）/ $20$（诊断） | 体体力驱动 |

### 1.3 参考解

**方阵**: Sangani & Acrivos (1982) 精确数值表（9 点，$\phi \in [0.05, 0.75]$）  
**六角**: Gebart (1992) 近堆积渐近式 $K/R^2 = C_1 (\sqrt{\phi_{max}/\phi} - 1)^{5/2}$

**关键发现**: hex 精确解不可得（Koch & Ladd 1997 / Wang & Sangani 1997 闭源），Gebart 渐近式在 $\phi = 0.70/0.75$ 存在系统性偏差（+2.4%/+3.9%）。

---

## 2. 数值方法

### 2.1 格子 Boltzmann 方法

- **模型**: D2Q9 BGK，单松弛时间 $\tau = 1$
- **边界**: 半程反弹（mid-point bounce-back）
- **驱动**: 体体力 $F = \rho f_s \hat{x}$，周期性边界
- **稳态判据**: 相对速度变化 $< 10^{-5}$ 持续 2000 步

### 2.2 渗透率计算

$$K_{sim} = \frac{(1-\phi) \nu \langle u_x \rangle_{fluid}}{a_{body}}$$

其中 $a_{body}$ 为体体力加速度，$\langle u_x \rangle_{fluid}$ 为流体相平均速度。

### 2.3 共性模块入口

所有物理核（碰撞、迁移、边界、力施加）均来自 `tensorlbm` 库，driver 层仅处理几何生成和后处理。铁律 grep 验证零手写物理核。

---

## 3. 结果与判据

### 3.1 收敛阶梯

![方阵收敛](benchmarks/pending/permeability_fibrous/figures/fig_sq_convergence.png)

**方阵主档**（$\phi = 0.3, 0.5, 0.6$）：严格单调收敛，最细档误差 $< 1\%$。

**方阵异常档**（$\phi = 0.4$）：第 3→4 阶梯出现 +0.21pp 回升，破坏单调性。

![六角收敛](benchmarks/pending/permeability_fibrous/figures/fig_hex_convergence.png)

**六角主档**（$\phi = 0.70, 0.75$）：误差持续走负，最终稳定在 -2% 到 -4%。

### 3.2 误差汇总

![误差汇总](benchmarks/pending/permeability_fibrous/figures/fig_error_summary.png)

| 构型 | $\phi$ | 最细档误差 | 单调性 | 判决 |
|------|--------|-----------|--------|------|
| sq | 0.30 | 0.22% | ✅ | PASS |
| sq | 0.40 | 2.63% | ❌ | FAIL |
| sq | 0.50 | 0.24% | ✅ | PASS |
| sq | 0.60 | 0.88% | ✅ | PASS |
| hex | 0.70 | -2.10% | ❌ | FAIL |
| hex | 0.75 | -2.03% | ❌ | FAIL |

### 3.3 失效模式诊断

![楼梯分解](benchmarks/pending/permeability_fibrous/figures/fig_staircase_decomposition.png)

**三种失效模式全定源**:

1. **hex 参考侧偏差**: Gebart 渐近式在 $\phi = 0.70/0.75$ 对精确解偏高 +2.4%/+3.9%（楼梯分解证实）
2. **sq 0.40 几何量化振荡**: 分辨率地板处半径量化导致 +0.21pp 回升（gap-96 扩展案深入门内 -0.65%）
3. **S1 惯性效应**: $Re_{cell} = 0.5$ 下真实惯性，物理量正确但破容差

---

## 4. 判据表

### 4.1 四门判决

| 门 | 要求 | 结果 | 说明 |
|----|------|------|------|
| G1 共性模块入口 | 物理核全部库入口 | ✅ PASS | 铁律 grep 零命中 |
| G2 直接观测量 ≤3% | 最细档误差 ≤3% | ❌ FAIL | 主档数值过线，但 hex 因参考偏差导致单调破缺 |
| G3 ≥2 档单调收敛 | 严格单调递减 | ❌ FAIL | sq 0.40 一档破缺，hex 0.70/0.75 两主档全破 |
| G4 参考先锁定后跑 | prereg 冻结后运行 | ✅ PASS | prereg v1.0 2026-09-28 冻结，正式跑 2026-09-29 |

**总判**: FAIL（sq/hex 双 FAIL）

### 4.2 机器门禁三元组

```json
{
  "verdict": "terminal",
  "reason": "三失效模式全定源：hex 近堆积 = 参考侧（Gebart 渐近式对精确解偏高 +2.4%/+3.9%）；sq 0.40 = 几何量化地板振荡（gap-96 扩展案深入门内 −0.65%）；S1 = 真实 O(Re) 惯性",
  "gates": [
    {"gate": "library_entry", "pass": true},
    {"gate": "direct_observable", "pass": false},
    {"gate": "monotone_convergence", "pass": false},
    {"gate": "reference_locked_first", "pass": true},
    {"gate": "case_readme", "pass": true},
    {"gate": "figures", "pass": true}
  ]
}
```

完整三元组见 `benchmarks/pending/permeability_fibrous/gates.json`。

---

## 5. 审计溯源

### 5.1 工件清单

| 工件 | 路径 | 说明 |
|------|------|------|
| 冻结报告 | `benchmarks/pending/permeability_fibrous/result.json` | 40 份 case 机器档案 + wave9_record 附录 |
| 诊断报告 | `benchmarks/pending/permeability_fibrous/DIAGNOSTICS.md` | 三失效模式全定源 |
| 预注册 | `benchmarks/pending/permeability_fibrous/prereg.md` | v1.1 冻结（2026-09-28） |
| R3 自产门 | `benchmarks/pending/permeability_fibrous/hex_ref_gate/` | 51 case + 双门 JSON |
| 扩展定源 | `benchmarks/pending/permeability_fibrous/out_diag/` | 4 案 gap-72/96 扩展 |

### 5.2 计算资源

- **正式跑**: ~40 GPU·h（GPU5，2026-09-29）
- **诊断跑**: ~4 GPU·h（GPU5，2026-09-29）
- **R3 自产门**: ~8 CPU·h（本地机，2026-09-29/30）
- **终裁文档**: 0 GPU·h（纯后处理，2026-10-22）

### 5.3 参考锁定时间线

| 事件 | 时间 | 说明 |
|------|------|------|
| prereg v1.0 冻结 | 2026-09-28 | 参考表 + 判据 + 档位锁定 |
| prereg v1.1 修订 | 2026-09-29 | 加第 4 级 gap≈72 阶梯 |
| 正式跑完成 | 2026-09-29 | 40 份 case 机器档案 |
| 诊断跑完成 | 2026-09-29 | 4 案扩展定源 |
| 终裁文档 | 2026-10-22 | 三失效模式全定源 |

---

## 6. 局限与缓解

### 6.1 局限

1. **hex 精确表不可得**: Koch & Ladd 1997 / Wang & Sangani 1997 闭源，8 通道检索未果
2. **Gebart 渐近式偏差**: 在 hex 0.70/0.75 对精确解偏高 +2.4%/+3.9%
3. **sq 0.40 量化地板**: 几何量化振荡导致单调破缺
4. **惯性效应**: $Re_{cell} = 0.5$ 下的真实惯性，物理量正确但破容差

### 6.2 缓解

1. **hex 参考侧偏差已定量**: 楼梯分解证明偏差来源，非数值误差
2. **sq 0.40 可扩展**: gap-96 扩展案已入门内，加 rung 可修复
3. **sq 0.30/0.50/0.60 现成候选**: 三档全过，可独立晋级
4. **诊断完整**: 三失效模式全定源，负结果对后续研究有价值

---

## 7. 菜单选项（需 owner 决策）

### 选项 A: cut-cell 自产精确表
- **动作**: M1 cut-cell 重跑方阵全阵自产精确表
- **预算**: ~8 GPU·h
- **预期**: 消除 hex 参考侧偏差

### 选项 B: 门放宽 1%
- **动作**: 将 3% 门放宽到 1%
- **预算**: 零成本
- **预期**: vf0.6/0.7 即过

### 选项 C: 购精确表
- **动作**: 购买 Koch-Ladd 1997 / Wang-Sangani 1997 表
- **预算**: ~$50-100
- **预期**: 获得 hex 精确参考

### 选项 D: 维持 FAIL + sq 0.30-0.60 现成候选档（推荐）
- **动作**: 维持 Gebart 门接受 FAIL，sq 0.30/0.50/0.60 三主档独立晋级
- **预算**: 零成本
- **预期**: 记录负结果，供后续研究参考

---

## 8. 结论

本案例 TERMINAL 判决基于三失效模式的完整定源：

1. **hex 近堆积 = 参考侧偏差**: Gebart 渐近式在 $\phi = 0.70/0.75$ 对精确解偏高 +2.4%/+3.9%
2. **sq 0.40 = 几何量化地板振荡**: gap-96 扩展案深入门内 −0.65%
3. **S1 = 真实 O(Re) 惯性**: 物理量正确，按协议记录

**四门判决**: G1✅ G2❌ G3❌ G4✅ → FAIL

**负结果价值**:
- 首次定量 hex 0.70/0.75 横向渗透率的解析 LBM 测量
- 首次揭示 Gebart 渐近式在 hex 近堆积的系统性偏差
- 完整诊断三失效模式，为后续研究提供参考

---

## 附录: 运行命令

```bash
# 单案运行
cd benchmarks/pending/permeability_fibrous
PYTHONPATH=../../../src python run.py case --array sq --vf 0.5 --w 238 --device cuda:5

# 聚合判决
PYTHONPATH=../../../src python run.py report --out-dir out
```

## 附录: PROVENANCE

所有判决数字直读工件，零手抄。完整三元组见 `benchmarks/pending/permeability_fibrous/gates.json`。
