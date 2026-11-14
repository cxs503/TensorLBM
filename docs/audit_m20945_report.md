# 交差审计报告：m20945 XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN

## 审计概要

**任务**：m20945 — XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN（两格子 Shan-Chen MCMP 交叉验证）
**审计方**：board-audit (m20975)
**日期**：2026-11-15
**判定**：**L2**（独立功能实现）

---

## ① 案例介绍

**物理背景**：两格子（双分布函数）Shan-Chen 多组元多相 LBM 模型。每个组元 σ 携带独立分布函数 f_σ，通过跨组件伪势力实现相分离：

```
F_σ = -G_12 · ψ_σ · Σ_i w_i · ψ_σ'(x ± c_i) · c_i
```

**几何/参数**：
- 格子：D2Q9
- 伪势：线性 ψ(ρ) = ρ
- 碰撞：BGK，每组件独立 τ
- 强迫：速度偏移 u_eq = u + τF/ρ
- 液滴验证：48×48 网格，τ=1.0，G_12=±2.5，2000 步

**外部锚点**：
- Shan & Chen (1993) Phys. Rev. E 47, 1815
- Shan & Doolen (1995) J. Stat. Phys. 81, 379
- Martys & Chen (1996) J. Comput. Phys. 126, 276
- Guo, Zheng & Shi (2002) Phys. Rev. E 65, 046308
- OSS: geoelements/mcmp-lbm (commit 015a26f)

---

## ② 怎么做的

**方法**：交叉验证（cross-validation）—— 在 run_xval.py 中独立重实现 MPLBM-UT 约定（forward-gather, G>0=排斥），与 TensorLBM 库函数 collide_sc_two_component（backward-gather, G<0=排斥）对比。数学上 G_TL = -G_MPLBM 时两者等价。

**验证流程**：
1. 共存密度：平界面弛豫 → 液相/气相密度
2. 静态液滴：多半径 → Laplace 定律 Δp = σ/R
3. 虚假电流：u_max 水平
4. 表面张力：Laplace 拟合 vs EOS 预测

**工具链**：PyTorch (float64), matplotlib, numpy。单核 CPU。

---

## ③ 计算结果

### 液滴交叉验证（TensorLBM vs MPLBM-UT）

| 量 | MPLBM-UT | TensorLBM | 误差 |
|---|---|---|---|
| σ (R=12) | 0.197684 | 0.197684 | **0.00%** |
| σ (R=16) | 0.212781 | 0.212781 | **0.00%** |
| Δp (R=12) | 0.014860 | 0.014860 | **0.00%** |
| Δp (R=16) | 0.013170 | 0.013170 | **0.00%** |
| u_max (R=12) | 0.012554 | 0.012554 | ratio=1.000 |
| σ 拟合 | 0.203784 | 0.203784 | cross-error **0.00%** |

### 分层 Poiseuille

| 来源 | ny | err_Q_mix | 状态 |
|---|---|---|---|
| xval runner | 25 | NaN | ❌ converged=false |
| xval runner | 33 | NaN | ❌ converged=false |
| 独立已验证基准 | 25 | **-0.14%** | ✓ PASS |
| 独立已验证基准 | 33 | **+0.62%** | ✓ PASS |

---

## ④ 结果图

工件包含三张图（由交付方生成，审计方验证文件存在且非空）：
- `fig1_laplace_law.png` (63970 B): Δp vs 1/R 线性拟合
- `fig2_sigma_comparison.png` (45130 B): σ 跨求解器比较
- `fig3_spurious_currents.png` (58470 B): 虚假电流水平

---

## ⑤ 注意点（缺陷清单）

### D1（阻塞）：测试文件 SyntaxError
`TensorLBM/tests/test_xval_mcmp_two_lattice_shan_chen.py` 第 1 行为 `import sys; sys.path.insert(...)`，第 15 行才是 `from __future__ import annotations`。违反 Python 语法，pytest 无法收集。

### D2（阻塞）：导入路径不匹配
测试导入 `collide_sc_two_component` 和 `sc_two_component_force` 自 `tensorlbm.multiphase`，但这些函数**不存在**于 `TensorLBM/src/tensorlbm/multiphase/__init__.py` 中。它们仅存在于 `m20945/src/tensorlbm/multiphase.py`。

### D3（非阻塞）：xval runner 分层 Poiseuille 发散
xval runner 使用 `stream()` 而无壁面反弹处理，导致通道流算例全部 NaN。`oss_solutions.json` 已正确标记为 partial/honest-fail。

### D4（结构性）：无机器裁判
`tensor_benchmark_cases` 表中无此 anchor 注册行 → L4 级结构性不可达。

---

## ⑥ 误差判定

| 关键量 | 基准 | 测量值 | 误差 | <3%? |
|---|---|---|---|---|
| σ cross-error (R=12) | 0 | 0.00% | 0.00% | ✓ |
| σ cross-error (R=16) | 0 | 0.00% | 0.00% | ✓ |
| Δp cross-error (R=12) | 0 | 0.00% | 0.00% | ✓ |
| Δp cross-error (R=16) | 0 | 0.00% | 0.00% | ✓ |
| err_Q_mix (ny=25, 独立基准) | 0 | -0.14% | 0.14% | ✓ |
| err_Q_mix (ny=33, 独立基准) | 0 | +0.62% | 0.62% | ✓ |

**重要说明**：
- 液滴交叉验证是 **code-to-code** 比较（TensorLBM vs MPLBM-UT），非 code-to-analytic。误差 0.00% 反映两实现位级一致，但不独立证明物理正确性。
- 分层 Poiseuille 的独立已验证基准（`benchmarks/verified/mcmp_stratified_poiseuille`）误差 <1%，满足 <3% 标准。

---

## 判定结论

**级别：L2**（独立功能实现）

- ✓ L0: 有真实代码，非纯文档
- ✓ L1: 有数学验证（Laplace 定律交叉验证）
- ✓ L2: 有真代码实现（输入→计算→输出），`collide_sc_two_component` 在 `src/tensorlbm/multiphase.py:534-750`
- ✗ L3: 测试基础设施损坏（SyntaxError + 导入路径不匹配），集成未通过测试验证
- ✗ L4: 无机器裁判（tensor_benchmark_cases 无注册行），xval runner 解析基准失败（NaN）
- ✗ L5: 非科研工程成果

**消账建议**：confirmed（L2 已达成，但需修复 D1/D2 才能升级）
