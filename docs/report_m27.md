# Mission 27 — SDAA Sweep: maneuvering_steady_turn Benchmark

## ① 案例介绍

### 物理背景
船舶操纵性定常回转（steady turning）是 IMO 操纵性衡准的核心指标之一。本案例验证 `tensorlbm.maneuvering` 模块中线性 sway-yaw 模型的定常回转固定点求解与瞬态积分精度。

### 几何与参数

| 参数 | 值 | 说明 |
|---|---|---|
| 船型 | KVLCC2-class 大型油轮（全尺度） | |
| L (m) | 320.0 | 垂线间长 |
| B (m) | 58.0 | 型宽 |
| Draft (m) | 20.8 | 吃水 |
| Cb | 0.81 | 方形系数 |
| U (m/s) | 7.97 | 航速（≈15.5 kn） |
| Mass (kg) | 3.127×10⁸ | 排水质量 |
| Izz (kg·m²) | 2.001×10¹² |  yaw 转动惯量 |

### 水动力导数（无量纲）

| 导数 | 值 |
|---|---|
| Y'v | −0.65 |
| Y'r | +0.05 |
| N'v | −0.095 |
| N'r | −0.08 |
| Y'δ | −0.16 |
| N'δ | +0.04 |

> **注意**：导数为 caller-supplied 的 KVLCC2 代表性线性系数，量级与 Yasukawa & Yoshimura (2015) / SIMMAN 2008 一致，但非精确摘录，也非 LBM 流场提取。

### 模态特性
- 特征值：λ₁ = −0.0237, λ₂ = −0.1400
- 时间常数：τ₁ = 42.14 s, τ₂ = 7.14 s
- 方向稳定性：✅ 稳定（两特征值均为负）

---

## ② 方法

### 验证流程
1. **原始基准**（CPU, float64）：`benchmarks/verified/maneuvering_steady_turn/run.py` 在 CPU 上运行，验证 8 条判据全部 PASS。
2. **SDAA 迁移**：编写独立脚本 `scripts/sdaa_sweep_maneuvering_steady_turn.py`，不改业务代码：
   - `torch.set_default_device('sdaa')` + `torch.set_default_dtype(float64)`
   - 所有 `torch.linalg.*`（solve, eigvals, matmul）在 SDAA 上执行
   - Monkey-patch `Tensor.numpy()` → `.cpu().numpy()`（仅最终 host 提取走 CPU）
   - 无需 TRITON_INTERPRET（纯 torch + numpy，无 triton kernel）

### 计算配置

| 参数 | 值 |
|---|---|
| 舵角 | 10°, 20° |
| 时间步长 | coarse=1.0 s, fine=0.25 s |
| 积分器 | semi_implicit_euler, rk4 |
| T_end | 300 s（原 600 s，因 SDAA kernel-launch 开销缩减；仍 >7× τ₁） |
| 稳态窗口 | t ∈ [150, 300] s（>3.5 e-folding） |
| 容差 | 3% |
| 总壁钟 | 59.89 s |

### 8 条判据
| ID | 判据 | 阈值 |
|---|---|---|
| C1 | 每层 \|R_err\| ≤ 3% | 3% |
| C2 | 每层 \|β_err\| ≤ 3% | 3% |
| C3 | 两层 span(R, β) ≤ 3% | 3% |
| C4 | 独立代数路径一致 < 1e-6% | 1e-6% |
| C5 | Euler 瞬态 r(t) 误差(fine) ≤ 3% | 3% |
| C6 | 瞬态加密比 > 3 | 3 |
| C7 | RK4 瞬态 r(t) 误差(fine) ≤ 3% | 3% |
| C8 | 方向稳定 | 特征值 < 0 |

---

## ③ 计算结果

### 稳态回转误差（vs 解析固定点）

| Level | δ (°) | 积分器 | R_ana (m) | R_meas (m) | R_err (%) | β_ana (°) | β_meas (°) | β_err (%) |
|---|---|---|---|---|---|---|---|---|
| coarse | 10 | Euler | 1296.0 | 1304.2 | +0.639 | 7.77 | 7.71 | −0.771 |
| fine | 10 | Euler | 1296.0 | 1304.6 | +0.663 | 7.77 | 7.70 | −0.800 |
| coarse | 20 | Euler | 665.5 | 669.5 | +0.599 | 15.26 | 15.14 | −0.745 |
| fine | 20 | Euler | 665.5 | 669.6 | +0.621 | 15.26 | 15.14 | −0.772 |
| fine_rk4 | 10 | RK4 | 1296.0 | 1304.7 | +0.672 | 7.77 | 7.70 | −0.810 |
| fine_rk4 | 20 | RK4 | 665.5 | 669.7 | +0.630 | 15.26 | 15.14 | −0.782 |

### 瞬态精度

| 指标 | 值 |
|---|---|
| Euler coarse r(t) err | 0.509% |
| Euler fine r(t) err | 0.124% |
| 加密比 (coarse/fine) | **4.09**（一阶，符合预期） |
| RK4 fine r(t) err | **5.83×10⁻⁸%** |
| 稳态 r 标准差 | 4.0×10⁻⁵ ~ 8.0×10⁻⁵ |

### 代数路径一致性

| 路径对比 | 偏差 (%) |
|---|---|
| NumPy solve vs library | 2.79×10⁻¹⁴ |
| Nomoto 闭式 vs library | 2.79×10⁻¹⁴ |

### 两层 Span

| 舵角 | R_span (%) | β_span (%) |
|---|---|---|
| 10° | 0.023 | 0.028 |
| 20° | 0.022 | 0.027 |

### 判据汇总

| ID | 判据 | 最大观测值 | PASS? |
|---|---|---|---|
| C1 | \|R_err\| ≤ 3% | 0.672% | ✅ |
| C2 | \|β_err\| ≤ 3% | 0.810% | ✅ |
| C3 | span ≤ 3% | 0.028% | ✅ |
| C4 | 代数路径 < 1e-6% | 2.8e-14% | ✅ |
| C5 | Euler transient ≤ 3% | 0.124% | ✅ |
| C6 | 加密比 > 3 | 4.09 | ✅ |
| C7 | RK4 transient ≤ 3% | 5.8e-8% | ✅ |
| C8 | 方向稳定 | λ = −0.024 | ✅ |

**总结论：8/8 PASS，verdict = PASS**

---

## ④ 结果图

![SDAA 定常回转误差与判据汇总](/tmp/m27_sdaa_results.png)

*左图*：6 个计算工况的回转半径误差 |R_err| 和漂角误差 |β_err|，红色虚线为 3% 阈值。所有工况远低于阈值。
*右图*：8 条判据的最大观测值，全部绿色（PASS）。

> **注**：本案例为 ODE 级操纵性验证（无 LBM 流场），无场量云图可生成。图中数据全部从 `sdaa_maneuvering_steady_turn.json` 程序化提取，零手抄。

---

## ⑤ 注意点

1. **非端到端 LBM 验证**：水动力导数为 caller-supplied 的代表性 KVLCC2 线性系数，非 LBM 流场提取，非 PMM 实验。无自由面、无舵/桨滑流、无高阶非线性。
2. **T_end 缩减**：SDAA 上 2×2 tensor op 的 kernel-launch 开销使原 600 s 配置超时，缩减至 300 s。稳态窗口仍捕获 >3.5 个 e-folding，不影响判据有效性。
3. **Monkey-patch**：`Tensor.numpy()` 被 patch 为 `.cpu().numpy()`。所有 torch.linalg 运算仍在 SDAA 执行，仅最终 host 提取走 CPU。数值结果不受影响。
4. **适用范围**：本验证仅覆盖线性 sway-yaw 模型的 ODE 求解精度。非线性 cross-flow（Abkowitz 高阶项）仅作为 diagnostic 记录，不在验收集内。
5. **端到端阻塞**：LBM → 操纵性耦合需要自由面求解器，这是结构性阻塞（structural blocker），当前未解决。

---

## ⑥ 误差判定

| 关键量 | 最大误差 | 阈值 | 是否 < 3% |
|---|---|---|---|
| 回转半径 R | 0.672% | 3% | ✅ **是** |
| 漂角 β | 0.810% | 3% | ✅ **是** |
| 两层 span | 0.028% | 3% | ✅ **是** |
| 代数路径一致性 | 2.8×10⁻¹⁴% | 1e-6% | ✅ **是** |
| Euler 瞬态 r(t) | 0.124% | 3% | ✅ **是** |
| RK4 瞬态 r(t) | 5.8×10⁻⁸% | 3% | ✅ **是** |
| 加密比 | 4.09 | >3 | ✅ **是** |

**所有关键量误差均 < 3%，SDAA 设备上的计算结果与解析解高度一致。**

### 无基准声明原因
本案例的水动力导数为 caller-supplied 的代表性系数（量级与 published PMM/MMG 数据一致但非精确摘录），因此不存在"实验基准值"可供对比。验证口径为：数值解 vs 同一组导数下的解析固定点——这是自洽性验证（self-consistency verification），而非实验对标。端到端实验对标需要 LBM 流场提取导数，当前为结构性阻塞。

---

*报告生成时间：2026-10-29 | 工件来源：artifacts/sdaa_maneuvering_steady_turn.json | 设备：SDAA (torch 2.7.1, float64)*
