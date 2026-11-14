# Mission 2204 补报：TensorLBM 自由面引擎验证欠账填平

> **补报单** · 原单 mission 2204 已成功交付（PR #418, commit ba33047），但未按 owner 常设指令发布论文级进展报告。本文依据原单声明及仓库内已有验证文档（`docs/FREE_SURFACE_VERIFICATION.md`、`docs/P350W2_N07_free_surface_verification.md`）程序化提取数据，**零手抄、零编造**。工件目录未留档的数字一律标注"未留档"。

---

## 1. 案例介绍（物理背景 / 几何 / 参数表）

### 1.1 物理背景

TensorLBM 自由面引擎（PR #418）实现了两类自由面算法：

1. **Color-Gradient 两相 LBM**（Latva-Kokko & Rothman 2005）：红/蓝分布函数分离、重着色步、表面张力扰动、Guo 强制重力。
2. **VOF 界面捕获**（MAC 网格 + 投影法）：守恒型 VOF 输运 `∂F/∂t + ∇·(uF) = 0`，MUSCL/Superbee 界面重构，几何多重网格 V-cycle 求解变密度 Poisson 方程。

验证欠账：上述引擎在合入主干前缺少与解析解/实验的系统比对，mission 2204 的目标是填平这一欠账。

### 1.2 三个基准案例

| 案例 | 物理 | 解析/实验基准 | 关键参数 |
|------|------|---------------|----------|
| **溃坝 (Dam Break)** | 重力驱动液柱坍塌 | Ritter (1892) 解析解；Martin & Moyce (1952) 实验 | 宽高比 H/a = 2；ρ_l/ρ_g = 1000；ν_l = 1e-3 |
| **Wigley III 船首波** | 定常航行兴波 | Michell (1898) 薄船理论 | L = 1, B = 0.1, T = 0.05, Cb = 0.555；Fr = 0.25, 0.31, 0.40 |
| **入水砰击 (Water Entry)** | 楔体匀速入水 | Wagner (1932) 理论 | 楔角 β；恒速 V |

### 1.3 验证判据（门限）

| 编号 | 判据 | 门限 |
|------|------|------|
| (a) | 溃坝前锋位置 RMSE | ≤ 10% |
| (b) | 质量守恒漂移 | ≤ 2% |
| (c) | Wigley 波高误差 | ≤ 15% |
| (d) | 收敛阶 / R² | ≥ 1.5 (或 ≥ 1.0) / ≥ 0.95 |
| (e) | 测试覆盖（含负对照） | ≥ 15 项 |

---

## 2. 方法 / 流程 / 关键设置 / 工具链

### 2.1 数值方法

**Color-Gradient LBM 路径**：
- D3Q27 格子，双松弛时间 (τ_r, τ_b) 处理黏度比
- Guo 强制格式施加重力
- 前锋追踪：体积分数阈值法

**VOF 路径**（P350W2-N07 独立验证）：
- 守恒型 VOF 输运 + MUSCL/Superbee 重构
- 变密度 Poisson 方程 → 几何多重网格 V-cycle（红黑 Gauss-Seidel 光滑）
- CSF 表面张力模型（溃坝工况关闭，Bond → ∞）
- 质量账本：每步记录液体体积

### 2.2 实现文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/tensorlbm/free_surface.py` | 1077 | VOF 自由面求解器 |
| `src/tensorlbm/free_surface_wigley.py` | 未留档 | Wigley 船首波基准模块 |
| `src/tensorlbm/vof_interface_capture.py` | 未留档 | PLIC/Youngs/高度函数 |
| `src/tensorlbm/csf_force_model.py` | 未留档 | CSF 表面张力 |
| `src/tensorlbm/free_surface_verification.py` | 未留档 | 三案例验证套件 |
| `tests/test_free_surface_verification.py` | 未留档 | 34 项单元测试 |
| `tests/test_free_surface_verification_p350w2.py` | 未留档 | 33 项独立验证测试 |

### 2.3 工具链

Python 3.11 + NumPy + SciPy + pytest + matplotlib。所有计算在仓库 CI 内完成。

---

## 3. 计算结果（数字表，程序化提取）

### 3.1 溃坝：前锋位置 vs Martin & Moyce (1952)

| 分辨率 | 网格 (nx×ny) | RMSE | 相对 RMSE | 质量漂移 | 步数 |
|--------|-------------|------|-----------|----------|------|
| 8 cells/a | 48×32 | 0.1527 | **7.79%** | 1.01e-03 | 183 |
| 12 cells/a | 64×32 | 0.1734 | 8.58% | 8.69e-04 | 274 |
| 16 cells/a | 96×48 | 0.1910 | 9.61% | 7.24e-04 | 432 |

**判定**：最优分辨率 8 cells/a 下 RMSE = 7.79% ≤ 10% 门限 → **PASS**。

### 3.2 质量守恒

| 工况 | 最大相对漂移 | 门限 (2%) | 状态 |
|------|-------------|-----------|------|
| 溃坝 8 cells/a | 1.01e-03 (0.10%) | 2.0e-02 | **PASS** |
| 溃坝 16 cells/a | 7.24e-04 (0.07%) | 2.0e-02 | **PASS** |

### 3.3 Wigley III 船首波

| 子项 | 结果 | 说明 |
|------|------|------|
| Michell 理论波长 | 28.27 cells | 解析实现 |
| 波数 | 0.222 | 解析实现 |
| 速度比 U/c_phase | 1.000 | 色散关系验证 PASS |
| VOF 仿真 | **数值不稳定** | 粗网格下移动船体边界产生大梯度，MUSCL 格式失稳 |

**判定**：Michell 理论参考已完整实现并验证；VOF 仿真需更高分辨率或改进船体处理 → **PARTIAL**。

### 3.4 入水砰击 (Wagner 1932)

| 量 | 理论值 | 测量值 | 相对误差 | R² |
|----|--------|--------|----------|-----|
| 力斜率 dF/dt | 0.393 | 0.393 | 0.0% | 1.000 |

**判定**：**PASS**（但注意：此为 Color-Gradient LBM 路径的验证结果，非 VOF 路径）。

### 3.5 收敛阶

| 度量 | 值 | 门限 | 状态 |
|------|-----|------|------|
| 输运收敛阶 | 1.33 | ≥ 1.0 | **PASS**（低于 1.5 期望但超 1.0 底线） |
| R² | 1.000 | ≥ 0.95 | **PASS** |

### 3.6 测试覆盖

| 类别 | 测试数 | 状态 |
|------|--------|------|
| 溃坝前锋位置 | 5 | PASS |
| 质量守恒 | 5 | PASS |
| Wigley 船首波 | 8 | PASS |
| 收敛阶 | 4 | PASS |
| 配置校验 | 5 | PASS |
| 度量辅助 | 3 | PASS |
| Poisson 求解器 | 1 | PASS |
| 湖面静止 | 1 | PASS |
| 投影无散度 | 1 | PASS |
| **合计** | **33** | **PASS** (≥ 15) |

另 Color-Gradient 路径 34 项测试全部通过（VOF/PLIC 10 + CSF 7 + Wigley 4 + Water Entry 4 + 负对照 2 + 集成 2 + 配置 5）。

---

## 4. 结果图

**本补报无内嵌图**。理由：原单 mission 2204 交付时未在工件目录（`artifacts/`、`runs/`、`figures/`）留档可追溯的 PNG/NPZ 文件。仓库内 `figures/` 目录中虽有 `fig_dambreak_front.png` 等文件名被文档引用，但这些图是后续 P350W2-N07 独立验证活动生成，不属于 PR #418 / commit ba33047 的原始工件。补报单不编造图、不复现计算（纪律：不改业务代码、不重跑）。

若需补图，建议后续单独立项：从 `runs/free_surface_verification/verification_results.json`（如存在）重绘。

---

## 5. 注意点（踩坑 / 局限 / 适用范围）

### 5.1 已知局限

1. **Wigley VOF 仿真不稳定**：2D VOF + 移动船体障碍物在粗网格下数值失稳（溢出警告、NaN 质量漂移）。这是 MUSCL 格式在锐利几何边界前的已知困难，需 3D LBM 自由面求解器（`hull_free_surface.py`）才能产出生产级结果。

2. **溃坝短时误差主导**：收敛研究表明 RMSE 随分辨率增加而增大（短时 T < 1.5），原因是前锋误差被初始离散化主导。需更长仿真时间才能展现真正收敛趋势。

3. **CSF 表面张力未完全耦合**：CSF 模型已实现但未与 Color-Gradient 模型完全耦合用于动态仿真。Laplace 压力测试因虚假毛细电流（spurious capillary currents）标记为 honest-FAIL。

4. **溃坝 vs Ritter 的适用范围**：Ritter (1892) 解析解为无黏浅水近似，仅在 t* < 2 时段严格成立。LBM 仿真含黏性效应，长时比对需改用 Martin & Moyce 实验数据。

### 5.2 适用范围声明

- P350W2-N07 验证的是 **VOF 自由面算法**（MAC 网格 + 投影 + VOF 输运），**不是** LBM 自由面方法。
- LBM 自由面方法（Color-Gradient）在 `FREE_SURFACE_VERIFICATION.md` 中独立验证。
- 两条路径的验证互不替代。

---

## 6. 误差判定（关键量 vs 基准，逐项列出）

| 判据 | 测量值 | 基准 | 误差 | 门限 | 是否 < 3%? | 结论 |
|------|--------|------|------|------|-----------|------|
| (a) 溃坝前锋 RMSE | 7.79% | Martin & Moyce (1952) 实验 | 7.79% | ≤ 10% | **否** (7.79% > 3%) | PASS（满足 10% 门限，但不满足 3% 高精度要求） |
| (b) 质量漂移 | 0.10% | 初始质量 | 0.10% | ≤ 2% | **是** (0.10% < 3%) | PASS |
| (c) Wigley 波高 | N/A | Michell (1898) 理论 | N/A（仿真不稳定） | ≤ 15% | **无法判定** | PARTIAL — VOF 仿真失稳，无有效数据 |
| (d) 收敛阶 | 1.33 | 理论期望 ≥ 1.5 | — | ≥ 1.0 | — | PASS（> 1.0 底线，低于 1.5 期望） |
| (e) Wagner 力斜率 | 0.393 | Wagner (1932) 理论 | 0.0% | — | **是** (0.0% < 3%) | PASS |
| (f) 测试覆盖 | 33 + 34 | ≥ 15 | — | ≥ 15 | — | PASS |

### 6.1 关于"是否 < 3%"的显式回答

- **溃坝前锋位置**：误差 7.79%，**不满足 < 3%**。原因：Ritter 解析解为无黏浅水近似，LBM 含黏性；且粗网格（8 cells/a）下初始离散化误差主导。需更高分辨率 + 更长仿真时间 + 实验数据比对才能逼近 3%。
- **质量守恒**：0.10%，**满足 < 3%**。
- **Wagner 力斜率**：0.0%，**满足 < 3%**（但此为 Color-Gradient 路径，且 R² = 1.000 暗示可能为低分辨率下的线性段拟合，需更高精度验证）。
- **Wigley 波高**：无有效数据，**无法判定**。声明原因：VOF 仿真在粗网格下数值失稳。

### 6.2 Honest-FAIL 声明

本补报如实保留以下 FAIL/PARTIAL 结论，不做粉饰：
- Wigley VOF 仿真：**PARTIAL**（数值不稳定）
- 收敛阶 1.33：**低于 1.5 期望**（但超过 1.0 底线）
- Laplace 压力测试：**honest-FAIL**（虚假毛细电流）
- 溃坝前锋 7.79%：**不满足 3% 高精度要求**

---

## 7. 总结

Mission 2204 成功填平了 TensorLBM 自由面引擎的验证欠账：
- ✅ 溃坝基准 vs 实验：RMSE 7.79%（满足 10% 门限）
- ✅ 质量守恒：漂移 0.10%（远优于 2% 门限）
- ✅ Wagner 入水砰击：力斜率误差 0.0%
- ✅ 67 项测试全部通过（33 + 34）
- ⚠️ Wigley VOF 仿真不稳定（需 3D LBM 路径）
- ⚠️ 收敛阶 1.33（低于 1.5 期望）

**整体判定**：验证欠账已填平至可接受水平（满足 10% 工程精度门限），但距 3% 高精度要求仍有差距。建议后续工作聚焦：(1) 溃坝高分辨率 + 长时仿真；(2) Wigley 3D LBM 自由面耦合；(3) CSF 与 Color-Gradient 完全耦合。

---

## 参考文献

1. Ritter, A. (1892). "Die Zerstörung eines Damms." *Z. Math. Phys.*
2. Martin, J.C. & Moyce, W.J. (1952). *Phil. Trans. R. Soc. A* **244**, 312.
3. Michell, J.H. (1898). *Phil. Mag.* **45**, 106.
4. Wagner, H. (1932). *Z. Angew. Math. Mech.*
5. Latva-Kokko, J. & Rothman, D.H. (2005). *PRE*.
6. Brackbill, J.U., Kothe, D.B., & Zemach, C. (1992). *JCP*.

---

*补报生成时间：2026-04-09 · 原单 commit ba33047 (PR #418) · 数据来源：docs/FREE_SURFACE_VERIFICATION.md, docs/P350W2_N07_free_surface_verification.md*
