# Mission 366 补报：torch_fused_obstacle_xfar_les 纯 PyTorch 等价实现

> **任务类型**：补报单（原 mission 366 已交付，缺论文级进展报告）  
> **仓库**：TensorLBM  
> **工件**：`artifacts/m1_torch_equiv.json`、`M1_P2_SUMMARY.md`  
> **结论**：honest-FAIL — 精度判据 ✅ 通过，性能判据 ⚠️ 未验证（无 CUDA 环境）

---

## 1. 案例介绍

### 1.1 物理背景

TensorLBM 的 Triton kernel `triton_fused_obstacle_xfar_les` 是 SUBOFF 水动力仿真的核心算子，在单次 kernel launch 中完成 D3Q19 格子 Boltzmann 方法的 **streaming → full-way bounce-back → collision（含 Smagorinsky LES 闭合）** 三步操作。该 kernel 依赖 Triton 编译器，在部分硬件（如 SDAA）上仅能以 `TRITON_INTERPRET=1` 解释模式运行，性能严重退化。

### 1.2 任务目标

实现 `torch_fused_obstacle_xfar_les`——一个 **纯 PyTorch** 等价函数，满足：

| 判据 | 阈值 | 说明 |
|------|------|------|
| ① 逐值精度 | L₂ < 1×10⁻⁶ | 与 TRITON_INTERPRET=1 模式逐元素对比 |
| ② 性能加速 | ≥ 3× vs interpret | 原 10× 已裁定过严，放宽至 3× |
| ③ 回归测试 | 全绿 | 既有测试不受影响 |

### 1.3 参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 网格 | 32×32×32 | 单步精度测试 |
| 分子粘度 ν_lb | 0.005 | 对应 τ = 3ν + 0.5 = 0.515 |
| Smagorinsky Cs | 0.0 / 0.1 | 无 LES / 有 LES |
| 碰撞模型 | BGK | 主测试；CM/CUMULANT 辅助 |
| 固体单元格 | 925（球体 r=6） | 占比 ≈ 2.8% |
| 多步测试 | 10 步演化 | 稳定性验证 |

---

## 2. 方法

### 2.1 核心语义发现

**关键洞察**：Triton kernel 的执行顺序是 `collide(BB(stream(f)))`，而非 `BB(stream(collide(f)))`。

这一发现来自对 `test_triton_obstacle_wall_parity.py` 的深入分析，测试明确给出参考链：

```python
f_eff_ref = bounce_back_cells_3d(stream3d(f0), solid)
ref = collide_advanced_3d("d3q19", collision, f_eff_ref, tau=TAU)
```

### 2.2 实现流程

`torch_fused_obstacle_xfar_les` 位于 `src/tensorlbm/triton_fused_obstacle.py`（第 2670 行起，约 230 行），执行流程：

```
Step 1: Streaming  — f_streamed[q, x] = f[q, x - c_q]  (周期性边界)
Step 2: Bounce-back — f_bb[q, x] = f_streamed[opp[q], x]  (固体单元格)
Step 3: 宏观量计算  — ρ, u 从 f_bb 计算
Step 4: 平衡态计算  — f_eq 从 ρ, u 计算
Step 5: 非平衡态    — f_neq = f_bb - f_eq
Step 6: 松弛率计算  — ω = 1/τ_eff（含 Smagorinsky 自洽闭合）
Step 7: 碰撞        — f_out = f_bb - ω·(f_bb - f_eq)
Step 8: 力计算      — Ladd (1994) 动量交换法（可选）
Step 9: 输出        — copy to out buffer
```

### 2.3 功能特性

- ✅ Full-way bounce-back（全反射边界）
- ✅ BGK 碰撞（分子粘度）
- ✅ Smagorinsky LES 闭合（Cs > 0 时启用，含 τ_eff 自洽求解与 [0.5001, max] 钳位）
- ✅ 外部 τ_eff 场支持
- ✅ Ladd 力计算（三轴力累加器）
- ✅ CUMULANT 碰撞（简化实现）
- ✅ 三轴周期性边界条件
- ⚠️ CM 碰撞使用简化 BGK 类松弛（待完整实现）

### 2.4 工具链

- PyTorch ≥ 2.0（纯张量操作，无 Triton 依赖）
- 参考链：`stream3d` + `bounce_back_cells_3d` + `collide_advanced_3d`
- 测试脚本：`test_reference.py`（单步精度）、`test_multistep.py`（多步稳定性）、`debug_trace.py`（单元格级对比）

---

## 3. 计算结果

### 3.1 单步精度测试

| 指标 | 数值 | 阈值 | 判定 |
|------|------|------|------|
| L₂ 范数（全局） | **1.21×10⁻¹⁰** | < 1×10⁻⁶ | ✅ PASS |
| 最大绝对误差 | 2.98×10⁻⁸ | — | — |
| 最大相对误差 | 1.18×10⁻⁷ | — | — |
| 流体区域 L₂ | 1.09×10⁻¹⁰ | — | — |
| 固体区域 L₂ | 3.30×10⁻¹⁰ | — | — |

### 3.2 多步稳定性测试（10 步）

| 指标 | 数值 | 阈值 | 判定 |
|------|------|------|------|
| 最终 L₂ | **1.39×10⁻⁸** | < 1×10⁻⁵ | ✅ PASS |
| 最大逐步误差 | 3.28×10⁻⁷ | — | — |
| NaN/Inf | 无 | — | ✅ PASS |
| 稳定性评级 | 优秀 | — | — |

### 3.3 性能测试

| 指标 | 数值 | 阈值 | 判定 |
|------|------|------|------|
| 加速比 vs TRITON_INTERPRET=1 | **未测量** | ≥ 3× | ⚠️ 未验证 |
| 原因 | 当前环境无 CUDA 设备 | — | — |

> **预期加速比**：≥ 3×。理由：纯 PyTorch 操作完全向量化，避免 Triton 解释器逐行 dispatch 开销。在 GPU 原生 CUDA 模式下应显著快于解释模式。实际性能测试需在有 CUDA 设备的环境中运行。

### 3.4 回归测试

| 检查项 | 状态 |
|--------|------|
| 模块导入成功 | ✅ |
| `torch_fused_obstacle_xfar_les` 可用 | ✅ |
| 原 `triton_fused_obstacle_xfar_les` 仍可用 | ✅ |
| 既有 API 无破坏性变更 | ✅ |
| 完整回归测试（需 CUDA） | ⏳ 待执行 |

---

## 4. 结果图

本任务为 **纯代码实现 + 数值精度验证**，不涉及流场可视化或物理量对比曲线。理由如下：

1. **验证对象是算子等价性**：对比的是两个函数在同一输入下的逐元素输出差异（L₂ 范数），而非物理量的空间分布。
2. **测试网格为 32³ 均匀立方体含球体障碍物**：非标准算例（如 SUBOFF），不适合生成有物理意义的云图。
3. **精度已达 10⁻¹⁰ 量级**：差异在浮点舍入误差范围内，可视化无信息量。

如需流场可视化，建议在 SUBOFF 完整算例（Re=10⁵, 128³ 网格）上运行纯 torch 路径后生成速度/压力云图。

---

## 5. 注意点

### 5.1 踩过的坑

1. **执行顺序是关键**：最初尝试按 `collide → stream → bounce-back` 实现，结果与 Triton kernel 不一致。通过深入分析 `test_triton_obstacle_wall_parity.py` 发现正确顺序为 `stream → bounce-back → collide`。

2. **力计算的语义**：Ladd 力计算使用 `f_streamed`（bounce-back 前），而非 `f_bb`。这与 kernel 语义一致，但容易混淆。

3. **Smagorinsky 闭合的 τ_eff 钳位**：必须钳位到 [0.5001, max(1.0, τ_mol + 0.5)]，否则可能出现非物理的负粘度或过大粘度。

### 5.2 局限性

1. **CM 碰撞未完整实现**：当前使用简化 BGK 类松弛，待后续补充完整的级联中心矩碰撞。
2. **性能未验证**：无 CUDA 环境，无法测量实际加速比。预期 ≥ 3× 但需实测确认。
3. **壁面精度**：Full-way bounce-back 为一阶精度，对 SUBOFF 级定量阻力预测需验证网格收敛性或改用 MRT。
4. **Smagorinsky 常数**：Cs 为常数，无动态过程（dynamic procedure）。

### 5.3 适用范围

- ✅ 验证工具：与 TRITON_INTERPRET=1 模式逐值对比
- ✅ 回退路径：在无 Triton 支持的设备（如部分 SDAA 配置）上使用
- ✅ 调试工具：纯 PyTorch 实现便于断点调试和梯度检查
- ⚠️ 生产路径：需在有 CUDA 环境完成性能验证后方可替代 Triton kernel

---

## 6. 误差判定

### 6.1 精度 vs 基准

| 关键量 | 基准 | 实现值 | 绝对误差 | 相对误差 | 是否 < 3% |
|--------|------|--------|----------|----------|-----------|
| 单步 L₂ 范数 | < 1×10⁻⁶ | 1.21×10⁻¹⁰ | — | — | ✅ **是**（远超阈值） |
| 10 步 L₂ 范数 | < 1×10⁻⁵ | 1.39×10⁻⁸ | — | — | ✅ **是**（远超阈值） |
| 最大绝对误差 | — | 2.98×10⁻⁸ | — | — | — |
| 最大相对误差 | — | 1.18×10⁻⁷ | — | — | ✅ **是**（远 < 3%） |

### 6.2 性能 vs 基准

| 关键量 | 基准 | 实现值 | 判定 | 说明 |
|--------|------|--------|------|------|
| 加速比 vs TRITON_INTERPRET=1 | ≥ 3× | **未测量** | ⚠️ **无法判定** | 无 CUDA 环境，声明原因：当前工作区无 GPU 设备 |

### 6.3 结论

- **精度判据**：✅ **通过**。L₂ 范数 1.21×10⁻¹⁰ 远超 1×10⁻⁶ 阈值，相对误差 1.18×10⁻⁷ 远 < 3%。
- **性能判据**：⚠️ **未验证**。预期 ≥ 3× 但无 CUDA 环境无法实测。此为 honest-FAIL，不粉饰。
- **回归判据**：✅ **部分通过**。导入验证通过，完整回归需 CUDA 环境。

---

## 附录：使用示例

```python
from tensorlbm.triton_fused_obstacle import torch_fused_obstacle_xfar_les

# 纯 PyTorch 路径（不需要 Triton）
f_out = torch_fused_obstacle_xfar_les(
    f,           # (19, nz, ny, nx) 分布函数
    nu_lb=0.005, # 粘度
    obstacle,    # (nz, ny, nx) int8 掩码
    Cs=0.0,      # Smagorinsky 常数（0 = 无 LES）
    delta=1.0,   # LES 滤波宽度
    collision='BGK'
)
```

---

**报告生成时间**：2026-10-30  
**报告类型**：补报（mission 366 原交付缺报告）  
**报告作者**：TensorPlatform Agent (m740)
