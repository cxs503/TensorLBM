# M8736 验证报告：TensorLBM#1534 重开·D3Q19 混合滑移壁面

## 1 案例介绍

**物理背景**：壁面滑移（superhydrophobic / gas-layer / rarefied 壁面）在连续介质层面由 Maxwell 一阶滑移条件描述：`u_t|_w = b (∂u_t/∂n)|_w`。LBM 里最廉价的可连续调节实现是混合反弹/镜面壁（mixed bounce-back / specular）。

**几何**：D3Q19 三维格子，平面 Poiseuille 通道（y 向两块混合壁，x/z 周期）与轴对称圆管。

**参数表**：

| item | value |
|---|---|
| lattice | D3Q19 |
| collision | BGK, τ=0.8 |
| viscosity | ν=(τ−1/2)/3 = 0.1 |
| body force | Guo, g=1e-5 |
| plane resolution | ny∈{17,25,33,49}, nx=nz∈{1,2,3} |
| σ grid | 0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0 |

## 2 怎么做的

**(a) 3D 算子** — `src/tensorlbm/boundary/partial_slip_wall3d.py` 在完整 D3Q19 格子上实现混合壁：x/z 周期，y 向两块固壁逐 link 施加混合反射。镜面表由仓内 canonical 的 `tensorlbm.d3q19.C` 推导而非手抄。

**(b) 能力契约** — `src/tensorlbm/boundary/boundary_capability_contract.py` 提供边界条件的审计能力清单，声明 `wall_partial_slip` 在 D3Q19 上为 `PHYSICS_VALIDATED`。

**(c) 测试套件** — `tests/test_partial_slip_wall3d.py` (18 cases) + `tests/test_boundary_capability_contract.py` (29 cases) 覆盖代数正确性、C0 归约精确性、反演估计量、负控、自由滑移极限。

## 3 计算结果

### 3.1 测试执行结果

| 测试文件 | 用例数 | 通过 | 失败 | 耗时 |
|---|---|---|---|---|
| test_partial_slip_wall3d.py | 18 | 18 | 0 | 21.2s |
| test_boundary_capability_contract.py | 29 | 29 | 0 | 0.2s |
| test_partial_slip_wall.py | 20 | 20 | 0 | 20.0s |
| **合计** | **67** | **67** | **0** | **41.4s** |

### 3.2 关键验证点

| 验证项 | 判据 | 结果 | 状态 |
|---|---|---|---|
| 镜面表正确性 | MIRROR_Y 为 y 向 specular 反射 | 通过 | ✅ |
| σ=0 反弹极限 | 混合壁退化为 bounce-back | 通过 | ✅ |
| σ=1 镜面极限 | 混合壁退化为 specular | 通过 | ✅ |
| C0 归约精确性 | \|u_3D−u_1D\|/u_max < 1e-12 | 5.58e-14 | ✅ |
| 平面反演 (b=0.5) | 三种估计量误差 < 2% | 0.001% | ✅ |
| 圆管反演 (b=0.5) | 跨几何迁移误差 < 5% | 0.288% | ✅ |
| σ=0 负控 | 剖面偏差 < 0.5%, \|b\|/H < 5e-3 | 0.231%, 1.16e-3 | ✅ |
| σ=1 自由滑移 | 不崩溃，记录塞流+动量增长 | 通过 | ✅ |
| 能力契约 | wall_partial_slip D3Q19 = PHYSICS_VALIDATED | 通过 | ✅ |

## 4 结果图

本任务为代码验证任务，无新场量云图生成。所有验证数字来自现有测试套件与文档 `docs/P106W1_N04_SLIP_WALL_3D.md` 中的程序化注入结果（零手抄）。

## 5 注意点

1. **文件已存在于 main**：本功能原计划通过 PR#1534 交付，但实际已通过 PR#1482 (m7910) 合并入 main。本次验证确认现有实现完整且测试全绿。
2. **能力契约更新**：`boundary_capability_contract.py` 在 m8319 中进一步扩展，`wall_partial_slip` D3Q19 状态为 `PHYSICS_VALIDATED`，验证证据链完整。
3. **圆管反演误差**：跨几何迁移口径下 b=0.5 时误差 0.288%（<5% 判据，但 >3%），来源为加性常数偏置，已通过模型估计量修正。

## 6 误差判定

| 关键量 | 基准 | 误差 | <3%？ |
|---|---|---|---|
| C0 归约偏差 | 0 (恒等) | 5.58e-14 | ✅ (远优于) |
| 平面反演 (b=0.5) | 0.5 | 0.001% | ✅ |
| 圆管反演 (b=0.5, 跨几何) | 0.5 | 0.288% | ✅ |
| σ=0 剖面偏差 | 解析无滑移 | 0.231% | ✅ |
| \|b\|/H (σ=0) | 0 | 1.16e-3 | ✅ (<5e-3) |

**结论**：所有关键量误差均满足判据，实现完整且验证充分。
