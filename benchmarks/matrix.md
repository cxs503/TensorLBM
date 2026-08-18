# TensorLBM 共性模块功能矩阵 Benchmark

> 思路：不逐个试错，而是按**共性模块功能矩阵（碰撞 × 湍流 × 边界 × 物理案例）**系统性组合排查。
> 每个组合跑一个物理案例，误差 ≤3%（真实模拟、无外推）才入库 verified/。

## 功能矩阵盘点（2026-08-18）

### 碰撞模型（Collision）
| 编号 | 模型 | 入口 | 格 |
|------|------|------|-----|
| C1 | BGK | solver.collide_bgk / collide_bgk3d | D2Q9/D3Q19 |
| C2 | MRT | collide_mrt / general_sim AUTO | D2Q9/D3Q19 |
| C3 | cumulant | collide_cumulant_d2q9/d3q19/d3q27 | 全 |
| C4 | cascaded | collide_cascaded_d3q19/d3q27 | D3Q19/D3Q27 |
| C5 | KBC/entropic | collide_kbc_d3q19 / entropic_kbc | D3Q19/D3Q27 |
| C6 | D3Q27 MRT | d3q27_collide | D3Q27 |

### 湍流模型（Turbulence/LES）
| 编号 | 模型 | 入口 |
|------|------|------|
| T0 | 无（DNS/层流） | — |
| T1 | Smagorinsky | collide_smagorinsky_bgk/mrt (2D/3D) |
| T2 | WALE | collide_wale_bgk/mrt (2D/3D) |
| T3 | Vreman | collide_vreman_bgk/mrt (2D/3D) |
| T4 | 动态 Smagorinsky | collide_dynamic_smagorinsky_* |
| T5 | DES/DDES | ddes.py / des_turbulence.py |
| T6 | RANS k-ε | rans_ke.py |

### 边界条件（Boundary）
| 编号 | 类型 | 入口 |
|------|------|------|
| B1 | Zou-He 速度入口 | zou_he_inlet_velocity(_3d) |
| B2 | Zou-He 压力出口 | zou_he_outlet_pressure(_3d) |
| B3 | far-field | far_field_bc_2d/3d |
| B4 | bounce-back（无滑移） | bounce_back_cells(_3d) |
| B5 | free-slip | free_slip_cells_3d / *_walls_3d |
| B6 | wall-function | wall_function_3d / wall_function_d3q27 |
| B7 | periodic | stream 模运算 / engine 自动 |

### 几何/物理案例（Problem）
| 编号 | 案例 | 参考基准 | 判别力 |
|------|------|---------|--------|
| P1 | Taylor-Green 2D 涡衰减 | 解析 γ=2νk² | **碰撞/耗散** |
| P2 | Poiseuille 2D 管流 | 解析抛物线 | **边界/粘性** |
| P3 | 方腔流 Re=100 | Ghia 1982 | **对流/角点** |
| P4 | 球 Re=100 阻力 | Cd=1.087 | **阻力/壁面** |
| P5 | 圆柱 Re=100 | Cd≈1.35 | **分离流/St** |

## 组合策略（同一问题只保留最优配置，精力放在不同问题上）

**原则**：每个物理问题只做 1 个 benchmark（用该问题下最优的碰撞/湍流/边界组合），
不扫描组合矩阵。矩阵的价值是**盘点能力**（知道有什么），不是穷举测试。

| 案例 | 采用配置 | 参考基准 | 状态 |
|------|---------|---------|------|
| P1 Taylor-Green | C1 BGK（已验证） | 解析 γ=2νk² | ✅ 已入库 -0.035% |
| P2 Poiseuille | C1 BGK + Zou-He | 解析抛物线 | ✅ 已入库 0.18% |
| P3 方腔流 Re=100 | C2 MRT + BB 壁面 | Ghia 1982 | 子 agent 进行中（GPU1） |
| P4 球 Re=100 | GeneralSimEngine（MRT+BB） | Cd=1.087 | 子 agent 进行中 |
| P5 圆柱 Re=100 | MRT+far-field | Cd≈1.35 | 子 agent 进行中 |
| P6 SUBOFF Re=1000 | GeneralSimEngine | Ct≈0.004 | 子 agent 完成，审阅中 |
| P7 NACA 翼型 | airfoil_benchmark | Cl 文献 | 子 agent 中断，待重派 |
| P8 Blasius 平板 | turbulent_channel | f'(η) 解析 | 子 agent 中断，待重派 |
| P9 空化气泡 | cavitation（需先修 EOS 缺口） | RP 理论 | 缺口分析完成 |
| P10 后向台阶 | backward_facing_step | Armaly 1984 | 12.2% 未达标，待改进 |

**新问题方向（未覆盖）**：多相/自由表面（Laplace/溃坝）、声学、RANS 通道、
D3Q27 高精度、AMR 网格收敛、壁面函数高 Re。

## 判定标准（不变）

- 真实模拟（无外推），误差 ≤3% 才入库 verified/
- 共性模块路径：GeneralSimEngine / 库 solver / 物理 run_* 入口（禁手写 collide-stream）
- 每个组合一个 result.json + README.md
