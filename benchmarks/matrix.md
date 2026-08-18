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

## 组合策略（不全扫描，按判别力聚焦）

| 案例 | 扫碰撞 (C) | 扫湍流 (T) | 扫边界 (B) | 说明 |
|------|-----------|-----------|-----------|------|
| P1 TG | C1/C2/C3 | T0 | B7 | **碰撞模型判别**（无湍流） |
| P2 Poiseuille | C1/C2 | T0 | B1/B2/B4 | **边界+粘性判别** |
| P3 方腔 | C1/C2/C3 | T0 | B4/B5 | 对流+角点 |
| P4 球 | C1/C2/C3 | T0/T1/T2 | B3/B4/B6 | **阻力+湍流+壁面**（最全） |
| P5 圆柱 | C2 | T1/T2 | B3/B4 | 分离流 |

## 优先级（先做判别力最强、最易达标的）

1. **P1×{C1,C2,C3}**：TG 扫碰撞——测碰撞模型数值耗散（已入库 C1 BGK -0.035%，补 C2 MRT/C3 cumulant）
2. **P4×{C2,T0/T1,T2}**：球扫湍流——LES 模型判别（Smag/WALE 对阻力影响）
3. **P3×{C1,C2}**：方腔扫碰撞（已入库候选）
4. **P5×{T1,T2}**：圆柱扫 LES

## 判定标准（不变）

- 真实模拟（无外推），误差 ≤3% 才入库 verified/
- 共性模块路径：GeneralSimEngine / 库 solver / 物理 run_* 入口（禁手写 collide-stream）
- 每个组合一个 result.json + README.md
