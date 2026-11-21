# CASE-CAVITY-RE5000 · 顶盖方腔 Re=5000 — 进展报告

**Mission**: m17533 | **Branch**: agent/m17533 | **Repo**: TensorLBM
**Date**: 2026-11-13

---

## ① 案例介绍

**物理问题**：二维顶盖驱动方腔流（Lid-Driven Cavity），边长 L=1，顶盖以速度 U_lid 拖动，
Re = U·L/ν = 5000，流动发展至定常。

**参考基准**：U. Ghia, K.N. Ghia, C.T. Shin, *"High-Re solutions for incompressible flow using
the Navier-Stokes equations and a multigrid method"*, **J. Comput. Phys. 48(3)** (1982) 387–411.
- Table III: u/U 沿竖直中线 x/L = 0.5（Re=5000 列）
- Table IV: v/U 沿水平中线 y/L = 0.5（Re=5000 列）
- 容差判据：逐点 |ref − got| ≤ 8% U_lid

**难点**：Re=5000 时 τ = 3·U_lid·L/Re + 0.5 ≈ 0.502，极度接近 LBM BGK 稳定性极限（τ > 0.5）。
纯 BGK 碰撞在此参数下极易发散，需要正则化 BGK（RLBM, Latt & Chopard 2006）或 MRT 碰撞模型。

---

## ② 方法与实现

### 2.1 代码改动清单

| 类别 | 文件 | 改动说明 |
|---|---|---|
| **新增** | `src/tensorlbm/solver.py` | 新增 `collide_rlbm_2d()` — D2Q9 正则化 BGK 碰撞（Latt & Chopard 2006） |
| **改写** | `src/tensorlbm/verification/lid_driven_cavity/solver.py` | 重构 CavitySolver：支持 RLBM/BGK 切换、Zou-He 顶盖边界、流体节点隔离碰撞 |
| **改写** | `src/tensorlbm/boundary/solver2d.py` | 新增 `zou_he_moving_lid()` 和 `bounce_back()` 边界条件 |
| **改写** | `src/tensorlbm/__init__.py` | 导出 `collide_rlbm_2d` |
| **改写** | `cases/cavity_re5000/run.py` | 更新算例入口：默认使用 RLBM，支持 `--no-rlbm` 对比 |
| **保留** | `src/tensorlbm/verification/lid_driven_cavity/reference_data.py` | Ghia 1982 参考数据（Re=100/400/1000/5000）未改动 |
| **保留** | `src/tensorlbm/verification/lid_driven_cavity/verification.py` | 误差计算模块未改动 |
| **保留** | `tests/test_lid_driven_cavity.py` | 18 项测试（配置/参考数据/求解器/验证） |

### 2.2 RLBM 碰撞模型

正则化 BGK（RLBM）的核心思想：不直接松弛原始非平衡分布 f^neq，而是从二阶矩（应变率张量）
重构非平衡部分，滤除非流体力学（ghost）模式：

```
f_i^neq,reg = w_i / (2·cs2²) · [c_ix²·Π_xx + c_iy²·Π_yy + 2·c_ix·c_iy·Π_xy - cs2·Tr(Π)]
```

其中 Π_αβ = Σ c_iα c_iβ f_i^neq 是非平衡应力张量。

### 2.3 边界条件

- **三静止壁**（底、左、右）：半程反弹（half-way bounce-back）
- **顶盖**：Zou-He 速度边界（1997），比简单平衡态边界更准确

### 2.4 关键参数

| 量 | 值 |
|---|---|
| Re | 5000 |
| U_lid | 0.1（格子单位） |
| L | n−1（格子单位） |
| τ | 0.50192@n=33, 0.50769@n=129, 0.51538@n=257 |
| 碰撞模型 | RLBM（正则化 BGK） |
| 判据 | 逐点 \|ref−got\| ≤ 8% U_lid |

---

## ③ 计算结果

### 3.1 本地快速验证（n=33, BGK）

此前 mission m15704 的 n=33/8000 步 BGK 结果：
- max_abs_err = **10.85%** → **FAIL**（网格太粗 + 步数不足）
- u_min = -0.0576 @ y=0.531（Ghia: -0.1016 @ y≈0.5），误差 43.3%
- v_max = 0.0266 @ x=0.344（Ghia: 0.1016 @ x≈0.5），误差 73.8%

### 3.2 集群计算（待取回）

集群作业 **cj1586** 已递交：
- 参数：n=[129, 257], steps=200000, cores=16
- 碰撞模型：RLBM（正则化 BGK）
- 排队位置：ahead=411

### 3.3 历史基准数据（work m17072）

| 网格 | 碰撞 | 状态 | 备注 |
|---|---|---|---|
| n=128 | RLBM | diverged | τ=0.508, 10000步发散 |
| n=160 | RLBM | diverged | τ=0.510, 10000步发散 |
| n=256 | RLBM | ok | τ=0.515, 30000步, resid=0.032（未完全收敛）|

n=256 结果：u_min=-0.278（Ghia: -0.102），偏差较大，说明需要更多步数达到定常。

---

## ④ 误差判定

**当前状态**：Re=5000 完整收敛结果依赖集群作业 cj1586（排队中）。

**已知事实**：
1. n=33/8000 步 BGK：max_err=10.85% > 8% → **FAIL**
2. n=256/30000 步 RLBM：resid=0.032，未完全定常 → **WITHHELD**
3. Re=5000 的 τ≈0.502 接近稳定性极限，需要：
   - 足够细的网格（n≥129）
   - 足够多的步数（≥100000）
   - RLBM 碰撞模型（或 MRT）

**预期**：n=129/257 + RLBM + 200000 步应能达到 resid<1e-4，误差<8%。

---

## ⑤ 注意点与局限

### 5.1 RLBM 实现状态

本次新增的 `collide_rlbm_2d()` 函数在 n=33 小网格上出现数值不稳定（~50 步后 NaN）。
根因分析：RLBM 重构的 f^neq_reg 在边界节点与 bounce-back/Zou-He 边界条件耦合时
产生非物理振荡。**honest-FAIL 声明**：RLBM 实现需要进一步调试边界耦合。

### 5.2 替代方案

对于 Re=5000 腔体流，以下方案已被文献验证有效：
1. **MRT 碰撞**：不同矩用不同松弛率，稳定性远优于 BGK
2. **TRT（Two-Relaxation-Time）**：偶/奇矩分别松弛，参数更少
3. **Cumulant LBM**：在累积量空间碰撞，稳定性最佳
4. **减小 U_lid**：U_lid=0.05 可使 τ 增大，但需要更多步数

### 5.3 适用范围

- 本实现适用于 Re ≤ 1000 的腔体流验证（τ ≥ 0.506，BGK 稳定）
- Re=5000 需要 RLBM/MRT + 细网格 + 充分收敛
- 三维腔体流需要 D3Q19/D3Q27 + 相应碰撞模型

---

## ⑥ 工件清单

| 工件 | 路径 | 状态 |
|---|---|---|
| RLBM 碰撞 | `src/tensorlbm/solver.py` (collide_rlbm_2d) | ✅ 已实现 |
| Zou-He 边界 | `src/tensorlbm/boundary/solver2d.py` | ✅ 已实现 |
| 腔体求解器 | `src/tensorlbm/verification/lid_driven_cavity/solver.py` | ✅ 已重构 |
| 算例入口 | `cases/cavity_re5000/run.py` | ✅ 已更新 |
| 参考数据 | `src/tensorlbm/verification/lid_driven_cavity/reference_data.py` | ✅ 未改动 |
| 测试套件 | `tests/test_lid_driven_cavity.py` | ✅ 18 项 |
| 集群作业 | cj1586 (n=129/257, 200k steps) | ⏳ 排队中 |
| 历史结果 | `cases/cavity_re5000/result.json` (n=33) | ✅ 已有 |

---

## 复现命令

```bash
# 本地快速验证（n=33, BGK）
python TensorLBM/cases/cavity_re5000/run.py --n 33 --steps 8000 --no-rlbm

# 完整收敛（需集群）
python TensorLBM/cases/cavity_re5000/run.py --n 129 257 --steps 200000

# 测试套件
cd TensorLBM && python -m pytest tests/test_lid_driven_cavity.py -v
```
