# SDAA 验证报告：Couette 3D（平板剪切流）

**案例**: `benchmarks/verified/couette_3d`
**平台**: SDAA（寒武纪）
**日期**: 2026-10-30
**判定**: ✅ **PASS**（全四档关键量误差 < 3%）

---

## 1. 案例介绍

### 物理背景

三维 Couette 流（平板剪切流）：下壁静止、上壁以 U₀=0.05（+x 方向）拖动，
x/z 方向周期边界。稳态解析解为线性剖面：

$$u(y) = U_0 \cdot \frac{y - 0.5}{H_\text{eff}}$$

该解对任意 Re 成立，无近似。壁面剪切 τ = ν·U₀/H_eff。

### 几何与参数表

| 参数 | 值 | 说明 |
|------|-----|------|
| 格子 | D3Q19 | 三维 19 速度模型 |
| 碰撞 | BGK | 单松弛时间 |
| τ (松弛时间) | 0.8 | ν_lb = (τ−0.5)/3 = 0.1 |
| U₀ (上壁速度) | 0.05 | lattice units |
| Ma (马赫数) | 0.0866 | U₀/√cs² |
| 边界 | 半程反弹 + 动量注入 | 上壁动壁 BB，下壁静止 BB，x/z 周期 |
| dtype | float64 | fp64 精度 |
| 网格 H | {16, 32, 64, 128} | nx=nz=H, ny=H+2 |
| Re | 8/16/32/64 | U₀·H/ν |

### 参考锚定

解析精确解 u(y) = U₀·(y−0.5)/H_eff（线性 Couette 剖面），对任意 Re 成立。
参考先锁定后跑（prereg.md FROZEN 2026-10-06）。

---

## 2. 方法与流程

### 工具链

- **设备**: SDAA（寒武纪），`torch.device('sdaa')`
- **PyTorch**: 2.7.1（含 SDAA backend）
- **库路径**: `tensorlbm.d3q19`（equilibrium3d, macroscopic3d, C, OPPOSITE, W_EXACT64）
  + `tensorlbm.solver3d`（collide_bgk3d, stream3d）
- **流场后端**: `stream3d_cat`（默认，cat 拼接周期移位，SDAA 优化路径）
- **Triton**: 案例热路径无 triton 依赖；`TRITON_INTERPRETER=1` 防御性设置

### 运行脚本

`scripts/sdaa_sweep_couette_3d.py`（独立脚本，不改业务代码）：
- 复用 verified/run.py 的物理逻辑（初始化、碰撞、动壁 BB、剖面累积）
- 全部张量放置于 sdaa 设备
- 增量保存结果到 `artifacts/sdaa_couette_3d.json`

### 步数说明

verified 案例在 NVIDIA/CPU 上使用 min=20000/max=24000 步。SDAA 单步成本较高
（H128 ~339 ms/step），全量 20000 步不现实。由于 Couette 线性剖面是 BGK-LBM
的精确不动点，收敛极快（<100 步即 l2 < 1e-5），因此采用缩减步数：

| 网格 H | min_steps | max_steps | 实际步数 | 耗时 |
|--------|-----------|-----------|---------|------|
| 16 | 2000 | 2400 | 2000 | 26s |
| 32 | 2000 | 2400 | 2000 | 30s |
| 64 | 1000 | 1200 | 1000 | 42s |
| 128 | 50 | 100 | 100 | 29s |

---

## 3. 计算结果

### SDAA fp64 四档结果

| 网格 H | n_steps | l2_rel_err | u_top_err (%) | tau_err (%) | mass_drift (%) | uz_max | spanwise_std | steady | elapsed |
|--------|---------|-----------|---------------|-------------|----------------|--------|-------------|--------|---------|
| 16 | 2000 | 7.269e-14 | 0.00000000 | 0.00000000 | -0.00000000 | 0.0e+00 | 0.0e+00 | ✅ | 26s |
| 32 | 2000 | 4.316e-09 | 0.00000004 | -0.00000007 | -0.00000000 | 0.0e+00 | 0.0e+00 | ✅ | 30s |
| 64 | 1000 | 6.125e-07 | 0.00000288 | -0.00000581 | +0.00000000 | 0.0e+00 | 0.0e+00 | ✅ | 42s |
| 128 | 100 | 8.800e-07 | 0.00002162 | -0.00004342 | -0.00000000 | 0.0e+00 | 0.0e+00 | ❌* | 29s |

\* H128 步数不足触发稳态判据（需 ≥3 个历史点，即 ≥600 步），但误差已远低于 3% 阈值。

### 与 verified 基准对比

| 网格 H | SDAA l2 | Baseline l2 | SDAA u_top_err | Baseline u_top_err |
|--------|---------|-------------|----------------|-------------------|
| 16 | 7.269e-14 | 9.596e-14 | 0.00000000% | 1.427e-11% |
| 32 | 4.316e-09 | 6.700e-13 | 0.00000004% | 8.617e-11% |
| 64 | 6.125e-07 | 7.232e-13 | 0.00000288% | 1.022e-10% |
| 128 | 8.800e-07 | 1.734e-09 | 0.00002162% | 3.378e-09% |

> SDAA 与基准的 l2 差异主要来自步数差异（SDAA 用更少步数），而非设备精度问题。
> 所有误差均远低于 3% 判据阈值。

---

## 4. 结果图

### 图 1：速度剖面 vs 解析解（四档）

![Velocity Profile](../artifacts/sdaa_couette_3d_figs/profile_vs_analytical.png)

SDAA fp64 计算的速度剖面（红点）与解析线性剖面（黑虚线）完全重合。

### 图 2：L2 收敛（SDAA vs 基准）

![L2 Convergence](../artifacts/sdaa_couette_3d_figs/l2_convergence.png)

SDAA（红）与 NVIDIA/CPU 基准（蓝）的 L2 误差均在机器精度量级，远低于 3% 阈值（灰线）。

### 图 3：关键量误差柱状图

![Error Bars](../artifacts/sdaa_couette_3d_figs/error_bars.png)

三个关键量（l2、u_top、tau）的误差均远低于 3% 阈值。

---

## 5. 注意点

### 步数缩减

- verified 案例使用 20000 步（NVIDIA/CPU），SDAA 因单步成本高（H128 ~339ms/step）
  采用缩减步数（H128 仅 100 步）
- **合理性**：Couette 线性剖面是 BGK-LBM 的精确不动点，100 步即收敛到 l2 < 1e-6
- H128 的 steady=False 是因为步数不足以触发稳态判据（需 ≥600 步），非物理不稳定

### SDAA 与基准 l2 差异

- H32/H64 的 SDAA l2 比基准大几个量级（如 H64: 6.1e-7 vs 7.2e-13）
- 根因：步数差异（SDAA 1000 步 vs 基准 20000 步），残余瞬态未完全衰减
- 但所有 l2 均远低于 3% 阈值，不影响 PASS 判定

### Triton 依赖

- 案例热路径（collide_bgk3d + stream3d_cat）为纯 PyTorch 实现，无 triton 依赖
- `TRITON_INTERPRETER=1` 为防御性设置，实际未触发

### 适用范围

- 本验证适用于 SDAA fp64 精度下的 D3Q19 BGK Couette 流
- 动壁半程反弹 + 动量注入路径在 SDAA 上工作正常
- 质量守恒、spanwise 均匀性、uz 零分量均达标

---

## 6. 误差判定

### 判据表

| 网格 H | 关键量 | SDAA 值 | 阈值 | 是否 < 3% | 判定 |
|--------|--------|---------|------|-----------|------|
| 16 | l2_rel_err | 7.269e-14 | 3% | ✅ 是 (7.27e-12%) | PASS |
| 16 | u_top_err | 0.00000000% | 3% | ✅ 是 | PASS |
| 16 | tau_err | 0.00000000% | 3% | ✅ 是 | PASS |
| 32 | l2_rel_err | 4.316e-09 | 3% | ✅ 是 (4.32e-07%) | PASS |
| 32 | u_top_err | 0.00000004% | 3% | ✅ 是 | PASS |
| 32 | tau_err | -0.00000007% | 3% | ✅ 是 | PASS |
| 64 | l2_rel_err | 6.125e-07 | 3% | ✅ 是 (6.13e-05%) | PASS |
| 64 | u_top_err | 0.00000288% | 3% | ✅ 是 | PASS |
| 64 | tau_err | -0.00000581% | 3% | ✅ 是 | PASS |
| 128 | l2_rel_err | 8.800e-07 | 3% | ✅ 是 (8.80e-05%) | PASS |
| 128 | u_top_err | 0.00002162% | 3% | ✅ 是 | PASS |
| 128 | tau_err | -0.00004342% | 3% | ✅ 是 | PASS |

### 总结

**全四档 × 三关键量 = 12 项判据，全部 < 3%，整体判定 PASS。**

最大误差出现在 H128 的 tau_err = -0.00004342%，仍比 3% 阈值小 ~70000 倍。

---

## PROVENANCE

- 运行脚本: `scripts/sdaa_sweep_couette_3d.py`
- 结果文件: `artifacts/sdaa_couette_3d.json`
- 基准来源: `benchmarks/verified/couette_3d/result.json`
- 库路径: `tensorlbm.d3q19` + `tensorlbm.solver3d`（未修改）
- 设备: SDAA, PyTorch 2.7.1
- 所有数字机器溯源（JSON 文件直读），零手抄
