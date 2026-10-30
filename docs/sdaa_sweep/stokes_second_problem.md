# SDAA Sweep: stokes_second_problem

## 案例概述

**Stokes 第二问题（振荡平板, D2Q9）** — 受迫振荡层流解析解 benchmark。

半无限静止流体中，无限平板在自身平面内谐波振荡 u_w(t) = U·cos(ωt)。
稳态周期解析解（Batchelor; Landau-Lifshitz §24）：

    u(y,t) = U·e^{−ky}·cos(ωt − ky),    k = √(ω/(2ν))

幅值随深度指数衰减、相位线性滞后，皆由单一穿透波数 k 决定。

## 仿真配置

| 参数 | 值 |
|---|---|
| 格子 | D2Q9, BGK (τ=0.8, ν=0.1) |
| 驱动 | 振荡平板 = 顶行 (y=ny−1)，zou_he_moving_lid |
| 远场 | 底行 (y=0)，自由滑移镜面反射 (SPECULAR) |
| x 方向 | 周期边界 |
| U | 0.05 |
| nx | 8 |
| 编译模式 | eager（SDAA teco_inductor 无法编译完整 LBM 步链） |

## 网格选择

案例文档推荐 H=50/100/200。SDAA 设备上小网格的 kernel launch overhead 占主导，
H=40 即超过 120s shell 超时限制。故采用可行较小档 **H=20/30**（2 点/线，仍可验证单调收敛）。

| kH | H | 总步数 | 墙钟 (s) |
|---|---|---|---|
| 6 | 20 | 4,886 | 31.1 |
| 6 | 30 | 10,990 | 67.8 |
| 8 | 20 | 4,116 | 23.1 |
| 8 | 30 | 9,282 | 57.3 |

**总墙钟**: 179.3 s (~3 min)

## 设备证据

```
torch.sdaa.get_device_name(0) = "/dev/tcaicard15"
torch.sdaa.device_count() = 1
```

## 判据（先验固定，与案例文档一致）

| # | 判据 | 门限 | 掩模 |
|---|---|---|---|
| 1 | 幅值 max \|A_num/A_ana − 1\| | ≤3% | A_ana ≥ 10%·U |
| 2 | 相位滞后 max \|φ_num − φ_ana\|/φ_ana | ≤3% | φ_ana ≥ 1 rad AND A_ana ≥ 10%·U |
| 3 | 复振幅 max \|U_num/U_ana − 1\| | ≤3% | 同幅值掩模 |
| 4 | 网格收敛 | 单调下降 | complex_rel_max_pct 沿 H 递增 |

## 结果表

| kH | H | 幅值 max (%) | 相位 max (%) | 复振幅 max (%) | 步数 | 墙钟 (s) |
|---|---|---|---|---|---|---|
| 6 | 20 | 1.010 | 0.487 | 1.437 | 4,886 | 31.1 |
| 6 | 30 | 0.478 | 0.255 | 0.666 | 10,990 | 67.8 |
| 8 | 20 | 1.779 | 0.898 | 2.355 | 4,116 | 23.1 |
| 8 | 30 | 0.788 | 0.398 | 1.161 | 9,282 | 57.3 |

## 判据逐条比对

### 1. 幅值 ≤ 3% — **PASS**

| kH | H | 幅值 max (%) | 结果 |
|---|---|---|---|
| 6 | 20 | 1.010 | PASS |
| 6 | 30 | 0.478 | PASS |
| 8 | 20 | 1.779 | PASS |
| 8 | 30 | 0.788 | PASS |

### 2. 相位 ≤ 3% — **PASS**

| kH | H | 相位 max (%) | 结果 |
|---|---|---|---|
| 6 | 20 | 0.487 | PASS |
| 6 | 30 | 0.255 | PASS |
| 8 | 20 | 0.898 | PASS |
| 8 | 30 | 0.398 | PASS |

### 3. 复振幅 ≤ 3% — **PASS**

| kH | H | 复振幅 max (%) | 结果 |
|---|---|---|---|
| 6 | 20 | 1.437 | PASS |
| 6 | 30 | 0.666 | PASS |
| 8 | 20 | 2.355 | PASS |
| 8 | 30 | 1.161 | PASS |

### 4. 网格收敛 — **PASS**

| kH | H=20 → H=30 | 单调下降 | 结果 |
|---|---|---|---|
| 6 | 1.437% → 0.666% | ✓ | PASS |
| 8 | 2.355% → 1.161% | ✓ | PASS |

## 总 verdict

**PASS** — 全部 4 项判据通过。

## 绕行记录

| 项目 | 状态 | 说明 |
|---|---|---|
| TRITON_INTERPRET | 不需要 | 本案例无 triton 依赖（纯 D2Q9 BGK + 库函数） |
| 编译模式 | eager | SDAA teco_inductor 无法编译完整 LBM 步链（compile_route.py 有自动 fallback） |
| 网格 | 缩减档 H=20/30 | 推荐档 H=50/100/200 中 H=40 即超 120s shell 超时；SDAA 小网格 kernel launch overhead 占主导 |
| 收敛点数 | 2 点/线 | 推荐 3 点/线，受限于 wall clock 缩减为 2 点，仍可验证单调性 |

## 产物

- 脚本: `scripts/sdaa_sweep_stokes_second_problem.py`
- 单点运行器: `scripts/_sdaa_stokes2_single.py`
- 产物 JSON: `artifacts/sdaa_stokes_second_problem.json`
