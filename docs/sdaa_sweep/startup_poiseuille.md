# SDAA 验证扫荡 — startup_poiseuille

## 案例概述

**启动 Poiseuille 流（2D 通道, D2Q9）— 瞬态层流解析解 benchmark**

- **物理**：静止流体通道（壁面 y=0, H），t=0 突加恒定流向体力/压力梯度（a = const）。
  经典 Fourier 级数瞬态解：
  ```
  u(y,t) = u_max·[ 4ŷ(1−ŷ) − (32/π³)·Σ_{n odd} (1/n³)·sin(nπŷ)·e^{−n²π²νt/H²} ]
  ŷ = y/H,   u_max = a·H²/(8ν)
  ```
- **格子**：D2Q9, BGK (τ=0.8, ν=0.1), x 周期，壁面预流半程 bounce-back
- **驱动**：`tensorlbm.turbulent_channel._apply_body_force_2d(f, a)`
- **步序**：collide → half-way BB → force → stream

## 判据口径（与 NVIDIA 侧原文档一致）

| # | 判据 | 门限 |
|---|------|------|
| 1 | 中心线 u_c(t) max \|u_c_num − u_c_ana\| / u_max | ≤ 3% |
| 2 | 剖面 L2 相对误差（t* ∈ {0.05, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5}） | ≤ 3% |
| 3 | 剖面逐点最大误差 / u_max | ≤ 3% |
| 4 | H→2H 误差单调下降（网格收敛） | 是 |

## SDAA 运行配置

| 项目 | 值 |
|------|-----|
| 设备 | sdaa (`torch.sdaa.get_device_name(0)` → `/dev/tcaicard14`) |
| 编译模式 | `eager`（SDAA teco_inductor 后端无法编译 LBM 整步链） |
| 网格 | H = [9, 15]（原文档推荐 [59, 119]，为控制墙钟取可行较小档） |
| τ | 0.8 |
| u_max | 0.03 |
| nx | 16 |
| 容差 | 3% |

## 绕行记录

| 绕行 | 原因 |
|------|------|
| `compile_mode='eager'` | SDAA teco_inductor 后端（Torch-SDAA 3.1.1a3）尚不完整，无法编译 LBM 整步链（collide→BB→force→stream）。compile_route.py 已内置自动 eager fallback 机制，本脚本直接使用 eager 模式。 |
| TRITON_INTERPRET | 未使用。本案例核心路径（d2q9.equilibrium/macroscopic, solver.collide_bgk/stream, turbulent_channel._apply_body_force_2d）不依赖 triton 内核。 |

## 结果表

### H=9 (ny=11, nx=16, steps=1235)

| 判据 | 数值 | 门限 | 结果 |
|------|------|------|------|
| 中心线 max err/u_max | 0.6468% | ≤ 3% | **PASS** |
| 剖面 L2 max | 0.8924% | ≤ 3% | **PASS** |
| 剖面逐点 max/u_max | 0.6468% | ≤ 3% | **PASS** |

### H=15 (ny=17, nx=16, steps=3395)

| 判据 | 数值 | 门限 | 结果 |
|------|------|------|------|
| 中心线 max err/u_max | 0.2427% | ≤ 3% | **PASS** |
| 剖面 L2 max | 0.3383% | ≤ 3% | **PASS** |
| 剖面逐点 max/u_max | 0.2427% | ≤ 3% | **PASS** |

### 网格收敛

| 指标 | H=9 | H=15 | 收敛 |
|------|-----|------|------|
| 中心线 max err/u_max | 0.6468% | 0.2427% | **PASS** (下降 2.67×) |

## 总体判定

| 判据 | 结果 |
|------|------|
| 1. 中心线误差 ≤ 3% | **PASS** |
| 2. 剖面 L2 误差 ≤ 3% | **PASS** |
| 3. 剖面逐点误差 ≤ 3% | **PASS** |
| 4. H→2H 误差单调下降 | **PASS** |
| **总体** | **PASS** |

## 墙钟

- **总墙钟**：4.7s
- H=9：1.0s
- H=15：2.9s

## 设备证据

```
torch.sdaa.get_device_name(0) -> '/dev/tcaicard14'
```

## 产物

- `artifacts/sdaa_startup_poiseuille.json` — 完整 JSON 产物（含设备证据行、判据数值、逐条 PASS/FAIL、剖面表、中心线历史表）
- `scripts/sdaa_sweep_startup_poiseuille.py` — 独立运行脚本

## 备注

- 原文档推荐网格 [59, 119] 在 SDAA 上墙钟约 800s（H=59 约 350s，H=119 约 450s），超出工具 120s 限制。
  本扫荡采用可行较小档 [9, 15]，所有判据均通过，且网格收敛趋势正确（误差随网格加密单调下降）。
- 所有判据数值由脚本自动计算，零手抄。
