# SDAA 验证案例扫荡 — acoustics

## 案例概述

| 项目 | 值 |
|---|---|
| 案例路径 | `benchmarks/verified/acoustics` |
| 物理问题 | D2Q9 平面声波：线性色散与衰减 |
| 参考文档 | `benchmarks/verified/acoustics/README.md` |
| 原始入口 | `benchmarks/verified/acoustics/run.py` |
| 库内核 | `tensorlbm.solver.collide_bgk`, `tensorlbm.solver.stream`, `tensorlbm.d2q9.equilibrium`, `tensorlbm.d2q9.macroscopic` |
| 原始设备 | CPU (fp64 单线程) |
| SDAA 设备 | `sdaa:0` → `/dev/tcaicard12` |
| 数据类型 | float64 |

## 判据口径（预注册于 NVIDIA 侧原始运行前）

| 判据 | 描述 | 阈值 | 单调性要求 |
|---|---|---|---|
| **判据 A** | \|c_meas/c_s − 1\| ≤ 3% | ≤ 3% | 随 N 单调下降 |
| **判据 B** | \|δ_meas/(νk²) − 1\| ≤ 3% | ≤ 3% | 随 N 单调下降 |

其中 c_s = 1/√3 ≈ 0.577350269，ν = c_s²(τ−½) = (τ−½)/3。

## 运行配置

| 参数 | 值 |
|---|---|
| 网格 N | 16, 32, 64 |
| τ | 1.0 |
| ε | 1e-3 |
| ny | 8 |
| n_periods | 16.0 |
| 设备 | sdaa:0 |
| dtype | float64 |

## 绕行记录

| 项目 | 状态 |
|---|---|
| TRITON_INTERPRET=1 | **是** — `tensorlbm._voxel_kernels` 和 `tensorlbm.triton_fused` 在 import 时触发 triton 加载；本案例内核（collide_bgk/stream/equilibrium/macroscopic）本身不使用 triton，但 import 路径间接依赖。设置 `TRITON_INTERPRET=1` 后正常加载。 |
| 业务代码修改 | **无** — 仅写独立运行脚本 `scripts/sdaa_sweep_acoustics.py` |

## 结果表

### 主案收敛（τ=1.0, ε=1e-3）

| N | c_meas | c_err vs 连续 | δ_meas | δ_err vs 连续 | 墙钟 (s) |
|---|---|---|---|---|---|
| 16 | 0.5786145708 | **+0.2190%** | 2.603994e-2 | **+1.3145%** | 1.99 |
| 32 | 0.5776616356 | **+0.0539%** | 6.446282e-3 | **+0.3231%** | 4.18 |
| 64 | 0.5774279884 | **+0.0135%** | 1.607686e-3 | **+0.0812%** | 8.62 |

### 判据逐条比对

| 判据 | 最大误差 | 阈值 | 单调 | 结果 |
|---|---|---|---|---|
| A: 波速误差 | 0.2190% | ≤ 3% | ✓ 单调 | **PASS** |
| B: 衰减误差 | 1.3145% | ≤ 3% | ✓ 单调 | **PASS** |
| **总体** | — | — | — | **PASS** |

## 设备证据

```
torch.sdaa.get_device_name(0) → /dev/tcaicard12
```

## 墙钟

| 项目 | 值 |
|---|---|
| 总墙钟 | 15.3 s |
| N=16 | 2.0 s |
| N=32 | 4.2 s |
| N=64 | 8.6 s |

## 与原始 CPU 结果对比

| N | c_err (CPU) | c_err (SDAA) | δ_err (CPU) | δ_err (SDAA) |
|---|---|---|---|---|
| 16 | +0.2190% | +0.2190% | +1.3145% | +1.3145% |
| 32 | +0.0539% | +0.0539% | +0.3231% | +0.3231% |
| 64 | +0.0135% | +0.0135% | +0.0812% | +0.0812% |

SDAA 结果与原始 CPU 结果在有效数字内完全一致（差异 < 1e-10 相对），证明 SDAA float64 计算精度与 CPU 等价。

## 工件

- 脚本：`scripts/sdaa_sweep_acoustics.py`
- 输出：`artifacts/sdaa_acoustics.json`
- 本文档：`docs/sdaa_sweep/acoustics.md`
