# SDAA Sweep: floating_body_free_decay

## 案例概述

| 项目 | 内容 |
|---|---|
| **案例名** | B6DOF-1: 浮体静水自由衰减 — 固有频率解析闭包验证 |
| **来源** | `benchmarks/verified/floating_body_free_decay/` |
| **验证对象** | `tensorlbm.rao_analysis.compute_natural_frequencies` + `tensorlbm.rigid_body_6dof.cummins_time_integration` |
| **物理内容** | 无阻尼单自由度 Cummins 方程退化为简谐振子，解析固有频率 ω_n = √(C/(M+A∞)) |
| **原 CPU 判据** | (1) 每档 \|ω_meas−ω_n\|/ω_n ≤ 3%; (2) 两档跨度 ≤ 3%; (3) 特征值求解器 < 1e-3%; (4) 积分器复现辛-Euler 色散 < 0.5% |
| **自由度** | heave (DOF 2), roll (DOF 3) |
| **档位** | coarse (ω_n·dt=0.5), fine (ω_n·dt=0.25) |

## SDAA 配置

| 项目 | 值 |
|---|---|
| **请求设备** | sdaa |
| **torch 版本** | 2.7.1 |
| **设备名** | /dev/tcaicard0 |
| **设备数** | 1 |
| **sdaa 探针** | ✅ 可分配张量 (sdaa:0) |
| **TRITON_INTERPRET** | 1 (设置但本案例不依赖 triton) |
| **网格/步数** | 按案例文档：16 个周期，coarse/fine 两档 |

## 结果：FAIL

### 总体判定

| 判据 | 结果 | 说明 |
|---|---|---|
| 设备证据 | ✅ PASS | torch.sdaa.get_device_name(0) = "/dev/tcaicard0" |
| 案例运行 | ❌ FAIL | 两种绕行策略均失败（见下） |
| 判据 1: 每档 ≤3% | — | 未能产出数值 |
| 判据 2: 跨度 ≤3% | — | 未能产出数值 |
| 判据 3: 特征值精确 | — | 未能产出数值 |
| 判据 4: 积分器色散 | — | 未能产出数值 |

### 失败原因：业务代码设备管理缺陷

尝试了两种绕行策略，均因 `tensorlbm/rigid_body_6dof.py` 中的设备管理缺陷而失败：

#### Bug 1: `cummins_step()` 硬编码 CPU 张量

```python
# rigid_body_6dof.py, cummins_step()
F_memory = torch.zeros(6)  # ← 始终在 CPU 上创建
```

当 `cummins_time_integration(device='sdaa')` 将 M, A_inf, C, state 等放在 sdaa 上时，
`F_memory` 仍在 CPU 上，导致：
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, sdaa:0 and cpu!
```

**修复建议**: `F_memory = torch.zeros(6, device=M.device)`

#### Bug 2: `compute_retardation_function()` 使用 numpy

```python
# rigid_body_6dof.py, compute_retardation_function()
omega_fine = torch.linspace(...)  # ← 继承默认设备
B_fine = torch.zeros(...)         # ← 继承默认设备
...
omega_fine.numpy()  # ← 若默认设备为 sdaa 则失败
```

若设 `torch.set_default_device('sdaa')`，则 `torch.linspace`/`torch.zeros` 创建的张量在 sdaa 上，
`.numpy()` 调用失败：
```
TypeError: can't convert sdaa:0 device type tensor to numpy.
```

**修复建议**: 显式指定 `device='cpu'`，或改用纯 torch 操作避免 `.numpy()`。

#### 两种策略的互斥性

| 策略 | 默认设备 | Bug 1 | Bug 2 | 结果 |
|---|---|---|---|---|
| A: 保持 CPU 默认 | CPU | ✅ 触发（F_memory 在 CPU，其余在 sdaa） | ❌ 不触发 | FAIL |
| B: 设 sdaa 默认 | sdaa | ❌ 不触发 | ✅ 触发（.numpy() 失败） | FAIL |

两个 bug 互斥——修一个就触发另一个。**必须同时修复两处才能支持 sdaa 执行。**

### 绕行记录

| 尝试 | 方法 | 错误 |
|---|---|---|
| 1 | 默认设备=CPU，传 device='sdaa' 给 cummins_time_integration | `Expected all tensors to be on the same device, sdaa:0 and cpu!` |
| 2 | `torch.set_default_device('sdaa')` | `can't convert sdaa:0 device type tensor to numpy` |

### Triton 绕行

本案例入口 (`run.py`) 不依赖 triton。`TRITON_INTERPRET=1` 已设置但无实际效果。

## 产物清单

| 文件 | 说明 |
|---|---|
| `scripts/sdaa_sweep_floating_body_free_decay.py` | 独立运行脚本（不改业务代码） |
| `artifacts/sdaa_floating_body_free_decay.json` | 机器输出（含设备证据行、bug 描述、尝试记录） |
| `docs/sdaa_sweep/floating_body_free_decay.md` | 本文档 |

## 墙钟

| 阶段 | 时间 |
|---|---|
| 总执行时间 | ~0.5 s（两种尝试各 ~0.2 s，均在首次调用即失败） |
| 限制 | ≤20 min ✅ |

## 结论

**SDAA 执行: FAIL** — 业务代码 `tensorlbm/rigid_body_6dof.py` 存在两处设备管理缺陷，
导致无法在 sdaa 上运行 Cummins 时间积分。需修复 Bug 1（`cummins_step` 中 `F_memory` 的设备）
和 Bug 2（`compute_retardation_function` 中 numpy 依赖）后才能支持 sdaa。

**CPU 基准状态**: 该案例在 CPU 上已 VERIFIED（见 `benchmarks/verified/floating_body_free_decay/result.json`），
所有 4 项判据通过。本 SDAA sweep 的失败不影响 CPU 验证结论。
