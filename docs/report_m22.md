# Mission 22 补报：SDAA Sweep — floating_body_free_decay

> **Mission ID**: 22 (backfill)  
> **Repo**: TensorLBM  
> **Scenario Key**: report-backfill  
> **日期**: 2026-10-29  
> **结论**: **honest-FAIL** — 业务代码设备管理缺陷导致 SDAA 执行无法完成

---

## 1. 案例介绍

### 1.1 物理问题

浮体静水自由衰减（Floating Body Free Decay）是船舶耐波性验证的经典基准。本案例验证 `tensorlbm.rao_analysis.compute_natural_frequencies` 与 `tensorlbm.rigid_body_6dof.cummins_time_integration` 两个核心模块。

**物理模型**：无阻尼单自由度 Cummins 方程退化为简谐振子：

$$M \ddot{\xi} + C \xi = 0 \quad \Rightarrow \quad \omega_n = \sqrt{\frac{C}{M + A_\infty}}$$

其中 $M$ 为质量矩阵，$A_\infty$ 为无穷频附加质量，$C$ 为恢复力系数。

### 1.2 验证对象

| 项目 | 内容 |
|---|---|
| **案例路径** | `benchmarks/verified/floating_body_free_decay/` |
| **验证模块** | `tensorlbm.rao_analysis` + `tensorlbm.rigid_body_6dof` |
| **自由度** | heave (DOF 2), roll (DOF 3) |
| **档位** | coarse ($\omega_n \cdot dt = 0.5$), fine ($\omega_n \cdot dt = 0.25$) |
| **周期数** | 16 个振荡周期 |

### 1.3 原 CPU 判据（NVIDIA 侧已有成熟判据）

| 编号 | 判据 | 阈值 |
|---|---|---|
| 1 | 每档 $\|\omega_{meas} - \omega_n\| / \omega_n$ | ≤ 3% |
| 2 | 两档跨度 | ≤ 3% |
| 3 | 特征值求解器精度 | < 1e-3% |
| 4 | 积分器复现辛-Euler 色散 | < 0.5% |

> **CPU 基准状态**：该案例在 CPU 上已 VERIFIED，所有 4 项判据通过。

---

## 2. 方法：SDAA 迁移策略

### 2.1 设备环境

| 项目 | 值 |
|---|---|
| **请求设备** | sdaa |
| **torch 版本** | 2.7.1 |
| **设备名** | /dev/tcaicard0 |
| **设备数** | 1 |
| **sdaa 探针** | ✅ 可分配张量 (sdaa:0) |
| **TRITON_INTERPRET** | 1（已设置但本案例不依赖 triton） |

### 2.2 独立运行脚本

编写 `scripts/sdaa_sweep_floating_body_free_decay.py`，遵循"不改业务代码"原则：
- 计算张量放 `sdaa` 设备
- 入口不依赖 triton，`TRITON_INTERPRET=1` 无实际效果
- 尝试两种绕行策略以隔离 bug

### 2.3 绕行策略

| 策略 | 默认设备 | 思路 |
|---|---|---|
| A | CPU | 保持默认设备为 CPU，传 `device='sdaa'` 给 `cummins_time_integration` |
| B | sdaa | 设 `torch.set_default_device('sdaa')` 使所有 `torch.zeros()` 自动在 sdaa 上创建 |

---

## 3. 关键结果

### 3.1 总体判定

| 判据 | 结果 | 说明 |
|---|---|---|
| 设备证据 | ✅ PASS | `torch.sdaa.get_device_name(0)` = `/dev/tcaicard0` |
| 案例运行 | ❌ **FAIL** | 两种绕行策略均失败 |
| 判据 1: 每档 ≤3% | — | 未能产出数值 |
| 判据 2: 跨度 ≤3% | — | 未能产出数值 |
| 判据 3: 特征值精确 | — | 未能产出数值 |
| 判据 4: 积分器色散 | — | 未能产出数值 |

### 3.2 失败根因：业务代码设备管理缺陷

在 `tensorlbm/rigid_body_6dof.py` 中发现两处互斥的设备管理 bug：

#### Bug 1: `cummins_step()` 硬编码 CPU 张量

```python
# rigid_body_6dof.py, cummins_step()
F_memory = torch.zeros(6)  # ← 始终在 CPU 上创建
```

当 `cummins_time_integration(device='sdaa')` 将 M, A_inf, C, state 等放在 sdaa 上时，`F_memory` 仍在 CPU 上，触发：

```
RuntimeError: Expected all tensors to be on the same device, 
but found at least two devices, sdaa:0 and cpu!
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

若设 `torch.set_default_device('sdaa')`，`.numpy()` 调用失败：

```
TypeError: can't convert sdaa:0 device type tensor to numpy.
```

**修复建议**: 显式指定 `device='cpu'`，或改用纯 torch 操作避免 `.numpy()`。

#### 互斥性分析

| 策略 | 默认设备 | Bug 1 | Bug 2 | 结果 |
|---|---|---|---|---|
| A: 保持 CPU 默认 | CPU | ✅ 触发 | ❌ 不触发 | FAIL |
| B: 设 sdaa 默认 | sdaa | ❌ 不触发 | ✅ 触发 | FAIL |

**两个 bug 互斥——修一个就触发另一个。必须同时修复两处才能支持 sdaa 执行。**

### 3.3 墙钟时间

| 阶段 | 时间 |
|---|---|
| 总执行时间 | ~0.5 s（两种尝试各 ~0.2 s，均在首次调用即失败） |
| 限制 | ≤20 min ✅ |

---

## 4. 关联工件概览

本 mission 同期还产出了 SUBOFF 系列 SDAA benchmark 结果（sdaa:20/21/22），汇总如下：

| Benchmark | 碰撞模型 | Re | 网格 | Cf_num | Cf_ref | 误差% | 状态 |
|---|---|---|---|---|---|---|---|
| B1: 4L+Cumulant | Cumulant D3Q19 | 1,000 | 320×120×120 | 0.04179 | 0.04200 (Blasius) | **0.5%** | ✅ |
| B2: RANS k-ε | RANS k-epsilon (MRT) | 100,000 | 200×80×80 | 0.00662 | 0.00833 (ITTC-57) | **20.5%** | ⚠️ |
| B3: RANS k-ε | RANS k-epsilon (MRT) | 10,000 | 200×80×80 | 0.01065 | 0.01875 (ITTC-57) | **43.2%** | ⚠️ |

**关键发现**：
- Cumulant D3Q19 在 4L 域上 Re=1000 达到 0.5% 误差，为 SUBOFF 最佳结果（比 MRT+Smag 2.3% 好 4.6 倍）
- RANS k-ε 在 Re=1e5 误差 20.5%，略优于 Smagorinsky (23.9%)，但逊于 Vreman (16.4%)
- RANS k-ε 在 Re=1e4 误差 43.2%，模型在过渡 Re 下过度生成涡粘性
- RANS k-ε 在两个高 Re 下均显著低估摩擦阻力——需要壁面函数改善近壁行为
- 所有 3 个模拟稳定（无发散），5000 步完成

---

## 5. 注意点与局限

### 5.1 证据边界

本 SDAA sweep 的 floating_body_free_decay 案例使用 **调用方合成的浮体和水动力系数**，不涉及 LBM 流场、船体耦合、自由面或辐射记忆效应的验证。因此：

- 设备证据行（sdaa 探针通过）仅证明硬件可达
- 案例运行失败归因于业务代码 bug，非算法问题
- CPU 基准验证结论不受影响

### 5.2 未完成的 SDAA 验证

由于业务代码 bug，以下物理判据在 SDAA 上 **均未验证**：
- 固有频率精度（≤3%）
- 档位一致性（≤3%）
- 特征值求解器精度（<1e-3%）
- 辛积分器色散（<0.5%）

### 5.3 修复路径

需同时修复 `tensorlbm/rigid_body_6dof.py` 中两处：
1. `cummins_step()`: `F_memory = torch.zeros(6, device=M.device)`
2. `compute_retardation_function()`: 避免 `.numpy()` 或显式 `device='cpu'`

修复后需重新运行 SDAA sweep 以完成设备迁移验证。

### 5.4 RANS 模型局限

SUBOFF RANS k-ε 结果（B2/B3）误差 >15%，不满足精度标准。主要原因：
- 缺少壁面函数，近壁湍流建模不足
- 过渡 Re 下涡粘性过度生成
- 5000 步可能不足以达到统计稳态

---

## 6. 误差判定

| 关键量 | 误差 | 是否 <3% | 说明 |
|---|---|---|---|
| floating_body ω_n (SDAA) | N/A | **无法判定** | 业务代码 bug 导致无法运行 |
| SUBOFF B1 Cf (Cumulant, Re=1e3) | 0.5% | ✅ 是 | 最佳结果 |
| SUBOFF B2 Cf (RANS, Re=1e5) | 20.5% | ❌ 否 | 缺壁面函数 |
| SUBOFF B3 Cf (RANS, Re=1e4) | 43.2% | ❌ 否 | 过渡 Re 模型失效 |

**floating_body_free_decay 在 SDAA 上的关键量误差无法给出数值**——原因是业务代码设备管理 bug 阻止了任何数值产出，非算法精度问题。CPU 基准已通过全部 4 项判据。

---

## 7. 结论

**SDAA 执行: honest-FAIL**

`tensorlbm/rigid_body_6dof.py` 存在两处互斥的设备管理缺陷（`cummins_step` 硬编码 CPU 张量 + `compute_retardation_function` 依赖 numpy），导致 floating_body_free_decay 案例无法在 sdaa 上运行。需同时修复两处后重新验证。

同期 SUBOFF 系列 benchmark 在 sdaa 上成功运行：Cumulant D3Q19 在 Re=1000 达到 0.5% 误差（优秀），RANS k-ε 在高 Re 下误差 >20%（需改进壁面处理）。

---

*报告生成: mission 22 backfill, 2026-10-29*
