# 湍流模型能力矩阵与 fail-closed 合同（r1）

`tensorlbm.turbulence_capability_contract` 是湍流模型的唯一通用审计合同。
它区分"仓库中存在的碰撞算子实现"和"可对外宣称已完成物理验证的湍流模型组合"。后者在当前版本**一律 fail-closed**：没有任何组合返回 `AVAILABLE`。

## 审计范围（源码直读，不从 docstring 外推）

| 文件 | 内容 |
|---|---|
| `turbulence.py` | LES 闭合：Smagorinsky（6 组合）、动态 Smagorinsky（2 组合）、WALE（3 组合）、Vreman（3 组合） |
| `core/turbulence.py` | 仅 identity 元数据（NONE/SMAGORINSKY/WALE），无数值实现 |
| `rans_ke.py` | RANS k-ε、Spalart-Allmaras、k-ω SST |
| `ddes.py` | DDES/SAS 混合模型（仅 2D） |
| `wall_model.py` | 壁函数（log-law/Reichardt）、壁面距离 FMM |
| `suboff_resistance.py` | RANS-KE 调用方（使用标量平均 workaround） |
| `turbulent_channel.py` | Smagorinsky BGK D2Q9 通道流 benchmark |

## LES 能力矩阵

### Smagorinsky（6 组合，全部 CONTRACT_TESTED）

| Lattice | Collision | 入口 | 测试证据 | 热路径 |
|---|---|---|---|---|
| D2Q9 | BGK | `collide_smagorinsky_bgk` | test_marine.py: shape, finite, mass, momentum, identity | — |
| D2Q9 | MRT | `collide_smagorinsky_mrt` | test_phase4.py: shape, mass, momentum, finite | — |
| D3Q19 | BGK | `collide_smagorinsky_bgk3d` | test_marine.py: shape, finite, mass, momentum, identity | — |
| D3Q19 | MRT | `collide_smagorinsky_mrt3d` | test_marine.py: shape, finite, mass, momentum, identity | — |
| D3Q27 | BGK | `collide_smagorinsky_bgk27` | test_d3q27.py: shape, finite, identity | — |
| D3Q27 | MRT | `collide_smagorinsky_mrt27` | test_d3q27.py: mass conservation | — |

### 动态 Smagorinsky（2 组合，CONTRACT_TESTED）

| Lattice | Collision | 入口 | 测试证据 | 热路径 |
|---|---|---|---|---|
| D2Q9 | BGK | `collide_dynamic_smagorinsky_bgk` | test_dynamic_smagorinsky.py: shape, finite | `float(...item())` GPU→CPU sync (turbulence.py:1096) |
| D3Q19 | BGK | `collide_dynamic_smagorinsky_bgk3d` | test_dynamic_smagorinsky.py: shape | `float(...item())` GPU→CPU sync (turbulence.py:1160) |

MRT 和 D3Q27：**无实现**。

### WALE（3 组合，仅 BGK，CONTRACT_TESTED）

| Lattice | Collision | 入口 | 测试证据 |
|---|---|---|---|
| D2Q9 | BGK | `collide_wale_bgk` | test_turbulence_extensions.py: shape, finite, mass, momentum, identity |
| D3Q19 | BGK | `collide_wale_bgk3d` | test_turbulence_extensions.py: shape, finite, mass, momentum, identity |
| D3Q27 | BGK | `collide_wale_bgk27` | test_turbulence_extensions.py: shape, finite, mass, identity |

MRT：**无实现**。WALE 无 MRT 变体。

### Vreman（3 组合，仅 BGK，CONTRACT_TESTED）

| Lattice | Collision | 入口 | 测试证据 |
|---|---|---|---|
| D2Q9 | BGK | `collide_vreman_bgk` | test_turbulence_extensions.py: shape, finite, mass, momentum, identity |
| D3Q19 | BGK | `collide_vreman_bgk3d` | test_turbulence_extensions.py: shape, finite, mass, momentum, identity |
| D3Q27 | BGK | `collide_vreman_bgk27` | test_turbulence_extensions.py: shape, finite, mass, identity |

MRT：**无实现**。Vreman 无 MRT 变体。

## RANS / 混合模型状态

| 模型 | Lattice | Collision | 实现状态 | 验证等级 | 入口 | 热路径 |
|---|---|---|---|---|---|---|
| k-ε (KESolver) | D3Q19 | MRT | IMPLEMENTED | IMPLEMENTED_ONLY | `collide_rans_ke + KESolver` | `mask.bool()` 每调用分配 (rans_ke.py:394)；suboff_resistance.py:617 使用 `nu_t.mean().item()` 标量平均 |
| Spalart-Allmaras | D3Q19 | MRT | IMPLEMENTED | IMPLEMENTED_ONLY | `collide_rans_sa + SASolver` | `nu_t.mean().item()` 标量平均 (rans_ke.py:821)，丢失逐单元信息；委托 `collide_smagorinsky_mrt3d(C_s=0.0)` |
| k-ω SST | D2Q9 | BGK | IMPLEMENTED | IMPLEMENTED_ONLY | `komega_sst_collision_d2q9 + KOmegaSSTSolver` | — |
| DDES | D2Q9 | BGK | IMPLEMENTED | IMPLEMENTED_ONLY | `apply_ddes_collision` | — |

**关键发现**：
- k-ε 的 `collide_rans_ke` 实现了逐单元 MRT 应力松弛率覆盖，但 `suboff_resistance.py` 实际调用路径使用 `nu_t.mean().item()` 标量平均，丢失了空间变化。
- SA 的 `collide_rans_sa` 同样使用标量平均并委托给 `collide_smagorinsky_mrt3d(C_s=0.0)`，不是真正的逐单元 SA 碰撞。
- DDES 所有辅助函数（`_strain_rate_magnitude`、`_gradient_magnitude`、`_laplacian`、`ddes_eddy_viscosity`、`sas_source_term`）均为 2D（ux, uy），无 3D 实现。无调用方，无测试。
- 以上四个模型**均无单元测试**。

## 壁模型状态

| 模型 | Lattice | Collision | 实现状态 | 验证等级 | 热路径 |
|---|---|---|---|---|---|
| 壁函数 (wall_function_3d) | D3Q19 | BGK | IMPLEMENTED | BENCHMARK_ONLY | `bool(turb.any())` sync (wall_model.py:276)；`.sum().item()` drag sync (wall_model.py:295, 299) |
| 壁面距离 FMM (2D) | D2Q9 | N/A | IMPLEMENTED | IMPLEMENTED_ONLY | — |
| 壁面距离 FMM (3D) | D3Q19 | N/A | IMPLEMENTED | IMPLEMENTED_ONLY | — |

壁函数 docstring 声称 "Validated: SUBOFF AFF-8 Re=2M, Ct_total 0.0040 vs experimental 0.004 (<1% error)"，但**无单元测试断言此结果**。按审计原则，docstring 断言不作为物理验证证据。

## 验证等级定义

| 等级 | 含义 | fail-closed 状态 |
|---|---|---|
| `CONTRACT_TESTED` | 存在 shape/finite/mass/momentum/identity 单元测试 | `WITHHELD_NO_PHYSICS_VALIDATION` |
| `BENCHMARK_ONLY` | 仅在 examples/benchmark 中使用，无单元测试 | `WITHHELD_NO_PHYSICS_VALIDATION` |
| `IMPLEMENTED_ONLY` | 实现存在，无测试 | `WITHHELD_NO_CONTRACT_TESTS` |
| `NO_IMPLEMENTATION` | 无实现 | `WITHHELD_NO_IMPLEMENTATION` |

**合同测试 ≠ 物理验证**：shape/mass/momentum/identity 测试验证算子代数性质（守恒、良构性），不证明湍流物理正确性、谱精度或壁面流动验证。

## 热路径分配审计

`turbulence_hot_path_audit()` 返回以下观察（冷路径审计，不修改数值模型）：

| 函数 | 文件:行 | 模式 | 严重性 | 说明 |
|---|---|---|---|---|
| `collide_dynamic_smagorinsky_bgk` | turbulence.py:1096 | `float(...item())` | SYNC | 全局 Cs 归约到 Python float；每步 GPU→CPU 同步 |
| `collide_dynamic_smagorinsky_bgk3d` | turbulence.py:1160 | `float(...item())` | SYNC | 同上 |
| `collide_rans_ke` | rans_ke.py:394 | `mask.bool()` | ALLOCATION | 每调用分配新 bool 张量；应由调用方预计算 |
| `collide_rans_sa` | rans_ke.py:821 | `nu_t.mean().item()` | SYNC | 标量平均 GPU→CPU 同步；丢失逐单元信息 |
| `wall_function_3d` | wall_model.py:276 | `bool(turb.any())` | SYNC | 分支决策 GPU→CPU 同步 |
| `wall_function_3d` | wall_model.py:295 | `.sum().item()` | SYNC | 摩擦阻力诊断 GPU→CPU 同步 |
| `wall_function_3d` | wall_model.py:299 | `.sum().item()` | SYNC | 压差阻力诊断 GPU→CPU 同步 |

## 使用

```python
from tensorlbm import require_turbulence_capability

# 当前会抛 TurbulenceWithheldError，而不是许可过度宣称。
require_turbulence_capability("smagorinsky", "D2Q9", "BGK")
# TurbulenceWithheldError: WITHHELD_NO_PHYSICS_VALIDATION: ...

from tensorlbm import turbulence_capability_matrix
matrix = turbulence_capability_matrix()
# matrix["wale"]["D2Q9"]["MRT"].implementation_status == "NO_IMPLEMENTATION"
# matrix["rans_ke"]["D3Q19"]["MRT"].verification_level == "IMPLEMENTED_ONLY"
```

该合同是声明/门禁层；它没有改动数值湍流算法，也没有把既有 contract 测试或 docstring 声称解释成物理准确性证据。

## 未知公共输入

`require_turbulence_capability` 会在访问能力矩阵前验证 `family`、`lattice` 和 `collision`。未知值不会泄漏 `KeyError`，而是稳定抛出 `TurbulenceWithheldError`，并在错误文本中分别携带机器可读的 `WITHHELD_UNKNOWN_FAMILY`、`WITHHELD_UNKNOWN_LATTICE` 或 `WITHHELD_UNKNOWN_COLLISION`。
