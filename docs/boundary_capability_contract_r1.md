# 边界条件能力矩阵与 fail-closed 合同（r1）

`tensorlbm.boundary_capability_contract` 是边界条件的唯一通用前端合同。
它区分"仓库中存在的实现/测试机械流程"和"可对外宣称已完成碰撞 × 物理 × 后端组合验证的合同"。后者在当前版本**一律 fail-closed**：没有任何 (boundary_kind, lattice) 组合返回 `AVAILABLE`。

## 审计范围

本合同审计以下 9 类边界条件 × 3 种 lattice（共 27 个单元）：

| 边界类型 | 说明 |
|---|---|
| `periodic` | 周期性边界（内建于 streaming 的 `torch.roll` / gather） |
| `zou_he_inlet` | Zou/He 速度入口（解析法或 NEBB） |
| `zou_he_outlet` | Zou/He 压力出口 |
| `wall_bounce_back` | 半程反弹壁面（含 D3Q27 link-wise 动壁 ME） |
| `wall_free_slip` | 镜面反射自由滑移壁面 |
| `farfield` | 自由流远场 Dirichlet 边界 |
| `sponge` | 海绵/吸收层出口（黏性 + 目标场两种策略） |
| `nscbc` | 非反射特征边界条件（简化单特征松弛） |
| `bouzidi_interpolated` | Bouzidi-Firdaouss-Lallemand 插值反弹（曲壁） |

碰撞族（collision-agnostic，但验证证据因碰撞而异）：`bgk`, `mrt`, `trt`, `smagorinsky`, `kbc`, `cascaded`

物理：`single_phase`, `turbulence`, `multiphase`, `free_surface`, `ibm`

后端：`torch_cpu`, `torch_cuda`

## 审计结论（源码 + 可执行测试，不从 docstring 外推）

| 边界类型 | lattice | 实现状态 | 验证证据 | 前端状态 |
|---|---|---|---|---|
| `periodic` | D2Q9 | `MECHANICS_TESTED` | `test_lattice.py`, `test_solver.py`：周期 streaming 质量守恒；无周期流物理验证 | `WITHHELD_NO_COMPLETE_COMPOSITION_EVIDENCE` |
| `periodic` | D3Q19 | `MECHANICS_TESTED` | `test_solver3d.py`：同上 | 同上 |
| `periodic` | D3Q27 | `MECHANICS_TESTED` | `test_d3q27.py`：同上 | 同上 |
| `zou_he_inlet` | D2Q9 | `IMPLEMENTATION_ONLY` | 无专用测试；`test_cylinder_cd.py` 用 `apply_simple_channel_boundaries`（平衡入口），非 Zou-He | 同上 |
| `zou_he_inlet` | D3Q19 | `MECHANICS_TESTED` | `test_full_wet.py`（回归）、`test_marine.py`（profile shape/finite）；无 Zou-He 精度物理验证 | 同上 |
| `zou_he_inlet` | D3Q27 | `MECHANICS_TESTED` | `test_d3q27.py`：速度 prescribed 验证、finite output；无物理验证 | 同上 |
| `zou_he_outlet` | D2Q9 | `IMPLEMENTATION_ONLY` | 无专用测试 | 同上 |
| `zou_he_outlet` | D3Q19 | `IMPLEMENTATION_ONLY` | 经 `apply_zou_he_channel_boundaries_3d` / `apply_wave_inlet_3d` 间接使用；无专用出口压力测试 | 同上 |
| `zou_he_outlet` | D3Q27 | `IMPLEMENTATION_ONLY` | 经 `apply_zou_he_channel_boundaries_27` 间接使用；无专用出口压力测试 | 同上 |
| `wall_bounce_back` | D2Q9 | `PHYSICS_VALIDATED` | `test_cylinder_cd.py`：Cd 验证（2.0 < Cd < 8.0 @Re=100），容差极宽，仅 BGK | 同上 |
| `wall_bounce_back` | D3Q19 | `PHYSICS_VALIDATED` | `test_sphere_cd.py`：Cd 验证（err < 120–150%），容差极宽，仅 BGK | 同上 |
| `wall_bounce_back` | D3Q27 | `MECHANICS_TESTED` | `test_d3q27.py`：shape；`test_d3q27_moving_wall_momentum_exchange.py`：ME 力单元测试；无物理验证 | 同上 |
| `wall_free_slip` | D2Q9 | `NO_IMPLEMENTATION` | — | `WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE` |
| `wall_free_slip` | D3Q19 | `IMPLEMENTATION_ONLY` | 无测试；docstring 引用 waLBerla FreeSlip 模式 | `WITHHELD_NO_COMPLETE_COMPOSITION_EVIDENCE` |
| `wall_free_slip` | D3Q27 | `NO_IMPLEMENTATION` | — | `WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE` |
| `farfield` | D2Q9 | `IMPLEMENTATION_ONLY` | 无测试；docstring 声称 ~9% Cd 误差，但不是测试，**不采信为验证证据** | `WITHHELD_NO_COMPLETE_COMPOSITION_EVIDENCE` |
| `farfield` | D3Q19 | `IMPLEMENTATION_ONLY` | 同上（docstring 声称 channel ~65% → far-field ~9%） | 同上 |
| `farfield` | D3Q27 | `NO_IMPLEMENTATION` | — | `WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE` |
| `sponge` | D2Q9 | `MECHANICS_TESTED` | `test_gap_improvements.py`：profile shape、damping、shape conservation；无波吸收物理验证 | `WITHHELD_NO_COMPLETE_COMPOSITION_EVIDENCE` |
| `sponge` | D3Q19 | `MECHANICS_TESTED` | 同上（3D target sponge no-damping） | 同上 |
| `sponge` | D3Q27 | `MECHANICS_TESTED` | 同上（lattice-agnostic） | 同上 |
| `nscbc` | D2Q9 | `IMPLEMENTATION_ONLY` | 无测试；简化单特征松弛，非完整 NSCBC | `WITHHELD_NO_COMPLETE_COMPOSITION_EVIDENCE` |
| `nscbc` | D3Q19 | `IMPLEMENTATION_ONLY` | 同上 | 同上 |
| `nscbc` | D3Q27 | `NO_IMPLEMENTATION` | — | `WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE` |
| `bouzidi_interpolated` | D2Q9 | `MECHANICS_TESTED` | `test_interpolated_bc.py`：shape、finite、halfway-q=standard BB、linear/quad branch；无物理验证 | `WITHHELD_NO_COMPLETE_COMPOSITION_EVIDENCE` |
| `bouzidi_interpolated` | D3Q19 | `MECHANICS_TESTED` | 同上 + `compute_q_sphere`；`sphere_bouzidi.py` benchmark 报告 ~13% Cd 误差但**不是测试**；无可执行物理验证 | 同上 |
| `bouzidi_interpolated` | D3Q27 | `NO_IMPLEMENTATION` | — | `WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE` |

## 实现状态等级

| 等级 | 含义 |
|---|---|
| `NO_IMPLEMENTATION` | 仓库中未发现该 (kind, lattice) 的实现代码 |
| `IMPLEMENTATION_ONLY` | 代码存在但无任何测试 |
| `MECHANICS_TESTED` | 有 shape/finite/unit 测试，但无物理验证 |
| `PHYSICS_VALIDATED` | 存在可执行物理验证测试（如 Cd 误差），即使容差很宽 |

**docstring 中的物理验证声明（如 farfield 的 ~9% Cd 误差）不采信为 `PHYSICS_VALIDATED` 证据**；只有可执行测试证据才被采纳。

## 组合约束规则

1. **无实现**：`implementation_status == NO_IMPLEMENTATION` → `WITHHELD_NO_IMPLEMENTATION_FOR_LATTICE`，先于物理检查。
2. **非单相物理**：`physics != single_phase` → `WITHHELD_NO_COUPLED_BC_PHYSICS_CONTRACT`。这些物理能力可在包的其他模块存在，但没有证据表明它们与边界条件形成已审计的端到端合同。
3. **有实现但无完整组合证据**：即使 `PHYSICS_VALIDATED`，也无碰撞 × 物理 × 后端的完整组合验证合同 → `WITHHELD_NO_COMPLETE_COMPOSITION_EVIDENCE`。

边界条件在 streaming 之后（或 streaming 内，对 periodic）施加，因此在实现层面是碰撞无关的（collision-agnostic）。但验证证据是碰撞特定的：例如 `wall_bounce_back` 的 `PHYSICS_VALIDATED` 仅来自 BGK 碰撞的 Cd 测试。

## 使用

```python
from tensorlbm import require_boundary_condition_capability

# 当前会抛 BoundaryConditionWithheldError，而不是许可过度宣称。
require_boundary_condition_capability(
    "wall_bounce_back", "D3Q19", "bgk", "single_phase", "torch_cpu",
)
```

```python
from tensorlbm import boundary_capability_matrix

matrix = boundary_capability_matrix()
cap = matrix["bouzidi_interpolated"]["D3Q19"]
print(cap.implementation_status)  # MECHANICS_TESTED
print(cap.status)                  # WITHHELD_NO_COMPLETE_COMPOSITION_EVIDENCE
print(cap.verification_evidence)  # test_interpolated_bc.py: ...
```

## 未知公共输入

`require_boundary_condition_capability` 会在访问能力矩阵前验证 `kind`、`lattice`、`collision`、`physics` 和 `backend`。未知值不会泄漏 `KeyError`，而是稳定抛出 `BoundaryConditionWithheldError`，并在错误文本中分别携带机器可读的 `WITHHELD_UNKNOWN_BOUNDARY`、`WITHHELD_UNKNOWN_LATTICE`、`WITHHELD_UNKNOWN_COLLISION`、`WITHHELD_UNKNOWN_PHYSICS` 或 `WITHHELD_UNKNOWN_BACKEND`。该验证先于实现/物理检查，因此不完整的调用方也不能掩盖未知请求。

该合同是声明/门禁层；它没有改动数值边界条件算法，也没有把既有 identity、shape 或 docstring 声明解释成物理准确性证据。
