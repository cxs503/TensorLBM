# 局部加密 / AMR 能力矩阵与 fail-closed 合同（r1）

`tensorlbm.amr_capability_contract` 是局部加密的唯一通用前端合同。
它区分“仓库中存在的 patch 机械流程”和“可对外宣称已完成物理耦合、守恒与精度验证的组合”。后者在当前版本**一律 fail-closed**：没有任何当前路径会返回 `AVAILABLE`。

## 审计结论（源码，不从 identity/shape 测试外推）

| 路径 | lattice | 实际机制 | 状态 |
|---|---|---|---|
| `adaptive_dynamic` | D2Q9 | `AdaptiveSolver2D`，动态 patch、比率 2、层级子循环；可选 FH helper，否则双线性/块平均 | `AVAILABLE_MECHANICS_ONLY`；前端 `WITHHELD_REQUIRED_METADATA_NOT_EMITTED` |
| `adaptive_dynamic` | D3Q19 | `AdaptiveSolver3D`，动态 patch、比率 2、层级子循环；FH helper 是此特定路径的实现 | 同上 |
| `multigrid_static` | D3Q19 | `MultiGridSolver` 静态 levels；粗细交换为普通三线性插值/块平均 | `AVAILABLE_MECHANICS_ONLY`；前端 withheld |
| `surface_shell` | D3Q19 | `SurfaceRefinementSolver` 固定三层、物体邻近/尾流及表面壳层；通过现有 plain helper 交换 | `AVAILABLE_MECHANICS_ONLY`；前端 withheld |
| `multipatch_static` | D3Q19 | `MultiPatchSolver` 静态多 patch；普通三线性/块平均（含面/边/角填充） | `AVAILABLE_MECHANICS_ONLY`；前端 withheld |
| 任意局部加密路径 | D3Q27 | 未发现 D3Q27 AMR/local-refinement solver 或交换实现 | `WITHHELD_NO_D3Q27_LOCAL_REFINEMENT` |

“FH”只描述所列 `adaptive_refinement.py` helper 的 f/f_eq/f_neq 重缩放加插值；不是对所有 MultiGrid/surface/multipatch 路径的通用保证，也不是跨碰撞模型或几何耦合的精度认证。

## 物理组合矩阵规则

对每一个现有路径：

- `single_phase`：有上述机械实现时仍为 `WITHHELD_REQUIRED_METADATA_NOT_EMITTED`；它不等于已证明质量、动量、通量或界面误差。
- `turbulence`、`multiphase`、`ibm`、`curved_wall`：均为 `WITHHELD_NO_COUPLED_AMR_PHYSICS_CONTRACT`。这些能力/算子可在包的其他模块存在，但没有证据表明它们与 AMR 的子循环、粗细交换、patch 重叠、几何更新和守恒账本形成已审计的端到端合同。
- D3Q27 先以 `WITHHELD_NO_D3Q27_LOCAL_REFINEMENT` 拒绝，不借用 D3Q19 实现。

## 将来前端必须提供的 metadata

一个将来可执行的组合必须由所选 runtime **实际产生并可追溯**下列全部字段，不能仅由调用者补一个字典升级当前路径：

1. `subcycling`：各 level 时间步和同步时点；
2. `ratio`：空间及时间 refinement ratio；
3. `exchange_scheme`：每个方向、边界层、碰撞/流化时序的交换方案；
4. `geometry_remesh_provenance`：掩膜、曲壁 q、IBM marker 或多相几何在每个 level 的重建来源/版本；
5. `flux_inventory_ledger`：质量、动量以及适用相/组分的跨 patch/界面收支；
6. `refinement_decision_evidence`：指标、阈值、patch 生命周期和决策时间戳。

使用：

```python
from tensorlbm import require_local_refinement_capability

# 当前会抛 LocalRefinementWithheldError，而不是许可过度宣称。
require_local_refinement_capability("adaptive_dynamic", "D3Q19", "multiphase")
```

该合同是声明/门禁层；它没有改动数值 AMR 算法，也没有把既有 identity 或形状测试解释成物理准确性证据。

## 未知公共输入

`require_local_refinement_capability` 会在访问能力矩阵前验证 `path`、`lattice` 和 `physics`。未知值不会泄漏 `KeyError`，而是稳定抛出 `LocalRefinementWithheldError`，并在错误文本中分别携带机器可读的 `WITHHELD_UNKNOWN_PATH`、`WITHHELD_UNKNOWN_LATTICE` 或 `WITHHELD_UNKNOWN_PHYSICS`。该验证先于 metadata 检查，因此不完整的调用方 metadata 也不能掩盖未知请求。
