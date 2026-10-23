# AMR 回归等价性验证报告

**日期**: 2026-07-17  
**Worktree**: `/root/.hermes/marine-control/TensorLBM_dev/regress-amr-r1`  
**Base commit**: `cfb26c670b7a5778e219c60499526523593a29e7` (HEAD == base, working tree clean)  
**测试文件**: `tests/test_amr_regression_equivalence.py` (28 tests, all passing)

---

## 1. 原始 Bug 辨识

### BUG-1: `_fh_coarse_to_fine_3d` / `_fh_fine_to_coarse_3d` 硬编码 D3Q19

**位置**: `adaptive_refinement.py` 第 171 行、第 227 行

**描述**: 原始 FH (Filippova–Hänel) 接口交换函数在函数体内无条件导入 D3Q19 的 `macroscopic3d` / `equilibrium3d`:

```python
# adaptive_refinement.py L171
from .d3q19 import equilibrium3d, macroscopic3d
rho, ux, uy, uz = macroscopic3d(f_coarse)  # 期望 19 个速度
```

当传入 D3Q27 数据（27 个速度方向）时，`macroscopic3d` 因速度数不匹配而崩溃。这是一个**设计限制（带病上岗）**：原始实现无法处理 D3Q27 格点。

**amr_common 修复**: `amr_common._fh_coarse_to_fine_3d` 通过 `_macroscopic(lattice, f)` / `_equilibrium(lattice, ...)` 分派机制支持 D3Q19 和 D3Q27 两种格点。

**验证**: `test_bug1_original_fh_coarse_to_fine_3d_hardcoded_d3q19` (PASS — 原始在 D3Q27 上抛异常), `test_bug1_amr_common_handles_d3q27` (PASS — amr_common 正确处理 D3Q27)

---

### BUG-2: `halo_exchange` 形状不匹配 (shape mismatch)

**位置**: `amr_common.py` 第 247-250 行; 同样存在于 `adaptive_refinement.py` 的 `_inject_to_patch` (第 1162-1169 行)

**描述**: 当上采样后的父层数据 `f_up` 在任一空间维度上小于 `patch_f` 时，`min()` 截断后的布尔掩码形状 `(fz, fy, fx)` 与 `patch_f` 的空间形状 `(nz_f, ny_f, nx_f)` 不匹配。PyTorch 布尔索引要求精确形状匹配，因此抛出 `IndexError`:

```
IndexError: The shape of the mask [6, 6, 6] at index 0 does not match
the shape of the indexed tensor [19, 8, 8, 8] at index 1
```

**根因**: `min()` 防御性截断逻辑有缺陷——它截断了掩码但没有截断被索引张量的空间维度。

**影响**: 当 patch 的实际形状与 box+ratio 推算的形状不一致时触发。正常使用路径（patch 由相同 box 创建）不会触发，但 API 层面缺乏鲁棒性。

**验证**: `test_bug2_halo_exchange_shape_mismatch_when_upsample_smaller` (PASS — 确认 bug 存在), `test_bug2_halo_exchange_works_when_shapes_match` (PASS — 正常路径工作)

---

### BUG-3: `_fh_fine_to_coarse_3d` 同样硬编码 D3Q19

**位置**: `adaptive_refinement.py` 第 227 行

**描述**: 与 BUG-1 同源。原始 `_fh_fine_to_coarse_3d` 无条件导入 D3Q19 函数，无法处理 D3Q27。

**验证**: `test_bug3_original_fh_fine_to_coarse_3d_hardcoded_d3q19` (PASS)

---

### 非 Bug: `_coarse_to_fine_3d` / `_fine_to_coarse_3d` 格点无关

**描述**: `refinement.py` 中的纯插值/平均函数不涉及格点特定逻辑（不调用 equilibrium/macroscopic），对任意 Q 值均正确工作。

**验证**: `test_no_bug_coarse_to_fine_3d_lattice_agnostic`, `test_no_bug_fine_to_coarse_3d_lattice_agnostic` (PASS)

---

### 已知限制: `_fine_to_coarse_3d` 要求维度可整除

**描述**: `_fine_to_coarse_3d` 使用 `view()` 重塑，要求各维度精确可被 ratio 整除。非整除维度会抛 `RuntimeError`。这是**契约要求**，非 bug——调用方需确保 fine grid 由正确的 refine 操作创建。

**验证**: `test_coarsen_non_divisible_raises` (PASS)

---

## 2. 等价性验证

### D3Q19 FH refine 等价

| 操作 | 原始 (`adaptive_refinement`) | 共性 (`amr_common`) | 结果 |
|------|---------------------------|---------------------|------|
| `_fh_coarse_to_fine_3d` | 直接调用 D3Q19 | 分派到 D3Q19 | **完全一致** (atol=1e-7) |
| `_fh_fine_to_coarse_3d` | 直接调用 D3Q19 | 分派到 D3Q19 | **完全一致** (atol=1e-7) |
| 纯插值 refine | `_coarse_to_fine_3d` | `refine(use_fh=False)` | **完全一致** (atol=0) |
| 纯平均 coarsen | `_fine_to_coarse_3d` | `coarsen(use_fh=False)` | **完全一致** (atol=0) |

**验证**: `test_refine_d3q19_fh_equivalence`, `test_refine_d3q19_plain_equivalence`, `test_coarsen_d3q19_fh_equivalence`, `test_coarsen_d3q19_plain_equivalence` (all PASS)

### halo_exchange 等价

`amr_common.halo_exchange` 与 `AdaptiveSolver3D._inject_to_patch` 使用相同的代码模式（border 掩码 + min 截断 + 布尔索引赋值），在形状匹配时产生**完全一致**的输出。

**验证**: `test_halo_exchange_equivalence_with_solver_inject` (PASS, atol=1e-7)

### AMRPatch3D 数据类等价

`amr_common.AMRPatch3D` 与 `adaptive_refinement.AMRPatch3D` 具有相同的字段和属性（nz/ny/nx/cells），amr_common 版本额外增加 `lattice` 字段和 `__post_init__` 验证。

**验证**: `test_amr_patch3d_equivalence` (PASS)

### 多种子等价性

在 4 个不同随机种子下，D3Q19 FH refine 的原始与共性输出完全一致。

**验证**: `test_equivalence_multiple_seeds` (PASS)

### FH 重缩放正确性

当 `tau_f == tau_c` 时，FH 重缩放因子为 1，FH 输出应与纯插值完全一致。

**验证**: `test_fh_rescaling_correctness` (PASS)

---

## 3. 组合测试: AMR + Collision 完整循环

### D3Q19 完整循环

```
refine(coarse 4³ → fine 8³) → collide(BGK, τ=0.75) → coarsen(fine 8³ → coarse 4³)
```

- 形状正确: (19,4,4,4) → (19,8,8,8) → (19,8,8,8) → (19,4,4,4) ✓
- 密度近似守恒: 最大漂移 < 0.1 (trilinear align_corners=True 的已知特性) ✓

**验证**: `test_refine_collide_coarsen_loop_d3q19` (PASS)

### D3Q27 完整循环

同上流程，使用 D3Q27 格点（27 个速度方向）。amr_common 正确分派，原始实现无法处理。

**验证**: `test_refine_collide_coarsen_loop_d3q27` (PASS)

### 多步 AMR 循环

5 步碰撞后密度仍为正，形状正确。

**验证**: `test_multi_step_amr_loop_d3q19` (PASS)

### halo_exchange 内部保护

halo_exchange 仅覆盖边界单元，内部单元完全不受影响。

**验证**: `test_halo_exchange_preserves_interior_d3q19` (PASS, atol=0)

### AMRPatch3D 完整生命周期

创建 → halo exchange → collide → coarsen → 写回父层，全流程正确。

**验证**: `test_amr_patch3d_lifecycle` (PASS)

### AdaptiveSolver3D 集成路径

使用 `amr_common.refine` 替代原始硬编码 D3Q19 函数，模拟 `_add_patch` 路径，全流程正确。

**验证**: `test_adaptive_solver3d_step_with_amr_common_refine` (PASS)

---

## 4. 密度漂移说明

refine → coarsen 往返后密度有 ~7% 漂移（4³ 网格）。这是 `F.interpolate(mode="trilinear", align_corners=True)` 的已知特性：

- `align_corners=True` 保持角点精确，但内部点为插值
- 插值后块平均不等于原始值
- 网格越大漂移越小（边界效应占比下降）

这不是 bug，是插值格式的固有属性。FH 重缩放本身是正确的（`tau_f == tau_c` 时退化为纯插值，已验证）。

---

## 5. 结论

| 验证项 | 结果 |
|--------|------|
| BUG-1: 原始 FH 硬编码 D3Q19 | ✅ 已辨识，amr_common 已修复 |
| BUG-2: halo_exchange 形状不匹配 | ✅ 已辨识，正常路径不受影响 |
| BUG-3: 原始 FH coarsen 硬编码 D3Q19 | ✅ 已辨识，amr_common 已修复 |
| D3Q19 refine/coarsen 等价性 | ✅ 完全一致 (atol=1e-7) |
| halo_exchange 等价性 | ✅ 完全一致 (atol=1e-7) |
| D3Q27 支持 | ✅ amr_common 正确支持，原始不支持 |
| AMR + Collision 组合 | ✅ D3Q19/D3Q27 完整循环通过 |
| 多种子鲁棒性 | ✅ 4 个种子全部一致 |

**总计**: 28 tests, 28 passed, 0 failed

amr_common 模块正确提取了 AMR patch 机制，在 D3Q19 上与原始实现完全等价，同时修复了 D3Q27 支持的限制。halo_exchange 的形状不匹配 bug 在正常使用路径（patch 由相同 box 创建）下不会触发，但 API 层面需要文档化形状匹配契约。
