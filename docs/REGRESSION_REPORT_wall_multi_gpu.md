# 回归验证报告：壁面函数 + 多卡域分解等价性验证

**日期**: 2026-07-17  
**Worktree**: `/root/.hermes/marine-control/TensorLBM_dev/regress-wall-multi-r1`  
**Base commit**: `cfb26c670b7a5778e219c60499526523593a29e7`  
**HEAD**: `cfb26c670b7a5778e219c60499526523593a29e7` (== base, clean)  
**测试文件**: `tests/test_regress_wall_multi_gpu.py` (24 tests, all PASSED)

---

## 1. 原始Bug辨识（带病上岗检测）

### 1.1 wall_shear.py — **CONFIRMED BUG: ImportError**

**文件**: `src/tensorlbm/wall_shear.py`, line 186  
**问题**: `wss_from_fneq_3d` 函数尝试从 `d3q19` 导入 `equilibrium`，但实际函数名为 `equilibrium3d`。

```python
# wall_shear.py line 186 (BUG)
from .d3q19 import equilibrium as eq3d  # ← 'equilibrium' 不存在于 d3q19
```

```python
# d3q19.py 实际导出
# ['C', 'OPPOSITE', 'W', 'equilibrium3d', 'macroscopic3d']  ← 是 equilibrium3d
```

**影响**: `wss_from_fneq_3d` 函数完全不可用，调用时抛出 `ImportError`。  
**验证**: `test_wss_from_fneq_3d_import_error_equilibrium_vs_equilibrium3d` ✓  
**注意**: 2D 版本 `wss_from_fneq_2d` 不受影响（d2q9 确实导出 `equilibrium`）。

### 1.2 wall_shear.py — **CODE SMELL: _W3D.to(device) 结果被丢弃**

**文件**: `src/tensorlbm/wall_shear.py`, line 191-192  
**问题**: `_W3D.to(device)` 的返回值未赋值，权重张量实际未移动到设备。

```python
from .d3q19 import W as _W3D  # noqa: PLC0415,F401
_W3D.to(device)  # ← 结果被丢弃，应为 _W3D = _W3D.to(device)
```

**影响**: 无功能性影响（`_W3D` 在后续计算中未使用），但属于死代码。  
**验证**: `test_wss_from_fneq_3d_W_to_device_discarded` ✓

### 1.3 wall_model.py — **INCONSISTENCY: y_val 不一致**

**文件**: `src/tensorlbm/wall_model.py`  
**问题**: `compute_wall_slip_velocity` 硬编码 `y_val = 1.5`（line 154），而 `wall_function_3d` 默认 `y_val = 0.5`（line 217）。

```python
# compute_wall_slip_velocity (line 154)
y_val = 1.5  # 硬编码

# wall_function_3d (line 217)
y_val: float = 0.5  # 默认参数
```

**影响**: 两个函数使用不同的壁面距离，可能导致不一致的壁面处理。但它们是不同功能的函数（slip velocity vs body force），设计上可能有意为之。  
**验证**: `test_wall_model_y_val_inconsistency` ✓

### 1.4 multi_gpu.py — **已修复: .contiguous() 非连续切片bug**

**文件**: `src/tensorlbm/multi_gpu.py`, `halo_exchange_3d` function, lines 178-179  
**状态**: 已修复。当前代码包含 `.contiguous()` 调用。

```python
# 当前代码（已修复）
left_ghost.copy_(left[:, :, :, -2 * ov:-ov].contiguous().to(left_ghost.device))
right_ghost.copy_(right[:, :, :, ov:2 * ov].contiguous().to(right_ghost.device))
```

**CPU等价性验证**: 在CPU上，`.contiguous()` 不改变结果（`copy_` 能正确处理非连续切片）。  
**验证**: `test_contiguous_vs_no_contiguous_cpu` ✓, `test_contiguous_vs_no_contiguous_with_streaming` ✓

### 1.5 roughness.py — **无Bug**

所有函数功能正常：
- `roughness_b_correction`: 三区制（光滑/过渡/全粗糙）正确
- `compute_rough_wall_slip_velocity`: Newton迭代+粗糙度修正正常
- `apply_rough_wall_damping_2d`: 2D阻尼正常
- `apply_rough_wall_bounce_back`: bounce-back正常

### 1.6 wall_model.py — **无功能性Bug**

- `compute_wall_distance_fmm`: FMM距离场计算正确
- `compute_wall_slip_velocity`: 壁面滑移速度正确（层流时sr=0，全无滑移）
- `wall_function_3d`: 体力壁面函数正确
- `apply_wall_model_bounce_back`: bounce-back正常

---

## 2. 等价性验证

### 2.1 wall_function_common vs wall_model.wall_function_3d

**结论**: **完全等价** (max diff = 0.0)

`wall_function_common.wall_function` 接受预计算的 `u_tau` 和 `y_plus`，而 `wall_model.wall_function_3d` 内联计算这些量。当使用相同的 `u_tau`/`y_plus` 输入时，两者产生完全相同的分布函数。

关键等价点：
- `_apply_body_force` (common) ≡ `ibm_apply_body_force_3d` (ibm) — 相同的Guo体力公式
- `compute_u_tau` (common) ≡ 内联Newton迭代 (wall_model) — 相同的log-law/Reichardt求解
- `_near_wall_mask` (common) ≡ 内联near掩码 (wall_model) — 相同的6连通检测

**验证**: 
- `test_wall_function_common_equals_wall_model_3d` ✓ (log-law)
- `test_wall_function_common_reichardt_equals_wall_model_reichardt` ✓ (Reichardt)

### 2.2 CPU单卡 vs CPU多卡域分解

**结论**: **完全等价** (max diff = 0.0)

| 配置 | max diff | allclose |
|------|----------|----------|
| 1卡 vs 2卡 (1步) | 0.0 | ✓ |
| 1卡 vs 4卡 (1步) | 0.0 | ✓ |
| 1卡 vs 2卡 (5步) | 0.0 | ✓ |

**验证**: 
- `test_single_vs_multi_2_cards` ✓
- `test_single_vs_multi_4_cards` ✓
- `test_multi_step_single_vs_multi` ✓

### 2.3 MultiGPUSolver3D vs MultiDeviceSolver3D

**结论**: **完全等价** (max diff = 0.0)

两个求解器使用相同的域分解策略和halo交换逻辑，在CPU上产生相同结果。

**验证**: `test_multi_gpu_solver_vs_multi_device_solver` ✓

### 2.4 halo_exchange_3d: .contiguous() 修复前后

**结论**: **CPU上完全等价** (max diff = 0.0)

`.contiguous()` 调用在CPU上不改变数值结果，但修复了GPU上非连续切片可能导致的问题。

**验证**:
- `test_contiguous_vs_no_contiguous_cpu` ✓
- `test_contiguous_vs_no_contiguous_with_streaming` ✓

---

## 3. 组合测试

### 3.1 wall_function + collision + multi_gpu

**结论**: **完全等价** (max diff = 0.0)

单卡参考：collide → wall_function_3d → stream  
多卡：分解 → 逐slab collide+wall_function → halo交换 → stream → gather

关键点：solid mask 需要按slab分解（含ghost层，使用modulo索引）以匹配slab形状。

**验证**: `test_wall_function_plus_collision_plus_multi_gpu` ✓

### 3.2 wall_function_common + collision + multi_gpu

**结论**: **完全等价** (max diff = 0.0)

使用 `wall_function_common.wall_function` 替代 `wall_model.wall_function_3d`，结果与单卡参考一致。

**验证**: `test_wall_function_common_plus_collision_plus_multi_gpu` ✓

### 3.3 multi_gpu + boundary_fn

**结论**: **完全等价** (max diff = 0.0)

`MultiGPUSolver3D.step` 的 `boundary_fn` 参数在周期域上与单卡等价。

**验证**: `test_multi_gpu_with_boundary_fn` ✓

### 3.4 多步组合: wall_function + collision + multi_gpu (3步)

**结论**: **完全等价** (max diff = 0.0)

3步迭代：collide → wall_function → halo → stream，多卡与单卡完全一致。

**验证**: `test_multi_step_multi_gpu_with_wall_function` ✓

---

## 4. 测试汇总

```
24 passed, 1 warning in 5.38s
```

| 类别 | 测试数 | 通过 | 失败 |
|------|--------|------|------|
| Bug辨识 | 5 | 5 | 0 |
| 等价性-壁面函数 | 2 | 2 | 0 |
| 等价性-多卡 | 4 | 4 | 0 |
| 等价性-halo交换 | 2 | 2 | 0 |
| 组合测试 | 4 | 4 | 0 |
| 粗糙度健全性 | 4 | 4 | 0 |
| 壁面剪切Bug详查 | 3 | 3 | 0 |
| **总计** | **24** | **24** | **0** |

---

## 5. 结论

### 5.1 带病上岗检测

| 模块 | 状态 | 严重程度 |
|------|------|----------|
| wall_model.py | 功能正常（y_val不一致为设计wart） | 低 |
| wall_shear.py | **wss_from_fneq_3d 不可用**（ImportError） | **高** |
| roughness.py | 功能正常 | 无 |
| wall_function_common.py | 功能正常，与wall_model等价 | 无 |
| multi_gpu.py | .contiguous()已修复，CPU等价 | 已修复 |

### 5.2 等价性结论

- **wall_function_common ≡ wall_model.wall_function_3d**: ✓ 完全等价
- **CPU单卡 ≡ CPU多卡**: ✓ 完全等价（2卡/4卡，单步/多步）
- **halo_exchange .contiguous() 修复前后**: ✓ CPU上完全等价
- **MultiGPUSolver3D ≡ MultiDeviceSolver3D**: ✓ 完全等价

### 5.3 组合结论

- **wall_function + collision + multi_gpu**: ✓ 完全等价
- **wall_function_common + collision + multi_gpu**: ✓ 完全等价
- **多步迭代组合**: ✓ 完全等价

### 5.4 需修复的Bug

1. **wall_shear.py line 186**: `from .d3q19 import equilibrium as eq3d` → 应改为 `from .d3q19 import equilibrium3d as eq3d`
2. **wall_shear.py line 192**: `_W3D.to(device)` → 应改为 `_W3D = _W3D.to(device)` 或删除死代码

### 5.5 Git状态

- HEAD == base commit `cfb26c6` (无commit/push)
- 仅新增未跟踪文件: `tests/test_regress_wall_multi_gpu.py`
- 工作树状态: clean (除新增测试文件外)
