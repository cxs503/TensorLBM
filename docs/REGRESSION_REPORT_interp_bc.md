# 回归验证报告：插值边界 + 滑移网格

**日期**: 2026-07-17  
**基线提交**: `cfb26c670b7a5778e219c60499526523593a29e7`  
**工作树**: `regress-interpolated-bc-r1`, `regress-sliding-mesh-r1`  
**状态**: 两个工作树均 clean（仅新增未跟踪测试文件，无源码修改，无 commit/push）

---

## 1. 辨识原始 Bug（带病上岗）

### 1.1 插值边界 `interpolated_bc.py`

#### Bug-A: BFL 二次插值公式使用错误的前步分布（严重）

**位置**: `bouzidi_bounce_back_3d` 第 262 行（3D），`bouzidi_bounce_back` 第 73 行（2D）

**现象**: 二次插值分支（q ≥ 0.5）使用 `fp_opp = f_prev[opp]`（前步**反方向**分布），而标准 BFL 公式应使用 `fp_d = f_prev[direction]`（前步**同方向**分布）。

**标准 BFL 公式**:
```
线性 (q < 0.5):  f_bc = 2q * f_opp + (1 - 2q) * f_prev[direction]     ✓ 代码正确
二次 (q ≥ 0.5):  f_bc = f_opp/(2q) + (2q-1)/(2q) * f_prev[direction]   ✗ 代码用 f_prev[opp]
```

**代码实际实现**:
```python
f_bc_quad = f_opp / (2*safe_q) + (2*safe_q - 1) / (2*safe_q) * fp_opp  # fp_opp = f_prev[opp]
```

**影响**: 当 `f_prev[direction] ≠ f_prev[opp]` 时（即流体非平衡态），二次插值结果偏离标准 BFL。实测：q=0.75 时代码结果 0.0532 vs 标准 BFL 0.0476，偏差约 12%。

**传播**: 该 bug 同时存在于原始 `interpolated_bc.py` 和共性模块 `interpolated_bc_common.py`——共性模块**忠实复制了原始 bug**。

**测试**: `TestBouzidiBounceBack3DBugs::test_quadratic_uses_fp_opp_not_fp_d`  
`TestBouzidiBounceBack3DBugs::test_quadratic_bug_reproduced_in_common`

#### Bug-B: 2D 版本死代码（轻微）

**位置**: `bouzidi_bounce_back` 第 65 行

**现象**: `f[direction][fluid_nodes]` 是一个无赋值的裸索引表达式，计算后丢弃。不影响输出，但暗示早期公式演变的残留。

**测试**: `TestBouzidiBounceBack3DBugs::test_2d_dead_code_line65`

#### Bug-C: 3D 线性分支 docstring 与代码不一致（文档）

**位置**: `bouzidi_bounce_back_3d` docstring 第 229 行

**现象**: docstring 称线性分支使用 "opposite-direction population from the previous step"，但代码实际使用 `fp_d = f_prev[direction]`（同方向）。2D 版本的 docstring 描述正确（"upstream neighbour"）。

### 1.2 滑移网格 `sliding_mesh.py`

#### Bug-D: Runner 调用 `collide_bgk` 参数不匹配（严重，运行时崩溃）

**位置**: `run_sliding_mesh_rotor` 第 366 行

**现象**: 代码调用 `collide_bgk(f, rho, ux, uy, tau)`，但实际签名为 `collide_bgk(f, tau)`。运行时抛出 `TypeError: collide_bgk() takes 2 positional arguments but 5 were given`。

**影响**: `run_sliding_mesh_rotor` benchmark runner 无法运行。核心 BC 函数（`apply_sliding_mesh_bc_2d` 等）不受影响。

**测试**: `TestSlidingMeshBugs::test_runner_collide_bgk_wrong_signature`

#### Bug-E: 2D BC 硬编码旋转中心（限制）

**位置**: `apply_sliding_mesh_bc_2d` 第 228-229 行

**现象**: 旋转中心硬编码为 `(nx/2, ny/2)`，不接受 `cx, cy` 参数。3D 共性模块 `apply_sliding_mesh_bc_3d` 正确接受 `cx, cy, cz`。

**测试**: `TestSlidingMeshBugs::test_apply_bc_2d_hardcoded_center`

#### Bug-F: 3D 插值函数硬编码中心（限制）

**位置**: `interpolate_interface_3d` 第 176 行

**现象**: `cx = cy = cz = 0.5`（归一化），不接受中心参数。与 `apply_sliding_mesh_bc_3d`（接受 cx/cy/cz）不一致。

**测试**: `TestSlidingMeshBugs::test_interpolate_3d_hardcoded_center`

---

## 2. 等价性验证

### 2.1 插值边界：原始 D3Q19 vs 共性模块 D3Q19

**结论**: ✅ **位级等价**（`torch.equal`）

| 测试维度 | 测试数 | 结果 |
|---------|--------|------|
| 参数化（6方向 × 8 q值） | 48 | 全部 PASS |
| 全 19 方向 × 5 q 值扫描 | 1 | PASS |
| 非流体节点不变 | 1 | PASS |
| q=0.5 标准反弹back | 1 | PASS |

**唯一差异**: 原始使用 `int(OPPOSITE3D[direction].item())`（GPU→CPU 同步），共性模块使用预计算列表 `_OPP19_LIST[direction]`（无同步）。结果值相同。

### 2.2 插值边界：D3Q27 物理合理性（共性模块新增）

**结论**: ✅ **物理合理**

| 验证项 | 结果 |
|--------|------|
| q=0.5 全 27 方向 = 标准反弹back | PASS |
| 全 27 方向 × 5 q 值输出有限 | PASS |
| 非流体节点不变 | PASS |
| `compute_q_sphere_27` 形状/范围 | PASS |
| D3Q27 边界节点数 ≥ D3Q19 | PASS |
| 共享方向 q 场一致（≥15/19） | PASS |

### 2.3 滑移网格：D2Q9 原始 vs 共性模块 3D

**结论**: ✅ **结构等价**（共性模块为 3D 推广，D2Q9 为单层特例）

| 验证项 | 结果 |
|--------|------|
| `rotate_velocity_field_2d` vs `rotate_velocity_field_3d`(axis=z) | PASS（位级等价） |
| 5 个角度参数化 | PASS |
| `interpolate_interface_2d` vs 3D 单层 | PASS（atol=1e-5） |
| 壁面速度公式 ω×r 一致 | PASS |
| 松弛公式 f-(1/τ)(f-f_eq) 一致 | PASS |
| 3D D3Q19 BC 有限 + 非接口不变 | PASS |
| `sliding_mesh_step` 自动检测 D3Q19 | PASS |

**注**: 共性模块无 D2Q9 实现；D2Q9 等价性通过 3D 单层（axis=z, nz=1）验证。平衡函数不同（D2Q9 vs D3Q19），但壁面速度和松弛公式结构相同。

### 2.4 滑移网格：D3Q27 物理合理性（共性模块新增）

**结论**: ✅ **物理合理**

| 验证项 | 结果 |
|--------|------|
| D3Q27 BC 有限 | PASS |
| `sliding_mesh_step` 自动检测 D3Q27 | PASS |
| D3Q27 壁面速度 = ω×r | PASS |
| 3 轴旋转保模长 | PASS |

---

## 3. 组合测试

### 3.1 插值边界 + 碰撞

| 测试 | 结果 |
|------|------|
| D3Q19: BGK 碰撞 → bouzidi BC（球面 q 场）→ 有限 | PASS |
| D3Q19: 碰撞后原始 vs 共性模块等价 | PASS |
| D3Q27: BGK 碰撞 → bouzidi BC → 有限 | PASS |
| 10 步碰撞+BC 稳定性（无 NaN/质量爆炸） | PASS |

### 3.2 滑移网格 + 碰撞

| 测试 | 结果 |
|------|------|
| 2D: BGK 碰撞 → `apply_sliding_mesh_bc_2d` → 有限 | PASS |
| 3D D3Q19: BGK 碰撞 → `sliding_mesh_step` → 有限 | PASS |
| 3D D3Q27: BGK 碰撞 → `sliding_mesh_step` → 有限 | PASS |
| 10 步碰撞+滑移网格稳定性 | PASS |
| 2D: 旋转速度场 → 接口插值 → 有限 | PASS |

---

## 4. 测试汇总

| 工作树 | 测试文件 | 测试数 | 通过 | 失败 |
|--------|---------|--------|------|------|
| regress-interpolated-bc-r1 | `tests/test_interp_bc_regression.py` | 65 | 65 | 0 |
| regress-sliding-mesh-r1 | `tests/test_sliding_mesh_regression.py` | 23 | 23 | 0 |
| regress-interpolated-bc-r1 | `tests/test_interpolated_bc.py`（已有） | 16 | 16 | 0 |
| **合计** | | **104** | **104** | **0** |

---

## 5. 结论

1. **辨识 bug**: 发现 6 个问题（2 严重 + 2 限制 + 2 文档/轻微）。最严重的是 BFL 二次插值公式使用错误的前步分布（Bug-A），该 bug 在共性模块中被忠实复制。

2. **等价性验证**: 原始 D3Q19 与共性模块 D3Q19 **位级等价**。D3Q27 新增实现**物理合理**。滑移网格 D2Q9 与 3D 共性模块**结构等价**。

3. **组合验证**: 插值边界+碰撞、滑移网格+碰撞均通过有限性和稳定性测试。

4. **建议修复**:
   - Bug-A: 将 `fp_opp` 改为 `fp_d`（二次插值分支）
   - Bug-D: 将 `collide_bgk(f, rho, ux, uy, tau)` 改为 `collide_bgk(f, tau)`
   - Bug-E/F: 为 2D BC 和 3D 插值函数添加中心参数
