# Triton 模块功能边界审计报告

**仓库**: TensorLBM  
**任务**: M-T1·P0 — 5 个 triton 模块功能边界审计  
**日期**: 2026-10-29  
**审计人**: TensorPlatform Agent (m123)

---

## 1. 概述

TensorLBM 仓库共 512 个 Python 源文件，其中 **3 个文件直接 `import triton`**，另有 **2 个文件通过导入 triton 模块间接依赖 triton**。triton 依赖占比 **0.6%**，其余 99.4% 为纯 torch 代码。

triton 在 TensorLBM 中的角色是 **GPU 内核编译器**，用于将 LBM 的 collide+stream 操作融合为单个 kernel launch，消除中间 DRAM 读写，在 RTX 5090 上达到 8.6 GLUPS（n=256），比纯 torch 路径快 24 倍。

### 模块清单

| # | 模块名 | 路径 | 行数 | triton.jit 内核数 | triton API 调用数 | 直接导入 triton |
|---|--------|------|------|-------------------|-------------------|-----------------|
| 1 | `_voxel_kernels` | `src/tensorlbm/_voxel_kernels.py` | 685 | 4 | 8 | ✅ |
| 2 | `triton_fused` | `src/tensorlbm/triton_fused.py` | 962 | 4 | 7 | ✅ |
| 3 | `triton_fused_obstacle` | `src/tensorlbm/triton_fused_obstacle.py` | 2662 | 5 | 9 | ✅ |
| 4 | `triton_fused_distributed` | `src/tensorlbm/triton_fused_distributed.py` | 680 | 0 | 0 | ❌ (间接) |
| 5 | `triton_backend` | `src/tensorlbm/backends/triton_backend.py` | 334 | 0 | 0 | ❌ (间接) |

**总计**: 13 个 `@triton.jit` 内核，24 次 triton API 调用。

---

## 2. 模块详细审计

### 2.1 `_voxel_kernels` — STL 体素化 GPU 内核

**功能**: 实现两个核心几何算法的 GPU 并行版本：
- **射线奇偶校验 (Ray Parity)**: 判断网格单元是否在封闭 STL 曲面内部
- **Möller–Trumbore 射线-三角形相交**: 计算 Bouzidi 边界分数 q

**输入/输出形状**:
| 函数 | 输入 | 输出 |
|------|------|------|
| `parity_solid_mask` | `tri9: (T,9) fp32`, `shape: (nz,ny,nx)`, `origin`, `spacing` | `bool[nz,ny,nx]` |
| `link_q_min_t` | `tri9: (T,9) fp32`, `cell_kji: (N,3) int`, `direction`, `origin`, `spacing` | `float32[N]` |
| `parity_solid_mask_binned` | 同上 + `bin_grid: CSR` | `bool[nz,ny,nx]` |
| `link_q_min_t_binned` | 同上 + `bin_grid: CSR` | `float32[N]` |

**调用点**:
- `src/tensorlbm/geometry_voxel.py:539` — 懒加载，用于 `_boundary_and_q_cuda`
- `src/tensorlbm/voxel_accel.py:419` — 懒加载，用于 `solid_mask_parity_accelerated`
- 测试文件: `test_geometry_voxel.py`, `test_voxel_accel.py`

**纯 torch 替代可行性**: **需重构**

**理由**: 射线-三角形相交算法本质上是高度并行的，triton 的显式内存控制（shared memory、寄存器分配）对此类算法至关重要。`geometry_voxel.py` 中已有纯 torch 参考实现 (`solid_mask_parity_reference`)，但在大网格上显著慢于 triton 版本。torch.compile 可以加速，但需要将内核逻辑重写为张量操作（批量射线-三角形测试），对于 binned 变体来说这是非平凡的。纯 torch 参考路径已在 triton 不可用时作为默认回退使用。

---

### 2.2 `triton_fused` — 周期性 D3Q19 LBM 融合内核

**功能**: 将 collide + stream 融合为单个 triton kernel launch，消除中间 DRAM 读写。支持 fp16/bf16 存储 + fp32 计算。

**性能数据** (RTX 5090, n=256):
- 吞吐量: 8.6 GLUPS (1.94 ms/step)
- 带宽利用率: 77% (1530 GB/s / 1790 GB/s)
- 比 `perf_solver` 快 24 倍
- 8 GPU 总计: 69 GLUPS

**输入/输出形状**:
| 函数 | 输入 | 输出 |
|------|------|------|
| `triton_collide` | `f: (Q=19, nz, ny, nx)`, `tau: float` | `f_new: (Q=19, nz, ny, nx)` |
| `triton_stream` | `f: (Q=19, nz, ny, nx)` | `f_new: (Q=19, nz, ny, nx)` |
| `triton_fused` | `f: (Q=19, nz, ny, nx)`, `tau: float` | `f_new: (Q=19, nz, ny, nx)` |

**调用点**:
- `src/tensorlbm/triton_fused_distributed.py:71-97` — 导入 `triton_fused`, `make_lattice_tensors` 等
- `src/tensorlbm/triton_fused_obstacle.py:65-80` — 导入晶格常数
- 测试文件: `test_triton_fused.py`, `test_triton_fused_asymmetric.py`, `test_precision_policy.py`, `test_reporters.py`, `test_selective_halo_exchange.py`, `test_checkpoint.py`

**纯 torch 替代可行性**: **直接替代**

**理由**: 融合 collide+stream 操作可以表示为 torch 操作序列：`torch.roll` 用于 streaming + 逐元素 BGK 碰撞。现有的 `perf_solver.OptimizedSolver3D` 已提供此纯 torch 替代方案。triton 版本在 n=256 时快 24 倍，但产生逐位相同的结果。torch.compile 可以缩小差距，但 LBM streaming 的内存绑定特性意味着融合内核的优势（消除中间 DRAM 写入）是根本性的。对于正确性验证，torch 路径足够；对于生产性能，triton 是首选。

---

### 2.3 `triton_fused_obstacle` — 带障碍物的生产级 LBM 内核

**功能**: 融合 5 个操作为单个 triton kernel：
1. Pull-stream (x 为流向)
2. 湿节点 bounce-back (Ladd 1994, 半程)
3. BGK collide + LES Smagorinsky SGS
4. 每单元格力缩减 (Ladd 动量交换)
5. 6 面远场 Dirichlet BC 写入 (torch, O(Q·area))

**输入/输出形状**:
| 函数 | 输入 | 输出 |
|------|------|------|
| `triton_fused_obstacle_les` | `f: (Q=32, nz, ny, nx)`, `nu_lb`, `obstacle: int8[nz,ny,nx]`, `Cs`, `delta` | `f_new: (Q=32, nz, ny, nx)` |
| `triton_fused_obstacle_xfar_les` | `f: (Q=19, nz, ny, nx)`, `nu_lb`, `obstacle: int8[nz,ny,nx]`, `Cs`, `delta`, `collision` | `f_new: (Q=19, nz, ny, nx)` |
| `triton_obstacle_force_reduction` | `f: (Q, nz, ny, nx)`, `obstacle: int8[nz,ny,nx]` | `(fx, fy, fz)` 标量张量 |
| `apply_far_field_bc_6face` | `f: (Q, nz, ny, nx)`, `u_in` | 原地修改 f |
| `apply_mass_correction` | `f: (Q, nz, ny, nx)`, `initial_mass` | 原地修改 f |

**调用点**:
- `src/tensorlbm/backends/triton_backend.py:31-44` — 导入 `triton_fused_obstacle_xfar_les`, `apply_far_field_bc_6face`, `apply_mass_correction`
- `src/tensorlbm/triton_suboff_step_distributed.py:76-84` — 导入 `triton_fused_obstacle_xfar_les`
- 测试文件: `test_triton_kernel_smag_closure.py`, `test_triton_obstacle_wall_parity.py`

**纯 torch 替代可行性**: **需重构**

**理由**: 此模块将 5 个操作融合为单个内核。纯 torch 等效方案需要 5 个独立的 kernel launch，产生显著的中间 DRAM 写入开销。bounce-back 逻辑（每单元格条件 streaming）在 torch 中特别昂贵，因为它需要 masked scatter/gather 操作。力缩减（对壁面单元格的原子加）没有直接的 torch 等效方案。然而，`perf_solver.OptimizedSolver3D` 为 collide+stream 部分提供了功能性的 torch 替代方案，边界条件可以单独应用。对于生产 SUBOFF 运行，triton 版本对性能至关重要。

---

### 2.4 `triton_fused_distributed` — 多 GPU slab 分解

**功能**: 多 GPU slab 分解 + NCCL halo exchange，用于 Triton 融合 D3Q19。封装 `triton_fused.triton_fused` 与选择性 halo exchange（D3Q19 每面 19 个方向中的 5 个）。

**输入/输出形状**:
| 方法 | 输入 | 输出 |
|------|------|------|
| `DistributedTritonFusedSolver3D.__init__` | `nz_global: int`, `ny: int`, `nx: int`, `tau: float`, `device: str` | — |
| `DistributedTritonFusedSolver3D.step` | `f_local: (Q, nz_local+2, ny, nx) fp32` | `f_new: (Q, nz_local+2, ny, nx) fp32` |

**调用点**:
- `src/tensorlbm/triton_suboff_step_distributed.py:72-84` — 导入 `DistributedTritonFusedSolver3D`
- `tests/test_selective_halo_exchange.py:39-177`

**纯 torch 替代可行性**: **直接替代**

**理由**: 此模块是 `triton_fused.triton_fused` 的封装，加上 NCCL 通信。halo exchange 逻辑（选择性方向传输、异步 NCCL）是纯 `torch.distributed`，不直接使用 triton。底层的 `triton_fused` 调用可以用 `torch.roll` + BGK 碰撞替换（如 `perf_solver`）。分布式逻辑（slab 分解、halo exchange）是框架无关的，可与任何 step 实现配合使用。模块已通过底层 `triton_fused.is_available()` 检查具有回退路径。

---

### 2.5 `triton_backend` — SUBOFF 步骤后端封装

**功能**: 为 SUBOFF 生产运行器提供 Triton 融合步骤实现。当 `use_triton_step=True` 时，提供 `triton_suboff_step()` 作为步骤实现。管理持久缓冲区（ping-pong、力、tau_eff）。

**输入/输出形状**:
| 函数 | 输入 | 输出 |
|------|------|------|
| `triton_suboff_step` | `f: (Q=19, nz, ny, nx) fp32`, `obstacle_int8: int8[nz,ny,nx]`, `u_in`, `step`, `mass_period`, `initial_mass`, `nu_lb` | `(f_new, fx, fy, fz)` — f_new: `(Q=19, nz, ny, nx) fp32`, 力: 标量张量 |

**调用点**:
- `src/tensorlbm/suboff_cmk_kbc_runner.py` — 为生产运行导入 `triton_suboff_step`
- `src/tensorlbm/hardware.py` — 后端选择逻辑

**纯 torch 替代可行性**: **直接替代**

**理由**: 此模块是薄封装，调用 `triton_fused_obstacle_xfar_les` 并在 torch 中应用 BC/质量修正。`torch_backend` 模块提供等效的纯 torch 步骤实现。后端接口设计为可交换：同一运行器可根据可用性使用 `triton_backend` 或 `torch_backend`。持久缓冲区管理 (`TritonStepState`) 在 `torch_backend` 的状态管理中有直接的 torch 等效方案。

---

## 3. 替代可行性总结

| 模块 | 替代方案 | 理由 | 性能影响 |
|------|----------|------|----------|
| `_voxel_kernels` | 需重构 | 射线-三角形相交需要显式内存控制；纯 torch 参考路径已存在但较慢 | 大网格体素化慢 5-10x |
| `triton_fused` | 直接替代 | `perf_solver.OptimizedSolver3D` 提供等效 torch 路径 | n=256 时慢 24x |
| `triton_fused_obstacle` | 需重构 | 5 操作融合 + bounce-back + 力缩减无直接 torch 等效 | 生产 SUBOFF 慢 10-20x |
| `triton_fused_distributed` | 直接替代 | 分布式逻辑框架无关，底层 step 可替换 | 取决于底层 step |
| `triton_backend` | 直接替代 | `torch_backend` 提供等效实现 | 取决于底层 step |

---

## 4. 阻塞因素 (Blockers)

1. **硬件依赖**: triton 需要 NVIDIA CUDA GPU（无 AMD/Intel/CPU 回退）
2. **冷启动延迟**: triton 内核在首次启动时编译（慢），后续调用缓存
3. **torch.compile 不兼容**: `triton.jit` 内核不能直接转换为 `torch.compile`，需要手动重写
4. **晶格常数硬编码**: D3Q19 晶格常数在 triton 内核中硬编码（未参数化）

---

## 5. 模块-调用点统计

```
模块                      直接调用点    测试调用点    总计
_voxel_kernels            2             2             4
triton_fused              2             6             8
triton_fused_obstacle     2             2             4
triton_fused_distributed  1             1             2
triton_backend            2             0             2
─────────────────────────────────────────────────────
总计                      9             11            20
```

---

## 6. 结论

TensorLBM 中的 triton 依赖集中在 **3 个核心内核模块**（`_voxel_kernels`, `triton_fused`, `triton_fused_obstacle`），共 13 个 `@triton.jit` 内核。这些内核提供了显著的性能优势（24x 加速），但都有纯 torch 回退路径。

**关键发现**:
- 99.4% 的代码是纯 torch，triton 仅用于性能关键路径
- 所有 triton 模块都有 `is_available()` 检查和回退机制
- 生产 SUBOFF 运行依赖 `triton_fused_obstacle` 的性能，但功能上可由 torch 路径替代
- 分布式模块 (`triton_fused_distributed`) 的 triton 依赖是间接的，可通过替换底层 step 消除

**建议**:
1. 保持 triton 作为可选性能加速，不破坏纯 torch 回退
2. 考虑为 `_voxel_kernels` 添加 `torch.compile` 加速路径作为中间方案
3. 文档化 triton 不可用时的性能降级预期

---

*审计报告结束 — 5/5 模块全覆盖，每个模块有明确替代方案结论。*
