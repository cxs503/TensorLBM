# D3Q19 / D3Q27 MRT 基础数值一致性审计（非精度排名）

## 范围与公平性

本证据只对两个可执行 MRT 入口分别施加相同类别的**lattice-local**数值一致性检查：

| lattice | 直接入口 | population shape |
|---|---|---|
| D3Q19 | `tensorlbm.solver3d.collide_mrt3d` | `(19, nz, ny, nx)` |
| D3Q27 | `tensorlbm.d3q27.collide_mrt27` | `(27, nz, ny, nx)` |

共同入口 `tensorlbm.advanced_collision_contract.collide_advanced_3d` 已将两者标记为可用 MRT；本审计进一步直接调用上述实现，避免 dispatch 层掩盖来源。此处不比较 D3Q19 与 D3Q27 的误差大小，也不声称任何一方具有更高物理精度：两者方向集、moment basis 以及可表示的高阶矩不同，跨 lattice 的数值残差横比不是 accuracy benchmark。

## 可重复探针与通过条件

实现：`tests/test_d3q19_d3q27_mrt_consistency.py`。每一项对两种 lattice 使用固定 CPU `float32` 输入、固定随机种子 `2718`、相同域大小 `(nz, ny, nx) = (2, 3, 4)` 及 `tau=0.8`：

1. **equilibrium fixed point**：由局部 `rho, ux, uy, uz` 生成 `feq`；碰撞后输出有限，且 `max abs(out - feq) <= 1e-6`。
2. **perturbed conservation**：在每一格加入小的、零 density/零 raw-momentum 的非平衡扰动；碰撞前后以各自 `macroscopic*` 计算，density 与 `rho*u` 三分量均以 `atol=1e-6, rtol=0` 一致。输出同时必须有限。
3. **repeated determinism**：相同 clone 输入的两次碰撞结果必须 `torch.equal`（bitwise exact），并且有限。
4. **直接源码绑定**：`inspect.getsource()` 的 SHA-256 固定为下表值。碰撞实现内容变更会令审计测试失败，须重跑并重新审计后才可更新指纹。

| lattice | direct callable source SHA-256 |
|---|---|
| D3Q19 | `d1b45c86a5c40fbbdc019962939aae9aedcde7f9849af086ff9a2929a57b4e54` |
| D3Q27 | `3e3d756dbc79847ea7729e74abb0ee8a227285ab6a9e02f5aeab064b8cf41ff0` |

执行命令：

```bash
pytest -q tests/test_d3q19_d3q27_mrt_consistency.py
```

## dtype 边界（重点）

**2026-08-23 复审更新**：`_get_d3q19_mrt_matrices` 与 `_get_d3q27_mrt_matrices` 增加了可选 `dtype` 参数（默认仍为 `float32`，全部既有 float32 调用点逐位不变），两个直接碰撞入口 `collide_mrt3d` / `collide_mrt27` 以及各 MRT-SGS 内核现在按 population dtype 构造 cached matrices——float64 populations 不再在 `matrix @ f` 处 dtype-mismatch 崩溃，而是执行 float64 MRT。动因是 B3 校准路径（`autograd_calib`）在 float64 域上运行 closure-family 轴。复审在同一探针族上新增 `test_mrt_accepts_float64_populations`（两个 lattice 的 fp64 fixed-point 与守恒，容差 1e-10/1e-12），float32 探针全部原样通过；源码指纹按下表更新。

这份 artifact 证明 float32 与 float64 CPU 路径下的基础数值一致性；不外推到 GPU/其他加速器、streaming/boundary/forcing 耦合、长时间稳定性或物理解精度。
