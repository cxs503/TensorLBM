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
| D3Q19 | `847e4b6d385ae9147e1a3b2e02a7de8f19fe1ff1c1ac66a8a900ac901d7f2b13` |
| D3Q27 | `4b1b55bf7b2aae49857f22d261e75666765764f5eeeb37050f105a17bafc10b5` |

执行命令：

```bash
pytest -q tests/test_d3q19_d3q27_mrt_consistency.py
```

## dtype 边界（重点）

审计输入明确限定为 `float32`，这不是 `float64` 精度声明。两个当前 MRT 实现的 cached moment matrices 均由 float32 tensor 构造；尤其 D3Q27 的 `_get_d3q27_mrt_matrices`（`d3q27.py`）无论 population dtype 都返回 `torch.float32` matrix 与 inverse。因此以 `float64` populations 调用 `collide_mrt27` 会在矩阵乘法处以 dtype mismatch `RuntimeError` 失败，而不是执行 float64 MRT。该限制由专门的 D3Q27 回归测试锁定。

这份 artifact 证明的是当前 float32 CPU 路径下的基础数值一致性；不外推到 float64、GPU/其他加速器、streaming/boundary/forcing 耦合、长时间稳定性或物理解精度。
