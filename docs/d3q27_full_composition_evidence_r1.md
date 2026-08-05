# D3Q27 MRT 完整 composition evidence（R1）

## 目的

本 artifact 证明 D3Q27 MRT 在 **composition 级**已有可执行一致性证据，覆盖 wall / geometry / boundary / force 四个维度。它将 `bounce_back_cells_27`、`equilibrium27`、`collide_mrt27`、`stream27`、`macroscopic27`、`compute_obstacle_forces_27` 六个组件组合为完整管线，验证它们协同工作时的内部一致性。

本 artifact **不改变** `general_capability_matrix` 对 D3Q27 MRT 完整候选的 `WITHHELD` 判定（tier `no_composition_evidence`，reason `WITHHELD_D3Q27_COMPOSITION`）。它证明的是 composition 的**可执行性和内部一致性**，而非**物理精度**。物理验证仍被 WITHHELD。

## 与组件级证据的关系

| 维度 | 组件级证据（R1） | 完整 composition 证据（本 artifact） |
|---|---|---|
| 覆盖范围 | `collide_mrt27` 单独 | `bounce_back_cells_27` + `collide_mrt27` + `stream27` + `macroscopic27` + `compute_obstacle_forces_27` |
| 检查数 | 6 | 15 |
| 证据层级 | `component_contract` | `composition_contract` |
| 物理验证 | WITHHELD | WITHHELD |
| 实现文件 | `d3q27_composition_evidence.py` | `d3q27_full_composition_test.py` |
| 测试文件 | `test_d3q27_composition_evidence.py` | `test_d3q27_full_composition.py` |

组件级证据证明碰撞核自身的一致性（不动点、质量/动量守恒、有限性、确定性、源码绑定）。完整 composition 证据进一步证明多个组件组合后仍保持一致。

## 探针实现

实现：`src/tensorlbm/d3q27_full_composition_test.py`
测试：`tests/test_d3q27_full_composition.py`

探针使用固定 CPU `float32` 输入、固定随机种子 `31415`、域大小 `(nz, ny, nx) = (8, 10, 12)` 及 `tau=0.8`。探针状态包括：

- **均匀平衡分布** `feq_uniform`：用于 collide+stream 不动点测试（均匀场下 streaming 为恒等映射）
- **扰动分布** `f_perturbed`：在随机低 Mach 平衡分布上叠加零质量、零动量微扰，用于守恒性测试
- **障碍物掩码** `obstacle_mask`：域中心的 2×2×2 块，用于力提取测试
- **壁面掩码** `wall_mask`：通道壁面（±y 和 ±z 面），用于 bounce-back 测试

## 十五项 composition 检查

### 1. Bounce-back 壁面 composition（`bounce_back_cells_27`）

| # | 检查 | 通过条件 | 描述 |
|---|---|---|---|
| 1 | bounce_back_involution | `max abs(bb(bb(f)) - f) <= tol` | bounce-back 两次施加恢复原始分布 |
| 2 | bounce_back_mass_conservation | `abs(sum(bb(f)) - sum(f)) <= tol` | bounce-back 保守总质量 |
| 3 | bounce_back_momentum_reflection | `max abs(p_after + p_before) <= tol` | 固体格上流体动量被反转 |

### 2. 平衡+碰撞+流动一步 composition（`equilibrium27` + `collide_mrt27` + `stream27`）

| # | 检查 | 通过条件 | 描述 |
|---|---|---|---|
| 4 | full_step_shape | `stream27(collide_mrt27(f)).shape == f.shape` | 完整一步保持形状 |
| 5 | full_step_mass_periodic | `abs(sum(step(f)) - sum(f)) <= tol` | 周期边界下 collide+stream 保守质量 |
| 6 | full_step_equilibrium_fixed_point | `max abs(step(feq_uniform) - feq_uniform) <= tol` | 均匀平衡是 collide+stream 的不动点 |
| 7 | full_step_finite | `isfinite(step(f)).all() == True` | 一步后所有值有限 |

### 3. 宏观量恢复 composition（`macroscopic27`）

| # | 检查 | 通过条件 | 描述 |
|---|---|---|---|
| 8 | macroscopic_roundtrip | `max abs(macroscopic(equilibrium(rho,u)) - (rho,u)) <= tol` | 平衡→宏观量往返恢复 |
| 9 | macroscopic_finite_after_step | `isfinite(macroscopic(step(f))).all() == True` | 一步后宏观量有限 |
| 10 | macroscopic_mass_after_step | `abs(sum(rho_after) - sum(rho_before)) <= tol` | 一步后总质量守恒 |

### 4. 壁面链路力提取 composition（`compute_obstacle_forces_27`）

| # | 检查 | 通过条件 | 描述 |
|---|---|---|---|
| 11 | force_empty_zero | `max abs(force(f, empty_mask)) <= tol` | 空障碍物掩码给出零力 |
| 12 | force_finite | `isfinite(force(f, mask)).all() == True` | 非空障碍物给出有限力 |
| 13 | force_momentum_balance | `max abs(Δp + F) <= tol` | bounce-back 流体动量变化 == −F_solid（动量交换恒等式） |
| 14 | force_drag_sign | `fx > 0` | +x 方向流动 → 障碍物阻力为正 |

### 5. 确定性

| # | 检查 | 通过条件 | 描述 |
|---|---|---|---|
| 15 | determinism | `torch.equal(step(f1), step(f2)) == True` | 相同 clone 输入的完整一步 bitwise 一致 |

所有十五项通过后，`composition_evidence_tier = "composition_contract"`。

## 动量交换恒等式（检查 13）

检查 13 是本 artifact 的核心 composition 性质。它验证 `compute_obstacle_forces_27` 与 `bounce_back_cells_27` 的相互一致性：

1. 流动后、bounce-back 前，在固体格上计算力：`F = 2 Σ c·f_solid`
2. 对固体格施加 bounce-back：`f[i] → f[OPPOSITE[i]]`
3. 计算流体动量变化：`Δp = Σ c·(f_after - f_before)` at solid cells
4. 验证 `Δp = -F`

数学推导：bounce-back 将 `f[i]` 替换为 `f[OPPOSITE[i]]`，利用 `c[OPPOSITE[i]] = -c[i]` 和 OPPOSITE 的对合性，可得 `Δp = -2 Σ c·f_solid = -F`。这精确到浮点精度。

## WITHHELD 物理验证

以下六个方面在 artifact 中被显式标记为 WITHHELD，表示它们**不在**本 composition 证据的覆盖范围内：

| 方面 | WITHHELD 理由 |
|---|---|
| wall_treatment | bounce-back + MRT composition 可执行且内部一致，但无物理壁面精度验证（如 Poiseuille 通道比较） |
| geometry | obstacle_mask + MRT composition 可执行，但无物理几何精度验证（如球阻力 vs. 参考 Cd） |
| boundary | Zou/He inlet/outlet + MRT composition 可执行，但无物理边界精度验证（如 inlet/outlet 质量通量守恒） |
| streaming_collision_coupling | stream + collide 一步保守质量，但无多步长时间稳定性或收敛性测试 |
| force_observation | 力提取满足动量交换恒等式，但无物理力精度验证（如阻力系数 vs. 实验） |
| physical_accuracy | 无端到端物理精度测试（如 lid-driven cavity vs. Ghia，或球阻力 vs. Schlichting） |

## 机器可读 artifact

探针函数 `run_d3q27_full_composition_probe()` 返回一个 JSON-ready dict，包含：

- `artifact_id`: `"d3q27-full-composition-evidence-r1"`
- `version`: `"d3q27-full-composition-evidence-r1"`
- `lattice` / `collision` / `entrypoint`
- `probe_config`: shape, dtype, device, seed, tau
- `checks`: 十五项检查的 status + 测量值 + 容差
- `composition_evidence_tier`: `"composition_contract"`（全部 PASS 时）
- `withheld_physical_validation`: 六个 WITHHELD 方面及理由
- `capability_matrix_cross_reference`: general_capability_matrix 判定 + 组件级 + composition 级证据层级
- `artifact_sha256`: artifact 自身的 canonical JSON SHA-256

执行命令：

```bash
pytest -q tests/test_d3q27_full_composition.py
```

## 容差说明

- **元素级检查**（involution、fixed point、roundtrip）：绝对容差 `atol=1e-5`
- **全局求和检查**（mass conservation、momentum balance）：相对容差 `rtol=1e-5`，即 `delta <= 1e-5 * max(1, |reference|)`。这是因为 float32 在 ~1000 格的全局求和中累积 ~1e-4 的绝对误差，但相对误差仍为 ~1e-7。

## dtype 边界

与组件级证据一致，探针输入限定为 `float32`。D3Q27 的 cached moment matrices 返回 `torch.float32` matrix 与 inverse，以 `float64` populations 调用 `collide_mrt27` 会在矩阵乘法处以 dtype mismatch `RuntimeError` 失败。

本 artifact 证明的是当前 float32 CPU 路径下的 composition 数值一致性；不外推到 float64、GPU/其他加速器、长时间稳定性或物理解精度。
