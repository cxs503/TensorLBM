# D3Q27 MRT 组件级 composition evidence（R1）

## 目的

本 artifact 证明 D3Q27 MRT 碰撞核在**组件级**已有可执行一致性证据，同时明确记录完整 wall/geometry/boundary/output composition 仍被 WITHHELD。

它不改变 `general_capability_matrix` 对 D3Q27 MRT 完整候选的 `WITHHELD` 判定（tier `no_composition_evidence`，reason `WITHHELD_D3Q27_COMPOSITION`），而是为该判定中"component-level executable evidence"这一陈述提供可重复的机器可读证据。

## 与 D3Q19 完整组合证据的差距

| 维度 | D3Q19 MRT | D3Q27 MRT |
|---|---|---|
| 碰撞核入口 | `tensorlbm.solver3d.collide_mrt3d` | `tensorlbm.d3q27.collide_mrt27` |
| advanced_collision_contract | AVAILABLE | AVAILABLE |
| 组件级一致性证据 | ✅ 已有（`test_d3q19_d3q27_mrt_consistency.py`） | ✅ 已有（同上 + 本探针） |
| 完整 composition 判定 | **SUPPORTED**（`EXECUTABLE_CONTRACT`） | **WITHHELD**（`NO_COMPOSITION_EVIDENCE`） |
| 完整 composition 理由 | R1 已验证 D3Q19/MRT/single-phase/static-wall/bounce-back/no-AMR/Torch 组合 | `WITHHELD_D3Q27_COMPOSITION`：有组件级证据，但无已验证完整平台组合 |

D3Q19 的完整组合证据来自 `general_capability_matrix._R1_SUPPORTED`：它声明了一个经过测试的完整执行路径（lattice + collision + turbulence + multiphase + boundary + geometry + wall_treatment + refinement + backend + outputs 全部匹配）。D3Q27 没有对应的 `_R1_SUPPORTED` 条目，因此任何 D3Q27 完整候选都被 WITHHELD。

## 探针实现

实现：`src/tensorlbm/d3q27_composition_evidence.py`
测试：`tests/test_d3q27_composition_evidence.py`

探针直接调用 `collide_mrt27`，不经过 dispatch 层，避免来源掩盖。使用固定 CPU `float32` 输入、固定随机种子 `2718`、域大小 `(nz, ny, nx) = (2, 3, 4)` 及 `tau=0.8`，与 `test_d3q19_d3q27_mrt_consistency.py` 的探针状态完全一致。

## 六项组件级检查（全部 PASS）

| # | 检查 | 通过条件 | 描述 |
|---|---|---|---|
| 1 | equilibrium fixed point | `max abs(out - feq) <= 1e-6` | 平衡分布是碰撞的不动点 |
| 2 | mass invariant | `abs(sum(out) - sum(f)) <= 1e-6` | 碰撞保守质量 |
| 3 | momentum invariant | `max abs(rho*u_after - rho*u_before) <= 1e-6` | 碰撞保守三方向动量 |
| 4 | finite output | `torch.isfinite(out).all() == True` | 输出无 NaN/Inf |
| 5 | determinism | `torch.equal(first, second) == True` | 相同 clone 输入的两次碰撞 bitwise 一致 |
| 6 | source hash | `SHA-256(inspect.getsource(collide_mrt27)) == 4b1b55...` | 证据绑定到碰撞实现源码 |

所有六项通过后，`component_evidence_tier = "component_contract"`。

## WITHHELD 完整 composition 方面

以下六个方面在探针 artifact 中被显式标记为 WITHHELD，表示它们**不在**本组件级证据的覆盖范围内：

| 方面 | WITHHELD 理由 |
|---|---|
| wall_treatment | `bounce_back_cells_27` + `collide_mrt27` 耦合未验证为完整 wall-treatment composition；无 bounce-back + MRT 的积分质量/动量平衡测试 |
| geometry | `static_solid_mask` + `collide_mrt27` 未验证为完整 geometry composition；无 obstacle-mask + MRT 的闭合质量/动量平衡测试 |
| boundary | Zou/He inlet/outlet + `collide_mrt27` 未验证为完整 boundary composition；无 inlet/outlet + MRT 质量守恒测试 |
| output | `macroscopic27` (rho/velocity 提取) + `collide_mrt27` 未验证为完整 output composition；无端到端输出精度测试 |
| streaming_collision_coupling | `stream27` + `collide_mrt27` 耦合未验证为完整时间步进 composition；无多步 stream + collide 质量/动量守恒测试 |
| force_observation | `compute_obstacle_forces_27` + `collide_mrt27` 未验证为完整 force-observation composition；无力 + MRT 积分动量平衡测试 |

## 机器可读 artifact

探针函数 `run_d3q27_mrt_composition_probe()` 返回一个 JSON-ready dict，包含：

- `artifact_id`: `"d3q27-mrt-composition-evidence-r1"`
- `version`: `"d3q27-composition-evidence-r1"`
- `lattice` / `collision` / `entrypoint`
- `probe_config`: shape, dtype, device, seed, tau
- `checks`: 六项检查的 status + 测量值 + 容差
- `component_evidence_tier`: `"component_contract"`（全部 PASS 时）
- `withheld_composition`: 六个 WITHHELD 方面及理由
- `capability_matrix_cross_reference`: general_capability_matrix 判定 + advanced_collision_contract 判定
- `artifact_sha256`: artifact 自身的 canonical JSON SHA-256

执行命令：

```bash
pytest -q tests/test_d3q27_composition_evidence.py
```

## dtype 边界

探针输入明确限定为 `float32`。D3Q27 的 cached moment matrices（`_get_d3q27_mrt_matrices`）无论 population dtype 都返回 `torch.float32` matrix 与 inverse，因此以 `float64` populations 调用 `collide_mrt27` 会在矩阵乘法处以 dtype mismatch `RuntimeError` 失败。该限制由 `test_d3q19_d3q27_mrt_consistency.py::test_d3q27_mrt_documented_float32_only_matrix_limitation` 锁定。

本 artifact 证明的是当前 float32 CPU 路径下的组件级数值一致性；不外推到 float64、GPU/其他加速器、streaming/boundary/forcing 耦合、长时间稳定性或物理解精度。
