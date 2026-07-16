# R1：D3Q19 CH production collision-only evolution adapter 合同

## 范围

`tensorlbm.phasefield.evolution_adapter` 是一个明确、窄范围的适配层：它以成对状态
`FreeEnergyCollisionOnlyState(f, g)` 驱动现有 production
`tensorlbm.multiphase3d.free_energy_step_3d`。适配层不改变该 production step，也不实现
streaming 或 boundary。

- 状态形状固定为 `f.shape == g.shape == (19, nz, ny, nx)`；空间尺寸为正数。
- `f` 与 `g` 必须都是浮点 tensor，且 shape、device、dtype 完全一致。
- 初始化入口 `initialize_free_energy_collision_only_state(phi)` 要求浮点三维
  `phi`，以真实 production `init_free_energy_g_3d(phi)` 创建 `g`，以单位密度、零速度
  D3Q19 equilibrium 创建 `f`。
- `FreeEnergyCollisionOnlyConfig.steps` 至少为 2，确保 runner 可以交接至少两次更新后的
  `(f, g)`。

## 阶段与明确冻结的限制

`run_free_energy_collision_only` 的每一步只调用一次真实 production
`free_energy_step_3d`，并将它返回的更新后 `(f, g)` 传入下一步。返回结果固定声明：

```python
result.stage == "collision_only"
result.status == "no_streaming_boundary_withheld"
result.physical is False
```

production step 内的微分算子是周期有限差分；该事实记录在
`result.differential_operator`。这不是 streaming/boundary 策略：本 R1 不搬运 population，
不施加任何边界、几何或墙面处理。后续 streaming/boundary owner 必须在此 adapter 外接入，
才能形成完整的演化链。

## 诊断账本

初值和每个 collision 后的样本都分别记录，绝不互相改名：

- `phi_integral = sum_x(phi)`，其中 `phi = sum_i(g_i)`；
- `f_mass = sum_i,x(f_i)`；
- `g_sum = sum_i,x(g_i)`。

这些是状态观测，不是物理接受、守恒、pressure 或 Laplace 结论。

## 最小使用示例

```python
import torch
from tensorlbm.phasefield import (
    FreeEnergyCollisionOnlyConfig,
    initialize_free_energy_collision_only_state,
    run_free_energy_collision_only,
)

phi = torch.zeros((3, 4, 5), dtype=torch.float32)
state = initialize_free_energy_collision_only_state(phi)
result = run_free_energy_collision_only(state, FreeEnergyCollisionOnlyConfig(steps=2))

assert result.state.f.shape == result.state.g.shape == (19, 3, 4, 5)
assert [sample.step for sample in result.diagnostics] == [0, 1, 2]
assert result.status == "no_streaming_boundary_withheld"
assert result.physical is False
```
