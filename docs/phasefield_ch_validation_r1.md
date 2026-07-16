# R1: 公共 D3Q19 Free-Energy CH 封闭周期域诊断链

## 目的与边界

`tensorlbm.phasefield.ch_validation` 提供一个小尺寸、确定性的 runner，真实调用现有
`tensorlbm.multiphase3d.free_energy_step_3d`。它只用于观察该 D3Q19 Free-Energy
碰撞步在无几何、无外力、周期有限差分环境中的数值输出；不是船体、气泡或溃坝案例。

结果始终明确标记：

- `status == "diagnostic_only"`
- `physical_acceptance == False`
- `conservation_claim == False`

它不宣称严格 CH 守恒，也不构成物理验证或验收门槛。该生产步本身是 collision-only；
本 runner 不附加 streaming、边界处理或物理标定。

## 记录量（刻意不混称）

每一个时间点均记录：

- `phase_integral = sum(phi)`，其中 `phi = sum_i(g_i)`；
- `f_mass = sum_i,x(f_i)`，即动量分布 `f` 的全 zeroth moment；
- `phi`、`f`、`g` 各自的 finite 标志和各自范围（min/max）。

没有将 `sum(g)` 另命名为质量，也没有将 phase integral 叫作 phase volume，更不会以
`f_mass` 代替上述任一 phase 量。

## 使用方式

```python
from tensorlbm.phasefield import (
    FreeEnergyCHValidationConfig,
    run_closed_periodic_free_energy_diagnostic,
    uniform_phase_capillary_force,
)

result = run_closed_periodic_free_energy_diagnostic(
    FreeEnergyCHValidationConfig(shape=(3, 4, 5), steps=2, seed=7)
)
assert result.status == "diagnostic_only"
assert result.physical_acceptance is False
assert len(result.series) == 3  # 初值加两个真实 free_energy_step_3d 调用后的样本
```

`uniform_phase_capillary_force` 使用同一公共 `DoubleWellFreeEnergy` 与
`force_minus_phi_grad_mu` 周期算子。常数 `phi` 的化学势也是常数，故 capillary force
三分量为零；这只是算子一致性检查。

## 可重复性

初始 `phi` 在 CPU 上由局部、指定 seed 的 `torch.Generator` 生成；初始 `f` 和 `g`
分别由 D3Q19 平衡分布和 `init_free_energy_g_3d` 创建。随后 runner 在每一步将更新后的
`f` 与 `g` 原样传回真实 `free_energy_step_3d`，最少要求并执行两步。
