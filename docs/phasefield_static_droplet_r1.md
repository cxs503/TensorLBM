# D3Q19 CH 静滴初始化与 Laplace-style 诊断 R1

`tensorlbm.phasefield.static_droplet` 提供纯张量、三维 `(z, y, x)` 的
周期静滴初始化：

```python
phi = initialize_static_droplet(
    (48, 48, 48), radius=10.0, interface_width=1.5
)
```

其剖面为 `tanh((R-r)/(sqrt(2)*interface_width))`，其中 `r` 是 minimum-image
周期距离；因此球心与界面可以跨越任一周期面。`radius`、`interface_width`、
shape 和 center 均会验证。初始化器不创建 D3Q19 population，也不改变生产
`multiphase3d`、自由能或公共差分算子。

`diagnose_static_droplet(phi, DoubleWellFreeEnergy(...))` 必经公共
`DoubleWellFreeEnergy.chemical_potential(..., boundary="periodic")` 与
`force_minus_phi_grad_mu(..., boundary="periodic")`。它报告：

- `phi > 0` 的阈值体积、等效半径、周期圆均值球心；
- 候选平均曲率 `2/R`（几何量，不是压力验证）；
- 平滑体积和 Korteweg `-phi grad(mu)` 的净力、范数、`mu` 范围。

## 明确 WITHHELD 的 Laplace-style 链

该路径没有真实热力学 pressure field，也不从 LBM distribution、`mu` 或
Korteweg force 虚构一个 pressure。故 `LaplaceStyleDiagnostic` 固定返回：

- `status="withheld"`；
- `observed_pressure_jump=None`；
- `expected_pressure_jump=None`；
- 说明文字明确禁止把 `mu` 或 force 当作 pressure。

顶层结果始终为 `status="diagnostic_only"` 和
`physical_acceptance=False`。这不是 Laplace PASS、表面张力标定、静滴平衡
证明，亦不构成生产 D3Q19/CH solver 的物理接受准则。