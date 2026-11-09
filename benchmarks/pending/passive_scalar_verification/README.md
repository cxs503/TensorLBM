# passive_scalar_verification（pending：待战役机门）

**状态：本地形式跑（formal）四门全过，等战役 DAG 机门独立复判。** 判据与产物见
`artifacts/passive_scalar_verification.json`（`verdict: PASS`，`criteria_pass` 四项全
true）。

本目录是 P73W2B-N05 节点（TensorLBM 被动标量输运验证面）的算例入口。模块本体在
`src/tensorlbm/passive_scalar_verification.py`，pytest 在
`tests/test_passive_scalar_verification.py`（17 例，34.6 s）。

## 物理问题

被动标量（示踪剂/盐度/温度）在剪切流中的**长期**输运不是分子扩散：横向剪切把标量拉成
细丝、横向扩散再把细丝抹平，等效地沿程扩散被放大 `1 + O(Pe²)`。Taylor (1953) / Aris
(1956) 给出两种经典几何的闭式：

```
平面缝道（缝高 H，平均流速 U，扩散系数 D）：D_eff/D = 1 + Pe²/210,  Pe = U H / D
圆管    （半径 a，平均流速 U，扩散系数 D）：D_eff/D = 1 + Pe²/48 ,  Pe = U a / D
```

⚠️ **本节点勘误**：战役种子把 `1 + Pe²/210` 标为「圆管」。`1/210` 是**平面缝道**系数；
圆管系数是 `1/48`（等价于 `1 + Pe_D²/192`，`Pe_D = 2aU/D`）。本验证面两种几何都验，
artifact 的 `criteria.G1.geometry_attribution` 里显式登记。

## 四门（判据先于动手，全部机器可算）

| 门 | 内容 | 判据 | 本次结果 |
|---|---|---|---|
| G1 | Taylor–Aris 剪切弥散解析锚，`Pe ∈ [0.1, 100]` | 误差 < 3% | **PASS**，最大 0.666%（LBM Pe=5）／离散 Aris ≤ 0.0002% |
| G2 | erf 扩散剖面 `C = 0.5·erfc(x/√(4Dt))` 二阶收敛，3 档网格 | L2 阶 ≥ 1.9 | **PASS**，阶 2.001 / 1.994（min 1.994） |
| G3 | Schmidt 数无关性，`Sc ∈ [0.7, 10]`，Pe 固定 | 剖面/比值最大偏差 < 2% | **PASS**，spread 1.606%，max\|dev\| 1.814% |
| G4 | 离散质量守恒 | \|ΔM\|/M < 1e-8 | **PASS**，最大 2.66e-12 |
| G5 | 机器可读工件 | 路径存在 | `artifacts/passive_scalar_verification.json` |

## 三条互不依赖的证据线（G1）

1. **闭式**（`taylor_aris_ratio`）：`1 + Pe²/210`、`1 + Pe²/48`。
2. **离散 Aris 辅助问题**：在格点实际所在的横截面节点上解 `D∇²χ = -u'`（壁面零通量、
   圆管含 `1/r` 柱算子与轴上零通量），`D_eff = D + <u'χ>`。它去掉了「节点平均速度 vs
   连续 U」的记账差，剩下的就是求积/曲率误差。`Pe ∈ {0.1,1,5,20,50,100}` × 两种几何
   共 12 行，误差 `≤ 2e-4 %`；横向加密 16→256 呈 O(h²)（缝道 −0.0209% → −0.0000%）。
3. **时推进 LBM**：D2Q9 标量核 + 半程零通量壁（`tensorlbm.taylor_aris`），速度场为解析
   Poiseuille 抛物线，`D_eff` 由基频模态衰减率给出（`method="mode"`）。

| Pe | u_mean | D_eff/D (LBM) | 闭式 | 离散 Aris | err vs 闭式 | err vs 离散 | 质量漂移 | r² |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.000235 | 0.995516 | 1.000048 | 1.000048 | −0.453% | −0.453% | 6.6e-13 | 1.000000 |
| 2 | 0.004706 | 1.014025 | 1.019048 | 1.019043 | −0.493% | −0.492% | 6.6e-13 | 1.000000 |
| 5 | 0.011765 | 1.111599 | 1.119048 | 1.119018 | −0.666% | −0.663% | 6.3e-13 | 1.000000 |
| 20 | 0.047059 | 2.900511 | 2.904762 | 2.904285 | −0.146% | −0.130% | 6.7e-13 | 1.000000 |
| 40（探针，**不计入门**） | 0.094118 | 9.367605 | 8.619048 | 8.617138 | +8.685% | +8.709% | 6.8e-13 | 1.000000 |

Pe=40 探针是**刻意保留的失效点**：`u = 0.094` 已逼近格子上限（`|u| < 0.1`），O(Ma²) 晶格
误差把 `D_eff` 抬高 8.7%。它说明「时推进 LBM 的可用 Pe 上限约 20–30（本设置）」，高 Pe 段
由离散 Aris 参考覆盖，而不是靠调参硬凑。

## G2：erf 扩散剖面

`dt ∝ dx²`（扩散标度，`τ_s = 0.8` 固定 → 晶格扩散系数恒为 0.1），三档 `nx = 128/256/512`，
`t = 200`。L2 误差只在 `|x−x0| ≤ 3√(4Dt)` 窗口内计（周期镜像在该窗口外 < 1e-19）：

| level | nx | dx | steps | L2 误差 | 阶 |
|---|---|---|---|---|---|
| 0 | 128 | 1.0 | 200 | 1.9665e-04 | — |
| 1 | 256 | 0.5 | 800 | 4.9116e-05 | 2.001 |
| 2 | 512 | 0.25 | 3200 | 1.2333e-05 | 1.994 |

若改用声学标度（`dt ∝ dx`）冻结 dx，`O(dt)` 项会把观测阶压到 1——这是本次唯一「参数选
错就测不出二阶」的坑。

## G3：Schmidt 数无关性

`nu` 由伴随水力松弛钉住（`τ_f = 0.8 → ν = 0.1`），`D = ν/Sc`；同时把施加流速按
`u_mean = Pe·D/H` 缩放，使 **Pe 固定为 3**（无量纲问题完全相同），只有 `τ_d = 3D + 0.5`
随 Sc 变。任何散布都是与 `τ_d` 有关的数值伪扩散。

| Sc | D | u_mean | steps | D_eff/D | 相对闭式 |
|---|---|---|---|---|---|
| 0.70 | 0.1429 | 0.0252 | 2023 | 1.023945 | −1.814% |
| 1.00 | 0.1000 | 0.0176 | 2890 | 1.029660 | −1.265% |
| 2.50 | 0.0400 | 0.0071 | 7225 | 1.037233 | −0.539% |
| 5.00 | 0.0200 | 0.0035 | 14450 | 1.039588 | −0.314% |
| 10.00 | 0.0100 | 0.0018 | 28900 | 1.040555 | −0.221% |
| 0.70（分辨率对照 ny=33） | 0.1429 | 0.0252 | 7623 | 1.033810 | −0.868% |

max|dev| = 1.814% < 2%，spread = 1.606%。趋势随 Sc 增大而收敛（`τ_d → 0.5` 侧误差更小），
且最差档 Sc=0.7 在 ny 17→33 时误差 −1.814% → −0.868%（约减半）——**残差是横向分辨率的
O(h²) 效应，不是 Sc 物理**。这是本门唯一的解释性结论，故把对照表一并入账。

## 复现

```bash
# 形式跑（CPU 约 2.5 min；本仓用集群 loong64 节点跑，153.5 s）
python benchmarks/pending/passive_scalar_verification/run.py --threads 16
# 分两步（跑 + 出图），便于无 matplotlib 的节点
python .../run.py --no-figs --payload artifacts/passive_scalar_verification.payload.json
python .../run.py --figs-from artifacts/passive_scalar_verification.payload.json
# 秒级冒烟（分辨率不足，门不保证过，只验管线）
python .../run.py --smoke
pytest tests/test_passive_scalar_verification.py -q
```

## 库入口（零手写物理核）

| 用途 | 入口 |
|---|---|
| D2Q5 标量平衡/碰撞/周期 streaming | `tensorlbm.scalar_transport.scalar_equilibrium / scalar_collision / scalar_stream / scalar_macroscopic` |
| 标量扩散系数 ↔ 松弛时间 | `tensorlbm.scalar_transport.tau_from_diffusivity` |
| D2Q9 标量 + 半程零通量壁 | `tensorlbm.taylor_aris.scalar_*`、`simulate_channel_dispersion` |
| 缝道/圆管解析速度剖面 | `tensorlbm.taylor_aris.poiseuille_channel_velocity / poiseuille_pipe_velocity` |
| 离散 Aris 参考（交叉校验用） | `tensorlbm.taylor_aris.aris_dispersion_coefficient` |
| 三对角求解（本模块内，LAPACK-free） | `passive_scalar_verification._thomas_solve` |

## 注意点

1. **`numpy.linalg.solve` 在集群 loong64 节点上不可信**：对 64 阶以上的 pin 过的近奇异
   三对角系统，返回残差 1.9e2 的解（scipy 同矩阵 1.1e-12）。本模块因此自带 Thomas 三对角
   求解，并把库内 `aris_dispersion_coefficient` 的解作为交叉校验入账
   （`library_cross_check.trustworthy` 在 n≥64 时为 false）。
2. **窗口/边界**：erf 前驱的周期域必须宽到「窗口边缘到对侧」仍远超扩散长度，否则回流镜像
   让误差不随网格收敛（第一版 nx=64 就踩了这个坑：三档误差全是 5.8e-2，阶 0）。
3. **模态法要求 `kh` 小**：nx=64（`kh≈0.8`）时 Pe=20 误差 +6.3%，nx=128 降到 −0.15%。
4. 速度场是解析抛物线（只验标量核）；圆管无时推进 LBM（库内标量核是笛卡尔，缺轴对称源项）。

## 局限

见 artifact 的 `limitations` 字段：圆管锚是离散柱 Aris 而非轴对称/3D LBM 实跑；LBM 段只
覆盖 Pe ≤ 20（探针给出 40 的失效证据）；Sc 只通过 `τ_d` 进入离散问题（未联动变 `τ_f`）。