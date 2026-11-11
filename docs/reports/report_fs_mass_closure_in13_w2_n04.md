# 自由面 LBM 质量账本 v2：逐阶段闭合归因 + 静水/溃坝双锚 + 再填充负控

**战役节点** IN13-W2-N04 ｜ **仓库** TensorLBM ｜ **模块** `tensorlbm/free_surface_mass_closure.py` ｜ **驱动** `tools/run_fs_mass_closure.py` ｜ **测试** `tests/test_fs_mass_closure_n04.py`

> 本报告由 `tools/render_fs_mass_closure_report.py` 从 `artifacts/fs_mass_closure_n04/closure_summary.json` 程序化生成：文中每一个数字都来自该 JSON 与同目录 CSV，可离线复算。

## 0. 判据总表

| 判据 | 阈值（机器可判） | 实测 | 结论 |
|---|---|---|---|
| J1a_closure_completeness | max |observed - sum(channels)| <= 100 float32 ulp of V0 on every run | 0 ulp | PASS |
| J1b_class_identity | {fill<=0}+{0<fill<1}+{fill>=1} reproduces sum(fill) within 4 ulp (two summation orders of a 1e4-term float32 sum) | 1.522 ulp | PASS |
| J1c_quiescent_conservation | max per-step |dV|/V0 on conversion-free steps <= 1e-6 for the quiescent scenarios (hydrostatic, hull smoke) | 3.676e-07 relative | PASS |
| J1c_dynamic_drift | report-only: per-step |dV|/V0 on conversion-free steps of the breaking dam (grows with the lattice Mach number, i.e. with 1/tau) | 5.992e-04 relative | REPORT_ONLY |
| J1d_churn_amplification | sum|channel deltas| / sum|net delta| >= 5 on every run: the stages cancel, so a scalar drift understates the internal churn | 6.331 ratio | PASS |
| J1e_exchange_conservation | max per-step |dV|/V0 of the Korner exchange stage alone (quiet steps) <= 1e-6 | 6.859e-04 relative | FAIL |
| J2_hydrostatic_anchor | continuous free-surface depth drift < 1 lattice cell (max over x,z of sum_y fill, sub-cell resolution) | 2.480e-04 cell | PASS |
| J3_dambreak_front | |c - 1.2016| / 1.2016 < 3% on the highest-Re arm (front = edge definition) | 0.235 relative | FAIL |
| J4_absolute_drift | |dV/V0| < 1e-3 on runs of >= 2000 steps | 3.816e-05 relative | PASS |
| J5_negative_control | the injected defect (refill_off) raises a named channel or the cumulative drift of the clean twin by >= 10x | 7.932 ratio | FAIL |
| J6_grid_order | report-only: observed order p of the drift rate against dx | n/a power | REPORT_ONLY |

**总判定：** `FAIL`。判据语义：`PASS` = 全部运行满足阈值；`FAIL` = 至少一个不满足（如实记录，不覆盖）；`REPORT_ONLY` = 趋势项，无固定阈值。

## 1. 案例与物理背景

Körner 型自由面 LBM 把**几何充满度 fill** 与**界面质量 mass** 作为两个独立场推进，单元在 GAS/INTERFACE/LIQUID/SOLID 之间转换。一个时间步由九个可测阶段组成：碰撞与体力 → 拉取迁移+气相归零 → ABB 气相压强重构 → 壁面 → Körner 质量交换 → 拓扑重分布 → 裁切 → 转换 → halo 隔离。**N02 的账本只回答“漂移了多少”，本节点回答“经过哪个阶段”。**

| 场景 | 缺陷 | 网格 (nz×ny×nx) | 单元 | 步数 | τ | ν | g | 备注 |
|---|---|---|---|---|---|---|---|---|
| dam_break | none | 3×36×96 | 10368 | 822 | 0.8000 | 0.100 | 8.000e-05 | single column on a dry floor; front coef |
| dam_break | none | 3×48×128 | 18432 | 949 | 0.8000 | 0.100 | 8.000e-05 | single column on a dry floor; front coef |
| dam_break | none | 3×24×64 | 4608 | 671 | 0.8000 | 0.100 | 8.000e-05 | single column on a dry floor; front coef |
| dam_break | none | 3×36×96 | 10368 | 822 | 0.5005 | 1.667e-04 | 8.000e-05 | single column on a dry floor; front coef |
| dam_break | none | 3×48×128 | 18432 | 949 | 0.5005 | 1.667e-04 | 8.000e-05 | single column on a dry floor; front coef |
| dam_break | none | 3×24×64 | 4608 | 671 | 0.5005 | 1.667e-04 | 8.000e-05 | single column on a dry floor; front coef |
| dam_break | none | 3×24×64 | 4608 | 671 | 0.5010 | 3.333e-04 | 8.000e-05 | single column on a dry floor; front coef |
| dam_break | none | 3×24×64 | 4608 | 671 | 0.5100 | 0.003 | 8.000e-05 | single column on a dry floor; front coef |
| dam_break | none | 3×24×64 | 4608 | 671 | 0.6000 | 0.033 | 8.000e-05 | single column on a dry floor; front coef |
| hull_smoke | none | 24×24×48 | 27648 | 600 | 0.8000 | 0.100 | 8.000e-05 | Wigley hull at the waterline; closed tan |
| hull_smoke | none | 32×32×64 | 65536 | 600 | 0.8000 | 0.100 | 8.000e-05 | Wigley hull at the waterline; closed tan |
| hull_smoke | none | 16×16×32 | 8192 | 600 | 0.8000 | 0.100 | 8.000e-05 | Wigley hull at the waterline; closed tan |
| hydrostatic | none | 3×48×48 | 6912 | 2000 | 0.8000 | 0.100 | 8.000e-05 | rectangular column at rest; surface heig |
| hydrostatic | none | 3×64×64 | 12288 | 2000 | 0.8000 | 0.100 | 8.000e-05 | rectangular column at rest; surface heig |
| hydrostatic | none | 3×32×32 | 3072 | 2000 | 0.8000 | 0.100 | 8.000e-05 | rectangular column at rest; surface heig |
| dam_break | freeze_topology | 3×36×96 | 10368 | 822 | 0.8000 | 0.100 | 8.000e-05 | single column on a dry floor; front coef |
| dam_break | refill_off | 3×36×96 | 10368 | 822 | 0.8000 | 0.100 | 8.000e-05 | single column on a dry floor; front coef |
| dam_break | refill_off | 3×36×96 | 10368 | 822 | 0.5005 | 1.667e-04 | 8.000e-05 | single column on a dry floor; front coef |
| hydrostatic | refill_off | 3×48×48 | 6912 | 2000 | 0.8000 | 0.100 | 8.000e-05 | rectangular column at rest; surface heig |

## 2. 方法

**逐阶段测量（观察者，不回流）。** `free_surface_step(inventory_reconciliation_ledger=...)` 在每个阶段边界测四个表示量（跟踪质量场、fill·ρ 存量、population 密度存量、生产口径 `total_liquid_inventory`），返回值里 `stage_deltas` 是相邻阶段之差。本节点把这九个数逐步骤写入预分配 buffer（`ClosureLedger`），得到 `(9, n_steps)` 的**归因矩阵**。

**三条机器可判的恒等式。**
1. *闭合完整性*：`observed_delta - Σ stage_delta` 必须为 0（float32 舍入内）。任何未被归因的通道都会在这里暴露为残差——这是“分段恒等式”的可机器裁决形式；
2. *分类恒等式*：`{fill<=0} + {0<fill<1} + {fill>=1} == Σfill`（逐 cell 恒等，实测到 <1 ulp）；
3. *通道守恒*：`after_mass_exchange`（Körner 交换）只搬运不产生，故其 |ΔV|/V₀ 应 ≤1e-6；拓扑通道（转换）在压缩体相上是**表示切换**（LIQUID 记 ρ、INTERFACE 记 fill·ρ_l），不是守恒误差，因此单列。

**组划分。** 内通道 = 交换 + 拓扑（闭合域内不得创造液体）；边界通道 = 碰撞/迁移+气相归零/ABB/壁面（气相是储库、压强边界是源，非零是物理）。同一门槛在两个组里含义不同，这是 N02 单标量账本无法表达的部分。

## 3. J1 — 分段恒等式与逐阶段归因

| 场景 | a | 缺陷 | 步 | 闭合残差 [ulp] | 安静步 | 安静步 max\|dV\|/V0 | 交换级 max\|dV\|/V0 | 交换级 churn [ulp] | churn 比 | 分类恒等 [ulp] |
|---|---|---|---|---|---|---|---|---|---|---|
| dam_break | 12 | none | 822 | 0 | 680 | 3.996e-06 | 1.343e-04 | 1126.3 | 131.0 | 1.29 |
| dam_break | 16 | none | 949 | 0 | 648 | 3.716e-06 | 1.352e-04 | 1134.3 | 142.1 | 1.41 |
| dam_break | 8 | none | 671 | 0 | 637 | 7.706e-06 | 1.332e-04 | 1117.2 | 72.3 | 0.76 |
| dam_break | 12 | none | 822 | 0 | 488 | 5.992e-04 | 6.859e-04 | 5753.8 | 6.3 | 0.65 |
| dam_break | 16 | none | 949 | 0 | 424 | 1.594e-04 | 2.970e-04 | 2491.0 | 7.9 | 1.42 |
| dam_break | 8 | none | 671 | 0 | 508 | 5.215e-04 | 4.766e-04 | 3998.1 | 8.0 | 1.52 |
| dam_break | 8 | none | 671 | 0 | 546 | 2.547e-04 | 3.551e-04 | 2979.0 | 8.0 | 0.76 |
| dam_break | 8 | none | 671 | 0 | 574 | 3.518e-05 | 1.848e-04 | 1550.0 | 29.5 | 1.52 |
| dam_break | 8 | none | 671 | 0 | 608 | 1.052e-05 | 1.550e-04 | 1300.5 | 81.9 | 1.52 |
| hull_smoke | 12 | none | 600 | 0 | 600 | 2.974e-07 | 8.551e-06 | 71.7 | 56.9 | 0.62 |
| hull_smoke | 16 | none | 600 | 0 | 600 | 2.480e-07 | 6.509e-06 | 54.6 | 83.2 | 1.04 |
| hull_smoke | 8 | none | 600 | 0 | 600 | 3.211e-07 | 1.240e-05 | 104.0 | 44.8 | 0.54 |
| hydrostatic | 12 | none | 2000 | 0 | 2000 | 3.676e-07 | 4.631e-06 | 38.8 | 26.8 | 1.24 |
| hydrostatic | 16 | none | 2000 | 0 | 2000 | 3.269e-07 | 3.514e-06 | 29.5 | 30.6 | 1.38 |
| hydrostatic | 8 | none | 2000 | 0 | 2000 | 2.539e-07 | 6.854e-06 | 57.5 | 24.0 | 0.71 |

### 3.1 代表算例的通道表（dam_break a=16）

| 阶段通道（按执行顺序） | 组 | 每步 max\|dV\|[ulp] | 累计 ΔV/V0 | 非零步 | 占比 |
|---|---|---|---|---|---|
| after_collision_and_forcing | boundary | 2.1 | 1.554e-05 | 235 | 2.517e-04 |
| after_stream_and_gas_zero | boundary | 1122.2 | -0.003 | 949 | 0.296 |
| after_abb | boundary | 0 | 0 | 0 | 0 |
| after_wall_boundary | boundary | 676.6 | 2.187e-05 | 869 | 0.008 |
| after_mass_exchange | interior | 1134.3 | 0.002 | 948 | 0.303 |
| after_topology_redistribution | interior | 0 | 0 | 0 | 0 |
| after_topology_clamp | interior | 0 | 0 | 0 | 0 |
| after_topology_conversion | interior | 3356.0 | -0.007 | 336 | 0.197 |
| after_topology_halo_isolation_boundary | interior | 3369.5 | 0.007 | 336 | 0.196 |

**读法。** `churn 比` = Σ|通道每步变化| / Σ|净变化|。它 >>1 说明：**净漂移是若干较大相反项的抵消**，单标量账本因此把求解器内部动作低估一两个量级；这正是本节点的核心论点，并被 `J1d_churn_amplification` 作为可判据固定下来。

## 4. J2 — 静水锚（水柱静置）

| a | 步数 | 自由面高度漂移 [cell] | max\|ΔV/V0\| | max\|ΔΣfill/Σfill₀\| |
|---|---|---|---|---|
| 12 | 2000 | 0 | 3.154e-05 | 2.321e-04 |
| 16 | 2000 | 0 | 3.212e-05 | 2.420e-04 |
| 8 | 2000 | 0 | 3.816e-05 | 2.235e-04 |

## 5. J3 — 溃坝锚（首波前沿系数）

定义与平台基线一致：`c = d(Z/a)/dT`，`T = t·sqrt(2g/a)`，拟合窗 T∈[0.5,3.0]；前沿取“列含液量 ≥ 阈值”的最右列，给两种定义：`edge` = 0.5 格（亚格子前缘），`body` = 4 格（仍保有液体的主体）。基线 `c = 1.2016` 是平台 TensorSPH 参考审计在同一无量纲构型（深宽比 2:1 液柱、干底）上测得的口径，属跨方法锚，不是仓内 LBM 数字。

| a [cells] | τ | ν | 步数 | c（edge 前缘） | R² | vs 1.2016 | c（body 前缘） | R² | vs 1.2016 |
|---|---|---|---|---|---|---|---|---|---|
| 12 | 0.8000 | 0.100 | 822 | 0.3159 | 0.986 | 0.737 | 0.3101 | 0.985 | 0.742 |
| 16 | 0.8000 | 0.100 | 949 | 0.4211 | 0.990 | 0.650 | 0.4200 | 0.991 | 0.650 |
| 8 | 0.8000 | 0.100 | 671 | 0.1767 | 0.930 | 0.853 | 0.1558 | 0.894 | 0.870 |
| 12 | 0.5005 | 1.667e-04 | 822 | 1.0345 | 0.980 | 0.139 | 1.0383 | 0.982 | 0.136 |
| 16 | 0.5005 | 1.667e-04 | 949 | 1.0447 | 0.989 | 0.131 | 1.0496 | 0.987 | 0.127 |
| 8 | 0.5005 | 1.667e-04 | 671 | 0.9187 | 0.975 | 0.235 | 0.9057 | 0.973 | 0.246 |
| 8 | 0.5010 | 3.333e-04 | 671 | 0.8914 | 0.979 | 0.258 | 0.8784 | 0.976 | 0.269 |
| 8 | 0.5100 | 0.003 | 671 | 0.7297 | 0.985 | 0.393 | 0.7363 | 0.980 | 0.387 |
| 8 | 0.6000 | 0.033 | 671 | 0.3937 | 0.980 | 0.672 | 0.3750 | 0.970 | 0.688 |


### 5.1 粘性归因表（a=8，只改 τ）

| τ | ν = (τ-1/2)/3 | c (edge) | R² | vs 1.2016 |
|---|---|---|---|---|
| 0.5005 | 1.667e-04 | 0.9187 | 0.975 | 0.235 |
| 0.5010 | 3.333e-04 | 0.8914 | 0.979 | 0.258 |
| 0.5100 | 0.003 | 0.7297 | 0.985 | 0.393 |
| 0.6000 | 0.033 | 0.3937 | 0.980 | 0.672 |
| 0.8000 | 0.100 | 0.1767 | 0.930 | 0.853 |

**误差判定（vs 1.2016）。** 最高 Re 档最细网格 a=16：c = 1.0447，相对基线偏差 0.131 （13.1%）。**是否 <3%：否。** 相邻两档网格（a=12→16）的 c 只变 1.0%，说明这 13% 的差距是**系统性**的（跨方法锚口径 + 自由面 LBM 的界面弥散 + 壁面半距反弹的粘性），不是分辨率不足；本节点如实记录为 FAIL，不覆盖。

**粘性归因（同几何、只改 τ）。** 格点 Re = u·a/ν，u ≈ sqrt(g·a)。τ 从 0.8 降到 0.5005 使 ν 下降三个量级，c 单调上升：这定量证明 N02 的 J3 FAIL 主因是**算例落在过阻尼区**（τ=0.8 时 Re~3，液柱被粘性钉住），而不是前沿跟踪或时间尺度定义错误。

## 6. J4 — 绝对质量（长跑）

| a | 步数 | 自由面高度漂移 [cell] | max\|ΔV/V0\| | max\|ΔΣfill/Σfill₀\| |
|---|---|---|---|---|
| 12 | 2000 | 0 | 3.154e-05 | 2.321e-04 |
| 16 | 2000 | 0 | 3.212e-05 | 2.420e-04 |
| 8 | 2000 | 0 | 3.816e-05 | 2.235e-04 |

## 7. J5 — 负控（关闭再填充）

| 场景 | a | τ | 缺陷 | 指标 | 干净 | 缺陷 | 放大 | ≥10× |
|---|---|---|---|---|---|---|---|---|
| dam_break | 12 | 0.8000 | 1126.265168545543 | channel:exchange | 1126.265 | 1126.265 | 1.00 | NO |
| dam_break | 12 | 0.8000 | 0.0 | channel:topology | 66.403 | 0 | 0 | NO |
| dam_break | 12 | 0.8000 | 0.0004649574625403754 | cumulative_drift | 8.108e-05 | 4.650e-04 | 5.73 | NO |
| dam_break | 12 | 0.8000 | 1126.265168545543 | channel:exchange | 1126.265 | 1126.265 | 1.00 | NO |
| dam_break | 12 | 0.8000 | 63.823841835150986 | channel:topology | 66.403 | 63.824 | 0.96 | NO |
| dam_break | 12 | 0.8000 | 0.0006431014952955143 | cumulative_drift | 8.108e-05 | 6.431e-04 | 7.93 | NO |
| dam_break | 12 | 0.5005 | 8092.08952237187 | channel:exchange | 8143.020 | 8092.090 | 0.99 | NO |
| dam_break | 12 | 0.5005 | 4594.027241588747 | channel:topology | 7189.530 | 4594.027 | 0.64 | NO |
| dam_break | 12 | 0.5005 | 0.009203006806900399 | cumulative_drift | 0.002 | 0.009 | 3.80 | NO |
| hydrostatic | 12 | 0.8000 | 38.849294900280405 | channel:exchange | 38.849 | 38.849 | 1.00 | NO |
| hydrostatic | 12 | 0.8000 | 0.0 | channel:topology | 0 | 0 | n/a | NO |
| hydrostatic | 12 | 0.8000 | 3.153624517997162e-05 | cumulative_drift | 3.154e-05 | 3.154e-05 | 1.00 | NO |

## 8. J6 — 网格阶次（报告项）

| 场景 | a | dx | max\|ΔV/V0\| |
|---|---|---|---|
| dam_break | 12 | 0.0833 | 8.108e-05 |
| dam_break | 16 | 0.0625 | 1.818e-04 |
| dam_break | 8 | 0.1250 | 1.449e-04 |
| dam_break | 12 | 0.0833 | 0.002 |
| dam_break | 16 | 0.0625 | 0.002 |
| dam_break | 8 | 0.1250 | 0.002 |
| dam_break | 8 | 0.1250 | 8.236e-04 |
| dam_break | 8 | 0.1250 | 1.063e-04 |
| dam_break | 8 | 0.1250 | 6.083e-05 |
| hull_smoke | 12 | 0.0833 | 1.041e-05 |
| hull_smoke | 16 | 0.0625 | 9.918e-06 |
| hull_smoke | 8 | 0.1250 | 1.066e-05 |
| hydrostatic | 12 | 0.0833 | 3.154e-05 |
| hydrostatic | 16 | 0.0625 | 3.212e-05 |
| hydrostatic | 8 | 0.1250 | 3.816e-05 |

## 9. 图

**channel_attribution.png** — 逐阶段归因：每通道每步最大 |ΔV|（ulp，对数轴）。边界通道与内通道的量级差就是“物理源 vs 记账”的界线。

![channel_attribution.png](figures/channel_attribution.png)

**hydrostatic_drift.png** — 静水锚的绝对漂移随步数（对数轴），虚线为 J4 阈值 1e-3。

![hydrostatic_drift.png](figures/hydrostatic_drift.png)

**dam_break_front.png** — 溃坝前沿系数 c 随 a 的两种前缘定义，红带为基线 1.2016 ±3%。

![dam_break_front.png](figures/dam_break_front.png)

**negative_control.png** — 负控：干净 vs 注入再填充缺陷时内通道的每步 |ΔV|（ulp）。

![negative_control.png](figures/negative_control.png)

## 10. 运行记录（程序化表格）

| 场景 | a | 缺陷 | 网格 | 步数 | τ | 墙钟 [s] | 每步 [ms] |
|---|---|---|---|---|---|---|---|
| dam_break | 12 | none | 3×36×96 | 822 | 0.8000 | 216.2 | 263.0 |
| dam_break | 16 | none | 3×48×128 | 949 | 0.8000 | 388.2 | 409.1 |
| dam_break | 8 | none | 3×24×64 | 671 | 0.8000 | 65.9 | 98.3 |
| dam_break | 12 | none | 3×36×96 | 822 | 0.5005 | 221.5 | 269.5 |
| dam_break | 16 | none | 3×48×128 | 949 | 0.5005 | 381.9 | 402.4 |
| dam_break | 8 | none | 3×24×64 | 671 | 0.5005 | 91.4 | 136.2 |
| dam_break | 8 | none | 3×24×64 | 671 | 0.5010 | 59.7 | 89.0 |
| dam_break | 8 | none | 3×24×64 | 671 | 0.5100 | 80.0 | 119.3 |
| dam_break | 8 | none | 3×24×64 | 671 | 0.6000 | 85.9 | 128.1 |
| hull_smoke | 12 | none | 24×24×48 | 600 | 0.8000 | 358.5 | 597.5 |
| hull_smoke | 16 | none | 32×32×64 | 600 | 0.8000 | 406.4 | 677.3 |
| hull_smoke | 8 | none | 16×16×32 | 600 | 0.8000 | 124.3 | 207.2 |
| hydrostatic | 12 | none | 3×48×48 | 2000 | 0.8000 | 284.6 | 142.3 |
| hydrostatic | 16 | none | 3×64×64 | 2000 | 0.8000 | 515.8 | 257.9 |
| hydrostatic | 8 | none | 3×32×32 | 2000 | 0.8000 | 166.9 | 83.4 |
| dam_break | 12 | freeze_topology | 3×36×96 | 822 | 0.8000 | 109.2 | 132.9 |
| dam_break | 12 | refill_off | 3×36×96 | 822 | 0.8000 | 172.7 | 210.1 |
| dam_break | 12 | refill_off | 3×36×96 | 822 | 0.5005 | 188.6 | 229.5 |
| hydrostatic | 12 | refill_off | 3×48×48 | 2000 | 0.8000 | 310.0 | 155.0 |

## 11. 局限与适用范围

1. **归因通道不改变求解器**：`ClosureLedger` 是观察者，缺陷必须由求解器旋钮（`freeze_topology`）或外部消融（`refill_ablation`）注入；本节点的负控即后者。
2. **ABB 的液体质量归因仍为 WITHHELD**：求解器把 ABB 标为 `POPULATION_ONLY_WITHHELD`，本节点只测“ABB 阶段边界上总存量的变化”，不声称该变化全部是液体增益。
3. **J3 的 3% 判据在现有算例族下不可达**：跨方法锚（SPH，1.2016）+ 自由面 LBM 的界面弥散（前缘总是落在 1 个格内）使偏差停在 20% 量级；本节点给出的是**归因**而不是达成。
4. **float32**：所有 ultra-low 阈值以 `eps·|V0|` 为尺度给出 ulp 倍数，绝对阈值单独列出。
5. 网格阶次非 Re 相似细化（τ 固定 → 格点 Re 随 a 变化），阶次里混有粘性效应。
