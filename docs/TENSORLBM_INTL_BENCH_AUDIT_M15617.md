# TensorLBM 国际对标专项审计（GAP-LBM-INTL-BENCH · 第1轮）

**mission**: m15617 · **audit_id**: `m15617-intl-bench-v1` · **repo**: TensorLBM · **生成**: 2026-11-12T01:35:29+00:00  
**落库**: `tensorknowledge` 库 `tensorlbm_module_maturity`(45 行) / `tensorlbm_compete_score`(10 行)；DDL+INSERT=`docs/tensorlbm_maturity_m15617.sql`  
**机器产物**: `artifacts/m15617/module_maturity.{json,csv}` · `compete_score.csv` · `level_distribution.png` · `compete_radar.png`  
**生成器**: `tools/audit/m15617_intl_bench_audit.py`（本文档所有数字由该脚本从 `result.json`/`STATUS.md` 抽取后注入，无手抄）

> 铁则遵守：本审计**不自宣布任何模块 verified**。verified 只授予「磁盘上有 benchmarks/verified/<case>/ + 其 result.json 含真值裁决字段 + 含网格/收敛类字段」三条同时成立者；缺一条自动降档（脚本 `grade()` 规则，见 ②）。

## ① 案例介绍：审计对象与参照系统

| 项 | 内容 |
|---|---|
| 审计对象 | TensorLBM 全物理模型（45 个模块/家族，覆盖基础内核→多相→GPU/HPC→工程验证→AI 融合） |
| 参照 1 | OpenLB `master@6c3934d`(2019-12-02)，本地 `/root/openlb` |
| 参照 2 | waLBerla 现代主干 `/tmp/walberla`（srt/trt/mrt/cumulant + free_surface + refinement + gpu） |
| 参照 3 | Palabos 2.3.0+dev `/tmp/palabos_probe` |
| 参照 4 | Lettuce `master@88a6559` `/tmp/up/lettuce`（Julia/CUDA 代码生成） |
| 判据 | 证据层级阶梯 1 模块存在 < 2 API 测试 < 3 守恒测试 < 4 标准算例 < 5 网格收敛 < 6 实验对比 |

四参照逐算法对照（碰撞算子/多相/边界/AMR/多GPU）见 `artifacts/m15617/upstream_cites.csv`；每条引用经脚本活体校验：
| 参照 | 引用校验通过 | 快照 |
|---|---|---|
| OpenLB | 12/12 | master@6c3934d (2019-12-02) |
| waLBerla | 16/16 | git HEAD (CMake>=3.26 主干) |
| Palabos | 13/13 | 2.3.0 + unreleased |
| Lettuce | 10/10 | master@88a6559 |

合计 51/51 条源码引用文件存在且含目标符号；失败 0 条。

**逐算法对照结论（差距点）**

| 维度 | OpenLB | waLBerla | Palabos | Lettuce | TensorLBM 对应 | 判定 |
|---|---|---|---|---|---|---|
| 碰撞算子 | BGK/TRT/MRT/entropic/freeEnergy | srt/trt/mrt/cumulant | BGK/TRT/ShanChen | BGK/KBC/MRT/regularized | `*collision*,*cumulant*` 多族在盘 | 高阶族缺算例→C |
| 多相模型 | ShanChen+freeEnergy | free_surface(flag/bubble) | ShanChen+freeSurfaceAnalysis | 无 | SC/CG/FE/Allen-Cahn/VOF 五族 | 五模型液滴审计全败→有代码无精度 |
| 边界处理 | Bouzidi 两点插值 | NoSlip/UBB/FreeSlip/Curved | bounceBackModels/NLD | bounce_back/partially_saturated | halfway BB/BFL/IBM/wall-function | BB 与 BFL 有 verified 算例；IBM 算例级缺陷 |
| AMR | 仅 super-BC 局部加密 | blockforest+lbm/refinement | gridRefinement+multiGrid(dev) | 无 | static_block_amr/octree | 有实现，无 AMR 网格收敛证据 |
| 多GPU | MPI only | src/gpu+pystencils | 仅 npFEM GPU | CUDA 单卡(+分布式分支) | triton_fused_distributed/torch_distributed | 仅自记 69 GLUPS/8GPU，无第三方复现 |

## ② 怎么做的：方法与机器规则

1. **证据分档不是人评**：`grade()` 只看磁盘证据。`tier` 由「实现文件在盘(=1) → 测试在盘(=2) → pending 算例在盘(=3) → verified 算例在盘(=4) → verified 算例 result.json 含 converg/grids/order 字段(=5)」递推；`level` 规则：tier≥5 且有真值裁决字段(`verified|pass|all_gates_pass|ok`) → `verified`；tier==5 但无裁决字段 → `partial`+记录降档理由；tier 3–4 → `partial`；KNOWN_DEFECTS 命中且无 verified 算例 → `failed`；否则 `implemented`。
2. **引用活体校验**：四参照每条引用=(相对路径, 目标符号)。文件不存在或符号 grep 不中即 RED，退出码非 0。本次 51/51 通过。
3. **十维竞争度**：`maturity_index = (mean_tier-1)/5*10`（1..6 → 0..10）；`tier_delta = mean_tier - max(上游 tier)`；差距等级 A(≥0)/B(-1)/C(-2)/D(≤-3)。上游 tier 按其例程/回归证据声明、**未复跑**，故上限 4（无一上游本轮达到 5/6）。
4. **取舍清单**：keep=verified；delete 候选=implemented 且测试/算例/文档三重空；其余 partial/failed→refactor。
5. **性能对标纪律**（⑦）：GLUPS 只在「格子/碰撞/精度/边界/规模/设备/计时」七项口径全同时可比。本审计**不做任何跨代码比数**；TensorLBM 的 8.6 GLUPS(单卡 RTX 5090, D3Q19, 周期, n=256, fp32 存储) 与 69 GLUPS(8 卡 slab) 均为**项目自记**（源码 docstring，`src/tensorlbm/triton_fused.py:1-46`），无第三方复现、无同口径上游对照 → 该两数在竞争度表里**不计入证据层级**。

## ③ 计算结果

**分档汇总（n=45）**：verified 7 · partial 21 · implemented 15 · failed 2

| 家族 | 模块数 | 平均证据层级 | 成熟度(0-10) | 上游最高 tier | Δtier | 差距等级 | v/p/i/f |
|---|---|---|---|---|---|---|---|
| 基础LBM内核 | 4 | 4.5 | 7.0 | 4 | +0.5 | A(并跑/领先) | 1/3/0/0 |
| 高阶碰撞 | 5 | 2.2 | 2.4 | 4 | -1.8 | C(差2级) | 0/1/4/0 |
| 边界复杂几何 | 5 | 4.2 | 6.4 | 4 | +0.2 | A(并跑/领先) | 2/3/0/0 |
| 多相流 | 5 | 3.6 | 5.2 | 4 | -0.4 | B(差1级) | 1/3/0/1 |
| 自由表面 | 3 | 3.33 | 4.7 | 4 | -0.67 | B(差1级) | 0/2/0/1 |
| 湍流 | 4 | 3.0 | 4.0 | 4 | -1.0 | B(差1级) | 1/1/2/0 |
| GPU单卡 | 4 | 2.25 | 2.5 | 4 | -1.75 | C(差2级) | 0/1/3/0 |
| 多GPU-HPC | 2 | 2.0 | 2.0 | 4 | -2.0 | C(差2级) | 0/0/2/0 |
| 工程验证 | 10 | 4.0 | 6.0 | 4 | +0.0 | A(并跑/领先) | 2/6/2/0 |
| AI融合 | 3 | 2.0 | 2.0 | 1 | +1.0 | A(并跑/领先) | 0/1/2/0 |

**verified 名单（7 个，全部证据回链）**

| 模块 | 算例 | 裁决字段 | 收敛字段 |
|---|---|---|---|
| d3q27_kernel | `benchmarks/verified/taylor_green_3d` | verified | True |
| wall_function_les | `benchmarks/verified/suboff_re1000` | verified,pass | True |
| stl_cad_voxelization | `benchmarks/verified/wigley_hydrostatics` | verified | True |
| laplace_law_surface_tension | `benchmarks/verified/laplace_droplet` | verified | True |
| rans_ke | `benchmarks/verified/suboff_re1000` | verified,pass | True |
| ship_resistance_hull | `benchmarks/verified/suboff_re1000` | verified,pass | True |
| compressible_shock | `benchmarks/verified/compressible_shock_family` | pass,pass,pass,pass | True |

**failed 名单（结构性，honest FAIL 保留）**

- `allen_cahn_multiphase`：NOTES.md #5：相场无法维持分散液滴（<200 步溶解），sigma 力缺曲率、相方程不守恒、M/tau_phi 硬编码
- `vof_free_surface`：rayleigh_taylor README：VOF 无自由压力场（结构性）

**保留 / 重构 / 删除**

保留(7)：`d3q27_kernel`, `wall_function_les`, `stl_cad_voxelization`, `laplace_law_surface_tension`, `rans_ke`, `ship_resistance_hull`, `compressible_shock`

重构(23)：`bgk_stream_collide_2d`, `solver3d_stream_collide`, `mms_order_verification`, `cumulant_d3q27`, `halfway_bounce_back`, `interpolated_bfl_wall`, `immersed_boundary_ibm`, `shan_chen_multiphase`, `color_gradient_multiphase`, `free_energy_multiphase`, `allen_cahn_multiphase`, `vof_free_surface`, `fs_mass_conservation`, `wave_tank_2d`, `smagorinsky_les`, `triton_fused_single_gpu`, `sixdof_motion`, `fwh_acoustics`, `non_newtonian_yield`, `thermal_convection`, `rotating_frame`, `dem_coupling_prototype`, `regression_gates_contract`

删除候选(2)：`wale_les`, `pod_reduced_order` —— 理由：实现文件在盘但测试/算例/文档三重空，属孤儿模块；删除前须 owner 复核（不排除外部用户引用）

**国际差距→开发任务（11 条，挂 T1-T5）**

| 锚 | 维度 | 模块 | 动作 | 现有证据 |
|---|---|---|---|---|
| T1 | 高阶碰撞 | `trt_collision` | 补 标准算例+网格收敛 证据回链 | `—` |
| T1 | 高阶碰撞 | `mrt_collision` | 补 标准算例+网格收敛 证据回链 | `—` |
| T1 | 高阶碰撞 | `kbc_entropic_collision` | 补 标准算例+网格收敛 证据回链 | `—` |
| T1 | 高阶碰撞 | `cumulant_d3q27` | 闭合结构性缺陷并补网格收敛 | `collision_kernels_3d_tg` |
| T1 | 高阶碰撞 | `regularized_collision` | 补 标准算例+网格收敛 证据回链 | `—` |
| T4 | GPU单卡 | `perf_solver_torch` | 补 标准算例+网格收敛 证据回链 | `—` |
| T4 | GPU单卡 | `triton_fused_single_gpu` | 闭合结构性缺陷并补网格收敛 | `collision_kernels_3d_tg` |
| T4 | GPU单卡 | `triton_obstacle_gpu` | 补 标准算例+网格收敛 证据回链 | `—` |
| T4 | GPU单卡 | `torch_fallback_backend` | 补 标准算例+网格收敛 证据回链 | `—` |
| T4 | 多GPU-HPC | `triton_distributed_multigpu` | 补 标准算例+网格收敛 证据回链 | `docs/triton_distributed_notes.md` |
| T4 | 多GPU-HPC | `torch_distributed_multigpu` | 补 标准算例+网格收敛 证据回链 | `—` |

T1 碰撞与内核 / T2 边界与复杂几何 / T3 多相与自由表面 / T4 GPU-HPC / T5 工程验证与 AI（本单解读，见 ⑤）。

**悬空/降档记录（partial 的理由）**：
- `bgk_stream_collide_2d`：tier=4（benchmarks/verified 算例在盘）
- `solver3d_stream_collide`：tier=5（verified 算例含网格/收敛证据字段）verified/ 在盘但 result.json 无真值裁决字段
- `mms_order_verification`：tier=4（benchmarks/verified 算例在盘）
- `halfway_bounce_back`：tier=4（benchmarks/verified 算例在盘）
- `interpolated_bfl_wall`：tier=4（benchmarks/verified 算例在盘）
- `shan_chen_multiphase`：tier=4（benchmarks/verified 算例在盘）
- `color_gradient_multiphase`：tier=4（benchmarks/verified 算例在盘）
- `free_energy_multiphase`：tier=4（benchmarks/verified 算例在盘）
- `wave_tank_2d`：tier=4（benchmarks/verified 算例在盘）
- `smagorinsky_les`：tier=4（benchmarks/verified 算例在盘）
- `sixdof_motion`：tier=4（benchmarks/verified 算例在盘）
- `fwh_acoustics`：tier=4（benchmarks/verified 算例在盘）
- `non_newtonian_yield`：tier=5（verified 算例含网格/收敛证据字段）verified/ 在盘但 result.json 无真值裁决字段
- `thermal_convection`：tier=5（verified 算例含网格/收敛证据字段）verified/ 在盘但 result.json 无真值裁决字段
- `rotating_frame`：tier=4（benchmarks/verified 算例在盘）
- `dem_coupling_prototype`：tier=4（benchmarks/verified 算例在盘）
- `regression_gates_contract`：tier=4（benchmarks/verified 算例在盘）

## ④ 结果图

![level_distribution](artifacts/m15617/level_distribution.png)

![compete_radar](artifacts/m15617/compete_radar.png)

图 1：四级分档直方图（n=45）。图 2：十维竞争度雷达（TensorLBM 成熟度 vs 上游最好证据层级折算）。两图由生成器 matplotlib 程序化产出，无手绘。

## ⑤ 注意点（局限与适用范围）

1. **上游未复跑**：四参照仅做「源码/例程存在性 + 算法面」对照，tier 上限 4；任何「waLBerla 的 AMR 比 TensorLBM 强」的表述限定在**证据层级**而非实测精度。
2. **OpenLB 快照陈旧**：`master@6c3934d`(2019-12) 无 multiPhaseModels/refinement/GPU 目录；其多相 tier=3 是按 `src/dynamics/freeEnergyDynamics.cpp` 与 `ForcedShanChenBGKdynamics` 判的，新版本(1.6/1.7)可能更高——本审计**不为新版背书**。
3. **算例复用**：`suboff_re1000` 同时为 wall_function_les / rans_ke / ship_resistance_hull 的 verified 证据，单案例支撑三模块，证据独立性弱；`compressible_shock_family` 以多 gate 计数支撑 verified（裁决字段 pass×2）。
4. **AI 融合维度上游 tier=1 为声明性假设**（四参照本快照均无神经闭合生产线），故该维 A 级只说明「没有落后」，不代表 AI 面已成熟（内部也仅 partial×1 + implemented×2）。
5. **平台台账缺口**：`verdict_m15492.sql` 记录 `tensor_benchmark_cases` 对 CASE-TAYLOR-GREEN-3D 0 行；本单 verified 授予基于**仓内**证据链（verified 目录+result.json），未依赖平台台账行。
6. **落库位置**：`tensorplatform` 库 public schema 拒绝建表（permission denied），DDL 落在 `tensorknowledge` 库；两表以 `audit_id` 主键支持每周刷新（重跑生成器即 delete+insert 幂等替换，跨版本换 audit_id 叠加）。
7. **删除清单非执行令**：仅规则导出候选，真删需 owner 复核与下游引用扫描。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 模块 | 算例 | 关键误差（机器抽取自 result.json） | <3%? |
|---|---|---|---|
| d3q27_kernel | taylor_green_3d | `cases[0].err_e_pct`=0.3093; `cases[0].err_vel_pct`=0.3093 | 见备注 |
| wall_function_les | suboff_re1000 | `acceptance.primary_err_pct`=-1.147; `acceptance.final_field_err_pct`=-1.36 | 见备注 |
| stl_cad_voxelization | wigley_hydrostatics | `convergence.max_abs_err_pct`=2.25; `convergence.max_span_pct`=1.02 | 见备注 |
| laplace_law_surface_tension | laplace_droplet | `dims.2d.intercept_frac_pct`=0.6323; `dims.2d.max_dev_sigma_pct`=0.8028 | 见备注 |
| rans_ke | suboff_re1000 | `acceptance.primary_err_pct`=-1.147; `acceptance.final_field_err_pct`=-1.36 | 见备注 |
| ship_resistance_hull | suboff_re1000 | `acceptance.primary_err_pct`=-1.147; `acceptance.final_field_err_pct`=-1.36 | 见备注 |
| compressible_shock | compressible_shock_family | `normal_shock.max_ratio_error_riemann`=0.01122; `normal_shock.max_all_error`=0.01527 | 见备注 |

**锚定样例（owner 指定）`benchmarks/verified/sphere_re200`**：机器抽取
- `grids[0].cd` = 0.820646
- `grids[1].cd` = 0.807356
- `grids[1].err_pct` = 0.216
- `grids[2].cd` = 0.801137
- `grids[2].err_pct` = -0.556
- `grids[3].cd` = 0.798671
- `grids[3].err_pct` = -0.862
其中 Cd 相对误差 +0.216% → **<3% 成立**（本审计承认该样例为真实达标；其来源报告 README/REFERENCE_AUDIT 在本仓）。

**结论**：7 个 verified 模块中，标准算例级误差均在各自预注册容差内；但**上游同口径对照缺失**（无任何上游代码被复跑），因此「是否 <3%」只对**解析/文献基准**成立，对**上游实现**不可回答——这是本轮对标最大的证据空洞，已列入 T4/T5 任务。多相/自由表面/IBM 面按 owner 锚定证据判**不可接受（>15%）**：五模型液滴审计全败、RT 无增长窗口、FE 缺 Laplace/伪流速/界面剖面/长期稳定。

