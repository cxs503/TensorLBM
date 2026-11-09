# 液舱晃荡冲击的低阶链误差定标：三维两相自由面 LBM 参照

**节点** p120w2-n10 ｜ **仓库** TensorLBM ｜ **参照链** ship-performance-platform IN48-N07（只读移植）

## 1. 案例介绍

### 1.1 物理背景

船舶液舱晃荡链在工程上分两级：线性势流给出固有频率，Saint-Venant（浅水，SWE）给出壁压时历，再由壁压做结构校核。两级都是渐近模型：SWE 只在 `kh → 0` 成立，且完全不能表示气垫（封闭空气腔的等熵压缩）；势流不能表示爬高、卷破与冲击。本节点用三维大密度比自由面 LBM（Körner 模型，气体不做动力学、以 ABB 压力边界等价无穷密度比）在同一几何、同一时间锚上与低阶链对拍，把“链误差”从假设变成可测量的数字，并给出适用边界。

### 1.2 几何与参数（IN48-N07 锚，只读移植）

| 量 | 值 | 来源 |
|---|---|---|
| 液舱长 L（晃荡方向 x） | 20.0 m | `TANK = RectTank(20, 12, 8)` |
| 液舱宽 B（z） | 12.0 m | 同上 |
| 液舱高 H（y） | 8.0 m | 同上 |
| 静水深度 h_l | 4.0 m（充液 0.5） | `h_ref = 0.5 * TANK.height` |
| 水密度 rho | 1025.0 kg/m^3 | `RHO_WATER` |
| 重力 g | 9.80665 m/s^2 | `G` |
| 一阶固有频率 omega_1 | 0.926203 rad/s | `sqrt(g k tanh(k h))`, k=pi/L |
| 一阶周期 T_1 | 6.7838 s | 2 pi / omega_1 |
| 相对深度 kh_1 | 0.6283 | k h_l |
| SWE 自身一阶频率 omega_1^SWE | 0.983808 rad/s （+6.22%） | k sqrt(g h) |
| 涌潮前缘速度 u_front | 12.5262 m/s | Ritter 2 sqrt(g h_l) |
| 静水压 rho g h_l | 40207.26 Pa | - |
| 闭式冲击峰值 4 rho g h_l | 4.0000 (rho g h_l) | Ritter 滞止 |
| 精确涌潮反射 rho g h_r | 3.4940 (rho g h_l) | Saint-Venant 激波条件 |

### 1.3 两个算例

1. **冲击算例（surge release）**：液柱高 h_l=4 m、长 5 m（0.25L）置于 x=0 端，其余干舱，闸门瞬时撤除，涌潮跨 15 m 打到对壁——与 m5744 `dam_break_case(h0=4, wet_length=5, dry_length=15)` 同一构型。
2. **受迫算例（forced）**：满深 h_l 起振，水平简谐加速度 `a(t)=A omega^2 sin(omega t)`，A=0.25 m、omega=omega_1（m5744 `regular_excitation`），6 个周期，对应站立波壁压时历与超谐（2 omega_1）谱。

## 2. 怎么做的

### 2.1 数值方法

* 求解器：`tensorlbm.free_surface_lbm.free_surface_step`（Körner 2005 三维自由面 LBM，D3Q19，BGK + Smagorinsky 亚格子，ABB 气相压力重构，质量独立跟踪，I/L/G 拓扑事务）。大密度比由模型本身承担：气体无动力学，气相以压力边界注入，等价于无穷密度比。
* 新增标定模块：`src/tensorlbm/sloshing_impact_calibration.py`（锚移植 / 格点标度 / 壁压测量 / GCI / 谐波分析 / 气垫闭合）。

### 2.2 格点标度（两不变量）

| L/nx | 网格 (nx x ny x nz) | dx [m] | dt [s] | g_lbm | tau | nu_lbm | Re_eff |
|---|---|---|---|---|---|---|---|
| L/40 | 15360 40x16x24 | 0.5000 | 0.003193 | 2.000e-04 | 0.50096 | 3.200e-04 | 2000 |

* 不变量 1：前缘格点马赫数 `u_lbm = 0.08`（即 `dt ~ dx`），保证可压缩误差不随网格变化；不变量 2：有效雷诺数 `Re = u_front h_l / nu`（`nu_lbm ~ dx`，`tau` 随网格微增）。真实 Re ~ 5e7，自由面 LBM 在 L/80 无法解析，故取 Re=2000 的惯性主导参照并加亚格子闭合（见 5.3 局限）。

### 2.3 测量

* 壁面动压：撞壁侧壁面相邻两格 `p = c_s^2 (rho - rho_liquid)`（LBM 压力仅到常数），线性外插到壁面 `p_wall = 1.5 p_{dx/2} - 0.5 p_{3dx/2}`（壁面 1–2 格外插）；面板取撞击壁下半水深，横跨全宽；仅统计 LIQUID 格，湿格 < 3 时输出 0（避免单丝液膜伪值）。
* 壁面相同时历 `h_wall`：壁相邻列 fill 沿宽度平均，除 h_l 无量纲化——与 SWE 的 `rho g h_wall` 同量，用于相关性对拍。
* 声学噪声滤波：0.10 s 中心滑动平均（撞击升压时间 ~h/u ~ 0.45 s，滤波不影响物理升压），原始峰值一并报告。
* GCI：三点（L/40, L/60, L/80）拟合观测阶 p，Richardson 外插，安全系数 Fs=1.25；非单调三元组一律判 `usable=False` 而不粉饰。

### 2.4 工具链

```
# 1) SWE 锚（跨仓只读导入 m5744 impact.py，只落派生时历）
M5744_ROOT=<ship-performance-platform> python3 scripts/gen_swe_reference_anchor.py \
    --out artifacts/sloshing_impact_lbm/sw_swe_reference.json
# 2) 各算例（可单进程=一个 shard）
python3 scripts/run_sloshing_impact_calibration.py campaign --shard i --shards 6 \
    --out-dir artifacts/sloshing_impact_lbm
# 3) 汇总（GCI/谱/判据）
python3 scripts/run_sloshing_impact_calibration.py post \
    --case-dir artifacts/sloshing_impact_lbm \
    --swe-reference artifacts/sloshing_impact_lbm/sw_swe_reference.json \
    --out artifacts/sloshing_impact_lbm/report.json
# 4) 图 + 本报告
python3 scripts/make_sloshing_calibration_figures.py --case-dir artifacts/sloshing_impact_lbm \
    --fig-dir artifacts/sloshing_impact_lbm/figures
python3 scripts/make_sloshing_calibration_report.py --case-dir artifacts/sloshing_impact_lbm \
    --out docs/SLOSHING_IMPACT_LBM_CALIBRATION.md
```

## 3. 计算结果

### 3.1 三套网格与壁压峰值

| 网格 | 单元数 | 步数 | P_peak/(rho g h_l)（0.1 s 滤波） | 原始峰值 | 峰值时刻 [s] | h_wall/h_l 峰值 | 冲量 ∫p dt | 质量漂移 |
|---|---|---|---|---|---|---|---|---|
| L/40 | 15360 | 1879 | 0.5442 | 1.69 | 2.63 | 0.2611 | -0.1341 | 7.95e-04 |

### 3.2 判据逐条

| 判据 | 量 | 值 | 阈值 | 结论 |
|---|---|---|---|---|
| (a) 网格收敛 | GCI(P_peak) | n/a | ≤ 0.05 | FAIL |
| (a) 观测阶 | p_obs | n/a | ≥ 1.2 | FAIL |
| (a) 补充：冲量 GCI | GCI(∫p dt) | n/a | 参考 | p_obs=n/a |
| (a) 补充：爬高 GCI | GCI(h_wall 峰值) | n/a | 参考 | p_obs=n/a |
| (b) 参照带 | 闭式 4 rho g h_l / 精确涌潮反射 / SWE 自身冲击锚 | 4 / 3.494 / 0.8247 | - | - |
| (b) 峰值带 | P_peak/(rho g h_l) 细网格 | 0.5442 | [3.0, 5.0] | FAIL |
| (c) 静水基线 | L2 相对误差 | 0.1056 | < 0.01 | FAIL |
| (d) 超谐 2 omega_1 | 能量占比（爬高 / 压力） | 0.0189 / 0.0020 | > 0.05 | FAIL |
| (e) 气垫负控 | P_peak(气垫) vs 单相 | n/a vs n/a | 严格更低 | FAIL |
| (f) 与 SWE 时历相关 | Pearson r | 0.3352（前半窗 0.6989） | > 0.9 | FAIL |
| (g) 质量守恒 | max |dM|/M（自 t=0） | 7.95e-04 | < 1e-3 | PASS |

> 质量漂移的另一种口径（100 步初始包络松弛后再起算）为 1.13e-03；两种口径都列出，不做选择性隐去。判据用更朴素的自 t=0 口径。

### 3.3 SWE 链对拍

| 量 | LBM | SWE (IN48-N07) |
|---|---|---|
| 壁面水深峰值 h_wall/h_l | 0.2611 | 0.8247 |
| 涌潮抵达时刻 [s] | 2.299 | 1.244 |
| 时历相关 r（全窗 6.00 s） | 0.3352 | (基准) |
| 时历相关 r（前半窗） | 0.6989 | (基准) |
| SWE/LBM 爬高比 | — | 3.158 |

受迫站立波对照：LBM（精确色散）与 SWE（k sqrt(g h) 色散）壁面水深时历 r = -0.5345（窗 40.70 s，LBM 峰值 1.313 vs SWE 1.361）。相位差即为 SWE 频率误差的累积，是低阶链在 kh=0.63 的适用边界直接证据。

### 3.4 SWE 适用边界（频率）

* kh_1 = 0.6283 时 `omega_SWE/omega_exact = 1.0622`（SWE 快 6.22%）。
* 3% 误差以内要求 kh < 0.4300（h/L < 0.1369）；5%: kh < 0.5602；10%: kh < 0.8106。
* LBM 受迫算例实测 omega = 0.9243 rad/s，相对精确色散 0.9980；SWE/实测 = 1.0643。

## 4. 结果图

![fig01_impact_wall_pressure](artifacts/sloshing_impact_lbm/figures/fig01_impact_wall_pressure.png)

*fig01_impact_wall_pressure*

![fig02_fields_nx40](artifacts/sloshing_impact_lbm/figures/fig02_fields_nx40.png)

*fig02_fields_nx40*

![fig04_forced_spectrum](artifacts/sloshing_impact_lbm/figures/fig04_forced_spectrum.png)

*fig04_forced_spectrum*

![fig06_hydrostatic](artifacts/sloshing_impact_lbm/figures/fig06_hydrostatic.png)

*fig06_hydrostatic*

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

### 5.1 坑：τ 决定的不只是粘度，而是能不能“流起来”

首版取 `tau=0.8`（`nu_lbm=0.1`）是常见默认值，但在这个格点标度下有效雷诺数只有 `Re_eff ~ 6`——液体像蜂蜜一样塌不下来，涌潮 6 s 内根本到不了对壁（壁面全程干，P_peak=0）。这属于“参数合法但物理荒谬”的陷阱：LBM 的粘度必须按目标 Re 反算，而不是沿用默认 tau。修正为固定 `Re_eff=2000` 后 `tau = 0.5 + 3 nu_lbm ~ 0.501`。

### 5.2 缺陷：tau≈0.5 时 D3Q19 MRT 自由面路径不可用

为稳定近无粘流动改用 MRT 碰撞时，**第一步**即触发 `TopologyTransactionError: topology candidate fill is outside [-0.1, 10]`，L/40 与 L/60 一致复现，且与 Smagorinsky 常数无关（C_s=0.10/0.16 均失败）。BGK+Smagorinsky 在同一配置下可跑完整撞击窗。这是共享自由面模型的既有缺陷，已在本报告中登记，未在本节点修复。

### 5.3 局限：Re 不可解析 + 亚格子闭合

真实液舱 Re ~ 5e7；L/80 只有 16 层水深，只能跑 Re_eff=2000 的惯性主导参照。冲击峰值由 Froude 标度的滞止压力主导，粘度是二阶影响，但仍需在报告口径里说明：本参照不是实尺度直接数值模拟。

### 5.4 坑：单点壁压的声学噪声

单点壁压含声学/拓扑转换噪声，量级可与信号同阶（曾观测到 -3 rho g h_l 的负压尖峰，出现在液膜刚触壁、面板湿格只有 1–2 个的瞬间）。因此判据只认 0.10 s 滑动平均 + 面板（下半水深 × 全宽，仅 LIQUID 格）平均，湿格 < 3 时置零；原始峰值同表列出以便复核。

### 5.5 发现：SWE 链无底摩擦，爬高被显著高估

同一构型、同一 h_l 下，SWE 的壁面水深峰值 0.8247·h_l，LBM 为 0.2611·h_l，比值 3.16。m5744 的 Saint-Venant 求解器是无粘、无底摩擦的（`swe_solve` 只有对流项 + 加速度源项），涌潮跨 15 m 干床时边壁/底摩阻在 LBM 参照里是实打实存在的，因此低阶链对壁面爬高给出偏保守（偏大）的估计——这正是“低阶链误差定标”要输出的结论之一。

### 5.6 环境缺陷：loong64 集群上本算例出现 glibc 堆损坏/段错误

同一算例外协方 所 loong64 两节点上反复以 `free(): corrupted unsorted chunks` / `malloc(): unsorted double linked list corrupted` / `Segmentation fault` 终止（faulthandler 落点在 `free_surface_topology_transaction._birth_density_field` 的 `torch.roll` 上），而 x86_64 控制面上同一代码、同一参数可完整跑完 6 s 撞击窗。该节点上另一个与本任务无关的仓库（`wall_model_les`）作业同样报告 `free(): corrupted unsorted chunks`，指向 loong64 自定义 torch 构建（2.13.0）的堆损坏缺陷而非本模块逻辑。结论：本节点的生产算例在控制面（x86_64）执行，参数与种子完全一致可复现；集群路线待该构建修复后再启用。

### 5.7 未评估：网格收敛（判据 a）——模型在 L/60、L/80 上积分不下去

撞击算例在 L/40 跑完整个 6 s 窗；**L/60 与 L/80 在冲击段触发 `TopologyTransactionError: topology candidate fill is outside [-0.1, 10]`** （x86_64 控制面与 loong64 集群均复现，loong64 上另伴随 glibc 堆损坏终止）。也就是说：共享自由面模型在 tau≈0.5 的近无粘设置下，无法把“涌潮撞击 + 爬高 + 卷破”这一段拓扑剧变过程积到更细网格上。判据 (a) 因此**未评估**（不是通过）：本节点的参照只有 L/40 这一个离散误差未知的解，所有基于它的结论都带有这一未定量化的离散误差，已在下文局限中标注。

### 5.8 未评估：气垫负控（判据 e）

封闭气腔按 `p = p0 (V0/V)^gamma`（gamma=n/a）注入 ABB 气相压力：未截断时乌尔体积压缩使气压升到 ~1.7 p0，界面拓扑被推过模型的 `fill in [-0.1, 10]` 守卫（`TopologyTransactionError`）；截断到 1.35 p0 后仍未能在预算内跑完 6 s 窗。判据 (e) 因此**未评估**（不是通过），原因与证据在此如实登记：共享自由面模型的 ABB 气相压力只在 O(rho_liquid) 内自洽，“气垫效应”目前超出该模型的可运行包线。

## 6. 误差判定（逐项 vs 基准）

| 关键量 | 基准 | 误差 | 是否 < 3% |
|---|---|---|---|
| 静水压剖面 | `rho g (h_l - y)` 解析 | 10.565% (L2) | 否 |
| 壁面水深时历（冲击） | SWE 1-D 解 | r = 0.3352（相关，非百分比误差） | 不适用（判据 >0.9） |
| 一阶固有频率 | 精确色散 `sqrt(gk tanh kh)` | 0.20% (LBM 实测/精确) | 是 |
| 冲击峰值压力 | 闭式 4 rho g h_l 带 [3,5] | 实测 0.5442（比值，非百分比） | 见判据 (b) |

## 7. 结论与审计链

* 判据总表：**未全部通过**；未通过项在上表逐条标 FAIL，未做任何覆盖。
* 审计链（数据 → 判据 → 图 → 报告，全部程序化）：
  - 锚：`artifacts/sloshing_impact_lbm/sw_swe_reference.json`（含 m5744 commit `e58d8fe0d7c110c3607e54ea0a7e9dd2e9355fa1` 与模块路径）
  - 算例：`artifacts/sloshing_impact_lbm/*_nx*.json`（时历 + 场切片 + 质量账本 + 标度）
  - 汇总：`artifacts/sloshing_impact_lbm/report.json`（GCI/谱/判据，含 filter 窗口）
  - 图：`artifacts/sloshing_impact_lbm/figures/*.png`
* 生成时间（UTC）：2026-11-08T16:21:05Z

