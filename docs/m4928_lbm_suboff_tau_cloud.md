# m4928 — SUBOFF tau-topic CLOUD artefact（场量云图证据）

任务：为课题 `lbm-suboff-tau` 补齐 CLOUD 缺口（缺「场量云图 / 空间分布证据」）。
全部数值由 `benchmarks/lbm_suboff_tau_cloud_run.py` 的本次 run 产物程序化生成，
本文件的数字由脚本从 `artifacts/m4928/lbm-suboff-tau-cloud/summary_L32.json` 与 `field_cloud.json` 注入（无手抄）。

## 1. 案例与配置（引自 frozen verified 驱动）
- 几何：DARPA SUBOFF 裸艇体（回转体），L=4.356 m，D=0.508 m，L/D=8.575
- Re = 1000.0（u_phys=1.0e-03 m/s，nu_phys=4.356e-06 m²/s）
- 格子：D3Q19 MRT，half-way bounce-back，tau=0.5048，nu_lb=0.001600
- 网格级别：L=32 cells/艇长，域 192x68x68，步数 6000
- 几何自校验（本仓解析积分）：排水体积 277.2 lu³ = 0.6992 m³；
  湿面积 323.2 lu² = 5.9882 m²；棱形系数 0.7919；
  L/D 校验 8.575（参考 4.356/0.508 = 8.575 ✓）

## 2. 怎么做的
1. `benchmarks/lbm_suboff_tau_cloud_run.py` 直接 **import** `benchmarks/verified/suboff_sail_resistance/run.py`
   的 `SuboffAppendageEngine`（frozen 驱动不改），按同一配置（bare_hull / Re=1000 / MRT / pressure_extrap=none /
   p0=near_wall / friction=standard / 质量修正 200 步）跑步进链 `lbm_step_correct + far_field_bc_3d + correct_mass3d`，
   每 25 步采样探针，每 750 步存中心剖面快照。
2. 后处理（`station_tables`）：中心剖面（z/cz）上的壁面压力 Cp、近壁速度梯度估计的局部 Cf、
   边界层位移厚度 delta*、以及 21 个 x/L 站位剖面；外流场统计矩、进出口质量流、驻点 Cp。
3. `benchmarks/lbm_suboff_tau_cloud_plot.py` 生成图与表，并写机器可读 `field_cloud.json`（坐标/色标单位/色标范围/md5）。
   本 md 亦由该链路的 JSON 数值注入。

## 3. 结果（可溯源）
| 量 | 值 | 参考 | 误差 | U95 |
|---|---|---|---|---|
| 驻点 Cp | 0.9995 | 1.0（解析驻点） | -0.05% | 2.22e-04 |
| 远场 u_x/U_inf | 0.99959 | 1.0 | -0.041% | 8.24e-05 |
| 进出口质量失衡 | 0.022% | 0 | — | — |
| Cd_total（力积分窗口） | 1.225023 | 0.041995（Blasius） | +2817.1% | 窗口漂移 0.0017% |

**误差是否 <3% 的显式回答**：驻点 Cp（-0.05%）、远场来流
(-0.041%)、质量守恒 (0.022%) **<3% → PASS**；
**阻力/壁面摩擦 NOT PASS**：L=32 时艇体直径仅 2R≈5.6 格，
壁面量（Cf/Cd）在此分辨率下不可用（Cd 误差 +2817%），
阻力结论只能引用 verified 阶梯（L=96/112/128，erratum E2 最细层 +0.95%）。

## 4. 图与表
- `fig1_field_cloud.png`：中心剖面 u_x/U_inf、Cp、omega_z 云图 + x/L=0.5 横截面云图（坐标 x/L、y/D；色标单位见 field_cloud.json）
- `fig2_reference_comparison.png`：壁面 Cp(x/L)、局部 Cf(x/L) vs Blasius、x/L=0.5 剖面 vs Blasius 相似解、网格阶梯收敛
- `table1_grid_levels.csv`、`table2_stations.csv`、`table1_and_gci.md`、`field_cloud.json`

## 5. 局限（honest）
- 本窗口只产出 **1 个场量网格级（L=32）**：GCI 需 3 级 → 不适用，`field_cloud.json:gci.*.fallback_reason` 已写明。
- 该级艇体仅 ~5.6 格直径，壁面摩擦/总阻力不可用；云图与 Cp/剖面属外流场证据，可信度由驻点/远场/质量三项判据支撑。
- 裸艇体（无 sail/鳍）：回转体无周向变化，"逐角度级"表不适用（已在 field_cloud.json 注明）。
- 更高分辨率场量级（L=48/64）已排队为后续补测单：脚本已支持 `--resolution` 一次产出，无需改代码。
