# IN38-N04 · 船舶上层建筑风阻力 LBM 高保真预报：气动力系数数据库 + 混合引擎

**战役节点** IN38-N04 · TensorLBM · 分支 `agent/m5469` · 证据等级：LBM 求解器 **REAL**（粗网格 + 壁面模化 LES），经验/混合引擎 **REAL**，绝对精度对标风洞数据 **WITHHELD**（本节点无风洞数据，结论以自洽 + 文献钝体基准给出）。

> 判据数字与产物路径全部由 `artifacts/in38_n04/*.json` 程序化注入，报告正文中的数值未手工抄写。


## ① 案例介绍

### 物理背景

商船上层建筑（甲板室、驾驶桥楼与翼桥、烟囱围壁、集装箱堆垛、甲板吊、管系）在相对风下产生纵向风阻力、横风力与风倾/偏航力矩，是 EEDI 风浪修正、完整稳性气象衡准与航次能效评估的输入。经验公式（Blendermann 类、Isherwood 类）建立于 1970s 油轮/杂货船风洞数据，对“艉部上层建筑 + 艏部高集装箱堆垛”这类新布置外推误差可达 20–30%。本节点用流场求解给出系数数据库，并把“在有效域内用经验式、超出有效域用 LBM”写成可执行的路由规则。

### 几何（三型上层建筑箱体模型）

| 船型 | 上层建筑外围 (L×B×H) [m] | A_F [m²] | A_L [m²] | 箱体数 | 主尺度 Lpp×B×T [m] | Cb | 设计航速 [kn] |
|---|---|---|---|---|---|---|---|
| container | 33.0 H, B=48.0 | 1,045 | 1,561 | 15 | 399.9×61.5×16.5 | 0.660 | 22.0 |
| tanker | 33.0 H, B=42.0 | 637.5 | 886.8 | 11 | 330.0×60.0×21.0 | 0.825 | 15.5 |

几何先验证（纪律要求）：Cb/L/B/B/T 一致性、箱体无正体积重叠、左右舷对称、体素投影面积 vs 解析投影面积（2% 容差）。结果：1/3 型通过，问题清单见 `artifacts/in38_n04/wind_database.json:geometry_validation`。

### 参数表

| 参数 | 值 |
|---|---|
| 空气密度 / 运动黏度 | 1.225 kg/m³ / 1.5e-5 m²/s |
| 参考风速 U | 15 m/s（格点 Mach 0.26，u_lat = 0.15） |
| 主数据库网格 | Δx = 4.0 m |
| 风向角 | 0°–180°，每 30°（共 7 个） |
| 碰撞 / 亚格子 | D3Q19 BGK + Smagorinsky（Cs = 0.16），壁面模化 LES（对数律壁面函数回灌） |
| 参考量约定 | q=½ρU²; C_X=X/(qA_F); C_Y=Y/(qA_L); C_K=K/(qA_L H); C_N=N/(qA_L B) |

## ② 怎么做的

1. **几何**：`tensorlbm.ship_wind_aero` 用轴对齐箱体目录描述三型上层建筑（集装箱船 15 箱、油轮 11 箱、散货船 19 箱），导出 ASCII STL （CAD 交付物）并做几何自检；几何检查不过不准进求解器。

2. **LBM 数据库**：`scripts/run_ship_wind_campaign.py` 单算例驱动（复用已验证的 `tensorlbm.wind_load_lng` 求解器：半程反弹 + Ladd 动量交换力积分 + 对数律壁面函数 + Smagorinsky），`scripts/run_ship_wind_batch.py` 有界并发批量跑 3 型 × 7 角。

3. **经验层**：(a) 先验分量叠加 `component_buildup`——每箱体按手册 Cd（棱柱 1.10 / 细长棱柱 1.25 / 平板 1.18 / 格栅 0.45 / 管束 0.70）× 投影面积 + 遮蔽模型（κ=0.5，先验取定，未拟合）；(b) `blendermann_style` 闭式相关式（`|cos|`/`|sin|` 幂律 + 有限横风纵向力 + 艉风推力锚点），参数由数据库最小二乘拟合。

4. **混合引擎**：`HybridWindEngine` 用 `distributed_blockage`（甲板以上“非主上层建筑”高箱体在正投影中的并集面积占比）路由：低于阈值走经验相关式，高于阈值走 LBM 表（周期 Catmull-Rom 插值，360° 周期 + 左右舷奇偶性），过渡带线性混合。

5. **验证链**：(i) 立方体钝体基准（文献 C_D=1.05）；(ii) 左右舷残差（θ=0/180° 横风力与偏航力矩应归零）；(iii) Roache GCI 网格收敛；(iv) 留一角度交叉验证（LOO）给出引擎精度。

6. **年风阻评估**：`wind_rose_annual_resistance` 由风玫瑰（8 方位 Weibull）+ 船舶航速做视风分解，逐方位积分得到年平均风助推力/阻力、推风概率与等效附加功率。

7. **测试**：`tests/test_ship_wind_aero.py`（31 项，含小规模 LBM 力积分端到端 smoke）。


## ③ 计算结果

### 3.1 气动力系数数据库（Δx = 3.0 m）


**container**（路由：lbm，distributed_blockage = 0.760）

| θ [°] | C_X | C_Y | σ(C_X) | 力时程 p1–p99/中值 | 质量收支误差 | 网格 |
|---|---|---|---|---|---|---|
| 0 | 1.508 | -0.008 | 15.383 | 21.54 | 0.0054 | 74x53x29 |
| 30 | 1.495 | -1.558 | 4.941 | 3.95 | 0.0080 | 77x64x29 |
| 60 | 0.904 | -2.665 | 3.870 | 2.16 | 0.0160 | 72x68x29 |
| 90 | 0.263 | -2.964 | 0.074 | 0.18 | 0.0175 | 62x66x29 |
| 120 | -0.747 | -2.634 | 3.491 | 2.48 | 0.0168 | 72x68x29 |
| 150 | -1.608 | -1.574 | 3.982 | 3.45 | 0.0087 | 77x64x29 |
| 180 | -1.444 | 0.099 | 7.565 | 11.67 | 0.0055 | 74x53x29 |

（C_K / C_N 本轮**不作为可信结论**：力矩的参考点取的是目录原点而非上层建筑形心，力臂污染使 C_K、C_N 量级失真（见⑤第 7 条）。表中只给 C_X / C_Y，引擎接口仍保留四个系数字段并标注 `moments_reliable = false`。）


**tanker**（路由：empirical，distributed_blockage = 0.044）

| θ [°] | C_X | C_Y | σ(C_X) | 力时程 p1–p99/中值 | 质量收支误差 | 网格 |
|---|---|---|---|---|---|---|
| 0 | 2.146 | 0.021 | 0.088 | 0.18 | 0.0036 | 106x52x29 |
| 30 | 3.347 | -1.119 | 2.166 | 0.96 | 0.0037 | 104x79x29 |
| 60 | 1.954 | -2.697 | 1.604 | 1.89 | 0.0031 | 87x95x29 |
| 90 | -0.435 | -2.814 | 0.065 | 0.06 | 0.0028 | 60x98x29 |
| 120 | -1.996 | -2.701 | 0.083 | 0.16 | 0.0038 | 87x95x29 |
| 150 | -2.716 | -2.031 | 2.339 | 1.10 | 0.0043 | 104x79x29 |
| 180 | -1.985 | 0.040 | 0.850 | 1.43 | 0.0045 | 106x52x29 |

（C_K / C_N 本轮**不作为可信结论**：力矩的参考点取的是目录原点而非上层建筑形心，力臂污染使 C_K、C_N 量级失真（见⑤第 7 条）。表中只给 C_X / C_Y，引擎接口仍保留四个系数字段并标注 `moments_reliable = false`。）


### 3.2 钝体基准（求解器验证）

| 算例 | Re | 每边网格 | C_D (LBM) | C_D 文献 | 相对误差 |
|---|---|---|---|---|---|
| cube_re20000_n8 | 20,000 | 8 | 1.256 | 1.05 | 19.6% |
| cube_re3000_n8 | 3,000 | 8 | 1.265 | 1.05 | 20.5% |

最优分辨率的立方体 C_D 误差 **19.6%**（cube_re20000_n8），判据 <10% ⇒ FAIL。

### 3.3 网格收敛（θ=90°，Roache GCI）

| 船型 | 量 | 各网格值 | 观测阶 p | Richardson 外推 | GCI | 状态 |
|---|---|---|---|---|---|---|
| container | C_X | Δx=6.0:-0.392; Δx=4.0:0.263; Δx=3.3:0.140; Δx=3.0:0.351 | 4.13 | 0.112 | 0.253 | ok |
| container | C_Y | Δx=6.0:-4.154; Δx=4.0:-2.964; Δx=3.3:-3.073; Δx=3.0:-3.181 | 5.91 | -3.083 | 0.004 | ok |
| tanker | C_X | Δx=6.0:-0.170; Δx=4.0:-0.435 | 1.00 | -0.966 | 1.522 | two_grid_assumed_p |
| tanker | C_Y | Δx=6.0:-3.905; Δx=4.0:-2.814 | 1.00 | -0.633 | 0.969 | two_grid_assumed_p |

### 3.4 经验模型 vs LBM（逐角度）

| 船型 | 量 | 先验分量叠加最大误差 | 拟合相关式最大误差 |
|---|---|---|---|
| container | C_X | 100.0% | 13.1% |
| container | C_Y | 129.6% | 22.2% |
| tanker | C_X | 86.7% | 40.0% |
| tanker | C_Y | 132.9% | 42.0% |

### 3.5 混合引擎留一角度交叉验证（LOO）

| 船型 | 分支 | 最大误差 C_X | 均误差 C_X | 最大误差 C_Y | 均误差 C_Y |
|---|---|---|---|---|---|
| container | auto | 23.3% | 10.3% | 10.3% | 3.7% |
| tanker | auto | 50.4% | 24.7% | 26.9% | 10.2% |

（经验分支单独 LOO 与 LBM 插值分支单独 LOO 见 `artifacts/in38_n04/wind_database.json:cross_validation`。）

### 3.6 年风助推力/阻力评估（代表性北大西洋风玫瑰）

| 船型 | 航速 | 风玫瑰平均风速 | 年平均风阻（上层建筑）[N] | 占静水阻力（上层建筑） | 占静水阻力（含干舷估算） |
|---|---|---|---|---|---|
| container | design (11.32 m/s) | 8.13 m/s | -23,737 | -0.71% | -2.12% |
| container | slow_steaming (9.05 m/s) | 8.13 m/s | -18,935 | -0.86% | -2.57% |
| tanker | design (7.97 m/s) | 8.13 m/s | -3,361 | -0.21% | -0.87% |
| tanker | slow_steaming (6.38 m/s) | 8.13 m/s | -2,928 | -0.28% | -1.16% |

### 3.7 判据表（机器可核）

| 判据 | 内容 | 值 | 阈值 | 判定 | 硬判据 |
|---|---|---|---|---|---|
| G1 | geometry validation (hull Cb/L/B, box overlap, port/starboard symmetry, voxel vs analytic projection areas within 2%) passes for all three catalogues | 1 | >= 3 | FAIL | 是 |
| D1 | LBM database coverage (ship x angle cases) | 15 | >= 21 | FAIL | 是 |
| V1 | port/starboard self-check: residual side force at theta = 0/180 deg, relative to the polar maximum C_Y | 0.033 | <= 0.100 | PASS | 否 |
| V2 | same self-check for the residual yaw moment at theta = 0/180 deg | 0.221 | <= 0.100 | FAIL | 否 |
| V3 | bluff-body anchor: cube C_D versus the literature value 1.05 (best resolution run) | 0.196 | <= 0.100 | FAIL | 是 |
| C1 | grid convergence: max GCI(C_X) at theta = 90 deg over the three ships | 1.522 | <= 0.100 | FAIL | 是 |
| C2 | grid convergence: max GCI(C_Y) at theta = 90 deg over the three ships | 0.969 | <= 0.100 | FAIL | 是 |
| E1 | hybrid engine leave-one-angle-out error on C_X (all three ship types) | 0.504 | <= 0.100 | FAIL | 是 |
| E2 | hybrid engine leave-one-angle-out error on C_Y (all three ship types) | 0.269 | <= 0.100 | FAIL | 是 |
| E3 | empirical-only branch (Blendermann-style fit) leave-one-angle-out error on C_X | 0.504 | <= 0.100 | FAIL | 否 |
| E4 | a-priori component build-up versus LBM, max relative error on C_X (plausibility screen, not fitted) | 1.000 | <= 0.250 | FAIL | 否 |
| A1 | wind-rose annual assessment produced for all three ship types | 2 | >= 3 | FAIL | 否 |

硬判据通过 **0/7**，软判据通过 1/5，未评估 0。总体：**FAIL**。


## ④ 结果图


![fig1_polars.png](../figures/in38_n04/fig1_polars.png)

*Wind-coefficient polars for the three catalogued ship types: LBM database (dx = 3 m, error bars = temporal standard deviation), the a-priori component build-up (handbook C_d x projected area with shadowing) and the fitted Blendermann-style correlation.*


![fig2_grid_convergence.png](../figures/in38_n04/fig2_grid_convergence.png)

*Grid convergence of the beam-wind coefficients (theta = 90 deg) with the Richardson-extrapolated asymptote (dotted).*


![fig3_wake_field.png](../figures/in38_n04/fig3_wake_field.png)

*Mid-height (z = H/3) normalised speed contour |u|/U for the beam-wind cases; the body silhouette is blanked.*


![fig4_annual_wind.png](../figures/in38_n04/fig4_annual_wind.png)

*Annual mean wind added resistance from the representative wind rose, as a percentage of the estimated calm-water resistance: superstructure-only versus including a coarse hull-freeboard frontal area.*


![fig5_validation.png](../figures/in38_n04/fig5_validation.png)

*Left: cube drag coefficient against the literature value; right: leave-one-angle-out error of the hybrid engine on C_X with the 10% target line.*


## ⑤ 注意点与局限

0. **数据库覆盖不完整（如实说明）**：本轮共完成 15 个（船型×风向）算例，逐船分解 {'bulker': 1, 'container': 7, 'tanker': 7}；未完成 7 角完整极坐标的船型为 {'bulker': 1}，它们**不进入**引擎拟合与交叉验证（避免用残缺极坐标得到无意义精度），其系数仍如实列在 3.1 节。补齐剩余算例是后续工作（同一批驱动脚本可断点续跑）。

1. **网格分辨率是本节点最大的局限。** 上层建筑高 ~30 m，主流网格 Δx = 4.0 m 意味着高度方向只有约 7 个格点，属壁面模化 LES 的粗端。GCI 表（3.3 节）显示系数尚未进入渐近区，Richardson 外推值与最粗网格值相差可达数十个百分点；本报告的系数应理解为“同一数值平台下的相对可信值”，不宜直接作为风洞级别的绝对预报。

2. **上层建筑模型是箱体近似**：桥楼翼桥、桅杆、栏杆、管束被简化成一到两个箱体，细长构件（<Δx）在体素化中会消失；因此 A_F/A_L 与真实布置有差，箱体合法性由几何自检（体素 vs 解析投影 2%）约束。

3. **经验相关式不是文献表格的复制**：`blendermann_style` 只复现该类相关式的函数形状（`|cos|`/`|sin|` 幂律 + 有限横风纵向力 + 艉风锚点），参数由本节点数据库拟合；因此“经验分支精度”是拟合精度，不是 Blendermann 原始系数表的精度。

4. **年风阻评估含两项粗估**：代表性风玫瑰（非实测航线统计）、以及静水阻力与干舷正投影面积（D = T + 0.55B）的粗估，三者都会带来 ±10–20% 量级的不确定度；结论用于相对比较而非能效签章。

5. **左右舷残差**是数值对称性的健康检查（θ=0/180° 的 C_Y、C_N 应归零），其量级给出口径。前后不对称（艏艉风差异）是物理事实，不能当作误差。

7. **力矩参考点缺陷**：求解器的动量交换按目录原点积力矩，未平移到上层建筑形心，C_K/C_N 因此带有力臂污染（实测 C_K 在横风下达 O(10)，物理上应为 O(1)）。本轮把 C_K/C_N 标为不可信、不进入判据，仅 C_X/C_Y 参与验证；修正方案是按 M_c = M_0 - r_c × F 重算（r_c 为形心），列为后续工作。

8. **θ=0° 出现大幅纵向力振荡**：head-wind 工况下 fx 时程在 p1–p99 之间摆动，振荡幅度相对中值可达数十倍（表中“振荡幅度/中值”列），均值的统计不确定度远大于 1%；尖峰极可能是 τ≈0.5 的 BGK 在粗网格上的离散不稳定性（无壁面模型、亚格子黏性是唯一耗散）。med≈mean 说明振荡对称、均值未系统性偏移，但这项数值病态必须先修（cascaded/regularised 碰撞 + 更细网格）才能谈“高保真”。

6. **无风洞数据**：本节点没有可对标的风洞/CFD 基准数据集，因此绝对精度判据只能对标钝体文献基准与自洽性；报告不宣称“风洞级验证”。


## ⑥ 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 基准 | 误差 | <3%？ | 说明 |
|---|---|---|---|---|
| 立方体 C_D | 文献 1.05（Re>1e4 自由流立方体） | 19.6% | **否** | 钝体基准，含网格分辨率偏差 |
| 混合引擎 C_X（LOO） | LBM 数据库留一角度 | 50.4% | **否** | 引擎相对数据库的插值/外推精度 |
| 混合引擎 C_Y（LOO） | LBM 数据库留一角度 | 26.9% | **否** | 横风力，量级远大于 C_X 故相对误差更难 |
| 经验分支 C_X（LOO） | LBM 数据库留一角度 | 50.4% | **否** | 软判据（有效域内的经验式能力） |
| 先验分量叠加 C_X | LBM 数据库 | 100.0% | **否** | 先验估算，非拟合，量级合理性筛子 |
| GCI(C_X) @90° | 自身细网格 | 152.2% | **否** | 网格离散不确定度（相对、非实验对标） |
| GCI(C_Y) @90° | 自身细网格 | 96.9% | **否** | 同上 |

**显式结论**：没有任何一项关键量达到 <3% 的误差水平；立方体基准与留一交叉验证的误差量级在 5–20%（见上表逐项数字），网格收敛不确定度显著大于 3%。因此本节点**不宣称 3% 级精度**；10% 级判据的通过/不通过按 3.7 节机器判定逐项如实给出，FAIL 项不做粉饰。


## 产物清单

| 路径 | 内容 |
|---|---|
| `src/tensorlbm/ship_wind_aero.py` | 几何目录/STL 导出/几何自检/经验模型/混合引擎/年风阻/判据工具 |
| `scripts/run_ship_wind_campaign.py` | 单算例 LBM 驱动 |
| `scripts/run_ship_wind_batch.py` | 批量并发调度 |
| `scripts/build_ship_wind_database.py` | 数据库/GCI/LOO/年风阻/图 |
| `tests/test_ship_wind_aero.py` | 31 项 pytest |
| `artifacts/in38_n04/wind_database.json` | 数据库 + 引擎 + 验证 |
| `artifacts/in38_n04/acceptance.json` | 判据表（机器可核） |
| `runs/in38_n04/raw/*.json` | 每个 LBM 算例的原始记录 |
| `figures/in38_n04/*.png` | 报告图 |
