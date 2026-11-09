# P92W3-N02 · TensorLBM 热辐射验证面：角系数闭式族 + 灰体封闭腔辐射度 + 太阳/天空边界 → 船舶甲板热载荷

**节点**：P92W3-N02（TensorLBM，物理域，证据等级目标 REAL）
**分支**：`agent/m8447`　**机器判据产物**：`artifacts/P92W3_N02_radiation_verification.json`
**基准脚本**：`benchmarks/radiation_closed_form_deck/run.py`（`python benchmarks/radiation_closed_form_deck/run.py` 可逐位复现）

## 摘要

本次把 TensorLBM 的热辐射能力从"有代码零验证"推到 REAL：

| 判据 | 要求 | 实测 | 结论 |
|---|---|---|---|
| (a) 六类角系数闭式误差 | ≤ 1 % | 最差 **0.00279**（蒙特卡洛统计极限） | PASS |
| (b) 封闭腔辐射度能量守恒残差 | ≤ 1e-6 | 最差 **4.54e-14** | PASS |
| (c) 甲板热载荷 vs 解析解 | ≤ 5 % | 最差 **9.68e-05** | PASS |

全部判据通过（`all_pass = True`），且每条判据都对着**独立于被测实现**的基准：
(a) 对多维高斯-勒让德求积或蒙特卡洛光线追踪（定义级估计量）；(b) 对串联热阻/辐射网络解析解；
(c) 对闭式解与独立 300 步二分参考解。

## 1 案例介绍

### 1.1 物理背景

船舶总体性能中热辐射进入三条链路：①上层建筑/舱室的灰体辐射换热；②甲板对太阳短波与天空长波的净交换（夜间天空辐射冷却是舰船红外特征与甲板结露的直接原因）；③热 LBM/导热求解所需壁面 Neumann 热流。原实现中角系数有 5 个函数、3 个靠数值求积，`benchmarks/verified/radiation_enclosure_ship` 的角系数是**手填 0.4/0.6 矩阵再归一化**，其自报 `energy_conservation_residual = 1.0`（100 % 不守恒，因为把太阳吸收项塞进了照射量 G）——即"有代码、零验证"。

### 1.2 几何与参数

| 案例 | 几何 | 参数 |
|---|---|---|
| 角系数族 1 | 平行正对等矩形 | X=a/c, Y=b/c ∈ [0.5, 5] |
| 族 2 | 共边垂直矩形 | H=a/L, W=c/L（**均对公共边长归一化**），∈ [0.2, 2] |
| 族 3 | 同轴平行圆盘 | R1,R2,S：等径/不等径，R1=R2=S 等 |
| 族 4 | 同心球 | r1/r2 = 0.5, 0.25 |
| 族 5 | 微元→同轴圆盘 | R/H ∈ [0.1, 8] |
| 族 6 | 天穹→倾斜平面 | β = 0…180° |
| 封闭腔 | 2 面平行板 / 2 面同心球 / 6 面立方体腔（3×3 温-ε 扫描）/ 3 面含再辐射面 / 甲板-天空开放腔 | T ∈ [300, 800] K，ε ∈ [0.05, 1.0] |
| 甲板 | 12 m × 8 m 甲板 + 3 m 高上层建筑块（x∈[3,7], y∈[1,4]） | ε=0.9, α=0.6, h=12 W/(m²K), T_air=303 K, T_dp=288 K, T_sea=293 K, G_b=800 W/m², 高度角 60°, G_d=120 W/m² |

天空有效温度用 Berdahl–Martin：ε_sky=0.711+0.56·t+0.73·t²（t=(T_dp−273.15)/100），T_sky=T_air·ε_sky^{1/4}；夜间工况（T_air=300 K, T_dp=285 K）给出 T_sky=282.62 K，白天工况（T_air=303 K, T_dp=288 K）给出 T_sky=287.47 K。设计甲板温度 T_design = 305 K（"甲板热载荷"在该温度下报价，因为在自由平衡温度上净负荷恒为 0，会退化成 0/0）。

## 2 怎么做的

### 2.1 模块地图（全部落在 TensorLBM 域内）

| 文件 | 内容 |
|---|---|
| `src/tensorlbm/radiative_view_factors.py` | 新增：6 类角系数闭式族 + 目录基准值 + 独立校验器（4D 高斯求积 4 个、天穹投影求积、通用矩形蒙特卡洛光线追踪） |
| `src/tensorlbm/gray_enclosure_radiosity.py` | 新增：灰体漫射封闭腔辐射度线性求解（J、G、q 三量全返）+ 能量/线性/闭合/求和/互易五项审计 + 再辐射面（ε=0, T=None）+ 解析锚（串联热阻、再辐射网络）+ 封闭腔构造器（同心球、立方体） |
| `src/tensorlbm/deck_thermal_load.py` | 新增：太阳/天空/海面边界 → 甲板能量平衡 → 平衡温度（Brent）+ 独立二分参考 + 无对流闭式解 + 蒙特卡洛长波估计 + 无限壁遮挡天空角系数 + LBM 壁面热流注入与注能审计 |
| `src/tensorlbm/thermal_radiation.py` | 最小修改：`compute_sky_radiation_flux` 增加 opt-in `view_factor_model="closed_form"`（F=(1+n_y)/2）与 `surface_temperature` 净交换模式；**默认路径逐位不变**（已用 verified 基准回归确认） |
| `src/tensorlbm/view_factor.py` | 仅加文档指针（不改变行为） |

### 2.2 验证方法（每条判据都有"独立第二算法"）

* **(a) 闭式 vs 独立基准**：平滑族用多维高斯-勒让德求积（平行矩形 200²、圆盘 40⁴=2.56e6 节点、微元圆盘 n=400）；接触边族（共边垂直矩形）的积分在公共边处被两个余弦同时压到 0、高级求积收敛极慢（n=160 仍偏高 3.4 %），因此改用**定义级**的余弦加权蒙特卡洛光线追踪（2e6 光线，σ≈3e-4）作基准。另与 Howell/Siegel/Menguc 目录值比对。
* **(b) 封闭腔**：直接 LU 解 (I−R F)J=e，q=J−G；能量守恒残差定义为 |Σ A_i q_i| / Σ A_i|q_i|（闭合腔内部换热必须严格抵消）。解析锚：两表面串联热阻 q=σ(T₁⁴−T₂⁴)/[1/ε₁+(A₁/A₂)(1/ε₂−1)]；含再辐射面的三面腔用标准辐射网络。再辐射面不臆造温度（ε=0、T=None，q≡0）。
* **(c) 甲板热载荷**：全链 = 闭式角系数 → 灰体封闭腔辐射度 → 耦合能量平衡（Brent 求根）；参考 = 同一标量方程的黑体环境闭式/独立 300 步二分。另用蒙特卡洛光线追踪独立复算长波交换。

### 2.3 工具链与关键设置

Python 3.11 / numpy 2.4.6 / scipy 1.16 / torch 2.7 / matplotlib 3.10；pytest（`tests/radiation/`，81 个新用例 + 原有 36 个）；
求积阶数与光线数见 §2.2；判据容差写死为 `TOL_A=1e-2, TOL_B=1e-6, TOL_C=5e-2` 并在 JSON 中逐条输出。

## 3 计算结果

### 3.1 判据 (a)：六类角系数闭式族

| 族 | 校验点数 | 最大相对误差 | 通过 |
|---|---|---|---|
| `parallel_rect` | 5 | 3.25e-14 | True |
| `perp_rect` | 3 | 0.00279 | True |
| `coax_disks` | 4 | 1.88e-08 | True |
| `conc_spheres` | 2 | 0.00198 | True |
| `diff_disk` | 4 | 2.43e-14 | True |
| `sky_dome` | 7 | 4.83e-05 | True |

目录值（Howell 2010 Table 5.1 等）比对，最大相对误差 2.45e-05：

| 族 | 参数 | 闭式 | 目录值 | 相对误差 |
|---|---|---|---|---|
| `parallel_rect` | [1.0, 1.0] | 0.199825 | 0.199820 | 2.45e-05 |
| `perp_rect` | [1.0, 1.0] | 0.200044 | 0.200040 | 1.89e-05 |
| `coax_disks` | [1.0, 1.0, 1.0] | 0.381966 | 0.381966 | 0 |
| `conc_spheres` | [2.0, 4.0] | 0.250000 | 0.250000 | 0 |
| `diff_disk` | [1.0, 1.0] | 0.500000 | 0.500000 | 0 |
| `sky_dome` | [0.0] | 1.000000 | 1.000000 | 0 |
| `sky_dome` | [60.0] | 0.750000 | 0.750000 | 0 |

### 3.2 判据 (b)：封闭腔辐射度能量守恒

| 案例 | 最差残差（能量/线性/闭合） | 解析锚误差 | 解析锚 |
|---|---|---|---|
| `parallel_plates_2_surface` | 8.7e-17 | 1.42e-16 | series resistance |
| `concentric_spheres_2_surface` | 1.69e-16 | 1.77e-16 | series resistance with closed-form (r1/r2)^2 |
| `cube_6_surface_sweep_9_configs` | 4.01e-15 | 0 | closed-form F_adj (perp rect) + summation-consistent F_opp = 1 - 4 F_adj |
| `three_surface_with_reradiating` | 8.82e-17 | 3.53e-16 | radiation network with reradiating surface |
| `deck_sky_open_enclosure` | 4.54e-14 | 0 | open enclosure with sky patch (reciprocity-consistent view factors) |

### 3.3 判据 (c)：甲板热载荷 vs 解析解

| 场景 | T_deck 模型 [K] | T_deck 参考 [K] | 载荷(305 K) 模型 [W/m²] | 载荷(305 K) 参考 [W/m²] | 相对误差 |
|---|---|---|---|---|---|
| `night_still_air_black_sky` | 282.618 | 282.618 | -116.05 | -116.05 | 1.22e-16 |
| `day_solar_plus_convection` | 325.151 | 325.151 | 370.60 | 370.60 | 1.75e-16 |
| `tilt_0deg` | 322.119 | 322.119 | 364.60 | 364.60 | 2.98e-14 |
| `tilt_20deg` | 317.146 | 317.146 | 256.86 | 256.86 | 8.3e-09 |
| `tilt_45deg` | 308.365 | 308.365 | 70.29 | 70.29 | 5.06e-07 |
| `tilt_70deg` | 302.224 | 302.224 | -57.50 | -57.50 | 2.58e-06 |
| `tilt_90deg` | 302.063 | 302.063 | -60.83 | -60.83 | 6e-06 |
| `gray_sky_eps_0.85` | 325.153 | 325.151 | 370.62 | 370.60 | 3.99e-05 |
| `gray_sky_eps_0.7` | 325.155 | 325.151 | 370.64 | 370.60 | 9.68e-05 |
| `longwave_analytic_vs_monte_carlo` | nan | nan | nan | nan | 1e-15 |

倾角扫描：

| 倾角 [°] | T 模型 [K] | T 参考 [K] | 载荷模型 [W/m²] | 载荷参考 [W/m²] | 相对误差 |
|---|---|---|---|---|---|
| 0 | 322.119 | 322.119 | 364.60 | 364.60 | 2.98e-14 |
| 20 | 317.146 | 317.146 | 256.86 | 256.86 | 8.3e-09 |
| 45 | 308.365 | 308.365 | 70.29 | 70.29 | 5.06e-07 |
| 70 | 302.224 | 302.224 | -57.50 | -57.50 | 2.58e-06 |
| 90 | 302.063 | 302.063 | -60.83 | -60.83 | 6e-06 |

灰天空（全链辐射度 vs 黑体环境解析）：

| ε_sky | T 全链 [K] | T 解析 [K] | 载荷全链 [W/m²] | 载荷解析 [W/m²] | 相对误差 |
|---|---|---|---|---|---|
| 0.85 | 325.153 | 325.151 | 370.62 | 370.60 | 3.99e-05 |
| 0.7 | 325.155 | 325.151 | 370.64 | 370.60 | 9.68e-05 |

长波交换解析 vs 蒙特卡洛：

| T_deck [K] | q_lw 解析 [W/m²] | q_lw MC [W/m²] | 相对误差 |
|---|---|---|---|
| 290 | -12.41 | -12.41 | 1e-15 |
| 310 | -122.77 | -122.77 | 1.16e-16 |
| 330 | -256.68 | -256.68 | 2.21e-16 |

### 3.4 甲板边界场（LBM Neumann 条件）与昼夜循环

* 天空角系数场：min 0.000，max 0.994，均值 0.762（上层建筑投影区 F→0，未遮挡区 F→1）。
* 壁面热流场（T_wall = 305 K）：-117.1 … 395.0 W/m²，面平均 297.8 W/m²。
* 注入审计：1536 个边界单元、dA=0.0625 m²、Δt=0.5 s → 注入焓 14294.629 J vs 理论 14294.629 J，相对误差 1.55e-11（差值为 300 K 温度偏移的 float64 舍入）。
* 24 h 昼夜循环（30°N，day 172，灰天空，准稳态）：T_deck ∈ [295.12, 329.87] K，305 K 定温载荷 ∈ [-173.0, 461.1] W/m²。

## 4 结果图

**图 1 — fig1_viewfactor_family**

![fig1_viewfactor_family](../figures/P92W3_N02_fig1_viewfactor_family.png)

*六类角系数闭式解（蓝点线）与独立基准（红叉：4D 高斯求积或蒙特卡洛光线追踪）逐点对比；各子图标注该族最大相对误差，全部 <1%*

**图 2 — fig2_radiosity_audit**

![fig2_radiosity_audit](../figures/P92W3_N02_fig2_radiosity_audit.png)

*灰体封闭腔辐射度审计：(左) 各案例能量/线性/闭合残差与 1e-6 判据线；(右) 六面立方体腔净辐射通量（热面放热、冷面吸热，总和为零）*

**图 3 — fig3_deck_load**

![fig3_deck_load](../figures/P92W3_N02_fig3_deck_load.png)

*船舶甲板热载荷：(左) 天空发射率 1.0/0.85/0.7 下全链（辐射度）与解析解冷却负荷；(中) 倾角扫描甲板平衡温度（模型 vs 基准重合）；(右) 24h 昼夜循环甲板温度与冷却负荷*

**图 4 — fig4_deck_field**

![fig4_deck_field](../figures/P92W3_N02_fig4_deck_field.png)

*甲板边界条件场（可直接作为 LBM 壁面 Neumann 条件）：(左) 超结构遮挡下天空角系数分布；(右) 稳态壁面热流 q''(x,y)，注意阴影带与边缘效应*

## 5 注意点（踩过的坑 / 局限 / 适用范围）

1. **共边垂直矩形的归一化约定陷阱（本节点实测被抓出）**：该闭式解的两个参数都是"边长 / 公共边长"（H=a/L, W=c/L），不是"边长 / 板间距"。在 a=c=L 时两种约定重合，故单元级冒烟测试无法发现；换成 H=1、W=0.2 后与蒙特卡洛相差 6 %，追查后确认是实现约定错误（原实现按 a/c、b/c 标注）。修正后 2D 极限 F(H,W→0) = (a+c−√(a²+c²))/(2a) = 0.292893 精确复现。**教训：只有"定义级"基准（光线追踪）能定位这类约定错误，求积与自己比只能自证一致。**
2. **接触边求积收敛陷阱**：同一几何用 4D 高斯求积 n=20/40/80/120/160 得 0.2338/0.2205/0.2120/0.2087/0.2069，向 0.20004 缓慢收敛；长公共边时更糟（H=W=0.02 时 n=160 给 0.87）。任何用中低阶求积"验证"该类角系数的做法都会假阳性。
3. `numpy.polynomial.legendre.leggauss(800)` 单次约 16 s（O(n²) 特征值），天穹求积默认降到 120×240（~1e-6 精度），否则测试被拖垮。
4. **同心球角系数方向约定**：(r₁/r₂)² 是"外球→内球"（外球凹面看内球），内球→外球恒为 1（凸面）。写反会让 ω 面辐射度方程整体走偏；本实现两个方向都提供并在 docstring 中写死。
5. **旧 API `compute_sky_radiation_flux` 的两处缺陷**（已 opt-in 修复、默认不变）：①天空角系数用 clamp(n_y)，对倾斜面系统性低估（45° 处 0.707 vs 闭式 0.854，−17 %）；②docstring 声称返回净通量，实际只返回入射项 εσT_sky⁴F，缺出射 εσT_s⁴。默认路径保持逐位不变，已用 `benchmarks/verified/radiation_enclosure_ship` 结果逐键比对确认无回归。
6. **旧 verified 基准自报残差 1.0**：`benchmarks/verified/radiation_enclosure_ship` 的 `energy_conservation_residual = 1.0`（把太阳吸收项并入照射量 G 后闭合腔内部换热不再抵消），叠加手填角系数矩阵——这是把该能力判为 MOCK 级证据的直接依据。新求解器同口径残差 4.54e-14。
7. **API 层既有缺口（不在本节点范围，未改动）**：`app/backend/routers/postprocess.py` 的 `/thermal-radiation` 端点构造 `RadiationEnclosureConfig` 时未提供 `view_factors`，撞上 fail-closed 契约后返回 500（`app/tests/test_engineering_gap_closure.py::TestThermalRadiationAPI` 3 例失败为既有现象）。建议后续节点让该端点接受角系数矩阵或几何块（可直接复用 `radiative_view_factors` 族与 `GrayEnclosure`）。
8. **灰天空修正在大天穹配比下极小**（ε_sky 1.0→0.7 只改 1e-4 量级载荷）：因为天空斑块面积比甲板大 1000 倍，其自身辐亮度由 εσT⁴ 主导。判据 (c) 的"通过"主要来自解析等价性而非灰天空物理，真实船舱（面积比 ~1）时该修正会显著变大——已在 `gray_enclosure_radiosity` 中留有任意面积比接口。
9. **边界场太阳分配是近似**：直射项用确定性阴影（射线-长方体求交，严格），散射+海面反射项按局部 F_sky 加权（各向同性天空假设）；有遮挡时的散射增强/多次反射未建模。
10. **昼夜循环为"准稳态序列"**：每个时刻单独解平衡，未含甲板热惯性（钢/木甲板 τ 为分钟量级），因此不能用于瞬态峰值温度；LBM 侧的瞬态由 `inject_deck_boundary` 每步注入 dT 承担。

## 6 误差判定（关键量 vs 基准，逐项回答"是否 < 3 %"）

| 关键量 | 基准 | 误差 | < 3 %？ |
|---|---|---|---|
| 6 类角系数（25 个工况点） | 4D 求积 / 蒙特卡洛光线追踪 | 最差 0.00279 | **是** |
| 角系数目录值（7 点） | Howell/Siegel/Menguc 2010 | 最差 2.45e-05 | **是** |
| 封闭腔辐射度能量守恒 | 物理守恒（应恒为 0） | 4.54e-14 | **是** |
| 两表面腔/再辐射腔净通量 | 串联热阻 / 辐射网络解析 | 最差 3.53e-16 | **是** |
| 甲板平衡温度 & 305 K 定温载荷 | 闭式解 / 独立二分 | 最差 9.68e-05 | **是** |
| 甲板长波交换 | 蒙特卡洛光线追踪（2e6 光线） | 1e-15 | **是** |

判据 (a) 的两个族（`perp_rect` 0.00279、`conc_spheres` 0.00198）的误差由蒙特卡洛统计量（σ≈2e-4~3e-4）而非闭式解本身决定：闭式解与目录值一致到 2.45e-05（角系数目录值仅给 5 位有效数字）。

## 7 审计链

* 代码：`src/tensorlbm/radiative_view_factors.py`、`gray_enclosure_radiosity.py`、`deck_thermal_load.py`（新增），`thermal_radiation.py`（opt-in 扩展），`view_factor.py`（文档指针）
* 测试：`tests/radiation/test_radiative_view_factors_closed_form.py`（41 例）、`test_gray_enclosure_radiosity.py`（19 例）、`test_deck_thermal_load.py`（21 例）；连同原有 `test_radiation.py` 共 **117 passed**
* 基准：`benchmarks/radiation_closed_form_deck/run.py` → `result.json` + `artifacts/P92W3_N02_radiation_verification.json`
* 回归：`benchmarks/verified/radiation_enclosure_ship` 结果逐键比对无变化（legacy 路径未被改动）
* 复现：`python benchmarks/radiation_closed_form_deck/run.py && python -m pytest tests/radiation -q`
