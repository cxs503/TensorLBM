# P93W3-N02 · TensorLBM 热辐射角系数内核与封闭腔辐射度解析锚：灰体封闭腔 + 太阳/天空辐射边界 → 船舶甲板温度场可算面

**节点**：P93W3-N02（TensorLBM，物理域，证据等级目标 REAL）
**机器判据产物**：`artifacts/P93W3_N02_radiation_deck_env.json`（`benchmarks/radiation_deck_env_p93/result.json` 同源）
**复现命令**：`python benchmarks/radiation_deck_env_p93/run.py`（约 40 s，含 6 张图）
**报告生成**：`python benchmarks/radiation_deck_env_p93/make_doc.py`（本文所有数字由 JSON 程序化注入，零手抄）

## 摘要

本次把 TensorLBM 的热辐射面从「板级热载荷 + 直接线性求解」推到「**球-平板/圆柱-平板角系数闭式族 + 高斯-赛德尔辐射度迭代 + 甲板温度场**」：

| 判据 | 要求 | 实测 | 结论 |
|---|---|---|---|
| (a) 5 构型角系数闭式误差 | ≤ 1 % | 最差 **0.0026** | PASS |
| (b) 封闭腔辐射度高斯-赛德尔迭代残差 | ≤ 1e-6 | 最差 **8.8909e-07** | PASS |
| (c) 等温封闭腔净辐射通量（能量守恒） | ≤ 0.1 % | 最差 **1.7814e-15** | PASS |
| (d) 甲板温度场 vs 解析稳态解 | ≤ 3 % | 最差 **7.5998e-05** | PASS |

四项全部通过（`all_pass = true`），且每条判据都对着**独立于被测实现**的基准：(a) 对多维高斯-勒让德求积 / 定义级蒙特卡洛光线追踪；(b) 对稠密 LU 解与代数方程残差；(c) 对 Stefan-Boltzmann 能量守恒极限；(d) 对两条互不相同的解析解（二分非线性平衡 + 余弦级数 2D 解）。

## 1 案例介绍

### 1.1 物理背景

船舶总体性能中的热辐射进入三条链路：①上层建筑/舱室灰体辐射换热；②甲板对太阳短波与天空长波的净交换（舰船红外特征、甲板结露、直升机甲板作业温度限制）；③热 LBM/导热求解所需的壁面 Neumann 边界。前一节点（P92W3-N02）已把「板级」热载荷与 6 类角系数做成 REAL，但留下两个缺口：**球-平板 / 圆柱-平板构型缺闭式解**（太阳盘、管系/烟囱对甲板）与**只有 0 维集总节点、没有温度场**；辐射度求解只有稠密 LU，无法在耦合导热时给出可核查的迭代收敛证书。本节点把这三处补齐并给出判据级验证。

### 1.2 几何与参数

| 案例 | 几何 / 取值 | 说明 |
|---|---|---|
| 角系数构型 1 | 平行正对等矩形 X,Y ∈ {0.5,1,2,3,4} | 已有闭式，本文复核 |
| 构型 2 | 共边垂直矩形 H,W ∈ {0.5,1,2} | 已有闭式 + 蒙特卡洛 |
| 构型 3 | 同轴平行圆盘 (R1,R2,S) | 已有闭式 + 极坐标求积 |
| 构型 4（新） | 球-平板：微元→球 (R/d)²、球→无限平板 ½ | 太阳盘 / 球状构件 |
| 构型 5（新） | 圆柱-平板：微元→无限长圆柱 R/h、圆柱→无限平板 ½ | 管系/烟囱/栏杆 |
| 封闭腔 | 立方腔（两 ε 组）、低 ε 立方腔、同心球、三面含再辐射面、甲板-天空-海面开放腔 | T ∈ [300,900] K，ε ∈ [0.05,1] |
| 甲板材料 | 船用钢 k=45 W/(m·K)，ρc=3.9e6 J/(m³K)，厚度 25 mm（板+骨材等效） | 等效集总厚度 |
| 环境 | T_air=303 K，露点 288 K → T_sky=287.47 K，T_sea=293 K，h_c=12 W/(m²K)，ε=0.9，α_s=0.6 | Berdahl-Martin 天空 |
| 太阳 | 直射 G_b=800 W/m²（高度角 60°）+ 散射 G_d=120 W/m² + 海面反照率 0.06 | 三分量显式 |
| 甲板内表面 | T_in=300 K，h_in=100 W/(m²K)（机械处所强制通风） | 下表面 Robin |

### 1.3 太阳边界三分量

| 倾角 β [°] | 直射吸收 [W/m²] | 天空散射吸收 [W/m²] | 海面反射吸收 [W/m²] | 合计 [W/m²] |
|---|---|---|---|---|
| 0 | 415.69 | 72.00 | 0.00 | 487.69 |
| 30 | 480.00 | 67.18 | 1.67 | 548.85 |
| 60 | 415.69 | 54.00 | 6.24 | 475.93 |
| 90 | 240.00 | 36.00 | 12.47 | 288.47 |

三分量之和与既有聚合函数 `absorbed_solar_flux` 的逐点相对偏差最差 **0**（同一浮点表达式，见 `tests/radiation/test_deck_thermal_field.py`）。水平甲板 F_sea=0 ⇒ 反射分量为 0；倾斜面（β=90°）反射分量 12.5 W/m²，说明三分量在几何上确实可分。

## 2 怎么做的

### 2.1 模块地图（全部落在 TensorLBM 域内）

| 文件 | 内容 |
|---|---|
| `src/tensorlbm/radiative_view_factors.py`（扩展） | 新增 4 类闭式：`vf_element_to_sphere`=(R/d)²、`vf_sphere_to_infinite_plane`=½、`vf_element_to_infinite_cylinder`=R/h、`vf_infinite_cylinder_to_parallel_plane`=½；新增 2 个独立校验器（3D 面积分求积、蒙特卡洛光线追踪）+ 目录基准值 + 族表登记 |
| `src/tensorlbm/radiosity_gauss_seidel.py`（新增） | 灰体漫射封闭腔辐射度的高斯-赛德尔迭代：逐行精确消去自视角系数、可选松弛、迭代残差/代数残差/能量闭合/闭合残差四项审计、谱半径 ρ(diag(1-ε)F) 预测收敛率；纯再辐射腔（全部 ε=0）判为奇异并显式报错 |
| `src/tensorlbm/deck_thermal_field.py`（新增） | 甲板温度场：D2Q5 热格子-Boltzmann 导热 + 上表面非线性辐射/对流源项 + 下表面 Robin；两条解析锚（一维二分非线性平衡 + 二维余弦级数线性化解）；三分量太阳边界 |
| `benchmarks/radiation_deck_env_p93/run.py` | 四项判据 + 数值研究 + 6 图，产物 `result.json` / `artifacts/P93W3_N02_radiation_deck_env.json` |
| `tests/radiation/test_view_factors_sphere_cylinder_plate.py` | 19 用例（闭式 vs 求积/蒙特卡洛、极限、守卫） |
| `tests/radiation/test_radiosity_gauss_seidel.py` | 13 用例（vs LU、等温腔闭合、再辐射、谱半径、松弛、诚实不收敛） |
| `tests/radiation/test_deck_thermal_field.py` | 13 用例（三分量一致性、解析锚、LBM 场 vs 解析，重算例标 `slow`） |

### 2.2 新增闭式解与推导要点

**球-平板（微元→球）**：把双面积分在球面上先积方位角，再对 u=cos t 积分，得

$$F_{dA\to S}=\frac{2R^2}{(\cdot)}\int_{-1}^{-R/d}\frac{(R+du)(Ru+d)}{(R^2+2Rdu+d^2)^2}\,du=\frac{R^2}{d^2}$$

（sympy 逐符号积分确认，且与 3D 面积分求积一致到 1e-9）。**球→无限平板 = ½** 与 **圆柱→无限平板 = ½** 来自凸体定理：凸体表面点向外半球不被自身遮挡，绕射向 d_z<0 的概率为 ½(1-n_z)，对球/圆柱 ⟨n_z⟩=0 ⇒ ½，与半径、高度无关。

**圆柱-平板（微元→无限长圆柱）= R/h**：用余弦投影单位圆盘技巧——漫射余弦加权方向与单位圆盘一一对应且**均匀**分布，于是 F 等于「射线打中圆柱」的方向集合在圆盘上的面积分数。射线 (x,z)=(τx, τ√(1-r²)) 与 x²+(z-h)²=R² 相交 ⇔ x² ≤ β²(1-y²)（β=R/h），即半轴 (β,1) 的椭圆内部，面积 πβ ⇒ **F = β = R/h**。该结果同时被 3D 求积（y=√D·tanψ 替换）与蒙特卡洛射线追踪两条独立路径确认。

### 2.3 辐射度迭代与收敛证书

$$J_i^{(k+1)}=(1-\omega)J_i^{(k)}+\omega\cdot\frac{\varepsilon_i\sigma T_i^4+(1-\varepsilon_i)\sum_{j\ne i}F_{ij}J_j}{1-(1-\varepsilon_i)F_{ii}}$$

矩阵 diag(1-ε)F 非负、闭环行和 ≤1；只要有一个 ε>0 即为严格次随机 ⇒ ρ<1 ⇒ 迭代是压缩映射（谱半径见判据表）。收敛判据用**迭代残差** max|ΔJ|/max|J|，判据 (b) 直接压在这一量上；能量闭合另用 |ΣA·q|/ΣA·εσT⁴（相对**发射功率**）——若按 |ΣAq|/ΣA|q| 归一，等温腔 q→0 时是 0/0，会被数值噪声放大到 O(1)（见 §5 的坑）。

### 2.4 甲板温度场：LBM 与两条解析锚

- 求解器：D2Q5 热 ADE（BGK），dx = L/ny 为方形格子，dt = c_s²(τ-1/2)dx²/α；上表面通量作为**边界行体源** S=q/dx 注入，下表面 h_in 对流同样以源项施加，四侧为反射边界（绝热）。
- 解析锚 1（厚度方向，全非线性）：q_net(T_s) = (T_s-T_in)/(L/k+1/h_in) 用 200 步二分求解（与既有 Brent 求解器算法不同），内部温度线性剖面 T(z)=T_in+q(1/h_in+z/k)；h_in=0 退化为绝热下表面时与既有 0 维平衡温度一致（测试钉住）。
- 解析锚 2（二维阴影边界，线性化）：上表面 Robin（h_eff=4εσT_m³+h_c）+ 下表面 Robin + 侧向绝热，用余弦级数 T=Σθ_n(z)cos(nπx/W) 精确求解，θ_n=a_n cosh λ_n z + b_n sinh λ_n z，逐模态 2×2 线性方程闭式给出。

## 3 计算结果

### 3.1 判据 (a)：5 构型角系数闭式 vs 独立裁判

| 构型 | 配置数 | 最差相对误差 | 独立裁判方法 |
|---|---|---|---|
| parallel_rect | 4 | 1.7004e-14 | 4D 高斯-勒让德（n=60）双面积分 |
| perp_rect | 3 | 0.0026 | 蒙特卡洛射线追踪（6e5 光线） |
| coax_disks | 3 | 4.0692e-15 | 极坐标高斯-勒让德求积（n=60） |
| sphere_plate | 4 | 0.0026 | 3D 面积分求积 / 蒙特卡洛 |
| cylinder_plate | 4 | 3.4012e-04 | 3D 面积分求积（y=√D·tanψ）/ 蒙特卡洛 |

逐条明细（闭式值 / 独立参考值 / 相对误差）：

| 构型 | 参数 | F 闭式 | F 参考 | 相对误差 |
|---|---|---|---|---|
| parallel_rect | X=0.5, Y=0.5 | 0.068590 | 0.068590 | 1.4163e-15 |
| parallel_rect | X=1, Y=1 | 0.199825 | 0.199825 | 2.5002e-15 |
| parallel_rect | X=1, Y=2 | 0.285875 | 0.285875 | 5.8254e-15 |
| parallel_rect | X=3, Y=4 | 0.587616 | 0.587616 | 1.7004e-14 |
| perp_rect | H=0.5, W=0.5 | 0.240636 | 0.241108 | 0.0020 |
| perp_rect | H=1, W=1 | 0.200044 | 0.200482 | 0.0022 |
| perp_rect | H=2, W=1 | 0.116426 | 0.116732 | 0.0026 |
| coax_disks | R1=1, R2=1, S=1 | 0.381966 | 0.381966 | 4.0692e-15 |
| coax_disks | R1=0.5, R2=1, S=1 | 0.468871 | 0.468871 | 3.9070e-15 |
| coax_disks | R1=2, R2=1, S=3 | 0.072949 | 0.072949 | 1.7122e-15 |
| sphere_plate | R=1, d=2, mode=element->sphere | 0.250000 | 0.250000 | 4.2188e-15 |
| sphere_plate | R=1, d=5, mode=element->sphere | 0.040000 | 0.040000 | 4.5103e-15 |
| sphere_plate | R=2, d=2.5, mode=element->sphere | 0.640000 | 0.640000 | 5.2042e-15 |
| sphere_plate | R=1, h=1.5, mode=sphere->infinite plane | 0.500000 | 0.501287 | 0.0026 |
| cylinder_plate | R=1, h=2, mode=element->infinite cylinder | 0.500000 | 0.500000 | 1.5099e-14 |
| cylinder_plate | R=0.4, h=2, mode=element->infinite cylinder | 0.200000 | 0.200000 | 1.5127e-14 |
| cylinder_plate | R=0.9, h=1, mode=element->infinite cylinder | 0.900000 | 0.900000 | 1.5543e-14 |
| cylinder_plate | R=0.8, h=2, mode=infinite cylinder->infinite plane | 0.500000 | 0.499830 | 3.4012e-04 |

### 3.2 判据 (b)：高斯-赛德尔辐射度迭代

| 封闭腔 | 面数 | 迭代数 | 迭代残差 | 代数残差 | vs LU 最大相对差 | ρ(diag(1-ε)F) |
|---|---|---|---|---|---|---|
| cube_two_eps | 6 | 13 | 4.4881e-07 | 1.3666e-07 | 1.9316e-07 | 0.5301 |
| cube_low_eps | 6 | 87 | 8.8909e-07 | 4.2988e-06 | 6.0480e-06 | 0.9332 |
| concentric_spheres | 2 | 5 | 1.2344e-07 | 2.1838e-09 | 2.1530e-09 | 0.5755 |
| three_surface_reradiating | 3 | 25 | 6.2751e-07 | 3.7890e-07 | 8.2796e-07 | 0.7914 |
| deck_sky_sea_open | 3 | 3 | 1.9038e-10 | 1.5934e-16 | 3.9771e-16 | 0.0400 |

五个封闭腔（含 6 面立方腔、同心球、含再辐射面的三面腔、甲板-天空-海面开放腔）迭代残差最差 **8.8909e-07** ≤ 1e-6，且与稠密 LU 解的最大相对差 **6.0480e-06**，即迭代没有引入新的解偏差。

### 3.3 判据 (c)：等温封闭腔能量守恒

| 腔体 | T [K] | ε | 净功率 [W] | 发射功率 [W] | 相对净功率 | max\|q\|/σT⁴ |
|---|---|---|---|---|---|---|
| cube | 300 | 0.05 | 6.8212e-13 | 551.2 | 1.2376e-15 | 4.4521e-07 |
| cube | 300 | 0.3 | 4.5475e-13 | 3307.0 | 1.3751e-16 | 1.6210e-07 |
| cube | 300 | 0.7 | 0 | 7716.2 | 0 | 2.2519e-08 |
| cube | 300 | 1 | 0 | 11023.2 | 0 | 0 |
| cube | 450 | 0.05 | -1.8190e-12 | 2790.2 | 6.5191e-16 | 4.4521e-07 |
| cube | 450 | 0.3 | 0 | 16741.5 | 0 | 1.6210e-07 |
| cube | 450 | 0.7 | -1.8190e-12 | 39063.5 | 4.6565e-17 | 2.2519e-08 |
| cube | 450 | 1 | 0 | 55805.0 | 0 | 0 |
| cube | 700 | 0.05 | 2.9104e-11 | 16337.5 | 1.7814e-15 | 4.4521e-07 |
| cube | 700 | 0.3 | -7.2760e-12 | 98024.9 | 7.4226e-17 | 1.6210e-07 |
| cube | 700 | 0.7 | -7.2760e-12 | 228724.8 | 3.1811e-17 | 2.2519e-08 |
| cube | 700 | 1 | 0 | 326749.7 | 0 | 0 |
| concentric_spheres | 500 | 0.8 | -1.1429e-11 | 320652.1 | 3.5643e-17 | 9.4754e-09 |

等温腔净功率相对值最差 **1.7814e-15**（门 1e-3），单面净通量相对 σT⁴ 最差 **4.4521e-07** —— 即辐射度方程在数值上精确恪守 Stefan-Boltzmann 极限。

### 3.4 判据 (d)：甲板温度场 vs 解析稳态解

**厚度方向（全非线性耦合）**

| ny | 步数 | 收敛 | T_top(LBM) [K] | T_top(解析) [K] | q_top(LBM) [W/m²] | q 解析 [W/m²] | 场最大相对误差 | 能量闭合 |
|---|---|---|---|---|---|---|---|---|
| 6 | 54400 | True | 304.0434 | 304.0837 | 387.596 | 386.882 | 7.3648e-05 | 9.6114e-08 |
| 8 | 96600 | True | 304.0535 | 304.0837 | 387.418 | 386.882 | 5.5189e-05 | 9.9818e-08 |

**二维阴影边界（冻结线性化系数）**：W=0.2 m，nx=32、ny=4，步数 24200，T∈[301.500, 302.874] K，与余弦级数解析解的最大场误差 **7.5998e-05**，能量闭合 9.1242e-08。

解析锚（厚度方向）：T_surface = 304.083750 K，q = 386.881541 W/m²，总热阻 0.010556 K·m²/W，二分残差 -1.9895e-12 W/m²。

### 3.5 数值研究：时间步（τ）敏感性与观测收敛阶

| τ | dt [s] | 步数 | T_top [K] | 场最大相对误差 |
|---|---|---|---|---|
| 1 | 0.2508 | 54400 | 304.0434 | 7.3648e-05 |
| 2 | 0.7523 | 18400 | 304.1139 | 1.5800e-04 |
| 5 | 2.2569 | 6600 | 304.7845 | 0.0024 |
| 20 | 9.7801 | 3800 | 312.9269 | 0.0291 |

观测收敛阶（ny=6→8）：**p = 1.003**（一阶，与「通量以边界行体源施加、等效边界内移半格」一致）。τ 增大可大幅减少步数，但**稳态离散解并非严格 τ 不变**（朴素 w_i·S 源项留下 O(τ) 一致性误差），τ=20 时误差已接近 3 % 门限——因此求解器默认 τ=1（大步长只用于瞬态估计，不用于判据）。

## 4 结果图

**图 1**（`figures/p93w3_n02/p93_fig_a_view_factors.png`）：Five-configuration closed-form view-factor family (parallel rectangles, perpendicular rectangles, coaxial disks, sphere-plate, cylinder-plate) against independent quadrature / Monte-Carlo estimators.

![view_factor_family](../figures/p93w3_n02/p93_fig_a_view_factors.png)

**图 2**（`figures/p93w3_n02/p93_fig_b_radiosity_convergence.png`）：Gauss-Seidel radiosity residual history for five enclosures; the annotated spectral radius rho(diag(1-eps)F) sets the asymptotic rate.

![radiosity_convergence](../figures/p93w3_n02/p93_fig_b_radiosity_convergence.png)

**图 3**（`figures/p93w3_n02/p93_fig_c_isothermal_cavity.png`）：Isothermal closed cavity: net radiative power normalised by the emitted power, over three temperatures and four emissivities (gate 0.1 %).

![isothermal_cavity](../figures/p93w3_n02/p93_fig_c_isothermal_cavity.png)

**图 4**（`figures/p93w3_n02/p93_fig_d_deck_1d_field.png`）：Through-thickness deck field: LBM (two resolutions) against the analytic bisection+linear solution, and the nonlinear surface balance whose root defines the surface temperature.

![deck_1d_field](../figures/p93w3_n02/p93_fig_d_deck_1d_field.png)

**图 5**（`figures/p93w3_n02/p93_fig_d2_shadow_edge.png`）：Two-dimensional deck field across a shadow edge: LBM pcolormesh and top/bottom-row comparison with the exact cosine-series solution.

![deck_2d_shadow](../figures/p93w3_n02/p93_fig_d2_shadow_edge.png)

**图 6**（`figures/p93w3_n02/p93_fig_solar_components.png`）：Absorbed solar flux split into direct beam, isotropic sky diffuse and ground(sea)-reflected components versus panel tilt.

![solar_components](../figures/p93w3_n02/p93_fig_solar_components.png)

## 5 注意点（踩过的坑 / 局限 / 适用范围）

1. **「每步温度变化小」不是收敛判据。** 第一版用 max|ΔT|/max|T| < 1e-6 判稳态，结果 4800 步就“收敛”，而能量闭合残差 0.29（净余 116 W/m² 还在加热）。单步 ΔT 只反映 dT/dt，慢模态漂移时它天然很小。改为**能量闭合残差** |Σq_top+Σq_bot|/Σ|q_top| 后，收敛步数升到 5e4 量级且残差 1e-7。
2. **能量残差的归一化分母不能取 ΣA|q|。** 等温腔解析上 q≡0，0/0 会把数值噪声放大成 O(1)（实测 residual=1.0 的假警报）；改用**发射功率** ΣAεσT⁴ 作分母后降到 1e-15 量级。
3. **2D 余弦级数的边界符号。** 首次实现把上下 Robin 条件整体写反（−k∂T/∂z），解析解出现“上冷下热”，与 LBM 差 10 %。以「z 向上、导热通量 = −k∂T/∂z」重推：上表面 k∂T/∂z|_L = A−h_eff T(L)，下表面 k∂T/∂z|_0 = h_in(T(0)−T_in)，两条件对 k∂T/∂z 都是**正号**。修好后 LBM/解析一致到 4e-5。
4. **线性化分支的双重计入。** 冻结线性化时若沿用非线性分支的 h_c·T 再叠加 h_eff=h_r+h_c，会重复扣除对流项，整场偏冷 30 K。两条分支必须各自完整、互不复用常量。
5. **D2Q5 稳态解对 τ 并非严格不变**（§3.5），这是朴素源项的 O(τ) 一致性误差；用 Guo 二阶强迫可望消除，本节点未做，列入后续。当前以 τ=1 运行并把 τ 敏感性作为已知局限公开。
6. **二维网格受 D2Q5 方形格子约束**：dx=L/ny 决定 nx。真实甲板长厚比 600（12 m / 20 mm）无法在面内解析；本文二维算例取 0.2 m 阴影边界贴片。这不是偷懒：钢甲板面内扩散时间 ~1e7 s 比厚度方向（~35 s）长七个数量级，真实甲板本来就接近**局部辐射平衡 + 弱侧向导热**——这也是判据 (d) 的厚度方向一维场为主锚的原因。
7. **通量边界是一阶处理**（体源内移半格，观测阶 p≈1.0）。对 25 mm 钢甲板、q~390 W/m²、k=45 的工况，半格偏移引起的温度误差 ~1e-2 K，比 3 % 门限低三个数量级；但换到低导热/厚板工况需按 §3.5 做网格敏感性检查。
8. **适用域**：灰体漫射（无镜反射、无参与介质）、封闭腔角系数须满足求和/互易律（`GrayEnclosure` 构造时审计）；天空按 Berdahl-Martin 有效温度、海面按灰体 0.96，未含云量/气溶胶时间序列。

## 6 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准 | 实测误差 | 是否 < 3 % |
|---|---|---|---|
| 5 构型角系数 | 独立求积 / 蒙特卡洛 / 目录值 | 最差 0.0026 | 是 |
| 辐射度迭代残差 | 迭代不动点（LU 解交叉验证 6.0480e-06） | 8.8909e-07 | 是 |
| 等温腔净功率 | Stefan-Boltzmann 能量守恒（应为 0） | 1.7814e-15 | 是 |
| 甲板温度场（厚度方向 ny=8） | 二分非线性平衡 + 线性剖面解析解 | 5.5189e-05 | 是 |
| 甲板温度场（二维阴影边界） | 余弦级数精确解 | 7.5998e-05 | 是 |
| 甲板表面热流 q_top | 解析 q | 0.0014 | 是 |
| 太阳三分量之和 | 既有聚合函数 `absorbed_solar_flux` | 0 | 是 |

**结论：关键量全部 < 3 %（实测最差 0.26 %，来自蒙特卡洛统计误差量级的垂直矩形构型）**；温度场判据最差 7.6e-5，远优于 3 % 门限。无判据项使用「无基准」豁免。

## 7 审计链

- 判据数字：`artifacts/P93W3_N02_radiation_deck_env.json` → `criteria.{a,b,c,d}`（本报告由 `make_doc.py` 从此 JSON 注入）。
- 单元/回归测试：`tests/radiation/test_view_factors_sphere_cylinder_plate.py`（19）、`test_radiosity_gauss_seidel.py`（13）、`test_deck_thermal_field.py`（13，其中 2 个 `slow`）。
- 图：`figures/p93w3_n02/*.png`，由 `run.py` 内 matplotlib 程序化生成（无手绘）。
- 与既有能力的关系：角系数族表 `CLOSED_FORM_FAMILY` 现含 10 类；辐射度 LU 解（P92W3-N02）未被替换，高斯-赛德尔解与其交叉验证（最大相对差 6e-6）。

## 8 局限与后续

- 镜面反射/参与介质未覆盖；腔体角系数仍要求外部给定并满足求和/互易（本节点只补构型族，不做自动腔体分割）。
- LBM ADE 的 τ 一致性与二阶通量边界（Guo 强迫 / 镜像粒子）是下一步；当前一阶边界已满足判据但值得改进。
- 二维阴影算例是 0.2 m 贴片；若要做整甲板温度分布，应改用「局部平衡 + 弱侧向导热」的薄板集总模型（面内时间尺度问题见 §5.6）。

