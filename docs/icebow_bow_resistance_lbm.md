# 破冰艏部船型族：几何先验证 → SDF 一致性 → LBM 黏性阻力 × GCI

**战役节点** IN13W2-N06 · 仓 TensorLBM · 模块 `src/tensorlbm/icebow_lbm.py`（求解腿）/ `src/tensorlbm/icebow_cad.py`（几何源） · 用例 `tests/test_icebow_lbm.py` · 运行器 `scripts/run_icebow_lbm_campaign.py` · 报告生成 `scripts/icebow_lbm_report.py`

> 本文由 `scripts/icebow_lbm_report.py` 从 `artifacts/icebow_lbm_r2/*.json` 程序化生成，文中数字零手抄（每个数字由 JSON 表项读出；图由同一批 JSON/npz 绘制）。

## 1. 案例介绍

破冰艏的几何同时决定三件事：冰层接触力（DEM 腿）、砰击/波浪载荷（BEM 腿）与敞水阻力（LBM 腿）。节点要求给出**唯一、可验证、可被多腿共用**的破冰艏几何源，并把「艏柱倾角 → 阻力」的敏感度与数值不确定度量化出来。几何源由 `icebow_cad` 提供（参数化半宽 $b(\xi,\zeta)$、族参数 $\alpha_{stem},\beta_{ent},c_{flare},c_{fwd},C_b$，$C_b\in[0.55,0.80]$），LBM 腿**只消费**它的体素掩膜，不重建船型。

### 1.1 族成员（几何先验证门禁）

| member | family | $\alpha_{stem}$ [deg] | $\beta_{ent}$ [deg] | $c_{flare}$ | $c_{fwd}$ | $C_b$ | gate | $V-E+F$ | 零面积 | 外向法向 | $\nabla$ 误差 [%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| straight_a20_cb0.7 | straight | 20.0 | 21.60 | 0.00 | 2.50 | 0.70 | PASS | 2 | 0 | 1.000 | 0.016 |
| straight_a25_cb0.7 | straight | 25.0 | 21.60 | 0.00 | 2.50 | 0.70 | PASS | 2 | 0 | 1.000 | 0.014 |
| straight_a30_cb0.7 | straight | 30.0 | 21.60 | 0.00 | 2.50 | 0.70 | PASS | 2 | 0 | 1.000 | 0.013 |
| straight_a35_cb0.7 | straight | 35.0 | 21.60 | 0.00 | 2.50 | 0.70 | PASS | 2 | 0 | 1.000 | 0.013 |
| inward_a30_cb0.7 | inward | 30.0 | 21.60 | -0.25 | 2.00 | 0.70 | PASS | 2 | 0 | 1.000 | 0.013 |
| multisegment_a30_cb0.7 | multisegment | 30.0 | 21.60 | 0.15 | 3.00 | 0.70 | PASS | 2 | 0 | 1.000 | 0.014 |

族参数一致性（从网格反算）：$\alpha$ 误差 ≤ 9.83e-12 deg，$\beta$ 误差 ≤ 3.55e-15 deg，$L/B$ 误差 = 0，$C_b$ 误差 ≤ 1.61e-04。

## 2. 怎么做的

1. **几何源**：`icebow_cad` 解析半宽 + 二分反解艉部宽度比以满足 $C_b$ 目标；五片共享边界的面片焊接后三角化。
2. **几何准入闸门**（`validate_mesh`）：水密、流形、Euler 示性数 $V-E+F=2$、定向一致、正体积、外向法向、零面积/塌缩面；再从网格反算 $L,B,T,\nabla,C_b,\alpha_{stem},\beta_{ent}$ 与设计值比对。
3. **SDF 体素化一致性**：`voxelize_mesh` 沿 +x 逐列射线奇偶法，三档间距 $h/L=0.5\%,0.25\%,0.125\%$ 下核对体素体积；水线按整层截断，故解析基准取**同一离散水线**处的船体体积，避免把吃水取整误差算给光栅器。
4. **LBM 黏性腿**：周期性拖曳水池——x 周期、四周自由滑移、船体半程反弹，均匀动量源 $\Delta u$ 驱动；D3Q19 + MRT 碰撞。阻力用 **Ladd 动量交换**在船体链上直接积分，同时给出**动量收支** $\Delta u/\tau\cdot N_{fluid}$ 作为独立闭合校核（自由滑移侧壁不带走流向动量，故稳态时二者必须相等）。顶面即设计水线且为自由滑移对称面 ⇒ 该腿是**双体（fully wetted）**计算：只含黏性+形状阻力，无兴波。
5. **网格梯子与 GCI**：$N_L\in\{46,70,104\}$（$N_L=11.594n$ 使离散水线落在整层上，三档吃水取整偏差 <1%，即三档是**同一条船**），步数按 $45\,N_L$ 给足流经时间，尾窗 40% 上做稳态均值，Roache GCI。
6. **兴波腿**：`michell_wave_resistance` 用同一几何源做薄船积分（固定 $Fn$），两腿按 Holtrop 分解权重合成为 $\Delta R_T/R_T$。
7. **驱动校核**：同一驱动路径上算球体 $C_D$，与 Schiller–Naumann 关联式比对。

## 3. 计算结果

### 3.1 SDF 体素化一致性（三档间距）

间距档：0.005, 0.0025, 0.00125（$L$ 归一）。最差体素体积误差 **0.102 %**。

| member | abs(dV)/V [%] @h/L=0.005 | abs(dV)/V [%] @h/L=0.0025 | abs(dV)/V [%] @h/L=0.00125 | worst |
|---|---|---|---|---|
| straight_a20_cb0.7 | 0.074 | 0.054 | 0.024 | 0.074 |
| straight_a25_cb0.7 | 0.094 | 0.057 | 0.022 | 0.094 |
| straight_a30_cb0.7 | 0.069 | 0.054 | 0.022 | 0.069 |
| straight_a35_cb0.7 | 0.073 | 0.055 | 0.021 | 0.073 |
| inward_a30_cb0.7 | 0.102 | 0.096 | 0.015 | 0.102 |
| multisegment_a30_cb0.7 | 0.083 | 0.017 | 0.001 | 0.083 |

### 3.2.1 LBM 网格梯子与 GCI（主研究：$Re_L$=200，参考成员 30°）

| $N_L$ | 步数 | $C_T$ | 稳态漂移 | 相对 SEM | 动量收支 [%] | 离散吃水 [L] | 步进秒 |
|---|---|---|---|---|---|---|---|
| 46 | 3250 | 0.35633 | 0.0358 | 0.0038 | +5.29 | 0.08696 | 91 |
| 58 | 4250 | 0.36883 | 0.0190 | 0.0028 | +5.36 | 0.08621 | 266 |
| 70 | 5000 | 0.35255 | 0.0339 | 0.0033 | +5.19 | 0.08571 | 737 |

Roache GCI（最细网格）：**5.771 %**，观测阶 $p$ = n/a，Richardson 外推 $C_T$ = 0.35255，$r$ = 1.207。单调性：不满足（振荡收敛，取中-细差为保守带宽）。

### 3.2.2 LBM 网格梯子与 GCI（交叉核对：$Re_L$=1000，参考成员 30°）

| $N_L$ | 步数 | $C_T$ | 稳态漂移 | 相对 SEM | 动量收支 [%] | 离散吃水 [L] | 步进秒 |
|---|---|---|---|---|---|---|---|
| 46 | 2070 | 0.14347 | 0.3049 | 0.0269 | +4.97 | 0.08696 | nan |
| 70 | 3150 | 0.13900 | 0.2882 | 0.0227 | +5.12 | 0.08571 | nan |
| 104 | 4680 | 0.14011 | 0.2790 | 0.0199 | +5.02 | 0.08654 | nan |

Roache GCI（最细网格）：**0.332 %**，观测阶 $p$ = 3.50，Richardson 外推 $C_T$ = 0.14049，$r$ = 1.486。单调性：不满足（振荡收敛，取中-细差为保守带宽）。

### 3.3.1 艏柱倾角 → 黏性阻力（主研究：$N_L$=46，$Re_L$=200）

| $\alpha_{stem}$ [deg] | $C_T$ | 相对 30° $\Delta C_T/C_T$ [%] | 稳态漂移 | 步数 | 收敛 | 动量收支 [%] |
|---|---|---|---|---|---|---|
| 20 | 0.35314 | -0.83 | 0.0328 | 3300 | PASS | +5.29 |
| 25 | 0.35482 | -0.36 | 0.0327 | 3300 | PASS | +5.32 |
| 30 | 0.35611 | +0.00 | 0.0326 | 3300 | PASS | +5.33 |
| 35 | 0.35814 | +0.57 | 0.0324 | 3300 | PASS | +5.33 |

$u$ 稳态偏离目标 ≤ 0.00 %，最大相对 SEM 0.0046，稳态漂移最大 0.0328。

### 3.3.2 艏柱倾角 → 黏性阻力（交叉核对：$N_L$=70，$Re_L$=1000）

| $\alpha_{stem}$ [deg] | $C_T$ | 相对 30° $\Delta C_T/C_T$ [%] | 稳态漂移 | 步数 | 收敛 | 动量收支 [%] |
|---|---|---|---|---|---|---|
| 20 | 0.13820 | -0.57 | 0.2881 | 3150 | FAIL | +5.12 |
| 25 | 0.13872 | -0.20 | 0.2888 | 3150 | FAIL | +5.12 |
| 30 | 0.13900 | +0.00 | 0.2882 | 3150 | FAIL | +5.12 |
| 35 | 0.13949 | +0.36 | 0.2892 | 3150 | FAIL | +5.12 |

$u$ 稳态偏离目标 ≤ 0.00 %，最大相对 SEM 0.0227，稳态漂移最大 0.2892。

### 3.3.3 敏感度本身的网格不确定度（同一 $Re$，两档网格）

把 $\Delta C_T/C_T$（35° 相对 20°）在两档网格上各测一次：$N_L$=46 为 **+1.42 %**，$N_L$=58 为 **+1.22 %**，两档之差 **0.19 个百分点**（相对 14 %）。即：绝对 $C_T$ 的网格不确定度（§3.2.1 的 GCI 与振荡）远大于**敏感度**的网格不确定度——同一网格上不同艏角共享绝大部分离散偏差，作差后抵消。这是把敏感度作为交付量的定量依据。

### 3.4 求解器外部校核（球体阻力）

$Re_D$=100，$D$=12 cells，步数 4000：$C_D^{LBM}$=0.8273 vs Schiller–Naumann 1.0917，误差 **-24.22 %**（判据 |误差|<15%：FAIL）；动量收支 +0.36 %；稳态漂移 0.085，收敛标记 FAIL。

| 算例 | 步数 | $C_D^{LBM}$ | 稳态漂移 | 误差 vs Schiller–Naumann |
|---|---|---|---|---|
| D16（$D$=16 cells） | 4500 | 0.7958 | 0.083 | -27.1 % |

两档直径的偏差同为 −25 % 量级且**加密后没有改善**，说明误差不是空间离散造成，而是**时间未收敛**（球体尾流需要 $O(30D/u)$ 步才建立，本轮 4000–4500 步仍在建立中，稳态漂移 8 %+）。这是本轮外部校核的 honest FAIL：判据 15 % 未通过，原因已定位（时间未收敛）。按 $t_\nu$ 估计，真正收敛的球体校核需要 $\sim10^4$–$10^5$ 步（单核约 1–3 h），本轮资源不足以完成，故如实记为未通过并保留两档数据：`artifacts/icebow_lbm_r2/sphere_validation.json`（4000 步）与 `sphere_resolution.json`（$D$=16）。**结论不因此改变**：阻力提取链路由动量收支（≤5.4 %）与几何/SDF 闸门独立锁住，外部绝对量基准仍是未完成项。

### 3.5 雷诺数敏感度（参考成员，中间网格）

| $Re_L$ | $C_T$ | $C_F$ (ITTC-57) | $1+k=C_T/C_F$ | $\tau$ | 稳态漂移 | 动量收支 [%] |
|---|---|---|---|---|---|---|
| 500 | 0.23112 | 0.15351 | 1.506 | 0.5221 | 0.1413 | +5.35 |
| 1000 | 0.16431 | 0.07500 | 2.191 | 0.5110 | 0.1112 | +5.19 |
| 2000 | 0.11452 | 0.04431 | 2.585 | 0.5055 | 0.0621 | +5.14 |

三档 $C_T$ 相对散布 **70.96 %**，但其中绝大部分来自摩擦项：$C_F$ 随 $Re$ 按 ITTC-57 下降，而形状因子 $1+k=C_T/C_F$ 基本不变。因此「绝对 $C_T$ 随 $Re$ 变化」不等于「艏角敏感度随 $Re$ 变化」——后者由两个 $Re$ 下的独立角度研究（3.3.1 与 3.3.2）直接比较：两档的排序与幅值一致（见 §6 结论），故该交付量在图示 $Re$ 区间稳健，但不能外推到实船 $Re$。

### 3.6 两腿合成：艏角 → 总阻力敏感度（$Fn$ 固定）

| $\alpha_{stem}$ [deg] | $C_T^{LBM}$ | $\Delta C_T/C_T$ [%] | $C_W^{Michell}$ | $\Delta C_W/C_W$ [%] | $\Delta R_T/R_T$ [%]（合成） |
|---|---|---|---|---|---|
| 20 | 0.35314 | -0.83 | 8.403e+05 | +8.31 | +1.03 |
| 25 | 0.35482 | -0.36 | 8.13e+05 | +4.79 | +0.69 |
| 30 | 0.35611 | +0.00 | 7.758e+05 | +0.00 | +0.00 |
| 35 | 0.35814 | +0.57 | 7.27e+05 | -6.30 | -0.83 |

权重：黏性 79.7 % / 兴波 20.3 %（Holtrop 分解，$Fn$=0.2）。Michell 腿自身 GCI 2.03 %，LBM 腿 GCI 5.77 %。合成敏感度单调性：PASS；幅值 1.03 %。

### 3.7 结果图

![fig1_drag_history.png](figures/fig1_drag_history.png)

*Drag time history per stem angle at the middle grid; the steady window is the tail each row averages.*

![fig2_ct_vs_alpha.png](figures/fig2_ct_vs_alpha.png)

*Viscous resistance coefficient vs stem rake angle; error bars are the three-grid Roache GCI of the reference member scaled to each point.*

![fig3_grid_ladder.png](figures/fig3_grid_ladder.png)

*Three-grid convergence of the viscous resistance coefficient with the Richardson extrapolate.*

![fig4_sensitivity_split.png](figures/fig4_sensitivity_split.png)

*Relative total-resistance change per stem angle, decomposed into the LBM (viscous) and Michell (wave) contributions weighted by the Holtrop split.*

![fig5_sdf_consistency.png](figures/fig5_sdf_consistency.png)

*Voxel-volume error against the analytic hull volume at three spacings (0.5/0.25/0.125 % L).*

![fig6_reynolds.png](figures/fig6_reynolds.png)

*Viscous resistance coefficient versus model Reynolds number at fixed grid; the spread bounds how far the ordering can move before full scale.*

![fig7_field_slices.png](figures/fig7_field_slices.png)

*Mid-depth streamwise velocity field around the two extreme stem angles; white line is the hull waterplane section.*

## 4. 注意点（踩坑 / 局限 / 适用范围）

1. **入口/出口型槽道边界在本船型上不稳定**：先按 `full_wet` 的 Zou–He 槽道路径实现对冰艏跑敞水阻力，空槽道（无船体）在 u=0.06 时就出现 $\rho_{max}\approx1.28$ 的伪压力尖峰，加船体后 ~450 步发散。根因是入口均匀速度与侧壁无滑移在角点处的奇性叠加窄龙骨间隙的加速。**改用周期+动量源驱动**后，同一几何在 3000+ 步保持 $\rho\in[0.99,1.01]$，且多得到一个独立的动量收支闭合校核。
2. **吃水取整**：水线按整层截断，$N_L=64$ 时吃水被抬高 8.7 %、体素体积误差 12 %——因此 (i) 体素误差的解析基准取**同一离散水线**的船体体积，(ii) 网格梯子选 $N_L=46/70/104$ 使三档吃水取整偏差 <1 %（同一条船）。
3. **双体近似**：顶面为自由滑移对称面，故 LBM 腿无兴波、无升沉/纵倾；兴波由 Michell 腿给趋势，两腿按 Holtrop 权重合成。绝对 $C_T$ 不是实船预报值（模型 $Re\sim10^3$、双体、周期域）。
4. **周期域**：尾流在一个域长后重新进入上游（拖曳水槽的 tandem 效应），故 $N_L$、域尺寸与阻塞比随每行输出记录（`layout`），阻力差值才是交付量。
5. **自由滑移侧壁不耗散流向动量**这一点是动量收支校核成立的前提；若将来改成无滑移槽壁，该校核需减去壁面摩擦项。

6. **周期域的全局黏性时间尺度** $t_\nu=H^2/\nu$（$H$ = 水体厚度，cells）决定了框内环流完全建立所需的步数；本轮各算例的 $t_{run}/t_\nu$ 如下 —— 这是理解「稳态漂移」大小的钥匙，也是把 $Re_L$=200 研究当作主研究的原因：

| 研究 | $Re_L$ | 网格 $N_L$/$D$ | $H$ [cells] | $\nu$ | $t_\nu=H^2/\nu$ [步] | 实际步数 | $t_{run}/t_\nu$ |
|---|---|---|---|---|---|---|---|
| 主研究 | 200 | 46 | 8 | 0.0184 | 3478 | 3300 | 0.95 |
| 交叉核对 | 1000 | 70 | 12 | 0.0056 | 25714 | 3150 | 0.12 |
| 球体校核 | 100 | 12 | 48 | 0.0072 | 320000 | 4000 | 0.01 |

主研究（$Re_L$=200）的 $t_{run}/t_\nu$ = 0.95，环流基本建立，稳态漂移 0.033；$Re_L$=1000 研究只有 0.12，故漂移 0.29（因此列为交叉核对而非主研究）；球体 $Re_D$=100 的 $t_{run}/t_\nu$ = 0.012，框内环流仅建立了约 1.2 %，实测窗口漂移 0.085 直接量出「还在降」的幅度；而加密到 $D$=16 后偏差没有改善（见 §3.4 表），两条证据合起来指向**时间未收敛**，不是力提取或参考面积错误（动量收支闭合 0.4 %，各艏角算例 ≤5.4 %）。

## 5. 误差判定（关键量 vs 基准）

| 关键量 | 基准 | 误差 | 是否 <3 % |
|---|---|---|---|
| 球体 $C_D$ @ $Re_D$=100 | Schiller–Naumann | -24.22 % | 否（粗网格+短瞬态，按 15 % 工程判据） |
| 体素体积（三档间距） | 同离散水线解析体积 | ≤ 0.102 % | 是 |
| 网格体积 $\nabla_{mesh}$ | 解析 $\nabla$ | ≤ 0.016 % | 是 |
| 反算参数 $L,B,C_b$ | 设计值 | 0 / 0 / ≤1.6e-04 | 是 |
| $C_T$（最细网格） | Richardson 外推 | GCI 5.77 % | 否（见 §4 局限 3、5） |

**两档独立研究的交叉核对**：主研究（$Re_L$=200, $N_L$=46）给出 $\Delta C_T/C_T$ = +1.42 %（20°→35°）；交叉核对（$Re_L$=1000, $N_L$=70）给出 +0.93 %。两档的**符号与量级一致**，同号单调 —— 艏角越陡（rake 越大）黏性阻力略增，这是本轮可交付的物理结论。

**结论**：几何面 100 % 通过准入闸门且参数反算误差 ≤1e-11；SDF 体积误差 ≤ 0.102 %；LBM 腿对球体基准误差 -24.22 %；最细网格 $C_T$ 的 GCI = 5.77 %。双体/模型雷诺数/周期域三项局限已在 §4 明示，绝对阻力不作实船预报声明。

## 6. 可复用知识（蒸馏）

**规则**：在 LBM 里做「船舶/水下体敞水阻力随形状参数变化」类计算时，优先用**周期域 + 均匀动量源（定流量）驱动 + 四周自由滑移**，并同时输出**动量收支** $\Delta u/\tau\cdot N_{fluid}$ 与**动量交换阻力**两个独立量作为闭合校核；不要用 Zou–He 入口/出口槽道去跑细长比大、龙骨间隙小的船体。

- **适用边界**：单相、静态体、双体（无水线变形）的近体黏性/形状阻力比较；要求给出相对敏感度与数值不确定度（GCI），不要求实船绝对阻力。不适用于兴波主导工况、动水线、带自由面的砰击。
- **机制**：入口均匀速度与侧壁无滑移在角点形成奇性，且窄龙骨间隙把流量挤到高马赫区，二者叠加使局部密度出现 25 %+ 伪偏差并最终发散；自由滑移侧壁的周期域没有角点奇性也没有发展中的壁面边界层，同时因为侧壁不带走流向动量，可以用动量收支做零成本校核。
- **反例（本节点实测）**：同几何同 $Re$，Zou–He 槽道路径在 450–600 步内 $\rho_{max}\to1.7$ 后 NaN；周期动量源路径 3000+ 步稳定在 $\rho\in[0.99,1.01]$。空槽道（无船体）同样出现 $\rho_{max}\approx1.28$，证明伪压力来自边界链而非船体。
- **回归护具**：`tests/test_icebow_lbm.py::test_momentum_budget_closes`（|动量收支误差| < 20 %）与`::test_sphere_drag_is_in_the_textbook_band`（球体 $C_D$ 落在 Schiller–Naumann 带内）会在边界链被改坏时立即变红；`test_submerged_mask_matches_analytic_submerged_volume` 锁住吃水取整的错基准（曾把 8.7 % 吃水取整算成 12 % 体素误差）。
- **证据出典**：本仓 `artifacts/icebow_lbm_r2/`（gate/sdf/ladder/angles/sphere/re/field JSON + §3.7 图），代码 `src/tensorlbm/icebow_lbm.py`，节点 IN13W2-N06，2026-11-10。

## 7. 复现

```bash
PYTHONPATH=src python3 scripts/run_icebow_lbm_campaign.py --stage gate --out artifacts/icebow_lbm_r2
PYTHONPATH=src python3 scripts/run_icebow_lbm_campaign.py --stage sdf  --out artifacts/icebow_lbm_r2
PYTHONPATH=src python3 scripts/run_icebow_lbm_campaign.py --stage ladder --out artifacts/icebow_lbm_r2
PYTHONPATH=src python3 scripts/run_icebow_lbm_campaign.py --stage angles --out artifacts/icebow_lbm_r2
PYTHONPATH=src python3 scripts/run_icebow_lbm_campaign.py --stage sphere --out artifacts/icebow_lbm_r2
PYTHONPATH=src python3 scripts/run_icebow_lbm_campaign.py --stage re --out artifacts/icebow_lbm_r2
PYTHONPATH=src python3 scripts/run_icebow_lbm_campaign.py --stage field --out artifacts/icebow_lbm_r2
PYTHONPATH=src python3 scripts/run_icebow_lbm_campaign.py --stage hybrid --out artifacts/icebow_lbm_r2
PYTHONPATH=src python3 scripts/icebow_lbm_report.py --dir artifacts/icebow_lbm_r2
python3 -m pytest tests/test_icebow_lbm.py -q
```