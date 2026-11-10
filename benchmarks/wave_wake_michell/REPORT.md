# P96W2-N06 · LBM 船舶兴波阻力与 Michell 薄船理论对标 —— 兴波阻力验证面

**仓**: TensorLBM · **分支**: `agent/m8148` · **节点**: P96W2-N06（物理域）·
**产物**: `src/tensorlbm/wave_wake_michell.py`、`tests/test_wave_wake_michell.py`、
`benchmarks/wave_wake_michell/{run.py, run_lbm_dispersion.py, results.json, figures/}`

> **一句话结论**：建成了**可复算的兴波阻力验证面**（Michell 积分 + Kelvin 波系几何 +
> hump/hollow + 片体干涉），用**闭式谱**做独立仲裁抓出并修掉了本模块 `f_x` 的 2 倍错误；
> Kelvin 散波角、浅水 Mach 角、Fn 标度不变性全部数值复现。**但**「LBM 自由面算出的 C_W
> 与 Michell 比 <15%」这条主线判据**未取得**：3-D Körner 自由面内核在控制面上实测
> **9.56 s/step @ 4e5 cells**，外推一个 λ/Δx≥20 的 Wigley 算例需 **≈35 h**，超出本单预算；
> 2-D 自由面机制（同一套质量账本）被用作替代验证，实测周期误差 **8.8–13.8%**、
> 1000 步质量漂移 **<0.7%**。C1 报 **WITHHELD**（不是 PASS），honest FAIL 不掩盖。

---

## ① 案例介绍

**物理背景**。兴波阻力 R_W 是总阻力里唯一不能用粘性/经验回归解释的分量：它由船体拖曳的
定常自由面波系决定，唯一的闭式锚是 Michell（1898）薄船积分。本节点要建立的对标面由三块组成：
(a) Michell 积分的 C_W(Fn)；(b) 波系几何（Kelvin 散波角、横波波长 λ=2πU²/g）；
(c) 干涉/峰谷结构（hump/hollow、双体片体干涉因子）。

**几何**（全部用闭式 Cb 与数值体积分双算，几何先验证再谈物理）：

| hull | L [m] | B [m] | T [m] | Cb closed | Cb numeric | Cb error | S_wet [m2] | L/B | B/T | thin-ship |
|---|---|---|---|---|---|---|---|---|---|---|
| wigley-standard | 1.000 | 0.1000 | 0.0625 | 0.444444 | 0.444444 | -0.00001% | 0.14879 | 10.00 | 1.60 | admitted |
| wigley-III | 3.048 | 0.4060 | 0.1520 | 0.554997 | 0.554997 | -0.00002% | 1.46083 | 7.51 | 2.67 | admitted |

两船型都落在薄船准入窗内（Cb≤0.60、L/B≥5）；Cb 闭式与数值体积分相差 ≤2e-5 %，
说明采样网格与解析形状一致（平台规矩：几何未验证前不许出 C_W）。

**参数**：ρ=1000 kg/m³, g=9.81 m/s²，Fn=0.20/0.30/0.40，深水（λ/2 ≪ 水深）。

## ② 怎么做的

1. **Michell 积分**（`michell_resistance`）：u=tanθ 变量代换（重量 = sec³θ dθ = sec u du），
   对 (n_x, n_z, n_theta) 三级加密；源强度用**解析导数** f_x = (B/2)(−8x/L²)g(ζ)，
   而非差分。C_W = R_W /(0.5ρU²S)，S 由双面积分 2∬√(1+f_x²+f_z²)dxdz 给出。
2. **闭式谱仲裁**（`michell_spectrum_analytic`）：抛物型 Wigley 的谱 P+iQ 可对 x、z **逐项
   解析积出**（I_z = T[1/a − 2/a³ + e**(-a)(2/a²+2/a³)]，X = −iL cos(mL/2)/m + 2i sin(mL/2)/m²），
   只对 θ 做数值积分 → 与采样实现**零共享离散**，两者一致即同时验证了 f_x、e^{κz} 与轴向相位。
3. **波系几何**（`caustic_angle_deg`）：远场相位 F(θ)=κ(θ)cos(θ+α)，κ=k0sec²θ；驻点存在的最大
   α 即 caustic，用二分求解 —— **不调用 sin=1/3**；再用 `caustic_angle_analytic_deg`
   （极大化 tanα=u/(2u²+1)）与浅水射线包络 `shallow_ray_envelope_deg` 两条独立路线交叉验证。
4. **波系合成**（`wake_field`）：驻相合成 Σ|I(θ_j)|cos(κ_j r cos(θ_j+α)+arg I)，
   在轴线上做 FFT 取横波波长，与 2πFn²L 比对。
5. **hump/hollow**（`hump_hollow`）：平滑后取斜率变号点 + 抛物线顶点细化 + **地形显著度**
   （与两侧最近的异类极值比）过滤抖动。
6. **干涉**（`catamaran_interference`）：薄船叠加 abs(I_cat)²=2abs(I)²(1+cosφ)，φ=k0secθ·Δx+k0secθtanθ·Δy，
   按 sec³θ|I|² 加权求谱平均；同时给出教科书单波近似。
7. **LBM 自由面支线**（`run_lbm_dispersion.py`）：2-D Körner GALS（与 3-D 同一套质量交换/相变规则），
   正弦自由面位移 + 静水压密度初场，1000 步，测质量账本漂移、自由面周期、λ/Δx=32/64 两档。
   3-D 船级算例的**成本墙**在 `blocker_measurements` 里实测记录。

## ③ 计算结果

**Michell 兴波阻力梯级**

| hull | Fn | k0 L | R_W [N] | C_W |
|---|---|---|---|---|
| wigley-standard | 0.20 | 25.000 | 2.609715e-02 | 8.939606e-04 |
| wigley-standard | 0.30 | 11.111 | 1.406573e-01 | 2.141436e-03 |
| wigley-standard | 0.40 | 6.250 | 3.175842e-01 | 2.719721e-03 |
| wigley-III | 0.20 | 25.000 | 8.602795e-01 | 9.847475e-04 |
| wigley-III | 0.30 | 11.111 | 5.706741e+00 | 2.903293e-03 |
| wigley-III | 0.40 | 6.250 | 1.377033e+01 | 3.940668e-03 |

**采样求积 vs 闭式谱（同一理论、不同离散）**

| Fn | grid (n_x x n_z x n_theta) | C_W sampled | C_W closed form | deviation |
|---|---|---|---|---|
| 0.20 | 161x61x600 | 8.939606e-04 | 8.875623e-04 | +0.721% |
| 0.20 | 321x121x1200 | 8.891647e-04 | 8.875623e-04 | +0.181% |
| 0.30 | 161x61x600 | 2.141436e-03 | 2.141666e-03 | -0.011% |
| 0.30 | 321x121x1200 | 2.136796e-03 | 2.141666e-03 | -0.227% |
| 0.40 | 161x61x600 | 2.719721e-03 | 2.733928e-03 | -0.520% |
| 0.40 | 321x121x1200 | 2.713061e-03 | 2.733928e-03 | -0.763% |

**求积极限（quadrature refinement）**

| hull | level | grid (n_x x n_z x n_theta) | C_W | vs finest |
|---|---|---|---|---|
| wigley-standard | 1 | 81x31x300 | 2.152677e-03 | +0.743% |
| wigley-standard | 2 | 161x61x600 | 2.141436e-03 | +0.217% |
| wigley-standard | 3 | 321x121x1200 | 2.136796e-03 | +0.000% |
| wigley-standard | converged | - | - | +0.217% |
| wigley-III | 1 | 81x31x300 | 2.920990e-03 | +0.858% |
| wigley-III | 2 | 161x61x600 | 2.903293e-03 | +0.247% |
| wigley-III | 3 | 321x121x1200 | 2.896154e-03 | +0.000% |
| wigley-III | converged | - | - | +0.247% |

**三方交叉验证（Fn=0.30）** —— 闭式谱为此模块的独立仲裁

| hull (Fn=0.30) | C_W this module | C_W closed form | dev vs closed form | C_W ship_resistance_closure | dev vs that |
|---|---|---|---|---|---|
| wigley-standard | 2.141436e-03 | 2.141666e-03 | -0.011% | 2.160719e-03 | -0.892% |
| wigley-III | 2.903293e-03 | n/a | n/a | 2.160719e-03 | +34.367% |

> 注：`wigley-III` 与 `ship_resistance_closure` 的对比**不可用**：后者只带标准 Wigley
> 几何（`hull_section_grid("wigley")`），与 III 型不是同一条船；该行的偏差是几何差异，
> 不是实现差异。标准 Wigley 一行才是实现对比（−0.89%）。

**hump / hollow（Michell 曲线，Fn=0.20–0.60）**

| type | Fn | C_W | relative prominence |
|---|---|---|---|
| hump | 0.2404 | 1.197678e-03 | 0.157 |
| hump | 0.3002 | 2.042975e-03 | 0.373 |
| hump | 0.4988 | 4.448513e-03 | 0.137 |
| hollow | 0.2103 | 8.135950e-04 | 0.086 |
| hollow | 0.2589 | 1.009846e-03 | 0.186 |
| hollow | 0.3462 | 1.281408e-03 | 0.594 |

**Kelvin 波系几何**

| quantity | value | reference | deviation |
|---|---|---|---|
| caustic angle (bisection on stationary points) | 19.471220728 deg | 19.471220634 deg (sin = 1/3) | +9.38e-08 deg |
| caustic angle (analytic max tan a = u/(2u^2+1)) | 19.471220634 deg | same | -6.09e-10 deg |
| wave-vector direction at the caustic | 35.2642 deg | arctan(1/sqrt 2) = 35.2644 deg | -2.22e-04 deg |
| on-axis wavelength (FFT of the synthesised field) | 0.571429 m | 2 pi Fn^2 L = 0.565487 m | +1.051% |

**浅水 Mach 锥（深水 Kelvin 角的正交分支）**

| Fr_h = U/sqrt(gh) | arcsin(1/Fr_h) | ray envelope | deviation |
|---|---|---|---|
| 1.2 | 56.442690 deg | 56.442690 deg | -1.79e-07 deg |
| 1.6 | 38.682187 deg | 38.682187 deg | -2.46e-10 deg |
| 2.0 | 30.000000 deg | 30.000000 deg | -4.53e-08 deg |
| 3.0 | 19.471221 deg | 19.471221 deg | -5.50e-08 deg |

> 战役判据 C3 写的 `sin θ = 1/Fn` 是**浅水**（非色散）Mach 锥关系，不是深水 Kelvin 角
> （19.4712°，sin=1/3，与 Fn 无关）。两条分支此处都算并都验过：Fr_h=3.0 时浅水式给出
> 19.4712°，恰好与深水 Kelvin 角重合 —— 这大概是判据里公式混用的来源。

**双体片体干涉因子（Fn=0.30）**

| mode | S/L | full spectrum | single-wave approximation | deviation |
|---|---|---|---|---|
| transverse | 0.2 | 1.3920 | 2.0000 | -30.4% |
| transverse | 0.3 | 1.1692 | 2.0000 | -41.5% |
| transverse | 0.4 | 0.9369 | 2.0000 | -53.2% |
| tandem | 0.2 | 0.3291 | 0.3937 | -16.4% |
| tandem | 0.3 | 0.4202 | 0.0183 | +2192.8% |
| tandem | 0.4 | 1.1110 | 0.7353 | +51.1% |

**LBM 自由面支线（2-D，同一套 Körner 质量账本）**

| lambda/dx | grid | mass drift / 1000 steps | measured period | linear-theory period | period error |
|---|---|---|---|---|---|
| 32 | 128x28 | -0.6866% | 330 | 290 | +13.80% |
| 64 | 192x28 | -0.2584% | 390 | 427 | -8.75% |

**3-D 自由面成本墙（实测，非估计）**

| measurement | value |
|---|---|
| 3-D Körner kernel, 393 216 cells, default threads | 9.562 s/step |
| 2-D kernel, 24 576 cells, 1 / 4 / 16 threads | 1: 0.0888 / 4: 0.0497 / 16: 0.2487 s/step |
| extrapolated Wigley case (2.1e6 cells, 2e4 steps, 4 threads) | 35.4 h |

## ④ 结果图

**图 1. cw_vs_fn.png** — `benchmarks/wave_wake_michell/figures/cw_vs_fn.png`

**图 2. kelvin_wake.png** — `benchmarks/wave_wake_michell/figures/kelvin_wake.png`

**图 3. catamaran_interference.png** — `benchmarks/wave_wake_michell/figures/catamaran_interference.png`

**图 4. quadrature_and_crosscheck.png** — `benchmarks/wave_wake_michell/figures/quadrature_and_crosscheck.png`

**图 5. lbm_free_surface_leg.png** — `benchmarks/wave_wake_michell/figures/lbm_free_surface_leg.png`

（图由 `benchmarks/wave_wake_michell/run.py` 用 matplotlib 生成，程序化注入，无手绘。）

## ⑤ 注意点 / 踩过的坑 / 局限

1. **抓到的真 bug**：模块 `half_beam_dx` 最初写成 (B/2)(−4x/L²)g（少一个 2），使 R_W 低 4 倍。
   三处独立证据锁定它：闭式谱、`ship_resistance_closure`（有限差分 + 逐 θ 循环）、
   以及 B² 标度检查。修后采样法与闭式谱差 **-0.011%**。
   → 教训：**任何薄船实现都必须有一个闭式谱仲裁**，否则 2 倍因子错误会被 C_W 的 4 倍误差
   静静吃掉，而所有单调性/收敛性测试仍然通过。
2. **sin θ=1/Fn 是浅水式**（见上），深水 Kelvin 角恒为 19.4712°；C3 若按 1/Fn 判会在 Fn≥0.3 全部
   「FAIL」，那是判据写错而不是物理错。
3. **包络角测量不可作为 caustic 估计**：驻相合成保留楔外非驻相泄漏，
   `measure_wake_angle_deg` 读到 33.33°（阈值 0.5），系统性高估。
   因此 caustic 由「驻点存在性二分 + 解析极大化」双路线给出，包络测量只作为**负面结果**登记。
4. **单波（教科书）干涉近似在其零点附近无用**：tandem 在 S/L=0.3 时 1+cos(k0S)≈0.018，
   全谱给 0.42，比值 +2192% —— 这不是物理矛盾，而是「只取一个波长」的近似在相位抵消点
   必然失效；工程上要么用全谱，要么声明误差量级。
5. **薄船理论的双体干涉不衰减**：横波系统只有单一波长 2πU²/g，相位差与间距无关，
   大间距下因子回到 1+cos(k0S) 而不是 1（实测 S/L=40 时 1.039）。真实船靠粘性/非线性解相关，
   薄船理论给不了 —— 用干涉因子时必须声明这一点。
6. **2-D ≠ 3-D**：2-D 支线只验证「机制」（质量账本、色散、相变规则），不产出船级 C_W。
   其周期误差 8.8–13.8%（λ/Δx=64/32）、质量漂移 <0.7%，说明距 3% 级对比还差一个量级。
7. **线程超订**：Körner 内核在 24k cells 上 1/4/16 线程为 0.089/0.050/0.249 s/step ——
   默认 128 线程比 4 线程慢 5 倍。任何 3-D 自由面预算表都必须先做这个测量。
8. **适用范围**：Cb≤0.60、L/B≥5、深水、微幅（薄船一阶）。T 型/方尾/跨声速（Fn>0.6）不行。

## ⑥ 误差判定（逐项 vs 基准，显式回答「是否 <3%」）

| criterion | status | evidence / reason |
|---|---|---|
| C1_wigley_cw_vs_michell | WITHHELD | the numerical (LBM free-surface) C_W does not exist at ship scale in this budget; the Michell reference itself is delivered and verified (quadrature convergence 0.22%, agreement with the closed-form spectrum 0.01%, with ship_resistance_closure 0.89%) |
| C2_fn_scaling_hump_hollow | PARTIAL | hump/hollow positions are extracted from the Michell curve and compared with the transverse-wave-only prediction; the Froude-scaling invariance is verified (max C_W change under a 3.048x scale change: 0.000%). The LBM-vs-Michell leg is withheld with C1. |
| C3_kelvin_angle | PASS | caustic angle reproduced to 9.38e-08 deg (gate 5 deg) by two independent routes; on-axis wavelength +1.05% of 2 pi Fn^2 L; shallow branch arcsin(1/Fr_h) reproduced to <=1.8e-07 deg |
| C4_series60_wave_profile | NOT_ATTEMPTED | requires a 3-D free-surface elevation field with experimental reference (Noblesse et al. 1986); blocked by the same cost wall as C1, and the module does not carry the Series-60 hull family (thin-ship offsets only for the Wigley family) |
| C5_catamaran_interference | PARTIAL | interference factor delivered from thin-ship superposition at S/L=0.2/0.3/0.4 (transverse and tandem) together with the textbook single-wave approximation; the numerical (LBM) leg is withheld with C1, so this is a theoretical reference, not a solver comparison |
| C6_grid_convergence | PARTIAL | quadrature refinement of the Michell integral converges to 0.22%; the free-surface resolution requirement is audited on the 2-D LBM mechanism (lambda/dx = 32/64, period error 13.8% / 8.8%) |
| C7_mass_conservation | PASS | 2-D Korner ledger drift 0.687% over 1000 steps (gate 1%) at lambda/dx = 32 and 64 |
| C8_regression | PASS | tests/test_wave_wake_michell.py (new) + existing LBM suite run in the report |

**逐项误差与 3% 门**：

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| C_W（采样 vs 闭式谱，全部 3 个 Fn、2 档网格） | 闭式解析谱 | 最大 0.763% | 是（见下表，需随网格报告） |
| C_W（本模块 vs ship_resistance_closure，Fn=0.3） | 独立实现 | 0.892% | 是 |
| 求积收敛（中→细） | 自身序列 | 0.217% | 是 |
| Kelvin caustic 角 | sin=1/3 = 19.4712° | 9.38e-08 deg (4.82e-07%) | 是 |
| 浅水 Mach 角（Fr_h=1.2/1.6/2.0） | arcsin(1/Fr_h) | ≤1.8e-07 deg | 是 |
| 横波波长（合成波场 FFT） | 2πFn²L | 1.05% | 是（1.05%，受 400 点 FFT 栅格限制） |
| Fn 标度不变性（3.048× 缩尺） | 自身 | 1.8e-13% | 是 |
| **C_W（LBM 自由面 vs Michell）** | Michell | **无数据** | **不适用 —— WITHHELD** |
| 自由面质量账本（2-D，1000 步） | 初始存量 | 0.687% | 是 |
| 自由面周期（2-D，λ/Δx=64/32） | √(gk tanh kh) | 13.80% | **否**（这是 LBM 自由面尚不能支撑 3% 级对比的直接证据） |

**诚实结论**：解析面（Michell/波系/hump/干涉）已达 3% 以内并可复算；**数值面未达**。
C1（LBM C_W vs Michell<15%）、C4（Series 60 波高分布）本单**未取得数据**，报 WITHHELD /
NOT_ATTEMPTED，并给出实测成本墙（≈35 h/算例 @ 4 线程）作为下一步的预算依据。
