# P82W5-N04 屈服应力 LBM 3D+移动壁（D3Q19 + Ladd 移动壁）

> 本文件由 `scripts/make_report_p82w5_n04.py` 生成的报告正文自动落盘（数字零手抄）。
> 判据数字与产物路径见 `artifacts/P82W5-N04/*.json`；图见 `figures/p82w5_n04/`。

## 1 案例介绍

屈服应力流体（Bingham / Herschel–Bulkley）在船–冰间隙（h/D）润滑阻力中承担「低于屈服应力即近似刚性」的关键角色。本节点把 TensorLBM 的屈服应力内核从 2D 静止壁推进到 **D3Q19 + 移动壁**，并在三类解析/高精度参考解上定量闭合：
① 3D 方管 Bingham 塞流剖面（无正则化的增广拉格朗日参考解）；② Bingham 平面 Couette–Poiseuille（一壁以 $U_w$ 移动 + 体积力，塞厚闭式 $\delta_p = 2\tau_y/G$）；③ $Bn\to0$ 牛顿极限。


几何与参数（lattice 单位，$\rho=1$，$c_s^2=1/3$）：方管 $N\times N$ 流体格，壁面位于 $y,z=0$ 与 $N+1$（半路反弹壁面位于格点面）；通道 $H$ 层流体、$x/z$ 周期、$y=H+1$ 层可移动。参数：$\mu_0=0.1$，$G=\rho a=1\times10^{-5}$（方管）、$2\times10^{-6}$（Couette–Poiseuille），$\tau_y=2\times10^{-5}$ / $4\times10^{-6}$，Papanastasiou 参数 $m=1\times10^{5}$，$U_w$ 由目标塞区位置反解得到。


## 2 方法与关键设置

* 内核：`src/tensorlbm/yield_stress_moving_wall_3d.py`（D3Q19 BGK + Guo 体积力，复用 `yield_stress_rheology_3d` 的 Papanastasiou/HB 应力与剪切率恢复）。
* 移动壁：**half-way bounce-back + Ladd 速度修正**（未用 Bouzidi）——本节点所有壁面都是落在格点面上的平面壁，此时半路反弹即二阶精度，Bouzidi 插值因子退化为 $q=1/2$；实现写作「stream 后覆写」：$f_i(x_b)=f_{\bar i}^{post}(x_b)+6w_i\rho(x_b)(c_i\cdot u_w)$，对每条 $x_b-c_i$ 为固体的连接。**选择声明**：Ladd/半路反弹方案，固体层每步重置为静止平衡态。
* 动量账本（MEM）：逐连接精确记账 $M=\sum c_i(f_i^{new}-f_i^{streamed}) = 2\sum c_i f_{\bar i}^{post} + \sum c_i 6w_i\rho(c_i\cdot u_w)$，给出离散恒等式 $P(t+1)-P(t)=M_{body}+M_{wall}$ 与稳态力平衡两条独立判据。
* 参考解：`src/tensorlbm/yield_stress_analytical_moving_wall.py`——Couette–Poiseuille 精确解（$\tau_0$ 二分到机器精度）、牛顿方管 Fourier 级数、Bingham 方管**无正则化**增广拉格朗日解（细网格节点与 LBM 格心完全重合）；纯 Couette 的退化性显式分类。
* 工具链：PyTorch float64 / `stream3d`（cat 后端）/ pytest / matplotlib。


参考解自身的网格收敛审计（同一物理问题、不同细网格 refine）：


| case | Q (per unit depth) | n_fine | AL iterations |
|---|---|---|---|
| N16_r8 | 0.094683 | [128, 128] | 4000 |
| N32_r8 | 2.561053 | [256, 256] | 4000 |
| N64_r4 | 49.932524 | [256, 256] | 4000 |
| N64_r8 | 49.881928 | [512, 512] | 4000 |


## 3 计算结果与判据

### (a) 3D 方管 Bingham 剖面 vs 无正则化参考解

| N | L∞ rel | L2 rel | Q rel | 参考解 refine | 力平衡 | 步数 |
|---|---|---|---|---|---|---|
| N8 | 53.00% | 28.09% | 7.56% | r4 | — | — |
| N16 | 19.24% | 18.48% | 18.86% | r8 | — | — |
| N32 | 2.35% | 2.91% | 3.22% | r8 | — | — |


判据 a：N≥32 时 $L_\infty\le3\%$ → **PASS**（实测 2.35%）


### (b) Bingham Couette–Poiseuille：塞厚与塞区刚性

| H | regime | U_w | δ_p 精确 | δ_p 应力交点 | 误差 | δ_p 阈值法 | 误差 | 塞区 CV | 剖面 L∞ |
|---|---|---|---|---|---|---|---|---|---|
| H16 | plug | 1.707e-04 | 5.33 | 4.575 | 14.23% | 8.00 | 50.00% | 0.006 | 10.95% |
| H32 | plug | 6.827e-04 | 10.67 | 9.432 | 11.58% | 16.00 | 50.00% | 0.006 | 6.84% |
| H64 | plug | 0.002731 | 21.33 | 19.149 | 10.24% | 31.00 | 45.31% | 0.004 | 3.97% |


正则化参数 m 扫描（同一 H，$M=m\tau_y/\mu_0$）：


| case | M | μ_plug | 塞区 CV | 剖面 L∞ |
|---|---|---|---|---|
| H16_m1e+04 | 0.53 | 0.153 | 0.014 | 47.72% |
| H16_m1e+06 | 53.33 | 5.433 | 0.011 | 10.83% |


结论：m 从 $10^5$ 增到 $10^6$ 时剖面误差不再改善、塞区 CV 反而变差——$m\dot\gamma\gg1$ 区 $\mu_{eff}=\mu_0+\tau_y/\dot\gamma$ 与 m 无关，继续加大 m 只增加数值刚度。


纯 Couette 退化控制（无体积力）：


| 情形 | τ_wall | 判定 | u_max/U_w 或 L∞ | 物理含义 |
|---|---|---|---|---|
| τ_w<τ_y (0.03τ_y) | 1.250e-07 | no_steady_solution | u_mean/U_w=0.491 | 理想 Bingham 无稳态解；正则化模型给出 $\mu_{eff}\approx\mu_0+m\tau_y$ 的蠕动线性剖面 |
| τ_w>τ_y | 5.000e-06 | fully_yielded | 0.03845 | 全间隙屈服、线性剖面 |


纯 Couette 控制说明：$\tau_w<\tau_y$ 分支在理想 Bingham 下**无稳态解**，正则化模型把它替换成 $\mu_{eff}\approx\mu_0+m\tau_y$ 的蠕动线性剖面（$u_{mean}\approx U_w/2$），因此任何「部分塞区」的 Couette 报告都是把不存在的解拟合出来的结果——这是本节点对历史口径的更正。


判据 b：$\delta_p$ 误差 ≤3% 且塞区 CV ≤1e-3 → **HONEST-FAIL**（δ_p 最大误差（应力交点估计量）14.23%，阈值法 50.00%，CV 最大 0.006）


### (c) 屈服面位置分辨率阶梯（纯压力驱动通道，$y_p=\tau_y/G$）

| N | y_p 精确 | y_p(u_max 反演) | 误差 | δ_p(应力交点) | 剖面 L∞ | 步数 |
|---|---|---|---|---|---|---|
| N8 | 2.000 | 1.8944 | 0.1056 | 1.149 | 22.86% | 900 |
| N16 | 2.000 | 1.9459 | 0.0541 | — | 4.82% | 4700 |
| N32 | 2.000 | 1.9704 | 0.0296 | — | 0.84% | 16200 |
| N64 | 2.000 | 1.9853 | 0.0147 | — | 0.18% | 53700 |


拟合斜率（log–log）：u_max 反演 -0.94，γ̇ 阈值交点 0.15，剖面 $L_\infty$ -2.36。剖面本身以 2.36 阶收敛（说明离散是二阶的），而**屈服面位置**这个泛函只以 ~1 阶收敛：$y_p=h-\sqrt{2\mu_0u_p/G}$ 在 $y_p\ll h$ 时条件数极大（$\partial y_p/\partial u_p\approx-\mu_0/(G(h-y_p))$），叠加正则化模型在屈服面附近的 $O(1/M)$ 偏差。

判据 c：斜率 ≥1.7 → **HONEST-FAIL**（取屈服面估计量的较大斜率 0.15）


### (d) MEM 动量账本

| 闭合量 | 实测 |
|---|---|
| force_balance_duct | 1.0000000000 |
| closure_window_duct | 1.0000000000 |
| closure_couette | 1.0000000013 |
| force_balance_cp | 1.0000000002 |
| closure_window_cp | 1.0000000000 |


判据 d：全部闭合率 ∈[0.97,1.03] → **PASS**


### (e) 2D/3D 流量比与矩形管形状因子

| case | τ_y | 形状因子 (LBM) | 形状因子 (解析) | 比值误差 | 参考解 | Q_3D 误差 |
|---|---|---|---|---|---|---|
| newtonian_limit_N16 | 1.000e-07 | 0.4204 | 0.4245 | 0.96% | fourier | 1.31% |
| newtonian_limit_N32 | 1.000e-07 | 0.4213 | 0.4226 | 0.32% | fourier | 0.40% |
| bingham_N16 | 2.000e-05 | 0.2297 | 0.2740 | 19.25% | 8 | 18.86% |
| bingham_N32 | 2.000e-05 | 0.3516 | 0.3603 | 2.48% | 8 | 3.22% |


判据 e：比值误差 ≤5% → **PASS**（判定集 bingham_N32,newtonian_limit_N16,newtonian_limit_N32，最大 2.48%；牛顿极限形状因子 $f=0.4217$）


### (f) Bn=0 牛顿极限

| case | L∞ rel | 步数 | 力平衡 | 形状因子 |
|---|---|---|---|---|
| duct_N16 | 0.58% | 2500 | 1.00000000 | 0.4196 |
| couette_H16 | 0.00% | 100 | — | — |
| duct_N32 | 0.15% | 8400 | 1.00000000 | 0.4212 |
| couette_H32 | 0.00% | 100 | — | — |


判据 f：≤1% → **PASS**（最大 0.58%）


## 4 注意点（踩过的坑与局限）

1. **纯 Bingham 平面 Couette 不存在部分塞区**：零压力梯度 ⇒ 剪应力沿间隙均匀，要么全间隙屈服（线性剖面，$\delta_p=0$），要么无稳态解。历史实现中的 $\delta_p=H-\mu_0U_w/\tau_y$ 不是 Couette 问题的解；本节点改用 Couette–Poiseuille 作塞厚闭式锚（$\delta_p=2\tau_y/G$），并显式分类纯 Couette 两种退化情形。
2. **Papanastasiou 的 $m$ 在 $m\dot\gamma\gg1$ 区失效**：此时 $\mu_{eff}\to\mu_0+\tau_y/\dot\gamma$ 与 $m$ 无关，继续增大 $m$ 不改善塞区；塞内残余剪切为 $\dot\gamma\approx(\tau_y/\mu_0)e^{-M}$，$M=m\tau_y/\mu_0$。
3. **剪切率恢复方式决定塞区行为**：由非平衡应力反解时 `|Π|~τ_y` 把等效黏度钉在 $\mu_0+\tau_y/\dot\gamma_{min}$，塞区不可能刚性；由速度梯度反解才能让塞区黏度升到 $\mu_0+m\tau_y$。本节点两种模式都保留（`use_neq`）并分别报告。
4. **稳态账本必须取最后一窗**：整段平均会把力平衡闭合率压到 ~0.98（起点瞬态贡献）。
5. **移动壁 Ladd 项的 $\rho$ 因子**：缺 $\rho$ 在 $\rho\approx1$ 时不可见，但密度有 20% 偏差时即为 33% 的修正误差（2D 侧审计结论，3D 沿用同一实现）。
6. 局限：壁面限定为格点面上的平面壁；方管 Bingham 的屈服面是闭合曲线，在 2 格厚屈服层下 $L_\infty$ 由近壁区主导（见 (a) 的误差分布）；高 $M$ 时塞区黏度可达 $\mu_0(1+M)$，需相应放宽 `nu_max`。


## 5 误差判定（逐项、显式回答是否 <3%）

| 判据 | 实测 | 门槛 | 满足？ |
|---|---|---|---|
| (a) 3D 方管 $L_\infty$ | 2.35% | ≤3% | 是 |
| (b) 塞厚 $\delta_p$ | 14.23% | ≤3% | 否 |
| (b) 塞区 CV | 0.006 | ≤1e-3 | 否 |
| (c) 收敛斜率 | 0.15 | ≥1.7 | 否 |
| (d) 动量闭合 | 见上表 | [0.97,1.03] | 是 |
| (e) 2D/3D 比值 | 2.48% | ≤5% | 是 |
| (f) Bn=0 | 0.58% | ≤1% | 是 |


合计 **4/7** 项满足。未满足项按纪律如实保留为 HONEST-FAIL，不以后处理掩盖：其数值含义、成因与后续路线写在第 4 节。


## 6 复现与产物

* 内核：`src/tensorlbm/yield_stress_moving_wall_3d.py`
* 参考解：`src/tensorlbm/yield_stress_analytical_moving_wall.py`
* 文档：`docs/yield_stress_moving_wall_3d.md`
* 基准：`scripts/bench_yield_stress_moving_wall_3d.py`（JSON 落 `artifacts/P82W5-N04/`，场数据 `.npy` 同名）
* 图：`scripts/gen_yield_stress_moving_wall_figures.py` → `figures/p82w5_n04/`
* 测试：`tests/test_yield_stress_moving_wall_3d.py`（11 项，含解析恒等式、AL 参考解交叉验证、移动壁牛顿锚、账本恒等式、质量守恒）

