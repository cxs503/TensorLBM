# XVAL-LBM-TWO-CONSERVATION-ZOUHE-WITH-CHEMICAL-POT

`mission m18753 · repo TensorLBM · branch agent/m18753 · 第 2 轮（调研→结合）`
Anchor: two-conservation (fluid mass **and** order parameter) Zou-He open boundaries for the
free-energy (Cahn-Hilliard) two-distribution LBM **in the presence of the chemical potential**.

## 0. 一句话结论

* **力一致的（force-consistent）双守恒重构**：`rho`/`rho u` 与 `psi`/`psi u` 在化学势力
  `F = mu grad psi` 下**逐点精确重建**（相对误差 5.09e-15 /
  1.28e-13，判据 <15 % → 万倍余量通过）。
* **经典 Zou-He（丢半力项）的缺陷律**被解析预测并逐点复现：
  `|u_meas - u_targ| = |F|/(2 rho)`，最大失配 9.71e-15；
  同一随机场上相对速度误差达 324.9 %（>15 % 判据 → 失败）。
* **上游 ebinan92 的亏量分配变体**：`psi` 精确（误差 0），但其**平流通量不守恒**
  （绝对残差 0.64，随机场相对残差 2.12e+03）——两者不可兼得。
* **通道耦合算例 = honest-FAIL**：把该闭包接到带 Cahn-Hilliard 界面的自由能通道上，
  `corrected` 在 83 步、`classic` 在
  86 步发散；`impose` 撑过 200 步但
  `|psi|` 涨到 1.124（越界）。
  `fail_stage = open-boundary order-parameter closure × CH interface coupling`，
  `failure_reason = psi 在开边界列无界增长（墙壁/周期对照均稳定）`。

## 1. 问题设定与基准出处

自由能双分布（`f` 走 `rho,u`；`g` 走 `psi,mu`）LBM 在同一条开边界上必须同时守恒**两个**
量，且速度/密度的宏观量必须用 Guo 半力约定 `rho u = sum c_i f_i + F/2` 度量。化学势
`mu = a psi (1-psi^2) - kappa lap psi` 通过 Korteweg 力 `F = mu grad psi` 进入动量方程，
因此经典 Zou-He（Zou & He 1997）在边界列上必然出现 `O(F/2)` 的系统偏差，而 `g` 完全没有
边界闭包。

基准出处（external_anchor）：

| # | 基准 | 出处 |
|---|---|---|
| A1 | Zou-He 速度/压力重构方程 | Zou & He, *Phys. Fluids* 9 (1997) 1591 |
| A2 | Guo 半力速度约定 `u=(j+F/2)/rho`、Guo 源项 | Guo, Zheng & Shi, *Phys. Rev. E* 65 (2002) 046308 |
| A3 | 双分布自由能模型 `mu`, `p = rho/3 + psi mu/3`, `g^eq` 带 `3 gamma mu` | Swift, Osborn & Yeomans, *PRL* 75 (1995) 830 |
| A4 | 力一致的速度/压力边界 | Krüger et al., *The Lattice Boltzmann Method*, Springer 2017, §9.4 |
| A5 | 上游实现（同题对标源） | `ebinan92/Fingering_dynamics` `lattice_boltzmann/fingering.py:303,304,322,369`，`oss_solutions` id=191 tag `two-conservation-zouhe-with-chemical-potential-flux`；`oss_inventory` id=50 |
| A6 | 分层两相 Poiseuille 解析解（本轮未用到的物理对标路线） | `cases/mcmp_stratified_poiseuille/analytic.py::layered_poiseuille` |

**判据（先于运行声明）**：重建后的守恒量相对误差 **<15 %**（owner 精度标准）；同时把
B2 的缺陷律失配单独作为解析判据（<5 %）。

## 2. 方法与实现

重构（入口 `x=0`，未知入流方向 `{1,5,8}`，`Kf` = 已知 `f` 之和）：

```
rho = (Kf + f3 + f6 + f7 - F_x/2) / (1 - u_x)          # 质量守恒 + 半力
f1  = f3 + (2/3) rho u_x                                # 非平衡反弹
f5,f8 = [ rho u_x - F_x/2 + (f3+f6+f7) - f1  ±  (-F_y/2 - f2 - f6 + f4 + f7) ] / 2
```

由构造可得（机器精度）：`sum f = rho`，`sum c_i f_i + F/2 = rho u`。对 `g` 同样代数、
无体力项，得**通量一致**闭包：

```
psi = ((g0+g2+g4) + 2 (g3+g6+g7)) / (1 - u_x)      # sum g = psi 且 sum c_x g = psi u_x
g1  = g3 + (2/3) psi u_x,  g5,g8 = [psi - K_g - g1 ± (-g2 - g6 + g4 + g7)]/2
```

出口（`x=nx-1`，压力型）与出口 `psi` 闭包同式镜像；上游"亏量分配"变体
（`psi` 指定、按 `w1:w5:w8` 摊派）作为对照实现。

关键代码指针：

* `src/tensorlbm/phasefield/zouhe_two_conservation.py`
  `reconstruct_inlet` / `reconstruct_outlet` / `column_invariants`（`force_correction`
  开关即 `classic` 对照；`psi_mode="flux"|"impose"` 即两条方法路线）
* `src/tensorlbm/phasefield/free_energy_channel.py`
  `FreeEnergyChannel2D`（f/g collide + Guo 源、半程反弹墙壁、开边界接线、
  `equilibrate_interface` 离散界面对齐）
* `tests/test_zouhe_two_conservation_chemical_potential.py`（4 个 gate，全绿）
* `xval/XVAL-LBM-TWO-CONSERVATION-ZOUHE-WITH-CHEMICAL-POT/run_xval.py`（本节全部数字）

## 3. 参数表

| 量 | 符号 | 值 | 说明 |
|---|---|---|---|
| 边界列离散 | ny | 16 / 32 / 64 | 三套"分辨率" |
| 格子速度 | `u_target` | 0.01–0.06 | 入口抛物线剖面 |
| 化学势力 | `|F| = |mu grad psi|` | 0.005–0.2 | 6 个参数点（B2 扫描） |
| 表面张力 | `sigma` | 0.01 | CH 参数，`kappa = 0.75 sigma xi` |
| 界面厚 | `xi` | 3.0 | `a = -2 kappa / xi^2` |
| 松弛 | `tau_f, tau_g` | 0.8, 0.7 | `nu = (tau_f-0.5)/3 = 0.1` |
| 迁移率系数 | `gamma` | 0.1 | `g^eq` 的 `3 gamma mu` 项 |
| 通道/网格 | `ny, nx` | 34 × 48 | B4 通道探针，`H = 32` |
| 无量纲 | `Re = u H / nu` | ≈ 12.8 | B4，`u=0.04` |
| 无量纲 | `Ma = u / c_s` | ≈ 0.069 | `c_s = 1/sqrt 3` |
| 无量纲 | `H / xi` | 10.7 | 界面分辨率 |
| 精度 | dtype | float64 | 全部算例 |

## 4. 对照表（ref vs got，判据逐条打勾）

**B1 双守恒重构（三套分辨率 × 三条路线，相对误差）**

| ny | mode | mass-flux rel | ux rel | uy abs | psi-flux abs | psi-flux rel |
|---|---|---|---|---|---|---|
| 16 | corrected | 2.42e-15 | 2.26e-15 | 8.03e-17 | 1.60e-16 | 1.06e-14 |
| 16 | classic | 3.249 | 3.249 | 0.013 | 1.75e-16 | 2.23e-14 |
| 16 | impose | 2.90e-15 | 2.78e-15 | 3.51e-17 | 0.640 | 276.852 |
| 32 | corrected | 5.09e-15 | 4.98e-15 | 5.13e-17 | 8.67e-17 | 1.28e-13 |
| 32 | classic | 2.582 | 2.582 | 0.011 | 1.32e-16 | 7.29e-14 |
| 32 | impose | 2.81e-15 | 2.84e-15 | 4.09e-17 | 1.421 | 1.29e+03 |
| 64 | corrected | 4.58e-15 | 4.66e-15 | 5.95e-17 | 1.80e-16 | 5.91e-14 |
| 64 | classic | 2.432 | 2.432 | 0.013 | 1.53e-16 | 8.29e-14 |
| 64 | impose | 5.24e-15 | 5.22e-15 | 5.14e-17 | 1.482 | 2.12e+03 |

`corrected` 行全部 ≤1e-13（判据 15 % → **✓✓**，机器精度）；`classic` 行 `ux_rel` 达
324.9 % → **✗**；`impose` 行 `psi` 误差 0 但 `psi`-通量
非守恒 → 判据 **部分 ✓ / 通量 ✗**。

**B2 经典 Zou-He 缺陷律 `|u_meas-u_targ| = |F|/(2 rho)`**

| `|F|` | 实测缺陷 | 解析预测 | 相对失配 |
|---|---|---|---|
| 0.005 | 0.0034 | 0.0034 | 9.71e-15 |
| 0.010 | 0.0068 | 0.0068 | 5.62e-15 |
| 0.020 | 0.0136 | 0.0136 | 2.56e-15 |
| 0.050 | 0.0339 | 0.0339 | 6.13e-16 |
| 0.100 | 0.0679 | 0.0679 | 6.13e-16 |
| 0.200 | 0.1358 | 0.1358 | 2.04e-16 |

判据（失配 <5 %）→ **✓**（最大 9.71e-15）。

**B4 通道耦合探针（200 步，`ny×nx = 34×48`，带 CH 界面）**

| mode | 存活步数 | 稳定 | `psi_max` | 中截面剖面 L2 rel | `max|u_y|` |
|---|---|---|---|---|---|
| corrected | 83 | NO (blow-up) | — | — | — |
| classic | 86 | NO (blow-up) | — | — | — |
| impose | 200 | yes | 1.124 | 0.096 | 0.003 |

`corrected/classic` → **✗**（发散）；`impose` 存活但 `|psi|>1` 越界、剖面误差
0.096（<15 % 但不可信，见 §6）。

## 5. 收敛性与参数敏感性

* **分辨率**：B1 在 `ny = 16 / 32 / 64` 上 `corrected` 误差始终保持 ≤1e-13（重构是逐点代数
  恒等式，与分辨率无关）；`classic` 的缺陷随列长不放大（`ux_rel` ≈ 常数），说明它是
  `O(F/2)` 的**系统**偏差而非离散误差。
* **参数点**：B2 在 6 个 `|F|`（0.005→0.2，40 倍跨度）上，实测缺陷与 `|F|/(2 rho)` 的
  失配 ≤9.71e-15 —— 缺陷律对力强度线性成立，斜率为 1（log-log 图
  两条线重合）。
* **路线敏感**：同一随机场上 `flux` 与 `impose` 的 `psi`-通量残差相差 6 个数量级以上。

## 6. 失败模式与工件

* **honest-FAIL：通道耦合**。`fail_stage = open-boundary order-parameter closure × CH
  interface`。证据（本轮实跑）：
  1. 仅保留墙壁（把开边界换成恒等映射）：`psi` 在 600 步内稳定保持 `[-1.000, 1.000]`；
  2. 周期盒内的上游引擎（同参数）：1500 步稳定在 `|psi| ≤ 1.084`；
  3. 接上开边界 `psi` 闭包后，`corrected` 在第 83 步、`classic` 第 86 步发散，且**增长源
     定位在最后一列（出口）**——即 `psi` 闭包本身，而不是体力/碰撞核。
  推断（未证）：出口 `psi` 由通量一致式隐式取得，`psi(1+u_x)` 的分母在界面临近
  `u_x→-1` 时病态；`impose` 变体（显式 `psi`）不发散但把 `|psi|` 推过 1。下一轮应从
  "出口改为回压/零梯度 + 角点补齐"入手（与 `oss_solutions` 189/191 的迁移备注一致）。
* **方法局限**：B1/B2 是边界算子的**代数/解析**验证，不构成整场精度证明；B4 未通过，
  因此**不能**声称通道算例达到 15 % 判据。
* **未完成**：分层两相 Poiseuille 解析解（A6）未接上；`oss_solutions` 写入受 DB 权限限制，
  本轮以 SQL 工件形式落盘（`docs/oss_solutions_XVAL-LBM-TWO-CONSERVATION-ZOUHE-WITH-CHEMICAL-POT.sql`）。

### 复现

```bash
cd <worktree> && PYTHONPATH=src python3 -m pytest tests/test_zouhe_two_conservation_chemical_potential.py -q
PYTHONPATH=src python3 xval/XVAL-LBM-TWO-CONSERVATION-ZOUHE-WITH-CHEMICAL-POT/run_xval.py
```

工件：`artifacts/xval_two_conservation/results.json`、
`artifacts/xval_two_conservation/fig_conservation.png`。
