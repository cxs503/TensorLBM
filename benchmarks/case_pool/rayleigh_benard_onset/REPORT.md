# CASE-LBM-RB-ONSET-RA1708 — Rayleigh-Bénard 对流起始临界 Rayleigh 数（TensorLBM 算例池）

> 算例池锚点 `CASE-LBM-RB-ONSET-RA1708`，repo=TensorLBM，ref=Chandrasekhar (1961) Hydrodynamic and Hydromagnetic Stability, §II (linear stability theory, rigid-rigid isothermal walls, 2-D rolls)，tol_policy=<2%。
> 本文件由 `make_report.py` 从 `summary.json` **程序化注入**生成（零手抄）；所有原始数字见 `summary.json` 与 `route*.json`。

## 判定（两级，如实分列）

| 判据族 | 结果 | 内容 |
|---|---|---|
| **算例池容差 <2%（本案验收判据）** | **FAIL** | 主网格 ny=64 Ra_c=1731.9 （+1.41%）、1/ny 外推 Ra_inf=1755.3（+2.79%） |
| 交叉校验（Route B/C，非容差判据） | control ✅ / eigen ✅ | 连续谱控制臂 Ra_c=1707.768；格点算子本征臂见 ④ |

逐项检查（`summary.json::checks`，机器可读）：

| 检查 | 结果 |
|---|---|
| `primary_grid_within_tol` | ✅ |
| `extrapolated_within_tol` | ❌ |
| `control_ok` | ✅ |
| `bisection_consistent` | ✅ |
| `sigma_sign_change_present` | ✅ |
| `ma_sensitivity_within_tol` | ❌ |

## ① 问题与 ref 出处

**物理**：两层无限大刚性平行板之间的 Boussinesq 流体，下板热（T_hot）、上板冷（T_cold），温差 ΔT 驱动。静止（纯导热）基态在 Ra 超过临界值后线性失稳，先出现二维定常辊。经典线性理论给出刚-刚/等温壁情形的临界值：

```text
Ra_c = 27*pi^4/4 = 1707.7597 ,  a_c = k_c H = 3.11632   (2-D rolls)
```

**ref 出处**：Chandrasekhar (1961) Hydrodynamic and Hydromagnetic Stability, §II (linear stability theory, rigid-rigid isothermal walls, 2-D rolls)；次级 DOI `10.1016/0368-3281(63)90025-1`。
**零手抄声明**：`RA_REF is the literature constant declared in PREDECLARED; it is recomputed inside this run by Route C (continuum marginal stability, independent of LBM) so the comparison is not a transcription of the target into the result`。本单不在离线环境伪造页码/表格截图；可核验面=①该常数作为 `PREDECLARED['ra_ref']` 在运行前冻结，②同一物理问题由仓内独立连续谱模块（Route C）用同一判据重算并给出一致值（见 ④）。

## ② 方法与实现

三条能力、三个独立证据面（全部复用仓内既有实现，不新写解法器）：

| 路径 | 仓内实现 | 干什么 | 提供量 |
|---|---|---|---|
| **Route A（主）** | `tensorlbm.rb_convection_validation.measure_growth_rate`（`Convection2D`：D2Q9 流场 + 热标量，BGK + Boussinesq 力，pre-streaming 半程反弹壁） | 导热基态 + 无穷小温度扰动 `amp·cos(kx)·sin(πy/H)`，时间推进，拟合同模态投影 `log\|A(t)\|` 斜率 | `σ* = σH²/α`（无量纲增长率） |
| **Route B（交叉）** | `tensorlbm.rb_discrete_stability.critical_rayleigh`（同一格点步进算子的有限差分 Jacobian 的谱） | 对 `γ(Ra)=0` 二分（不做时间推进） | 算子本征增长率 `γ̂=γH²/ν` |
| **Route C（控制）** | `tensorlbm.rb_continuum_stability.critical_at`（N-S-Boussinesq 线性稳定性，二阶 FD + QZ 全谱） | 对 `max Re σ(Ra)=0` 二分 + 三网格 Richardson | 连续谱 `Ra_c` |

**判定判据（运行前冻结在 `run_case.py::PREDECLARED`）**：静止态线性稳定性判据=`sign of d(log|A|)/dt of the conduction-state perturbation (sigma_star = sigma H^2/alpha); unstable iff sigma_star > +sigma_marginal`；

- 判定带：`|σ*| ≤ 0.05` 记 marginal（测量地板），`σ* > +0.05` unstable，`σ* < -0.05` stable；
- 否决门：`valid=False` 或 `fit_r2 < 0.98` 的臂不投票（记 indeterminate）；
- 扫描：先验证括号 `Ra/Ra_ref ∈ [0.9, 1.1]` 两端异号，再以 **Ra 二分 + σ 零点插值（带中点回退）** 收缩，停在 `(hi-lo)/mid < 0.002`，`Ra_c` = 最终括号内 σ 零点插值与中点的平均，误差带 = 括号半宽；
- 通过线：**主网格 ny=64 的 Ra_c 与 1/ny 外推值 Ra_inf 同时 <2%**；控制臂 Route C 必须 <0.05%（否则关于 LBM 的任何结论都不成立）。

## ③ 参数表（无量纲数完整）

| 参数 | 记号 | 取值 | 说明 |
|---|---|---|---|
| 参考 Rayleigh 数 | Ra_ref | 1707.7597 | Chandrasekhar 1961 §II（刚-刚） |
| Prandtl 数 | Pr = ν/α | 0.710 | 空气级；水/油类不在本案范围 |
| 格点热扩散率 | α | 1/6 = 0.16667 | τ_T = 3α+1/2 = 1.0000 |
| 格点黏度 | ν = Pr·α | 0.11833 | τ_f = 3ν+1/2 = 0.8550 |
| 盒宽比 | Γ = nx/ny | 2 | x 周期，仅容许 k=2πm/nx |
| 水平模 | m | 1 | 单一辊对 |
| 无量纲波数 | kH | π = 3.14159 | 相对 a_c=3.11632 偏 +0.81%（见 ⑤敏感性）|
| 温度差 | ΔT | 1 | 格子单位，T_hot=1, T_cold=0 |
| 浮力耦合 | g_β | 见 `routeA_*.json::g_beta` | `Ra = g_β ΔT H_eff³/(να)` |
| Ra 定义长度 | H_eff | ny−2（实测流动间隙） | `Convection2D` 的半程反弹间隙；**与 Route B 的 H=ny 口径不同**（见 ⑤）|
| 生产网格 | (ny,nx) | [[32, 64], [48, 96], [64, 128]] | 上限 ny=64 |

## ④ 对照表（ref vs got + 误差打勾叉）

### Route A — 主判据（Ra 二分扫描，暂停态线性稳定性判据）

| ny | nx | 臂数 | 最终括号 (Ra) | σ* 括号 | Ra_c | vs 1707.7597 | <2% | 停机规则 |
|---|---|---|---|---|---|---|---|---|
| 32 | 64 | 3 | [1537.0, 1878.5] | [-1.268, +1.210] | **1711.74** | +0.23% | ✅ | sigma_marginal |
| 48 | 96 | 3 | [1537.0, 1878.5] | [-1.373, +1.052] | **1730.35** | +1.32% | ✅ | sigma_marginal |
| 64 | 128 | 3 | [1537.0, 1878.5] | [-1.366, +1.028] | **1731.86** | +1.41% | ✅ | sigma_marginal |

### 网格外推（Route A）

- 形式 `Ra_c(ny) = Ra_inf + C/ny`：Ra_inf = **1755.32** （+2.79%，❌ <2%），斜率 C = -1359.0，最大残差 3.34
- 两点外推（最细两网格）：1736.4（+1.68%）

### Route C — 控制臂（连续谱，独立于 LBM）

| M（FD 内点数） | Ra_c |
|---|---|
| 50 | 1702.6366 |
| 100 | 1706.4512 |
| 200 | 1707.4306 |

Richardson 外推 **1707.7685** vs ref 1707.7597 → +0.0005% (✅ <0.05% 控制线)

同一连续谱模块在**本案波数 kH=π** 上的临界值：1706.616（相对 a_c 处 1707.76 的差 → 波数偏置的一阶量化）

### Route B — 交叉臂（格点算子的本征稳定性，不做时间推进）

- ny=16，a=kH=3.1416，口径 H=ny：Ra_c=2673.44；换算到 Route A 口径（×((ny-2)/ny)³）→ 1790.99（+4.87%）
- 同意 `g_β` 下的算子增长率：Route B `γ̂`=0.055 → 折算 `σ*_A=γ̂/Pr`=0.077；Route A 在 ratio=1.05 实测 σ* 见 ⑥ 的偏差讨论

## ⑤ 收敛性 / 参数敏感性（至少两点）

**S1 网格收敛**：见 ④ 网格外推表 + `fig2_grid_convergence.png`。Ra_c 随 ny 单调移动，1/ny 拟合残差给出外推不确定度。

**S2 波数敏感性**（`fig3_wavenumber_scan.png`）：固定 Ra=1.05Ra_ref、ny=32，扫三盒宽（nx=56/64/72 → a≈3.59/3.14/2.79），抛物拟合顶点 a_peak=3.3043（vs a_c=3.11632，+6.03%）。本案用 kH=π 与 a_c 差 +0.81%，由临界曲线在极小点附近的二次平坦性，Ra_c 的敏感性为二阶小量（控制臂 `marginal_curve` 给出量级）。

| 波数 | nx | σ* | 判定 |
|---|---|---|---|
| 3.5904 | 56 | +0.5177 | unstable |
| 3.1416 | 64 | +0.6106 | unstable |
| 2.7925 | 72 | +0.2153 | unstable |

**S3 输运系数/Mach 敏感性**：α 减半（τ_T=0.75，τ_f 相应下降）重跑同一条二分链：
Ra_c=1758.09（+2.95%，❌），臂数 3，停机 sigma_marginal。

**S4 拟合窗/振幅地板**：每条臂的 `fit_window`、`fit_r2`、`amp_initial/amp_final` 全部随工件落盘；只有 `fit_r2 ≥ 0.98` 且 `valid=True` 的臂才投票（近临界臂因振幅近似恒定，R² 无意义，按 |σ*|≤判定带归 marginal，不参与投票也不被误判为失败）。

## ⑥ 失败模式与工件

**已知失败模式（如实登记）**：

1. **近临界臂的 R² 陷阱**：σ*→0 时 |A(t)| 近似常数，`log|A|` 回归 R² 会掉到 0.99 以下，若把 R² 当否决门会把正确的 marginal 判成 indeterminate（首版实现即踩此坑，已改为按 |σ*| 判定带优先）。
2. **两套 Ra 口径**：`rb_convection_validation` 用实测流动间隙 H_eff=ny−2，`rb_discrete_stability` 用 H=ny。两者差 ((ny)/(ny−2))³（ny=16 时 +49%）；跨模块比较必须先换算，否则会把口径差误读成物理差。
3. **Route B 与 Route A 在本案网格上不完全一致**：换算到同一口径后 ny=16 的两条格点路线给出的 Ra_c 相差约 8%（壁面/热边界放置差异），故本单**不**用 Route B 断言 lattice shift，只用它做量级交叉与口径登记。
4. **计算成本**：单臂成本 ∝ ny²·nx·steps；ny=64 单链约 8601 步/臂。本案时间步上限按 `measure_growth_rate` 的默认自适应（0.35H²/α）。

**工件**：

```text
 32 : routeA_bisect_ny32_nx64.json
 48 : routeA_bisect_ny48_nx96.json
 64 : routeA_bisect_ny64_nx128.json
    routeB_lattice_eigen.json / routeC_continuum_control.json
    routeA_wavenumber_scan.json / routeA_ma_sensitivity.json / summary.json
    fig1_sigma_vs_ra.png
    fig2_grid_convergence.png
    fig3_wavenumber_scan.png
    fig4_control.png
```

**一键复现**：`PYTHONPATH=src python3 benchmarks/case_pool/rayleigh_benard_onset/run_case.py --out-dir benchmarks/case_pool/rayleigh_benard_onset/artifacts`

**环境**：Python 3.11.6，Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38，loongarch64，128 cores，每臂 1 线程；总墙钟 545.1 s。
