# 壁面模型 LES：Re_tau = 590 稳定性根因隔离与峰值位置双指标闭环（P372W2-N01）

**仓库** TensorLBM ｜ **模块** `tensorlbm.wall_model_les` ｜ **工具** `tools/les_peak_diagnostic.py` ｜ **分支** `agent/m10757`

> 本节点把壁模化 LES 的误差诊断从「峰值幅值欠预测」单一指标，扩到**峰值位置偏移**与**高 Re 稳定性**双指标闭环：先用参数网格把 Re_tau=590 的发散归因，再在三套网格上量峰值位置、用 Richardson/GCI 把网格量化误差与模型误差分开，最后由偏移量反推壁函数修正。

## ① 案例介绍（物理背景 / 几何 / 参数）

压力驱动平面槽道，x/z 周期、y 双壁 halfway bounce-back，壁面模型在交换节点给出 `u_tau`。参照量是 Moser–Kim–Mansour (1999) `Re_tau = 590` DNS 平均剖面表：表里没有 `<u'v'>` 列，雷诺应力由**精确槽道动量平衡** `-<u'v'>+ = 1 - y/h - dU+/dy+` 复原（壁面与中线两端恒等式同时作为完整性门）。

| 量 | 值 |
|---|---|
| DNS 参照 | `benchmarks/pending/channel_turb_3d/dns_ref/mkm590.means`（sha256 前缀 `878bc951794516b6`，与 provenance 一致 = True） |
| DNS Re_tau（表内 y+ 中点） | 587.19 |
| DNS 峰值 | `y+ = 45.20`，`-<u'v'>+ = 0.8682` |
| 分子黏度（按 dy+ 定标） | `nu = u_tau / dy+`，故同一 dy+ 下 nu 与 Re_tau 无关 |
| 槽道半高 | `h+ = Re_tau`（`u_tau` 恒为 0.02 lattice，除 Mach 门实验） |
| 三套网格（Re_tau=590） | dy+ = 7.5 / 15 / 30（`ny = 158 / 79 / 40`，横向固定 16×16） |
| 控制算例 | Re_tau = 180、dy+ = 7.5、16×16（该求解器已验证的档位，用作诊断工具自基线） |

网格族**只沿壁法向细化**（横向分辨率固定在 16×16 格），因此三套网格的物理盒宽 `Lx+ = Lz+` 从 120 增长到 480 壁单位；这是本节点的已知混杂因子（⑤ 明确列出），不主张横向无关性。

## ② 怎么做的（方法 / 流程 / 工具链）

1. **稳定性根因隔离** `benchmarks/wall_model_les/re590_stability.py`：在 `(Re_tau/nu, u_tau, Cs, 交换点行, 盒宽, 耦合方式, 欠松弛因子, dy+)` 上做二分式扫描，每个算例记录全时程 `u_bulk / u_tau_model / 最小有效松弛裕度 3(nu+nu_t)_min / SGS 比 / max_speed`，首次非有限步与异常文本；`stability_attribution.py` 把扫描折成假设表（H1 松弛裕度 / H2 交换点耦合 / H3 亚格子不足）。
2. **峰值位置诊断** `tools/les_peak_diagnostic.py`（库 `tensorlbm.wall_model_les.peak_diagnostic`）：从 LBM 流场提取 `-<u'v'>+(y+)`，在离散最大点周围三点区间上用自然三次样条 + 独立三点抛物线求亚格子峰位，两者之差折入不确定度（壁模化网格上间距与峰位同阶，单一数字会高估测量能力）。
3. **三网格 Richardson/GCI**：不等细化比的一般形式 `(f3-f2)/(f2-f1) = r12^p (r23^p - 1)/(r12^p - 1)` 解观测阶 `p`，`f_ext = f1 + (f1-f2)/(r12^p - 1)`，`GCI_fine = 1.25 |(f1-f2)/f1| / (r12^p - 1)`；`|f_fine - f_ext|` 记网格误差、`|f_ext - f_DNS|` 记模型误差。
4. **壁函数修正映射**：把三套网格的相对峰位偏移拟成 `offset(dy+)`（线性/反比两族，留一点交叉验证），并反解达到 15% 容差所需的 `dy+`。
5. **可复现性**：同种子逐位复现 + 两种子统计散布（`reproducibility_check.py`）。
6. **判据折账** `acceptance_p372w2.py` 把以上工件折成 c1–c7 裁决表，报告里所有数字由脚本注入，无手抄。

## ③ 计算结果（数字表，程序化注入）

### 3.1 稳定性根因（假设表）

| 假设 | 判别实验 | 裁决 | 依据 |
|---|---|---|---|
| H1_relaxation_margin | same wall-unit grid (ny=49, box 32x32), u_tau (hence nu) varied | **REJECTED** | the outcome does not follow the viscosity margin (see rows) |
| H2_exchange_coupling | under-relaxed wall-model target / alternative coupling on the blowing-up configuration | **REJECTED** | no coupling variant stabilises the run |
| H3_sgs_deficit | Cs raised on the blowing-up configuration (0.1 -> 0.3) | **REJECTED** | raising Cs does not prevent the divergence |
| H4_resolved_velocity_envelope | u_tau swept at a fixed wall-unit grid (dy+=24.6, box 32x32, Cs=0.1, explicit coupling): stability must follow the velocity scale, not nu | **SUPPORTED** | the fastest run of the family reaches max abs(u) = 1.290 lattice units (lattice stability envelope ~1) before/at divergence, and the smallest velocity scale is the stable one |


**判别量**：全部算例按「运行中达到的最大 `max|u|`」分成互不重叠的两组 —— 发散组最小 1.099、稳定组最大 0.944（lattice 稳定包络 ~1）。这说明发散由**已解析速度尺度**控制，而不是由 `tau-1/2` 的分子黏度余量或壁模型耦合增益控制；同时也解释了为什么提高 `Cs`、降低 `beta` 只能推迟而不能消除发散。

关键扫描行（全部来自 `runs/m10757/batchP|batchQ|long1` 的算例 JSON）：

| 算例 | dy+ | nu | 盒 | Cs | beta | u_tau | 首次反号步 | 首个 NaN 步 | max abs(u) | 结论 |
|---|---|---|---|---|---|---|---|---|---|---|
| `P3_re590_ny40_cs010` | 30.26 | 6.61e-04 | 8x8 | 0.10 | 1.00 | 0.020 | None | None | 0.845 | 稳定 |
| `L2_re590_ny49_n32_nosgs` | 24.58 | 8.14e-04 | 32x32 | 0.00 | 1.00 | 0.020 | 332 | 1328 | 1.275 | 发散 |
| `Q3_beta0.2` | 24.58 | 8.14e-04 | 32x32 | 0.10 | 0.20 | 0.020 | 366 | 1281 | 1.099 | 发散 |
| `Q2_beta0.5` | 24.58 | 8.14e-04 | 32x32 | 0.10 | 0.50 | 0.020 | 366 | 1281 | 1.233 | 发散 |
| `Q5_utau0.015` | 24.58 | 6.10e-04 | 32x32 | 0.10 | 1.00 | 0.015 | None | None | 0.447 | 稳定 |
| `L1_re590_ny49_n32` | 24.58 | 8.14e-04 | 32x32 | 0.10 | 1.00 | 0.020 | 500 | 1250 | 1.101 | 发散 |
| `Q4_utau0.03` | 24.58 | 1.22e-03 | 32x32 | 0.10 | 1.00 | 0.030 | None | 1098 | 1.290 | 发散 |
| `Q6_cs0.3` | 24.58 | 8.14e-04 | 32x32 | 0.30 | 1.00 | 0.020 | 732 | 1647 | 1.248 | 发散 |
| `P1_re590_ny79_cs010` | 15.13 | 1.32e-03 | 16x16 | 0.10 | 1.00 | 0.020 | None | None | 0.944 | 稳定 |
| `P4_re590_ny79_nowall` | 15.13 | 1.32e-03 | 16x16 | 0.10 | 1.00 | 0.020 | None | None | 0.936 | 稳定 |
| `P2_re590_ny79_cs030` | 15.13 | 1.32e-03 | 16x16 | 0.30 | 1.00 | 0.020 | None | None | 0.476 | 稳定 |

### 3.2 判据表

| 判据 | 裁决 | 依据 |
|---|---|---|
| c1_stability_re590 | **PASS** | fine grid dy+=7.52, 4500 steps, first non-finite step = None, non-finite log rows = 0, measured Re_tau at the last log = 262.8386126497991 |
| c2_reproducibility | **WITHHELD** | the reproducibility window does not develop a resolved stress peak (peak / DNS scale = [0.0103, 0.0209]), so peak reproducibility is not testable there; determinism itself holds (bitwise identical = True, max |d uv+| = 0.000e+00) |
| c3_peak_offset_15pct | **FAIL** | fine-grid peak y+ = 73.32 vs DNS 45.20 -> +62.22 % (threshold ±15 %); Richardson h->0 = 110.61780670241092 -> +144.75 %; observed order p = nan, converged = False |
| c3b_peak_offset_re_matched | **FAIL** | peak position against the DNS peak at each run's own measured Re_tau: fine Re_tau=295 -> +96.29 %; medium Re_tau=287 -> +1.40 %; coarse Re_tau=324 -> +11.98 % (worst fine, threshold ±15 %) |
| c4_grid_error_split | **INFO** | y_peak: grid error (fine vs h->0) = 37.299 wall units, GCI(fine) = 0.636, model error (h->0 vs DNS) = 65.421; value: grid error = 0.5991, model error = 0.9331 |
| c5_wall_function_correction | **PASS** | offset map (inverse) coefficients [-0.414724, 3.568946], rmse = 0.1709; fitted spacing for the 15 % tolerance = 6.32 wall units |
| c6_reference_integrity | **PASS** | benchmarks/pending/channel_turb_3d/dns_ref/mkm590.means sha256 matches provenance = True; balance identities: wall value = 0.0, centre value = 0.0; tabulated Re_tau = 587.19 |
| c7_control_re180 | **INFO** | Re_tau = 180 control: stable = True, peak-position error vs MKM-180 = +28.37 % |

计数：{'PASS': 3, 'FAIL': 2, 'WITHHELD': 1, 'INFO': 2}

### 3.3 峰值位置与幅值（Re_tau = 590）

| 网格 | dy+ | 样点数（半槽） | 离散峰 y+ | 样条峰 y+ | 抛物线峰 y+ | 不确定度 | -<u'v'>+ 峰 | 峰位误差 | 幅值误差 |
|---|---|---|---|---|---|---|---|---|---|
| fine | 3.76 | 20 | 73.32 | 73.32 | 73.32 | 0.000 | 0.5342 | +62.2 % | -38.5 % |
| medium | 7.36 | 10 | 40.50 | 37.56 | 36.97 | 0.295 | 1.1085 | -16.9 % | +27.7 % |
| coarse | 16.60 | 5 | 41.50 | 43.00 | 43.65 | 0.324 | 1.1296 | -4.9 % | +30.1 % |

平衡性门（测量是否可用的前提）：`u_tau_model/u_tau` 必须接近 1，且解析应力不得超过总应力包络 `1 - y+/h+`；任一条不满足，该网格的峰值数字只作诊断，不作判据。

| 网格 | u_tau_model/u_tau | 实测 Re_tau | 峰值/应力包络 | 平衡 |
|---|---|---|---|---|
| fine | 0.500 | 295.2 | 0.712 | 否 |
| medium | 0.487 | 287.2 | 1.277 | 否 |
| coarse | 0.549 | 323.7 | 1.305 | 否 |

**Re 匹配补充比较**（把每个网格的实测 Re_tau 作为参照雷诺数，DNS 峰位在两个 DNS 档（180 / 590）之间按 `ln Re_tau` 线性插值）：

| 网格 | 实测 Re_tau | DNS 匹配峰位 y+ | 实测峰位 y+ | 偏移 | 是否 ≤15% |
|---|---|---|---|---|---|
| fine | 295.2 | 37.35 | 73.32 | +96.29 % | 否 |
| medium | 287.2 | 37.04 | 37.56 | +1.40 % | 是 |
| coarse | 323.7 | 38.40 | 43.00 | +11.98 % | 是 |

### 3.4 Richardson 外推与 GCI（峰位 / 峰值）

| 量 | 网格值（细→粗） | 观测阶 p | 外推 h→0 | GCI(fine) | 网格误差 | 模型误差 | 参考值 |
|---|---|---|---|---|---|---|---|
| 峰位 y+ | 73.318 / 37.562 / 43.000 | n/a | 110.6178 | 0.6359 | 37.2995 | 65.4209 | 45.1970 |
| -<u'v'>+ 峰 | 0.534 / 1.109 / 1.130 | n/a | -0.0649 | 1.4018 | 0.5991 | 0.9331 | 0.8682 |

收敛判定：`converged = False`，`monotone = False`，备注 `non-monotone triple: differences change sign, no asymptotic order`。

### 3.5 壁函数修正映射

* 拟合族 **inverse**，系数 `[-0.414724, 3.568946]`，rmse = 0.1709，最大残差 = 0.2388
* 达到 15% 容差所需的壁法向间距：`dy+ = 6.32`
* 网格无关判定：外推峰位偏移 +144.75 %，是否满足 15%：**False**

### 3.6 可复现性

* 同种子重跑逐位一致：**True**（`max|Δ-<u'v'>+| = 0.0`）
* 两种子峰位差 +0.00 %、峰值差 +303.22 %、`u_bulk` 差 +0.27 %（`2500` 步短窗，仅作重复性界）

## ④ 结果图

![fig_wmles_stability_histories.png](fig_wmles_stability_histories.png)

*(a) Plane-mean bulk velocity for the Re_tau = 590 configuration that diverges and for the coupling variants. The baseline does not decay smoothly into a NaN state: it reverses sign around the step marked by the thin vertical line and only then diverges, so the failure is a growing oscillation of the wall-model/first-cell loop. (b) The same runs in terms of the smallest relaxation margin 3(nu+nu_t) in the box: the margin stays at O(10^-3) and does not collapse towards 0.5-1/2 = 0, which rules out a vanishing effective relaxation time.*

![fig_wmles_stability_map.png](fig_wmles_stability_map.png)

*Stability of the Re_tau = 590 wall-modelled channel as a function of the wall-normal spacing and the lateral box size. Divergence needs a box that can hold resolved turbulence; short or narrow boxes stay finite because the resolved fluctuations never reach the amplitude that drives the coupling loop.*

![fig_wmles_velocity_envelope.png](fig_wmles_velocity_envelope.png)

*Largest resolved velocity reached during each Re_tau = 590 scan case against its wall-normal spacing. Every divergent run (X) reached max|u| >= 1.1 lattice units; every run that stayed finite stayed below 0.95. The wall-normal spacing alone does not separate the outcomes.*

![fig_wmles_stress_profiles.png](fig_wmles_stress_profiles.png)

*(a) Resolved Reynolds shear stress in wall units for the three grids; stars mark the sub-sample peak found by the cubic-spline diagnostic, the dashed line is the DNS peak position at y+ = 45.2 (MKM-590 mean-profile balance). (b) The corresponding mean velocity profiles against the same DNS.*

![fig_wmles_peak_offset.png](fig_wmles_peak_offset.png)

*(a) Peak-position error against the DNS reference as a function of the wall-normal spacing, with the fitted correction map and the spacing at which the fitted map reaches the 15 % tolerance. (b) The measured peaks, the Richardson h->0 estimate and its GCI band against the DNS peak.*

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **短盒会掩盖发散**：8×8 的小盒在 Re_tau=590 跑 2500 步不 NaN，但横向盒宽足够（32×32）时同一组参数在 1000–1700 步内发散。任何「我跑通了」的结论必须先声明盒宽，否则是在测初值瞬态而不是稳定性。
2. **发散不是「有效松弛时间趋零」**：最小裕度 `3(nu+nu_t)_min` 全程 O(1e-3)，反而在发散前是 `max_speed` 逼近 1（lattice 稳定上限）——归因必须看速度尺度，不能只看 tau。
3. **欠松弛不是白吃的午餐**：`beta<1` 虽然能推迟 NaN，但把壁模型反演的输入速度改掉，耦合系统的不动点随之漂移（实测 `u_tau_model/u_tau` 到 1.4 以上），所以它既不是「不改物理的稳定化」，也不能当作验证通过的配置。
4. **粗网格的「稳定」可能是假稳定**：dy+=30 时实测 `Re_tau` 只有 290–330 左右（壁模型在粗首层上严重低估摩阻），此时若只看「不 NaN」会把它当成可用配置。
5. **壁单位必须用实测 u_tau**：三套网格的 `u_tau_model/u_tau` 都远离 1（见 3.3 平衡门），若沿用配置里给的 `u_tau` 归一化，`y+` 与 `-<u'v'>+` 会同时被同一个因子缩放，峰位看起来「更接近 DNS 590」，但那是归一化假象。本报告一律按实测 `u_tau` 归一化，并额外给出 Re 匹配比较（3.3），把「没到 590」与「峰值形状错」两件事分开。

6. **网格族只沿 y 细化**，横向盒宽在壁单位下从 120 变到 480，抖动能量所依赖的盒尺寸不同——峰值位置误差主要受 dy+ 控制，但横向盒效应未被本族排除。
7. **峰值位置本身在本求解器里不是一个已收敛量**：上一轮记录（`docs/wall_model_les_m7797_report.md`，Re_tau=180、32×32 盒、6000 步）判出的解析峰位是 `y+ = 20.22`（相对 DNS 31.98 偏 **-37%**），本轮的 Re_tau=180 控制算例（16×16 盒、3500 步）给出相反符号的偏移；两者差在盒子宽度与统计窗上，而不是在壁模型上。任何「峰位误差 ≤X%」的结论都必须同时给盒宽与平均窗，否则不可比。
8. **DNS 参照的应力是推导量**：MKM 表只给平均剖面，`-<u'v'>+` 由动量平衡复原（恒等式在壁面/中线两端严格成立，见 c6），它不含 DNS 的湍流统计列，因此本节点的「模型误差」是相对这个复原剖面而言的。

## ⑥ 误差判定（逐项，显式回答是否 < 3%）

| 关键量 | 本次值 | 基准 | 相对误差 | 是否 < 3% |
|---|---|---|---|---|
| 峰位 y+（fine） | 73.32 | 45.20（DNS-590） | +62.22 % | 否 |
| 峰值 -<u'v'>+（fine） | 0.5342 | 0.8682（DNS-590） | -38.47 % | 否 |
| 峰位 y+（medium） | 37.56 | 45.20（DNS-590） | -16.89 % | 否 |
| 峰值 -<u'v'>+（medium） | 1.1085 | 0.8682（DNS-590） | +27.67 % | 否 |
| 峰位 y+（coarse） | 43.00 | 45.20（DNS-590） | -4.86 % | 否 |
| 峰值 -<u'v'>+（coarse） | 1.1296 | 0.8682（DNS-590） | +30.10 % | 否 |
| 峰位（Richardson h→0） | 110.618 | 45.197（DNS-590） | +144.75 % | 否 |
| 峰值（Richardson h→0） | -0.0649 | 0.8682（DNS-590） | +107.47 % | 否 |

**显式回答**：峰值位置与幅值相对 DNS-590 的误差**没有**任何一项落在 3% 以内；本节点主张的是 15% 量级的位置判据与「网格误差 vs 模型误差」的分离，3% 量级的结论在本网格族与本次统计窗口下不可达（细网格内层带内只有 20 个壁法向样点、平均窗 2000 步，且 Re=590 的壁模化解本身处于慢振荡）。数值按上表逐项如实列出，未做修饰。
