# Schnerr–Sauer 空化质量传递源项：文献闭式 + Rayleigh–Plesset 锚定验证

> 单：selfstart-*（m6298，TensorLBM）。改动面：`src/tensorlbm/cavitation.py`
> （只动 `schnerr_sauer_source` 及其新拆出的 `schnerr_sauer_bubble_radius`）、
> 新增 `tests/test_cavitation_schnerr_sauer_reference.py`、
> `artifacts/cavitation_schnerr_sauer_20261108/`（脚本+results.json+图）。
> 本文所有数字由 `artifacts/cavitation_schnerr_sauer_20261108/run_validation.py`
> 程序化注入（`results.json` / `tables.md`），零手抄。
> 重跑：`PYTHONPATH=src python artifacts/cavitation_schnerr_sauer_20261108/run_validation.py`

## 1. 案例介绍（为什么取这条）

**物理背景。** 空化是螺旋桨/舵/泵喷推进器的核心性能与噪声机理。TensorLBM 里唯一
的相变质量传递模型是 `cavitation.py` 的 `schnerr_sauer_source`（Schnerr & Sauer 2001），
它被 `apply_cavitation_force` 与 `run_cavitation_flow` 调用，是"空化能力"链路的物理心脏。

**取此条的三个理由（先读账再选题）。**

| 账本/审计面 | 现状 | 本单处置 |
|---|---|---|
| `benchmarks/gaps.md` G3 | `psi_cavitation` 共存态/符号约定已闭环（`docs/g_ledger_g3_cavitation_coexistence.md`，m4969） | 不重跑；G3 明确只覆盖伪势与力调用，**不含源项** |
| `docs/analytic_verification_burndown_20261103.md` / `docs/lbm_verification_ladder_r2.md` | A–E 五个解析解案例覆盖碰撞核/反弹/Guo 强迫/方管 | 均不含相变源项 |
| `tests/` 全库检索 | `schnerr_sauer_source` **零直接测试**（`test_sc_eos_coexistence.py` 只测 `psi_cavitation` / `apply_cavitation_force`） | 本单补上 |

即：**已合并的空化源项从未被文献闭式或解析基准裁决过**——典型的"验证欠账"。

**参数表（格点单位，`_RHO_L`/`_RHO_V` 是 `psi_cavitation` 的标定密度尺度）。**

| 量 | 值 | 出处 |
|---|---|---|
| ρ_l | 2.65 | `cavitation._RHO_L` |
| ρ_v | 0.038 | `cavitation._RHO_V` |
| 混合密度 ρ（α_v = 0.5） | 1.3440 | ρ_l − α_v(ρ_l−ρ_v) |
| n_nuc（格点） | 0.0001 | 新增常量 `N_NUC_LATTICE_DEFAULT`（1 核 / 10⁴ 单元） |
| C_vap / C_cond | 1.0 / 0.2 | 函数默认值 |

## 2. 怎么做的（方法）

**参考式（Schnerr & Sauer 2001；interPhaseChange 家族同形）**

```
α_v = (ρ_l − ρ)/(ρ_l − ρ_v)                       # 混合密度 → 空泡分数
R_b = ( α_v/(1−α_v) · 3/(4π n_nuc) )^(1/3)        # 单分散泡群平均半径
ṁ   = C · (3 ρ_l ρ_v / ρ) · α_l α_v / R_b · sqrt( 2|p−p_sat| / (3 ρ_l) )
C = C_vap (p<p_sat 蒸发) ；C = C_cond (p>p_sat 凝结) ；ṁ(p=p_sat) ≡ 0
```

**三条彼此独立的锚（都不读实现里的常数）：**

1. **A：Rayleigh–Plesset 壁速。** 直接 RK4 积分惯性主导的 RP 方程
   `R R̈ + 1.5 Ṙ² = ±Δp/ρ_l`，远端壁速应收敛到 `sqrt(2Δp/(3ρ_l))`——
   这正是源项里那个 `sqrt(2|p−p_sat|/(3ρ_l))`。这条锚把"源项前因子从哪来"钉在方程上。
2. **B：Rayleigh 溃灭时间。** 同一族 ODE 的溃灭支给出 `t_c = 0.914681 R₀√(ρ_l/Δp)`
   （Rayleigh 1917），用来反证 A 的积分路径没错。
3. **C：闭式逐点比对 + 性质断言。** α_v 扫描逐点对闭式；并断言四条物理性质：
   饱和点恒零、单相极限趋于零、`|ṁ| ∝ sqrt(|Δp|)`（拟合斜率 = 0.5）、`R_b(α_v)` 可反演。

**鉴别力负控（D）。** 把历史上合并态的两个缺陷形态重建为 `_legacy_*` 变体
（(i) 只乘 α_l/α_v 之一；(ii) `R_b` 被硬 clamp 到 0.5 lu），确认上面的断言会把它们判红；
否则这些断言只是摆设。

**工具链。** torch 2.x / CPU / float64；RK4 常步长（溃灭 4×10⁴ 步，
生长 dt=1e-4 至 R/R₀ = 10）；pytest 用例 14 条 6.8 s；本机 load15 > 100，
故不跑任何三维空化算例（只做 ODE + 标量源项，符合容量纪律）。

## 3. 计算结果

### T1 Rayleigh–Plesset 溃灭（解析 t_c = 0.914681 R₀√(ρ/Δp)）

| Δp | t_c 数值 | t_c 解析 | 相对误差 | <1% |
|---|---|---|---|---|
| 0.5 | 2.105701 | 2.105754 | 0.0025% | 是 |
| 1 | 1.488956 | 1.488993 | 0.0025% | 是 |
| 4 | 0.744478 | 0.744496 | 0.0025% | 是 |

### T2 Rayleigh–Plesset 惯性生长壁速（解析 √(2Δp/(3ρ_l))）

| Δp | Ṙ 数值 | Ṙ 解析 | 相对误差 | <1% |
|---|---|---|---|---|
| 0.004 | 0.03170620 | 0.03170620 | 0.0000% | 是 |
| 0.01 | 0.05013190 | 0.05013190 | 0.0000% | 是 |
| 0.04 | 0.10026380 | 0.10026380 | 0.0000% | 是 |

### T3 α_v 扫描（Δp = 1e-2，c_vap = c_cond = 1）

| α_v | 新实现 | Schnerr–Sauer 闭式 | 相对误差 | legacy(R_b clamp) | legacy(单 α 因子) |
|---|---|---|---|---|---|
| 0.0001 | 9.216883e-07 | 9.216883e-07 | 4.683e-11% | 1.143578e-06 | 5.717888e-03 |
| 0.001 | 4.276759e-06 | 4.276759e-06 | 1.466e-12% | 1.143563e-05 | 5.717814e-03 |
| 0.01 | 1.978858e-05 | 1.978858e-05 | 3.082e-13% | 1.143414e-04 | 5.717068e-03 |
| 0.05 | 5.703736e-05 | 5.703736e-05 | 8.316e-14% | 5.713584e-04 | 5.713584e-03 |
| 0.2 | 1.353407e-04 | 1.353407e-04 | 2.003e-14% | 2.278989e-03 | 5.697472e-03 |
| 0.5 | 2.108883e-04 | 2.108883e-04 | 1.285e-14% | 5.637063e-03 | 5.637063e-03 |
| 0.8 | 2.039139e-04 | 2.039139e-04 | 7.975e-14% | 8.652349e-03 | 5.407718e-03 |
| 0.95 | 1.197007e-04 | 1.197007e-04 | 2.944e-13% | 8.537844e-03 | 4.493602e-03 |

### T4 饱和点与单相端点（p = p_sat）

| 状态 | 新实现 ṁ | legacy ṁ |
|---|---|---|
| alpha_v=0.5 | 0.000000e+00 | -2.254825e-08 |
| alpha_v=0 (pure liquid) | 0.000000e+00 | -0.000000e+00 |
| alpha_v=1 (pure vapour) | 0.000000e+00 | -1.594992e-06 |

### T5 R_b(α_v) 映射（n_nuc = 1e-4 lu）

| α_v | 新 R_b (lu) | 解析 R_b (lu) | legacy R_b (lu) |
|---|---|---|---|
| 0.01 | 2.8891 | 2.8891 | 0.5000 |
| 0.05 | 5.0086 | 5.0086 | 0.5000 |
| 0.2 | 8.4195 | 8.4195 | 0.5000 |
| 0.5 | 13.3650 | 13.3650 | 0.5000 |
| 0.8 | 21.2157 | 21.2157 | 0.5000 |
| 0.95 | 35.6633 | 35.6633 | 0.5000 |

### T6 sqrt(|Δp|) 标度拟合斜率

| 实现 | 拟合斜率 | 与 0.5 的偏差 |
|---|---|---|
| 新实现 | 0.500000000000 | 1.055e-15 |
| legacy(R_b clamp) | 0.500000000000 | 3.331e-16 |


## 4. 结果图

| 图 | 内容 |
|---|---|
| `artifacts/cavitation_schnerr_sauer_20261108/fig1_rp_collapse.png` | (a) RP 积分溃灭轨迹 vs Rayleigh 常数；(b) 生长壁速 vs √(2Δp/(3ρ_l)) |
| `artifacts/cavitation_schnerr_sauer_20261108/fig2_alpha_sweep.png` | 源项 vs α_v：新实现 / legacy(R_b clamp) / legacy(单 α 因子) |
| `artifacts/cavitation_schnerr_sauer_20261108/fig3_dp_scaling.png` | 源项 vs \|Δp\| 双对数与 0.5 斜率 |

## 5. 注意点（踩到的坑 / 局限）

1. **修复前源项在饱和点不为零。** `torch.where(dp < 0, evap, cond)` 在 `dp == 0`
   时落进凝结分支，实测 p = p_sat、α_v = 0.5 时留下
   `1.595e-06` 的伪传质（表 T4）。已改为显式零。
2. **修复前缺一个相因子。** 闭式是 `α_l · α_v`，历史实现蒸发只乘 α_l、凝结只乘 α_v。
   后果在单相端最严重：α_v→0 时 `R_b→0` 而传质不随之消失，纯液（α_v=1e-8）在 p<p_sat
   下仍产出 `1e+3` 倍于修复后的伪蒸发（负控用例
   `test_negative_control_missing_phase_factor_is_detected` 守住）。
3. **修复前 `R_b` 与 α_v 无关。** SI 的 `n_nuc=1.6e13` 乘 `1e-30` 得到 `1.6e-17` 格点值，
   `(3/(4πn))^(1/3) ≈ 2.4e5` 再被 `clamp(max=0.5)` 砍成常数 0.5：源项退化成定半径模型，
   且量级被放大约 10.0 倍（α_v=0.05 处）。
   现在 `n_nuc` 明确是格点量（默认 1e-4），并提供 `n_nuc_si` + `lattice_length_m` 的
   显式换算入口，**不接受隐式 1e-30 缩放**。
4. **`apply_cavitation_force` 的注入方式未改。** 它把 `ṁ·τ` 直接加进 `feq` 的密度，
   是"单步硬注入"的粗口径；本单只修源项的物理形式与量级，没有重写耦合方式
   （量级已小 ~1–2 个数量级，反而更稳，见回归用例
   `test_apply_cavitation_force_stays_finite_with_new_source`）。
5. **`run_cavitation_flow` 仍是已知 stub 且不在本单范围**：其入口 BC 索引
   `f[:, :, 0]` 与库的 `(9, ny, nx)` 布局不符，`test_sc_eos_coexistence.py` 里
   strict xfail 仍在（本单未触碰、未粉饰）。因此**不能**把本单读作"空化全流程已验"。
6. **未覆盖**：表面张力/粘性对 RP 壁速的修正、泡间相互作用、非单分散泡群、
   可压缩修正——本单裁决的是"源项闭式 + RP 前因子 + 三条极限性质"，不是空化 CFD 精度。
7. **单相端点的浮点陷阱**：原实现分母带 `+1e-12`，ρ = ρ_v 时 α_v = 1−3.8e-13，
   留下 `1.1e-18` 的单相泄漏（本单的端点断言逐点抓出）；已去掉该 epsilon 并改为
   `rho_l > rho_v` 显式校验。

## 6. 误差判定（逐项 vs 基准，是否 <3%）

| 关键量 | 基准来源 | 相对误差 | <3% | <1% |
|---|---|---|---|---|
| 溃灭时间 t_c（Δp = 0.5/1/4） | Rayleigh 1917 闭式 0.914681 R₀√(ρ/Δp) | 最大 0.0025% | 是 | 是 |
| 生长壁速 Ṙ（Δp = 4e-3/1e-2/4e-2） | RP 一阶积分 √(2Δp/(3ρ_l)) | 最大 0.000000% | 是 | 是 |
| 源项逐点值（α_v ∈ [1e-4, 0.95]） | Schnerr–Sauer 闭式 | 最大 4.683e-11% | 是 | 是 |
| sqrt(\|Δp\|) 标度指数 | 解析 0.5 | 1.05e-15（残差） | 是 | 是 |
| 饱和点 ṁ | 物理必需 0 | 精确 0（`saturation_is_exactly_zero = True`） | 是 | 是 |

**显式回答：关键量全部 < 1%，远优于 3% 门。** 需要说明的是这里的基准是解析式/ODE 积分
而非实验，因此误差衡量的是"实现 vs 文献闭式"，不含湍流/可压缩/表面张力等建模误差；
`run_cavitation_flow` 的全流程精度依旧未验（见 §5.5），本单不对它作任何精度声明。

**判据自检（`results.json.verdict.passed`）= True**，由
`run_validation.py` 与 `pytest tests/test_cavitation_schnerr_sauer_reference.py`
（14 passed, 6.8 s）双侧共同裁决；`tests/test_sc_eos_coexistence.py`（13 passed, 1 xfailed）
证明未破坏 G3 已闭环面。

## 7. 复现

```bash
# 断言（CI 门）
python -m pytest tests/test_cavitation_schnerr_sauer_reference.py -q
# 数字 + 图（本文全部表格与图片的来源）
PYTHONPATH=src python artifacts/cavitation_schnerr_sauer_20261108/run_validation.py
# G3 面回归
python -m pytest tests/test_sc_eos_coexistence.py -q
```
