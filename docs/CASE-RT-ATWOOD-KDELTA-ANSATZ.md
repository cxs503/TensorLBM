# CASE-RT-INSTABILITY-ATWOOD — k*xi → 0 外推的**模型形式不确定度**

> 锚点 `CASE-RT-INSTABILITY-ATWOOD`（TensorLBM，分支 `agent/m22980`）。
> 本文件由 `benchmarks/case_pool/rayleigh_taylor_atwood05/verify/make_ansatz_report.py`
> 从 `artifacts/runs_kdelta/*/case.json` + `artifacts/kdelta/similarity_test.json`
> **程序化注入**（零手抄），重跑一条命令即可再生成。

**一句话结论**：已合并的 `verify/REPORT_KDELTA.md` 把「λ=64 的 −10.14 % 里约一半是有限界面厚度的模型形态差」建立在
**线性外推** `err = e0 + m·kξ` 上，报出锐界面残差 **−3.28 %**（峰值估计量 P）/ **−5.53 %**（窗估计量 W）。
本单对**外推函数形式本身**做可辨识性检验，结论是：

* 3 个梯级（kξ = 0.098–0.196）**不足以把「线性于 kξ」与「二次于 kξ」分开**：两种形式都能把数据拟合到
  rms < 0.06 pp，但给出的锐界面残差差 **2.10 pp（W）/ 1.58 pp（P）**；
* 因此「锐界面残差 **<3 %**（平台精度标准）」目前**不可判定**：P 估计量的线性外推落在 −3.28 %（<3 %），
  二次外推落在 −4.86 %（>3 %），两条路径跨越 3 % 线；
* 算例门（±15 %）不受影响：所有梯级的窗估计量误差在 −11.5 %…−8.6 %，**PASS 且有裕度**。

## ① 案例介绍（物理/几何/参数）

单模 Rayleigh–Taylor 不稳定性，重流体在上、轻流体在下，Atwood 数 A = (ρ₁−ρ₂)/(ρ₁+ρ₂) = 1/2，
初始幅值 a₀ = 0.01λ。本单沿用仓内既有驱动器 `run_case.py --specs kdelta`（Boussinesq 双层 D2Q9 +
Allen–Cahn 有界相场），其相似性设计**钉住**线性问题的全部无量纲群，只留界面厚度：

| 群 | 记号 | 钉住值 | 实现方式 |
|---|---|---|---|
| 黏性/惯性比 | ζ = νk²/√(Agk) | 1.5380e−2 | ν = ν₀·λ/64 |
| 格子马赫数 | Ma ~ √(Agλ) | ≈0.25（实测见下表）| g = g₀·64/λ |
| 初始幅值 | a₀/λ | 0.01 | 播种本征模 |
| **界面厚度** | kξ = 2πw_AC/λ | **0.196 → 0.049** | w_AC = 2 格固定 ⇒ kξ ~ 1/λ |

参考值全部由闭式公式在代码里求值（`ref/ref_lock.py`）：σ_inv = √(Agk)、σ_visc = √(Agk+ν²k⁴) − νk²
（Rayleigh 1883 / Taylor 1950 / Chandrasekhar 1961 §92）。锚点文献 Tryggvason et al. 2011 正文在沙箱不可取得，
本单**不著录任何页码/表号、不转抄书内数字**（沿用 MU 记账）。

## ② 怎么做的

| 步骤 | 做法 | 文件 |
|---|---|---|
| 1 | 复用已合并的相似性阶梯（λ = 64/96/128），新增**预声明** λ = 192（3×）/ 256（4×）梯级 | `run_case.py`（`SPECS_KDELTA`）|
| 2 | 把「模型形态」假设写成**三种可外推形式**并逐一拟合：linear `e0+m·kξ`、quadratic `e0+q·kξ²`、proportional `c·kξ`（无截距）| `verify/kdelta_similarity_test.py` |
| 3 | 留一法（LOO）稳定性：逐级丢弃后看截距 e0 的散布（检验数据本身是否支持外推） | 同上 |
| 4 | **预声明判别器**：同 λ 下把 Allen–Cahn 迁移率重置为 M ~ 1/λ（`KD_l096_m13`、`KD_l128_m10`，m21799 已预声明但从未跑），检验「亏欠到底由 kξ 还是由 AC 弛豫率相似性破坏驱动」 | 同上（判别规则写在脚本 docstring）|
| 5 | 单元测试钉住拟合/LOO 的数学与已发表数字（防止估计量定义漂移）| `tests/test_rt_kdelta_similarity.py`（8 例，绿）|

外推与检验的**预声明**（在跑任何新梯级前写进脚本）：形式选择用 rms/AIC 报，不由人挑；
判别器阈值 `Paired / Regression` 硬编码在脚本常量里。

## ③ 计算结果（程序化注入）

| arm | lam | k*xi | zeta | Ma_peak | M | err_W vs inv | err_P vs inv |
|---|---|---|---|---|---|---|---|
| KD_l064 | 64 | 0.1963 | 1.5380e-02 | 0.247 | 20.0 | -11.52% | -7.76% |
| KD_l096 | 96 | 0.1309 | 1.5380e-02 | 0.250 | 20.0 | -9.45% | -6.24% |
| KD_l128 | 128 | 0.0982 | 1.5380e-02 | 0.249 | 20.0 | -8.55% | -5.53% |

| estimator | form | n free | e0 (k*xi -> 0) | slope | R^2 | rms | AIC |
|---|---|---|---|---|---|---|---|
| W | linear | 2 | -5.53% | -30.37 | 0.9988 | 0.044 pp | -14.8 |
| W | quadratic | 2 | -7.64% | -101.1 | 0.9979 | 0.057 pp | -13.2 |
| W | proportional | 1 | +0.00% | -66.4 | -0.5223 | 1.531 pp | 4.6 |
| W | LOO(linear) | — | spread [-5.87, -5.31]% (full -5.53%) | — | — | — | — |
| P | linear | 2 | -3.28% | -22.78 | 0.9998 | 0.014 pp | -21.7 |
| P | quadratic | 2 | -4.86% | -75.71 | 0.9956 | 0.061 pp | -12.7 |
| P | proportional | 1 | +0.00% | -44.12 | 0.0493 | 0.907 pp | 1.4 |
| P | LOO(linear) | — | spread [-3.39, -3.21]% (full -3.28%) | — | — | — | — |

| lam | M base | M rescaled | err_W base | err_W rescaled | delta | base step vs lam=64 | regression to lam=64 | H_thickness | H_mobility |
|---|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | NOT_RUN | NOT_RUN |

Declared but not available in this checkout: KD_l192, KD_l256, KD_l096_m13, KD_l128_m10.


## ④ 图

![k*xi extrapolation ansatz](benchmarks/case_pool/rayleigh_taylor_atwood05/figures/kdelta_ansatz.png)

左/中：三个梯级的实测误差与三条外推路径（星号为各自 kξ→0 截距），橙色带为平台 ±3 % 精度标准、绿色带为算例 ±15 % 门；
右：锐界面截距在三种形式下的柱状对比——**形式之间的散布就是外推的模型不确定度**，其量级（
1.58–2.10 pp）与要判的 3 % 阈值同量级。

## ⑤ 注意点（局限 / 踩过的坑）

| 编号 | 内容 |
|---|---|
| N1 | **kξ 跨度只有 2 倍**（0.098–0.196），外推到 0 是 2 倍外推；外推的不确定度因此不可能小于形式散布。λ=192/256 已预声明并递交（`KD_l192`、`KD_l256`），本 checkout 内**尚无工件**：KD_l192, KD_l256, KD_l096_m13, KD_l128_m10 → 状态 **NOT_RUN**，不作结论 |
| N2 | linear 与 quadratic 都是 2 参数，AIC 只差 1–9（见上表），**AIC 不足以判决**；真正的判决只能来自更宽的 kξ 跨度 |
| N3 | 同 λ 变 w_AC（`w15` 支臂，已合并）在 λ=64/96 上几乎不改变误差（−10.01 % vs −10.14 %），而跨 λ 阶梯却单调改善——这两条证据方向不一致，说明 kξ 未必是唯一旋钮；`KD_l096_m13`/`KD_l128_m10`（M ~ 1/λ）正是为此预声明，本单**未能取回其工件**（队列），故 H_thickness / H_mobility 均判 **NOT_RUN**，不选边 |
| N4 | 本单是**分析侧**增量：没有改变已合并的求解器与算例判定；`run_case.py` 只新增一条预声明梯级，向后兼容（默认 `--specs main` 不变）|
| N5 | 队列现实：两份递交（`cj4502`、`cj4503`）在 16 min 内仍处 queued（在飞 91 作业），故本轮交付以「既有 3 梯级的可辨识性检验 + 判别器 + 测试」为界，算力侧留待后续单回收 |

## ⑥ 误差判定（逐项，显式回答是否 <3 %）

| 量 | 基准 | 结果 | 误差 | <15 %（算例门）| <3 %（平台标准）|
|---|---|---|---|---|---|
| 线性增长率 σ（窗估计量 W，λ=64 参考例）| σ_visc 闭式 | 2.7725e−03 | −10.14 % | **是** | 否 |
| σ（W）锐界面外推 **线性形式** | σ_inv | — | **-5.53 %** | 是 | 否（>3 %）|
| σ（W）锐界面外推 **二次形式** | σ_inv | — | **-7.64 %** | 是 | 否 |
| σ（P）锐界面外推 **线性形式** | σ_inv | — | **-3.28 %** | 是 | 名义上是（−3.28）但形式不确定度 ±0.79 pp 使其**跨越 3 % 线** |
| σ（P）锐界面外推 **二次形式** | σ_inv | — | **-4.86 %** | 是 | 否 |
| 外推的模型形式不确定度 | 三种形式的截距散布 | — | W 2.10 pp / P 1.58 pp | — | **与 3 % 同量级 ⇒ 不能宣称达标** |
| 判别器 H_thickness / H_mobility | 预声明阈值 | 工件未取回 | — | NOT_RUN | NOT_RUN |

**判定**：算例门 **PASS**（−10.14 %，±15 %，与已合并报告一致，本单未改动该结论）；
「锐界面残差 < 3 %」= **不可判定（WITHHELD）**，因为外推函数形式尚未被数据识别，形式散布 1.6–2.1 pp
与 3 % 阈值同量级。这是对已合并报告一处**过强表述的收敛性收敛**（把单点 −3.28 % 改为区间），不是对算例判定的推翻。

## ⑦ 给下次自找活的建议线索

1. 回收 `cj4502`/`cj4503`（`KD_l096_m13`、`KD_l128_m10`、`KD_l192`、`KD_l256`）后重跑
   `verify/kdelta_similarity_test.py` —— 判别器会自动给出 H_thickness/H_mobility 的 yes/no，无需人工判断；
2. 若两者皆否，则下次该做的是**把 AC 迁移率按物理率缩放**（M ∝ 1/σ）而非按 λ，或直接换压力演化/双分布变密度模型
   （`docs/CASE-RT-TWOFLUID-PRECOND.md` ⑦-1 的结论）；
3. 有限密度比尖峰增长率（锚点的字面问题）仍为 INCOMPLETE，SCMP 参数区已用尽（寄生电流地板与目标速度信噪比 ≈ 1）。

## ⑧ 产物

| 路径 | 内容 |
|---|---|
| `benchmarks/case_pool/rayleigh_taylor_atwood05/verify/kdelta_similarity_test.py` | 三种外推形式 + LOO + 预声明判别器（一命令复现）|
| `benchmarks/case_pool/rayleigh_taylor_atwood05/verify/make_ansatz_report.py` | 本报告与图 5 的生成器（程序化注入）|
| `benchmarks/case_pool/rayleigh_taylor_atwood05/artifacts/kdelta/similarity_test.json` / `.md` / `ansatz_report.md` | 全部数字与表 |
| `benchmarks/case_pool/rayleigh_taylor_atwood05/figures/kdelta_ansatz.png` | 图 5（外推形式对比）|
| `tests/test_rt_kdelta_similarity.py` | 8 例单元测试（拟合/LOO/已发表数字复算）|
