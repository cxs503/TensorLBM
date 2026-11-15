# XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION

**锚点** `XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION`（lbm-principles-practice/code 的 shan-chen-laplace-pressure-calibration 做法）｜**仓库** TensorLBM｜**分支** `agent/m21141`
**判据** 对标误差 <15%（owner 精度标准），关键量显式回答是否 <3%｜**日期** 2026-11-15

---

## ① 案例介绍（物理背景 / 几何 / 参数）

**做法来源（零手抄，逐行引用）**：`lbm-principles-practice/code`（MIT）`chapter9/shanchen.cpp`

| 环节 | 上游位置 | 上游做法 |
|---|---|---|
| 压力场 | `:126-139` | `p = Σ_s ρ_s/3 + Σ_s Σ_ss g[s][ss] ψ_s ψ_ss / 6` |
| 汽/液压力 | `:333-353` | 角上 **4×4** 盒平均（汽）+ 域中心 **4×4** 盒平均（液） |
| 半径 | `:357-367` | 中线行上找 ρ 穿过 `ρ_av=(ρ_g+ρ_l)/2` 处线性插值 |
| 表面张力 | `:369` | `γ = Δp · r`（Young–Laplace 标定） |
| 伪势 | `:74-76` | `ψ(ρ)=ρ₀[1−exp(−ρ/ρ₀)]`，ρ₀=1 |
| 上游缺陷 | `:358` | `double rad;` **未初始化**；扫描不 break 时打印未定义 γ（实测 R=24 得 γ=−4.86） |

**本文的验证对象**：TensorLBM 已合并的 SCMP 压力/Laplace 标定链——`tensorlbm.multiphase`（`psi_exp`/`sc_force`/`collide_guo`/`step_sc`）与 `tensorlbm.wetting.measure_young_laplace`/`run_young_laplace`。

**几何与参数**：二维 D2Q9，周期性方形域；液滴半径扫描 R=10/12/14（48²），对照大域 64²（R=16）；τ=1.0；G=5.0（库约定 G>0 = 吸引）；ρ₀=1，ψ=1−e^{−ρ}；液/汽初值 1.96/0.16；界面 tanh 宽度取 1.5（库默认）/3.0。

---

## ② 怎么做的（方法 / 工具链）

1. **压力项符号与系数的一阶推导**（解析）：库力 `F = G·cs²·ψ·Σ_i w_i c_i ψ(x+c_i)`（`G>0` 吸引，见 `sc_force` docstring），且 `Σ_i w_i c_i ψ(x+c_i) ≈ cs²∇ψ`。连续极限 `F = −∇p_int` ⇒ **`p_int = −(G cs²/2)ψ²`**，即
   `p(ρ) = ρ/3 − G ψ²/6`（吸引使压力**降低**）。
2. **解析基准（Maxwell 构造）**：对同一状态方程做等压 + 等化学势（`μ=∫(1/ρ)dp/dρ dρ`）构造，数值求共存密度 ρ_v/ρ_l。该基准不依赖任何格子模拟，用来独立判定压力项的**符号与系数**。实现在 `src/tensorlbm/multiphase/laplace.py::sc_maxwell_coexistence`。
3. **上游标定器移植 + 加固**：`laplace_pressure_calibration`（4×4 盒、中线插值半径、`γ=Δp·r`）与 `interface_radius`；对上游 UB 加护栏：找不到穿越即抛 `RadiusEstimationError`，绝不返回未定义半径。
4. **Laplace 律拟合**：`fit_laplace`（过原点最小二乘 `Δp=σ/R`，给 σ、R²、最大相对残差）。
5. **端到端复核尝试**：用库 `step_sc` 跑半径扫描（`xval/.../run_xval.py`），试图给出 σ 的格子实测值与 `SIGMA_DEFAULT=0.05611` 的偏差。
6. 工具链：Python 3.11 / torch（float64）/ numpy / matplotlib；测试 `pytest`。

---

## ③ 计算结果（程序化产出，零手抄）

`artifacts/laplace_calibration_analytics.json`、`artifacts/xval_results.json`。

**表 1｜压力项符号判定（G=5，ρ_l=1.957，ρ_v=0.1596）**

| 压力式 | Δp = p_l − p_v | 与正确符号之比 |
|---|---|---|
| `p=ρ/3 − Gψ²/6`（本文推导，与库力一致） | **+2.770e−3** | 1 |
| `p=ρ/3 + Gψ²/6`（`wetting.measure_young_laplace` 实际所用） | +1.195 | **431.6×** |

**表 2｜Maxwell 共存的独立复核**

| 组别 | ρ_l | ρ_v | 基准 | ρ_l 误差 | ρ_v 误差 |
|---|---|---|---|---|---|
| Maxwell（G=5） | 1.9289 | 0.15289 | 库文档常数 1.957/0.1596 | **1.44%** | **4.20%** |
| Maxwell（G=4.7） | 1.6780 | 0.20322 | 参考 C++ 实测 1.6707/0.18473 | **0.43%** | **10.0%** |

**表 3｜端到端 Laplace 扫描（库 `step_sc`）——诚实记录**

| 臂 | 设置 | 结果 |
|---|---|---|
| 库当前力（含 `cs²` 前置因子） | G=5，R=10，48²，width=3，G 斜坡 800 步，4000 步 | **未相分离**：ρ 均匀 0.4173 ⇒ 无法标定 |
| 去掉 `cs²` 前置因子（k=G/3=1.667，与参考 C++ 同量级） | G=5，R=10/12/14，48²，width=3，2500 步 | **发散**，步数 24/23/21 |
| 库自带 `run_young_laplace(radius=16, 64²)` | G=5 与 G=2，各 1500 步 | **status='diverged'** |
| 参考 C++ 默认（width≡1.5 陡界面，无斜坡） | G=5，R=14，48²，600 步 | **发散**（第 14 步） |

⇒ **σ 的格子实测值：未取得（WITHHELD）**，不能声称与 `SIGMA_DEFAULT=0.05611` 达标或超标。

---

## ④ 结果图

- `artifacts/fig1_eos_maxwell.png`：SC 状态方程 `p(ρ)` 与两组 Maxwell 共存线（场量曲线 + 基准交点）。
- `artifacts/fig2_coexistence.png`：共存密度柱状对比（库常数 / Maxwell G=5 / Maxwell G=4.7 / 参考实测）。
- `artifacts/fig3_pressure_sign.png`：两种压力符号下的 Δp（对数轴，431.6× 差）。

（云图项：本次未取得相分离的稳定液滴场，故无 ρ/p 云图；原因见 ③ 表 3，属 honest FAIL 说明。）

---

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **上游 UB 必须吸收而不照抄**：`chapter9/shanchen.cpp:358` 的未初始化 `rad` 已用异常护栏替代；本文全部半径来自带护栏的 `interface_radius`。
2. **符号约定不能混用**：参考 C++ 用 `F=−g ψΣw c ψ`（g<0 吸引），库用 `F=G cs² ψΣw c ψ`（G>0 吸引），二者是对方的负数。压力项必须随约定翻符号；`wetting.measure_young_laplace` 取了参考的 `+gψ²/6` 形式却套库的 `G>0` 约定 ⇒ 分支取反（表 1）。
3. **`wetting.measure_young_laplace` 另外两处不可标定性**：`G=5.0` 被硬编码（`run_young_laplace(G=...)` 传非 5 的 G 时压力式与求解不一致）；σ 用的是**输入的名义半径**而非实测半径（质量不守恒时半径会漂移，见 OSS inventory §4.3：初值 R=15 → 平衡 16.32）。
4. **库 SC 求解器在文档默认参数下不可用**：`run_young_laplace` 默认 G=5 也 `diverged`；去掉 `cs²` 前置因子后以 Guo 碰撞在 R=10–14、48² 上于 20 余步发散。参考实现用「feq 用含 0.5F 的重心速度」的注入方式，库用标准 Guo（feq 用未移位速度），二者稳定性不同——这是**待办**，不是本文结论。
5. 适用范围：本文的解析结论（压力式符号/系数、Maxwell 共存）与求解器稳定性无关，可直接复用；Laplace 实测结论仅覆盖上述 β 参数。

---

## ⑥ 误差判定（逐项，显式回答 <3%）

| 关键量 | 本文值 | 基准 | 误差 | <15%? | <3%? |
|---|---|---|---|---|---|
| ρ_l（Maxwell G=5） | 1.9289 | 库常数 1.957 | 1.44% | ✅ | ✅ |
| ρ_v（Maxwell G=5） | 0.15289 | 库常数 0.1596 | 4.20% | ✅ | ❌ |
| ρ_l（Maxwell G=4.7） | 1.6780 | 参考 C++ 1.6707 | 0.43% | ✅ | ✅ |
| ρ_v（Maxwell G=4.7） | 0.20322 | 参考 C++ 0.18473 | 10.0% | ✅ | ❌ |
| Δp 符号正确性 | +2.770e−3 | 吸引必使 p_l>p_v | 定性一致 | ✅ | —（定性判定） |
| σ（Laplace 拟合） | **未取得** | `SIGMA_DEFAULT=0.05611` | — | **WITHHELD** | **未判定** |

**结论**：压力项符号/系数**已由解析 Maxwell 构造独立证实**（ρ_l 1.44%/0.43%，ρ_v 4.2%/10.0%，均 <15%）；`wetting.measure_young_laplace` 的 `+Gψ²/6` 分支被**否定**（Δp 放大 431.6×）。端到端 σ 标定因求解器发散/未相分离而**未取得**，按纪律不粉饰为达标。

---

## 附：产物与复现

- 新模块：`src/tensorlbm/multiphase/laplace.py`（`sc_pressure` / `interface_radius` / `laplace_pressure_calibration` / `fit_laplace` / `sc_maxwell_coexistence`）
- 测试：`tests/test_xval_shan_chen_laplace_pressure_calibration.py`（9 项，全绿）
- 复现：`python xval/XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION/run_xval.py`；`python .../make_figures.py`
- 已知未修（后续工单建议）：①`wetting.measure_young_laplace` 压力符号 + 硬编码 G + 名义半径；②`multiphase.sc_force` 额外 `cs²` 前置因子导致 G 的物理含义偏移（k=G/9）；③SC 求解器启动稳定性（**本条已在本单内在 code 上验证为未解决**，需要与参考实现一致的力注入方案）。
