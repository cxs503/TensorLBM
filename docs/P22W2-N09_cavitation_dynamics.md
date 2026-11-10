# P22W2-N09 — 空化与通气验证面：Kunz 输运、片-云转变、通气数标度律、推力降额

**仓库** TensorLBM · **模块** `tensorlbm.cavitation_dynamics` · **节点** `p22w2-n09`

## 0. 结论摘要（机器判据）

- 判据通过 **7/7**（未通过：无）
- Kunz 单常数标定 `C_dest = C_prod = C = 437.229`（对实测空泡长度 1.0, 0.8, 0.6 做最小二乘，目标 0.25, 0.4, 0.65 → 计算 0.2917, 0.4, 0.525）

### 判据表

| 判据 | 目标 | 计算值 | 容差 | 结果 |
|---|---|---|---|---|
| E1 | cavity length vs experiment < 20 % over sigma = 0.8-1.5 | 0.1933 | 0.2 | PASS |
| E2 | critical cavitation number vs experiment < 15 % | -0.09365 | 0.15 | PASS |
| E3 | thrust derating at sigma = 0.6 in 10-15 % | -0.1345 | [0.1, 0.15] | PASS |
| E4 | alpha_v -> 0 as sigma -> infinity | 8.451e-06 | 0.001 | PASS |
| E5 | p_inf >> p_v degenerates to single phase | 1.542e-05 | 0.01 | PASS |
| E6 | evaporation = condensation at steady state | 8.283e-14 | 0.01 | PASS |
| E7 | sheet-cloud shedding St_L in 0.2-0.3 below sigma_crit | 0.2505, 0.249, 0.2477 | [0.2, 0.3] | PASS |

## 1. 案例介绍（物理背景 / 几何 / 参数表）

空化预报是船舶总体性能的短板之一：叶片/翼型在低 σ 下产生片空化，片空化越过再入射流阈值后断裂为云空化并周期性脱落，造成升力下降、推力降额、结构激振。本节点搭建**时间精确**的验证面，与仓内已有的准定常链（`tensorlbm.cavitation_chain`，节点 P151W1-N04：等压空泡长度 + 给定Strouhal 数）互补：这里空泡长度、脱落频率、相变质量收支均由**输运方程解出来**。

| 项 | 取值 |
|---|---|
| 翼型 | NACA0015，α = 5.0° |
| 雷诺数 | 1e5-1e6 (panel Cp is inviscid; the experimental digest is Re = 1e5) |
| 来流 | U = 10 m/s, c = 0.1 m water at 20 C |
| 饱和蒸气压 | 2329.58 Pa（20 °C，Antoine） |
| 空化数定义 | sigma = (p_inf - p_v)/(0.5 rho U^2); local sigma(x) = sigma + Cp(x) |
| Kunz 形式 | Kunz (2000) with availability factors: evap C_dest alpha_l (-s), cond C_prod alpha_v s |

## 2. 方法（流程 / 关键设置 / 工具链）

1. **压力场**：Hess-Smith 面元法求 NACA0015 湿表面 Cp(x)（复用 `cavitation_chain.section`，200 面元，σ_i = −Cp_min = 1.97）。局部驱动量 s(x) = σ + Cp(x)：s<0 蒸发、s>0 凝结。
2. **Kunz 质量传递 + 一维非定常有限体积输运**：
   `∂α_v/∂t + ∂(u α_v)/∂x = S`，`S = C_dest(1-α_v)(-s)/t∞`（蒸发）、`S = -C_prod α_v s/t∞`（凝结），t∞ = c/U；通量一阶迎风（守恒通量形式），源项隐式分裂（保正，任意 dt 不越界）。来流边界 α=1e-5，出口零梯度。
3. **片-云转变（再入射流）**：汽相包络 L_cav 超过 L/c = 0.5（再入射流判据，带 0.40–0.60）后，闭合区形成以 U_jet = 0.25 U 上行的射流；射流扫过空泡（卷吸为云，质量守恒）、到达脱体点后整片空泡断裂为云，云以 0.9U 对流并在压力恢复区凝结，周期重复。**脱落频率是输出量**。
4. **推力降额**：叶元-动量理论（BEMT，固定点迭代求 a, a′）+ 剖面升力亏损（等压平台 `-∫(Cp+σ)dx` 归一化），逐方位角 θ 通过浸深 h(r,θ) 得到 σ_r(θ)，方位平均后给 ΔK_T/K_T0。
5. **工具链**：numpy 一维 FV 求解器 + matplotlib 图 + pytest 回归；全部数字写入 `artifacts/p22w2_n09/validation.json`，图与 sha256 写入 `figures.json`，本报告由脚本从 JSON 生成。

### 标定协议（声明，不隐藏）

Kunz 常数取 C_dest = C_prod = C，**单参数**在全部实测空泡长度点 σ = 1.0, 0.8, 0.6 上做最小二乘标定，得 C = 437.229（对数残差目标值 0.0694）。σ = 0.9 / 1.2 / 1.4 的长度、转变空化数 σ_critical、脱落频率与推力降额均为该单常数标定的**盲预测**；`kunz_constant_sensitivity` 表给出 E1/E2 随 C 的权衡，便于审计标定选择。**刻意不用验收判据本身去挑 C**（那会构成循环论证）：标定只对准实测长度，因此 E1 的 19.3 % 接近容差并不奇怪——表中 C = 874 处 E1 可降到 5.6 %，但那是按判据挑参数，本文不采用。

## 3. 计算结果（数字表，程序化注入）

### E1 空泡长度（σ = 0.8–1.5，容差 <20 %）

| σ | L_cav/c 计算 | L_cav/c 实验（数字化曲线） | 相对误差 | 角色 |
|---|---|---|---|---|
| 1.00 | 0.292 | 0.255 | +14.4 % | calibration point (used by the least-squares fit of C) |
| 0.80 | 0.400 | 0.386 | +3.6 % | calibration point (used by the least-squares fit of C) |
| 0.90 | 0.342 | 0.310 | +10.1 % | blind prediction |
| 1.20 | 0.217 | 0.182 | +19.3 % | blind prediction |
| 1.40 | 0.158 | 0.136 | +16.2 % | blind prediction |

窗口内（含标定点）最大误差 **+19.3 %**，盲预测（3 点）最大误差 **+19.3 %**（容差 20 %）。空泡长度定义：vapour extent (alpha_v >= 0.1) from the leading edge, time mean。

### E2 临界空化数（容差 <15 %）

- 模型：σ_critical = **0.6313**（在 `sigma_critical_dynamics` 中用二分法定位汽相包络穿过 L/c = 0.5 的位置）
- 参考：σ_critical = **0.6965**（log-log inversion of the digitised measured L/c(sigma) at L/c=0.5）
- 相对误差 **-9.4 %** → PASS

### E3 推力降额（σ_0.7R = 0.6，经验带 10–15 %）

- 面积分数约定（仓内经验链约定）：ΔK_T/K_T0 = **-13.5 %**
- 平台积分约定（分剖面完整升力亏损，上界）：-26.8 %
- K_T0 = 0.1809，K_T,cav = 0.1565；方位角分辨范围 +13.3 % … +13.6 %
- σ 约定：sigma_0.7R = (p_atm + rho g h - p_v)/(0.5 rho W_0.7R^2) at theta=pi/2；σ 扫描（面积分数约定）：0.40→-23.1%, 0.50→-17.6%, 0.60→-13.5%, 0.70→-10.3%, 0.80→-7.9%, 1.00→-4.7%, 1.20→-2.8%, 1.60→-0.9%, 2.50→-0.0%

### E4 / E5 负控

| σ | α_v 平均 | α_v 最大 |
|---|---|---|
| 2.0 | 2.078e-07 | 1.130e-05 |
| 3.0 | 1.111e-07 | 1.010e-05 |
| 4.0 | 9.321e-08 | 9.402e-06 |
| 8.0 | 7.575e-08 | 8.451e-06 |

E4（σ→∞ 时 α_v→0）：单调衰减 = True，σ=8 处 α_v,max = 8.45e-06 → PASS

E5（p_∞ ≫ p_v 退化单相）：σ = 20.0（p_∞/p_v ≈ 5.27）时相变总量 / 空化工况 = 1.54e-05，α_v,max = 7.93e-06 → PASS

### E6 质量守恒（稳态：蒸发 = 凝结 + 流出）

- 蒸发 0.1856 kg/m²、凝结 0.1856 kg/m²、流出 1.12e-37 kg/m²、限制器 0 kg/m²
- 收支残差 **8.28e-14**（容差 1 %）→ PASS
- 统计窗口：chord times >= 8.0 (start-up inventory excluded)（排除启动充装，得到真正的稳态收支）

### E7 片-云转变与脱落频率（St_L ∈ 0.2–0.3）

| σ | 脱落 | 事件数 | f_shed [U/c] | 谱基频 | L_break-off/c | St_L | St_c |
|---|---|---|---|---|---|---|---|
| 0.70 | False | 0 | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.000 |
| 0.60 | True | 12 | 0.4714 | 0.4707 | 0.531 | 0.251 | 0.471 |
| 0.55 | True | 11 | 0.4374 | 0.4377 | 0.569 | 0.249 | 0.437 |
| 0.50 | True | 10 | 0.4133 | 0.4127 | 0.599 | 0.248 | 0.413 |

喷射比 c_j = 0.25；St_L 定义 St_L = f·L_break-off/U，L_break-off 为再入射流起始时刻的汽相包络长度；谱基频为汽相质量信号 FFT 的基频（独立交叉校验）。

再入射流速度比敏感性（实测带 0.2–0.4）：

| c_j | f_shed [U/c] | St_L |
|---|---|---|
| 0.20 | 0.3497 | 0.199 |
| 0.25 | 0.4374 | 0.249 |
| 0.30 | 0.5246 | 0.299 |

St_L 区间 0.199, 0.299。

### Kunz 单常数敏感性（模型形 vs 标定）

| C | E1 最大误差 | σ_critical | E2 误差 |
|---|---|---|---|
| 109.3 | +60.6 % | 0.7836 | +12.5 % |
| 218.6 | +37.7 % | 0.6898 | -1.0 % |
| 437.2 | +19.3 % | 0.6313 | -9.4 % |
| 874.5 | +5.6 % | 0.5961 | -14.4 % |
| 1749 | +9.4 % | 0.5609 | -19.5 % |

> E1 and E2 pull in opposite directions along the single constant C: the least-squares calibration (C = 437.2) gives E1 19.3 % / E2 -9.4 %, and the best joint margin over the scanned C is 0.96 of the tolerances (C = 874.5: E1 5.6 %, E2 -14.4 %).  A single constant cannot drive both errors below the few-percent level -- that is the model-form error, not the calibration.

## 4. 图

### fig1_cavity_length_vs_sigma.png

![fig1_cavity_length_vs_sigma.png](../artifacts/p22w2_n09/fig1_cavity_length_vs_sigma.png)

*空泡长度—空化数：模型 vs 数字化实测点（±20 % 带）* · sha256 `84e193011f5db5b4…`

### fig2_alpha_v_spacetime.png

![fig2_alpha_v_spacetime.png](../artifacts/p22w2_n09/fig2_alpha_v_spacetime.png)

*α_v(x,t) 时空图：片空化生长—射流断裂—云对流（σ=0.55）* · sha256 `c0ce3294e1fb6057…`

### fig3_shedding_frequency.png

![fig3_shedding_frequency.png](../artifacts/p22w2_n09/fig3_shedding_frequency.png)

*汽相包络时序与脱落谱，标注事件频率与 St_L* · sha256 `6ffc8f002ad46880…`

### fig4_thrust_derating.png

![fig4_thrust_derating.png](../artifacts/p22w2_n09/fig4_thrust_derating.png)

*推力降额—σ（两种约定）与逐方位角（叶面位置）分辨* · sha256 `efdf76bea834c847…`

### fig5_profiles_and_budget.png

![fig5_profiles_and_budget.png](../artifacts/p22w2_n09/fig5_profiles_and_budget.png)

*收敛汽相剖面（多 σ）与稳态相变收支（E6）* · sha256 `b336b286e8a9536c…`

## 5. 注意点（踩过的坑 / 局限 / 适用域）

1. **压力钳制会造假**：最初在 α_v ≥ 0.5 处把驱动压差置零（“空泡内压力等于 p_v”），空泡变成冻结态，网格加密后 σ=0.8 的 L/c 从 0.41 跳到 0.72（非唯一解）。改为始终用湿表面 s(x) 驱动后解在 n_cells = 140/200/280 上收敛到 1 % 以内。
2. **源项线性化必须与隐式格式一致**：蒸发项写作 `C_dest(1-α)` 的线性形式时不能把 (1-α) 塞进常数项，否则 α→1 时出现负源，稳态长度整体偏移（该 bug 由 pytest 的可用性因子用例捕获）。
3. **质量收支要排除启动充装**：整体累计量里含初始充装，直接相减会给出 1.3 % 的假残差；按 t ≥ 8c/U 重新起算后残差降到 1e-13 量级。出口通量与限制器通量也必须入账（5 项收支）。
4. **低 σ 段偏短是模型形限制**：σ ≤ 0.7 时模型汽相包络比实测短约 25–30 %（Re=1e5 下实测空泡受层流分离/三维增厚支配，无粘面元 Cp + 一维输运无法复现），因此 E2 转变空化数偏低。这是**模型形误差**而非标定误差：单常数敏感性表显示，把 C 调小可让 E2 进入 15 %，但 E1 随即退化到 20 % 边缘，说明一个常数无法同时压住两处——需要粘性/三维闭合模型。
5. **推力降额的数值依赖约定**：分剖面平台亏损（完整物理）在 σ_0.7R = 0.6 给出 26.8 %，面积分数约定给 13.5 %。经验 10–15 % 带是在面积分数约定下定义的（仓内经验链同约定），两种口径并列报告，避免“选口径凑判据”。
6. **脱落频率与射流比强耦合**：周期 ≈ 生长时间 + 射流渡越时间，生长时间只占约 5 %，故 St_L ≈ 0.95 c_j。因此 St_L 落在 0.2–0.3 主要来自实测射流速度带（0.2–0.4）的约束，不是对脱落频率的独立预测；Hammond/Arndt 类实验若给出 c_j > 0.32，本模型 St_L 将越出该带。

## 6. 误差判定（逐项 vs 基准，显式回答“是否 < 3 %”）

| 量 | 基准 | 误差 | < 3 % ? |
|---|---|---|---|
| L_cav/c（σ=0.8…1.5，窗口内全部点） | 实测数字化曲线 | 最大 +19.3 % | 否（>3 %） |
| σ_critical | 实测曲线反演 | -9.4 % | 否（>3 %） |
| ΔK_T/K_T0（σ=0.6） | 经验 10–15 % 带 | 13.5 %（带内） | 否（带为基准，非点值） |
| 稳态质量收支残差 | 严格守恒 | 8.28e-14 | 是（1e-5 量级） |
| St_L | 实测带 0.2–0.3 | 0.251, 0.249, 0.248 | 见上：带内，但由 c_j 约束 |

**是否 < 3 % 的显式回答**：空泡长度、临界空化数与推力降额三项均**未达 3 %**（工程容差为 15–20 %，本节点判据即按工程容差设定）；只有守恒量（E6）达到机器精度量级。无基准的量（脱落频率的独立预测）已在第 5 节声明其参数依赖。

## 7. 审计链

- 判据数字：`artifacts/p22w2_n09/validation.json`（本报告所有数字的唯一来源）
- 图与 sha256：`artifacts/p22w2_n09/figures.json`
- 代码：`src/tensorlbm/cavitation_dynamics/{kunz_fv,thrust,references,validation}.py`
- 回归：`tests/test_cavitation_dynamics.py`（26 个用例，pytest 全绿）
- 复现：`PYTHONPATH=src python -m tensorlbm.cavitation_dynamics.validation` / `PYTHONPATH=src python scripts/p22w2_n09_cavitation_dynamics.py` / `PYTHONPATH=src python scripts/p22w2_n09_report.py`
- 运行环境：3.11.6 / numpy 2.4.6
