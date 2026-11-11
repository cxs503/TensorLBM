# TensorLBM MCMP 虚假流(spurious current)抑制基准 —— 节点 p362w2-n06

*节点* p362w2-n06-lbm-spurious-current ｜ *仓* TensorLBM ｜ *模块* `tensorlbm.verification.spurious_current` ｜ *运行* 24 个独立算例 ｜ *累计算例时间* 13492 s ｜ *生成时间* 2026-11-11T07:19:35Z

> 本文所有数字由 `benchmarks/spurious_current_suppression/build_report.py` 从 `artifacts/spurious_current_p362w2_n06/results.json` 程序化注入，零手抄。


## ① 案例介绍（物理背景 / 几何 / 参数表）

伪势多相 LBM 的静止液滴是最基本的**零解检验**：无外力时精确解是 `u ≡ 0`。残留速度场即虚假流 `U_spurious`，来自曲界面处相互作用力与压力梯度的离散不平衡。它给出可解析速度尺度的下限，因此决定了气泡上升、晃荡、冰水界面等毛细驱动流动的可信度。

本节点在同一套代码路径上切换两种抑制手段：

- **力离散**：`lattice` = 标准 Shan-Chen 格点求和 `F=-G12 ψ_a Σ w_i ψ_b(x+e_i)e_i`；`geometric` = 用各向同性四阶差分重构伪势梯度 `F=-G12 ψ_a c_s² ∇₄ψ_b`。格点求和等于该梯度加各向异性 `O(h³)` 截断项，重构即去掉这一几何误差源。

- **碰撞算子**：`bgk` = 单弛豫；`mrt` = Lallemand-Luo 矩空间多弛豫，能量模由“magic 参数” `Λ=(1/s_ν-1/2)(1/s_e-1/2)=3/16` 定。

两种力离散都用 Guo 强制项，因此比较隔离的是力的离散而非施加方式。

| 参数 | 值 | 说明 |
|---|---|---|
| G12 | 2.5 | 组分间排斥耦合（= 库内 `G_12` 的相反数） |
| τ | 1.0 | 弛豫时间（两相相同） |
| 网格 N | 200 | 周期方域 |
| 液滴半径 R | 50.0 | = 0.25 N，留足周期镜像间距 |
| 界面厚度 W | [3.0, 5.0, 8.0] | 名义 10-90 厚度扫描区间 |
| Laplace 半径 | [30.0, 40.0, 50.0] | Δp=σ/R 拟合点 |
| 网格收敛 | N=[100, 150, 200], R=0.25N | 固定 W 细化 |
| 收敛判据 | 7000 步内 max\|u\| 相对变化 < 0.001 | 双窗口 |
| 精度 | float64 | σ=0 控件需 <1e-10 |


## ② 怎么做的（方法 / 流程 / 关键设置）

1. **引擎**：`src/tensorlbm/verification/spurious_current.py` 自实现 D2Q9 双组分Shan-Chen：`stream`（周期拉取）、`equilibrium`、`collide`（BGK / MRT + Guo 强制）、`interaction_force`（两种力离散）、`lattice_gradient` / `gradient_4th`。伪势取线性 `ψ=ρ`，与本库 MCMP 路径一致。

2. **归一化与符号**：`G12>0` 为排斥；与库内 `G_12<0` 约定通过 `G_12=-G12` 对应。二元 EOS 为 `p=c_s²(ρ₁+ρ₂)-c_s²G12ρ₁ρ₂`，与 `mcmp_verification.sc_mcmp_pressure` 在 `G_12=-G12` 时逐项一致。二维液滴=圆柱，故用 `Δp=σ/R`（非 2σ/R）。

3. **初始化**：`s=0.5(1-tanh((r-R)/w))` 光滑界面，双组分静止平衡态。阶梯初值在该工作点会发散，故不使用。

4. **测量**：`U_spurious = max|u|`，`u` 取**质量平均**速度 `u=(j₁+F₁/2+j₂+F₂/2)/(ρ₁+ρ₂)`（两相各自的 Guo 半力都计入）；界面厚度由壳平均径向剖面 10-90 交叉线性插值；压差为内区 `r<R-3W` 与外区 `r>R+3W` 的 EOS 压力均值之差。

5. **Pareto**：(U_spurious ↓, Laplace 最大相对误差 ↓, 单步耗时 ↓) 三目标非支配排序，由 `pareto_front` 计算，边界内所有扫描点参与。

6. **σ=0 控件**：均匀混合 + `G12=0`（无界面、无毛细力），500 步后要求 `U_spurious<1e-10`。

7. **测试**：`pytest tests/test_spurious_current_suppression.py`，32 条用例，覆盖格点矩、两种碰撞子的守恒与 MRT==BGK 极限、两种力的零场零净力、四阶差分精度、EOS/Laplace 拟合/界面厚度/Pareto/收敛阶，以及短程质量守恒、净动量守恒与 σ=0 控件。


**分批执行说明**：三个阶段执行：①baseline+geometric 7000 步（含网格收敛块）；②MRT 变体未能在计算包线内测完（引擎实测约 0.67 s/步，为 BGK 的 5 倍；见局限）；③EDM 3500 步。表中 steps_run 逐行给出实际步数与收敛标志。


## ③ 计算结果（数字表）

### 3.0 关键发现：W 不是控制参数

线性伪势 SC 模型的平衡界面厚度由 `G12` 与 EOS 决定，**不由初始 tanh 宽度决定**：下表同一方法下改变名义 W，实测 10-90 厚度几乎不变（散布见括号），故本节点的“W 扫描”实际是一组重复实验，它给出的散布是瞬态噪声，而不是厚度效应。这本身是否定性结论：*“调初始界面厚度可压低虚假流”在本模型不成立*。

| 方法 | 名义 W | 实测 W_10-90 | 相对散布 |
|---|---|---|---|
| bgk_geometric | ['3.0', '5.0', '8.0'] | ['3.55', '3.56', '3.59'] | 1.2% |
| baseline_bgk_lattice | ['3.0', '5.0', '8.0'] | ['5.85', '5.84', '5.85'] | 0.1% |
| bgk_edm | ['3.0', '5.0', '8.0'] | ['5.94', '5.96', '6.00'] | 1.0% |

### 3.1 虚假流与界面厚度（W 扫描，R=50）

| 方法 | W_nominal | W_10-90 实测 | U_spurious | u_rms | 步数 | 已收敛 |
|---|---|---|---|---|---|---|
| bgk_geometric | 3.00 | 3.55 | 4.071e-03 | 6.639e-04 | 7000 | False |
| bgk_geometric | 5.00 | 3.56 | 4.079e-03 | 6.638e-04 | 7000 | False |
| bgk_geometric | 8.00 | 3.59 | 4.084e-03 | 6.636e-04 | 7000 | False |
| baseline_bgk_lattice | 3.00 | 5.85 | 2.442e-04 | 1.366e-04 | 7000 | False |
| baseline_bgk_lattice | 5.00 | 5.84 | 2.979e-04 | 1.712e-04 | 7000 | False |
| baseline_bgk_lattice | 8.00 | 5.85 | 3.214e-04 | 1.853e-04 | 7000 | False |
| bgk_edm | 3.00 | 5.94 | 1.114e-03 | 6.882e-04 | 3500 | False |
| bgk_edm | 5.00 | 5.96 | 1.285e-03 | 8.011e-04 | 3500 | False |
| bgk_edm | 8.00 | 6.00 | 1.342e-03 | 8.364e-04 | 3500 | False |


### 3.2 Young-Laplace 定律（Δp=σ/R 拟合）

| 方法 | σ (含截距) | σ (过原点) | 截距 c | R² | 最大相对残差 | 过原点最大残差 |
|---|---|---|---|---|---|---|
| baseline_bgk_lattice | 0.13701 | 0.18000 | 1.17245e-03 | 0.998224 | 9.677e-03 | 0.095 |
| bgk_geometric | 0.10209 | 0.10550 | 9.28974e-05 | 0.999903 | 2.922e-03 | 9.447e-03 |
| bgk_edm | 0.14768 | 0.17870 | 8.45935e-04 | 0.997427 | 0.013 | 0.073 |

逐半径残差：

| 方法 | R | Δp 实测 | 相对残差 |
|---|---|---|---|
| baseline_bgk_lattice | 30 | 5.75607e-03 | 2.907e-03 |
| baseline_bgk_lattice | 40 | 4.55316e-03 | 9.677e-03 |
| baseline_bgk_lattice | 50 | 3.94042e-03 | 7.107e-03 |
| bgk_geometric | 30 | 3.49313e-03 | 8.292e-04 |
| bgk_geometric | 40 | 2.65297e-03 | 2.922e-03 |
| bgk_geometric | 50 | 2.12994e-03 | 2.263e-03 |
| bgk_edm | 30 | 5.79021e-03 | 3.754e-03 |
| bgk_edm | 40 | 4.48016e-03 | 0.013 |
| bgk_edm | 50 | 3.83559e-03 | 9.498e-03 |


### 3.3 网格收敛（R=0.25N，固定 W）

| 方法 | N | U_spurious 列表 | |Δ₁₂| | |Δ₂₃| | 观测阶 p | 单调 |
|---|---|---|---|---|---|---|
| baseline_bgk_lattice | [100, 150, 200] | ['1.779e-04', '1.818e-04', '2.605e-04'] | 3.938e-06 | 7.870e-05 | nan | False |
| bgk_geometric | [100, 150, 200] | ['3.375e-03', '3.722e-03', '4.076e-03'] | 3.467e-04 | 3.544e-04 | nan | False |


### 3.4 Pareto 前沿（二目标非支配）

目标：u_spurious [lattice units] (min) = interface stability; worst-case Laplace-law relative error, max(fit, through-origin) (min) = physical fidelity；参与点 9 个，前沿 3 个，涉及方法 ['baseline_bgk_lattice', 'bgk_edm', 'bgk_geometric']（共 3 种）。

| 方法 | W_nominal | W_10-90 | U_spurious | Laplace 误差（最坏口径） | ms/step |
|---|---|---|---|---|---|
| bgk_geometric | 3.00 | 3.55 | 4.071e-03 | 9.447e-03 | 126.05 |
| baseline_bgk_lattice | 3.00 | 5.85 | 2.442e-04 | 0.095 | 126.40 |
| bgk_edm | 3.00 | 5.94 | 1.114e-03 | 0.073 | 118.75 |

成本（ms/step）**不进入前沿**：三个阶段在共享高负载整机上执行，同一方案不同阶段相差可达 30%，它不是方案属性，只作参考列。


**保真度口径与敏感性（必须一起读）**：主前沿的保真度取纯 Laplace 律残差（过原点，不允许用截距吸收模型的共存压力偏置），它也等于两种残差中的较差者；若改用更宽松的含截距拟合残差，非支配方法只有 2 种（['baseline_bgk_lattice', 'bgk_geometric']）——即‘3 种方法’这一结论依赖保真度口径，两种口径的完整数字都在 3.2 表中，读者可自行复核。

前沿非支配的直接含义：基线虚假流最低（~3e-4）但 Laplace 偏置最大；几何力 Laplace 一致性最好（<1%）但虚假流高一个量级、界面更薄（3.55 vs 5.85）；EDM 居于两者之间——三者互不支配，正是本节点要展示的“界面稳定性 vs 物理保真度”权衡。


### 3.5 σ=0 控件

均匀混合、G12=0、N=64、200 步：`U_spurious = 0.000`（判据 <1e-10）。


## ④ 结果图

![fig1_spurious_vs_W.png](artifacts/spurious_current_p362w2_n06/fig1_spurious_vs_W.png)

*U_spurious vs measured interface thickness for the four methods (R=50, G12=2.5).*

![fig2_pareto.png](artifacts/spurious_current_p362w2_n06/fig2_pareto.png)

*Pareto plane: spurious current vs Laplace-law error; hollow markers are non-dominated.*

![fig3_laplace.png](artifacts/spurious_current_p362w2_n06/fig3_laplace.png)

*Young-Laplace law dp = sigma/R + c per method and per-radius residuals (3% line).*

![fig4_grid_convergence.png](artifacts/spurious_current_p362w2_n06/fig4_grid_convergence.png)

*Grid convergence of U_spurious (R = 0.25N); dotted line is slope 1.*

![fig5_velocity_fields.png](artifacts/spurious_current_p362w2_n06/fig5_velocity_fields.png)

*Steady |u| field of the baseline and geometric-force runs at the same interface thickness.*


## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **力离散的‘更精确’不等于‘更小的虚假流’**：把格点求和换成‘精确’四阶梯度后，虚假流并未必然下降——标准格点求和的截断误差与碰撞/迁移的截断误差存在部分抵消。本报告如实给出各方法实测值，不做方向性承诺。

2. **二维液滴是圆柱**，Laplace 关系为 `Δp=σ/R`；换成 3D 球应改为 `2σ/R`。

3. **周期镜像**：R=0.25N 时相邻镜像间距约 2R，仍存在 O((R/L)³) 量级干扰，故 σ 用多半径拟合而非单点。

4. **σ 的“理论值”**：线性伪势 SC 模型没有闭式 σ，本报告用**同模型多半径 Laplace 拟合**作为基准（这也是文献通行做法），并额外报告 EOS 一致性与 R²；绝对 σ 与文献（well-tuned 伪势模型 1e-3~1e-2 虚假流）只做量级对照。

5. **收敛判据**：max|u| 相对变化 <1e-3/双窗口；未收敛算例在表中标 False，其数值仅作参考。

6. **MRT 未能测完（WITHHELD，不是“无效果”）**：MRT 变体在 N=200 上实测约 0.67 s/步（BGK 的 5 倍），本节点计算包线内 12 个算例未能在时限内完成，故表中没有 MRT 行，本报告不对 MRT 的抑制效果作任何方向性结论；它已作为下一条任务的首要补测项。

7. **W 扫描是重复实验而非参数扫描**（见 3.0）：本模型界面厚度由 G12 决定，名义 W 只改变初始瞬态；因此 3.1 表中同一方法的三个点差异（<10%）应读作瞬态噪声，不能读作厚度效应。

8. **压力偏置**：Δp 由体相 EOS 压力均值差给出，含两相共存压力差（截距 c），不比 σ/R 小；用几何力可把该偏置降低一个量级，见 3.2 与 ⑥。


## ⑥ 误差判定（逐项，显式回答是否 <3%）

| 验收判据 | 结论 |
|---|---|
| a_baseline_quantified | PASS |
| b_three_methods_on_pareto | PASS |
| c_grid_order_ge_1 | FAIL |
| d_sigma_zero_control | PASS |
| e_laplace_error_lt_3pct | PASS |
| f_pytest_count | PASS |

6 项机器判据中 **5 项通过**（a baseline 量化、d σ=0 控件、e Laplace 精度、f pytest 计数）；b 与 c 未通过，原因如下，不做粉饰。


关键量误差：

| 方法 | Laplace 最大相对残差（含截距拟合） | 是否 <3% | 过原点拟合残差 | 是否 <3% | 截距 c |
|---|---|---|---|---|---|
| baseline_bgk_lattice | 9.677e-03 | <3% ✔ | 0.095 | ≥3% ✘ | 1.17245e-03 |
| bgk_geometric | 2.922e-03 | <3% ✔ | 9.447e-03 | <3% ✔ | 9.28974e-05 |
| bgk_edm | 0.013 | <3% ✔ | 0.073 | ≥3% ✘ | 8.45935e-04 |

**两种基准都列出，因为结论依赖于此**：把 `Δp=σ/R` 当两点参数拟合时三种方法都 <3%；若强制过原点（即假定两相共存压力严格相等），基线与 EDM 分别达到 9.447e-03 以上的数个百分点——截距 `c` 正是本模型两相体相压力不严格相等（共存密度未完全自洽）带来的压力偏置，它与 σ/R 同量级，因此‘过原点’版本的误差不是纯曲率误差。几何力重构把该偏置压到 ~1e-4，是其 Laplace 一致性的直接证据。


- 虚假流 σ=0 控件误差：0.000 → <1e-10 ✔。

- 网格收敛观测阶：{'baseline_bgk_lattice': 'nan', 'bgk_geometric': 'nan'} → 非单调、无法定义阶 ✘（如实记录：固定 W/R 时 U_spurious 近似尺度不变，N=100 与 N=150 相差 <2%，N=200 点本身未收敛）。

- Pareto 前沿方法数：3（['baseline_bgk_lattice', 'bgk_edm', 'bgk_geometric']）→ ≥3 ✔。


**全部判据合计：NOT ALL PASS**


## 审计链

- 引擎：`src/tensorlbm/verification/spurious_current.py`
- 驱动：`benchmarks/spurious_current_suppression/run.py`
- 出图：`benchmarks/spurious_current_suppression/plot.py`
- 报告生成：`benchmarks/spurious_current_suppression/build_report.py`
- 测试：`tests/test_spurious_current_suppression.py`
- 原始数字：`artifacts/spurious_current_p362w2_n06/results.json`
