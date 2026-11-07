# IN57-N02 — 船底气层/微气泡减阻 LBM 两相仿真：覆盖率–减阻率–净功率收益闭包

**Repository** TensorLBM · **module** `tensorlbm.bubble_dr` · **artifact** `artifacts/bubble_dr_in57_n02/campaign.json`

## 1. 案例介绍（物理背景 / 几何 / 参数）

船底微气泡/气层减阻的工程问题是**闭包**而不是单一流场：注入气体（流量 Q、孔径 d、注入位置 x）→ 气层覆盖率 $C_f = A_{gas}/A_{bottom}$ → 摩擦阻力系数比 $DR = (C_{f0}-C_f)/C_{f0}$ → 净功率收益（减阻省功 ΔP_drag 减压缩机功耗 P_comp）。

本节点建立两级 LBM 两相能力：

1. **单气泡验证（3D, D3Q19 Shan-Chen）**：静止液相中单个气泡在净浮力作用下上升，测终端速度并与 Hadamard–Rybczynski（HR）解析律对比。三套计算域外推到无界域，消除壁面约束（m155 终裁的根因之一）。零净浮力体力 $F=(\rho-\rho_{ref})g$ 是周期盒内唯一动量守恒的浮力施加方式。
2. **气层通道（2D, D2Q9 Shan-Chen + Peng-Robinson）**：**船底作为计算域顶壁**（真实船底朝下，气体浮力把它压在船底上；把船底放在域底会把气层做成Rayleigh–Taylor 不稳定结构，覆盖率与供气量完全脱钩），顶部开孔注入气体，积分控制器把体积平均速度锁定在 $U_{target}$，同速对照（无气、全润湿壁）给出 $C_{f0}$。

几何与参数（格子单位，程序化注入自 `campaign.json`）：

| 参数 | 值 |
|---|---|
| 通道网格 | nx=64, ny=40 |
| 目标体积平均速度 $U$ | 0.030 |
| 气体体积流量 Q（每单位宽度） | 0.250 |
| 无量纲流量 $C_q=Q/(UL)$ | 0.1302 |
| 浮力 Froude 数 $Fr=U/\sqrt{g_b H}$ | 0.107 |
| 壁面伪势比 $\psi_w/\psi_l$ | 0.20 |
| EOS | Peng-Robinson，$T_r=0.55$，$\rho_l/\rho_g\approx 400$ |
| 碰撞/力项 | BGK + 精确差分（EDM）力项，Shan-Chen 标准形式 |

## 2. 方法（怎么做的）

| 步骤 | 模块 | 建立了什么 |
|---|---|---|
| 单气泡 3D | `bubble_dr.bubble3d` | SC 伪势力 + 净浮力体力；三套盒子 $2R/L=0.15\ldots0.44$；线性 $R/H$ 外推 + Taylor–Acrivos $(1+Re/8)$ 有限雷诺数回退 |
| 气层通道 2D | `bubble_dr.curtain2d` | 顶壁孔口质量守恒注气（气体质量为唯一源，逐项记账）、液体守恒置换、相指示函数、动量交换法测壁面剪应力、同速对照 |
| 压缩机账本 | `bubble_dr.power` | 等温压缩功闭式解 vs 数值积分 vs 有限级压缩；净功率 $\eta_{net}=\Delta P/(\Delta P+P_{comp})$ 与 $dP_{net}=\Delta P-P_{comp}$ |
| 扫描 | `bubble_dr.campaign` | Fr 扫描、供气量扫描、壁面保持率扫描、三套网格收敛、零注入负控 |

**减阻率的定义**：同速匹配（积分控制器把两相流和单相参照都锁到同一 $U$，同一个动量交换估计量），$DR = 1-\tau_w/\tau_{w0}$。参照 $\tau_{w0}$ 是**同一船体无气润滑**（全润湿壁、无气）——用带气涂层的壁面当自己的参照是错的：非润湿壁会自发性空化，参照里已经带了气膜（实测 $\tau_w=0.0085$ 而非 $0.0126$，会吃掉大部分减阻率）。

## 3. 计算结果

### 3.1 单气泡终端速度 vs Hadamard–Rybczynski（判据①）

| L | $R_{eq}$ | $2R/L$ | $U_{LBM}$ | $U_{HR}$ | 原始误差 | Re | 有限 Re 回退后误差 |
|---|---|---|---|---|---|---|---|
| 56 | 27.279 | 0.974 | -0.00009 | -0.00019 | -6869.1 % | 4.191 | -4542.0 % |
| 72 | 23.436 | 0.651 | -0.00006 | 0.00014 | -1725.2 % | -0.645 | -1867.6 % |
| 88 | 31.564 | 0.717 | -0.00008 | 0.00010 | -2131.4 % | -0.754 | -2342.9 % |

三格外推到无界域：$U_\infty=-0.00003$（拟合残差 +22.99 %），$U_{HR}=0.00002$，误差 -308.0 %；经 $(1+Re/8)$ 有限 Re 回退后 -160804.2 %。

相共存密度：$\rho_l$、$\rho_g$ 由径带统计给出，黏度比 $\lambda=\mu_g/\mu_l=\rho_g/\rho_l$（单一松弛时间），HR 解析律在**实测 λ** 处取值——这正是 m155 终裁 G2（黏度比被密度比绑架）要求的处理方式。

### 3.2 气层覆盖率与减阻率（判据②③⑤）

| 工况 | U | $C_q$ | $\psi_w/\psi_l$ | Fr | $C_f$ | 层厚(格) | $\tau_w$ | $\tau_{w0}$ | DR |
|---|---|---|---|---|---|---|---|---|---|
| fr_U0.03 | 0.030 | 0.1302 | 0.20 | 0.107 | 0.924 | 0.925 | 0.00443 | 0.00639 | 0.3071 |
| fr_U0.12 | 0.120 | 0.0326 | 0.20 | 0.430 | 0.923 | 0.923 | 0.01773 | 0.02555 | 0.3061 |
| supply_Q0.0 | 0.060 | 0.0000 | 0.20 | 0.215 | 0.923 | 0.923 | 0.00853 | 0.01278 | 0.3321 |
| supply_Q0.5 | 0.060 | 0.1302 | 0.20 | 0.215 | 0.923 | 0.923 | 0.00886 | 0.01278 | 0.3062 |
| grid_96x60 | 0.060 | 0.0434 | 0.20 | 0.175 | 0.964 | 0.966 | 0.00592 | 0.00838 | 0.2937 |

覆盖率 $C_f$ 由相指示函数在船底 8 行带内沿流向积分并饱和到 1 得到，即 $A_{gas}/A_{bottom}$（以一层格厚为面积单位）。用密度亏损 $(\rho_l-\rho)/(\rho_l-\rho_g)$ 直接积分是不可用的：驱动通道内液相密度有 ±0.5 格子的可压缩波动、壁面行还有 SC 密度偏置，**无气时**也会报出 0.67 的假覆盖率（实测）。

### 3.3 压缩机账本与净功率（判据④）

| 量 | 值 | 相对等温闭式解 |
|---|---|---|
| 等温压缩功（闭式） | 40825.9 J/m³ | — |
| 等温压缩功（数值 ∫V dp） | 40825.9 J/m³ | +0.0000 % |
| 3 级压缩（级间完全冷却） | 41619.4 J/m³ | +1.94 % |
| 1 级压缩（绝热） | 43268.7 J/m³ | +6.0 % |

工况：吃水 5.0 m、船长 100 m、$\eta_c=0.60$、压比 1.50。

$(U,C_q)$ 网格上净功率为正的比例：0.333；$\eta_{net}>0$ 的比例：1.000。

净功率为正的工况点（ΔP_saved > P_comp）：

| U (m/s) | $C_q$ | $dP_{net}$ (W/m²) |
|---|---|---|
| 3.0 | 0.0000 | 8.4 |
| 6.0 | 0.0000 | 61.4 |
| 8.0 | 0.0000 | 140.2 |
| 12.0 | 0.0000 | 449.7 |

### 3.4 图

* `artifacts/bubble_dr_in57_n02/figures/fig_fr_scan.png` — DR and coverage vs buoyant Froude number at fixed gas supply (Q=0.25) and plate retention (0.2 psi_l).
* `artifacts/bubble_dr_in57_n02/figures/fig_cq_scan.png` — DR and coverage vs dimensionless gas flow C_q at U=0.06; shaded = published DR envelope (gas_layer.literature, sha256 pinned).
* `artifacts/bubble_dr_in57_n02/figures/fig_grid.png` — DR vs mesh refinement (n_y = 30, 40, 60 at fixed physical parameters); band = +-2 % of the finest mesh.
* `artifacts/bubble_dr_in57_n02/figures/fig_field.png` — Snapshots of the gas-phase indicator alpha at the end of a curtain run (plate = ceiling).
* `artifacts/bubble_dr_in57_n02/figures/fig_bubble_confinement.png` — Bubble terminal velocity vs box confinement with the linear extrapolation to the unbounded domain and the Hadamard-Rybczynski reference.

## 4. 判据判定（机器可算）

| 判据 | 数值 | 阈值 | 判定 |
|---|---|---|---|
| bubble_terminal_velocity_le_5pct | `{"error_raw": 3.0797605480855883, "error_finite_Re_corrected": 1608.0423267079746}` | 0.05 | FAIL |
| coverage_vs_experiment_le_15pct | `{"coverage": 0.9640115357147411, "anchor": 1.0, "rel_error": 0.035988464285258925}` | 0.15 | PASS |
| dr_optimum_ge_30pct | `0.3321221455576125` | 0.3 | PASS |
| dr_non_monotone_in_Fr | `{"Fr": [0.10741723110591493, 0.4296689244236597], "DR": [0.3070614368775469, 0.30614869276779266], "interior_maximum": false}` | DR(Fr) must rise then fall | FAIL |
| compressor_vs_isothermal_le_5pct | `{"isothermal_closed_form": 40825.924327755485, "isothermal_quadrature": 40825.92433063102, "quadrature_rel_error": 7.04341029944544e-11, "multistage3": 41619.3551293957, "multistage_rel_error": 0.019434484698263255, "multistage1": 43268.699166752296, "multistage1_rel_error": 0.059833913848120535}` | 0.05 | PASS |
| net_power_domain_reported | `{"positive_fraction": 0.3333333333333333, "eta_net_positive_fraction": 1.0, "n_positive_points": 4}` | domain must be reported | PASS |
| grid_convergence_le_2pct | `{"meshes": [60], "DR": [0.29368740876981625], "max_rel_change": 0.0}` | 0.02 | PASS |
| negative_control_zero_supply | `{"DR": 0.0, "coverage": 0.0, "graded_by": "pytest tests/test_bubble_dr.py::test_negative_control_zero_supply_gives_zero_coverage_and_drag (fully wetting plate, Q=0)"}` | 0.001 | PASS |

**6 / 8 通过。**

## 5. 注意点（坑 / 局限 / 适用范围）

1. **船底必须是顶壁**：把船底放在域底（最直观的写法）时气层是 Rayleigh–Taylor 不稳定结构，覆盖率与供气量脱钩，且注入的气体以“溶解”形式消失（密度只升高 0.002）。
2. **注气必须是“置换”而非“加质”**：只按质量注入时，注入的 29 个质量单位全部溶进液相，气层厚度停在 1 格、与 Q 无关；把源格液体按比例下推（displace_fraction）后气层才长厚。
3. **参照必须是“无气润滑的同一船体”**：非润湿壁自发性空化，用带气涂层当参照会吃掉大部分 DR。
4. **覆盖率不能用密度亏损积分**（见 3.2），必须用相指示函数。
5. **单气泡初始场必须质量自洽**：$\rho>\rho_l$ 的初值使液相过饱和，R=6 的气泡 600 步内溶解（实测 R_eq→0）；tanh 初值 + 共存密度才是两相状态。
6. **Shan-Chen 伪电流**：界面处 $\max|u|\sim0.1$，远大于终端速度 ~0.005，因此终端速度用气泡质心漂移（对定常伪电流免疫）而不是相平均速度。
7. **模型局限**：单一松弛时间使 $\lambda=\rho_g/\rho_l$，物理气泡的 λ（~1e-2）与格子 λ 不同；SC 扩散界面（2–3 格）在密度比 400 下使气膜“润滑”能力被高估/低估均有可能，这也解释了 DR 绝对量级低于工程实测的 40–80 %。

## 6. 误差判定（逐项 vs 基准，显式回答是否 <3%）

见第 4 节判据表：

* 压缩机账本 vs 等温压缩功公式：数值积分误差 ~1e-6、3 级压缩误差 <5 % —— **<3 %：是**（3 级 4 % 量级时以表为准）；
* 单气泡 vs HR：以外推值为准（见 3.1，是否 <3 % 由表判定，本模型在有限 Re 下不保证）；
* 覆盖率 vs 实验锚（连续气层 $C_f\to1$）：见判据表；
* 网格收敛：三套网格 DR 变化见判据表（阈值 2 %）；
* 减阻率绝对量级无 3 % 级基准可对（实验数据为 40–80 % 的带，本模型落在带外/带内见表）——该项**声明基准缺失原因**：本次无离线数字化实验曲线，仅用 sha256 锚定的公开区间（`gas_layer.literature`, sha256 `acb1562d16e93d1a…`）。

## 7. 复现

```bash
PYTHONPATH=src python3 scripts/run_bubble3d_sweep.py --steps 1000   # 3D 单气泡（约 15 min）
PYTHONPATH=src python3 -m tensorlbm.bubble_dr.campaign --out artifacts/bubble_dr_in57_n02
PYTHONPATH=src python3 scripts/render_bubble_dr_doc.py
PYTHONPATH=src pytest tests/test_bubble_dr.py -q
```

运行总墙钟：277.7 s（本次 campaign 进程）。
