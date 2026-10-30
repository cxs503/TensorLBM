# 相场（phase-field）入水：GitHub 公开实现调研与本项目对照

> **英文标题** — Phase-field water entry on GitHub: survey and comparison with TensorLBM
> **检索日期** — 2026-10-30 ｜ **检索面** — GitHub（`gh search repos` / `gh search code`，API 直查）+ 文献
> **用途** — 为「基于 phase-field 的入水（water entry）benchmark」选定模型路线、参考锚与验收门
> **状态** — 调研报告（不是 benchmark 结果）。本文所有"我方数字"均为探针/现有归档实测值，所有"参考数字"均标注来源。

---

## 0. 结论（TL;DR）

1. **GitHub 上不存在可直接复用的"phase-field LBM 入水"实现。** 代码搜索 `"water entry" "phase-field"`（33 命中）、`"water entry" "lattice Boltzmann"`（37 命中）、`water_entry language:Python`（507 命中）中，**唯一**的相场/LBM 入水代码仓库就是本项目 TensorLBM 自己；其余命中全是论文列表、BibTeX、期刊数据库。仓库搜索 `water entry lattice boltzmann`、`phase field water entry` 均为 **0 结果**。
2. **公开的相场两相 LBM 实现都停在"液滴 / 分层 Poiseuille / RTI / 泰勒气泡 / 液滴撞击润湿"这一基准集合**（TCLB、Fakhari Fortran 参考码、pytorch-lbm、JAX-LaB、mcmp-lbm 等 10 个仓库）。**没有一个跨到"刚体穿越自由面 + 冲击载荷 + 运动耦合"**——这正好是入水 benchmark 的核心。
3. **入水的定量锚只能来自文献，而文献里确实有同族蓝本**：De Rosis & Tafuni (2022, *CACAIE* 37:832–847, DOI 10.1111/mice.12651) 是目前唯一"保守相场 + 双分布 LBM + 运动固体 + 与实验/SPH 对比"的公开方法学，给出 **4 个 2D 算例与机器可读参考数字**（自由落体圆柱、上浮圆柱、楔形入水、半潜方块俯仰衰减）。其模型族（Allen–Cahn 保守相场 + h/g 双分布）与本仓库 `tensorlbm.cac_lbm` **同族**，可直接作为 benchmark 蓝本。
4. **本项目差距是"数据通路"而非"模型"**：`cac_lbm`（D3Q19 CAC）已通过 3D Laplace/σ 标定验证（verified），但没有运动固体、没有流固界面动量交换（MEM）力、没有接触角/湿润边界；而仅有的 2D 自由能（FE/Swift）路线是 Boussinesq（ρ≡1，**无密度比**），不能用于入水。
5. **先定锚，再谈误差**：仓库现成锚 `wagner_theory.wagner_wedge_total_force` 与其自身 `wagner_wedge_pressure` 的积分相差 **4/π = 1.2732（+27.3%）**，且与文献 Eq.(45)/实测表不符（详见 §5.3）。Wagner 楔形锚在用于验收前必须先定版。

---

## 1. 检索口径与统计

| 查询方式 | 查询串 | 命中 | 其中可作为"相场入水"的代码 |
|---|---|---|---|
| repo 搜索 | `phase-field LBM` | 2 | 0（1 个 AN 相场模型、1 个仅含 README 的空壳） |
| repo 搜索 | `lattice Boltzmann phase field` | 8 | 0 |
| repo 搜索 | `multiphase LBM` | 8 | 0（全部为伪势/多组分/多孔介质） |
| repo 搜索 | `lattice-boltzmann cahn-hilliard` | 3 | 0（纯 CH 方程，无 NS 耦合入水） |
| repo 搜索 | `water entry lattice boltzmann` / `phase field water entry` / `sphere water entry` | 0 / 0 / 1 | 0（唯一命中是实验数据仓库） |
| repo 搜索 | `water entry in:name` | 10 | 0（实验装置、边界元、势流、数据仓库） |
| code 搜索 | `"water entry" "phase-field"` | 33 | **1：cxs503/TensorLBM `scripts/bench_fe_water_entry.py`** |
| code 搜索 | `"water entry" "lattice Boltzmann"` | 37 | **1：cxs503/TensorLBM（README）** |
| code 搜索 | `water_entry language:Python` | 507 | **1：cxs503/TensorLBM（7 文件）** |

> 口径说明：仓库搜索只看 `name/description/topics`，对"论文库/awesome-list"类仓库天然误命中，所以上面逐条打开了 `git/trees` 或 README 确认。代码搜索的命中量大但绝大多数是同一批论文库的 `.bib/.json/.txt` 文本，唯一含可执行 LBM 相场入水代码的仓库是本项目。

**结论**：本项目要做的 benchmark 在公开 GitHub 上没有"参照实现"，因此必须**自建算例定义 + 用文献数字定锚**（这也是本仓 `benchmarks/README.md` 的既有规范：区分解析校验、内部一致性、外部实验验证）。

---

## 2. 公开相场两相 LBM 仓库清单（10 个）

| # | 仓库 | 模型族 | 维数/格子 | 密度比 | 固体/运动体 | 公开算例 | 定量验证 | 许可 / ★ / 最近提交 |
|---|---|---|---|---|---|---|---|---|
| 1 | **CFD-GO/TCLB**<br>`example/multiphase/d2q9_pf_velocity`、`d3q27_pf_velocity` | 保守相场（Fakhari–Mitchell，h/g 双分布） | 2D D2Q9 / 3D D3Q27 | 参数化（文献至 1000） | 静态壁 + 接触角（`ContactAngle_45*`）；**无运动体** | `cm_dam_break`、`mrt_dam_break`、`Droplet_Falling_2D`、`RayleighTaylorInstability(_3D)`、`annularTaylorBubble_DasC`、`bubbleRiseBenchmark` | 层流 Poiseuille / RTI / 泰勒气泡（论文级）；**无入水** | GPL-3.0 / 206★ / 2026-03 |
| 2 | **JunjunQing/pytorch-lbm** | Allen–Cahn 相场 + 几何放大湿润 | 3D D3Q19 | **828 : 1** | **曲面固体 + 体积惩罚 + 湿润边界**（ridge/凸半球/凹腔） | 液滴撞击曲面（spread/recoil/接触角） | 静态接触角、网格收敛、伪电流脚本（无第三方复核、无许可） | 无许可 / 0★ / 2026-08 |
| 3 | **AbbasHub/Improved_Phase-Field_LBM_2D**（Fakhari 等 2017 PRE 参考实现） | 保守相场 | 2D 周期 | 高密度比 | 无 | 分层 Poiseuille / RTI / 泰勒气泡 | 论文 Table 级 | GPL-3.0 / 56★ / 2018 |
| 4 | **pavinpp/MSc-MCM-JAX**（JAX-LaB） | **伪势**（Cascaded/MRT、Peng–Robinson EOS），**非相场** | 2D/3D D2Q9/D3Q19 | ~350 : 1 | 壁面润湿、可微 | `droplet_impact_on_wall`、**`droplet_impact_on_liquid`**、`capillary_rise`、`capillary_fingering` | 与实验/解析对比若干 | NOASSERTION / 0★ / 2026-02 |
| 5 | **2317942351/LBMD3Q27** | TCLB `d3q27_pf_velocity` 审计包 | 3D | — | 球面湿润边界诊断 | 球面静态液滴接触角（PRE 2025 Table II 风格） | **自述 `exploratory_not_validation`** | NOASSERTION / 0★ / 2026-07 |
| 6 | **nataneb/LBM_Cahn_Hilliard** | CH 方程 LBM（含 NS 耦合） | 2D | 低 | 无 | CH 经典/时间分数阶基准 | 方程级 | GPL-3.0 / 10★ / 2021 |
| 7 | **AminZar69/ompPhaseField**、**CudaPhaseField** | Fakhari 族相场 | 2D（OpenMP）/ 3D（CUDA Fortran） | 参数化 | 无（自述"润湿条件待加"） | 单液滴平衡 | 弱 | GPL-3.0 / 1★ / 2026-05 |
| 8 | **Extremumm/CGLBM** | 改进色梯度（Lafarge 2021），**非相场** | 2D D2Q9 | 参数化 | 静态 | Laplace、RTI | 有（run 脚本 + pytest） | GPL-3.0 / 1★ / 2026-10 |
| 9 | **geoelements/mcmp-lbm** | 伪势 MRT 3D | 3D D3Q19 | 参数化 | 静态多孔 | 液滴、接触角、砂柱注入 | 弱 | NOASSERTION / 30★ / 2023 |
| 10 | **harrislab-brown/WavyWaterEntry**、**Deltares-research/WaterImpactEntry**、**pwellens/entry_water** | **非 LBM**：实验数据 / 2D 轴对称边界元 / 势流 | — | — | 真实刚体 | 入水实验与势流 | 实验级 | 各异 / 0★ |

### 2.1 逐条要点（只写与"入水 benchmark"相关的部分）

**#1 TCLB** — 公开代码里**最成熟的保守相场 LBM**，也是文献 D3Q27 路线的母体（本仓 `2317942351/LBMD3Q27` 审计的对象）。它的算例库完整覆盖了"界面 + 壁面"的所有非入水场景，并**提供了接触角案例**——这两点正是我们缺的。它没有入水案例，也没有运动固体（壁面是静态 mask）。对我们：**可作同族交叉校验器**（把我们的 dam break / droplet falling 与 TCLB 的 xml 配置对齐后比同一物理量），不可作入水锚。

**#2 pytorch-lbm** — 唯一公开的"高密度比（828:1）+ 曲面固体 + 湿润边界"相场实现，工程结构（`lbm/fe_ac_solver.py`、`fe_wetting.py`、`fe_chemical_potential.py`）与本仓 `phasefield/` 很接近。它的**曲壁湿润（几何放大法）**是我们要补的接触角能力的最佳参考；但它研究的是**液滴撞干壁/曲面**（无液池、无入水冲击载荷），而且未给许可、无第三方复核。

**#3 Fakhari 2D Fortran** — 保守相场原始参考实现（PRE 96, 053301），周期域、无固体。价值在于**方程级 oracle**：我们 CAC 模块的实现说明里已写明"以其为方程级 oracle，未做行级抄写"，这条纪律应保持。

**#4 JAX-LaB** — 公开仓库里唯一有 `droplet_impact_on_liquid.py`（液滴撞液池）的代码，是"入水类"最接近的公开算例；但它是**伪势 + 真实 EOS** 路线，不是相场，不能拿它的数字当相场锚，只能作为"物理趋势"的旁证（spread/crown/cavity 形态）。

**#5 LBMD3Q27** — 与 #1 同源，专门诊断 TCLB D3Q27 相场的**球面湿润边界**，并明确自述 `exploratory_not_validation`。对我们是一个重要警示：**相场 + 曲壁 + 湿润在公开工程里也仍是未闭合项**，不要指望一个现成实现。

**#6–#9** — CH 方程、2D/3D Fakhari 移植、色梯度、伪势 MRT：提供"界面演化正确性"的局部校验点（Laplace、RTI、分层 Poiseuille），但都不处理刚体入水。

**#10** — 非 LBM，但**是入水实验数据的现成来源**（尤其 `WavyWaterEntry` 与 `WaterImpactEntry`），可作为后期"外部实验验证"阶段的候选（需逐条核对几何/无量纲数，不能直接套用）。

---

## 3. 文献标杆：目前唯一的"相场 LBM 入水/出水"数字

**A. De Rosis, A. Tafuni (2022).** *A phase-field lattice Boltzmann method for the solution of water-entry and water-exit problems.* Computer-Aided Civil and Infrastructure Engineering **37**(7):832–847. DOI [10.1111/mice.12651](https://doi.org/10.1111/mice.12651)（accepted manuscript 开放版：`pure.manchester.ac.uk/ws/files/177869532/CACAIE_DeRosis_Tafuni.pdf`）

### 3.1 方法与公共参数（原文 §3）

- 两相：水 ρ_H = 1000 kg/m³, ν_H = 1e-6 m²/s；空气 ρ_L = 1.225 kg/m³, ν_L = 1.5e-5 m²/s ⇒ **密度比 816、黏度比 15**。
- 相场：**Allen–Cahn 保守型**，界面厚度 **ξ = 4** 格，迁移率 **M = 0.1**，表面张力 **σ = 1e-5**（格子单位）。
- 碰撞：**中心矩 MRT（CM-MRT）**；D2Q9 双分布（f 动量/pressure，g 相场）；运动固体用 **IBM（Peskin 直接力）** + 二阶 Adams–Bashforth 刚体积分。
- 重力 a = 9.81 m/s²；"除非另述，浸没刚体主尺寸用 100 个格点离散"。
- **单位换算（原文 Appendix B，可直接照抄的做法）**：D = 1 m 用 100 格 ⇒ s_l = 0.01 m；物理 1 m/s 取格子 0.005 ⇒ s_v = 200 m/s；s_ρ = 1000 kg/m³。由此 s_t = s_l/s_v = 5e-5 s、s_ν = s_l²/s_t = 2 m²/s ⇒ **ν_lattice ≈ 5e-7、τ ≈ 0.5**。也就是说：文献为了把 Re 匹配到 1e6，是把 τ 顶在 0.5 上跑的——这正是它必须用 CM-MRT（而不能用 BGK）的根本原因，也预告了我们的 BGK 版本必须降低格子 Re。
- **关键稳定性结论（原文 Fig.5）**：同一算例换成 **BGK 在 t ≈ 0.2 s 崩掉**，只有 CM-MRT 能跑完 1.5 s。这是选用碰撞算子的硬证据。

### 3.2 四个基准算例与参考数字

| 算例 | 几何/驱动 | 网格 | 参考数据源 | 参考峰值（机器读自原文表） |
|---|---|---|---|---|
| **B-1 自由落体圆柱入水** | D = 0.3 m，初速 v_i = 2.55 m/s 撞静止水面，自由下落（6DOF） | 8000 × 3200 格，直径 400 格 | 实验 + level-set（Colicchio et al. 2009）+ SPH | max(q̇_y): Exp 0.8912 / Num 0.7868 / SPH 0.8798；−min(q_y): Exp 0.4423 / Num 0.4185 / SPH 0.4455 |
| **B-2 上浮圆柱** | D = 2 m，ρ_s = 0.6ρ_w，初始中心深度 Y_C = 2.5D，域宽 3D、水深 3.5D | 300 × 500 | Fekken (2004) + Campbell & Vignjevic (2012) + SPH | max(q̇_y): Num 2.3165 / SPH 2.6184；max(q_y): Num 5.2368 / SPH 5.5396 |
| **B-3 楔形恒速入水** | 半楔长 D_w = 1 m（500 格），恒速 v = 1 m/s，死升角 ζ = 2°/3°/4° | 2000 × 1000 | Wagner (1932) 解析 Eq.(45) + SPH | P–t 斜率相对误差 ε_W,LB = **6.33% / 29.47% / 44.23%**；ε_W,SPH = 5.16% / 32.29% / 46.33% |
| **B-4 半潜方块俯仰衰减** | L = 0.1 m 立方，半浸没，给初角速度 | 500 × 500 | Fekken (2004) VOF | 稳定角 45°（解析），振荡包络对比 |

### 3.3 该文献对我们最重要的三条"预期管理"

1. **BGK 会崩**：入水是高 Re + 高密度比 + 强瞬态，BGK/单松弛在公开结果里撑不过 0.2 s。我们 CAC 模块是 **BGK 家族**（h-LBE + g-LBE 单松弛），必须优先评估稳定性，必要时上 CM/MRT 或两相 regularized。
2. **即使成熟实现，Wagner 对比也只在小死升角接近**（2° 时 6.3%，3°/4° 时 29%/44%，SPH 同样差）。所以**"楔形 vs Wagner"的门限必须按死升角分档**，或在 2° 档设 ≤10%，不能拿 3% 一刀切。
3. **界面厚度（ξ = 4）会带来"力-时移"**：原文对自由落体圆柱明确把回弹阶段的时序偏差归因于弥散界面厚度，并说明**加密一档也没消除**。这决定了我们的验收应放在**峰值/积分量**而非逐点波形。

---

## 4. 本项目现状（能力矩阵）

命令与文件均为仓库内可复核路径。

| 能力 | TensorLBM 现状 | 证据（模块 / 归档） | 入水 benchmark 够用吗 | 缺口 |
|---|---|---|---|---|
| 保守相场（CAC）双分布 LBM | ✅ D3Q19，h-LBE（CAC）+ g-LBE（HCH），各向同性模板，壁面 full-way bounce | `src/tensorlbm/cac_lbm.py`（`CACSim`，`init_phi_layers`，`diagnostics`） | **足够作物理内核** | — |
| 相场 σ 标定 | ✅ Route-B：`σ_model = σ_target / 0.923896`（W = 4），R 与密度比迁移成立 | `benchmarks/verified/laplace_cac_phasefield/`（R=20/32/48，ρ 比 10/100，误差 ≤0.445%，伪流 ≤6.7e-5） | 够用（须沿用同一标定） | — |
| 重力/静水压 | ⚠️ 有 `gravity_mode="rho"`；实测 dp/dy 偏差 2.1%（g=1e-4）/ 22%（g=1e-5） | 本报告 §5.1 探针 `probe/p0_hydrostatic.py` | **不足** | 需低 g 下的静水平衡修正或抬高 g 下界 |
| 相质量守恒（带壁 + 重力） | ⚠️ 周期域 8.6e-13（verified 归档）；**带壁 + g=1e-4 时 6000 步漂 +1.5e-3，界面单向爬升 +0.61 格** | 本报告 §5.1 | **不足** | 需墙-相场通量审计/修正 |
| 静态固体（任意 mask） | ✅ `wall_mask` 任意布尔体 | `cac_lbm.CACSim` | 部分够用 | 无流固 MEM 力 |
| 运动固体（给定速度） | ⚠️ 仅 CG/SC 2D/3D 两相路线有 moving bounce-back | `src/tensorlbm/multiphase_water_entry.py`（Ladd MEM，fresh-node） | **CAC 路线没有** | 需为 CAC 实现运动壁 + MEM |
| 6DOF 刚体自由落体 | ❌ 相场路线没有；单相有 | `src/tensorlbm/sphere_water_entry.py`（迎风系/无相场）、6DOF 浮体模块 | **不足** | 需刚体积分器 + 加质量稳定化 |
| 接触角 / 湿润边界 | ❌ 无 | `2317942351/LBMD3Q27` 的自述失败可作为反例 | **差距最大** | 需 TCLB/pytorch-lbm 路线的一个最小实现 |
| 无相场的高密度比两相（对照） | ⚠️ CG（`color_gradient2d` 双 τ 已验证分层 Poiseuille）；SC 3D 结构性失败 | `verified/poiseuille_two_phase_cg`；`pending/two_phase_poiseuille`（26.8%/33.2% FAIL） | 仅作对照 | — |
| 2D 自由能（Swift）相场 | ❌ 不可用于入水：ρ≡1，仅 Boussinesq 浮力 | `multiphase.free_energy_step`；`pending/hadamard_rybczynski_droplet/prereg.md`（模块清单："Free-Energy 2D λ≡1，DEAD"） | **禁用** | — |
| Allen–Cahn 3D 早期模块 | ❌ 少数相液滴 25→1000 步完全溶解（不守恒） | `pending/hadamard_rybczynski_droplet/README.md` E4 | **禁用** | — |
| 解析锚：Wagner 楔/球 | ⚠️ 球体公式自洽；**楔形总力与其自身压力积分差 4/π** | `src/tensorlbm/wagner_theory.py`，见 §5.3 | 定版后可用 | 需修正锚或改用文献 Eq.(45) |
| 解析锚：入水砰击 | ✅ 公式级回归测试 | `tests/test_water_entry_slamming.py`，`docs/reports/WATER_ENTRY_SLAMMING_SUMMARY.md` | 够用 | — |
| 基准治理 | ✅ `pending/` ↔ `verified/`，验收 = 每检查点 ≤3% 且两套网格跨度 ≤3% | `benchmarks/STATUS.md` | 够用 | 入水门限需按 §3.3 分档 |

---

## 5. 本次实测的探针数字（可复跑）

探针目录：`benchmarks/pending/phasefield_water_entry/probe/`（脚本入库，输出落在 `/tmp`）。

### 5.1 P0 静水压/静水平衡（`p0_hydrostatic.py`）

设置：闭合壁箱 `(nz,ny,nx) = (8,96,8)`，`init_phi_layers` 半高界面，`gravity_mode="rho"`，ν = 0.1、σ = 0.01、W = 4、M = 0.05（与 verified Laplace 同参数），6000 步。

| 臂 | 界面位移（步→格，末值） | dp/dy ÷ (ρ_l g) | max\|u\| | φ 积分相对漂移 | NaN |
|---|---|---|---|---|---|
| ρ 比 100，g = 1e-5 | +0.35 → **+0.79** | −1.12 → **−1.22** | 2.25e-4 | +9.1e-4 → **+1.88e-3** | 无 |
| ρ 比 100，g = 1e-4 | +0.10 → **+0.61** | −1.022 → **−1.021** | 2.79e-4 | +4.9e-4 → **+1.52e-3** | 无 |
| ρ 比 100，g = 1e-3 | −2.14 → −0.69 | −1.040 → −1.016 | 1.12e-3 | −3.3e-3 → −7.7e-4 | 无 |
| ρ 比 816，g = 1e-4 | +0.10 → **+0.61** | −1.022 → **−1.021** | 2.79e-4 | +4.9e-4 → **+1.53e-3** | 无 |

读法（这是入水 benchmark 的**前置门**）：

- **静水压梯度偏差 2.1%**（g = 1e-4）在"≤3%"的整体门内，但已经吃掉 2/3 预算；g = 1e-5 时 22% 不可用 → **入水算例的 g 不能取得太小**，这对"低 Fr 入水"是硬约束。
- **界面单向爬升 + φ 积分单向增长**（同一符号、近线性）说明存在**系统性相质量源**（墙—相场通量），不是随机噪声。6000 步 +1.5e-3 ⇒ 若跑 3e4 步量级约 +0.8%，对"液面高度"类观测量是**一阶误差**。**必须在 benchmark 前置门里显式设门**（建议 |Δφ/φ| ≤ 1e-4/1e3 步，或先修墙通量）。
- 密度比 816（= 水/空气）与 100 的探针结果**逐位接近**，说明密度比本身不是当前误差主因。

### 5.2 P0b 静态固体跨越界面（`p0b_static_body.py`）

（`(8,96,64)` 壁箱 + 中心落在液面上的静态圆盘，g = 1e-4，4000 步；机器数字见 `NOTES.md` §P0b。）

### 5.3 Wagner 楔形锚的内部不一致（**必须先修**）

用仓库自身函数算 ζ = 2°/3°/4°、ρ = 1000、V = 1 m/s、t = 0.002 s：

| 量 | 值（ζ = 2°） | 值（ζ = 3°） | 值（ζ = 4°） |
|---|---|---|---|
| `wagner_wedge_total_force` | 8.093 kN/m | 3.593 kN/m | 2.018 kN/m |
| `wagner_wedge_pressure` 主导项积分（−c…c） | **6.357 kN/m** | **2.822 kN/m** | **1.585 kN/m** |
| 含 convective 项的积分 | 291.9 kN/m（边缘奇点发散） | 87.3 kN/m | 37.1 kN/m |
| 文献 Eq.(45)（De Rosis & Tafuni 2022） | **6.357 kN/m** | 2.822 kN/m | 1.585 kN/m |
| 原文 Table（t = 0.002 s）：Pres / **W** / SPH | 6.8 / **6.3** / 6.1 | 4.1 / **2.8** / 4.3 | 2.7 / **1.6** / 3.0 |

> 表注：原文 ζ = 2° 的表头为 `Pres. / W / SPH`；ζ = 3°、4° 的两张表在文本抽取后是"三列未标注序列"，按 Eq.(45) 逐值反推，第三列为 Wagner 值（2.8、1.6 kN/m，与本节复算的 2.822、1.585 一致），第一列为 Present。原文自报斜率误差 ε_W,LB = 6.33% / 29.47% / 44.23%；用表内（已四舍五入到 0.1 kN/m 的）数值反算得 7.7% / 30.7% / 43.8% ⇒ **表格精度不足以做锚**，锚必须取解析式（Eq.(45) 或本仓压力积分）而不是论文表格读数。

结论：

1. `wagner_wedge_total_force` = **1.2732 ×** 其自身压力积分 = 4/π × 文献 Eq.(45)。**三者中两个自洽（压力积分 ⟷ 文献 Eq.(45) ⟷ 原文 W 列），一个孤立（仓库的总力函数）。**
   > 补充（诚实披露）：若按"平板在自由面上"的附加质量 `m_a = ρπc²/2`、`F = d(m_a V)/dt` 自行推导，得到 ρπ²cV²/(2tanβ)，即 **π²/2** —— 与上两者都不同。这说明"楔形砰击力"的系数在文献里至少有三套约定（π / π²/4 / π²/2 的差别来自附加质量取全平面还是半平面、以及是否把 π/2 喷溅修正同时计入 c 与力）。**本报告的立场**：锚一律取"与文献数字 + 仓库自身压力式一致"的 **π²/4 形式**，并在 S2 开跑前用 Zhao & Faltinsen (1993) 的原始压力式做一次独立核对定版；**不擅自改库**。
2. `wagner_wedge_slamming_coefficient` 返回 π/tan β（ζ = 2° 时 89.96），与其 docstring 里的 C_s = π²/(2 tan²β) 也不是同一个量 → 该函数**不能被当作 C_s 使用**。
3. 含 convective 项的解析积分在边缘发散（这是 Wagner 的已知奇点，原文也明确"用 P–t 斜率而不是 P 本身"来比），所以**验收锚必须用主导项/加质量形式**，或对压力场做 0.95c 截断后再积分。
4. 动作项：修 `wagner_theory` 的楔形总力与 C_s（或标 `deprecated` 并新增 `wagner_wedge_added_mass_force` 与文献 Eq.(45) 对齐），并把上表做成 `tests/` 的锚回归。

---

## 6. 建议的 phase-field 入水 benchmark（阶梯 + 门限）

详细锁定见 [`benchmarks/pending/phasefield_water_entry/prereg.md`](../../benchmarks/pending/phasefield_water_entry/prereg.md)。要点：

| 阶段 | 内容 | 参考锚 | 建议门 |
|---|---|---|---|
| **S0** | 静水压 + 相质量守恒（带壁 + 重力） | 解析 dp/dy = −ρ_l g；φ 守恒 | dp/dy 偏差 ≤ 3%；\|Δφ/φ\| ≤ 1e-4/1e3 步；NaN-free |
| **S1** | 静态圆盘跨越界面（无运动） | 界面不平移、伪流有界 | 远/近场界面差 ≤ 0.5 格；max\|u\| ≤ 1e-3 |
| **S2** | 楔形恒速入水 ζ = 2°/3°/4° | Wagner Eq.(45) P–t 斜率；文献 ε_W,LB 分档 | ζ=2°: ≤10%；ζ=3°: ≤35%；ζ=4°: ≤50%（分档，见 §3.3） |
| **S3** | 圆柱恒速入水（**prescribed motion**；6DOF 自由落体留 S5，未实现） | Colicchio et al. (2009) 实验峰值：max(q̇_y) 0.8912 m/s、−min(q_y) 0.4423 m | max(q̇_y) ≤ 12%（原文 LB 自身 +11.7%，故门=「达到参考实现水平」）；−min(q_y) ≤ 5%；时序平移只记录不设门 |
| **S4** | 上浮圆柱 / 3D 球体（可选，预算门后） | Fekken (2004)、Wagner 球体式 C_F = π√(h/2R) | 峰值 ≤ 5%；两网格跨度 ≤ 5% |

计算量级（本机 2 线程 CPU 实测 0.711 µs/格/步，仅量级外推）：S2（4 × 300 × 600 = 7.2e5 格，2.2e3 步）≈ **20 min/臂**；S3（4 × 300 × 500 = 6.0e5 格，1.4e4 步）≈ **1.8 h/臂**；S4（96³ = 8.8e5 格，4e3 步）≈ 50 min/臂。S0/S1 在 CPU 分钟级，可进 CI。真实 Re（7.65e5）在格子 Re（1e3）下不可能匹配，S3 的"定量"仅指**力峰值**，空腔/夹带只做形态比对。

---

## 7. 风险与未决项（按优先级）

1. **碰撞算子稳定性（P0 风险）**：CAC 现为 BGK 家族；文献证据显示入水场景 BGK 在 t ≈ 0.2 s 崩。→ 先做"BGK 极限探针"（把 v_entry 抬到目标 Re，记录发散步），再决定是否上 CM/MRT。
2. **墙—相场质量源（P0 风险）**：§5.1 的 +1.5e-3/6000 步单向漂移。→ 归因（bounce-back 对 g 的通量 vs 壁面化学势梯度），修或设门。
3. **运动固体缺失（P1）**：需为 CAC 实现 moving bounce-back + MEM（可移植 `multiphase_water_entry.py` 的 Ladd/fresh-node 实现）+ 6DOF 积分。
4. **湿润/接触角缺失（P1）**：入水的喷溅/空腔/闭合对接触角敏感（文献 §3.4 与 Kharmiani 2021 三相结合部都强调这一点）。最小可用方案：几何放大法（pytorch-lbm）或 TCLB `ContactAngle_45` 的壁面 φ 外推。
5. **锚定版（P1）**：Wagner 楔形（§5.3）+ 球体与 Colicchio 数字的独立复算，全部落成 `tests/` 回归。
6. **单位换算（P2）**：必须固定一套"格子↔物理"映射并在 prereg 里锁死（原文 Appendix B 给了示例），否则 Re/Fr/We 三数不可比。

---

## 8. 参考

- De Rosis, A., & Tafuni, A. (2022). A phase-field lattice Boltzmann method for the solution of water-entry and water-exit problems. *Computer-Aided Civil and Infrastructure Engineering*, 37(7), 832–847. DOI [10.1111/mice.12651](https://doi.org/10.1111/mice.12651)
- Fakhari, A., Mitchell, T., Leonardi, C., & Bolster, D. (2017). Improved locality of the phase-field lattice-Boltzmann model for immiscible fluids at high density ratios. *Phys. Rev. E*, 96, 053301.
- Mitchell, T., Leonardi, C., & Fakhari, A. (2018). Development of a three-dimensional phase-field lattice Boltzmann method for the study of immiscible fluids at high density ratios. *Int. J. Multiphase Flow*, 107, 1–15.
- Liang, H., Xu, J., Chen, J., Wang, H., Chai, Z., & Shi, B. (2018). Phase-field-based lattice Boltzmann modeling of large-density-ratio two-phase flows. *Phys. Rev. E*, 97, 033309.
- Wagner, H. (1932). Über Stoß- und Gleitvorgänge an der Oberfläche von Flüssigkeiten. *ZAMM*, 12(4), 193–215.
- Colicchio, G., Greco, M., Miozzi, M., & Lugni, C. (2009). Experimental and numerical investigation of the water-entry and water-exit of a circular cylinder. *24th Int. Workshop on Water Waves and Floating Bodies*.
- Fekken, G. (2004). *Numerical simulation of free-surface flow with moving rigid bodies*. PhD thesis, Rijksuniversiteit Groningen.
- Kharmiani, S. F., et al. (2021). Numerical study of free-fall cylinder water entry using an efficient three-phase lattice Boltzmann method. *Ocean Engineering*, 235, 109328.
- Truscott, T. T., Epps, B. P., & Belden, J. (2014). Water entry of projectiles. *Annu. Rev. Fluid Mech.*, 46, 355–378.
- 仓库内锚与归档：`src/tensorlbm/wagner_theory.py`、`docs/reports/WATER_ENTRY_SLAMMING_SUMMARY.md`、`benchmarks/verified/laplace_cac_phasefield/`、`benchmarks/pending/dam_break_2d_multimodel/REPORT.md`、`benchmarks/pending/hadamard_rybczynski_droplet/`。
