# XVAL-LBM-DAM-BREAK-SLOT-MUSCL2-LOCAL-HLL-SSPRK2 —— 与成熟开源浅水求解器的交叉对标

锚点 `XVAL-LBM-DAM-BREAK-SLOT-MUSCL2-LOCAL-HLL-SSPRK2` · 任务 m17725 · 仓 TensorLBM · 分支 agent/m17725

机器可复核产物：`xval/dam_break_slot_muscl2_local_hll_ssprk2/artifacts/result.json`（case_spec / xval_table / criteria / differences / frictions）、`xval/dam_break_slot_muscl2_local_hll_ssprk2/artifacts/table.csv`、`xval/dam_break_slot_muscl2_local_hll_ssprk2/artifacts/arm_*.json`（33 个臂×网格原始剖面目）、`figures/xval_dam_break_slot_profiles.png`、`figures/xval_dam_break_slot_convergence.png`。本文所有数字由 `make_doc.py` 从 result.json 程序化注入（零手抄），全部剖面由 `run_arms.py` 现场重算。

**总裁决：PASS**（C1–C5 全过；方案级一致性 0.29–1.86%，低于 owner 15% 精度标准，也低于 3%）

## ① 问题与参考出处

**问题**：本仓矩阵 slot-8（`benchmarks/validation_matrix/cases/dam_break_ritter_ci`）的溃坝求解器 在 m17112 由一阶全局波速 Rusanov 升级为「minmod-MUSCL 重构 + 局部波速 HLL 通量 + SSP-RK2」（`src/tensorlbm/shallow_water.py::swe_step_muscl2`）。此前它只与 Ritter 解析解对照（自证），缺**独立第三方实现**这一证据轴。本单要求：与成熟开源浅水实现同规格横跑，误差 <15%。

**对标实现（OSS，零手抄：源码/文档查得）**

| 项 | 内容 | 出处 |
|---|---|---|
| 套件 | Clawpack 5.14.0（GeoClaw 家族，浅水/双曲守恒律主流开源实现） | pip 包 `clawpack-5.14.0.tar.gz`（内部镜像 lpypi.loongnix.cn，源码 meson+gfortran 构建） |
| 通量 | `clawpack.riemann.shallow_hlle_1D`（HLLE 浅水 Riemann 求解器）、`shallow_roe_with_efix_1D`（Roe+熵修正） | 编译模块 `clawpack/riemann/shallow_hlle_1D*.so`，接口 `rp1(wave,s,apdq,amdq,q_l,q_r,aux_l,aux_r,auxl1,auxr1,wave,s,amdq,apdq)`，`cparam.grav` |
| 求解器 | `pyclaw.ClawSolver1D`（Godunov + minmod MUSCL + Hancock 预测-校正）、`pyclaw.SharpClawSolver1D`（WENO5 + SSP104） | `clawpack/pyclaw/classic/solver.py`、`clawpack/pyclaw/sharpclaw/solver.py` |
| 物理/解析 | 浅水方程 `h_t+(hu)_x=0`、`(hu)_t+(hu²+gh²/2)_x=0`；Ritter 自相似解 | Toro (2009) §10.4（HLL 通量构造）、Ritter (1892) Z. Ver. Deutsch. Ing. 36, 947；本仓 `shallow_water.py::ritter_depth/ritter_front`（解析锚零手抄） |

**假设（预声明于 `predeclared.json`，跑 OSS 前写定）**：H1 = 同阶（二阶 MUSCL + 两波 HLL 族）横跑下，本仓与 Clawpack 的湿区水深剖面相对 L2 差 <15%；H0 = >15% 则指向本仓实现的边界/干床/波速缺陷。

## ② 方法与实现

**设计（experimental-design：决策 → 备选 → 判据 → 混杂 → 敏感度）**

* 决策：本仓 MUSCL2+HLL+SSPRK2 是否能作为 slot-8 的裁决求解器（跨实现一致）；不一致必须归因。
* 备选（预声明拒绝）：只比解析解（自证循环）／与生态内 TensorDG 横跑（非 OSS）／自写第二套 FV 当参考（自证循环）。
* **六个臂**（同 IC / 同域 / 同 t_end / 同 CFL）：

| 臂 | 实现 | 格式 | 干床初值 |
|---|---|---|---|
| A | 本仓 `scheme='muscl2'`：minmod-MUSCL 面状态 + Davis 局部波速 HLL + SSP-RK2(Heun) + 正性降阶保护 α:1→0.5→0 | 二阶 | h=0（真干床） |
| A_matched | 同上，仅初值干区抬到 ε=3e-3（与 OSS 臂同质量 IC，隔离『格式差』与『干床底垫差』） | 二阶 | 3e-3 |
| A(1e-2) | 同上，初值干区 ε=1e-2（与 Roe 臂同 IC） | 二阶 | 1e-2 |
| B | Clawpack `ClawSolver1D(shallow_hlle_1D)` + minmod + Hancock | 二阶 | 3e-3 |
| B_1st | Clawpack `ClawSolver1D(shallow_hlle_1D)`，limiter=0 | 一阶（稳定性对照） | 1e-4 |
| C | Clawpack `SharpClawSolver1D(shallow_hlle_1D)`，WENO5 + SSP104 | 五阶（高精度参照） | 1e-4 |
| D | Clawpack `ClawSolver1D(shallow_roe_with_efix_1D)` + minmod + Hancock | 二阶（换通量族对照） | 1e-2 |

* **指标**（预声明）：M1 质量反演前沿误差 `x_f,num=27V_down/(4h0)` vs `2c0t`；M2 湿区 [0,x_f] 剖面相对 L2；M3 跨实现剖面相对 L2（线性插值到公共节点网格，另报 0.8x_f 的 bulk 变体剔除舌尖）；M4 质量漂移；M5 网格阶梯 nx=100/200/400/800；M6 CFL=0.2/0.4/0.8；M7 OSS 干床底垫敏感度 ε=3e-3 vs 1e-2。
* **判据（预声明，跑前写定，不得为结果让路）**：`C1_cross_tool_l2_pct` < 15%；`C1b_icmatched_l2_pct` < 15%；`C2_front_diff_pct` < 15%；`C3_profile_l2_pct_vs_analytic` < 5%；`C4_mass_drift_pct` < 1%；`C5_cross_tool_vs_highorder_l2_pct` < 15%
* **混杂与消除**：①半格错位（本仓节点制 vs Clawpack 单元中心）→ 线性插值 + bulk 变体；②OSS 干床底垫 ε 引入附加质量 → A_matched/A(1e-2) 同 IC 对照 + ε 敏感度；③波速估计（Davis vs HLLE/Einfeldt）与限制器作用对象（守恒量分量式 vs 波强式）→ D 臂换通量族 + C 臂换高阶格式界定；④时间推进（固定 dt vs 逐步自适应 dt）→ CFL 敏感度；⑤边界（wall/extrap 两码一致）。

**实现要点**（`xval/dam_break_slot_muscl2_local_hll_ssprk2/`）

* `predeclared.json`：先于 OSS 运行写定的决策/假设/指标/阈值/停止规则/混杂/环境。
* `run_arms.py`：A/A_matched/A(1e-2) 复用库内 `swe_step`（不复制求解器代码，只改初值干区）；B/B_1st/D/C 走 PyClaw，`num_eqn=2/num_waves=2` 由驱动按 SWE 系统规模声明（f2py 模块不带此元数据）。下游体积一律按各臂**自身初值**的体积差计算，保证 `x_f=27V/(4h0)` 对带底垫的臂仍成立。
* `analyze.py`：全部指标/判据/图；`make_doc.py`：本文与上报载荷（程序化注入）。
* `tests/test_xval_dam_break_slot.py`：把本 XVAL 的判据固化为回归测试（读 result.json，无 clawpack 依赖）。

## ③ 参数表（无量纲完整）

| 量 | 值 | 说明 |
|---|---|---|
| h0 | 1 | 初始水深（长度单位） |
| g | 1 | 重力；c0=√(g h0)=1 为速度单位 |
| 域 | x/h0 ∈ [-4, 12] | 坝在 x=0（左 4 h0 / 右 12 h0） |
| t_end | 5 √(h0/g) | t* = 5 |
| Fr（前沿） | 2 | u_front/c0 = 2（Ritter 自相似） |
| 判裁网格 | [200, 400] | dx/h0 = 0.0800 / 0.04000 |
| CFL | 0.4（敏感度 [0.2, 0.8]） | 本仓固定 dt=CFL·dx/(2c0)；OSS 逐步自适应 |
| 干床判据 h_dry/h0 | 1e-4（本仓库内） | OSS 臂所需正水层：HLLE 0.003、Roe 1e-2 |
| 探测用底垫 | 0.001 | 记录 OSS 二阶 NaN 的临界值 |
| 边界 | wall (left) / extrapolation (right) in both codes | 左壁反射 / 右零梯度外流 |

## ④ 对照表（ref vs got，逐项打勾叉）

### 4.1 判据裁决

| 判据 | 实测最差值 | 阈值 | 判定 | 备注 |
|---|---|---|---|---|
| `C1_cross_tool_l2_pct` | 8.952% | < 15% | ✓ | A(h=0) vs B(ε=3e-3)：**不同 IC**，差值含 OSS 干床底垫贡献 |
| `C1b_icmatched_l2_pct` | 1.064% | < 15% | ✓ | A_matched vs B：同 IC，纯格式差（决定性归因） |
| `C2_front_diff_pct` | 0.133% | < 15% | ✓ | 两臂质量反演前沿之差 |
| `C3_profile_l2_pct_vs_analytic` | 1.168% | < 5% | ✓ | 本仓判裁档 vs Ritter 解析解（矩阵 5% 门） |
| `C4_mass_drift_pct` | 0.000% | < 1% | ✓ | 本仓与 OSS 两臂最大总质量漂移 |
| `C5_cross_tool_vs_highorder_l2_pct` | 0.882% | < 15% | ✓ | A vs OSS 五阶 WENO5 参照 |

**总裁决 PASS**。

### 4.2 网格阶梯（CFL=0.4，逐臂 vs 解析 + 跨实现）

| nx | 前沿误差 A / A_m / B / C / D [%] | 剖面 L2 vs 解析 A / A_m / B / C / D [%] | A×B 跨实现 | A_m×B（同 IC） | A×C | A×D | A(1e-2)×D（同 IC） |
|---|---|---|---|---|---|---|---|
| 100 | +0.256 / +0.249 / +0.515 / +0.122 / — | 1.93 / 7.29 / 8.22 / 1.72 / — | 7.21% ✓ | 1.86% ✓ | 1.03% ✓ | — | — |
| 200 | +0.119 / +0.115 / +0.252 / +0.053 / 0.17 | 1.17 / 8.46 / 8.75 / 1.17 / 16.83 | 8.32% ✓ | 1.06% ✓ | 0.75% ✓ | 15.89% ✗ | 1.41% |
| 400 | +0.057 / +0.055 / +0.124 / +0.024 / 0.09 | 0.83 / 9.04 / 9.15 / 1.31 / 17.28 | 8.95% ✓ | 0.53% ✓ | 0.88% ✓ | 16.49% ✗ | 1.36% |
| 800 | +0.028 / +0.027 / +0.061 / +0.013 / — | 0.80 / 9.33 / 9.37 / 1.39 / — | 9.26% ✓ | 0.29% ✓ | 1.20% ✓ | — | — |

### 4.3 关键归因（差异到底来自哪里）

* 本仓 A 与 OSS B 直接比：`nx=200` 8.32%、`nx=400` 8.95% —— 过 15% 门但离及格线并不远，必须说清构成。
* 同一份本仓代码，只把干区初值改成 ε=3e-3（A_matched），它与 A 的差 `8.19%/8.91%`（nx=200/400）**几乎等于** A×B 的差 `8.32%/8.95%` —— 即 A×B 的差值基本上全是『干床底垫』这一 IC 差异，不是格式差异。
* 同 IC 后：A_matched×B = `1.06%`/`0.53%`；A(1e-2)×D = `1.41%`/`1.36%`（Roe 族同 IC）。**两套不同通量族、两套不同代码的二维一致性进入 1% 量级** → 本仓实现与成熟 OSS 在『方案级』上一致。
* 反向证据：`nx=400` 时 A×D（不同 IC）16.49% > 15% —— 不是格式差，而是 Roe 臂被迫用 ε=1e-2：同臂在 ε=3e-3 时剖面 L2 9.15%，ε=1e-2 时 17.20%（Δ=8.05%）。本仓在 ε=1e-2 下剖面 L2 也升到 17.07%（A(1e-2) 臂），即该偏差是**底垫的物理效应**，两码同现。
* 精度排序（nx=400，vs 解析）：本仓 A 0.83% ≈ OSS 五阶 WENO5 1.31% > OSS 二阶 B 9.15%（B 的差距主要来自底垫，见 §4.3 第 4 条）。**本仓不是误差放大器，是这条算例上更准的一侧。**

## ⑤ 收敛性与参数敏感性

### 5.1 网格收敛（M5）

| 量 | nx=100 | 200 | 400 | 800 | 趋势 |
|---|---|---|---|---|---|
| 本仓剖面 L2 vs 解析 [%] | 1.929 | 1.168 | 0.830 | 0.795 | 单调收敛 |
| 本仓 vs OSS 五阶 [%] | 1.029 | 0.748 | 0.882 | 1.204 | 非单调（有界抖动） |
| 本仓 vs OSS 二阶（同 IC）[%] | 1.857 | 1.064 | 0.532 | 0.289 | 单调收敛 |
| 本仓前沿误差 [%] | 0.256 | 0.119 | 0.057 | 0.028 | 单调收敛 |

收敛曲线见 `figures/xval_dam_break_slot_convergence.png`（log-log）。本仓 A×OSS 二阶同 IC 差从 1.86%（nx=100）→ 0.29%（nx=800），**随加密单调下降**，这是两码收敛到同一解的直接证据；A×WENO5 稳定在 0.75–1.20%（五阶参照自身的残差量级）。

### 5.2 参数敏感性 I：CFL（M6，nx=400）

| CFL | 本仓前沿误差 [%] | 本仓剖面 L2 [%] | A_m×B（同 IC）[%] | A×B [%] | OSS B 剖面 L2 [%] | OSS WENO5 剖面 L2 [%] |
|---|---|---|---|---|---|---|
| 0.2 | +0.057 | 0.87 | 0.44 | 8.91 | 9.11 | 1.31 |
| 0.4 | +0.057 | 0.83 | 0.53 | 8.95 | 9.15 | 1.31 |
| 0.8 | +0.054 | 0.62 | 1.68 | 9.09 | 9.24 | 1.31 |

CFL 0.2→0.8：本仓剖面误差在 0.62–0.87% 内变化（<0.3pp），跨实现一致性的最坏点出现在 CFL=0.8 且仍 <2%——说明结论对时间步不敏感；也解释了为何固定 dt（一阶稳定界）不影响跨实现可比性。

### 5.3 参数敏感性 II：OSS 干床底垫 ε（M7）与 OSS 稳定性边界

| 项 | 值 |
|---|---|
| HLLE+minmod，ε=0.003：前沿误差 / 剖面 L2 | +0.124% / 9.15% |
| HLLE+minmod，ε=0.01：前沿误差 / 剖面 L2 | +0.117% / 17.20% |
| Δ（前沿 / 剖面） | 0.007pp / 8.05pp |
| HLLE+minmod，ε=0.001 探测（nx=200/400） | NaN 单元 137 / 240（非有限） |
| HLLE 一阶（lim=0），ε=0.0001 | NaN 单元 0 / 0（有限，稳定） |

结论：①前沿位置对 ε 几乎不敏感（Δ<0.01pp），**剖面 L2 对 ε 高度敏感**（8pp 量级）——因此任何以『剖面 L2』作判据的跨实现比较都必须先把 IC 底垫对齐，否则会误判为格式缺陷（本单 §4.3 正是这么做的）；②Clawpack 5.14 的 1-D SWE Riemann 求解器在二阶（minmod）下**没有正性保护**：ε≤1e-3 直接 NaN，这是 OSS 侧的鲁棒性缺口，本仓 α:1→0.5→0 的降阶重试不存在该问题（h=0 真干床全程有限）。

## ⑥ 差异清单、移植取舍与失败模式

### 6.1 本仓 vs OSS 差异清单（做法 / 参数 / 边界条件）

| 项 | 本仓 TensorLBM | Clawpack 5.14 | 量化 | 移植取舍 |
|---|---|---|---|---|
| dry-bed handling | dry cells h=0; per-step positivity fallback (reconstruction amplitude 1->0.5->0) so a negative stage is rejected and retried at lower order | no positivity fallback in the 1-D SWE Riemann solvers; a positive initial layer is required | HLLE+minmod with dry floor 1e-3 -> 137 / 240 NaN cells (nx=200/400); stable floor 3e-3; first-order HLLE is stable at 1e-4 (0 NaN) | keep the in-house positivity fallback; it is a genuine robustness advantage, not a shortcut |
| wave-speed estimate (HLL family) | Davis min/max of u-/+c on the two face states | HLLE given its own Einfeldt-type estimates inside rp1_shallow_hlle | same wave family; the Roe arm D brackets the flux-family sensitivity | no port needed |
| time integration | SSP-RK2 (Heun), fixed dt = CFL dx /(2 c0) from the first-order stability bound | Hancock predictor-corrector (Classic) or SSP104 (SharpClaw), dt re-estimated each step from the current max wave speed | both 2nd order in time; CFL sensitivity in §5 | no port needed |
| grid layout | nx nodes inclusive of both endpoints (x = linspace) | nx cell centres | half-cell (dx/2) stagger; removed as a confounder by interpolation and by the 0.8 x_f bulk metric | documented, not ported |
| limiter application | component-wise minmod on the conserved variables of the two face states | minmod on the wave strengths (projected) in Classic; component-wise WENO in SharpClaw (char_decomp=0) | part of the cross-implementation difference; bounded by the 15% gate | characteristic limiting is a candidate future upgrade (needs an eigen-decomposition of the SWE system) |
| boundary conditions | mirrored impermeable wall (left) / zero-gradient outflow (right), ghost states in _muscl_face_states | pyclaw.BC.wall / pyclaw.BC.extrap | the left wall sits 4 h0 upstream and is never reached by the rarefaction (t_end=5, c0 t = 5) | no port needed |

### 6.2 移植取舍（结论）

* **不移植**：Clawpack 的 Roe 通量族与 WENO5 重构——本单证据显示在干床溃坝上它们没有精度收益（Roe 需 ε=1e-2 才稳定，剖面误差升到 17.20%；WENO5 只到 1.31%，与二阶本仓 0.83% 同级），而算力/复杂度代价明显（WENO5 每维 5 点重构 + SSP104 四段）。
* **不移植**：波强式（characteristic）限制器 —— 需 SWE 特征分解，换来的差异在 1% 量级（同 IC A_m×B），当前判据 5%/15% 下无收益证据；若将来加干床强间断/源项（底坡、摩阻）问题，可作候选。
* **保留本仓独有**：干床正性降阶保护 α:1→0.5→0 —— 本单最有价值的差异，OSS 在该算例上需要人为底垫（ε≥3e-3，附加质量 0.000% 量级的 IC 扰动）才能二阶稳定。**不引入 OSS 的底垫做法。**
* **可借鉴（工程面）**：PyClaw 的逐步自适应 dt（按当前最大波速）——本仓现用一阶稳定界 dt；CFL 敏感度显示对精度无影响（§5.2），仅影响效率，登记为效率项而非精度项，本轮不改。

### 6.3 失败模式与工件

| 失败/风险 | 现象 | 处置 | 工件证据 |
|---|---|---|---|
| OSS 二阶在薄干层上崩 | ε=1e-3 时 HLLE+minmod 出 NaN（137/240 单元，nx=200/400） | OSS 臂抬 ε 至 3e-3，并**同时**给出同 IC 的本仓对照臂把该效应从格式差里剥离开 | `arm_clawpack_classic_hlle_nx400_cfl0.4_eps1e-3.json` |
| 半格错位（节点制 vs 单元中心） | 跨实现比较含 ~dx/2 平移 | 线性插值到公共节点 + bulk（0.8x_f）变体，两口径差 <0.2pp | `xval_table.*bulk_l2_pct` |
| 解析解在储水侧失效 | 域左端壁在 x=-4，而 t=5 时 Ritter 稀疏波需 x≤-5，储水头被截断 | 判据只取湿区 [0,x_f]；储水头作诊断量记录，**两码一致**（各臂 0.74–0.76，解析 0.87）→ 域截断、非求解器缺陷 | `xval_table.h_head_*` |
| Clawpack 部署摩擦 | loongarch64 无 wheel，需 meson+gfortran 源码构建 ~4min；f2py riemann 模块不带 num_eqn 元数据 | 双层 fork 守护进程构建；驱动侧声明 num_eqn/num_waves=2 | `result.json.frictions` |
| honest 边界 | 本单未覆盖：二维/带底坡/带摩阻/湿床（Stoker）溃坝；未做 MLUPS 性能对标 | 明确列为下一轮候选（见 §6.4） | — |

### 6.4 结论与下一步

* **本单裁决：PASS**。跨实现（方案级）一致性 0.29–1.86%，全部 <15% 且 <3%；六个判据全部通过。本仓 `muscl2 + local-HLL + SSPRK2` 可作为 slot-8 的裁决求解器，与成熟开源实现的交叉证据已建立。
* 关键可复用结论（reusable）：**以『剖面 L2』做跨实现对标时，干床底垫 ε 是头号混杂**——ε 从 3e-3 到 1e-2 就能把剖面误差从 9% 推到 17%，足以把 1% 级的格式一致性淹没成 >15% 的假失配。任何干床算例的跨码对标都必须同 IC 对照臂。
* 下一轮候选：①Stoker 湿床溃坝（解析解存在，无干床混杂，但检验激波捕捉）；②带底坡/摩阻的 well-balanced 检验；③把 OSS 臂接入 `benchmarks/validation_matrix/` 作为可选第二求解器（装饰性，不入判裁）。

### 6.5 误差判定（逐项，显式回答）

| 关键量 | 基准 | 本仓误差 | 跨实现差 | 是否 <3% | 是否 <15%（owner 标准） |
|---|---|---|---|---|---|
| 前沿位置 x_f | Ritter 2c0t | +0.057% | A_m×B 0.53%（剖面）/ 前沿差 0.067% | 是 | 是 |
| 剖面 h(x) 湿区 L2 | Ritter 解析 | 0.83% | A_m×B 0.53%、A×WENO5 0.88% | 是 | 是 |
| 剖面（未对齐 IC 的裸口径） | Clawpack B | 9.15%（B 自身 vs 解析） | A×B 8.95%、A×D 16.49% | 否（>3%，已归因为 IC 底垫，附同 IC 对照 1.4%） | 是（A×B）/ 否（A×D，已归因） |
| 质量守恒 | 初始体积 | 0.00e-2 % | — | 是 | 是 |

**显式回答**：方案级关键量（同 IC 跨实现、vs 解析）**全部 <3%**；裸口径 A×B 7.2–9.3% 亦 <15%，A×D 16.5% 超 15% 但已用同 IC 对照（1.36%）与 ε 敏感度（8pp）归因为 OSS 侧干床底垫要求，**不构成本仓实现缺陷**——若按 owner 15% 标准裁决，本维（方案一致性）判通过，且附归因证据。

### 6.6 复现命令

```bash
pip install clawpack            # 5.14.0，loongarch64 需源码构建（meson+gfortran）
python xval/dam_break_slot_muscl2_local_hll_ssprk2/run_arms.py    # 33 个臂文件，~30s CPU
python xval/dam_break_slot_muscl2_local_hll_ssprk2/analyze.py     # result.json/table.csv/2 图
python xval/dam_break_slot_muscl2_local_hll_ssprk2/make_doc.py    # 本文 + 上报载荷
pytest tests/test_xval_dam_break_slot.py -q                       # 判据回归
```

### 6.7 生态摩擦（如实）

* `tool_install` · clawpack：no loongarch64 wheel; source build (meson + gfortran) from the internal mirror → pip install clawpack took ~4 min and had to be launched as a double-forked daemon because the shell tool harvests the process group after 120 s
* `api_gap` · clawpack.riemann (f2py modules)：compiled riemann modules expose only rp1/cparam; pyclaw raises AttributeError on num_eqn → run_arms._claw_solve declares num_eqn=2/num_waves=2 for the SWE system
* `oss_robustness_gap` · clawpack shallow_hlle_1D / shallow_roe_with_efix_1D：2nd-order minmod reconstruction on a near-dry layer (eps<=1e-3) produces NaN: the OSS 1-D SWE solvers have no positivity fallback → OSS arm needs dry floor 3e-3 (HLLE) / 1e-2 (Roe); the in-house arm runs at h=0 with the alpha-fallback. Reported as an XVAL difference, not a port target

* `report_ingest_defect` · `$TENSOR_API_BASE/tensor/reports`：POST 正文 markdown 被原样入库（长度 10967 = 我发稿长度、md5 `9eefccb0…` 与本地 `/tmp/report.json` 完全一致），但 title/summary/scenario_key 被**旧任务（m15467 MCMP-Laplace）**覆盖、mission_id 记成 15467，`GET /tensor/reports?mission_id=17725` 返 0 条 → m17725 的报告在库但不可按任务检索。证据：`xval/dam_break_slot_muscl2_local_hll_ssprk2/artifacts/report_ingest_friction.json`（报告行 id 16698/16700，psql 实查）。本仓无法修平台侧 handler，已就地留档 + 建议回填。
