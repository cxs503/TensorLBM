# XVAL-LBM-CONSERVATIVE-ALLEN-CAHN-H-POPULATION — 预注册（判据先于实验）

任务 m17678｜仓库 TensorLBM｜分支 `agent/m17678`｜对标对象 CFD-GO/TCLB
`models/multiphase/d3q27_pf_velocity`（保守 Allen-Cahn **h-population** 模型）
本文件在跑任何基准**之前**写定；后续结果不得回改阈值（如回改须在报告中显式声明并说明理由）。

## 0. 决策（Decision）与备选（Alternatives）

**决策**：TensorLBM 是否把现有的 h-population 保守 Allen-Cahn 核当作"与成熟开源实现
（TCLB d3q27_pf_velocity）同族可对标"的核——即两者在同一算例上的偏差是否 < 15%（owner 精度标准）。
若 < 15%：判定"已对齐"，差异清单只作为参数/移植说明。
若 ≥ 15%：判定"未对齐"，决策改为**采用 TCLB 等价核（本单新增 `phasefield/cac_h_population.py`）
作为 OSS-平权参考核**，并把差异溯源到具体项（∂t(φu) 修正项 / 平衡态阶数 / 梯度模版 / 法向截断）。

**备选实现（A/B 两臂，同一算例、同一初值、同一时间步长）**
- **Arm A**＝TensorLBM 现有 h-population 保守 AC 核（`tensorlbm.cac_lbm2d` / `tensorlbm.cac_lbm`
  的 h 分量函数：`f_equilibrium` / `f_source` / `lambda_phi` / `iso_gradient_*` / 流式），
  即 Liang et al. PRE 97,033309 的 CAC h-LBE（含 ∂t(φu) 二阶修正项，D2Q9 / D3Q19 模版）。
- **Arm B**＝本单新增的 **TCLB 平权核**（`tensorlbm.phasefield.cac_h_population`）：
  h_i^eq=w_iφ[1+3c·u+…]、F_phi,i=w_i[(1−4(φ−1/2)²)/W](c_i·n̂)、
  h_i←h_i−ω(h_i−h_i^eq+½F_phi,i)+F_phi,i、ω=1/(3M+½)、D3Q27/D3Q19/D2Q9 模版，
  全部系数由 TCLB 源码**程序化抽取**（见 §5），不是抄写。
- **参考（Ref）**＝解析解/守恒律（非 OSS 需要跑 GPU 的二手数字）：
  ① 平/曲界面平衡剖面 φ=½[1−tanh(2(r−R)/W_eff)]；② 均匀平移精确解 φ(x−Ut)；
  ③ 周期域内 Σφ 精确守恒（通量形式 ⇒ 机器精度）；④ 界面宽度标定 W_eff=3·M·W（推导见报告②）。

## 1. 假设（Hypothesis）

- H1（核等价）：u≡0 时 Arm A 与 Arm B 的每步更新在机器精度内相同（因为 Arm A 的唯一额外项是
  ∂t(φu)，u≡0 时恒为 0）。可证伪点：若差 >1e-12，则两臂在模版/法向/系数上另有差异。
- H2（物理正确性）：Arm B 在静态与平移算例上满足解析参考（界面半径、质心、守恒量）<15%；
  若 H2 失败，则说明"TCLB 式"离散在本仓实现条件下不可用，OSS 对标结论为 no-go。

## 2. 度量（Metrics）与预注册阈值（Thresholds）

| 代号 | 度量 | 阈值（先定） | 类型 |
|---|---|---|---|
| P1 | F_phi 场 vs TCLB 源码抽取表达式的相对 L2 误差 | **< 1e-13** | 结构恒等（非物理精度） |
| P2 | 梯度/拉普拉斯模版 vs TCLB `model.R` IsotropicGrad/myLaplace 公式原文求值 | **< 1e-13** | 结构恒等 |
| P3 | 静态界面质量漂移 \|M(t)−M(0)\|/M(0)，t=2000 步 | **< 1e-10** | 守恒律 |
| P4 | 静态界面半径漂移 \|R(t)−R(0)\|/R(0) | **< 5%** | 精度 |
| P5 | 静态界面剖面 vs 解析 tanh 的相对 L2 误差 | **< 15%** | 精度（XVAL 判据） |
| P6 | 均匀平移：质心位移相对精确解误差、φ 场 vs 精确平移场的相对 L2 误差 | **< 5% / < 15%** | 精度（XVAL 判据） |
| P7 | 跨臂 A↔B 场相对 L2 误差（同一算例、同一步数） | **< 15%** | **XVAL 主判据** |
| P8 | 界面宽度标定：W_measured/W_pred（W_pred=3·M·W） | **1 ± 15%** | 参数律 |
| P9a | 单步核恒等：本仓模块 vs TCLB 源码抽取公式（u=0 与 u≠0） | **< 1e-13** | 结构恒等 |
| P9b | 单步跨臂恒等：Arm A vs Arm B（u=0，同一初值，一步） | **< 1e-6** | 结构恒等（见 §2.1 修订） |

### 2.1 预注册修订（跑数之前，2026-11-13，原因可查）

写完 §2 后、跑任何基准之前，读码发现 `src/tensorlbm/d2q9.py`（以及 `d3q19.py`/`d3q27.py`）
的格子权重表以 **float32** 存储（如 1/9 → 0.1111111119389534，相对误差 7.4e-9），
Arm A 的内部算子（`iso_gradient_2d` 等）直接用该表。因此"跨臂单步恒等"的物理下限是
浮点往返 ~1e-8，不可能达到 1e-12。故把原 P9 拆为 P9a（本仓模块 vs **源码抽取公式文本**，
两侧均 float64 精确，阈值 1e-13 不变）与 P9b（跨臂单步，阈值放宽到 1e-6 并记录残差量级）；
多步跨臂偏差仍由 **P7（<15%）** 裁定，不因该换算放宽。该 float32 权重表问题本身记入报告 ⑤ 注意点。

判"对齐"⇔ P7 < 15%（且在 P3–P6 上两臂都过）。任一项 > 阈值即如实记 FAIL 项，不得事后放宽。

## 3. 变量、混杂与对照（Variables / confounders / control）

- 受控（两臂相同）：初始 φ 场（同一 tanh 剖面函数）、网格、步数、W、M（⇒ 同一 τ=0.5+3M）、
  周期性流式、float64。
- 被改（臂间）：平衡态阶数（A：一阶 O(u) ；B：二阶 O(u²)，另设 `eq_order=1` 消偏置）、
  ∂t(φu) 项（A 有 / B 无）、速度模版（D2Q9 vs D3Q27）、法向截断 minGradient。
- 混杂控制：**u≡0 的静态算例**把 ∂t(φu) 与 O(u²) 项都关掉，是 A/B 的"同核"对照组；
  均匀平移算例才是差异的暴露组。float64 + 同一随机种子（本单无随机项）。
- 参数敏感性（≥2 轴，§5 of 报告）：①移动率 M∈{1/12,1/6,1/3,1/2}（W 固定）→ 验 W_eff=3MW；
  ②界面宽度 W∈{2,3,4,6}（M=1/3 固定）→ 网格分辨率敏感性；③平移速度 U∈{0.01,0.02,0.04}（CFL）。

## 4. 重复与停止规则（Repetitions / stopping）

- 每次运行确定（无随机），重复 = 换网格与换参数，不做同构重复；数值不确定性用 float64 下
  "步数加倍后指标变化"衡量（时间收敛性，§5 报告）。
- 停止规则：任一场出现 NaN/Inf ⇒ 停止该 run，记录发散步号与末态 CFL，作为失败模式上报；
  指标在预注册步数处读数（静态 2000 步、平移 1 个整周期 512 步），**不因结果好看而延长或截短**。

## 5. 环境与溯源（Environment / provenance）

- 主机 CPU；`torch 2.7.1`（float64）、`sympy 1.14.0`；单进程；无 GPU。
- TCLB 源码：`/tmp/TCLB`（`models/multiphase/d3q27_pf_velocity/Dynamics.c.Rt`、
  `Dynamics.R`、`model.R`、`src/lib/lattice.R`），文件 sha256 在运行时计算并入 JSON 工件。
- **零手抄**：所有 TCLB 侧数字（权重、tmp1 系数、更新式结构、默认参数 M/IntWidth/omega_phi、
  模版系数 16/4/1/72 与 /36）由脚本**解析源码文本**得到并落盘哈希；本仓侧数字由代码运行时产出。
- 许可：TCLB 为 GPL-3.0（`docs/oss_inventory/2026-11-09_TCLB_deep_dive.md` 第 20 行结论
  =reference-only）；本单**不复制、不改写其源码**，只读公式并独立实现（离散格式属公开文献）。

## 6. 分析（Analysis，先定）

- 误差定义：相对 L2 = ‖a−b‖₂/‖b‖₂（b=参考）；半径由 φ≥0.5 面积（2D）/体积（3D）反演；
  质心由 φ 加权一阶矩；宽度由 tanh 拟合（最小二乘，拟合区间 0.05<φ<0.95）。
- 无量纲参数表必须完整：M（移动率）、W、ω、U（格子速度）、Re 无关（无粘性/纯 AC 核）、
  CFL=U·dt/dx、R/W、L/R。
- 结论只允许写成"在 §5 环境与 §3 受控条件下"的陈述；不得外推到有流体反馈（g-LBE）的耦合情形
  ——**本单比较的是纯 AC 核（prescribed u），与全耦合精度无关**（解释边界）。

## 7. 安全/资源限制

单 run ≤ 2 min CPU（2D 128² ≤ 4000 步；3D 48³ ≤ 400 步），全部 run 合计 ≤ 15 min，
不递交集群（规模远低于 10 min 门槛）；内存 <2 GB。