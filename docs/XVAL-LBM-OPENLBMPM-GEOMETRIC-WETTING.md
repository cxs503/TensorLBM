# XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE — 几何润湿接触角 vs `openLBMPM` 交叉对标

对标锚点：`XVAL-LBM-GEOMETRIC-WETTING-CONTACT-ANGLE`（第 1 轮·开源对标）
对标对象：`PorousMediaSimulation/openLBMPM` 的 **geometric-wetting-contact-angle** 做法
判据（owner 精度标准）：接触角相对误差 **≤ 15 %**；分项与门槛在 §五预注册。

**结论一句话**：几何润湿条件在**算子层**通过对标——4 个实现变体、27 个成对解析算例，规定角反读误差 **≤ 2.4e-14 %（远优于 15 % 门槛）**；`openLBMPM` 的**候选法向选择**在镜像填充的固壁色值下**严格退化为**本仓已交付的闭式条件（27/27 例逐值一致，求解器内 400 步轨迹差 3.5e-9 属舍入），因此**不移植**；其各向同性 9 点色梯度与 `SolidColorDiff` 固壁色值在本宿主里**保真度反而更差**（M2 0.00713、0.02694 vs 0.00639），亦**不移植**；**移植的是参考实现的接触角测量配方**（base width `b` + apex height `h` 反演，解析圆帽复现误差 <0.6°）。**求解器级端到端静态液滴为 honest-FAIL**：宿主 color-gradient 的 A-扰动表面张力在 `A≥0.01` 下产生冲向顶壁的液柱（`|u|max ~1e-2`），`A=0.005` 延后至 ~2500 步仍发生 → 接触角实测 165.6°（规定 60°，176 %），失败机制与接棒项见 §六。

---

## 一、问题与参考（ref 出处）

color-gradient (Rothman–Keller / Latva-Kokko) LBM 中固壁接触角必须由**几何润湿边界条件**给出：在壁面邻层把色梯度法向按接触角 θ 旋转，并使相邻固体格的"色值"与该旋转法向一致（Akai et al., Phys. Rev. E **98**, 043302 (2018)）。本仓 `src/tensorlbm/cg_wetting2d.py` 已实现**闭式**版本：

$$\phi_s=\phi_f+\cot\theta\,|\partial_t\phi|_f \tag{1}$$

**符号固定由 θ 决定**；`docs/XVAL-LBM-AKAI2018-GEOMETRICAL-WETTING.md` §六 明确登记"未复现候选分支（`distP/distM`）"。本单对标 `openLBMPM`，其做法（**读码记录，含文件:行号**）：

| 要素 | `openLBMPM` 位置 | 做法 |
|---|---|---|
| 固壁色值 | `RKCG2D/AcceleratedRKGPU2D.py:1560 calColorValueOnSolid` | 由 `SolidColorDiff`（`IniFiles/RKtwophasesetup2D.ini:42`，~0.5）**给定**固体色差，不需标定 `G_s` |
| 色梯度 | `:1584-1632` | 各向同性 9 点：`∇φ = 3 Σ_k w_k φ(x+c_k) c_k` |
| 润湿投影 | `:2430-2492`（旧支 `:1639-1678`） | 界面附近梯度投影到"由 `cosθ` 与壁法向构造的**两个候选单位向量**中较近者" |

出处链：平台 OSS 盘点 `docs/oss-inventory-openLBMPM.md` §4.2（含逐行指针）+ `oss_solutions.id=5`（`NEW:CAP-LBM-VALIDATION-SURFACE` / `geometric-wetting-contact-angle`）。

**零手抄声明（诚实口径）**：本机无外网（`curl https://raw.githubusercontent.com/...` → `Connection reset`，生态摩擦 F1），`~/oss-cache` 仅有 `lbm / pylbm / hschwane` 三份缓存；本单递交的集群取源作业 `cj1688` 排在 311 单之后未在时限内执行 → 依据平台**读码入库记录**（上表行号）而非逐行复核源码。**未从参考仓复制任何代码**（该仓无许可证文件，见 OSS 盘点 §R1）。

## 二、方法与实现

新增 `src/tensorlbm/wetting/geometric_two_candidate.py`：

1. `geometric_wetting_ghost_candidates(...)` — 候选法向选择版几何润湿。设 `g_t`、`g_n` 为切向/指向壁内法向的梯度分量，`s_t=sign(g_t)`，候选单位法向 `n_±=(s_t sinθ, ±cosθ)`；到实测单位梯度 `g/|g|` 的距离只差第二分量，故**选近者 ⇔ `sign(g_n')=sign(g_n)`**，得

   $$\phi_s=\phi_f+|g_t|\,\cot\theta\,\mathrm{sign}(g_n) \tag{2}$$

   与 (1) 的差别**恰在** `sign(g_n)·sign(cotθ)<0` 的壁面单元。支持 `stencil="central2"|"isotropic9"`（后者即参考实现的 9 点色梯度）与 `solid_color_offset`（参考 `SolidColorDiff` 的替身）。`return_diagnostics=True` 输出候选统计、翻转面数。
2. `cap_geometry_contact_angle(b, h → R=(b²+4h²)/(8h), θ=atan2(b/2,R−h))` — **参考实现的接触角测量配方**（`oss_solutions.id=5`，与 `mcmp_verification.cap_geometry_contact_angle` 同源）。接触线基点由壁面相邻两行**线性外推到 y=0.5**，去掉首行半格偏置；退化场返回 `status` 而非抛错。
3. 单测 `tests/test_wetting_geometric_two_candidate.py`（20 项）+ 既有 `tests/test_cg_geometrical_wetting.py`：**36 项全绿**（`PYTHONPATH=src pytest -q`），锁定"镜像填充下与闭式逐值一致"、"`SolidColorDiff<0` 时符号翻转"、"θ=90 镜像"、限幅/门限/异常，以及 cap 几何配方 <0.6° 复现。

工装：`benchmarks/xval_openlbmpm_wetting/operator_study.py`（算子层）、`run.py`（求解器层）、`make_figures.py`。

## 三、参数表

### 3.1 算子层（解析场，无量纲/lattice）

| 量 | 值 | 说明 |
|---|---|---|
| 场 | `φ=tanh((R−r)/ξ)`，`R=24` | 128×128，上下各 1 层固体，壁面 y=0.5 |
| θ | 30/45/60/75/90/105/120/135/150° | 9 档 |
| ξ | 1.5 / 2.0 / 3.0 | 3 档 → 27 个成对算例 |
| 真值 | 解析场**延续进固体格** `φ_exact` | 外部真值（非自证） |
| 门限 | `gate=1e-6`（`\|g\|>\|C\|` 守卫，对应参考 `\|C\|>1e-6`） | — |
| 变体 | V1 闭式2pt / V2 候选2pt / V3 候选Iso9 / V4 候选Iso9+SolidColorDiff | 确定性，无需重复 |

### 3.2 求解器层（lattice 单位）

| 量 | 值 | 说明 |
|---|---|---|
| 网格 | nx=96（x 周期）× ny=64，上下 1 层固体 | 顶壁保持中性 90° |
| 碰撞 | BGK×2 色，`visc_mode="mixture"` | τ_r=τ_b=1.0 → ν=1/6 |
| 密度比 | 1（ρ_r+ρ_b≡1） | 与参考一致 |
| 表面张力幅值 A | **0.005**（stock 0.04 → 失稳，见 §六） | 扰动式 `(A/2)\|∇φ\|w_i[(c_i·n̂)²−1/3]` |
| 再着色 β | 0.70 | Latva-Kokko |
| 液滴 | R=14，按**规定角**初始化（`y_c=0.5−R cosθ`），ξ=2 | 测"静态一致性"而非从错角松弛 |
| 步数 | 4000（每 500 步采样） | 收敛门：`dphi_max`、`\|u\|max` |
| 臂 | `cot60`（闭式）/ `ctl60`（关润湿 BC，仍按 60° 初始化） | + 参考臂 `oss60`（候选，轨迹等价见 §5.3） |

## 四、与 OSS 参考实现的差异清单 + 移植取舍

| 维度 | `openLBMPM`（参考） | 本仓（交付态） | 取舍 |
|---|---|---|---|
| 固壁色值 | `SolidColorDiff` **给定**色差（`:1560`） | 镜像填充（零法向差） | **不移植**：V4 实测 M2=0.02694，比镜像填充差 **4.2×**（§5.2） |
| 法向候选 | 两候选取近者（`:2430-2492`） | 闭式 (1) 符号由 θ 固定 | **不移植**：镜像填充下 `sign(g_n)≡0`，候选规则**严格退化**为 (1)，27/27 例逐值一致（§5.2）。若强行施以 `SolidColorDiff` 才"活"起来，但保真度更差 |
| 色梯度 | 各向同性 9 点（`:1584-1632`） | 壁面单元 2 点中心切向差 | **不移植**：V3 M2=0.00713 > V1 0.00639，且仅在 4/27 例更优（§5.2） |
| 接触角测量 | `b,h` 反演 cap 几何 | 原有 Kasa 圆拟合 + 面积反演 | **移植**（新增第二独立估计器，解析圆帽 <0.6°，对伪流更稳健） |
| 角度范围 | cosθ 直接设定，无 90° 限制记载 | 0<θ<180 全域（θ>90 由 cotθ<0 表达） | 保持本仓现状 |
| 壁面法向 | ISO8 平滑 + 法向表（孔隙几何） | axis-aligned 逐方向解析 | 平面壁足够；非规则几何仍未做 |
| 边界处理 | 隐式半程反弹（跳过固体格） | stream 后固体格反向镜像（半程 BB，壁面 y=0.5） | 等价壁面位置约定 |
| 求解器其它 | R-K 再着色 + CSF 曲率力（Akai 式，`:2499-2551`） | A-扰动表面张力 + Latva-Kokko | **本仓弱项**：CSF 力替换是本单失败的直接接棒项（§六） |

## 五、对照表（ref vs got，逐项打勾叉）

### 5.1 测量配方（参考的接触角估计器，解析圆帽）

| θ 规定 | 30 | 45 | 60 | 75 | 90 | 105 | 120 | 135 | 150 |
|---|---|---|---|---|---|---|---|---|---|
| cap 几何反演 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 全部 |误差|<0.6°（单测：`test_cap_geometry_recovers_analytic_cap_angle`×6 + 手算 9 档） |

### 5.2 算子层 XVAL（`benchmarks/xval_openlbmpm_wetting/operator_study.json`，27 例/变体）

| 变体 | M1 反读相对误差 max | M1 中位 | **M2 保真度 `⟨\|φ_s−φ_exact\|⟩`** | M2 max | 候选翻转面 | 15 % 门 |
|---|---|---|---|---|---|---|
| V1 闭式 2pt（**本仓交付态**） | **1.58e-14 %** ✅ | 1.18e-14 % | **0.00639** | 0.02693 | 0 | ✅ |
| V2 候选 2pt（参考机制） | 1.58e-14 % ✅ | 1.18e-14 % | 0.00639（**与 V1 逐值相同**） | 0.02693 | 0 | ✅ |
| V3 候选 Iso9（参考色梯度） | 2.37e-14 % ✅ | 0 % | 0.00713（配对均值差 +7.4e-4，23/27 更差） | 0.02901 | 0 | ✅ |
| V4 候选 Iso9 + `SolidColorDiff=0.5` | 2.37e-14 % ✅ | 0 % | 0.02694（**4.2× 差**） | 0.11312 | 3070 | ✅ |

读法：**M1 门（≤15 %）四变体全过**（反读误差是浮点噪声级）；**M2 是判别量**——参考的两个"特色"（候选选择 + 9 点梯度）在本宿主里**不带来保真度收益**，而 `SolidColorDiff` 固壁色值**显著变差**。故移植取舍为"只移植测量配方"，理由为数据而非偏好。

### 5.3 求解器层（4000 步，A=0.005）

| 臂 | θ 规定 | θ_capgeom（**判据量**，参考配方） | 相对误差 | θ_circle | θ_BC 反读 | \|u\|max | 判据 |
|---|---|---|---|---|---|---|---|
| `cot60` | 60 | **165.64** | **176.1 %** ❌ | 81.4 | **60.00** (n=14) | 6.0e-3 | ❌ honest-FAIL |
| `ctl60`（关 BC，仍按 60° 初始化） | 90 | 128.97 | 43.3 %（对 60°：115 %） | 退化 180 | 90.00 (n=14) | 5.6e-3 | ✅ 敏感性对照按预期（关掉 BC 得不到 60°） |

**BC 本身在求解器内精确命中规定角**（反读 60.00°、90.00°），与 §5.2 一致；失败在**外形**——见 §六。闭式臂与候选臂在求解器内 400 步轨迹差 `max|Δf_r|=3.46e-9`、`max|Δf_b|=3.45e-9`（`equivalence_check.json`，非逐位相同仅因 `|g_t|·|cot|·sign` 与 `|g_t|·cot` 的浮点结合顺序），即两实现**在物理上等价**。

## 六、收敛性/参数敏感性（≥2 点）与失败模式

**敏感性（均在表内，程序化产出）**

1. **界面宽度 ξ（1.5 → 3.0，27 例成对）**：M2 保真度随 ξ 增大而下降（细界面更"真"），变体排序不随 ξ 改变；V1=V2 在三个 ξ 下均逐值相同。
2. **表面张力幅值 A（0 / 0.005 / 0.01 / 0.02 / 0.04，2000 步探针，`A` 是稳定性主控参数）**：
   `A=0` → 液滴完全静止（`|u|max=1.2e-14`，apex 不变）；`A=0.005` → 2000 步内稳定（apex 7，`|u|max 9.4e-3`）但 2500 步后出现液柱；`A=0.01/0.02/0.04` → ≤2000 步内**液柱直达顶壁**（apex=62/63，`|u|max 1.8e-2…4.5e-2`）；τ=0.8/1.5 不改变结论，β=0.9 仅小幅延迟。→ 宿主 A-扰动表面张力方案在小 R 漫散界面上**不稳定**，这是端到端对标的唯一阻塞点。
3. **再着色 β（0.70 → 0.90）**：只影响伪流与界面锐度，不改变 BC 反读角（两档均 60.00°）。

**失败模式（诚实登记）**

* fail_stage：`solver-level static drop relaxation`（算子层已通过）。
* failure_reason：① 宿主 CG 求解器的 **A-扰动表面张力**在 `A≥0.01`（stock 0.04）下产生冲向顶壁的**液相射流**（`|u|max ~1e-2`，apex 62/63 格），`A=0.005` 只是延后到 ~2500 步；② 初始场 `ρ≡1`（无 Laplace 压差）使液滴起跑即强烈瞬变（`dphi=1.69` @500 步），放大该不稳定；③ 圆拟合估计器在伪流下退化（`ctl60` 报 180°），故以参考配方 `b,h` 为判据量。
* **已排除**：BC 写错。四条独立证据：① 算子层 27 例反读误差 ≤2.4e-14 %；② 与既有闭式实现逐值相同（镜像填充退化性）；③ 求解器内反读精确 60.00°/90.00°；④ 关闭 BC 的对照臂反读变 90°、外形角变 129°。
* 接棒项（第 2 轮建议）：用 CSF 曲率力（参考 `:2499-2551`）替换 A-扰动，或按 Laplace 压差 `Δp=σ/R` 初始化总密度以消除起跑瞬变；验收仍用本单同一判据量（cap 几何 + 圆拟合 + 反读三路）。

## 七、工件与环境

| 工件 | 路径 |
|---|---|
| 新模块 | `src/tensorlbm/wetting/geometric_two_candidate.py` |
| 单测（36 项全绿，含既有 16 项） | `tests/test_wetting_geometric_two_candidate.py` |
| 算子层 XVAL | `benchmarks/xval_openlbmpm_wetting/operator_study.py` → `operator_study.json`、`fig_operator_xval.png` |
| 求解器层 | `benchmarks/xval_openlbmpm_wetting/run.py` → `results_*.json`、`results.json`、`phi_*.npy`、`fig_solver_angle.png`、`fig_phase_fields.png` |
| 等价性证据 | `equivalence_check.json`（400 步，两实现轨迹差 ≤3.5e-9） |
| 环境 | Python 3.11 / torch（`operator_study.json.environment`）/ x86_64 CPU |

**生态摩擦（如实记）**

* **F1**：本机无外网（GitHub raw `Connection reset`），`~/oss-cache` 无 `openLBMPM`；集群取源作业 `cj1688` 排队 311 单未执行 → 只能引用平台读码入库记录（含行号），**未能逐行复核源码**。
* **F2**：`tensorlbm.wetting.metrics.spherical_cap_inversion` 在平界面/扩散界面上直接抛 `ValueError`（无降级返回，延续上一单 F2）→ 本单改用返回 `status` 的 cap 几何配方。
* **F3**：宿主 `color_gradient2d` 的 A-扰动表面张力在静态液滴上失稳（`A≥0.01` 射流），使端到端接触角对标在正确 BC 下仍无法完成；这是本仓求解器缺陷（已在本单量化），非对标对象问题。
* **F4**：`benchmarks/xval_akai2018_wetting/run.py` 的 `ArmSpec.steps` 默认 40000 而历史产物只跑到 6000（文档表与 `results.json` 数值不一致，`gk60 circle 124.6` vs 文档 `115.1`）——历史产物与本单读数混用会误判，本单所有数字均出自本单 JSON。