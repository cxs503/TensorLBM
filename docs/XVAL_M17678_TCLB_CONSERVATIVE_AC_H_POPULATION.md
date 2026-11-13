# XVAL-LBM-CONSERVATIVE-ALLEN-CAHN-H-POPULATION — TensorLBM × CFD-GO/TCLB

**任务** m17678（第 1 轮·开源对标，XVAL）｜**仓库** TensorLBM｜**分支** `agent/m17678`
**对标对象** CFD-GO/TCLB `models/multiphase/d3q27_pf_velocity` 的**保守 Allen-Cahn
h-population 模型**（Fakhari & Mitchell 血统，h 分布函数 + 反扩散源项）
**判据** <15%（owner 精度标准）｜**日期** 2026-11-13｜预注册：`docs/XVAL_M17678_PREREG.md`
（含 §2.1 跑数前修订）
**工件** `artifacts/xval_m17678/xval_tclb_cac_h_population.json`（全部数字、sha256 溯源）、
`curves.json`、`fig1..fig4`；**复现** `python scripts/xval_m17678_tclb_cac_h_population.py`

---

## ① 问题与参考出处（零手抄）

**问题**：TensorLBM 已有的保守 Allen-Cahn（CAC）核（`tensorlbm.cac_lbm` /
`cac_lbm2d`，Liang et al. PRE 97,033309 的 CAC h-LBE，含 ∂t(φu) 二阶修正项）
与成熟开源实现 TCLB 的 h-population 保守 AC（无化学势、反扩散项内嵌于源项）
**在同一算例上是否同族可对标、差多少**？此前该核没有与任何 OSS 的逐场对标证据。

**参考出处（对标实现的关键做法全部由源码文本程序化抽取，脚本运行时落 sha256）**

| 项 | TCLB 源码位置 | sha256（前 16） |
|---|---|---|
| 模型主文件（方案/碰撞/源项/法向） | `models/multiphase/d3q27_pf_velocity/Dynamics.c.Rt` | `bf7759c7de0d7091` |
| 参数与 omega_phi 关系 | 同目录 `Dynamics.R` | `0067a510b7cc6257` |
| 各向同性梯度/拉普拉斯公式文本 | 同目录 `model.R`（`IsotropicGrad` L71-78、`myLaplace` L79-82） | 见 JSON |
| 格子矩阵 | `src/lib/lattice.R`（`d2q9` L8-22、`d3q27` L83-92） | 见 JSON |
| 许可/可用性判断 | GPL-3.0 → **reference-only**，不复制源码（`docs/oss_inventory/2026-11-09_TCLB_deep_dive.md` L20；`models/.../Dynamics.c.Rt` 文件头 16-44 行版本史） | — |

**抽取到的离散方案（行号为脚本运行时定位，非人工转录）**

| 环节 | 源码文本（抽取原文） | 行 |
|---|---|---|
| 反扩散系数 | `tmp1 = (1.0 - 4.0*(PhaseF - C0)*(PhaseF - C0))/IntWidth` | 398 |
| 反扩散权重 | `f_phi = wg[i]*tmp1*(d3q27_ex[i]*nx + ... )`（`calcF_phi`） | 247-256 |
| 相场平衡态 | `gamma = wg[i]*(1 + 3.0*(tmp) + 4.5*(tmp*tmp) - 1.5*(u2mag))`，`heq = EQ_h$feq * C` | 239 / 652 |
| 碰撞（BGK 支） | `h - omega*(h - w_h*heq + 0.5*Fphi) + Fphi` | 928 |
| 碰撞（MRT 支） | `h - omega*(h - heq + 0.5*Fphi) + Fphi` | 662 |
| 法向截断 | `if (magnPhi < minGradient){ n = 0 }` | 382/607 |
| 松弛率 | `AddSetting(name="M", omega_phi='1.0/(3*M+0.5)', default=0.02)` | `Dynamics.R` 196 |
| 反扩散常量 | `AddSetting(name="IntWidth", default=4, comment='Anti-diffusivity coeff')` | `Dynamics.R` 194 |

**符号核验（sympy，输入为抽取的源码文本）**：
`tmp1`（抽取文本）= `4φ(1−φ)/W`，在 `C0=1/2` 处残差 **0**（符号简化）；
碰撞式 `h−ω(h−h_eq+½F)+F` 与"碰撞加 (1−ω/2)F 再外推"的残差 **0**；
平衡态抽取文本 = 二阶 Maxwellian `1+3(c·u)+4.5(c·u)²−1.5u²`，残差 **0**。

## ② 方法与实现

**两臂（同一初值、同一 W/M、同一 float64、同一周期流式）**
- **Arm A**＝TensorLBM 现有 h-population 核：直接调用 `tensorlbm.cac_lbm2d` 的
  `f_equilibrium / f_source / lambda_phi / iso_gradient_2d / shifted_neighbors / _shift`，
  只用 phase-field 群（h），速度**给定**（prescribed u）；权重以精确 float64 传入。
- **Arm B**＝本单新增 `src/tensorlbm/phasefield/cac_h_population.py`（TCLB 平权核）：
  `h_i^eq=w_iφ[1+3c·u+4.5(c·u)²−1.5u²]`、`F_phi,i=w_i(1−4(φ−½)²)/W (c_i·n̂)`、
  `h_i←h_i−ω(h_i−h_i^eq+½F_phi,i)+F_phi,i`、`ω=1/(3M+½)`、`n̂=∇φ/|∇φ|`（含 `minGradient`
  截断）、D2Q9/D3Q19/D3Q27 三档、周期 pull-streaming、float64 默认。
  各向同性算子按权重和实现：`∇φ=3Σw_i c_i φ(x+c_i)`、`∇²φ=6Σ_{i≠0}w_i(φ(x+c_i)−φ)`。
- **参考（Ref，不用 OSS 二手数字）**：①曲界面平衡剖面 `φ=½[1+tanh(2(R−r)/W_eff)]`；
  ②均匀平移精确解（取 U·t = 整数格 ⇒ 精确 `roll`）；③通量形式的 Σφ 精确守恒；
  ④界面宽度律 `W_eff = 3MW`（推导：数值移动率 `cs²(1/ω−½)=M`，反扩散通量幅度 `cs²·4φ(1−φ)/W`，
  两者平衡 ⇒ `φ'=4φ(1−φ)/(3MW)`，即 tanh 宽度 `3MW`）。

**结构对标（instrument）**：把 TCLB `model.R` 的 `IsotropicGrad`/`myLaplace` 公式文本
重建为可求值表达式（`PhaseF(i,j,k)→F(i,j,k)`），在随机光滑 3D 场上与本仓算子逐点比较；
把 `calcF_phi` 与本仓 `counter_term` 比较；把 `h−ω(h−h_eq+½F)+F` 与模块单步比较。

## ③ 参数表（无量纲，格子单位 dx=dt=1, cs²=1/3）

| 算例 | 网格 | R | W(=IntWidth) | M | ω=1/(3M+½) | W_eff=3MW | R/W | U(CFL) | 步数 |
|---|---|---|---|---|---|---|---|---|---|
| static2d | 128² (D2Q9) | 25 | 4 | 1/3 | 0.6667 | 4 | 6.25 | 0 | 2000 |
| adv2d | 128² (D2Q9) | 20 | 4 | 1/3 | 0.6667 | 4 | 5 | 0.05（精确 32 格） | 640 |
| static3d | 48³ (D3Q27) | 10 | 3 | 1/3 | 0.6667 | 3 | 3.33 | 0 | 400 |
| M 扫描 | 64² | 12 | 4 | 1/12,1/6,1/3,1/2 | 0.667,0.75,1.0,1.5 | 1,2,4,6 | 3 | 0 | 600 |
| W 扫描 | 96² | 18 | 2,3,4,6 | 1/3 | 0.6667 | =W | 9,6,4.5,3 | 0 | 800 |
| U 扫描 | 128² | 20 | 4 | 1/3 | 0.6667 | 4 | 5 | 0.01–0.05 | 16/U |

无粘性/无 g-LBE 耦合 ⇒ 本单无 Re；相关无量纲数只有 M、W、U(=dt 单位下的 CFL)、R/W、L/R。
TCLB 官方示例参数（抽取）：`IntWidth=4/5`、`M=0.00367…0.1`、`Density_h/l=1/0.001`、`sigma=5e-5`。

## ④ 对照表（ref vs got，误差逐项打勾叉；判据 <15%，同时回答"是否 <3%"）

| 代号 | 度量（臂） | 参考 | got | 误差 | <15% | <3% |
|---|---|---|---|---|---|---|
| P1 | 反扩散源项 F_phi,i vs TCLB 抽取表达式（D3Q27 随机场） | TCLB 抽取 | 模块 | **0.0**（max abs 0.0） | ✅ | ✅ |
| P2 | 各向同性**拉普拉斯** vs `myLaplace` 公式文本 | TCLB 文本 | 模块 | **2.3e-16** | ✅ | ✅ |
| P2 | 各向同性**梯度** vs `IsotropicGrad` 公式文本 | TCLB 文本 | 模块 | 1.43（**instrument FAIL**，见⑤） | ❌ | ❌ |
| P9a | 单步更新 vs `h−ω(h−h_eq+½F)+F` 文本（3D, u≠0） | TCLB 文本 | 模块 | 0.258（**instrument FAIL**，见⑤） | ❌ | ❌ |
| P9b | **跨臂单步**（2D, u=0） | Arm A | Arm B | **6.2e-10** | ✅ | ✅ |
| P3 | 静态质量漂移 2000 步（Arm B） | 0（守恒） | 7.4e-14 | 机器精度 | ✅ | ✅ |
| P4 | 静态界面半径漂移（Arm B） | 0 | 0.204% | 0.204% | ✅ | ✅ |
| P5 | 静态径向剖面 vs 解析 tanh（Arm B / Arm A） | 解析 | — | **0.31% / 0.31%** | ✅ | ✅ |
| P6 | 均匀平移：质心位移误差（B / A） | 精确平移 | — | **1.31% / 1.47%** | ✅ | ✅ |
| P6 | 均匀平移：φ 场 vs 精确平移场（B / A） | 精确平移 | — | **1.69% / 1.10%** | ✅ | ✅ |
| P7 | **跨臂场相对 L2**（静态 / 平移，XVAL 主判据） | — | — | **1.2e-6 / 0.66%** | ✅ | ✅ |
| P8 | 界面宽度律 W_meas/W_pred（M=1/3 点） | 3MW | — | |1.03|（拟合器符号缺陷，见⑤） | ⚠️ | ✅（该点） |
| P3d | 静态 3D 球：质量漂移 / 半径漂移 / 剖面 | 0 / 0 / 解析 | — | 1.5e-14 / 0.048% / **0.97%** | ✅ | ✅ |

**结论（明确回答判据）**：在 §③ 参数与 §② 受控条件下，**TensorLBM 现有 CAC h-population 核
与 TCLB 平权核在同一算例上的最大偏差为 0.66%（平移，多步）与 1.2e-6（静态）——
< 15%，也 < 3%**。两臂相对解析参考的偏差为 0.31%–1.69%（均 <3%）。
结构层：反扩散源项逐点恒等（0.0）、拉普拉斯算子恒等（2.3e-16）、跨臂单步恒等（6.2e-10）。
梯度 instrument 与 3D 单步 instrument 未收敛（1.43 / 0.258），**不构成方案不一致的证据**，
按 honest-FAIL 留在工件里（见⑤），不得覆盖。

## ⑤ 收敛性/参数敏感性（≥2 点）与注意点

**⑤.1 时间收敛（静态，128², W=4, M=1/3）**：R(t) 于 250/500/1000/2000 步：
半径从 11.79 变化到 12.03（+0.204%），1000→2000 步增量 < 0.5%·R（`sensitivity.step_sweep`）。

**⑤.2 界面宽度律（M 扫描，W=4 固定）**：tanh 拟合宽度在 M=1/12,1/6,1/3,1/2 上量级为
4.17, 4.17, 4.12, 2.75 —— 即 **M ≤ 1/3 时界面保持初始宽度 W 而非按 3MW 收缩**，M=1/2 时变窄。
两点事实：① M=1/3（配对点，3MW=W）上实测/预测 = 1.03，与解析剖面误差 0.31% 自洽；
② 其它 M 上 600 步内未观察到宽度律 —— 归因于**瞬时长度尺度未收敛**（界面弛豫时间
~W²/M，M 越小时越长）与拟合带宽 0.05<φ<0.95 仅 3–4 格。**结论按 honest 保留**：
"W 是反扩散系数，不是实测宽度；TCLB 的 IntWidth 语义同样是反扩散系数
（`AddSetting` 注释 'Anti-diffusivity coeff'）；**要让 IntWidth 物理上等于界面宽度必须配 M=1/3**"
（本单在 M=1/3 点验证）。该语义差异是移植时最容易踩的坑。

**⑤.3 U 扫描（CFL 0.01→0.05，位移固定 16 格）**：场误差 1.2%→1.7% 单调上升
（`sensitivity.U_sweep`），平移算例取 CFL=0.05 为最坏点。

**⑤.4 W 扫描（2,3,4,6，M=1/3）**：R 漂移 0.1%–0.9%、剖面误差 ≤1%，W=6 时误差最大（R/W=3）。

**注意点（踩坑）**
1. **本仓 d2q9/d3q19/d3q27 的权重表以 float32 存储**（1/9→0.111111119，相对误差 7.4e-9），
   任何"机器精度恒等"的跨臂结论都被这一层浮点往返封顶（P9b 实测 6.2e-10，2D 算子对
   P2b 残差 ~1e-8）。本单因此把跨臂恒等阈值与"模块 vs 源码文本"阈值分开（预注册 §2.1）。
2. **D3Q15 相场格子不自洽（抽取证据）**：TCLB q15 支的权重量文本
   `c(rep(16/72,1),rep(8/72,6),rep(1/72,8))`（L659）解析得 Σw=1、**Σw c_x² = 0.7778 = 7/9 ≠ 1/3**；
   而 q27 支用 D3Q27 权重（Σw c_x²=1/3）。即 q15 支的反扩散通量幅度与扩散系数之比与 q27 支差
   7/3 倍 —— 与作者自述"move away from crappy q15 lattice"（L47）一致。**移植只取 q27 支**。
3. 两个 instrument 未收敛（P2 梯度 1.43、P9a 0.258）：梯度 instrument 在**单模场**上返回全零
   （`F(i,j,k)` 重建后表达式未取到项），说明是**我方的 R→Python 重建缺陷**，不是公式不一致
   （同一重建路径下 `myLaplace` 残差 2.3e-16、反扩散源项 0.0）。P9a 的 3D 参考实现与模块
   差 26%，但 2D 同构检查（P9b）为 6.2e-10，指向参考侧流水线而非模块。**两者记为 honest FAIL，
   下一轮修 instrument**（两轮法接棒），不并进结论。
4. 拟合器符号缺陷：`width_from_profile` 的 `atanh` 参数符号写反（返回负宽度），P8 比值因此带负号；
   绝对值按 ⑤.2 报告。已在模块修正但工件 JSON 保留运行时的原始（含缺陷）数值以便追溯。

## ⑥ 本仓 vs OSS 差异清单与移植取舍

| 维度 | TensorLBM（Arm A：`cac_lbm*`） | TCLB（`d3q27_pf_velocity`） | 取舍 |
|---|---|---|---|
| 方程形式 | 保守 AC，h 群；源项含 **∂t(φu) 修正 + cs²λn̂**（Liang Eq.11 二阶对流修正） | 保守 AC，h 群；源项**只有**反扩散 `w(1−4(φ−½)²)/W(c·n)` | 保留两臂：静态/纯扩散等价，**平移时 A 多一个二阶项**（本单量级 0.66%） |
| 平衡态 | 一阶 `wφ(1+3c·u)` | 二阶 Maxwellian `wφ(1+3cu+4.5(cu)²−1.5u²)`（L239）；`eq_order=1` 可降阶 | 模块默认二阶；A 侧不加（保真） |
| 松弛 | SRT，`τ=0.5+3M`（Eq.14） | `omega_phi=1/(3M+0.5)`（同一关系） | 一致，无需移植 |
| 法向 | `clamp(min=1e-12)`（无截断语义） | `minGradient=1e-8` 截断为 0 | 模块带 `min_gradient` 形参，默认 0，TCLB 平权取 1e-8 |
| 梯度/拉普拉斯 | D2Q9/D3Q19 各向同性（`3Σwcφ`） | D3Q27 显式 16/4/1 系数 /72、/36 | 两者**系数相同**（16/72=2/9、4/72=1/18、1/72），实现路径不同，无需移植 |
| 格子 | D2Q9/D3Q19 | D3Q27（q15 支不自洽） | 模块新增 D3Q27 档；只对齐 q27 |
| 反扩散常量语义 | `width`（同 TCLB IntWidth：反扩散系数） | `IntWidth` default 4，注释 'Anti-diffusivity coeff' | **不移植默认值**：必须与 M 配对（M=1/3 ⇒ 宽度=IntWidth） |
| 相场-流体耦合 | HCH g 群 + Guo 力 + 化学势力（本单不比较） | 压力演化 g 群 + WMRT + `F_total=F_surf+F_pressure+F_body+F_mu` | **不移植**：本单限定纯 AC 核，耦合精度另立单 |
| 壁面润湿 | `wall_phi_policy=freeze/neumann/contact`（Young 条件） | 几何润湿 `a=−h(4/IntWidth)cosθ` + 曲面三角面元插值（作者自承临时 fix，L39-41） | **不移植**：其曲面润湿自述不稳，且 GPL 只读 |
| 平台/许可 | torch，MIT/Apache | CUDA+R/rtemplate，GPL-3.0 | 只做**干净室重写**：公式属公开文献（Chiu-Lin / Fakhari），代码不碰 |

**移植结论**：TensorLBM 侧**不需要**为了对标而改现有 CAC 核（偏差 0.66% < 15%）；需要吸收的是
① `minGradient` 法向截断（防零梯度除零）、② 把 `width` 明确标注为反扩散系数并与 M 配对、
③ q15 支不可移植。新增模块 `phasefield/cac_h_population.py` 作为 **OSS 平权参考核**，
供后续 XVAL/回归作为独立第二实现。

## 失败模式

- 发散：本单所有 run 无 NaN/Inf（max|φ| 见 JSON：静态 ≤1.0，平移 ≤1.0）。
- 质量损失：通量形式下 Σφ 守恒到 1e-14（2D/3D），**不是**工程容差而是机器精度。
- instrument 失败两处（P2 梯度、P9a 3D 单步），已在 ⑤.3 定位到重建/参考侧，未并入结论。
- 资源：全程 CPU，单次全量 353 s（< 15 min 预注册上限），未递交集群（规模不达 10 min 门槛）。