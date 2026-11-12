# T1-01 · 多相模型数学模型与离散误差系统比较

**锚点** GAP-LBM-T1-MULTIPHASE（第 1 轮 · 方法研究，**本单不改任何求解器代码**）
**仓库** TensorLBM · **基线** `a9870455`（分支 `agent/m15618`）
**数字来源** 全部引用程序生成的 [`T1_NUMERIC_APPENDIX.md`](T1_NUMERIC_APPENDIX.md)（生成器
`analysis/t1_phase_budget.py`，机器可读全量 `artifacts/T1_multiphase/t1_phase_budget.json`）。
**东家判据** 五条物理验收（Laplace / 静态伪流 / 长期稳定 / ρ比与μ比独立 / HR 终端速度）见
[`T1-02`](02_ACCEPTANCE_CRITERIA.md)。

本文件回答一个问题：**五个模型的失败是"参数没调好"还是"结构上做不到"？**
结论先行：四个家族里有三个是结构性不足（数学上不可能同时满足 HR 五判据），
第四个（相场）是实现未完成。这直接决定 [`T1-04`](04_VERDICT_AND_ROADMAP.md) 的选型裁决。

---

## 0. 记号与统一框架

单松弛(BGK)格子 Boltzmann 方程（本仓 `d3q19.py`，19 个离散速度，$c_s^2=1/3$，$\Delta x=\Delta t=1$）：

$$
f_i(\mathbf x+\mathbf c_i,\,t+1)=f_i(\mathbf x,t)-\frac{1}{\tau}\big[f_i-f_i^{eq}(\rho,\mathbf u)\big]+S_i,
\qquad
\rho=\sum_i f_i,\quad \rho\mathbf u=\sum_i \mathbf c_i f_i+\tfrac{1}{2}\mathbf F .
\tag{0.1}
$$

多相模型族之间的差别只在三处：① 非理想压力从哪来（界面如何产生 Laplace 压差）；
② 密度对比 $\rho_d/\rho_c$ 由谁承载（EOS / 被平流的总密度 / 序参量插值）；
③ 黏度场 $\mu(\mathbf x)$ 如何从状态量得到（$\tau$ 的选取）。五条验收判据恰好逐条压在这三处之上，
所以"哪个模型能同时满足五条"是个纯结构问题，不需要先跑大规模算例。

---

## 1. 模型 A：Shan–Chen 伪势（单组分，SCMP）

**数学形式**（本仓 `multiphase.py::psi_carnahan_starling` / `make_psi_carnahan_starling`）：

$$
\mathbf F(\mathbf x)=-G\,\psi(\rho(\mathbf x))\sum_i w_i\,\psi(\rho(\mathbf x+\mathbf c_i))\,\mathbf c_i,
\qquad
p(\rho)=\rho c_s^2+\frac{G c_s^2}{2}\psi^2 .
\tag{1.1}
$$

$\psi$ 由目标状态方程反解：$\psi=\sqrt{2(p_{EOS}-\rho c_s^2)/(G c_s^2)}$。取
Carnahan–Starling $p_{EOS}=\rho RT\frac{1+\eta+\eta^2-\eta^3}{(1-\eta)^3}-a\rho^2$，$\eta=b\rho/4$。

**离散误差的三个来源**

1. **力的非梯度部分 = 伪流的唯一来源。** Shan (PRE 73, 047701, 2006) 的结论：对**平直**界面，
   (1.1) 的离散求和可写成某个标量的离散梯度，故 $\nabla_h\times\mathbf F_h=0$，伪流为零；
   对**弯曲**界面，$\mathbf F_h$ 不再可积（离散旋度 $O(\Delta x^2)$ 非零），驻涡由该旋度驱动，
   $u_{sp}\sim \dfrac{\sigma}{\mu}\cdot\mathcal O\!\left(\dfrac{\Delta x^2}{W^2}\right)$。
   ⇒ **伪流不是"力大"，是"力的旋度大"**；加密只能把它压成 $O(\Delta x^2)$（且 $W/\Delta x$ 需同步放大）。
2. **力注入格式的离散不平衡。** 速度平移法与 Guo/EDM 法的平衡态不一致会留下
   $(2-\frac{1}{2\tau})$ 量级的过量注入——本仓有实测缺陷记录：
   `docs/mcmp3d_stability_findings.md §1`（D3Q19 的 Guo 分支过量注入）。这是实现级误差，
   与模型无关，可单点修好。
3. **状态方程无气相分支。** 历史 `psi_carnahan_starling` 硬编码 $a=0.5,b=4,RT=1/3$，该组参数下
   $\partial p/\partial\rho>0$ 全域成立 ⇒ **无 van der Waals 环、无 spinodal、根本不能相分离**
   （附录 A1 第一张表 `has_vdw_loop=false`；模块自身 docstring 亦已记载）。合法参数 $a=5.0$ 才有环。

**共存与密度比上限**（附录 A1，Maxwell 等面积构造，$a=5,b=4,RT=1/3$）：
$\rho_g=9.9\times10^{-3}$、$\rho_l=0.3548$、$\rho_l/\rho_g=35.8$；
沿共存曲线降温（$T_r=0.6\to0.95$）密度比 $4403\to53$。

> **重要更正（相对既有叙事）**：SC 的瓶颈**不是**密度比上限。密度比 35.8（甚至更高）在共存曲线上
> 是拿得到的；真正杀死 HR 的是下面两条结构性锁 + 伪流地板。

**锁 I（密度对比与可压缩性绑定）：** $p=\rho c_s^2+\frac{Gc_s^2}{2}\psi^2$ 意味着界面压差由
密度场承担，因此流体自身必须是可压缩的；液滴内部密度由 EOS 固定，不能当自由参数用。
对 HR 需要的 $\rho_d/\rho_c=2$，$T_r\approx0.9$ 附近可以给到，但代价是伪流随 $\Delta\rho$ 增长
（实测 $u_{sp}$ 0.137–0.140，见 §6）。

**锁 II（黏度比锁，致命）：** BGK 里 $\mu=\rho c_s^2(\tau-\tfrac12)$。

- $\tau$ 全局常数 ⇒ $\mu$ 均匀 ⇒ $\lambda=\mu_d/\mu_c=1$（**没有黏度比**）；
- 让 $\tau=\tau(\rho)$ 去要黏度比 ⇒ $\lambda=\rho_d/\rho_c$（**黏度比被密度比绑死**）。

附录 A1 第三张表就是这个二选一：`lambda_uniform_tau = 1`、`lambda_tau_of_rho = ρ_ratio`。
HR 判据要的恰恰是"给定密度比下自由选 λ"，故 SCMP 结构上不可能满足判据④。

**判据映射**：①Laplace 可（需修 EOS 参数）②伪流不可（§7 量化）③长期稳定可 ④**不可** ⑤随④失败。

---

## 2. 模型 B：Shan–Chen 双组分（SC-MCMP）

**数学形式**（本仓 `multiphase3d.py::sc_two_component_force_3d` / `collide_sc_two_component_3d`）：

$$
\mathbf F_\sigma=-G_{12}\,\psi_\sigma(\rho_\sigma)\sum_i w_i \psi_{\bar\sigma}(\mathbf x+\mathbf c_i)\mathbf c_i
+\rho_\sigma\mathbf g,\qquad \sigma\in\{1,2\},\ \ G_{12}<0\ \text{相分离}.
\tag{2.1}
$$

两组分各有自己的分布 $f_1,f_2$ 与 $\tau_1,\tau_2$，总密度 $\rho=\rho_1+\rho_2$。

**为什么它比 A 更接近 HR、但仍然不行**

- **比 A 好**：$\tau_1\ne\tau_2$ 在形式上给了独立黏度比（全仓唯一"看起来能调 λ"的 SC 路径）。
- **不行 1（动量守恒）**：本仓 docstring 自述（源码确认）——不等 $\tau$ 时每步多出源项
  $\partial_t\mathbf M=(\frac{1}{\tau_2}-\frac{1}{\tau_1})\frac{\rho_1\rho_2(\mathbf u_1-\mathbf u_2)}{\rho_1+\rho_2}$，
  强分离下结构性失稳（实测 $\tau=(1,0.75)$、$G_{12}\le-1.5$ 发散）。$\tau$ 相等时该项为零，
  但那时又回到 $\lambda=1$。**要么不守恒，要么没有黏度比。**
- **不行 2（压力压塌/组分反转）**：干预力是原始密度的函数（$\psi=\rho$ 线性）时，环境组分被
  "泵"进液滴核心。仓库包络实验（附录 A6，`probe/p0b_envelope.json`，14 例，对比度 1.05–2.1×、
  $G\in[-2.5,+1.5]$）末帧 $R_{eq}=0$（液滴消失）或组分混合；$G=+2.5$ 两例在 6000 步内 NaN。
  可用对比度窗口 $\gtrsim1.05$ 即失效 ⇒ 零浮力。
- **不行 3（λ 与 σ 不可解耦）**：不等 $\tau$ 同时改变界面张力与分离程度，"调 λ 就动了 σ"，
  不满足判据①②④的正交性要求。

**判据映射**：①②可（在混合态/低对比度内）③**不可** ④形式可、实测不可 ⑤不可。

---

## 3. 模型 C：Color-Gradient（CG）

**数学形式**（本仓 `multiphase3d.py::color_gradient_step_3d`）：两组分分布 + 颜色梯度扰动

$$
\mathbf F_{pert}=-\tfrac12 A\,\kappa_c\,\nabla\!\left(\frac{\rho_r-\rho_b}{\rho_r+\rho_b}\right),\qquad
f_i^{\sigma*}=\frac{\rho_\sigma}{\rho}f_i+\beta\frac{\rho_r\rho_b}{\rho}\,w_i\cos^2\!\left(\frac{\theta}{2}\right)
\tag{3.1}
$$

而**总密度 $\rho=\rho_r+\rho_b$ 是被平流的标量，压力取等温理想气体形式 $p=\rho c_s^2$**。

**这条链的数学后果（本单最干净的"结构必败"证明）**

静态液滴的力学平衡是 Laplace 定律 $p_{in}-p_{out}=(d-1)\sigma/R$（$d=3$：$2\sigma/R$）。
既然 $p=\rho c_s^2$，**能承受的密度对比上限被 EOS 定死**：

$$
\boxed{\ \Delta\rho_{max}=\frac{(d-1)\sigma}{R\,c_s^2}\ }\tag{3.2}
$$

超出部分无处可去，只能被声波带走（vented acoustically）。附录 A2 数值：$\sigma=10^{-2}$、$R=12$
时 $\Delta\rho_{max}=5.0\times10^{-3}$，即 $\rho_r/\rho_b\le1.005$；要维持 HR 的密度比 2（$R=12$）
需要 $\sigma=2.0$——比典型取值大约 400 倍。而 $\sigma=2.0$ 在显式表面张力力格式下已越过稳定界
（$\sigma\lesssim\rho\Delta x^3/\Delta t^2$ 量级，$\rho\approx1$ 即 $\sigma\lesssim1$），
所以"用大 σ 换密度比"这条退路也被封死。

**仓库实测与 (3.2) 的自洽性检验**（附录 A2 σ 反解表）：`probe/p0_static.json` 两个初值
（$r_{in}=0.5$ 与 $r_{in}=2.0$）各自塌缩到平台 $1.0027$ 与 $1.0024$，对应
$\sigma_{implied}=4.1\times10^{-3}$ 与 $6.0\times10^{-3}$——两个独立初值给出同一量级的 σ，
且塌缩后的对比度几乎相同。这正是 (3.2) 预测的"由 σ 与 R 决定的平台"，不是数值扩散的偶然结果。
$r=2.0$ 例的 $R_{eq}$ 从 11.03 涨到 14.87（多余密度排空后液滴反而膨胀），与"总密度不可压缩性缺失"一致。

**黏度比锁**：单一共享 $\tau$ ⇒ $\lambda=\rho_d/\rho_c$（与 A 的锁 II 同型）。判据④结构不可。

**伪流的假象**：CG 静滴末帧 $u_{max}=4.9\times10^{-4}$，看起来比 SC 的 $1.5\times10^{-1}$ 好两个
数量级，但此时密度对比已被压到 $\sim2.4\times10^{-3}$（附录 A2）。用同一无量纲
$S=u_{sp}\mu/\sigma$ 记账：CG $\approx0.02$、SC $\approx7.0$（附录 A7）——CG 的"低伪流"是
用扔掉密度对比买来的，不能作为模型质量证据。这一条必须写进判据②的口径：
**伪流必须与"密度对比存活"同时报告**。

**判据映射**：①可（低对比度内）②读数须扣除对比度塌缩 ③可 ④**不可** ⑤不可（无浮力）。

---

## 4. 模型 D：自由能（Swift / Cahn–Hilliard 变密度）

**数学形式**（本仓 `phasefield/free_energy.py`）：双阱自由能 + Korteweg 应力

$$
f(\phi)=-\frac{A}{2}\phi^2+\frac{B}{4}\phi^4+\frac{\kappa}{2}|\nabla\phi|^2,\qquad
\mu_\phi=-A\phi+B\phi^3-\kappa\nabla^2\phi,\qquad
\mathbf F_s=-\phi\nabla\mu_\phi
\tag{4.1}
$$

$\phi$ 按 Cahn–Hilliard（Model B，守恒）演化：$\partial_t\phi+\nabla\!\cdot(\phi\mathbf u)=\nabla\!\cdot(M\nabla\mu_\phi)$。

**优点（判据④的唯一"干净"解）**：$\rho$ 与 $\mu$ 都通过 $\phi$ 插值 ⇒ $\lambda$ 与密度比在代数上
互不约束，且这是逐点恒等式（本仓 `two_phase_viscosity.py` +
`docs/benchmarks/hadamard_rybczynski_droplet_revival_2026-11.md §3.5` 已把该解耦做到机器精度：
随机相场 4×4 组合 λ 误差 $\sim10^{-16}$）。CH 守恒 ⇒ 判据③（液滴不溶解）在模型层面成立。

**为什么本仓的 FE 路径仍不满足 HR**：现有 `phasefield/` 是 $\lambda\equiv1$ 的 Boussinesq 单密度
形式（审计 MP-004 / `terminal_verdict_m155.md` 第 7 行），**没有密度对比** ⇒ 无浮力。且它
**没有压力场**：`tests/test_phasefield_static_droplet.py` 明确断言 Laplace 诊断 `status="withheld"`、
`observed_pressure_jump is None`（源码确认）⇒ 判据①目前无法测量。不是算错，是测不出。

**判据映射**：①②暂不可测（缺 p 场）③模型可 ④**可**（已机器精度）⑤不可（无密度对比）。

---

## 5. 模型 E：Allen–Cahn 相场（两类：非守恒 AC / 守恒 cAC）

### 5.1 非守恒 Allen–Cahn（Model A）——数学上必然溶解，与网格无关

AC 方程：$\partial_t\phi+\nabla\!\cdot(\phi\mathbf u)=\Gamma(\mu_\phi)$
（右侧是反应项，不是散度 ⇒ 序参量积分不守恒）。曲率驱动前锋速度 $v_n=-\tilde M\sigma\kappa$，
球面 $\kappa=2/R$：

$$
\frac{dR}{dt}=-\frac{2\tilde M\sigma}{R}\ \Longrightarrow\ R^2=R_0^2-4\tilde M\sigma\,t
\ \Longrightarrow\ t^*=\frac{R_0^2}{4\tilde M\sigma}.\tag{5.1}
$$

**$t^*$ 与分辨率无关**：加密网格不改变崩塌时刻。仓库探针（$R_0\approx11.4$，
$\phi<-0.5$ 体积 6272→400@100→0@200 步）反演 $\tilde M\sigma\approx0.16$，代入 (5.1)：
$t^*=11.4^2/(4\times0.16)=203$ 步（附录 A3）。**观测 200 步，预测 203 步**——这不是"参数没调好"，
是 Model A 的解析预言。凡"液滴 200 步消失"的路径，改 $W$、改 $\sigma$、改网格都是白改
（旧终审已在 $W\in\{2,4,8\}\times\sigma\in\{0,0.01\}$ 六格全试，全消失）。

### 5.2 守恒相场（cAC：Chiu–Lin / Geier–Fakhari–Lee；本仓 `cac_lbm.py`）

$$
\partial_t\phi+\nabla\!\cdot(\phi\mathbf u)=\nabla\!\cdot\Big[\underbrace{M\big(\nabla\phi-\frac{4\phi(1-\phi)}{W}\hat{\mathbf n}\big)}_{\text{扩散}+\ \textbf{界面锐化反扩散通量}}\Big],\qquad
\hat{\mathbf n}=\frac{\nabla\phi}{|\nabla\phi|}
\tag{5.2}
$$

(5.2) 相比 5.1 有两处结构性差别：① 右侧是散度形式 ⇒ $\int\phi\,dV$ 守恒（判据③）；
② 反扩散通量维持解析界面厚度 $W=\tanh$ 剖面，使 $W/\Delta x$ 成为**输入**而非数值涌现量
（判据①的网格-界面关系可控）。

**本仓已有该实现**：`src/tensorlbm/cac_lbm.py`（D3Q19 双 LBE：$h$-LBE 输运 φ、$g$-LBE 输运压力/动量，
$\rho(\phi)$、$\mu(\phi)$、化学势、Guo 力、界面法向锐化通量；引用 [L1] Liang, Xu, Chen, Wang, Chai,
Shi, PRE 97, 033309 (2018)，arXiv:1710.09541；[L2] Wang, He, Wang, arXiv:2306.11320）。
`tests/test_cac_smoke.py` + `tests/test_phasefield_static_droplet.py` 本单实跑 **14 项全通过**（§9）。

**但它的 HR 实跑是诚实失败**：`benchmarks/pending/hadamard_rybczynski_droplet/gates_verified.json`
记录 `terminal_buoyancy` 终端速度相对误差 **−43.46%**（`direct_observable` 门
`recomputed_pass=true`，即误差是真实测出来的，不是漏测）。43% 属于"能跑但不对"，是
实现/标定问题（可修），与 A/B/C 的"跑不了"（结构问题）性质不同。

**判据映射**：①②待补测（本仓 cAC 尚无 p 场诊断——见 T1-02 的 S0）③模型可、实测待验
④**可**（$\mu(\phi)$ 插值 + `two_phase_viscosity.py`）⑤**已跑、−43%**（需修）。

---

## 6. 六轴系统比较（定量）

六轴 = ①非理想压力来源 ②密度比控制 ③黏度比控制 ④界面厚度/网格 ⑤稳定性条件
⑥伪流（$S=u_{sp}\mu/\sigma$ 记账）。

| 轴 | A SCMP | B SC-MCMP | C Color-Gradient | D FE/CH(本仓) | E1 AC 非守恒 | E2 cAC 守恒(本仓) |
|---|---|---|---|---|---|---|
| ① 压差来源 | EOS 不可积力 $\psi\nabla\psi$ | 组分间核力 $G_{12}\psi_1\nabla\psi_2$ | 颜色梯度扰动 + 理想气体 EOS | Korteweg $-\phi\nabla\mu$ | $-\phi\nabla\mu$ | $-\phi\nabla\mu$ + h-LBE 压力场 |
| ② 密度比 | EOS 共存曲线决定（35.8 @基准；可更高） | 两组分密度和，界面处受核力挤压 | **被 $p=\rho c_s^2$ 卡死**：$\Delta\rho\le2\sigma/(Rc_s^2)$ | 序参量插值（代数自由） | 插值（代数自由） | 插值（代数自由） |
| ③ 黏度比 | **锁**：常数τ⇒λ=1；τ(ρ)⇒λ=ρ比 | 不等τ可调但破坏动量守恒（源码自述源项） | 共享τ⇒λ=ρ比 | $\mu(\phi)$ 独立插值 ✔ | $\mu(\phi)$ ✔ | $\mu(\phi)$ ✔（`two_phase_viscosity.py` 机器精度） |
| ④ 界面厚度/网格 | 数值涌现，$W\sim4$ 格；$W/\Delta x$ 不可输入 | 同左 | recolouring 维持，随对比度塌缩失效 | $W=\sqrt{2\kappa/B}$ 可输入 | $W$ 可输入但被反应-扩散失配吃掉（实测 $W_{eq}\approx0.8$ 格） | **$W$ 显式输入**，锐化通量维持 |
| ⑤ 稳定性 | τ>½；力注入离散不平衡（Guo 过量注入已记录）；伪流随Δρ增长 | 不等τ 源项 + 压力压塌 + NaN（$G=+2.5$） | 需 $\sigma\gtrsim1$ 才能托住密度比 ⇒ 越稳定界 | CH 四阶刚性；M 与 τ_g 受稳定界约束 | 无守恒 ⇒ 溶解（不是失稳） | cAC 二阶，刚性远低于 CH；M、W、τ_g 需联合标定 |
| ⑥ 伪流 S | **7.0**（实测归约，附录 A7） | 0.027–0.13（末帧 $u_{sp}$，σ 未标定） | 0.02（但对比度已塌缩到 $2\times10^{-3}$，不可比） | 未测（无 p 场） | — | **未测**（HR −43% 是间接信号） |
| **HR 五判据可达性** | ①②部分/④✗/⑤✗ | ③✗④实测✗⑤✗ | ②④✗⑤✗ | ②缺⑤✗ | ③✗（解析预言） | **①②③④可，⑤待修** |

**判据④的必要条件（引理，可直接用于评审）**：若 $\mu$ 只由全局常数 $\tau$ 与密度决定，则
$\lambda=\rho$ 比；要独立控制，必须存在**与密度无关的第二标量**（相场/序参量）进入 $\tau$。
五族中只有 D、E2 满足；B 只在"牺牲动量守恒"的意义上满足。

---

## 7. 伪流判据的定量口径（本单的方法学贡献）

把伪流和"体积力/毛细速度"放进同一个无量纲数，判据才不会各说各话。定义

$$
S\ \equiv\ \frac{u_{sp}\,\mu_o}{\sigma},\qquad \text{要求}\quad S<0.1\,Ca\quad(\text{终端速度误差}<10\%).
\tag{7.1}
$$

而 HR 的 $Ca$ 不是自由参数（推导见 T1-02 §5）：$Ca=Bo\,(1+\lambda)/(6(2+3\lambda))$，
$Bo=\Delta\rho g(2R)^2/\sigma$。$Bo=0.005,\lambda=1$（球形液滴、等黏度）⇒ $Ca=3.33\times10^{-4}$，
$S<3.3\times10^{-5}$。

**这一条把问题钉死了**：SC 族实测 $S=7.0$，差 $2\times10^5$ 倍。这说明 SC 族的失败是
**5 个数量级**的，不是 2 倍标定误差，因此不是"再调调参数"能救的。

---

## 8. 文献与开源实现对照

> 证据级说明：**引文本身**（作者/刊/年/结论要点）已核对；**离线环境无法核验上游源码行号**，
> 凡涉及具体代码结构的条目一律标 `[源码未核验]`，不得当作本仓事实使用。

| 模型族 | 关键文献 | 上游库中的对应实现 | 对本单的可迁移要点 |
|---|---|---|---|
| SC 伪势 | Shan & Chen, PRE 47, 1815 (1993)；Shan, PRE 73, 047701 (2006)（离散力/压力一致性与伪流来源）；Sbragaglia et al., PRE 75, 026702 (2007)（多程伪势） | OpenLB `shanChenForcedSingleComponent`；Palabos `multiphaseShanChen` `[源码未核验]` | 平直界面零伪流、弯曲界面 $O(\Delta x^2)$ 的结论决定"必须报 $S$ 而不是 $u_{sp}$" |
| SC 力格式 | Li, Luo, Li, Kang, PRE 86, 016709 (2012)（forcing scheme / EDM）；Connington & Lee, JCP 250, 446 (2013)（伪流综述） | OpenLB `chemicalPotential`（EOS 驱动、可换 Peng–Robinson）`[源码未核验]` | EDM/化学势格式是 SC 族唯一仍有希望的分支（作为备选路线，见 T1-04） |
| Color-Gradient | Gunstensen et al., PRE 43, 4320 (1991)；Grunau, Chen, Eggert, Phys. Fluids A 5, 2557 (1993)（密度对比版）；Latva-Kokko & Rothman, PRE 72, 046701 (2005)（recolouring）；Reis & Phillips, J. Phys. A 40, 4033 (2007) | Palabos 颜色模型示例；Lettuce 文档列 "shan-chen / free-surface" `[源码未核验]` | recolouring 的 β 与对比度上限；(3.2) 与文献"CG 密度比靠大 σ 换"的经验一致 |
| 自由能 CH | Swift, Osborn, Yeomans, PRL 75, 830 (1995)；Swift et al., PRE 54, 5041 (1996)；Briant & Yeomans, PRE 69, 031603 (2004)（变密度）；Cahn & Hilliard, J. Chem. Phys. 28, 258 (1958)（Model B 守恒） | waLBerla `phasefield` 模块（CH 与 LBM 耦合）`[源码未核验]`；OpenLB `freeEnergy` `[源码未核验]`；Palabos `freeEnergy` `[源码未核验]` | 守恒性来自 Model B；Lee & Lin, JCP 206, 16 (2005) / Lee & Fischer, PRE 74, 046709 (2006) 给出高密度比下消除寄生流的离散化处方 |
| 非守恒 AC | Allen & Cahn, Acta Metall. 27, 1085 (1979)；Bray, Adv. Phys. 43, 357 (1994)（Model A/B 粗化动力学） | 少见作为生产求解器 | (5.1) 的 $t^*\propto R_0^2$ 是 Model A 的教科书结论 ⇒ 判据③用它做否决性预测 |
| 守恒 cAC / 相场 LBM | Chiu & Lin, JCP 230, 185 (2011)（守恒相场）；Geier, Fakhari, Lee, PRE 91, 063309 (2015)（守恒相场 LBE）；Fakhari, Mitchell, Leonardi, Bolster, PRE 96, 053310 (2017)（局部化改进、高密度比）；Liang et al., PRE 97, 033309 (2018) ＝ 本仓 `cac_lbm.py` [L1]；Mitchell, Leonardi, Fakhari (2018) 三维高密度/黏度比 `[题名/卷期待核]` | waLBerla `phasefield` + EDM 力 `[源码未核验]`；OpenLB 无 cAC 生产路径 `[源码未核验]` | $\rho(\phi)$、$\mu(\phi)$、锐化通量、h-LBE 压力场 = 判据①②③④同时可满足的唯一"文献里已被证明可行"的组合；平衡力（Francois et al., JCP 213, 141 (2006)）是伪流能否进 $10^{-5}$ 的关键 |
| 验收参照 | Hadamard, C. R. Acad. Sci. 152, 1735 (1911)；Rybczynski, Bull. Int. Acad. Sci. Cracovie A, 40 (1911)；Clift, Grace & Weber, *Bubbles, Drops and Rigid Particles*, Academic Press (1978)；Hasimoto, JFM 5, 317 (1959)（周期阵列 Stokes 阻力 ⇒ 盒子尺寸系统偏差） | — | 判据⑤的口径：$Re<0.1$、$Bo<0.01$、盒子 $L/R$ 必须显式做收敛而非"设大点" |

**上游库平台引文**：OpenLB — Krause et al., Comput. Math. Appl. 81, 258 (2021)；
waLBerla — Bauer et al., Comput. Math. Appl. 81, 478 (2021)；Palabos — Latt et al.,
Comput. Math. Appl. 81, 334 (2021)；Lettuce（JAX LBM）— 文档级，`[源码未核验]`。

---

## 9. 本单已实跑的验证（不是纸面推理）

| 项 | 命令 | 结果 |
|---|---|---|
| cAC 冒烟 + 相场静滴诊断不变量 | `python3 -m pytest tests/test_cac_smoke.py tests/test_phasefield_static_droplet.py -q -x` | **14 passed**, 93.85 s |
| 本附录全部数字 | `python3 docs/multiphase/T1/analysis/t1_phase_budget.py` | `artifacts/T1_multiphase/t1_phase_budget.json` + 5 图 + 附录 |
| 源码级断言 | §1/§2/§4/§5.1/§5.2 引用的 `src/...` 文件与 docstring | 源码确认 |

**未做**：任何新数值算例（本单职责是方法研究，不写实现）；−43% 的根因定位（见 T1-04 的
H1–H6 假设表，需 S4 阶段实测判定）。

---

## 10. 结论（→ 移交 T1-04 裁决）

1. **SC（A/B）与 CG（C）是结构性不足**：各自有一条不可绕过的代数锁（λ=ρ 或 $\lambda$=1；
   $p=\rho c_s^2$ 造成的 $\Delta\rho$ 天花板），不是参数问题。
2. **非守恒 AC（E1）在数学上被判死刑**：$t^*=R_0^2/(4\tilde M\sigma)\approx203$ 步，
   与观测 200 步吻合，且与网格无关。
3. **只剩下含"第二标量"的两条路**：D（自由能/CH）与 E2（守恒 cAC，本仓已实现）。
   两者在判据④上已是机器精度；差别是 D 缺密度对比、E2 已有完整变量密度框架但 HR 实测 −43%。
4. **失败量级决定策略**：$S$ 差 $10^5$ 的路线（A/B/C）不值得再投算力；差 1 个数量级
   （E2，需实测确认）的路线值得投入修——这就是 T1-04 的裁决依据。