# T1-02 · 多相物理验收判据（owner 定五条 → 可执行口径）

**锚点** GAP-LBM-T1-MULTIPHASE · 第 1 轮 → 后续实现单的唯一验收口径
**性质** 判据**先于实现**冻结（本文件即冻结文本，后续不得在看到结果后放宽；改判据需新开提案并留痕）
**数字依据** [`T1_NUMERIC_APPENDIX.md`](T1_NUMERIC_APPENDIX.md)（程序生成）+ [`01_MODEL_MATH_COMPARISON.md`](01_MODEL_MATH_COMPARISON.md)
**铁则** 实现单**不得自宣布通过**；每条判据由**独立 Agent + 机器门**复判（见 §7）。

---

## 0. 五条判据与依赖序

| ID | owner 判据 | 一句话口径 | 主要判别对象 |
|---|---|---|---|
| **T1-C1** | Laplace 定律 | 静态液滴内外压差 vs $2\sigma/R$ 线性、斜率即 σ | 界面力与压力场是否自洽 |
| **T1-C2** | 静态液滴伪流速 | 无量纲 $S=u_{sp}\mu_o/\sigma$ 与**存活密度对比**同时报告 | 离散力的非梯度部分 |
| **T1-C3** | 液滴长期稳定 | 2 万步内体积/密度对比不漂移，且**不得**呈 Model A 溶解律 | 界面方程的守恒性 |
| **T1-C4** | 密度比与黏度比独立控制 | 2×2 物性阵上 λ 实测与 ρ 比**正交** | τ 的构造是否含第二标量 |
| **T1-C5** | 液滴上浮终端速度（HR） | $|U_{meas}-U_{HR}|/U_{HR}$ 在 $Re\ll1,Bo\ll1$ 下 | 全链条：浮力→界面→黏性拖曳 |

**依赖序（不许跳级）**：C1 → C2 → C3 → C4 → C5。
理由：C2/C5 的读数只有在"界面已经处于力学平衡"时才有意义——旧审计的最大教训正是
**在混合态/塌缩态上测表面张力与终端速度，测出的数全无效**
（`benchmarks/pending/hadamard_rybczynski_droplet/NOTES.md` G3、审计 MP-002）。

---

## T1-C1 · Laplace 定律

**定义**：对静态球滴，$\Delta p \equiv \langle p\rangle_{r<R/2}-\langle p\rangle_{r>1.5R}=2\sigma/R$
（$d=3$；$d=2$ 时 $\sigma/R$）。$p$ 必须是**热力学压力场**，由分布函数矩构造：
SC 族 $p=\rho c_s^2+\frac{Gc_s^2}{2}\psi^2$；相场族 $p=c_s^2\sum_i f_i+\text{(力项二阶修正)}$。
**禁止**用 $\mu_\phi$ 或 Korteweg 力冒充压力（现有 `phasefield/static_droplet.py` 正是据此
`status="withheld"`——这是正确行为，不是缺陷）。

**协议**
- 周期盒或全反弹壁；$R\in\{8,12,16\}$ 格点，$W/\Delta x\in\{2,4\}$，两档分辨率（盒长加倍、R 同倍）。
- 初始 $\tanh$ 剖面；跑 5000 步或 $|\,\mathrm d\Delta p/\mathrm dt\,|<10^{-6}\sigma/R$ 后取 500 步平均。
- 输出：每例 $\Delta p$、$R_{eq}$（$|\phi|$ 或颜色指标 0.5 等值面）、$\sigma_{meas}=\Delta p\,R_{eq}/2$。

**通过门（全部满足）**
1. $\Delta p$ vs $2/R$ 线性拟合 $R^2\ge0.99$；
2. $|\sigma_{meas}/\sigma_{in}-1|\le10\%$（$W=4$）与 $\le15\%$（$W=2$）；
3. 两档分辨率的 $\sigma_{meas}$ 相对差 $\le10\%$，且误差随分辨率不增；
4. $R_{eq}$ 与初始 $R$ 的偏差 $\le1$ 格（界面漂移）；
5. 全程无 NaN/Inf。

**失败签名**：$\sigma_{meas}$ 随 $R$ 单调漂移（定标错，$S=u_{sp}\mu/\sigma$ 不可比）；
$\Delta p<0$ 或 $\Delta p$ 由初始瞬态主导（$t<100$ 的读数一律作废）。

---

## T1-C2 · 静态液滴伪流速

**定义**：伪流 $u_{sp}=\max_{\mathbf x}\sqrt{u_x^2+u_y^2+u_z^2}$（混合速度，**不取单组分速度**——
`docs/mcmp3d_stability_findings.md` 末节已记录单组分 $u_{max}$ 含稀相 $F/\rho$ 不可作诊断）。
还一必须同报的无量纲数（本单引入，见 T1-01 §7）：

$$
S\ \equiv\ \frac{u_{sp}\,\mu_o}{\sigma},\qquad
\text{门：}\ S\ \le\ 0.1\,Ca_{bench},\quad Ca_{bench}=Bo\frac{1+\lambda}{6(2+3\lambda)} .
\tag{C2.1}
$$

$Bo=0.005$、$\lambda=1$ ⇒ $Ca_{bench}=3.33\times10^{-4}$ ⇒ **$S\le3.3\times10^{-5}$**（附录 A4）。

**为什么必须用 $S$**：绝对 $u_{sp}$ 不可跨模型比较。CG 静滴末帧 $u_{max}=4.9\times10^{-4}$ 看似
优于 SC 的 $1.5\times10^{-1}$，但 CG 的密度对比同时塌缩到 $2.4\times10^{-3}$，$S_{CG}\approx0.02$
vs $S_{SC}\approx7.0$（附录 A7）——两者的"好坏"必须放在同一张无量纲表上。

**协议**
- 与 C1 同一批算例（复用），每档分辨率跑 5000 步，记录 $u_{sp}(t)$ 与 $\rho_d/\rho_c(t)$、$R_{eq}(t)$。
- **三件必报**：$u_{sp}(t)$ 曲线、$u_{sp}$ 的空间分布图、$\rho_d/\rho_c(t)$。
- 判定"稳态伪流"：$t\in[2000,5000]$ 的 $u_{sp}$ 标准差 $\le5\%$ 均值。

**通过门**
1. $S\le3.3\times10^{-5}$（对应权重流 $u_{sp}\le0.1\,U_{HR}$，$U_{HR}$ 取 C5 的基准例）；
2. $\rho_d/\rho_c$ 在 5000 步内保持 $\ge0.95$ 初值（**伪流达标但对比度塌缩 ⇒ FAIL**，
   这一条专门排除 CG 型假通过）；
3. $u_{sp}$ 随分辨率不增（两档）；
4. 伪流场不明显偏向坐标轴（各向异性比 $\max/\text{mean}$ 有界，记录不作硬门）。

**失败签名**：$u_{sp}$ 不随分辨率下降（说明来源是力的旋度而非离散截断，见 T1-01 §1）；
$u_{sp}$ 平台随 $\Delta\rho$ 增长且 $S$ 量级 $>1$（SC 族的实测形态，$S\approx7$）。

---

## T1-C3 · 液滴长期稳定（含 Model A 否决性检验）

**定义**：20000 步（≈ 6.7× C5 基准例的特征时间 3000 步，附录 A5），静滴在周期盒内。

**通过门**
1. 体积漂移 $|\Delta V/V_0|\le10^{-3}$（$V$ 为 $\phi>0$ 或颜色指标体积，**必须是守恒量**的定义）；
2. 密度对比漂移 $|\Delta\rho_d/\rho_d|\le2\%$；$R_{eq}$ 漂移 $\le2\%$；
3. 无 NaN/Inf，$u_{sp}$ 不随时间增长（线性拟合斜率 $\le0$）；
4. **Model A 否决检验（必须做）**：对 $R_{eq}^2(t)$ 做线性拟合，
   若 $R^2\ge0.9$ 且斜率显著为负 ⇒ 判 **FAIL（非守恒界面方程）**。
   依据：非守恒 AC 的解析预言 $R^2=R_0^2-4\tilde M\sigma t$（T1-01 (5.1)），
   仓库探针 $R_0\approx11.4$、$\tilde M\sigma\approx0.16$ ⇒ $t^*=203$ 步，观测 200 步，吻合。

**失败签名**：体积单调减少且 $\Delta V>1\%$（非守恒）；$R^2(t)$ 线性（Model A）；
界面出现格点尺度振荡（$\sigma\lesssim$ 稳定界的征兆）。

---

## T1-C4 · 密度比与黏度比独立控制

**定义**：设目标物性 $(\rho_d/\rho_c,\ \lambda)$，实测
$\lambda_{meas}=\mu_d^{meas}/\mu_c^{meas}$，$\rho^{meas}$ 比由场量直读。

**协议（三层，缺一不可）**
1. **代数层（必要条件）**：逐格点映射 $\mu(\phi)=\mu_c[1+(\lambda-1)\phi]$、
   $\tau(\mathbf x)=\frac12+\mu(\phi)/(\rho(\phi)c_s^2)$；随机相场上 4×4 组合，误差须 $\le10^{-12}$
   （本仓 `two_phase_viscosity.py` + HR 复活报告 §3.5 已做到 $\sim10^{-16}$，此处只做回归）。
2. **水动力层（充分条件）**：两层 Couette（周期流方向、壁面驱动），由速度剖面斜率比反解
   $\lambda_{meas}$；偏差 $\le3\%$。**这一层必须有**，否则"映射对了但力项把黏性吃掉"无法排除。
3. **正交性层**：$2\times2$ 阵 $(\rho$ 比 $\in\{1.5,3\})\times(\lambda\in\{0.1,10\})$，固定 $\sigma_{in}$。
   要求：① $\lambda_{meas}$ 与目标偏差 $\le5\%$；② $\lambda_{meas}$ 对 $\rho$ 比的**相关系数**在
   舍入量级（即 $\partial\lambda_{meas}/\partial(\rho$ 比$)\approx0$）；
   ③ $\sigma_{meas}$（由 C1 同批测）在各物性组合间差异 $\le5\%$。

**通过门**：三层全过。**失败签名**：$\lambda_{meas}$ 随 $\rho$ 比变化（τ 仍被密度绑定，SC 族的形态）；
调 λ 引起 $\sigma_{meas}$ 漂移（λ-σ 耦合，SC-MCMP 不等 τ 路径的形态）。

---

## T1-C5 · 液滴上浮终端速度（Hadamard–Rybczynski）

**参照量**

$$
U_{HR}=\frac{2}{3}\,\frac{\Delta\rho\, g\, R^2}{\mu_o}\cdot\frac{1+\lambda}{2+3\lambda},
\qquad \lambda=\frac{\mu_d}{\mu_c},
\tag{C5.1}
$$

**双基准（必须两个都对）**：
- 解析 (C5.1)（Hadamard 1911 / Rybczynski 1911；Clift–Grace–Weber 1978）；
- 本仓**独立**的锐界面 Q1 轴对称 Stokes 求解器 `src/tensorlbm/hadamard_rybczynski.py`
  （最小耗散原理 + 三容器外推，λ 阶梯 8 点，窗口 A 最大误差 0.80%，见
  `docs/benchmarks/hadamard_rybczynski_droplet_revival_2026-11.md §3.1`）。
  它同时给出**容器偏差的一阶系数** $c_1$（$F(H)=F_\infty(1+c_1R/H+\dots)$，实测 $c_1\approx1.8$–3.1），
  是本判据"盒子收敛"步骤的量化依据，不需要凭经验设 $L/R$。

**有效窗（不可省）**：$Re=2\rho_oU R/\mu_o\le0.1$；$Bo=\Delta\rho g(2R)^2/\sigma\le0.01$；
且 $Ca$ 不是独立的——见下。

**关键结构性事实（本单推导，实现单必须遵守）**：把 (C5.1) 代进 $Bo$、$Ca$ 定义，得到**恒等式**

$$
Ca=\frac{\mu_o U}{\sigma}=\frac{Bo\,(1+\lambda)}{6(2+3\lambda)} .\tag{C5.2}
$$

即：**HR 液滴一旦取定 $Bo$，$Ca$ 就定了**。$Bo\le0.01$ ⇒ $Ca\le6.7\times10^{-4}$（几乎不变形），
所以"C5 是最难的判据"不是因为变形，而是因为**信号只有毛细速度的 $10^{-4}$ 量级**，
伪流地板必须低到 $S\le3.3\times10^{-5}$（C2）。这条恒等式同时把参数设计变成可解的：
给定 $(R,\rho$比$,\lambda,\nu_o,Re,Bo)$ 可闭式解出 $(g,\sigma)$（附录 A5 五个候选例）。

**协议**
1. 先按 C1–C3 在选定 $(\rho$比$,\lambda,\sigma)$ 上过关；
2. 基准例（推荐，附录 A5 第 2 行）：$R=12$、$\rho$比 $2$、$\lambda=1$、$\nu_o=0.1$、$Re=0.05$、$Bo=0.005$
   ⇒ $U_{HR}=2.08\times10^{-4}$、$g=5.43\times10^{-7}$、$\sigma=0.0625$、$\tau_o=0.80$、$\tau_i=0.65$、
   特征时间 $\approx4.3\times10^3$ 步，跑 $\ge1.3\times10^4$ 步；
3. **盒子收敛**：$L/R\in\{8,12,16\}$ 三档（不是"设大点"），用 $\phi=0$ 面或颜色界面测 $R_{eq}$，
   $U$ 对 $a/L$ 线性外推并报残差；外推后仍不足 3% 视为盒子未收敛；
4. **终端判据**：$|\mathrm dU/\mathrm dt|\cdot R/U<10^{-4}$ 持续 1000 步；取 500 步平均；
5. **交叉验证**：与 `hadamard_rybczynski.py` 在同一 $(\rho$比$,\lambda)$ 的 $F_\infty$ 对比；
6. 必报：$U(t)$ 曲线、液滴内部环流速度场图、$\Delta\rho/\rho$ 与 $R_{eq}$ 的时间序列。

**通过门**
- 主门：$|U_{meas}-U_{HR}|/U_{HR}\le10\%$（T1 阶段收口线）；
- 结题门：$\le5\%$，且 $\lambda\in\{0.1,1,10\}$ 三点全部达标（一点达标=偶然，三点才是模型能力）；
- 硬否决：$Re>0.1$ 或 $Bo>0.01$ 的算例读数不计（超出 (C5.1) 适用域）；
- **诚实条款**：−43.46%（现有 cAC 实跑，`gates_verified.json`）作为**基线记录**保留，
  修好后必须**在同一协议、同一网格**下重测并给出前后对照，不允许"换个小盒子重测到通过"。

---

## 6. 判据-模型可行性总表（引用 T1-01 的推导）

| 判据 | A SCMP | B SC-MCMP | C CG | D FE/CH | E2 cAC |
|---|---|---|---|---|---|
| C1 Laplace | 需修 EOS 参数（$a$=0.5 无 spinodal） | 混合态下无效 | 仅低对比度内有效 | 缺 p 场（WITHHELD） | 缺 p 场（**S0 要先补**） |
| C2 伪流 | ✗ $S\approx7$ | ✗ | ✗（假通过，对比度塌缩） | 未测 | 未测（**关键未知**） |
| C3 长期稳定 | 可 | ✗ 反转/NaN | 可（但无对比度） | 模型可 | 待验 |
| C4 ρ/μ 正交 | ✗ 代数锁 | ✗ 实测被 τ 污染 | ✗ 代数锁 | ✔ 机器精度 | ✔ 机器精度 |
| C5 HR | ✗ | ✗ | ✗ | ✗ 无密度对比 | 已跑 −43.46%，可修 |

---

## 7. 反自宣布机制（gate 规则）

1. 每条判据对应一个**机器门脚本**（读工件 JSON，不读人话），输入 schema 固定：
   ```
   {case, model, grid, params{...}, metrics{dp, Req, sigma_meas, u_sp, S, rho_ratio_t,
    lambda_meas, U_meas, U_ref, t, ...}, verdict{pass: bool, reason: str},
    artifacts{...}, sha256{...}}
   ```
2. 实现单只允许写 `verdict.pass=false` 的**自测**结果；`pass=true` 必须由**独立验证 Agent**
   在独立工作树上重跑同一门脚本得出（独立验证单 + gate）。
3. 任何"数字对上了但判据口径被改"的情况：门脚本 diff 非空 ⇒ 自动转评审，
   不得作为通过依据（口径改动只能通过新提案）。
4. 每份通过记录必须含：命令、commit、随机种子/确定性说明、硬件、工件 sha256、以及
   **未通过判据的 honest 列表**（只报通过项 = 报告不完整）。

---

## 8. 与既有锚点的关系

本文件不新立锚点：五条判据是 `GAP-LBM-T1-MULTIPHASE` 伞下的**子判据**，
既有单点锚（MCMP-POISEUILLE 的 ③界面伪流、GAP-PP-2 的 3D Laplace、GAP-PP-3 的伪流标度、
GAP-PP-1 的稳定性）逐条挂靠关系见 [`03_ANCHOR_CONSOLIDATION.md`](03_ANCHOR_CONSOLIDATION.md)。