# 环形 Taylor-Couette 层流（D2Q9，2D r-θ 平面）— 解析剖面 + 壁面力矩验证（R_eff 剖面反演方法）

## 概述

- **问题**：同心圆柱环隙内的稳态层流（环形 Couette 流），内柱以 $\omega_i$ 旋转，
  外柱静止，半径比 $\eta=r_i/r_o=0.5$，轴向周期（2D r-θ 平面是精确表述：层流解
  无轴向依赖）。**新问题族**：旋转流（非直线驱动），库内 verified 目录中原无同类。
- **解析解**（稳态、轴对称、无滑移壁面；精确解，无需文献数据）：

$$u_\theta(r) = A\,r + \frac{B}{r},\qquad
A=\frac{\Omega_o r_o^2-\Omega_i r_i^2}{r_o^2-r_i^2},\quad
B=\frac{r_i^2 r_o^2(\Omega_i-\Omega_o)}{r_o^2-r_i^2}\ \ (\Omega_o=0)$$

  壁面力矩（单位轴向长度，两壁等值反号，交叉验证观测量）：

$$M = 4\pi\mu L\,B,\qquad \mu=\rho_0\nu,\ L=1\ (\text{2D 每单位深度格单位})$$

- **无量纲参数**：$Re_i=\omega_i r_i^2/\nu = 20 \ll Re_{i,\text{crit}}(\eta{=}0.5)\approx 68.2$
  （Esser & Grossmann 1996；$\eta=0.5$ 时 gap 基 Re 与 $Re_i$ 相等），深居层流
  轴对称稳定区，无 Taylor 涡。$\nu=0.05$（$\tau=0.65$），壁速
  $u_{\text{wall}}=0.0417/0.0208/0.0104$，Ma $=0.072/0.036/0.018$。

## 判定方法：有效阶梯半径 R_eff（剖面反演，移植自 poiseuille_3d_pipe 的 R_eff^Q）

**核心物理点（与 pipe 先例同一逻辑）**：数字（staircase）反弹环隙**不是**名义半径
$(r_i,r_o)$ 的连续环隙。楼梯阶壁沿不同方向的链接具有不同的有效壁面位置，名义半径
做参考即循环论证/系统偏差。pipe 先例用**流量反演**得 $R_{\text{eff}}^Q$；环隙无流量
可观测量，故把反演移植到**剖面本身**（唯一可观测量）：

1. 测量方位角平均剖面 $u_\theta(r)$（径向 bin $k$：$k\le d<k+1$，逐 bin 逐格平均）；
2. 在中心区（用名义解析解选区 $u_{\text{nom}}>0.2\,u_{\text{wall}}$）对**精确 Couette
   解族** $u=Ar+B/r$ 做加权（格子数）二参数最小二乘拟合 → $(A,B)$；
3. 由拟合解的**无滑移条件反演有效半径**（$\omega_i$ 是施加量，不是拟合量）：

$$r_{o,\text{eff}}=\sqrt{-B/A}\ \ (u_\theta=0\ \text{零速条件}),\qquad
r_{i,\text{eff}}=\sqrt{B/(\omega_i-A)}\ \ (u_\theta=\omega_i r\ \text{无滑移条件})$$

4. 偏移量 $\delta_i=r_{i,\text{eff}}-r_i$、$\delta_o=r_{o,\text{eff}}-r_o$ 必须是
   **格无关的格单位常数**（pipe 先例 $R+0.11$ 的环隙对应物）；
5. **主指标**：逐 bin 剖面对**有效几何**解析剖面 $u_{\text{ref}}=Ar+B/r$ 的误差，
   绝对归一化 $U_{\max}^{\text{ref}}=\omega_i r_{i,\text{eff}}$（有效壁上施加的壁速；
   恒等式 $A r_{i,\text{eff}}+B/r_{i,\text{eff}}=\omega_i r_{i,\text{eff}}$ 实测符合到
   $10^{-16}$），中心区（$u_{\text{ref}}>0.2U_{\max}^{\text{ref}}$）max 与加权 L2；
6. **独立交叉验证通道——壁面力矩** $M_{\text{ref}}=4\pi\nu B$：力矩在边界动量交换
   上测量，与剖面拟合是完全不同的可观测量，且精确解要求两壁力矩等值反号（牛顿第
   三律自检）。

**为什么这不是"凑值"**：$(A,B)$ 来自对**精确解函数族**的全剖面拟合（17/35/69 个
bin），不是逐点调参；有效半径由无滑移条件从 $(A,B)$ **唯一确定**；力矩通道完全
独立于拟合。名义半径比较另行**披露**（展示 R_eff 方法消除的偏差）。

### 力矩测量通道（约定无关的界面通量记账）

主通道为**界面通量**：对每条边界链接（流体格 $x_f$、指向固体的方向 $q$），真实
离场种群 $g[q,x_s] c_q$（$x_s$ 为固体端点，post-stream 图像）离开流体、真实回场
种群 $g[\bar q,x_f] c_{\bar q}$ 到达流体 —— 取链接中点力臂，对 z 力矩求和。这是
**纯测量**（不假设反射模型），对运动壁/静止壁对称成立。次通道：静止外壁上的
full-way（湿格）plain MEM（`compute_obstacle_forces` 约定 $F_\alpha=2\sum_q c_{q\alpha}f_q$
的逐格推广 + 力臂）。库原语 `d3q27.moving_wall_linkwise_me_force_torque`
（理想半程反射模型）在本方案上系统性高估约 2 倍，作为**约定失配披露**（见披露 5）。

## 实现（真实模拟，无外推，extrap: none）

- **格子/碰撞**：D2Q9，BGK `solver.collide_bgk` + `solver.stream`（周期 gather），
  `d2q9.equilibrium` / `d2q9.macroscopic`。
- **几何**：方域 $n=2r_o+3$，圆心 $((n{-}1)/2,(n{-}1)/2)$；流体 $=\{r_i<d\le r_o\}$
  （`cylinder_mask`：内盘 $d\le r_i$、外部 $d>r_o$）；全域边界均为固体，周期流进
  流出只连接固体-固体。
- **边界条件（全部库函数）**：
  - 内柱（整个障碍掩码，与 `run_rotating_cylinder` 用法一致）：
    `rotating_cylinder.moving_wall_bounce_back`（Ladd 1994 运动壁反弹，
    $f_{\bar i}=f_i-2w_i\rho(c_i\cdot u_w)/c_s^2$），壁速场由
    `rotating_cylinder.rotating_wall_velocity` 给出 $u_w=\omega_i\times r$；
  - 外柱（静止）：`boundaries.bounce_back_cells`（post-streaming 反弹）。
- **主循环**：collide（全格）→ stream（周期）→ 内柱运动壁反弹 → 外柱反弹。
- **初始化**：$\rho=1$ 静止（真实 spin-up 暂态，不用解析解初始化）。
- **稳态**：$u_{\max}$ 漂移（最近 10 个采样点，间隔 200 步）$<10^{-5}$，最小
  $3.5\,(r_o{-}r_i)^2/\nu$ 步、上限 $10\times$；剖面与力矩取末 **1000 步**时间平均。
- **加速**：`compile_route.route_step`（共享封装，全步函数 torch.compile，
  监控保持 eager）。

## 结果

### 网格阶梯（$\eta=0.5$，$Re_i=20$，$\nu=0.05$，$\tau=0.65$）

| $r_i$ | $r_o$ | $n$ | $\omega_i$ | Ma | 步数 | 稳态 | 流体格 |
|-------|-------|-----|-----------|-----|------|------|--------|
| 24 | 48 | 99 | 1.7361e-3 | 0.072 | 40 400 | 是 | 5 420 |
| 48 | 96 | 195 | 4.3403e-4 | 0.036 | 161 400 | 是 | 21 704 |
| 96 | 192 | 387 | 1.0851e-4 | 0.018 | 645 200 | 是 | 86 864 |

### 主指标：方位角平均逐 bin 剖面（有效几何，绝对归一化 $U_{\max}^{\text{ref}}=\omega_i r_{i,\text{eff}}$）

| $r_i$ | 剖面 max | 剖面 L2 | 逐格 max（披露） | 名义半径 max（披露） | 中点 $u$ vs 有效（披露） |
|-------|----------|---------|------------------|---------------------|------------------------|
| 24 | **0.746%** | 0.303% | 2.876% | 2.133% | −0.008% |
| 48 | **0.363%** | 0.102% | 1.851% | 0.974% | −0.005% |
| 96 | **0.152%** | 0.035% | 1.244% | 0.435% | −0.023% |

**三档 ≤3% 且随网格细化单调收敛（0.746% → 0.363% → 0.152%，近似一阶）→ 达标。**
全部变体（L2、形状归一化、逐格、名义）同步单调收敛。

### 几何常数：有效半径偏移（格单位）

| $r_i$ | $r_{i,\text{eff}}$ | $r_{o,\text{eff}}$ | $\delta_i$ | $\delta_o$ | $\delta_i$（全 bin 拟合） | $\delta_o$（全 bin 拟合） | $\eta_{\text{eff}}$ |
|-------|--------------------|--------------------|-----------|-----------|--------------------------|--------------------------|---------------------|
| 24 | 23.871 | 47.959 | −0.129 | −0.041 | −0.128 | −0.049 | 0.4977 |
| 48 | 47.888 | 95.961 | −0.112 | −0.039 | −0.112 | −0.039 | 0.4990 |
| 96 | 95.898 | 191.864 | −0.102 | −0.136 | −0.115 | −0.061 | 0.4998 |

$\delta_i$ 三档散布 **0.027** 格（两档间一致到 0.017）；$\delta_o$ 中心区拟合散布
0.097、全 bin 拟合散布 **0.022** 格 —— 均为格无关常数（容差 0.25 格），量级与
pipe 先例的 $R+0.11$ 相当（内壁为负偏移、绝对值略大，与运动壁反弹/Ladd 修正的
链接位置一致）。$\delta_o$ 在最细档的两种选区之差见披露 6。

**参数不变性辅证**（$r_i=24$，$n=99$）：$\delta_i$ 在 $Re_i=20\to30$（同 $\tau=0.65$）
仅移动 0.001（−0.129 → −0.130），即与驱动强度无关；$\nu=0.05\to0.08$
（$\tau\,0.65\to0.74$）时 $\delta_i$ 移动 −0.038（−0.129 → −0.167）——半程反弹
有效壁面位置的 $\tau$ 依赖是文献已知效应（Ginzbourg & d'Humières 1996），正是
R_eff 必须**从流场反演**而不能先验假定的重要原因（本阶梯固定 $\tau=0.65$，反演
常数格无关）。

| 辅助工况 | $\tau$ | $\delta_i$ | $\delta_o$ | 剖面 max / L2 | 力矩内壁 | 平衡 |
|----------|--------|-----------|-----------|---------------|----------|------|
| $Re_i=30,\nu=0.05$ | 0.65 | −0.130 | −0.058 | 0.692% / 0.286% | +0.124% | −0.001% |
| $Re_i=20,\nu=0.08$ | 0.74 | −0.167 | −0.041 | 0.542% / 0.227% | +0.053% | +0.004% |

### 交叉验证：壁面力矩 vs $M_{\text{ref}}=4\pi\nu B$

| $r_i$ | $M_{\text{ref}}$ | 界面通量 内壁 | 界面通量 外壁 | 两壁平衡 | full-way MEM 外壁（静止壁） |
|-------|------------------|---------------|---------------|----------|----------------------------|
| 24 | 0.82629 | +0.185% | −0.189% | −0.004% | −0.305% |
| 48 | 0.83279 | +0.203% | −0.159% | +0.043% | −0.148% |
| 96 | 0.83577 | +0.727% | −0.495% | +0.232% | +0.149% |

- **全部 ≤3%**；两壁力矩等值反号（牛顿第三律）到 **≤0.23%**；
- 恒等式 $4\pi\nu B(r_{i,\text{eff}},r_{o,\text{eff}},\omega_i)=4\pi\nu B_{\text{fit}}$
  符合到 $10^{-14}$（有效半径反演自洽）；
- 力矩是独立于剖面拟合的可观测量（边界动量交换），其 ≤0.73% 误差与剖面误差
  同量级、同方向单调 —— 物理一致；
- 内柱净力 $\sim10^{-5}$–$10^{-3}\,M$（纯旋转对称，应无净力；量级与 float32
  噪声一致）。

## 披露（不构成判定项）

### 披露 1：逐格（per-cell）最严格指标

2.876% → 1.851% → 1.244%（最差格位于 $r\approx24.04/48.09/96.08$，内壁楼梯
过渡层）。性质与 pipe 先例相同：同半径格子因楼梯阶壁局部链接构型不同存在单格
散布，是 staircase 离散固有的 $O(1/r_i)$ 单格几何效应，径向（+方位角）平均后
消失，随细化单调收敛。

### 披露 2：名义半径比较（R_eff 方法消除的偏差）

2.133% → 0.974% → 0.435%（max）；中点速度 vs 名义解析 −1.61% → −0.71% →
−0.45%。用名义半径做参考会把阶梯壁位置偏差（$\delta\approx-0.1$ 格）计入剖面
误差 —— 修正后（vs 有效几何）中点误差降至 ≤0.023%。这是参考系修正，不是求解器
或边界条件缺陷。

### 披露 3：形状归一化变体（次要诊断）

1.014% → 0.467% → 0.212%（逐 bin max，参考按最内 bin 幅度重标定）——同样单调
收敛，说明剖面形状与幅度误差一致。

### 披露 4：质量漂移（float32 累积）

0.043% / 0.122% / 0.491%（645 200 步累积，未做任何质量校正 —— 校正属于人工
修正，铁律禁止）。漂移是单调累积不是发散；Couette 解在不可压极限与 $\rho$ 水平
无关，剖面误差比漂移小一个量级以上且单调收敛，故污染居次。内柱净力随漂移同步
增长（$2\times10^{-3}M$ 于最细档）亦与此一致。

### 披露 5：库 linkwise 运动壁 MEM 原语的约定失配（gap 记录，非库缺陷）

`d3q27.moving_wall_linkwise_me_force_torque`（理想半程反射模型
$f_r=f_o-2\rho w(c\cdot u_w)/c_s^2$）在本方案的环隙上给出 $T/M_{\text{ref}}=
2.06/2.14/2.14$（内壁，$\tau=0.65$；外壁 −2.03/−1.95/−1.91；$\tau=0.74$ 时降为
1.69）。**症状**：约 2 倍系统性高估。**根因**：该原语假设回场种群是离场种群的
理想镜像，而本方案 collide 在全部格子（含固体格）上运行、反弹 swap 在 post-stream
进行，回场种群实为被 BGK 弛豫过的镜像（内盘固体格宏观场实测携带 $\omega\times r$
刚体旋转，其平衡分布替换了理想反射内容）；理想反射模型只对"固体格上以反弹替代
碰撞"的方案精确。**倍数随 $\tau$ 漂移**（0.65 → 0.74 时 2.06 → 1.69）与碰撞污染
机制一致。**建议**（供库维护者参考）：在原语 docstring 标注适用方案（固体格不参与
碰撞/反弹即碰撞）；需要约定无关力矩时可用本 benchmark 的界面通量记账（对实际
离场 + 实际回场种群取链接中点力臂，运动壁/静止壁对称成立，实测精度 ≤0.73%、
两壁平衡 ≤0.23%）。本 benchmark 判定不使用该原语。

### 披露 6：$\delta_o$ 反演在最细档的选区敏感度

$r_{o,\text{eff}}=\sqrt{-B/A}$ 中 $|A|\propto 1/r_i^2$（最细档 $|A|=3.6\times10^{-5}$），
外区 $u\to0$ 对 $A$ 的约束弱，残余 bin 散度被放大进 $r_{o,\text{eff}}$：中心区拟合
与全 bin 拟合在 $r_i=96$ 相差 0.075 格（粗档仅 0.008/0.010 格）。两种选区给出的
$\delta_o$ 均在 $[-0.14,-0.03]$ 内、格无关性均在容差内（散布 0.097 / 0.022 格），
但最细档的 $\delta_o$ 应视为反演噪声 ±0.04 格量级。$\delta_i$（由强信号无滑移条件
反演）无此问题（两选区之差 ≤0.013 格）。

## 判定

- **真实模拟（无外推、无人工修正）**：是。全部 BC 为库函数；从静止 spin-up；
  无校正因子、无结果调参；$R_{\text{eff}}$ 由剖面拟合 + 无滑移条件反演（非逐点
  拟合参考），力矩通道独立于拟合。`extrap: "none"`。
- **误差 ≤3% 且收敛**：是。主指标三档 0.746% → 0.363% → 0.152%（max），
  0.303% → 0.102% → 0.035%（L2），单调收敛；力矩交叉验证全部 ≤0.73%、两壁平衡
  ≤0.23%；几何常数 $\delta_i/\delta_o$ 格无关（散布 0.027/0.022 格，全 bin 拟合）。
- **铁律 1（公共入口）**：run.py 只做胶水（几何装配、bin/拟合分析、力矩记账），
  无手写 collide/stream/equilibrium/bounce/zou_he/far_field（提交前 grep 自检
  通过：`def (collide|stream|equilibrium|bounce|zou_he|far_field)` 零匹配）。
- `result.json` 记录 `verdict: "verified"`、`passed_3pct_and_converged: true`、
  `err_decreased: true`、`geometry_constants_grid_independent: true`、
  `torque_crosscheck.all_within_3pct: true`，`per_grid` 含全部指标变体。

## 入口声明（全部公共库函数 + 共享封装）

| 模块 | 函数 | 用途 |
|------|------|------|
| `tensorlbm.rotating_cylinder` | `rotating_wall_velocity` | 旋转壁速场 $u_w=\omega\times r$ |
| `tensorlbm.rotating_cylinder` | `moving_wall_bounce_back` | Ladd 运动壁反弹（内柱） |
| `tensorlbm.solver` | `collide_bgk` / `stream` | D2Q9 BGK 碰撞 / 周期流动 |
| `tensorlbm.d2q9` | `equilibrium` / `macroscopic` | 平衡分布 / 宏观量 |
| `tensorlbm.boundaries` | `cylinder_mask` / `bounce_back_cells` / `compute_obstacle_forces` | 几何 / 静止壁反弹 / 净力自检 |
| `tensorlbm.d3q27` | `moving_wall_linkwise_me_force_torque` | 力矩诊断（披露变体，见披露 5） |
| `benchmarks/compile_route` | `route_step` 等 | torch.compile 共享封装 |

## 工件

- `run.py` — 本 benchmark（single / scan / aggregate 三子命令）；
- `case_ri24.json` / `case_ri48.json` / `case_ri96.json` — 三档完整指标（含逐 bin 剖面）；
- `result.json` — 汇总（收敛、几何常数、力矩交叉验证、判定）；
- `log_ri24.txt` / `log_ri48.txt` / `log_ri96.txt` — 各档运行日志（含稳态判定行）；
- `aux_ri24_re30.json` / `aux_ri24_nu08.json`（+ 对应 log）— 参数不变性辅助工况。

## 运行

```bash
PYTHONPATH=/nfs/wangxi/worktrees/bm_relaunch/src:/nfs/wangxi/worktrees/bm_relaunch/benchmarks \
  CUDA_VISIBLE_DEVICES=6 /nfs/wangxi/venvs/tensorlbm/bin/python run.py scan \
    --out-dir /nfs/wangxi/runs/bm_widen_20260919/taylor_couette --ri 24 48 96
# 单例：
... run.py single --ri 24 --out case_ri24.json
# 辅助不变性：
... run.py single --ri 24 --re-i 30 --out aux_ri24_re30.json
... run.py single --ri 24 --nu 0.08 --out aux_ri24_nu08.json
# 汇总（从 case json 重建 result.json）：
... run.py aggregate --out-dir /nfs/wangxi/runs/bm_widen_20260919/taylor_couette --ri 24 48 96
```

（run.py 已内置上述两个 sys.path 插入，可直接运行。）

## 参考

- 解析解 / 环隙 Couette 流：教科书标准结果（e.g. Kundu & Cohen, *Fluid Mechanics*）。
- Ladd, A.J.C. (1994). Numerical simulations of particulate suspensions via a
  discretized Boltzmann equation. Part I. *J. Fluid Mech.* 271, 285–309.
  （运动壁反弹修正 $f_{\bar i}=f_i-2w_i\rho(c_i\cdot u_w)/c_s^2$）
- Esser, A. & Grossmann, S. (1996). Analytic expression for Taylor-Couette
  stability boundary. *Phys. Fluids* 8, 1814–1819.（$Re_{i,\text{crit}}(\eta{=}0.5)\approx68.2$）
- Ginzbourg, I. & d'Humières, D. (1996). Local second-order boundary methods
  for lattice Boltzmann models. *J. Stat. Phys.* 84, 927–971.（半程反弹有效壁面
  位置及其 $\tau$ 依赖）
- 阶梯壁/数字圆柱有效半径：Sukop & Thorne, *Lattice Boltzmann Modeling* (2006)；
  Krüger et al., *The Lattice Boltzmann Method* (2017).
- R_eff^Q 方法学先例：`benchmarks/verified/poiseuille_3d_pipe/`（本目录同款逻辑）。
