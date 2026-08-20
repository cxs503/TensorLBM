# 3D 椭圆管 Poiseuille 流（D3Q19）— 解析解扩展验证（s^Q 方法）

**状态：未达标记录（not_verified）**——s^Q 单尺度方法在 a=20 档 5.11% > 3%，
根因为数字（staircase）椭圆壁面的**各向异性水力几何**无法用单一尺度因子表达
（详见"判定"）。仿真本身正确：剖面是精确椭圆抛物面（拟合残差 <0.2%）。

## 概述

- **问题**：三维椭圆截面直管内稳态层流（Poiseuille 流），速度入口 + 压力出口驱动，
  管壁无滑移。截面半轴 $a$（沿 $y$）、$b$（沿 $z$），轴比 $a:b=2:1$。
- **解析解**（稳态、充分发展、不可压、无滑移壁面）：

  $$u(y,z) = U_{\max}\left(1 - \frac{y^2}{a_{\text{eff}}^2} - \frac{z^2}{b_{\text{eff}}^2}\right),\qquad U_{\max}=2\bar u = 2u_{\text{in}}$$

  （椭圆抛物面的平均速度恰为 $U_{\max}/2$，由质量守恒 $U_{\max}=2u_{\text{in}}$。
  标准参考：White, *Viscous Fluid Flow*——$Q=\pi ab\,U_{\max}/2$。）
- **无量纲参数**：$a=20$ 时 Re $= \bar u D_h/\nu = 5.2$，$a=40$ 时 Re $=10.4$
  （$D_h=4A/P$，$P$ 用 Ramanujan 周长近似，$\nu=(\tau-1/2)/3=0.1$，
  $u_{\text{in}}=0.02$，Ma $\approx 0.07$）。

## 判定方法：面积等效尺度 s^Q（R_eff^Q 的椭圆类比）

**核心物理点**：数字（staircase）半程反弹椭圆管**不是**名义半轴 $(a,b)$ 的连续椭圆管。
楼梯阶壁沿不同方向、不同曲率处具有不同的有效壁面位置。按圆管 benchmark 的
$R_{\text{eff}}^Q$ 方法（单参数、独立积分观测量、不拟合剖面）推广：椭圆几何族是
**单尺度**的（$a_{\text{eff}}=a s$、$b_{\text{eff}}=b s$），由流量反演唯一确定：

$$s^Q = \sqrt{\frac{2Q}{\pi\, a\, b\, U_{\max}}},\qquad U_{\max}=2u_{\text{in}}\ \text{（施加量）}$$

比较参考剖面 $u(y,z)=U_{\max}(1-\lambda/s^Q{}^2)$，$\lambda=(y/a)^2+(z/b)^2$。
（注：$s^Q$ 是"数字截面面积等效尺度"——对圆管即 $R_{\text{eff}}^Q=\sqrt{A_{\text{digital}}/\pi}$；
实测 $s^Q$ 与 $A_{\text{digital}}/(\pi ab)$ 的一致性受入口/中段密度差异影响，见下。）

**为什么这不是"凑值"**：$s^Q$ 只来自流量 $Q$（独立积分观测量）与施加的 $U_{\max}$；
双参数剖面拟合 $(a_{\text{fit}}, b_{\text{fit}})$ 仅作诊断披露（若用它做比较参考即
为循环论证，故本 benchmark 一律不用，同圆管口径）。

**物理一致性自检**（质量守恒 + 有效尺度）：

- 中心速度 $u_c = 2u_{\text{in}}/s^Q{}^2$（连续入口通量 $u_{\text{in}}\pi ab$ 与椭圆
  抛物线通量 $\pi a_{\text{eff}}b_{\text{eff}}u_c/2$ 相等）：实测偏差 **+0.23%**（a=20）/
  **+0.11%**（a=40）——质量守恒自洽；
- $Q/Q_{\text{ana}}^Q \equiv 1$（$s^Q$ 构造恒等）；
- 充分发展检查（$x=n_x-8$ vs $x=n_x/2$ 逐 bin 最大偏差）：1.54%（a=20）/ 0.87%（a=40）。

## 实现（真实模拟，无外推，extrap: none）

- **格子/碰撞/流场**：D3Q19，BGK `solver3d.collide_bgk3d` + `solver3d.stream3d`
  （周期 gather），`d3q19.equilibrium3d` / `macroscopic3d`。
- **几何**：流动沿 $x$；截面 $(y,z)$ 网格 $n_y=2a+3$、$n_z=2b+3$（每侧 1 格固体边距，
  同圆管 $2R+3$ 约定），轴心在 $(y,z)=(a+1,b+1)$；流体格
  $\lambda=(y/a)^2+(z/b)^2\le 1$，固体格 $\lambda>1$。管长 $L=n_x=6a$。
- **边界条件（全部库函数）**：
  - 入口 $x=0$：`boundaries3d.zou_he_inlet_velocity_3d`（Zou-He 速度入口，$u_x=u_{\text{in}}$）；
  - 出口 $x=n_x-1$：`boundaries3d.zou_he_outlet_pressure_3d`（Zou-He 压力出口，$\rho_{\text{out}}=1$）；
  - 管壁：`boundaries3d.bounce_back_cells_3d`（post-streaming 半程反弹，作用于全部固体格）。
- **主循环**：collide → stream（周期）→ 入口 BC → 出口 BC → 管壁反弹。
- **初始化**：$\rho=1$ + 椭圆抛物线剖面（仅作初值）。
- **稳态**：≥20000 步；测量面 $x=n_x/2$ 的 $u_{\max}$ 在 2000 步窗口内相对漂移
  $<10^{-5}$ 即停（上限 60000 步）。剖面取末 400 步时间平均。
- **测量与比较**：按 $\lambda=(y/a)^2+(z/b)^2$ 分环（bin $k$：$k\Delta\lambda\le\lambda<(k+1)\Delta\lambda$，
  $\Delta\lambda=1/b$，共 $b$ 个 bin——椭圆抛物面仅依赖 $\lambda$，这是圆管径向分环的
  自然类比），逐 bin 平均 $u(\lambda)$，对比解析剖面（逐 bin 逐格平均，消除分环偏差）；
  比较尺度用 $s^Q$，$U_{\max}=2u_{\text{in}}$（施加量，绝对归一化）。**主指标 = $\lambda$ 分环
  平均剖面在中心区（$|u_{\text{ana}}|>0.2U_{\max}$）的最大相对误差**；逐格（per-cell）
  最严格变体另行披露。

## 结果（s^Q 比较，速度入口模式）

### 主指标：λ 分环平均剖面（绝对归一化 $U_{\max}=2u_{\text{in}}$）

| a | b | Re | 步数 | 剖面 max | 剖面 L2 | 中心速度 vs $2u_{\text{in}}/s^Q{}^2$ | 拟合残差 | $a_{\text{fit}}-a$ / $b_{\text{fit}}-b$ |
|---|-----|----|------|----------|---------|---------------------------------------|---------|----------------------------------------|
| 20 | 10 | 5.2 | 25600 | **5.11%** | 2.02% | +0.23% | 0.19% | +1.18 / −0.15 |
| 40 | 20 | 10.4 | 21200 | **2.43%** | 0.85% | +0.11% | 0.18% | +1.06 / −0.15 |

**两档未全部 ≤3%（a=20 为 5.11%），但随网格细化单调收敛（5.11% → 2.43%）→ 未达标（not_verified）。**

误差位置：a=20 的最大误差在最后一个中心 bin（bin 7，$\lambda\in[0.7,0.8)$，+5.11%）；
核心区 bin 0–6 误差全部 ≤1.9%。a=40 同位置 bin 15 误差 2.43%。

### 披露 1：逐格（per-cell）最严格指标（s^Q，绝对归一化）

| a | 逐格中心区 max | 位置/性质 |
|---|---------------|----------|
| 20 | 15.86% | 楼梯过渡层单格散布（中心 bin 最外沿） |
| 40 | 6.18% | 同上，随细化减半（一阶收敛） |

### 披露 2：形状归一化变体（$U_{\max}=u_{\text{center}}$，次要诊断）

| a | 逐 bin max | 逐格 max |
|---|-----------|---------|
| 20 | 7.24% | 14.15% |
| 40 | 3.36% | 7.15% |

### 披露 3：名义半轴 (a,b) 参考（无 s^Q 修正）

| a | 逐 bin max |
|---|-----------|
| 20 | 11.81% |
| 40 | 5.95% |

这就是被 s^Q 方法修正消除的**近壁 bin 系统偏差**（用名义半轴比较所致，非求解器或
边界条件缺陷）；修正后主指标降至 5.11% → 2.43%。

## 判定（未达标——阶梯椭圆壁面的各向异性）

- **真实模拟（无外推、无人工修正）**：是。全部 BC 为库函数；无校正因子、无结果调参；
  $s^Q$ 为独立积分观测量（流量反演），非剖面拟合。
- **误差 ≤3% 且收敛**：**否**。主指标（λ 分环剖面 max，s^Q 绝对归一化）
  5.11% → 2.43%，**单调收敛**但 a=20 档 5.11% > 3%，未满足"两档均 ≤3%"。
- **根因（记录在案）：数字椭圆壁面是各向异性的，单尺度 s^Q 无法表达其水力几何。**
  - 面积等效尺度实测：$(a_{\text{eff}}-a,\ b_{\text{eff}}-b)=(+0.225,\ +0.113)$（a=20）/
    $(+0.204,\ +0.102)$（a=40）——长轴位移约为短轴的 2 倍，且两档网格都保持该
    各向异性（对比圆管的各向同性 $+0.11$，两档一致）；
  - 双参数剖面拟合（仅诊断）：$(a_{\text{fit}}-a,\ b_{\text{fit}}-b)=(+1.18,\ -0.15)$ /
    $(+1.06,\ -0.15)$——长轴端（单格楼梯"尖端"，曲率半径 $\sim b^2/a=5$ 格）有效壁面
    外凸约 1.1 格，短轴端（近平直壁，曲率半径 $\sim a^2/b=40$ 格）内缩约 0.15 格；
  - 但实测剖面**本身是精确椭圆抛物面**（加权拟合残差 0.19%/0.18%，中心区拟合最大
    误差 0.43%/0.37%），且质量守恒自检 <0.3%——**求解器与边界条件正确**，失败的是
    "单一尺度参数化参考几何"这一比较口径，而非仿真；
  - 收敛性：5.11% → 2.43%（一阶于 1/a，与楼梯壁单格效应一致）；若取更高分辨
    档位（如 a=40/80）单尺度方法可能达标，但按圆管同口径的 a=20/40 两档判定
    标准，本案例如实记为 not_verified。未做任何凑格/调参。
- `result.json` 中 `extrap: "none"`、`verdict: "not_verified"`、`verified: false`、
  `comparison_method`、`primary_metric` 与 `notes`（含各向异性诊断）字段记录了判定
  口径；`per_grid` 含全部指标变体。

**给后续工作的提示**：若需椭圆管 ≥3% 达标，需改用**两观测量**水力反演
（流量 $Q$ + 压降 $\Delta p$ 联合确定 $(a_{\text{eff}}, b_{\text{eff}})$，均为独立
积分观测量、非剖面拟合）或曲率相关的局部壁面位置修正——本记录未采用，因为它超出
"R_eff^Q 类似方法（流量反演有效轴长）"的任务口径。

## 运行

```bash
PYTHONPATH=/home/wxsc/cxs/TensorLBM/src \
  /home/wxsc/anaconda3/envs/ftw-env/bin/python run.py scan <out_dir> \
    --a 20 40 --ratio 2.0 --min-steps 20000 --max-steps 60000 --device cuda:2
# 单例：
PYTHONPATH=/home/wxsc/cxs/TensorLBM/src \
  /home/wxsc/anaconda3/envs/ftw-env/bin/python run.py single 40 case_a40_b20.json
```

注：本目录的 `case_a20_b10.json` / `case_a40_b20.json` 为实测数据；`result.json` 由
`run.build_summary` 从两档 case 聚合生成（`python -c "import json; from run import
build_summary; build_summary([json.load(open(f'case_a{a}_b{b}.json')) for a,b in
[(20,10),(40,20)]], [20,40], 2.0, 0.8, 0.02, 20000, 60000, '.')"`，与 `scan` 命令
输出完全一致）。

## 参考

- Zou, Q. & He, X. (1997). On pressure and velocity boundary conditions for the
  lattice Boltzmann BGK model. *Phys. Fluids* 9, 1591–1598.
- Ginzbourg, I. & d'Humières, D. (1996). Local second-order boundary methods for
  lattice Boltzmann models. *J. Stat. Phys.* 84, 927–971.（半程反弹有效壁面位置）
- 阶梯壁/数字管有效半径：Sukop & Thorne, *Lattice Boltzmann Modeling* (2006)；
  Krüger et al., *The Lattice Boltzmann Method* (2017).
- 椭圆管 Poiseuille 解析解：White, *Viscous Fluid Flow* (1974)（或标准流体力学教材：
  $u=U_{\max}(1-y^2/a^2-z^2/b^2)$，$Q=\pi ab\,U_{\max}/2$；Ramanujan 椭圆周长近似）。
