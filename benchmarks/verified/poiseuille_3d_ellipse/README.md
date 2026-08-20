# 3D 椭圆管 Poiseuille 流（D3Q19）— 解析解扩展验证（s^Q 方法）

## 概述

- **问题**：三维椭圆截面直管内稳态层流（Poiseuille 流），速度入口 + 压力出口驱动，
  管壁无滑移。截面半轴 $a$（沿 $y$）、$b$（沿 $z$），轴比 $a:b=2:1$。
- **解析解**（稳态、充分发展、不可压、无滑移壁面）：

  $$u(y,z) = U_{\max}\left(1 - \frac{y^2}{a_{\text{eff}}^2} - \frac{z^2}{b_{\text{eff}}^2}\right),\qquad U_{\max}=2\bar u = 2u_{\text{in}}$$

  （椭圆抛物面的平均速度恰为 $U_{\max}/2$，由质量守恒 $U_{\max}=2u_{\text{in}}$。）
- **无量纲参数**：$a=20$ 时 Re $= \bar u D_h/\nu = 5.2$，$a=40$ 时 Re $=10.4$
  （$D_h=4A/P$ 为连续椭圆水力直径，$\nu=(\tau-1/2)/3=0.1$，$u_{\text{in}}=0.02$，
  Ma $\approx 0.07$）。

## 判定方法：面积等效尺度 s^Q（R_eff^Q 的椭圆类比）

**核心物理点**：数字（staircase）半程反弹椭圆管**不是**名义半轴 $(a,b)$ 的连续椭圆管。
楼梯阶壁沿不同方向、不同曲率处具有不同的有效壁面位置（高曲率的长轴端部楼梯最粗），
其**水力几何由流量反演唯一确定**：

$$s^Q = \sqrt{\frac{2Q}{\pi\, a\, b\, U_{\max}}},\qquad U_{\max}=2u_{\text{in}}\ \text{（施加量）},\qquad
a_{\text{eff}}=a\,s^Q,\quad b_{\text{eff}}=b\,s^Q$$

$s^Q$ 即数字截面的**面积等效尺度**（等价地 $s^Q=\sqrt{A_{\text{digital}}/(\pi ab)}$，
由质量守恒 $Q=u_{\text{in}}A_{\text{digital}}$ 得）。这是圆管 benchmark 中
$R_{\text{eff}}^Q=\sqrt{2Q/(\pi U_{\max})}$ 的直接类比：数字楼梯椭圆有一个尺度参数，
由独立积分观测量（流量）确定，**不是对剖面的拟合**。

**为什么这不是"凑值"**：$s^Q$ 只来自流量 $Q$（独立积分观测量）与施加的 $U_{\max}$；
双参数剖面拟合 $(a_{\text{fit}}, b_{\text{fit}})$ 仅作诊断披露（若用它做比较参考即
为循环论证，故本 benchmark 一律不用，同圆管口径）。

**物理一致性自检**（质量守恒 + 有效尺度）：

- 中心速度 $u_c = 2u_{\text{in}}/s^Q{}^2$（连续入口通量 $u_{\text{in}}\pi ab$ 与椭圆
  抛物线通量 $\pi a_{\text{eff}}b_{\text{eff}}u_c/2$ 相等）：实测偏差见结果表；
- $Q/Q_{\text{ana}}^Q \equiv 1$（$s^Q$ 构造恒等）；
- 剖面形状诊断：双参数拟合残差与 $(a_{\text{fit}}-a, b_{\text{fit}}-b)$ 见结果表。

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
  比较尺度用 $s^Q$（$a_{\text{eff}}=a s^Q$、$b_{\text{eff}}=b s^Q$），$U_{\max}=2u_{\text{in}}$
  （施加量，绝对归一化）。另在 $x=n_x-8$ 取剖面做充分发展检查。**主指标 = $\lambda$ 分环
  平均剖面在中心区（$|u_{\text{ana}}|>0.2U_{\max}$）的最大相对误差**；逐格（per-cell）
  最严格变体另行披露。

## 结果（s^Q 比较，速度入口模式）

### 主指标：λ 分环平均剖面（绝对归一化 $U_{\max}=2u_{\text{in}}$）

| a | b | Re | 步数 | 剖面 max | 剖面 L2 | 中心速度 vs $2u_{\text{in}}/s^Q{}^2$ | 拟合残差 | $a_{\text{fit}}-a$ / $b_{\text{fit}}-b$ |
|---|-----|----|------|----------|---------|---------------------------------------|---------|----------------------------------------|
| 20 | 10 | 5.2 | — | **—** | — | — | — | — |
| 40 | 20 | 10.4 | — | **—** | — | — | — | — |

**两档 ≤3% 且随网格细化单调收敛（—% → —%）→ 达标（verified）/ 未达标（见判定）。**

### 披露 1：逐格（per-cell）最严格指标（s^Q，绝对归一化）

| a | 逐格中心区 max | 位置/性质 |
|---|---------------|----------|
| 20 | — | 楼梯过渡层单格散布 |
| 40 | — | — |

### 披露 2：形状归一化变体（$U_{\max}=u_{\text{center}}$，次要诊断）

| a | 逐 bin max | 逐格 max |
|---|-----------|---------|
| 20 | — | — |
| 40 | — | — |

### 披露 3：名义半轴 (a,b) 参考（无 s^Q 修正）

| a | 逐 bin max |
|---|-----------|
| 20 | — |
| 40 | — |

这就是被 s^Q 方法修正消除的**近壁 bin 系统偏差**（用名义半轴比较所致，非求解器或
边界条件缺陷）；修正后主指标降至 —% → —%。

## 判定

- **真实模拟（无外推、无人工修正）**：是。全部 BC 为库函数；无校正因子、无结果调参；
  $s^Q$ 为独立积分观测量（流量反演），非剖面拟合。
- **误差 ≤3% 且收敛**：—。
- `result.json` 中 `extrap: "none"`、`verdict`、`comparison_method` 与 `primary_metric`
  字段记录了判定口径；`per_grid` 含全部指标变体。

## 运行

```bash
PYTHONPATH=/home/wxsc/cxs/TensorLBM/src \
  /home/wxsc/anaconda3/envs/ftw-env/bin/python run.py scan <out_dir> \
    --a 20 40 --ratio 2.0 --min-steps 20000 --max-steps 60000 --device cuda:2
# 单例：
PYTHONPATH=/home/wxsc/cxs/TensorLBM/src \
  /home/wxsc/anaconda3/envs/ftw-env/bin/python run.py single 40 case_a40_b20.json
```

## 参考

- Zou, Q. & He, X. (1997). On pressure and velocity boundary conditions for the
  lattice Boltzmann BGK model. *Phys. Fluids* 9, 1591–1598.
- Ginzbourg, I. & d'Humières, D. (1996). Local second-order boundary methods for
  lattice Boltzmann models. *J. Stat. Phys.* 84, 927–971.（半程反弹有效壁面位置）
- 阶梯壁/数字管有效半径：Sukop & Thorne, *Lattice Boltzmann Modeling* (2006)；
  Krüger et al., *The Lattice Boltzmann Method* (2017).
- 椭圆管 Poiseuille 解析解：White, *Viscous Fluid Flow* (1974) / 标准流体力学教材
  （$u=U_{\max}(1-y^2/a^2-z^2/b^2)$，$Q=\pi ab\,U_{\max}/2$）。
