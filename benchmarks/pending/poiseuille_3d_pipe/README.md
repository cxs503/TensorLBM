# 3D 圆管 Hagen-Poiseuille 流（D3Q19）— 解析抛物线剖面验证

## 概述

- **问题**：三维圆管内稳态层流（Hagen-Poiseuille 流），速度入口 + 压力出口驱动，管壁无滑移。
- **解析解**（稳态、充分发展、不可压、无滑移壁面）：

  $$u(r) = U_{\max}\left(1 - \frac{r^2}{R_{\text{eff}}^2}\right)$$

  其中 $r$ 为距管轴距离，$U_{\max}=2\bar u$（抛物线平均速度的两倍）。
  速度入口模式（库函数）下 $U_{\max}=2u_{\text{in}}$（质量守恒）。
  $R_{\text{eff}}$ 为圆管有效半径：半程反弹（half-way bounce-back）把无滑移
  壁面置于流体格（$d\le R$）与固体格（$d>R$）连线的中点。对直壁（2D 通道）
  精确为 $R+0.5$；对圆周阶梯（staircase）壁，实测有效半径
  $R_{\text{eff}} \approx R+0.23$（见下文"有效半径"一节）。

- **无量纲参数**：$R=20$ 时 Re $= \bar u D/\nu = 8.2$，$R=40$ 时 Re $=16.2$
  （$D=2R_{\text{eff}}$，$\nu=(\tau-1/2)/3=0.1$，$u_{\text{in}}=0.02$，Ma $\approx 0.07$）。

## 实现（真实模拟，无外推，extrap: none）

- **格子/碰撞/流场**：D3Q19，BGK `solver3d.collide_bgk3d` + `solver3d.stream3d`
  （周期 gather），`d3q19.equilibrium3d` / `macroscopic3d`。
- **几何**：流动沿 $x$；截面 $(y,z)$ 网格 $n_y=n_z=2R+3$，轴心在
  $(y,z)=(R+1,R+1)$；流体格 $d=\sqrt{(y-y_c)^2+(z-z_c)^2}\le R$，固体格
  $d>R$（含全部边界行与四角）。管长 $L=n_x=6R$。
- **边界条件（主模式 velocity，全部库函数）**：
  - 入口 $x=0$：`boundaries3d.zou_he_inlet_velocity_3d`（Zou-He 速度入口，$u_x=u_{\text{in}}$）；
  - 出口 $x=n_x-1$：`boundaries3d.zou_he_outlet_pressure_3d`（Zou-He 压力出口，$\rho_{\text{out}}=1$）；
  - 管壁：`boundaries3d.bounce_back_cells_3d`（post-streaming 半程反弹，作用于全部固体格）。
- **交叉验证模式（pressure）**：压力差驱动 — 入口 $x=0$ 施加
  Zou-He 压力（$\rho_{\text{in}}=1\pm\Delta\rho/2$，$\Delta\rho=4\nu L U_{\max}/(c_s^2 R_{\text{eff}}^2)$，
  手写约 10 行，为库函数 `zou_he_outlet_pressure_3d` 的精确镜像，标准教材公式，
  非外推）；出口与管壁同主模式。
- **主循环**：collide → stream（周期）→ 入口 BC → 出口 BC → 管壁反弹。
- **初始化**：$\rho=1$ + 抛物线剖面 $u_x=U_{\max}(1-(r/R_{\text{eff}})^2)$。
- **稳态**：≥20000 步；测量面 $x=n_x/2$ 的 $u_{\max}$ 在 2000 步窗口内
  相对漂移 $<10^{-5}$ 即停（上限 60000 步）。剖面取末 400 步时间平均。
- **测量**：测量面径向分环（bin $k$：$k\le d<k+1$，$k=0..R$）平均 $u(r)$，
  对比解析抛物线（逐 bin 平均，消除分环偏差）；另在 $x=n_x-8$ 取剖面做
  充分发展检查（相对偏差 <0.7%）。

## 结果

### 主模式：速度入口（库函数 BC）

| R (格) | Re | 步数 | 剖面 L2 误差* | 剖面形状 L2 | 中心速度 vs $2u_{\text{in}}$ | 抛物线拟合残差 | $R_{\text{fit}}-R$ |
|--------|-----|------|--------------|-------------|----------------------------|----------------|-------------------|
| 20 | 8.2 | 20000 | 2.28% | 1.58% | −1.12% | 0.25% | +0.227 |
| 40 | 16.2 | 23600 | 1.20% | 0.69% | −0.81% | 0.13% | +0.238 |

\* 剖面 L2 相对误差：径向平均剖面 vs 解析抛物线（$U_{\max}=2u_{\text{in}}$，
$R_{\text{eff}}=R+0.5$），$\|u_{\text{num}}-u_{\text{ana}}\|_2/\|u_{\text{ana}}\|_2$。

### 交叉验证：压力驱动（Zou-He 压力入口镜像）

| R (格) | 步数 | 入口 ρ 误差 | 中心速度 vs dp 预测 | 剖面 L2 | 抛物线拟合残差 | 质量漂移 |
|--------|------|------------|--------------------|---------|---------------|---------|
| 20 | 20000 | +0.0004% | −2.31% | 3.28% | 0.25% | 0.006% |
| 40 | 22600 | −0.0002% | −1.36% | 1.85% | 0.07% | 0.005% |

### 有效半径（关键物理点）

实测径向剖面与抛物线 $u(r)=A(1-(r/R_{\text{fit}})^2)$ 的加权最小二乘拟合
给出 $R_{\text{fit}}$：

- 速度入口 R=20：$R_{\text{fit}}=20.227$；R=40：$R_{\text{fit}}=40.238$。
- 压力驱动 R=20：$R_{\text{fit}}=20.227$（与速度模式一致到 4 位小数）。

即阶梯（staircase）半程反弹圆管的有效（水力）半径
$R_{\text{eff}} \approx R+0.23$，介于面积半径（$\sqrt{N_{\text{cells}}/\pi}\approx R$）
与壁面中点半径（$R+0.5$）之间，是阶梯几何的固有离散属性（与驱动方式、网格
均无关的几何常数）。**若以实测有效半径比较，剖面最大相对误差仅 0.13–0.24%。**
以先验壁面中点 $R+0.5$ 比较时，外侧 bin（$d\gtrsim 12$）出现系统性
2–7%（R=20）/ 1–4%（R=40）偏差，随网格细化一阶收敛（减半）——这是阶梯
壁离散误差，不是压力 BC 或管壁反弹的缺陷。

## 判定

- **真实模拟（无外推、无人工修正）**：是。全部主模式 BC 为库函数；
  压力入口为库函数出口的教材镜像；无校正因子、无结果调参（$R_{\text{fit}}$
  仅作诊断，两档网格与两种驱动给出同一几何常数）。
- **误差 ≤3% 且收敛**：是。剖面 L2 相对误差 2.28% → 1.20%（速度入口，
  绝对 $U_{\max}$ 参考），随网格细化严格减小；抛物线形状残差 <0.3%；
  中心速度与 $2u_{\text{in}}$ 偏差 ≤1.2%；充分发展检查 <0.7%。
- **备注**：近壁 bin 的最大相对误差（vs 先验 $R+0.5$）为 7.1%（R=20）→
  4.1%（R=40），一阶收敛，源于阶梯壁有效半径偏移（$R_{\text{eff}}=R+0.23$），
  已在结果中完整披露。
- `result.json` 中 `extrap: "none"`。

## 运行

```bash
PYTHONPATH=/home/wxsc/cxs/TensorLBM/src \
  /home/wxsc/anaconda3/envs/ftw-env/bin/python run.py scan <out_dir> \
    --R 20 40 --mode velocity --min-steps 20000 --max-steps 60000 --device cuda:2
# 压力驱动交叉验证：
PYTHONPATH=/home/wxsc/cxs/TensorLBM/src \
  /home/wxsc/anaconda3/envs/ftw-env/bin/python run.py scan <out_dir> \
    --R 20 40 --mode pressure --min-steps 20000 --max-steps 60000 --device cuda:1
# 单例：
PYTHONPATH=/home/wxsc/cxs/TensorLBM/src \
  /home/wxsc/anaconda3/envs/ftw-env/bin/python run.py single 40 case_R40.json
```

## 参考

- Zou, Q. & He, X. (1997). On pressure and velocity boundary conditions for the
  lattice Boltzmann BGK model. *Phys. Fluids* 9, 1591–1598.
- Latt, J. & Chopard, B. (2008). Lattice Boltzmann method with regularized
  pre-collision distribution functions. *Math. Comput. Simul.* (非平衡反弹重构).
- 阶梯壁/半程反弹圆管有效半径讨论：Sukop & Thorne, *Lattice Boltzmann
  Modeling* (2006)；Krüger et al., *The Lattice Boltzmann Method* (2017).
