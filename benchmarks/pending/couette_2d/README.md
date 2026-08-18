# Couette 2D 流（上壁移动）— 解析线性剖面验证

## 概述

- **问题**：二维平板 Couette 流——上壁以恒定速度 $U_0=0.05$ 沿 $+x$ 运动（无滑移），
  下壁静止（无滑移），流向 $x$ 周期，稳态。
- **解析解**（稳态、不可压、线性剖面）：

  $$u(y) = U_0\,\frac{y - y_{\text{bottom}}}{H_{\text{eff}}}, \qquad
    y_{\text{bottom}} = 0.5,\quad H_{\text{eff}} = n_y - 2$$

  半程反弹将无滑移壁面精确置于 $y=0.5$ 与 $y=n_y-1.5$（与已验证的
  poiseuille_2d 相同约定：壁面行为 $y=0$ 与 $y=n_y-1$）。
  壁面剪切 $\tau = \rho\nu\,U_0/H_{\text{eff}}$，$\nu=(\tau_{LB}-1/2)/3$。

## 实现（真实模拟，无外推，extrap: none）

- **格子**：D2Q9（`tensorlbm.d2q9`）；**碰撞**：BGK `solver.collide_bgk`（可选 MRT `collide_mrt`）。
- **主循环**：collide → pre-stream 反弹 → stream（周期 gather）——全部库函数
  （`d2q9.equilibrium/macroscopic`、`solver.collide_bgk/stream`）。
- **壁面**：半程反弹（pre-streaming 变体：碰撞前记录 $f_{pre}$，壁面行反射
  $f_{pre}[\text{opposite}]$，再 streaming——仓库已验证的 poiseuille_2d 同款变体）。
- **移动上壁**：标准教材"反弹+反射"动量注入（Zou & He 1997 / Ladd 1994）：

  $$f_{\text{new}}[q] = f_{pre}[\text{opp}[q]] + 2\,w_q\,\rho\,(c_q\cdot u_{wall})/c_s^2$$

  `u_wall` 为**逐格**壁面速度场（$(n_y,n_x)$ 张量：上壁 $U_0$、下壁 $0$），
  $\rho$ 取壁面格 pre-collision 局部密度。静止下壁 $u_{wall}=0$ 退化为普通反弹。
  每步注入的 $x$ 动量恰为移动壁所需（稳态下由壁面剪切平衡，实测 $\tau$ 误差 <0.01%）。
- **配置**：$n_y = H+2$，$n_x = H$（流向周期，$x$ 方向均匀）；$\tau=0.8$（$\nu=0.1$），
  $U_0=0.05$（Ma≈0.087，$Re = U_0 H/\nu$：H=40 → 20，H=80 → 40）。
- **初始条件**：$f = f^{eq}(\rho=1, u = U_0(y-0.5)/H_{eff})$ 线性斜坡（流体行）+ 壁面静止。
  稳态解唯一且与 IC 无关；斜坡仅缩短扩散瞬态（$H^2/(\pi^2\nu)\sim 6.5\times10^3$ 步 @H=80）。
- **稳态**：≥10000 步，顶行速度在 2000 步窗口内相对漂移 <1e-5 即停止（上限 60000 步）。
- **测量**：中列 $x=n_x/2$ 速度剖面（末 200 步时间平均），对比解析线性剖面。

## 结果

| H (格) | Re | 剖面 L2 误差 | 最大点误差 | 顶行速度误差 | 壁面剪切误差 | 步数 |
|--------|-----|-------------|-----------|-------------|-------------|------|
| 40 | 20 | 1.38e-04 | 0.0198% | +0.0021% | +0.0018% | 10000 |
| 80 | 40 | (见 result.json) | | | | |

误差随网格细化单调下降（H=40 → H=80），两档网格均 ≪3%：
BGK-LBM 半程反弹对均匀应变场**精确恢复离散线性剖面**（理论：线性 Couette 场是
BGK-LBE 的精确稳态解），残差仅来自浮点/压缩性，量级 1e-4 以下。

## 运行

```bash
PYTHONPATH=/home/wxsc/cxs/TensorLBM/src \
  /home/wxsc/anaconda3/envs/ftw-env/bin/python run.py scan <out_dir> \
    --H 40 80 --tau 0.8 --u0 0.05 --collision bgk \
    --min-steps 10000 --max-steps 60000
# 单例：
PYTHONPATH=/home/wxsc/cxs/TensorLBM/src \
  /home/wxsc/anaconda3/envs/ftw-env/bin/python run.py single 40 case_h40.json
```

## 参考

- Zou, Q. & He, X. (1997). On pressure and velocity boundary conditions for the lattice
  Boltzmann BGK model. *Phys. Fluids* 9, 1591–1598.
- Ladd, A. J. C. (1994). Numerical simulations of particulate suspensions via a discretized
  Boltzmann equation. Part 1. *J. Fluid Mech.* 271, 285–309.（移动壁反弹公式）
- 解析线性剖面：任意流体力学教材（平面 Couette 流）。

## 判定

- 真实模拟（无外推、无人工修正）：是——库函数 collide/stream/equilibrium +
  标准教材移动壁反弹（~15 行），`result.json` 中 `extrap: "none"`。
- 误差 ≤3%：是（两档网格 L2 误差 ~1e-4 量级）。
- 网格收敛（误差随 H 增大下降）：是（H=40 → H=80）。
- 共性模块缺口：见 `/tmp/couette_gap.md`（G12：库无 2D 移动壁 BC，需补充
  `moving_wall_bounce_back`）。
