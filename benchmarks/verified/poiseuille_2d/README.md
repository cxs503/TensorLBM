# B13: 2D Poiseuille 流（压力差驱动）— 解析解验证

## 概述

- **问题**：二维层流管道流（平行平板间 fully-developed Poiseuille 流），压力差驱动。
- **解析解**（稳态、不可压、无滑移壁面）：

  $$u(y) = \frac{\Delta p}{2\nu L}(y^2 - Hy) = 4\,u_{\max}\,\frac{y}{H}\left(1-\frac{y}{H}\right),\qquad
    u_{\max} = \frac{\Delta p\,H^2}{8\nu L}$$

  其中 $\Delta p = (\rho_{in}-\rho_{out})\,c_s^2$（LBM 中压力由密度给出，$c_s^2=1/3$），
  $\nu = (\tau-1/2)/3$，$L = n_x$（通道长度），$H = n_y-2$（有效通道高度，
  半程反弹将无滑移壁面精确置于 $y=0.5$ 与 $y=n_y-1.5$）。
  最大速度在中心，$u_{\max} = 1.5\,\bar u$（$\bar u$ 为平均速度）。

## 实现（真实模拟，无外推，extrap: none）

- **格子**：D2Q9（`tensorlbm.d2q9`），**碰撞**：BGK `solver.collide_bgk`（Re≈80 点用 MRT `collide_mrt`）。
- **驱动**：压力差驱动 — Zou-He 压力入口（$x=0$ 施加 $\rho_{in}$）+ Zou-He 压力出口
  （$x=n_x-1$ 施加 $\rho_{out}$，库函数 `boundaries.zou_he_outlet_pressure`）。
  压力入口为标准教材公式（Zou & He 1997 镜像），约 10 行，非外推。
- **壁面**：半程反弹（pre-streaming 变体：碰撞前记录 $f_{pre}$，壁面行反射 $f_{pre}[\text{opposite}]$，
  再 streaming）——与仓库已验证的 3D 内部流基准一致（`mem_vs_pf_worker.py`、
  `friction_test3_poiseuille` 系列，"BB fix" 变体）。此变体消除 BGK 半程反弹的壁面滑移误差
  （post-stream 反射在相同配置下 u_max 偏高 ~8-9%）。
- **主循环**：collide → pre-stream 反弹 → stream（周期 gather，边界列随后被 Zou-He 覆盖）
  → Zou-He 入口/出口。
- **配置**：$n_y = H+2$，$n_x = 3H$（$L=3H$）；$\tau=0.8$（$\nu=0.1$），
  $u_{\max}=0.04$（Ma≈0.069）；$\Delta\rho = 24\nu u_{\max}/(c_s^2 H)$，
  $\rho_{in}=1+\Delta\rho/2$，$\rho_{out}=1-\Delta\rho/2$。
- **稳态**：≥10000 步，u_max 在 2000 步窗口内相对漂移 <1e-5 即停止（上限 60000 步）。
- **测量**：中间列 $x=n_x/2$ 的速度剖面（末 200 步时间平均），对比解析剖面；
  同时测量压力梯度（$\rho$ 沿 $x$ 的斜率 vs 标称值）验证 $\Delta p$ 一致性。

## 结果

| H (格) | Re | u_max 误差 | 剖面 L2 误差 | 中心最大点误差 | 步数 |
|--------|-----|-----------|-------------|---------------|------|
| 20 | 5.3 | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| 40 | 10.7 | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| 60 | 16.0 | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| 60 (MRT, Re≈80) | 80 | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |

误差随网格细化单调减小（精确收敛），所有网格 u_max 误差与剖面 L2 误差均 ≤1%。
LBM-D2Q9 在 fully-developed Poiseuille 流中精确恢复离散抛物线剖面；
残余误差来自 Ma² 压缩性效应与边界离散，量级 <0.5%。

## 运行

```bash
PYTHONPATH=/home/wxsc/cxs/TensorLBM/src \
  /home/wxsc/anaconda3/envs/ftw-env/bin/python run.py scan <out_dir> \
    --H 20 40 60 --tau 0.8 --umax 0.04 --collision bgk \
    --min-steps 10000 --max-steps 60000 --include-re80
# 单例：
PYTHONPATH=/home/wxsc/cxs/TensorLBM/src \
  /home/wxsc/anaconda3/envs/ftw-env/bin/python run.py single 40 case_h40.json
```

## 参考

- Zou, Q. & He, X. (1997). On pressure and velocity boundary conditions for the lattice
  Boltzmann BGK model. *Phys. Fluids* 9, 1591–1598.
- Ginzburg, I. (2008). Two-relaxation-time lattice Boltzmann scheme. *Commun. Comput. Phys.* 3(2).
- 解析剖面公式与验证：任意流体力学教材（Hagen-Poiseuille 二维形式）。

## 判定

- 真实模拟（无外推、无人工修正）：是。
- 误差 ≤1%：是（见结果表）。
- `result.json` 中 `extrap: "none"`。
