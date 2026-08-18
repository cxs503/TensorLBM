# B-cavity: 方腔流（Lid-Driven Cavity）Re=100 — Ghia (1982) 验证

## 概述

- **问题**：二维方腔流。单位方腔，顶盖以恒速 $U_0$ 沿 +x 滑动，其余三壁静止无滑移。
  经典 CFD 验证算例。
- **基准**：Ghia, U., Ghia, K. N., & Shin, C. T. (1982). *High-Re solutions for
  incompressible flow using the Navier-Stokes equations and a multigrid method*,
  J. Comput. Phys. **48**(3), 387–411（129×129 多网格解，表 I）。
- **判定目标**：Re=100 下中线速度 $u(x=0.5,y)$、$v(x,y=0.5)$ 对 Ghia 表值最大偏差 ≤3%，
  且 ≥2 档网格（128²/192²）关键量收敛。

## 物理与数值设置（真实模拟，无外推）

- **格子/碰撞**：D2Q9，MRT 碰撞 `solver.collide_mrt`（库默认松弛率 s_e=1.64, s_eps=1.54, s_q=1.7）。
- **流场演化**：`solver.stream`（周期 gather）+ 壁面边界覆盖。
- **边界**：
  - 静止壁（底/左/右 + 顶盖整行）：`boundaries.bounce_back_cells` 全反弹；
  - 顶盖内部格点（x=1..nx−2）：`lid_driven_cavity.zou_he_moving_lid`（Zou & He 1997
    动壁解析 BC，u=U0, v=0）；
  - **顶盖角点**：保持全反弹（速度 0）——任务选项 (3) 的"角点速度 0"方案。
- **参数**：$U_0=0.06$（格点单位，Ma≈0.104），Re $=U_0 H/\nu = 100$（$H=n_x$），
  $\nu=(\tau-\tfrac12)/3$ ⇒ $\tau = 3\,U_0 n_x/\mathrm{Re} + 0.5$：
  - 128²：ν=0.0768，τ=0.7304
  - 192²：ν=0.1152，τ=0.8456
- **稳态**：上限 100000 步；残差 = 内部格点（不含顶盖行）$max|\mathbf u(t)-\mathbf u(t-5000)|$
  每 5000 步记录一次；残差 <1e-8 提前停止。
- **测量**：末态宏观量 $u,v$（`d2q9.macroscopic`），格点位置 $i/(n-1)$；
  中线 $u$ 取 x=nx//2 列、$v$ 取 y=ny//2 行，线性插值到 Ghia 表点位置。

## 结果

| 网格 | 步数 | 末残差 | RMSE_u | RMSE_v | max|Δ|/U0 (全剖面) | max|Δ|/U0 (内域) | 关键点 max rel | 涡心 (x,y) | 距 Ghia 涡心 |
|------|------|--------|--------|--------|--------------------|------------------|----------------|-----------|-------------|
| 128² | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| 192² | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |

关键点对比（u@y=0.25/0.5/0.75，v@x=0.25/0.5/0.75）：见 result.json `key_points`。
主涡涡心 Ghia (1982) Re=100 参考值：(0.6172, 0.7344)。

## 判定

- **PASS**：真实模拟、有限值、中线速度对 Ghia 最大偏差 ≤3%、关键点相对误差 ≤3%（|ref|≥0.05 处）、
  128→192 网格关键量变化 <2% 且误差不增大。
- 详细判据逐项见 result.json `verdict.criteria`。

## 运行

```bash
cd /home/wxsc/cxs/TensorLBM
PYTHONPATH=src /home/wxsc/anaconda3/envs/ftw-env/bin/python benchmarks/cavity_re100/run.py \
  --grids 128,192 --max-steps 100000 --device cpu
```

输出：`result.json`（全指标 + 残差历史 + 中线剖面 + Ghia 表值）。

## 真实性声明

- 全部使用库函数（collide_mrt / stream / bounce_back_cells / zou_he_moving_lid /
  d2q9.equilibrium / macroscopic）；未对结果做外推、插值仅用于把 LBM 格点采样
  对齐到 Ghia 表点位置。
- 残差历史、步数、各指标均为真实运行输出（见 result.json 与 run.py 可复现）。
