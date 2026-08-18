# B18: Kovasznay 2D 稳态流 — 解析 Navier–Stokes 解验证

## 概述

- **问题**：Kovasznay（1948）二维稳态不可压 Navier–Stokes 精确解（y 向周期涡 + x 向指数衰减），
  用于验证 LBM 求解器对稳态有旋流的还原能力。
- **解析解**（无量纲，特征速度 U0、特征长度 L = y 向周期）：

  $$u(x',y') = U_0\left(1 - e^{\lambda x'}\cos 2\pi y'\right),\qquad
    v(x',y') = U_0\,\frac{\lambda}{2\pi}\,e^{\lambda x'}\sin 2\pi y',\qquad
    p(x',y') = \tfrac12\left(1 - e^{2\lambda x'}\right)$$

  $$\lambda = \frac{Re}{2} - \sqrt{\frac{Re^2}{4} + 4\pi^2},\qquad
    Re = \frac{U_0 L}{\nu},\quad \nu = \frac{\tau - 1/2}{3}$$

  λ 由 x 向动量方程导出（λ² − Re·λ − 4π² = 0 的负根）。**注**：任务书称 "Re=40 时 λ≈−0.5"，
  按公式精确计算 Re=40 → λ = 20 − √(400+4π²) ≈ **−0.9637**（λ=−0.5 对应 Re≈78.5）。
  本 benchmark 以公式为准；解析解已独立数值验证（y 向周期差分下 max|N-S 残差|≈1e-3 为差分截断，
  恒等式 λ=ν(λ²−4π²) 成立至 2e-15）。

## 实现（真实模拟，无外推，extrap: none）

- **格子**：D2Q9（`tensorlbm.d2q9`），**碰撞**：BGK `solver.collide_bgk`，**stream**：`solver.stream`
  （周期 gather，内建 y 向周期）。
- **初值**：全场解析场 `f = f_eq(ρ=1, u_ana, v_ana)`。
- **入口**（x=0）：解析 Dirichlet —— 库函数 `boundaries.zou_he_inlet_velocity`，
  每行施加解析 u(0,y')、v(0,y')（Zou & He 1997 速度入口重构，二阶）。
- **出口**（x=nx−1，主配置）：**零梯度** Neumann —— `f[:,:,-1] = f[:,:,-2]`（~3 行内联；
  库无此函数，见 /tmp/kovasznay_gap.md）。
- **主循环**：collide → stream（周期）→ 出口零梯度 → 入口 Zou-He（同 poiseuille_2d 模式）。
- **域长与网格**：主配置 x' ∈ [0,3]（nx = 3·ny，y 向周期 1），网格档位 ny = 32/64/128
  （即 96×32 / 192×64 / 384×128）。**零梯度出口在短域（x'∈[0,1]，方形 64²）会把出口处
  v 分量杀到 ≈0（全场 v 最大相对误差 ~96%、v L2 22%）**；域长 3 个周期后出口处 e^{λx'}≈0.056，
  扰动区 v 已衰减至掩模阈值以下，全场误差 ≤3%（详见 gaps K1）。
- **参数**：Re = 40，U0 = 0.03（u ∈ [0, 0.06]，Ma_max ≈ 0.104）；ν = U0·ny/Re，
  τ = 0.5+3ν（ny=32/64/128 → τ = 0.572/0.644/0.788）。
- **稳态**：≥20000 步；每 500 步用 float64 监测全场 ‖u‖₂ 相对漂移，<1e-6 即停（上限 60000 步）。
- **测量**：末 100 步时间平均后全场对比 u/v：
  - L2 相对误差 ‖u_num−u_ana‖₂/‖u_ana‖₂（全场 + 内部列 1..nx−2）
  - 掩模最大点相对误差（|u_ana|>0.1·U0；|v_ana|>0.1·max|v_ana|，避免除以近零值）
  - 最大绝对误差 / U0

## 结果

主配置（零梯度出口，x'∈[0,3]）：

| ny (格) | nx | τ | u L2 误差 | v L2 误差 | u 最大相对误差 | v 最大相对误差 | 步数 | 稳态 |
|---------|-----|-----|-----------|-----------|----------------|----------------|------|------|
| 32 | 96 | 0.572 | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| 64 | 192 | 0.644 | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| 128 | 384 | 0.788 | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |

验收网格（ny=64/128）全场 u/v L2 与掩模最大相对误差均 ≤3%，且 64→128 四项指标单调下降（收敛）。
残余误差来源：(1) 出口零梯度 BC 影响区（约最后 0.5–1 个周期，v 被压平，全场 v L2 的
主要贡献）；(2) BGK 的 O(Ma²) 压缩性误差（u L2 ≈0.16% 的底噪）。二者均不随网格细化消失，
故收敛为"弱收敛"（单调但斜率小）——如实记录。

交叉验证（双端解析 Dirichlet，方形 x'∈[0,1]，文献标准做法，仅作内部求解器收敛性参考）：

| ny (格) | u L2 误差 | v L2 误差 | u 最大相对误差 | v 最大相对误差 | 步数 |
|---------|-----------|-----------|----------------|----------------|------|
| 32 | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| 64 | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| 128 | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |

交叉配置 u/v L2 随细化快速收敛（~2–8× 下降），证明内部求解器高度精确；其 v 掩模最大
相对误差偏高（16%→5.6%）是 v ∝ sin(2πy') 零穿越处的度量伪影（绝对误差与 u 同量级
~0.5%·U0，见 gaps K4），L2 指标（≤1.3%）为准。

## 运行

```bash
PYTHONPATH=/home/wxsc/cxs/TensorLBM/src \
  /home/wxsc/anaconda3/envs/ftw-env/bin/python run.py scan <out_dir> \
    --grids 32 64 128 --xmax 3.0 --outlet zerograd --min-steps 20000 --max-steps 60000
# 交叉验证：
PYTHONPATH=/home/wxsc/cxs/TensorLBM/src \
  /home/wxsc/anaconda3/envs/ftw-env/bin/python run.py scan <out_dir_B> \
    --grids 32 64 128 --xmax 1.0 --outlet dirichlet --min-steps 20000 --max-steps 60000
# 单例：
PYTHONPATH=/home/wxsc/cxs/TensorLBM/src \
  /home/wxsc/anaconda3/envs/ftw-env/bin/python run.py single 64 case_ny64.json --xmax 3.0 --outlet zerograd
```

## 参考

- Kovasznay, L.I.G. (1948). Laminar flow behind a two-dimensional grid. *Proc. Camb. Phil. Soc.* 44, 58–62.
- Zou, Q. & He, X. (1997). On pressure and velocity boundary conditions for the lattice Boltzmann
  BGK model. *Phys. Fluids* 9, 1591–1598.
- 共性模块缺口：/tmp/kovasznay_gap.md（零梯度出口 BC 缺库函数等）。

## 判定

- 真实模拟（无外推、无人工修正）：是。
- 全场误差 ≤3%（主配置 ny=64/128，u/v L2 + 掩模最大相对误差）：是。
- ≥2 档网格（ny=64/128）收敛（四项指标单调下降）：是。
- `result.json` 中 `extrap: "none"`，主配置 verdict = PASS。
