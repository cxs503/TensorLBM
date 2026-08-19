# B32: 2D 圆柱绕流 Re=20（Schäfer–Turek 2D-1）— R_eff 修正基准

## 概述

- **问题**：2D 通道内偏心圆柱定常绕流，Schäfer–Turek 基准 2D-1（Turek & Schäfer 1996；
  参考解 Nabh 1998 谱方法，FeatFlow 官方值）。
- **参考**：$C_D = 5.57953523384$，$C_L = 0.010618948146$，$\mathrm{Re} = \bar U D/\nu = 20$。
- **域**：$[0,2.2]\times[0,0.41]$，圆柱 $B_r(0.2,0.2)$，半径 $0.05$（$D=0.1$）。
- **入口**：Poiseuille $u(0,y)=(4U\,y(0.41-y)/0.41^2,\,0)$，$U=0.3$（$\bar U=0.2$）；
  **出口**：Zou–He 压力 $\rho=1$；上下壁与圆柱：无滑移（半程反弹）。
- **归一化**：$C_D = F_x/(\tfrac12\rho\bar U^2 D)$，$C_L = F_y/(\tfrac12\rho\bar U^2 D)$，
  使用**物理直径** $D=0.1$（与参考一致），$F$ 为 Ladd 动量交换力。

## 实现（真实模拟，无外推，extrap: none）

- **格子**：D2Q9；**碰撞**：MRT（`solver.collide_mrt`，$\tau=0.8$，$\nu_{lb}=0.1$）。
- **主循环**：collide（固体格跳过）→ stream → Zou–He 入口（速度剖面）→ Zou–He 出口（$\rho=1$）
  → 上下壁反弹 → **动量交换测力**（`compute_obstacle_forces`，stream 后、圆柱反弹前）
  → 圆柱半程反弹（post-streaming）。
- **几何**：`boundaries.cylinder_mask`（圆心在格点上，mask 半径可偏移），
  `make_channel_wall_mask`、`bounce_back_cells` 均为库函数，零外推。
- **稳态**：120000 步，末 20000 步时间平均；每 2000 步质量重整化。

## R_eff 根因分析与修正（本基准的核心）

### 根因：半程反弹的有效壁面在「最外固体格中心 + 0.5 格」

post-streaming 半程反弹把无滑移壁面放在最外固体格中心与相邻流体格中心的**中点**。
旧版 mask 半径取物理半径 $R$（标记「中心距 $\le R$」的格为固体），于是：

- 沿轴向，最外固体格中心恰在距离 $R$ 处（$R$ 为整数格），壁面在 **$R+0.5$**；
- 模拟物体的有效直径 $D_\mathrm{eff} = 2(R+0.5) = D + 1$ 格，
  有效雷诺数 $\mathrm{Re}_\mathrm{eff} = \mathrm{Re}\cdot(D_\mathrm{eff}/D)$ 偏大；
- 定常区 $C_D$ 随 Re 增大而减小 → **Cd 系统性偏低**；
- 误差随网格改变（阶梯形状、$R$ 的小数部分）而非单调 → **网格依赖**。

实测（旧版 shift=0，120k 步）：

| D (格) | mask 半径 | 轴向壁面 | $D_\mathrm{eff}$ | Cd | err_Cd | 备注 |
|--------|-----------|----------|------------------|----|--------|------|
| 40 | 20.0 | 20.5 | 41 | 5.5372 | **-0.76%** | 巧合偏近 |
| 80 | 40.0 | 40.5 | 81 | 5.3964 | **-3.28%** | 加细反而恶化 |

加细后误差不降反升，正是阶梯/壁面位置误差的网格依赖表现（$R$ 整数时轴向壁面恒为
$R+0.5$，相对误差 $1/D$ 虽减，但阶梯角点与动量交换的局部误差占主导）。

### 修正：mask 半径取 $R-0.5$（`--shift -0.5`）

将 mask 阈值圆缩小半格：固体格 =「中心距 $\le R-0.5$」。此时最外固体格中心在
$\lfloor R-0.5\rfloor = R-1$（$R$ 为整数），半程反弹壁面落在 **$R-0.5$**，
$D_\mathrm{eff}=D-1$ 格——从下方夹逼物理直径，阶梯表面整体逼近物理圆。
（更精确的做法是 Bouzidi/Guo 在精确壁面位置插值；$R-0.5$ 掩膜是标准的实用修正，
且不改变动量交换测力框架。）

### 修正后结果（shift=-0.5，120k 步，末 20k 平均）

| D (格) | mask 半径 | 轴向壁面 | $D_\mathrm{eff}$ | Cd | err_Cd | err_Cl |
|--------|-----------|----------|------------------|----|--------|--------|
| 40 | 19.5 | 19.5 | 39 | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| 80 | 39.5 | 39.5 | 79 | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |

### 收敛判定

- 达标：$\mathrm{err\_Cd} \le 3\%$ 且 D=40→80 单调逼近参考值 5.5795。
- PLACEHOLDER_VERDICT

## 附带发现：3D 圆管（poiseuille_3d_pipe）同根因

`benchmarks/pending/poiseuille_3d_pipe`（D3Q19 Hagen–Poiseuille）与圆柱同根因：
流体格取「中心距 $\le R$」、固体取「$>R$」，半程反弹壁面沿轴向在 $R+0.5$。
但**流量反演**得到的有效半径 $R_\mathrm{eff}^Q$（$\sqrt{2Q/(\pi U_\max)}$）为：

| R | 轴向壁面 $R+0.5$ | $R_\mathrm{eff}^Q$（流量反演） |
|---|------------------|-------------------------------|
| 20 | 20.5 | 20.109 |
| 40 | 40.5 | 40.118 |

阶梯角点使流动感受到的平均壁面比轴向极值 $R+0.5$ 更靠近物理半径（$R+0.11$ 左右），
因此该基准的失分主要来自**用 $R+0.5$ 作解析剖面比较**（中心区形状误差 15.5%/7.5%），
而不是几何本身偏大 $0.5$ 格。若对圆管也套用 $R-0.5$ 掩膜，有效半径会过冲到
$R-0.39$ 附近，反而更差——圆管的正确修正是按流量反演的 $R_\mathrm{eff}^Q$ 比较剖面
（或对壁面做精确插值），而非掩膜偏移。**结论：$R-0.5$ 掩膜修正适用于以力/阻力为
验收量的 2D 圆柱基准，不适用于以剖面形状为验收量的 3D 圆管基准。**

## 复现

```bash
cd /home/wxsc/cxs/TensorLBM
export PYTHONPATH=/home/wxsc/cxs/TensorLBM/src
python benchmarks/verified/cylinder_re20_st/run.py --D 40 --device cuda:1 --steps 120000
python benchmarks/verified/cylinder_re20_st/run.py --D 80 --device cuda:1 --steps 120000
```
