# 圆柱 Re=20（Schäfer-Turek 通道基准 2D-1）— 待达标

**状态：🔶 未达标（D=40 巧合精度，D=80 加密恶化；R_eff 修复方向明确）**

## 物理问题

Schäfer-Turek 通道内圆柱（CFD 标准验证）：
- 域 [0,2.2]×[0,0.41]，圆柱 D=0.1 偏心 (0.2,0.2)，Poiseuille 入口 + 压力出口
- **Re=20（U_mean·D/ν）**：Cd=**5.57953523384**、Cl=0.010618948146（Nabh 1998 谱方法，FeatFlow 官方引用值）
- OpenLB cylinder2d.cpp 同配置（Poiseuille 入口、压力出口、Cd 按 U_mean 归一）

## 共性模块

库 solver 路径（D2Q9）：solver.collide_mrt + stream + d2q9.equilibrium
+ boundaries（zou_he_inlet_velocity 剖面 / zou_he_outlet_pressure / bounce_back_cells / cylinder_mask / make_channel_wall_mask / compute_obstacle_forces）

## 结果（主 agent 实测，真实模拟）

| 网格 | Cd | 误差 | Cl | 步数 |
|------|-----|------|-----|------|
| D=40（880×164） | 5.537 | **-0.76%** | -0.014 | 100000 |
| D=80（1760×328） | 5.396 | **-3.28%** | -0.0016 | 80000 |

## 未达标原因

1. **加密恶化**：D=40 的 -0.76% 是巧合精度，D=80 恶化到 -3.28%——按用户"加密恶化=假结果"标准不入库
2. **Cd 随网格下降**（5.54→5.40）：嫌疑 **half-way BB 有效半径 R_eff=R+0.5**（有效直径偏大 → Cd 偏低）或入口剖面/单位网格依赖
3. **Cl 符号反**（-0.014 vs +0.0106）：y 轴约定或参考符号约定问题

## 下一步

- **R_eff 修正**：用 R-0.5 或精确壁面位置构造圆柱 mask（与 poiseuille_3d_pipe 同根因，可同时修复）
- 复核入口 Poiseuille 剖面格子单位（U_char_lb 换算）
- 目标：两档网格 Cd 单调收敛到 5.5795 且 ≤3%

## 运行方式

```
cd /home/wxsc/cxs/TensorLBM
PYTHONPATH=src python /tmp/b31_cyl40.py --D 40 --steps 100000
```
