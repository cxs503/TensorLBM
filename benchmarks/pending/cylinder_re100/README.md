# 圆柱 Re=100（自由流涡脱落）— 待达标

**状态：🔶 未达标（涡脱落播种后数值发散，需出口吸收层）**

## 物理问题

自由流圆柱绕流 Re=100，Kármán 涡街：
- **Braza et al. 1986（JFM 165）**：Cd=1.35±0.05（实验），St≈0.164
- 2D 数值文献：Cd 1.31-1.39、St 0.164-0.168

## 共性模块

库 solver 路径（D2Q9）：solver.collide_mrt + stream + d2q9.equilibrium
+ boundaries（cylinder_mask/far_field_bc_2d/compute_obstacle_forces）

## 结果（三轮实验，真实模拟）

| 版本 | 配置 | Cd | 问题 |
|------|------|-----|------|
| v1（子agent） | MRT 24D×12D 无播种 | 1.237（-8.4%） | **涡脱落未建立**（St 异常，稳态分支值） |
| v2（主agent） | MRT 40D×40D 无播种 | 1.10（-18.5%） | 同 v1（域更大 Cd 更低） |
| v3（主agent） | MRT 20D×20D **播种** | **NaN @20000 步** | 播种触发涡脱落但 far-field 零梯度出口反射尾流→发散 |

## 未达标原因（根因链）

1. Re=100 涡脱落**需播种**触发（纯数值噪声增长极慢：10k 步 Cl std=0.007 → 50k 步才 0.40）
2. 播种后流场非稳态 → **far_field_bc_2d 零梯度出口反射尾流压力波**
3. 反射 + 质量修正相互作用 → 20000 步数值发散（NaN）

## 下一步（修复方向明确）

- 出口加**吸收层（sponge layer）**：下游最后 5-10D 区域逐格增大 τ（或加阻尼力）
- 或下游加长（出口 ≥30D）让尾流在到达前衰减
- OpenLB/Palabos 圆柱案例的标准做法
- 目标：涡脱落建立后时均 Cd→1.35、St→0.165

## 运行方式

```
cd /home/wxsc/cxs/TensorLBM
PYTHONPATH=src python /tmp/b4v3_seed.py --D 48 --steps 60000   # 播种版（需先加吸收层）
```
