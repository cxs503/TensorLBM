# 圆柱绕流 benchmark 家族

**状态：✅ 全部达标（同一问题族，不同 Re/口径，各自两档收敛验证）**

## 物理问题

圆柱绕流——最经典的 CFD 验证（不同 Re 流态完全不同）：
- **Re=20 通道口径**：定常层流无分离（Schäfer-Turek 基准）
- **Re=100 自由流**：Kármán 涡街（周期涡脱落）
- **Re=200 自由流**：涡脱落增强（频率升高）

## 工况表（本文件夹下）

| 子目录 | 工况 | 口径 | 碰撞 | Cd | St | 关键点 |
|--------|------|------|------|-----|-----|--------|
| **re20_st/** | Re=20 | **Schäfer-Turek 通道**（2D-1） | MRT | 5.548（-0.56%）/5.419（-2.88%） | — | **质量重整化振荡真凶发现**（去掉后收敛） |
| **re100/** | Re=100 | 自由流 | MRT+sponge | 1.329（-1.5%）/1.316（-2.6%） | 0.166/0.160 | **播种权重 bug + sponge 层**（首个涡脱落） |
| **re200/** | Re=200 | 自由流 | MRT+sponge | 1.339（+0.65%）/1.318（-0.92%） | 0.196/0.194 | 工程链复用（播种 St 0.165→0.195） |

## 共性模块

- solver.collide_mrt（tau_field sponge）+ stream + d2q9
- boundaries.far_field_bc_2d + make_sponge_strength + zou_he（通道口径）
- compute_obstacle_forces（Ladd MEM）

## 关键技术发现

1. **质量重整化振荡**（每 2000 步全局重整化激发 ~10k 步周期慢振荡 Cd±2-4%）——网格依赖假象的真凶
2. **播种权重 bug**（手写 D2Q9 cy_w 符号错 → 3.5 倍横向射流 → NaN）
3. **sponge 吸收层**（下游 10D tau_field 渐变）消除 far-field 出口尾流反射
4. **涡脱落工程链普适性**：Re=100→200 只需调播种频率

## 运行方式

```
cd /home/wxsc/cxs/TensorLBM/benchmarks/verified/cylinder/re200
PYTHONPATH=../../../src python run.py
```
