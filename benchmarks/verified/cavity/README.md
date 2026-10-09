# 方腔流 benchmark 家族（lid-driven cavity）

**状态：✅ 全部达标（同一问题族，不同工况/维度，各自两档收敛验证）**

## 物理问题

顶盖驱动方腔流——经典基准（**Ghia, Ghia & Shin 1982, JCP 48:387**）：
- 顶盖以 U 移动驱动腔体内流动，Re = U·H/ν 决定流态
- Re 增大：主涡下沉、角涡出现、流场结构复杂化

## 工况表（本文件夹下）

| 子目录 | 工况 | 碰撞 | 网格 | 精度 | 关键点 |
|--------|------|------|------|------|--------|
| **re100/** | Re=100（层流单涡） | MRT | 128²/192² | 0.75%→0.73% | V3 修复迁移 |
| **re400/** | Re=400（主涡+次级涡） | MRT | 128²/192² | 1.50%→0.83% | **V3 pre-streaming 反弹修复（根因发现）** |
| **re1000/** | Re=1000（高 Re 多涡） | **RLBM** | 192²/256² | 1.14%→1.05% | MRT 在 Re≥576 失稳→RLBM 修复 |
| **3d/** | 3D 展向周期 Re=400 | MRT | 96²×24/128²×32 | 1.96%→1.38% | 3D 顶盖 BC（z 对角对 17/18 修复） |

## 共性模块

- solver.collide_mrt / collide_rlbm + stream + d2q9
- boundaries.zou_he_moving_lid（2D）/ zou_he_moving_lid_3d（3D）
- pre-streaming 半程反弹（V3，三静止壁）

## 关键技术发现

1. **V3 修复**（post-streaming 反弹 + 周期 stream 使顶盖动量绕入底壁 → 23.6% 偏差）——方腔家族全部受益
2. **MRT 在 Re≥576 失稳**（τ→0.5 发散）→ RLBM（正则化 BGK）修复
3. **3D 顶盖 Zou-He z 对角对 17/18 颠倒** → 虚假 jz 动量，修复后守恒

## 运行方式

每个子目录独立 run.py（100k-200k 步）：

```
cd /home/wxsc/cxs/TensorLBM/benchmarks/verified/cavity/re400
PYTHONPATH=../../../src python run.py
```
