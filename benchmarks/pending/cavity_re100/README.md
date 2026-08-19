# 方腔流 Re=100（lid-driven cavity）— 待达标

**状态：🔶 未达标（128/192 网格 22.5% 偏差，待顶盖 BC 修复）**

## 物理问题

顶盖驱动方腔流，经典基准：
- **Ghia, Ghia & Shin 1982（JCP 48:387）**：Re=100/400/1000 中线速度表值 + 涡心
- Re=100：主涡心 (0.6172, 0.7344)，中线 u(0.5,0.5)=-0.20581

## 共性模块

**src/tensorlbm/lid_driven_cavity.py**（现成模块！）：
- D2Q9、`zou_he_moving_lid` 顶盖动壁 BC、任意壁全反弹
- **内置 GHIA_RE100/400/1000 论文表值**（脚本直接对比）
- 复用：solver.collide_mrt + solver.stream + boundaries.bounce_back_cells

## 结果（子 agent 实测，真实模拟）

| 网格 | 最大偏差 | 收敛 |
|------|---------|------|
| 128² | ~22.5% | err 不随网格下降 |
| 192² | ~22.5% | 同上 |

## 未达标原因

- 顶盖移动壁 BC（zou_he_moving_lid）在**角点**处理与 Ghia 参考不一致（角点奇异性）
- 128→192 误差不降（err_decreased=False）——系统性问题非分辨率
- 注意：任务上下文给的"y=0.25→0.0138"等近似值与论文不符，应以库内 GHIA_RE100 表值（u(0.5,0.5)=-0.20581）为准

## 下一步

- 检查 zou_he_moving_lid 角点处理（角点速度应为 0 或平均）
- 对比 Ghia 参考时注意 y 坐标降序（np.interp 陷阱）
- 目标：中线速度对 Ghia 表值偏差 ≤3%

## 运行方式

```
cd /home/wxsc/cxs/TensorLBM
PYTHONPATH=src python benchmarks/cavity_re100/run.py   # 128/192 网格，10 万步上限
```
