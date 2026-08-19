# 方柱绕流 Re=100（尖锐角验证）— 待达标

**状态：🔶 未达标（Cd=-26.7%，参考口径 + MEM 偏差待查）**

## 物理问题

自由流方柱绕流 Re=100（尖锐角前缘分离）：
- **Sohankar et al. 1997**：Cd≈2.05（通道配置，范围 2.01-2.10）
- 自由流方柱 Re=100：文献范围 ~1.5-2.1（依赖域/涡脱落状态）

## 共性模块

库 solver 路径（D2Q9）：solver.collide_mrt + stream + d2q9.equilibrium
+ boundaries（compute_obstacle_forces MEM + far_field_bc_2d + 方柱 BB）
（GeneralSimEngine POLYGON_2D 是死代码——无分支，已记录缺口）

## 结果（子 agent 实测，真实模拟）

| 配置 | Cd | St | Cl_std | 状态 |
|------|-----|-----|--------|------|
| S=40（30S×30S 域） | **1.504（-26.7%）** | 0.153 | 0.099 | 涡脱落建立、统计收敛 |
| S=80 | 后台跑中 | — | — | — |

**关键点**：涡脱落**已真实建立**（St=0.153 正常、Cl_std=0.099、10 块均值漂移 0.0034）——不是流态问题。Cd 统计收敛但系统性偏低 26.7%。

## 未达标原因（两个候选）

1. **参考口径**：Sohankar 2.05 是**通道配置**；自由流方柱 Re=100 文献范围 1.5-2.1——1.50 可能接近自由流真实值（需更多文献确认）
2. **MEM 全固体格求和**：`compute_obstacle_forces` 对所有固体格求和（F=2·Σ_{solid}c·f），方柱内部 ~1944 ghost 格可能贡献 O(ρU·A) 偏置（与 B3/B4 球圆柱的 MEM 问题同源）——需 ghost_clean A/B 诊断确认

## 下一步

- **ghost_clean 诊断**：只用表面格（surface mask）测力 vs 全固体格对比（脚本已备）
- 参考值口径确认（自由流 vs 通道）
- 目标：Cd 对确认参考 ≤3%

## 运行方式

```
cd /home/wxsc/cxs/TensorLBM
PYTHONPATH=src python /tmp/sq_analyse.py   # 或子 agent 的 /tmp/run_sq_*.sh
```
