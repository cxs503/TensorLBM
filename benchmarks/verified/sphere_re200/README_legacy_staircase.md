# 球 Re=200（GeneralSimEngine 绕流）— 待达标

**状态：🔶 未达标（压力积分 -24.6%、MEM +268%，力法缺口阻塞 G12/G15）**

## 物理问题

自由流绕球 Re=200，阻力系数：
- **Schiller-Naumann**：Cd = 24/200·(1+0.15·200^0.687) = **0.8056**（精确值；任务书 0.769 是算术错误）
- Clift-Gauvin：0.7810

## 共性模块

**GeneralSimEngine PARAMETRIC_SPHERE**（src/tensorlbm/general_sim.py）：
- D3Q19、MRT、BB 壁面、质量修正、extrap='none'

## 结果（真实模拟，16000 步收敛）

| 力法 | Cd | 误差 | 分析 |
|------|-----|------|------|
| 压力积分（extrap=none） | 0.607 | **-24.6%** | 近壁压力无外推→系统性低估（G12） |
| **MEM（MOMENTUM_EXCHANGE）** | 2.97 | **+268%** | Ladd 和式曲面平衡背景未扣除（G15） |

（D=60/D=80 压力积分都 ~0.62，网格加密不改善——系统性偏差非分辨率）

## 未达标原因

1. **G12 压力积分**：extrap='none' 时 drag_pressure_integration 取近壁单元压力（距壁 0.5-1 格）积分，无壁面外推→Cd_p 系统性低估（Cd_p≈0.41 vs 文献 0.55 量级）
2. **G15 MEM**：momentum_exchange_standard 是 Ladd 和形式（f_i+f_opp），平壁平衡项抵消但**曲面球壁不抵消**→O(ρU·A) 虚假力

## 顺带发现的引擎 bug

- `_sample_forces` 中 cd_mem（ForceMethod.BOTH）错存为 (fx_p+fx_f)/dpS 而非 MEM 力/dpS——**已修复**（82ebdaf，MOMENTUM_EXCHANGE 单独可选 + cd_mem 正确）

## 下一步

- MEM 减平衡背景（F -= ρU·Σn̂·dA）或用 galilean 变体（momentum_exchange_galilean）
- 或压力积分加二阶壁面外推（但禁外推规则下不可用——需改用 MEM 正确实现）
- 球绕流系列（B1/B2/B3）因力法缺口暂缓

## 运行方式

```
cd /home/wxsc/cxs/TensorLBM
PYTHONPATH=src python benchmarks/pending/sphere_re200/run.py 60 16000 cuda:0
```
