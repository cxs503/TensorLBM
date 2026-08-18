# B6: SUBOFF 潜艇 Re=1000 阻力 — 待达标

**状态：🔶 验证中（历史 3.8% 超 3% 标准，子 agent 用 GPU2 重跑中）**

## 物理问题

DARPA SUBOFF（Standard 自推进潜艇模型）均匀来流阻力：
- 全尺寸实验（Re=1.2e7）：Ct ≈ 0.0031（Roddy 1990）
- **Re=1000 数值参考**：Ct ≈ 0.004（本库历史基线：Cd_tot=0.0439，误差 3.8% CUDA / 3.6% SDAA）
- 艇型：L/D≈8.57（bow 0-0.233L / midbody 0.233-0.748L / stern 0.748-1.0L）

## 共性模块

**GeneralSimEngine PARAMETRIC_SUBOFF**（src/tensorlbm/general_sim.py）：
- GeometryConfig: PARAMETRIC_SUBOFF（suboff_length, suboff_radius）
- SolverConfig: D3Q19, resolution（艇长 80-160 格）, wall=BOUNCE_BACK
- ForceMethod: PRESSURE_FRICTION, extrap='none'

## 运行方式

```
cd /home/wxsc/cxs/TensorLBM
PYTHONPATH=src python benchmarks/pending/suboff_re1000/run.py
```

## 结果

- 历史：Re=1000 BB 基线 Cd_tot=0.0439（**3.8%**，超 3% 未达标）
- 当前：子 agent GPU2 重跑中（加密网格 + 更长收敛）

## 未达标原因与下一步

1. 3.8% 主要来自网格分辨率 + 摩擦公式（SUBOFF 阻力以摩擦为主 ~70%）
2. 需：艇长 L≥128 格、域 ≥5L、≥20000 步收敛
3. 摩擦系数分布对比实验数据（验证近壁处理）
4. 目标：Ct 误差 ≤3%
