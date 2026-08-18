# B2: 球体 Re=100 阻力系数（D60 网格加密）— GeneralSimEngine 共性模块

**状态：🔶 框架/验证中（未达标不入库，本目录为 run.py 占位）**

## 物理问题

均匀来流绕球（直径 D=1.0 m），Re=100。阻力系数参考：
- Schiller-Naumann: Cd = 24/Re·(1+0.15·Re^0.687) = **1.087**
- Clift-Gauvin: Cd = 24/Re·(1+0.1315·Re^(0.82-0.05·log10(Re))) = 1.0875

## 共性模块

**GeneralSimEngine**（src/tensorlbm/general_sim.py）：
- GeometryConfig: PARAMETRIC_SPHERE（radius=0.5, center=(0,0,0)）
- SolverConfig: D3Q19, collision=AUTO(→MRT), resolution=60, wall=BOUNCE_BACK
- ForceMethod: PRESSURE_FRICTION, extrap='none'（真实模拟，禁外推）
- 主循环: lbm_step_correct + far_field_bc_3d + drag_pressure/drag_friction 积分

## 运行方式

```
cd /home/wxsc/cxs/TensorLBM
PYTHONPATH=src python benchmarks/verified/sphere_re100/run.py 60 quadratic 4000
```

## 结果（2026-08-18 实测）

- D40 (extrap=none): Cd≈0.92-0.95（**-13~-15% 未达标**）
- D40 (quadratic 外推): 0.948（-13.2%——外推不作达标依据）
- D60: 待测（网格加密看收敛趋势）

## 未达标原因与下一步

D40 网格分辨率不足（域 140³/180³），Cd 偏低 ~13%。需：
1. D60/D80/D100 加密扫描收敛趋势
2. 检查压力/摩擦积分公式（历史 0.98% quadratic 表明压力外推有效但禁外推）
3. 目标：真实网格（extrap=none）误差 ≤3%

## 判定标准

- 真实模拟（extrap='none'），禁外推凑精度
- 误差 = |Cd_sim − Cd_ref|/Cd_ref ≤ 3% 才入库
- 网格收敛趋势为可信度依据
