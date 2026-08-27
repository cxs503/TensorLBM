# Couette 流（3D 平板剪切，D3Q19）— 解析线性剖面验证

**状态：✅ 已入库（2026-08-20）**

## 物理问题

两平行平板间剪切流：上壁以 U0 运动、下壁静止，x/z 方向周期（展向均匀），
稳态线性速度剖面（与 2D 版完全相同的解析解，对任意 Re 成立）：
- **u(y) = U0·y/H**（精确线性剪切）
- 壁面剪切 τ = ν·U0/H

## 共性模块与边界处理

库 solver 路径（tensorlbm.solver3d.collide_bgk3d + stream3d + d3q19.equilibrium3d/macroscopic3d）：
- 网格 (nz=8, ny=H+2, nx=8)，壁行 y=0 / y=ny-1，x/z 周期
- 碰撞：BGK（τ=0.8, ν=0.1）
- 上壁移动、下壁静止：**pre-streaming half-way 反弹 + 动量注入**
  f_new[q] = f_pre[opp[q]] + 2·w_q·ρ·(c_q·u_wall)/cs²（Zou-He 移动壁反弹，
  与已验证 2D Couette benchmark 同一方案，方向索引对照 D3Q19 C 矩阵核对：
  顶壁入流方向 cy<0 = {4,8,9,16,18}）
- U0 = 0.1（Ma ≈ 0.173），≥20000 步稳态（drift < 1e-5）

## 结果（2026-08-20 实测，真实模拟无外推）

| 网格 H | Re | L2 误差 | max_rel | u_top 误差 | 稳态 |
|--------|-----|---------|---------|-----------|------|
| 40 | 40 | ... | **...%** | ...% | ✅ |
| 80 | 80 | ... | **...%** | ...% | ✅ |

**收敛性**：误差随 H 增大单调下降（...%→...%）✅ 真收敛

真实性检查：uz_max ~1e-7（展向无驱动，z 向速度消失）、ux 跨 z 均匀、
质量漂移 ~1e-3%、总 x 动量守恒。

## 运行方式

```
cd /home/wxsc/cxs/TensorLBM
PYTHONPATH=src python benchmarks/verified/couette_3d/run.py scan /tmp/couette3d_scan --H 40 80 --tau 0.8 --u0 0.1
```

## 判定

- 真实模拟（无外推），max_rel ≤3% 且两档网格收敛 → 达标
- 2D Couette 的 3D 推广：验证 D3Q19 流形 + 移动壁 BC 迁移正确性
