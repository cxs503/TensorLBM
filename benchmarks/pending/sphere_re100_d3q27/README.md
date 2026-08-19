# 球 Re=100 D3Q27（高精度碰撞模型）— 待达标

**状态：🔶 验证中（D3Q27 共性路径搭建完成，正式网格待跑）**

## 物理问题

自由流绕球 Re=100，D3Q27 高精度碰撞模型验证：
- Schiller-Naumann Cd=1.087 / Clift-Gauvin Cd=1.109
- 目的：对比 D3Q19（历史 2.3%），验证 D3Q27 各向异性精度收益

## 共性模块（D3Q27 库函数路径）

GeneralSimEngine 的 `LatticeModel.D3Q27` 是**死枚举**（setup 无条件用 d3q19：equilibrium3d/collide/macroscopic）——已记录缺口 G7。走**库函数路径**：
- 碰撞：`d3q27_collide.collide_cumulant_geier_d3q27` / `collide_mrt27`（cumulant.py）
- 流播：`d3q27.stream27`（solver3d.stream3d 是 D3Q19 硬编码 q=19，不可用）
- 远场 BC：`boundaries_d3q27.far_field_bc_27`（已存在）
- 力：`obstacles.compute_obstacle_forces_27`（Ladd MEM）

## 状态

- run.py 已写好（benchmarks/pending/sphere_re100_d3q27/run.py，~7200 字节，全库函数零手写）
- 后台脚本已启动（D=40 mrt27/cumulant_geier + D=60 mrt27）
- GPU 性能实测：D=40 mrt27 499ms/步、cumulant_geier 96ms/步（1.5M 格）——GPU 必要（CPU cumulant 9s/步不可行）

## 共性模块缺口（G7/G8）

- GeneralSimEngine D3Q27 死枚举（需 setup 分派 D3Q27 分支）
- solver3d.stream3d / far_field_bc_3d / momentum_exchange_standard 全部 D3Q19 硬编码
- drag_pressure 压力+摩擦积分无 D3Q27 路径

## 运行方式

```
cd /home/wxsc/cxs/TensorLBM
PYTHONPATH=src python benchmarks/pending/sphere_re100_d3q27/run.py \
  --diam 40 --collision cumulant_geier --steps 25000 --device cuda:1
```
