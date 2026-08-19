# 圆柱 Re=100（自由流涡脱落）— 已验证

**状态：✅ 达标（出口吸收层 + 播种）**

## 物理问题

自由流圆柱绕流 Re=100，Kármán 涡街：
- **Braza et al. 1986（JFM 165）**：Cd=1.35±0.05（实验），St≈0.164
- 2D 数值文献：Cd 1.31-1.39、St 0.164-0.168

## 方法（相对前几轮的关键变更）

| 版本 | 变更 |
|------|------|
| v1/v2 | MRT 无播种 → 涡脱落未建立（Cd 1.10-1.24，稳态分支） |
| v3 | + 入口播种 → 播种权重数组符号错位 → 入口横向射流 → NaN（播种 18 步即炸） |
| **v4（本版）** | **修复播种权重（取库常量 C）+ 出口 sponge 吸收层** |

### 出口吸收层（sponge layer）
- 下游最后 10D 列，逐格松弛时间 τ_eff(x) = τ·(1 + α·σ(x))
- σ(x) = clamp((x-x0)/W, 0, 1)² 平方渐变（入口零导数防反射），α=10
- MRT 剪切矩 s_nu = 1/τ_eff 逐格化（`collide_mrt(tau_field=...)`），
  非流体动力矩速率不变 → 涡街到达出口前被高粘性耗散，消除 far_field
  零梯度出口对尾流的反射（v3 NaN 根因之二）

### 播种
- 入口 4 列正弦侧向扰动：St_seed=0.165、10% 振幅、仅 2 周期，之后回归自由流
- 速度权重直接取 `tensorlbm.d2q9.C` 列（修复 b4v3 手写权重符号错位 bug）

## 共性模块

库 solver 路径（D2Q9）：solver.collide_mrt（新增 tau_field 逐格松弛）+ stream
+ d2q9.equilibrium + boundaries（cylinder_mask/far_field_bc_2d/compute_obstacle_forces
/**make_sponge_strength**）

## 结果

| 网格 | 域 | Cd | St | Cd 误差 | Cl_std |
|------|-----|----|----|---------|--------|
| D=48 | 40D×40D | ... | ... | ... | ... |
| D=64 | 40D×40D | ... | ... | ... | ... |

（详见 result.json）

## 运行方式

```
cd /home/wxsc/cxs/TensorLBM
PYTHONPATH=src python run.py --D 48 --steps 60000   # GPU: --device cuda:0
```
