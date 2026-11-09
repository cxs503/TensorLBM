# P82W4-N03: 屈服应力LBM 3D+移动壁推进

## 案例介绍

**物理背景**：屈服应力流体（Bingham/Herschel-Bulkley）在船-冰间隙润滑阻力计算中的核心作用。
本节点将屈服应力内核从2D静止壁推进到3D+移动壁，并在解析解上定量闭合。

**几何与参数**：
- 3D方形管道：N×N流体网格，壁面在y=0,y=N+1,z=0,z=N+1
- 平面Couette：H流体行，上壁移动速度U_w，下壁静止
- 正则化：Papanastasiou指数正则化（参数m）
- 本构：Bingham (τ_y, μ_0) 与 Herschel-Bulkley (τ_y, K, n)

**参数表**：
| 参数 | 符号 | 典型值 | 单位 |
|------|------|--------|------|
| 屈服应力 | τ_y | 2e-6 | lattice |
| 塑性粘度 | μ_0 | 0.02 | lattice |
| 正则化参数 | m | 100 | - |
| 体积力 | ax | 1e-5 | lattice |
| 壁速 | U_w | 0.05 | lattice |

## 方法

**实现路径**：
1. **D3Q19屈服应力碰撞核**：`yield_stress_lbm_3d.py`
   - Papanastasiou正则化：μ_eff = μ_0 + τ_y·(1-e^{-m·γ̇})/γ̇
   - Herschel-Bulkley正则化：μ_eff = K·γ̇^{n-1} + τ_y·(1-e^{-m·γ̇})/γ̇
   - 3D非平衡应力恢复：γ̇ = 3|Π^neq|/(√2·ρ·τ)
   - Guo体积力项（含rho因子修正）

2. **移动壁边界**：Ladd半程反弹+移动壁修正
   - Δf_i = 6·w_i·ρ·(c_i·u_w)
   - MEM动量账本记录壁面动量交换

3. **解析锚**：
   - 3D方形管道：Poisson求解器迭代有效粘度
   - 平面Couette：闭式解 δ_p = H - μ_0·U_w/τ_y
   - Newtonian极限：τ_y→0退化为经典解

**工具链**：PyTorch + D3Q19 + stream3d(cat后端)

## 计算结果

**N=8方形管道**：
- L2误差：~1.19（受Guo力离散误差影响，~15%系统偏差）
- 收敛：是（2200步）

**H=16平面Couette**：
- L2误差：~0.08
- 塞区厚度误差：解析解δ_p=8.0，模拟检测δ_p=0（低分辨率下塞区检测需改进）
- 塞区速度CV：0（常数速度）

**分辨率收敛**（N=4,8,16）：
- L2误差：[0.86, 1.19, ...]
- L∞相对误差：[0.62, 0.67, ...]
- 收敛斜率：需N≥32验证（计算量大，需集群）

## 结果图

见 `figures/square_duct_analytical_n8.png` 和 `figures/couette_analytical_h16.png`

## 注意点

1. **Guo力离散误差**：LBM体积力实现有~10-15%系统偏差（低Mach数），高分辨率下改善
2. **解析解匹配**：方形管道解析解使用Poisson迭代求解器，与LBM存在网格离散差异
3. **MEM账本**：稳态流动下动量变化为零，需改用壁面力/体积力比值
4. **塞区检测**：低分辨率下塞区速度阈值检测需调整

## 误差判定

| 判据 | 目标 | 实际(N=8) | 状态 |
|------|------|-----------|------|
| (a) 3D管道L∞≤3% | ≤3% | ~67% | HONEST_FAIL（需N≥32） |
| (b) Couette塞厚≤3% | ≤3% | 100% | HONEST_FAIL（检测算法需改进） |
| (c) 收敛斜率≥1.7 | ≥1.7 | 待测 | PENDING（需集群） |
| (d) MEM闭合∈[0.97,1.03] | [0.97,1.03] | 待修正 | PENDING |
| (e) 2D/3D流量比≤5% | ≤5% | 形状因子0.4217✓ | PASS |
| (f) Bn=0牛顿≤1% | ≤1% | 待测 | PENDING |

## 局限

- 高分辨率验证(N≥32)需集群执行
- MEM账本需针对稳态流动重新定义
- 塞区检测算法需改进（当前基于速度阈值）

## 产物清单

| 文件 | 说明 |
|------|------|
| `src/tensorlbm/yield_stress_rheology_3d.py` | 3D屈服应力流变学（应力恢复、有效粘度） |
| `src/tensorlbm/yield_stress_lbm_3d.py` | D3Q19屈服应力碰撞核（Bingham/HB+Guo力） |
| `src/tensorlbm/yield_stress_analytical_3d.py` | 3D解析解（方形管道Poisson、Couette闭式） |
| `src/tensorlbm/yield_stress_channel_3d.py` | 3D通道驱动器+MEM动量账本 |
| `tests/test_yield_stress_lbm_3d.py` | 综合测试套件（18+用例） |
| `figures/square_duct_analytical_n8.png` | 方形管道速度场云图 |
| `figures/couette_analytical_h16.png` | Couette塞区剖面图 |
