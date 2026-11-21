# Force-Corrected NEBB + Body-Force Driver Scaffold

## 概述

本模块实现了 **force-corrected NEBB**（力修正非平衡反弹）边界条件和 **body-force driver scaffold**（体力驱动脚手架），用于 LBM 中体力驱动流动的精确模拟。

## 物理背景

在 LBM 中，当存在体力（如重力、压力梯度）时，标准的半程反弹（half-way bounce-back）边界条件会在壁面引入速度滑移误差。这是因为体力对动量平衡有贡献，但标准 BB 没有考虑壁面节点处的体力贡献。

Force-corrected NEBB 通过添加基于 Guo (2002) 体力模型的修正项来消除这种滑移误差，即使在体力存在的情况下也能保持壁面速度为零。

## 实现

### 模块结构

```
src/tensorlbm/
├── boundary/
│   ├── __init__.py                    # 导出 ForceCorrectedNEBB 等
│   └── force_corrected_nebb.py        # 力修正 NEBB 边界条件
└── forcing/
    ├── __init__.py                    # 导出 BodyForceDriver 等
    └── body_force_driver.py           # 体力驱动脚手架
```

### 关键算法

#### Guo 体力模型 (Guo et al., 2002)

物理速度：
```
u = (Σ f_i c_i + F/2) / ρ
```

平衡分布使用物理速度计算：
```
feq = feq(ρ, u)
```

碰撞后添加体力项：
```
f* = f - ω(f - feq)
f_new = f* + S_i
```

其中体力项：
```
S_i = (1 - ω/2) · w_i · [3(c_i - u) + 9(c_i·u)c_i] · F
```

#### Force-Corrected NEBB

在壁面流体节点，未知分布函数通过以下方式重建：
```
f_i = f_opp + (feq_i - feq_opp) + Δf_force
```

其中力修正项：
```
Δf_force = (1 - 1/(2τ)) · w_i · [3(c_i - u_w) + 9(c_i·u_w)c_i] · F
```

对于静止壁面 (u_w = 0)：
```
Δf_force = (1 - 1/(2τ)) · w_i · 3·c_i · F
```

## 验证状态

### 已验证
- ✅ 质量守恒：Guo 体力项零和，密度漂移 < 1e-12
- ✅ 壁面滑移：force-corrected NEBB 消除壁面速度（slip < 1e-5）
- ✅ 力修正项线性缩放：修正项与体力大小成正比
- ✅ 力修正项依赖 τ：修正系数 (1 - 1/(2τ)) 正确

### 待修复
- ⚠️ 速度剖面误差 ~15%：存在流/BC 交互的微妙问题，需要进一步调试
  - 可能原因：拉Convention 与现有 lbm_channel.py 的推Convention 不一致
  - 建议：对齐现有工作代码的流Convention

## 使用示例

```python
from tensorlbm.forcing.body_force_driver import BodyForceDriver, PoiseuilleConfig

cfg = PoiseuilleConfig(
    height=32,       # 流体行数
    nx=4,            # 流向列数
    tau=0.8,         # 松弛时间
    force=1e-5,      # 体力
    steps=10000,     # 总步数
    forcing="guo",   # Guo 二阶体力
    bc="force_corrected_nebb",  # 力修正 NEBB
)

driver = BodyForceDriver(cfg)
result = driver.run()
print(f"L2 error: {result.eps_L2:.6f}")
print(f"Wall slip: {result.wall_slip:.2e}")
```

## 参考文献

1. Guo, Z., Zheng, C., & Shi, B. (2002). "Discrete lattice effects on the forcing term in the lattice Boltzmann method." Phys. Rev. E, 65, 046308.
2. Chen, H., & Doolen, G. D. (1998). "Lattice Boltzmann method for fluid flows." Annu. Rev. Fluid Mech., 30, 329-364.
3. haimingz/vivsim: force-corrected-nebb implementation pattern.

## 文件清单

| 文件 | 说明 |
|------|------|
| `src/tensorlbm/boundary/force_corrected_nebb.py` | 力修正 NEBB 边界条件实现 |
| `src/tensorlbm/boundary/__init__.py` | 更新导出 |
| `src/tensorlbm/forcing/__init__.py` | 体力驱动模块 |
| `src/tensorlbm/forcing/body_force_driver.py` | BodyForceDriver 脚手架 |
| `tests/test_force_corrected_nebb.py` | 测试套件 |
| `docs/force_corrected_nebb_report.md` | 本报告 |
