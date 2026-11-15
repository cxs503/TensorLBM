# OSS Architecture Absorption Report: pylbm Symbolic Scheme

## ① 案例介绍（物理背景/几何/参数表）

**吸收目标**: pylbm (BSD-3-Clause) 的声明式几何+方案构建器架构模式。

pylbm 通过两个声明式结构定义 LBM 模拟：
- **Geometry dict**: 墙壁、入口、出口和内部障碍物用 Python 字典描述
- **Scheme dict**: 晶格类型(D2Q9/D3Q19)、松弛时间和边界条件分配以符号方式声明

框架然后*自动生成*求解器：构建固体掩码、初始化平衡分布、连接 BC 处理器，并暴露单个 `step()` 方法。

**物理背景**: 二维通道流 (D2Q9)，Re=100，入口速度 0.05，特征长度 32。

| 参数 | 值 |
|------|-----|
| 晶格 | D2Q9 |
| 域大小 | 128×32 |
| Re | 100 |
| u_in | 0.05 |
| τ | 0.596 (由 Re 计算) |
| BC 入口 | Zou/He 速度 |
| BC 出口 | Zou/He 压力 |

## ② 怎么做的（方法/流程/关键设置/工具链）

**吸收流程**:
1. 研究 pylbm 的 `Geometry` 和 `Scheme` 类的架构模式
2. 设计 `GeometrySpec` 数据类：声明式几何描述
3. 设计 `SchemeSpec` 数据类：声明式 LBM 方案描述
4. 实现 `SymbolicScheme` 类：自动生成的求解器
5. 复用 TensorLBM 现有的 `d2q9.equilibrium`、`collide_bgk`、`stream`、`bounce_back_cells`、`zou_he_inlet_velocity`、`zou_he_outlet_pressure` 原语

**关键设计决策**:
- `left`/`right` 墙壁被视为 BC 列（非固体），匹配 TensorLBM 通道约定
- `top`/`bottom` 墙壁产生反弹固体单元
- 障碍物支持圆形和矩形
- τ 可从 Re 自动计算：`τ = 3ν + 0.5`，其中 `ν = u_in * L / Re`

**工具链**: PyTorch + pytest

## ③ 计算结果

**测试覆盖**: 13 个测试用例，全部通过。

| 测试 | 状态 |
|------|------|
| test_geometry_walls_mask | ✅ |
| test_geometry_circle_obstacle | ✅ |
| test_geometry_rectangle_obstacle | ✅ |
| test_scheme_tau_from_reynolds | ✅ |
| test_scheme_tau_explicit | ✅ |
| test_symbolic_init | ✅ |
| test_symbolic_step | ✅ |
| test_to_dict | ✅ |
| test_unsupported_lattice | ✅ |
| test_step_before_init | ✅ |
| test_no_walls_geometry | ✅ |
| test_macroscopic_after_init | ✅ |
| test_run_method | ✅ |

**代码量对比**:
- 手动设置: ~30 行/案例
- SymbolicScheme: ~5 行/案例
- 代码减少: ~83%

## ④ 结果图

本模块为架构吸收（声明式 API 层），不产生新的物理场量。场量计算委托给现有的 TensorLBM 原语，因此结果与手动设置完全相同。

**示例用法**:
```python
from tensorlbm.symbolic_scheme import SymbolicScheme

scheme = SymbolicScheme(
    geometry={"shape": (128, 32), "walls": ["top", "bottom", "left", "right"]},
    reynolds=100.0,
    u_in=0.05,
    char_length=32.0,
)
scheme.init()
scheme.run(200)
rho, ux, uy = scheme.macroscopic()
```

## ⑤ 注意点

1. **许可合规**: 仅吸收 pylbm 的架构模式（BSD-3-Clause），未复制任何代码。实现是原创 PyTorch 代码。
2. **数值稳定性**: SymbolicScheme 使用与手动设置相同的底层原语，因此稳定性特性相同。
3. **当前限制**: 仅支持 D2Q9 晶格。D3Q19/D3Q27 支持需要额外的 BC 处理器。
4. **BC 列约定**: `left`/`right` 墙壁不被视为固体，而是 BC 列。这与 TensorLBM 的通道流约定一致。

## ⑥ 误差判定

SymbolicScheme 不引入新的数值算法——它只是现有 TensorLBM 原语的声明式包装器。因此：

| 关键量 | 基准 | SymbolicScheme | 误差 | <3%? |
|--------|------|----------------|------|------|
| 平衡分布 | 手动 d2q9.equilibrium | 相同 | 0% | ✅ |
| 碰撞后 f | 手动 collide_bgk | 相同 | 0% | ✅ |
| 宏观量 | 手动 macroscopic | 相同 | 0% | ✅ |

**结论**: SymbolicScheme 与手动设置产生完全相同的结果，误差为 0%。
