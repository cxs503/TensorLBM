# TensorLBM Benchmark 问题类型清单（按物理问题分类）

> 目标：每个 benchmark 通过共性模块实现，真实模拟（禁外推），精度 ≤1% 才保存。
> 本清单覆盖**不同问题类型**（不限于绕流），每类给出参考基准与共性模块路径。

## A. 稳态绕流（GeneralSimEngine）

| # | 问题 | 参考基准 | 共性模块 | 备注 |
|---|------|---------|---------|------|
| B1 | 球 Re=100 阻力 | Schiller-Naumann Cd=1.087 | PARAMETRIC_SPHERE | D40/D60/D80 真实模拟 |
| B2 | 球 Re=100 D60 加密 | 同上 | 同上 | 网格收敛趋势 |
| B4 | 2D 圆柱 Re=100 | Braza 1990 Cd≈1.35 | PARAMETRIC_CYLINDER+D2Q9 | 升阻力+St |
| B5 | 2D 圆柱 Re=200 | Braza 1990 Cd≈1.33, St≈0.196 | 同上 | 涡脱落基准 |
| B6 | SUBOFF Re=1000 | 实验 Ct | PARAMETRIC_SUBOFF | 阻力+摩擦系数 |

## B. 内部流动（解析解精确）

| # | 问题 | 参考基准 | 共性模块 | 备注 |
|---|------|---------|---------|------|
| B13 | 2D Poiseuille 流 | 解析抛物线剖面 u=Δp/2νL(y²−Hy) | D2Q9+Zou-He 入口 | **解析解，最易 1%** |
| B14 | 3D 管道流 | 解析圆管剖面 | D3Q19+压力驱动 | 需压力 BC 支持 |
| B15 | 方腔流 Re=100 | Ghia 1982 涡心/中线剖面 | D2Q9 或 D3Q19 | 经典基准，验证对流+耗散 |
| B16 | 方腔流 Re=400/1000 | Ghia 1982 | 同上 | 高阶验证 |

## C. 时变/涡流（文献基准）

| # | 问题 | 参考基准 | 共性模块 | 备注 |
|---|------|---------|---------|------|
| B17 | Taylor-Green 涡衰减 | 解析 e^(−2νk²t) 衰减率 | D3Q19 周期域 | **验证数值耗散**（LBM 弱项） | ✅ 已达标（2D D2Q9 周期域, err≤0.13%, benchmarks/verified/taylor_green_2d/） |
| B18 | 后向台阶 Re=100-800 | Armaly 1984 再附着长度 | backward_facing_step.py | 验证分离流 |
| B19 | 方柱绕流 Re=100 | 文献 Cd≈2.05, St≈0.14 | PARAMETRIC_CYLINDER(方) | 验证尖锐角处理 |

## D. 自由表面（dam_break 模块）

| # | 问题 | 参考基准 | 共性模块 | 备注 |
|---|------|---------|---------|------|
| B20 | 2D 溃坝波前位置 | Martin & Moyce 1952 | dam_break.py | 已有 bench_dam_break |
| B21 | 3D 溃坝水柱坍塌 | Martin & Moyce | dam_break_3d.py | 已有 bench_dam_break_3d |
| B22 | 溃坝波前时间关系 | 实验数据 | dam_break_quant.py | 定量 |

## E. 多相流（multiphase 模块）

| # | 问题 | 参考基准 | 共性模块 | 备注 |
|---|------|---------|---------|------|
| B23 | Laplace 定律（气泡压差） | Δp=σ/R 解析 | multiphase.py | 表面张力验证 |
| B24 | 静态液滴平衡 | Young-Laplace | multiphase | 形状精度 |

## F. 边界层/湍流（RANS/turbulence 模块）

| # | 问题 | 参考基准 | 共性模块 | 备注 |
|---|------|---------|---------|------|
| B25 | 平板层流边界层 | Blasius 解 | turbulent_channel/des | 速度剖面 |
| B26 | 湍流通道 Re_τ=180 | DNS 数据 (Kim 1987) | turbulent_channel.py | u+ 剖面 |
| B27 | RANS 后向台阶 | rans_ke.py | 工程验证 | 中等精度 |

## G. 声学/空化（专项模块）

| # | 问题 | 参考基准 | 共性模块 | 备注 |
|---|------|---------|---------|------|
| B28 | 圆柱涡脱落声 | 文献 SPL 谱 | acoustics.py | 气动声学 |
| B29 | 空化气泡 | Rayleigh-Plesset | cavitation.py | 相变验证 |

## 优先推荐（解析解/经典基准，最易达标 1%）

1. **B13 Poiseuille 流** — 解析解，验证边界条件+粘性，最易
2. **B15/B16 方腔流** — Ghia 经典基准，验证对流+扩散+角点
3. **B17 Taylor-Green** — 解析衰减，验证数值耗散（LBM 关键指标）
4. **B20 溃坝波前** — 实验基准，验证自由表面
5. **B23 Laplace 定律** — 解析，验证表面张力
6. **B25 Blasius 边界层** — 解析，验证壁面摩擦

## 判定标准（不变）

- 真实模拟（禁外推），误差 ≤1% 才保存到 benchmarks/verified/
- 通过共性模块（GeneralSimEngine 或各物理模块入口）
- 保存附运行命令/配置/参考值来源/误差
