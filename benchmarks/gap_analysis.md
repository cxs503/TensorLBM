# CFD 常用 Benchmark 分析——当前覆盖与缺口

> 对照：OpenLB examples（克隆源码）/ Palabos / 经典 CFD 验证体系（Schäfer-Turek、Ghia、
> Martin-Moyce、Kim-Moin-Moser 等）。分类列出：✅ 已达标、🔶 已测未达标、❌ 不具备。

## 1. 层流内部流（解析解/经典基准）

| Benchmark | 参考 | 覆盖 | 状态 |
|-----------|------|------|------|
| **2D Poiseuille**（平板管道） | 解析抛物线 | ✅ verified 0.18% | ✅ |
| **3D Poiseuille**（OpenLB poiseuille3d） | 解析抛物线 | 🔶 pending（近壁 staircase 15.5%→7.5%） | R_fit 修正后可达 |
| **Couette**（平板剪切） | 解析线性 | ✅ verified 0.015% | ✅ |
| **方腔流 Re=100**（OpenLB cavity2d） | Ghia 1982 | 🔶 pending（顶盖 BC 22.5%） | 角点修复 |
| **方腔 Re=400/1000** | Ghia 1982 | ✅ verified（Re=400 0.83%、Re=1000 1.05%，2026-08-19） | Re=1000 需 RLBM 碰撞（MRT 失稳） |
| 3D 方腔（OpenLB cavity3d） | 3D Ghia | ❌ 不具备 | — |
| **后向台阶**（OpenLB bstep2d/3d） | Armaly 1984 | 🔶 pending（12.2%） | 加密/入口 |
| 3D 后向台阶（OpenLB bstep3d） | Armaly | ❌ 不具备 | — |
| **功率律流体**（OpenLB powerLaw2d） | 解析 | ❌ **不具备** | 非牛顿模型需查 |

## 2. 层流外部流（绕流）

| Benchmark | 参考 | 覆盖 | 状态 |
|-----------|------|------|------|
| **圆柱 Re=20**（Schäfer-Turek 2D-1） | Cd=5.5795 | 🔶 pending（R_eff 网格依赖） | R_eff 修复 |
| **圆柱 Re=100**（自由流涡脱落） | Braza Cd=1.35 | 🔶 pending（播种后 NaN） | 出口吸收层 |
| 圆柱 Re=200（OpenLB 同系） | Braza Cd≈1.33 | ❌ **不具备** | 依赖 Re=100 |
| **圆柱 3D**（OpenLB cylinder3d） | 3D 文献 | ❌ 不具备 | — |
| **球 Re=100/200** | SN 1.087/0.806 | 🔶 pending（力法 G12/G15） | MEM 修正 |
| **方柱 Re=100** | Sohankar 2.05 | 🔶 pending（-26.7% 口径） | 口径确认 |
| **SUBOFF 潜艇** | 实验/数值 | 🔶 pending（-9.85%） | L≥128 |
| NACA 0012 翼型 | Cl 文献 | 🔶 pending（流态网格依赖） | 低攻角可试 |

## 3. 时变/耗散验证

| Benchmark | 参考 | 覆盖 | 状态 |
|-----------|------|------|------|
| **Taylor-Green 2D**（OpenLB tgv 系） | 解析衰减 | ✅ verified -0.035% | ✅ |
| **剪切波衰减** | 解析 e^(-νk²t) | ✅ verified 0.011% | ✅ |
| **Kovasznay 稳态** | 解析 N-S 解 | ✅ verified 2.71% | ✅ |
| **Stokes 第一问题**（起动平板） | 解析 erf | ✅ verified 0.08% | ✅ |
| **3D Taylor-Green**（OpenLB tgv3d） | 解析衰减 | ❌ **不具备** | 3D 周期域，湍流前验证 |

## 4. 多相/自由表面（OpenLB multiComponent）

| Benchmark | 参考 | 覆盖 | 状态 |
|-----------|------|------|------|
| **Laplace 液滴**（youngLaplace2d/3d） | Δp=σ/R | ✅ verified 0.8% | ✅ |
| **接触角**（contactAngle2d/3d） | Young 定律 | ❌ **不具备** | 需多相修复 |
| **相分离**（phaseSeparation2d/3d） | 理论共存 | ❌ 不具备 | EOS 参数修复（G3/G4） |
| **Rayleigh-Taylor**（rayleighTaylor2d/3d） | 增长率理论 | ❌ **不具备** | 需大密度比 |
| **溃坝**（dam break） | Martin-Moyce | 🔶 多相缺口阻塞 | G9 大密度比 |
| 微流控（microFluidics2d） | 应用 | ❌ 不具备 | — |

## 5. 湍流/工程（OpenLB turbulence）

| Benchmark | 参考 | 覆盖 | 状态 |
|-----------|------|------|------|
| **湍流通道 Re_τ=180** | MKM DNS | 🔶 pending（2D 无转捩） | 需 3D（G10） |
| **衰减各向同性湍流 DIT** | DNS 谱 | ❌ **不具备** | 3D 周期域 + LES |
| 主动脉流（aorta3d） | 医学 CFD | ❌ 不具备 | 工程应用 |
| 喷嘴/文丘里（nozzle3d/venturi3d） | 实验 | ❌ 不具备 | 工程应用 |

## 6. 边界层/其他

| Benchmark | 参考 | 覆盖 | 状态 |
|-----------|------|------|------|
| **Blasius 平板** | f'(η) 解析 | 🔶 pending（前缘台阶 +48%） | 中置薄板 |
| 方腔高 Re（400/1000） | Ghia | ❌ 不具备 | — |
| 圆柱涡脱落 St（声学源） | Roshko | 🔶 pending（6.8%） | 格子离散 |

## 缺口总结（❌ 不具备的，按优先级）

### P0（修复一个解锁多个）
1. **MEM 曲面力修正**（G12/G15）——解锁：球 Re=100/200、圆柱 Re=100/200、方柱（5 个 pending）
2. **R_eff 修正**（half-way BB 壁面位置）——解锁：圆柱 Re=20、3D Poiseuille（2 个）

### P1（新案例，共性模块可达）
3. **方腔 Re=400/1000**（Ghia 表值齐全，库内 lid_driven_cavity 现成）
4. **3D Taylor-Green**（OpenLB tgv3d 同款，3D 周期域，湍流前验证）

### P2（需模块开发）
5. **Rayleigh-Taylor**（多相大密度比，G9）
6. **接触角**（Young 定律，需多相修复）
7. **DIT 衰减湍流**（3D 周期域 + LES）
8. **非牛顿 powerLaw**（需模型开发）

## 结论

- **已覆盖**：解析解类最全（Poiseuille/Couette/TG/剪切波/Kovasznay/Stokes1/Laplace——7 个），与 OpenLB 的 laminar+解析系持平
- **最大缺口**：**绕流力法**（MEM/R_eff 两个修复解锁 7 个 pending）——这是"算得准"的关键路径
- **第二缺口**：**多相动态**（RT/接触角/相分离）——被 EOS 参数（G3/G4）阻塞
- **OpenLB 有而我们没有的**：3D 圆柱、3D 方腔、3D 后向台阶、powerLaw、contactAngle、RT、微流控、主动脉、喷嘴——多为工程应用类
