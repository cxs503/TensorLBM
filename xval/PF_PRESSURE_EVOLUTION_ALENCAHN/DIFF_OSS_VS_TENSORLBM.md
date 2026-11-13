# 差异清单：TensorLBM 移植 vs CFD-GO/TCLB `d2q9_pf_pressureEvolution`

锚点 `XVAL-LBM-PRESSURE-EVOLUTION-ALLENCAHN-PHASEFIELD`（m18194）。
OSS 侧数字全部由 `scripts/xval_m18194_pf_pressure_evolution.py` 在运行时从源码文本抽取
（`artifacts/xval_m18194/oss_extract.json` 落 sha256 + 行号），零手抄。
逐条数字与判据见 `docs/XVAL-LBM-PRESSURE-EVOLUTION-ALLENCAHN-PHASEFIELD.md`。

## 一、做法差异（结构化，P1–P14 全部 residual ≤ 8.3e-17）

| # | 环节 | TCLB（抽取原文，行号见 oss_extract.json） | TensorLBM 移植 | 差异 |
|---|---|---|---|---|
| P1 | 矩矩阵 | `M = matrix(c(...),9,9)`，列主序 = 9 个矩 | `D2Q9_MOMENTS`（行=矩） | 等价（residual 0） |
| P2 | 松弛矩 | `S[7] *= 1/(tau+0.5); S[8] *= ...`（0 基） | `VISCOUS_MOMENTS=(7,8)` = p_xx, p_xy | 等价 |
| P3 | gbar 平衡态 | `G[i]*rho/3.0 + w[i]*(pressure-rho/3.0)` | `gbar_equilibrium()` | 等价（resid 0） |
| P4 | 速度 | `(3.0/rho)*(sum f c + (0.5/3.0)(mu grad + F_b))` | `velocity()` | 等价（8.3e-17） |
| P5 | 压力回收 | `sum(f) + dRho*(grad.u)/6.0` | `pressure_from_populations()` | 等价（0） |
| P6 | 化学势 | `4.0*(12.0 sigma/W)(...)-1.5 sigma W lap`，lap=(角+4边−20心)/6 | `chem_potential()`（各向同性 9 点拉普拉斯） | 等价（2.1e-17） |
| P7 | 梯度 | `(φ_(1,0)−φ_(−1,0))/3 + 对角/12` | `isotropic_gradient()` | 等价（8.3e-17） |
| P8 | 定向差分（碰撞前一半） | `Rc[i] = 0.5(φ_(+c)−φ_(−c))` | `directional_centred()` | 等价（0） |
| P9 | 定向差分（碰撞后一半） | `Rm[i]=0.25(−φ_(+2c)+5φ_(+c)−3φ_0−φ_(−c))` | `directional_mixed()` | 等价（0） |
| P10 | h 平衡态 | `w_i φ[1+3c·u+4.5(c·u)²−1.5u²] + θ w_i (c·n)` | 同 | 等价（2.8e-17） |
| P11 | 反扩散幅值 | `θ = 3M(1−4(φ−φ_avg)²)/W` | 同 | 等价（0） |
| P12 | 相场松弛率 | `ω_φ = 1/(3M+0.5)` | 同 | 等价（0） |
| P13 | 初始界面 | `½(1 − tanh(2(r−R)/W))` | `tanh_drop()` | 等价（0） |
| P14 | 参数默认 | `W=4, M=0.05, Viscosity_l=0.16666666, omega_l=1/(3 Viscosity_l)` | 本 XVAL 取 ν=1/6（与 OSS 默认一致） | 一致 |

## 二、参数与边界条件差异

| 项 | TCLB | TensorLBM 本单 | 影响 |
|---|---|---|---|
| 边界 | `Solid`/`Wall` 全反弹、`EVelocity`/`WPressure`/`WVelocity`/`EPressure` | **仅周期**，壁面未移植 | 圆/方腔、RT 算例本单不做（见报告 §6） |
| 流式计算 | CUDA C，templated 生成器（Python 2） | PyTorch float64 CPU | 精度更高（G3 mass 6e-13）；速度慢 ~10 倍 |
| 碰撞 | MRT（BGK 分支只 `printf("NOT IMPLEMENTED")`） | MRT | 同 |
| 体积力 | `Buoyancy*`、`Gravitation*`、`Gmatched*` | `body` 参数同构（本单置 0） | 未验证 |
| 浮力/重力耦合算例 | 报头称已测 layered Poiseuille / RT / 平面气泡 | 未做 | 二轮 |
| 时刻顺序 | stage：`BaseIter`(碰撞+流式) → `calcPhase` | 同（h 用碰撞后 f 的速度） | 同（P10 用新 u） |
| ν 场插值 | `tau = 1/(ω_l + Δω·phase)`，与 ρ 同线插值 | 同 | ratio 1e3 下黏度场非调和，OSS 继承 |

## 三、移植取舍（建议）

1. **采纳**：f 群承压 + gbar 平衡态 + 仅两个黏性矩松弛的 MRT；AC h 群把反扩散项放进平衡态（Σθw(c·n)=0 ⇒ 相质量逐点精确守恒，本单实测 6e-13）；
   碰撞前用中心差分、碰撞后用混合（迎风）差分这一对"半力"拆分。
2. **保留 TensorLBM 自有**：p 仅作为端口变量 `PFPressureEvolution`，不改动既有 Shan-Chen / CAC 核；float64 torch 与既有 `phasefield` 模块共用 `_shift/_neighbours` 与各向同性算子。
3. **不移植**：节点类型边界机（`lib/boundary.R` 的 `FullBounceBack` 等）、MRT 的 `S0..S6` 占位可调参数（OSS 中注释掉）、BGK 分支。
4. **需注意**：混合差分对 Σ_i I_i 只是二阶守恒（本单实测源项 3.0e-5 vs 力标度 3.0e-4 ≈ 10%，W=4），它是 ratio=1 算例压力缓慢漂移（6000 步仍 −1.9%）的主因；如需长时间精确静压平衡，建议改用中心差分 + 时间半步，或在后处理中扣除源项。
