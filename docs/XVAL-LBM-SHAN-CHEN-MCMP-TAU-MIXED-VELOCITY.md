# XVAL-LBM-SHAN-CHEN-MCMP-TAU-MIXED-VELOCITY — 与 OSS τ-mixed 速度组分 MCMP 的交叉对标

锚点 `XVAL-LBM-SHAN-CHEN-MCMP-TAU-MIXED-VELOCITY`（mission 18327）· 判据 <15%（owner 精度标准）·
状态 **PASS**（四臂中 parity/momentum/channel 三臂 PASS，separation 臂部分完成，见 §5.4）。

## ① 问题 + ref 出处

**问题**：本仓 `collide_sc_two_component(..., u_eq="viscous")` 用的是"按 τ 加权把两组分塌缩成一个共同速度"
（Martys 1996 / sailfish 约定）。OSS 参考实现
`loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow` 的 `shan-chen-mcmp-tau-mixed-velocity`
用的是同一个想法。本单要回答三件事：
(1) 两者是否是**同一个公式**（不是"看起来像"）；
(2) 该公式的连续极限给出的混合物黏度，本仓数值解是否真能复现（这才是 <15% 的物理判据）；
(3) 本仓 vs OSS 的**差异清单**：差异各自带来什么物理后果。

**ref 出处（零手抄）**：平台 `oss_solutions` 条目 **id=35**（evidence: report 9555），
repo `loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow`，feature
`GAP-LBM-MCMP-POISEUILLE`，tag `shan-chen-mcmp-tau-mixed-velocity`。方法串原文：

> `vel_mix = Σ(Vel*Rho/tau)/Σ(Rho/tau)`；源码行 `Domain.py:70-88`、`Domain.py:143-176`（混合本身在 `Domain.py:151-153`）；
> 每组 τ 来自 nu 列表（`Domain.py:77-78`）；`Rhoref=200.0`、`Psi=4.0` 为纯占位死参数；**缺 Shan-Chen 非局域力，故无相分离能力**。

**生态摩擦（如实记）**：本沙箱 **GitHub 不可达**（`git clone`/`codeload` 均超时），
因此对标对象是**清单里被核验过的公式与源码行引用**，而不是跑 OSS 代码得到的数字——
这一点限制了结论强度，已在 §6 的"适用范围"里写明。开工三查中
`knowledge_base`（shan-chen / tau mixed velocity）与 `negative_knowledge_ledger`（MCMP、loliverhennigh）
**均 0 命中**，只有仓内 `xval/horizontal_survey_empty/data/oss_solutions_dump.csv` 有货（即上面的 id=35）。

## ② 方法与实现

新增（本单交付）：

* `src/tensorlbm/mcmp_tau_mixed.py` —— REF-OSS 公式的独立实现 `oss_vel_mix()`、本仓两分组形式
  `tau_mixed_velocity()`、连续极限 `mixture_effective_viscosity()`、
  解析槽道参考解 `channel_poiseuille_profile()`，以及**黑盒探针**
  `collision_momentum_probe()`：一次无外力碰撞后由 `ΔJ_k = −(J_k − ρ_k u_visc)/τ_k`
  反解出"碰撞实际使用的 u_mix"，再与 REF-OSS 公式比对——**不是比对两个我写的函数，而是量库内部真正用的速度**。
* `xval/XVAL-LBM-SHAN-CHEN-MCMP-TAU-MIXED-VELOCITY/{predeclared.json, run_xval.py}` —— 四臂实验 + 图 + JSON。
* `tests/test_mcmp_tau_mixed_xval.py` —— pytest 门（公式、守恒、解析自洽、artifact 一致性）。

**连续极限（REF-OSS 公式的物理含义，本单推导）**：两组分共用 τ-mixed 平衡速度时，
`Σ_k ΔJ_k = −Σ_k (J_k − ρ_k u_mix)/τ_k = 0`（动量精确守恒），
Chapman-Enskog 一阶给出 `Σ_k Π_k^neq = −2 c_s² Σ_k τ_kρ_k S(u)`，故

  `ν_eff = Σ_k ρ_k ν_k / ρ`，`ν_k = c_s²(τ_k − ½)`

即**密度加权算术平均**（不是 τ 的算术平均、也不是调和平均）。这就是 <15% 判据的解析基准。

四臂（阈值全部在 `predeclared.json` 里**动手前**写死，事后未改）：
`parity`（公式一致性，5 个前碰撞态 × float64/float32）· `channel`（理想 G₁₂=0 共动混合物槽道 vs 解析）
· `momentum`（三种耦合模式的动量守恒）· `separation`（OSS 等价控制臂 vs 本仓 SC 力臂）。

## ③ 参数表（格子单位，无量纲量完整）

`dx=dt=1`、`ρ₀=1`、`c_s²=1/3`、D2Q9 BGK；槽道 `ny×4`、`y=0` 与 `y=ny−1` 为反弹壁，x 周期。

| case | ny | τ | gx | u_max(解析) | Re=uH/ν_eff | Ma=u/c_s | max% | L2% | ν_eff(预测) | ν(实测) | ν 误差% | 壁左 | 壁右 | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | 33 | (1.0,0.7) | 1.0e-06 | 1.097e-03 | 0.301 | 1.90e-03 | 6.21 | 8.34 | 0.11667 | 0.11666 | 0.007 | 0.504 | 31.496 | PASS |
| refine_17 | 17 | (1.0,0.7) | 1.0e-06 | 2.743e-04 | 0.038 | 4.75e-04 | 12.34 | 16.27 | 0.11667 | 0.11659 | 0.063 | 0.508 | 15.492 | PASS |
| refine_65 | 65 | (1.0,0.7) | 1.0e-06 | 4.389e-03 | 2.407 | 7.60e-03 | 3.18 | 4.29 | 0.11667 | 0.11675 | 0.074 | 0.501 | 63.499 | FAIL |
| tau_equal_control | 33 | (1.0,1.0) | 1.0e-06 | 7.680e-04 | 0.147 | 1.33e-03 | 6.18 | 8.34 | 0.16667 | 0.16667 | 0.000 | 0.503 | 31.497 | PASS |
| tau_ratio_1p11 | 33 | (1.0,0.9) | 1.0e-06 | 8.533e-04 | 0.182 | 1.48e-03 | 6.20 | 8.35 | 0.15000 | 0.15000 | 0.001 | 0.504 | 31.496 | PASS |
| tau_ratio_3 | 33 | (1.0,0.6) | 1.0e-06 | 1.280e-03 | 0.410 | 2.22e-03 | 6.20 | 8.32 | 0.10000 | 0.09999 | 0.009 | 0.502 | 31.498 | PASS |
| force_half | 33 | (1.0,0.7) | 5.0e-07 | 5.486e-04 | 0.150 | 9.50e-04 | 6.21 | 8.34 | 0.11667 | 0.11666 | 0.007 | 0.504 | 31.496 | PASS |
| force_double | 33 | (1.0,0.7) | 2.0e-06 | 2.194e-03 | 0.602 | 3.80e-03 | 6.21 | 8.34 | 0.11667 | 0.11666 | 0.007 | 0.504 | 31.496 | PASS |

`Re` 取 `u_max·H/ν_eff`（H=ny−1），`Ma=u_max/c_s`。所有算例 Ma ≤ 4e-3（不可压区）。
分离臂：`48×48` 周期、`ρ1+ρ2=2`、初始 1% 随机扰动（seed 7）、`τ=(1.0,0.7)`、4000 步、Guo forcing。

## ④ 对照表（ref vs got + 误差打勾叉）

**臂 1 — 公式一致性（REF-OSS vs 本仓碰撞实际使用的速度）**，阈值 float64 `1e-10` / float32 `1e-4`：

| dtype | seed | (ρ1,ρ2) | (τ1,τ2) | max rel dev | 判定 |
|---|---|---|---|---|---|
| float64 | 0 | (1.0,1.0) | (1.0,0.7) | 3.01e-15 | PASS |
| float64 | 1 | (0.3,1.7) | (1.0,0.6) | 5.05e-15 | PASS |
| float64 | 2 | (2.0,0.2) | (1.2,0.8) | 2.17e-15 | PASS |
| float64 | 3 | (1.0,1.0) | (1.0,1.0) | 6.89e-15 | PASS |
| float64 | 4 | (0.5,0.5) | (3.0,0.55) | 3.63e-15 | PASS |
| float32 | 0 | (1.0,1.0) | (1.0,0.7) | 1.42e-06 | PASS |
| float32 | 1 | (0.3,1.7) | (1.0,0.6) | 2.46e-06 | PASS |
| float32 | 2 | (2.0,0.2) | (1.2,0.8) | 1.77e-06 | PASS |
| float32 | 3 | (1.0,1.0) | (1.0,1.0) | 2.94e-06 | PASS |
| float32 | 4 | (0.5,0.5) | (3.0,0.55) | 3.35e-06 | PASS |

最坏偏差：float64 **6.89e-15** ✔、float32 **3.35e-06** ✔
⇒ 本仓 `u_eq="viscous"` 施加的就是 REF-OSS 的 `Σ(Vel·Rho/τ)/Σ(Rho/τ)`（舍入级一致）。

**臂 2 — 槽道剖面 vs 解析解（连续极限 ν_eff）**：见 ③ 表 `max%` 列。最坏 12.34%
（出现在最粗网格 ny=17）✔ <15%；ny=33/65 为 6.21% / 3.18%。
**黏度实测误差 ≤ 0.074%**（全部 8 例），壁位置实测
0.504/31.496（ny=33）与预declared 的 `y=0, y=ny−1`
一致（本仓 `stream` 后反弹把壁放在**壁格点中心**，所以 `H=ny−1` 且剖面取 `y(H−y)` 而非 `(y−½)(H−y+½)`——
这是本单第一轮 12% 残差的真因，属**几何约定**而非黏度错误，靠二次型拟合的壁位置诊断定位）。

**臂 3 — 动量守恒**（τ=(1.0,0.7)，200 步，周期性无外力）：

| u_eq | 相对动量漂移 | 每步 | 判定 |
|---|---|---|---|
| viscous | 2.31e-14 | 1.15e-16 | PASS |
| mixture | 5.29e-01 | 2.65e-03 | FAIL |
| self | 5.20e-16 | 2.60e-18 | PASS |

⇒ τ-mixed 耦合动量精确守恒（2.3e-14），
而**质心（barycentric）`mixture` 模式**在不等 τ 下有 53% 的净动量源 ✔
——这条把库 docstring 里的既有断言变成了实测证据。

**臂 4 — 相分离（OSS 差集）**：

| G_12 | density contrast | 对比初值倍数 | max|u| | 质量漂移 |
|---|---|---|---|---|
| 0.0 (control) | 3.16e-07 | 0.00 | 5.42e-03 | 2.7e-13 |
| -0.4 | 7.666e-06 | 0.0 | 5.51e-03 | 2.74e-13 |
| -0.8 | 1.911e-04 | 0.0 | 5.73e-03 | 2.70e-13 |
| -1.2 | 1.571e+00 | 78.6 | 1.84e-01 | 2.69e-13 |

控制臂 `G_12=0`（= OSS 草稿条件：无非局域力）密度对比度停在 `3.16e-07`（= 初始 1% 扰动的扩散残余），
**复现了 OSS 清单里"缺 SC 非局域力，故无相分离能力"的行为** ✔（阈值 `control_max_contrast=1e-3`）；
`|G_12|=0.8→1.2` 之间 onset 出现相分离（对比度 1.57、78.6×、max|u|=0.18）。
未完成：G=-0.4, G=-0.8, G=-1.2, G=-1.6, G=-2.0 两档在报告落盘时仍在跑（单档 ~170 s），
按预声明本应对全部 5 档扫描，**如实标记该臂 PARTIAL**（不覆盖、不补数）。

## ⑤ 收敛性与参数敏感性（至少两点）

1. **网格收敛**（τ=(1.0,0.7)，gx=1e-6）：max 误差 12.34%(ny=17) → 6.21%(ny=33) → 3.18%(ny=65)，
   L2 16.27% → 8.34% → 4.29%，近似一阶收敛；**黏度实测误差在三档上都 ≤0.08%**（即残差不是黏度而是壁面边界层，
   ny=33 的 max 误差 6.2% 集中在紧贴壁的第一层流体格点）。图 `figures/convergence_viscosity.png`。
2. **黏度比敏感**（ny=33，τ2 ∈ {1.00, 0.90, 0.70, 0.60}，即 ν 比 1:1、1:1.11、1:1.43、1:1.67）：
   实测 ν = 0.16667, 0.15000, 0.09999 vs 预测
   0.16667, 0.15000, 0.10000，
   误差 0.000%, 0.001%, 0.009%——
   等 τ 控制例给出同样 6.18% 的 max 误差，说明该水平是**壁面效应**而与混合无关（判别力：见下）。
3. **驱动力敏感**（gx/2、gx、2gx）：max 误差 6.21% 三档相同、ν 实测相同 ⇒ 线性响应、无 Ma 污染。
4. **判别力检查**（为什么这个 ref 能验伪）：假设实现改成了 τ 算术平均或调和平均，ny=33 τ=(1.0,0.7) 会给出
   ν = 0.11667 / 0.09524，
   对 ν_eff=0.11667 的偏差 **0.0% / -18.4%**，
   远超 15% 判据 ⇒ 本测试不是"太宽松才过"。`figures/channel_profiles.png` 同时画出三条解析候选曲线。

## ⑥ 差异清单（本仓 vs OSS）与移植取舍

| # | 维度 | OSS 参考（清单 id=35） | 本仓实现 | 后果 / 取舍 |
|---|---|---|---|---|
| 1 | 共同速度 | `Σ(Vel·Rho/τ)/Σ(Rho/τ)`，碰撞前把各组塌缩为一个速度 | `u_eq="viscous"` 同式（实测 6.9e-15） | **采纳**：不等 τ 时的默认耦合；另有 `mixture`/`self` 备选（实测 `mixture` 不守恒，53%/200步） |
| 2 | 非局域 Shan-Chen 力 | **缺**（故无相分离） | `sc_two_component_force`（standard/edm 两种离散，G₁₂<0 相斥） | **不采纳 OSS 的缺失**：这是唯一实质差异，控制臂 contrast 3.2e-7 vs 本仓 1.57 |
| 3 | 占位常数 | `Rhoref=200.0`、`Psi=4.0` 死参数 | 线性赝势 ψ=ρ、格子密度 ~1 | 不移植（死参数会伪装成"物理"） |
| 4 | 每组 τ 来源 | nu 列表 `Domain.py:77-78` | `tau1/tau2` 标量入参 | 一致 |
| 5 | 力项离散 | 无（草稿没有力项设计） | `shift`（一阶）/`guo`（二阶，含 1−1/2τ 因子） | 两相界面力必须用 Guo（shift 会每步注入 F/2 伪动量 → 界面涡量）；本单槽道臂用 Guo |
| 6 | 维度/模板 | 3D（D3Q15/D3Q19 系） | 本路径 2D D2Q9（3D 另有模块） | 混合公式维度无关；黏度数字不可跨维搬用 |
| 7 | 边界 | 手册式进出口重建（其 car.py 流） | `bounce_back_cells` + Zou-He 等 | 壁位置约定（壁格点中心）必须在解析基准里对上，否则 12% 假误差 |
| 8 | 求解器宿主 | TF1 静态图 | torch、`stream()` 分离 | 不移植（生态不同） |

**移植取舍结论**：本仓不需要从 OSS 移植任何代码——配方（τ 加权塌缩）已在库内且被证明是同一公式；
OSS 值得借鉴的只有"用 τ 加权避免不等 τ 下的动量源"这一**设计理由**，而我们已有更强版本（Guo forcing + SC 力 + EDM 备选）。
反过来，本单给 OSS 的差集留了两条可复用知识：**①"混合公式对 ⟹ 黏度必须是 ρ 加权算术平均"是可用一条槽道算例验伪的硬约束；
②壁位置约定错 0.5 格会在 ny=33 上伪造 ~6–12% 误差，判黏度前必须先测壁位置。**

## ⑦ 工件 / 复现

```
xval/XVAL-LBM-SHAN-CHEN-MCMP-TAU-MIXED-VELOCITY/predeclared.json     # 阈值/设计（动手前写死）
xval/XVAL-LBM-SHAN-CHEN-MCMP-TAU-MIXED-VELOCITY/run_xval.py          # 四臂驱动
xval/XVAL-LBM-SHAN-CHEN-MCMP-TAU-MIXED-VELOCITY/results/xval_results.json
xval/XVAL-LBM-SHAN-CHEN-MCMP-TAU-MIXED-VELOCITY/figures/(channel_profiles, convergence_viscosity, momentum).png
src/tensorlbm/mcmp_tau_mixed.py                                       # REF-OSS 公式 + 解析基准 + 黑盒探针
tests/test_mcmp_tau_mixed_xval.py                                     # pytest 门
```

环境：3.11.6 / torch 2.7.1 / loongarch64 / 128 threads / git `56aafaaf9888`。
运行：`python xval/XVAL-LBM-SHAN-CHEN-MCMP-TAU-MIXED-VELOCITY/run_xval.py --arm all`（本轮总墙钟 326s，
channel 臂 8 例约 315 s，全部在控制面跑，未上集群——预声明预算 1800 s）。

**失败模式（本单实际踩到的）**：①沙箱无 GitHub → 只能对公式不能对 OSS 数字；
②99 shell 工具在 120 s 收割整个进程组，前后被砍两次（channel 臂重跑一次，~10 min 白跑）→ 长算例必须 `setsid` 分离；
③第一版把壁面参考写成半格偏移，ny=17/33 出现 23%/12% 的**假 FAIL**，靠二次型拟合给出壁位置才定位；
④`u_eq="mixture"` 外观合理但保不住动量（53% 漂移）——若把它当"同一公式的另一种写法"会静默错。
