# XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE — 两组分 Shan-Chen 交叉力算符交叉验证

> 锚点 `XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE`，mission 18101，repo TensorLBM，kernel commit `fc86f5080e86`，torch 2.7.1 / numpy 2.4.6，dtype float64。
> 判据（总账）：**<15%**（owner 精度标准）。预登记阈值见 `predeclared.json`；全部数字由 `run_xval.py` 写入 `artifacts/summary.json`，本文件由 `make_report.py` 程序化渲染（零手抄）。

## 判定一览

| 判据 | 门限 | 实测 | 判定 |
|---|---|---|---|
| 主判据：真实求解态（松弛分层界面）vs 谱精确梯度 | 15.0% | 7.01% (standard) / 6.43% (edm) | PASS |
| 次判据：解析单模扫描（≥8 cell/波长）vs 谱精确梯度 | 15.0% | 9.97% (axis) / 7.75% (diagonal) | PASS |
| 不变式：kernel == OSS 前向 gather stencil（g_oss=-G12） | 1e-6 | 7.45e-09 | PASS |
| 不变式：实测误差 == 闭式符号 \|1-S(k)\| | 1e-6 | 7.44e-09 | PASS |
| 不变式：均匀场零力 | 1e-14 | 0.00e+00 | PASS |
| 不变式：D2Q9 对称等变（旋转 90°/转置） | 1e-12 | 7.33e-16 | PASS |
| 不变式：周期域净力（总动量）标准式 | ≤1e-13 | 1.83e-17 | PASS |
| 可证伪性：6 个种植缺陷均被 15% 门捕获 | 全部 ≥15% | 最小 12.45%（5/6 ≥15%） | **FAIL（如实）** |

**总判定**：精度门（主/次判据）**PASS**；不变式 6/6 **PASS**；可证伪性判据**部分达标**（见 ⑥）。`gate_pass` = False（该字段要求全部检查通过，含如实记 FAIL 的可证伪性项）。

## ① 问题设定与基准出处

**物理对象**。TensorLBM 的 D2Q9 两组分 Shan-Chen（MCMP）模型里，组分 σ 受到另一组分的交叉力驱动：`src/tensorlbm/multiphase.py:534` 的 `sc_two_component_force`，由 `collide_sc_two_component`（:`788`）在每次碰撞前调用。它决定界面张力、假速度（spurious currents）与相分离阈值，是 MCMP 路线唯一的相间耦合项（`G_aa = G_bb = 0`，只有交叉项 `G_12`）。本单验证的是**这个算符本身**与其连续极限的一致性，不是模型级（线性 ψ 的）表面张力精度。

**ref（三条互不共享代码路径的参考）**：

1. `ref/oss`（实现横向）：开源 MCMP 实现的**前向 gather**写法 `F_σ = -g ψ_σ Σ_i w_i c_i ψ_σ'(x + c_i)`，取自 `omalaspinas/palabos` `src/multiPhysics/shanChenProcessor3D.hh:304`（`t_i` 各向同性权重，ψ 由策略给定，`PsiIsRho` 即线性）与 `sorush-khajepor/listLBM` `mcmp_mod.f90`。TensorLBM 写成**反向 gather** `-G_12 ρ_σ Σ_i w_i c_i ρ_σ'(x - c_i)`，故约定映射应为 `g_oss = -G_12`（上游正=排斥，本仓负=排斥）。
2. `ref/symbol`（闭式）：同一 stencil 的 D2Q9 符号 `S(k) = Σ_i w_i (c_i·k̂) sin(k·c_i) / (c_s²|k|)`，轴上退化为教科书的 `sin(k)/k`，所以离散算符的截断误差是**可计算量**而不是拟合量。
3. `ref/spectral`（连续极限）：采样场的精确梯度（Fourier 符号 `i k`，周期场下对光滑场超代数精度），交叉力的连续极限由 ② 推导为 `F_1 = G_12 c_s² ρ_1 ∇ρ_2`。

**方法谱系出处（external_anchor）**：Shan & Chen 1993 PRB 47 1815 / 1994 PRE 49 2941（伪势耦合原始式）；Martys & Chen 1996 PRE 53 743（两组分压力关系）；Kupershtokh, Medvedev & Karpov 2009 Comput. Math. Appl. 58 965（EDM 精确差分分裂，即本仓 `sc_form="edm"`）；Sbragaglia et al. 2009 PRE 79 057701（各向同性 stencil 与假速度）；Krüger et al. *The Lattice Boltzmann Method: Principles and Practice* 2017 第 9 章（伪势多相与力项）。**本机无外网，页码级引用不可核验，故只著录卷期页到文献级，不伪造页码**；可核验面 = 上述三条可复算的参考对象。

## ② 方法与实现

**连续极限推导**（决定 ref/spectral 的形式，也决定符号）。D2Q9 下 `Σ_i w_i c_i f(x - c_i) ≈ -c_s² ∇f + (c_s⁴/6) ∂_α∇²f + …`，因此

    F_1 = -G_12 ρ_1 Σ_i w_i c_i ρ_2(x - c_i) = +G_12 c_s² ρ_1 ∇ρ_2 + O(h²),   c_s² = 1/3

G_12 < 0 时力指向上游方向（背离富组分 2 的区域）⇒ 排斥 ⇒ 分相。两者的**总和** `F_1+F_2 = G_12 c_s² ∇(ρ_1ρ_2)` 是纯梯度，故与库内 `sc_mcmp_pressure`（`p = c_s²(ρ_1+ρ_2) + G_12 c_s² ψ_1ψ_2`）自洽：`∇p = F_1+F_2`。这一步把 ref/symbol 与 ref/spectral 联系起来，也说明**本单测的是算符，不是 EOS 常数**。

**实现与代码指针**：

| 环节 | 文件 | 提供量 |
|---|---|---|
| 被测核 | `src/tensorlbm/multiphase.py:534` `sc_two_component_force`（:`788` 被求解器调用） | `(Fx1,Fy1,Fx2,Fy2)` |
| OSS 参考 | `xval/.../xval_lib.py::oss_force` | 前向 gather 独立 NumPy 实现（float64，自带权重表） |
| 闭式符号 | `xval_lib.py::symbol_ratio` | `S(k)` |
| 连续参考 | `xval_lib.py::spectral_force` | `G_12 c_s² ρ_1 ∇ρ_2` |
| 真实求解态 | `tensorlbm.mcmp_verification.simulate_planar_slab` | 松弛后的两组分分层场 + `u_max` 平台判据 |
| 驱动 | `run_xval.py`（arms A–F） | `artifacts/summary.json` |
| 测试 | `tests/test_xval_mcmp_cross_force.py`（9 项，全绿） | 不变式断言 |

**六条臂**：A 单模扫描（轴/对角，闭式对照）；B 分辨率细化（固定物理波长，n=32…256）；C 真实求解态（松弛分层界面，n=64/96，G_12=-2.5/-3.0，standard 与 edm 两种离散）；D 动量（作用-反作用）缺陷；E 不变式（均匀场零力、对称等变、OSS 恒等、闭式恒等）；F 可证伪性（在独立参考实现上种植 6 个缺陷）。

## ③ 参数表（lattice units，无量纲齐全）

| 参数 | 值 | 说明 |
|---|---|---|
| 格子 | D2Q9，`c_s²=1/3` | 权重 `4/9, 1/9×4, 1/36×4` |
| 伪势 | `ψ(ρ) = ρ`（线性） | 本仓 MCMP 约定（同 Palabos `PsiIsRho`） |
| 耦合 `G_12` | −2.5（解析臂、C 臂之一），−3.0（C 臂之二） | 负=排斥=分相 |
| 松弛时间 | τ₁ = τ₂ = 1.0 ⇒ ν = 1/6 | C 臂用库默认 velocity-shift forcing |
| 网格 | 解析臂 n=64（k = 2πm/n，m=1…31）；细化 n=32/64/128/256；C 臂 n=64/96 | 全周期，无壁面无体力 |
| 步数 / 收敛判据 | C 臂上限 4000/3000/3000，`stop_tol=1e-4`（`u_max` 连续两窗口相对变化） | 步数与 `u_max` 逐例报告 |
| 密度范围（实测） | n=64,G12=-2.5: ρ₁∈[0.133,0.896]；n=96,G12=-2.5: ρ₁∈[0.137,0.882]；n=96,G12=-3.0: ρ₁∈[0.066,0.957] | 无发散 |
| 无量纲数 | Ma = u_max/c_s ≤ 0.0370（C 臂实测 u_max ≤ 0.0213） | 远小于 0.1，不可压近似成立；解析臂无流动（纯算符） |
| 浮点/设备 | float64 / CPU，1 线程 | 库内权重表为 float32，故恒等判据取 1e-6（实测 7.5e-09） |

## ④ 对照表（逐项 ref vs got + 误差）

**表 1｜单模扫描（ref = 谱精确梯度；`err_closed = |1-S(k)|` 为闭式预测）**

| 方向 | m | k [rad/cell] | λ [cell] | got/ref | `S(k)` | err 实测 | err 闭式 |
|---|---|---|---|---|---|---|---|
| axis | 1 | 0.0982 | 64.00 | 0.998394400 | 0.998394393 | 0.16% | 0.16% |
| axis | 2 | 0.1963 | 32.00 | 0.993586859 | 0.993586851 | 0.64% | 0.64% |
| axis | 4 | 0.3927 | 16.00 | 0.974495366 | 0.974495358 | 2.55% | 2.55% |
| axis | 5 | 0.4909 | 12.80 | 0.960321554 | 0.960321547 | 3.97% | 3.97% |
| axis | 8 | 0.7854 | 8.00 | 0.900316323 | 0.900316316 | 9.97% | 9.97% |
| axis | 10 | 0.9817 | 6.40 | 0.846927999 | 0.846927993 | 15.31% | 15.31% |
| axis | 12 | 1.1781 | 5.33 | 0.784213309 | 0.784213304 | 21.58% | 21.58% |
| axis | 16 | 1.5708 | 4.00 | 0.636619777 | 0.636619772 | 36.34% | 36.34% |
| axis | 20 | 1.9635 | 3.20 | 0.470527986 | 0.470527982 | 52.95% | 52.95% |
| diagonal | 1 | 0.1388 | 45.25 | 0.996791886 | 0.996791879 | 0.32% | 0.32% |
| diagonal | 2 | 0.2777 | 22.63 | 0.987223028 | 0.987223020 | 1.28% | 1.28% |
| diagonal | 4 | 0.5554 | 11.31 | 0.949769018 | 0.949769011 | 5.02% | 5.02% |
| diagonal | 5 | 0.6942 | 9.05 | 0.922523702 | 0.922523695 | 7.75% | 7.75% |
| diagonal | 8 | 1.1107 | 5.66 | 0.812417474 | 0.812417468 | 18.76% | 18.76% |
| diagonal | 10 | 1.3884 | 4.53 | 0.721461328 | 0.721461322 | 27.85% | 27.85% |
| diagonal | 12 | 1.6661 | 3.77 | 0.622844020 | 0.622844015 | 37.72% | 37.72% |
| diagonal | 16 | 2.2214 | 2.83 | 0.424413185 | 0.424413182 | 57.56% | 57.56% |
| diagonal | 20 | 2.7768 | 2.26 | 0.253664236 | 0.253664234 | 74.63% | 74.63% |

闭式恒等最大偏差 7.44e-09；符号恒等最大偏差 7.44e-09。轴向上 `err → |1 - sin(k)/k|` 精确成立，这就是教科书级二阶中心差截断。

**表 2｜真实求解态（ref = 松弛后密度场的谱精确梯度）**

| G_12 | n | 步数 | 收敛 | u_max | err standard | err edm |
|---|---|---|---|---|---|---|
| -3.0 | 96 | 3000 | False | 0.0213 | 7.01% | 6.43% |
| -2.5 | 64 | 2250 | True | 0.0092 | 3.65% | 3.54% |
| -2.5 | 96 | 3000 | False | 0.0088 | 3.49% | 3.38% |

**表 3｜不变式（exact references）**

| 项 | got | 门限 |
|---|---|---|
| 均匀场交叉力 | 0.0e+00 | ≤1e-14 |
| kernel vs OSS stencil（g_oss=-G_12） | 7.45e-09 | ≤1e-6 |
| 同上但**错误映射**（g_oss=+G_12）：证明映射不是可任意选的 | 200.0% | — |
| 旋转 90° 等变 | 7.3e-16 | ≤1e-12 |
| 转置等变 | 7.1e-16 | ≤1e-12 |
| 周期域净力（standard / edm） | 1.83e-17 / 1.83e-19 | ≤1e-13 |
| 逐点不抵消 `mean|F₁+F₂|/mean(|F₁|+|F₂|)`（standard / edm） | 0.7446 / 0.7441 | 诊断量 |

## ⑤ 收敛性与参数敏感性

**分辨率细化**（固定物理波长 = 域长，n 翻倍）：

| n | k | err | err 闭式 |
|---|---|---|---|
| 32 | 0.19635 | 6.413e-03 | 6.413e-03 |
| 64 | 0.09817 | 1.606e-03 | 1.606e-03 |
| 128 | 0.04909 | 4.015e-04 | 4.015e-04 |
| 256 | 0.02454 | 1.004e-04 | 1.004e-04 |

实测收敛阶 = 1.998, 1.999, 2.000（均值 **1.999**），即算符二阶一致 —— 与 D2Q9 对称 stencil 的理论阶一致。

**参数敏感性**：① `G_12`：−2.5 → −3.0（n=96）误差 3.49% → 7.01%，界面越陡（|G_12| 越大）高频含量越高，误差按谱权上升；② 分辨率 n=64 → 96（G_12=−2.5）误差 3.65% → 3.49%；③ 离散形式 standard vs edm 几乎不差（7.01% vs 6.43%），说明误差来自公共 stencil 的截断而非分裂方式。

**可用分辨率下界（工程结论）**：15% 门在 λ ≥ **7.11 cell/波长** 时满足，λ 更粗即破门（首个破门点 λ=2.06 cell）。按 tanh 界面 10–90 宽度 w 与主波长 λ≈2πw/… 的经验换算，界面宽度低于 ~2 cell 时交叉力的方向性误差已不可忽略。

## ⑥ 失败模式与工件

**（1）可证伪性判据部分达标（如实 FAIL）**：预登记「6 个种植缺陷全部 ≥15%」未满足——最小者为 `M5 psi=sqrt(rho)` 12.45%（12.45%）。原因：被测模量为常数密度 0.5 的单模场，把 ψ=ρ 换成 √ρ 在该点退化为**常数幅度缩放**（d√ρ/dρ = 1/(2√ρ)），只改变力的整体幅度而非形状，因此弱检出；相对测量底噪（基线 2.55%）仍是 **4.9×**。其余 5 个缺陷（符号翻转、丢对角权重、丢 c_s² 归一、gather 方向反、权重表错）误差 35%–197%，全部门外。**该条如实记为 FAIL，不改门限**。

**（2）预登记假设 H3 被证伪（负面结果入账）**：原假设「standard 形式在周期域留下 O(h²) 净动量缺陷、edm 精确为零」。实测两者**都**精确为零（standard 1.8e-17，edm 1.8e-19）：因为 standard 的域和 `Σ_x ρ₁(x)[ρ₂(x−c_i)+ρ₂(x+c_i)]` 是 c_i 的偶函数，被 `Σ_i w_i c_i` 的奇性抵消，故两组分 SC 交叉力在周期域上**总动量严格守恒**；`F₁+F₂` 的缺陷只存在于**逐点**层面（`local_defect` = 0.7446 vs 0.7441，两种形式几乎相同），即它是界面上的局部分布力，不是净动量漂移。文献中「SC 不守恒动量」的说法在本仓这个 stencil 上不成立，**edm 的价值也不在动量而在消除 O(h²) 离散旋度**——本单的度量未能分辨这一点，留给后续单（负面/未判部分如实标注）。

**（3）适用范围与局限**：
- 只覆盖 D2Q9 `ψ=ρ` 的 MCMP 交叉力；`multiphase3d.py`（D3Q19）与 `multiphase3d_d3q27.py` 未覆盖，属独立 XVAL；
- 只覆盖无壁面、无 solid_mask 分支（`wall_mode='dry'/'mirror'` 未进入）；
- 测的是**算符**与连续极限的一致性；模型级（线性 ψ）表面张力绝对精度不在本单范围——同仓 `benchmarks/case_pool/mcmp_laplace` 已如实记录该路线绝对 σ_EOS 锚 0/12 不达标；
- 两个 n=96 的 C 臂案例达到步数上限（`converged=false`），其碟/界面仍在缓慢松弛，误差数值保留但已标注；由于判据是对**当时密度场**的算符截断误差，松弛度不影响该量的含义，只影响所测界面陡度。

**工件与复现**：

```bash
cd <worktree>
PYTHONPATH=src python xval/XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE/run_xval.py            # 解析臂+汇总 (<1 s)
PYTHONPATH=src python xval/XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE/run_xval.py --slab-case 64:-2.5:4000
PYTHONPATH=src python xval/XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE/run_xval.py --slab-case 96:-2.5:3000
PYTHONPATH=src python xval/XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE/run_xval.py --slab-case 96:-3.0:3000
PYTHONPATH=src python xval/XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE/make_report.py            # 本文件 + 图
python -m pytest tests/test_xval_mcmp_cross_force.py -q                                       # 9 passed
```

- `xval/XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE/{predeclared.json,xval_lib.py,run_xval.py,make_report.py,REPORT.md}`
- `xval/XVAL-LBM-MCMP-TWO-COMPONENT-CROSS-FORCE/artifacts/{summary.json,slab_cases.jsonl,fields.npz,fig_xval.png}`
- `tests/test_xval_mcmp_cross_force.py`

![xval figure](artifacts/fig_xval.png)

**图**：(a) 轴/对角单模上算符截断 vs 闭式 `S(k)`（虚线）与 15% 门；(b) 分辨率细化（斜率 −2）；(c) 真实松弛界面上交叉力剖面 vs 谱参考；(d) 6 个种植缺陷的检出幅度 vs 15% 门与基线底噪。

