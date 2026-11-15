# XVAL-LBM-PER-STEP-SETUP-HOOK-AND-MASK-BLEND · R3: 子格（部分饱和单元）掩膜壁面的解析验证

> 本文件由 `benchmarks/xval_per_step_hook_mask_blend/generate_report_r3.py` 从
> `artifacts/xval_frac_mask_wall_position/result.json` **程序化生成**，数字零手抄；
> 预宣言见 `benchmarks/xval_per_step_hook_mask_blend/predeclared_r3_fractional.json`。
> 复现：`python benchmarks/xval_per_step_hook_mask_blend/run_r3_fractional.py`（调度器上执行），
> 回归：`pytest tests/test_fractional_mask_wall_position.py`。

## 0. 与 round 1/2 的关系

round 1/2（`docs/XVAL-LBM-PER-STEP-SETUP-HOOK-AND-MASK-BLEND.md`）验证的是**二元掩膜**
（壁面落在格心面上，界面 = `Ny-2` 半格约定）。本 round 不动那套结论，只补上后来合入的
**子格**能力（`partial_cell_layer`）的验证欠账：把 `ε_s` 从 {0,1} 放开到 (0,1)，壁面落在
格子内部时，库里的权重公式是否真的把壁放在解析位置上。

## 7. 产物

| 文件 | 内容 |
|---|---|
| `benchmarks/xval_per_step_hook_mask_blend/predeclared_r3_fractional.json` | 测量前冻结的设计与 P1–P5 |
| `benchmarks/xval_per_step_hook_mask_blend/run_r3_fractional.py` | 运行脚本（复用 round 1/2 harness） |
| `benchmarks/xval_per_step_hook_mask_blend/generate_report_r3.py` | 报告/本文档生成器 |
| `artifacts/xval_frac_mask_wall_position/result.json` | 全部 KPI、判定、环境 |
| `artifacts/xval_frac_mask_wall_position/table.csv` | 臂 × 配置全表 |
| `artifacts/xval_frac_mask_wall_position/profiles.npz` | 实测与解析剖面 |
| `artifacts/xval_frac_mask_wall_position/figures/*.png` | 4 张图 |
| `tests/test_fractional_mask_wall_position.py` | 4 条物理级回归（含「原始 ε_s 当权重会失败」的反例测试） |

## 1. 案例介绍

**物理背景。** 掩膜混合（mask blend）边界把边界条件写成一条张量表达式
`f = f_post*(1-M) + Σ_k m_k f_bc_k`，而不是对单元索引做分支；这是
loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow 的核心做法，本仓库已按该做法移植并验证
了**二元掩膜**（壁面落在格心面上）。随后库内新增 `partial_cell_layer`——Noble & Torczynski (1998)
的部分饱和单元（partially saturated cell），允许**壁面落在格子内部**，用固体分数 `ε_s ∈ [0,1]`
表示，混合权重为

`B = ε_s (τ - 1/2) / ((1 - ε_s) + (τ - 1/2))`，反射的是**碰撞后**的分布。

代码与单元测试已合入，但**从未发布过一个验证数字**：壁面究竟落在解析解说的位置上吗？把上游的
"原始固体分数当权重"直接移植（`f*(1-ε_s) + bounce*f*ε_s`，碰撞前分布）会错多少？本单就是补这笔
验证欠账（XVAL 锚点的 round 3）。

**几何与约定**（格心在整数 `y`，单元 `k` 覆盖 `[k-0.5, k+0.5]`）：第 0 行全固，第 1 行为部分
饱和行，固体占据其下表面，因此无滑移界面在

`y_b = 0.5 + ε_s`，通道 `y_b ≤ y ≤ Ny-1.5`，`H = (Ny-1.5) - (0.5+ε_s)`。

`ε_s = 1` 退化为 round 1/2 的阶梯壁（界面在共享面 `y=1.5`），这就把约定钉死在旧结果上。

| 量 | 符号 | 值 |
|---|---|---|
| 松弛时间 | τ | 0.80（τ 扫描 0.6/0.7/0.8/0.9） |
| 黏度 | ν = (τ-1/2)/3 | 0.1 |
| 壁面/驱动速度 | U | 0.02 |
| Mach | Ma = √3 U | 0.034641 |
| 通道高度 | Ny | 16、24 |
| 固体分数 | ε_s | 0.25 / 0.50 / 0.75 / 1.00 |
| 网格 | nx | 8 |

参考解是闭式解，且用的是**真实子格壁位**而非吸附到格点后的位置：Couette `u = U (y-y_b)/H`，
Poiseuille `u = g y (H-y)/(2ν)`。界面位置 `d_eff` 由**仅流体行**的最小二乘拟合给出（部分饱和行
自身的宏观速度是混合量，被排除在拟合与误差范数之外）。

## 2. 怎么做的

* 设计在测量**之前**冻结：`benchmarks/xval_per_step_hook_mask_blend/predeclared_r3_fractional.json`，
  写明 5 条可证伪预测 P1–P5、阈值（界面 0.05 dx、内部目标 3%、owner 门槛 15%）与停止规则。
* 复用 round 1/2 的 harness（`run_xval.run_fractional_wall` / `run_fractional_sweep`）**未加改动**：
  同样的几何约定、同样的解析解、同样的停止规则（Couette = 12 个黏性 e-folding + 200 步；
  Poiseuille = 3.5 H²/ν）。唯一改变的变量是"子格壁如何表达"。
* 三个对照臂：`naive_eps`（上游直接移植，权重=原始 ε_s，碰撞前）、`nt_pre`（NT 权重，碰撞前反射）、
  `nt_post`（NT 权重，碰撞后反射 = 已发表 NT 形式）。
* 结构极限另做位级检查：`ε_s=1` 必须与二元 bounce 层**位级相同**，`ε_s=0` 必须是恒等，内部部分
  饱和团块必须守质量。
* 工具链：torch 2.7.1 / python 3.11.6，torch.float64，CPU 4 线程，
  递交调度器执行（本机只做短测试），全量 1.6 s。
* 结果程序化注入：本报告的每张表、每个数字都从 `artifacts/xval_frac_mask_wall_position/result.json`
  生成，无手抄。

## 3. 计算结果

### 3.1 预宣言判定

| 预测 | 判定 |
|---|---|
| `P1_nt_post_places_the_wall` | FALSIFIED |
| `P2_naive_is_biased` | HOLDS |
| `P3_tau_robustness` | FALSIFIED |
| `P4_reflect_choice_matters` | FALSIFIED |
| `P5_eps_one_collapse` | FALSIFIED |
| `P_limits` | HOLDS |

* P1：`nt_post` 的最大界面偏差 **0.1454 dx**，最大剖面 L2 **3.3822 %**
  （阈值 0.05 dx / 3 %）。
* P2：`naive_eps` 在 ε_s=0.50 的界面偏差 **0.3589 dx**，随 (1-ε_s) 单调：**True**。
* P3：`nt_post` 在不同 τ 下界面偏差最大 **0.0841 dx**。
* P4：`nt_pre` 是否一致地差于 `nt_post`：**False**（逐 ε_s 见下表）。
* P5：ε_s=1 三臂是否退化到同一阶梯结果：**False**。
* 结构极限：ε_s=1 位级等于 bounce **True**；ε_s=0 位级恒等 **True**；
  部分饱和团块质量相对漂移 **0.000e+00**。
* NT 权重：B(ε=0.25)=0.0714, B(ε=0.5)=0.1875, B(ε=0.75)=0.4091, B(ε=1.0)=1.0000。

### 3.2 臂 × 配置全表

### Plane Couette, Ny=16, tau=0.80 (interface at y_b = 0.5 + eps_s)

| eps_s | arm | d_eff_dx | offset_err_dx | profile_l2_pct | max_err_pct |
|---|---|---|---|---|---|
| 0.25 | `naive_eps` | +1.1885 | +0.4385 | 3.0286 | 3.1080 |
| 0.50 | `nt_pre` | +1.1117 | +0.1117 | 0.7882 | 0.8020 |
| 0.75 | `nt_post` | +1.1046 | -0.1454 | 1.0550 | 1.0644 |
| 0.25 | `nt_pre` | +0.8592 | +0.1092 | 0.7360 | 0.7552 |
| 0.50 | `nt_post` | +0.9633 | -0.0367 | 0.2562 | 0.2609 |
| 1.00 | `naive_eps` | +1.5000 | +0.0000 | 0.0001 | 0.0001 |
| 0.25 | `nt_post` | +0.7592 | +0.0092 | 0.0613 | 0.0628 |
| 0.75 | `naive_eps` | +1.4452 | +0.1952 | 1.4533 | 1.4661 |
| 1.00 | `nt_pre` | +1.5000 | +0.0000 | 0.0001 | 0.0001 |
| 0.50 | `naive_eps` | +1.3589 | +0.3589 | 2.5808 | 2.6262 |
| 0.75 | `nt_pre` | +1.3124 | +0.0624 | 0.4596 | 0.4635 |
| 1.00 | `nt_post` | +1.1000 | -0.4000 | 2.9848 | 2.9850 |

### Plane Couette, Ny=24, tau=0.80 (interface at y_b = 0.5 + eps_s)

| eps_s | arm | d_eff_dx | offset_err_dx | profile_l2_pct | max_err_pct |
|---|---|---|---|---|---|
| 0.25 | `naive_eps` | +1.1886 | +0.4386 | 1.9524 | 1.9852 |
| 0.50 | `nt_pre` | +1.1117 | +0.1117 | 0.5042 | 0.5097 |
| 0.75 | `nt_post` | +1.1046 | -0.1454 | 0.6675 | 0.6716 |
| 0.25 | `nt_pre` | +0.8592 | +0.1092 | 0.4789 | 0.4868 |
| 0.50 | `nt_post` | +0.9633 | -0.0367 | 0.1644 | 0.1664 |
| 1.00 | `naive_eps` | +1.5000 | +0.0000 | 0.0003 | 0.0003 |
| 0.25 | `nt_post` | +0.7592 | +0.0092 | 0.0402 | 0.0406 |
| 0.75 | `naive_eps` | +1.4452 | +0.1952 | 0.9111 | 0.9161 |
| 1.00 | `nt_pre` | +1.5000 | +0.0000 | 0.0003 | 0.0003 |
| 0.50 | `naive_eps` | +1.3589 | +0.3589 | 1.6390 | 1.6573 |
| 0.75 | `nt_pre` | +1.3124 | +0.0624 | 0.2893 | 0.2907 |
| 1.00 | `nt_post` | +1.1001 | -0.3999 | 1.8688 | 1.8691 |

### Plane Poiseuille, Ny=16, tau=0.80

| eps_s | arm | d_eff_dx | offset_err_dx | profile_l2_pct | max_err_pct |
|---|---|---|---|---|---|
| 0.25 | `naive_eps` | +1.1720 | +0.4220 | 8.8605 | 11.1850 |
| 0.50 | `nt_pre` | +1.0936 | +0.0936 | 2.0092 | 2.5606 |
| 0.75 | `nt_post` | +1.1070 | -0.1430 | 3.3822 | 4.0850 |
| 0.25 | `nt_pre` | +0.8408 | +0.0908 | 1.8583 | 2.3949 |
| 0.50 | `nt_post` | +0.9566 | -0.0434 | 1.0304 | 1.2050 |
| 1.00 | `naive_eps` | +1.4946 | -0.0054 | 0.2269 | 0.1657 |
| 0.25 | `nt_post` | +0.7487 | -0.0013 | 0.1334 | 0.1445 |
| 0.75 | `naive_eps` | +1.4375 | +0.1875 | 4.2834 | 5.3351 |
| 1.00 | `nt_pre` | +1.4946 | -0.0054 | 0.2269 | 0.1657 |
| 0.50 | `naive_eps` | +1.3478 | +0.3478 | 7.6314 | 9.5461 |
| 0.75 | `nt_pre` | +1.2995 | +0.0495 | 1.0882 | 1.4021 |
| 1.00 | `nt_post` | +1.1157 | -0.3843 | 9.4121 | 11.3818 |

### Tau sweep, plane Couette, Ny=16, eps_s=0.50

| tau | arm | d_eff_dx | offset_err_dx | profile_l2_pct | max_err_pct |
|---|---|---|---|---|---|
| +0.6000 | `naive_eps` | +1.4492 | +0.4492 | 3.2526 | 3.3097 |
| +0.7000 | `nt_pre` | +1.1312 | +0.1312 | 0.9274 | 0.9436 |
| +0.9000 | `nt_post` | +0.9159 | -0.0841 | 0.5852 | 0.5955 |
| +0.6000 | `nt_pre` | +1.1511 | +0.1511 | 1.0695 | 1.0881 |
| +0.7000 | `nt_post` | +1.0150 | +0.0150 | 0.1054 | 0.1072 |
| +0.6000 | `nt_post` | +1.0645 | +0.0645 | 0.4540 | 0.4618 |
| +0.9000 | `naive_eps` | +1.3226 | +0.3226 | 2.3132 | 2.3538 |
| +0.7000 | `naive_eps` | +1.4012 | +0.4012 | 2.8945 | 2.9453 |
| +0.9000 | `nt_pre` | +1.0945 | +0.0945 | 0.6659 | 0.6775 |



### 3.3 owner 门槛

全部 45 个臂中最差的是 `nt_post`，剖面 L2 = **9.4121 %**；
是否全部 < 15 %：**True**。

## 4. 结果图

* 图 1：三种子格壁表达测得的界面位置 vs 解析 `y_b = 0.5 + ε_s`。
* 图 2：剖面 L2 误差 vs ε_s（含 3 % 内部目标与 15 % owner 门槛）。
* 图 3：ε_s=0.50 时界面位置随 τ 的变化（NT 权重依赖 τ，壁位不应依赖）。
* 图 4：ε_s=0.50 的 Couette 剖面（实测 vs 解析）。

## 5. 注意点

* **约定陷阱**：`d_eff` 的正确性完全取决于把格子单元看成"格心 + 半格覆盖"；若沿用 round 1 记录的
  节点壁约定（H = Ny-1），同一批数据会系统性偏移约 0.5 dx，那比本节要判定的效应还大。判定前先
  用 ε_s=1（界面正好在共享面）把约定钉住。
* **部分饱和行自身的速度不可用**：它是流体与壁的混合量，只能作诊断，不能进误差范数；本单的 `d_eff`
  与 L2 都只用流体行，这也意味着靠近壁的第一层流体行的分辨率损失没有被计入误差，报告的是"通道
  **内部**的壁位正确性"。
* **τ 依赖**：NT 权重显式含 τ，所以 τ 扫描是必需的独立轴；若只看 τ=0.8 一个点，就无法区分"权重
  公式对"与"系数碰巧对上"。P3 就是为此设的。
* **没有与上游二进制对比**：上游是 TF 1.x/Py2/float32 GPU，无法在此执行；本单的"naive"臂就是对上游
  做法的公式级移植，误差数字针对解析解，不对上游二进制。
* **适用范围**：只验证了平板（Couette/Poiseuille）子格壁位；曲面/运动子格壁（NT 的动量项）未在本单。

## 6. 误差判定（是否 <3 %）

| 关键量 | 基准 | 误差 | < 3 % | < 15 % |
|---|---|---|---|---|
| `nt_post` 剖面 L2（Couette/Poiseuille，ε_s<1，全部） | 闭式 Couette / Poiseuille（真实子格 y_b） | ≤ 3.3822 % | 否 | 是 |
| `nt_post` 界面位置 | `y_b = 0.5 + ε_s` | ≤ 0.1454 dx | 否（判据 0.05 dx，非百分比） | — |
| `naive_eps` 界面位置（上游移植） | 同上 | 0.3589 dx @ ε_s=0.50 | 否 | — |
| `naive_eps` 剖面 L2 最差 | 闭式解 | 8.8605 % | 否 | 是 |
| 结构极限（ε_s=1 / ε_s=0 / 质量） | 位级等价 / 恒等 / 守恒 | 0 / 0 / 0.0e+00 | 是 | 是 |

**结论（honest）**：NT 后碰撞反射显著优于上游「原始 ε_s 当权重」的直译（界面偏差 0.3589 dx → 最差 0.1454 dx，剖面 L2 8.8605 % → 3.3822 %），但仍**没有**达到 0.05 dx 的壁位判据（P1 被证伪），也不满足 τ 无关性（P3 被证伪）。正确表述是：该层可作为「精度约 0.15 dx 量级的亚格子壁」使用，在所有固体分数上把它当作精确壁位是**未经证实**的。

**已知带病点（本单新增证据，不粉饰）**：

1. `nt_post` 的界面位置偏差随固体分数增大：ε_s=0.25/0.50/0.75 逐个见 §3.2 全表，
   最大 0.1454 dx @ ε_s=0.75，因此**预宣言的 0.05 dx 判据被证伪**；但剖面 L2
   全部 < 15 % owner 门槛，最差 9.4121 %（`nt_post`，Poiseuille，ε_s=0.75）。
2. **`reflect="post"` 在 ε_s=1 时不退化为 bounce**：测得界面 d_eff = 1.10 dx（偏差 −0.40 dx），
   而 `naive_eps`/`nt_pre` 为 1.50 dx（偏差 0）。这与库内 docstring「ε_s = 1 gives B = 1（pure
   bounce-back, bit-wise the bounce layer）」的表述冲突：该位级等价只在 `reflect="pre"` 下成立
   （`run_fractional_limits` 正是用 pre 测的）。默认值是 `post`，所以在 ε_s→1 的极限下默认用法会
   给出错位壁面。**未在本单修复**（NT 权重在固相极限要回退碰撞，属算子语义改动），已加 xfail 特征化
   测试把这个行为钉住：`tests/test_fractional_mask_wall_position.py::test_nt_post_does_not_collapse_to_bounce_back_at_eps_one`。
3. 预宣言 P3（τ 无关）也被证伪：ε_s=0.50 时界面偏差从 +0.0645 dx
   (τ=0.60) 单调漂到 -0.0841 dx (τ=0.90)，说明残余偏差不是
   纯几何误差，权重公式的 τ 依赖没有被完全吸收。
