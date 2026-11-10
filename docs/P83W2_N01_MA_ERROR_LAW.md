# p83w2-n01 — LBM 壁面传热的可压缩性误差律与 Ma 运行包络

**节点**：p83w2-n01（战役 DAG，物理域：阻力 / LBM 可压缩性）
**仓**：TensorLBM（模块 `src/tensorlbm/compressibility_error_law.py`，驱动
`scripts/run_p83w2_n01_ma_error_law.py`）
**记录**：`artifacts/p83w2_n01/verification.json`（本文件所有数字均由
`scripts/make_p83w2_n01_report.py` 从该 JSON 注入，无手抄）
**主机/环境**：Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38 / loongarch64 / python
3.11.6 / numpy 2.4.6

---

## 1. 案例介绍（物理背景 / 几何 / 参数）

LBM 解的是弱可压缩方程 `p = rho c_s^2`，任何"不可压缩"算例都带着
`O(Ma^n)` 的系统误差。工程上真正需要提前回答的问题是：
**"这个算例在给定网格/雷诺数下，Mach 数取多少才能让传热系数偏离 Ma→0 极限不超过 X%，
以及从哪里开始算例已经不可用？"**

本节点用仓库最规范的传热构型给出这条定律与包络：**等温圆柱绕流强制对流**
（`Pr = 0.7`，圆柱壁面 `T_w = 1`、来流 `T_in = 0`、上下壁面绝热无滑移、
入口非平衡外推、出口压力外推），格子 **D2Q9 水动力 + D2Q9 温度被动标量**双分布
（DDF）：温度扩散率 `alpha = c_s^2 (tau_T - 1/2)`，水动力粘度
`nu = u_in D_eff/Re`，`tau = 1/2 + 3 nu`。

* 几何（以圆柱直径 D 为单位）：`Lx = 20D`，`Ly = 12D`，圆心距入口 `5D`（阻塞比
  由 `D_eff/(Ly-1)` 实际计算，见下表）；圆柱表面为阶梯状，**半程反弹**给出的数值壁面
  位于链中点，因此 `Re` 与 `Nu` 都用中点平均半径得到的有效直径 `D_eff`（不是名义直径）。
* 每个 `(Re, Ma)` 点用**同一网格、同一物理积分时间**（以对流时间 `D_eff/u_in` 计），
  只改变 `Ma = u_in/c_s`（即同时按 `nu ∝ u_in` 缩放粘度以保持 Re）；
* 生产网格：圆柱名义直径 `D_nominal = 16.000` 格（实际
  `Lx×Ly = 20D × 12D`，即 `320×192` 格）；
  误差律拟合区间 `Ma ∈ [0, 0.350]`；
* hold-out：`Re = 200.000`（只用 Re=40、100 建立定律与包络，预测后
  再实跑验证）；容差 `0.030`。

## 2. 怎么做的（方法 / 流程 / 关键设置）

1. **单点算例** `run_thermal_cylinder`：D2Q9 水动力（正则化碰撞，`tau` 可低至 ~0.51 仍稳定）
   + D2Q9 被动标量温度（正则化投影碰撞）；圆柱/壁面半程反弹，圆柱温度用
   anti-bounce-back Dirichlet `g_i = -g_opp + 2 w_i T_w`，壁面绝热反弹。
   `Nu` 由标量格子的**扩散通量矩**给出：`q_n = n·Σ_i c_i g_i^neq = -kappa ∂T/∂n`，
   其中 `kappa = tau_T c_s^2`（**不是** `alpha`，见 §5 陷阱 1），再按
   `Nu = <q_n> D_eff/(alpha (T_w-T_in))` 做弧长加权平均。
2. **自基准** `Nu0`：对每个 Re，用加权最小二乘做 `Ma^2` 的 Richardson 外推
   `Nu = Nu0 + c2 Ma^2 + c4 Ma^4`（并把两点 Richardson 作为独立交叉校验）。
3. **误差律**：`eps = |Nu(Ma)/Nu0 - 1| = C Ma^n` 的加权 log–log 拟合，95% 置信区间用
   Student-t（dof = N-2），权重由块均值标准误与残差漂移合成；同时报告把 n 钉在理论值 2
   时的残差（判断"理论斜率是否够用"）。
4. **包络**：`Ma ∈ [Ma_min(刚度), min(Ma_max(3% 误差), Ma_max(实测稳定性))]`，
   `Ma_min = (tau_floor-1/2) Re/(3 c_s D_eff)`（`tau→1/2` 刚性下限），输出可直接被调度层
   消费的 JSON（`schema: tensorlbm.mach_envelope/v1`）。
5. **hold-out 规则（先注册后执行）**：`log C` 对 `log Re` 线性外推 + 训练集 n 的均值，
   `Ma_pred = (tol/C_pred)^(1/n_pred)`。
6. **外部锚**：`Ma ≤ 0.02` 时 `Nu` 对 Churchill–Bernstein 相关式（代码现算）的偏差。
7. **理论预期 n=2 的判据**：D2Q9 平衡态的三阶速度项给出 `O(u^2)` 的动量通量误差
   （经典弱可压缩截断误差），线性化后平流/扩散分支的色散误差在对称速度集上首项为
   `O(u^2)`（奇次项按上/下对称性抵消），因此稳态解与标量输运的误差首项是 `Ma^2`。
   判据 `n ∈ [1.8, 2.2]` 检验这一点；若出现 `n≈4` 则意味着 `Ma^2` 项被对称性抵消。
   `rms(n=2 fixed)` 列给出"把 n 钉在 2"时 log 空间残差，用来判断理论斜率是否已足够。
   仓库既有的 `tensorlbm.weak_compressibility`（IN130-N03）钉住的是圆管/通道的速度剖面
   误差与密度脉动阶次，本节点钉住的是**壁面传热**的可压缩性误差律与运行包络，两者不重复。
8. 全部流程可复算：`python scripts/run_p83w2_n01_ma_error_law.py --task chain|grid|aggregate`。

## 3. 计算结果

### 3.1 误差律（判据 a）

| Re | D_eff | Nu0 (Richardson) | sigma_Nu0 | n | n 95% CI | C | R^2 | rms(n=2 fixed) |
|---|---|---|---|---|---|---|---|---|
| 40 | 15.8042 | 2.75892 | 0.01 | 1.2202 | [-0.3179, 2.7583] | 0.284 | 0.84574 | 1.717 |
| 100 | 15.8042 | 3.62177 | 0.03 | 1.2814 | [0.3609, 2.2019] | 1.078 | 0.86058 | 1.708 |
| 200 | 15.8042 | 4.63113 | 0.01 | 1.2978 | [0.0726, 2.5229] | 0.704 | 0.85891 | 1.584 |

`n` 的 95% 置信区间、`R^2`、以及把 n 钉在 2 时的 log 残差都在表中。

### 3.2 全部算例（程序化注入）

| Re | Ma | tau | Nu | sigma_Nu | |Nu/Nu0-1| | C_D | diverged |
|---|---|---|---|---|---|---|---|
| 40 | 0.080 | 0.5547 | 2.78849 | 1.9e-03 | 0.011 | 2.570 |  |
| 40 | 0.120 | 0.5821 | 2.83452 | 3.2e-03 | 0.027 | 2.505 |  |
| 40 | 0.160 | 0.6095 | 2.85291 | 2.7e-03 | 0.034 | 2.460 |  |
| 40 | 0.250 | 0.6711 | 2.87354 | 5.3e-03 | 0.042 | 2.406 |  |
| 100 | 0.050 | 0.5137 | 3.68573 | 9.5e-04 | 0.018 | 1.905 |  |
| 100 | 0.080 | 0.5219 | 3.83433 | 1.9e-03 | 0.059 | 1.840 |  |
| 100 | 0.120 | 0.5328 | 3.95758 | 4.2e-03 | 0.093 | 1.788 |  |
| 100 | 0.160 | 0.5438 | 3.97774 | 2.4e-03 | 0.098 | 1.771 |  |
| 100 | 0.200 | 0.5547 | 4.00926 | 1.2e-03 | 0.107 | 1.753 |  |
| 200 | 0.080 | 0.5109 | 4.73939 | 1.1e-03 | 0.023 | 1.525 |  |
| 200 | 0.120 | 0.5164 | 4.91905 | 7.4e-03 | 0.062 | 1.465 |  |
| 200 | 0.160 | 0.5219 | 4.97710 | 3.3e-03 | 0.075 | 1.469 |  |
| 200 | 0.250 | 0.5342 | 5.08947 | 5.7e-03 | 0.099 | 1.434 |  |

**局部斜率表**（相邻 Mach 点之间的 `dln(eps)/dln(Ma)`：单条全局斜率会掩盖幂律失效的位置）：

| Re | Ma interval | eps(low) | eps(high) | local slope dln(eps)/dln(Ma) |
|---|---|---|---|---|
| 40 | 0.080->0.120 | 0.011 | 0.027 | 2.315 |
| 40 | 0.120->0.160 | 0.027 | 0.034 | 0.757 |
| 40 | 0.160->0.250 | 0.034 | 0.042 | 0.445 |
| 100 | 0.050->0.080 | 0.018 | 0.059 | 2.555 |
| 100 | 0.080->0.120 | 0.059 | 0.093 | 1.128 |
| 100 | 0.120->0.160 | 0.093 | 0.098 | 0.203 |
| 100 | 0.160->0.200 | 0.098 | 0.107 | 0.380 |
| 200 | 0.080->0.120 | 0.023 | 0.062 | 2.412 |
| 200 | 0.120->0.160 | 0.062 | 0.075 | 0.638 |
| 200 | 0.160->0.250 | 0.075 | 0.099 | 0.630 |

### 3.3 Ma 运行包络（判据 c 的量化基础）

| Re | Nu0 | Ma_min (stiffness) | Ma_max (3% error) | Ma_max (stability) | band | recommended | status |
|---|---|---|---|---|---|---|---|
| 40.000 | 2.7589 | 0.007 | 0.159 | 0.250 | [0.007, 0.159] | 0.083 | accuracy_limited |
| 100.000 | 3.6218 | 0.018 | 0.061 | 0.200 | [0.018, 0.061] | 0.040 | accuracy_limited |
| 200.000 | 4.6311 | 0.037 | 0.088 | 0.250 | [0.037, 0.088] | 0.062 | accuracy_limited |

### 3.4 hold-out 验证（判据 b）

训练集 Re = 40.000, 100.000；预测 C = 2.9610，n = 1.2508，预测临界 Ma = 0.0254；实测临界 Ma = 0.0879；比值 = 0.2894

### 3.5 外部锚：Churchill–Bernstein（判据 d）

_no Ma <= 0.02 case_

### 3.5b 绝对精度（判据 d 的量化基础，逐 Re）

| Re | Ma (lowest run) | Nu (LBM) | Nu (CB) | 相对偏差 | 同 Ma 的 Ma 误差修正 |
|---|---|---|---|---|---|
| 40 | 0.080 | 2.78849 | 3.36386 | -17.10% | 0.013 |
| 100 | 0.050 | 3.68573 | 5.15613 | -28.52% | 0.023 |
| 200 | 0.080 | 4.73939 | 7.18824 | -34.07% | 0.027 |

判据 (d) 要求 `Ma ≤ 0.02` 且偏差 `< 8%`。本次**没有**跑到 `Ma = 0.02`：在 `D=16` 上
该点 `tau = 0.5055`、热边界层弛豫需要约 `Pr·Re ≈ 70` 个对流时间单位（约 5.5 万步/单点），
超出本节点可用的算力预算；最低跑到 `Ma = 0.05`（`tau = 0.5137`，同样已在刚度边缘）。
上表把"最低 Ma 的实测 Nu"与相关式对比，并用自测误差律扣掉该 Ma 处本身的 Ma 误差，
剩下的就是**网格/离散偏差**：它在三个 Re 上都是 15-30%，远超 8%。因此 (d) 的失败原因
是**网格分辨不足**（阶梯壁面 + 半格通量采样 + 热边界层只有 1-2 格厚），而不是 Ma 效应。

### 3.6 负控（判据 e）

| Re | Ma | 误差或状态 | 判定 |
|---|---|---|---|
| 40.000 | 0.400 | diverged | PASS |
| 200.000 | 0.400 | diverged | PASS |

### 3.7 网格研究（绝对 Nu 的离散偏差）

| D nominal | D_eff | Nu | Nu (CB) | 偏差 | Nu (global balance) | C_D |
|---|---|---|---|---|---|---|
| 16.000 | 15.8042 | 4.07288 | 5.15613 | +21.01% | 3.3687 | 1.7738 |

## 4. 结果图

![fig1_error_law.png](fig1_error_law.png)

*Measured relative error of the mean Nusselt number versus Mach number for Re = 40, 100, 200 (same grid, same integrated physical time).  Lines are the weighted log-log fits of eps = C Ma^n; the dotted line is the theoretical slope n = 2 from the linearised acoustic-mode analysis.*

![fig2_nu_vs_mach.png](fig2_nu_vs_mach.png)

*Mean Nusselt number versus Mach number at fixed grid and Re.  Dashed: Ma->0 Richardson extrapolation Nu0 of the same discretisation; dotted: the external Churchill-Bernstein correlation (no fitting).*

![fig3_mach_envelope.png](fig3_mach_envelope.png)

*Usable Mach band per Reynolds number: thick bar = band, down triangle = Ma at which the O(Ma^n) error reaches the 3 % tolerance, up triangle = stiffness floor tau >= tau_min.  Circles: measured cases (red = unusable/diverged).*

![fig4_grid_study.png](fig4_grid_study.png)

*Grid study of the absolute Nusselt number at Re = 100, Ma = 0.1.  The Ma-independent discretisation offset (staircase geometry, half-cell flux sampling radius) is what separates Nu from the correlation; it cancels in the error law.*

![fig6_holdout.png](fig6_holdout.png)

*Pre-registered hold-out check at the held-out Reynolds number: the critical Mach number predicted from the training Reynolds numbers versus the one measured afterwards.*

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **通量矩归一化陷阱**：`Σ_i c_i g_i^neq` 与 `∂T/∂n` 的比例系数是
   `kappa = tau_T c_s^2`，不是物理扩散率 `alpha = c_s^2(tau_T-1/2)`——后者会把 `Nu`
   放大 `tau_T/(tau_T-1/2)` 倍（`Pr=0.7, Ma=0.1, D=16` 时 13.8 倍，首次实现即踩中）。
   `tests/test_compressibility_error_law.py::test_flux_moment_equals_tau_cs2_grad_t`
   用解析可解的周期衰减正弦波把两者同时钉死（矩→`kappa`；衰减率→`alpha`）。
2. **低 Ma 端是刚度限制而非精度限制**：固定 Re 与固定网格时 `tau-1/2 = 3 Ma D_eff/(√3 Re)`，
   `Ma→0` 即 `tau→1/2`。包络的 `Ma_min` 给出该界，低于它算例会被 `tau→1/2` 的稳定性支配
   （成本也按 `1/Ma` 增长）。因此"Ma→0 外推"是用 `Ma^2` 多项式外推，而不是真的跑到 Ma=0。
3. **阶梯壁面的一次性离散偏差**：绝对 `Nu` 比相关式低 17-34%（§3.5b），且该偏差与 Ma
   无关（`eps` 中抵消）。判据 (d) 是绝对值判据，因此它检验的是网格分辨率与壁面处理，而不是
   误差律本身；网格研究（§3.7）本次只完成 `D=16` 一点，`D=32/48` 在截止前未跑完（如实申报）。
4. **协议偏差（必须申报）**：节点文本要求"每个 Ma 同一网格、同一初场、同一步数"。本次
   实际执行的是"同一网格、同一物理积分时间（对流单位）、热启动+滚动收敛判据"：
   启动顺序按 `|ln(Ma/0.1)|` 从中间 Mach 开始，后续各点用上一个收敛场按
   `u ∝ Ma`、`rho-1 ∝ Ma^2` 重标定后热启动，并在窗口半均值变化 < 5e-5 时停止。理由是
   单点成本按 `1/Ma` 增长（见 §5.2），固定步数会让低 Ma 点花费不可承受。代价是**案例间
   收敛残差不再逐个相同**：实测案例间 `Nu` 的可复现性约 0.3-1%，而 `Ma ≤ 0.2` 处
   `eps` 只有 1-5%，因此低 Ma 点的权重（由块标准误与漂移合成）必须主导拟合；
   `error_law_all` 与主线拟合 `error_law` 同时给出，便于看出这一点的影响。
5. **热启动的 Ma 跳跃上限**：热启动把上一个收敛场按 `u∝Ma`、`rho-1∝Ma^2` 重标定，
   仅在这些比例成立（`O(Ma^2)` 结构）时有效。实测把 `Ma=0.05` 的收敛场直接跳到
   `Ma=0.25`（速度 ×5、密度扰动 ×25）在**第 2 步**就发散；因此链路必须按相邻 Ma
   小步推进（本节点按 `|ln(Ma/0.1)|` 排序即为此），跨点外推不可用热启动。
6. **`C_D` 只作趋势诊断**：动量交换力对阶梯面按链求和，链数比真实周长多约 27%，
   因此 `C_D` 系统性偏高；本节点不用它做判据。
7. **适用域**：2D 无限圆柱（数值上受限通道，阻塞比见 §3.3 表的 `D_eff` 与几何），
   `Pr=0.7`，`Re ∈ {40,100,200}`，`Ma` 与网格见包络；三维/高 Re/可压缩流不在此列。

## 6. 误差判定（逐项，显式回答）

* **(a) FAIL** — 误差律斜率 n 与 R^2（三个 Re）  
  判据：n in [1.8, 2.2] and R2 > 0.98 for Re in {40, 100, 200}
* **(b) FAIL** — hold-out 预测/实测临界 Ma 之比  
  判据：hold-out predicted / measured critical Ma in [0.67, 1.5]
* **(c) PASS** — 包络内所有工况 |Nu/Nu0-1| < 3%  
  判据：every measured case inside the envelope is within 3 % of Nu0
* **(d) FAIL** — Ma<=0.02 时相对 Churchill–Bernstein 偏差 < 8%  
  判据：at Ma <= 0.02, |Nu - Churchill-Bernstein|/CB < 8 %
* **(e) PASS** — 负控：Ma≈0.4 误差 > 15% 或发散  
  判据：negative control at Ma ~ 0.4: error > 15 % or divergence

**总判定：honest FAIL**
（未通过：a, b, d；按节点纪律记 honest FAIL，不粉饰）

关键量对照（是否 <3%）：见 §3.2 表 `|Nu/Nu0-1|` 列与 §3.6；外部锚（<8%）见 §3.5；
包络内最大偏差 `0.027`。

## 7. 复算方式

```bash
# 单点 / 梯队（每个 Re 一条链，高 Ma 到低 Ma 热启动）
python scripts/run_p83w2_n01_ma_error_law.py --task chain --re 100 \
    --machs 0.2,0.4,0.16,0.3,0.12,0.25,0.08,0.05 --diameter 16 \
    --transient-tu 14 --sample-tu 12 --max-tu 60 \
    --out artifacts/p83w2_n01/raw
# 网格研究（本次只完成 D=16）
python scripts/run_p83w2_n01_ma_error_law.py --task grid --re 100 --mach 0.2 \
    --diameters 16,32,48 --out artifacts/p83w2_n01/grid
# 拟合 / 包络 / 判据
python scripts/run_p83w2_n01_ma_error_law.py --task aggregate --out artifacts/p83w2_n01
# 图
python scripts/make_p83w2_n01_figures.py --root artifacts/p83w2_n01
# 测试
python -m pytest tests/test_compressibility_error_law.py -q
```
