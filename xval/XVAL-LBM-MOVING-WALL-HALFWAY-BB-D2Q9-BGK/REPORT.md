# XVAL-LBM-MOVING-WALL-HALFWAY-BB-D2Q9-BGK — 动壁半程反弹（D2Q9+BGK）交叉验证

**verdict: PASS** · 主判据 max |Δu|/U0 = 0.4744 %（判据 <15 %，另 <3 %）

## ① 问题设定与基准出处

**被检验的主张.** TensorLBM 的动壁闭包 —— *pre-streaming 半程反弹（half-way bounce-back，
`f_new[q] = f_pre[opp(q)]`，有效壁面在格子坐标 0.5 处）+ Ladd 动量项
`+2 w_q rho_w (c_q·u_w)/cs²`* —— 在 D2Q9 + BGK 下是否达到 owner 精度标准（对标误差 <15%）。
闭包本体就是仓内已交付的实现：`benchmarks/verified/couette_2d/run.py::moving_wall_bounce_back`
与 `benchmarks/case_pool/couette_2d/run_case.py`；本单**未修改**它，只做独立交叉验证。

**两个基准（external anchor，出处见文献）**

| 算例 | 解析基准 | 判别力 |
|---|---|---|
| 稳态平面 Couette | `u = U0 (y-0.5)/H_eff`，给定边界下 N-S 精确解（Schlichting & Gersten, *Boundary-Layer Theory*, ch. 5；Krüger et al. 2017 §9 标准算例） | 线性剖面任何"一致"的半程反弹都精确复现 ⇒ 只检验**壁位约定与质量/动量记账**，无法区分闭包 |
| Stokes 第二问题（振荡平板） | `u(y,t) = U0 exp(-k(y-0.5)) cos(ωt - k(y-0.5))`，`k = sqrt(ω/2ν)`（Stokes 1851；Schlichting & Gersten ch. 5） | 壁速随时间变化 ⇒ 逐时步检验动量项、相位（有效壁位）与幅值（壁面滑移），本次**主判据** |

**方法谱系/上游出处.** Ladd 1994 (J. Fluid Mech. 271:285) 动壁动量修正；Zou & He 1997
(Phys. Fluids 9:1591) 速度边界；pylbm `LinearBounceBack`（`alpha=2, delta=1/2`）的等效化简见
`OSS_DIFF.md`（含源码行号与 commit `f9f5d16`）；`oss_inventory#4`（lbm-principles-practice：
"half-way bounce-back (rest + moving wall with Ladd momentum correction)"）、
`oss_inventory#10`（pylbm 边界族）、`oss_solutions#435`（同族 Couette 选型，跨单一致性）。

## ② 方法与实现

- 只用仓内原语：`tensorlbm.d2q9.{equilibrium,macroscopic}`、`tensorlbm.solver.{collide_bgk,stream}`；
  本目录新增的只有**壁面闭包**与**判分口径**，关键指针 `mwbb.py::make_step`。
- 单步顺序：`f_pre ← f` → `collide_bgk(f, τ)` → 壁行替换为 `f_pre[opp]` → 壁行加 Ladd 动量项
  （`rho_w = Σ f_pre` 取自壁行碰撞前分布）→ `stream`（x/y 周期 gather，壁行留在域内）。
- **顺序是硬约束**：动量项必须加在"反弹后"的分布上；加在碰撞前会被壁行的 `f_pre[opp]` 覆盖而完全失效
  （实测 `ladd_precollision` 与 `static_bb` 逐位同值，误差 93.94 %）。
- 判分口径（预注册 `PREREG.md`）：误差**恒以 U0 归一**（不用局部 u，避免远场发散）；主判据取
  `η = y-0.5 ≤ 3δ` 的可解层；远场截断底噪 `exp(-gap/δ)` 单列报告。
- Stokes 第二问题：底壁 `u_w(t) = U0 cos(ωt)`，`ω = 2ν/δ²`，通道深 `gap = 6δ`，统计窗口 = 第 6 个
  周期（瞬态已衰减），每 4 步采样，基频投影得幅值/相位。
- 全部 `float64`；无标定因子、无外推、无结果调参。

## ③ 参数表（格子单位，主判据工况）

| 量 | 值 | 说明 |
|---|---|---|
| 格子/碰撞 | D2Q9 / BGK | `collide_bgk` |
| τ | 0.8 | ν = (τ−0.5)/3 = 0.1 |
| U0 | 0.02 | Ma = U0/√(1/3) = 0.0346 |
| ω | 0.003125 | T = 2π/ω = 2011 步 |
| δ = √(2ν/ω) | 8 | 黏性穿透深度（8 格） |
| Re_δ = U0·δ/ν | 1.6 | 振荡层内惯性/黏性同阶 |
| gap / ny / nx | 48 / 50 / 4 | x 周期（一维流动），壁行 y=0 与 y=49 |
| 步数 | 12064 | 6 个周期 |
| 采样 | 每 4 步，末周期 | 基频投影 |
| 参考算例数 | 14 | 见 artifacts/case_*.json |

## ④ 对照表（ref vs got，判据逐条打勾）

**主判据：Stokes 第二问题幅值/相位逐层对比**

| η/δ | row | A/U0 解析 | A/U0 LBM | 幅值误差 % | 相位解析 ° | 相位 LBM ° | 相位误差 ° |
|---|---|---|---|---|---|---|---|
| 0.188 | 2 | 0.829 | 0.8297 | 0.086 | 10.7 | 10.4 | -0.297 |
| 0.438 | 4 | 0.6456 | 0.6462 | 0.0874 | 25.1 | 24.8 | -0.301 |
| 0.938 | 8 | 0.3916 | 0.3919 | 0.0655 | 53.7 | 53.4 | -0.285 |
| 1.44 | 12 | 0.2375 | 0.2377 | 0.0627 | 82.4 | 82.1 | -0.237 |
| 1.94 | 16 | 0.1441 | 0.1442 | 0.11 | 111 | 111 | -0.186 |
| 2.94 | 24 | 0.053 | 0.05305 | 0.0928 | 168 | 168 | -0.0853 |

**总账判据（机器复核，`artifacts/summary.json:gate`）**

| id | 判据 | 实测 | 阈值 | 判定 |
|---|---|---|---|---|
| M1 | Stokes-2nd max |du|/U0 over eta<=3 delta_bl (reference config) | 0.4744 | 15 | PASS |
| M2 | Stokes-2nd amplitude error at eta=delta and 2delta | 0.1096 | 5 | PASS |
| M3 | Stokes-2nd phase error at eta=delta and 2delta | 0.2847 | 5 | PASS |
| M4 | Couette pointwise max |du|/umax (H=32, tau=0.8, U0=0.05) | 3.441e-10 | 2 | PASS |
| M5 | Couette fitted effective wall position |y_wall| | -1.026e-10 | 0.01 | PASS |
| M6 | max |mass drift| over all cases | 3.331e-10 | 1e-06 | PASS |
| M7 | ablation ranking err(ladd) < err(eq_wall_node) < err(static_bb) | ladd_pre_local=0.4744, eq_wall_node=4.577, static_bb=93.94 | strict | PASS |
| M8 | resolution err(delta=8) < err(delta=4) | d4=1.745, d8=0.4744 | strict | PASS |
| M9 | Couette ablation: static wall cannot reproduce u(y)=U0(y-.5)/H | ladd=3.441e-10, static=100 | strict | PASS |

总体 verdict = **PASS**（失败项：无）。

**Couette（稳态精确解）全部工况**

| tag | H_eff | τ | U0 | max abs err % | L2 rel | 拟合壁位 y_wall | 判稳 |
|---|---|---|---|---|---|---|---|
| `couette_H16_tau0.8_U0.05_ladd_pre_local` | 16 | 0.8 | 0.05 | 1.16e-11 | 8.3e-14 | 2.71e-13 | True |
| `couette_H32_tau0.8_U0.05_ladd_pre_local` | 32 | 0.8 | 0.05 | 3.44e-10 | 4e-12 | -1.03e-10 | True |
| `couette_H32_tau0.8_U0.05_static_bb` | 32 | 0.8 | 0.05 | 100 | 1 | 3.3e+06 | False |
| `couette_H32_tau1.2_U0.05_ladd_pre_local` | 32 | 1.2 | 0.05 | 8.57e-12 | 7.5e-14 | -3.85e-13 | True |

**动壁闭包消融（同一 Stokes 工况）**

| 变体 | max abs err % (η≤3δ) | 幅值误差 % @η=δ | 相位误差 ° @η=δ | 质量漂移 % |
|---|---|---|---|---|
| `ladd_pre_local` | 0.4744 | 0.06548 | -0.2847 | -7.5e-11 |
| `ladd_pre_unitrho` | 0.4764 | -0.009057 | -0.3973 | -4.9e-11 |
| `eq_wall_node` | 4.577 | -3.712 | 1.763 | -4.4e-13 |
| `ladd_precollision` | 93.94 | -100 | -53.71 | 0 |
| `static_bb` | 93.94 | -100 | -53.71 | 0 |
| `ladd_sign_flip` | 187.9 | -0.009057 | 179.6 | -5e-11 |

关键读数：参考闭包 0.4744 % ≪ 15 % 判据（也 <3 %）；拿掉动量项
（`static_bb`）壁面根本不动（93.9 %）；"壁节点置平衡态"
的常见偷懒写法（`eq_wall_node`）4.58 %，比参考闭包差
9.65 倍；`ladd_pre_unitrho`
（rho_w≡1）与参考相差 0.0019 %
（Ma≤0.17 时 ρ_w−1 ~ O(Ma²)）⇒ "取局部 ρ_w"不是精度关键，**方向与施加顺序**才是。

## ⑤ 收敛性与参数敏感性

**分辨率扫描（δ = 4 与 8，同 τ、同 U0）**

| δ | ny | 步数 | max err % (η≤3δ) | max err % (全域) | 远场底噪 % |
|---|---|---|---|---|---|
| 4 | 26 | 3016 | 1.745 | 1.745 | 0.25 |
| 8 | 50 | 12064 | 0.4744 | 0.4744 | 0.25 |

δ: 4→8 误差 1.745 % → 0.4744 %，单调下降约
3.68 倍（判据 M8 PASS）。注：δ 变化同时改变
ω 与 Re_δ，属"相似条件下细化"而非严格网格收敛阶，**不报阶数**。

**参数点（τ / U0）**

| tag | τ | ν | U0 | Ma | δ | max err % |
|---|---|---|---|---|---|---|
| `stokes2_tau1.8` | 1.8 | 0.433 | 0.02 | 0.0346 | 8 | 1.724 |
| `stokes2_U0.1` | 0.8 | 0.1 | 0.1 | 0.173 | 8 | 0.4744 |
| `stokes2_U0.005` | 0.8 | 0.1 | 0.005 | 0.00866 | 8 | 0.4744 |

τ=1.8（ν=0.4333）误差 1.724 %，仍远小于 15 %；这是半程反弹壁位对 τ 的已知
依赖（BGK 壁面滑移项随 τ 增大），属适用范围而非缺陷。U0 0.005→0.1（Ma 0.0087→0.173）误差在 4 位有效数字内
不变（0.4744 % vs 0.4744 %）⇒ 误差**不是** Ma² 压缩性，
而是线性壁模型误差（相位/壁位），归一化后与 U0 无关。

## ⑥ 失败模式与工件

- **已排除的失败模式（负面知识，已入 ledger）**：动量项加在碰撞前 → 被反弹覆盖、壁面静止（93.94 %）；
  不加动量项 → 同；符号反 → 187.9 %（壁面反向拖曳）；壁节点置平衡态 → 4.58 %（可用但明显劣于参考）。
- **范围限制**：仅平面壁、壁法向 y、单分量切向速度、D2Q9+BGK、float64；弯曲/斜壁需 Bouzidi 插值反弹
  （pylbm `NoSlipLinearBouzidi` 谱系）；float32 与湍流 Re 不在本单范围。
- **复现**：`cd xval/XVAL-LBM-MOVING-WALL-HALFWAY-BB-D2Q9-BGK && python3 run_xval.py --stage chunk1 &&
  python3 run_xval.py --stage chunk2 && python3 run_xval.py --stage finalize && python3 make_figures.py`
  （分块各 <2 min；判据见 `artifacts/summary.json:gate`）。
- **工件**：`artifacts/summary.json`、`artifacts/case_*.json`、`artifacts/fig1..fig4*.png`；
  代码 `mwbb.py` / `run_xval.py` / `make_figures.py`；预注册 `PREREG.md`；OSS 对照 `OSS_DIFF.md`；
  门禁测试 `tests/test_xval_moving_wall_halfway_bb_d2q9_bgk.py`。

## ⑥′ 误差判定（显式回答 owner 提问）

- 主判据 M1：Stokes 第二问题 max |Δu|/U0 = **0.4744 %** → <3 %？**是**；<15 %？**是**（PASS）。
- 幅值/相位 M2/M3：0.1096 % / 0.2847 ° → **是**（PASS/PASS）。
- Couette M4：3.44e-10 % → <2 %？**是**（线性剖面精确复现）。
- 壁位 M5：y_wall 拟合 = -1.03e-10 格（阈 0±0.01）→ **是**。
- 质量守恒 M6：全工况 max |Δm/m| = 3.33e-10 % → 无虚假质量源。
