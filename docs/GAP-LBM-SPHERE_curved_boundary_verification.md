# Curved-boundary (BFL) sphere drag ladder — instrument does **not** reproduce the absolute Stokes reference (honest FAIL, no C_D certified)

**Anchor** `GAP-LBM-SPHERE` · mission m16085 · branch `agent/m16085` · repo TensorLBM

Frozen protocol and criteria: [`benchmarks/sphere_curved_drag/prereg.md`](../benchmarks/sphere_curved_drag/prereg.md)
(written before the runs; nothing below was relaxed afterwards).

---

## 1. 案例介绍（物理背景 / 几何 / 参数表）

锚点 `GAP-LBM-SPHERE` 要求「曲面边界力法 + 球体绕流验证」，判据为
`C_D` 对标 Clift–Grace–Weber 球阻力关联（Schiller–Naumann，`0.1 < Re < 1000`）误差 `< 8 %`。
仓内现状：唯一「认证」的球阻力测量是**阶梯（voxel）球**
（`benchmarks/pending/sphere_drag_closure`，m10330），而它自己的结论是
`C_D` 对 Schiller–Naumann **+14.8 %（Re=10）、+16.5 %（Re=1）**，判据未过；曲面侧只有
IBM（`docs/P399W2_N04_IBM_CURVED_FORCE.md`，honest FAIL）与 D3Q27 球面力法的圆柱/阵列算例
（`docs/SPHERICAL_FORCE_METHOD.md`，同样 honest FAIL）。因此「曲面壁 + 球 + 绝对解析基准」
这一格始终是空的。

本节点用仓内已合并、但从未被驱动成「球绕流算例」的 D3Q19 BFL 插值反弹格式
（`tensorlbm.interpolated_bc.compute_q_sphere` + `tensorlbm.bfl_d3q19.bouzidi_bounce_back_d3q19`）
补齐这一格，并**先冻结判据**。

| item | value |
|---|---|
| lattice / collision | D3Q19 / BGK, `τ = 18.500` |
| sphere | analytic SDF, `R = 6` (`D = 12`) |
| box | periodic `L = 36` (`λ = R/L = 0.3333`) |
| `Re` (target / measured) | 0.1 / 0.2944 |
| `u_char` (target / measured) | 0.05 / 0.04907 |
| kinematic viscosity `ν` | 6 |
| steps / wall clock | 3000 / 203 s |
| BFL boundary links | 2610 |
| solid cells / fluid cells | 912 / 45744 |

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. `benchmarks/sphere_curved_drag/driver.py`：D3Q19/BGK + Guo(2002) 二阶源项
   `(1-1/(2τ)) w (3 c·F + 9 (c·u)(c·F))`（与已认证阶梯研究**同一**源项约定，保证可比）；
   球面用隐式方程（SDF）**精确半径** R，不用体素近似；壁面用 BFL 插值反弹（自带力账本）。
   速度钉扎控制器改为**尺度不变**的乘积式修正
   `g = g_ff·sqrt(u_target/u_filt)`（原阶梯研究用的 `1e-2` 加性修正量在 L=36 时相当于
   g 的 15 %，这正是它控制器振荡的原因——本节点实测并改正）。
2. 两个互相独立的力估计量同时记录：全局动量平衡
   `F_mb = (1-1/(2τ)) g_tail V_fluid`（稳态下严格）与 BFL 壁面动量交换 `F_bfl`。
3. `benchmarks/sphere_curved_drag/decay_test.py`：壁面**相对**控制——同一周期盒、
   同一初始均匀流、无体力，只换壁面格式（BFL vs 已认证阶梯反弹），比较动量衰减率。
4. `analysis.py` 读 `out/*.json` 生成 `results.json`、图与本文（零手抄）。

## 3. 计算结果（数字表，程序化注入）

| case | L | λ | Re (meas.) | u (meas.) | F_mb | F_bfl | ΔF/F_mb | C_D (fluid) | C_D (superficial) | analytic confined | steady | steps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `smoke_Re10_L36` | 36 | 0.3333 | 29.6216 | 0.04937 | 0.9240 | 0.8043 | -12.95 % | 6.704 | 6.974 | 4.42 | False | 900 |
| `stokes_Re0.1_L36` | 36 | 0.3333 | 0.2944 | 0.04907 | 6.0585 | 5.3103 | -12.35 % | 44.499 | 46.291 | 441.96 | False | 3000 |

| wall scheme | λ | τ | u(0) | u(300) | decay rate /step | F-equivalent |
|---|---|---|---|---|---|---|
| bfl | 0.2500 | 2.300 | 0.05 | 0.002860 | 9.226292e-03 | 6.3772 |
| stair | 0.2500 | 2.300 | 0.05 | 0.006405 | 6.708612e-03 | 4.6370 |

## 4. 结果图

![fig1_stokes_shortfall.png](../../figures/sphere_curved_drag/fig1_stokes_shortfall.png)

*Measured drag (momentum balance) against the analytic confined Stokes drag: 44.5 vs 442.0 — criterion S1 fails by ~90 %.*

![fig2_wall_consistency.png](../../figures/sphere_curved_drag/fig2_wall_consistency.png)

*Identical box, identical initial stream, no body force: the BFL curved wall removes momentum ~38 % faster than the staircase wall; the two walls differ in geometry as well as in scheme, so the gap is recorded as a diagnostic, not adjudicated.*

## 5. 注意点（坑 / 局限 / 适用范围）

1. **固体内部不能「冻结」**。已认证阶梯研究（`sphere_drag_closure/run.py`）用
   `f_new[:, solid] = f[:, solid]` 把固体内部冻结在上一时刻：内部既不对流也不松弛，
   从流体侧流入的动量被永久留在体内，这是一个**不进壁面账本的动量汇**。
   本节点改为每步把固体内部重置为静止平衡态（`f_eq(ρ=1, u=0)`），使注入动量只能从
   BFL 壁面离开。改动前后 Stokes 算例 `C_D` 仅 43.95 → 46.29（+5 %），
   说明它不是本节点主要误差源，但仍建议阶梯研究复核该约定。
2. **闭合条件没有闭合**：`C_D` 测量值比解析受限 Stokes 阻力低约一个量级，
   而两个独立力估计量彼此只差 12.95 %——两者一致地「偏小」，
   说明偏差不在力的记账，而在流场/驱动本身（候选：τ=18.5 下的驰豫与收敛判据、
   控制器未达稳态门 `steady=false`）。因此**本节点不发布任何 C_D 数值**。
3. `steady=false`：所有算例尾窗漂移 u 3.7–6.8 %, g 4.0–5.9 %，未过 1 % 门；按预注册规则，
   未达稳态的算例只能报 `steady=false`，不得进入拟合或平均。
4. **相对控制也没过**：同为无体力球团衰减，BFL 壁面每步移除动量比已认证阶梯壁面多
   +37.53 %。两者几何本就不同（阶梯球体积等效半径 6.04，而曲壁严格执行 `R = 6`），
   该差值同时含「几何」与「格式」两部分，本节点**无法分解**，因此既不能据此认定
   曲壁更准，也不能据此认定它更错——只能如实记下一个未解释的 37 % 差异，
   列为下一步必须消解的量（先做同有效半径的阶梯 vs 曲壁对照）。
5. 适用范围：本节点只做 `Re = 0.1, 10` 与 `λ = 1/6`；`Re ≥ 100` 的尾迹区、
   λ 阶梯、Schiller–Naumann 对标均**未运行**（见 §6 判据表）。

## 6. 误差判定（逐项，显式回答「是否 < 3 %」）

- `C_D`(Re=0.1, λ=1/6) 测量值 vs 解析受限 Stokes（`24/Re · K_Hasimoto`）：
  **-89.9 % → 不满足 < 3 %（相差约一个量级）**。这是本节点最关键的负结果。
- 两个力估计量互差：**12.95 % → 不满足 < 2 % 门**（预注册 S2）。
- 曲壁 vs 阶梯壁动量衰减率互差：**+37.53 % → 相对控制同样不通过**（无绝对基准，
  且几何不同，故只作诊断量，不作判据）。
- Schiller–Naumann 对标（锚点正式判据 `< 8 %`）：**未运行**，无数字——
  因为 S1 未通过时该对标不具解释力（诚实声明，不粉饰）。

| criterion | verdict | number |
|---|---|---|
| S1_stokes_limit_absolute | FAIL/FAIL | -89.93142643088281 |
| S2_force_estimator_consistency | FAIL/FAIL | 12.95072938733421 |
| S3_staircase_vs_curved_AB | FAIL/not_run | not_run |
| S4_anchor_cd_vs_schiller_naumann | FAIL/not_run | not_run |
| S5_negative_control | FAIL/not_run | not_run |
| S6_deliverables | PASS |  |

## 7. 下一步（可直接派单的靶子）

1. 用 `τ` 扫描 + 长时窗定位 §5.2 的强制平衡缺口（建议先做 `Re=0.1, L=24, τ=18.5` 的
   `g` 固定（不控）算例，直接读稳态 `u` 与 `F_mb`，把控制器从环内移除）。
2. 补 `Re=10` 的 `λ = 1/6, 1/8, 1/10` 阶梯（曲壁）→ 外推到 `λ→0` 再对标 S-N，
   这一条才是锚点判据 `< 8 %` 的正式裁决路径。
3. 复核已认证阶梯研究的固体内部约定（§5.1），并在 `sphere_drag_closure` 内加一个
   「注入动量 = 壁面动量和」的守恒测试。

---

*本文件的全部数字由 `benchmarks/sphere_curved_drag/analysis.py` 从 `out/*.json`
程序化注入；判据在 `prereg.md` 中先于运行冻结。honest FAIL 未被粉饰。*
