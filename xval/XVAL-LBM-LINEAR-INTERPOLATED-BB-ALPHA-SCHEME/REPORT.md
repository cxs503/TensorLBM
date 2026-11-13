# XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME — 插值反弹 α-格式族 vs OSS 精确 Poiseuille 判据

**状态：honest FAIL（主判据 15% 未达成，实测最好臂 2.86%）。**
本单完成三件事：① 把 TensorLBM 的两点线性插值反弹按 OSS 源码**逐项推广为 α 格式族**
（含 pylbm 的碰撞后修正项 `K̂`），默认路径与改造前算符**逐位一致**（max|Δ| = 5.6e-17）；
② 在同一判据（力驱动平面 Poiseuille，壁面亚格子距离 δ）上与 OSS 声明的精确性条件做交叉对标；
③ 实测出**闭包不是瓶颈、载体与壁面落位是瓶颈**，且本仓算符的**有效壁面位置比 δ 系统性多出
0.5 cell 量级**——给出 OSS 差异清单与移植取舍。

## 1 问题 + 基准出处

**基准**：x 周期、y 两直壁的力驱动充分发展平面 Poiseuille 流，壁面位于首/末流体节点外侧 δ cell，
物理通道高 `H = (N−1) + 2δ`，流体节点 `y_j = δ + (j−1)`；精确解
`u(y) = F/(2ν)·y·(H−y)`（White, *Viscous Fluid Flow* ch.3；Bouzidi-Firdaouss-Lallemand,
*Phys. Fluids* 13 (2001) 3452 §III.A 同款算例）。

**OSS 锚（源码直读，未搬运代码）**
* `~/oss-cache/pylbm/pylbm/boundary.py:920-993` `LinearBounceBack`：
  `α̂ = −1 + α(1/2+δ)`、`β = 1 − αδ`、`β̂ = 1 − α/2`、`K̂ = 1 − α̂`（`Par*` 子类开修正）；
  `BFLBounceBack` α = 2 (δ≤1/2) / 1/δ；`CLIBounceBack` α = 4/(1+2δ)（⇒ α̂=1，无需修正）；
  `YLIBounceBack` α = 2/(1+δ)。
* `~/oss-cache/pylbm/tests/test_linear_bounce_back.py`：几何 `H = N−1+2δ`；**精确性条件**
  “the scheme is exact when `Λ⁺Λ⁻ = δ²/2` and the wall momentum is `Λ⁻F`”，力在碰撞后按各向同性权重加入，
  报告速度 = 存储动量 − F/2。
* `site-packages/lbmpy/boundaries/boundaryconditions.py:149-230` `NoSlipLinearBouzidi`：
  同一三分支闭包（`q≥1/2`：`(f_q + (2q−1)f_q̄)/(2q)`；`q<1/2`：`2q f_q + (1−2q) f*`），第二份独立实现。

## 2 方法与实现

**(a) 能力改造（本仓新增，向后兼容）** — `src/tensorlbm/boundary/interpolated_bounce_back.py`
新增 `alpha`（`None`=BFL / float / callable）与 `post_collision_correction` + `f_old` 参数，
按 pylbm 的统一形式求值：

    val = α̂ f_pre(i) + β̂ f_pre(opp i) + β f_new(i) + K̂·½[(f_pre(i)−f_old(i)) − (f_pre(opp)−f_old(opp))]

导出 `alpha_bfl` / `alpha_cli` / `alpha_yli` 三个族成员。**等价性证明见 tests**：
默认路径与改造前两分支算符 `allclose(atol=1e-15)`；`α̂=1`（CLI）时修正项恒为 0；`q=1/2` 时 BFL 与 CLI
都退化为半程交换。

**(b) 对标驱动** — `xval/.../run_xval.py`：`carrier ∈ {bgk, trt_d2(Λ⁺Λ⁻=δ²/2), trt_c(3/16), trt:λ}`
× `closure ∈ {halfway, bfl, bfl_par, cli}` × δ × N；迁移用 `solver.stream`（x 周期、y 开口：最外两行重置为静止平衡，
堵住两壁互相 wrap 的污染），力按 OSS 方式在碰撞后加入（`s_i = 3 w_i c_ix F`），TRT 复用生产核
` solver.collide_trt(f, τ⁺, Λ)`，报告速度取 `momentum − F/2`（OSS 约定）。

**观测（无自由参数）**：`L2 = ‖u−u_ref‖₂/‖u_ref‖₂`；`w_bot/w_top` = 数值剖面最小二乘抛物线的根到首/末流体节点的距离
（`w = δ` 表示壁面落位正确）；`fit_res` = 该抛物线拟合的相对残差（判别剖面是否仍为抛物）。

## 3 参数表（无量纲/格子量）

| 量 | 值 |
|---|---|
| ν（格子） | 0.1（τ⁺ = 0.8；BGK τ = 0.8） |
| F（单位质量体力） | 8ν·u_max/H²，u_max=0.02 |
| Ma = u_max/c_s | 0.035 |
| Re_H | 3.1 (N=8) / 6.5 (N=16) |
| N（流体行数） | 8, 16 |
| δ | 0.25, 0.5, 0.9 |
| Λ⁺Λ⁻ | δ²/2（OSS）、3/16（经典）、0.02–0.32（扫描） |
| nx / 壁厚 | 2 / 3 行 |
| 收敛判据 | 200 步内 max|Δu| < 1e-12·u_max（≥4000 步后开始判） |

## 4 对照表（ref vs got，逐点打勾叉）

ref = 解析 Poiseuille（同一节点坐标）；误差 = L2 剖面相对误差。主判据 `L2 < 15%`：

| N | δ | carrier | Λ⁺Λ⁻ | closure | L2 (%) | max (%) | w_bot | w_bot−δ | fit_res | conv | src |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.250 | bgk | — | bfl_par | 38.84 | 27.93 | +0.733 | +0.483 | 1.4e-15 | ✓ | fast/results.json |
| 8 | 0.250 | bgk | — | halfway | 49.06 | 35.28 | +0.852 | +0.602 | 3.4e-15 | ✓ | fast/results.json |
| 8 | 0.250 | trt_c | 0.1875 | bfl_par | 40.14 | 28.87 | +0.749 | +0.499 | 2.3e-15 | ✓ | fast/results.json |
| 8 | 0.250 | trt_c | 0.1875 | halfway | 40.28 | 28.97 | +0.750 | +0.500 | 1.2e-15 | ✓ | fast/results.json |
| 8 | 0.900 | bgk | — | bfl_par | 42.53 | 33.02 | +1.567 | +0.667 | 1.7e-15 | ✓ | fast/results.json |
| 8 | 0.900 | bgk | — | halfway | 2.86 | 2.22 | +0.852 | -0.048 | 1.6e-15 | ✓ | fast/results.json |
| 8 | 0.900 | trt_c | 0.1875 | bfl_par | 43.41 | 33.70 | +1.579 | +0.679 | 2.0e-15 | ✓ | fast/results.json |
| 8 | 0.900 | trt_c | 0.1875 | halfway | 8.74 | 6.78 | +0.750 | -0.150 | 1.4e-15 | ✓ | fast/results.json |
| 16 | 0.250 | bgk | — | bfl | 18.55 | 13.39 | +0.750 | +0.500 | 8.9e-15 | ✓ | run_log_N16_partial.txt (N=16) |
| 16 | 0.250 | bgk | — | bfl_par | 18.20 | 13.14 | +0.741 | +0.491 | 1.6e-14 | ✓ | run_log_N16_partial.txt (N=16) |
| 16 | 0.250 | bgk | — | halfway | 23.49 | 16.95 | +0.879 | +0.629 | 1.6e-14 | ✓ | run_log_N16_partial.txt (N=16) |
| 16 | 0.250 | trt_c | 0.1875 | bfl | 18.85 | 13.61 | +0.758 | +0.508 | 1.2e-14 | ✓ | run_log_N16_partial.txt (N=16) |
| 16 | 0.250 | trt_c | 0.1875 | bfl_par | 18.50 | 13.35 | +0.749 | +0.499 | 1.3e-14 | ✓ | run_log_N16_partial.txt (N=16) |
| 16 | 0.250 | trt_c | 0.1875 | halfway | 19.15 | 13.82 | +0.766 | +0.516 | 1.3e-14 | ✓ | run_log_N16_partial.txt (N=16) |
| 16 | 0.250 | trt_d2 | 0.0312 | bfl | 18.37 | 13.26 | +0.746 | +0.496 | 4.7e-15 | ✓ | run_log_N16_partial.txt (N=16) |
| 16 | 0.250 | trt_d2 | 0.0312 | bfl_par | 18.02 | 13.01 | +0.737 | +0.487 | 1.1e-14 | ✓ | run_log_N16_partial.txt (N=16) |
| 16 | 0.250 | trt_d2 | 0.0312 | halfway | 33.84 | 24.42 | +1.141 | +0.891 | 1.2e-14 | ✓ | run_log_N16_partial.txt (N=16) |
| 16 | 0.500 | bgk | — | halfway | 13.27 | 9.73 | +0.879 | +0.379 | 1.4e-14 | ✓ | run_log_N16_partial.txt (N=16) |

**判定**：`L2 < 15%` 的算例 3 个；最好算例 bgk/halfway
（N=8, δ=0.9）L2 = **2.86%** ⇒ **主判据未达成（✗）**。
唯一低误差点来自「δ=0.9 的 halfway」——那是半程反弹把壁面钉在 0.5 cell、与 δ=0.9 的几何**偶然接近**，
不是闭包正确性（见 `w_bot` 列：halfway 恒为 0.85，与 δ 无关）。

## 5 收敛性 / 参数敏感性（≥2 点）

1. **δ（壁距）敏感性**：`w_bot` 对闭包/载体几乎不随 δ 变化（halfway 恒 0.85；BFL/K 修正臂在 δ=0.25 给 0.74–0.75，
   在 δ=0.9 给 1.57），即「有效壁面 = δ + 常数」而非「= δ」，这是 L2 随 δ 单调恶化的直接原因。
2. **分辨率**：同一 δ 下 N=8 → 16 的 L2 变化见上表（例如 δ=0.25 bgk/bfl_par 38.8% → 18.2%），
   收敛但收敛到**带偏差的抛物线族**（`fit_res` ~1e-14 ⇒ 剖面严格为抛物线，只是通道宽度错）。
3. **载体**：`fit_res ≈ 1e-14` 说明数值解是**精确抛物线**，误差全部来自有效壁位；BGK 与 TRT 之差别体现在
   壁位偏移量（δ=0.25：halfway 0.879 → trt_c 0.766 → trt_d2 0.746），而不是剖面形状。
4. **OSS 精确性条件复现失败**：把 `Λ⁺Λ⁻=δ²/2` 原样移植（`trt_d2`）后，δ=0.25 的 L2 = 18.02%，
   仍远离 OSS 声称的机器精度 ⇒ 缺口不在 TRT 参数，而在闭包取值约定（下述第 6 节）。

## 6 失败模式、差异清单与工件

**差异清单（本仓 vs OSS，做法/参数/边界条件）**

| 维度 | OSS（pylbm/lbmpy） | TensorLBM（本单） | 影响 |
|---|---|---|---|
| 闭包系数 | `α̂,β,β̂,K̂` 统一 α 参数化 | 已按同式实现（本单新增） | 无差（等价性已测） |
| δ 度量 | 壁面到**首/末流体节点**的链路距离 | 同 | 无差 |
| 低分支第二项 | `f̂_q̄(x_f,t)`（碰撞后、同节点、反方向） | `f_new(x_f,i)`（t+1/2 迁移后、本方向） | **疑为 +0.5 cell 壁位偏差来源**（本单实测，未修） |
| 精确性条件 | `Λ⁺Λ⁻=δ²/2` + 壁面动量 `Λ⁻F` | `Λ⁺Λ⁻=δ²/2` 已移植；壁面值项未实现（直壁 x 流向链路 `c_i·u_w=0` 时该项恒 0，故本算例不受影响） | 参数已对齐，结果仍不精确 |
| 力项 | 碰撞后各向同性权重 `3w_i c_ix F`，速度报 `momentum − F/2` | 同 | 无差 |
| 修正项来源 | `g = f(t) − f(t−1)`（步间变化，含源项） | `f_pre − f_old`（碰撞+源变化） | 需与 OSS 逐位复核（未完成） |
| 周期方向 | x 周期、y 壁 | 同 + y 开口两行重置（防 wrap） | 本仓额外防护 |

**移植取舍**：① 保留 α 参数化 + `K̂` 修正（OSS 已验证的族，本单已入生产模块，向后兼容、有 10 项单测）；
② 不搬运 pylbm 代码（仅取公式与几何/参数约定）；③ 壁面动量项（移动壁/Dirichlet 值）**未移植**，
本算例链路垂直、该项恒 0，故留作 strain：斜壁/移动壁算例需先补该项；
④ 判据保持 owner 的 15%（不因实测不达标而放宽）。

**失败定位（诚实，未闭环）**：所有插值臂的 `w_bot ≈ δ + 0.5`（δ=0.25：0.74–0.75 ⇒ +0.49~0.50；
δ=0.9：1.57 ⇒ +0.67），而半程臂恒 0.85（+0.35）。剖面本身是精确抛物线（`fit_res`~1e-14），
故偏差是**壁面落位**而非离散精度。最可能的机制是低分支第二项用了 `f_new(x_f,i)`（迁移后本方向）
而非 OSS 的 `f̂_q̄(x_f,t)`（碰撞后反方向）——两者在 q=1/2 恰好相同（这正是本仓「q=1/2 退化为半程」测试
能通过、而 q≠1/2 出错的原因）。**该假设本单未做修复验证（预算内仅完成定位），留第 2 轮**。

**工件**：`xval/XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME/{PREREG.md, run_xval.py, summarize.py,
artifacts/fast/results.json, artifacts/run_log_N16_partial.txt, xval_alpha_*.png}`；
测试 `tests/test_interpolated_bounce_back_alpha.py`（10 passed）。
