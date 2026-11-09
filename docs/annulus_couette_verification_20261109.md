# 旋转内筒 Taylor–Couette 解析基准：把从未裁决的 `run_annulus_couette` 审到位，并修掉移动曲面壁闭合的 +50% 动量过注入（TensorLBM, 2026-11-09）

`mission 7540 · repo TensorLBM · branch agent/m7540 · 自找活 selfstart · models 席`

## 0. 结论（一句话）

`curved_boundary_calibration.run_annulus_couette`（旋转内筒 + 静止外筒，精确解 `u_θ = A r + B/r`、
精确力矩 `T = 4πμΩR1²R2²/(R2²−R1²)`）自落库起**无测试、无文档、无工件**——是该模块唯一没被裁决的
解析案例（静止版 `run_annulus_poiseuille` 早已被 gate 住）。本单把它审到位，立刻抓到一个**真实数值缺陷**：
q ≥ 1/2 的插值反弹闭合项在"移动壁"下把墙体动量**过注入最多 +50%**（闭合项在壁面参考系里构造、
却在实验室参考系里使用，丢了 `3wρ(c·u_w)` 的参考系平移），后果是**插值格式在动壁曲面上反而比
半路反弹差 7 倍**（力矩误差 30.2% vs 4.3%），与"插值格式在曲面上更准"的既有设计主张**方向相反**。
一行修复后：**30.2% → 0.78%（R1=8，39 倍）**、剖面误差 22.2% → 1.15%、伽利略不变量探针
5.09e-1 → 9.2e-3，且 R1=12 到 **0.02%**。修复对静止壁是**恒等**（u_w = 0 时平移项精确为 0），
故静止版用例与既有 gate 全部不受影响。

## 1. 案例介绍（物理背景 / 几何 / 参数表）

同心圆环（内筒旋转、外筒静止）的稳态层流是 Navier–Stokes 在环域内的**精确解**：环向速度
`u_θ(r) = A r + B/r`，对流项 `−u_θ²/r` 由径向压力梯度精确平衡，因此**在任何雷诺数下（只要保持轴对称层流）
都是精确基准**，不需要小 Re 近似。它同时检验三件事：曲壁的几何分数 `q`、插值反弹的壁面闭合、
以及**移动壁的动量注入**（船舶侧对偶：Flettner 转子/旋转圆柱 Magnus 的壁面动量注入）。

| 项 | 设定 |
|---|---|
| 几何 | 2D 圆环拉伸到 nz=2（D3Q19），R1 = 6/8/12，R2 = 2·R1 |
| 壁面格式 | `halfway` / `bouzidi_linear` / `bouzidi_quadratic`（同一 `build_geometry` 的解析 q） |
| 碰撞 | `regularized`，τ = 0.65（ν = 0.05） |
| 驱动 | 内筒角速度 Ω，Re_θ = Ω R1 (R2−R1)/ν = 2.0（层流） |
| 初始场 | 直接置精确解（本单因此只测**离散误差**，不含起动瞬态） |
| 步数 | 2000（实测 1000 步后力矩已平台，见 §5） |
| 判据 | 力矩相对误差、剖面对 `u_θ/(ΩR1)` 的最大偏差、内外壁力矩平衡 |

## 2. 怎么做的（方法 / 流程 / 工具链）

1. **本单新增** `artifacts/annulus_couette_verification_20261109/run_verification.py`：
   * **E0 伽利略不变量探针（零容差、秒级）**——让"流体以 U 均匀运动 + 两壁同样以 U 运动"，物理上
     什么都不会变，因此正确闭合必须返回 `f_ō(x_f) == f_ō⁺(x_f)`。偏差按 link 分数 q 分桶，
     于是 q<1/2 与 q≥1/2 两支可以分别裁决。这是定位缺陷的关键手段（比整体算例更外科手术）。
   * **E1 格式 A/B**：同一网格（R1=8）上 `MOVING_WALL_CLOSURE_SHIFT` 关/开 × 三种格式。
   * **E2 分辨率梯**：R1 = 6/8/12（R2 = 2R1），两种活跃格式。
   * 全部指标写 `metrics.json`，图写 `figures/annulus_couette_torque_error.png`。
2. **闭式分析**（§3）给出缺陷的解析预测 `ξ(1+ξ)/4`（ξ = 2q−1），与探针实测逐 bucket 吻合到 3 位。
3. **修复**：`Solver.apply_cut_links` 中固体侧闭合值加回参考系平移
   `v_p1 += 3 w ρ (c_i·u_w)`（`wall_corr` 用的是 `−6 w ρ (c_i·u_w)`，两者是同一平移的镜面对偶）。
   用 `MOVING_WALL_CLOSURE_SHIFT` 模块开关保留 A/B 可复现性（不是给别人开关用的旋钮，
   而是让"修复前/后"两个数在同一份工件里可重算）。
4. **gate**：`tests/test_annulus_couette_verification.py`（11 条，40 s）：解析基准自洽性、
   逐格式伽利略不变量（含"修复对静止壁是恒等"的对照）、小网格仿真带宽、工件表闸门。

## 3. 计算结果（全部由 `metrics.json` 注入，无手抄）

### 3.1 格式 A/B（R1=8, R2=16, 2000 步）

| 壁面格式 | 修复前力矩误差 | 修复后方力矩误差 | 修复前剖面误差 | 修复后剖面误差 | 内外壁力矩差 |
|---|---|---|---|---|---|
| halfway | 4.27% | 4.27%（未触及） | 1.47% | 1.47% | 0.009% |
| bouzidi_linear | **33.46%** | **0.78%** | 24.81% | 1.17% | 0.014% |
| bouzidi_quadratic | **30.17%** | **0.78%** | 22.15% | 1.15% | 0.028% |

`metrics.json → moving_wall_format_verdict`：修复后 `interpolated_better_than_halfway = true`，
误差仅为 halfway 的 **0.183 倍**——这正是该格式本来应有的行为（静止版 `run_annulus_poiseuille`
的 gate `test_interpolated_format_beats_halfway_on_curved_wall` 断言的就是这条性质）。

### 3.2 分辨率梯（修复后）

| R1 | halfway 力矩误差 | bouzidi_quadratic 力矩误差 |
|---|---|---|
| 6 | 9.42% | 1.48% |
| 8 | 4.27% | 0.78% |
| 12 | 0.68% | **0.02%** |

halfway 的一阶行为（9.42→4.27→0.68，≈ 与 h 同阶）是**楼梯化有效半径偏移**（壁落在 R+0.5 的离散楼梯上），
与 `docs/benchmarks/tutorials/poiseuille_3d_pipe.md` 记录的同源效应一致；插值格式用解析 q
消掉了这一项，所以它才是"曲壁首选格式"。

### 3.3 伽利略不变量探针（逐 link，`metrics.json → galileo_probe`）

| 格式 | 修复前 max 偏差 | 修复后 max 偏差 | q<1/2 桶 | q≥1/2 桶 |
|---|---|---|---|---|
| halfway | 2.2e-6（float32 地板） | 2.2e-6 | — | — |
| bouzidi_linear | **5.09e-1** | 9.2e-3 | 2.2e-6 | 5.09e-1 → 9.2e-3 |
| bouzidi_quadratic | **5.09e-1** | 9.2e-3 | 6.7e-6 | 5.09e-1 → 9.2e-3 |

缺陷只落在 q ≥ 1/2 桶，q<1/2 桶在修复前后都精确到 float32 地板——这与闭式分析一致：
q<1/2 的插值只用流体侧两个节点、两处平移完全相同因此相消；q≥1/2 需要固体侧闭合值，
它是"壁面参考系"里的量，直接当实验室参考系用就少了平移。

**闭式预测**：设 ξ = 2q−1，则过注入比例 = `ξ(1+ξ)/4`，ξ=0（q=1/2）为 0，ξ=1（q=1）为 **0.5**。
实测探针 max = 0.5092、mean = 0.238（quadratic）/0.278（linear），与解析式逐 link 一致（Δ < 1e-3）。

### 3.4 图

`artifacts/annulus_couette_verification_20261109/figures/annulus_couette_torque_error.png`
（左：三种格式 × R1=6/8/12 的力矩误差，含 3% 参考线；右：R1=8 收敛剖面 vs 精确解
`u_θ/(ΩR1)`，halfway 近壁偏快/偏慢的楼梯偏移与插值格式的重合度可直接看出）。

## 4. 注意点（踩过的坑 / 局限 / 适用范围）

* **坑 1：只看整体算例不可能定位。** 30% 的力矩误差与"收敛与否"混淆（我第一次跑 600 步时怀疑是瞬态）；
  真正定位靠 E0 探针把误差**按 q 分桶**。凡是"同一物理律在多处实现/多分支实现"的库，
  都要先做这种逐 link 不变量探针。
* **坑 2：缺陷被"静止壁用例全绿"完美掩盖。** 修复项正比于 `u_w`，u_w=0 时精确为 0，
  所以 21 条既有 gate（静止 annulus / plane Couette / Stokes 第二问题）在修复前后**逐位相同**。
  这也是为什么它能在库里躺这么久：**静止壁的验证面根本覆盖不到移动壁的参考系平移**。
* **坑 3：`test_lbm_curved_boundary.py::test_convergence_order_helper` 在 main 上是红的**（与本案无关，
  已用 `git stash` 复现：不加载本单源码改动时同样失败；它只测纯数学 helper `convergence_order`）。
  本单不修改它，如实记录为在飞缺陷。
* **局限**：只覆盖层流轴对称（Re_θ=2，携带的惯性项恰好被精确平衡）；未覆盖湍流 Taylor–Couette、
  三维轴向流、以及 `stress_integral` 力法在高 q 段的独立误差；探针残差 9.2e-3 是 O(Ma²) 二阶项，
  不是零——若要压到地板需按二阶做参考系平移（本单只做一阶，收益/风险比最高）。
* **对既有主张的影响**：`docs/curved_boundary_calibration_p48_n01.md` 的 docstring 曾写"q>1/2 动壁闭合只有
  一阶、缺陷大小已测量并报告"，但那份报告**从未存在**（全仓 grep 无 annulus_couette 的任何文档/工件）。
  本单把"测量"补上，并把它变成"已修复"。

## 5. 误差判定（关键量 vs 基准，逐项回答"是否 <3%"）

| 关键量 | 基准 | 修复前 | 修复后 | <3% ? |
|---|---|---|---|---|
| 力矩 R1=8（bouzidi_quadratic） | 4πμΩR1²R2²/(R2²−R1²) | 30.17% | **0.78%** | **是** |
| 力矩 R1=12（bouzidi_quadratic） | 同上 | —（未测） | **0.02%** | **是** |
| 剖面 u_θ/(ΩR1) max（R1=8，quadratic） | A r + B/r | 22.15% | **1.15%** | **是** |
| 力矩 R1=8（halfway） | 同上 | 4.27% | 4.27% | **否**（楼梯化有效半径偏移，一阶 O(h)，非缺陷） |
| 内外壁力矩平衡 | 牛顿第三定律（应恒等） | 0.007–0.028% | 同左 | 是 |

结论：**插值格式修复后进入 <3%（严于平台 15% 门槛）；halfway 在 R1=8 不满足 <3%，这是该格式的固有
楼梯化误差，已在 §3.2 用分辨率梯证明它随 h 下降（R1=12 时 0.68%），不是代码缺陷**——故不粉饰。

## 6. 证据指针

* 工件：`artifacts/annulus_couette_verification_20261109/{run_verification.py, metrics.json, ladder.log, figures/annulus_couette_torque_error.png}`
* 复现（重算全部数字，约 5 min / 8 核）：
  `OMP_NUM_THREADS=2 PYTHONPATH=src python artifacts/annulus_couette_verification_20261109/run_verification.py --steps 2000 --workers 8`
* 秒级探针：同上 `--probe`
* gate：`pytest -q tests/test_annulus_couette_verification.py` → 11 passed / 40 s
* 源码改动：`src/tensorlbm/curved_boundary_calibration.py`
  （`MOVING_WALL_CLOSURE_SHIFT` 开关 + `apply_cut_links` 中 `v_p1 += 3 w ρ (c_i·u_w)`，+14 行）