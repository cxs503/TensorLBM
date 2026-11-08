# LBM 解析解验证梯 Step A–E（r3：在 r2 证据链上补三维方管级）

> 文件名沿用 `_r2`（同一份文档的增量修订，避免外部链接/图引用漂移）；r3 的新增内容是
> **Step E（3D 方管 Poiseuille，Boussinesq 级数解）**，A–D 的结论与数字保持原样。
>
> 本文件的全部数字由 `scripts/lbm_verification_report.py` 从
> `artifacts/lbm_verification_ladder/metrics.json`（`tests/test_lbm_verification.py --refine --json`
> 的产物）程序化生成，无手抄；改动求解器后重跑两条命令即可刷新。

## 1. 结论（TL;DR）

- 五个闭式解析解案例（剪切波 / Couette / Poiseuille / 壁面摩擦 / 方管）在 CPU 上
  **5/5 通过** 1% 判据；最大误差 **0.416%（Step E）**，其次 0.337%（Step C），
  其余三项 ≤ 6.44e-05%。
- A / B / D 的残差是纯方法误差（float64 下 6.4e-05% 量级，
  且 float32 下大 1–2 个数量级 ⇒ 已到单精度舍入地板）；C / E 的 0.337% / 0.416% 是**离散误差**：
  Step C 固定步数不降（1000/1500/2000/3000 步均为 0.3367%），与 Ma 无关，但与网格二阶收敛
  （ny=12/H=10 → 0.3367%，ny=16/H=14 → 0.1709%，比值 1.97 ≈ (14/10)²）；
  Step E 在 1000→2000 步下**误差完全不变**（0.415905%），随方管边长加密单调下降
  （L=6 → 0.827%，L=10 → 0.416%，L=14 → 0.235%）。两者的 float32/float64 误差比都 ≈1，
  即 dtype 不敏感——离散误差主导，不是舍入。
- 这五个案例此前长期躺在 `tests/ci_quarantine.txt` 里：缺陷不是物理，而是**验证框架**——
  测试签名把设备钉在 `sdaa:0`、函数 `return bool` 而不是 assert。quarantine 行已删除，
  本文件是删除的依据。

## 2. 为什么有 r2 / r3

2026-11-03 的 v1 重建已经修好了"能红能绿"（设备无关 + 真断言 + 向量化），那部分在 main 上；
PR#703 因交付形态被关闭，r2 从当前 main 重新切枝。r2 相对 main 补的是**证据链**，不是物理：

1. **机器可读指标**：`tests/test_lbm_verification.py --json <path>` 把每个 Step 的关键量、
   误差、判据写成 JSON——报告/文档的数字从 JSON 注入，杜绝"文档写一套、代码跑一套"。
2. **dtype 成为被判据约束的事实**：dtype 由 `TENSORLBM_TEST_DTYPE` 解析（缺省 float64），
   并有 `test_step_c_poiseuille_float32_error_is_not_rounding_bound`：float32 误差必须仍 <1%
   且与 float64 同量级（比值落在 [0.3, 3]），否则说明判据已贴到单精度舍入地板、失去鉴别力。
3. **报告生成器**：`scripts/lbm_verification_report.py` 只读 JSON、只写 `artifacts/`，
   产出数字表与对照图（数值 vs 解析解；float64 vs float32）。

**r3（本单）补的是覆盖面**：A–D 全是"单方向受限 + 另一方向周期"的通道流，也就是**二维几何**。
D3Q19 在三维里最容易露馅的地方恰恰是**方管角区**——两条壁的反弹算子在此叠加、角区
剪应力由两个方向共同贡献，而此前没有任何解析解覆盖这一级。r3 因此加 Step E，并顺手修掉
报告脚本的一个真 bug（见第 10 节）。

## 3. 五个案例的物理设定

| Step | 物理问题 | 几何 / 格点 | 关键参数 | 解析解 | 校验的内核 |
|---|---|---|---|---|---|
| A | 剪切波衰减 | 周期盒 64×4×4 | τ=1，A₀=0.01，100 步 | u_y=A₀cos(kx)e^(−νk²t) | 碰撞核给出的 ν=(τ−½)/3 |
| B | Couette 流 | 12 层通道（y 向），nx=80 | τ=1，u_top=0.05，1000 步 | u_x=u_top(y−½)/(H−½) | 半路反弹无滑移 + 壁面动量交换 |
| C | Poiseuille 流 | 12 层通道，上下壁 | τ=1，G=0.000166667，1000 步 | u=G/(2ν)(y−½)(H+½−y) | Guo 体力注入 + 抛物线剖面 |
| D | 壁面摩擦 | 同上（Couette 型剪切） | τ=1，u_top=0.05，gap=10.5 | Cf=2ν/(L·U) | 壁面剪应力提取 τ_w=ν·∂u/∂y |
| E | 方管 Poiseuille | 10×10 截面（y、z 双壁）+ x 向周期 4 | τ=1，g=6.6667e-04，2000 步 | Boussinesq 级数解（见 §4） | 3D 双向往复受限：角区 + 两壁反弹叠加 |

量纲：格子单位（lu / ts），密度 ρ=1。判据统一为**相对误差 < 1%**（每个 Step 取一个关键量）。
Step E 的判据量取"截面逐格点最大偏差 / u_max"（比中心点更严：中心点误差仅 0.221%，
场量最大偏差 0.416%）。

### 4. Step E 的解析解（Boussinesq 级数）

截面 $[-a,a]^2$（边长 $L=2a$）、四壁无滑移、沿管轴均匀体力 $g$ 下，充分发展流满足
$\nabla^2 u=-g/\nu$、壁面 $u=0$，级数解

$$u(Y,Z)=\frac{16\,(g/\nu)\,a^2}{\pi^3}\sum_{n\ \text{odd}}\frac{(-1)^{(n-1)/2}}{n^3}
\left[1-\frac{\cosh(n\pi Z/2a)}{\cosh(n\pi/2)}\right]\cos\frac{n\pi Y}{2a}$$

半路反弹把壁面放在 $j=\pm(L+1)/2$，流体格点为 $Y,Z\in\{\pm\frac12,\dots,\pm(a-\frac12)\}$，
因此 LBM 与解析解是**逐格点**比较（不插值）。实现细节：

- `cosh` 比值一律用指数形式 $e^{-k(a-|Z|)}(1+e^{-2k|Z|})/(1+e^{-2ka})$ 计算，
  否则 $n$ 大时 `cosh(nπ/2)` 溢出（$n=2001$ 时指数已达 3100 量级）；
- 基准先自检再使用（`test_step_e_duct_series_satisfies_poisson`）：壁面值为 0（<1e-12 相对）、
  对称性精确成立、**解析求导**得到的 $\nabla^2u$（无需差分）与 $-g/\nu$ 的最大相对偏差
  随项数单调下降（201 项 1.9e-02 → 801 项 4.8e-03，判据 <1e-2）。
  这遵循"先给工具本身造解析基准，再做负控"的做法：级数若写错（系数/符号/坐标原点错位），
  后面 0.4% 的 LBM 误差就没有意义。

## 5. 实测结果（float64，设备 `cpu`）

| Step | 案例 | 关键量 | 数值解 | 解析解 | 误差 % | 判据 % | 判定 |
|---|---|---|---|---|---|---|---|
| A | A_shear_wave | 运动粘度 ν (lu²/ts) | 0.166667 | 0.166667 | 1.7218e-05 | 1 | PASS |
| B | B_couette | Couette 剖面峰值偏差 Δu_max (lu) | Δu_max = 1.0339e-08 | 0（解析剖面） | 2.0678e-05 | 1 | PASS |
| C | C_poiseuille | Poiseuille 峰值 u_max (lu) | 0.0124167 | 0.012375 | 0.336699 | 1 | PASS |
| D | D_friction | 壁面摩擦系数 Cf | 0.63492 | 0.634921 | 6.4436e-05 | 1 | PASS |
| E | E_duct_poiseuille | 方管截面最大速度 u_max | 0.0289062 | 0.0289704 | 0.415905 | 1 | PASS |

通过 **5/5**。"误差 %" 为报告量，"判据 %" 为门限。E 的其它口径（同一份 JSON）：
中心点误差 0.221%，L2 相对误差 0.403%，逐格点场图见 `fig_ladder.png` 右下两块。

## 6. dtype 敏感度（同判据、两个精度）

| Step | 误差 % (float64) | 误差 % (float32) | f32/f64 | 误差性质判定 |
|---|---|---|---|---|
| A | 1.7218e-05 | 0.00270013 | 157 | float32 舍入地板主导 |
| B | 2.0678e-05 | 0.00133678 | 64.6 | float32 舍入地板主导 |
| C | 0.336699 | 0.337789 | 1 | 离散/物理误差主导（dtype 不敏感） |
| D | 6.4436e-05 | 0.00404464 | 62.8 | float32 舍入地板主导 |
| E | 0.415905 | 0.415603 | 0.999 | 离散/物理误差主导（dtype 不敏感） |

读法：A/B/D 的 float32 误差比 float64 大 1–2 个数量级 ⇒ float64 残差是真方法误差，
float32 数字由舍入地板决定；C/E 的比值 ≈1.00 ⇒ 该残差是离散误差、与精度无关。
两个精度下都 5/5 通过：CI 用缺省 float64 断言，float32 作守卫。

## 7. Step E 的离散误差归因（网格 / 步数）

| 方管边长 L (lu) | 截面最大相对误差 % | 判据 % |
|---|---|---|
| 6 | 0.827292 | 1 |
| 10 | 0.415905 | 1 |
| 14 | 0.234596 | 1 |

| 时间步 | 截面最大相对误差 % |
|---|---|
| 1000 | 0.415905 |
| 2000 | 0.415905 |

读法：加密单调下降、加步数完全不变 ⇒ E 的 0.416% 是离散误差（稳态已到），
与 Step C 同性质。pytest 侧的守卫用例 `test_step_e_duct_error_is_discretization_limited`
断言 L=6→10 至少改善 1.5×（实测 1.99×）；若这条变红，先查网格映射，
不要靠放宽 1% 门限遮过去。

## 8. 判据的鉴别力（负控）

`test_no_dynamics_restore_must_use_pre_collision_snapshot` 刻意把 NoDynamics 回填错成
"碰撞后快照"，要求两种顺序 200 步后场量差 > 1e-3。实测（float64, 1000 步）该缺陷把
Couette 剖面误差从 2.1e-05% 抬到 **4.76%**、Poiseuille 从 0.337% 抬到 **17.17%**
——即 1% 判据确实抓得住"边界实现写错"这一类回归，不是恒绿的装饰。
Step E 的负控是"基准自检"（第 4 节）：级数解本身先被 Poisson 方程裁决，才允许用来裁决 LBM。

## 9. 误差判定（逐项 vs 基准）

| 关键量 | 基准来源 | 数值解 | 基准值 | 相对误差 | < 3%? | < 1%? |
|---|---|---|---|---|---|---|
| ν（Step A） | 解析 ν=(τ−½)/3 | 0.1666666380 | 0.1666666667 | 1.722e-05% | 是 | 是 |
| Couette 剖面 Δu_max | 解析线性剖面 | 1.034e-08 lu | 0 | 2.068e-05% | 是 | 是 |
| u_max（Step C） | Guo 稳态抛物线 | 0.01241667 | 0.01237500 | 0.3367% | 是 | 是 |
| Cf（Step D） | 2ν/(L·U) | 0.63492023 | 0.63492063 | 6.444e-05% | 是 | 是 |
| u_max（Step E） | Boussinesq 级数解 | 0.02890623 | 0.02897037 | 0.4159% | 是 | 是 |
| R=gL²/(νū)（Step E） | 级数解同批格点平均 | 28.0554 | 28.1213 | 0.2345% | 是 | 是 |
| R（Step E） | 文献 f·Re=56.91 ⇔ 28.455 | 28.0554 | 28.455 | 1.404% | 是 | 否（见下） |

六项关键量全部 **< 3%**；五项 < 1%。唯一 >1% 的是"R 对文献值"这一行，**不是求解器误差**：
Stokes 流在离散格点上取算术平均，等价于用节点值近似截面积分，本身带 ~1% 的求积偏差——
同一份级数解在 L=10 的节点平均是 28.1213，与连续积分值 28.455 差 1.17%；
加密到 L=14 后该偏差降到 0.60%（LBM 侧 28.2461 vs 28.455 → 0.734%）。
因此代码里对文献只设 3% 门（`test_step_e_duct_poiseuille`），对"同口径级数解"才设 1% 门；
两种门限的差别写在断言注释里，不拿口径差冒充精度。

## 10. 注意点与局限

1. **Step C 的 0.34% 是离散误差，不是"再跑久一点就归零"**：固定 1000→3000 步数值不变
   （0.3367%），与 Ma 无关（u_max 缩到 1/8 误差不变），但随网格二阶下降
   （ny=12 → 0.3367%，ny=16 → 0.1709%，比值 1.97 ≈ (14/10)²）。要用它做"更严的判据"就得加密网格，
   而不是压阈值。
2. **网格加密实测的坑**：用 ny=20 但沿用 1000 步时误差反而升到 0.54%——那是**没到稳态**
   （H²/ν ≈ 2.3k 步），不是离散误差变大。做加密研究必须同时放大步数，否则会得到错误的收敛方向。
   Step E 已把这条做进用例（L=10 在 1000 步与 2000 步误差完全相同 ⇒ 稳态判据成立）。
3. **节点平均 ≠ 截面积分**：把"节点算术平均"当连续积分去对文献值，会凭空造出 ~1.2% 的
   "求解器误差"，且它随网格加密才收敛（L=10 1.17% → L=14 0.60%）。报告积分量时必须写清口径。
4. **Cf 不是工程阻力系数**：Step D 的 Cf≡2τ_w/(ρU²)=2ν/(L·U) 是 Couette 型摩擦系数，
   与船体阻力系数 C_D（按湿表面积归一）不同尺度，不能直接互比。Step E 的 R=gL²/(νū)
   才是可与文献 f·Re 对表的无量纲阻力。
5. **A/B/D 的 float64 残差已接近双精度可分辨极限**（6e-05% 量级），
   再压阈值只会放大舍入噪声；鉴别力应靠网格/步数，而不是把 1% 改成 1e-6。
6. **未覆盖**：MRT/正则化碰撞、D3Q27、**曲面/浸没边界**（Step E 只覆盖阶梯方管）、
   多相与自由面、非稳态（Stokes 第二问题/Womersley）——这些案例裁决的是
   BGK + 半路反弹 + Guo 强迫这条最小路径在 2D 通道与 3D 方管上的正确性。
   > **补记（2026-11-08，m6517）**：其中「MRT/正则化碰撞」一项已由 **Step F** 补上——
   > A/B/C/E 四个算例对 bgk/mrt/trt/rlbm 各跑一遍，16 个组合全部 <1%（最大 0.4159%），
   > 且 TRT(Λ=3/16) 把 Step C 残差从 0.3367% 压到 6e-8，独立证明该残差是壁面位置误差。
   > 详见 `docs/operator_ladder_20261108.md`、用例 `test_step_f_*`。
   > 仍缺：D3Q27 / KBC / cascaded、完整 Guo 强迫耦合、τ→0.5 低粘区。
7. **运行时**：整条梯子 9 个用例在本机（128 核、load>100）约 81 s；`--refine` 的报告路径约 87 s；
   float32 视点约 35 s。案例都钉 1 线程（小张量上多线程反而慢），远低于 CI 600 s/用例预算。

## 11. 工具链与复现

```bash
# 主判据（float64）+ Step E 的网格/步数敏感性
PYTHONPATH=src python tests/test_lbm_verification.py --refine \
    --json artifacts/lbm_verification_ladder/metrics.json
# dtype 对照（float32）
PYTHONPATH=src TENSORLBM_TEST_DTYPE=float32 python tests/test_lbm_verification.py \
    --json artifacts/lbm_verification_ladder/f32/metrics.json
# 表格 + 图
PYTHONPATH=src python scripts/lbm_verification_report.py
# pytest 门（CI 跑的就是这条）
PYTHONPATH=src pytest -q tests/test_lbm_verification.py
```

产物：`artifacts/lbm_verification_ladder/{metrics.json, f32/metrics.json, tables.md,
fig_ladder.png, fig_dtype_error.png}`。`fig_ladder.png` = A 衰减曲线 + B/C 剖面 + D 摩擦系数
\+ E 方管对角剖面与截面云图（数值场 + 解析等值线），均与解析解同图；
`fig_dtype_error.png` = 两精度误差对照（对数轴 + 1% 阈值线）。

## 12. 变更清单

| 文件 | 变更 |
|---|---|
| `tests/test_lbm_verification.py` | **r3**：加 Step E（`duct_series` / `duct_profile_exact` / `duct_poiseuille_case` + 3 条用例：级数基准自检、方管剖面 1% 判据、离散误差归因）；`collect_metrics(..., refine=True)` 导出网格/步数敏感性；r2：`--json` 指标导出、`resolve_dtype()`、设备不可用 skip、Step A 衰减历史、Step B/C 的 y 坐标、float32 守卫用例 |
| `scripts/lbm_verification_report.py` | **r3**：`STEPS` 扩到 A–E、Step E 的两块新图（对角剖面 + 截面云图叠加解析等值线）、`refinement_table()` 离散误差归因表；**修 bug**：图函数用了 `np` 却没导入，matplotlib 可用时直接 `NameError`（原脚本在无 matplotlib 的机器上"看起来正常"） |
| `docs/lbm_verification_ladder_r2.md` | 本文件（r3：覆盖面 + 口径说明 + 变更清单） |
| `artifacts/lbm_verification_ladder/*` | 重新生成的实测 metrics / 表 / 图（含 E 与 refinement 段） |
| `tests/ci_quarantine.txt` | 无需改动：五个案例的 quarantine 行已在 2026-11-03 修复中删除（`grep lbm_verification` 为空） |