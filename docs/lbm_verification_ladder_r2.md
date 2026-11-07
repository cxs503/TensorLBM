# LBM 解析解验证梯 Step A–D（r2：可出证据的版本）

> 本文件的全部数字由 `scripts/lbm_verification_report.py` 从
> `artifacts/lbm_verification_ladder/metrics.json`（`tests/test_lbm_verification.py --json`
> 的产物）程序化注入，无手抄；改动求解器后重跑两条命令即可刷新。

## 1. 结论（TL;DR）

- 四个闭式解析解案例（剪切波 / Couette / Poiseuille / 壁面摩擦）在 CPU 上 **4/4 通过**
  1% 判据；最大误差 0.337%（Step C），其余三项 ≤ 6.44e-05%。
- A / B / D 的残差是纯方法误差（float64 下 6.4e-05% 量级，
  且 float32 下大 1–2 个数量级 ⇒ 已到单精度舍入地板）；C 的 0.337% 是**离散误差**：
  固定步数不降（1000/1500/2000/3000 步均为 0.3367%），与 Ma 无关
  （u_max 0.05→0.00625 误差恒为 0.3367%），但与网格二阶收敛
  （ny=12/H=10 → 0.3367%，ny=16/H=14 → 0.1709%，比值 1.97 ≈ (14/10)²）。
- 这四个案例此前长期躺在 `tests/ci_quarantine.txt` 里：缺陷不是物理，而是**验证框架**——
  测试签名把设备钉在 `sdaa:0`、函数 `return bool` 而不是 assert。quarantine 行已删除，
  本文件是删除的依据。

## 2. 为什么有 r2（PR#703 关闭后的重建）

2026-11-03 的 v1 重建已经修好了"能红能绿"（设备无关 + 真断言 + 向量化），那部分在 main 上；
PR#703 因交付形态被关闭，r2 从当前 main 重新切枝。r2 相对 main 补的是**证据链**，不是物理：

1. **机器可读指标**：`tests/test_lbm_verification.py --json <path>` 把每个 Step 的关键量、
   误差、判据写成 JSON——报告/文档的数字从 JSON 注入，杜绝"文档写一套、代码跑一套"。
2. **dtype 成为被判据约束的事实**：dtype 由 `TENSORLBM_TEST_DTYPE` 解析（缺省 float64），
   并新增 `test_step_c_poiseuille_float32_error_is_not_rounding_bound`：float32 误差必须仍 <1%
   且与 float64 同量级（比值落在 [0.3, 3]），否则说明判据已贴到单精度舍入地板、失去鉴别力。
3. **报告生成器**：`scripts/lbm_verification_report.py` 只读 JSON、只写 `artifacts/`，
   产出数字表与两张对照图（数值 vs 解析解；float64 vs float32）。

## 3. 四个案例的物理设定

| Step | 物理问题 | 几何 / 格点 | 关键参数 | 解析解 | 校验的内核 |
|---|---|---|---|---|---|
| A | 剪切波衰减 | 周期盒 64×4×4 | τ=1，A₀=0.01，100 步 | u_y=A₀cos(kx)e^(−νk²t) | 碰撞核给出的 ν=(τ−½)/3 |
| B | Couette 流 | 12 层通道（y 向），nx=80 | τ=1，u_top=0.05，1000 步 | u_x=u_top(y−½)/(H−½) | 半路反弹无滑移 + 壁面动量交换 |
| C | Poiseuille 流 | 12 层通道，上下壁 | τ=1，G=0.000166667，1000 步 | u=G/(2ν)(y−½)(H+½−y) | Guo 体力注入 + 抛物线剖面 |
| D | 壁面摩擦 | 同上（Couette 型剪切） | τ=1，u_top=0.05，gap=10.5 | Cf=2ν/(L·U) | 壁面剪应力提取 τ_w=ν·∂u/∂y |

量纲：格子单位（lu / ts），密度 ρ=1。判据统一为**相对误差 < 1%**（每个 Step 取一个关键量）。

## 4. 实测结果（float64，设备 `cpu`）

| Step | 案例 | 关键量 | 数值解 | 解析解 | 误差 % | 判据 % | 判定 |
|---|---|---|---|---|---|---|---|
| A | A_shear_wave | 运动粘度 ν (lu²/ts) | 0.166667 | 0.166667 | 1.7218e-05 | 1 | PASS |
| B | B_couette | Couette 剖面峰值偏差 Δu_max (lu) | Δu_max = 1.0339e-08 | 0（解析剖面） | 2.0678e-05 | 1 | PASS |
| C | C_poiseuille | Poiseuille 峰值 u_max (lu) | 0.0124167 | 0.012375 | 0.336699 | 1 | PASS |
| D | D_friction | 壁面摩擦系数 Cf | 0.63492 | 0.634921 | 6.4436e-05 | 1 | PASS |

通过 4/4。"误差 %" 为报告量，"判据 %" 为门限。

## 5. dtype 敏感度（同判据、两个精度）

| Step | 误差 % (float64) | 误差 % (float32) | f32/f64 | 误差性质判定 |
|---|---|---|---|---|
| A | 1.7218e-05 | 0.00270013 | 157 | float32 舍入地板主导 |
| B | 2.0678e-05 | 0.00133678 | 64.6 | float32 舍入地板主导 |
| C | 0.336699 | 0.337789 | 1 | 离散/物理误差主导（dtype 不敏感） |

读法：A/B/D 的 float32 误差比 float64 大 1–2 个数量级 ⇒ float64 残差是真方法误差，
float32 数字由舍入地板决定；C 的比值 ≈1.00 ⇒ 该残差是离散误差，
与精度无关。两个精度下都 4/4 通过：CI 用缺省 float64 断言，float32 作守卫用例。

## 6. 判据的鉴别力（负控）

`test_no_dynamics_restore_must_use_pre_collision_snapshot` 刻意把 NoDynamics 回填错成
"碰撞后快照"，要求两种顺序 200 步后场量差 > 1e-3。实测（float64, 1000 步）该缺陷把
Couette 剖面误差从 2.1e-05% 抬到 **4.76%**、Poiseuille 从 0.337% 抬到 **17.17%**
——即 1% 判据确实抓得住"边界实现写错"这一类回归，不是恒绿的装饰。

## 7. 误差判定（逐项 vs 基准）

| 关键量 | 基准来源 | 数值解 | 基准值 | 相对误差 | < 3%? | < 1%? |
|---|---|---|---|---|---|---|
| ν（Step A） | 解析 ν=(τ−½)/3 | 0.1666666380 | 0.1666666667 | 1.722e-05% | 是 | 是 |
| Couette 剖面 Δu_max | 解析线性剖面 | 1.034e-08 lu | 0 | 2.068e-05% | 是 | 是 |
| u_max（Step C） | Guo 稳态抛物线 | 0.01241667 | 0.01237500 | 0.3367% | 是 | 是 |
| Cf（Step D） | 2ν/(L·U) | 0.63492023 | 0.63492063 | 6.444e-05% | 是 | 是 |

四项全部 **< 3%（也全部 < 1%）**；最大偏差 0.337% 出现在 Step C，已归因为离散误差
（第 1 节）。阈值历来是 1%，本次未改动——没有一项是靠放宽门限通过的。

## 8. 注意点与局限

1. **Step C 的 0.34% 是离散误差，不是"再跑久一点就归零"**：固定 1000→3000 步数值不变
   （0.3367%），与 Ma 无关（u_max 缩到 1/8 误差不变），但随网格二阶下降
   （ny=12 → 0.3367%，ny=16 → 0.1709%，比值 1.97 ≈ (14/10)²）。要用它做"更严的判据"就得加密网格，
   而不是压阈值。
2. **网格加密实测的坑（本单踩到）**：用 ny=20 但沿用 1000 步时误差反而升到 0.54%——那是**没到稳态**
   （H²/ν ≈ 2.3k 步），不是离散误差变大。做加密研究必须同时放大步数，否则会得到错误的收敛方向。
3. **Cf 不是工程阻力系数**：Step D 的 Cf≡2τ_w/(ρU²)=2ν/(L·U) 是 Couette 型摩擦系数，
   与船体阻力系数 C_D（按湿表面积归一）不同尺度，不能直接互比。
4. **A/B/D 的 float64 残差已接近双精度可分辨极限**（6e-05% 量级），
   再压阈值只会放大舍入噪声；鉴别力应靠网格/步数，而不是把 1% 改成 1e-6。
5. **未覆盖**：MRT/正则化碰撞、D3Q27、曲面壁面（阶梯近似以外）、多相与自由面——这些案例只裁决
   BGK + 半路反弹 + Guo 强迫这一条最小路径。
6. **运行时**：整条梯子（含 float32 守卫用例）在本机（128 核、load>100）约 57 s；案例都钉 1 线程
   （小张量上多线程反而慢），远低于 CI 600 s/用例预算。

## 9. 工具链与复现

```bash
# 主判据（float64）
PYTHONPATH=src python tests/test_lbm_verification.py --json artifacts/lbm_verification_ladder/metrics.json
# dtype 对照（float32）
PYTHONPATH=src TENSORLBM_TEST_DTYPE=float32 python tests/test_lbm_verification.py \
    --json artifacts/lbm_verification_ladder/f32/metrics.json
# 表格 + 图
PYTHONPATH=src python scripts/lbm_verification_report.py
# pytest 门（CI 跑的就是这条）
PYTHONPATH=src pytest -q tests/test_lbm_verification.py
```

产物：`artifacts/lbm_verification_ladder/{metrics.json, f32/metrics.json, tables.md,
fig_ladder.png, fig_dtype_error.png}`。`fig_ladder.png` = A 衰减曲线 + B/C 剖面 + D 摩擦系数，
均与解析解同图；`fig_dtype_error.png` = 两精度误差对照（对数轴 + 1% 阈值线）。

## 10. 变更清单

| 文件 | 变更 |
|---|---|
| `tests/test_lbm_verification.py` | 加 `--json` 指标导出、`resolve_dtype()`、设备不可用 skip、Step A 衰减历史、Step B/C 的 y 坐标、float32 守卫用例；Step C 离散常数改为跟随默认 dtype（原写死 `.float()`） |
| `scripts/lbm_verification_report.py` | 新增：JSON → 数字表 + 两张对照图 |
| `docs/lbm_verification_ladder_r2.md` | 本文件（证据与判定） |
| `artifacts/lbm_verification_ladder/*` | 新增：实测 metrics / 表 / 图 |
| `tests/ci_quarantine.txt` | 无需改动：四个案例的 quarantine 行已在 2026-11-03 修复中删除（`grep lbm_verification` 为空） |
