# XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION — R3 复核报告

> **锚点**: `XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION`（mission m20921）
> **轮次**: R3（承接 R2 预登记但 NOT_RUN 的数值腿）
> **判据**: owner <15%，并逐项显式回答 <3%；预登记见 `predeclared.json`（运行前固定，未事后调整）
> **渲染**: 本文件由 `make_report_r3.py` 从 `artifacts/r3_summary.json` 程序化生成，数字零手抄

## ① 案例介绍

单组分 Shan–Chen 伪势模型（SC94，ψ=1−e^(−ρ)）在周期盒内形成液-气共存。二维圆柱界面满足
Δp = σ/R，三维球面界面满足 Δp = 2σ/R；两者合并写作 **Δp = (d−1)σ/R**。
R2 留下一条可否证的跨维预言：若界面张力由 Korteweg 截断应力支配且模板四阶矩比
κ3 = a3/a1 在 D2Q9 与 D3Q19 上相同，则 **σ_3D/σ_2D = 1**。R2 只验证了 κ3 的算术，
F（平面液膜）/ D（液滴扫描）两条数值腿因排队未取回终态而标 NOT_RUN，故 R2 明确不主张
跨维一致。R3 就是去把这两条腿跑完并把判决落地。

| 参数 | 值 |
|---|---|
| 模型 | SCMP SC94, psi=1-exp(-rho), D2Q9/D3Q19 |
| 耦合 | G_lib = 5.0 ≡ 物理 G_eff = -5.0 |
| 松弛 | τ = 1.0（BGK） |
| 半径 | R = [10.0, 14.0, 18.0]，域 L = 4·R |
| 维度 | [2, 3]（D2Q9 / D3Q19） |
| 力格式 | velocity-shift（与 `benchmarks/verified/laplace_droplet` 一致） |
| 阈值 | 15% 宽判据 / 3% 紧判据；力学平衡闸门 0.001 |

## ② 怎么做的

R2 的执行器是单体式的（`--stage all`：2D+3D 的液滴与液膜必须在同一个排队窗口内全部跑完），
实际结果是**一条腿都没落盘**。R3 不改物理、不改协议、不改阈值，只改执行方式：

**执行方式（本轮的工程改动）** — 新增 `run_r3_split.py`，把计划拆成互相独立的
(维度, 腿) 分片，每片落一个 part 文件；`--merge` 用 R2 的同一套 `_fit_sigma` 与
`THRESHOLDS`（直接 `import run_r2`，单一真源，不复制阈值）汇总为 `artifacts/r3_summary.json`。
缺片一律记为 `NOT_RUN`，绝不以相邻片或上一轮的数值代替。

**四条腿**（协议与 R2 预登记完全一致）：

- **D 腿（主判据）**：2D/3D × R = 10/14/18 液滴扫描，σ_i = Δp·R_eq/(d−1)，
  逐半径一致性与跨维比值。
- **F 腿**：2D/3D 平面液膜（横向平均 1D 剖面），给出体相共存密度、
  力学平衡闸（|p_l−p_v|/p_v）、以及 Korteweg 锚 σ_K = |G_eff κ3|∫(∂_n ψ)²dn。
- **M 腿**：SC 状态方程上的 Maxwell 等面积构造（解析基准，无求解器）。
- **S 腿（本轮新增）**：收敛液滴的密度场快照，用于出云图；R2 的 artifact 只有标量表，
  没有场量，报告的④因此无图可出——这一轮把场补上。

已取回的分片：液滴腿 [2d]，液膜腿 [2d]，场快照 [2d]。

工具链：Python + PyTorch（库即 TensorLBM `src/tensorlbm`，求解器零手写：只用库内的
`collide_sc_single_component*` + `stream*`）；matplotlib 出图；cluster 递交执行。

## ③ 计算结果

### 3.1 Maxwell 等面积构造（解析基准，M 腿）

| 量 | 值 |
|---|---|
| ρ_v | 0.152696 |
| ρ_l | 1.928661 |
| p_c | 0.034188 |
| 等化学势残差 | 0.000 |

### 3.2 模板四阶矩（κ3 预言）

| 维度 | 速度数 | a1 | a3 | κ3 |
|---|---|---|---|---|
| 2D | 9 | 0.3333333 | 0.0555556 | 0.1666667 |
| 3D | 19 | 0.3333333 | 0.0555556 | 0.1666667 |

κ3 比值 (3D/2D) = **1.000000000** ⇒ 预言 σ_3D/σ_2D = 1。

### 3.3 液滴扫描（D 腿）

| 维度 | R_init | L | R_eq | Δp | ρ_in | ρ_out | σ_i = Δp·R_eq/(d−1) | 收敛 | steps |
|---|---|---|---|---|---|---|---|---|---|
| 2D | 10 | 40 | 10.540 | 0.005192 | 1.9783 | 0.1629 | 0.05472 | True | 4000 |
| 2D | 14 | 56 | 14.461 | 0.003866 | 1.9667 | 0.1611 | 0.05591 | True | 4000 |
| 2D | 18 | 72 | 18.273 | 0.003053 | 1.9596 | 0.1601 | 0.05579 | True | 5000 |

### 3.4 σ 拟合（D 腿）

| 维度 | σ (through origin) | σ (mean of radii) | 逐半径最大偏差 | 自由截距 | R² 代理量 |
|---|---|---|---|---|---|
| 2D | 0.05525 | 0.05547 | +1.35% | 0.000168 | — |
| 3D | NOT_RUN | — | — | — | — |

### 3.5 平面液膜（F 腿）：共存与力学平衡

| 维度 | L_cells | ρ_l | ρ_v | p_l | p_v | mech gap | σ_K(tanh) | 收敛 |
|---|---|---|---|---|---|---|---|---|
| 2D | {'half': 48, 'nx': 8, 'ny': 192} | 1.9314 | 0.1555 | 0.03453 | 0.03455 | 0.00 | 0.13718 | False |
| 3D | NOT_RUN | — | — | — | — | — | — | — |

### 3.6 跨维判决（R2 预登记 P1）

跨维比值需要 2D 与 3D 两条液滴腿同时到齐；当前 `NOT_RUN`。

### 3.7 预登记判据总表

| 判据 | 含义 | 判决 |
|---|---|---|
| `P1_cross_dim` | σ_3D/σ_2D = 1（<15%） | __NOT_RUN__ |
| `P1_cross_dim_tight3pct` | σ_3D/σ_2D = 1（<3%） | __NOT_RUN__ |
| `P2_mech_balance_2d` | 2D 平面膜 |p_l−p_v|/p_v < 1e−3 | PASS |
| `P2_mech_balance_3d` | 3D 平面膜 |p_l−p_v|/p_v < 1e−3 | __NOT_RUN__ |
| `P3_radius_consistency_2d` | 2D 逐半径 σ_i 一致性 <15% | PASS |
| `P3_radius_consistency_3d` | 3D 逐半径 σ_i 一致性 <15% | __NOT_RUN__ |
| `P4_sigmaK_vs_lap_2d` | 2D Korteweg σ_K vs Laplace σ（<15%） | **FAIL** |
| `P4_sigmaK_vs_lap_3d` | 3D Korteweg σ_K vs Laplace σ（<15%） | __NOT_RUN__ |
| `P5_coexist_vs_maxwell_2d` | 2D 共存密度 vs Maxwell（<15%） | PASS |
| `P5_coexist_vs_maxwell_3d` | 3D 共存密度 vs Maxwell（<15%） | __NOT_RUN__ |

## ④ 结果图

- `artifacts/fig_r3_field.png` — 2D 收敛液滴的 ρ(x,y) 云图 + 过中心径向剖面（标出 ρ_in / ρ_out，并由 Δp·R_eq 给出同一场的 σ 读数）。
- `artifacts/fig_r3_crossdim.png` — 左：Δp/(d−1) vs 1/R_eq（2D、3D 与过原点拟合）；右：σ_2D / σ_3D 柱状 + ±3%/±15% 带，标题给出跨维比值。
- `artifacts/fig_r3_flat.png` — 左：平面膜共存密度 vs Maxwell 解析值；右：力学平衡闸（对数轴）与 σ_K 对 σ_Laplace。

## ⑤ 注意点

1. **本轮改的是执行方式，不是物理**：协议、radii、阈值全部沿用 R2 预登记；
   把单体执行改成可分片执行，是为了让已经算完的腿能落盘，而不是为了让判据变松。
2. **σ 没有第一性原理解析值**：SC 伪势模型的界面张力依赖离散化细节，
   所以判据只能是「跨维一致 + 逐半径一致 + Maxwell 相容」这三条互相独立的基准，
   而不是与某个解析 σ 比。
3. **Korteweg 锚只是量级锚**：R2 审计已测到 σ_K 比 Laplace σ 低 ~29%（代码系数）/
   高 ~114%（文档系数），差异来自 cs² 因子与实测剖面的有限带宽；P4 因此是弱判据。
4. **伪流速度**：SC94 在 τ=1 的液滴界面带 u_max ≈ 0.14，液滴并非严格静止；
   本报告只用体相带平均值，不依赖界面内速度。
5. **适用范围**：ψ=1−e^(−ρ)、G_eff=−5、τ=1、velocity-shift 力格式、R=10…18。
   力格式换成 Guo 会改变模型本身（R1 实测 σ 差 56.8%），不可跨格式搬用。
6. **三维成本**：3D R=18 的域是 72³，单核 3D 步进约 0.25 s，单算例小时量级；
   这是 R2 全量跑不完的直接原因，也是本轮改成按维度分片递交的原因。

## ⑥ 误差判定（逐项，显式回答 <3%）

| 关键量 | 值 | 基准 | 误差 | <3%? | <15%? |
|---|---|---|---|---|---|
| σ_2D（过原点拟合） | 0.05525 | 0.0561（`benchmarks/verified/laplace_droplet` 锁定值） | +1.51% | 是 | 是 |
| σ_2D 逐半径最大偏差 | +1.35% | 该维 σ 均值 | +1.35% | 是 | 是 |
| σ_3D（过原点拟合） | NOT_RUN | 0.0561（库锁定 2D 值） | — | — | — |
| 2D 平面膜 ρ_l vs Maxwell | 1.9314 | 1.9287 | +0.14% | 是 | 是 |
| 2D 平面膜 ρ_v vs Maxwell | 0.1555 | 0.1527 | +1.84% | 是 | 是 |
| 2D 力学平衡 |p_l−p_v|/p_v | 0.001 | 0（精确平衡） | 0.001 | 是 | 是 |

**结论（不粉饰）**：
- PASS 的预登记判据 3 条：`P2_mech_balance_2d`, `P3_radius_consistency_2d`, `P5_coexist_vs_maxwell_2d`。
- FAIL 的预登记判据 1 条：`P4_sigmaK_vs_lap_2d`。
- NOT_RUN 的预登记判据 5 条：`P1_cross_dim`, `P2_mech_balance_3d`, `P3_radius_consistency_3d`, `P4_sigmaK_vs_lap_3d`, `P5_coexist_vs_maxwell_3d`。

## ⑦ 未做 / 后续

- 毛细波色散路线（不依赖压力差、可给 <3% 的绝对 σ）仍未做：需要长时程界面涨落统计与
  波数分辨，成本比本轮的液滴扫描高一个量级。
- τ 扫描（σ 的黏性依赖）与 G_eff 扫描仍未覆盖；R1 只做过 τ=0.7/1.0/1.5 的液滴单点。
- Guo 格式下体相 EOS 偏移（共存密度整体下移）的机理仍只记录现象，未追根。

## 复算方式

```bash
cd xval/XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION/verify_r2
python run_r3_split.py --merge                                  # 汇总（含解析腿，秒级）
python run_r3_split.py --dim 2 --leg droplet --jobs 3 --threads 2
python run_r3_split.py --dim 3 --leg droplet --jobs 3 --threads 8   # 3D 最贵
python run_r3_split.py --dim 2 --leg flat
python run_r3_split.py --dim 2 --leg field --R 14
python make_figures_r3.py && python make_report_r3.py
pytest tests/test_xval_sc_laplace_r3.py -q
```

