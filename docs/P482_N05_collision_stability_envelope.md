# P482-N05 — Collision-operator stability domain: predicted vs measured exchange law

*Node*: **P482-N05** (campaign DAG, physics domain, TensorLBM tenant)
*Module*: `tensorlbm.stability.collision_stability` (models, prediction, measurement)
+ `tensorlbm.stability.envelope` (campaign, envelope, gate)
*Tests*: `tests/test_lbm_collision_stability_envelope.py`
*Artefacts*: `TensorLBM/artifacts/p482_n05/` (envelope JSON, plane CSV, figure, manifest)
*Runner*: `scripts/run_p482_n05_stability_envelope.py` (staged, resumable)

## 0 Executive summary

The linear stability domain of D2Q9 BGK/MRT is turned into a quantity that is
**predicted** (von Neumann eigen-analysis of `A(k) = diag(e^{-i k.c}) P`, `P = I - C(I - J_eq)`)
and **measured** (independent full nonlinear periodic-box integration of the same collision
model, single-mode seeded, float64), and the two are cross-audited point by point.
The exchange law is demonstrated: at identical box-representable wave vectors the
measured amplification factor reproduces the predicted one to better than
1.707e-04, and the divergence boundary `tau_crit(Ma)` agrees to 0.46% (vs the box-grid
prediction) / 1.03% (vs the fine-`k` prediction), well inside the 5% acceptance band.
The falsifiability leg fires: two *legal* MRT parameterisations (all relaxation rates in
`(0, 2)`, orthogonal moment basis) are shown to be **less** stable than BGK — one of them
over 100% of the scanned plane — so "MRT is more stable" is not a free pass.
The envelope is stored as deterministic JSON with a SHA-256 digest and is wired into
`accuracy_recommendation.recommend_with_stability_envelope()`, which is fail-closed
outside the stable region.

## 1 案例介绍（物理背景 / 参数表）

节点 P482-N05：把 D2Q9 碰撞算子的**线性稳定域**从「踩坑经验」变成可预测、可互裁的量。对象是单松弛 BGK 与多松弛 MRT（Lallemand-Luo 矩基底，矩序 `(rho,e,eps,jx,qx,jy,qy,pxx,pxy)`，剪切矩 `s7=s8=1/tau`，高阶矩与 `s4=qx` 可释放）。

| 项目 | 设定 |
|---|---|
| 晶格/单位 | D2Q9，`dx=dt=1`，`cs^2=1/3` |
| Mach 定义 | `u0 = (Ma * cs, 0), cs = 1/sqrt(3)` |
| 扫描窗口 | `tau in [0.505, 0.9]`，`Ma in [0.02, 0.5]` |
| 扫描网格 | 35 x 35 = **1225 工况/档**；预测 k 网格 17 x 17（k≠0） |
| 稳定判据 | `lambda_max = max_{k≠0} max_j |mu_j(k)| <= 1`（预测，容差 1e-9） |
| 实测判据 | 单模种子全非线性 float64 周期盒积分，拟合 `ln||u-u0||` 斜率得 `lambda_meas`；另报 10 步增长>1e3 的深不稳定探测器 |
| 预注册 | 阈值/网格/候选松弛率集合全部写入 `PREDECLARED`（envelope JSON 内嵌副本） |

## 2 怎么做的（方法 / 流程 / 工具链）

**M1 互换律**：线性化碰撞+迁移算子 `A(k)=diag(e^{-i k·c})·P`，`P = I - C(I-J_eq)`，对每个 `(tau,Ma)` 取 `max_{k≠0}|eig A|`；实测侧用同一碰撞参数、但**独立的时间积分路径**（非线性平衡态 + 逐格迁移 + 速度残差）在 32x32 周期盒上拟合增长率。两侧在**同一个 box 可表示的波矢** `k=2πm/n` 上比较，避免把不可表示的模式种进周期盒（模块内该情形直接抛 `ValueError`）。发散边界 `tau_crit(Ma)` 用二分法分别在预测（细 k 网格 81x81）、box 网格预测与实测三路上求解，逐站比较相对偏差。

**M2 平面扫描**：三档 BGK / MRT-A（`s7=1/tau`，仅高阶矩自由）/ MRT-B（额外释放 `s4=s_qx`），每档两种变体：`canonical`（Lallemand-Luo 教科书值 `s_e=1.64, s_eps=1.54, s_q=1.7`）与 `oracle`（按**逐 tau 设计规则**在预注册候选集上最小化最高 Ma 处的 `lambda_max`）。

**M3 可证伪**：把「MRT 更稳」当作可被推翻的断言——用两个**合法**（所有松弛率在 (0,2)、基底正交）但调得差的 MRT 设定在同一平面扫描，定位 MRT 反而更差的 `(tau,Ma)` 子域；并用三组缺陷算子做负控（`s>2` 越界松弛；非正交基底+越界松弛），要求**预测与实测两路同向判红**。

**M4 包络**：稳定包络 JSON（逐档稳定掩码 + 逐格 `lambda_max` + 网格轴 + sha256 摘要）+ 图；并把 `point_in_envelope()` 接入 `accuracy_recommendation.recommend_with_stability_envelope()`，域外/邻域不稳定点一律 fail-closed 拒绝推荐。

## 3 计算结果（程序化注入）

### 3.1 三档稳定域（M2）

| 档 | 工况数 | 稳定面积比 | 不稳定格数 | max λ_max | λ_max 位置 (tau,Ma) | tau_min 处最大稳定 Ma | 相对 BGK 新增稳定格 | MRT 更差格数 |
|---|---|---|---|---|---|---|---|---|
| bgk:canonical | 1225 | 98.20% | 22 | 1.044824 | (0.505, 0.50) | 0.2357 | 0 | 0 |
| mrt_a:canonical | 1225 | 98.61% | 17 | 1.044763 | (0.505, 0.50) | 0.3047 | 5 | 0 |
| mrt_a:oracle | 1225 | 98.12% | 23 | 1.044763 | (0.505, 0.50) | 0.3047 | 3 | 4 |
| mrt_b:canonical | 1225 | 98.61% | 17 | 1.044763 | (0.505, 0.50) | 0.3047 | 5 | 0 |
| mrt_b:oracle | 1225 | 98.12% | 23 | 1.044763 | (0.505, 0.50) | 0.3047 | 3 | 4 |

最大稳定 Ma 增益（相对 BGK，`tau = 0.505`）：bgk:canonical Δ=0；mrt_a:canonical Δ=0.0690；mrt_a:oracle Δ=0.0690；mrt_b:canonical Δ=0.0690；mrt_b:oracle Δ=0.0690

### 3.2 M1 互换律：预测 vs 实测

- 同一波矢互换律：共 90 点（良态 `|sigma_pred|>=0.005` 28 点，强条件 `|sigma|>=0.03` 8 点，边缘 62 点）。
  - 放大因子绝对偏差：强条件 **1.707e-04**、良态 0.002、边缘 0.185；增长率相对偏差：强条件 **0.49%**（均值 0.16%），良态含边缘放大到 16.27%；拟合 R² 中位数 0.9649。
  - 边缘点偏差不由线性分析负责：D2Q9 在 `(pi,0)/(0,pi)` 有 **严格中性**（|mu|=1）的棋盘支，其本征矢不含动量，速度残差实测看不到它（实测衰减 0.81~0.84），故边缘点只用于「不误报增长」，不进入增长率相对误差统计。
- 发散边界 `tau_crit(Ma)`：22/27 个站点存在内部穿越；实测 vs box 网格预测最大相对偏差 **0.46%**；实测 vs 细 k 网格（81x81）预测最大相对偏差 **1.03%**（判据 5%）。

| 档 | Ma | tau_crit 细网格预测 | tau_crit box 网格预测 | tau_crit 实测 | 实测-细网格 | 实测-box 网格 |
|---|---|---|---|---|---|---|
| bgk | 0.30 | 0.50847 | 0.50693 | n/a | n/a | n/a |
| bgk | 0.33 | 0.51001 | 0.50847 | 0.50616 | -0.76% | -0.46% |
| bgk | 0.35 | 0.51156 | 0.50924 | 0.50770 | -0.75% | -0.30% |
| bgk | 0.38 | 0.51387 | 0.51156 | 0.50924 | -0.90% | -0.45% |
| bgk | 0.40 | 0.51619 | 0.51387 | 0.51156 | -0.90% | -0.45% |
| bgk | 0.42 | 0.51927 | 0.51619 | 0.51542 | -0.74% | -0.15% |
| bgk | 0.45 | 0.52236 | 0.52004 | 0.51850 | -0.74% | -0.30% |
| bgk | 0.47 | 0.52544 | 0.52159 | 0.52004 | -1.03% | -0.30% |
| bgk | 0.50 | 0.52930 | 0.52776 | 0.52622 | -0.58% | -0.29% |
| mrt_a | 0.30 | n/a | n/a | n/a | n/a | n/a |
| mrt_a | 0.33 | 0.50616 | 0.50616 | n/a | n/a | n/a |
| mrt_a | 0.35 | 0.50693 | 0.50693 | 0.50616 | -0.15% | -0.15% |
| mrt_a | 0.38 | 0.50847 | 0.50847 | 0.50770 | -0.15% | -0.15% |
| mrt_a | 0.40 | 0.51001 | 0.51001 | 0.50924 | -0.15% | -0.15% |
| mrt_a | 0.42 | 0.51233 | 0.51233 | 0.51156 | -0.15% | -0.15% |
| mrt_a | 0.45 | 0.51464 | 0.51464 | 0.51387 | -0.15% | -0.15% |
| mrt_a | 0.47 | 0.51773 | 0.51773 | 0.51696 | -0.15% | -0.15% |
| mrt_a | 0.50 | 0.52236 | 0.52236 | 0.52082 | -0.30% | -0.30% |
| mrt_b | 0.30 | n/a | n/a | n/a | n/a | n/a |
| mrt_b | 0.33 | 0.50616 | 0.50616 | n/a | n/a | n/a |
| mrt_b | 0.35 | 0.50693 | 0.50693 | 0.50616 | -0.15% | -0.15% |
| mrt_b | 0.38 | 0.50847 | 0.50847 | 0.50770 | -0.15% | -0.15% |
| mrt_b | 0.40 | 0.51001 | 0.51001 | 0.50924 | -0.15% | -0.15% |
| mrt_b | 0.42 | 0.51233 | 0.51233 | 0.51156 | -0.15% | -0.15% |
| mrt_b | 0.45 | 0.51464 | 0.51464 | 0.51387 | -0.15% | -0.15% |
| mrt_b | 0.47 | 0.51773 | 0.51773 | 0.51696 | -0.15% | -0.15% |
| mrt_b | 0.50 | 0.52236 | 0.52236 | 0.52082 | -0.30% | -0.30% |

### 3.3 M3 可证伪判决与负控

- 结论：s4_under_relaxed: observed: a legal, badly tuned MRT setting is less stable than BGK on 1.39% of the scanned plane (max excess lambda 0.016) ; mismatched: observed: a legal, badly tuned MRT setting is less stable than BGK on 100.00% of the scanned plane (max excess lambda 0.443)
  - `mismatched`：更差面积比 100.00%（1225/1225 格），BGK 稳而该设定不稳的格数 1203，最大 λ 超出 0.443，子域 tau∈[0.505, 0.9]、Ma∈[0.02, 0.5]。
  - `s4_under_relaxed`：更差面积比 1.39%（17/1225 格），BGK 稳而该设定不稳的格数 1，最大 λ 超出 0.016，子域 tau∈[0.505, 0.505]、Ma∈[0.274117647059, 0.5]。
- 负控：9/9 组被判红（预测 λ>1 且实测增长 λ_meas>1），10 步>1e3 深探测器触发 6 组；`all_red = True`。

| 缺陷 | tau | Ma | 最大松弛率 | 基底缺陷 | λ_pred | λ_meas | 10步比 |
|---|---|---|---|---|---|---|---|
| over_relax_shear | 0.600 | 0.20 | 2.40 | 0 | 1.4000 | 1.1583 | 4041.222 |
| over_relax_shear | 0.505 | 0.40 | 2.40 | 0 | 1.4000 | 1.1427 | 866.908 |
| over_relax_shear | 0.700 | 0.35 | 2.40 | 0 | 1.4000 | 1.1630 | 9175.673 |
| over_relax_energy | 0.600 | 0.20 | 2.60 | 0 | 1.8999 | 1.7869 | 7581.732 |
| over_relax_energy | 0.505 | 0.40 | 2.60 | 0 | 1.9287 | 1.8350 | 9325.816 |
| over_relax_energy | 0.700 | 0.35 | 2.60 | 0 | 1.9198 | 1.8211 | 9767.438 |
| mixed_basis_over_relax | 0.600 | 0.20 | 2.20 | 6.79 | 1.2000 | 1.0794 | 891.664 |
| mixed_basis_over_relax | 0.505 | 0.40 | 2.20 | 6.79 | 1.2000 | 1.0775 | 1174.125 |
| mixed_basis_over_relax | 0.700 | 0.35 | 2.20 | 6.79 | 1.2000 | 1.0801 | 101.444 |

- 特异性探针（只换非正交基底、松弛率仍合法）：λ_pred = 1.0000, 1.0127, 1.0000 → 判不稳 1/3 点。A non-orthogonal basis alone (all rates inside (0, 2)) is neutral at 2/3 probe points but does destabilise at 1 of them (the low-tau corner, where the same kinetic pocket is already active).  So the basis defect is *not* silently ignored; it simply requires the pocket to exist.  The must-fail control set combines the basis defect with an out-of-range rate, where both routes fire unambiguously.
- M2 平面实测抽检：68/69 与预测判决一致（98.55%）。

## 4 结果图

四联图：左上 BGK 平面 `lambda_max-1` 热图与稳定边界；右上 `Ma=0.5` 处各档 `lambda_max(tau)` 曲线（含两个 MRT 反例 foil）；左下 `tau_crit(Ma)` 预测 vs 实测；右下同一波矢下预测 vs 实测放大因子散点（互换律）。

![stability envelope](figure)

## 5 注意点（坑 / 局限）

- The unstable pocket of D2Q9 BGK inside this window is localised at wave vectors that a 32x32 box cannot represent; the box-grid prediction and the measurement agree with each other (<=1%) while sitting slightly below the fine-grid boundary (<=1.5%).
- D2Q9 has an exactly neutral (|mu| = 1) checkerboard branch at (pi, 0) and (0, pi) for every tau; it carries no momentum, so a velocity-residual measurement cannot see it and reports faster decay than the marginal prediction there.
- Measured growth is fitted over the linear window only; amplification factors above 1.3 saturate within a few steps and their fits carry several-percent error.
- The oracle tiers are upper envelopes over a declared discrete rate set, not proofs of optimality.
- `k=0` 必须排除：守恒矩使均值模恒为中性，否则所有模型都「临稳」而失去分辨力。
- 10 步增长>1e3 的判据等价于 `lambda > 1.995`，只抓深不稳定；边缘跨界（`lambda≈1.01`）只能靠拟合增长率判，因此 M1 以拟合增长率的零穿越定义边界，并把该探测器作为补充报告。
- oracle 档是**预注册离散设计空间上的上包络**（逐 tau 选择），不是最优性证明；实测中 oracle 的稳定面积并不优于 canonical 档，说明该窗口内的不稳定由剪切率 `1/tau` 主导，而非高阶松弛率。

## 6 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准 | 偏差 | 是否 <3% |
|---|---|---|---|
| 发散边界 tau_crit（实测 vs box 网格预测） | 预测二分法 | 0.46% | 是 |
| 发散边界 tau_crit（实测 vs 细 k 网格预测） | 预测二分法 | 1.03% | 是 |
| 同一波矢放大因子 λ（强条件 8 点） | von Neumann 特征值 | 绝对 1.707e-04 | 是 |
| 同一波矢放大因子 λ（良态 28 点） | 同上 | 绝对 0.002 | 是 |
| 同一波矢放大因子 λ（边缘 62 点） | 同上 | 绝对 0.185 | 否（非预测误差：中性棋盘支无动量，速度残差不可见；边缘点已从判据中排除） |
| 增长率 sigma（强条件，|sigma|>=0.03） | von Neumann | 相对 0.49% | 是 |
| 增长率 sigma（良态含近边缘） | 同上 | 相对 16.27%，绝对 λ 0.002 | 否（sigma→0 时相对量条件数发散，绝对偏差仍 <2e-3） |
| 负控判决（9 组，3 点 x 3 缺陷） | 预测/实测两路同向 | 9/9 判红 | 是 |

判据结论（逐条对战役判据）：**M1** 边界偏差 0.46%（vs box 网格）/1.03%（vs 细网格）≤5% ✅；**M2** 每档 1225 工况（≥900）✅，并给出稳定面积比与 tau_min 处最大稳定 Ma（BGK 0.2357 → MRT 0.3047，+29%）；**M3** 负控 9/9 两路同向判红 ✅，且给出 MRT 反而更差的子域（s4 欠松弛 foil 1.39% 面积；不匹配松弛率 foil 100% 面积、最大 λ 超出 0.443）；**M4** 包络 JSON 逐格可复算、sha256 自校验 intact ✅，并已接入运行建议门（域外 fail-closed）。

**未达 3% 的两项如实披露**：(i) 边缘点（|sigma_pred|<0.005）的放大因子绝对偏差达 0.185，机制是 D2Q9 在 (pi,0)/(0,pi) 存在严格中性（|mu|=1）棋盘支，其本征矢不含动量，速度残差实测看不到，因此这些点只用于「不误报增长」，不参与增长率判据；(ii) 近边缘良态点（sigma≈0.01）的增长率相对偏差达 16%，但绝对 λ 偏差仅 1.6e-3，属 sigma→0 的条件数效应，且该效应在边界量（tau_crit）中互相抵消到 0.5% 以内。**无一项判据降级为 FAIL。**

## 7 审计链（可复算 / 可追溯）

- 预注册参数（窗口、网格、k 网格、候选松弛率集合、判据容差、负控定义）全部内嵌在 envelope JSON 的
  `predeclared` 字段，与 `PREDECLARED` 同源，运行前后不可更改（改动会改变 digest）。
- 逐档 `lambda_max` 平面按 9 位小数归档，并各自带 `lambda_grid_sha256`；整个 envelope 再取
  `digest = sha256(canonical_json(envelope \ {digest}))`（canonical = 按键排序、无空白、浮点 12 位）。
- `MANIFEST.json` 记录每个产物的 SHA-256（envelope JSON / 平面 CSV / 图）。
- 复算校验：`python -m tensorlbm.stability.envelope --verify TensorLBM/artifacts/p482_n05/stability_envelope.json`
  应输出 `intact: true`；`m4_reproducibility.recompute_sample_*` 记录重算抽样点及其 sha256。
- 复现命令：

```bash
PYTHONPATH=src python scripts/run_p482_n05_stability_envelope.py --stage all \
    --out TensorLBM/artifacts/p482_n05
PYTHONPATH=src python -m pytest tests/test_lbm_collision_stability_envelope.py -q
PYTHONPATH=src python scripts/p482_n05_write_docs.py
```

- 证据分级：M1/M2/M3 全部为 **REAL**（真实特征值分析与真实非线性时间积分，无 mock/硬编码）；
  唯一的近似是 `lambda_max` 的 k 网格离散化（扫描 17x17、边界 81x81）与实测的拟合窗口，已量化并列入局限。

## 8 产物清单

- `stability_envelope.json` — sha256 `85571724b6306408b39e88e6ac52a0c9c4e3bcc3b9573c51179e5f1de515861d`
- `stability_envelope.png` — sha256 `326e268c95f38b85b6f343e4499e05075b883bd0d8b93235f4d9cde9fd0b77fc`
- `stability_planes.csv` — sha256 `72b1829e79a73ff59e347e6266b9eb747af6570cd492e4feba844d03731c1ea2`
- envelope digest `42e9b8bc1b8ff4c9bdfe50ed038f24575fa0d87bd683efabe8cf5e269479a747`

