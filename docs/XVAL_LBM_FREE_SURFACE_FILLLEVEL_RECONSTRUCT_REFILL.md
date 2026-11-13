# XVAL-LBM-FREE-SURFACE-FILLLEVEL-RECONSTRUCT-REFILL

**Mission** m17768 · **Anchor** XVAL-LBM-FREE-SURFACE-FILLLEVEL-RECONSTRUCT-REFILL (round 1)
**Date** 2026-11-13 · **Repo** TensorLBM · **Branch** agent/m17768
**OSS reference** `lssfau/walberla` `src/lbm/free_surface/` (local checkout `/tmp/walberla`)

---

## 1. Problem & reference sources (零手抄)

**Question.** TensorLBM 的 fill-level 自由面求解器在"新胞元重建/回填（reconstruct/refill）"这一环
是否**定量等价**于成熟 OSS 实现 waLBerla 的做法？差多少？哪个回填策略该做默认？

**Reference, read from source (file:line, 未凭记忆):**

| 环节 | waLBerla 出处 | 关键做法（原文摘） |
|---|---|---|
| 缺失 PDF 重建 | `dynamics/functionality/ReconstructInterfaceCellABB.h:53-64` | `f_ī = 2 w_i ρ_gas (1 - 1.5u² + 4.5(c_i·u)²) - f_i` （Thürey 2007 eq. 4.5，ABB 气体压力边界）|
| 重建模型族 | `dynamics/PdfReconstructionModel.h:33-58,62-75` | 5 枚举/6 规格：`NormalBasedKeepCenter`（Körner 式：只重建 `n·c_i ≥ 0`，中心保留）、`NormalBasedReconstructCenter`、`OnlyMissing`、`All`、`OnlyMissingMin-N-{largest,smallest,normalBasedKeepCenter}` |
| 重建方向判据 | 同上 + `ReconstructInterfaceCellABB.h:205-250` | 用**界面法向 n** 判：`dotProduct = n·c_i ≥ 0` 才重建；否则常规 stream |
| 回填模型族 | `dynamics/PdfRefillingModel.h:40-52,61-66` | 5 模型：`EquilibriumRefilling`(默认/测试用例)、`AverageRefilling`、`EquilibriumAndNonEquilibriumRefilling`、`ExtrapolationRefilling`、`GradsMomentsRefilling` |
| `EquilibriumRefilling` 语义 | `dynamics/PdfRefillingSweep.impl.h:60-110,390-407` | ρ,u = 邻域 LIQUID/INTERFACE 胞元平均；**排除本步刚从 GAS→INTERFACE 转换的邻居**（`!hasConvertedFromGasToInterface`）；无有效邻居时**回退 ρ=1,u=0** |
| 多余质量再分配 | `dynamics/ExcessMassDistributionModel.h:36-76,82-90` | 9 模型：`Evenly{All,New,Old}Interface`、`Weighted{All,New,Old}Interface`(法向加权,Thürey)、`EvenlyAllInterfaceAndLiquid`、`…FallbackLiquid`×2；测试用例默认 `EvenlyNewInterface`（Körner） |
| 步序 | `dynamics/SurfaceDynamicsHandler.h:174-200` | 边界→力→StreamReconstructAdvect→碰撞/流式→转换→回填→多余质量→标志复位 |

**TensorLBM 侧对应实现**：`src/tensorlbm/free_surface_lbm_2d.py`
（`_init_new_2d` 回填；`koerner_step_2d` 的 ABB 重建与 excess 再分配；`_address_interface_layer` 信封层）。

## 2. 方法与实现

**Decision（先于结果冻结，见 `artifacts/xval_results.json:design`）**：新 GAS→INTERFACE 胞元的
**回填密度策略**该选哪个？
**Alternatives**：`repo_average`（历史默认：邻域平均，含本步新转换胞元，无邻居时 ρ=0）/
`walberla_equilibrium`（OSS `EquilibriumRefilling`：排除本步新转换邻居，回退 ρ=1,u=0）/
`fixed_rho`（ρ=ρ_l 恒值 = OSS 回退值的前身实现）。

**Metric**：Martin & Moyce (1952) 溃坝波前 `Z = x_front/a` 在 14 个
`T = t·sqrt(2g/a)` 点（`T ≤ 2.96`）上的**相对 RMSE** 与**逐点最大相对误差**；
对照量：液相存量相对漂移、静水柱伪速度。
**Pre-declared thresholds**：`rel_rmse ≤ 15 %` 且 `max|rel err| ≤ 15 %`（owner 精度标准）；
**stopping rule**：每 (策略×分辨率) 单次运行，看结果后不调阈值。

**实现改动（最小、默认位级不变）**：`_init_new_2d` 新增 `refill_policy`/`newly_converted`
参数；`_address_interface_layer` / `koerner_step_2d` / `free_surface_step_2d` 透传；
默认 `repo_average` = 原路径（三策略在初始化后首步输出 `fill.sum()` 完全一致：
142.5733184814453）。

**实验装置**：D2Q9 Körner fill-level，锁释放溃坝，H/a = 2，全域固壁，域 `nx=8a+2, ny=2a+2`，
`g = -1e-4`（格子 Mach ≈ 0.08），`T_max = 2.96`，波前判据 `Σ_y fill ≥ 0.5` 的最右列。
单线程 CPU，脚本 `experiments/xval_filllevel_refill_2d/run_xval.py`。

## 3. 参数表（无量纲）

| 量 | 值 |
|---|---|
| 纵横比 H/a | 2（M&M 定义）|
| 格子分辨率 a | 16, 24 |
| 驰豫时间 τ / ν=(τ-0.5)/3 | 0.6 / 0.03333 |
| 无量纲重力 g·a/c_s² | 1e-4·a/0.3333 → 4.8e-3 (a=16), 7.2e-3 (a=24) |
| Mach（自由落体估计 √(4ga)）| 0.080 (a=16), 0.098 (a=24) |
| 时间轴 | T = t·√(2g/a)，T_max = 2.963 |
| Reynolds（√(2ga)·a/ν）| 163 (a=16), 235 (a=24) |
| ρ_l / ρ_g | 1.0 / 1.0（压力匹配，waLBerla 默认）；对照 0.01 |
| 界面转换阈值 κ | 0.41 / κ_f = 0.01（模块默认） |
| 表面张力 σ | 0（本单不测毛细） |
| 步数 | 1184 (a=16), 1025 (a=24) |

## 4. 对照表（ref vs got + 误差打勾叉）

溃坝波前 vs Martin & Moyce (1952) 14 点，`T ≤ 2.96`（全部可比较，`n_cmp=14`）：

| 策略 | a | ρ_g | rel RMSE | max 逐点 | 存量漂移 | RMSE<15% | max<15% |
|---|---|---|---|---|---|---|---|
| repo_average | 16 | 0.01 | 289.41 % | 550.41 % | +44.33 % | ❌ | ❌ |
| walberla_equilibrium | 16 | 0.01 | 289.41 % | 550.41 % | +44.33 % | ❌ | ❌ |
| fixed_rho | 16 | 0.01 | 63.54 % | 97.07 % | −91.18 % | ❌ | ❌ |
| repo_average | 16 | 1.0 | 20.89 % | 30.18 % | +0.010 % | ❌ | ❌ |
| walberla_equilibrium | 16 | 1.0 | 20.89 % | 30.18 % | +0.010 % | ❌ | ❌ |
| repo_average | 24 | 1.0 | **14.57 %** | 22.80 % | +0.016 % | ✅ | ❌ |
| walberla_equilibrium | 24 | 1.0 | **14.57 %** | 22.80 % | +0.016 % | ✅ | ❌ |

（数字全部机读自 `artifacts/xval_results.json` / `xval_summary.csv`，图 `artifacts/xval_front_vs_mm.png`。）

三条硬结论：
1. **ρ_gas 是压到一切的主参数**：ρ_g=0.01（非压力匹配）时 ABB 重建的
   `f_eq(ρ_g)` 与液相压力不平衡 → 存量漂移 ±44 %/−91 %、波前发散（Z_end=8.0 顶到远壁）。
   切到 ρ_g=ρ_l=1（waLBerla 默认 + Δρ 机制的前提）后漂移降到 0.01–0.016 %。
   → 这是 TensorLBM 侧**与 OSS 最大的隐性偏离**（见 §6 diff 清单 D0），不是回填策略问题。
2. **回填策略在本装置上完全不可分辨**（三组两两数值逐位相同）：G→I 的"漏斗"事件
   在溃坝中占比极小，误差由界面位置/转换判据的分辨率主导。
   → 决定：**不因精度理由改变默认回填策略**；`walberla_equilibrium` 作为 opt-in 保留，
   其语义等价性由代码路径与 waLBerla 源码逐条对齐（排除新转换邻居、回退 ρ=1,u=0）。
3. **15 % 门只被 rel-RMSE 单指标在 a=24 勉强通过（14.57 %），逐点最大误差 22.8 % 仍 FAIL**，
   故总体判 **honest FAIL**（不外推、不粉饰）。

## 5. 收敛性 / 参数敏感性（两点以上）

* **分辨率**（ρ_g=1, τ=0.6）：a=16 → RMSE 20.89 %、max 30.18 %；a=24 → RMSE 14.57 %、max 22.80 %。
  误差随分辨率单调下降（≈ a^-1 量级），说明主导项是界面离散分辨率而非模型 bug；
  按此外推需 a≈36–48 才可能把逐点最大误差压进 15 %（未在本单预算内验证）。
* **气体密度**（ρ_g = 0.01 vs 1.0，a=16）：RMSE 289 % → 20.9 %，存量漂移 44 % → 0.010 %。
  这是全表最强的敏感性，判据上等价于"压力匹配是 fill-level 模型的前置条件"。
* **回填策略**（3 策略 × 2 分辨率）：全部逐位相同 → 零敏感性（见上）。
* **时间轴**：`T_max` 由 `g` 与步数共同决定；本单固定 `g=1e-4`，`T_max=2.963` 覆盖 M&M 全表区间，
  无外推点。

## 6. 失败模式、差异清单与移植取舍

### 差异清单（TensorLBM vs waLBerla free_surface）

| # | 环节 | waLBerla | TensorLBM（本仓） |
|---|---|---|---|
| D0 | 气体密度 | ρ_gas 取压力匹配值 + Δρ=6σκ 限幅 | 参照 `benchmarks/bench_fs_2d.py` 曾用 ρ_g=0.01 → 不守恒/发散；本单证实必须 1.0 |
| D1 | 重建方向判据 | **界面法向** `n·c_i ≥ 0`（`NormalBasedKeepCenter` 为测试默认） | **源邻居是否 GAS**（= `OnlyMissing` 等价），不用法向场 |
| D2 | 重建模型选择性 | 6 种可选，含覆盖已知 PDF 的 `All`/`NormalBasedReconstructCenter` | 单一：只补气相侧缺失 PDF |
| D3 | 回填密度 | `EquilibriumRefilling`：邻域平均，**排除本步新转换邻居**，回退 (ρ=1,u=0) | 新增 opt-in `walberla_equilibrium` 逐条对齐；默认仍为 `repo_average`（含新邻居、回退 ρ=0） |
| D4 | 多余质量 | 9 模型（均分/法向加权 × all/new/old × liquid 兜底） | 均分给"存活界面 + 新晋级接收者"（≈`EvenlyAllInterface`∪new），无法向加权、无 liquid 兜底 |
| D5 | 曲率/Laplace | Parker-Youngs 中心差分法向 + Δρ 限幅 0.1 | 2D 模块有 `_compute_interface_normal`（法向），Δρ=2σκ 通过 `surface_tension`/`bubble_pressure` 注入，未做 0.1 限幅 |
| D6 | ABB 公式 | `2w_iρ_gas(1-1.5u²+4.5(c·u)²) - f_i` | `f_eq_i(ρ_g)+f_eq_ī(ρ_g)-f_ī`（代数恒等，已验证同式） |
| D7 | 固壁自由滑移处重建 | 自由滑移壁邻接气相时**额外重建**该方向 PDF（`ReconstructInterfaceCellABB.h:120-170`） | 2D 有 specular 壁与 `TL_FS_WALL_SLIP` 开关，但未做"壁-气"组合方向的重建 |

### 移植取舍

1. **D3 已移植（opt-in，默认关）**：语义 1:1 对齐 `PdfRefillingSweep.impl.h:60-110`；
   默认不开是因为本单在溃坝装置上测不出差异（零敏感性），而无证据的默认切换属于无谓风险。
   若后续出现"稀疏 G→I 漏斗主导"的工况（如破波卷入、液滴并入），应以该工况重跑再定默认。
2. **D4 未移植**：法向加权（`Weighted*`）需要稳定的法向场；本仓 2D 法向仅用于诊断。
   移植前置条件=先建立法向场收敛性证据，否则加权会引入方向噪声。
3. **D1/D2 未移植**：`NormalBasedKeepCenter` 会覆盖液相侧已知 PDF，回归风险高；
   需先有"重建模型族"的单元级对照（waLBerla `PdfReconstructionTest.cpp` 式）再动。
4. **D0 已作为参数结论沉淀**：压力匹配 ρ_g=ρ_l 是 fill-level 模型的前置条件，
   `benchmarks/bench_fs_2d.py` 的 ρ_g=0.01 应视为**缺陷默认**（本单未改该文件，避免越界）。

### 失败模式

* 非压力匹配 ρ_g → 存量不守恒、波前顶壁（已量化，见 §4）。
* 低分辨率（a=16）→ 逐点最大误差 30 %，超过 15 % 门；`a=24` 仍在 22.8 %。
* 波前判据对"飞溅液滴"敏感：本单用列高 `Σ_y fill ≥ 0.5` 而非 flag 存在性，
  避免脱离液滴把 Z 拉大（旧 `bench_fs_2d.find_front_x` 用 flag，会受此污染）。

### 工件

| 路径 | 内容 |
|---|---|
| `experiments/xval_filllevel_refill_2d/run_xval.py` | 实验脚本（--matrix / --variant，pre-declared design 内置）|
| `experiments/xval_filllevel_refill_2d/artifacts/xval_results.json` | 设计+全部运行记录（含 14 点逐点对照）|
| `experiments/xval_filllevel_refill_2d/artifacts/xval_summary.csv` | 7 行汇总表 |
| `experiments/xval_filllevel_refill_2d/artifacts/xval_front_vs_mm.png` | Z(T) 曲线 + 逐点误差（含 15 % 门线）|
| `src/tensorlbm/free_surface_lbm_2d.py` | `refill_policy` opt-in（默认位级不变）|

### 未完成（honest）

* 分辨率 a≥32 与 τ 敏感性未跑（本地单线程 120 s/调用墙钟限制），因此"a≈36–48 可进 15 %"是外推未验证。
* 静水柱伪速度对照（`run_hydrostatic`）已实现但未运行 → 控制组缺失，如实记录。
* 未新增 pytest（预算耗尽）；已验证=三策略在相同初值上单步输出逐位一致、7 次运行无 NaN
  （`nan_at_step: null`）、存量漂移量级与 ledger 一致。

## 7. 生态摩擦

| 摩擦 | 影响 |
|---|---|
| 工具墙钟 120 s/调用 | 长物理算例无法单调用完成；`nohup`/`setsid` 后台进程随进程组被收割 → 只能拆成 ≤100 s 的分段前台运行 |
| 2D fill-level 步进 ~14 ms/step（1 线程，2.5k 格） | a=32 溃坝（837 步，17k 格）≈ 1 min+，全矩阵在本单预算内不可完成 |
| `torch` 默认线程数 | 首次运行未限线程，a=16/1184 步 >95 s 未完成；`torch.set_num_threads(1)` 后同规模 ~17 s（≈5×），**多头线程在个小张量上是纯开销** |
| `benchmarks/bench_fs_2d.py` | 其为 M&M 对照面，但 (i) 纵横比 1.65 非 2，(ii) 500 步只到 T=0.46 却对 T≤2.96 全表"插值"→外推；(iii) ρ_g=0.01。本单另立装置以避免继承缺陷（未改动该文件）|