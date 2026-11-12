# T1-03 · 既有锚点与在跑单点收编（T1 伞下映射，不新立 slug）

**锚点** GAP-LBM-T1-MULTIPHASE（伞）· 第 1 轮
**目的** 把仓库里既有的多相锚点、基准记录与在跑单点**逐条挂到 T1 的五条子判据**上，
避免"另立重复锚点 + 白做"。**本文件不新建任何 slug**，也不改写历史 slug。

---

## 0. 收编规则

1. **历史 slug 不改名、不删除**（总账历史不可改）。T1 只在文档层做映射：`slug → T1-Cx`。
2. **一个能力只允许一个主承载**：某 slug 若已承载某子判据，T1 不再开同义新单，
   只在其上追加轮次（round）。
3. **不属多相的锚点一律不并入**（单相边界/单相绕流/单流体自由液面），防止 T1 变成万金油。
4. 新出现的多相需求先在本文件登记 → 若能映射到 C1–C5 就挂靠；不能且确属新能力，
   才走新锚提案（附本文件的"不重复"检索记录）。

---

## 1. 映射总表

| # | 既有 slug / 记录 | 现状（证据级） | T1 伞下位置 | 本轮动作 |
|---|---|---|---|---|
| 1 | **GAP-LBM-MCMP-POISEUILLE**（2D 分层 Poiseuille 解析对拍） | R3 已闭 ①界面位置 ②流量 ④分辨率趋势 = PASS；③**界面伪流 = honest FAIL**；R4 完成共存标定 + 方法评审，③ 仍 FAIL（`docs/MCMP_STRATIFIED_POISEUILLE_R4.md`） | **T1-C2 的平界面支路**（$S$ 在平直界面本就应≈0，其非零恰恰证明来源不是曲率而是格式） | 不再新开 Poiseuille 单；其 ③ 的判据口径由 T1-C2 统一（用 $S$ 而非绝对 $S_{ifc}$ 门） |
| 2 | **GAP-PP-1-3DMCMP-STABILITY** | 已修**实测缺陷**：D3Q19 Guo 分支过量注入 $(2-\frac{1}{2\tau})$，$u/(gt)$ 1.50002→1.00001，有回归测试；$\Delta\rho\ge100$ = honest FAIL（`docs/mcmp3d_stability_findings.md`） | **T1-C3（稳定性）** + T1-01 §1 的"力注入离散不平衡"轴 | 保留其缺陷修复结论；后续 T1 实现单**必须**在 Guo/EDM 路径上复跑该测试，防止回归 |
| 3 | **GAP-PP-2-3DMCMP-LAPLACE** | 轮 2 待集群作业（cj1075/cj1077）回；当前无终判（`docs/GAP_PP_2_3DMCMP_LAPLACE.md`） | **T1-C1 的主承载** | 并入 T1 伞，**不新开 Laplace 单**；作业回后直接用 T1-C1 的门脚本裁决 |
| 4 | **GAP-PP-3-3DMCMP-SPURIOUS-SCALING** | 仪器（协议+runner+拟合+证伪门）完备并有负测试；生产 $u_{max}$ 矩阵未完成 = honest FAIL（`docs/3D_MCMP_SPURIOUS_SCALING_STATUS.md`） | **T1-C2 的标度支路**（给"分辨率要多高才够"的定量外推） | 保留；其 `wb_psi2` 实测已并入 T1-01 §7 的 $S$ 记账（$S\approx7.0$，附录 A7） |
| 5 | **GAP-PP-4-opt-loop**（τ/Re 反演） | 受门误差 0.4988%（<3%），有反证控制臂（`docs/OPT_LOOP_TAU_RECOVERY.md`） | **T1-C4 水动力层（第二层）的测量仪器**：从衰减率反解 ν，用来实测 $\lambda_{meas}$ | 不改动；在 T1-C4 协议里引用为仪器 |
| 6 | 基准记录 **`hadamard_rybczynski_droplet`**（W9-B 终审 FAIL） | 五模块逐个排除，结论仍是"库内无能跑 HR 的模块"（`benchmarks/pending/hadamard_rybczynski_droplet/terminal_verdict_m155.md`）；其中 G1（G_12 符号文档）已在后续 docstring 修复（源码确认现为 `G_12 < 0 → 相分离`） | **T1-C5 的父记录（基线）** | **只追加轮次，不覆盖**。−43.46% 的 cAC 实跑（`gates_verified.json`）作为 C5 的基线读数保留 |
| 7 | 基准 **`hadamard_rybczynski_droplet_revival`**（IN26-L2） | REAL：锐界面 Q1 轴对称 Stokes 解 + 三容器外推，$F_\infty$ 误差 $\le0.80\%$；容器一阶系数 $c_1\approx1.8$–3.1（`docs/benchmarks/hadamard_rybczynski_droplet_revival_2026-11.md`） | **T1-C5 的参照量发生器**（第二基准 + 盒子收敛系数来源） | 不改动；T1-C5 协议直接调用 |
| 8 | `GAP-LBM-STAIRCASE-WALL-OFFSET`（阶梯壁偏移） | 边界几何修正，与界面物理无关（`docs/STAIRCASE_WALL_OFFSET_20261111.md`） | **不并入 T1**（单相边界） | 仅在"带壁 HR 算例"处标注为依赖 |
| 9 | `GAP-LBM-SPHERE`（球绕流阻力 BFL） | 单相流固耦合 | **不并入 T1** | 不动 |
| 10 | 湿相接触角线（`docs/WETTING_CONTACT_ANGLE_BENCHMARK.md`、`benchmarks/in32_l01_wetting_ladder`） | 多相界面**边界条件**能力，判据是接触角而非五条液滴判据 | **T1 邻接、本轮不扩张**（登记为 T1-Cx 候选，不进本轮判据集） | 本轮**只登记**：接触角判据需独立定义，避免五判据膨胀为七判据 |
| 11 | free-surface 族（`dam_break*`、`hull_free_surface`、`sloshing_tank`、`multiphase_water_entry`） | 单流体 + 体积/flag，无第二相黏度 | **不并入 T1** | 不动（审计 §2 已明确区分） |
| 12 | `benchmarks/eos_maxwell`（SC 共存窗口与伪流） | 真算例 + 判据 D1–D4 + `CRITERION.md` | **T1-C1/C2 在 SC 族上的证据源**；其 `wb_psi2` 行是附录 A7 唯一同时给出 $u_{max}$ 与 σ 的记录 | 引用，不改动 |

---

## 2. 在跑单点（in-flight）点归

| 单点 | 状态 | 归并后不再做什么 |
|---|---|---|
| GAP-PP-2-3DMCMP-LAPLACE 轮 2（集群 cj1075/cj1077） | 等待集群回 | 不再单独开 3D Laplace 新单（T1-C1 唯一口径） |
| GAP-PP-3-3DMCMP-SPURIOUS-SCALING 生产扫描 | 缺机时（不是缺方法） | 不再重写协议；直接按 README 递集群，结果喂 T1-C2 的分辨率外推 |
| MCMP-POISEUILLE ③ 界面伪流 | honest FAIL 挂账 | 不再用绝对 $S_{ifc}$ 门；改用 $S=u_{sp}\mu/\sigma$（口径变更需在本文件留痕） |

---

## 3. 反重复检索记录（提出新锚前必须填）

本单已做的检索（2026-11-12，基线 `a9870455`）：

| 检索 | 方法 | 结果 |
|---|---|---|
| 是否有 HR / 液滴终端速度锚 | 仓内 `grep -r "Hadamard\|Rybczynski"` | 命中 13 个文件，主记录 = 第 6/7 行（已有父记录 + 参照发生器）⇒ **不新立** |
| 是否有 3D Laplace 锚 | `grep -r "GAP-PP-2"` | 已有 GAP-PP-2-3DMCMP-LAPLACE ⇒ **不新立** |
| 是否有伪流锚 | `grep -r "SPURIOUS"` | 已有 GAP-PP-3 ⇒ **不新立** |
| 是否已有守恒相场模型 | `ls src/tensorlbm/cac_lbm.py` + `tests/test_cac*` | **已存在**（263 行，D3Q19 双 LBE，14 项测试通过）⇒ 裁决不能提"从零建相场" |
| 是否已有黏度比解耦 | `src/tensorlbm/two_phase_viscosity.py`（HR 复活报告 §3.5） | **已存在**（机器精度）⇒ T1-C4 代数层直接复用 |

**结论**：本轮**新增 0 个 slug**；T1 只是把 11 条既有资产按五条子判据重组，并给出裁决与路线图
（[`04_VERDICT_AND_ROADMAP.md`](04_VERDICT_AND_ROADMAP.md)）。