# 交差审计 — m15544 `GAP-PP-3-3DMCMP-SPURIOUS-SCALING`

| 项 | 值 |
|---|---|
| 审计对象 | mission 15544（返修轮1，原单 m15363；status=succeeded） |
| 交付分支 / commit | `agent/m15544` / `30e2f7d4e96b16cc30d559623740355d199108b2`（m15544: GAP-PP-3-3DMCMP-SPURIOUS-SCALING 返修轮交付） |
| 被审工作树 | `/root/tp-agent/work/m15544`（只读取证 + 只读重跑） |
| 审计工作树 | `/root/tp-agent/work/m15650`，证据 `artifacts/audit_m15650/`（p1/p2/p3.json = 审计自跑探针数据，fig1_m15544_audit.png） |
| 审计方式 | 工件逐项核验 + 引擎路径**逐位复现** + DB 基准裁判 + 集群作业台账 + 物理有效性探针（非重做任务） |
| 判级结论 | **L2**（工具链成立），**锚点科学交付未达成**（该锚点机器门自判 FAIL） |
| verdict 落表 | `tensor_maturity_verdicts` (repo=TensorLBM, scope=`m15544 GAP-PP-3-3DMCMP-SPURIOUS-SCALING`, mission_id=15650, level=2, verdict_by=board-audit) |

## 1. 三件套核验（要求「真存在」）

| 项 | 结论 | 指针 |
|---|---|---|
| impl_files（非占位实现） | **存在**（923 行真实数值实现：D3Q19 权重/梯度、SC 力、EDM vs velocity-shift 两套 forcing、Maxwell 等面积构造、静态液滴推进） | `src/tensorlbm/pseudopotential_edm3d.py:1-923`；`static_droplet` `:642`；`step_sc3d` `:454`；`maxwell_coexistence` `:504`；`droplet_init3d` `:575`；`sc_interaction_force_3d` `:351` |
| entry（入口） | **存在**（可执行 CLI + 判据机） | `benchmarks/mcmp3d_spurious_scaling/run_incremental.py:152 main()`（`--out/--threads/--steps/--resume`）；`run.py:90` 调引擎；`fit_report.py:272-276` 计算 C1/C2/C3 与 Z/P/M 门，`all_pass` 为合取 |
| run_artifacts（真实运行工件） | **存在但残缺**：2 个算例（协议矩阵 30 算例），两例 `converged=false / finite=false / dp=NaN` | `artifacts/mcmp3d_spurious_scaling/results.json`（`meta.n_cases=2`, `machine=loongarch64`, `git_commit=da452346…`, `command=/usr/local/bin/python benchmarks/mcmp3d_spurious_scaling/run_incremental.py`）；`verdict.json`（`all_pass:false`, `unconverged_cases:[main_n32_G-4.4_kupershtokh, main_n32_G-4.4_velocity_shift]`）；`tables.md`；`REPORT.md`；`figures/`（4 张） |
| benchmark（ref vs got 数值） | **缺失（L4 断点）** | `tensor_benchmark_cases` 中该锚点 **0 行**（psql 核验；全表仅 `GAP-LBM-MCMP-POISEUILLE` 一行，id=7）；平台 `GET /api/v1/tensor/bench/l4-verdict/15544` → `CLAIMED_NOT_PROVEN, rows=0, failed_checks=[execution_success, source_cluster_job, cluster_job_bound, ref_got_error_present, within_tolerance]` |
| repro_cmd（可执行） | **存在且已实证可复现** | 文档命令 `python benchmarks/mcmp3d_spurious_scaling/run_incremental.py`；审计用引擎入口独立重跑该算例，得到 `u_max=9.642752295209812e-3`，与工件 `results.json` **逐位相同**（相对差 0.0）；40.54 s 内跑完 50 项测试 |

## 2. 基准判据（唯一裁判）逐项核

判据来自本锚点预注册协议 `benchmarks/mcmp3d_spurious_scaling/README.md:12-24`（C1/C2/C3）与交付 artifacts 自带机器裁决 `verdict.json`：

| 判据 | 阈值 | 实测 | 裁判 |
|---|---|---|---|
| C1 表覆盖 | ≥3 分辨率 × ≥3 密度比 | grids=1, Δρ levels=1（`verdict.json: C1_table_coverage`） | **FAIL** |
| C2 幂律指数不确定度 | σ_p/\|p\| < 20% | p 未定（无拟合点），rel_unc=NaN（两种 forcing 均 NaN） | **FAIL** |
| C3 力格式交叉对照 | ≥2 密度比，两指数 2σ 内一致 | `levels_per_forcing{kuperstokh:1, velocity_shift:1}` → 不足 2 档 | **FAIL** |
| 控制 Z（Δρ→0 截距） | u_max 塌缩 ≥1 个量级 | NaN | **FAIL** |
| 控制 P（平台） | 后 3 检 ≤5% 且全部收敛 | spread=0 但 `all_converged=false` | **FAIL** |
| 控制 M（质量） | \|ΔM\|/M<1e-9 | 9.484e-15 | PASS |
| 总账 | — | `all_pass:false` | **该锚点科学交付 = FAIL（作者自判，诚实未粉饰）** |

> 关键点：mission 15544 在池内 `status=succeeded`，但它 succeeded 的是**交付物存在性**（返修轮把 L2 三件套补齐了），**不是**锚点判据。审计不改写这一点：succeeded ≠ 基准通过。

## 3. 独立复现与物理有效性探针（审计自跑，数据在 `artifacts/audit_m15650/p{1,2,3}.json`）

复现（p1.json，`n=32, R=4, W=2, G_eff=−4.4, kupershtokh`）：

| step | u_max | ρ_max | ρ_min | R_eq | dp | 说明 |
|---|---|---|---|---|---|---|
| 0（初值） | 0 | 1.3441 | 0.2838 | 4.058 | — | tanh 液滴存在 |
| 50 | 7.16e-2 | 0.8907 | 0.2822 | 3.368 | −6.97e-3 | 液核已塌 |
| 100 | 1.49e-2 | 0.3156 | 0.2535 | 18.65 | NaN | 只剩均匀场 |
| 200 | **9.6428e-3**（= 工件值） | 0.3045 | 0.2867 | 18.36 | NaN | 工件里被当作「稳态伪电流」 |

结论 1（测量对象错位）：报告为 spurious current 的数，取自在 **ρ 全场 0.2867–0.3045 的均匀场**上（实测 Δρ=0.0178 vs 协议横坐标 Δρ=1.106，**低 62 倍**），`dp=NaN`、`R_eq=18.36≈0.57n`（"内部"占半个盒子）。液滴在 100 步内蒸发，u_max 是该瞬态的衰减尾（0.0716→0.0149→0.0096），协议自己的平台控制 P 也因此 FAIL。

结论 2（根因，可证伪的定量）：这是**协议参数选择**问题，不是引擎崩溃。
- 同 G 下把液滴放大：`n=48, R=6, W=2`（p2.json）→ 液核保住（ρ_max 1.384→1.270 @100 步）但仍在溶解，dp 由 −4.85e-3 到 −8.59e-3 持续为负 → 仍未达力学平衡；
- 采用超临界半径 + 更窄界面：`n=32, R=8 (=n/4), W=1.5, G_eff=−4.6`（p3.json）→ ρ_max 1.587→1.688（≥ρ_l=1.587），**dp 由负转正 +2.90e-3 @200 步**（σ≈9.6e-3 格子单位），液滴稳定 → 同一工装能测到物理的伪电流。
- 机理：`G_eff=−4.4` 附近 EOS 极平（dp/dρ = c_s²(1+Gψψ′) ≈ 0.059），Laplace 压 2σ/R≈0.01–0.05 需要 Δρ≈0.2–0.8；界面宽 W=2 且 R=n/8=4 时液滴半径与界面厚度同量级，亚临界 → 蒸发/溶解。**协议 `R=n/8`、`W=2` 三个分辨率（4/6/8）全部落在亚临界区**，故全矩阵即便跑完也测不到 u_max(Δρ)。

结论 3（生产矩阵从未执行）：commit message 自述「完整生产矩阵需集群计算（cj1156 排队中），当前工件为部分数据」；审计时 `cj1156` 仍 `queued`（age 42.3 min）→ 3×5×2=30 例从未落数。

结论 4（测试断言的是代数/机器门不变量，不是物理）：50 项测试全绿（`tests/test_mcmp3d_spurious_scaling.py` 12 项 + `tests/test_pseudopotential_edm3d.py` 38 项，审计实跑 40.54 s），覆盖权重矩、梯度精确性、动量注入、拟合器恢复已知指数（1/2/3）、伪造/瞬态/漏质检测；**不构成 u_max(Δρ) 标度律的物理验证**，不能支撑 L4。

## 4. 判级（L0–L5）

| 级 | 判定 | 依据 |
|---|---|---|
| L0 文档 | 过 | `docs/M15544_3D_MCMP_SPURIOUS_SCALING_FINAL.md`、`artifacts/.../REPORT.md` |
| L1 数学/单元验证 | 过 | 50 项测试；Maxwell 等面积残差 5e-14–1e-7（6 档 G 全 converged） |
| **L2 独立功能实现** | **过（判级=L2）** | 真代码 923 行 + 入口 + 真实运行工件（loongarch64, float64, git_commit 绑定），审计逐位复现 |
| L3 系统集成 | **不过** | 引擎只被 benchmark 脚本 import（`run.py:90`、`run_incremental.py:72`、`fit_report.py` 外围），**无主求解器/平台侧调用点**，无 CI/平台注册（`tensor_benchmark_cases` 0 行） |
| L4 基准验证 | **不过** | 无 `tensor_benchmark_cases` 行 → 无 ref/got/误差；平台 l4-verdict = CLAIMED_NOT_PROVEN；权威判据 C1/C2/C3 全 FAIL；覆盖 2/30 例且两例不收敛 |
| L5 科研工程成果 | 不过 | 标的量（幂律指数 p 及不确定度、力格式对照）未产出（p=NaN） |

百页报告不改变级别：级别只看功能实质。本单不再重做交付，只出裁决 + 返修输入。

## 5. 返修轮2 输入（fail_stage / failure_reason 建议）

- **fail_stage**：`benchmark/physics-config`（工具链 OK，配置使测量对象消失）。
- **failure_reason**：协议半径 `R=n/8`（4/6/8）在 `W=2` 下全部亚临界，液滴 100 步内蒸发/溶解 → `dp=NaN`、Δρ 实测低 62 倍 → C1/C2/C3 无数据可拟合；生产矩阵作业 `cj1156` 从未出队。
- **已试路径**（审计已证）：`R=n/4, W=1.5, G=−4.6` 可得到 ρ_l 以上液核且 dp>0（+2.90e-3 @200 步）。
- **下一轮最小可证单元**：(a) 用 `(R=n/4, W≤1.5)` 重扫 3 分辨率 × ≥3 Δρ × 2 forcing，先在每例上过「ρ_core ≥ ρ_l − 2%」「dp>0」两道前置闸（不满足即记为配置失败，不许进拟合）；(b) 收敛判据改用 u_max 与 dp 双变量平台（现仅 dp）；(c) 把 ref/got 落进 `tensor_benchmark_cases`（ref=文献/收敛外推 p，tolerance=σ_p/|p|<20%）；(d) 若 3 分辨率下 p 的网格趋势不收敛，如实报 FAIL 并写尺度窗口结论。

## 6. 2026-11-12 审计的局限与未跑项

- 未跑完整 30 例矩阵（单例 n=32 200 步约 20 s；n=64 会超单命令时限），故「全矩阵皆测不到」是由 (i) 亚临界判据的定量机理 + (ii) 三个代表性探针 推出的**强证据推断**，非穷举实测。
- 未做求解器无关的第三方实现交叉（如另一套 SC 引擎）验证 σ 与 dp/dρ 的数值；`dp` 使用模块自带的 `pressure_mech`。
- 图 `fig1_m15544_audit.png`：A 液核密度衰减（三配置）、B 工件 u_max 的瞬态本质、C 实测 Δρ vs 协议 Δρ。
## 7. 落表与门禁预演（机器复核）

| 项 | 值 |
|---|---|
| verdict 行 | `tensor_maturity_verdicts` id=356（repo=TensorLBM, scope=`m15544 GAP-PP-3-3DMCMP-SPURIOUS-SCALING`, mission_id=15650, level=2, verdict_by=board-audit, gate_state=pending） |
| evidence 键 | impl_files / entry / run_artifacts / tests / integration_callers / benchmark / repro_cmd / external_anchor（+ impl_files_detail / run_artifacts_detail / audit_findings / anchor_criteria / level_justification / not_run） |
| 机器复核预演 | 直接调用 `/root/tp-agent/maturity_gate.py:143 judge(level=2, evidence, 'TensorLBM', 15650)` → `gate_level=2, why=[]`（impl_files/entry/run_artifacts 路径存在且 impl 非占位）→ 审计单 succeeded 后应裁 `confirmed` |
| 失败信号另存 | `negative_knowledge_ledger` id=169（entry_id=`NEG-M15650-3DMCMP-SUBCRITICAL`, failure_kind=`parameter_config_invalid`）：亚临界协议参数的证伪证据 + 反证配置（R=n/4, W=1.5, dp>0）——供返修轮2 直接站上去，避免重蹈 |
| 证据副本 | `artifacts/audit_m15650/m15544_copy/`（被审字节原样副本 + sha256 见 `PROVENANCE.md`；引擎 sha256 与生态 main 一致，证明 m15544 复用既有引擎） |
| 审计自产数据 | `artifacts/audit_m15650/p1.json`（含 n48 项在 p2.json）、`p2.json`、`p3.json`、`fig1_m15544_audit.png` |

> 注意 L2 与锚点验收是两本账：本裁决只认定「工具链达到独立功能级」，**不认定** `GAP-PP-3-3DMCMP-SPURIOUS-SCALING` 已解决——该锚点的 C1/C2/C3 在作者自己的 `verdict.json` 里是 FAIL，审计逐项复核一致，且平台 `l4-verdict` 判 `CLAIMED_NOT_PROVEN`。
