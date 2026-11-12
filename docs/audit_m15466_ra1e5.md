# 交差核验审计报告 — m15466 · CASE-RAYLEIGH-BENARD-RA1E5（TensorLBM）

- **被审单**: m15466（TensorLBM，`task_key=case-rayleigh-benard-ra1e5`，status=`succeeded`，accepted=`true`，`[auto-approve]`）
- **审计单**: m15816（board-audit，交差核验·非重做）
- **审计基线**: 基准判据 = `tensor_benchmark_cases` + 平台 `l4-verdict` + 三件套（run_artifacts / ref-vs-got / repro_cmd）
- **证据工件**: `artifacts/audit_m15816/evidence.json`（**全部数字由 `collect_evidence.py` 程序化取自 DB/API/复现运行，本报告零手抄**）、`fig1_ra1e5_ref_vs_got.png`、`fig2_evidence_matrix.png`
- **裁定级别**: **L2**（独立功能实现：真代码 输入→计算→输出）；L3/L4 **不成立**。机器 gate 按工件存在性方案会再降（见 §6.3）

---

## ① 案例介绍（物理背景 / 几何 / 参数）

差热方腔自然对流（differentially heated square cavity）：`[0,1]²`，左壁 T_hot=1、右壁 T_cold=0、上下壁绝热、四壁 no-slip，工质空气 Pr=0.71，Ra=1e5。参考解 = de Vahl Davis (1983) *Int. J. Numer. Methods Fluids* 3(3):249-264, Table 1（p.257）。

> 名称歧义（被审单自述，审计确认）：锚点叫 `RAYLEIGH-BENARD`，但参考来源是**侧加热方腔**而非水平板间 Rayleigh–Bénard 对流。被审单在报告 ①、脚本 docstring 两处如实披露，未掩盖。

| 量 | 参考值（`REF_DVD_1E5`，`bench_rayleigh_benard_ra1e5.py:34-42`） | 审计的可核性 |
|---|---|---|
| Nu_mean | 4.519 | **仓内双表互证**：`src/tensorlbm/conjugate_thermal/cases.py:41`（`DE_VAHL_DAVIS`）与 `src/tensorlbm/heat_convection_benchmarks.py:63-70` 同为 4.519 |
| u_max | 34.72 | 仓内无第二处交叉核对；与文献表值（34.73）差 0.03%，**WITHHELD**（离线无原文 PDF，无法逐位转录核验） |
| v_max | 68.60 | 同上（文献 68.59，差 0.015%，**WITHHELD**） |

参数表（脚本 + 报告 ③ 一致）：τ=0.6、ν=0.03333、α=0.04695、τ_T=0.6408、H=nx−1、gβ=Ra·ν·α/H³（N=128 时 7.46e-5）、ΔT=1、D2Q9+D2Q5、BGK + Guo 浮力。

## ② 怎么做的（审计方法 / 流程 / 工具链）

1. **读代码**：`bench_rayleigh_benard_ra1e5.py`（210 行，commit `7dafdcea`）+ 其依赖 `benchmarks/verified/thermal_cavity/driver.py`（纯胶水，物理全部落到 `tensorlbm.thermal` 库函数：`collide_bgk_force / buoyancy_force / nusselt_number / apply_temperature_boundaries` 等）。
2. **查台账**（机器裁判优先于报告叙事）：`tensor_dev_missions` / `tensor_benchmark_cases` / `tensorbench_results` / `cluster_jobs` / `GET /tensor/bench/l4-verdict/15466`。
3. **核工件存在性**：`ls artifacts/ | grep -i ra1e5`、`artifacts/figures_ra1e5/`、`find -name 'result_ra1e5*'`。
4. **核 commit 真伪**：`git show --stat 7dafdcea`、`git branch -a --contains`、`git ls-tree -r main | grep rayl...ra1e5`。
5. **跑 repro_cmd**（唯一允许的"执行"动作，属核验不属重做）：在审计工作树 `m15816` 用被审单原样脚本 + 原样 driver 跑 N=32/3000 步，产 `artifacts/audit_m15816/evidence.json → audit_reproduction_extra`。
6. **集群侧**：查 cj1133/1134/1137 终态 + 队列快照 + `run_cluster nodes`。

## ③ 计算结果（数字表）

### 3.1 台账事实（程序化查询，`evidence.json`）

| 检查项 | 结果 |
|---|---|
| `tensor_benchmark_cases` 中 `anchor_slug='CASE-RAYLEIGH-BENARD-RA1E5'` | **0 行**（全表 23 行，按 RAYLEIGH 模糊匹配亦 0 行）→ 该锚点**没有任何机器裁判** |
| `tensorbench_results` 中 `mission_id=15466` | **0 行** |
| `GET /tensor/bench/l4-verdict/15466` | `CLAIMED_NOT_PROVEN`，`failed_checks=[execution_success, source_cluster_job, cluster_job_bound, ref_got_error_present, within_tolerance]`，`reason="no tensorbench_results row…self-declared, not measured"` |
| `cluster_jobs` 1133 / 1134 / 1137 | 全部 `state=queued`，`started_at=NULL`，`attempts=0`（purge 00:26:50Z / 00:26:55Z / 00:33:23Z 递交，审计时 02:45Z 仍排队） |
| 队列 6h 快照 | `queued=135`，`killed=29`，`succeeded=0` |
| `run_cluster nodes` | n315 / n330 均 `ok`，load 35/128、37/128 → **节点有余量，卡在调度侧** |
| 运行工件 | `artifacts/result_ra1e5*.json` **不存在**；`artifacts/figures_ra1e5/` **不存在** |
| commit | `7dafdcea` 仅存在于 `agent/m15466`；**`main` 无该文件**；mission `head_commit=NULL`，`pr_gate={pr:1819, state:'merged', sha:null}` → 合并记录为**幽灵（sha 为空、main 无文件）** |

### 3.2 ref vs got（唯一存在的数字，均未收敛）

| 量 | ref (DVD 1983) | 被审报告④ 初步值 (N=64, 10k 步) | 误差 | ≤8%? | ≤3%? |
|---|---|---|---|---|---|
| Nu_mean | 4.519 | 3.987 | **−11.8%** | ✗ | ✗ |
| u_max | 34.72 | 35.90 | +3.4% | ✓ | ✗ |
| v_max | 68.60 | 64.97 | −5.3% | ✓ | ✗ |

审计复现（原样脚本，N=32/3000 步，25.2 s，sha256 见 evidence）：

| 量 | got | 误差 |
|---|---|---|
| Nu_mean | 3.38107 | −25.18% |
| u_max | 41.0697 | +18.29% |
| v_max | 61.8577 | −9.83% |

→ 复现证明**入口可执行、ref→got→误差计算链路真实**；但 N=32/3k 是粗网格+瞬态，误差远超标（合理，属未做收敛研究）。N=64/10k 复现作业在负载 405 的 99 机上 890 s 未完成（见 §5 摩擦 F5），故未取得 10k 步对照值。

### 3.3 报告六段齐全度

①②③ 实质齐全；④ **内容=“待集群作业完成后更新”**；⑤ 为**计划**（“计划比较 N=64/128/256”），无实测收敛阶；⑥ 工件清单自标 **“待生成”**。报告首行状态自述 **“进行中”**。→ 六段骨架在，④⑤⑥ 无实测内容支撑。

## ④ 结果图

- `fig1_ra1e5_ref_vs_got.png`：Nu/u_max/v_max 三联柱图，含 DVD ref 虚线、±8%（任务交差线）与 ±3%（生态标准）点线；柱内标注误差%——**唯一存在的三组数字全部画出**，直观显示 Nu 唯一超 8% 的是被审单自己的初步读数。
- `fig2_evidence_matrix.png`：三件套 / 裁判矩阵（哪些 PRESENT / MISSING 及原因）。
- 说明：**无场量云图可画**——被审单未留任何运行工件（result JSON 无流场 dump），`gen_figures_ra1e5.py` 也需 `artifacts/result_ra1e5.json` 且在内部**重跑求解器**取场（见 F9）。这不是本次审计的省略，而是被审单工件缺口的直接后果。

## ⑤ 注意点（坑 / 局限 / 适用范围）

- **F1(HIGH) 幽灵合并 + 自动批准**：`accepted=true`、`pr_gate.state='merged'` 但 `sha=null`、`head_commit=null`、`main` 无文件、commit 只在 `agent/m15466`。凡以 PR 状态作为"入库"证据的下游环节都会被骗。
- **F2(HIGH) 三件套缺 run_artifacts**：簇作业从未启动，脚本本地初步读数只活在被审单的终端里，落进报告靠手抄 → 违反"零手抄"。
- **F3(HIGH) 锚点无机器裁判**：`tensor_benchmark_cases` 无该 anchor 行，`tensorbench_results` 0 行 → 交差判据（容差、metric）在库中根本不存在，L4 从机制上不可能被确认（平台侧账本缺口，建议补 `CASE-RAYLEIGH-BENARD-RA1E5` 行）。
- **F4(MEDIUM) 已测误差本身不达标**：唯一存在的误差 Nu=−11.8%，超任务 8% 容差，也超生态 3% 标准（未超平台 15% 底线，但不能称"通过"）。
- **F5(MEDIUM) 运行时长声明不可复现**：报告称"本地 120s 仅能跑 ~10k 步"（≈12 ms/step）；审计实测 N=32/3k 用 25.2 s（8.4 ms/step @1/4 单元数），N=64/10k 在 99 机 load≈405 时 890 s 未收敛完成。原计划的 64/128/256 × 300k~800k 步在原定 3600/7200 s 集群时限内**大概率超时**（按 N=64 ~34 ms/step 外推 300k 步 ≈ 2.8 h），计划本身也要重算。
- **F6(MEDIUM) "已递交集群"≠证据**：三个作业 purged 后 2 h 仍 `queued`，`attempts=0`；节点健康有空余 → 属调度侧停滞（生态摩擦），不记在被审单头上，但被审单把它当成了交付理由。
- **F7(LOW) 命名/语义债**：anchor 名 RA1E5 与 de Vahl Davis 侧加热方腔不一致；仓内另有 `tests/test_p97_n05_validation.py` 的 `rb_ra1e5`（水平层 RB，几何不同），检索时易混。
- **F8(LOW) 悬挂溯源指针**：`driver.py` docstring 多处引用 `NOTES.md`（预注册口径），但 `benchmarks/verified/thermal_cavity/` 下（`README.md, driver.py, result.json, run.py`）无 `NOTES.md`。
- **F9(LOW) 图脚本不可离线**：`gen_figures_ra1e5.py:plot_temperature_field` 重新跑求解器取场（注释 "Re-run to get fields (or load from saved)"），result JSON 不含场 → 工件不可复用于出图。
- **局限**：审计未取得 de Vahl Davis 原文 PDF，u_max/v_max 参考值为 **WITHHELD**（仅以仓内 Nu 双表互证 + 文献记忆差 <0.05% 作旁证）；未复跑 300k 步收敛算例（核验单非重做，且集群排队 135）。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

| 关键量 | ref | 现有唯一 got（10k 步，N=64） | 误差 | <3%? |
|---|---|---|---|---|
| Nu_mean | 4.519 | 3.987 | 11.82% | **否** |
| u_max | 34.72 | 35.90 | 3.40% | **否**（3.40% > 3%） |
| v_max | 68.60 | 64.97 | 5.29% | **否** |

**结论：无一项 <3%，且无一项是收敛值；Nu 亦超任务 8% 容差。** 本案当前状态 = 基准**未通过**（非"通过"、非"部分通过"）。

### 6.3 L 级判定与 gate 预期

| 级 | 判据 | m15466 |
|---|---|---|
| L0 文档 | 报告存在 | ✓ |
| L1 数学/单元验证 | 测试 | ✗ 无任何测试引用该脚本（`grep -rn ra1e5 tests/` 仅命中无关的 `test_p97_n05_validation.py`） |
| **L2 独立功能（真代码 输入→计算→输出）** | 实现 + 入口 | **✓ 成立**：210 行脚本 + 真实 `tensorlbm.thermal` 链路 + 审计复现产出合法 JSON（含 `err_pct`/`pass`） |
| L3 系统集成 | 集成调用方 | ✗ 无主求解器/平台侧调用；仅作为库 API 的**外部消费者** |
| L4 基准验证 | 完整算例+误差+可复现 | ✗ 无收敛算例、无运行工件、无 bench 行、`l4-verdict=CLAIMED_NOT_PROVEN` |
| L5 科研工程 | +外部锚 | ✗ |

**裁定：L2。** 附百页报告不改变级别；此次未通过的是 L4，不是 L2。按平台 `maturity_gate.py` 的 `SCHEME`（L2 需 `impl_files`+`entry`+`run_artifacts` 文件存在），本单 `run_artifacts` **客观缺失**，gate 会自动降到 L1/L0 并开返修轮——这与本审计的"L2(功能实质) / L4(未证)"双判断不矛盾：**功能是真的，证据链是断的**，返修要做的是跑完并落工件，而非重写代码。

**gate 预演（用平台自身 maturity_gate.judge() 对已落表 evidence 复核）**：gate_level=0，原因 [L2 缺: run_artifacts(工件缺失), L1 缺: tests] —— 与本案结论一致：run_artifacts 客观缺失、脚本无任何测试；impl_files/entry 均通过存在性与非占位检查，即实现为真这半由机器确认。