# m18159 交差核验报告 — m18068 `XVAL-LBM-NO-MULTIPHASE-LAYOUT-PATTERN-ONLY`（TensorLBM）
> 审计任务 m18159 · 对象任务 m18068 · 判级 L0 · gate 预期 `reopened` · 证据源 `docs/board_audit/m18159/evidence.json`（程序化采集，零手抄）
> **一句话结论**：m18068 的 runner 状态是 `succeeded`，但交付实质为空——0 个提交、0 个工件、0 个测试、0 条基准算例、0 篇报告；唯一产物是两个未跟踪的接口探针脚本。判 L0，锚点能力缺口完全未闭合，交差结论 = **返修（reopened）**。

## ① 案例介绍（对象/背景/参数）
**锚点**：`XVAL-LBM-NO-MULTIPHASE-LAYOUT-PATTERN-ONLY`，任务榜「开源对标（XVAL）」第 1 轮，目的=把 TensorLBM 的多组分分布布局与成熟开源实现 2b-t/LB-t 的做法对齐，判据为对标误差 <15%。
**对标实现**：`oss_solutions` id=103 → 仓 `2b-t/LB-t`，feature_slug `GAP-LBM-MCMP-POISEUILLE`，tag `no-multiphase-layout-pattern-only`。
> 平台记录的该做法原义：「该仓无多相实现；唯一可借的是 NPOP 并排存多套分布的布局机制与"组分索引 p 进算子签名"的接口范式（从未实例化 NPOP>1，故只是范式不是实现）。」
即 LB-t 侧唯一可借的是**布局范式**（`AaPopulation<LT,NP>` 把 NP 套分布并排存成 `NZ*NY*NX*NP*ND` 的一维数组，`write(x,y,z,n,d,p=0)` 把组分索引 p 放进算子签名，`spatialToLinear` 给出索引映射，`aa_pattern` 分 even/odd 选路），且上游从未实例化 NP>1。因此本锚点的物理内容=**单组分（NP=1）情形下的 AoS/AA 布局与索引**，对标对象是「做法」而非「数值解」。

**对象任务参数表（全部取自 `tensor_dev_missions`）**

| 项 | 值 |
|---|---|
| 任务 id / 仓 | 18068 / TensorLBM |
| runner 状态 | `succeeded` |
| claim / 模型 | `tp-qw38-15` |
| 起止 | 2026-11-13T11:22:46.713321+00:00 → 2026-11-13T11:38:32.259174+00:00（15.8 min） |
| 预算 | tool_calls ≤ 80，wall ≤ 7200 s，tokens ≤ 3000000 |
| 实际用量 | tool_calls 32（40%），wall 15.8 min（13%） |
| 自报 fail_stage | `no_progress+forced_finish` |
| 交付分支 / commit | `None` / `None`（均 null） |
| changed_files | `.scratch_m18068_kb.py, .scratch_m18068_kb2.py` |
| 验收请求 | tensor_approval_requests#12176 kind=`dev_acceptance` status=`approved` decided_by=`eco_admin` |
| 报告湖（tensor_reports） | 0 篇 |
| 基准算例（tensor_benchmark_cases） | 0 条（全平台共 91 条） |

## ② 怎么做的（方法/流程/工具链）
**判据先于取证**（预声明于 `scripts/m18159_audit.py` docstring，未在看到结果后调整）：

| 规则 | 内容 |
|---|---|
| R1 唯一裁判 | 数值主张只有存在 `tensor_benchmark_cases` 行（ref/metric/tolerance）且 got 落在容差内才成立；无行 ⇒ 不可判 |
| R2 L2 | 必须有真代码：能 input→compute→output 的入口，且在交付分支上提交 |
| R3 L3 | 另需能 grep 到主求解器集成调用方 |
| R4 L4 | 另需完整算例 + 误差分析 + 可执行 repro_cmd |
| R5 L5 | 科研/工程级成果，超出一道基准通过 |
| R6 三件套 | run_artifacts/tests/repro_cmd 必须是文件，不是文字 |
| R7 诚实性 | 「预算耗尽」类归因须用预算数字自证 |

**取证链（四个独立源，互相交叉）**：

1. `tensor_dev_missions`（任务记录 + result/evidence JSON）→ 预算、用量、fail_stage、changed_files；
2. `tensor_benchmark_cases`（按 anchor_slug 精确匹配）→ 是否存在机器裁判；
3. `tensor_reports`（mission_id 精确匹配）→ 是否发布六段报告；
4. 对象工作树 `/root/tp-agent/work/m18068` 的 git 状态 → 提交数(`git rev-list --count origin/main..HEAD`)、未跟踪文件及其 sha256。

**工具链**：`scripts/m18159_audit.py`（采集→`evidence.json`）→ `scripts/m18159_figures.py`（作图）→ `scripts/m18159_render_report.py`（六段报告，assert 绑定）→ `docs/board_audit/m18159/insert_verdict.py`（幂等 upsert verdict）。复跑 `scripts/m18159_verify.py` 可重判。

## ③ 计算结果（核验矩阵，数字表）

### 三件套 + 五要素逐项核验

| # | 核验项 | 判据 | 结果 | 证据 |
|---|---|---|---|---|
| 1 | `impl_files` | delivered source files (src/xval/tests) on the delivery branch? | ❌ MISSING | only two untracked probe scripts at repo root |
| 2 | `entry` | executable entry (input->compute->output) for the anchor? | ❌ MISSING | no runnable driver references the anchor |
| 3 | `run_artifacts` | results/artifacts (json/csv/png) produced for the anchor? | ❌ MISSING | {"xval_layout_dirs": [], "xval_multiphase_dirs": []} |
| 4 | `tests` | tests asserting the new behaviour? | ❌ MISSING | no test file added; existing tests/test_aa_pattern_parity.py predates the mission |
| 5 | `integration_callers` | reachable from the main solver (L3)? | ❌ MISSING | no new symbols exist to be called |
| 6 | `benchmark` | tensor_benchmark_cases row for the anchor with ref/metric/tolerance? | ❌ MISSING | {"rows": [], "count": 0} |
| 7 | `repro_cmd` | repro_cmd that reproduces the numbers? | ❌ MISSING | no repro_cmd in the mission result or worktree |
| 8 | `external_anchor` | external OSS anchor identified with a real pointer? | ✅ MET | {"id": 103, "repo": "2b-t/LB-t", "feature_slug": "GAP-LBM-MCMP-POISEUILLE", "tag": "no-multiphase-layout-patte… |
| 9 | `report_six_sections` | six-section report published to the report lake? | ❌ MISSING | — |

### 交付文件实况（git + sha256）

交付分支 `agent/m18068` 相对 `origin/main` 的**自有提交数 = 0**（工作树 HEAD 为 `4903d86c Merge #2085 (m17738) via merge-master`，即 main 的合并点，非本单产物）。未跟踪文件：

| 路径 | 字节 | sha256 | 性质 |
|---|---|---|---|
| `.scratch_m18068_kb.py` | 1385 | `dc04831aac52b645…` | 接口探针（非交付代码） |
| `.scratch_m18068_kb2.py` | 1090 | `5c79738efd103a98…` | 接口探针（非交付代码） |

`xval/` 下匹配 `*layout*` / `*multiphase*` 的算例目录：[] / [] —— 空。

### 判级推导

| 台阶 | 前置条件 | 是否满足 |
|---|---|---|
| L1 数学/单元验证 | 有可被测试断言的实现 | ❌（无新增测试） |
| L2 独立功能实现 | impl_files ∧ entry | ❌ / ❌ |
| L3 系统集成 | L2 ∧ integration_callers | ❌ |
| L4 基准验证 | L3 ∧ benchmark ∧ repro_cmd ∧ run_artifacts | ❌ |

**判级 = L0**：仅保留「开源做法侦察」这一知识性产出。no implementation, no artifact, no test, no repro_cmd, no benchmark case for the anchor -> nothing above L0 is admissible; capability gap stays open

### 预算主张核验（R7）

任务自述失败原因为「预算耗尽于侦察阶段」。机器数字：tool_calls 32/80 = **40%**，wall 15.8/120 min = **13%**。两项均未耗尽，故该归因为**不实陈述**（不影响其「未做实现」这一 self-report 的诚实性）。

## ④ 结果图

![evidence matrix](figures/fig1_evidence_matrix.png)

**图 1** 左：10 项预声明证据检查，仅 `external_anchor` 通过（1/10）；右：L2–L4 台阶前置条件全未满足，判级钉在 L0，gate 预期 `reopened`。

![budget utilisation](figures/fig2_budget_utilization.png)

**图 2** 失败归因与预算实况对照：tool_calls 40%、wall 13%，红色虚线为预算天花板——「预算耗尽」的主张在图上即为伪。

## ⑤ 注意点（坑/局限/适用范围）

1. **`status='succeeded'` ≠ 交付成功**。本轮任务的锚点描述写「已 succeeded」，实际是 runner 生命周期状态；判定必须看 `result.fail_stage`、`branch`、`head_commit`、`changed_files`。这是本单最重要的口径陷阱。
2. **锚点没有机器裁判**：`tensor_benchmark_cases` 里该 anchor_slug **0 行**（全表 91 行，其它 XVAL-LBM 锚点普遍有行）。没有 ref/metric/tolerance，「误差 <15%」这条判据在本锚点上**无法被机器执行**——这是 dispenser 侧的缺口，不是执行侧的问题。
3. **auto-approve 放行了 no-progress**：`tensor_approval_requests#12176` 以 `approved` 收尾，`accepted=true`，`decided_by=eco_admin`（自动批准路径）。诚实 FAIL 被批准为 accepted，账面上会掩盖空交付，须靠本类交差核验反向消账。
4. **代理角色无 SELECT 权限**：`tensor_benchmark_cases` / `tensor_maturity_verdicts` 对代理账号 `permission denied`，本审计改用只读 foundation 连接（DSN 运行时解析，不入库、不落盘）；平台缺一个「benchmark case 只读」API，这是可复现性上的生态摩擦。
5. **未跟踪探针不能计入交付**：两个 `.scratch_m18068_kb*.py` 是 KB/oss_solutions 端点探测脚本，且修复不了 404 问题（端点确实不存在）；把它们算成「代码交付」会虚增到 L2。
6. **独立性限制**：本审计未重跑 m18068（无对象可跑，任务未产出任何可执行入口）；结论全部由 primary source（DB 行 + git 对象 + 文件 sha256）重新推导，未采信其自述的任何未经验证的说法；LB-t 源码侧只核到 `oss_solutions` 记录的范式层，`/tmp/LB-t` 检出已不在（临时目录已被回收），故本审计不声称复核了上游源码逐行。

## ⑥ 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准/期望 | 实测（got） | 误差 | 是否 <3% |
|---|---|---|---|---|
| 对标误差（VS LB-t 布局范式） | 需 <15%（owner 精度标准） | **未产生任何数值** | 不可计算 | **N/A（无 got 值，非通过）** |
| benchmark 算例行数 | ≥1 | 0 | 缺 1 条裁判 | **否（0 ≠ 1）** |
| 交付提交数 | ≥1 | 0 | 缺 1 次提交 | **否（0 ≠ 1）** |
| 六段报告 | 1 篇 | 0 | 缺 1 篇 | **否（0 ≠ 1）** |
| 预算用量（诚实性） | 「耗尽」⇒ ≈100% | tool_calls 40% / wall 13% | 剩余 60% / 87% | **否（主张与实测不符）** |

**显式回答「是否 <3%」**：不适用且**不构成通过**——m18068 未产出任何可比较的数值量，「误差 <3%」这一问在本题上无被测量；真正被量化的是**证据覆盖率**：10 项预声明检查中 1 项满足（仅外部锚存在），核心数值判据 <15% 从未被执行。

## 处置（disposition）

- **判级**：L0（文档/知识性侦察），不授 L1+；
- **gate 预期**：`reopened` —— 能力缺口 `GAP-LBM-MCMP-POISEUILLE`（no-multiphase layout pattern）完全未闭合，无「已确认」对象；
- **返修建议（给下一轮）**：①先为该 anchor 在 `tensor_benchmark_cases` 注册一行（ref=LB-t 的 `AaPopulation<LT,1>` 索引映射公式，metric=索引/写读往返一致性，tolerance=0），否则该锚点永远不可判；②实现范围收窄为「NPOP=1 的 AoS 布局 + p 进算子签名 + spatialToLinear 索引映射」，在 TensorLBM 侧落 `tests/` 往返一致性与 even/odd 选路两项断言即可先登 L2；③侦察阶段（读上游源码、找端点）本单已证明 32 tool_calls 足够，不需要再花整单预算。
- **附带的平台级发现**（超出本单裁决范围，供 dispenser/gate 侧修）：F3（失败归因不实）、F4（auto-approve 放行 no-progress）、F5（知识库端点 404、锚点无基准行）。
