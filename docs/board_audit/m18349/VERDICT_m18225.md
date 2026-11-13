# 交差审计 m18349 — m18225 `XVAL-LBM-PF-EOS-MAXWELL-HIGH-DENSITY-RATIO`（TensorLBM）

**结论（disposition）: request changes / 返修（reopened）。判级 L0。**
m18225 是 honest-FAIL 的调研轮：**零提交、零运行、零报告**。本锚点的全部实现与数据
（`benchmarks/eos_maxwell/*`、`tests/test_eos_maxwell_stability.py`）都来自 **m8675**（merge
`b245924c`，2026-11-10 campaign，5922.9 s），不属于本单。**把 m8675 的能力记到 m18225 账上=虚标**，
故不授 L2+；本单连仓库内文档（选型沉淀 / oss_solutions 行）都没有，判 **L0**。

- 审计单：m18349（branch `agent/m18349`），只读核验，不重做（未重跑 campaign）
- 落表：`tensor_maturity_verdicts` id=**738**（repo=TensorLBM, scope=`m18225 …`, mission_id=18349, level=0, verdict_by=board-audit）
- 机核：`judge(declared=0, evidence.tests=[]) ⇒ gate_level=0`，`link_original ⇒ 18225`（预期返修）
- 证据包：`docs/board_audit/m18349/{verify_m18225.py,verify_m18225.out,evidence_m18225.json,gate_dryrun_m18225.txt,fig_m18225_anchor_status.png}`

---

## ① 案例介绍（问题设定 + 基准出处）

- 物理：单组分 SCMP/PR-EOS（YS2006 参数，G=+1，2D D2Q9，float32，周期边界）两相共存，
  目标密度比 10 / 100 / 1000；判据来自 **Maxwell 等面积构造**给出的共存对（ρ_l, ρ_v, p_sat）。
- 任务形式：owner 2026-11-11 任务榜第 2 轮（前轮 m18132 honest-FAIL=no_progress）；本单要求
  ①集群 `~/oss-cache` 横向调研（lssfau/lbmpy `pf-eos-maxwell`）②方法谱系 ③选型沉淀进 `oss_solutions`
  ④结合实现，总账判据 **误差 <15%**。
- 基准出处（**唯一裁判**）：`tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-PF-EOS-MAXWELL-HIGH-DENSITY-RATIO'`
  ⇒ **0 行**（`ILIKE '%MAXWELL%' OR '%EOS%'` 亦 0 行）。即本锚点**没有注册基准案例**，
  ref/got/tolerance 无从机裁 → L4 对本锚点结构性不可达（对任何执行者，不只 m18225）。
- 既有（m8675）预注册门：密度比误差 ≤15%、伪流 Ma ≤0.10（PASS ≤0.05）、保持 ≥20000 步、
  质量漂移 ≤5e-08/步、ψ clamp 不触发（`benchmarks/eos_maxwell/CRITERION.md:5`）。

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

审计流程（全部可复跑，脚本 `verify_m18225.py`，输出 `verify_m18225.out`）：

1. **提交身份**：`git -C /root/tp-agent/work/m18225 rev-list --count main..HEAD` ⇒ **0**；HEAD=`3e2bf36d`（m17881 合并）；
   `git status --porcelain` ⇒ 仅 `.tmp_kb1..5.py`（12:37–12:42 API/DB 查询草稿）。
2. **实现溯源**：`git log --diff-filter=A -1 -- benchmarks/eos_maxwell/{probe_lib,run}.py` 与
   `tests/test_eos_maxwell_stability.py` ⇒ 三者均 `b245924c Merge #1598 (m8675)`。
3. **三件套存在性**：`benchmarks/eos_maxwell/out/results.json` ⇒ **不存在**（bytes=0），
   `git ls-files benchmarks/eos_maxwell/out` ⇒ 空（工件从未入库）。
4. **基准裁判**：psql 直查 `tensor_benchmark_cases`（同 ①）。
5. **repro_cmd 可执行性**（只跑 `--help`，不起算）：`python3 benchmarks/eos_maxwell/run.py --help` ⇒ rc=0，`--mode` 存在。
6. **报告**：`tensor_reports WHERE mission_id=18225` ⇒ **0 行**（`report_owed=false`，succeeded 自动放行，报告纪律漏网）。
7. **原单自述理由复核**：`cluster_jobs` id=1834 `state=queued`、`cmd='ls ~/oss-cache/'`、`mid` 空 ⇒ 队列积压属实；
   m18132 `status=failed / fail_stage=no_progress / 38 tool_calls / 0 changed_files` ⇒ 前轮死因描述属实。
8. **树内 provenance 测试独立复跑**（审计侧动作，不构成本单交付）：
   `pytest tests/test_eos_maxwell_stability.py -k 'results_artefact_matches or criterion_artefact_matches or maxwell_construction_hits_target or consumer_anchors_exist'`
   ⇒ **3 passed, 2 skipped**（skip：`campaign artefact not present in this tree` / `criterion artefact not present in this tree`）。
9. **证据与图**：`make_figure_m18225.py` 程序化解析 `RESULTS.md §2` 表格画图（零手抄）。

## ③ 计算结果（数字表）

**A. m18225 交付三件套核验（本单视角）**

| 项 | 要求 | 实测 | 判定 |
|---|---|---|---|
| 仓库内提交 | ≥1 | `main..HEAD` = 0 提交；HEAD 属他单 | ✗ |
| impl_files（本单源码） | 存在且非占位 | 无；仅 5 个未跟踪 `.tmp_kb*.py` 草稿 | ✗ |
| run_artifacts | 运行工件非空 | `out/results.json` 缺失且未入库（0 bytes） | ✗ |
| benchmark ref vs got | 注册案例机裁 | `tensor_benchmark_cases` 该 anchor = 0 行 | ✗（无裁判） |
| repro_cmd | 可执行 | 本单未留；既有 driver `--help` rc=0（真复现需 5922.9 s） | △ |
| tests（本单） | 测试名列表 | 无（既有测试属 m8675） | ✗ |
| integration_callers | 接入调用方 | 无本单接线 | ✗ |
| external_anchor（任务③选型沉淀） | `oss_solutions` 行/文献 | `feature_slug ILIKE '%EOS-MAXWELL%'` = 0 行 | ✗ |
| 论文级报告 | `tensor_reports` | mission_id=18225 ⇒ 0 行 | ✗ |

**B. 锚点既有（m8675）数值现状 —— 只作背景，不计入 m18225**（源：`benchmarks/eos_maxwell/RESULTS.md` §2，2026-11-10 机生成）

| run | forcing | 稳定/步数 | got 密度比 | ref（Maxwell） | 密度比误差 | 伪流 Ma | 预注册复合门 |
|---|---|---|---|---|---|---|---|
| tier_ratio10_edm | edm | True/20000 | 10.45 | 10.01 | **0.0437（4.37%）** | 0.115 | FAIL（Ma>0.10） |
| tier_ratio10_guo | guo | True/20000 | 13.81 | 10.01 | 0.379 | 0.127 | FAIL |
| tier_ratio100_edm | edm | True/20000 | 293.3 | 99.97 | 1.93 | 0.333 | FAIL |
| tier_ratio100_guo | guo | False/23 | — | 99.97 | — | — | FAIL（塌缩） |
| tier_ratio1000_edm | edm | False/100 | — | 995.2 | — | — | FAIL（塌缩） |
| tier_ratio1000_guo | guo | False/16 | — | 995.2 | — | — | FAIL（塌缩） |

- `RESULTS.md:165`：`tiers passed: 0 / 6`。
- **对总账判据（<15%）的准确读法**：仅 `tier_ratio10_edm` 的密度比误差 4.37% <15%（若只看密度比则达标），
  但同一 run 的伪流 Ma=0.115 超预注册 0.10 门，按预注册复合门该 tier 仍 FAIL；
  ratio 100/1000 **无任何存活 tier**。→ 该锚点当前状态下，"高密度比两相可用"不成立。
- **D2 格式偏差（重要）**：平面界面对照给出 实测/Maxwell = 1.10（ratio10×edm）/4.56（ratio100×edm），
  即 Maxwell 锁值本身不是离散格式的不动点（`CRITERION.md` D2 表）。任何"支持密度比 X"的声明须附此偏差。

## ④ 结果图

`fig_m18225_anchor_status.png`（两联图，数字由 `RESULTS.md §2` 表格机解析，脚本 `make_figure_m18225.py`）：
- 左：密度比误差 vs 目标比（对数轴）+ 15% 门线；右：伪流 Ma vs 目标比 + 0.10 门线；`×`=塌缩档。
- 图明确标注 **"NOT m18225 output; m18225 ran nothing"**——避免图被误引为本单成果。
- 无本单计算 → 无本单场量云图；本图为审计侧对既有数据的可视化，理由已在图题与本节声明。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **工件断链（最高危）**：`benchmarks/eos_maxwell/out/results.json` 从未入库，导致
   `test_results_artefact_matches_tree_code_hashes` / `test_criterion_artefact_matches_the_artefact_it_describes`
   **永久 skip**（`tests/test_eos_maxwell_stability.py:331,371`）。即：`RESULTS.md`/`CRITERION.md`
   的数字**当前无法在本仓库内自证**（sha256 provenance 环在"工件缺失"处断开）。任何下游引用
   CRITERION 数字都应知道这一点。
2. **本锚点无基准案例**：`tensor_benchmark_cases` 0 行 ⇒ maturity_gate 的 L4 机器判据
   （benchmark + repro_cmd）在本锚点永远缺一项；需要有人把 `<15%` 判据登记成 case 才能让后续单"可判 L4"。
3. **报告纪律漏网**：m18225 succeeded 且 `report_owed=false`、`report_id` 空 ⇒ 论文级报告 0 行仍被放行，
   守望线未触发。机制缺口在"honest-FAIL 也得交报告"未被强制。
4. **集群排队**：cj1834 是一条 `ls ~/oss-cache/` 的只读作业，却与 1800+ 号作业一起长期 `queued`；
   同一时刻 1874/1842/1829/1798… 均 queued。轻量只读作业应走旁路（这属平台侧观察，非本单判级依据）。
5. **范围限制**：本单不重跑 5922.9 s campaign（交差核验非重做）；不判 m8675 的 L 级；
   3D / 两相独立黏度未测，CRITERION D4 已声明不得外推。

## ⑥ 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准（ref） | got | 误差 | 是否 <15% |
|---|---|---|---|---|
| m18225 交付覆盖率（三件套+报告） | 4/4 存在 | **0/4** | 100% 缺失 | **否** |
| m18225 提交数 | ≥1 | 0 | — | **否** |
| 密度比 ratio=10 (edm，既有 m8675) | 10.01（Maxwell） | 10.45 | **4.37%** | 是（单看该量） |
| 伪流 Ma ratio=10 (edm，既有) | ≤0.10（预注册门） | 0.115 | 超门 15.0% | **否** → 复合门 FAIL |
| 密度比 ratio=100 (edm，既有) | 99.97 | 293.3 | **193%** | **否** |
| 密度比 ratio=1000 | 995.2 | 塌缩（100 步/16 步） | 不可测 | **否** |
| 外部锚点可溯源（lbmpy 源码/pin） | 需源码片段+hash | 无（oss_solutions 0 行） | — | **否** |

**显式回答"是否 <3%"**：本单 **无任何本单计算量**，故对本单不存在 <3% 可判量；
上表既有数据的最好一项 4.37% 也 **不满足 <3%**（且该量不属本单）。本单 L0 的依据是"零交付"，
不依赖精度数字。

---

## 发现问题（按严重度）

| # | 严重度 | 问题 | 证据 / 复现 |
|---|---|---|---|
| F1 | blocking | m18225 零提交、零运行、零报告，"结合实现④"完全未做 | `rev-list main..HEAD`=0；`tensor_reports` 0 行；`verify_m18225.out` |
| F2 | blocking | 声明的调研成果（lbmpy 方向）未沉淀：`oss_solutions(feature_slug='…EOS-MAXWELL…')`=0 行，任务③未完成 | psql 查询（`evidence.external_anchor`） |
| F3 | high | 锚点 run_artifacts 从未入库 ⇒ CRITERION/RESULTS sha256 provenance 环断开，2 条 provenance 测试永久 skip | `git ls-files benchmarks/eos_maxwell/out` 空；pytest `-rs` 输出 |
| F4 | high | 本锚点 `tensor_benchmark_cases` 0 行 ⇒ 无基准裁判，L4 结构性不可达（对全锚点） | psql 直查 |
| F5 | medium | succeeded 单报告缺失仍放行（`report_owed=false`）：报告纪律未强制到 honest-FAIL | `tensor_dev_missions` id=18225 |
| F6 | medium | 单次工具预算（80 calls / 2515 s）被只读调研吃尽，调研与实现之间无检查点，导致"调研完就死" | `result.tool_calls=80`、`fail_stage=budget_exhausted+forced_finish` |
| F7 | low | 集群队列让 `ls` 级只读作业也长时间 queued，调研类作业被饿死 | `cluster_jobs` id∈{1834,1842,1874,1829,1798} 全 queued |

**未运行的检查（unknown / not_run，不视为 passed）**：
- 未重跑 `benchmarks/eos_maxwell/run.py --mode tiers,tau,boundary` 全量 campaign（5922.9 s，交差非重做）；
- 未在 loong64 集群复算（本审计只在控制面 99 上跑 `--help` 与 8 s 级 pytest 子集）；
- 未验证 lbmpy 上游 `pf-eos-maxwell` 实现是否存在/可用（集群 oss-cache 作业长期 queued，m18225 亦未拿到）；
- 未判 m8675 的成熟度 L 级（越界，另单）。

## Overstated claims（对 m18225 自述的核对）

| 自述 | 核对结果 |
|---|---|
| "honest-FAIL，未进入实现/验证阶段" | **属实**（且属诚实申报，不惩罚） |
| "cluster cj1834 因队列积压 308 作业始终未执行" | **属实**（`cluster_jobs` id=1834 state=queued；同期多单同状） |
| "读前轮工件与核心库、DB 查询"（只读调研） | **属实**，但产出只落在 mission.result 文本，未成文档/未入 oss_solutions |
| 隐含"本锚点调研有可复用产出" | **不成立**：TD 侧可复用知识 0 条（oss_solutions 0 行、无 docs 文件） |

## 判级理由（L0）

- **L0 = 文档成果**：m18225 唯一"成果"是 mission.result 内的调研文字；**连一份仓内文档都没有**
  （选型沉淀/REPORT 均无）→ 取 L0 而不是更低（任务榜承认 honest-FAIL 留痕），也不取 L1+。
- **L1 不可授**：L1 要求 evidence.tests（数学/单元验证）非空；本单未跑任何测试。
- **L2 不可授**：L2 要求本单 impl_files + entry + run_artifacts；三者皆无。若把 m8675 的
  `benchmarks/eos_maxwell/*` 填进 impl_files，gate 可能被"文件存在+非占位"骗到 L2/L3 —— 这正是本次
  拒填的原因（**虚标=把别人的能力记到本单账上**）。
- **L4 结构性不可达**：`tensor_benchmark_cases` 该 anchor 0 行。
- 机核自证（`gate_dryrun_m18225.txt`）：声明 L0 ⇒ `gate_level=0`；即便声明 L2/L3/L4，机器也逐级降到 0
  （`L1 缺 tests, L2 缺 impl_files/entry/run_artifacts, …`）——判级不是"审计者意见"，是机器复核结果。

## 返修建议（给 reopened 的 successor 单）

1. 先在 `tensor_benchmark_cases` 为本锚点登记案例（metric=密度比误差 vs Maxwell 构造，tolerance=15%，
   ref_source=CRITERION.md D1/D2 + Maxwell 等面积构造），否则 L4 永远不可判。
2. 把 m8675 的 `out/results.json`（或等价工件）入库，恢复 provenance 链，让 2 条 skip 测试转绿。
3. 交付必须含：单锚点 run 工件 + `repro_cmd` + 报告（六段），并在 `oss_solutions` 落
   `feature_slug='XVAL-LBM-PF-EOS-MAXWELL-HIGH-DENSITY-RATIO'` 的选型行。
4. 若仍要"调研先行"，把调研与实现拆成两个节点（预算隔离），调研产物落 docs/ 再进实现节点。

---

*审计人：m18349（board-audit）｜只读核验，未改被审仓代码｜全部数字机取（`verify_m18225.out` / `RESULTS.md` / psql），零手抄。*