# 分级门禁语义的可执行验证（XVAL-LBM-GITLAB-CI-STAGED-GATES）

> 锚点：`XVAL-LBM-GITLAB-CI-STAGED-GATES`（lssfau/lbmpy 的 gitlab-ci-staged-gates 做法）
> 结论一句话：**阶梯此前只有"声明结构"被机器检查，语义（停机/屏障/豁免/事件作用域）从未被执行过；
> 本轮把四条声明变成可证伪的检查，并当场发现"第一级阻塞门永远是红的"这一实质性缺陷——整条阶梯在实践中不可达。**

## 1. 案例介绍（问题背景）

`.gitlab-ci.yml` 头部（ALT-2 移植）写死了四条阶梯主张：

| 主张 | 文本 |
|---|---|
| C1 | 每个 stage 是一道质量门；阻塞门失败则流水线不再前进 |
| C2 | `needs: []` 取消 DAG 依赖，但**不取消** stage 屏障 |
| C3 | `allow_failure: true` 的咨询门会跑，但**不能**让流水线停 |
| C4 | `rules:` 限定的条件门只在特定事件上运行 |

既有验证（`tests/test_ci_staged_gates_xval.py`, 18 passed）只做**结构普查**与标记覆盖率 KPI
（`gate_sim.ladder_structure`），即"声明了几个 stage / 几条 needs 边 / 哪些用例被门看到"。
没有任何测试执行过 C1–C4 本身。这与锚点要防的病一致：**声明了但从未被使用过的门禁阶梯**。

实例规模（本仓 `.gitlab-ci.yml`，本轮实测）：8 个声明 stage（其中 `integration` 为声明未启用），
16 个 job，13 个阻塞门，3 个咨询门。

## 2. 怎么做的（方法 / 工具链）

新建 4 件产物，全部在本仓内、无外部服务依赖：

1. `development/verification/gitlab_ci_staged_gates/stage_semantics.py` —— **独立第二解析器**（直接读原始 YAML，
   不走 `ci_spec` 的规范化模型；两个独立实现对不上本身就是一条发现）+ 执行语义模型：
   - 事件上下文：`mr / branch_push / main_push / schedule`，用合成变量表求值 `rules: if`；
   - 无法判定的表达式（如 `=~` 正则）抛 `UnverifiableCondition`，记为 **WITHHELD**，绝不猜测；
   - 执行推进：按 `stages:` 顺序，阻塞失败后 pruned 后续全部 stage；`needs:` 被 prune 则连带；
     `needs: []` 仍受屏障约束；`allow_failure` 失败不阻断；
   - 结构检查：stage 未声明、`needs` 指向不存在的 job、`needs` 指向更晚 stage（GitLab 直接拒绝流水线）、
     needs 环、事件上 needs 目标缺席（流水线不可创建）；
   - 制品契约：`artifacts:reports:*` 必须落在 `artifacts:paths` 内（否则报告永远传不上去），
     `coverage combine` 必须有上传 `.coverage.*` 的上游生产者；
   - 指标：`gate_power(event)` = 单个阻塞门失败时被 prune 的下游 job 数（0 = 装饰门）。
2. `tools/verify_ci_staged_gates.py` —— 收据生成器：对真实 `.gitlab-ci.yml` 做 67 项故障注入断言，
   可选 `--probe` 本地实跑便宜阻塞门；退出码即门禁判据。
3. `tools/ci_ratchet.py` + `tools/ci_ratchet_baseline.json` —— **棘轮门**（见 §4 的修复）。
4. `tests/test_ci_staged_gate_semantics.py` —— 15 条验收测试（含合成阶梯的反向变异测试，
   证明检查不是恒真）。

## 3. 计算结果（程序化注入，无手抄）

收据文件：`docs/ci/staged_gate_semantics_receipt.json` / `.md`（由 `tools/verify_ci_staged_gates.py` 生成）。

| 量 | 值 |
|---|---|
| 结构/制品契约 error 数 | **0** |
| 阶梯主张 C1–C4 故障注入断言 | **67 / 67 通过** |
| 不可判定规则（WITHHELD） | 0 |
| 装饰性阻塞门（power=0 且非末级 stage） | 0 |

门禁力量表（单个门失败时被 prune 的下游 job 数，数字由模型算出）：

| event | lint | format-check | typecheck-core | testsuite-fast | platform-test | repository-layout | architecture-audit | package-build | coverage-report | build-documentation | pages |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mr | 8 | 8 | 8 | 3 | 3 | 3 | 3 | 3 | 2 | 0 | – |
| main_push | 9 | 9 | 9 | 4 | 4 | 4 | 4 | 4 | 3 | 1 | 0 |
| schedule | – | – | – | – | – | – | – | – | – | – | – |

读法：`quality` 三道阻塞门失败会连坐 8–9 个下游 job（含 `pages` 部署），`docs`/`pages`
在非默认分支 pipeline 上没有下游（power 0 属正常，末级 stage 已排除）。

## 4. 关键发现与修复

### 发现 F1（阻塞性，已修）：第一级阻塞门按实测永远为红 ⇒ 整条阶梯不可达

实测（本机 ruff 0.15.0，路径 `src tests examples benchmarks`）：

| 门 | 声明命令 | 实测结果 | 后果 |
|---|---|---|---|
| `lint` | `ruff check src tests examples benchmarks` | **3028 findings** | 红 |
| `format-check` | `ruff format --check …` | **2101 个文件待重排** | 红 |

`quality` 是第一个 stage 且两者 `allow_failure: false`，**任何 pipeline 都到不了 stage 2**；
矩阵里 lint 的 power=8 恰好说明这条"永不通过的门"卡住了 8 个下游 job。这与 C1 的主张相悖。

**修复（最小、可验证、不放大 diff）**：把第一级改成**棘轮门**——规则集不缩小、阻塞力不降低
（"不得新增债务"），冻结基线 `tools/ci_ratchet_baseline.json`
（`ruff {total:3028, by_code:{…}}`、`format {total:2101, fingerprint}`），只许降不许升；
按**每条规则码**独立棘轮，修 `F401` 不能赎买新 `E722`；`format` 另存文件集指纹，
"改好一个、改坏一个"计数不变也能被抓到。原全量 `ruff check` 降级为咨询门 `lint-full`
（债务可见但不再堵死阶梯）。棘轮自身的回归性有测试：把基线人为调低后 `rc=1`。

### 发现 F2（阻塞性，未修，记账）：`repository-layout` 阻塞门当前也是红的

`tools/check_repository_layout.py` 在本 worktree 实测 `rc=1`：根目录遗留 `verdict_*.sql` 等文件、
以及**被 git 跟踪的 `__pycache__/*.pyc`**（`xval/XVAL-LBM-ADDEDMASS-KC-RESPONSE-ENVELOPE/`）。
按检查器自身提示（"不要为了隐藏新发现而去重生成例外清单"），本轮**不**改策略、不删用户数据，
只如实入账：该门在本地探针中记为 `fail`，是本锚点之外的既有仓库债，建议单独开单清理。

### 发现 F3（方法学）：C1/C2 任何合规 pipeline 都恒真，真正可证伪的是配置层

故障注入显示：只要 stage 屏障生效，C1（失败停机）与 C2（`needs: []` 不脱屏障）**无法**被 YAML
变异破坏——它们是 GitLab 语义的推论，不是本仓的成就。把"阶梯有效"归因于 C1/C2 通过属于**过度声明**。
因此新增两条**配置层**可证伪性质并纳入测试：
① 无常量装饰门（`gate_power>0`，末级 stage 除外）；② C4b 默认分支限定门的声明与行为一致
（变异实验：把 `if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH` 改成恒真的 `if: $CI_DEFAULT_BRANCH`
后，C4b 转红——检查确实会失败，不是恒真）。

## 5. 注意点 / 局限 / 适用范围

1. **模型是 GitLab 的子集**：只实现了本仓实际用到的语法（`rules: if` 的 `$VAR` 比较 + `&&/||`、
   `needs`、`allow_failure`、`extends`、`artifacts`）。`=~` 正则、`parallel:matrix`、`include`、
   动态子流水线未建模——遇到不可判定条件会显式记为 WITHHELD（本轮为 0 条）。
2. **未在真 GitLab runner 上跑过**：本仓 CI 实际跑在 GitHub Actions（`ci-loong64`），
   `.gitlab-ci.yml` 是移植产物；本轮验证的是**语义模型 + 配置一致性**，不是 GitLab 服务端行为。
3. **`typecheck-core` / `testsuite-fast` / `platform-test` / `package-build` / `build-documentation`
   本地未跑**（超短预算或需专有环境），收据里标 `not_run` 并给理由，**不记作通过**。
   已递交集群作业 `cj4170` 做 mypy 复核，本单未取回结果。
4. 棘轮基线绑定 `ruff 0.15.0`；升级 ruff 会移动计数，工具会把版本漂移写进 `warnings` 而不是静默接受。

## 6. 误差判定（本单为工程/门禁任务，基准是"声明语义"而非解析解）

| 关键量 | 判定基准 | 实测 | 是否 <3% |
|---|---|---|---|
| C1–C4 主张 | 与 YAML 声明逐条比对 | 67/67 一致 | 是（0 偏差） |
| 结构/制品契约 error | 0 容忍 | 0 | 是 |
| 装饰性阻塞门 | 0 容忍 | 0（末级 stage 豁免） | 是 |
| `lint` 门当前债务 vs 基线 | 基线 3028 | 3028 | 是（0%） |
| `format` 门当前债务 vs 基线 | 基线 2101 | 2101 | 是（0%） |
| `repository-layout` 门 | 期望 rc=0 | **rc=1** | **否——honest FAIL，见 F2** |

复现：
```bash
python tools/verify_ci_staged_gates.py --probe          # 收据 + 本地探针，VERDICT
PYTHONPATH=. pytest tests/test_ci_staged_gate_semantics.py -q     # 15 passed
PYTHONPATH=. pytest tests/test_ci_staged_gates_xval.py -q         # 18 passed（原有）
python tools/ci_ratchet.py ruff && python tools/ci_ratchet.py format
```