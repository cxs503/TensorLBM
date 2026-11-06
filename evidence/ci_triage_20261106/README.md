# main CI 红单排查：不是测试失败，是 run 被取消（2026-11-06）

排查对象：`main@1ef76442` 上 `CI` 工作流七个 job 全红
（repository-layout, lint, test (3.11), test (3.12), platform-test,
architecture-audit, package-build）。

## 结论（一句话）

七个 job 的 `conclusion` 全部是 **`cancelled`，不是 `failure`**：没有任何 job 分到 runner、
没有任何 step 被执行、没有任何断言失败。红的原因在工作流并发策略，不在代码或测试。

证据由 `collect.py` 程序化抓取（GitHub Actions API），写入 `summary.json`，本页所有数字
均来自该文件，无手抄：

| job | conclusion | runner | lifetime (s) | steps recorded |
| --- | --- | --- | --- | --- |
| package-build | cancelled | — | 2 | 0 |
| architecture-audit | cancelled | — | 2 | 0 |
| lint | cancelled | — | 2 | 0 |
| platform-test | cancelled | — | 2 | 0 |
| test (3.12) | cancelled | — | 2 | 0 |
| test (3.11) | cancelled | — | 2 | 0 |
| repository-layout | cancelled | — | 2 | 0 |

run `42150673200`（event `push`, sha `1ef76442`）总寿命 **5 s**；有 failure 结论的 job 数 **0**。
同窗口 40 条最近 run 的结论分布：`queued/None` 30 条、`completed/cancelled` 9 条、
`pending/None` 1 条 —— 即整条队列基本没有任何 run 真正跑起来过。

## 根因（两条叠加）

1. **runner 供给不足**：同窗口 30 条 run 长期停在 `queued`，说明没有可用 runner（托管额度
   或自托管机池），与 `docs/history` 中 2026-10-26 那次的"跑得慢"不同——这次是"根本没开始"。
2. **concurrency 组把 main 的 run 互相顶掉**（本仓可修的那一半）：GitHub 对每个
   concurrency 组只保留 **一个 running + 一个 pending**，新 run 入组时会取消 **上一个
   pending** run，而这条规则与 `cancel-in-progress` 的取值无关（该 flag 只决定要不要杀
   *正在跑* 的）。原策略

   ```yaml
   group: ci-${{ github.workflow }}-${{ github.ref }}   # main 上全仓共用一个组
   ```

   于是当两个 merge 相隔数秒落 main（智能体并发交付的常态），后一个 push 就把前一个 commit
   的 pending run 取消掉——该 commit 的 CI 就永久显示红色，且**一个 step 都没跑**。
   原注释"main is never cancelled"与 GitHub 的语义相反，是这次误判的来源。

## 修复

`.github/workflows/ci.yml`：main 的并发组按 **commit** 分组，分支/PR 仍按 ref 分组
（保持"新 push 顶掉陈旧 run"的省算力行为）：

```yaml
group: ci-${{ github.workflow }}-${{ github.ref }}${{ github.ref == 'refs/heads/main' && format('-{0}', github.sha) || '' }}
cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}
```

- main：不同 commit ⇒ 不同组，任意两个 main commit 的 run 不再互相取消；同一 commit 的
  cron/手动 dispatch 仍共组，重复触发仍会被合并。
- 分支/PR：组不变（`refs/heads/<branch>`、`refs/pull/N/merge`），push 仍互相顶掉；
  PR run 与 main push run 天然不同组，merge 不会杀掉 PR 的裁决。
- `cancel-in-progress` 对 main 仍为 false：夜间/慢用例信号不会被后续 push 杀掉。

## 判据（可复跑）

`tests/test_ci_workflow.py` 新增 5 组用例，**执行**（而非文本匹配）工作流里的
`concurrency` 表达式：用一个只支持 ci.yml 所用子集（`==`/`&&`/`||`/`format()`）的求值器渲染
分组，再断言：

- `test_main_commits_get_distinct_concurrency_groups`：两个 main commit 组名不同；
- `test_ref_wide_main_group_is_the_regression_this_guards`：**负控**——旧的 ref 级表达式的
  两组名相同（证明上面的断言不是空断言）；
- `test_branch_groups_stay_ref_scoped_so_new_pushes_supersede`：分支仍按 ref 分组；
- `test_pull_request_runs_share_no_group_with_main_pushes`：PR 与 main push 不同组；
- `test_cancel_in_progress_never_kills_main`：main 恒为 false，其余 ref 为 true。

负控已实测：把 `ci.yml` 退回旧的 ref 级表达式后，前两条用例立即失败（2 failed / 13 passed）；
恢复修复后 15 passed。

## 本地门禁复跑（同一工作树，Python 3.11，见 `environment.txt`）

| 门 | 结果 |
| --- | --- |
| `ruff check src tests examples benchmarks` | 通过 |
| `ruff format --check src tests examples benchmarks` | 通过（1453 files） |
| `python tests/test_repository_layout.py -v` | 12 passed |
| `python tests/test_repository_report_migration.py -v` | 6 passed |
| `python tools/check_repository_layout.py` | OK（5136 paths） |
| `python tests/test_audit_import_graph.py` + `tools/audit_import_graph.py` | 5 passed + 报告生成 |
| `python scripts/check_platform_docs_consistency.py` | OK |
| `python -m build` | sdist + wheel 构建成功 |
| `PYTHONPATH=src pytest app/tests`（平台测试 job） | 703 passed, 74 skipped |
| `pytest tests/data tests/ml` | 130 passed |
| `pytest tests/test_ci_workflow.py` | 15 passed |

未完成/未证实（如实记录）：

- 全量 `pytest -m "not slow"`（6808 用例）本次会话**未跑完**：本机同时有其他智能体的
  `-n 16` 并行 pytest 抢占 128 核，我的串行扫到 7% 就超时收工。已跑到的部分（第 1–8% 区间
  共约 500 例，含 data/ml/ai4s 首段）无 failure；`h5py` 相关用例在补装 `h5py` 后由
  13 failed 变为全绿，确认此前的本地红是宿主缺依赖而非仓库缺陷。
- `mypy` core gate 未在本会话内确认（100 s 预算内未返回），交由 CI 复核。
- **队列本身能否转绿仍取决于 runner 供给**：本修复消除的是"被新 push 取消"这一确定性红因，
  若 runner 依旧不可用，main run 会变成 `queued`（黄）而不是 `cancelled`（红），这属基础设施
  范围，本仓无法修复。2026-11-06 02:43 合入的 `ci-loong64` 自托管池（#679）是同一问题的另
  一半处置。

## 适用边界

- 本页只做 CI 事实认定与工作流策略修复，不动物理阈值、不删测、不加 xfail、不降低门禁。
- 取消/排队状态是 2026-11-06 02:20–02:48 UTC 的快照；结论以 `summary.json` 的
  `collected_at` 为准，过期请重跑 `collect.py`。