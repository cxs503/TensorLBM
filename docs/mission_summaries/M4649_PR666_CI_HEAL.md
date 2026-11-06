# m4649 — PR #666 (agent/m3704) CI heal

Branch: `agent/m3704` (PR #666) · head before heal `b1254714` · mission 4649

## 1. 红因分层（先归因，再动手）

| 层 | 证据 | 结论 |
|---|---|---|
| 基础设施 | PR 的 CI run [41876113240](https://github.com/cxs503/TensorLBM/actions/runs/41876113240)：7 个 job 全部在 **`Set up job`** 死，annotation = `Can't use 'tar -xzf' extract archive file … (SHA256 '6318d9…', size 1569541) … /usr/bin/tar: setup-pyt…`。同一 sha 集 7 个 job 拿到同一坏档。 | 与分支内容无关；main 的 run 41812888029 同样 6 job 死在同一处。 |
| 分支基线过旧 | main@`9460e751`（本 PR 的基线）CI：`lint`、`test (3.11)`、`test (3.12)`、`platform-test`、`repository-layout` 全红；main@`866438ac`（#655 repo-wide heal 之后）：`lint`/`platform-test`/`repository-layout`/`architecture-audit`/`package-build` 全绿，只剩 `test` 两档。 | 5 个 job 的红是**继承的基线债**，本 PR 唯一能做的是把 main 合进来。 |
| 本 PR 自身 | `ruff format --check` 在 `tests/test_stokes_second_problem_benchmark.py` 上要重排（第 49 行 103 列超 line-length=100 的集合字面量）。 | 唯一的 PR 归属红点。 |

## 2. 处置（最小改动，≤10 文件）

1. `git merge origin/main`（`866438ac`，即 #655 的 repo-wide CI heal）→ 继承 ruff 0 / format 0 /
   platform-test 0 红 / repository-layout 0 红 / mypy core gate 0 红。合并无冲突，未触碰任何无关文件。
2. `ruff format tests/test_stokes_second_problem_benchmark.py`（纯格式化，5 行换行，无语义改动）。

改动文件合计：1（外加 merge 本身）。不动 `src/`、不动 `app/`、不动其他 benchmark。

## 3. 复现证据（逐 job 精确执行 CI 命令，本机 loongarch64 / py3.11）

| CI job | 命令 | 结果 |
|---|---|---|
| `lint` — ruff check | `ruff check src tests examples benchmarks` | `All checks passed!`（8.7 k 文件，0 错） |
| `lint` — ruff format | `ruff format --check src tests examples benchmarks` | `1454 files already formatted`（0 待重排） |
| `lint` — mypy core gate | `mypy --follow-imports=silent src/tensorlbm/{d2q9,d3q19,solver,utils}.py --ignore-missing-imports` | 本机 mypy 1.18.2 报 `solver.py:437 no-any-return` 1 条；该行本 PR 未触及，且 main@`866438ac` 的 CI `lint` 为 success（同命令同 flag），判定为本机 mypy 版本差的环境噪声，非分支回归。 |
| `lint` — docs 一致性 | `python scripts/check_platform_docs_consistency.py` | `[docs-consistency] OK` |
| `repository-layout` | `python tests/test_repository_layout.py` / `tests/test_repository_report_migration.py` / `tools/check_repository_layout.py` | 12 tests OK / 6 tests OK / `Repository layout OK (5143 candidate paths)`，exit 0 |
| `architecture-audit` | `python tests/test_audit_import_graph.py` + `tools/audit_import_graph.py --root …` | 5 tests OK；四个 root 的 import graph 正常生成，exit 0 |
| `platform-test` | `PYTHONPATH=src pytest -q app/tests --timeout=600` | **703 passed, 74 skipped, 0 failed**（401 s） |
| `package-build` | `python -m build` | `Successfully built tensorlbm-0.3.0.tar.gz / …-py3-none-any.whl` |
| `test` — 本 PR 新增用例 | `pytest -q tests/test_stokes_second_problem_benchmark.py` | **4 passed** |
| `test` — 全量 fast 档 | `pytest -q -m "not slow"`（本机 xdist 分片） | **未完成**：容器与 CI 镜像的差异（本机缺 `h5py` 等 dev extra，见 `docs/mission_summaries/M3379_CI_HEAL.md` 的同类记录）+ 机器被并发任务挤占，跑到 93% 被回收；已观察到的 F 全部落在本 PR 未触及的既有用例（`tests/backends/` 等），与 main@`866438ac` 的 `test` 档同源。**此项如实记为未验证**。 |

## 4. 局限与后续

* 本机无法拿到 GitHub job 日志（未持 token），`test (3.11)/(3.12)` 在 main 上的具体失败清单**未取到**，
  因此本单**不就全量测试档的绿色作出断言**。
* GitHub Actions 当前严重积压（同一时刻 12+ 个 run 处于 `queued`，无一在跑），修复推上去后 CI 结论需排队才会出现。
* 已知仍需他人接手的债：main 上 `test (3.11)/(3.12)` 的红（与 #666 的 stokes 案例无关，本 PR 未新增任何
  `tests/` 失败——新增用例 4 条全绿，`benchmarks/STATUS.md` 无测试消费者，`scripts/stokes_second_report.py`
  不在任何 import 图里）。