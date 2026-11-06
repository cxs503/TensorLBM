# m4684 — PR #676 重开绿（TensorLBM）

分支：`agent/m4684`（从 `agent/m4649` 内容重开到当前 `origin/main@1ef76442`）· mission 4684 · 第 2 轮

## 1. 红因分层（先归因，再动手）

第 2 轮入场时 PR #676（`agent/m4649`，head `8c8c46c3`）报 13 条 check 全部 `cancelled`。
用仓库 API 逐层取证（只读）：

| 层 | 证据 | 结论 |
|---|---|---|
| ① 本 PR 的 run | run `42147611612`（`pull_request`，8c8c46c3）：7 个 job 均在 `02:01:20/21` **开始执行**，全部在同一秒 `02:04:27` 被 cancelled（`lint`/`test`/`repository-layout` 等无一例外）。 | 不是逐步失败，是**整 run 级取消**：只有整 run cancel 才会让 7 个不同时长的 job 同秒终止。 |
| ② 谁取消 | PR #676 `closed_at=2026-11-06T02:21:04Z`，issue event = `closed by cxs503`，评论原文「裁定：修复 ≥2 轮未绿（owner 2026-11-05：修正成本高于重开），关闭重开。原分支留档。」 | 全 cancelled = PR 被关闭导致其在飞 run 终止（另一条 run `42146941684`@01:54 同样 3 分钟内被取消）。**分支内容不是红因**。 |
| ③ 重开的真实阻塞 | `main@866438ac` 的 push-run `41995371431`：`repository-layout`/`architecture-audit`/`package-build`/`lint`/`platform-test` **全绿**，`test (3.11)`/`test (3.12)` **红**。日志取得 5 条 FAILED。 | 新 PR 的基线本身带红；不修则重开也绿不了。 |
| ④ 哪些红会在 PR 档生效 | 5 条 FAILED 中 2 条在 `tests/test_multiphase_physical_benchmarks.py`，带 `@pytest.mark.slow` → PR 跑 `-m "not slow"` 不会选中。其余 **3 条在 fast 档**，必须修。 | 修复面收缩到 2 个文件、3 条断言。 |

## 2. 处置（最小改动）

1. `agent/m4649` 的内容原样重开到 `agent/m4684`（基线 `origin/main@1ef76442`，含 #667）：stokes_second_problem 案例档 + 回归守卫测试 + 验证文档 + 报告脚本。
2. 修 3 条 fast 档红断言（均为**测试侧**陈旧/脆弱断言，不动 `src/`）：
   * `tests/test_control_volume_force.py:220` — `pytest.approx(bfl_force[0])` 把 0-d `Tensor` 当 `expected`，pytest≥8.4 的 `approx` 会迭代 `expected` → `TypeError: iteration over a 0-d tensor`。改为 `pytest.approx(float(bfl_force[0]), abs=1e-8)`（float() 与左值 `.item()` 对称）。
   * `tests/test_regress_wall_multi_gpu.py::TestBugIdentification::test_wss_from_fneq_3d_import_error_equilibrium_vs_equilibrium3d` 与 `TestWallShearBugs::test_wss_from_fneq_3d_broken_import` — 两条负控断言"`wall_shear` 从 d3q19 导错名（`equilibrium` vs `equilibrium3d`）必抛 ImportError"。该 bug 早已修好（`wall_shear` 现导入 `equilibrium3d`），负控因此永久 `DID NOT RAISE ImportError`。改为**反向守卫**：调用必须成功且返回有限场（`torch.isfinite(wss).all()`）。
3. 删除上一轮已被 owner 关闭的 PR #676 的过程档 `docs/mission_summaries/M4649_PR666_CI_HEAL.md`（其内容已被本文第 1 节取代，避免两份互相矛盾的 CI 归因）。

改动文件合计 3 个（2 个测试 + 1 个新增本文；案例档为 m4649 原样重开）。

## 3. 复现与验证证据（逐 job 精确执行 CI 命令，本机 loongarch64 / py3.11 / `PYTHONPATH=src`）

| 红测 | 修复前 | 修复后 |
|---|---|---|
| `test_control_volume_force.py`（整档 9 条） | `1 failed, 8 passed`（`TypeError: iteration over a 0-d tensor`，完整的 pytest.approx 栈） | **9 passed** |
| `test_regress_wall_multi_gpu.py::TestBugIdentification+TestWallShearBugs` | `DID NOT RAISE ImportError`（`tests/test_regress_wall_multi_gpu.py:88`） | **7 passed, 1 xfailed** |
| `test_stokes_second_problem_benchmark.py`（本 PR 新增） | — | **4 passed** |
| 合并复跑 | — | `13 passed in 90.66s`（`artifacts/m4684/pytest_fixed.log`） |

| CI job | 命令 | 结果 |
|---|---|---|
| `lint` ruff check | `ruff check src tests examples benchmarks` | `All checks passed!` |
| `lint` ruff format | `ruff format --check …` | `1454 files already formatted` |
| `repository-layout` | `tools/check_repository_layout.py` + `tests/test_repository_layout.py`（12 tests）+ `test_repository_report_migration.py`（6 tests） | `OK (5146 candidate paths)` / `OK` |
| `architecture-audit` | `tests/test_audit_import_graph.py` | `OK`（5 tests） |
| `lint` docs 一致性 | `scripts/check_platform_docs_consistency.py` | `[docs-consistency] OK` |
| `package-build` | `python -m build`（在 m4649 同树上执行） | `Successfully built tensorlbm-0.3.0.tar.gz / …-py3-none-any.whl`（exit 0） |
| `lint` mypy core gate | `mypy --follow-imports=silent d2q9 d3q19 solver utils` | 本机 1 条 `solver.py:437 no-any-return`；该行本单未触及，且 `main@866438ac` 的 CI `lint` 为 success（同命令同 flag）⇒ 本机 mypy/torch-stub 版本噪声，非分支回归 |

日志：`artifacts/m4684/ci_gates.log`（lint/layout/arch/docs 全绿原文）、`artifacts/m4684/pytest_fixed.log`（红→绿复跑）。

## 4. 局限（如实记）

* **全量 fast 档未在本机跑完**：本机 load average 619（多智能体共享），500 个测试文件的整档在两轮内无法跑完；CI 的 `test` 档结论以 GitHub 为准。等价性依据：本单只新增 1 个测试文件（4 条全绿）与 2 处测试断言修改，未触碰 `src/`，其余 `test` 档信号与基线 `main` 同源。
* 本机无法执行 `platform-test` 整档（401 s，被并发挤占）；该 job 在 `main@866438ac` 为 success，且本单未改 `app/`。
* GitHub Actions 严重积压（`main` 的 run `42145612061` 自 01:42 起 `queued` 逾 1 h），新 PR 的 CI 结论需排队才会出现——**重开后若仍显示 cancelled/queued，先看是否又是同秒整 run 取消（PR 被关）而非内容红**。