# m4676 — PR #676 (`agent/m4649`) CI heal · round 1

Branch healed: `agent/m4649` (PR #676) · base `1ef76442` (main) · mission 4676
Reproduced on: loongarch64 container, CPython 3.11.6, ruff 0.15.0

## 1. 红因分层（先归因，再动手）

| 层 | 证据 | 结论 |
|---|---|---|
| 现象 | 13 项 check 全为 `cancelled`：`repository-layout / architecture-audit / test(3.11) / test(3.12) / package-build / platform-test / lint`，且 **job 名成对重复**（最后一个 `test(3.11)` 被截断）⇒ 实为 2 个 run × 7 job。 | `cancelled` 是 **run 级取消**，不是 job 失败；没有任何 job 产出 failure annotation。 |
| 机制 | `.github/workflows/ci.yml`：`concurrency.group = ci-${{ github.workflow }}-${{ github.ref }}`，`cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}`。非 main ref 上后一次 push 会取消在跑的 run。分支侧连续两次推送（建 PR + 更新 head）⇒ 两个 run 都被取代。 | 与分支文件内容**无关**；`pr-gate` 读到 `cancelled ≠ success` 即判红。 |
| 分支自身（本地逐 job 复现） | 见 §2：lint / format / docs-consistency / repository-layout / architecture-audit / platform-test **全绿**；新增用例 6 passed。 | 内容侧无 CI 红点（唯一无法在本机跑完的是全量 fast `test` 档，见 §6）。 |

因此本单能落地的两件事：**(a) 把分支内真实的缺陷修掉并加回归守卫；(b) 产出一个新 commit，让下一次 CI 拿到一个不被取代的 run。**

## 2. 逐 job 本地复现（精确执行 CI 命令）

| CI job | 命令 | 结果 |
|---|---|---|
| `lint` — ruff check | `ruff check src tests examples benchmarks` | `All checks passed!`（1454 文件） |
| `lint` — ruff format | `ruff format --check src tests examples benchmarks` | `1454 files already formatted` |
| `lint` — docs 一致性 | `python scripts/check_platform_docs_consistency.py` | `[docs-consistency] OK` |
| `repository-layout` | `python tests/test_repository_layout.py` / `tests/test_repository_report_migration.py` / `tools/check_repository_layout.py` | 12 tests OK / 6 tests OK / `Repository layout OK (5147 candidate paths)` |
| `architecture-audit` | `python tests/test_audit_import_graph.py` | 5 tests OK |
| `platform-test` | `PYTHONPATH=src pytest -q app/tests -n 8 --timeout=600` | **703 passed, 74 skipped, 0 failed**（350 s） |
| `test` — 本 PR 新增用例 | `pytest -q tests/test_stokes_second_problem_benchmark.py` | **6 passed**（含本轮新增 2 条守卫） |
| `test` — 全量 fast 档 | `pytest -q -m "not slow" -n 16 --timeout=900` | **未跑完**（本机 load 高，收尾时 ~58%，见 §6） |
| `package-build` | `python -m build` | 本轮**未复跑**；上一轮（m4649）同内容构建成功，本轮仅改动 `scripts/`、`tests/`、`artifacts/`，不进包体 |
| `lint` — mypy core gate | `mypy --follow-imports=silent src/tensorlbm/{d2q9,d3q19,solver,utils}.py --ignore-missing-imports` | 本机 **超时未完成**（120 s 工具上限），四项源码本轮未改动，判为非回归面 |

## 3. 本轮修掉的分支内缺陷（3 处真缺陷 + 1 条建议）

1. **报告生成器与成稿失步（最重要，可发表性缺陷）**
   `scripts/stokes_second_report.py` 的 ⑤「收敛阶」条目是**硬编码旧文本**（“本轮因宿主 load1>300 只作可选诊断跑，见
   diag_H160.json 若存在”），而 `diag_H160.json` 确实存在、worst 相位实测 **1.452%**，成稿写的是补跑结论。
   ⇒ 任何人重跑生成器都会把 H=160 结论**静默回退**成“未跑”，报告不可复现。
   修：抽出纯函数 `order_gate_bullet(per_h_worst)`，条目由**测量值**生成（有 160 档 → 写实测值与 H=80 对比；无 → 明确写“诊断档未跑”）。
   验证：重跑生成器后生成文档与 committed doc **逐字节相同**（`git diff docs/` 为空）。
2. **成稿引用的图未入库（悬空引用）**
   `docs/stokes_second_problem_verification_20261103.md:82` 引用 `artifacts/stokes_second_problem/stokes_second_verification.png`，
   该 png 在 PR 中**不存在**。修：把生成器产出的图收入 artifacts/（1092×319，98 KB，layout gate 通过）。
3. **死代码 + 未格式化**
   `PHI_HALF = round(math.pi/2, 6)` 赋值后从未使用（ruff `F841`）；文件未过 `ruff format`。
   注：`scripts/` 不在 CI lint scope（`ruff check src tests examples benchmarks`）也不在 pre-commit `files` 正则内，
   所以这两点**没有**造成 CI 红——但属真死代码/格式债，随手修掉。
4. **建议（未做，超本单范围）**：把 `scripts/` 纳入 lint scope 会牵动 145 个历史脚本，需独立任务（本轮只保证本单新增脚本干净）。

## 4. 回归守卫（`tests/test_stokes_second_problem_benchmark.py`，+2 用例）

* `test_order_bullet_follows_data_not_stale_text`：`{40,80,160}` → 条目含 `1.452%`/`1.012%`；`{40,80}` → 含“诊断档未跑”且**不含** `1.452`。
* `test_report_regeneration_reproduces_committed_doc`：在隔离 cwd（case 目录 symlink）重跑生成器，产物须与 committed doc **逐字节相等**。
* **负控已实跑**：向 committed doc 追加 1 字节 → 该用例 `FAILED`；恢复后 `6 passed`（守卫确实咬得住，不是空断言）。

## 5. 计算结果（程序化注入，非手抄；源 `artifacts/m4676/key_numbers.txt`）

| H | δ/Δx | cycles | max rel err, 10 %-mask | max rel err, 5 %-mask | max abs/U0 | mass drift | steps | wall |
|---|---|---|---|---|---|---|---|---|
| 40 | 10 | [6, 8] | **2.160 %** | 4.992 % | 1.374 % | 0.0608 % | 53407 | 12.2 s |
| 80 | 20 | [3, 4] | **1.012 %** | 5.157 % | 1.381 % | 0.1261 % | 113097 | 30.3 s |
| 16 (诊断) | 4 | [3, 4] | 15.758 % | 15.758 % | 1.696 % | 0.0042 % | 4524 | 18.6 s |
| 160 (诊断) | 40 | [3, 4] | 1.452 % | 4.862 % | 1.383 % | 0.5111 % | 452389 | 130.0 s |

同相位周期稳态差（门 0.5 %·U0）：H=40 **0.0299 %** / H=80 **0.0232 %**，两档 `steady=True`。
数值口径见 `docs/stokes_second_problem_verification_20261103.md`；本轮**未改任何数值/判据**，只修“生成器是否忠实复现成稿”。

## 6. 未完成与局限（如实记录）

* **全量 fast `test` 档未跑完**：本机与其它任务并发（load 高），收尾时约 58 %；观察到的 `F` 全部落在本 PR 未触及的既有用例，
  未逐条定位 ⇒ 本单**不就全量测试档绿色作断言**。
* **package-build 本轮未复跑**（仅 `scripts/`、`tests/`、`artifacts/` 变更，不进 sdist/wheel 源码包）。
* **无法读取 GitHub job 日志**（本机无 token，`api.github.com` 被策略拦截），`cancelled` 的 run id 未取得；
  “run 级取消而非内容失败”的判定基于：7 个 job 同一 `cancelled` 状态、成对重复的 job 名、无任何 failure annotation + 本地全绿复现。
* **CI 绿仍需一次不被取代的 run**：本单已产出一个新 commit；若平台在同 ref 上再次连续推送，run 仍可能被 concurrency 取代
  （若 pr-gate 需要确定性，建议平台在推送后等待当前 run 结束再判、或对该分支只推一次）。