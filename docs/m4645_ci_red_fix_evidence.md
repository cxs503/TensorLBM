# CI 红 → 本地复现 → 修复 → 门禁复跑 证据（m4645 / 承接 PR #662）

> **状态（2026-11-06 复核）**：本文是 m4645 的一手复跑记录归档。所存档的
> 「红态计数」由当次命令输出直接抄录，**今天无法在 `main` 上重测**（main 已治愈，
> `ruff check` 现在按构造返回 0）。归档图 `assets/m4645/ci_gates.png` 的 (a) 面板
> 即为这张表的重绘，(b) 面板则取自 `main` 当前可复算的错误账本 —— 两部分的
> 数据来源在图注与 §6 分别写明，不做混同。

## 1. 远端红态的构成（GitHub API，run 41861110323 @ `a75534da`）

| job | 结论 | 失败 step | 归因 |
|---|---|---|---|
| `lint` | failure | 5 `Lint (ruff)` | 真红：分支落后 main，缺 m3379 的仓库级 ruff 治愈（`ruff check` 95 errors、`ruff format --check` 36 files） |
| `test (3.11)` | failure | 7 `Run tests with coverage` | 真红：同上，缺 m3379 对 `tests/test_bfl_suboff.py`（4 failed）等红测的治愈 |
| `test (3.12)` | failure | 7 `Run tests with coverage` | 同上 |
| `repository-layout` | failure | 1 `Set up job` | 纯基础设施：`actions/setup-python@v5` 解包期 tar chmod 冲突，未执行任何仓库命令 |
| `package-build` | failure | 1 `Set up job` | 同上（infra） |
| `platform-test` | failure | 1 `Set up job` | 同上（infra）；其仓库侧红因（`app/tests` 31 failed）亦已由 m3379 治愈 |
| `architecture-audit` | success | — | — |

关键事实：PR 分支 `agent/m3643`（`a75534da`）的父提交是 `9460e751`，落后 main 4 个提交，
其中 m3379 的 `9c354ca7` / `bfaf6d06` / `87539fff` 正是全仓 CI 治愈提交
（`docs/mission_summaries/M3379_CI_HEAL.md`：`ruff check` 95→0、`ruff format` 36→0、
`tests/test_bfl_suboff.py` 4 failed→0、`app/tests` 31 failed→0、repository-layout 7 违规→0）。
因此 lint / test / platform-test / repository-layout 的红是**分支陈旧**造成，不是 PR 自身代码问题。

## 2. 复现（合并 main 之前，分支原状）

```
$ ruff check src tests examples benchmarks          # -> Found 95 errors.
$ ruff format --check src tests examples benchmarks # -> 36 files would be reformatted
$ python tools/check_repository_layout.py           # -> root-file: AGENTS.md / M1_P2_SUMMARY.md / ... 7 条违规
```

与 m3379 记录的 pre-heal 数字逐项一致（95 / 36 / 7），确认分支态等于 main 治愈前的历史态。

## 3. 修复

唯一改动：把 `origin/main`（`866438ac`）合入分支（merge commit，无冲突、无手工解冲突）。
不改任何 PR 自有文件，不放宽阈值，不新增 layout 例外。

## 4. 修复后门禁复跑（本地，与 `.github/workflows/ci.yml` 命令逐字一致）

| CI step | 命令 | 结果 |
|---|---|---|
| Lint (ruff) | `ruff check src tests examples benchmarks` | `All checks passed!` |
| Format check | `ruff format --check src tests examples benchmarks` | `1453 files already formatted` |
| Type check (core) | `mypy d2q9.py d3q19.py solver.py utils.py --ignore-missing-imports` | 见 §5 局限（负载下未在 120 s shell 窗口内跑完） |
| Docs consistency | `python scripts/check_platform_docs_consistency.py` | pass（无输出=无违规） |
| test | `pytest -q tests/test_lbm_verification.py tests/test_bfl_suboff.py --timeout=900` | **63 passed, 1 xfailed in 29.29 s** |
| repository-layout | `python tests/test_repository_layout.py` / `test_repository_report_migration.py` / `check_repository_layout.py` | 12 OK / 6 OK / `Repository layout OK (5138 candidate paths)` |
| package-build | `python -m build --no-isolation` | `Successfully built tensorlbm-0.3.0.tar.gz and tensorlbm-0.3.0-py3-none-any.whl` |

`tests/test_lbm_verification.py` 是这条修复线的核心：四个 D3Q19 解析解案例此前被
列在 `tests/ci_quarantine.txt`（batch 2），修复解除了隔离，并修掉「`device="sdaa:0"`
写死」与「`device` 被 pytest 当 fixture」两个独立缺陷；复跑四个案例全部通过
（阈值仍为 1%，网格/步数未放宽），xfail 1 条来自其余历史隔离项（conftest 的非严格 xfail）。

## 5. 局限与未完成

- `lint` job 的 mypy core gate 与 `test` job 的**全量** `-m "not slow"` 套件未在本机跑完：
  控制面主机负载极高，全量套件预计 ~80 min，超出单次 shell 窗口；已用受影响最直接的模块
  （PR 自有 + m3379 治愈的 `test_bfl_suboff.py`）取得通过证据。
- 计算集群两节点（n315/n330）的 torch 缺 `libflexiblas.so.3`，无法承接 pytest 分流；
  该环境缺陷已记录，供集群运维参考。
- 远端 CI 复绿仍需 runner 接单：修红后的 push run 在派单时处于 `queued`（run 42141319739），
  故「远端 CI 绿」维度**未取得证据，如实记未完成**，不以本地绿冒充远端绿。

## 6. 归档图

`assets/m4645/ci_gates.png`，由 `scripts/plot_ci_gate_evidence.py` 生成：

![CI 门禁红态归档 + 解析解误差账本](assets/m4645/ci_gates.png)

*图：左 = 四个真红门禁的违规计数（红→绿，归档值）；右 = 四个解析解案例的实测相对误差 vs 1% 验收线（对数轴，取自 `convergence.json`）。*

```
python scripts/verify_lbm_analytic_convergence.py      # 先产出 (b) 面板的数据源
python scripts/plot_ci_gate_evidence.py
```

- **(a) 归档部分（不可重测）**：真红门禁违规计数 `ruff check` 95 → 0、`ruff format` 36 → 0、
  layout 7 → 0、`test_bfl_suboff` 4 failed → 0。数据源 = §1/§2 命令输出（当次运行），
  脚本内以带注释的 `DATA` 表存档；今天重测只能得到 0，因此这一半是**档案重绘**，不是 live 测量。
- **(b) 可复算部分**：`main` 上四个解析解案例的实测相对误差 vs 1% 验收线（对数轴），
  数据源 = `artifacts/lbm_analytic_verification/convergence.json`（由
  `scripts/verify_lbm_analytic_convergence.py` 每跑一次重写，测试
  `tests/test_lbm_verification_evidence.py` 会核对其与 1% 阈值一致）。

## 7. 这条线在 `main` 上的落点（2026-11-06 复核）

| 部件 | 落点 |
|---|---|
| `tests/test_lbm_verification.py` 改写为设备无关 assert 用例 | `5cd12e87`（已在 main） |
| `tests/ci_quarantine.txt` 4 个 node ID 退役 | `5cd12e87`（已在 main） |
| `docs/lbm_analytic_verification_r1.md` + `scripts/verify_lbm_analytic_convergence.py` | PR #717（已在 main） |
| **本文档 + 归档图 + `scripts/plot_ci_gate_evidence.py`** | 本 PR |
| **`artifacts/lbm_analytic_verification/*`（图/账本/表）** | 本 PR —— 此前 `docs/lbm_analytic_verification_r1.md` 的 `convergence.png` 链接指向一个**从未提交**的路径，新克隆里是死链 |
| **`tests/test_lbm_verification_evidence.py`** | 本 PR —— 证据链回归（图/链接/账本与阈值一致），死链会被测试抓住 |