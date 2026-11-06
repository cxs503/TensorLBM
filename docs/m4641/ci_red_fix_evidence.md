# PR #658 CI 修红报告（TensorLBM / branch agent/m3600）

日期：2026-11-06　执行单：m4641　目标：让 PR #658（分支 `agent/m3600`，head `ff30fe68`）CI 转绿。

## 1. 红在哪：两层原因

### 1.1 代码层（真实红）——lint / format
`ruff format --check src tests examples benchmarks` 在 `agent/m3600` 上失败：

```
Would reformat: tests/test_spalding_wall_moment_balance.py
1 file would be reformatted, 1453 files already formatted
```

同一提交的 `scripts/verify_spalding_wall_moments.py` 还有 2 处 `E402`
（`sys.path.insert` 之后 import torch / tensorlbm）。该文件虽不在 CI 的
`ruff check src tests examples benchmarks` scope 内，但会在推送前置 ruff 门
（tp_pusher `_ruff_gate`，作用于改动面 `*.py`）被判红。

### 1.2 基础设施层（假红）——所有 7 个 job 死在 "Set up job"
run `41840767704`（push, 2026-11-04T13:43Z，head ff30fe68）的 7 个 job
（lint / test 3.11 / test 3.12 / platform-test / repository-layout /
architecture-audit / package-build）**全部**只有一步失败：`Set up job`。
job 日志（API `/actions/jobs/{id}/logs`）里的错误是：

```
Download action repository 'actions/setup-python@v5' (SHA:a26af69b...)
##[error]Can't use 'tar -xzf' extract archive file: .../setup-python-a26af69be9.../__tests__/data/inner/poetry.lock:
Cannot change mode to rwxr-xr-x: Operation not permitted
Action being checked out: actions/setup-python@v5. return code: 2.
```

即自建 runner `node099-loong64-tl`（loong64，`runs-on: ubuntu-latest` 由它承接）
在展开 `actions/setup-python` 的 action bundle 时 chmod 失败，job 在任何仓库命令
执行之前就已经失败——**与 PR 代码无关**。

旁证（同仓 main 的 run `41995371431`，2026-11-05）：只有并发的两个 `test (3.1x)`
job 因同一 tar/chmod 错误失败，串行执行的 lint / repository-layout /
architecture-audit / package-build / platform-test 全部 success。
说明该失败发生在多个 job 并发下载并展开同一个 action bundle 时（共享
`_work/_actions` 缓存，mode 冲突时 tar 无法 chmod），不是代码问题。

## 2. 修了什么（3 个文件，最小改动）

| 文件 | 改动 | 针对 |
|---|---|---|
| `tests/test_spalding_wall_moment_balance.py` | `ruff format` 规范化（纯格式，无语义改动） | lint / format 真实红 |
| `scripts/verify_spalding_wall_moments.py` | `import torch` / `import tensorlbm...` 加 `# noqa: E402` + `ruff format` | 推送前 ruff 门 |
| `.github/workflows/ci.yml` | 6 个 job 的 `actions/setup-python@v5` 换成 "Provision Python (runner-local venv)"：优先用 runner 上已存在的 `python<version>`，缺失则回退 `python3` 并 `::warning::`；venv 建在 `$RUNNER_TEMP`（仓库外，避免污染 repository-layout 扫描面） | 1.2 的假红 |

`src/tensorlbm/spalding_wall_model.py`（PR 原有内容：D3Q19/D3Q27 权重归一化，
去掉 float32 求和偏差造成的系统性质量源）与 `docs/spalding_wall_moment_balance_2026-11-03.md`
未改动，保持 PR 原样。

## 3. 本地复现与修复证据（均在本 worktree 内执行，loongarch64 / Python 3.11.6）

```
$ ruff format --check src tests examples benchmarks      # 修前: tests/test_spalding_wall_moment_balance.py would reformat
                                                          # 修后:
1454 files already formatted
$ ruff check src tests examples benchmarks
All checks passed!
$ ruff check scripts/verify_spalding_wall_moments.py
All checks passed!
$ python -m pytest -q tests/test_spalding_wall_model.py tests/test_spalding_wall_moment_balance.py
27 passed, 1 warning in 7.05s
$ python -m pytest -q tests/test_wall_resolved_channel3d.py      # 同模块下游用户
5 passed, 1 warning in 10.21s
$ python tools/check_repository_layout.py
Repository layout OK (5140 candidate paths; historical exceptions retained).
$ python tests/test_repository_layout.py && python tests/test_repository_report_migration.py
OK (12 tests) / OK (6 tests)
$ python tests/test_audit_import_graph.py
OK (5 tests)
$ python tools/audit_import_graph.py --root src/tensorlbm=tensorlbm ...
audit_ok (import-audit.json / .md 生成成功)
$ python -m build --sdist --outdir /tmp/dist
Successfully built tensorlbm-0.3.0.tar.gz
```

CI 新 provisioning 步骤的等价本地演练（`REQUESTED_PYTHON=3.12`，本机只有 3.11）：

```
$ base=python3.12 → 缺失 → ::warning:: 回退 python3
Python 3.11.6 (main, May  9 2024) [GCC 12.3.0 20230508] Anolis OS 12.3.0-2
$ python -m venv --system-site-packages $RUNNER_TEMP/ci-venv   → 成功
$ .../bin/python -m pip install --upgrade pip                   → 成功 (pip 26.3)
```

## 4. 注意点 / 局限（如实声明）

1. **CI 队列卡死**：2026-11-06T01:04Z 起该仓 20+ 个 run 一直 `queued`，
   runner `node099-loong64-tl` 显示 `online, busy=false` 却不接单（main 11-05 10:08
   的 run 还能跑）。因此本轮**无法拿到远端绿状态**，"CI 绿" 只能等 runner 恢复后
   由新 run 验证；本地复现与修复证据见第 3 节。
2. **3.12 矩阵在 loong64 runner 上不存在**（仅有 `/usr/bin/python3.11`）。
   新步骤会 `::warning::` 回退到 3.11 并在日志里打印真实解释器版本，矩阵行因此
   目前退化为 3.11 双跑；真正的 3.12 覆盖需要 runner 侧安装 3.12。
3. **沙箱里 `pip install -e ".[dev]"` 失败**：镜像只给 `hypothesis` 的 sdist，
   构建需 maturin/cargo（`cargo metadata` 失败）。runner 容器里同一命令在
   11-05 的 lint job 上是成功的，故判断为沙箱镜像差异，未改 CI 安装命令；
   若 runner 复现，应把 `hypothesis` 固定到带 wheel 的版本。
4. `app/tests`（platform-test）在本沙箱单次 120 s shell 上限内未跑完，PR 未触碰
   `app/`，其红同样来自 1.2 的 job setup 失败。

## 5. 判定

- 代码层红（ruff format / E402）：**已修复并本地复现→验证转绿**（第 3 节）。
- 基础设施层红（7 job "Set up job"）：**已用 repo 侧改动规避**（去掉
  setup-python action 依赖），但远端绿需 runner 队列恢复后由 CI 复验。
- 未完成项：远端 CI 实绿 / 合并进 main（pr-gate 记账）——受 runner 队列阻塞。