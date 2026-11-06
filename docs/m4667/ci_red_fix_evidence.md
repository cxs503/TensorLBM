# PR #669 CI 修红证据（TensorLBM / branch `agent/m4641` / 单 m4667）

日期：2026-11-06　工作树：`/root/tp-agent/work/m4667`（分支 `agent/m4667`，基线 = PR head `b9da075a`）

## 1. 红在哪（远端事实，非推测）

PR #669（`agent/m4641` → `main`，head `b9da075a`）的 check-runs 里唯一 completed 的失败项：

| 项 | 值 |
|---|---|
| workflow run | 42143763252（push）/ 42144261001（pull_request） |
| 失败 job | `127650153616` **test (3.12)**，失败步骤 = `Run tests with coverage`（step 7） |
| 汇总（job 日志实测） | `3 failed, 6487 passed, 177 skipped, 14 deselected, 116 xfailed, 6 xpassed in 1465.96s` |

三条红测（原文摘录，来自 job 日志）：

```
FAILED tests/test_control_volume_force.py::test_curved_moving_slip_impulse_is_invariant_across_nested_control_volumes
       - TypeError: iteration over a 0-d tensor
FAILED tests/test_regress_wall_multi_gpu.py::TestBugIdentification::test_wss_from_fneq_3d_import_error_equilibrium_vs_equilibrium3d
       - Failed: DID NOT RAISE ImportError
FAILED tests/test_regress_wall_multi_gpu.py::TestWallShearBugs::test_wss_from_fneq_3d_broken_import
       - Failed: DID NOT RAISE ImportError
```

`test (3.11)` 这一路 job 从未开始执行（runner 队列自 2026-11-06T01:04Z 起整体停摆，20+ run 停留在
`queued`），所以「只有 3.12 红」不等于「3.11 绿」——三条失败都与解释器无关（见第 2 节归因），
因此同一修复同时覆盖 3.11 矩阵行。

## 2. 根因（两条独立原因，都在测试侧，不在被测物理内核）

### 2.1 `pytest.approx()` 的 expected 不能是 0-d tensor

`tests/test_control_volume_force.py:220` 原写法：

```python
assert tight_force[0].item() == pytest.approx(bfl_force[0], abs=1e-8)
```

`bfl_force[0]` 是 **0-d** `torch.Tensor`。pytest ≥ 8.4 在
`ApproxScalar._warn_on_inexact_decimal_tolerance` → `_contains_decimal(expected)` 里会**迭代**
expected 去检测 `Decimal`，于是落到 `torch.Tensor.__iter__` 的
`if self.dim() == 0: raise TypeError("iteration over a 0-d tensor")`。
这是纯测试写法问题（两侧都已是 float 语义），与被测的嵌套控制体力不变性无关。

### 2.2 两条「bug-alive」回归测试在缺陷被修好后必然变红

`tensorlbm/boundary/wall_shear.py::wss_from_fneq_3d` 历史上误写
`from d3q19 import equilibrium`（D3Q19 实际导出的是 `equilibrium3d`），导致该函数一调用即
`ImportError`（仓库自带报告 `docs/reports/REGRESSION_REPORT_wall_multi_gpu.md` §1.1 记录了该缺陷）。
缺陷随后已被修复（当前源码 `from tensorlbm.d3q19 import ... equilibrium3d`），但两条测试仍以
`pytest.raises(ImportError)` 断言「缺陷存在」，于是**修复本身把 CI 顶红**——典型的陈旧 bug-alive 测试。

## 3. 改了什么（3 个文件，其中 2 个测试文件 + 1 处文档引用）

| 文件 | 改动 |
|---|---|
| `tests/test_control_volume_force.py` | `pytest.approx(bfl_force[0])` → `pytest.approx(bfl_force[0].item())`，并加注释说明 pytest ≥ 8.4 的行为（语义不变） |
| `tests/test_regress_wall_multi_gpu.py` | 两条 bug-alive 测试改写为 **bug-fixed 回归守卫**：`test_wss_from_fneq_3d_import_error_equilibrium_vs_equilibrium3d` → `test_wss_from_fneq_3d_uses_equilibrium3d_not_equilibrium`；`test_wss_from_fneq_3d_broken_import` → `test_wss_from_fneq_3d_functional`。新断言：调用不抛 `ImportError`、返回形状 `(nz,ny,nx)`、`isfinite`、固体节点被 mask 置零、且对**非平衡扰动输入**（`f_eq + 0.01·randn`, `manual_seed(0/1)`）产生非零 WSS（原先用纯 `f_eq`，`f_neq≡0`，即便函数能跑也只会得到全零场，测不出应力公式） |
| `docs/reports/REGRESSION_REPORT_wall_multi_gpu.md` | 更新被改名的测试引用，并注明缺陷已修复 / 测试已转为 bug-fixed 守卫 |

未触碰：`src/`（`spalding_wall_model.py` 保持 PR 原样）、CI 工作流、其它任何测试。

**为什么这三条红与本 PR 内容无关（前置事实）**：
`git diff --stat 866438ac b9da075a -- tests/test_control_volume_force.py tests/test_regress_wall_multi_gpu.py src/tensorlbm/boundary/wall_shear.py`
为空——PR 分支完全没碰这些文件，红是 main 继承下来的存量欠账，本单在 PR 分支上把它清掉以满足 pr-gate。

## 4. 本地复现 → 修复证据（全部在本 worktree 内执行；loongarch64 / Python 3.11.6 / pytest 9.2.0）

**修前（复现红）**：

```
$ python3 -m pytest -q \
    tests/test_control_volume_force.py::test_curved_moving_slip_impulse_is_invariant_across_nested_control_volumes \
    tests/test_regress_wall_multi_gpu.py::TestBugIdentification::test_wss_from_fneq_3d_import_error_equilibrium_vs_equilibrium3d \
    tests/test_regress_wall_multi_gpu.py::TestWallShearBugs::test_wss_from_fneq_3d_broken_import
3 failed, 1 warning in 21.00s
FAILED tests/test_control_volume_force.py::test_curved_moving_slip_impulse_is_invariant_across_nested_control_volumes
FAILED tests/test_regress_wall_multi_gpu.py::TestBugIdentification::test_wss_from_fneq_3d_import_error_equilibrium_vs_equilibrium3d
FAILED tests/test_regress_wall_multi_gpu.py::TestWallShearBugs::test_wss_from_fneq_3d_broken_import
```

与 CI 3.12 的三条失败**逐条同名、同错误类**（0-d tensor TypeError / DID NOT RAISE ImportError）。

**修后**：

```
$ python3 -m pytest -q tests/test_control_volume_force.py tests/test_regress_wall_multi_gpu.py tests/test_suboff_postprocess_runtime.py
26 passed, 13 xfailed, 1 warning in 25.69s

$ python3 -m pytest -q -k "control_volume or wall_shear or wss" tests/        # 改动面邻域全扫
71 passed, 1 skipped, 6777 deselected, 2 warnings in 93.32s

$ ruff check src tests examples benchmarks            → All checks passed!
$ ruff format --check src tests examples benchmarks   → 1454 files already formatted
$ python3 scripts/check_platform_docs_consistency.py  → [docs-consistency] OK
$ python3 tools/check_repository_layout.py            → Repository layout OK (5139 candidate paths)
```

图：`docs/m4667/fig_ci_red_fix.png`（左：修前 CI 通过/失败计数 vs 修后本地计数，对数轴；右：三条红测逐条 before/after 状态）。
机器可读数据：`docs/m4667/before_after.json`（由脚本从 CI 日志与本轮实测注入，非手抄）。

## 5. 注意点 / 局限（如实声明）

1. **远端绿状态本单未能观测**：GitHub Actions runner 自 2026-11-06T01:04Z 起队列停摆，`b9da075a` 的 7 个 job 中只有
   `repository-layout`(success) 与 `test (3.12)`(failure) 真正执行，其余（含 `test (3.11)`）一直 `queued`。
   因此「PR #669 CI 绿 + 合并进 main」在 runner 恢复前无法由我确认；本单交付的是**红测的本地复现 + 修复 + 校验**。
2. **未在本机跑完整 6500 例套件**：宿主机 load average ≈438（128 核，其它智能体并发跑测试），单进程全量套件预计 >2h 且结果会被
   资源竞争污染，故以「CI 已给出的全量基线（6487 passed / 3 failed）+ 三条红的定向复现/修复 + 改动面邻域 71 例」为证据链。
3. 移动/重命名了 2 条测试函数名：仓库内仅有 `docs/reports/REGRESSION_REPORT_wall_multi_gpu.md` 引用旧名，已同步更新；
   若外部看板按 pytest node id 引用旧名，需相应更新。
4. `TestBugIdentification.test_wss_from_fneq_3d_W_to_device_discarded`（断言 `_W3D.to(device)` 代码异味仍存在）本轮**未动**：
   它是「smell-alive」测试，一旦有人清理该行就会变红，属同类隐患，留作后续清理（不在本单范围）。
5. 三条红均与解释器无关，故同一修复同时覆盖 `test (3.11)` 与 `test (3.12)` 两个矩阵行；但 3.12 行在 loong64 runner 上
   依赖 runner 是否真的存在 `python3.12`（job 日志显示 `ci-venv/lib/python3.12/...`，即本次 3.12 行确实用 3.12 跑）。