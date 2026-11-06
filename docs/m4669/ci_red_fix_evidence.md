# PR #669 CI 修红报告（第 2 轮）— TensorLBM / agent/m4641 → agent/m4669

- 日期：2026-11-06
- 执行单：m4669
- 目标：让 PR #669（分支 `agent/m4641`，head `b9da075a`）CI 转绿
- 基线：`866438ac`（main）→ `b9da075a`（m4641 交付提交）

## 1. 第 1 轮遗留结论的更新

第 1 轮（`docs/m4641/ci_red_fix_evidence.md`）判定 CI 红为「基础设施层假红」：
runner 在 `Set up job` 展开 `actions/setup-python` bundle 时 tar/chmod 失败。
该轮把 6 个 job 的 `actions/setup-python@v5` 换成 runner-local venv 后，
**job 能起来了**——本轮 run `42143763252` 里 `Set up job` / checkout /
Provision Python 全部 success，`test (3.12)` 真的跑完了 24 分 25 秒。
因此第 2 轮的红是**代码层（用例层）真实红**，不再是基础设施问题。

## 2. 红在哪（CI 日志证据）

run `42143763252`（push, head `b9da075a`），job `127650153616` = `test (3.12)`，
step 7 `Run tests with coverage` = failure：

```
FAILED tests/test_control_volume_force.py::test_curved_moving_slip_impulse_is_invariant_across_nested_control_volumes
       - TypeError: iteration over a 0-d tensor
FAILED tests/test_regress_wall_multi_gpu.py::TestBugIdentification::test_wss_from_fneq_3d_import_error_equilibrium_vs_equilibrium3d
       - Failed: DID NOT RAISE ImportError
FAILED tests/test_regress_wall_multi_gpu.py::TestWallShearBugs::test_wss_from_fneq_3d_broken_import
       - Failed: DID NOT RAISE ImportError
3 failed, 6487 passed, 177 skipped, 14 deselected, 116 xfailed, 6 xpassed,
33 warnings, 81 subtests passed in 1465.96s
```

其余 job（lint / platform-test / repository-layout / architecture-audit /
package-build / test 3.11）在 run `42143763252` 里是 `cancelled`
（`concurrency: ci-<workflow>-<ref>` + `cancel-in-progress`，同一 ref 的新 push
`42144261001` 顶掉了在跑的一批），**不是失败**：它们没有跑完，不能据此判红。

同一提交在 main 上也是红的：run `42008133677`（main `866438ac`, 2026-11-05）
`test (3.11)` 与 `test (3.12)` 均 failure，其余 6 个 job success。
说明这 3 个失败是**仓库既有欠账**，与本 PR 的数值改动（Spalding 壁面矩平衡）
无关，但 PR 要过门必须清偿。

## 3. 三个失败的根因与修法

### 3.1 `test_curved_moving_slip_impulse_is_invariant_across_nested_control_volumes`

```
tests/test_control_volume_force.py:220
    assert tight_force[0].item() == pytest.approx(bfl_force[0], abs=1e-8)
...
_pytest/approx.py:1037: in _contains_decimal
    return any(isinstance(value, Decimal) for value in values)
torch/_tensor.py:1217: TypeError: iteration over a 0-d tensor
```

`pytest.approx()` 的实参是 0 维 `torch.Tensor`；pytest ≥8 在 `_contains_decimal`
里对实参做 `any(... for value in values)`，对 0-d tensor 迭代即抛 TypeError。
（左值已经 `.item()`，只有右侧漏了。）

修法（1 行）：`pytest.approx(float(bfl_force[0]), abs=1e-8)`，与同文件
`:79` 已有写法 `pytest.approx(float(expected), ...)` 一致。

### 3.2 / 3.3 两个「缺陷确认测试」：`..._import_error_equilibrium_vs_equilibrium3d` / `..._broken_import`

这两个用例是**把缺陷本身写成了断言**的「缺陷确认测试」：

```python
with pytest.raises(ImportError, match="cannot import name 'equilibrium'"):
    wss_from_fneq_3d(f, rho, ux, uy, uz, tau=0.8, mask=mask)
```

而现网代码已经修好了这个缺陷——`src/tensorlbm/boundary/wall_shear.py:186`
导入的是 `equilibrium3d`（`src/tensorlbm/wall_shear.py` 只是把
`tensorlbm.boundary.wall_shear` 别名成同一模块对象）。缺陷消失后，
「必须抛 ImportError」的断言必然失败（`DID NOT RAISE ImportError`）。

修法：改写为**正控回归守护**（保留原意图：防止 3-D WSS 再次带病上岗）：

- `test_wss_from_fneq_3d_import_resolves_equilibrium3d`：叠加确定性
  `f += 1e-4 * c_x c_y` 的 xy 剪切非平衡态，要求返回 `(nz,ny,nx)` 有限场、
  固体格点被 mask 清零、流体格点 `wss > 0`（若 f_neq 路径死掉会返回全零）。
- `test_wss_from_fneq_3d_is_callable`：纯平衡态 `f_neq = 0`，要求可调用并
  返回全零有限场（数值上也验证了 `f - f_eq` 的定义）。
- 同步更新 `docs/reports/REGRESSION_REPORT_wall_multi_gpu.md` 的验证用例名与
  修复状态说明。

不改用 `tests/ci_quarantine.txt`（non-strict xfail）：那是给「未修的既有失败」
留口的机制，这 3 个用例的失败原因已经被定位并修好，挂 quarantine 会永久掩盖
真回归。

## 4. 本地复现 / 修复证据（loongarch64, Python 3.11.6, torch 2.7.1）

修前（HEAD = `b9da075a`，未改动）：

```
$ python -m pytest -q tests/test_regress_wall_multi_gpu.py::TestBugIdentification::test_wss_from_fneq_3d_import_error_equilibrium_vs_equilibrium3d \
    tests/test_regress_wall_multi_gpu.py::TestWallShearBugs::test_wss_from_fneq_3d_broken_import \
    tests/test_control_volume_force.py::test_curved_moving_slip_impulse_is_invariant_across_nested_control_volumes
3 failed, 1 warning in 49.62s
```

修后（同三条命令，整文件跑，覆盖改动面全部 33 个用例）：

```
$ python -m pytest -q tests/test_control_volume_force.py tests/test_regress_wall_multi_gpu.py
20 passed, 13 xfailed, 1 warning in 68.67s
```

本 PR 自身的验收面（Spalding 壁面矩平衡）：

```
$ python -m pytest -q tests/test_spalding_wall_model.py tests/test_spalding_wall_moment_balance.py tests/test_wall_resolved_channel3d.py
32 passed, 1 warning in 10.09s
```

静态门（与 CI lint / repository-layout / architecture-audit 对应）：

```
$ ruff format --check src tests examples benchmarks     -> 1454 files already formatted
$ ruff check src tests examples benchmarks              -> All checks passed!
$ python scripts/check_platform_docs_consistency.py     -> [docs-consistency] OK
$ python tests/test_repository_layout.py                -> OK
$ python tests/test_repository_report_migration.py      -> OK
$ python tools/check_repository_layout.py               -> Repository layout OK (5139 candidate paths; ...)
$ python tests/test_audit_import_graph.py               -> OK
$ python -m pytest -q --collect-only tests               -> 6822 tests collected
```

## 5. 注意点 / 局限（如实声明）

1. **未能在本沙箱重跑整仓 6822 例**：本机单次 shell 上限 120 s（进程组被收割），
   load average ≈ 680（多智能体共享 128 核），后台/pytest-xdist 化都会互相拖死。
   替代证据是：CI 日志在父提交 `b9da075a` 上给出了**完整失败清单**（只有这 3 例），
   而本轮 diff 只动这 2 个测试文件 + 1 篇文档，不可能影响其余 6487 例；受影响的
   两个测试文件已整文件跑绿（第 4 节）。
2. **本机只有 Python 3.11.6**，3.12 无法本地复跑；CI 日志显示 runner 上确有
   `python3.12`（venv 路径 `_temp/ci-venv/lib/python3.12/`），而三处失败都与解释器
   版本无关（一处是 pytest.approx 对 0-d tensor 的迭代，两处是缺陷已修）。
3. **`cancelled` ≠ `failure`**：本轮除 `test (3.12)` 外其余 job 是被 concurrency
   顶掉的，不得当成红；复活它们只需一次新 push。
4. 本轮不改 `src/`：三个失败都在用例/文档层，库代码无误，故改动面压到 3 个文件。
5. 远端 CI 实绿与合并进 main（pr-gate 记账）需等新 push 触发新 run 后由 CI 复验，
   本沙箱无法代跑 GitHub Actions。
## 6. 数值验证（改写后的回归守护不是空断言）

改写后的用例不再是「抛不抛异常」的空壳断言，它检查的是一条有闭式解的应力路径。
状态取 `f = f_eq(rho=1, u=0) + A c_x c_y`（A = 1e-4，D3Q19 面内四对角），则

```
f_neq = A c_x c_y
sigma_xy = -(1 - 1/(2 tau)) * A * sum_q c_x^2 c_y^2 ,   sum_q c_x^2 c_y^2 = 4 (D3Q19)
wss = |sigma_xy|   (sigma_xz = sigma_yz = 0)
```

`python docs/m4669/make_figures.py` 逐 τ 对拍（程序化输出，未手抄）：

| τ (-) | 数值 σ_xy | 闭式 σ_xy | 相对误差 |
|---|---|---|---|
| 0.55 | 3.63636364e-05 | 3.63636364e-05 | 6.15e-15 |
| 0.60 | 6.66666667e-05 | 6.66666667e-05 | 6.10e-15 |
| 0.70 | 1.14285714e-04 | 1.14285714e-04 | 6.05e-15 |
| 0.80 | 1.50000000e-04 | 1.50000000e-04 | 6.14e-15 |
| 1.00 | 2.00000000e-04 | 2.00000000e-04 | 6.10e-15 |
| 1.20 | 2.33333333e-04 | 2.33333333e-04 | 6.04e-15 |
| 1.50 | 2.66666667e-04 | 2.66666667e-04 | 6.10e-15 |

- 关键量：无量纲相对误差（σ_xy 数值 vs 闭式）最大 6.15e-15，**是否 <3%：是**。
- 场量云图 + 误差曲线：`docs/m4669/fig_wss_regression.png`
  （(a) 16³ 域、球形固体 `sphere_mask` r=3.5 的中间切面 τ_w，固体格点严格为 0
  （`solid max wss = 0.0`），流体格点 `1.500e-04`；(b) τ∈[0.55,1.5] 的相对误差曲线，
  远低于 3% 验收带）。
- **适用范围**：这是实现层（调用路径 + 应力求和 + mask 语义）的**验证**，
  不是物理基准的对标；合成扰动有闭式解，因此误差可判为机器精度量级。
  若要判物理精度，需 Couette/槽道算例对标实验或 DNS 参考值，不在本单范围。
