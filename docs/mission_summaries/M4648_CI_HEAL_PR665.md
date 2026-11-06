# PR #665（分支 `agent/m3697`）CI 复红修复 —— round 1（m4648）

## 1 案例与红因定位

PR #665 head `bc543d8f` 的基点是旧 main `9460e751`；当前 main 已到 `866438ac`
（含 `87539fff` layout 迁移、`bfaf6d06` mypy core gate、`9c354ca7` m3379 成果）。
CI 作业日志证实 `git checkout -B agent/m3697 refs/remotes/origin/agent/m3697`
（push 事件测**分支自身内容**，非 merge ref），于是 main 上一轮的"清零"成果被整段
回退，红因全部可归因：

| 作业 | 证据 | 归结 |
|---|---|---|
| lint | `Found 96 errors`（F401/E722/I001/F841，多在 `benchmarks/verified/suboff_sail_resistance/*`、`tests/test_torch_fallback.py`，main 已清理）+ 分支新文件 `tests/test_thermal_radiation_analytic.py` I001、`src/tensorlbm/thermal_radiation.py`/同测试 ruff-format 不合规 | 分支落后 + 分支新文件格式 |
| repository-layout | 7 条 root-file 违规（AGENTS.md、M1_P2_SUMMARY.md、MS2_SUMMARY.md、SDAA_ROUND3_SUMMARY.md、bench_sdaa_baseline.py、debug_trace.py、validate_m2m3.py） | 未含 main 的 87539fff |
| test (3.11/3.12) | 7 failed：`tests/test_bfl_suboff.py`×4（main 已修）+ `test_control_volume_force.py`×1 + `test_regress_wall_multi_gpu.py`×2 | 前者落后；后 3 例为真测试缺陷 |
| platform-test | `##[error]Can't use 'tar -xzf' extract archive file ...` | loong64 self-hosted runner 抖动，非代码 |

## 2 方法

1. `git worktree add /root/tp-agent/work/m3697wt agent/m3697` → `git switch -C agent/m3697 bc543d8f`
   校准到 PR head（本地同名分支原为更早的 `d63ade54`）。
2. `git merge --no-edit origin/main`（866438ac）：**无冲突**，一次性吸收 lint/layout/bfl 三处红因。
3. 分支新文件 lint：`ruff check --fix tests/test_thermal_radiation_analytic.py`（I001）+
   `ruff format src/tensorlbm/thermal_radiation.py tests/test_thermal_radiation_analytic.py`。
4. 两处真测试缺陷（仅测试文件，算法未动）：
   - `tests/test_control_volume_force.py:220` 用 `pytest.approx(bfl_force[0])`，实参是
     **0-d torch tensor**：pytest 在构造失败说明时迭代它 → `TypeError: iteration over a
     0-d tensor`，把真实数字藏掉。改为 `pytest.approx(float(bfl_force[0]))`。
   - `tests/test_regress_wall_multi_gpu.py` 两例断言 `pytest.raises(ImportError)`，即
     "要求历史 bug 常驻"；而 `src/tensorlbm/boundary/wall_shear.py::wss_from_fneq_3d`
     已正确导入 `d3q19.equilibrium3d` → 必然 "DID NOT RAISE"。改写为正向回归：
     函数可运行、`f == f_eq` 时 WSS 恒 0，并用解析 σ_xy = -(1-1/(2τ))·Σf_neq·c_x·c_y 校验
     人工 `f_neq = a·c_x c_y` 扰动（Σ(c_x c_y)² 由 `d3q19.C` 程序化计算）。

## 3 计算结果（程序化，float64）

球体 D3Q19 曲线滑移首步冲量（shape=(32,32,48)，r=6，u=0.06）：

| 量 | tight CV | wide CV | 差 |
|---|---|---|---|
| F_x | 6.4754 | 6.4754 | 1.42e-14 |
| F_y | 3.66e-15 | -3.44e-15 | 7.11e-15 |
| F_z | 1.11e-16 | 3.56e-14 | 3.55e-14 |

嵌套 CV 不变性（`atol=1e-11`）逐分量 ≤3.6e-14 通过；与 BFL 冲量本体 6.4754 偏差 <1e-8。

验证（本地 99，`PYTHONPATH=src`，torch 2.7.1 / py3.11.6）：

```
pytest -q tests/test_control_volume_force.py tests/test_thermal_radiation_analytic.py \
      tests/test_regress_wall_multi_gpu.py::TestBugIdentification \
      tests/test_regress_wall_multi_gpu.py::TestWallShearBugs
-> 46 passed, 1 xfailed
ruff check src tests examples benchmarks          -> All checks passed!
ruff format --check src tests examples benchmarks -> 1454 files already formatted
python tools/check_repository_layout.py           -> Repository layout OK (5138 candidate paths)
```

## 4 注意点 / 残留（honest）

- `tests/test_multiphase_physical_benchmarks.py::test_spinodal_2d_physically_separates`
  **在 main 866438ac 上本身即红**：`relative_mass_drift = 1.21822e-05`，门槛 `<1e-5`
  （+21.8%）。该文件在分支基点与 main 间零差异（`git diff 9460e751 origin/main --` 空），
  红因来自 main 的 `07cbd913`（QUAL-CI m310）后的漂移变化，**非本 PR 引入**。
  本轮未修：需针对多相质量守恒（forcing/碰撞算子质量通量）单独定题；改门槛即造假。
- `test_spinodal_3d_physically_separates` 标 `@pytest.mark.slow`，PR/push 事件
  `-m "not slow"` 会 deselect（CI 日志 `14 deselected`）。
- `platform-test` 红为 self-hosted loong64 runner 解包 action 归档失败，重跑即恢复。
- 本任务**无场量云图**：交付物是 CI 门禁与测试契约修复，关键量以数值表 + 解析式对齐
  给出，没有可绘制的空间场/基准曲线族，故不生成装饰性图像。

## 5 误差判定（vs 基准，逐项）

| 关键量 | 基准 | 实测 | 相对误差 | <3%？ |
|---|---|---|---|---|
| 嵌套 CV 冲量不变性 | tight vs wide CV 同量 | 逐分量差 ≤3.55e-14（F_x=6.4754） | 5.5e-15 | 是 |
| `wss_from_fneq_3d` σ_xy | 解析 -(1-1/(2τ))·a·Σ(c_x c_y)² | 同式数值输出（rel 断言 1e-5） | 机器精度量级 | 是 |
| `f==f_eq` 时 WSS | 0（解析） | 0（`count_nonzero==0`） | 0 | 是 |
| spinodal 2D 质量漂移 | 门槛 1e-5 | 1.21822e-5 | **+21.8%（未达标）** | **否 —— 残留 red（main 继承）** |