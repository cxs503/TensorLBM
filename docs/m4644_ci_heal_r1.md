# m4644 CI heal note — analytic verification re-opened from current `main`

**背景.** PR #673（m4644，验证债 burn-down）由 owner 于 2026-11-05 裁定关闭，
本次从 **当前 `main`（`b3d9ba46`）重切分支**重开，只补原 PR 被提交门禁滤除、
且尚未落在 `main` 上的产物。这份笔记记录**重开后的验证路径**与**哪些部分不需要
重做、为什么**，供复核者快速判定"这条分支相对 main 到底加了什么"。

## 1. 起点核对：哪些部分已经在 main 上了

重开前先对表，避免把已合入的内容再提一遍（也避免把已修的问题当成本次成果）：

| 原 PR #673 的文件 | 当前 main 状态 | 证据 |
|---|---|---|
| `tests/test_lbm_verification.py`（+378/−251，改写为 assert 用例） | **已在 main** | `5cd12e87`（m4765），该提交同时改了此文件 525 行 |
| `tests/ci_quarantine.txt`（−4 行） | **已在 main**，且无残留 | `git show 5cd12e87 -- tests/ci_quarantine.txt` 删除 4 行；`grep test_lbm_verification tests/ci_quarantine.txt` 无命中 |
| `docs/lbm_analytic_verification_r1.md`（+164） | 缺失 | 提交门禁滤除，`ls` 不存在 |
| `scripts/verify_lbm_analytic_convergence.py`（+135） | 缺失 | 同上 |

因此本次变更**不重复提测**已合入的测试改写，而是补两件 provenance 产物，
并让脚本能对上 main 上**现版本**的测试模块（原脚本调用的
`shear_wave_viscosity()` / `poiseuille_profile()` / `couette_profile()` /
`tv.DEVICE` 在当前模块里都不存在——直接照搬会 `ImportError`/`AttributeError`）。

## 2. 本次变更做了什么

1. **`scripts/verify_lbm_analytic_convergence.py` 重写为对当前模块的 provenance 驱动**：
   直接 `import test_lbm_verification as tv`，调用 `shear_wave_case` /
   `couette_case` / `poiseuille_case` / `friction_case` 得到 canonical 值，
   再用同一批函数的关键字参数做网格细化与负控。脚本自身是 gate：canonical 超限、
   细化不收敛、负控没被检出，任一发生即非零退出。
2. **测试模块只加关键字参数、默认值不变**：`shear_wave_case(..., nx=64, steps=100, tau=1.0)`、
   `couette_case(..., ny=12, steps=1000, bounce_back=True)`、
   `poiseuille_case(..., ny=12, steps=1000)`、`friction_case(..., ny=12, steps=1000)`。
   断言算例仍是**默认参数**，因此 `pytest tests/test_lbm_verification.py`
   的判定范围与 main 完全相同；好处是公开表格与断言走同一段代码，不会漂移。
   另把剪切波初值的 `arange(dtype=torch.float32)` 改为跟随 torch 默认 dtype
   （默认仍是 float32，CI 判据不变），使脚本能在 float64 下测到真实收敛阶数。
3. **`docs/lbm_analytic_verification_r1.md` 按 main 现版本的实测结果重写**，
   表格不是抄的：由脚本生成 `artifacts/lbm_analytic_verification/tables.md` 后
   原样内联。**没有**沿用已关闭 PR 里的旧表（旧表是 float64 参考实现 +
   不同案例划分的数字，与 main 现模块不同源）。
4. **提交 provenance 产物** `artifacts/lbm_analytic_verification/{convergence.json,convergence.png,tables.md}`，
   避免文档引用悬空（原 PR 正是因"引用的产物没进仓库"被判悬空 provenance）。

`src/` 一个字节都没动：`git diff origin/main --stat` 只有下述 7 个文件。

## 3. 逐门复现（本机 loongarch64 / Python 3.11 / ruff 0.15.0）

命令与 `.github/workflows/ci.yml` 同名作业一致：

| CI 作业 | 命令 | 结果 |
|---|---|---|
| `lint` ruff check | `ruff check src tests examples benchmarks` | **1 error，main 既有**：`benchmarks/verified/mmg_imo_msc137_uncert/run_uncert.py:351` 的 `I001`（import 排序），非本变更文件（见 C1） |
| `lint` ruff format | `ruff format --check src tests examples benchmarks` | **6 files would be reformatted，全部 main 既有**（`benchmarks/…couette_poiseuille/run.py`、`…run_uncert.py`、`tests/test_couette_poiseuille_tutorial.py`、`tests/test_mmg_imo_msc137_uncert.py`、`tests/test_sdaa_ledger_v2.py`、`tests/test_vortex_identification_analytic.py`）；`tests/test_lbm_verification.py` **already formatted** |
| 本变更脚本 lint | `ruff check scripts/verify_lbm_analytic_convergence.py` + `ruff format --check` | **All checks passed / 1 file already formatted** |
| `lint` docs | `python scripts/check_platform_docs_consistency.py` | **`[docs-consistency] OK`** |
| `repository-layout` | `python tools/check_repository_layout.py` | **`Repository layout OK (5169 candidate paths)`** |
| `repository-layout` 测试 | `pytest -q tests/test_repository_layout.py tests/test_repository_report_migration.py` | **18 passed, 81 subtests passed** |
| 解析解用例 | `pytest -q tests/test_lbm_verification.py` | **5 passed（25 s）** |
| provenance 脚本 | `python scripts/verify_lbm_analytic_convergence.py` | **退出码 0**，`OK: canonical cases < 1 %, refinement converges, controls detected`（176 s） |

## 4. Caveats / 没做的事

* **C1 — main 上已有 lint/format 欠账。** `ruff check` 的 1 个 `I001` 与 6 个
  format 待改文件都在本变更**从未触碰**的路径上（`benchmarks/`、其它 test 文件），
  在未改动的 main 上同样存在。本变更不修它们：那属于另一个单点，混进来会破坏
  "单 PR 逻辑单一"的交付纪律。若 CI 因此红，责任归属可由
  `git diff origin/main --name-only` 直接判读（本分支只动 7 个文件）。
* **C2 — mypy 核心门有 1 个 main 既有报错。**
  `mypy --follow-imports=silent src/tensorlbm/{d2q9,d3q19,solver,utils}.py --ignore-missing-imports`
  在本机报 `src/tensorlbm/solver.py:437: error: Returning Any from function declared to
  return "Tensor" [no-any-return]`（`Found 1 error in 1 file (checked 4 source files)`）。
  这 4 个文件本变更零改动、`src/` 与 main 字节一致，所以该报错是**继承状态**，
  不由本分支引入，也不在本变更内修（改 `src/` 会越界到求解器改动）。
* **C3 — 没有跑全量 `tests/`。** 全量套件在本机需要数十分钟且依赖 CI 才装的
  dev extras（`h5py` 等）。本变更触碰的路径（解析解模块、layout/report 迁移、
  docs 一致性）与 provenance 脚本都已单跑通过；其余用例不受影响（`src/` 零改动）。
* **C4 — 文档数字只覆盖"当前 main 的模块"**：τ=1、float32 断言、CPU。
  局限清单见 R1 文档 §6，不在本笔记重复。
* **C5 — 没有重用已关闭 PR 的分支内容。** 分支从 `origin/main` 重切（`agent/m4886`），
  旧 PR 的旧表/旧脚本未 cherry-pick——它们与 main 现模块不同源，照搬会造出
  "文档说 7.1e-13%，代码跑出 1.3e-3%" 的悬空 provenance。

## 5. 变更文件清单（7 个）

| 路径 | 性质 |
|---|---|
| `tests/test_lbm_verification.py` | 仅加关键字参数（默认值 = 断言算例）+ dtype 跟随默认 |
| `scripts/verify_lbm_analytic_convergence.py` | 新增：provenance + gate |
| `docs/lbm_analytic_verification_r1.md` | 新增：按当前 main 实测重写 |
| `docs/m4644_ci_heal_r1.md` | 新增：本笔记 |
| `artifacts/lbm_analytic_verification/convergence.json` | 新增：全部数字 |
| `artifacts/lbm_analytic_verification/convergence.png` | 新增：三联图 |
| `artifacts/lbm_analytic_verification/tables.md` | 新增：R1 文档内联的表格源 |