# fixed_items.md — QUAL-CI（m310）修复清单

**append-only**。每条：类别 / 根因 / 修法 / 证据（证据一律指向原始工件文件，数字直读，不手抄）。
基线：`main@6ab2d9e4`，CI run `40075503423`（nightly，2026-10-24T07:47Z）→ lint failure + test(3.11) failure + test(3.12) failure，package-build success。
原始 CI 日志已落盘于 `agent_work/m310/baseline/ci_logs/`（经 GitHub REST `/actions/jobs/{id}/logs` 抓取，非记忆）。

---

## FIX-001 [lint] ruff check — 623 → 0

- **类别**：静态质量 / 存量漂移（main 侧累积，非任何单一 PR 自有文件）
- **基线证据**：`baseline/ci_logs/job_120817846451.log`（第 9438 行 `Found 623 errors.`；同 job 步骤 6 `Lint (ruff)` = failure）
- **根因**：main 长期累积的脚本式写法——`E701/E702` 单行多语句（集中在 `cut_cell_bgk_2d.py` 等短名模块，97/39/36/33 处）、`I001` 导入乱序、`F401` 未用导入、`E401` 一行多导入、`F541` 空 f-string、`F841` 未用变量、`E722` 裸 except。
- **修法**：`ruff check --fix`（197 项自动可修）+ `ruff format` 全量重排（`E701/E702` 等 416 项由格式化消解）+ 6 项手工：
  - `E722` `benchmarks/pending/channel_turb_3d/plot_figures.py` → 显式 `except (KeyError, TypeError, ValueError)`（保持"跳过坏行"的原语义，不再吞 KeyboardInterrupt/SystemExit）
  - `F841` ×4：`ai/suboff_inference.py`（去无用 `pred_result = None`）、`cut_cell_bgk_2d.py`（去未用 `incoming` mask）、`maneuvering.py` ×3（`mass_matrix()/damping_matrix()` 改为只保留校验副作用调用；`np.linalg.solve` 结果改名 `_c` 并注明 mode 分析用的是 `q_ss`）
- **未改测试断言**：本次 lint 修复零触碰 `tests/` 的断言逻辑（`tests/` 侧仅受 `ruff format` 重排，无断言语义变化）。
- **修后**：`All checks passed!`，退出码 0 → `evidence/ruff_check_after.txt`

## FIX-002 [lint] ruff format --check — 116 文件未格式化 → 0

- **类别**：静态质量
- **基线证据**：`baseline/ruff_format.txt`（`116 files would be reformatted, 1417 files already formatted`）。CI 上该步骤因前序 `Lint (ruff)` 失败而 **skipped**（`job_120817846451.log` 步骤 6 skipped），即从未真正执行过——属于"红 CI 掩盖的下一层红"。
- **根因**：同 FIX-001 的漂移。
- **修法**：`ruff format src tests examples benchmarks`（116 files reformatted）
- **修后**：`1533 files already formatted`，退出码 0 → `evidence/ruff_format_check_after.txt`

## FIX-003 [test] test_compile_utils::TestRunnerDedupRegression ×4 — 真缺陷（merge 丢代码）

- **类别**：真实代码缺陷 / 非环境、非 lint
- **基线证据**：`baseline/ci_logs/job_120817846396.log` 与 `job_120817846337.log` 的 short test summary：
  `TypeError: SuboffCmkKbcConfig.__init__() got an unexpected keyword argument 'compile_mode'`（4 例，py3.11 与 py3.12 双矩阵一致）
- **根因（逐版 git 比对，勿凭记忆）**：merge commit `8b2672bc`（PR #342，2026-10-10）合入分支 `83810c6b` 时，`SuboffCmkKbcConfig` 的 `compile_mode` 字段、`__post_init__` 三态校验、及步进循环里的 `compile_step` 分支整段丢失。逐版行数：`4ba836c9` = 12 行、`666a00ea` = 12 行、`536cc9c5` = 12 行、`da73d017` = 13 行、`83810c6b` = **0 行**，合并结果 = 0 行。而其使用方（`tensorlbm/compile_utils.py`、`suboff_torch_distributed.py`、`aircraft_icing.py`、`tests/test_compile_utils.py`）全在——是典型 merge 静默丢代码。
- **修法**：恢复 `SuboffCmkKbcConfig.compile_mode: str | None = None` 与 `__post_init__` 校验（① 走 `compile_utils.validate_compile_mode` 单一真源白名单；② 与 `use_triton_step` 互斥；③ KBC 拒绝，因为熵二分每步 `.item()` 不可编译）。**未改任何测试断言**（测试原文保持逐字不变）。
- **说明（honest）**：本次只恢复了配置契约（字段 + 校验），**未**恢复循环内的 `compile_step` 数值路径——重接编译路径需 CUDA + 数值等价复验，超出本次 CI 清零范围。已在下方 [待办] T-002 登记。
- **修后**：`20 passed, 2 skipped, 1 xpassed` → `evidence/test_compile_utils_after.log`

---

## 未清偿项（honest 披露，登记为 QUAL 审计候选，不静默修）

> 依任务纪律第 5 条：发现红背后是**真物理/数值缺陷**时不静默修，登记候选，只修确证非物理语义的部分。

### QUAL-CI-SPINODAL-1 — spinodal 在 1200 步内根本未相分离（真物理语义）

- **基线证据**：`baseline/ci_logs/job_120817846396.log`
  - `test_spinodal_2d_physically_separates - assert 1.21822e-05 < 1e-05`（py3.12）/ `1.23526e-05`（py3.11）
  - `test_spinodal_3d_physically_separates - assert 1.0 > 2.0`
- **本地等价复跑取证**（`evidence/test_spinodal_after.log` + 诊断扫描）：
  - 2D(64×64, G=-4.0, τ=1.0, ρ0=0.7, noise 0.05, 1200 步)：`density_ratio = 1.0002`，`phase_separated = False`
  - 3D(24³, 同参数)：`density_ratio = 1.0`，`rho_liquid = rho_gas = 0.6999→0.7004`，`phase_separated = False`
  - 时间序列（每 100 步）：`rho_std` 单调**衰减** 5.1e-4 → 3.3e-5 —— 噪声被扩散抹平，从未进入 spinodal 失稳带
  - float64 复跑：3D `relative_mass_drift = 0.0` 而 `density_ratio` 仍 = 1.0 —— **证明质量漂移不是分离失败的原因**，两个断言里前者只是伴生读数
- **判读**：这不是阈值口径问题，而是 Shan-Chen 单组分势场在当前 `psi_exp` / `G=-4.0` / `ρ0=0.7` 组合下没有跨过相分离临界点（或 `collide_sc_single_component` 的强制力项归一有偏）。属真物理/数值缺陷。
- **处置**：**不修**（既不静默改实现去迎合断言，也不改断言阈值放行）。交 owner 裁决。
- **候选待查**：`psi_exp` 势阱深度与 G 的临界值关系；SC 交互力 `F = -G·ψ·Σ w ψ(x+c) c` 的 stencil 归一；benchmark 参数（G=-4.0 是否本就落在两相区外）；`G` 符号约定历史（`776db80b` 曾翻转 SCMC 的 `G12` 符号为 "< 0 才分离"）。

### QUAL-CI-NAVIERSLIP-1 — Navier slip 质量守恒断言的绝对阈值口径（实现无缺陷）

- **基线证据**：`baseline/ci_logs/job_120817846396.log`
  `AssertionError: Navier slip should conserve mass, but drift is 0.0001220703125` / `assert 0.0001220703125 < 1e-06`
- **本地等价复跑取证**（`evidence/test_navier_slip_after.log` + 对照实验）：
  | 条件 | drift | 相对误差 |
  |---|---|---|
  | float32 + Navier slip（测试原样） | 1.2207e-4 | 9.08e-8 |
  | float32 **不加** slip BC（同循环） | 1.2207e-4 | 9.08e-8 |
  | float64 + Navier slip | 0.0 | 0.0 |
- **判读**：漂移与 slip BC **完全无关**——不加 BC 时数值一模一样；漂移全部来自 float32 下 `equilibrium3d`/`stream3d` 的舍入。断言用**绝对**阈值 `1e-6`，而初始总质量 1344，实际相对误差 9.08e-8。是断言口径与环境不可比，**不是** BC 物理缺陷。
- **处置**：**不修**（未确证为 BC 缺陷，按纪律不静默改实现；也不改断言放行）。交 owner 裁决。
- **候选处置建议（供 owner 参考，本次未执行）**：断言改为相对误差（如 `< 1e-6 * mass_initial`）或加 float64 对照臂。

---

## [待办] 交给后续班次

- **T-001**：`tests/ci_quarantine.txt` 仍有 141 行历史豁免（本次未动）。QUAL-CI 的目标是"存量红清零"，quarantine 是**非严格 xfail**，会掩盖真红——建议后续班次逐条复跑，把"确实已修好"的行删掉、把"仍红"的行登记为 QUAL 候选。
- **T-002**：FIX-003 只恢复了 `compile_mode` 的配置契约，**未**重接 `suboff_cmk_kbc_runner` 循环内的 `compile_step` 数值路径（需 CUDA + eager/compiled 等价复验）。
- **T-003**：`test (3.11)` 与 `test (3.12)` 是两个独立矩阵作业，本次本地只能在单一解释器（3.12）上等价复跑；py3.11 侧结论以 CI 日志同源失败为准（本次 7 个失败在两矩阵完全一致）。
- **T-004**：`package-build` 在 CI 为 success，本环境无 `pip`/`build`，**不可本地复现**，仅记录 CI 结论。
- **T-005**：本环境无 `triton`/`fastapi`，`tests/conftest.py` 自动忽略 16 个测试文件（5 fastapi + 11 triton）；CI 装了 triton，实际覆盖面比本地大。残余风险：这 11 个 triton 文件在 CI 上的通过性本次未取得本地等价证据。