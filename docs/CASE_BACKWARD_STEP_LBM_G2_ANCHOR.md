# CASE-BACKWARD-STEP-LBM · 管到底第1轮（m17819）：交付物核验 + 入口测试补齐 + g2 锚点在途

锚点 `CASE-BACKWARD-STEP-LBM`（TensorLBM 算例池）｜判据 `|X1/s − ref| ≤ 15 %`（refs.json）｜
ref: Armaly et al. 1983 *J. Fluid Mech.* 127:473（tier `SECONDARY_IN_REPO`，原文获取受阻四次登记于 refs.json）。

## ① 本单核验结论（管到底对表）

原 PR#1984（m15519 交付物）已裁定关闭，本单从当前 main 切枝重新对表：

| m15519 交付物 | main 状态 | 证据 |
|---|---|---|
| `benchmarks/case_pool/backward_step_armaly/refs.json` | **在库**（60 行，后续轮次已扩充） | `git log --follow`：#1831 (m15519) 并入，#1871 (m15929) 再扩 |
| `benchmarks/case_pool/backward_step_armaly/run.py` | **在库**（193 行，与原 PR +193 一致） | 同上 |
| `docs/CASE_BACKWARD_STEP_LBM.md` | **在库**（69 行，与原 PR +69 一致） | 同上 |

即 m15519 交付物本体**早已经 #1831 并入 main**，#1984 为重复 PR，关闭无损失。
本单的真实缺口 = **判分测试恒 skip**：
`tests/test_case_pool_backward_step_armaly.py::test_committed_benchmark_meets_reference_tolerance`
按 `artifacts/case_backward_step_armaly/g2_re{100,200,400}/result.json` 对 refs.json 逐项判分，
而 g2 档（Armaly 精确膨胀比 101/52 细网格）工件从未入库（m16958 只交 h1 减配阶梯，工件同样未入库）。

## ② 本单改动（≤10 文件，单一功能点：让算例入口可测、锚点工件在途可追）

1. **新增 `tests/test_run_case_entry.py`（4 测试，全绿）**：此前 `run_case.py` 入口
   （preset 表 `GRID` + `--smoke` 端到端 + 未知 preset 拒绝）**零测试覆盖**，本单补齐：
   - `test_grid_table_is_consistent`：9 档 preset 唯一、Armaly 档 ER=101/52 机器精度、h1 档 +0.99% 偏差钉死；
   - `test_build_config_keeps_tau_in_lattice_window`：τ≥0.55 晶格窗契约钉死，
     **唯一例外 h1_re400（τ=0.538）如实登记**（m16958 文档已记其 lattice 条件失败，由 verdict 门判）；
   - `test_smoke_preset_runs_end_to_end_and_writes_schema`：smoke 档（16×174 格、400 步、~10 s）
     端到端跑通，工件三件套（result.json/wall_shear.csv/field.npz）+ 遥测 schema 钉死；
   - `test_unknown_preset_is_rejected`：坏 preset 必须 SystemExit。
2. **本状态文档**：锚点工件在途登记（见③），供第2轮接棒。

**未改任何库/测试既有代码**；热文件（`src/tensorlbm/*`）零触碰。

## ③ g2 锚点工件在途（集群作业号，第2轮接棒）

| 作业 | 内容 | 命令 |
|---|---|---|
| cj1668 | g2_re100 锚点（hi=104, s=98, 90000 步预算, 稳态自停 5e-3） | `PYTHONPATH=src python3 benchmarks/case_pool/backward_step_armaly/run_case.py --preset g2_re100` |
| cj1671 | g2_re200 | 同上 `--preset g2_re200` |
| cj1672 | g2_re400 | 同上 `--preset g2_re400` |
| cj1683 | 全套 fast 测试（CI 口径 `-m "not slow" --timeout=600`） | `python3 -m pytest -m "not slow" --timeout=600 -o timeout_func_only=false -q` |

第2轮接棒：`run_cluster ps/fetch` 取回 `artifacts/case_backward_step_armaly/g2_re*/`，
过 `verdict.py` 五条件门后入库 → 判分测试由 skip 转实跑。
**honest 登记**：本单窗口内集群队列深（~320 作业在列），g2 工件未及取回，
故判分测试在本 PR 内仍为 skip——**不粉饰为已激活**，欠账登记于此。

## ④ 本单测试证据（本地，loong64 99 控制面）

- `pytest tests/test_run_case_entry.py` → **4 passed**（15 s，含 400 步 smoke 端到端）。
- `pytest tests/test_run_case_entry.py tests/test_case_pool_backward_step_armaly.py tests/test_bfs_converged_verdict.py -k "not short_run"` → **15 passed, 2 skipped**（skip = 工件缺失的判分测试，如实）。
- 全套 fast 口径（`-m "not slow"`，CI 同参）：集群作业 cj1683 在途，结果随第2轮补报。
