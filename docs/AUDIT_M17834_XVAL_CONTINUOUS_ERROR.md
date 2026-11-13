# 交差核验报告 · m17834 XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS（board-audit）

- 审计单：m17875（仓库 TensorLBM，分支 agent/m17875）
- 被审单：m17834（status=succeeded，分支 agent/m17834，工作树 `/root/tp-agent/work/m17834`）
- 锚点：`XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS`
- 基准裁判：`tensor_benchmark_cases` id=63（唯一裁判，报告好看不算数）
- 审计时刻：2026-11-13

---

## ① 案例介绍：这一单到底该交什么

锚点的验收物不是"任意解析解对标"，而是基准表里声明的**那一个**算例与**那两个**数：

| 项 | 基准表 id=63 原文 |
|---|---|
| case_name | vivsim Re=200 free-stream cylinder: continuous error print vs DOI refs (30D × 20D, cyl at 10D, 150 D/U0) |
| ref_source | `haimingz/vivsim @ fb84b12f examples/2d/flow_pass_cylinder.py:78-85`（CD_REF=1.389, CL_REF=0.718；DOIs 10.1016/j.ijmecsci.2024.108961 / 10.1016/j.camwa.2014.05.013 / 10.1017/S0022112086003014） |
| metric | `Cd_mean vs 1.389` **且** `Cl_peak 同口径 vs 0.718`（时间记录后半段的 mean/max，对齐 OSS 源码 207-208 行） |
| tolerance | `<15%`（owner 精度标准，移植时预声明 TOL_PCT=15.0，严格小于） |
| 先前状态 | 第2轮 got = WITHHELD（3 次新跑 0 个数字；2000/2500/500 步均 NaN） |

**判据唯一**：m17834 若没有对 vivsim 圆柱算例给出 Cd_mean 与 Cl_peak，则锚点未闭合，无论附了多少别的对标。

## ② 怎么做的（取证链，可重放）

1. 读工作树 `git status` 与产物清单：m17834 的交付物**全部未提交**（1 个 modified + 3 个 untracked），无 commit、无 PR；main（我仓 `804636b0`）里并不存在这些文件。
2. 取基准裁判行：`SELECT case_name, metric, tolerance, ref_source, status FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS'` → 1 行，status 仍为 `pending`。
3. 三件套核验：impl_files / entry / run_artifacts 逐个落到文件与行号（见 `artifacts/m17875/verdict_evidence.json`）。
4. **复现运行**（不是读它自己的 json）：执行被审单自带入口
   `cd TensorLBM/benchmarks/continuous_error_xval && python run_continuous_error_xval.py --out /tmp/xval_repro`
   → 3/3 PASS，三个误差数字与随单工件**逐位一致**（数值真实、确定性、非手抄）。
5. **测试执行**：`python -m pytest tests/test_continuous_error_xval.py` → 收集期报错；定位根因后把 `parents[2]` 修为 `parents[1]` 再跑（在 `/tmp/verify2` 用符号链接复刻同深度路径，未改被审工作树）→ 12 passed / 2 failed。
6. **锚点覆盖检索**：`grep -n 'Cd|Cl|cylinder'` 命中 0；交付的 3 个算例是 Poiseuille H16 / Couette H20 / Womersley α=5 H31。

## ③ 计算结果（数字表，全部来自复现运行）

### 3.1 交付物真实做了什么（ref vs got）

| 算例 | 参考解（引用） | L2_rel | L∞_rel | n_pts | <15%? | <3%? |
|---|---|---|---|---|---|---|
| poiseuille_2d_H16 | Schlichting&Gersten 2017 §III.2.1 | 0.4921% | 0.3608% | 16 | ✅ | ✅ |
| couette_2d_H20 | Schlichting&Gersten 2017 §III.2.2 | 0.9836% | 0.9633% | 20 | ✅ | ✅ |
| womersley_alpha5_H31 | Womersley 1955, doi:10.1113/jphysiol.1955.sp005276 | 0.0156% | 0.3210% | 4 | ✅ | ✅ |

复现命令输出与 `continuous_error_xval.json`（ts 2026-11-13T09:43:09Z，sha256 `6cdba43b…9497`）一致。

### 3.2 锚点基准的 got 值

| 要求量 | ref | m17834 got | 误差 | 判定 |
|---|---|---|---|---|
| Cd_mean | 1.389 | **不存在** | — | **WITHHELD** |
| Cl_peak | 0.718 | **不存在** | — | **WITHHELD** |

覆盖度 **0/2**；基准行状态保持 `pending`。交付的解析解算例与被要求的圆柱算例在物理（定常槽道/脉动槽道 vs 钝体绕流+涡街）、几何、参考量（速度剖面范数 vs 力系数）上**都不是同一件事**，属口径替换，不构成锚点闭合。

### 3.3 测例执行

| 命令 | 结果 |
|---|---|
| `python -m pytest tests/test_continuous_error_xval.py`（随单原样） | **collection ERROR**：`ModuleNotFoundError: tensorlbm.verification.xval` |
| 同上，仅把测试文件第 13 行 `parents[2]` → `parents[1]` | **12 passed, 2 failed**（7.48 s） |

失败两项：`test_large_error_fails`（`assert 0.5 > 0.5` 边界断言）、`test_zero_reference`（`l2_rel 0.03742 != l2_abs 0.02160`，相对范数按求和、绝对范数按 RMS，归一化口径不一致）。
根因（可复现）：测试文件把 `ROOT` 定到**工作树根**而非 `TensorLBM/`，而 `<worktree>/src` 下存在一份陈旧 `tensorlbm` 包（全平台 394 个工作树都有同样的 stray `src/tensorlbm`），插入 `sys.path[0]` 后把正确的 `TensorLBM/src` 顶掉，后续 `verification.xval` 自然找不到。

### 3.4 报告六段

`benchmarks/continuous_error_xval/REPORT.md` ①–⑥ 标题齐全，但：④ 明确声明无图；⑥ "基准值"列**全为 0.0**（误差对零真值，不是 ref-vs-got）；⑤ 无收敛性/敏感性两点分析；平台侧 `tensor_reports` **无 mission_id=17834 的记录**（0 行）。

## ④ 结果图

`artifacts/m17875/fig1_m17834_error_vs_tolerance.png`（由 `make_audit_fig.py` 读复现工件程序化生成，零手抄）：
左栏=三个交付算例的 L2/L∞ 相对误差 vs 15% 容差线；右栏=锚点真正要求的 Cd_mean/Cl_peak——两根柱子高度为 0，标注 "no number produced"。

## ⑤ 注意点（坑与局限）

1. **口径替换是最容易骗过"报告好看"的失效模式**：m17834 的报告自称"3/3 PASS，误差 <1%，远低于 15%"，数字真实、可复现，但比的不是锚点要求的量。只看报告会误判为 L4。
2. **测试绿/红与"能否收集"是两件事**：原样命令连收集都过不了；把那行路径错误修掉后仍是 12/2，报告里"7 个单元测试 + 1 集成测试"的表述与实际 14 个测例、2 红不符。
3. **交付物未提交**：全部改动停留在工作树（untracked），无 commit/PR，main 上不存在——"成果入库"未发生，任何基于 main 的复核都会 0 命中。
4. **外部锚未真取**：REPORT.md §⑤ 自认"vivsim 具体实现未获取"，故 mission goal 要求的"本仓 vs OSS 差异清单"无源可依；引用的 DOI 也是另一套文献（Schlichting / Womersley），非基准行的三篇。
5. 审计侧限制：我只复现了对方跑得动的那条命令，未替它实现圆柱算例；Cd/Cl 的 WITHHELD 结论来自"交付物中不存在该计算"（grep 0 命中 + 工件无字段），不是"我跑了并失败"。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 基准 | 计算值 | 相对误差 | <3%? |
|---|---|---|---|---|
| Poiseuille L2_rel | 0（解析自洽） | 0.4921% | 0.49% | ✅ 是 |
| Couette L2_rel | 0（解析自洽） | 0.9836% | 0.98% | ✅ 是 |
| Womersley L2_rel | 0（解析自洽） | 0.0156% | 0.02% | ✅ 是 |
| **Cd_mean** | **1.389** | **无** | **无法计算** | ❌ **否——基准量缺失（WITHHELD）** |
| **Cl_peak** | **0.718** | **无** | **无法计算** | ❌ **否——基准量缺失（WITHHELD）** |

结论：交付的解析解误差远小于 3%，但**锚点基准（Cd_mean/Cl_peak）0 覆盖**，基准判据未闭合。

---

## 判级与落表

**L2**。理由：impl_files 是真代码（非占位）、entry 有 CLI、run_artifacts 可复现且数值真实、并真实调用了仓内 `ice_hydro.lbm_channel` / `verification.womersley_pulsatile`（弱 L3 迹象），但——

- L3 不成立：锚点算例未接入主求解器/平台入口，无平台侧调用方；
- L4 不成立：基准 ref-vs-got（Cd_mean 1.389 / Cl_peak 0.718）**缺失**，且 repro_cmd 中的 pytest 条不可执行、2 个测例红；
- L5 不成立。

落表：`tensor_maturity_verdicts (repo='TensorLBM', scope='m17834 XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS', level=2, verdict_by='board-audit')`，evidence 见 `artifacts/m17875/verdict_evidence.json`（impl_files/entry/run_artifacts/tests/integration_callers/benchmark/repro_cmd/external_anchor 八键均给真实指针）。

**处置建议（reopened 返修）**：① 实现 vivsim Re=200 圆柱算例并给出 Cd_mean/Cl_peak（对齐 fb84b12f:78-85 与 207-208 行口径）；② 修 `parents[2]`→`parents[1]` 与两个红测例（含相对/绝对范数归一化口径统一）；③ commit 并推分支；④ 发 `tensor_reports` 六段报告。

## 附：落表与 gate 预演（机器复核口径）

- 落表：`tensor_maturity_verdicts` id=**654**，`repo='TensorLBM'`，`scope='m17834 XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS'`，`mission_id=17875`，`level=2`，`verdict_by='board-audit'`，`gate_state='pending'`（等守护周期裁定）。
- 证据包：`artifacts/m17875/verdict_evidence.json`（八键齐备）。因原件在 m17834 工作树**未提交**，审计侧另存只读快照 `artifacts/m17875/reviewed_m17834/`（逐字节 + sha256），使 gate 的 `files_exist`/`impl_real` 能对真实文件复核。
- 预演（直接调用守护自己的判据函数，非口头声明）：
  `maturity_gate.judge(2, evidence, 'TensorLBM', 17875) -> (2, [])` —— L2 三键 `impl_files / entry / run_artifacts` 机器复核全通过，无降级项。
- 说明：L3/L4 键虽写在 evidence 里（供返修对照），但本 verdict 只声明 L2；`benchmark.got = null` 与 `tests_detail` 的 2 红如实入包，不做粉饰。
