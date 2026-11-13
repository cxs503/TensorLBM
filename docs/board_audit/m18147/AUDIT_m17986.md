# 交差审计 m18147 — m17986 `XVAL-LBM-JOB-STATUS-CHECKPOINT-MACHINE`（TensorLBM）

审计对象：TensorLBM commit `56c30f13`（"MF-LBM-style job-status/wall-clock checkpoint machine"），
mission m17986，已 succeeded。本单=交差核验（非重做）。independence：审计在 `git archive 56c30f13` 的
干净抽取 `/tmp/m18147_rerun` 内独立运行，不读交付进程的中间态；外部锚在本地 vendored 上游树
`/tmp/mflbm_build/src`（lanl/MF-LBM `11be324a`）逐行核对。未检查项见文末。

## 1. 基准裁判（唯一裁判）

`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-JOB-STATUS-CHECKPOINT-MACHINE'`
→ **0 行**（全表 91 行 96 id；`%JOB%|%CHECKPOINT%|%WALL-CLOCK%|%STATUS%|%MF-LBM%` 亦 0 行）。
⇒ 该 anchor **没有平台机器裁判行**，L4（基准验证）无法机器裁定。这是判级的第一约束。

## 2. 三件套核验（逐项实测）

| 件 | 交付声明 | 审计实测 | 结论 |
|---|---|---|---|
| run_artifacts | `artifacts/xval_job_checkpoint/{summary.json, CHECKPOINT_MANIFEST.json, BENCHMARK_REF_VS_GOT.json, figures/×4, cases/×9, repro.sh}` | 文件全在；`summary.json` sha256=`be5e46294c479a99…`；独立重跑产出 `rerun_summary.json` sha256=`dfad50e3bcf7af0e…`（差异仅在墙钟量） | **存在且可独立再生** |
| benchmark ref vs got | 9 行全 PASS | 独立重跑：**8/9 行数值与交付完全相同**，第 7 行（墙钟定时精度）1.54591% → 1.79390%，两者均 <3%；`all_pass` 仍为 True | **数字可复现** |
| repro_cmd | `artifacts/xval_job_checkpoint/repro.sh` | 在**干净 out 目录**完整跑通（4 阶段，~4 min CPU，`{"all_pass": true}`）；在**已填充 out 目录**上重跑 ⇒ `RuntimeError: simulation already reached a terminal checkpoint state`（两次独立复现） | **可执行，但非幂等（F2）** |

物理与不变量复现（`rerun_compare.json`）：连续基线场 sha256 `b3283b3a236e5cbc…`
与交付 `b3283b3a236e5cbc…` **一致**（KE 比 0.852928 vs 0.852928）；
段数 1/2/4/8 全部 `steps_executed=240, steps_recomputed=0, max|Δf|=0.0`，末态 digest 与交付逐段相同；
双备份救援 `resume_source=backup, max|Δf|=0`，单备份阴性对照 `recovered=False`；
终态臂 `reached_max_step @25 步`、体积触发 2 次 == floor(1.25/0.5) —— **全部位级/整数级复现**。

## 3. 外部锚真伪（关键：XVAL 的“ref”到底是什么）

上游代码**真实存在且位置正确**：`IO_multiphase.F90:561-641` = `save_checkpoint`（写
`out2.checkpoint/`、`out2.checkpoint/2rd_backup/`，主写成功后写 `./job_status.txt='continue_simulation'`）；
`Main_multiphase.F90:222-303` = `MOD(ntime,ntime_clock_sum)` + `t_run_time`（小时）对
`simulation_duration_timer / checkpoint_save_timer / checkpoint_2rd_save_timer` 的比较与 MPI_Bcast；
`Main_multiphase.F90:139-141` = `continue_simulation` 分支；`Main_multiphase.F90:270-300` = 终态
`simulation_reached_max_step` / `simulation_done`。**锚不是编的。**

但 **9 行 ref-vs-got 里有 8 行的 ref 是规格（定义），不是上游测出来的数**：ref=0 步重算、ref=1 比例、
ref=5 状态数、ref=2 触发次数；第 9 行（定时精度）ref=0% 是理想值。⇒ 该表度量的是
**对 MF-LBM 语义的自一致性**，不是与某个外部运行结果的数值比对。因此它支持“语义/不变式级验证”，
不足以支撑“与外部解数值吻合”的 L4 主张。

## 4. 判级：**L3**

- 真实现（非 MOCK/STUB）：`job_checkpoint.py` 656 行，状态机 `LEGAL_TRANSITIONS` 显式表、原子落盘、
  sha256 载荷 digest、config-signature 守卫、`VolumeTrigger`。
- 系统集成：`LBMSimulation.enable_checkpointing/run_with_checkpointing`，`step()` 内驱动
  `maybe_checkpoint/should_stop→finish(reached_max_step)`；包级 `__init__` 惰性导出 7 个符号 ⇒ 属主求解器入口，不是旁挂脚本。
- 测试：`pytest -q tests/test_job_checkpoint.py tests/test_xval_job_checkpoint_artifact.py` → **26 passed in 6.00s**。
- 不判 L4：无 `tensor_benchmark_cases` 裁判行；ref 列定义型；repro 非幂等；开销轴跨机不复现（见 F5）。
- 不判 L5：无新物理/新数值方法，属基础设施能力。

## 5. 发现（按严重度）

- **F1 blocking（交付/发布）**：`56c30f13` **不在 main**——`git cat-file -e HEAD:src/tensorlbm/job_checkpoint.py` 失败，
  main 内 `grep job_checkpoint|enable_checkpointing` 0 命中，`artifacts/xval_job_checkpoint/*` 0 命中。
  能力对平台**不可消费**，且这正是历史审计（m17991/m17968）被 gate 以“工件缺失”reopen 的成因。
- **F2 blocking（可复现性）**：`repro.sh` 不清理输出目录 ⇒ 二次运行崩溃（机制：`begin()` 读到残留
  `job_status.txt=done` → `finished=True`，`run_with_checkpointing` 的 seg-0 路径无 `force_restart` 亦无断言 ⇒ `step()` 抛错）。
  探针：`evidence_snapshot/repro_probe.sh`。交付报告⑤.1 只记录了“并发写者”现象，未覆盖“同目录重跑”这一更常见路径。
- **F3 major（基准语义）**：见 §3——XVAL 表 8/9 行 ref 为定义型；交付报告⑥“9/9 行 <3%”在字面上成立，
  但其中 8 行的误差是**构造性为 0**（ref 与 got 同源），不构成精度证据。
- **F4 minor（语义漂移未声明）**：上游状态词是 `new_simulation/continue_simulation/simulation_done/simulation_reached_max_step`，
  本仓为 `new/continue/done/failed/reached_max_step`；“5 states vs 5 states”只比数量。报告差异清单有“changed”条目，
  但 ref-vs-got 表未标出这一点（另新增 `failed` 状态）。
- **F5 minor（不可复现轴，交付已自身声明）**：checkpoint 开销 delivered −1.84%/3.04% vs 审计 13.46%/4.17%
  （主机 CV 8.77%）⇒ 仅量级参考；定时精度轴稳定（1s: 1.55%→1.79%，2s: 1.42%→0.39%）。交付⑤.3 已如实写“共享控制面开销不可直接相减”。
- **F6 positive**：位级不变量（重启 0 差、0 重算、备份救援、终态阻断、体积触发）跨机器/跨进程**精确复现**，
  且基线场 digest 一致 —— 这是本单最硬的证据面。

## 6. 未运行的检查（unknown / not_run）

- 最坏情形中断（SIGKILL 于 `atomic_torch_save` 中间）的断电语义只由单测覆盖，**未在审计侧做 kill -9 注入**。
- 跨 device（CUDA↔CPU）/ `stream=True` 下的位级相等**未测**（交付亦声明只保证物理一致）。
- 对流出口 BC 的 `f_convec_bc/g_convec_bc/phi_convec_bc` 路径**未实现**（交付如实标注 not ported），本单无此 BC 可测。
- 大规模（≥512²、多进程）性能与可靠性**未测**；开销轴在审计主机上不可分辨（F5）。

## 7. 主张核对（overstated claims）

| 交付主张 | 审计裁定 |
|---|---|
| “XVAL against lanl/MF-LBM，9/9 行 PASS，<3%” | 数字可复现（8/9 位同，定时行 1.79% <3%），但**8 行 ref 是定义型**，措辞宜为“语义一致性 + 不变式”，不是外部数值验证 |
| “已 matured/可复现” | repro 在干净目录成立；**非幂等**，且 commit 未进 main ⇒ 平台侧不可消费 |
| “状态机 5 状态与上游一致” | 数量一致、**词表不同**且新增 `failed`（F4） |

## 8. Disposition

**approve with notes**（能力实质 L3 成立，独立复现通过）；**blocking 修复项**=F1（合入 main）+F2（repro 前置清理或 `--force`）。
L4 前置条件：为 anchor 建 `tensor_benchmark_cases` 裁判行，并把 ref 列改为**上游实测或解析参照**，而非规格常量。

## 工件

`docs/board_audit/m18147/`：`verdict_evidence.json`（落表 evidence）、`rerun_compare.json`（逐行对照，程序化生成）、
`figures/fig_audit_rerun_compare.png`、`insert_verdict_m18147.sql`、`evidence_snapshot/`（job_checkpoint.py、
test_job_checkpoint.py、integration_diff.patch、delivered/rerun summary 与 REF_VS_GOT、repro_probe.sh）。
