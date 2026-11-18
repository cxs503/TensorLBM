# m20976 交差核验 — XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS（对象 m20893，R2 分辨率扫描）

**取证基线**：`TensorLBM` 工作树 `/srv/tensor/agent/work/m20976`（HEAD `d210ab595` = main）；
交付 commit **`578410df6`**（只在 `refs/heads/agent/m20893`，**不在 main**）；
`tensor_dev_missions.id=20893.result`（status=succeeded, fail_stage=`budget_exhausted+forced_finish`, tool_calls=60）；
`tensor_reports.id=18989`；`tensor_benchmark_cases`；原始作业回执 `/srv/tensor/agent/cluster/jobs/cj3032.log|cj3062.log`。
**对象工作树** `/srv/tensor/agent/work/m20893` 已不可访问（路径越界）⇒ 全部以交付 commit + 作业回执 + 平台表为证。

## 裁决

| 项 | 结论 |
|---|---|
| 级别 | **L2**（独立功能实现：真代码 690 行，输入→计算→输出闭环，数字可由作业回执逐位对上） |
| maturity_gate | **reopened（返修）** |
| 三件套 | run_artifacts ✅（但只存标量最大值）／benchmark ref-vs-got ❌ **WITHHELD**（裁判表 0 行）／repro_cmd ✅ 可执行（本审计实跑 a=16 T≤0.5 400 步 12.2 s） |
| 科学结论 | R2 的 **honest FAIL 成立**：全窗 T≤4 下 5 档网格仅 a=32 过 15 %，最小误差 13.8 % ≫ 3 %；H1（E_inf<3 %）**否证**（本审计用交付的 `fit_order` 复算：无可行 (E_inf,p) 解） |

## 1. 三件套逐项核验

| 项 | 判据 | 结果 |
|---|---|---|
| run_artifacts | 文件真实存在、内容与声称一致 | ✅ `xval/oracle_minimal_d2q9_fs/artifacts/r2_cj3032_maxima.json`（12 行标量）＋ `figures/fig_r2_error_vs_a_measured.png`（122 034 B）＋ `docs/XVAL_LBM_ORACLE_MINIMAL_D2Q9_FS_R2_RESOLUTION.md`，均在 `578410df6` 内。**12 行数值与 `cj3032.log` 的 12 条汇总行逐字段相等**（`audit/m20976/evidence/job_log_recovery.json`）⇒ 抽取链无篡改 |
| benchmark ref vs got | `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS'` | ❌ **0 行**（全表 133 行/94 锚点，本锚从未登记）。`tensor_xval_cases`、`repro_pool` 同锚亦 0 行 ⇒ **机器裁判缺席**，只能以任务自带锚 M&M 1952 + 作者预声明 15 % 门作替代判据（与 m18218 同） |
| repro_cmd | 可执行、无隐参 | ✅ `python xval/oracle_minimal_d2q9_fs/resolution_study.py --plan|--run|--analyze`（:602 main，:344 plan，:480 analyze）。本审计实跑最小档 `run_tensorlbm(a=16, T≤0.5)`：400 步、12.21 s、`Z_cell` 18.103 %／`Z_sub.05` 18.137 %，与声明 steps=ceil(T/√(g/h0)) 一致；全窗重算递交调度作业 **cj3101**（见 §5） |
| 报告六段 | ①案例②方法③数字表④图⑤注意点⑥误差判定 | ✅ 六段齐全（`tensor_reports.id=18989`，4069 字符；树内 md 6350 B 同文）；⑥显式回答「<15 %？仅 a=32」「<3 %？否」——**honest** |

## 2. 基准判据（唯一裁判：预声明 15 % 门；裁判表缺失已记 WITHHELD）

关键量 `Z_sub(0.05)`／`Z_cell` 相对 Martin & Moyce (1952) 全窗 T∈[0.5,4] 的最大相对误差（交付工件 = 作业回执）：

| a | steps | Z_cell % | Z_sub.05 % | u_max | inv drift % | <15 % | <3 % |
|---|---|---|---|---|---|---|---|
| 16 | 3200 | 28.16 | 27.39 | 0.2377 | 0.0322 | ✗ | ✗ |
| 24 | 2133 | 16.67 | 15.88 | 0.2406 | 0.0585 | ✗ | ✗ |
| 32 | 1600 | 13.81 | 14.66 | 0.2442 | 0.0531 | ✓(仅此档) | ✗ |
| 48 | 1070 | **28.05** | **27.37** | 0.2517 | 0.0783 | ✗ | ✗ |
| 64 | 203 | 42.86 | 42.19 | 0.2587 | 0.0991 | ✗ | ✗ |

* **误差非单调**（28.16→16.67→13.81→28.05→42.86），细化后变差 ⇒ 不支持「网格无关 <15 %」。
* **H1 否证（本审计独立复算）**：把 5 个梯级最大误差喂给交付的 `fit_order`，`Z_cell` 与 `Z_sub.05` 均返回
  `ok=false, reason="no admissible (E_inf, p) in grid"` ⇒ 不存在满足约束的收敛拟合（`audit/m20976/evidence/fit_and_richardson.json`）。
* **Richardson 外推不可交付**：`richardson()` 需要逐 T `points`，工件只存标量 → `NOT_COMPUTABLE_FROM_ARTEFACT (KeyError:'points')`。
  预声明（设计 §②3）的拟合与外推**在交付物上无法产出**，报告也未给 p / E_inf 数值。

## 3. 级别判定（L0–L5）

* **L2 成立**：`resolution_study.py` 690 行真代码（`:173 run_tensorlbm` 调**生产求解器** `tensorlbm.free_surface_lbm_2d.free_surface_step_2d`，`:238 run_oss_port`，`:278 hydrostatic_control`，`:328 run_task`，`:403 fit_order`，`:438 richardson`，`:480 analyze`，`:602 main`），另有 `collect_job_stdout.py`、`make_figures_r2.py`、`publish_r2_report.py`；输入→计算→输出闭环，KPI 与作业回执逐字段一致。
* **L3 不成立**：交付物只是验证 harness，仓内**零反向调用点**（`grep -rn resolution_study` 仅命中自身；被测 `free_surface_step_2d` 属既有能力），未接入主求解器入口／平台／CI；且**未合入 main**（见 F5）。
* **L4 不成立**：①裁判表 0 行（无可登记的 ref/got 判据行）；②精度门 1/5 档通过、最小误差 13.8 % ≫ 3 %；③误差分析所需的逐 T 序列不在工件内，拟合/外推不可复算；④测试红（F4）。
* **L5 不适用**。

## 4. 缺陷清单（严重度排序）

| # | 严重度 | 认定 | 证据 | 影响 / 处置 |
|---|---|---|---|---|
| F1 | **major（报告数字）** | 报告 §3.1 与 §⑥ 的 **a=48 行打印的是 OSS port 臂的数字**（79.37/70.05/0.8165/609.79），该梯级真值 28.05/27.37/0.2517/0.0783 | `r2_cj3032_maxima.json` 中 `tensorlbm_koerner_a48_tau1` vs `oss_port_a48_tau1`；`cj3032.log` 汇总行同；报告 §⑤ 正文自己用 28.05 | 结论不变（两者都 ≫15 %）但**验收表本身错位**、与正文自相矛盾；属「报告好看不算数」靶心，必须修表后重发 |
| F2 | **major（证据链）** | 预声明的观测阶拟合与 Richardson **未交付**，且交付物上不可算 | §2 复算；`richardson` KeyError；报告无 p/E_inf 数值，H1 否证仅以「非单调」定性论证 | 误差分析不完整 ⇒ 不能声称 L4；下一轮必须落盘逐 T 序列（`runs.json`）再算 |
| F3 | **blocking（裁判）** | 锚点在 `tensor_benchmark_cases` 仍 **0 行**（m18218 已提「若要冲 L4 需注册」，R2 未做） | 上述 SQL 三表皆 0 行 | gate reopened 的结构性原因之一 |
| F4 | **blocking（仓内红测）** | `tests/test_xval_oracle_minimal_d2q9_fs.py::test_tensorlbm_arm_smoke` **仍失败**（1 failed / 4 passed，5.89 s），`run_xval.py:137 ModuleNotFoundError`（sys.path 顺序污染） | 本审计实跑 pytest；m18218 F3 已开单，R2 未修 | main 上 CI 红；返修项 |
| F5 | **major（入库）** | 交付只存在于 `refs/heads/agent/m20893`（`578410df6`），**main 树无** `docs/XVAL_..._R2_RESOLUTION.md`、`xval/.../resolution_study.py`；而 `mission.result.pr_gate` 记 `PR 2418 merged` | `git merge-base --is-ancestor`=NO；`for-each-ref --contains` 仅 agent/m20893 | 与树证据矛盾，需平台侧核对 PR 号归属；未合入即不能计消账 |
| F6 | moderate（取证） | 声称「逐 T 序列丢失」：`cj3032` rc=1 崩在 `KeyError:'label'`（保留 tail 仅 1561 B，无 payload）；但 `cj3062` tail 内**仍存** a=32 τ=0.8/1.2/1.4 的逐 T 序列 ⇒ 丢失范围应限定为**梯级臂**；声明的补算 `cj3085/cj3087` 在盘上**无日志**，从未落地 | `cluster/jobs/` 实查 | 报告⑤6 表述需限定范围；补算须重递 |
| F7 | minor（数字） | §①「a=64→6400 步」错：`steps=ceil(T/√(g/h0))`，h0=2a ⇒ a=16→3200、**a=64→203**（差 32×） | `:220 n_steps = ceil(t_max/dtT)`，`dtT=sqrt(g/h0)`；本地实跑 a=16 T≤0.5→400 步自洽 | 参数表错误，不影响 KPI |
| F8 | minor（外部锚） | `DESIGN` 取 M&M 工件自带 `sha256=e6300b9a…`，而文件实算 = `6f766bfa461dc0d7…`（m18218 F8 的同一处过期摘要，R2 以程序方式继承） | `sha256sum src/artifacts/martin_moyce_1952_dam_break.json` | 外部锚溯源不可核；改为现算 |
| F9 | minor（控制组） | 静水控制组仍用「半域宽水柱」初始化 ⇒ 是溃坝不是静水层，1e-6 伪流门不适用；报告如实记 FAIL 但未修 | `:278 hydrostatic_control`；cj3062 tail 两条 control `pass:false`，u_max 0.2395 / 0.8165 | 下轮改满域静水层 |
| F10 | 正面 | headline 与 ⑥ 无粉饰：明写「不能声称网格无关 <15 %」「<3 % 否」「H1 否证」；a=32 13.81 % 与 R1 13.806 % 同量级 | 报告 ⑥／`r2_cj3032_maxima.json` | 按 honest-FAIL 计，不作负面计入 |

## 5. 本审计实际跑了什么

| 检查 | 结果 |
|---|---|
| 作业回执重建（`audit/m20976/parse_job_logs.py`） | `cj3032` 12 条汇总行全解析 ＆ 与工件 12 行逐字段相等；`cj3062` rc=0、tail 8 000 B 内含 τ 臂逐 T 序列；`cj3085/cj3087` 无日志 |
| 拟合/外推复算（`audit/m20976/fit_from_artifact.py`） | `fits.Z_cell/Z_sub0.05 = ok:false（no admissible (E_inf,p)）`；`richardson = NOT_COMPUTABLE_FROM_ARTEFACT` |
| pytest | `tests/test_xval_oracle_minimal_d2q9_fs.py -q` → **1 failed, 4 passed in 5.89 s**（F4） |
| repro 实跑（本地短测，1 核） | `run_tensorlbm(a=16, T≤0.5)`：400 步 12.21 s，`Z_cell` 18.103 %／`Z_sub.05` 18.137 % |
| 全窗独立重算（调度作业 cj3101） | 递交 `python repro/verify_r2.py`（a=16/32/48/64 四档重跑，比 28.16/13.81/28.05/42.86）→ 截至本单收尾仍在排队，结果如实标 **not_run**（见 evidence/repro_check.json 若已回传） |

**checks not run**：cj3101 全窗四档重跑（排队中）、cj3085/cj3087（无工件可核）、`--analyze` 全链（缺逐 T 序列）、MPI/集群双节点（本单无需）。

## 6. 处置

**request changes（reopened）**，级别维持 **L2**（不因百页文本升，也不因 honest FAIL 降）。返修项：
1. 修报告 §3.1/§⑥ 的 a=48 行（28.05/27.37/0.2517/0.0783）并重发报告（F1）；修 §① steps 行（a=64→203）（F7）。
2. 落盘逐 T 序列（`runs.json`）后补出 `fit p/E_inf` 与 Richardson，或显式撤回该预声明分析（F2）。
3. 修 `tests/test_xval_oracle_minimal_d2q9_fs.py::test_tensorlbm_arm_smoke` 的 sys.path 顺序污染（F4）。
4. 把 `578410df6` 真合入 main 并核对 PR 2418 归属（F5）。
5. 若要冲 L4：向 `tensor_benchmark_cases` 注册本锚 `case_name/metric/tolerance` 行，并把 `sha256` 改为现算值（F3/F8）；同时说明 48/64 档误差反转的机理（否则「网格无关」永久不可声明）。

*独立性限制*：审计者未参与 m20893；对象工作树不可访问，只能核交付 commit 与作业回执；本审计的重算臂在单机单线程重跑，非原 12 worker 环境。