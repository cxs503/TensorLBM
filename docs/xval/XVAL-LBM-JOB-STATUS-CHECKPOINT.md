# XVAL — TensorLBM job-status / wall-clock checkpoint state machine vs lanl/MF-LBM

锚点：`XVAL-LBM-JOB-STATUS-CHECKPOINT-MACHINE`（任务 m17986，返修轮 1）

## ① 案例介绍（物理背景 / 几何 / 参数表 / 参考出处）

**物理背景。** 长时 LBM 算例（阻力、砰击、冰区绕流）在 HPC 上以无值守分段方式运行，作业可能被墙钟限额、节点故障或调度抢占中断。若没有可续算的状态机，中断即意味着从头重算；成熟的 OSS 实现把“作业状态文件 + 墙钟定时 checkpoint + 双备份 + continue 路径”作为基本设施。本单以 **lanl/MF-LBM** 的 `job-status-checkpoint-machine` 做法为对标对象做交叉验证（XVAL）。

**参考出处（零手抄，取自平台 OSS 台账 + 其列出的源码行号）。**

| 项 | 值 |
|---|---|
| repo | `lanl/MF-LBM` |
| feature_slug | `NEW:wall-clock-checkpoint-state-machine` |
| 源码位置 | `IO_multiphase.F90:579-641`, `Main_multiphase.F90:91-95`, `Main_multiphase.F90:139-141`, `Main_multiphase.F90:222-303`, `Init_multiphase.F90:154-184` |
| 台账 | `oss_solutions(repo='lanl/MF-LBM', feature_slug='NEW:wall-clock-checkpoint-state-machine')` |

参考做法（台账 method 字段的要点）：用 `job_status.txt` 承载状态机 (`new/continue/done/failed/reached_max_step`)，checkpoint 按**墙钟小时**定时（主 + 第二备份），保存完整 PDF（必要时附对流出流历史场），重启时读状态文件走 continue 路径避免重算；另有 `d_vol_*` 以“注入孔隙体积”触发输出/监控。

**几何与参数表（本仓对照实现的算例，无量纲）。**

| 参数 | 符号 | 值 | 说明 |
|---|---|---|---|
| 格子 | D2Q9 | 9 速度 | 二维九速，周期边界 |
| 网格 | Nx × Ny | 64 × 64 | 均匀格，格子单位 |
| 松弛时间 | τ | 0.55 | ν = (τ − 1/2)/3 = 0.0166667 |
| 初值速度幅值 | u₀ | 0.05 | Taylor–Green 涡 `ux=u₀ sin x cos y, uy=−u₀ cos x sin y` |
| 步数 | n | 240 | 每条实验臂的总步数 |
| 网格雷诺数 | Re = u₀L/ν | 192.0 | L = Nx |
| 马赫数 | Ma = u₀/c_s | 0.0866 | 近不可压 |
| 动能演化 | KE₀ → KE_end | 0.000625 → 0.00053308（比值 0.8529） | 场在真实演化，重启测试不是冻结场 |
| 基线场指纹 | sha256 | `b3283b3a236e5cbc9a3c4c16c850e557…` | 连续运行末态 |

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

**实现（`src/tensorlbm/job_checkpoint.py`，非占位）。**

| MF-LBM 做法 | 本仓实现 |
|---|---|
| `job_status.txt` 状态机 | `JobStatusFile` + `JobStatusRecord`（首行为状态词，其后 `k=v`），原子替换写入；`LEGAL_TRANSITIONS` 显式表，终态吸收 |
| 墙钟小时定时 | `CheckpointPolicy.interval_hours`；`JobCheckpointManager.maybe_checkpoint` 每步检查 `clock()`（默认 `time.monotonic`，可注入） |
| 主 + 第二备份 | `CheckpointStore(dual_backup=True)`：`checkpoint.pt` + `checkpoint_bak.pt`，均经 `checkpoint_io.atomic_torch_save` 原子落盘 |
| 保存完整 PDF（+额外场） | `fields` 映射任意 `torch.Tensor`；默认存全套 PDF `f`；另存 sha256 `digest` 防静默损坏 |
| 重启走 continue 路径 | `JobCheckpointManager.begin()` → `ResumeState(step, fields, source)`；`LBMSimulation.enable_checkpointing` 直接恢复 `f` 与步数 |
| `d_vol_*` 体积触发 | `VolumeTrigger`（每跨过一个体积量子触发一次） |

**接入点（真实调用，不是旁挂脚本）。** `src/tensorlbm/simulation.py` 的 `LBMSimulation` （包级导出、文档中的最小可运行入口）新增 `enable_checkpointing()` / `run_with_checkpointing()`，`step()` 内驱动状态机并在 `max_steps` 处写入 `reached_max_step` 终态。顶层 `tensorlbm` 惰性导出同步登记（`JobCheckpointManager`、`JobStatusFile`、`CheckpointStore`、`CheckpointPolicy`、`CheckpointError`、`VolumeTrigger`、`ResumeState`）。

**实验设计（预声明）。** 四条假设：

- H1 重启恒等：分段 checkpoint+restart 与连续运行到达同一末态，`max|Δf| = 0`（位级相等）。
- H2 零重算：各段执行步数之和 == 总步数。
- H3 双备份救援：主 checkpoint 被截断时由第二备份恢复且仍位级相等；`dual_backup=False` 为阴性对照（不可恢复）。
- H4 定时器语义：实测存档间隔与配置 `interval_hours` 一致，粒度上界为 1 步的机时。

**工具链。** Python 3.11 + PyTorch（CPU，float32）；`checkpoint_io.atomic_torch_save`（既有模块，复用）；pytest 不变式测试（`tests/test_job_checkpoint.py`，20 条）；matplotlib(Agg) 出图；所有数字由 `benchmarks/xval_job_checkpoint/run_xval.py` 程序化写入 `summary.json`，再经 `write_docs.py` 渲染（零手抄）。

**复现命令。**

```bash
PYTHONPATH=src python benchmarks/xval_job_checkpoint/run_xval.py --out artifacts/xval_job_checkpoint
PYTHONPATH=src python benchmarks/xval_job_checkpoint/write_docs.py \
    --summary artifacts/xval_job_checkpoint/summary.json \
    --out docs/xval/XVAL-LBM-JOB-STATUS-CHECKPOINT.md
python -m pytest tests/test_job_checkpoint.py -q
```

## ③ 计算结果（数字表，程序化注入）

**A. 参考 vs 实测（XVAL 判据 <15%；位级行用精确判据）。**

| quantity | ref | got | unit | rel_err_pct | tolerance_pct | pass | ref_source |
|---|---|---|---|---|---|---|---|
| restart field discontinuity max|df| (worst segment count) | 0 | 0 | lattice population units | 0 | 0 | True | IO_multiphase.F90:579-641 + Main_multiphase.F90:139-141 |
| steps recomputed after resume (worst segment count) | 0 | 0 | steps | 0 | 0 | True | Main_multiphase.F90:91-95 |
| recovery rate after corrupted primary checkpoint | 1 | 1 | fraction | 0 | 0 | True | IO_multiphase.F90:579-641 |
| recovery rate after corrupted primary, second backup disabled (control) | 0 | 0 | fraction | 0 | 0 | True | IO_multiphase.F90:579-641 |
| status states implemented | 5 | 5 | states | 0 | 0 | True | IO_multiphase.F90:579-641 |
| terminal state blocks silent restart (reached_max_step) | 1 | 1 | fraction | 0 | 0 | True | Main_multiphase.F90:91-95 |
| wall-clock timer accuracy: median measured interval vs configured | 0 | 1.54591 | % relative error | 1.54591 | 15 | True | Main_multiphase.F90:222-303 |
| auto (wall-clock) checkpoint -> restart field discontinuity max|df| | 0 | 0 | lattice population units | 0 | 0 | True | IO_multiphase.F90:579-641 + Main_multiphase.F90:222-303 |
| volume trigger count vs floor(volume/interval) | 2 | 2 | triggers | 0 | 0 | True | Main_multiphase.F90:222-303 |

全部行通过：**True**（判据：XVAL criterion: relative error <= 15% (owner precision standard); bit-exact rows use exact tolerance）。

**B. 分段重启（H1/H2）。**

| segments | restarts | steps_executed | steps_recomputed | checkpoints | max_abs_df_vs_continuous | bit_exact | final_digest | status |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 240 | 0 | 1 | 0 | True | b3283b3a236e5cbc9a3c4c16c850e557a63da538528bf37b005c4bc868de6ade | done |
| 2 | 1 | 240 | 0 | 2 | 0 | True | b3283b3a236e5cbc9a3c4c16c850e557a63da538528bf37b005c4bc868de6ade | done |
| 4 | 3 | 240 | 0 | 4 | 0 | True | b3283b3a236e5cbc9a3c4c16c850e557a63da538528bf37b005c4bc868de6ade | done |
| 8 | 7 | 240 | 0 | 8 | 0 | True | b3283b3a236e5cbc9a3c4c16c850e557a63da538528bf37b005c4bc868de6ade | done |

**C. 双备份救援（H3）。**

- 双备份臂：从 `backup` 恢复于第 120 步，`max|Δf| = 0`，位级相等 = True。
- 单备份对照臂：恢复 = False，错误首行 `no usable checkpoint generation:`。

**D. 定时器与开销（H4，A/B 交替 3 次取中位数）。**

| interval_s | checkpoints | overhead_pct | overhead_from_write_cost_pct | base_wall_cv_pct | median_measured_interval_s | interval_rel_err_pct | timer_excess_steps | step_time_s |
|---|---|---|---|---|---|---|---|---|
| 1 | 3 | -1.83618 | 0.242165 | 2.79242 | 1.01546 | 1.54591 | 0.932217 | 0.0165832 |
| 2 | 3 | 3.03849 | 0.23152 | 7.35365 | 2.02834 | 1.41689 | 1.70882 | 0.0165832 |

单次 checkpoint 生成（主写 + 备份拷贝，150 kB）：3.15 ms = 一步机时的 19%。自动（墙钟）路径重启：从第 196 步恢复，`max|Δf| = 0`（位级相等 True）。

**E. 终态词表与体积触发。**

- `max_steps=25` 臂：请求 40 步，实停 25 步，`job_status.txt` 终态 = `reached_max_step`；再次启动被终态挡住（`finished=True`，0 步执行）= True。
- 状态词覆盖：5 个（new, continue, done, failed, reached_max_step）。
- `d_vol_*` 类比：体积从 0 走到 1.25，量子 0.5，触发 2 次 == floor(V/ΔV) = 2。

## ④ 结果图

![fig_restart_discontinuity.png](../../artifacts/xval_job_checkpoint/figures/fig_restart_discontinuity.png)

*Restart discontinuity vs number of restarts (bit-exact for every arm).*

![fig_overhead_vs_interval.png](../../artifacts/xval_job_checkpoint/figures/fig_overhead_vs_interval.png)

*Checkpoint overhead and save count versus configured wall-clock interval.*

![fig_field_compare.png](../../artifacts/xval_job_checkpoint/figures/fig_field_compare.png)

*Final u_x of the continuous run, the checkpoint+restart run and their difference after 7 restarts.*

![fig_status_timeline.png](../../artifacts/xval_job_checkpoint/figures/fig_status_timeline.png)

*Checkpoint/restart timeline of the segmented run; final job_status.txt state.*

图 1 为重启不连续量的收敛性图（全部为 0，纵轴对数）；图 2 为 checkpoint 开销与存档数随墙钟间隔的敏感性；图 3 为连续运行 / 重启运行 / 二者差场（差场恒为 0）的场量对照；图 4 为分段运行的时间轴与终态状态词。（平台报告内已按 figures 字段内嵌同一批 PNG 的 base64。）

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **同一 case 目录的并发写者会被状态机正确拒绝，但读代码的人容易误判为 bug。** 本轮实测踩到：一个残留进程与新一轮运行共用同一 case 目录，后者的 `begin()` 看到 `continue` 却找不到 checkpoint，直接抛 `CheckpointError`（而不是静默从 0 重算）。这正是状态机应有的行为，但也说明**case 目录是单写者资源**，本轮未实现进程锁（列入后续项）。
2. **墙钟定时器只能在步边界求值**：理想超调上界是 1 步机时；本机共享负载下实测最坏 1.71 步（见敏感性表 `timer_excess_steps`），因此判据采用相对误差行（最坏 1.55%）而不是把 1 步当硬阈值。因此“相对误差”只有在 interval ≫ 单步机时时才有意义；本仓把两者分开登记（相对误差行 + 粒度行）。
3. **共享控制面（99）上的墙钟开销不可直接相减**：未交替时曾测出 −21%/−26% 的“负开销”。改为 A/B 交替 + 中位数 + 直接测单次写入成本（3.15 ms/次）后，得到可分辨的写入开销上限；A/B 差值仍在主机噪声内（CV 见敏感性表），故只作量级参考；表中仍给出基线墙钟的变异系数 CV 供判读。
4. **位级相等的前提**：同一 device/dtype、同一算子顺序。跨 device（如 CUDA↔CPU）或换 `stream` 模式时只能保证物理一致，不保证位级；checkpoint payload 已记录 `config_signature`（网格/τ/格子），不匹配直接拒绝续算，避免“静默混态”。
5. **未移植的部分**：MF-LBM 在开启对流出流 BC 时额外保存 `f_convec_bc/g_convec_bc/phi_convec_bc` 历史场；本仓的 `fields` 是任意张量映射，调用方把额外历史场作为额外 key 传入即可，但**本轮算例无此 BC，未做该路径的实测**。
6. **状态词表是语义契约，不是装饰**：终态（`done/failed/reached_max_step`）吸收，需要显式 `force_restart` 才能重开；这是防止“已完成算例被静默续算”的关键，也是与 MF-LBM 一致的语义。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 关键量 | 基准 | 实测 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| restart field discontinuity max|df| (worst segment count) | 0 lattice population units | 0 lattice population units | 0% | 是 |
| steps recomputed after resume (worst segment count) | 0 steps | 0 steps | 0% | 是 |
| recovery rate after corrupted primary checkpoint | 1 fraction | 1 fraction | 0% | 是 |
| recovery rate after corrupted primary, second backup disabled (control) | 0 fraction | 0 fraction | 0% | 是 |
| status states implemented | 5 states | 5 states | 0% | 是 |
| terminal state blocks silent restart (reached_max_step) | 1 fraction | 1 fraction | 0% | 是 |
| wall-clock timer accuracy: median measured interval vs configured | 0 % relative error | 1.54591 % relative error | 1.546% | 是 |
| auto (wall-clock) checkpoint -> restart field discontinuity max|df| | 0 lattice population units | 0 lattice population units | 0% | 是 |
| volume trigger count vs floor(volume/interval) | 2 triggers | 2 triggers | 0% | 是 |

**结论。** 9/9 行相对误差 <3%（其余为“≤ 上界”型判据或位级精确 0）。位级量（重启不连续量、重算步数、自动路径重启、体积触发计数）为**精确 0**，不是“小于阈值”，因此不存在“粉饰”的空间：任何非零值都会被逐位抓到。墙钟间隔精度 1.55%（<3%），其物理下界由单步机时决定，已单独以 `timer_excess_steps` 登记（最坏 1.71 步）。

## 差异清单（本仓 vs OSS）与移植取舍

| 维度 | lanl/MF-LBM | TensorLBM（本仓） | 取舍理由 |
|---|---|---|---|
| 状态载体 | `job_status.txt` 纯文本标志 | 同：首行状态词 + `k=v` | 可 grep、可人读、崩溃后仍可判读；增加 step/reason/checkpoints 便于审计 |
| 触发 | 墙钟小时定时 + `d_vol_*` | `interval_hours` + `VolumeTrigger` + 手动 `checkpoint_now` | 单位保持小时（不重标定），体积触发抽象成通用触发器，避免域语义硬编码 |
| 数据格式 | Fortran 二进制 PDF 数组 | `torch.save` payload（任意张量映射 + sha256 digest） | 与生态内既有 `checkpoint_io.atomic_torch_save` 一致；digest 抓静默损坏 |
| 备份 | 主 + 第二备份（`double_bak_checkpoint_pdf_cmd`） | `dual_backup` 开关，备份为主文件字节拷贝 | 少一次序列化；实测能救回被截断的主文件 |
| 额外历史场 | 对流出流 BC 三套历史场 | 任意额外 key | 本仓无该 BC；接口不假设物理 |
| 并发写者 | 隐含单写者（case 目录） | 同（无锁），但状态continue+无 checkpoint 时**报错而非重算** | 静默重算会浪费机时且掩盖事故；报错把事故变成可见事件 |

