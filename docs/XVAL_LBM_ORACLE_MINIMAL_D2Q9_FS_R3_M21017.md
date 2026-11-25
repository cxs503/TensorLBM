# XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS · R3（m21017 返修轮1）：指标定义审计 + 逐 run 工件/可复现链路硬化

**锚点** `XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS` · 仓 `TensorLBM` · 轮次 R3 · 承接 R2（m20893，`cj3032/cj3062`）
**本轮性质**：返修轮（原单交付经 maturity_gate 复核仅 L1：impl_files/entry/run_artifacts 三键无机器可核工件）。
本轮**不重做**物理扫描，而是把交付变成机器可核的形态，并修掉审计列出的阻塞项（red tests、0 行基准登记、逐 T 序列丢失）。

## ① 案例介绍
二维矩形水柱瞬时坍塌（Körner/Thürey 自由面 D2Q9 + BGK）：水柱宽 `a`、高 `h0=2a`，域 `nx=5a`、`ny=3a`（顶部 1a 气相裕量），
四边固壁、无表面张力；`g=5e-5`、`τ=1`。无量纲 `Z=x_front/a`、`T=t·√(g/h0)`。
基准：Martin & Moyce (1952) 实验前锋（[Martin–Moyce digitization](../src/tensorlbm/verification/refdata/martin_moyce_1952_dam_break.json)，17 个数字化点）。
本轮算例：`a=16`（`nx=80, ny=48`），`steps=3200`，覆盖声明窗口 `T∈[0,4]` 全程；另有 CI 档 `a=16, T≤1`。

## ② 怎么做的（本轮实际改动）
1. **指标定义审计（新发现，实测）**：冻结的 R1 指标 `front_cells()/a` 取「最右 LIQUID|INTERFACE 列」，
   而 `init_flags_from_fill_2d` 在液块正前方放了一列**空的** INTERFACE 包络 ⇒ `Z(t=0)=(a+1)/a`，
   即 +1 格（`+1/a`，a=16 时 `+0.0625`）的系统偏高。`resolution_study` 的 fill 判据（`fill>1e-6`）给出 `Z(t=0)=1` 正确值。
   两个定义现在**并列记录**：`run_xval.py` 保留 `Z`（R1 可比）并新增 `Z_fill`；`resolution_study.py` 新增 `Z_cell_flag`（legacy 定义）。
   若断言只写 `Z(t=0)==1`，本轮之前会红——这正是原 `test_tensorlbm_arm_smoke` 变红的原因（口径未被声明）。
2. **逐 run 工件 + 日志级序列回收**：`--run` 现在为每个 run 落 `artifacts/runs_r3/<label>.json`，
   并打印一行 `R2SER:`（含全 T 序列 `[T,Z_cell,Z_cell_flag,Z_sub.02,Z_sub.05,Z_sub.1]`）；
   `collect_job_stdout.py` 从 `cluster/jobs/cjN.log` 还原成同样的 per-run json，`--analyze` 可合并多作业（分片可组合）。
   这直击 R2 的取证坑（只有标量极大值活下来、逐 T 序列丢失）。
3. **入口**：新增仓内入口 `benchmarks/xval_oracle_minimal_d2q9_fs/run.py`（`make xval-oracle-d2q9` / `--full`），
   它 import `xval.oracle_minimal_d2q9_fs.resolution_study` 并驱动生产求解器 `tensorlbm.free_surface_lbm_2d.free_surface_step_2d`，
   产出可判工件 `result.json`（含 ref_vs_got 与 gate 判定，退出码即判定）。
4. **修红测试**：`tests/test_xval_oracle_minimal_d2q9_fs.py` 从「1 failed/4 passed」→「8 passed」：
   (a) `test_port_matches_upstream_snapshot` 的根因是 OSS 快照 `lbm/` 无 `__init__.py`（命名空间包），
   被 sys.path 上另一 checkout 的**常规** `lbm` 包（editable install）抢走 ⇒ `No module named 'lbm.util'`；
   修法是给 `oss_original.import_oss` 装一个 `__path__` 指向快照的合成包模块，把解析钉死（不是加 skip 蒙混）。
   (b) `test_tensorlbm_arm_smoke` 断言 `Z(t=0)=1`，与未声明的 +1 格口径冲突；现按两种口径分别精确断言。
5. **基准登记**：`tensor_benchmark_cases` 该锚点由 **0 行 → 1 行**（id=141，
   `case_name=dam_break_lbm_fs_martin_moyce`, tolerance 15, repo TensorLBM, status pending），机器裁判不再是 WITHHELD。

## ③ 计算结果（a=16, τ=1, T∈[0.5,4]，程序化注入自 `artifacts/r3_a16_summary.json`）

| T | Z_ref(M&M) | Z_cell | Z_cell_flag | Z_sub.05 | err_cell % | err_flag % | err_sub.05 % |
|---|---|---|---|---|---|---|---|
| 0.5 | 1.45 | 1.1875 | 1.1875 | 1.187 | -18.1 | -18.1 | -18.14 |
| 0.75 | 1.65 | 1.3125 | 1.3125 | 1.3197 | -20.45 | -20.45 | -20.02 |
| 1.0 | 1.85 | 1.4375 | 1.4375 | 1.473 | -22.3 | -22.3 | -20.38 |
| 1.25 | 2.025 | 1.625 | 1.625 | 1.6087 | -19.75 | -19.75 | -20.56 |
| 1.5 | 2.2 | 1.75 | 1.75 | 1.7388 | -20.45 | -20.45 | -20.96 |
| 1.75 | 2.36 | 1.875 | 1.875 | 1.8579 | -20.55 | -20.55 | -21.27 |
| 2.0 | 2.52 | 1.9375 | 1.9375 | 1.9691 | -23.12 | -23.12 | -21.86 |
| 2.25 | 2.66 | 2.0625 | 2.0625 | 2.0511 | -22.46 | -22.46 | -22.89 |
| 2.5 | 2.8 | 2.125 | 2.125 | 2.143 | -24.11 | -24.11 | -23.46 |
| 2.75 | 2.925 | 2.1875 | 2.1875 | 2.2192 | -25.21 | -25.21 | -24.13 |
| 3.0 | 3.05 | 2.3125 | 2.3125 | 2.2862 | -24.18 | -24.18 | -25.04 |
| 3.25 | 3.165 | 2.375 | 2.375 | 2.3502 | -24.96 | -24.96 | -25.75 |
| 3.5 | 3.28 | 2.4375 | 2.4375 | 2.4117 | -25.69 | -25.69 | -26.47 |
| 3.75 | 3.38 | 2.4375 | 2.4375 | 2.4707 | -27.88 | -27.88 | -26.9 |
| 4.0 | 3.48 | 2.5 | 2.5 | 2.5267 | -28.16 | -28.16 | -27.39 |

逐项误差判定（关键量 = 窗口内 `max|rel err|`）：

| 量 | 值 | <15 %？ | <3 %？ |
|---|---|---|---|
| `Z_cell`（物理格前锋） | 28.16 % | 否 | 否 |
| `Z_cell_flag`（legacy 口径） | 28.16 % | 否 | 否 |
| `Z_sub(0.05)`（亚格子趾） | 27.39 % | 否 | 否 |
| CI 档（T≤1, 入口 `run.py`）`Z_sub0.05` | 20.378 % | 否 | 否 |
| 质量账漂移（液量） | 0.0322 % | 是（门 1 %） | — |
| u_max | 0.2377（Ma≈0.05） | 是（门 0.5） | — |

**本轮结论：`a=16` 在全窗口仍 FAIL（27.39 % ≫ 15 %），与 R2 的 28.16/27.39 % 逐位一致（同代码同参数复现）。**
R2 的核心否证结论不变：误差随 `a` 非单调（a=16→32 降到 13.81 %，a=48/64 回升），H1（E_inf<3 %）被否证。

## ④ 结果图
![r3 a16](../xval/oracle_minimal_d2q9_fs/figures/fig_r3_a16_series.png)
*左：前锋 `Z(T)` 与 M&M 数字化点（cell / 亚格子趾两种度量）；右：三个度量定义各自的 |相对误差|，红虚线 15 % 验收、绿点线 3 % 目标。*

## ⑤ 注意点（踩过的坑 / 局限）
1. **+1 格是指标口径，不是物理偏差**：实测 `Z_cell_flag(t=0)=(a+1)/a`。在 `T≥0.5` 的判定窗口内两者最大误差同值（28.16 %），
   说明**R2 的 13.81 %/28.16 % 不是该口径造成的**；但 a=16 时 1 格 = `Z` 的 0.0625，相对 `Z_ref≈1.45` 约 4.3 %，报告必须带口径声明。
2. **审计环境差异**：OSS 快照目录 `/root/oss-cache/Maarten-vd-Sande_lbm` 存在，但 `lbm/` 是命名空间包；
   本机还装了**另一 worktree 的 editable `lbm`**，这是红测试的真因（非本仓代码缺陷，但可在仓内彻底消除）。
3. **R2 报告的 §3.1 表错行**（审计 F1：a=48 行混入 OSS port 数值）本轮未重发 R2 正文；R3 表格全部由脚本从工件注入，无手抄。
4. **分片复算未落地（honest）**：为补齐 R2 承诺的 a=24..64 阶梯与 τ 扫描，本轮递交了 `cj3125`（阶梯 5 臂）/`cj3126`（τ 扫描+控制组），
   递交时队列有 7-9 个在跑作业，本单签收时两作业**仍在排队、无日志**，故本报告不含其数据；其取证路径已就绪（`R2SER:` + `collect_job_stdout.py`）。
5. **范围**：本轮的实体增量是「交付形态 + 指标口径 + 红测试 + 基准登记」，不是新的物理结论。

## ⑥ 误差判定（显式回答）
- 关键量 `Z_sub(0.05)` vs Martin & Moyce 的最大相对误差：**27.39 %（a=16, T∈[0.5,4]）**
- 是否 <3 %：**否**（差约 9.1 倍）；是否 <15 %：**否**
- 复现性：本轮 a=16 结果与 R2（`cj3032` 纪录 28.16 %/27.39 %）**逐位一致**，即误差是模型/离散口径的真实水平，不是环境噪声
- 无基准的项：`cj3125/cj3126` 的 a=24..64 与 τ 扫描**无数据**，按 WITHHELD 处理，未做任何声明

## 复现命令
```
python xval/oracle_minimal_d2q9_fs/resolution_study.py --run --only a16 --workers 1 --tag smoke_r3.json   # 65 s, 1 核
python xval/oracle_minimal_d2q9_fs/resolution_study.py --analyze --tag smoke_r3.json
make xval-oracle-d2q9        # 入口 is CI 档，产出 benchmarks/xval_oracle_minimal_d2q9_fs/result.json
python -m pytest tests/test_xval_oracle_minimal_d2q9_fs.py -q                                            # 8 passed
```
