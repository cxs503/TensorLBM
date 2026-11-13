# 交差核验报告 m18154 —— 对象 m17994 `XVAL-LBM-MPI-1D-SLAB-UNPACKED-HALO-TESTALL-OVERLA`

| 项 | 值 |
|---|---|
| 审计任务 | m18154（board-audit，非重做） |
| 被审任务 | m17994（TensorLBM，status=succeeded，runner=deepseek-v4-flash） |
| 被审分支 commit | `9ae15ea20918ff220210ae2958a2d5d4626b1018`（`git log -1` in `/root/tp-agent/work/m17994`） |
| 工作树 | `/root/tp-agent/work/m17994`（被审）/ `/root/tp-agent/work/m18154`（本单，main 检出） |
| 证据冻结 | `docs/audit_m17994/evidence/`（delivered 工件原件 + 实现/测试/驱动快照 + 复跑结果） |
| sha256 | `slab_unpacked.py` = `f76649e5…f477`；`test_parallel_slab_unpacked.py` = `017f5111…5973`；`run_xval.py` = `17352ad8…e77d`；复跑结果全文件 = `e6e85bc7…f371` |
| 环境 | python 3.11.6 / numpy 2.4.6 / loongarch64 / 128 核 / **无 mpi4py，无 mpirun、mpiexec** |
| 结论 | **L2（独立功能实现）**；delivered 声称隐含 L4（完整交叉对标）→ 差 2 级，主因=无集成调用方 + 无登记基准判据行 + OSS ref 未执行 |

---

## 1. 取证：三件套逐项核验

| 声称 | 核验结果 | 证据 |
|---|---|---|
| `run_artifacts` 文件存在 | **真** | `evidence/{RESULTS.md,kpi.csv,tables.md,fig1_field_and_profile.png(76KB),fig2_comm_accounting.png(51KB),fig3_overlap_and_polling.png(59KB)}`；原件路径 `/root/tp-agent/work/m17994/artifacts/xval_mpi_slab_unpacked_halo_overlap/` |
| `results.json` 数值 | **真，且逐项复现** | 复跑 `python xval/.../run_xval.py --out /tmp/audit17994/results.json` → 9 臂 × 6 个确定性 KPI **相对差 = 0**（figA）；3.3MB 全文件 sha256 `e6e85bc7…`，下表 6 个键值逐位同号 |
| `repro_cmd` 可执行 | **真** | `cd /root/tp-agent/work/m17994 && python xval/XVAL-LBM-MPI-1D-SLAB-UNPACKED-HALO-TESTALL-OVERLA/run_xval.py`（无参数即全量 300 步 @128²，9 臂 + 收敛臂 + 轮询臂 + overlap 臂，本机 ~65 s，`COMPLETE` 落盘） |
| `tests` 通过 | **真** | `python -m pytest tests/test_parallel_slab_unpacked.py -q` → **10 passed in 8.65 s**（两次独立运行均 10 passed） |
| `impl` 非占位 | **真** | `src/tensorlbm/parallel/slab_unpacked.py` 617 行：`UnpackedHaloExchange:386`、`PackedHaloExchange:449`、`SlabSolver.step:553`（Testall 轮询 583-591）、`shear_wave_analytic:69`、4 种传输替身 `SerialComm:293 / QueueChannel:161 / ThreadPeers:228 / SocketPeers:317` |
| `integration_callers` | **假（无调用方）** | `grep -rn "slab_unpacked\|tensorlbm.parallel" --include=*.py` 在 `src/` 仅命中 `parallel/__init__.py:3` 自身；`/root/tensor-ecosystem/TensorLBM` 该子包尚未合并。既有的 `domain_decomposition.py` 由 `blockforest.py:7,101,779` 调用，新模块与它并行重复而非接入 |
| `benchmark` 登记判据行 | **缺（无 oracle）** | `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-MPI-1D-SLAB-UNPACKED-HALO-TESTALL-OVERLA'` → **0 行**；`%SLAB%`/`%UNPACKED%`/`%MPI%` 在 `tensor_benchmark_cases`、`tensor_xval_cases`、`repro_pool` 皆 0 行。本锚在平台上**没有机器裁判行** |

### 1.1 确定性 KPI：delivered vs 审计复跑（全部读自 results.json，无手抄）

| arm | probe err % | 场 L2 % | identity vs NP=1 | wire B/步 | staging B/步 | 质量漂移 | 复跑一致 |
|---|---|---|---|---|---|---|---|
| unpacked_np1 | 0.0137799 | 0.0137799 | —（基准臂） | 32768 | 0 | 1.887e-14 | 0 差 |
| unpacked_np2 | 0.0137799 | 0.0137799 | **0.0** | 32768 | 0 | 1.887e-14 | 0 差 |
| unpacked_np4 | 0.0137799 | 0.0137799 | **0.0** | 32768 | 0 | 1.876e-14 | 0 差 |
| unpacked-selective_np2/4 | 0.0137799 | 0.0137799 | **0.0** | 24576 | 12288 | 1.887e-14 | 0 差 |
| packed-selective_np2/4 | 0.0137799 | 0.0137799 | **0.0** | 24576 | 12288 | 1.887e-14 | 0 差 |

- `32768/24576 = 1.33333…` = 8/6 闭式值 ✔；unpacked staging = 0 B ✔（fig2 的可信基础）。
- 收敛臂（等物理衰减时间，`convergence`）：N=64×75 步 0.055118%；N=128×300 步 0.013780%（复跑同值）。
- 判定：**误差 0.0138% ≪ 15%（owner 标准），亦 ≪ 3%**；H1/H2/H4 三条被判据支持。

---

## 2. 判级（L0–L5）：**L2**

| 级 | 判据 | 本单 |
|---|---|---|
| L0 文档 | 只有文档 | 越过（有 617 行真代码） |
| L1 数学/单元验证 | 测试断言不变量 | ✔ 10 tests：`assert_array_equal` 逐位相等（NP=2/4 × 3 种 halo 约定）、`convergence` 闭式字节比 `8/6`、剪切波 <1%、`|uy|max<1e-12`——是**不变量断言**，非「跑通即过」 |
| **L2 独立功能实现** | 输入→计算→输出真代码 | **✔ 达标**：SlabSolver 走完 collide→isend(零拷贝)→分块 stream+Testall→waitall→ghost 粘贴；9 臂 × 3 NP 场量输出；工件可复现 |
| L3 系统集成 | 接入主求解器/平台 | ✘ 零调用方（见 §1 `integration_callers`）；模块 docstring 自述 "verification instrument, not a performance port"，且与既有 `domain_decomposition` 功能重叠未归一 |
| L4 基准验证 | 完整算例+误差分析+可复现 | ✘（连锁失败）L3 未过；且 **平台无该锚基准裁判行**，外部 ref（OSS 二进制）**未执行**（无 mpi4py/mpirun，见 §5）——「XVAL 交叉对标」在数值上只对到「本仓 NP=1 自洽 + 解析剪切波」，对不到 OSS 输出 |
| L5 科研工程 | L4 + 外部锚（文献/实验/前后对照） | ✘ |

**与声称的差**：mission `checks` 把本单写成「预声明驱动…600 步 @128²」并给出完整对标表，隐含 L4 语义；实测功能实质为**独立验证模块（L2）**，差 2 级。差距不来自造假（所有 KPI 均可复现），而来自 ①未接入任何求解器 ②对标另一侧未被真正执行 ③平台侧无判据行，机器无法独立复核误差。

---

## 3. 严重度排序发现（adversarial review）

### F1（中·证据链完整性）步数与预声明不符，且交差文本写错
- `predeclared.json:predeclared_thresholds.stopping_rule` = 「every arm runs the step count fixed before the run (**600** at N=128)」；`results.json.config.steps` = **300**；`RESULTS.md` ③ 写「步数 300（固定，跑前声明，未因结果延长）」；mission `checks` 写「**600 步 @128²**，跑前写死步数未延长」。
- 复现：`python -c` 直调 `SlabSolver(128,128,tau=0.8)` → t=300 探针 0.013780%、t=**600** 0.014709%（`evidence/audit_step_amendment_check.json`）。**物理结论不受影响**（两者都 ≪3%），但「预声明 600 却跑 300」未在预声明处留 amendment，交差文本还反向写成 600 —— 属 predeclaration 纪律缺陷，非数值错误。
- 处置：要求补 amendment（把 600→300 的理由与两次步数误差并列写进 predeclared 的 `amendment` 段）；交差文本以工件为准更正。

### F2（中·声明强度）"语义精确移植"的对照侧从未执行
- 无 mpi4py/mpirun（本机实测 `ModuleNotFoundError`、`no mpirun`），OSS `lbm-principles-practice/code` 的 C++/CUDA+MPI 实现**未被执行**；ref 是"从平台 OSS 台账读到的做法"（`oss_solutions` 行，零手抄），对照臂是**自写替身**（`QueueChannel` 线程队列 / `SocketPeers` socketpair）。
- 因此 H1（identity=0）证明的是**本仓自洽**（替身 vs 本仓 NP=1），不是"与 OSS 实现逐位一致"。报告 ① 已诚实标注该限制（记为 comparison limit），故**不判造假**；但锚题「交叉对标」在证据分级上应记 **ref=WITHHELD（未执行）**，R=「语义移植+解析锚」，不是 REAL 双实现对照。

### F3（中·可复现性）墙钟类结论不可复现，量级跨运行漂移 2.76×
- 同机复跑：MLUPS 最大比值 **2.763×**（`evidence/audit_timing_spread.json`，figC 右）；overlap/blocking speedup delivered {np1 1.161, np2 0.475, np4 0.456} vs 复跑 {1.102, 0.205, 0.420}（**符号稳定、量级不稳**）；轮询扫描 delivered {1行 17.53, 8行 13.45, 64行 16.25 s} vs 复跑 {11.37, 6.11, 6.32 s}——**"64 行比 8 行慢"的排序在复跑中翻转**。
- 结论：`kpi.csv` 的 `mlups` 列与 ④⑤ 的墙钟表格**不能作为跨 arm 性能结论**；H3 的 honest-negative 判读方向可保留（替身限制下 overlap 无收益），但"轮询越密越慢"应降级为"轮询次数随粒度单调（29.3↔3.45↔1.00 次/步，这一条复跑复现），墙钟差落在噪声带内"。
- 处置：报告须给墙钟数加噪声带/重复次数（本单 n=1）。

### F4（低-中·API 隐患）七标量归约里塞了 max 型量
- `slab_unpacked.py:541-556 local_scalars()` 第 7 项 `max|u|` 走 `allreduce_sum`（`:554 global_scalars`）→ 返回值实为"各 rank 局部 max 之和"（NP=4 得 0.258 而真值 0.064）。报告 ⑥ 已在文字里披露，但**公开 API 仍以 `global_scalars()` 暴露该语义错误值**，后续调用方会踩。
- 建议：拆 `allreduce_max`（或键名改 `sum_local_max_norm` 并在 docstring 标红）。

### F5（低·重复实现）与既有 `domain_decomposition.HaloExchange` 未归一
- 新模块复刻了 `PackedHaloExchange`（selective+staging）作为对照臂，与仓内 `domain_decomposition.py`（已被 `blockforest.py` 调用）并存；两份 halo 语义将来会分叉。建议：`PackedHaloExchange` 改为调用既有实现，或反向把主求解器切到新 `parallel` 包（这才是 L3 路径）。

### F6（低·口径）`kpi.csv` `identity_maxrel` 在 np1 行为空
- np1 是基准臂，空值语义正确但下游 CSV 消费者会当作缺失；建议填 `0.0` 或 `"ref"`。

---

## 4. 未跑 / 未知（不写成通过）
- **真实 MPI 4 rank**：环境无 mpi4py/mpirun，`not_run`。overlap 收益、wire 字节的"真实网络"结论在本镜像不可测（报告已声明）。
- **OSS 二进制对照**：`not_run`（同上，F2）。
- **集群（loong64 两节点）多进程**：未递交；本单为交差核验，不需重算（复现已在本机完成）。
- **长时间稳定性 / 600 步全 9 臂**：仅补跑 NP=1 串行 600 步（F1），全臂 600 步 `not_run`。

## 5. 检查项与命令（可复算）

```
cd /root/tp-agent/work/m17994
python -m pytest tests/test_parallel_slab_unpacked.py -q            # 10 passed
python xval/XVAL-LBM-MPI-1D-SLAB-UNPACKED-HALO-TESTALL-OVERLA/run_xval.py \
       --out /tmp/audit17994/results.json                            # COMPLETE, 9 arms
python docs/audit_m17994/make_audit_figures.py                       # figA/B/C + 两个 JSON
runuser -u postgres -- psql -d tensorplatform -tA -c \
  "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases \
   WHERE anchor_slug='XVAL-LBM-MPI-1D-SLAB-UNPACKED-HALO-TESTALL-OVERLA'"   # 0 rows
```

## 6. 裁定（disposition）

**Approve with notes → 级别 L2（confirmed，消账不返修）。**
理由：本单交付的**功能**是真的、可跑、可复现、精度远超 owner 标准（0.0138% vs 15%，且 <3%），测试是真断言；但功能**定位**是独立验证模块，未接入任何求解器，且对标另一侧未执行、平台无判据行 —— 按「级别=功能实质」只能是 L2。非返修项（无造假、无 MOCK）：F1/F3/F4 属报告与 API 纪律问题，建议随下一个同类节点一并修；若 owner 要求该能力上 L3，唯一路径是把 `parallel` 包接进 `blockforest`/主求解器（F5）。

---

## 7. 图

| 图 | 内容 | 读数 |
|---|---|---|
| `figures/figA_kpi_parity.png` | 9 臂 × 6 确定性 KPI 的 \|复跑−交付\|/\|交付\|（log10） | 全 0 → 交付数值可复现（唯一非 0 是 testall 轮询计数，属线程调度量） |
| `figures/figB_field_vs_analytic.png` | u_x 场云图 + y 剖面 LBM vs 解析剪切波 + 逐点误差；t=300 与 t=600 | 0.01378% / 0.01471%，剖面与解析重合 |
| `figures/figC_timing_spread.png` | overlap speedup、轮询扫描墙钟、MLUPS 交付 vs 复跑 | 符号稳定、量级不稳（2.76×）→ F3 |