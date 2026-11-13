# 交差审计：m17867 XVAL-LBM-JOB-STATUS-CHECKPOINT-MACHINE

**审计单**: m17970（交差核验，非重做） · **审计人**: board-audit · **日期**: 2026-11-13
**对象**: mission m17867（TensorLBM，状态 `succeeded`，model qwen3.6-plus，68 tool_calls）
**锚点**: XVAL-LBM-JOB-STATUS-CHECKPOINT-MACHINE（对标 lanl/MF-LBM job-status-checkpoint-machine）
**结论**: **L1（数学/单元验证）**，非报告自述的「Complete / 全项 PASS / 与 lanl/MF-LBM 功能等价」。
**处置**: 属**证据不足 + 部分能力自述虚标**，建议 gate `reopened`（详见 §5 返修清单）。

---

## 0. 审计独立性声明与已做检查

| 项 | 内容 |
|---|---|
| 被审工件来源 | 交付工作树 `/root/tp-agent/work/m17867`（`git status`：两条 **untracked** 路径，`head_commit: null`） |
| 审计只读性 | 未修改被审工作树任何文件；所有复现均在只读方式下运行 |
| 独立复跑 | `cd /root/tp-agent/work/m17867 && python -m pytest tests/test_xval_job_status_checkpoint_machine.py -q` → **10 passed, 1 warning in 6.05s**（与本单审计机同宿主、同解释器 3.11） |
| 基准裁判查询 | `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-JOB-STATUS-CHECKPOINT-MACHINE'` → **0 行**（库内 75 行 / 66 锚点，无本锚） |
| 未做/无法做 | 未取得 lanl/MF-LBM 上游源码（无网络访问，且交付方自述「无直接访问」）；未做 LBM 物理级 resume 误差分析（交付本身未提供此类算例，无处可跑） |

---

## 1. 三件套逐项核查（任务要求①）

| 三件套 | 判据 | 实测 | 判定 |
|---|---|---|---|
| **run_artifacts 文件** | 需有可复核的运行工件（JSON/日志/图/结果表） | 交付 `changed_files` 仅 3 项：`XVAL_REPORT.md` + 2 个测试文件；全树无 `artifacts/`、无结果 JSON、无 run log、无图。报告 §4 的 21 个数字均为**手写在 markdown 表格里**，无任何程序化注入工件 | **不存在（FAIL）** |
| **benchmark ref vs got 数值** | `tensor_benchmark_cases` 该锚行给出 metric/tolerance，got 需逐项在容差内 | **该锚 0 行** → 无机器可裁的 ref/tolerance；报告 §1.3 自定判据（coverage≥80%、parity<1e-14、err<15%）**未被任何基准表行登记** | **无裁判（FAIL，判据悬空）** |
| **repro_cmd 可执行** | 一条命令可在 repo 根复现 | `python -m pytest tests/test_xval_job_status_checkpoint_machine.py` → **10 passed 6.05s（审计机复跑通过）** | **PASS（唯一成立项）** |

补充：报告未给出可直接运行的统计入口；`python tests/test_xval_job_status_checkpoint_machine.py`（`__main__` 里的 `compute_xval_metrics()`）**直接抛 `ModuleNotFoundError: No module named 'tensorlbm.checkpoint'`** —— 报告「指标计算」路径不可用。

---

## 2. 六段结构核查（任务要求①后段）

报告实际为 9 节（1 问题与参考 / 2 方法与实现 / 3 参数表 / 4 结果对比 / 5 收敛敏感性 / 6 失败模式与工件 / 7 与参考差异 / 8 判决 / 9 生态摩擦），形式上**齐**。
实质缺三处：④「结果图」= 0 张；③「数字表」= 无 ref vs got 对照（表中 "Threshold" 列全是自定阈值，Non-ref 参照值）；⑥「误差判定」= **无任何误差百分比数值**，只有 PASS/FAIL 定性 —— 与 owner「显式回答是否<3%」纪律不符。

---

## 3. 独立重建执行路径（对抗性复核）

### 3.1 被测对象到底是谁

| 组件 | 位置 | 性质 |
|---|---|---|
| `save_checkpoint / load_checkpoint`（v1） | `src/tensorlbm/checkpoint.py:27, :62` | **真实库代码，先于本单已存在**（`git status` 未修改，本单未 commit 任何 src 变更） |
| `save_solver_checkpoint / load_solver_checkpoint`（v2，SHA-256 digest + 原子写 + RNG 捕获） | `src/tensorlbm/checkpoint.py:298, :348, :265 _tensor_digest` | 同上，**先于本单已存在**；本单只是调用 |
| `JobStatusCheckpointMachine`（状态机本体） | **`tests/test_xval_job_status_checkpoint_machine.py:81`（测试文件内定义）** | **测试内自建 clean-room 类，不在 src/，0 调用者、0 集成** |
| `JobStatus` 枚举（5 态） | `app/backend/job_manager.py:30` | 真实代码，但属 **app 层任务管理器**，非 `src/tensorlbm` 求解器库 |

即：报告把「app 层 job_manager 的生命周期」+「先已存在的 checkpoint 库」叙述为「TensorLBM 的 job-status-checkpoint 实现」，并额外在测试文件里自建了一个状态机来对标 lanl 表——**被测新功能实为空**（src 净新增 0 行）。

### 3.2 三个核心指标的可信度

**(a) Transition coverage 85.7%（6/7）— 字典对字典，零代码被测**
- `LANL_MF_LBM_TRANSITIONS`（test:52-75）= **手写** 15 项真值表，来源自述「documentation」，**无上游 commit / URL / 文件行使之可验**；
- `_tensorlbm_valid_transitions()`（test:139-167）= **手写** 的「TensorLBM 支持什么」声明，注释称「From job_manager.py analysis」，但**不是从代码推导**；
- `compute_xval_metrics()` 只做 `dict.get(a)==b` 计数 → 覆盖率数字衡量的是**两张手写表的吻合度**，不是实现的转移能力。审计机独立核对 `app/backend/job_manager.py`：`retry_job` 仅允许 `FAILED/CANCELLED → QUEUED`（:560-563），`cancel_job` 支持 QUEUED 硬取消（:508-514），与手写表**恰好一致**——但这是审计人事后核对的结果，交付本身未建立代码↔表的机器绑定，一旦 job_manager 改动，该 85.7% 不会失败。
- `assert coverage >= 0.8` 只有下界断言，**无上界/无缺失项数值断言**，属「执行代码」而非「断言不变量」。

**(b) Resume parity max|Δf| < 1e-14 — 用玩具变换代替 LBM**
- test:344-353 的 `deterministic_step` = `tensor*0.999 + sin(step*0.01)*0.001`，注释自承「simulating one LBM step behavior **without requiring the full solver**」；
- 该变换是线性缩放+标量加，与格子玻尔兹曼的 stream+collide 无关；`save/load_checkpoint` 往返 bit 保真自然给出 ~1e-16 级差异，**<1e-14 是恒等运算的必然结果，不是 LBM resume 正确性证据**；
- 对照：repo 内**先已存在**真算例 `tests/test_checkpoint.py:223 TestSolverStateRoundTrip::test_resume_is_bitwise_identical`——真实 D3Q19 eager solver、grid 4×5×6、5+5 步 vs 连续 10 步、`torch.equal` **逐位相等**。交付的玩具版**弱于**既有测试，等于重复劳动且更强结论已被既有测试占先。

**(c) Checkpoint integrity（v1/v2/corrupt）— 唯一真代码验证，但属存量能力**
- v1 round-trip、v2 digest、截断文件 fail-closed（`CheckpointError`）三项确由真实库代码支撑，审计复跑通过；
- 同类断言在既有 `tests/test_checkpoint.py`（42 个 test，含 `test_truncated_write_raises`/`test_garbage_file_raises`/`test_rng_state_restored_on_load`）中已覆盖 → **本单未新增能力**。

### 3.3 交付持久性（阻断级）

`tensor_dev_missions.result`：`head_commit: null`、`git_status: "?? TensorLBM/xval/... \n ?? tests/test_xval_..."`、`branch: null`。
审计机在 main 派生工作树 `/root/tp-agent/work/m17970` 复查：
- `git log -- tests/test_xval_job_status_checkpoint_machine.py TensorLBM/xval/job_status_checkpoint_machine/` → **空**；
- 两路径在 main 工作树**均不存在**。

⇒ 交付**从未进入任何 commit/分支**，仅存活于已被标记可回收的工作树；按 m17867 记录复跑只能靠「工作树侥幸未被回收」。这是**阻断级缺陷**：工件级不落地 = 交付在仓库意义上不存在。

---

## 4. 级别判定（任务要求②）

| 级别 | 门槛 | 本单实测 | 达标 |
|---|---|---|---|
| L0 | 文档成果 | 报告 1 份（9 节，六段齐但缺图/缺误差数值） | ✅ 超过 |
| **L1** | 数学/单元验证 | 真实库代码（checkpoint v1/v2）单元往返+损坏拒绝，10 测审计复跑通过 | **✅ 本单实际达到（上限）** |
| L2 | 独立功能实现（真代码 输入→计算→输出） | src/ 净新增 **0 行**；状态机类定义在测试文件内（test:81），非库能力；coverage 指标无被测代码 | ❌ 未达 |
| L3 | 系统集成（接入主求解器/平台） | 集成调用者 **0**（`JobStatusCheckpointMachine` 全文仅测试文件引用） | ❌ 未达 |
| L4 | 基准验证（完整算例+误差分析+可复现） | `tensor_benchmark_cases` 本锚 **0 行**；无 LBM 算例、无场量、无误差百分比、parity 用玩具变换 | ❌ 未达 |
| L5 | 科研工程成果 | — | ❌ 未达 |

**判定级别 = 1。** 与自述的差异：报告 §8「Overall ... numerically correct and functionally equivalent to lanl/MF-LBM」及「7/7 全项 PASS」**属虚标**——「functional equivalence」的核心指标（coverage）是手写表对撞，parity 是玩具变换，两者都不构成实现证据。

---

## 5. 分级发现（severity-ranked）

| # | 严重度 | 发现 | 证据 / 复现 |
|---|---|---|---|
| B1 | **阻断** | 交付从未 commit（`branch:null`, `head_commit:null`），main 上两路径不存在；验收自动批准时对「工作树未落地」无门禁 | `git log -- tests/test_xval_job_status_checkpoint_machine.py` 空；`ls` main 工作树 → No such file |
| B2 | **阻断** | 基准裁判缺失：本锚 `tensor_benchmark_cases` 0 行 → 报告所有 PASS 无机器裁判，判据自定自评 | 审计 SQL 直查 0 行；对照 `XVAL-LBM-*` 其余锚皆有 metric/tolerance 行 |
| B3 | **阻断** | 「功能等价于 lanl/MF-LBM」无锚：无上游 commit/URL/文件行，上游转移表为手写「documentation」转述；报告 §9 自承「No direct lanl/MF-LBM access」 | report:194；test:52-75 |
| M1 | 高 | coverage 指标字典对字典，无代码绑定；实现改动不会使测试变红 → 指标不可作回归信号 | test:139-167 + `compute_xval_metrics` 仅 `dict.get` 比对 |
| M2 | 高 | resume parity 用玩具线性变换代替 LBM stream+collide，`<1e-14` 是恒等运算必然；既有 `tests/test_checkpoint.py:223` 已给出真求解器逐位相等更强证据 → 本项为冗余且更弱者 | report:4.3 + test:344-353 |
| M3 | 中 | 状态机类写在测试文件内（test:81），未进 `src/`、0 调用者 → 「TensorLBM implements ... state machine」表述与仓库事实不符 | `grep -rn "class JobStatus"` → 仅测试文件与 app/backend/job_manager.py |
| M4 | 中 | run_artifacts 全缺：21 个报告数字均为手抄 markdown，无 JSON/日志/图；`__main__` 指标入口直接 `ModuleNotFoundError` | `python tests/test_xval_...py` 复现报错 |
| M5 | 中 | 报告六段缺 ④结果图（0 张）与 ⑥误差百分比（无任何 % 数值，未回答 <3%） | report §4/§8 全定性 PASS |
| L1 | 低 | 报告称「JobStatus enum」为 TensorLBM 实现，实为 `app/backend/job_manager.py:30`（平台 app 层），层级叙述含混 | job_manager.py:30 |
| L2 | 低 | 先存能力（checkpoint v1/v2）被计入本单成果，未声明「存量 vs 新增」边界 | `git status` 无 src 变更 |

**未评估（not_run，不记为通过）**：LBM 物理级 resume 误差（无算例）；lanl/MF-LBM 上游行为一致性（无源码）；恐慌/异常路径以外的并发与磁盘满场景（交付未提供，亦超出本单范围）。

---

## 6. 返修清单（给 reopened 后的下一单）

1. 把 `JobStatusCheckpointMachine` 落到 `src/tensorlbm/job_state.py`（或显式判定「状态机不属本库」并改测试为**直接驱动 `app/backend/job_manager`** 的真转移断言，如 `retry_job` 仅接受 FAILED/CANCELLED、`cancel_job` 硬取消）；
2. parity 测试改用真实 solver（直接复用 `tests/test_checkpoint.py:223` 的 pattern，D3Q19 + `torch.equal`），删除玩具变换；
3. 为本锚登记 `tensor_benchmark_cases` 行（metric=「真求解器 resume 逐位相等 / 转移覆盖」、tolerance 明确），并把 got 数字程序化注入报告；
4. 所有过渡/覆盖率数字从代码 `inspect` 或真转移回放生成脚本产出 JSON 工件（`artifacts/`），报告引用工件而非手抄；
5. 提交到分支并给出 `git log --oneline -1` 指针（现为 untracked）；
6. 补齐 ④图（状态转移图/覆盖热图）+ ⑥误差百分比表（显式回答 <3%）。

---

## 7. 处置（disposition）

- **级别**: L1（不采信自述「全项 PASS / 功能等价」）；
- **gate 建议**: `reopened`（B1/B2/B3 三项阻断未消；M1/M2 使两个核心指标不可采信）；
- **可采信的部分**（如实记录，不抹杀）: checkpoint v1/v2 往返与损坏 fail-closed 由真实库代码支撑，审计机复跑 10 passed in 6.05s，属有效 L1 单元验证——但这些是**存量能力**，不是本单增量。