# XVAL-LBM-MAXREGCOUNT-PINNED-BEFORE-TUNING

**Mission** 17957 · **Repo** TensorLBM · **Branch** `agent/m17957` · **类型** 开源对标（XVAL）
**对标对象** `AminZar69/CudaPhaseField` @ `d6181e677c413bbec443570010bca60d896022b5`（GPL-3.0，inventoried）
**判据** owner 精度标准 <15%；本单另有预登记阈值（`xval/XVAL-LBM-MAXREGCOUNT-PINNED-BEFORE-TUNING/predeclared.json`）
**结论（先给）** 协议移植**成立**；"钉预算即得可复现计时"在共享主机上**按预登记阈值不成立**（honest partial）：

| 判据 | 预登记 | 实测 | 判 |
|---|---|---|---|
| 钉预算臂选择可复现（5 次独立进程 rank-1 一致） | 5/5 | **5/5**（wall 与 CPU 双时钟） | ✓ |
| 钉预算臂计时相对散布 | <10% | **47.8%**（wall） / 40.8%（CPU） | ✗ |
| 未钉预算臂暴露混杂（rank-1 抖动 or 预算敏感性 ≥15%） | — | CPU 时钟 rank-1 **3/5**；预算敏感性 **25.6%–3050%** | ✓ |
| 物理锚 vs 解析 Poiseuille（u_max） | <15%（目标 <3%） | **0.361%** | ✓✓ |
| 物理锚剖面 L2 | <15%（目标 <3%） | **0.492%** | ✓✓ |
| 钉是否物理中性 | <1e-10 | **0.0**（两臂逐点相同） | ✓ |
| OSS 钉值 64 是否可移植 | — | 本仓 fp64 需 **80–140** words | ✗（值不可抄，需重算） |

---

## ① 问题 + ref 出处

**问题**：调优（选哪个 collision 实现、哪个 `torch.compile` mode、哪个 tile/block）只有在**资源预算被先钉住**时才可归因。预算漂移时，"最优配置"是预算的函数而不是 kernel 的属性；两个入口脚本继承不同预算，各自记录一个"调优结果"，两个都不可归因。

**ref（零手抄，程序化读取自 `TensorPlatform/docs/oss_inventory`，落盘 `artifacts/xval_maxregcount_pinned/oss_citation.json`，含 md/evidence 两文件 sha256）**：

- `CudaPhaseField.md:115` —— `collision_h` 单核 286 行，同时读 18 条 h + 18 条 heq + 18 条 hlph，历史上因寄存器溢出**直接 launch 失败**；修法是编译期 `maxregcount:64` 写死进 Makefile，并注明"不是诊断开关，是永久修复"（`Makefile:61`、`:64`）。
- `CudaPhaseField.md:271-272`（`tag: maxregcount-pinned-before-tuning`）：寄存器压力而非带宽才是最大 kernel 的真实瓶颈；上限作为**永久约束**施加到**每一个 build mode**，且在 tile/launch 调优**之前**钉住。
- `CudaPhaseField_evidence.py:438-439`：`("Makefile", 61)`/`("Makefile", 64)` → `maxregcount:64`（行级源码取证）。
- 许可：GPL-3.0 → transfer ∈ {reference, adapt}，**不搬代码**；本单移植的是**协议**与其算术，实现全为本仓原创。

**移植到本仓的现实缺口**：本仓无 CUDA、无 GPU、无寄存器文件（`nvidia-smi` 不存在），但**同类混杂确实存在且更隐蔽**——`grep src/tensorlbm` 显示各入口脚本各设各的线程预算（`actuator_line.py:957` 默认 16、`curved_wall_force_scaling.py:579` 默认 1、其余走 OMP 环境变量或 torch 默认 128），而**任何调优结果都不记录它是在哪个预算下得到的**。GPU 上的预算溢出是**硬失败**（launch 报错），CPU 上的预算漂移是**软失败**（不报错，只是结论错）——后者更需要机制化。

## ② 方法与实现

**新增机制（可复用能力）** `src/tensorlbm/bench_pinning.py`：

| API | 作用 |
|---|---|
| `ResourcePin(label, threads, dtype, compile_modes, warmup_steps, timed_steps, repeats, maxnreg)` | 不可变预算声明；构造即校验（线程数/精度/协议步数/maxnreg≥16） |
| `pinned_resources(pin)` | 上下文管理器：施加预算、产出审计、退出复原 |
| `assert_pinned(pin)` | 测量时刻实测态 ≠ 声明预算 → `PinnedResourceViolation`（禁止"预算漂移下的调优结果"入库） |
| `pin_audit(pin, **extra)` | 预算 + 实测态 + `pin_sha256`（内容寻址）随每个数字落盘 |
| `maxnreg_launch_kwargs(pin)` | GPU 侧等价拼写：`nvcc -maxrregcount=N` ↔ Triton `maxnreg=N` |
| `register_budget_words` / `register_cap_verdict` | 把外来钉值**重算**到本仓 kernel（活值数 × 每值字长 + 预留给指针/索引的 16 words） |

**实验臂（`xval/.../run_arm.py`，一次一个进程，串行，禁并行）**：

- 调优网格：D2Q9 BGK collide+stream 整步，3 实现 × 2 编译模式 = 6 配置
  （`solver.collide_bgk` / `collide_bgk_fused` / `collide_bgk_matmul` × eager / `torch.compile('default')`），
  N=128×128 周期剪切波，`tau=0.8`。
- **U（现状）**：5 次独立进程，每次继承一个预算 `{128,16,1,4,8}`（顺序预登记，取自本仓真实存在值），调优后把胜者记为"调优结果"，不带预算。
- **P（移植）**：5 次独立进程，预算钉 8 线程，`pinned_resources` + `assert_pinned`，每行数字带 `pin_sha256`。
- **ladder（敏感性）**：单进程内交错测 `threads ∈ {1,8,16,64,128}` × 6 配置，同一配置跨预算比较（不是跨时刻比较）。
- **物理锚**：同一 D2Q9/BGK 格子的平面 Poiseuille（Guo 强制 + 半程反弹壁面，H=16 流体行，6000 步），与解析抛物线比 —— 用来证明"钉是性能旋钮、不是物理旋钮"。

**分析** `analyze.py`：读全部原始 JSON（报告里没有一个数字是手抄的），按预登记阈值判定，输出 `summary.json` + 3 张 CSV + 4 张图。

**测量有效性守卫**：首版 runner 用闭包工厂产生 step 函数，三个实现共享同一 code object → Dynamo 命中 `recompile_limit (8)` 后**静默回落 eager**（"compiled" 臂实际在跟 eager 比）。已改为每实现一个模块级函数 + `cache_size_limit=64`，并在每行测量里记录 `compiled` 标志（`hasattr(step, "_torchdynamo_orig_callable")`）。这是本单发现的真实测量缺陷，不是调参。

## ③ 参数表（无量纲数完整）

| 量 | 值 | 说明 |
|---|---|---|
| 格子 | D2Q9, BGK/SRT | `cs² = 1/3` |
| `tau` | 0.8 | `nu = (tau-0.5)/3 = 0.1` |
| 调优域 | 128 × 128 周期 | 剪切波 `u_x = 0.01 sin(2πy/N)` |
| `Ma`（调优） | 0.0173 | `= 0.01/cs`，远小于 0.1 |
| 精度 | float64 | 与 `register` 算术对齐（fp64 → 2 words/value） |
| 计时协议 | warm-up 10 步 + 5×(40 步) 取中位数 | 主时钟 `perf_counter`（wall）；副时钟 `process_time`（CPU，amendment 1） |
| 预算 | P=8 线程；U 漂移集 {128,16,1,4,8}；ladder {1,8,16,64,128} | 128 = 本机核数，16/1 = 本仓脚本默认值 |
| 物理锚 | H=16 流体行，`g=1e-5`，6000 步（尾 200 步平均） | `u_max = g H²/8ν = 3.1875e-3`，`Ma = 5.5e-3` |
| 质量守恒 | `|Δm|/m = 4.0e-13` | 60 步~6000 步全程 |

## ④ 对照表（ref vs got，误差打勾叉）

**(a) 协议/选择面**

| 项 | ref（OSS 做法/本仓预登记） | got（本仓实测） | 判 |
|---|---|---|---|
| 钉预算臂 rank-1 一致性 | 5/5 | 5/5（wall 5/5，CPU 5/5），`bgk_fused/default` | ✓ |
| 钉预算臂计时散布 | <10% | 47.8%（wall） / 40.8%（CPU） | ✗ |
| 未钉臂 rank-1 一致性 | 预登记"应暴露混杂" | wall 5/5、**CPU 3/5**（1、4 线程下 `bgk_matmul/default` 反超） | ✓（cpu 时钟） |
| 预算敏感性（同一配置） | ≥15% 即构成混杂 | 25.6%（`bgk/default`）… **3050.8%**（`bgk_matmul/default`） | ✓ |
| rank-1 随预算变化 | — | wall：预算 1/16/64/128 → `bgk_fused/default`，预算 8 → `bgk_matmul/default`；CPU：1、8 → `bgk_matmul/default` | 2 种胜者 |

**(b) 物理面（精度判据）**

| KPI | ref（解析 Poiseuille） | got（arm P / arm U） | 误差 | <15% | <3% |
|---|---|---|---|---|---|
| `u_max` | 3.1875e-3 | 3.1760e-3（两臂相同） | **0.361%** | ✓ | ✓ |
| 剖面 `L2` | — | — | **0.492%** | ✓ | ✓ |
| 钉 vs 未钉的物理差 | 0 | **0.0（逐点相同）** | — | — | ✓（<1e-10） |

**(c) 寄存器钉值算术（OSS 64-word 上限 vs 本仓 kernel）**

| kernel | 活值数 | 精度 | 需 words（含 16 预留） | 64 是否够 |
|---|---|---|---|---|
| D2Q9 `collide_bgk` | 23 | fp64 | 62 | ✓ |
| D2Q9 `collide_bgk_fused` | 32 | fp32 | 48 | ✓ |
| D2Q9 `collide_bgk_fused` | 32 | fp64 | 80 | ✗（+16） |
| D2Q9 `collide_bgk_matmul` | 27 | fp64 | 70 | ✗（+6） |
| D3Q19 fused 移植（19 方向） | 62 | fp32 | 78 | ✗（+14） |
| D3Q19 fused 移植 | 62 | fp64 | **140** | ✗（+76） |
| **OSS `collision_h`** | 54 | fp32 | **70** | ✗（+6） |

**读法**：64 连 OSS 自己的 kernel 都不够（需 70）——说明 64 是**故意接受的少量溢出**换可 launch；本仓 fp64 融合 kernel 需 **80–140** words，即钉值必须**按 kernel×精度重算**（D3Q19 fp64 ≥128），照抄 64 会把融合 kernel 直接压进溢出区。**协议可移植，值不可移植。**

## ⑤ 收敛性 / 参数敏感性（≥2 点）

1. **预算敏感性（主敏感性）**：同一配置跨预算 `1→128` 线程，`bgk/default` 25.6%、`bgk_fused/default` 34.0%、`bgk_matmul/default` **3050.8%**、eager 三档 392%–930%（`table_ladder.csv`）。
   **同时**（如实）：同一预算内**配置之间**的散布同样巨大——`1/8/16/64/128` 线程分别为 1437% / 511% / 484% / 3045% / 6000%（`convergence.json: within_budget_config_spread_pct`）。所以正确的结论不是"预算比配置重要"，而是**两个旋钮强烈交互**：预算一变，rank-1 就换人（wall：8 线程 → `bgk_matmul/default`，其余 → `bgk_fused/default`；CPU 时钟：1、8 线程 → `bgk_matmul/default`）。**不钉预算就把两者混在同一个数字里，排名不可解释**——这正是 OSS 先钉预算的理由，也解释了为什么本仓"各脚本各设线程"的现状会让调优结论不可迁移。
2. **时序/收敛**：Poiseuille 锚 6000 步（≈2.3 个跨槽扩散时间 `H²/ν=2560`），尾 200 步平均；质量漂移 4.0e-13；`u_max` 在步数 5000→6000 的**相对变化 6.2e-9**（`convergence.json`，瞬态已完全消失），故 0.361% 主要是半程壁面位置与离散平衡的 O(网格) 偏差，不是瞬态残留。
3. **重复级噪声**：同一配置同一次启动内 5 次重复的散布可达 2.6×（pilot），跨启动 modal 配置散布 47.8% —— 见 ⑥ 的 friction，属宿主机负载所致，与预算无关（钉住的 P 臂同样受影响）。

## ⑥ 失败模式 / 工件 / 生态摩擦

**(i) honest partial FAIL（预登记 H1 未全绿）**
`h1_pinned_reproducible = false`。失败条款：散布 47.8% ≥ 预登记 10%。原因可归因：运行窗口内宿主 load average 74–171 / 128 核（其他 agent 的计时作业并发）。**结论边界**：钉预算消掉的是**可控制的**混杂（预算漂移），消不掉**不可控制的**混杂（共享主机抢占）；后者的正确疗法是独占节点（cluster_queue），不是把阈值放宽——按预登记纪律，阈值未改、结果不覆盖。

**(ii) 已发现并修复的测量缺陷**：Dynamo `recompile_limit` 静默回落 eager（见 ②），若不修，则"compiled 胜出 5/5"可能只是 eager 与 eager 自比。修复后每行带 `compiled` 标志。

**(iii) 生态摩擦（friction，如实记）**
- `knowledge_base` 检索引擎离线：`/api/v1/tensor/knowledge/search?q=maxrregcount` 返回 `engine_offline`（`http://127.0.0.1:8021` connection refused）→ 开工三查只能走 `negative_knowledge`（0 命中）+ `oss_inventory` 文件（可用）。记为本生态病：知识检索依赖单个 8021 进程，挂了没有降级检索面。
- 首版 ladder（8 预算 × 2 sweep = 96 cells）在 load 171 下 13.5 min 未完成 → 按 amendment 2 缩到 5 预算 × 1 sweep（**范围裁剪，非阈值改动**，预登记 amendments 已留痕）。
- 与另一个并发计时作业（`m17863`）同机竞争 CPU，双方计时互相污染 → 计时类 XVAL 应有独占节点门（建议入平台侧机制，不属本仓）。

**(iv) 工件清单**
- 预登记：`xval/XVAL-LBM-MAXREGCOUNT-PINNED-BEFORE-TUNING/predeclared.json`（含 2 条 amendments）
- 机制：`src/tensorlbm/bench_pinning.py`（+ `tests/test_bench_pinning.py`，8 tests 全绿）
- 运行器/分析：`run_arm.py` / `analyze.py` / `drive.sh`（串行驱动，禁并行）
- 原始：`artifacts/xval_maxregcount_pinned/tune_{U,P}_{0..4}.json`、`ladder.json`、`summary.json`、`convergence.json`、`table_arms.csv`、`table_ladder.csv`、`table_registers.csv`、`oss_citation.json`
- 图：`figures/xval_maxregcount_pinned/fig1_tuning_winner_vs_budget.png`（胜者随预算变化）、`fig2_selection_stability.png`（两臂 5 次启动）、`fig3_poiseuille_pin_neutrality.png`（物理中性）、`fig4_register_budget_vs_oss_cap.png`（钉值不可抄）

## 本仓 vs OSS 差异清单 + 移植取舍

| 维度 | OSS CudaPhaseField | 本仓（移植前） | 本仓（本单后） | 取舍 |
|---|---|---|---|---|
| 钉什么 | 寄存器上限（`-maxregcount 64`，编译期，全部 build mode） | 无（各脚本各设线程） | `ResourcePin`：threads + dtype + compile mode 白名单 + `maxnreg`（GPU 侧预留） | 移植"预算"抽象，不移植具体旋钮语义 |
| 何时钉 | 调 tile/launch **之前**，永久约束 | 无 | 调优前声明 + `assert_pinned` 测量时校验 | 照搬 |
| 钉值来源 | `64`（fp32, 54 活值, 允许少量溢出） | — | **重算**：`register_cap_verdict`（D2Q9 fp64 80 / D3Q19 fp64 140） | 值不抄，只抄算法 |
| 失败模式 | 溢出 → launch 硬失败（显性） | 预算漂移 → 静默错结论（隐性） | 漂移 → `PinnedResourceViolation`；结果无 `pin_sha256` 即视为不完整 | 把隐性变显性 |
| 结果携带 | Makefile 是唯一真相 | 无 provenance | 每行数字带 `pin` + `pin_sha256` + 实测态 | 照搬并落地为 JSON |
| 边界条件 | CUDA 寄存器文件、单 GPU | CPU 线程池 + 编译缓存 | CPU 主战场；GPU `maxnreg` 仅提供拼写与算术，**未实测** | 明确标注未实测部分 |

**未完成/未做（如实）**：GPU 侧 `maxnreg` 只有算术与 API（无 GPU 可跑，不做假实测）；结论的可复现性只在"同一共享主机、负载 74–171"这一环境下测量，独占节点下散布应显著更小但**本单未测**——不做无证据推断。