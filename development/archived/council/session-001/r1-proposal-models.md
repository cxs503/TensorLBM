author-model: qwen3.7-plus

# session-001 · 提案（模型与算法视角）：求解器能力深化 —— 10 库谁第二个上卡 / SDAA 性能基准线

> 提案人槽位文件。只写我自己的槽位，不动议事板其他文件；不派发任何任务、不碰 GitHub。
> 数字来源：`council/session-001/00_context.md`（标记 [ctx]）+ 我在本 worktree（TensorLBM @ agent/m37）实读到的代码/文件（标记 [repo]，附路径）。
> 我核不到的东西一律标 **待实测 / WITHHELD**，不写推测数字。

---

## 0. 一句话立场

**先立度量，再谈能力。** 我的排序是：今天先把 **SDAA 性能基准线 v0**（kernel MLUPS + 显存天花板 + 端到端 case 计时 + 跨卡带宽，全部落 JSON 且进门禁）打出来，同时用**已校准的普查脚本**（先用 TensorLBM 的 507/512 复现来验证脚本本身）判定"10 库第二个上卡是谁"；3D/多相/自由面上卡排在这两条之后，因为它最贵且最容易在"卡显存/计时不可控"上翻车。

理由（模型与算法视角）：现在平台手上只有一个性能事实——1200² matmul ~15ms vs CPU 64 线程 212ms（≈14×）[ctx]——而这对 LBM 的**逐格点带宽受限**内核是**完全不同**的算力画像。用 matmul 的优势去推断 LBM/SPH/DEM 上卡的收益，是把计算密集结论挪到访存密集问题上，会系统性高估。没有基准线，任何"哪个库先上卡"的裁决都是猜。

---

## 1. 我用到的既有事实（只列判据相关的，不复述背景表）

| # | 事实 | 来源 |
|---|---|---|
| F1 | SDAA 32 设备 `/dev/tcaicard0-31`；triton 3.4 无 SDAA 后端，只能 `TRITON_INTERPRET=1` 解释（慢）；纯 torch 全速 | [ctx] 硬约束① |
| F2 | TensorLBM 512 模块 = 507 纯 torch（A 类，可直接上卡）+ 5 triton 依赖（B 类，仅解释模式） | [ctx] m11 |
| F3 | 2D Couette 已上卡：L2 1.80e-4，跨卡 5→0 逐位复现 | [ctx] m10 |
| F4 | 43 个 verified 案例待全覆盖；扫荡线 m12–m23 共 12 案例在飞 | [ctx] |
| F5 | 性能入口已存在且已带设备选择：`benchmarks/bench_mlups.py`，参数 `--device cpu\|sdaa\|cuda\|mps`、`--collisions`、`--warmup`、`--steps`；内核函数 `_mlups_2d(ny,nx,...)` 已参数化但**无网格尺寸 CLI** | [repo] |
| F6 | 设备解析/能力探测已有：`tensorlbm.utils.resolve_device`（`sdaa`/`sdaa:N`，不可用即抛错）、`tensorlbm.hardware.probe()`（backend + collectives 探测，含 sdaa 插件探测） | [repo] `src/tensorlbm/utils.py:101`、`src/tensorlbm/hardware.py` |
| F7 | 验收口径已存在且机器可判：`benchmarks/verified/<case>/{README.md,result.json,run.py}`，`result.json` 含 `grids.*.l2_rel_err`、`convergence.err_decreased`、`verified:true`；STATUS 口径 = 每 checkpoint ≤3% 且两档网格 span ≤3% | [repo] `benchmarks/verified/couette_2d/result.json`、`benchmarks/STATUS.md` |
| F8 | 门禁可复用：`src/tensorlbm/regression_gate.py::evaluate_regression_gate / write_regression_manifest` + `scripts/evaluate_benchmark_gate.py`（离线评估产物） | [repo] |
| F9 | 已有"一次命令出 pass/fail JSON"的先例可抄：`benchmarks/gpu_smoke_suite.py`（9 项检查，每项阈值有出处） | [repo] |
| F10 | 分块路线已有设计文档：`docs/L1_BLOCK_SHARDING_DESIGN.md` | [repo] |
| F11 | **我核不到**：TensorFVM/FEM/DEM/SPH/BEM/DG/CAD 与 TensorPhysics 这 9 个库的代码现状（不在本 worktree，我未实读）；SDAA 单卡显存容量；torch_sdaa 对 bf16/tf32 的支持；TCCL 3.1.0 跨卡实测带宽 | 诚实标注，见 §7 |

---

## 2. 方向 D1（首推）：SDAA 性能基准线 v0 —— 让"快不快"变成可复现的数字

**① 目标一句话**：48 小时内产出一条可复现、可进门禁的 SDAA 性能基线（kernel MLUPS / 显存天花板 / 端到端 case 计时 / 跨卡带宽四张表），此后所有上卡投资都用它算 ROI。

**② mission 粒度拆解**

| ID | mission | repo | 机器判据（可执行/可判） | 预算 |
|---|---|---|---|---|
| M-P1 | 单卡 kernel 微基准网格 | TensorLBM | 最小改动给 `bench_mlups.py` 加 `--shape ny,nx` 与 `--json out.json`（**不改既有默认输出**）；在独占的 `/dev/tcaicard0` 上跑 bgk/mrt/trt/rlbm × {128²,256²,512²,1024²}，`--steps 100 --warmup 10`，每配置 3 次重复取 median；**判据：每条目 3 次 CV ≤2%**，超阈值该条目记 `INVALID`（不接受"慢"当结论）；同参数在 CPU 64 线程跑一遍做分母 | 1 卡 × 6 卡时 |
| M-P2 | 单卡显存天花板 | TensorLBM | 二分网格至 OOM：2D 出 (ny,nx)→peak_bytes 表，3D 出最大可跑 D3Q19/D3Q27 立方网格；**判据：显存随网格单调、OOM 边界 2/2 次可复现**；这张表直接决定 D3 线是否可行（F11 未给显存，必须实测） | 1 卡 × 3 卡时 |
| M-P3 | 端到端 case 计时 + 精度不退化 | TensorLBM | 取 `verified/couette_2d` 与 `verified/poiseuille_3d_duct` 的 `run.py` 原口径跑卡；**判据：`l2_rel_err` 与 CPU 基线差 ≤ max(1e-6 绝对, 1% 相对)，且 `result.json` 仍 `verified:true`（3% 门槛不放宽）**；记录 wall time 与显存 | 2 卡 × 4 卡时 |
| M-P4 | 跨卡通信带宽 / TCCL 可用性 | TensorLBM | 先 `hardware.probe()` 落 collectives 探测结果；再做 2/4/8 卡 all-reduce 与 halo-exchange 口径带宽表；**判据：all-reduce 结果 vs float64 CPU 参照在 float32 内核噪声内一致 + 带宽可复现（重复 2 次差 ≤5%）**；TCCL 探测 fail 则本 mission 转 `WITHHELD`，不阻塞他线 | 2 卡 × 3 卡时 |
| M-P5 | 基线进门禁 | TensorLBM | 把 P1/P3 的 JSON 接进 `regression_gate` 同口径（可新增 `evaluate_perf_gate`）；**判据：注入一次 −20% 假数据必须 FAIL，真实基线重跑 PASS**；新增 `tests/test_perf_gate.py` 进 pytest | 1 卡 × 1 卡时 + CPU |

**③ 卡位**：5 卡（P1 独占 1 卡；P2 独占 1 卡；P3 2 卡；P4 2 卡借 P3 释放后窗口）。全部要求**独占窗口**，禁与扫荡线共卡（共卡的前 2 卡时是废数）。

**④ 风险与依赖**
- 依赖 F5/F6/F8 都在（已核）；`bench_mlups` 加 `--shape` 是唯一代码改动，风险低。
- 风险 R-timing：卡上计时被别线抢占污染 → 靠 CV≤2% 自曝；不解则本方向整体作废（见 §8 放弃条件 1）。
- 风险 R-driver：torch_sdaa 3.1.1a3 的首次 kernel 编译开销会污染小网格条目 → 固定 `--warmup 10` 并把 warmup 时间单列。

**⑤ 放弃条件（不做）**：若 3 个工作日内拿不出 CV≤2% 的可复现基线（说明卡上计时本身不可控），D1 停止扩项，本视角全面放弃"性能承诺"，卡只做长时批量算例；产出转为一份"为什么不可控"的负面报告。

---

## 3. 方向 D2：10 库"第二个上卡"裁决 —— 用被校准过的普查脚本判，不靠感觉

**① 目标一句话**：用可复跑的静态普查 + 卡上冒烟，3 个工作日内给出"除 TensorLBM 外第二/第三个上卡的库是谁"的裁决与排序，并给出否决理由。

**② mission 粒度拆解**

| ID | mission | repo | 机器判据 | 预算 |
|---|---|---|---|---|
| M-C1 | 移植性静态普查（10 库） | 各库（只读扫描）+ 脚本落 TensorPlatform | 扫 import 面：纯 torch / triton / CUDA 扩展（`.cu`、`cpp_extension`）/ MPI / cupy / numpy fallback；产出 `docs/council/portability_census_<date>.json`，字段 `{repo, files, torch_only_pct, triton_hits, ext_hits, blockers[]}`；**关键判据（脚本自证）：普查脚本跑 TensorLBM 必须复现 m11 的 507/512 纯 torch 结论（偏差 0），否则脚本本身判 INVALID、结论不得引用** | 1 卡（仅冒烟）+ 12 卡时 CPU |
| M-C2 | 卡上 op 覆盖/CPU 回退探测 | 各库最小算例 | 每库跑最小算例（无算例则 profile 3 个代表 op），检测显式 `.cpu()`/同步/插件 fallback warning；**判据：给出 `(fallback_ops, hot_ops)` 表；至少 1 个库在卡上完成一次完整迭代且无 CPU 回退** | 1 卡 × 8 卡时 |
| M-C3 | 裁决书（含第三方复核） | TensorPlatform/docs | 打分 = 纯 torch 比例 × 单案例可判性 × 算术强度先验；**判据：得票 ≥2/3 且该库在 M-C2 有一次跑通；另一 worker 独立复跑普查脚本，JSON sha256 一致** | 0 卡 |

**③ 我的先验排序（待 M-C1/C2 证伪，不得当结论引用）**
TensorLBM 已上卡（F3），所以问的是**第二个**。按"SDAA 的强项是 GEMM（1200² ~15ms vs CPU 212ms）"，高算术强度 → 高收益：
1. **BEM / FVM**（dense GEMM 主导，最容易吃到 matmul 红利）
2. **FEM**（稀疏刚度阵，可块稠密化）
3. **DEM / SPH**（粒子 sort / scatter / 邻居搜索，op 覆盖与动态形状风险最高，SDAA 上大概率回退）
4. **DG**（每元素小算子、算术强度最低；CPU 可能已够快，收益最小 → 最后上）

**④ 风险与依赖**
- **我未实读这 9 个库**（F11）。因此本方向的结论**必须**由普查脚本产出，我的排序是"待验证先验"。若议事会拿我的排序当结论派活，那是误用，请直接驳回。
- 风险 R-xcompile：若某库核心是 CUDA C++ 扩展，LoongArch 无 Docker 镜像、github 443 不通 [ctx]，重编译几乎不可行 → 该库判 NOT_SUPPORTED 而非"努力一下"。

**⑤ 放弃条件（不做）**：若 M-C1 显示除 TensorLBM 外**全部**库 ≥50% 依赖 CUDA 扩展或 triton，则取消"第二个库上卡"，把卡全部并入 TensorLBM 的 3D/多相满卡路线（D3），普查报告转为负面结论归档。

---

## 4. 方向 D3：3D / 多相 / 自由面在卡上跑通并定量（最贵、最能对口船舶耐波与绿色航行）

**① 目标一句话**：把 TensorLBM 已有的 3D（cylinder_3d / sphere_re100 / poiseuille_3d_*）、自由面（dam break 3D）、相场（laplace_droplet / poiseuille_two_phase_cg）三类在 SDAA 上跑到与 CPU 同级精度，并给出跨卡逐位复现证据。

**② mission 粒度拆解**

| ID | mission | repo | 机器判据 | 预算 |
|---|---|---|---|---|
| M-S1 | 3D 单相上卡（cylinder_3d → sphere_re100 D40） | TensorLBM | **判据：Cd 与 CPU 基线差 ≤1%，且对参考值误差仍 ≤3%（repo 门槛不变）；`/dev/tcaicardA` vs `/dev/tcaicardB` 诊断量 sha256 一致（沿用 m10 的 5→0 逐位口径）；记录 wall time 与峰值显存** | 2 卡 × 8 卡时 |
| M-S2 | 自由面 dam break 3D 上卡 | TensorLBM | 入口 `benchmarks/bench_dam_break_3d.py` + `tests/test_free_surface_application.py`；**判据：同 seed 下液面高度场与 CPU 基线 L2 差 ≤1%；质量漂移 ≤0.1%/1e4 步（阈值先测 CPU 基线再锁，锁后写进 README）** | 2 卡 × 8 卡时 |
| M-S3 | 多相/相场上卡 | TensorLBM（`src/tensorlbm/phasefield`） | 沿用既有 verified 判据（Laplace 定律 Δp·R/σ 与解析值差 ≤3%；`poiseuille_two_phase_cg` 同口径）；卡上跨卡逐位 | 2 卡 × 8 卡时 |
| M-S4 | 满卡/分块预研（仅当 M-P2 判定单卡不够） | TensorLBM（`docs/L1_BLOCK_SHARDING_DESIGN.md`） | 2 卡 halo-exchange 跑同一 case，**判据：与单卡解 L2 差 ≤1% 且加速 ≥1.5×**（否则多卡无意义） | 2 卡 × 6 卡时（从 D3 池借） |

**③ 卡位**：6 卡（S1/S2/S3 各 2 卡）；S4 用同一 6 卡的空档，不新增卡。

**④ 风险与依赖**
- **强依赖 M-P2 的显存表**。3D 主力网格若超单卡，则全部转 S4 分块路线。
- 风险 R-threshold：S2 的"质量漂移阈值"目前没有 CPU 基线可引用（我没在 worktree 里找到已锁的 dam break 3D 质量漂移数），**先测后锁**，不许拍数字。
- 风险 R-precision：自由面/相场对 fp32 累加顺序敏感，跨卡逐位可能不成立 → 逐位判据 fail 时降级为"≤1% 统计一致 + 明确记录不逐位"。

**⑤ 放弃条件（不做）**：若 M-P2 显示单卡跑不动 D40 球且 S4 分块在 2 个 mission-day 内拿不到收敛解，3D 线下马，本视角只保留 2D + 多相（2D 网格便宜），并如实上报"3D 满卡需要先解决显存/分块"。

---

## 5. 方向 D4：5 个 triton 依赖模块"去 triton 化" + dtype 策略（小、快、直接解硬约束①）

**① 目标一句话**：把 m11 里的 5 个 B 类模块改写成纯 torch，让 TensorLBM 从 507/512 变成 512/512 全程全速；并给出 fp32/tf32/bf16 的"精度-速度"推荐清单。

**② mission 粒度拆解**

| ID | mission | repo | 机器判据 | 预算 |
|---|---|---|---|---|
| M-T1 | 5 模块纯 torch 重写 | TensorLBM | **判据：`TRITON_INTERPRET` 关闭时 import 成功且相关 pytest 子集全绿；纯 torch 版 vs triton 解释模式数值相对差 ≤1e-6；单步耗时 ≤ 现有 CPU 路径**（triton 解释模式只做正确性对照，禁作性能依据） | 2 卡 × 6 卡时 |
| M-T2 | dtype 策略探测 | TensorLBM | 在 Couette 2D + Taylor-Green 2D（解析判据）上测 fp32/tf32/bf16；**判据：仅当对解析解误差 ≤0.1% 且加速 ≥1.3× 的 dtype 才进推荐清单，否则写"不推荐"**；torch_sdaa 对 bf16/tf32 的支持先探测（F11 未给） | 1 卡 × 4 卡时 |

**③ 卡位**：3 卡（T1 2 卡，T2 1 卡）。

**④ 风险与依赖**：依赖 m11 的 B 类清单落地（[ctx] 只给了数量 5，**具体是哪 5 个模块我未在本 worktree 核到**，M-T1 的第一步就是把清单钉死成文件路径）；重写有数值漂移风险 → 用 1e-6 相对差卡住。

**⑤ 放弃条件（不做）**：若 5 个模块中的某个在纯 torch 下必然引入全局同步（例如必须 host 侧时间步控制），改写后性能反而更差，则保留原实现并标注 `TRITON_ONLY`，不硬改。

---

## 6. 32 卡卡位分配总表（合计 32）

| 车道 | 卡数 | 时段 | 约束 |
|---|---|---|---|
| 扫荡线（既有 in-flight m12–m23，12 案例）[ctx] | 12 | 连续 | 非本提案，先占先得 |
| **D1 P-线（性能基准线）** | 5 | 第 1–2 日**独占**；之后缩为 1 卡常驻门禁，4 卡归还机动池 | 计时窗口禁 co-tenancy |
| **D2 C-线（普查/裁决）** | 2 | 第 1–3 日 | 1 卡冒烟 + 1 卡（实际吃 CPU） |
| **D3 S-线（3D/多相/自由面）** | 6 | 第 2 日起连续 | 3 mission × 2 卡 |
| **D4 T-线（去 triton 化 + dtype）** | 3 | 第 1 日起连续 | 2 卡数值 + 1 卡性能 |
| 机动/热备（值班长抢修、失败重跑） | 4 | 连续 | 任何线可借，借还登记，不计入基线测算 |

派发口径：P-线 2 人（impl×1 + ver×1）、C-线 1 人（arch×1）、S-线 3 人（impl/ver 混）、T-线 1 人（impl）。**议事会批了才派**，我不派发。

**今日就能开工的 3 条（建议第一批）**：M-P1（1 卡）、M-P2（1 卡）、M-C1（CPU+1 卡）——三条互不依赖，当天出数。
次日：M-P3/M-P5（门禁）+ M-C2；第三日：M-S1/S2/S3 与 M-T1 视 P-线结论放行。

---

## 7. 不确定 / 待实测（诚实清单）

1. SDAA 单卡显存容量、单卡可持续功耗/降频行为 —— 上下文未给，M-P2 第一产出。
2. torch_sdaa 3.1.1a3 对 bf16/tf32/torch.compile 的实际支持面 —— 未测，M-T2 先探测再谈加速。
3. TCCL 3.1.0 跨卡 all-reduce 实测带宽与稳定性 —— 未测，M-P4 若探测 fail 则多卡线全部 `WITHHELD`。
4. m11 的 5 个 triton 依赖模块具体是哪 5 个 —— 本 worktree 未核到清单文件，M-T1 第一步钉死。
5. 其余 9 个 tensor 库的代码现状 —— 未实读，D2 结论只能由普查脚本产出；我的排序是先验。
6. dam break 3D 的质量守恒 CPU 基线数 —— 仓库里未找到已锁数值，M-S2"先测后锁"。

---

## 8. 全局放弃条件（触发即停，并出负面报告归档）

1. **度量不可控**：3 个工作日内 D1 拿不出 CV≤2% 的可复现基线 → 全线停止性能承诺，卡转为长时批量算例。
2. **生态不可移植**：M-C1 显示 9 库普遍依赖 CUDA C++ 扩展/triton（LoongArch 无镜像、github 443 不通）→ 取消"第二个库上卡"，集中 TensorLBM 满卡。
3. **显存不够**：3D 主力网格超单卡且分块 2 mission-day 无收敛 → 3D 线下马，只留 2D/多相 2D。
4. **扫荡线要卡**：43 案例全覆盖 [ctx] 需要更多卡时，D1/D2 让卡（可暂停），D3/D4 保留最小 4 卡。
5. **基线数值不可信**：任何 baseline JSON 缺环境指纹（torch/torch_sdaa 版本、device 名、git sha）→ 门禁拒收，该结论作废重跑。

---

## 9. 我请议事会裁决的 3 个点

1. D1 的 5 卡**独占窗口**能否拿到（这是本视角全部数字的前提；共卡 = 白跑）。
2. D2 的裁决权是否归"普查脚本 + 第三方复跑"，而不是归任何一个提案人的先验排序（包括我的）。
3. D3 的 3D 是否接受"先看 M-P2 显存表再放行"，而不是今天就把 6 卡锁死给 3D。