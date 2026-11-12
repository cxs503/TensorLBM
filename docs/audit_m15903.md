# 交差审计 · m15658 = GAP-LBM-T4-PHYSICS-AI-LOOP（TensorLBM）

- **审计单**：m15903（任务榜，`verdict_by='board-audit'`）
- **被审单**：m15658，status=succeeded，fail_stage=`budget_exhausted+forced_finish`（mission 自报 honest PARTIAL）
- **被审工作树**：`/root/tp-agent/work/m15658`（HEAD 7e71cfab `m15658: matured mission deliverables`）
- **verdict**：`tensor_maturity_verdicts.id=406`，**level = 2**（独立功能实现），scope `m15658 GAP-LBM-T4-PHYSICS-AI-LOOP`
- **结论一句话**：真实 LBM 流场生产器 + 带 sha256 的数据集 + 可执行的 FNO/反哺代码路径 = 成立（L2 实打实、可复现到逐位）；
  但**锚点的另一半「AI 闭环」只是代码未跑通**，审计方代跑后实测精度 45.6%/107%，**远出 15% 不可接受线**，
  且无基准裁判行、无 platform 报告、工件未随 PR 入 main → L3/L4/L5 全部不成立。

---

## ① 案例介绍（物理背景 / 几何 / 参数表）

锚点要求「T4 真实流场数据生产与 AI 闭环」。被审单选的物理算例是多模态衰减 Taylor–Green 涡（D2Q9 BGK）：

| 项目 | 值（来自 `dataset/manifest.json` provenance，非手抄） |
|---|---|
| 求解器 | tensorlbm D2Q9 BGK（`equilibrium` / `collide_bgk` / `stream` / `macroscopic`） |
| 初始场 | 6 模态 `[(1,1),(2,1),(1,2),(2,2),(3,1),(1,3)]`，single_mode=False |
| 网格 | n=128（18 个工况）+ n=64（6 个工况） |
| u0 | 0.05 lu，Mach = u0·√3 = 0.0866 |
| Re 定义 | Re = u0/(k1·ν)，k1 = 2π/n → ν = u0/(k1·Re)，τ = 3ν+0.5 |
| Re 集合 | train 50/100/200/400；**holdout 75/300（训练中从不出现）** |
| 步数 / 采样 | 5000 步，每 250 步采样 → 21 快照/工况，共 **504 快照** |
| 认证标尺 | Taylor–Green 单模态解析解 u_rms = u0/√2·exp(−2νk²t)（G. I. Taylor 1923） |

## ② 怎么做的（方法 / 关键设置 / 工具链 / 代码指针）

| 环节 | 代码指针 | 审计判读 |
|---|---|---|
| 流场生产 | `src/tensorlbm/ai/flow_field_loop.py:249 run_case` | 真 LBM 步进（collide_bgk→stream），非解析注入 |
| 生产端认证 | `:292 verify_producer`（判据 field<1%、decay<2%） | 真跑 + 解析标尺；审计方**独立重推解析式**复核（不调用其 `analytic_single_mode`） |
| 数据集资产 | `:393 produce_dataset` → npz(21,128,128)×24 + `manifest.json` sha256 + `catalog.sqlite`（assets 25 / metadata 52 / lineage 24） | 真实文件、哈希可复核 |
| 代理模型 | `:641 train_flow_map`（FNO2d，width=12，2 层，modes 6×6，12 epoch，best_val_mse=0.0548，train_wall_s=599.68） | 真训练，checkpoint 97049 B 可加载 |
| 留出评价 | `:786 evaluate_holdout`（fno vs viscous_prior vs persistence） | **mission 从未运行** |
| 反哺 LBM | `:887 feedback_acceleration`（代理 rollout k 步 → 真 LBM 接手 → healing） | **mission 从未运行** |
| 驱动 | `bench_t4_physics_ai_loop.py --quick/--stage all|produce|train` | 可执行 |
| 报告渲染 | `tools/t4_report_from_results.py` 读 `artifacts/.../results.json` | **该文件不存在 → 报告无法生成** |

工具链：Python 3.11 + PyTorch（torch 线程需限制，见 ⑤）；审计方复现脚本
`artifacts/audit_m15903/audit_repro.py`（A/B/C/D 四段，逐段落 `evidence.json`）。

## ③ 计算结果（数字表，全部程序化注入 `artifacts/audit_m15903/evidence.json`）

### 3.1 工件存在性（A 段）

| 检查 | 结果 |
|---|---|
| manifest 声明文件数 / 实际找到 / sha256 匹配 | 24 / 24 / **24（0 不匹配）** |
| 声明快照数 / 声明墙钟 | 504 / 263.89 s |
| `results.json` | **不存在** |
| `figures/` | **空目录** |
| `docs/T4_PHYSICS_AI_LOOP.md` | **不存在** |
| `model/fno_flow_map.pt` | 存在（97049 B），但 **不在 git**（`git cat-file -e 7e71cfab:...pt` 失败） |
| `tensor_reports` 中 mission_id=15658 的行 | **0**（平台报告未发） |
| 并入 main 的工件 | **无**（`git ls-tree HEAD artifacts/t4_physics_ai_loop/` 为空；PR #1846 → merge 0898713d 只带 5 个代码/测试文件） |

### 3.2 求解器可复现性（B 段，审计方重跑 `tgv_n128_re200_s0_t4`）

| 量 | 审计方重跑 | mission 记录 | 结论 |
|---|---|---|---|
| 墙钟 | 12.334 s | 12.054 s（manifest） | 一致量级 |
| max\|Δux\|/peak | **0.0** | — | **逐位复现** |
| max\|Δuy\|/peak | **0.0** | — | **逐位复现** |
| u_rms(t) 单调下降 | 是（0.05000 → 0.04149） | — | 真耗散解 |
| 初始场 FFT 显著谱点 | 12 个 bin >1% 峰值 | 声明 6 模态 | 非单模态解析场（反造假检查通过） |

### 3.3 生产端认证（C 段，审计方独立标尺）

| n | Re | 审计方 max 相对误差（场） | 衰减率相对误差 | 判据（<1% / <2%） |
|---|---|---|---|---|
| 64 | 50 | 0.2964% | 0.1244% | ✅ / ✅ |
| 64 | 200 | 0.3096% | 0.7123% | ✅ / ✅ |
| 128 | 50 / 200（mission 自报） | 0.1954%（max） | 0.0438% | ✅ / ✅ |

审计方解析式自行推导（u=(u0 sin kx cos ky, −u0 cos kx sin ky) → rms=u0/√2，∇²u=−2k²u → exp(−2νk²t)），
与 mission 的 `analytic_single_mode` 一致 → 标尺未被"调参迎合"。**生产器子部件：过。**

### 3.4 AI 闭环（D 段，审计方用 mission 交付的 checkpoint 代跑——**非 mission 工件**）

留出集 = 8 个 run（Re=75/300，n=128×6 + n=64×2），一步窗口 = Δt = 250 LBM 步：

| 指标（rel L2 平均） | FNO（交付模型） | viscous_prior 基线 | persistence 基线 |
|---|---|---|---|
| one_step | **0.4563** | 0.9525 | 0.9519 |
| rollout_final（4×Δt = 1000 步） | **1.0730** | 1.1950 | 1.2844 |

反哺协议（holdout Re=300，k=2 代理步 → LBM 接手，healing 250 步）：

| run | handback rel L2 | healed rel L2 | 墙钟加速比 |
|---|---|---|---|
| re300_s0 | 0.6567 | 0.7412 | 33.1× |
| re300_s1 | 0.6687 | 0.7658 | 33.1× |
| re300_s2 | 0.7047 | 0.7495 | 29.5× |
| n64_re300_s0 | 0.9818 | 0.9903 | 45.7× |

## ④ 结果图

`artifacts/audit_m15903/fig_audit_m15903.png`（左：留出集误差三方对比；右：生产器 u_rms 与独立解析标尺），
并由报告 figures 内嵌。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **torch 线程数**：不设 `torch.set_num_threads(8)` 时，n=128/5000 步从 12 s 膨胀到 >120 s（本机 128 核线程抖动）。审计脚本已固化。
2. **checkpoint 不在 git**：`fno_flow_map.pt` 只存在于 mission 工作树 → 仓库侧无法复现 D 段，必须重训 600 s。
3. **多模态初始场**：数据集的 24 个工况是 6 模态非线性场，不能用单模态解析解直接判；生产端认证用的是另跑的 single_mode 工况（`verify_producer`），两者不可混判。
4. **一步误差窗口是 250 LBM 步**（采样间隔），不是 1 步；所以 persistence 基线 one-step 也接近 0.95。FNO 的 0.456 是"2× 于持久性基线"但绝对误差仍不可用。
5. **加速比是墙钟比，不是正确性声明**（代码 docstring 也如此写）；33× 加速下 66% 的 handback 误差不能作为"加速"成果。
6. 审计方 D 段结果**归属审计方**，不能记到 mission 账上；本次用它证明"功能未闭合"，而不是替 mission 补作业。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 关键量 | 基准 | got | 误差 | <3%? |
|---|---|---|---|---|
| LBM 场 vs 解析（single-mode, n=128，mission 自报并审计方 n=64 复核） | Taylor–Green 解析解 | 0.1954%（n=128）/ 0.3096%（n=64） | — | ✅ 是 |
| 衰减率 vs 解析 2νk² | 解析 | 0.0438%（n=128）/ 0.12%~0.71%（n=64） | — | ✅ 是 |
| 代理 one-step rel L2（holdout） | 无可接受基准 = <15%（平台线）；无 mission 判据 | 0.4563 | **+45.6 pp** | ❌ 否（差 3 倍以上） |
| 代理 rollout rel L2 | 同上 | 1.0730 | **+107 pp** | ❌ 否 |
| 反哺 handback rel L2 | 同上 | 0.657~0.982 | ❌ | ❌ 否 |
| 基准裁判行（anchor 级） | `tensor_benchmark_cases` 应有 case | **0 行**（全表 25 行；TensorLBM 仅 3 行，均非本锚点） | — | N/A —— **无机器裁判，L4 不可确认** |
| `tensorbench_results` 行（mission 15658） | ≥1 | **0** | — | N/A |

**判据结论**：生产器侧三项全部 <3%（成立）；AI 闭环侧无任何 <3% 指标，主指标 45.6%/107%（不成立）。

## ⑦ 判级与处置（adversarial review 汇总）

**Level = 2**。理由链：

- L1 以上成立：不是文档成果，有真代码与真跑。
- **L2 成立**：真实 LBM 求解器 + 数据集资产 + 可训练/可评价/可反哺代码路径，输入→计算→输出闭合，且审计方独立复现（含逐位复现 12.33 s 单算例、24/24 sha256、10/10 pytest）。
- **L3 不成立**：`flow_field_loop` 无任何调用方（`grep -rn 'flow_field_loop' src/ app/ tools/` 仅命中报告工具 docstring；`tensorlbm/ai/__init__.py` 未导出）。仅"代码进了 main"（merge 0898713d），服务/平台侧无消费者。
- **L4 不成立**：锚点无 `tensor_benchmark_cases` 行（无裁判）；mission 无 `tensorbench_results` 行；工件未并入 main；AI 侧定量主张被审计方实测**反证**。
- **L5 不成立**。

**severity-ranked findings**

| # | 严重度 | 发现 | 证据 |
|---|---|---|---|
| F1 | **blocking** | 锚点"AI 闭环"未闭合：留出评价/反哺量化/图/`results.json`/docs/平台报告全部缺失；审计方代跑显示精度不可用（45.6% / 107%） | D 段表；`find artifacts/... results.json` 空；`tensor_reports` 0 行 |
| F2 | **blocking** | 无机器裁判：anchor 在 `tensor_benchmark_cases` 无行，`tensorbench_results` 0 行 → L4 客观不可证 | 两次 SQL 查询 |
| F3 | high | 数据集/checkpoint/认证工件**未随 PR 入 main**（gate 过滤）。仓库侧无法复现 D 段；只有 mission 工作树里的 64 MB 副本 | `git ls-tree HEAD artifacts/t4_physics_ai_loop/` 空；`git cat-file -e` 失败 |
| F4 | medium | 平台报告未发（mission 单自身违反报告纪律） | `SELECT count(*) … mission_id=15658` = 0 |
| F5 | medium | 训练不足即停：12 epoch、val_mse 0.0548 仍下降，无早停解释；且训练/留出协议（250 步窗口）使 persistence 基线也达 0.95，评价窗口选择未论证 | `train_summary.json` history |
| F6 | low | `tools/t4_report_from_results.py` 依赖不存在的 `results.json`，无降级路径 | 文件/工件清单 |

**positive（也记）**：生产器侧是被审单最硬的部分——真 LBM、24 个工况哈希自洽、审计方独立标尺 0.30% 场误差 / 0.12~0.71% 衰减率误差、单算例逐位可复现。这一块不虚。

**disposition**：`approve with notes`（L2 成立，不得升格）；锚点"AI 闭环"部分**建议保留在账上继续**（reopen 余量），
不应因本 verdict 被整体划掉。另外建议平台为 TensorLBM 补一条 anchor 级
`tensor_benchmark_cases`（TGV 衰减 / 留出 rel L2 阈值），否则同类锚点永远无法 L4。

**未跑 / 未知（not_run，不得记为 passed）**：
- mission 侧原始 `bench_t4_physics_ai_loop.py --stage all` 端到端（含重训 600 s）未由审计方完整重跑（预算）；
  审计方以 D 段"加载已交付 checkpoint + 真数据集"覆盖了留出/反哺路径，但**训练环节未重训验证**。
- `catalog.sqlite` 与平台数据资产表（TD/TK）的同步未核（本地 sqlite 有 assets 25 / metadata 52 / lineage 24）。
- 集群/GPU 路径未核（本机 CPU 复现）。

## 复现命令

```bash
cd /root/tp-agent/work/m15658
python -m pytest tests/test_flow_field_loop.py -q                     # 10 passed, 14.63 s
python3 /root/tp-agent/work/m15903/artifacts/audit_m15903/audit_repro.py A   # 24/24 sha256
python3 /root/tp-agent/work/m15903/artifacts/audit_m15903/audit_repro.py B   # 逐位复现 12.33 s
python3 /root/tp-agent/work/m15903/artifacts/audit_m15903/audit_repro.py C   # 独立标尺认证
python3 /root/tp-agent/work/m15903/artifacts/audit_m15903/audit_repro.py D   # 留出 + 反哺（审计方）
```