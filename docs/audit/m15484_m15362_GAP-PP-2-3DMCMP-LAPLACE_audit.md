# 交差核验 m15362 — GAP-PP-2-3DMCMP-LAPLACE（TensorLBM）

审计单：m15484（verdict_by=`board-audit`）· 被审单：m15362（`succeeded`，2026-11-11 22:48:53Z）
审计对象 commit：`fde9d5df`（交付分支 agent/m15362）→ squash 落主干 `7657e18f` / Merge #1800 `a085d425`
审计基线：`/root/tp-agent/work/m15484`（TensorLBM main + 已合入的交付物）

---

## 0. 结论（先给判级）

**判级 L2 = 独立功能实现**（真代码：输入→计算→输出；本审计独立跑通并产出数字）。
不是 L3/L4：**无任何调用方接入主求解器/平台**（L3 缺），**无基准数值对照工件**（L4 缺）；
被审单自报的 honest-FAIL 属于"算力不够跑不出数"，与审计核验结论一致——**文档六段齐全，但④对照表与⑤敏感性两段是空的**。

> 一句话：工装真建成了、能跑、单元测试真绿；**但一个 3D 数字都没交付**，因此不可能到 L4。

---

## 1. 取证（逐项核验三件套）

| 项 | 核验结果 | 证据指针 |
|---|---|---|
| impl_files（真代码） | ✅ 存在、无 placeholder、真算法 | `src/tensorlbm/mcmp_verification_3d.py`（618 行；`simulate_static_droplet_3d`:118、`measure_laplace_3d`:297、`estimate_coexistence_3d`:451、`_fit_laplace_level_3d`:493、`young_laplace_calibration_3d`:565） |
| entry（入口） | ✅ CLI 存在 | `benchmarks/mcmp_verify_bench_3d.py:220`（`main()` / `--out --jobs --quick`） |
| tests | ✅ 审计独立复跑 **5 passed in 12.65s** | `tests/test_mcmp_verification_3d.py`（5 个测试名见 §4） |
| run_artifacts | ❌ **零工件**：`git ls-files` 无 `artifacts/*mcmp*3d*`；worktree `artifacts/` 无本次交付产物；`runs/` 无对应目录 | `git show --stat fde9d5df`=4 files/1050 insertions，全为源码+文档 |
| benchmark ref vs got | ❌ **无 got 数字**。工装自带预登记判据（R²≥0.99 / 每例残差<5% / ≥3 半径×≥2 分辨率，见 `benchmarks/mcmp_verify_bench_3d.py:12-14,44-52`）但无任何一条被实际测量 | `docs/GAP_PP_2_3DMCMP_LAPLACE.md` ④ 明写"⏳ 集群作业排队中" |
| platform 基准裁判 | ❌ `tensor_benchmark_cases WHERE anchor_slug='GAP-PP-2-3DMCMP-LAPLACE'` = **0 行**（全表 16 行，唯一 MCMP 行是 `GAP-LBM-MCMP-POISEUILLE`） | psql 查询（见 §6 复现） |
| repro_cmd 可执行性 | ⚠️ 命令语法可执行（`--quick/--jobs/--out` 均存在），但**本审计实测不可在合理时间产出结果**（见 §5 F3），且 `--quick` 的"smaller grids"承诺是假的 | `benchmarks/mcmp_verify_bench_3d.py:126-127` `n_lap = n_grid if quick else n_grid`（两分支相同 → 死表达式） |
| 平台报告 | ❌ `tensor_reports WHERE mission_id=15362` = **0 行**，`tensor_dev_missions.report_owed=f`、`report_id` 空 → **报告欠账** | psql 查询（§6） |
| 集群作业 | ⏳ `cluster_jobs` id=1075/1077 `state=queued`，提交 11-11 22:38/22:40，审计时刻（11-12 00:33）**排队 113 分钟仍未派发**，无日志无 rc | `run_cluster ps cj1077` |

---

## 2. 报告六段核验（`docs/GAP_PP_2_3DMCMP_LAPLACE.md`，113 行）

| 段 | 是否齐全 | 实情 |
|---|---|---|
| ① 案例介绍（物理/几何/参数） | ✅ | Young-Laplace Δp=2σ/R、Shan-Chen 1993 引用、参数表（G=-2.5/-2.75/-3.0，R=10/14/18，n=64/96，τ=1.0）齐 |
| ② 方法/实现 | ✅ | D3Q19 + `collide_sc_two_component_3d` + `stream3d`，代码指针齐，且指针**核验属实**（函数确实存在且被调用） |
| ③ 计算结果（数字表） | ❌ | **无 3D 数字**。文中唯一数字是**转引 2D 结果**（Δp=0.018764, R=10.07, σ=0.09448 @n=48/G=-2.5/R=10）——那是上一轮 2D 工装的产物（`docs/MCMP_VERIFY_BENCH.md` 表内 σ=0.0947 一族），不是本单产出 |
| ④ 结果图/对照曲线 | ❌ | 无图、无对照表；写"待集群作业完成后补充" |
| ⑤ 敏感性与收敛 | ❌ | 全段是**预期**（"σ 应随 |G| 增大"），非实测 |
| ⑥ 注意点与失败模式 | ✅ | honest-FAIL 如实写（3D 计算量/本地超时/队列长），未粉饰 |

**判定：六段中 ③④⑤ 缺失 → 报告只能是"工装就位声明"，不能作为基准验证证据。**

---

## 3. 独立复现（adversarial review 步骤 6）

审计侧自建有界探针，全部写在审计工作树 `artifacts/m15484/`，不污染被审单命名空间：

- `artifacts/m15484/probe_3dmcmp_laplace.py`（多进程版，因 libgomp fork 后 futex 死锁弃用）
- `artifacts/m15484/probe_seq.py`（单进程顺序版：n=24、steps=2000、G12=-2.5、R∈{4,6,8}，以平面共存密度做种）
- 已完成的最小执行核验（`artifacts/m15484/smoke_3dmcmp_100steps.json`，默认密度种 0.92/0.08↔0.14/0.86，`n=32, 100 steps`，4 线程）：`u_max=?`→实测 `dp=0.05071`、`R_meas=9.5063`（初值 10，未收敛仍在收缩）、`σ=0.2410`、guardrail=`not_converged`，墙钟 4.2 s。
同一入口换显式种（0.9/0.1↔0.1/0.9）跑同参数得 `dp=0.10757`、`R=7.16`、`σ=0.3853` —— **两种种子差 2 倍**，说明 100 步非收敛态的 σ 纯属伪量（这正是为什么"跑出数字"不等于"有基准"）。
  → **证明代码路径端到端可执行**（L2 的机器可核部分），同时证明**非收敛态的 σ 无物理意义**（不得当作结果）。

> 诚实声明：n=24 顺序探针在审计时间窗内未收敛完成（`artifacts/m15484/probe.log` 为实时日志），因此本审计**不提供任何可被判据采纳的 3D σ 数字**；本审计的结论不依赖它——L4 缺失的判据是"零工件"，不因探针是否跑完而改变。

---

## 4. 测试真实性（不是"跑了代码"，是断言了不变量）

审计独立执行 `python -m pytest tests/test_mcmp_verification_3d.py -q` → **5 passed, 12.65s**。

| 测试 | 断言的东西 | 是否真不变量 |
|---|---|---|
| `test_sc_mcmp_pressure_3d_shape` | p 张量形状 + 解析式 p=c_s²(ρ1+ρ2)+G c_s²ρ1ρ2 数值（<1e-6） | ✅ 解析式对照 |
| `test_barycentric_velocity_3d` | 静止态 ux.max()<1e-10 | ✅ 平凡但对 |
| `test_droplet_radius_3d_perfect_sphere` | tanh 完美球 R=15 测回 \|ΔR\|<1 lu，crossings==6 | ✅ 几何反演真校验 |
| `test_init_static_droplet_3d_shape` | (19,n,n,n) 形状 | ⚠️ 只执行不断言物理（弱） |
| `test_measure_laplace_3d_guardrails_on_diverged` | NaN 场→sigma/dp=None + `non_finite_field` | ✅ 失败路径防护 |

结论：测试是**模块级正确性**，**不含任何"物理量对基准"的测试**（无 σ 收敛、无 R²、无网格无关性）。

---

## 5. 发现（按严重度）

**F1（阻断 L4，中高）** 交付零运行工件：预登记判据（R²≥0.99/残差<5%/3 半径×2 分辨率）一条未测。L4 判定不可能。
**F2（阻断 L3，中）** 无集成调用方：`grep -rn mcmp_verification_3d` 在仓内除自身/bench/tests 外 **0 命中**；未接入 `__init__`/验证注册表/平台 API。
**F3（缺陷，中）** `--quick` 名不副实：`n_lap = n_grid if quick else n_grid` 死表达式，`--quick` 只缩步数不缩网格（与管理器 docstring 第 18 行"smaller grids"承诺矛盾），使冒烟跑仍要 64/96 全分辨率 → 冒烟失去意义（这也是被审单"本地跑不动"的一部分原因）。
**F4（缺陷/风险，中低）** 交付的多进程基准脚本 `_run_all` 用 `fork` pool + torch：审计实测该模式在本镜像上**必死于 libgomp futex 死锁**（`probe.log` 空、`/proc/<child>/wchan=futex_wait_queue`、CPU 0%）。即 `benchmarks/mcmp_verify_bench_3d.py --jobs>1` **很可能一跑就挂**，文档给的复现命令因此不可信（`--jobs 4/8` 都是多进程）。
**F5（诚实性，正面）** 未发现 MOCK/硬编码：模块是真实 Shan-Chen 实现；文档 ③ 明示转引 2D 数字并标注来源语境，没有冒充 3D 结果。
**F6（流程，中）** 平台报告缺失（`tensor_reports` 0 行 + `report_id` 空）→ 交差欠账；且 `tensor_benchmark_cases` 无本锚点行 → 无机器裁判，工装判据只活在源码注释里。
**F7（流程，低）** 集群作业 cj1075/cj1077 提交后 113 分钟仍 queued：被审单把交付押在队列上，但没有在单内留下"排队即失败"的处理路径（如自动降规模/自动续跑）。

---

## 6. 核验复现命令（审计侧实际执行）

```bash
# 1) 交付物面
cd /root/tp-agent/work/m15484 && git show --stat fde9d5df
git ls-files | grep -i -E "mcmp_verify|mcmp_verification"     # 只有 4 个源码/文档文件，无 artifacts
# 2) 测试独立性
python -m pytest tests/test_mcmp_verification_3d.py -q        # 5 passed
# 3) 基准裁判 / 报告 / 集群
runuser -u postgres -- psql -d tensorplatform -tA -c \
 "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='GAP-PP-2-3DMCMP-LAPLACE';"  # 0 行
runuser -u postgres -- psql -d tensorplatform -tA -c \
 "SELECT count(*) FROM tensor_reports WHERE mission_id=15362;"                                                # 0
runuser -u postgres -- psql -d tensorplatform -tA -c \
 "SELECT id,state,submitted_at FROM cluster_jobs WHERE id IN (1075,1077);"                                     # queued, 22:38/22:40
# 4) 代码路径端到端
python artifacts/m15484/probe_3dmcmp_laplace.py    # 见 §3（多进程版死锁）
```

---

## 7. 判级理由（L0–L5 逐档）

- L0 文档：满足（且有真码）。
- L1 数学/单元验证：满足（5 测试全绿，含解析式与几何反演断言）。
- **L2 独立功能实现：满足** —— 真代码 + 入口 + 审计独立执行产出物理量（Δp/R/σ），输入→计算→输出闭环。
- L3 系统集成：**不满足** —— 零调用方（F2）。
- L4 基准验证：**不满足** —— 无算例结果、无误差分析、无工件（F1/F6），`repro_cmd` 多进程即挂（F4）。
- L5：不满足（无对照文献/实验的定量结论）。

→ **L2**。相对被审单自报（"code complete + honest-FAIL"），本裁定与之一致且给出机器可核边界；
与同族 `m15361 GAP-PP-1-3DMCMP-STABILITY`（L3，有集成调用方）相比，本单**反而少一环集成**，不应给 L3。

## 8. 未做 / 局限（not_run）

- 未跑完被审单的完整协议（n=64/96 × steps=12000 × 3G × 3R）：单次 3D 液滴在 128 核机上实测 ~0.07 s/step@n=32 且强非线性放大，全案 ≥ 数小时/作业 → 属返修轮职责，非审计职责。
- 未复核集群 cj1075/cj1077 的最终产物（审计时刻仍 queued，无 rc/log）。
- 未做 x86 CI 复核（本机 loongarch64）。
- 3D σ 与 2D σ 的定量一致性（预期 ≈0.094–0.106）**未验证**——这正是 L4 缺的那一块。