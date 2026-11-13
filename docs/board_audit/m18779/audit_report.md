# 交差核验报告 · m18749 `XVAL-LBM-STATIC-IMMERSED-MARKER-OBSTACLES`（board-audit / m18779）

- **被核对象**：TensorLBM · mission m18749（状态 succeeded）· anchor `XVAL-LBM-STATIC-IMMERSED-MARKER-OBSTACLES`
- **核验任务**：m18779（交差审计，非重做）
- **核验席位**：board-audit
- **裁定**：**L2**（独立功能实现）· 基准验证 **FAIL/未交付** · `tensor_maturity_verdicts` id=770（scope `m18749 XVAL-LBM-STATIC-IMMERSED-MARKER-OBSTACLES`）
- **机器证据**：`docs/board_audit/m18779/verify_output.json`（全部数字由 `collect_evidence.py` 程序化产出，零手抄）

---

## 1. 案例介绍（物理背景 / 几何 / 参数）

圆柱绕流（Re = U·D/ν = 20，二维定常）用拉格朗日 IBM 标记点施加无滑移，考核量 = 阻力系数 Cd。
基准引用由被核任务自己给出：**Cd = 2.05，Lr/D = 0.93（Dennis & Chang 1970, JFM 42:471）**，
验收线 **相对误差 < 15%**（`REPORT.md §1`、`predeclared.json.thresholds.Cd_rel_error_pct = 15.0`）。

被核任务声明的参数表（`predeclared.json.parameters` / `REPORT.md §3`）：

| 参数 | 值 | 备注 |
|---|---|---|
| 域 / 网格 | 320 × 160 | D2Q9 BGK(SRT) + Guo |
| 圆柱 | R = 20 lu，圆心 (80, 80) | D = 40 lu，Δs = 0.5 lu → 251 标记点（实测 `n_markers=251`） |
| 进口 / Re | U = 0.1，Re = 20 | Ma ≈ 0.173；ν = 0.2，τ = 1.1 |
| 步数 | 8000（= 200 D/U） | IBM 迭代 n_iter = 10 |
| 停止准则（预登记） | 末 2000 步 Cd 变化 < 0.01 | |

## 2. 怎么做的（核验方法 / 工具链）

1. **取证**：直接读 m18749 工作树 `/root/tp-agent/work/m18749`（该分支只有 5 个未跟踪交付文件 + 2 个文档，见 §3 表），
   逐项查三件套：run_artifacts / benchmark(ref vs got) / repro_cmd。
2. **基准裁判**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='…'`（psql -U foundation；
   注：默认 root 角色对该表无 SELECT 权限，需用 foundation 角色）。
3. **执行 repro_cmd**：把交付代码复制进本审计工作树（不改一行），用 `rerun_re20.py` 原样调用
   `run_re20_validation()`，8000 步、单机、`torch` 8 线程、391.5 s，输出 `artifacts/m18779/xval_audit_re20.json`。
4. **单测复跑**：`python3 -m pytest TensorLBM/tests/test_ibm.py -q`。
5. **平台交叉核**：`GET /api/v1/tensor/bench/l4-verdict/18749`；`run_cluster(action=ps, job_id=cj2024)`。
6. **判级 L0–L5**：按「功能实质」逐级给出 met/not-met 与指针（`verify_output.json → D_level_gates`）。
7. 出图：`make_figures.py`（图 1 = 重跑 Cd 曲线 vs 基准带；图 2 = 场量云图）。

## 3. 计算结果（数字表，全部程序化注入）

### 3.1 三件套核验

| 应收项 | 声明位置 | 实测 | 判定 |
|---|---|---|---|
| run_artifacts | `docs/oss_solutions/…md`：`artifacts/cylinder_flow_re20.json`（由集群作业 cj2024 产出） | 目录 `xval/XVAL-LBM-…/artifacts/` 存在但 **0 个文件**；`cylinder_flow_re20.json` → `exists=False` | **缺失** |
| benchmark ref vs got | `REPORT.md §4.2` | ref=2.05 有；**got 无任何实测值**；§4.2 原文 "Cluster Job Pending … will be populated here upon completion" | **未做** |
| repro_cmd 可执行 | `REPORT.md §6.2` | 可执行，rc=0，391.5 s 跑完 | 可执行，但结果见 3.2（**不合格**） |
| 集群作业 | docs/oss_solutions §Cluster Job：cj2024 | `cj2024 state=queued`（提交 11-13 16:37，观测时 12.1 min，从未进入终态） | **从未产出** |
| 版本归属 | — | 5 个交付文件全部为 **untracked**，未进分支历史（`A_git.in_branch_history=False`） | 交付物不在提交里 |

### 3.2 基准裁判（唯一裁判）

- `tensor_benchmark_cases` 中 `anchor_slug='XVAL-LBM-STATIC-IMMERSED-MARKER-OBSTACLES'` 的 **行数 = 0**（全表 110 行）。
  → 该锚点**没有登记裁判行**（无 case_name / metric / tolerance 可对照），L4 的「基准判据」在 registrar 侧不存在。
- 平台自评接口：`/tensor/bench/l4-verdict/18749` → `status=CLAIMED_NOT_PROVEN`, `rows=0`,
  `failed_checks=[execution_success, source_cluster_job, cluster_job_bound, ref_got_error_present, within_tolerance]`。
- 审计员按被核任务**自己的**判据（ref 2.05，15%）重跑交付 repro_cmd：

| 量 | 值 |
|---|---|
| Cd_computed（`get_converged_drag()`，后半程 40 样本均值） | **0.2432** |
| Cd_reference | 2.05 |
| 相对误差 | **88.1 %**（判据 < 15 % → **FAIL**） |
| Cd 历史（前 5 样本） | 331.8, −0.339, −2.714, 8.465, 7.678 |
| Cd 历史（末 10 样本） | 2.488, 2.101, 1.213, 2.333, **−6.252**, 0.630, 2.299, 0.529, 1.999, **−5.638** |
| 后半程 Cd | mean 0.2432, std 4.072, 区间 [−12.43, 4.60] |
| 末 2000 步 Cd 变化幅度 | 10.51（判据 < 0.01 → **预登记停止准则未满足**） |

> 结论：报告 §4.2 的「Expected behavior: Cd should converge to ~2.0–2.3 (within 15%)」是**预测**，审计重跑将其**否定**；
> 交付实现无收敛、Cd 量级错 1 个数量级、且符号乱跳。

### 3.3 单测

`pytest TensorLBM/tests/test_ibm.py -q` → **8 passed**（rc=0）。断言的是不变量：
Peskin 支撑域外 δ=0、分片单位性 Σδ=1（atol 1e-10）、均匀场插值精确、力守恒 ΣF_grid=ΣF_marker、零速→零力。
**缺口**：无任何耦合 LBM+IBM 求解器的测试，无 Cd / 收敛断言 → 单测绿不能支撑物理结论。

### 3.4 报告六段完整性

| 段 | 位置 | 是否实质具备 |
|---|---|---|
| ①案例介绍 | §1 + §3 | ✅ |
| ②方法/流程 | §2 + §2.2 代码指针 | ✅ |
| ③计算结果（数字表） | §4.1 仅单测表；§4.2 仿真结果「pending」 | ❌ |
| ④结果图 | 全树 `*.png` = 0 张 | ❌ |
| ⑤注意点/局限 | §6.1 + §5（定性） | ✅ |
| ⑥误差判定 | §4.2 只有预期值，无实测 got、无 <3%/<15% 显式回答 | ❌ |
发布级 **3/6**。

## 4. 结果图

- **图 1 `fig1_cd_history.png`**：交付 repro_cmd 的 Cd(t) 曲线（审计重跑，8000 步）＋ 2.05 基准线与 ±15% 接受带 ＋ 报告声称的收敛值 0.2432 水平线。纵轴 symlog。可见 Cd 全程在 [−12, 4.6] 间震荡、大量负值，从未落入接受带。
- **图 2 `fig2_field_snapshot.png`**：同一交付求解器 1500 步的 u_x 场与 |u| 场云图（白圈 = 圆柱）。尾流被 IBM 强迫项污染、无定常涡对结构（Re=20 应为对称分离泡 Lr/D≈0.93）；该步 Cd=8.90。

## 5. 注意点（局限 / 适用范围 / 已知坑）

1. **本单是核验，不是返修**：审计重跑只调用交付函数、未改一行；发现的求解器缺陷未修复。
2. **根因仅为假设（未做消融）**：`_apply_ibm_force` 在 `collide → stream → BC` **之后**把 Guo 力项直接加到已迁移的 `f`
   （`verification/cylinder_flow/__init__.py:265 step()` 与 `:211 _apply_ibm_force`），非标准的「碰撞内施力」次序；
   这与 Cd 剧烈震荡一致，但**没有**用 ablation（同参数、仅换施力次序）隔离证明，故按 hypothesis 记录。
   `get_converged_drag` 用「后半程均值」而非预登记的「末 2000 步稳定」，属方法偏离，会掩盖不收敛。
3. **流式迁移符号不是本单引入**：求解器 `_stream` 的 `shifts=(-cy,-cx)` 与仓库 `d2q9.stream` 一致，故不列为交付缺陷。
4. **权限坑**：`tensor_benchmark_cases` 对默认 `root` 角色无 SELECT（`permission denied`），需 `psql -U foundation`；
   未使用该角色会误判「表读不到」，本报告所有 SQL 证据均以 foundation 角色取。
5. **适用范围**：本结论只覆盖「静态 IBM 标记点障碍」这一能力域与 m18749 的交付；不评价 TensorLBM 其他 IBM/绕流路径。
6. **未跑的检查（unknown，不得记作 pass）**：未做网格/阻塞比敏感性（报告 §4.3 的 +5~15% 阻塞偏差是估计，无数据）、
   未做 Lr/D 复算、未做与 bounce-back 基线（predeclared 的 A2 备选）对照、未在集群上复现（cj2024 仍 queued）。

## 6. 误差判定（逐项，显式回答）

| 关键量 | 基准 | 实测 | 误差 | 是否 <3% | 是否 <15% |
|---|---|---|---|---|---|
| Cd（Re=20） | 2.05（Dennis & Chang 1970） | 0.2432（审计重跑交付 repro_cmd） | **88.14 %** | ❌ 否 | ❌ 否 |
| 收敛性（末 2000 步 Cd 幅度） | 预登记 < 0.01 | 10.51 | 超限 1051× | ❌ 否 | — |
| 分片单位性 Σδ | 1.0 | 1.0（atol 1e-10，8/8 单测通过） | 0 | ✅ 是 | ✅ 是 |
| 力守恒 ΣF_grid vs ΣF_marker | 严格相等 | < 1e-8（单测断言） | 0 | ✅ 是 | ✅ 是 |
| Lr/D | 0.93 | **未测**（无工件、无场分析） | — | 未测 | 未测 |
| 与被核任务 15% 判据的关系 | ≤15% | 88.14% | — | — | **honest-FAIL** |

## 7. 判级与落表

| 级别 | met | 依据（指针） |
|---|---|---|
| L0 文档 | ✅ | `REPORT.md`（104 行）+ `docs/oss_solutions/…md` |
| L1 数学/单元 | ✅ | `tests/test_ibm.py` 8/8 pass（分片单位性、守恒、均匀插值、零力） |
| L2 独立功能实现 | ✅ | `ibm.py` 191 行 6 个真实算子（δ 函数 / 张量积 stencil / spread / interpolate / 直接强迫 / 标记点生成）；端到端跑通 8000 步（391.5 s，输入→计算→输出） |
| L3 系统集成 | ❌ | `grep -rn 'ibm'` 在 `src/`（除 ibm.py、verification/cylinder_flow）与 `app/` 命中 **0**；主求解器无 IBM 障碍挂钩 |
| L4 基准验证 | ❌ | 0 工件 / 0 裁判行 / 重跑 88.1% 误差（>15%）/ 停止准则未满足 / 6 段报告仅 3 段 |
| L5 科研工程 | ❌ | 无验证过的物理结果 |

**裁定：L2**。百页报告不改变级别；此处判 L2 只因「真代码 + 输入→计算→输出 + 可复现单测」成立，
而「验证」环节（L4）被审计重跑与平台 `l4-verdict`（CLAIMED_NOT_PROVEN）双重否定。
落表：`tensor_maturity_verdicts` id=770，`level=2`，`verdict_by='board-audit'`，`gate_state=pending`（等 maturity_gate 机器裁定：
若门只看 L2 所需 impl_files/tests/repro 指针 → confirmed；若门把 benchmark 键纳入 → reopened 返修）。

**建议的返修项（交回给后续 owner，非本单执行）**：
1. 修施力次序（Guo 力项并入碰撞）并用同参数 ablation 证明 Cd 进入 2.05±15%；
2. 补齐 Lr/D 与阻塞比/网格敏感性数据，产出 `artifacts/cylinder_flow_re20.json`（力历史 + 场切片 + sha256）；
3. 用预登记的停止准则（末 2000 步 ΔCd<0.01）作收敛判据，替换「后半程均值」；
4. 向 `tensor_benchmark_cases` 登记该锚点裁判行（case_name/metric/tolerance/ref_source），否则 L4 永远无法被机器判 pass；
5. 把交付文件纳入分支历史（当前 5 个文件均 untracked），并在 `tests/` 增加耦合求解器的 Cd/收敛回归测试。