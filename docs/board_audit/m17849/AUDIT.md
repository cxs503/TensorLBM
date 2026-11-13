# m17849 · 交差审计 — m17713 `XVAL-LBM-BENCHMARK-METHOD-SELECTION`（TensorLBM）

**对象**：m17713（task_key=`anchor-xval-lbm-benchmark-method-selecti-rework1`，status=succeeded）
**产物**：分支 `agent/m17713` @ `24b4af6a`（**未并入 main**：main=`be78a87b`，见 F3）
**裁定**：**L3**（L2/L3 满足，L4 明确不达）｜verdict 落表 `tensor_maturity_verdicts`，`verdict_by='board-audit'`
**处置**：L3 消账；L4 返修项=F1/F2（见 §6）

---

## 0. 独立性声明与审计范围

**独立性**：我是独立审计席（m17849），未参与 m17713 任何节点；所有数字从被审工件重算，不采信报告叙述。
**已知局限**（如实申报）：

* 被审分支**未并入 main**，因此全部证据取自 `agent/m17713 @ 24b4af6a` 的字节快照
  `docs/board_audit/m17849/refs/`（`git archive agent/m17713 | tar -x`，file:line 指针即指该快照）；
* 我没有重跑 D=14 梯级（被审件自身申报 not_run，见 §4/F2）；
* 我重跑单臂 D=10 时宿主 load average ≈155（128 核，生态内 300+ 作业在跑），
  单臂墙钟从交付件的 6.4 s 劣化到 >10 min（重跑进程在 10 min 22 s 时由审计终止），重跑状态见 §2 第 5 行（**未完成即标 not_run，不算通过**）；
* 我不复核 OSS 侧上游仓库的最新状态，只核 pinned commit `573a01c` 的 sha256 与执行结果。

**未运行的检查**：D=14 收敛梯级重跑、`run_matrix.sh` 全矩阵重跑、图 3（分辨率梯级）重绘、
OSS 全周期流动长期对照（被审件已声明不可做：OSS 边界不对称，见 §5 工件表）。

---

## 1. 三件套取证（run_artifacts / ref vs got / repro_cmd）

| 判据项 | 结论 | 真实指针 |
|---|---|---|
| **impl_files**（非占位实现） | ✅ 真代码：`run_arm.py` 395 行，构造算例→驱动库内核→写 result/fail JSON；`oss_crosscheck.py` 270 行，真**执行** OSS 源码 | `refs/xval/benchmark_method_selection/run_arm.py:1-395`、`oss_crosscheck.py:1-270`、`analyze.py`、`make_report.py:1-291` |
| **entry**（入口） | ✅ 三条：单臂 CLI、矩阵脚本、pytest 走的 bench 入口 | `run_arm.py`（`--arm/--D/--threads`）、`run_matrix.sh:1-17`、`refs/benchmarks/xval_method_selection_bench.py:1-98` |
| **run_artifacts**（真实运行工件） | ✅ 3 个成功臂 + 2 个诚实失败臂，含 trace/state npz | `refs/xval/benchmark_method_selection/runs/{A1,A2,A4}_D10/result.json`、`runs/{A3,A5}_D10/fail.json`（`A3` step 600、`A5` step 5800 发散）、`summary.json`、`table.csv` |
| **tests** | ✅ 我自己跑过：`5 passed in 49.16s`（其一 `test_oss_kernel_crosscheck_agrees` 真执行 OSS 源码） | `refs/tests/test_xval_benchmark_method_selection.py`（5 个测试，断言不变量而非只执行代码） |
| **integration_callers**（接入主求解器） | ✅ 基准直接调用库内核（不是私有重写）：`collide_bgk/collide_mrt/collide_trt/stream`、`far_field_bc_2d`、`make_sponge_strength`、`compute_obstacle_forces`、`equilibrium` | `refs/xval/benchmark_method_selection/run_arm.py:51-57`（`from tensorlbm.solver import ...` 等） |
| **benchmark**（ref vs got） | ⚠️ **无机器裁判**：`tensor_benchmark_cases` 中该 anchor **0 行**（58 个 anchor 里无 METHOD-SELECTION 类）；内部判据（A1 vs OSS 臂 <15%）**实测不通过**，外部锚 WITHHELD | 见 §3；`refs/xval/benchmark_method_selection/summary.json`、`docs/CASE-LBM-SQUARE-CYL-NORBERG-RE200.md`（primary Re=200 表 WITHHELD） |
| **repro_cmd** | ✅ 命令存在且我被审分支上实跑过 pytest 路径；单臂重跑见 §2 | `bash xval/benchmark_method_selection/run_matrix.sh`；`python xval/benchmark_method_selection/run_arm.py --arm A2 --D 10` |
| **external_anchor** | ✅ 真执行上游源码 + sha256 存档：`Maarten-vd-Sande/lbm @ 573a01c4…`（5 个文件 sha256 入 `oss_crosscheck.json`），numba shim 只改速度 | `refs/xval/benchmark_method_selection/oss_crosscheck.json`（`oss_source_sha256_unmodified`、`env_shim`） |

补充取证（不采信自述，重算）：

* 四张图**非空图**（PNG 非白像素占比 5.6 %–57.1 %，256 灰阶）；
* 交付 KPI 与自身 `table.csv`/`summary.json` 一致（重算偏差 0.0 %）；
* D=14 日志 **8 个里 3 个是 0 字节**（`A1_D14.log/A2_D14.log/A4_D14.log`），`summary.json.convergence == {}`
  ⇒ 收敛梯级确实**没跑**，报告 §5.1 的 not_run 自述为真（不是隐瞒）。

---

## 2. 我自己跑的检查

| # | 检查 | 命令 | 结果 |
|---|---|---|---|
| 1 | 集成测试 | `pytest tests/test_xval_benchmark_method_selection.py -q`（worktree `/root/tp-agent/work/m17713`） | **5 passed, 49.16 s**（含执行 OSS 源码的 kernel 对照） |
| 2 | 工件自洽 | `compute_audit_numbers.py` 从 `runs/*/result.json` 重算全部偏差 | 与交付 `table.csv` 一致（0.0 %） |
| 3 | 参数自洽 | `A5` 实参 `lambda_trt=0.01125, tau_minus=2.0` vs 交付文档 §5.2 称"字面移植 Λ=3/16" | **不一致**（F4） |
| 4 | 图像真伪 | PNG 非白像素占比 | 4/4 非空 |
| 5 | 单臂可复现（D=10, A2） | `run_arm.py --arm A2 --D 10 --threads 4`（`/tmp/audit_m17849/wt`，branch tarball） | 见 §2.1（宿主过载，如实标注） |
| 6 | 集群复算 | `run_cluster submit` cj1699（TensorLBM, stage_dir=快照） | 递交时排队 309 位，**未出结果，标 not_run** |

### 2.1 单臂重跑状态

宿主 load average ≈ 155（128 核），交付件记录的 A2 墙钟 6.4 s 在本轮重跑中劣化到 > 10 min
（`ms_per_step` 从 0.301 退化到 >20）。**该重跑未在审计窗口内完成 ⇒ 记为 not_run，不记为通过。**
可复现性的正面证据由第 1 行（pytest 真执行 OSS 内核对照，49 s 全绿）与第 2 行（工件自洽）承担；
数值可复现的**强**证据（逐位一致）本轮**未取得**，如实登记为 L4 缺口的一部分（F2 附注）。

---

## 3. 基准裁判（唯一裁判）

```sql
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='XVAL-LBM-BENCHMARK-METHOD-SELECTION';   -- 0 rows
SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug ILIKE '%METHOD%';  -- 0
```

**结论：该 anchor 在基准表里没有注册行 ⇒ 没有机器裁判。** 按"基准判据是唯一裁判"的规则，
L4 无法被确认（这不是报告写得好不好的问题，是裁判缺位）。审计据以判分的替代证据只有两层，逐项列：

**层一 · 交付件自定的内部判据（owner 15 %，ref=A1 记录选型，got=OSS 做法移植臂，同 D=10）**

| KPI | ref A1 | got A2 | Δ vs A1 | got A4 | Δ vs A1 | <15 % ? |
|---|---|---|---|---|---|---|
| Cd mean | 1.614326 | 1.466136 | −9.18 % | 1.642362 | +1.74 % | ✅ / ✅ |
| Cl amplitude | 0.657925 | 0.351552 | **−46.57 %** | 0.556625 | **−15.40 %** | ❌ / ❌ |
| St (FFT) | 0.168996 | 0.176629 | +4.52 % | 0.171767 | +1.64 % | ✅ / ✅ |

⇒ H1（"OSS 做法逐 KPI <15 % 可移植"）**被否**，最大偏差 **46.57 %**（Cl 幅值，A2）。这是被审件自己的负结论，
报告 §7 也如实写了 ⇒ **诚实 FAIL，不扣分**；但它同时说明本单**没有**达到"对标误差 <15 %"这一 owner 精度线。

**层二 · 代码级对照（ref = 真正执行的 OSS 源码）**：碰撞 rel-L2 `1.18e-07`、内部域流动 rel-L2 `2.71e-08`，
k0 方向映射 8/8 一致 ⇒ **<3 % 达标（实为 <1e-6）**。这一层是 L1/L2 级正确性证据，不能替代物理量对标。

**外部锚**：仓库自己的 `docs/CASE-LBM-SQUARE-CYL-NORBERG-RE200.md` 里 Re=200 primary 表 **WITHHELD**，
只给趋势带 Cd∈[1.30,1.45]、St∈[0.16,0.18] 且**声明不参与判据**。实测 A1：St=0.169 **在带内**；
Cd=1.614 比带顶高 **11.33 %**、比带心高 **17.41 %** ⇒ 若以带心为参照则 >15 %。**基准缺位 + 该带不可裁**
⇒ 精度维度判 **WITHHELD**，不判 REAL。

---

## 4. 判级（L0–L5）

| 级 | 判据 | 结论 | 依据 |
|---|---|---|---|
| L0 文档 | 有无成体系的文档成果 | ✅ | `refs/docs/XVAL_LBM_BENCHMARK_METHOD_SELECTION.md`（7 节，含误差判定） |
| L1 数学/单元验证 | 内核与解析/上游实现逐位一致 | ✅ | OSS 执行级对照 rel-L2 1.18e-07 / 2.71e-08；pytest 5 passed |
| L2 独立功能实现 | 真代码 输入→计算→输出 | ✅ | `run_arm.py` + `runs/*/result.json`（3 成功 + 2 诚实失败，各 21259 步） |
| L3 系统集成 | 接入主求解器 | ✅ | 基准驱动库内核 `tensorlbm.solver/boundary/d2q9`（`run_arm.py:51-57`），非私有重写；入口 + 集成测试在位 |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现 | ❌ | **三缺**：(a) 基准表 0 行，无机器裁判（§3）；(b) 收敛梯级 D=14 全 0 字节日志、`convergence=={}`，算例不完整（F2）；(c) 逐位可复现本轮未取得（§2.1）+ 分支未并入 main（F3） |
| L5 科研工程 | 领域级成果 | ❌ | 无外部实验/文献锚可裁，Cd 出带，不能主张科研级精度 |

**级别 = L3（功能实质，不因报告篇幅提升）。** 与上一轮审计（m17627 对 m17499 判 L3）级别持平：
m17713 的增量是**证据质量**（真执行 OSS 源码、5 臂矩阵工件、集成测试、预注册文件），
不是功能级跃迁——L4 的三处缺口一件都没补上。

---

## 5. 发现（按严重度）

**F1（blocking，L4 维度）· 该 anchor 无基准注册行**
`tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-BENCHMARK-METHOD-SELECTION'` 返回 **0 行**（全表 58 anchor，无 METHOD 类）。
影响：L4 的"唯一裁判"缺位，该单永远无法机器消账。复现：§3 的两条 SQL。
建议：按已交付工件注册一行（metric=逐 KPI 相对 A1 偏差、ref=A1_D10、tolerance=<15 %、status=**failed**，
实测 max 46.57 %；同时把 Cd 出带与 St 在带内记为诊断列），审计不代注册（非本单授权范围，避免审计席自造裁判）。

**F2（blocking，L4 维度）· 收敛梯级未跑，算例不完整**
`logs/A1_D14.log`、`A2_D14.log`、`A4_D14.log` **均为 0 字节**；`summary.json.convergence == {}`；
`predeclared_v2.json` 的 `convergence_rule`（D=10→D=14 <10 %）**无任何观测**。
被审件在报告 §5.1 如实写 not_run（诚实 ✅），但预注册的收敛判据因此空转 ⇒ 不满足"完整算例"。
建议：D=14 三臂递交集群（预注册命令已在 `run_matrix.sh 的 D=14 段`），出 `convergence.json` 后复判 L4。

**F3（non-blocking，落库维度）· 交付件未并入 main**
`agent/m17713 @ 24b4af6a` 只在分支上；main `be78a87b` 的 `xval/benchmark_method_selection/` 仅 6 个文件
（无 `runs/`、无 `oss_crosscheck.py`、无 `predeclared_v2.json`），`tests/test_xval_benchmark_method_selection.py`
与 `benchmarks/xval_method_selection_bench.py` 在 main 上**不存在** ⇒ 仓库档案里查不到本单证据。
本审计已用 `docs/board_audit/m17849/refs/` 快照固定证据（先例：m17738/m17650）。

**F4（non-blocking，文档缺陷）· A5 参数描述与工件矛盾**
报告 §5.2 写 "A5（**字面移植 Λ=3/16**）：τ₊=0.5075 ⇒ τ₋=**2.00**"，但 `runs/A5_D10/fail.json` 记录
`lambda_trt=0.01125, tau_minus=2.0`（= 1.5·(τ₊−0.5)，即 `run_arm.py:18-21` 所述的**重导出**值）。
Λ=3/16 在 τ₊=0.5075 下只能得 τ₋=25.5（那才是 A3 的 `lambda_trt=0.1875, tau_minus=25.5`）。
矛盾仅在文档句，工件是对的；但读者会据此认为"字面移植也能到 τ₋=2"，属实质性误解风险。建议改一句。

**F5（non-blocking，聚合缺陷）· verdict 映射漏掉 A4**
`summary.json.verdict.per_arm_within_15pct` 只有 `{"A2_D10": false}`，而 A4 的 Cl 幅值偏差 **−15.40 %**
（>15 % 带）也应记 false。结论方向不变（H1 已由 A2 否掉），但"逐臂判定"不完整；
且该 −15.40 % 意味着**只动 sponge 一个旋钮就已擦线出带**，这是比"46.57 % 由碰撞算子造成"更细的结论，值得入表。

**F6（non-blocking，工件卫生）· 工件不对称 + 空日志无标记**
`A4_D10` 缺 `state.npz`（A1/A2 有）；D=14 空日志与"仍在跑"不可区分（无 `not_run.json` 落盘）。

---

## 6. 处置

* **L3 确认消账**（三件套 + 入口 + 集成 + 测试齐备，且失败如实记录）。
* **L4 返修**（reopened 项，按优先级）：F1 注册基准行 → F2 集群补 D=14 收敛梯级 → F4/F5 文档与聚合修正。
  三件做完可复判 L4；仅补文档不升 L4。
* 本审计**不做**任何被审代码改动（交差核验单，非返修单）。

---

## 7. 证据索引

```
docs/board_audit/m17849/
  AUDIT.md                      本文件
  audit_numbers.json            全部被引数字的重算结果（compute_audit_numbers.py 生成）
  compute_audit_numbers.py      重算脚本（run_arm 工件 → 数字，零手抄）
  make_audit_figures.py         审计图（fig1 KPI 守恒带 / fig2 Cd vs 仓库趋势带 / fig3 复现）
  figures/fig1_kpi_parity.png   A2/A4 vs A1 逐 KPI 偏差 vs ±15 % 带
  figures/fig2_cd_external.png  Cd 逐臂 vs 仓库趋势带（非判据）
  refs/xval/benchmark_method_selection/   被审交付件字节快照 @24b4af6a（runs/figures/logs/表/预注册）
  refs/tests/test_xval_benchmark_method_selection.py
  refs/benchmarks/xval_method_selection_bench.py
  refs/docs/XVAL_LBM_BENCHMARK_METHOD_SELECTION.md
  verdict_m17849.sql            verdict 落表 SQL
```