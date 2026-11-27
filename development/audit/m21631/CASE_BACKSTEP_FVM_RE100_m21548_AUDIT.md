# 交差核验 · m21548（CASE-BACKSTEP-FVM-RE100 · 后台阶 Re=100 · TensorLBM）

审计单：**m21631**｜verdict_by=`board-audit`｜verdict 行：`tensor_maturity_verdicts#887`
（repo=TensorLBM, scope=`m21548 CASE-BACKSTEP-FVM-RE100`, level=**1**）

**一句话结论**：交付里有**真代码、也有单元验证**，但算例本身**没有任何运行工件**——
`X1/s` 从未产出，`tensor_benchmark_cases` 对该 anchor **0 行**，交付分支**未并入 main**，
交付测试**实测 1 failed**。功能实质 = **L1**（数学/单元验证）；L2/L3/L4 逐条不可达。
裁定方向：**返修**（不是消账）。

---

## ① 案例（核验对象）

| 项 | 内容 |
|---|---|
| 锚点 | `CASE-BACKSTEP-FVM-RE100`（feature_ledger：domain=TensorLBM，status=**open**，created 2026-11-14） |
| 物理 | 后台阶层流分离—再附；观测量 `X1/s`（下壁主再附长度，台阶高 s 归一） |
| 几何/参数 | `Re = ū·D_h/ν = 100`，`D_h = 2h_in`；生产档 `h_in=52, s=49` → ER=101/52=1.9423077（=Armaly 10.1/5.2） |
| 参考 | Armaly 1983 实验 **3.05**；Erturk 2008 二维 **2.878**；两门独立、各 10 %（refs.json） |
| 被核验单 | m21548（status=succeeded；`fail_stage=budget_exhausted+forced_finish`；`server_quality_audit.gate_valid=false`） |
| 交付分支 | `agent/m21548`，commit `8c1a35b3e`（**未并入 main**） |

## ② 怎么做的（核验方法，非重做）

1. **三件套取证**：磁盘（worktree / root/tp-agent/work/m21548）、DB（missions/benchmark/reports/ledger）、git（分支归属、trunk 差异）。
2. **基准裁判**（唯一裁判）：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BACKSTEP-FVM-RE100'` → **0 行**（全表 139 行；`%BACKSTEP%` 0 行，`%STEP%` 只命中 FORWARD-STEP 等 5 条）。故**无登记容差、无 got 数值可比**，误差不可计算。
3. **repro_cmd 可执行性**：`run_case.py --help` 正常；在审计机上跑了两条短链（`h1_re100` 6000 步 / `g1_re100 --smoke` 3000 步），rc=0，产出 `result.json/field.npz/wall_shear.csv` → 证明**代码链能算**（L2 的"能力"存在），但这些是**审计者的复现**，不属被核验单的交付工件。
4. **测试裁判**：复跑交付的 `tests/test_case_pool_backward_step_armaly_verdict.py`。
5. **机器裁定预览**：直接调用 `/srv/tensor/agent/maturity_gate.py::judge()` 复核本 verdict 的 evidence 键。

所有原始指针在 `development/audit/m21631/verdict_evidence_m21548.json`；复现工件在 `development/audit/m21631/audit_repro/`；图脚本 `development/audit/m21631/make_figs.py`。

## ③ 计算结果（核验数字表，全部程序化注入）

### 3.1 三件套核对

| 项 | 声称 | 实测 | 判定 |
|---|---|---|---|
| run_artifacts | doc ③ 列 4 个 `run_compute` 作业，cj3511 succeeded | `artifacts/case_backward_step_armaly/` **空目录（0 项）**；全 work 盘扫描只命中 `m15519/runs/case_backward_step_armaly/*`（**另一单**，ER=2.0） | **缺** |
| benchmark ref vs got | doc ⑥：X1/s "待产出，未判定" | 注册表 0 行；`got = null`；误差无法计算 | **缺** |
| repro_cmd 可执行 | doc「复现」段 3 条命令 | `run_case.py --help` OK；6000 步实测 rc=0（X1/s=2.9054）；`verdict.py` 无工件时诚实输出 `runs=0 all_judged_pass=False` | **部分可执行**（可通过性无法判定） |

### 3.2 交付测试复跑

| 命令 | 结果 |
|---|---|
| `pytest tests/test_case_pool_backward_step_armaly_verdict.py -q`（m21548 工作树） | **1 failed, 10 passed, 1 skipped in 0.31 s** |
| 失败断言 | `test_each_condition_fails_independently:162` → `assert 'caliper' in rows['caliper']['note_missing']` 得 `[]` |
| doc §2.3 声称 | "10 passed, 1 skipped"、**"五条件逐项可独立失败"** → 与实测**矛盾** |

### 3.3 分支/集成核对

| 检查 | 结果 |
|---|---|
| `git branch -a --contains 8c1a35b3e` | **仅 `agent/m21548`**（未并 main） |
| trunk `verdict.py` | `:46 TOLERANCE_PCT = 15.0` —— 本单自称修掉的"硬编码容差"缺陷**在 main 上仍然存在** |
| trunk `ladder.py` / 本单测试文件 | **不存在** |
| trunk `refs.json` | 2924 B，无 `judgement` 段（`grep -c judgement = 0`；分支版 4259 B 有门定义） |
| `tensor_reports WHERE mission_id=21548` | **0 行**（平台报告从未发布，违反 owner 报告纪律） |

### 3.4 审计者复现（非验收点，仅供链路证明）

| 配置 | 步数 | `X1/s` | `steady_at_step` | `max_speed` | 质量通量相对误差 |
|---|---|---|---|---|---|
| `h1_re100`（h_in=26, ER=1.9615） | 6000 | 2.9054 | **null** | 10.586 | 12.2 % |
| `g1_re100 --smoke`（h_in=8, ER=1.875） | 3000 | 2.6499 | **null** | 3.678（峰值） | 3.2 % |

## ④ 图

1. `audit/m21631/figs/fig1_ux_field_audit_repro.png` —— 审计复现的平均流场 `u_x` 云图（h1_re100，6000 步），
   台阶线（虚线）与再附线（绿）标注 `X1/s=2.905`。**标题显式标注"未收敛、非本单工件"**，防止被误读为算例验收图。
2. `audit/m21631/figs/fig2_x1s_vs_steps.png` —— `X1/s` 随步数轨迹 × 双参考 ±10 % 带：两条复现曲线**都未进稳态**，
   直观说明「6 k 步不能当验收点」，也说明本单 doc 把预算抬到 170–200 k 步的判断是对的。
3. `audit/m21631/figs/fig3_evidence_matrix.png` —— evidence 键 × L1–L5 判据矩阵（绿=有真指针，红=必需但缺）；
   机器裁定即由图 3 的红格决定：`run_artifacts` 红 ⇒ L2 以下。

## ⑤ 注意点（坑 / 局限 / 适用范围）

* **"作业 succeeded"≠"有工件"**。cj3511 `succeeded rc=0`，但 `run_compute fetch` 回的是递交时的 stage 快照（206 MB 源码 tar），
  run 目录不在其中。取证必须在磁盘上看到 `result.json/field.npz/wall_shear.csv`，不能以队列终态代替工件。
* **"分支有增量"≠"主线有增量"**。m21548 的 7 个改动文件全在 `agent/m21548`，trunk 上一份没有；
  若按"交付必须入库"的严口径，实质只剩 docs（L0/L1）。本审计按"功能实质"给 L1，并在 evidence 里显式登记该差异。
* **自声明门 ≠ 基准裁判**。`refs.json` 里的 3.05/2.878 + 10 % 是仓内自设门，代替不了 `tensor_benchmark_cases` 的登记行；
  本 anchor 至今 0 行，属于"未登记算例"，任何"通过"声称都无平台裁判背书。
* **历史数值不可继承**。m15519 同算例 `re100_m16.json` 的 2.693（vs 3.0 → −10.23 %）是 ER=2.0 几何、12000 步的老结果，
  几何与预算都与 52:49 生产档不等价，**不构成本单证据**（本审计已在 evidence 里点名排除）。
* **F6 观察未定论**：审计复现在 6000 步就出现 `max_speed=10.59`（`ū=0.1`）与 12 % 质量通量缺口。单次短跑不足以下结论，
  但提示生产点应先加 `max_speed`/进出口通量监控，否则"收敛点"可能是被出口海绵层伪平衡的态。
* 锚点 slug 含 `FVM`，本单只做 LBM 臂（doc §5 已声明）；TensorFVM 臂未被覆盖——若总账本意是 FVM 求解读，需另派单。

## ⑥ 误差判定（逐项 + 显式回答）

| 量 | ref | got | 误差 | <10 %（ledger 门） | <3 %（owner 标准） |
|---|---|---|---|---|---|
| `X1/s` @Re=100 vs Armaly 1983 | 3.05 | **无** | **不可计算** | **未判定**（无 got） | **未判定** |
| `X1/s` @Re=100 vs Erturk 2008 | 2.878 | **无** | **不可计算** | **未判定**（无 got） | **未判定** |
| 观测阶 / Richardson | — | 无阶梯工件 | — | n/a | n/a |
| 审计复现（h1, 6000 步，非验收） | 3.05 / 2.878 | 2.9054 | −4.7 % / +0.95 % | 名义在带内，但**未稳态 ⇒ 不得据此判通过** | 同上 |

**是否 <3 %：无法回答。** 本算例没有可判定的收敛点数值，任何"通过/接近"的说法都属虚标；
审计者自己的 6 k 步短跑**故意不升级为验收结论**（`steady_at_step=null`、`max_speed` 超 `ū` 100 倍）。

## 判级与裁定

| 级 | 判据（maturity_gate SCHEME） | 本单实况 | 是否达 |
|---|---|---|---|
| L1 数学/单元验证 | `tests` 非空 | 11 个用例存在、10 过 1 败 1 skip | **达**（含 1 败，见 F2） |
| L2 独立功能 | impl+entry+`run_artifacts` 非空 | run_artifacts **真空** | 不达 |
| L3 系统集成 | +`integration_callers` | 分支未并 main、无调用方、注册表 0 行 | 不达 |
| L4 基准验证 | +`benchmark` 数值对照 + `repro_cmd` | got=null，注册表 0 行 | 不达（WITHHELD） |
| L5 科研工程 | +`external_anchor` 机读独立锚 | 3.05 为二手 digitisation（PRIMARY_PDF_CLOSED） | 不达 |

**机器裁定预览**（直接调用 gate 的 judge）：

```
declared 1 -> gate_level 1  reasons []
declared 2 -> gate_level 1  reasons ['L2 缺: entry(工件缺失),run_artifacts']
link_original -> 21548
```

⇒ verdict#887 落表后，周期 gate 会写 `gate_state='reopened'` 并为 m21548 开返修轮（≤1 且原单 succeeded）。

### 发现清单（severity 排序）

| 编号 | 严重度 | 证据 | 复现 |
|---|---|---|---|
| F1 `X1/s` 无工件，算例验收不可判定 | **blocking** | `artifacts/case_backward_step_armaly/` 0 项；cj3511 fetch 无 run 目录 | `ls -A` / 全盘 `find` |
| F2 交付测试 1 failed；doc 漏报并声称五条件独立 | **blocking** | `test_...:162 assert 'caliper' in []` | `pytest ... -q` |
| F3 基准注册表 0 行，无容差裁判 | **blocking** | `SELECT ... WHERE anchor_slug='CASE-BACKSTEP-FVM-RE100'` → 0 rows | psql |
| F4 分支未合并，trunk 仍硬编码 15 % | **blocking** | `verdict.py:46`；`git branch --contains` | git/grep |
| F5 平台报告未发布 | medium | `tensor_reports mission_id=21548` → 0 行 | psql |
| F6 复现中 max_speed/质量通量异常，未定论 | low（观察） | `audit_repro/result.json` | 6000 步复跑 |

### 处置（disposition）

**request changes / 返修**。返修最小闭环：
1. `g1_re100`（h_in=52,s=49，ER 精确）170 k 步生产点**跑完并取回** `result.json/field.npz/wall_shear.csv`；
2. 用分支版 `verdict.py` 双门（3.05 / 2.878，各 10 %）判一次，报 got 与两条误差；
3. 修 `caliper` 独立失败用例（或在 doc 里删掉该声称）；
4. 把 `ladder.py` + `judgement` refs + 测试并入 main，并在 `tensor_benchmark_cases` 登记一行（ref_source/metric/tolerance）后重判。

### 未跑的检查（不当作通过）

170k/200k 步生产点（本单非重做，且本机禁长跑）· TensorFVM 臂（不在白名单）· Armaly 1983 一手 Table/Fig.4（PDF 无 OA）。