# 交差核验审计 — m15776 / CASE-CYLINDER-DRAG-RE100（TensorLBM）

审计单：m15884（board-audit）· 被审任务：m15776（succeeded, PR #1850, merge 53046bcc 后）
锚点：`CASE-CYLINDER-DRAG-RE100`（Schäfer & Turek 1996 NNFM-48 **2D-2**：通道内圆柱 Re=100，D/H=0.244）
本单性质：**取证核验，非重做**。判据=平台基准表 + 可复现性 + 六要素报告，报告好看不算数。

## 0. 结论（disposition: request changes / 返修）

| 项 | 结论 |
|---|---|
| 实现（真代码 输入→计算→输出） | **存在且可执行**（L2 成立） |
| 工件（run_artifacts） | 存在，但**只有 1 个网格档（D=20）**，且 **dp 指标工件相对 HEAD 代码已过期** |
| ref vs got（唯一裁判） | **machine_pass=false**：Cd +8.22%（FAIL）、Cl_amp +24.84%（FAIL）、St +0.77%（PASS）、dp −18.79%（FAIL，且不可复现） |
| repro_cmd | **不可照跑**（两条硬缺陷 + 与注册工件档位不匹配） |
| 六要素报告 | `tensor_reports` 中 **0 行**（任务自己也在 summary 里写"Tensor API 报告"未跑） |
| 平台基准裁判行 | `tensor_benchmark_cases` 中 anchor **0 行** → 无机器裁判 |
| 判级 | **L2**（L3/L4 已证伪，见 §5） |

m15776 自身的诚实性：它的 result JSON 明确写 `honest-FAIL`、`machine_pass=false`，**没有粉饰**（这点应予记账）；但它"补齐 L2/L3 证据面"的自述里，L3 与 dp 的可靠性不成立。

## 1. 取证：三件套是否真存在

* 实现：`src/tensorlbm/validation/cylinder_dfg.py`（565 行；`REFERENCE_BANDS` L100–120、`probe_columns` L278、`run_case` L295、dp 外插 L401–414、CLI L512–565）。数值全部由库原语拼装（`solver.collide_mrt/stream`、`boundaries.zou_he_*`、`bounce_back_cells`、`compute_obstacle_forces`），不是另写一套格气。
* 入口：`benchmarks/verified/cylinder/re100_dfg/run.py`（网格序列 + 聚合）、`python3 -m tensorlbm.validation.cylinder_dfg --D.. --steps.. --out <FILE>`
* 工件（main，已合并）：
  - `artifacts/level_D20_u0833.json` sha256:592446e9eafb899a（936668 B，wall 178.9 s，D=20/12000 步）
  - `artifacts/level_D20_u0833.series.npz` sha256:aeb2273c45defb09
  - `artifacts/result.json` sha256:57c9490242b5129c（`machine_pass=false, verdict=not_verified, convergence={}`）
  - **缺**：`figures/` 从未入库（fig1/fig2 只存在于 m15776 工作树）；`runs/m15776/{D28,D40}.log|json` 亦未入库
* 测试：`tests/test_cylinder_dfg_reference.py` → 审计侧复跑 **6 passed / 7.10 s**（band-vs-register 冻结、D=20/28/40 探针不落固体、合成信号 St 回收、smoke 有限性）。测试断言的是**不变量**（探针位、band 冻结、St 估计器），不是"跑通即过"——这一点做得好。
* 复现审计（审计侧独立党案，**不是** m15776 工件）：`artifacts/audit_m15884/repro/repro_D20_6000.json`（D=20/6000 步/4 线程，wall 94.1 s）→ Cd 3.53285、St 0.29492、Cl_amp 1.22815、**dp 5.44944**。聚合链零手抄已核：用 `--aggregate-only` 从 level 工件重算，Cd/St/Cl_amp/dp 与 result.json **逐位一致**（差异 <1e-12）。

## 2. ref vs got（对照 `tensor_benchmark_cases` 口径逐项）

**注意：平台 `tensor_benchmark_cases` 中 `anchor_slug='CASE-CYLINDER-DRAG-RE100'` 为 0 行**（全表 25 行，无任何 cylinder 锚点）。故本次用 m15776 冻结的 register（`REFERENCE.md` + `references.json` 的 `cylinder_dfg2d2_*_re100`）作裁判口径，并把"平台无锚点行"作为台账缺口记入 verdict。容差=注册 case tol 5%；平台底线 15%。

| 指标 | 冻结带 | 工件 got（D20/12000） | 误差 vs 带中值 | 5% 判据 | 审计侧重跑（HEAD，D20/6000） | 审计侧误差 |
|---|---|---|---|---|---|---|
| Cd | 3.22–3.24 | 3.4955 | **+8.22%** | FAIL | 3.5328 | **+9.38%** |
| Cl_amp | 0.99–1.01 | 1.2484 | **+24.84%** | FAIL | 1.2282 | **+22.82%** |
| St | 0.295–0.305 | 0.3023 | **+0.77%** | PASS | 0.2949 | −1.69%（PASS） |
| dp* | 2.46–2.50 | 2.0140 | **−18.79%** | FAIL | **5.4494** | **+119.74%** |

三点关键核验结果：

1. **dp 工件过期（结构性问题）**：入库的 `level_D20_u0833.json` **没有 `probes{}` 块**，即由"探针直接取固体格点"的旧版代码产出（模块 docstring 自己承认该做法系统性偏低 ~19%，也就是 result.json 里那个 −18.79%）；HEAD 的"最近流体列 + 线性外插"实现从未产出过 level 工件。审计侧 HEAD 重跑 dp=5.4494，是工件的 **2.71 倍** → 入库数字在 HEAD 上不可复现。
2. **dp 量纲约定未闭合**：`REFERENCE.md` 同时写 `dp_raw 2.46–2.50` 与 `dp_dimensionless 4.92–5.00`（×2），而代码 `q=0.5u²` 归一后仍与 2.46 比。按 2.46 判，HEAD 是 +120%；按 4.92–5.00 判，HEAD 是 +10%，旧工件是 −59%。**两种口径下"−18.79%"都不是 HEAD 的真值**，该指标当前不可判。
3. **St 估计器不稳**：同一 D=20/12000 档，m15776 自己的 `runs/m15776/D20_u0833.json` 记录 St=0.5187、`D20_v2.json` 记录 0.3809，而入库工件是 0.3023（`st_ambiguous=false`）。单档单窗口 + FFT 峰值锁定，PASS 结论有偶然性（审计 `六要素` 需标注）。

## 3. repro_cmd 可执行性（逐条实测）

matrix.json 注册：`python benchmarks/verified/cylinder/re100_dfg/run.py --levels 28 40 --steps 60000 --threads 4`

| # | 缺陷 | 证据 |
|---|---|---|
| D1 | `run.py` L37 `REPO_ROOT = HERE.parents[2]` = `<repo>/benchmarks`，插入 `<repo>/benchmarks/src`（不存在）→ 实际 import 到环境里任意一个已装 `tensorlbm` | 实测：`ImportError: cannot import name 'cylinder_dfg' from 'tensorlbm.validation' (/root/tp-agent/work/m13963/src/tensorlbm/validation/__init__.py)`。同目录兄弟 `cylinder/re100/run.py` 用的是 `parents[3]` |
| D2 | `aggregate()` 对 `--out-dir` 做 `path.relative_to(REPO_ROOT)` | `ValueError: ... is not in the subpath of '<repo>/benchmarks'`（仓库外/相对路径一律崩） |
| D3 | 注册命令跑的档位（D=28/40）**没有工件**，而唯一工件是 D=20/12000 → 注册命令复现不出注册工件 | `artifacts/` 只有 `level_D20_u0833.*`；D28 的 `runs/m15776/D28_u0833.json`（40000 步，cd=3.3879, dp=5.5333）未入库 |

可行跑法（审计已实测）：`PYTHONPATH=<repo>/src python3 benchmarks/verified/cylinder/re100_dfg/run.py --aggregate-only --out-dir <绝对路径且位于 benchmarks/ 之下>`。

## 4. 六要素报告

`tensor_reports` 中 `mission_id=15776` **0 行**；仓库内也没有该锚点的六段报告（只有 `REFERENCE.md` 参考带登记与 m15432 的旧审计）。→ 报告要素**全部缺失**（案例/方法/数字表/图/注意点/误差判定），且**"是否<3%"这一问从未被显式回答**（实际三者远超 3%）。

## 5. 判级（L0–L5）

* **L2 = 独立功能实现** ✅：真代码、真输入→计算→输出；审计侧两个独立证据（pytest 6 passed；全新 6000 步运行产出完整可观测量与工件）。
* L3=系统集成 ✗：`tensorlbm/validation/__init__.py` 未导出 `cylinder_dfg`；无主求解器/平台侧调用路径；调用者只有 `tests/`、benchmark runner，以及 `matrix.json` 里一行清单 → 属清单级接线，不是运行时集成。
* L4=基准验证 ✗：`machine_pass=false`；4 指标中 3 项超出 5% 容差；只跑 1 个网格档（`convergence={}`，D28/D40 无工件），无误差收敛分析可谈；`tensorbench_results` 0 行，平台 l4-verdict=`CLAIMED_NOT_PROVEN`；repro_cmd 不可照跑且不复现注册工件。
* 附：m15776 的 `accepted=true` 来自 `TP_AUTO_APPROVE_DEV_ACCEPTANCE` 自动驾驶，且 `server_quality_audit={gate_valid:false, key_present:false, gate_missing_warning:true}` —— 验收不是证据门给的。

**判定：L2**（附"返修清单"）。

## 6. 返修清单（最小可验证）

1. 修 `run.py` 的 `REPO_ROOT`（`parents[3]`）+ `relative_to` 容错，使注册 repro_cmd 可照跑（+ CI 冒烟）。
2. 统一 dp 口径（选定 raw 或 doubled 其一），把 `REFERENCE_BANDS` 与 `REFERENCE.md` 对齐，并**用 HEAD 代码重出 D20 工件**覆盖 result.json（现工件是旧代码）。
3. 补 D=28/40（或 20/28/40）网格序列与 GCI/收敛表，否则 `convergence={}` 不能支撑 L4。
4. 平台侧落锚点行：`tensor_benchmark_cases(anchor_slug='CASE-CYLINDER-DRAG-RE100', metric∈{cd,cl_amp,st,dp}, tolerance='<5%')`，并把结果写 `tensorbench_results`（否则永远 CLAIMED_NOT_PROVEN）。
5. 补六要素报告（含 Cl_amp 与 dp 的 FAIL 与 St 估计器不稳的适用性说明）。