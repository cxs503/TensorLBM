# m15893 — 交差核验：m15469 `CASE-MCMP-CAPILLARY-RISE`（TensorLBM）

**裁定：L2**（`level=2`，`verdict_by='board-audit'`）· 交付实质=算例池驱动 + 真实 SC-LBM 端到端求解 + 诚实的负结果；
精度面 **全基准 FAIL（-86 % / -91 % / +116 %）**，不达 L4；无 app/API 调用方，不达 L3。

- 被审任务：m15469（TensorLBM，状态 `succeeded`，PR #1824 merged，merge commit `8d1d30aa`）
- 审计任务：m15893　审计分支：`agent/m15893`
- 单一裁判：`tensor_benchmark_cases`（task 指定）→ **该 anchor 0 行**；因此容差只能取自仓内预声明（见 F6）
- 全部数字由 `artifacts/audit_m15893/verify_m15469.py` 程序化产出，落在 `artifacts/audit_m15893/audit_m15893.json`（零手抄）

---

## ① 取证：三件套实测

| 三件套 | 判定 | 证据（可复核） |
|---|---|---|
| **run_artifacts 文件** | **部分存在** | main 内 `benchmarks/case_pool/capillary_rise/artifacts/` 只有 `jurin_closed_box.json`(20007 B)、`jurin_closed_box_report.md`(3981 B)、`route_a_vent_diagnostics.md`(1853 B)；**文档承诺的 `jurin_closed_box.png` 未进 main**（m15469 工作树有 62569 B，合并时被滤除）→ F8 |
| **benchmark ref vs got 数值** | **存在且可用，但全 FAIL** | JSON `analysis` 段：`h_ref_jurin=1.3133`（θ=88.15° 曲面弧拟合）、`h_ref_jurin_young=33.2477`（实为 θ=35.03° wall_psi 反算）、`h_ref_closed_box_eq1=26.8372`；`meas.h=2.83950`。审计独立复算三档全重现（`abs_diff=0.0`）→ F1 |
| **repro_cmd 可执行** | **是，且逐位可观** | `PYTHONPATH=src python3 benchmarks/case_pool/capillary_rise/sc_closed_box.py`（审计在 `.audit/repro/` 隔离目录复跑）：exit 0，wall 207.6 s（原 198.3 s），`h=2.83950` vs 提交值 `2.83950` → **abs_diff=0.0**，`rho_gas=0.12812412` vs `0.12812412` → **abs_diff=0.0**，`theta_arc=88.1464°` 逐位一致，判词同为 FAIL |

**基准裁判（`tensor_benchmark_cases`，task 指定唯一裁判）**

```sql
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
WHERE anchor_slug='CASE-MCMP-CAPILLARY-RISE';   -- → 0 rows
SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug ILIKE '%CAPILL%';  -- → 0
```

表内共 25 个 anchor，**无任何 CAPILL 类 anchor**。故"平台级容差行"不存在，容差唯一来源=仓内预声明
`CFG['tol_pct']=10.0`（`sc_closed_box.py:74`）与 README §2。审计按该 10 % 判据裁（与 m15489 对
`CASE-CAPILLARY-WAVE` 的处理口径一致）。

**报告六段（①问题/②方法/③参数/④ref vs got/⑤收敛敏感/⑥失败模式）**：`artifacts/jurin_closed_box_report.md`
六段齐全、数字与 JSON 同源；`docs/CASE_MCMP_CAPILLARY_RISE.md` 与 `README.md` §1–§6 亦齐全 →
但存在跨文档口径冲突（F2/F4/F5）。

---

## ② 判级：L0–L5

| 级别 | 判据 | 本单 |
|---|---|---|
| L0 文档 | 仅文档 | 超出（有可执行求解器与工件） |
| L1 数学/单元 | 公式/单元验证 | 超出（有 12000 步端到端物理算例） |
| **L2 独立功能实现** | 真代码：输入→计算→输出 | **命中**：`sc_closed_box.py`（SC D2Q9 + Guo forcing + bounce-back + Young wall + 浮力重力 + 子网格弯月面拟合）真跑，确定性输出 JSON/PNG/表；`run_case.py`（route A）作为被证伪的替代保留 |
| L3 系统集成 | 接入主求解器/平台 | **不达**：`grep -rn 'CAPILLARY-RISE|capillary_rise' app/ .github/` 无路由/服务注册；唯一调用面=`.github/workflows/ci.yml:95 pytest -q` 的通用收集（tests/ 目录）+ 复用 `tensorlbm.sc_surface_tension` / `wetting.metrics` 内核。是"用库"，不是"入库" |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现 | **不达**：误差分析/可复现已成立，但**精度全项 FAIL**（F1），且注册表无该 anchor 行（F6） |
| L5 科研工程成果 | — | 不达 |

> **级别=功能实质，报告厚度不改变级别**：交付的确实是可复现的求解器驱动与诚实的负结果（有价值，属 S-N 否定性成果轨），
> 但"毛细上升 Jurin 基准通过"这一能力**不成立**。

---

## ③ 发现（按严重度；均已复现）

### 阻塞级（决定级别）

**F1 — 精度全基准 FAIL，无任何口径可救。**（`audit.json: independent_reference_ladder`）
`h_got = 2.83950`（lattice units）。审计独立重推 `h = 2σcosθ/(Δρ g w)`：

| 参考 θ | h_ref | 误差 | <10 % ？ |
|---|---|---|---|
| 声明输入 θ=60.0°（**任务书 ref 口径**） | **20.3005** | **-86.0 %** | 否 |
| wall_psi 反算 θ=35.03°（文档称 "Young"） | 33.2477 | -91.5 % | 否 |
| 本算例弯月面弧拟合 θ=88.15°（文档头条口径） | 1.3133 | **+116.2 %** | 否 |
| 闭盒式 eq.(1)（含气相加压） | 26.8372 | -89.4 % | 否 |

三档独立参考全数为 FAIL ⇒ m15469 自报的 `FAIL` 属实，无粉饰。级别因此封顶 L2。

**F2 — 参考量标签错位：名为 "Young" 的参考不是输入角。**
`sc_closed_box.py:317-318` 用 `theta_from_wall_psi_deg`（35.03°）而非 `theta_input_deg`（60°）构造
`h_ref_jurin_young`。审计复算：`h_ref_jurin_young = 33.2477 = Jurin(θ=35.03°)`（`match_wall_psi_theta=true`），
而 **`Jurin(θ=60°) = 20.3005`（`match_declared_theta=false`）**。⇒ **全部交付件中都不存在"以声明输入角为锚"的
Jurin 参考**；gate 从未锚在输入参数上（数值结论不变，仍 FAIL，但口径不可接受）。

**F3 — 报告表存在硬编码假 PASS 行。**
`sc_closed_box.py:386-388` 直接写死 `"0.0 deg"` 与 `"PASS"`，而该行同时打印 `60.0 deg` 与 `35.0 deg`：

```
| theta (from wall_psi, Young relation) | input theta | 60.0 deg | 35.0 deg | 0.0 deg | PASS |
```

真实 |Δ| = **25.0°**（README §4 的同一比较是 `+53.1 deg / FAIL`）。同一物理量在两张表里一个 PASS 一个 FAIL ⇒
**证据面有自相矛盾行**，属可修复的实现缺陷（非结论翻转）。

### 非阻塞（不改变级别，但必须修）

**F4 — 一次运行两个头条误差。** `artifacts/jurin_closed_box_report.md` 头条 `+116.2 %`（arc 参考），
`README.md` §4 头条 `-91 %`（wall_psi 参考），`docs/CASE_MCMP_CAPILLARY_RISE.md` 摘要 "far below"。README §2 声明了三个参考，
但**未预声明哪个是主判据**，导致同一次运行可挑出不同头条误差（"选参考"空间）。修复方向：把"以声明 θ 的 Jurin"设为主判据。

**F5 — 文档数字陈旧。** `docs/CASE_MCMP_CAPILLARY_RISE.md` 写 "Young's relation claims 31.5 deg"，JSON 实为
**35.0264°**（`theta_from_wall_psi_deg`），README 又写 35.0°（`doc_theta_stale=true`）。

**F6 — 基准未在平台注册。** `tensor_benchmark_cases` 该 anchor 0 行（全表无 CAPILL anchor）⇒ 平台门无法机器裁，
容差只活在仓内代码里。与 m15489 记录的 F4 同类，属**算例池注册缺口**（池化令的系统债，不独属本单）。

**F7 — 无 tensor 报告。** `SELECT count(*) FROM tensor_reports WHERE mission_id=15469` → **0**（同批 `m15467 case-pool/mcmp-laplace` 有报告）。
按 owner 常设报告纪律"报告未发=本单未完成"，该交付在报告面是缺的。

**F8 — 交付件在 main 上缺 PNG。** 文档交付表与 README §5 均承诺 `artifacts/jurin_closed_box.png`；
merge 后 main 内**不存在**（`ls benchmarks/case_pool/capillary_rise/artifacts/` 仅 3 个文本/JSON 文件）。
`run_artifacts` 因此只能部分计入。

### 通过项（正面证据，一并入账）

- **可复现性 A 级**：审计隔离目录复跑，`h` / `rho_gas` / `theta_arc` 三项 `abs_diff` 全为 0.0（确定性逐位一致），
  判词 FAIL 一致（`reproduction.h_reproduced_to_1e-3=true`）。
- **测试真实**：`python -m pytest tests/test_case_pool_capillary_rise.py -q` → **4 passed in 5.98 s**（审计本单实跑）。
  覆盖面=几何/掩膜闭合性、初始 ρ 分布、子网格弯月面插值、Jurin 的 1/g 与 1/w 标度（**测的是不变量，不是"跑一遍"**）。
- **收敛旗标诚实**：`converged=False` 与预声明停止规则完全一致（`final_window_spread 1.39e-4 < 0.02` 但
  `u_max_liquid 0.0989 > u_tol 0.02`）→ `convergence_flag_correct=true`；未把"h 收敛"冒充"流动收敛"。
- **质量账诚实**：`mass_drift 9.04e-5`（全相）/ `liquid_mass_drift 8.79e-2`（液相），与文档"conserved to 1e-2"一致。
- **负结果有诊断**：`artifacts/route_a_vent_diagnostics.md` 记录 route A 双向质量漂移与 CAC 路线 NaN，属可复用知识。
- 审计独立复算 `Bond = Δρ g w²/σ = 0.7879`，与 README 0.79 一致。

---

## ④ 被夸大的申报

| 申报 | 实况 |
|---|---|
| 文档交付表：`artifacts/jurin_closed_box.png` 为交付件 | main 内不存在（F8） |
| 文档摘要：Young 关系"claims 31.5 deg" | 实际反算 35.03°（F5） |
| 报告表：`theta (from wall_psi)` 行 "0.0 deg / PASS" | 真实 |Δ|=25.0°（F3） |
| README 与 artifacts 报告的"主判据" | 口径不一致，未预声明（F4） |
| 任务状态 `succeeded` + `dev_acceptance` 自动批准（`server_quality_audit.gate_valid=false, gate_missing_warning=true`） | 交付的是**诚实 FAIL**：mission 系统把"跑完+自报 FAIL"记成 succeeded，平台面缺少"基准未过"的机器判据（F1/F6） |

**未夸大**：m15469 自身 headline 明确 "Verdict = honest FAIL"，未声称通过；本审计不推翻其诚实性，只封级别与列缺陷。

---

## ⑤ 审计范围与独立性限制

- 只读：m15469 工作树 + main 内合并结果 + `tensor_benchmark_cases` / `tensor_reports` / `tensor_maturity_verdicts`。
  未修改被审仓任何被审文件；审计复跑写在 `.audit/repro/`（隔离目录，`OUT` 相对脚本路径）。
- 独立性限制：审计者与被审者共享同一 `tensorlbm` 库与同一 `sigma=0.05611` 标定来源
  （`contact_line.engine.SIGMA_DEFAULT`）；本审计**未独立重测表面张力**，若该标定偏差，`h_ref` 会整体平移
  （但 -86 %～-91 % 的量级偏差不可能由 σ 标定误差解释，需 σ 偏差 ≈7 倍）。
- **未做**：未做 θ（f_w）标定梯级、未做 Bond 数/缝宽扫描（属返修范围，非核验范围）；
  未在 loong64 集群节点复跑（本地 x86_64 复跑已逐位一致，故未追加跨架构判读）。

## ⑥ 处置（disposition）

**request changes（reopened）**——理由与最小返修包：

1. **口径归位**：gate 主判据改为"以声明输入角 θ=60° 的 Jurin"（h_ref=20.3005，现误差 -86.0 %），
   其余两档降为诊断列；README/报告/文档统一同一条头条误差（修 F2/F4）。
2. **删假 PASS 行**：`sc_closed_box.py:386-388` 改为按真实 |θ_psi − θ_input| 计算并在超 10° 时判 FAIL（修 F3，附负对照测试）。
3. **补工件与注册**：把 `jurin_closed_box.png` 纳入交付（或从交付表删除），并在 `tensor_benchmark_cases`
   登记 `anchor_slug='CASE-MCMP-CAPILLARY-RISE'`（`h`，10 % 容差）（修 F6/F8）。
4. **能力面真问题**：SC 壁面 `f_w` 与 Young 关系不匹配（88.1° vs 35°/60°）是本次 FAIL 的根因；
   返修应先用 sessile-drop 梯级标定 `f_w→θ`，再重跑本算例，或改用带角度边界条件的自由能/相场路线。
5. 补发 tensor 报告（六要素 + ref/got 表）（修 F7）。

**账目结论**：该 anchor 的"Jurin 毛细上升基准通过"能力**不成立**；已交付的是可复现驱动的**否定性成果**（S-N 轨），
按 L2 记账，不得计入 L4 能力台账。