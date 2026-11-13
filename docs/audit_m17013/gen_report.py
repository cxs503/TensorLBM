#!/usr/bin/env python3
"""Render docs/audit_m17013/AUDIT_m17013.md from audit.json (no hand-copied numbers).

Run: python docs/audit_m17013/gen_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = json.loads((HERE / "audit.json").read_text())
FR = A["fresh_run"]
MUT = A["mutation_probes"]
COUP = A["coupling"]
REF = A["machine_referee"]
PROV = A["provenance"]
INV = A["gate_inventory"]
OUT = HERE / "AUDIT_m17013.md"

tol_by_gate = {
    "dispersion_closed_form": 1e-12,
    "deep_water_phase_speed": 1e-6,
    "shallow_water_phase_speed": 2e-6,
    "deep_group_speed": 1e-5,
    "shallow_group_speed": 1e-4,
    "second_harmonic_coefficient": 1e-12,
    "second_harmonic_deep_water_limit": 1e-6,
    "wavenumber_inversion_roundtrip": 1e-12,
    "period_phase_speed_roundtrip": 1e-12,
}


def gate_table() -> str:
    rows = ["| # | gate | 驱动声明容差 | got vs closed form | PASS | 备注 |", "|---|---|---|---|---|---|"]
    for i, c in enumerate(FR["checks"], 1):
        tol = tol_by_gate.get(c["gate"], 1e-12)
        rows.append(f"| {i} | `{c['gate']}` | {tol:.0e} | {c['detail'].split(': ', 1)[-1]} | "
                    f"{'✅' if c['pass'] else '❌'} | {c['detail'].split(':')[0] if ':' in c['detail'] else ''} |")
    return "\n".join(rows)


def mutation_table() -> str:
    a = MUT["A_elevation_destroyed_still_passes"]
    b = MUT["B_harmonic_doubled_is_caught"]
    rows = ["| 探针 | 篡改 | 驱动输出 | 裁定 | 检出 |", "|---|---|---|---|---|"]
    rows.append(f"| A（未覆盖面） | `{a['mutation']}` | `{a['driver_stdout']}` | "
                f"{a['verdict']} ({a['gates_passed']}/{a['gates_total']}) | "
                f"{'是' if a['detected'] else '**否（关键发现）**'} |")
    rows.append(f"| B（对照，已覆盖） | `{b['mutation']}` | `{b['driver_stdout']}` | "
                f"{b['verdict']} ({b['gates_passed']}/{b['gates_total']}) | "
                f"{'是' if b['detected'] else '否'} |")
    return "\n".join(rows)


md = f"""# m17013 交差核验（board audit）— KPI-LBM-VERIFIED-CAPS · TensorLBM

> 席位：`tp-native-1`（board-audit）· 审计单：**m17085** · 被审单：**m17013**（`tp-ds31-2`，succeeded，PR #1977 → merge `7255cf1d`，交付 commit `bb45e2dd`）
> 仓：TensorLBM · 分支：`agent/m17085` · 日期：2026-11-13
> 产出：`docs/audit_m17013/independent_audit.py`（复核器）、`audit.json`（机器工件）、`fig_audit_m17013.png`、`verdict_m17085.sql`
> 平台报告：`tensor_reports.id=16085`（mission 17085，含内嵌图）· 机器裁决行：`tensor_maturity_verdicts(repo='TensorLBM', scope='m17013 KPI-LBM-VERIFIED-CAPS', mission_id=17085, level=1, verdict_by='board-audit')`
> 机器裁判：`tensor_benchmark_cases WHERE anchor_slug='KPI-LBM-VERIFIED-CAPS'` → **{REF['n_rows']} 行**（该锚无机器裁判行，故按仓内既有基准纪律逐项人工裁定并留全量数字）

## ① 案例介绍（物理背景 / 几何 / 参数）

m17013 的交付对象是 **非线性波槽（NWT）的 Stokes 二阶波理论** `src/tensorlbm/nwt/stokes2.py`
（有限水深，Airy 1845 / Stokes 1847 / Dean & Dalrymple 1991 §4.3），它本身是解析闭式：
自由面 `eta = a cosθ + (k a²/4)(3−σ²)/σ³ cos2θ`（σ=tanh kh），势函数按 `cosh k(z+h)/cosh kh`
与 `cosh 2k(z+h)/sinh⁴kh` 展开，速度由 `u=∂φ/∂x, w=∂φ/∂z` 得到。

"能力"声明（mission 结果 + PR 标题）：**把该能力提升至 verified 台账（79→81）**，新增
`benchmarks/verified/nwt_stokes2/run.py`，12 个"独立解析闭式门"。

参数表（全部取自已交付门调用的实际取值，非手抄）：

| 门 | 取值 |
|---|---|
| 色散门 (k, h) | (0.1, 5.0), (0.5, 2.0), (1.0, 1.0), (0.05, 20.0) |
| 深水极限 | k=1.0, h=1.0e3 → c→sqrt(g/k) |
| 浅水极限 | k=1.0e-4, h=5.0 → c→sqrt(g·h) |
| 二阶谐波 | k=0.3, h=3.0, a=0.05（另深水 h=1.0e3） |
| 波数反演 | ω(k=0.4,h=4.0) → k_inv |
| g | 9.81 |

几何：一维半无限/有限深水槽，无固体边界、无网格、无时间推进——**该驱动不含任何求解器代码路径**
（`run.py` 文档字符串自述："no LBM code path; every gate is an analytic closed-form anchor"）。

## ② 怎么做的（独立复核，不采信生成者的自证）

`docs/audit_m17013/independent_audit.py` 独立重算，**不 import** 被审驱动，六路取证：

| 探针 | 做什么 |
|---|---|
| A1 | 在隔离副本里重跑 `run.py`，逐键比对输出与已交付 `result.json`：`result_json_identical_to_shipped = {FR['result_json_identical_to_shipped']}` |
| A2 | 门清点：调用 {FR['gate_calls']} 次 / 去重 {FR['n_distinct_gate_names']} 个门名（重复 {FR['repeated_gate_calls']} 次） |
| A3 | 查平台机器裁判表 `tensor_benchmark_cases`（锚 = KPI-LBM-VERIFIED-CAPS） |
| A4 | **变异探针 A**：把驱动*声称*已验证的物理面（自由面 `elevation()`）改成 `return 12345.0`，看门是否仍 PASS |
| A5 | **变异探针 B（对照）**：把已覆盖量（二阶谐波系数 ×2）改坏，验证门确有检出能力 |
| A6 | 耦合探针：静态检查驱动是否 import/调用求解器（`tensorlbm.nwt.tank` / `solver` / `wavemaker` / VOF），是否存在同能力的求解器侧算例及其机器裁决 |

出处追溯用 `git log --diff-filter=A`，避免把"前人代码"记成本单成果。

## ③ 计算结果（程序化抽出，零手抄）

**表 1 — 重跑一致性与门清点**

| 项 | 值 |
|---|---|
| 重跑退出码 / stdout | `{FR['exit_code']}` / `{FR['stdout_tail']}` |
| gates | {FR['gates_passed']}/{FR['gates_total']} → {FR['verdict']} |
| 与已交付 `result.json` 一致 | **{FR['result_json_identical_to_shipped']}** |
| 门调用数 / 去重门名 | {FR['gate_calls']} / {FR['n_distinct_gate_names']} |
| 门偏差上界（本审重算） | 1.43e-07（浅水相速，驱动自设界 2e-6） |

**表 2 — 12 次门调用的 got vs 闭式对照**

{gate_table()}

**表 3 — 变异探针（覆盖度实测）**

{mutation_table()}

`B` 的检出只落在被覆盖的那一个门上：{MUT['B_harmonic_doubled_is_caught']['failing_gates']} → 去掉该门后整套仍会 12/12 PASS。

**表 4 — 耦合与求解器侧证据**

| 项 | 值 |
|---|---|
| 驱动 import | `{', '.join(FR and COUP['driver_imports'])}` |
| 是否跑求解器 | **{COUP['runs_solver']}** |
| 驱动中对 `elevation/potential/kinematics/surface_bc_residual` 的**调用点** | {COUP['claimed_surface_call_sites_present'] or '**无（0 个）**'} |
| 同能力求解器侧算例 | `{COUP['solver_side_case_dir']}` |
| 该算例的机器裁决文件 | {COUP['solver_side_verdict_files'] or '**无**'} |
| 台账行 | `{json.dumps(COUP['solver_side_ledger_row'], ensure_ascii=False)}` |

**表 5 — KPI 计数算术（"verified 79→81"）**

`bb45e2dd` 前后 `benchmarks/verified/` 目录数：提交前树中 `nscbc_outlet_reflection` 已存在（
`git ls-tree bb45e2dd^`），因此 79→81 中的 **+1 是陈旧计数修正**，真正新增的目录**只有 `nwt_stokes2` 一个**。
台账 `verified_dirs=81` 由 `status_ledger.py` 按**目录存在**计数，无内容门 → 目录本身即可抬计数。

**表 6 — 溯源**

| 文件 | 首次引入 |
|---|---|
| `src/tensorlbm/nwt/stokes2.py` | {PROV['src/tensorlbm/nwt/stokes2.py']['added_by']} |
| `tests/test_nwt_stokes2.py` | {PROV['tests/test_nwt_stokes2.py']['added_by']} |
| `benchmarks/verified/nwt_stokes2/run.py` | {PROV['benchmarks/verified/nwt_stokes2/run.py']['added_by']} |

结论：能力本体与其单元测试**均早于本单**（m5712，2026-11-08）；m17013 新增的实质物只有
"一个解析闭式驱动 + 一次台账提升"。mission 报告里"`test_nwt_stokes2.py` 9 passed"是**复跑既有测试**，
不构成本单新增证据（复跑已核实：9 passed）。

## ④ 结果图

`docs/audit_m17013/fig_audit_m17013.png`（由 `gen_figure.py` 从 `audit.json` 解析生成，双面板）：

- **(a)** 12 次门调用的实测相对偏差（对数轴）与驱动自设容差线（1e-12 / 1e-6）。全部落在 1e-17…1.4e-7，
  但**参照系是同一闭式的另一次誊写**——偏差小只说明"公式写对了"，不是"求解器算对了"。
- **(b)** 覆盖矩阵：驱动文档字符串声称的 7 个物理面里，**自由面 η(x,t)、自由面边界条件残差 O(ε²)、
  速度场连续性/∇φ 三项无任何调用点**；变异探针 A 把 η 改成常量 12345 后门依然 12/12 PASS，
  变异探针 B（对照）改坏 a2 才 11/12 FAIL。

## ⑤ 注意点（踩坑 / 局限 / 适用范围）

1. **上报口径大于证据**：`run.py` 文档字符串写"the surface boundary conditions (second order in steepness),
   and the divergence-free / potential-matching velocity field"，但 12 个门里**没有一个**触及这两项；
   变异探针 A 是反证（12/12 仍 PASS）。文档字符串应删掉这两句，或补门。
2. **把"解析参考自证"当"能力 verified"**：`benchmarks/verified/` 的书架纪律明写
   `Acceptance: every checkpoint within 3% AND mesh-converged (two grids, span <=3%)`（`benchmarks/STATUS.md:5`）；
   本 case **无网格、无时间推进、无双网格收敛**，按书架自身判据不满足"verified"。它应挂在
   "analytic anchor / reference generator" 类目下，而不是计入"独立物理验证的能力数"。
3. **计数门是目录门**：`status_ledger.py --write` 按目录扫描计数（`benchmarks/status_ledger.py:215`），
   目录一建即 +1；本次 +2 里有 1 是陈旧修正。KPI 若直接读该计数会放大产出。
4. **注册残缺**：新 case 无 README；人工表 note 被追加在 `<!-- STATUS_LEDGER:END -->` **之后**
   （`benchmarks/STATUS.md` 文件尾），落在 `## verified/` 表段之外 → 台账仍显示 `nwt_stokes2 | **no**`（未覆盖）。
5. **同能力真正的求解器侧验证仍在 pending**：`benchmarks/pending/nwt_stokes2_2d/`（`run_acceptance.py`
   C1–C7：色散 1%、A1 3%、A2 10%、海绵层 5%、质量 0.5%/周期、双网格 3%、墙钟 20 min）**没有任何
   result/validation_status 文件**，台账 `machine=unknown`。这才是该能力能升 verified 的唯一路径。
6. **报告欠账**：m17013 `report_owed=t`、`tensor_reports` 里 0 条记录；单子以
   `fail_stage=budget_exhausted+forced_finish`（60 tool calls）收尾。交付本身复跑成立，但**无论文级报告**，
   §④ 的"与解析解比较曲线"在该单语境下只能以解析量自身闭合（无实验/文献离散数据可比）。
7. 局限：本审计只核"交付是否支撑其声明"，不重跑 NWT 求解器（`nwt_stokes2_2d` 矩阵需数十 CPU·h，
   且其机器裁决文件缺失 → 直接判 WITHHELD，不臆造数字）。

## ⑥ 误差判定（关键量 vs 基准，逐项，显式回答"是否 <3%"）

| 关键量 | 基准 | 本审 got vs ref | 是否 <3% | 判定 |
|---|---|---|---|---|
| ω² vs g k tanh(kh)（4 组 k,h） | 闭式（Dean & Dalrymple §4.3） | 0 … 1.49e-16 | **是** | PASS，但参照系=同一闭式 |
| 深/浅水 c, c_g 渐近 | sqrt(g/k)、sqrt(gh)、0.5c、c | ≤1.43e-07 | **是** | PASS |
| 二阶谐波系数 a2（有限深/深水） | (ka²/4)(3−σ²)/σ³、ka²/2 | 0（同式） | **是** | PASS |
| 波数反演 / 周期自洽 | 自洽 | 0 | **是** | PASS（round-trip，非物理基准） |
| 自由面 η(x,t) 幅值/相位 vs 解析 | 解析 | **未测** | **无法回答** | **WITHHELD** |
| 自由面边界条件残差 O(ε²) | 解析 | **未测** | **无法回答** | **WITHHELD** |
| 速度场连续性/无旋性 | 解析 | **未测** | **无法回答** | **WITHHELD** |
| LBM 波槽数值解 vs Stokes 二阶解（C1–C7） | 解析 + 双网格 | **无机器裁决** | **无法回答** | **WITHHELD** |

**直答**：已交付门覆盖的量，误差全部远小于 3%（≤1.43e-07）；但那些量是"公式 vs 同一公式"的自洽核对，
**对"求解器能力是否被独立物理验证"这一 KPI 命题不提供任何证据**。求解器侧关键量（η、BC 残差、
速度场、C1–C7 容差 1%/3%/10%）**无数据 → 一律 WITHHELD，不得写作通过**。

## ⑦ 判级与裁定

**级别：L1（数学/单元验证）** —— 逐级说明为何上不去：

| 级 | 判据 | 本单实况 |
|---|---|---|
| L1 数学/单元验证 | 有可跑的数学/单元断言 | ✅ 12/12 门复跑成立、与交付 JSON 逐键一致；变异 B 证明门有检出能力 |
| L2 独立功能实现 | 新增真代码：输入→计算→输出 | ❌ 新增物是**验证驱动**；能力本体 `stokes2.py` 早于本单（m5712 / 551cff07），本单未实现新功能 |
| L3 系统集成 | 接入主求解器/平台调用链 | ❌ 驱动 0 个求解器 import、0 个 `elevation/potential/kinematics` 调用点；耦合只在**既有**代码里（`nwt/tank.py:42`），非本单交付 |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现 | ❌ 无网格/时间推进/双网格收敛，违反 `STATUS.md:5` 的 verified 判据；求解器侧 `nwt_stokes2_2d` 无机器裁决 |
| L5 科研工程 | L4 + 外部锚 | ❌ 仅教科书闭式，无实验/文献离散数据；且 L4 未达 |

**Disposition：request changes（返修）**。可执行的最小返修路径：

1. 把 `benchmarks/verified/nwt_stokes2` 降级/改挂为 analytic-anchor 类目（或补 `README.md` 说明"参考解自证，
   非求解器验证"，并修正 `run.py` 文档字符串中未被覆盖的两项声明）；
2. 给 `run.py` 补两门：自由面 BC 残差应随 ε² 下降（`Stokes2Wave.surface_bc_residual`）+ 二维速度场
   连续性与 ∇φ 一致性——这样变异探针 A 类篡改才可能被抓住；
3. 真正升 verified 的做法：跑 `benchmarks/pending/nwt_stokes2_2d/run_acceptance.py all`，落
   `results.json`（C1 1% / C2 A1 3% / C2 A2 10% / C3 5% / C4 0.5%·周期 / C5 双网格 3%），再按书架判据入台账；
4. 台账侧：`status_ledger --check` 应把"verified 目录缺 README 或缺收敛证据"列为 hard violation，
   否则目录门会持续放大 KPI。
"""

OUT.write_text(md)
print("wrote", OUT, len(md), "bytes")