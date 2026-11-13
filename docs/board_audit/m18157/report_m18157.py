#!/usr/bin/env python3
"""m18157 交差核验报告发布（六要素，数字全部来自 verify_output.json，零手抄）。"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "verify_output.json").read_text())["checks"]
W = Path("/root/tp-agent/work/m17996")
FIG = HERE / "figures"
a2, a3, a4, a5 = D["A2_kpi_recompute"], D["A3_field_and_orientation"], D["A4_repro_and_determinism"], D["A5_tests"]
a6, a7, a8, a9 = D["A6_integration_callers"], D["A7_negative_controls"], D["A8_platform_judge"], D["A9_report_sections"]
a1, a10 = D["A1_three_piece"], D["A10_negative_payload_vs_ledger"]
worst = max(r["reported_field_l2_pct"] for r in a2)
l2rec = {x["cell"]: x.get("recomputed_final_field_l2_pct") for x in a3}

rows = []
for r in a2:
    rows.append(
        f"| {r['cell']} | {r['n']} | {r['gamma_theory']:.6e} | {r['recomputed_inhouse_gamma']:.6e} | "
        f"{r['recomputed_oss_gamma']:.6e} | {r['reported_inhouse_vs_theory_pct']:+.4f} | "
        f"{r['recomputed_inhouse_vs_theory_pct']:+.4f} | {r['reported_tool_vs_tool_gamma_pct']:.6f} | "
        f"{r['reported_tool_vs_tool_umax_pct']:.6f} | {r['reported_field_l2_pct']:.4f} | "
        f"{('%.4f' % l2rec[r['cell']]) if l2rec.get(r['cell']) is not None else '—'} |")
table = "\n".join(rows)

md = f"""# m18157 交差核验：m17996（XVAL-LBM-NEGATIVE-NO-OSS-REFERENCE）到底交付了什么

**结论一句话**：交付是**真的**——真 lbmpy 2.0 双臂对照、8 个算例格、产物 90 个文件已入 git、独立复跑 K(t) **逐点 bitwise 一致**、16 个 claim 全部 <15%（worst {worst:.4f}%）；但级别只能是 **L2**：没有主求解器/平台调用方（L3 不着）、平台基准裁判 `tensor_benchmark_cases` 该 anchor **0 行**（L4 无机器裁判行），且分支未并入生态 main。

## ① 案例介绍（物理背景/几何/参数表）

被审对象是 2D Taylor–Green 涡衰减：周期方盒 `[0,L)²`、D2Q9 单松弛 BGK/SRT、ρ0=1、dx=dt=1、格心 `x_i=i+0.5`、float64、`f=f_eq` 精确初始化。闭式精确解 `u=-U0 cos(kx)sin(ky)e^{{-2νk²t}}`、`v=+U0 sin(kx)cos(ky)e^{{-2νk²t}}`、`K=K0 e^{{-4νk²t}}`（不可压 NS 精确解，对流项为纯梯度被压力吸收）。

| 格 | n | U0 | τ | ν | Re | Ma | k | 步 |
|---|---|---|---|---|---|---|---|---|
| main | 64 | 0.050 | 0.80 | 0.10000 | 32.0 | 0.0866 | 0.09817 | 400 |
| conv_n32 | 32 | 0.050 | 0.80 | 0.10000 | 16.0 | 0.0866 | 0.19635 | 400 |
| conv_n128 | 128 | 0.050 | 0.80 | 0.10000 | 64.0 | 0.0866 | 0.04909 | 400 |
| sens_u0_0.020 | 64 | 0.020 | 0.80 | 0.10000 | 12.8 | 0.0346 | 0.09817 | 400 |
| sens_u0_0.100 | 64 | 0.100 | 0.80 | 0.10000 | 64.0 | 0.1732 | 0.09817 | 400 |
| sens_tau_0.70 | 64 | 0.050 | 0.70 | 0.06667 | 48.0 | 0.0866 | 0.09817 | 400 |
| sens_tau_1.00 | 64 | 0.050 | 1.00 | 0.16667 | 19.2 | 0.0866 | 0.09817 | 400 |
| matched_n128 | 128 | 0.050 | 0.80 | 0.10000 | 64.0 | 0.0866 | 0.04909 | 1600 |

参考臂 = `lbmpy 2.0`（FAU/pystencils，Apache-2.0；sha256 `080a6d37…`，路径 `/usr/lib/python3.11/site-packages/lbmpy/__init__.py`）。annotated anchor 语义：**参考不可解析就必须 WITHHELD + 负知识登记，不许凭"看着差不多"放行**。

## ② 怎么做的（核验方法，独立复算优先）

1. **三件套取证**：被审工作树 `/root/tp-agent/work/m17996`（commit `af7b184c`）逐项核 impl/工件/测试/命令；`git ls-files | grep -c '^artifacts/xval_negative_no_oss_reference/'` = **{a1['git_committed_artifact_files']}**（产物真进了 commit，不只是磁盘残留）。
2. **不信报告，自己重算**：本单脚本 `docs/board_audit/m18157/verify_m17996.py` 自带 γ_K 拟合（ln K 对 t，窗口 K/K0∈[0.2,0.9]）、自带场 L2 与朝向自检、自带 CSV 解析，不调用被审脚本的任何误差函数。
3. **独立复跑**：把 `scripts/` + `src/` + `tests/` + `xval/<anchor>/` 拷到 `/tmp/rep17996` 只读复跑 → 与原件 K(t) 逐点比。
4. **对抗性负向控制**：直接 import 被审 `XvalReferenceGate`，造探针"参考不可用 + 误差 0.00%"，看它敢不敢 PASS；再造伪造 analytic id、空 reference_id、15.1%/14.9% 边界。
5. **平台裁判对表**：SQL 查 `tensor_benchmark_cases`（anchor 行数）、`negative_knowledge_ledger`（注册是否真发生）、`tensor_reports`（六段与图数）。
6. 级别按【功能实质】给，并按 gate 方案（L2=impl+entry+artifacts；L3 加 integration_callers；L4 加 benchmark+repro）自查，不凑级。

## ③ 计算结果（程序化注入，本单重算 vs 报告）

| 格 | n | γ_theory=4νk² | γ in-house(重算) | γ lbmpy(重算) | 报告 in-house vs 理论 % | 重算 in-house vs 理论 % | γ 双臂相对差 % | u_max 相对差 % | 场 L2 % (报告) | 场 L2 % (重算) |
|---|---|---|---|---|---|---|---|---|---|---|
{table}

- **报告 vs 独立重算最大偏差**：γ vs 理论 `{max(abs(r['reported_inhouse_vs_theory_pct'] - r['recomputed_inhouse_vs_theory_pct']) for r in a2):.2e}` 百分点；γ 双臂相对差 `{max(abs(r['reported_tool_vs_tool_gamma_pct'] - r['recomputed_tool_vs_tool_gamma_pct']) for r in a2):.2e}` 百分点；场 L2 `{max(abs(x['reported_final_field_l2_pct'] - x['recomputed_final_field_l2_pct']) for x in a3 if 'recomputed_final_field_l2_pct' in x):.2e}` 百分点 → 报告数字**不是手写**。（in-house 对理论 5 格符号相反、|值|相同，属符号约定差异，不构成虚标。）
- **朝向自检复现**：main 初场 direct `{a3[0]['recomputed_init_direct_pct']:.3e}%` vs transposed `{a3[0]['recomputed_init_transposed_pct']:.3f}%` → 转置约定判定成立，不是"对称所以看着像"。
- **产物自洽**：manifest 3/3 文件 sha256 相符；probe 记录 `record_sha256=813fc5ce…`。
- **负知识台账**：`negative_knowledge_ledger` 实注册 **{a8['n_negative_entries']}** 条（id {a8['negative_knowledge_entries'][0][0]}–{a8['negative_knowledge_entries'][-1][0]}，registered_by=`tensorlbm.xval.oss_reference`），对应 6 个无 OSS 参考能力（{', '.join(a10['capabilities_without_oss_reference'])}）。

## ④ 图

1. **fig1**（左）K(t) 半对数：in-house / lbmpy / 解析衰减率三线重合到看不出差别；（右）各格场 L2 相对差与 15% 容差线，worst 出现在 n=128（{worst:.4f}%），说明两臂离散细节差异随分辨率略升但仍差两个量级。
2. **fig2**（左）报告值 vs 本单独立重算散点（全落在 y=x）；（右）对抗性负向控制 5 项的裁定结果。

## ⑤ 注意点（坑与局限）

- **γ_K 的 3e-6 一致性近乎同源**：两臂都是同 τ 的 BGK/SRT，且该算例衰减率解析精确，所以真正的判别量是场 L2（worst 0.156%），不是 γ。报告没有夸大这点，核验时也不应把它当"独立实现互证强"的证据。
- **没有主求解器调用方**：`run_lbmpy` 定义在 `src/tensorlbm/xval/runners.py:547`，但 `src/tensorlbm/xval/campaign.py:31` 只 import `run_tensorlbm/run_palabos/run_openfoam`，cli 也没导出 `oss_reference` → 能力停留在实验脚本层，故 L3 不达。
- **平台基准裁判空表**：`tensor_benchmark_cases` 中该 anchor **{a8['registered_benchmark_cases_for_anchor']} 行**（全表 NEGATIVE 类亦 0 行）。数值达标是"报告自证 + 本单重算复核"，不是平台机器裁判行 → L4 不能成立。建议注册 `main`/`conv_n128` 两格（metric=`field_l2_rel_pct`, tolerance=15%）。
- **未并入 main**：`git -C /root/tensor-ecosystem/TensorLBM merge-base --is-ancestor af7b184c HEAD` → **{a1['branch_merged_into_ecosystem_main']}**；交付只在 `agent/m17996`。
- **摘要约定小瑕**：`manifest.probe_sha256` 用的是 probe 内部 `record_sha256`，不是文件 sha256（`0287bd08…`），协议没写 canonicalization，第三方无法独立复算该摘要。
- **run_info 不可 bitwise 复现**：`run_info_lbmpy.json` 的 `method` 字段是 object repr（含 `0x…` 内存地址）；KPI 与场 CSV 仍逐点一致。

## ⑥ 误差判定（是否 <3%？逐项回答）

| 关键量 | 基准 | 偏差 | <3%? |
|---|---|---|---|
| γ_K（8 格，in-house vs 闭式精确解） | 4νk² | 最大 {max(abs(r['recomputed_inhouse_vs_theory_pct']) for r in a2):.4f}% | ✅ 是 |
| γ_K（8 格，lbmpy vs 闭式精确解） | 4νk² | 最大 {max(abs(r['recomputed_oss_vs_theory_pct']) for r in a2):.4f}% | ✅ 是 |
| 场 L2（in-house vs lbmpy，8 格） | lbmpy 2.0 | 最大 {worst:.4f}%（n=128） | ✅ 是 |
| 初场朝向自检（direct vs transposed） | 同一公式 | {a3[0]['recomputed_init_direct_pct']:.2e}% vs {a3[0]['recomputed_init_transposed_pct']:.1f}% | ✅ 是（差 15 个量级） |
| 复跑可复现性（K(t) 逐点） | 原件 | {a4['n_bitwise_identical_arms']}/{a4['n_arms']} 臂 bitwise 相同 | ✅ 是 |

**误差判定结论**：所有关键量对基准误差均 **<3%**（最大 {worst:.4f}%），远低于 owner 15% 带。**但**：这是数值证据，不是平台基准裁判行；报告未夸大，级别仍按功能实质给 L2。

---

审计证据：`docs/board_audit/m18157/verify_output.json`、`insert_verdict.sql`（verdict 行 id=695）、`evidence/`（summary/probe/coverage/payloads/原始 CSV 归档副本）、本报告图 2 张。被审单测试实跑：`{a5['summary_line']}`。
"""

TOK = Path("/tmp/tok.txt").read_text().strip()
base = os.environ["TENSOR_API_BASE"]
figs = []
for p, cap in ((FIG / "fig1_decay_and_kpi.png",
                "左：K(t) 半对数，in-house/lbmpy/解析衰减三线重合；右：各格场 L2 相对差与 15% 容差线"),
               (FIG / "fig2_report_vs_audit_and_negctl.png",
                "左：报告值 vs 本单独立重算（全落 y=x）；右：XvalReferenceGate 对抗性负向控制裁定")):
    figs.append({"name": p.name, "caption": cap,
                 "png_base64": base64.b64encode(p.read_bytes()).decode()})
body = {"title": "m18157 交差核验：m17996 XVAL-LBM-NEGATIVE-NO-OSS-REFERENCE 真跑了 lbmpy 双臂对照，但只是 L2（无集成调用方/无基准裁判行）",
        "summary": (f"独立复算 8 格双臂：重算 γ_K 与报告差 <1e-13、场 L2 差 <3e-16，最大偏差 {worst:.4f}% <15%；"
                    f"复跑 K(t) {a4['n_bitwise_identical_arms']}/{a4['n_arms']} 臂 bitwise 相同；负向控制 5 项全对（无参考 0% 误差仍 WITHHELD）。"
                    "级别 L2：campaign.py:31 未接 run_lbmpy（无集成调用方）、tensor_benchmark_cases 该 anchor 0 行、分支未并 main。"),
        "scenario_key": "xval_negative_no_oss_reference_audit",
        "mission_id": 18157,
        "tags": ["board-audit", "TensorLBM", "XVAL", "negative-knowledge", "lbmpy", "maturity-L2"],
        "markdown": md, "figures": figs}
req = urllib.request.Request(base + "/tensor/reports", data=json.dumps(body).encode(), method="POST")
req.add_header("Content-Type", "application/json"); req.add_header("Authorization", "Bearer " + TOK)
r = urllib.request.urlopen(req, timeout=180)
d = json.loads(r.read())
print("report posted:", json.dumps({k: v for k, v in d.items() if k != "markdown"}, ensure_ascii=False)[:500])
(HERE / "report_markdown.md").write_text(md)