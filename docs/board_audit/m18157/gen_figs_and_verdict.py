#!/usr/bin/env python3
"""m18157：交差核验图 + verdict 落表（board-audit 席位）。

图只画本审计单重算出来的数字（来源 verify_output.json，程序化注入，零手抄）。
落表 evidence 的每个指针都是本单实测的真实路径/行号/命令行；级别按【功能实质】给，
不为凑级把「测试引用了模块」写成 integration caller。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
D = json.loads((HERE / "verify_output.json").read_text())["checks"]
W = Path("/root/tp-agent/work/m17996")
ART = W / "artifacts/xval_negative_no_oss_reference"
LEVEL = 2


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16] if Path(p).is_file() else "MISSING"


# ── 图 1：K(t) 衰减，双臂 + 精确解（半对数）──────────────────────────────
rows = D["A2_kpi_recompute"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for c, style in (("main", "-"), ("conv_n128", "--")):
    for arm, fn, col in (("in-house tensorlbm", "tensorlbm/k_history_tensorlbm.csv", "tab:blue"),
                         ("lbmpy 2.0", "lbmpy/k_history_lbmpy.csv", "tab:orange")):
        t, k = [], []
        for i, line in enumerate((ART / "cells" / c / fn).read_text().splitlines()):
            if i and line.strip():
                a = line.split(","); t.append(float(a[0])); k.append(float(a[1]))
        ax[0].semilogy(t, k, style, color=col, lw=1.6, label=f"{c} {arm}")
        gammas = [r for r in rows if r["cell"] == c][0]
        gam = gammas["recomputed_inhouse_gamma"] if arm.startswith("in-house") else gammas["recomputed_oss_gamma"]
        ax[0].semilogy(t, [k[0] * pow(2.718281828459045, -gam * tt) for tt in t], ":", color=col, lw=1.0,
                       label=f"{c} analytic exp(-γt), γ=4νk²" if arm.startswith("in-house") else None)
ax[0].set_xlabel("time step"); ax[0].set_ylabel("K = 0.5·<|u|²>")
ax[0].set_title("TG decay: in-house vs lbmpy vs closed-form (audit re-fit)")
ax[0].legend(fontsize=7, ncol=1); ax[0].grid(alpha=.3, which="both")

cells = [r["cell"] for r in rows]
l2 = [max(r["reported_field_l2_pct"], 1e-6) for r in rows]
ax[1].semilogy(range(len(cells)), l2, "o-", color="tab:red", label="field L2 in-house vs lbmpy (%)")
ax[1].axhline(15.0, color="k", ls="--", lw=1.2, label="owner tolerance 15%")
for i, v in enumerate(l2):
    ax[1].annotate(f"{v:.3g}", (i, v), textcoords="offset points", xytext=(0, 6), fontsize=7)
ax[1].set_xticks(range(len(cells))); ax[1].set_xticklabels(cells, rotation=30, ha="right", fontsize=7)
ax[1].set_ylabel("field L2 rel. diff (%)"); ax[1].set_title("tool-vs-tool KPI per cell (recomputed from raw CSVs)")
ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig(FIG / "fig1_decay_and_kpi.png", dpi=140); plt.close(fig)

# ── 图 2：重算 vs 报告（一致性）+ 负向控制结果 ──────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
rep = [r["reported_tool_vs_tool_gamma_pct"] for r in rows]
rec = [r["recomputed_tool_vs_tool_gamma_pct"] for r in rows]
ax[0].scatter(rep, rec, s=42, color="tab:green")
lim = [min(rep + rec) * 0.5, max(rep + rec) * 2]
ax[0].plot(lim, lim, "k--", lw=1, label="y = x")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("reported γ_K tool-vs-tool (%)"); ax[0].set_ylabel("audit-recomputed (%)")
ax[0].set_title("report vs independent re-fit (raw CSVs)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, which="both")

nc = D["A7_negative_controls"]
labels = ["no ref\n(err=0%)", "fake analytic", "resolved ref\n(err=0%)", "err=15.1%", "err=14.9%"]
keys = ["no_ref_zero_error", "fake_analytic", "resolved_ref_zero_error", "over_tolerance", "under_tolerance"]
vals = [nc[k]["status"] if isinstance(nc.get(k), dict) else nc.get(k) for k in keys]
cmap = {"WITHHELD": "tab:orange", "PASS": "tab:green", "FAIL": "tab:red"}
ax[1].bar(range(len(vals)), [1] * len(vals), color=[cmap.get(v, "grey") for v in vals])
for i, v in enumerate(vals):
    ax[1].text(i, 0.5, v, ha="center", va="center", fontsize=9, rotation=90, color="white", weight="bold")
ax[1].set_xticks(range(len(vals))); ax[1].set_xticklabels(labels, fontsize=7)
ax[1].set_yticks([]); ax[1].set_title("adversarial negative controls on XvalReferenceGate")
fig.tight_layout(); fig.savefig(FIG / "fig2_report_vs_audit_and_negctl.png", dpi=140); plt.close(fig)
print("figures:", sorted(p.name for p in FIG.glob("*.png")))

# ── verdict evidence（全部真实指针）─────────────────────────────────────
a1, a2, a3, a4, a5 = D["A1_three_piece"], D["A2_kpi_recompute"], D["A3_field_and_orientation"], \
    D["A4_repro_and_determinism"], D["A5_tests"]
a6, a7, a8, a9, a10 = D["A6_integration_callers"], D["A7_negative_controls"], D["A8_platform_judge"], \
    D["A9_report_sections"], D["A10_negative_payload_vs_ledger"]
worst = max(r["reported_field_l2_pct"] for r in a2)
evidence = {
    "impl_files": [
        "/root/tp-agent/work/m17996/src/tensorlbm/xval/oss_reference.py",
        "/root/tp-agent/work/m17996/src/tensorlbm/xval/runners.py",
        "/root/tp-agent/work/m17996/scripts/xval_negative_no_oss_reference.py",
        "/root/tp-agent/work/m17996/xval/XVAL-LBM-NEGATIVE-NO-OSS-REFERENCE/predeclared.json",
    ],
    "entry": [
        "/root/tp-agent/work/m17996/scripts/xval_negative_no_oss_reference.py",
        "/root/tp-agent/work/m17996/src/tensorlbm/xval/oss_reference.py",
    ],
    "run_artifacts": [
        "/root/tp-agent/work/m17996/artifacts/xval_negative_no_oss_reference/summary.json",
        "/root/tp-agent/work/m17996/artifacts/xval_negative_no_oss_reference/oss_reference_probe.json",
        "/root/tp-agent/work/m17996/artifacts/xval_negative_no_oss_reference/capability_coverage.json",
        "/root/tp-agent/work/m17996/artifacts/xval_negative_no_oss_reference/negative_knowledge_payloads.json",
        "/root/tp-agent/work/m17996/artifacts/xval_negative_no_oss_reference/manifest.json",
        "/root/tp-agent/work/m17996/artifacts/xval_negative_no_oss_reference/REPORT.md",
        "/root/tp-agent/work/m17996/artifacts/xval_negative_no_oss_reference/cells/main/cell_result.json",
        "/root/tp-agent/work/m17996/artifacts/xval_negative_no_oss_reference/cells/conv_n128/lbmpy/k_history_lbmpy.csv",
        "/root/tp-agent/work/m17996/artifacts/xval_negative_no_oss_reference/cells/conv_n128/tensorlbm/k_history_tensorlbm.csv",
        "/root/tp-agent/work/m18157/docs/board_audit/m18157/evidence/summary.json",
        "/root/tp-agent/work/m18157/docs/board_audit/m18157/evidence/oss_reference_probe.json",
        "/root/tp-agent/work/m18157/docs/board_audit/m18157/evidence/negative_knowledge_payloads.json",
        "/root/tp-agent/work/m18157/docs/board_audit/m18157/evidence/capability_coverage.json",
        "/root/tp-agent/work/m18157/docs/board_audit/m18157/evidence/cells/main/lbmpy/k_history_lbmpy.csv",
    ],
    "impl_files_pointers": [
        f"{W}/src/tensorlbm/xval/oss_reference.py:1-{a1['impl'][0]['lines']} "
        f"(probe_oss_references:330 / audit_capability_coverage:386 / XvalReferenceGate:488 / decide:541 / "
        f"negative_knowledge_payloads:586) sha256={sha(W / 'src/tensorlbm/xval/oss_reference.py')}",
        f"{W}/src/tensorlbm/xval/runners.py:547 run_lbmpy（OSS 臂适配器，同 TG2DSpec、同 CSV 格式）"
        f" sha256={sha(W / 'src/tensorlbm/xval/runners.py')}",
        f"{W}/scripts/xval_negative_no_oss_reference.py:1-{a1['impl'][2]['lines']}（8 cell × 2 臂驱动 + 出表出图）"
        f" sha256={sha(W / 'scripts/xval_negative_no_oss_reference.py')}",
        f"{W}/xval/XVAL-LBM-NEGATIVE-NO-OSS-REFERENCE/predeclared.json（动手前写死判据：15% owner 带、"
        f"K/K0∈[0.2,0.9] 拟合窗、400 步/臂、wall 600 s）sha256={sha(W / 'xval/XVAL-LBM-NEGATIVE-NO-OSS-REFERENCE/predeclared.json')}",
    ],
    "entry_pointers": [
        f"{W}/scripts/xval_negative_no_oss_reference.py:512 main()（--out/--mission-id/--cells），"
        "本单只读复跑实跑通过（见 repro_cmd）",
        f"{W}/src/tensorlbm/xval/oss_reference.py:673 main()（审计包 CLI）",
    ],
    "run_artifacts_pointers": [
        f"{W}/artifacts/xval_negative_no_oss_reference/summary.json（16 claim PASS/0 FAIL/0 WITHHELD，"
        f"worst={worst:.4f}%，probe_sha256=813fc5ce…）",
        f"{W}/artifacts/xval_negative_no_oss_reference/oss_reference_probe.json"
        f"（7 候选 OSS，available=lbmpy 2.0 + palabos 二进制，5 项 honest 不可用；文件 sha256={sha(ART / 'oss_reference_probe.json')}）",
        f"{W}/artifacts/xval_negative_no_oss_reference/capability_coverage.json（13 能力：7 有 OSS 参考 / 6 无）",
        f"{W}/artifacts/xval_negative_no_oss_reference/negative_knowledge_payloads.json",
        f"{W}/artifacts/xval_negative_no_oss_reference/cells/{{{','.join(r['cell'] for r in a2)}}}/"
        "{lbmpy,tensorlbm}/{k_history,field,run_info}*（8 cell × 2 臂；audit 归档副本 "
        f"{HERE}/evidence/cells/）",
        f"git 入库核验：`git ls-files | grep -c '^artifacts/xval_negative_no_oss_reference/'` = "
        f"{a1['git_committed_artifact_files']}（产物真在 commit af7b184c 里，不是只在磁盘上）",
        f"manifest 内部一致性：3/3 文件 sha256 与 manifest 相符；manifest.probe_sha256=813fc5ce… 与 probe 内 "
        f"record_sha256 一致（≠ 文件 sha256 0287bd08…，协议未说明 canonicalization → 可复现性小瑕）",
    ],
    "tests": [
        "tests/test_xval_oss_reference.py 本单实跑：`PYTHONPATH=<wt>/src python3 -m pytest "
        "tests/test_xval_oss_reference.py -q -p no:cacheprovider` → " + a5["summary_line"],
    ],
    "integration_callers": (
        "NONE —— 诚实核验（不为凑级凑调用方）：对 tensorlbm.xval.oss_reference / XvalReferenceGate / "
        "probe_oss_references / run_lbmpy 做仓内 grep，唯一命中是定义处 src/tensorlbm/xval/runners.py:547；"
        "src/tensorlbm/xval/campaign.py:31 `from .runners import run_openfoam, run_palabos, run_tensorlbm` "
        "未把 run_lbmpy 接进调度，cli/__main__ 未导出 oss_reference。→ L3(系统集成) 不达"),
    "benchmark": {
        "registered_case": "NONE —— tensor_benchmark_cases 中 anchor_slug="
                          "'XVAL-LBM-NEGATIVE-NO-OSS-REFERENCE' 行数 = 0（平台基准裁判空表，"
                          "全表 NEGATIVE 类 anchor 亦为 0 行）→ 无机器裁判行，L4 不能按平台判据成立",
        "measured_tool_vs_tool": {r["cell"]: {"ref": "lbmpy 2.0",
                                              "got_inhouse_field_l2_pct": r["reported_field_l2_pct"],
                                              "audit_recomputed_field_l2_pct":
                                                  [x["recomputed_final_field_l2_pct"] for x in a3
                                                   if x.get("cell") == r["cell"]][0]
                                              if any(x.get("cell") == r["cell"] and "recomputed_final_field_l2_pct" in x
                                                     for x in a3) else None,
                                              "gamma_K_rel_pct": r["reported_tool_vs_tool_gamma_pct"],
                                              "audit_recomputed_gamma_rel_pct": r["recomputed_tool_vs_tool_gamma_pct"]}
                                           for r in a2},
        "audit_recompute_max_abs_delta_vs_report": {
            "gamma_vs_theory_pct": max(abs(r["reported_inhouse_vs_theory_pct"] - r["recomputed_inhouse_vs_theory_pct"])
                                       for r in a2),
            "tool_vs_tool_gamma_pct": max(abs(r["reported_tool_vs_tool_gamma_pct"] - r["recomputed_tool_vs_tool_gamma_pct"])
                                          for r in a2),
            "field_l2_pct": max(abs(x["reported_final_field_l2_pct"] - x["recomputed_final_field_l2_pct"])
                                for x in a3 if "recomputed_final_field_l2_pct" in x),
        },
        "worst_error_vs_tolerance_pct": {"worst": worst, "tolerance": 15.0, "pass": worst < 15.0},
        "orientation_check_reproduced": [{"cell": x["cell"], "direct_pct": x["recomputed_init_direct_pct"],
                                          "transposed_pct": x["recomputed_init_transposed_pct"]} for x in a3
                                         if "recomputed_init_direct_pct" in x],
    },
    "repro_cmd": a4["repro_cmd"],
    "external_anchor": (
        "lbmpy 2.0（FAU Erlangen / pystencils 代码生成，Apache-2.0）: "
        "path=/usr/lib/python3.11/site-packages/lbmpy/__init__.py "
        "sha256=080a6d37a69806f528c6b458590e68f61b3d90e60115b5d293c4cd5652152415（独立于 torch 求解器的代码路径）；"
        "解析锚=2D Taylor-Green 闭式精确解 u=-U0 cos(kx)sin(ky)e^{-2νk²t}, v=+U0 sin(kx)cos(ky)e^{-2νk²t}, "
        "K=K0 e^{-4νk²t}（不可压 NS 精确解，对流项为纯梯度）；平台负知识台账 6 条实注册 "
        f"(negative_knowledge_ledger id {a8['negative_knowledge_entries'][0][0]}-"
        f"{a8['negative_knowledge_entries'][-1][0]}, registered_by=tensorlbm.xval.oss_reference)"),
    "level_rationale": (
        "L2（独立功能实现）：真代码（lib 4 处 + 驱动脚本）+ 真跑（8 cell × 2 臂、16 claim、"
        f"产物 {a1['git_committed_artifact_files']} 个文件已入 git）+ 独立复跑逐点 bitwise 一致"
        f"（{a4['n_bitwise_identical_arms']}/{a4['n_arms']} 臂 K(t) 完全相同）。"
        "L3 不达：无主求解器/平台调用方（campaign.py:31 未接 run_lbmpy；cli 未导出）。"
        "L4 不达：tensor_benchmark_cases 无该 anchor 注册行，平台基准裁判无法给出机器判定；"
        "且 L4 在 gate 方案里以 L3 为前置。数值层面 15% 判据确实满足（worst "
        f"{worst:.4f}%），但这是报告自证 + 本单重算复核，属「证据强、登记缺」。"),
    "findings": {
        "F1_no_benchmark_registration": "anchor 无 tensor_benchmark_cases 行 → 平台裁判空转；建议把 main/conv_n128 两格注册为 case（metric=field_l2_rel_pct, tolerance=15%）",
        "F2_no_integration_caller": "run_lbmpy 只被本单脚本调用（runners.py:547 定义、仓内无其它调用点），xval 调度 campaign.py 未接入 → 能力停留在实验脚本层",
        "F3_not_merged": "commit af7b184c 未进入生态 main（merge-base --is-ancestor → NOT_MERGED），交付仅存在于 agent/m17996 分支",
        "F4_probe_digest_convention": "manifest.probe_sha256 用的是 probe 内部 record_sha256(813fc5ce…)，不是文件 sha256(0287bd08…)；协议未写 canonicalization，第三方无法独立复算该摘要 → 建议直接存文件 sha256",
        "F5_negctl_pass": "对抗性负向控制 5 项全部正确：无参考且误差 0% → WITHHELD(negative_knowledge_required=true)、伪造 analytic id → WITHHELD、空 reference_id → UnreferencedClaimError、15.1% → FAIL、14.9% → PASS；PASS 分支在 resolved 判定之后，代码中不存在无参考也 PASS 的路径",
        "F6_gamma_agreement_partly_tautological": "γ_K 双臂差 3e-6（相对）不宜当独立验证强度：两臂同为 BGK/SRT 且衰减率解析精确，判别量其实是场 L2（worst 0.156%）；报告并列了两臂对理论误差，未夸大",
        "F7_report_honest": "报告六段齐全（平台 report 16912，3762 字符 + 4 图；仓内 md 13681 字符 20 表），数值与本单独立重算一致（γ 差 <1e-13、场 L2 差 <3e-16；符号约定差异仅 in-house 侧 5 格，绝对值相同）",
        "F8_run_info_nondeterminism": "cells/*/lbmpy/run_info_lbmpy.json 的 method 字段存的是 object repr（含内存地址 0x…），逐次不同 → 工件不可 bitwise 复现（KPI 与场 CSV 仍 bitwise 一致）",
    },
}

payload = json.dumps(evidence, ensure_ascii=False)
sql = (
    "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
    "VALUES ('TensorLBM', 'm17996 XVAL-LBM-NEGATIVE-NO-OSS-REFERENCE', 18157, %d, $json$%s$json$::jsonb, 'board-audit') "
    "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level=EXCLUDED.level, evidence=EXCLUDED.evidence, "
    "mission_id=EXCLUDED.mission_id, gate_state='pending' RETURNING id, gate_state;"
) % (LEVEL, payload)
(HERE / "insert_verdict.sql").write_text(sql)
user, pw, host, db = Path("/tmp/pgcred").read_text().split("\n")[:4]
env = dict(os.environ); env["PGPASSWORD"] = pw
r = subprocess.run(["psql", "-h", host.split(":")[0], "-U", user, "-d", db, "-tA", "-F", "|", "-f",
                    str(HERE / "insert_verdict.sql")], capture_output=True, text=True, env=env)
print("psql rc=", r.returncode, r.stdout.strip(), r.stderr.strip()[:400])
(HERE / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=1))