#!/usr/bin/env python3
"""m18213 结单：生成证据图 + INSERT tensor_maturity_verdicts + 发布平台报告。"""
import base64
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
W = Path("/root/tp-agent/work/m18213")
BASE = "http://127.0.0.1:8010/api/v1"
v = json.loads((HERE / "verify.json").read_text())
nc = json.loads((HERE / "nc_single_rung.json").read_text())
MISSION = 18213
SUBJECT = 17990

# ---------------- 图：① 选型级 |Δ| vs 15% 判据  ② TRT τ₋ vs 分辨率 ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
kpis = ["cd_mean_last50", "cl_amp", "st_fft"]
for i, arm in enumerate(["A2", "A4"]):
    vals = [abs(v["recalc_delta_vs_A1"][arm][k]["delta_pct"]) for k in kpis]
    ax1.bar([x + i * 0.36 for x in range(len(kpis))], vals, width=0.36, label="A%s vs A1" % arm)
ax1.axhline(15, color="crimson", ls="--", lw=1.2, label="owner gate 15%")
for x in range(len(kpis)):
    ax1.text(x, 50, "", ha="center")
ax1.set_xticks(range(len(kpis)))
ax1.set_xticklabels(["Cd (last 50%)", "Cl_amp", "St"], fontsize=9)
ax1.set_ylabel("|Δ| vs A1 at D=10  [%]")
ax1.set_title("(a) selection-level gap (auditor recomputed from runs/*/result.json)", fontsize=9)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3, axis="y")

labs = ["D=10\n(τ+=0.5075)", "D=14\n(τ+=0.5105)"]
tm = [v["trt_tau_minus"]["D10"]["tau_minus_artifact"], v["trt_tau_minus"]["D14"]["tau_minus_artifact"]]
steps = [v["trt_tau_minus"]["D10"]["fail_step"], v["trt_tau_minus"]["D14"]["fail_step"]]
ax2.plot(labs, tm, "o-", color="darkorange")
for x, (t, s) in enumerate(zip(tm, steps)):
    ax2.annotate("τ₋=%.2f\nblowup @step %d" % (t, s), (x, t), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=8)
ax2.set_ylabel("τ₋ = 0.5 + Λ/(τ₊−0.5), Λ=3/16")
ax2.set_title("(b) literal OSS TRT magic-Λ fails at both resolutions", fontsize=9)
ax2.grid(alpha=0.3)

fig.suptitle("m17990 cross-audit (m18213): recomputed evidence — no registered benchmark case (0 rows)", fontsize=9)
fig.tight_layout()
png = HERE / "fig_audit_m17990.png"
fig.savefig(png, dpi=130)

# ---------------- evidence JSON ----------------
evidence = {
    "subject_mission": SUBJECT,
    "audit_mission": MISSION,
    "audit_doc": "docs/board_audit/m18213/AUDIT_m17990.md",
    "audit_scripts": ["audit/m18213/verify_m17990.py", "audit/m18213/nc_single_rung.py"],
    "raw_outputs": ["audit/m18213/verify.json", "audit/m18213/nc_single_rung.json", "audit/m18213/fig_audit_m17990.png"],
    "declared_level_by_subject": None,
    "impl_files": [
        "xval/benchmark_method_selection/run_arm.py (drives tensorlbm.solver: collide_bgk/collide_mrt/collide_trt/stream, m17990 branch)",
        "xval/benchmark_method_selection/analyze.py:280-296 (verdict block)",
        "xval/benchmark_method_selection/analyze.py:289-290 (defect F1: H_gap_survives_refinement)",
        "xval/benchmark_method_selection/predeclared_v3.json (pre-registration, mtime 10:56:32 < A3_D14 fail.json 10:58:11)",
        "xval/benchmark_method_selection/run_ladder.sh, run_ladder_cluster.sh (repro entry)",
        "xval/benchmark_method_selection/publish_report.py",
    ],
    "entry": "python3 xval/benchmark_method_selection/run_arm.py --arm A1 --D 10 ; python3 xval/benchmark_method_selection/analyze.py (rewrites summary.json/table.csv/figures)",
    "run_artifacts": {
        "runs/A1_D10/result.json": "Cd_last50=1.6132947378116644, Cl_amp=0.6579248352005606, St=0.16899580031695963, 21259 steps",
        "runs/A2_D10/result.json": "Cd_last50=1.4660336012540012",
        "runs/A4_D10/result.json": "Cd_last50=1.642771191938185",
        "runs/A3_D10/fail.json": "blowup @step 600/21259, tau_minus=25.500000000000163",
        "runs/A3_D14/fail.json": "blowup @step 1000/29763, tau_minus=18.357142857142936",
        "summary.json": "schema xval-method-selection/v3, auditor rerun reproduces it identically",
        "oss_crosscheck.json": "k1 rel-L2 1.1757884876259976e-07, k2 rel-L2 2.709855973692391e-08, 5 OSS source sha256 pinned @573a01c",
    },
    "tests": "tests/test_xval_benchmark_method_selection.py -> 6 passed in 54.67s (auditor rerun); invariants asserted (arms table, drag sign/O(1), <=50% arm disagreement, summary/ladder self-consistency); does NOT cover verdict.H_gap_survives_refinement (defect F5)",
    "integration_callers": [],
    "integration_note": "grep -rl 'run_arm|benchmark_method_selection' -> only tests/ and xval/benchmark_method_selection/ itself; no production path calls the harness. run_arm.py imports tensorlbm.solver kernels (library-driven experiment) but no main entry delegates to it => L3 not granted.",
    "benchmark": {
        "registered_case": "SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-METHOD-SELECTION-FIRST-EVIDENCE' -> 0 rows",
        "platform_rows": "SELECT count(*) FROM tensorbench_results WHERE mission_id=17990 -> 0 rows",
        "l4_verdict_api": "GET /tensor/bench/l4-verdict/17990 -> CLAIMED_NOT_PROVEN, l4_benchmark_pass=false, reason='no tensorbench_results row'",
        "ref_vs_got_internal": "A1 vs A2 at D=10: cl_amp -46.56648513639526% (auditor recomputed, bit-identical to summary.verdict.largest_abs_delta_pct); cd -9.13%; St +4.52%",
        "external": "Norberg/Sohankar Re=200 tables WITHHELD (report §6); absolute accuracy of A1 Cd=1.613 unadjudicated",
        "verdict_vs_gate": "internal contrast only: the OSS selection misses the owner 15% band on cl_amp -> H1 rejected; but nothing is measured against a registered/platform referee",
    },
    "repro_cmd": "cd xval/benchmark_method_selection && python3 run_arm.py --arm A1 --D 10 (21259 steps, 72.8 s recorded) && python3 analyze.py; tests: python3 -m pytest tests/test_xval_benchmark_method_selection.py -q (6 passed, 54.67 s)",
    "external_anchor": "Maarten-vd-Sande/lbm @573a01c4d2e6321f3634ff271adca4ce91872e6f — executed-source kernel cross-check (collision 1.18e-07, streaming 2.71e-08 rel-L2), 5 source files sha256-pinned in oss_crosscheck.json; upstream fidelity verified by execution, not by re-reading",
    "level_rationale": "L2 独立功能实现：真代码真计算（harness 驱动本仓 tensorlbm.solver 内核，D=10 21259 步，机器可读工件，KPI 由审计独立重算一致，analyze.py 聚合可位级复现，6 个不变量测试全过）。不授 L3：无生产调用方（grep 0 命中，主求解器路径未触）。不授 L4：平台判据 0 行 + tensorbench_results 0 行 + l4-verdict=CLAIMED_NOT_PROVEN，收敛梯级 honest not_run（D=14 仅 A3 失败臂，D=20 全无），外部锚 WITHHELD ⇒ 完整算例+基准误差分析+可复现三件套不同时成立。不授 L5：交付未合并主线。",
    "findings": [
        {"id": "F1", "severity": "high", "what": "verdict.H_gap_survives_refinement=true with a SINGLE rung (gap_ladder.A2.rungs=[10])", "where": "analyze.py:289-290 + analyze.py:150", "repro": "audit/m18213/nc_single_rung.py (D=14/20 artifacts removed, analyze.py unmodified) -> published rungs=[10], flag=True, corrected semantics=False", "impact": "machine-readable field contradicts report §5 (ladder not_run); downstream gate would read 'gap survives refinement' as proven", "fix": "require len(e['rungs'])>=2 and out-of-band at the highest rung; add single-rung negative-control test"},
        {"id": "F2", "severity": "medium", "what": "deliverable not merged to main", "detail": "5f81352b only on agent/m17990; main lacks predeclared_v3.json/run_ladder*.sh/figures/new tests+docs, summary.json still v2"},
        {"id": "F3", "severity": "medium", "what": "no registered benchmark case for the anchor (0 rows) => L4 unreachable for every round of this anchor"},
        {"id": "F4", "severity": "low", "what": "ladder not_run (honest, disclosed in report §5/§6) — not counted as a defect, caps the level at L2"},
        {"id": "F5", "severity": "low", "what": "tests pass but do not cover the F1 verdict semantics"},
    ],
    "overstated_claims": [
        "summary.verdict.H_gap_survives_refinement=true — false (single rung; F1)",
        "'双分辨率' in the report summary applies only to the A3 failure probe; the KPI ladder remains single-resolution",
    ],
    "checks_run": [
        "pytest tests/test_xval_benchmark_method_selection.py -q -> 6 passed in 54.67s (subject worktree)",
        "analyze.py rerun on a /tmp copy -> regenerated summary.json equals the published one (key-by-key, no numeric drift)",
        "independent KPI delta recompute from runs/*/result.json: A2 cl_amp -46.56648513639526%, cd -9.127974765318703%, St +4.516952960575163%; A4 cl_amp -15.396855863488955%",
        "independent TRT tau_minus recompute 0.5+Λ/(τ₊-0.5): D=10 -> 25.500000000000163, D=14 -> 18.357142857142936 (both match artifacts), both status=blowup",
        "DB referee: tensor_benchmark_cases (anchor) = 0 rows; tensorbench_results (mission 17990) = 0 rows; /tensor/bench/l4-verdict/17990 = CLAIMED_NOT_PROVEN",
        "ladder completeness scan: A1_D14/A4_D14 empty dirs, A2_D14/A5_D14 state.npz only, *_D20 absent",
        "merge state: git branch --contains 5f81352b -> agent/m17990 only; main summary schema = v2",
        "negative control: single-resolution world -> shipped flag True, corrected flag False",
    ],
    "checks_not_run": [
        "full ladder arms A1/A2/A4 at D=14 and D=20 (29763+ steps each): not_run by the auditor",
        "external literature tables (Norberg 2003 / Sohankar 1998, Re=200): not_run, no network; report itself marks WITHHELD",
        "OSS end-to-end full case (121 h/arm): not_run and declared infeasible in this ecosystem (loongarch64, no numba wheel)",
        "re-verification of cluster job cj1744 history/state: not_run",
    ],
    "disposition": "request changes: (1) fix F1 + add single-rung negative-control test; (2) merge agent/m17990 into main; (3) register a tensor_benchmark_cases row for this anchor (L4 is otherwise structurally unreachable); (4) limit the 'dual-resolution' wording. Honest not_run (F4) is NOT a repair requirement.",
    "verdict_by": "board-audit",
    "level": 2,
}
ev_json = json.dumps(evidence, ensure_ascii=False)
sql = ("INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
       "VALUES ('TensorLBM', 'm17990 XVAL-LBM-METHOD-SELECTION-FIRST-EVIDENCE', %d, 2, $j$%s$j$, 'board-audit');"
       % (MISSION, ev_json))
(HERE / "insert_verdict.sql").write_text(sql)

# ---------------- 报告 ----------------
md = (HERE.parent.parent / "docs/board_audit/m18213/AUDIT_m17990.md").read_text(encoding="utf-8")
b64 = base64.b64encode(png.read_bytes()).decode()
report = {
    "title": "m17990 交差审计（m18213）：XVAL-LBM 方法选型第一证据 = L2，且判定字段 H_gap_survives_refinement 单分辨率即真",
    "summary": "基准裁判 0 行（anchor 无 tensor_benchmark_cases 登记、tensorbench_results 0 行、l4-verdict=CLAIMED_NOT_PROVEN）⇒ L4 结构不可达；真代码真计算+聚合位级可复现+6 测试通过 ⇒ L2；审计负控证明 analyze.py:289-290 在只有 1 个分辨率时仍输出 H_gap_survives_refinement=true。",
    "scenario_key": "tensorlbm.xval.method_selection",
    "mission_id": MISSION,
    "tags": ["TensorLBM", "board-audit", "XVAL", "method-selection", "verdict-L2", "negative-control"],
    "markdown": md,
    "figures": [{"name": "fig_audit_m17990.png",
                 "caption": "左：审计独立重算的 D=10 选型级 |Δ|（A2/A4 vs A1，红线=owner 15% 判据）；右：字面移植 OSS TRT 磁参数 Λ=3/16 的 τ₋ 与发散步数（D=10 τ₋=25.50@600、D=14 τ₋=18.36@1000）。两图均自 audit/m18213/verify.json 程序化生成。",
                 "png_base64": b64}],
}
(HERE / "report.json").write_text(json.dumps(report, ensure_ascii=False))

# ---------------- 执行：INSERT + POST ----------------
ins = subprocess.run(["su", "postgres", "-c",
                      "psql -d tensorplatform -v ON_ERROR_STOP=1 -f %s" % (HERE / "insert_verdict.sql")],
                     capture_output=True, text=True)
print("INSERT rc=", ins.returncode, ins.stdout.strip()[:200], ins.stderr.strip()[:300])

token = json.load(open("/tmp/tok.json"))["access_token"]
req = urllib.request.Request(BASE + "/tensor/reports",
                            data=json.dumps(report).encode(),
                            headers={"Content-Type": "application/json",
                                     "Authorization": "Bearer " + token})
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        print("REPORT", r.status, r.read().decode()[:400])
except Exception as e:  # noqa
    print("REPORT ERROR", e)
sys.exit(0 if ins.returncode == 0 else 1)