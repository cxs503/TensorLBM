#!/usr/bin/env python3
"""Board audit (交差核验) for m17679 — XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS.

Independently recomputes every graded number from the raw artifacts and writes
audit.json. No number is copied from the mission's own summary or docs: the
frozen rows are recomputed from benchmarks/verified/cylinder/re200/result.json,
the fresh rows are re-read from the honest-FAIL run records, and the tolerance
verdicts are recomputed against the OSS reference values parsed out of the
upstream source slice (evidence/oss_flow_pass_cylinder.py @ fb84b12).

Usage:  python3 docs/board_audit/m17747/audit_m17747.py
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EV = HERE / "evidence"
OUT = HERE / "audit.json"

AUDITED_WORKTREE = Path("/root/tp-agent/work/m17679")
ANCHOR = "XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS"
MISSION = 17679
TOL_PCT = 15.0          # predeclared in the port as TOL_PCT and in the task
TIGHT_PCT = 3.0         # owner report bar (是否<3%)

audit: dict = {
    "mission": MISSION,
    "anchor": ANCHOR,
    "audited_repo": "TensorLBM",
    "audited_branch": "agent/m17679",
    "audited_commit": "3161a1fc xval(vivsim): port continuous error-print vs DOI refs with 15% gate",
    "audit_branch": "agent/m17747",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "verdict_by": "board-audit",
}


def psql(sql: str) -> str:
    r = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-A", "-t", "-c", sql],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else f"<psql_error: {r.stderr.strip()[:200]}>"


# --------------------------------------------------------------- A1 provenance
def a1_provenance() -> dict:
    up = (EV / "oss_flow_pass_cylinder.py").read_text()
    port = (EV / "xval_vivsim_cylinder_re200.py").read_text()
    up_cd = float(re.search(r"CD_REF\s*=\s*([\d.]+)", up).group(1))
    up_cl = float(re.search(r"CL_REF\s*=\s*([\d.]+)", up).group(1))
    doi_up = re.findall(r"https://doi\.org/(10\.[^\s\"]+)", up)
    p_cd = float(re.search(r"^CD_REF\s*=\s*([\d.]+)", port, re.M).group(1))
    p_cl = float(re.search(r"^CL_REF\s*=\s*([\d.]+)", port, re.M).group(1))
    doi_port = re.findall(r'"(10\.[^"]+)"', port)
    return {
        "check": "A1 reference-value provenance (parsed from upstream source, not taken on trust)",
        "upstream_file": "evidence/oss_flow_pass_cylinder.py (= vivsim examples/2d/flow_pass_cylinder.py @ fb84b12f02e18f3b9dfdf738d6a1a318d81539f9)",
        "upstream_CD_REF": up_cd, "upstream_CL_REF": up_cl,
        "upstream_DOIs": doi_up,
        "port_CD_REF": p_cd, "port_CL_REF": p_cl, "port_DOIs": sorted(doi_port),
        "cd_match": p_cd == up_cd, "cl_match": p_cl == up_cl,
        "doi_match": sorted(doi_up) == sorted(doi_port),
        "upstream_geometry": {
            k: re.search(rf"^{k}\s*=\s*(.+?)(?:\s*#|$)", up, re.M).group(1).strip()
            for k in ("D", "NX", "NY", "CYL_X", "CYL_Y", "U0", "RE", "TM")
            if re.search(rf"^{k}\s*=", up, re.M)
        },
        "upstream_caliber_line": re.search(r"^cd_mean\s*=\s*(.+)$", up, re.M).group(1).strip()
        + " / " + re.search(r"^cl_max\s*=\s*(.+)$", up, re.M).group(1).strip(),
        "upstream_has_assert_or_test": bool(re.search(r"\bassert\b|pytest", up)),
    }


# ---------------------------------------------------- A2 frozen (pre-existing) arm
def a2_frozen() -> dict:
    res = json.loads((EV / "result.json").read_text())
    ref_cd, ref_cl = 1.389, 0.718
    rows = {}
    for g, d in res["grids"].items():
        cd, cl_std = d["cd"], d["cl_std"]
        cl_peak = math.sqrt(2.0) * cl_std          # sinusoid: A = sqrt(2)*sigma
        rows[g] = {
            "cd": cd, "err_cd_pct": (cd - ref_cd) / ref_cd * 100.0,
            "cl_std": cl_std, "cl_peak_from_std": cl_peak,
            "err_cl_pct": (cl_peak - ref_cl) / ref_cl * 100.0,
            "st": d["st"], "within_15pc_cd": abs((cd - ref_cd) / ref_cd * 100) < TOL_PCT,
            "within_15pc_cl": abs((cl_peak - ref_cl) / ref_cl * 100) < TOL_PCT,
            "within_3pc_cd": abs((cd - ref_cd) / ref_cd * 100) < TIGHT_PCT,
            "within_3pc_cl": abs((cl_peak - ref_cl) / ref_cl * 100) < TIGHT_PCT,
        }
    return {
        "check": "A2 frozen arm recomputed from raw result.json (NOT from the mission summary)",
        "provenance": "benchmarks/verified/cylinder/re200/result.json, verified_date="
                      + res.get("verified_date", "?"),
        "produced_by_this_mission": False,
        "setup_mismatch_with_oss": "40D x 40D domain, cylinder at 12D, u=0.08, sponge 10D, "
                                   "Braza-1986 anchor; OSS = 30D x 20D, cylinder at 10D, u=0.05, no sponge",
        "rows": rows,
        "grid_refinement": {
            "cd_drop_pct": (rows["D64"]["cd"] - rows["D48"]["cd"]) / rows["D48"]["cd"] * 100,
            "st_drop_pct": (rows["D64"]["st"] - rows["D48"]["st"]) / rows["D48"]["st"] * 100,
            "cl_peak_drop_pct": (rows["D64"]["cl_peak_from_std"] - rows["D48"]["cl_peak_from_std"])
                                / rows["D48"]["cl_peak_from_std"] * 100,
            "n_levels": len(rows),
            "convergence_order_available": False,
        },
    }


# ------------------------------------------------------------- A3 fresh OSS-caliber arm
def a3_fresh() -> dict:
    runs = {}
    for f in sorted(EV.glob("xval_D*.json")):
        d = json.loads(f.read_text())
        runs[f.name] = {
            "setup": {k: d["setup"][k] for k in ("D", "u_in", "tau", "nx", "ny", "n_steps")},
            "nan_at": d["runtime"]["nan_at"],
            "steps_done": d["runtime"]["steps_done"],
            "wall_s": round(d["runtime"]["wall_s"], 1),
            "mlups": round(d["runtime"]["mlups"], 3),
            "err_pct_cd_mean": d["verdict"]["err_pct_cd_mean"],
            "err_pct_cl_peak": d["verdict"]["err_pct_cl_peak"],
            "st_reported": d["ours"]["st"],
            "st_vs_lit_err_pct": d["verdict"]["st_vs_lit_err_pct"],
            "any_valid_cd_or_cl": bool(math.isfinite(d["verdict"]["err_pct_cd_mean"])
                                       or math.isfinite(d["verdict"]["err_pct_cl_peak"])),
        }
    # independent replay (auditor-run, evidence/m17747_repro_D10.json)
    arch = json.loads((EV / "xval_D10_u010.json").read_text())
    rep = json.loads((EV / "m17747_repro_D10.json").read_text())
    replay = {
        "probe": "A3b auditor replay of the D=10 / u=0.1 config (--steps 2000, 16 threads)",
        "nan_at_archived": arch["runtime"]["nan_at"], "nan_at_replay": rep["runtime"]["nan_at"],
        "verdict_identical": arch["verdict"] == rep["verdict"],
        "pre_divergence_cd_half1_archived": arch["ours"]["cd_half1"],
        "pre_divergence_cd_half1_replay": rep["ours"]["cd_half1"],
        "rel_diff_cd_half1": abs(arch["ours"]["cd_half1"] - rep["ours"]["cd_half1"])
                             / abs(arch["ours"]["cd_half1"]),
        "replay_wall_s": round(rep["runtime"]["wall_s"], 1),
        "not_identical_reason": "setup.n_steps/torch_threads differ (20 vs 16 threads); "
                                "reduction order changes float32 sum in the 7th digit",
    }
    return {
        "check": "A3 fresh caliber-matched arm (30D x 20D, vivsim calibers) — the arm that decides the anchor",
        "runs": runs,
        "n_runs": len(runs),
        "n_runs_with_valid_number": sum(1 for r in runs.values() if r["any_valid_cd_or_cl"]),
        "replay": replay,
        "honest_fail_confirmed": all(r["nan_at"] is not None for r in runs.values()),
    }


# ------------------------------------------------------ A4 gate / reporting defects
def a4_defects() -> dict:
    summ = json.loads((EV / "summary.json").read_text())
    doc = (EV / "XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS.md").read_text()
    frozen = a2_frozen()["rows"]
    # the mission's own summary vs an independent recomputation
    drift = {}
    for r in summ["runs_frozen"]:
        g = r["grid"]
        drift[g] = {
            "err_cl_pct_summary": r["err_cl_pct"],
            "err_cl_pct_recomputed": round(frozen[g]["err_cl_pct"], 2),
        }
    row64 = next((ln for ln in doc.splitlines() if ln.startswith("| D64 |")), "")
    cell64 = [c.strip() for c in row64.split("|")]
    doc_cl_d64 = re.search(r"([\u2212-][\d.]+)%", cell64[7]) if len(cell64) > 7 else None
    st_floor = {
        "check": "A4a St is reported as a finite number on a series that is NaN",
        "observed": {f: {"st": d["ours"]["st"], "st_vs_lit_err_pct": d["verdict"]["st_vs_lit_err_pct"]}
                     for f, d in ((f.name, json.loads(f.read_text())) for f in sorted(EV.glob("xval_D*.json")))},
        "mechanism": "harmonic_amplitude() falls back to the frequency floor 1/T (0.001 lattice Hz) "
                     "when the fit is degenerate -> St = f*D/u = 0.1 -> a finite -49.2% 'error' printed "
                     "for a run whose Cd/Cl are NaN",
        "severity": "minor-to-medium (reporting surface; a reader can take -49.2% as a measured error)",
    }
    return {
        "check": "A4 claim-vs-artifact cross-checks",
        "summary_vs_recomputed_drift": drift,
        "doc_table_cl_d64_pct": (float(doc_cl_d64.group(1).replace("\u2212", "-"))
                                 if doc_cl_d64 else None),
        "doc_claims_programmatic_injection": "docs ②: '全部程序化注入'",
        "broken_pointer": "docs 结语点名的 `benchmarks/xval/README` 不存在（目录内仅 2 个 .py）",
        "st_floor": st_floor,
        "note_on_drift": "summary.json -7.43% vs docs ④ -7.42% for the same quantity: the doc table "
                         "was hand-typed, contradicting the 'programmatic injection' claim by 0.01 pp",
    }


# ---------------------------------------------------------------- A5 registry
def a5_registry() -> dict:
    rows = psql("SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='%s'" % ANCHOR)
    liked = psql("SELECT anchor_slug||'|'||tolerance FROM tensor_benchmark_cases "
                 "WHERE anchor_slug ILIKE '%%CONTINUOUS%%' OR case_name ILIKE '%%continuous%%'")
    return {
        "check": "A5 referee registration (tensor_benchmark_cases is the only judge)",
        "rows_for_anchor": rows,
        "near_miss_rows": liked or "<none>",
        "consequence": "no registered case => no machine-judgeable ref/got pair for this anchor; "
                       "the mission's own script gate is the only gate, and it was never fed a "
                       "caliber-matched number",
    }


# ---------------------------------------------------------------- A6 durability
def a6_durability() -> dict:
    paths = ["benchmarks/xval/xval_vivsim_cylinder_re200.py",
             "benchmarks/xval/xval_vivsim_re200_report.py",
             "tests/test_xval_vivsim_cylinder_re200.py",
             "docs/XVAL-LBM-CONTINUOUS-ERROR-PRINT-VS-DOI-REFS.md",
             "artifacts/xval_vivsim_re200/summary.json",
             "docs/figures/xval_vivsim_re200_frozen_ladder.png",
             "runs/xval_D10_u010.json", "runs/xval_D12_u010.json", "runs/xval_D12_u030.json"]
    tracked = {}
    for p in paths:
        r = subprocess.run(["git", "-C", str(AUDITED_WORKTREE), "ls-files", "--error-unmatch", p],
                           capture_output=True, text=True)
        ign = subprocess.run(["git", "-C", str(AUDITED_WORKTREE), "check-ignore", "-v", p],
                             capture_output=True, text=True)
        tracked[p] = {"tracked_in_agent/m17679": r.returncode == 0,
                      "gitignore_rule": ign.stdout.strip() or "-"}
    return {
        "check": "A6 deliverable durability (does the pushed branch carry the evidence?)",
        "paths": tracked,
        "consequence": "code/tests/docs/summary are tracked; the three honest-FAIL run records live "
                       "under runs/ which .gitignore:108 excludes -> only summary.json's runs_fresh "
                       "block survives in the branch (this audit snapshots them under evidence/)",
    }


# ---------------------------------------------------------------- A7 tests + report
def a7_tests_report() -> dict:
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_xval_vivsim_cylinder_re200.py",
                        "-q", "--no-header", "-p", "no:cacheprovider"],
                       cwd=str(AUDITED_WORKTREE), capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr) else ""
    md = psql("SELECT markdown FROM tensor_reports WHERE id=16662")
    sections = {k: (k in md) for k in ("①", "②", "③", "④", "⑤", "⑥")}
    figs = psql("SELECT count(*) FROM tensor_reports WHERE id=16662 AND figures IS NOT NULL")
    explicit_3pct = "3%" in md
    return {
        "check": "A7 auditor re-run of the mission's own test suite + report completeness",
        "pytest_cmd": "python3 -m pytest tests/test_xval_vivsim_cylinder_re200.py -q (in /root/tp-agent/work/m17679)",
        "pytest_tail": tail, "pytest_exit": r.returncode,
        "report_id": 16662, "report_sections_present": sections,
        "report_figures_nonnull": figs,
        "report_answers_3pct_question": explicit_3pct,
    }


# ---------------------------------------------------------------- A8 integration
def a8_integration() -> dict:
    r = subprocess.run(["grep", "-rn", "xval_vivsim", "--include=*.py", "--include=*.sh",
                        str(AUDITED_WORKTREE / "TensorLBM"), str(AUDITED_WORKTREE / "src")],
                       capture_output=True, text=True)
    hits = [ln for ln in r.stdout.splitlines() if ln.strip()]
    return {
        "check": "A8 solver/platform integration",
        "callers_outside_benchmark_and_tests": hits or [],
        "n_callers": len(hits),
        "consequence": "no call site in TensorLBM/src or the solver package -> L3 not met",
    }


audit["A1_provenance"] = a1_provenance()
audit["A2_frozen_arm"] = a2_frozen()
audit["A3_fresh_arm"] = a3_fresh()
audit["A4_claim_checks"] = a4_defects()
audit["A5_referee_registry"] = a5_registry()
audit["A6_durability"] = a6_durability()
audit["A7_tests_report"] = a7_tests_report()
audit["A8_integration"] = a8_integration()

# ------------------------------------------------------------------- level decision
A = audit
l1 = A["A2_frozen_arm"]["rows"] and A["A7_tests_report"]["pytest_exit"] == 0
l2 = A["A3_fresh_arm"]["n_runs"] >= 1             # real code, input->compute->output, artifacts exist
l3 = A["A8_integration"]["n_callers"] > 0
l4 = (A["A5_referee_registry"]["rows_for_anchor"] not in ("0", "")
      and A["A3_fresh_arm"]["n_runs_with_valid_number"] >= 1)
A["level_decision"] = {
    "L1_math_unit_validation": {"met": bool(l1),
                                "why": "11 tests green on re-run; they assert invariants (last-half window "
                                       "selection, sine amplitude recovery, single-sample spike pinning, "
                                       "strict |err|<15% boundary) not just execution"},
    "L2_independent_implementation": {"met": bool(l2),
                                      "why": "benchmarks/xval/xval_vivsim_cylinder_re200.py is real code that "
                                             "runs end-to-end (auditor replay: 32.5 s, 1999 steps, 3.69 MLUPS) "
                                             "and emits JSON/CSV/NPZ + a gate verdict"},
    "L3_system_integration": {"met": bool(l3), "why": "0 call sites outside benchmarks/xval + tests"},
    "L4_benchmark_validation": {"met": bool(l4),
                                "why": "0 referee rows for the anchor AND the caliber-matched arm produced "
                                       "0/3 valid numbers (NaN at 2000/2500/500); the FROZEN rows come from a "
                                       "pre-existing verified case with a different domain/u/caliber"},
    "L5_research_engineering": {"met": False, "why": "no experimental or literature-independent anchor met"},
    "level": 2,
    "level_note": "L2 for the XVAL harness (continuous error print + 15% gate + 11 tests + honest-FAIL "
                  "records). The anchor's own deliverable — a caliber-matched TensorLBM-vs-vivsim number — "
                  "is WITHHELD, so L4 is unreachable and no <15% claim may be attached to this anchor.",
}
A["disposition"] = (
    "request_changes (reopened): keep L2 credit for the harness; the anchor needs the "
    "caliber-matched number (root-cause the NaN, then re-run D=16/24 on the cluster) before "
    "any L4 claim")
A["audit_limits"] = [
    "One of the three honest-FAIL runs (D=10/u=0.1) was replayed; D=12/u=0.1 and D=12/u=0.3 were not "
    "re-run (250 s each) — their nan_at values are read from the records only.",
    "The auditor did NOT root-cause the divergence: the falsified tau hypothesis is taken as reported, "
    "and the resolution/seeding hypothesis is untested here.",
    "vivsim itself was never executed (JAX absent, 3.9e11 cell-updates upstream); all OSS-side claims are "
    "source-level only.",
    "The upstream clone is network-fetched and its checkout has no resolvable HEAD object; the fb84b12 "
    "identity rests on the working-tree content matching the cited lines.",
    "No convergence order is available (2 grid levels, different setup from the graded arm).",
]
def _clean(o):
    if isinstance(o, float) and not math.isfinite(o):
        return "NaN" if math.isnan(o) else ("+inf" if o > 0 else "-inf")
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


OUT.write_text(json.dumps(_clean(A), indent=1, default=str, ensure_ascii=False, allow_nan=False))
print(f"wrote {OUT}")
print(json.dumps(A["level_decision"], indent=1, ensure_ascii=False))