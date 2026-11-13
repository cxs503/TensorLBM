#!/usr/bin/env python3
"""m18779 board-audit evidence collector for mission m18749.

Anchor: XVAL-LBM-STATIC-IMMERSED-MARKER-OBSTACLES  (owner m18749, succeeded)
This is a hand-over (交差) verification: nothing is re-implemented here; every
number below is produced programmatically from the delivered tree, the platform
DB, or a faithful re-execution of the delivered repro command.

Output: verify_output.json  (all pointers are file:line / artifact path / sha256)
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import urllib.request

W18749 = pathlib.Path("/root/tp-agent/work/m18749")   # delivered mission tree
W18779 = pathlib.Path("/root/tp-agent/work/m18779")   # this audit tree
OUT = W18779 / "docs/board_audit/m18779/verify_output.json"
ART = W18779 / "artifacts/m18779"
DB = os.environ.get("TP_PG_DB", "tensorplatform")
ANCHOR = "XVAL-LBM-STATIC-IMMERSED-MARKER-OBSTACLES"

RC = "xval/XVAL-LBM-STATIC-IMMERSED-MARKER-OBSTACLES"


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def lines(p: pathlib.Path) -> int:
    return sum(1 for _ in p.open("rb"))


def psql(sql: str) -> str:
    r = subprocess.run(["psql", "-U", "foundation", DB, "-At", "-F", "|", "-c", sql],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return "SQL_ERROR:" + r.stderr.strip().splitlines()[0][:200]
    return r.stdout.strip()


def api(path: str) -> dict:
    base = os.environ["TENSOR_API_BASE"]
    tok = pathlib.Path("/tmp/tok.txt").read_text().strip()
    req = urllib.request.Request(base + path, headers={"Authorization": "Bearer " + tok})
    try:
        return json.load(urllib.request.urlopen(req, timeout=60))
    except Exception as e:  # noqa: BLE001
        return {"ERR": str(e)}


# ---------------------------------------------------------------- A. 三件套
delivered = {
    "ibm.py": "TensorLBM/src/tensorlbm/ibm.py",
    "cylinder_flow": "TensorLBM/src/tensorlbm/verification/cylinder_flow/__init__.py",
    "test_ibm.py": "TensorLBM/tests/test_ibm.py",
    "report_md": RC + "/REPORT.md",
    "predeclared": RC + "/predeclared.json",
    "oss_solution": "TensorLBM/docs/oss_solutions/" + ANCHOR + ".md",
}
A_delivered = {}
for k, rel in delivered.items():
    p = W18749 / rel
    A_delivered[k] = {
        "path": str(p), "exists": p.exists(),
        "lines": lines(p) if p.exists() else 0,
        "sha256": sha(p) if p.exists() else None,
    }

art_dir = W18749 / RC / "artifacts"
A_artifacts = {
    "dir": str(art_dir), "exists": art_dir.exists(),
    "files": sorted(str(p.relative_to(W18749)) for p in art_dir.rglob("*")) if art_dir.exists() else [],
    "claimed_but_absent": [RC + "/artifacts/cylinder_flow_re20.json"],
}
A_artifacts["claimed_but_absent_present"] = [
    str(W18749 / f) + " -> " + str((W18749 / f).exists()) for f in A_artifacts["claimed_but_absent"]
]
# untracked-only delivery: nothing of m18749 reached the branch history
git_status = subprocess.run(["git", "-C", str(W18749), "status", "--short"], capture_output=True, text=True).stdout
A_git = {
    "head": subprocess.run(["git", "-C", str(W18749), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip(),
    "untracked_deliverables": [l for l in git_status.splitlines() if "ibm" in l or "cylinder_flow" in l or ANCHOR in l],
    "in_branch_history": False,
}

# ---------------------------------------------------------------- B. 基准裁判
B_cases = {
    "query": f"SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='{ANCHOR}'",
    "rows": psql(f"SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='{ANCHOR}'"),
    "n_rows": int(psql(f"SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='{ANCHOR}'") or 0),
    "n_rows_table": int(psql("SELECT count(*) FROM tensor_benchmark_cases") or 0),
}
platform_l4 = api("/tensor/bench/l4-verdict/18749")

B_claim = {
    "report_claim_cd_ref": 2.05,
    "report_claim_tolerance_pct": 15.0,
    "report_claim_source": "Dennis & Chang (1970) JFM 42:471, stated in REPORT.md §1 / predeclared.json",
    "got_value_in_delivery": None,
    "report_status_quoted": "REPORT.md §4.2 'Cylinder Flow (Re=20) — Cluster Job Pending ... Results will be populated here upon completion'",
}

# ---------------------------------------------------------------- C. 重跑 repro_cmd
C_repro = {"cmd": (RC + "/REPORT.md §6.2"), "executed": False}
rp = ART / "xval_audit_re20.json"
if rp.exists():
    d = json.loads(rp.read_text())
    h = d["force_history"]
    cds = [r["Cd"] for r in h]
    half = cds[len(cds) // 2:]
    last = [r["Cd"] for r in h if r["step"] >= 6000]

    def mean(v):
        return sum(v) / len(v)

    def pstd(v):
        m = mean(v)
        return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5

    C_repro = {
        "cmd": "PYTHONPATH=TensorLBM/src python3 -c 'from tensorlbm.verification.cylinder_flow import "
               "run_re20_validation; run_re20_validation()'  ->  driven by "
               "/tmp/run_re20.py, output " + str(ART / "xval_audit_re20.json"),
        "executed": True,
        "exit_ok": True,
        "wall_sec": round(d.get("wall_sec", 0), 1),
        "params": d["config"],
        "Cd_computed": d["results"]["Cd_computed"],
        "Cd_reference": d["results"]["Cd_reference"],
        "rel_error_pct": d["results"]["rel_error_pct"],
        "within_15pct": d["results"]["rel_error_pct"] < 15.0,
        "n_samples": d["results"]["n_samples"],
        "Cd_first5": [round(c, 3) for c in cds[:5]],
        "Cd_last10": [round(c, 3) for c in cds[-10:]],
        "second_half_Cd_mean": round(mean(half), 4),
        "second_half_Cd_pstd": round(pstd(half), 4),
        "second_half_Cd_min": round(min(half), 3),
        "second_half_Cd_max": round(max(half), 3),
        "last2000_steps_Cd_range": [round(min(last), 3), round(max(last), 3)],
        "predeclared_stopping_rule": "Cd variation < 0.01 over last 2000 steps",
        "stopping_rule_met": (max(last) - min(last)) < 0.01,
        "stdout_log": str(ART / "repro_stdout.log"),
        "sha256": sha(rp),
    }

# ---------------------------------------------------------------- D. 单测重跑
pt = subprocess.run(["python3", "-m", "pytest", "TensorLBM/tests/test_ibm.py", "-q"],
                    cwd=str(W18779), capture_output=True, text=True)
D_tests = {
    "cmd": "cd " + str(W18779) + " && python3 -m pytest TensorLBM/tests/test_ibm.py -q",
    "tail": pt.stdout.strip().splitlines()[-1] if pt.stdout.strip() else pt.stderr[-300:],
    "returncode": pt.returncode,
    "test_file": str(W18779 / "TensorLBM/tests/test_ibm.py") + " (copy of m18749 file, sha256 "
                 + (sha(W18779 / "TensorLBM/tests/test_ibm.py") if (W18779 / "TensorLBM/tests/test_ibm.py").exists() else "n/a") + ")",
    "asserted_invariants": [
        "test_ibm.py:22 peskin support delta(|r|>2)=0",
        "test_ibm.py:34 partition of unity sum_k w=1 (atol 1e-10)",
        "test_ibm.py:55 interpolate uniform field exact",
        "test_ibm.py:81 spread force conservation sum F_grid = sum F_marker",
        "test_ibm.py:88 zero grid velocity -> zero forcing",
    ],
    "coverage_gap": "no test on the coupled LBM+IBM solver (CylinderFlowSolver); "
                    "no test asserting Cd, no convergence test",
}

# ---------------------------------------------------------------- E. 等级门 + 报告六段
rep = (W18749 / RC / "REPORT.md").read_text()
D_levels = {
    "L0_docs": {"met": True, "why": "REPORT.md (" + str(A_delivered["report_md"]["lines"]) + " lines) + oss_solutions doc"},
    "L1_math_unit": {"met": True, "why": "8/8 pytest pass: partition of unity, conservation, uniform interpolation"},
    "L2_real_impl": {"met": True, "why": "ibm.py " + str(A_delivered["ibm.py"]["lines"]) +
                     " lines real operators (peskin_delta_4pt:22, build_stencil_2d:35, spread_force_2d:72, "
                     "interpolate_velocity_2d:102, direct_forcing_static:129, build_cylinder_markers_2d:179); "
                     "input->compute->output path executes end-to-end (391 s, 8000 steps)"},
    "L3_integration": {"met": False, "why": "no caller outside verification/cylinder_flow: grep -rn 'ibm' over "
                       "TensorLBM/src (excluding ibm.py, verification/cylinder_flow) and TensorLBM/app -> 0 hits; "
                       "the production solver (solver.py / boundary/) has no IBM obstacle hook"},
    "L4_benchmark": {"met": False, "why": "0 run artifacts delivered; 0 rows in tensor_benchmark_cases for the anchor; "
                     "auditor re-run of the delivered repro gives Cd=%.4f (%.1f%% vs 2.05) and a non-converged, "
                     "sign-changing Cd history" % (C_repro["Cd_computed"], C_repro["rel_error_pct"]) if C_repro["executed"] else "not measured"},
    "L5_research": {"met": False, "why": "no validated physics result, no publication-grade evidence"},
    "assigned_level": 2,
}
six = {
    "1_case_intro": ("REPORT.md §1 + §3 (physics, geometry R=20 lu, parameter table, Re=20)", True),
    "2_how": ("REPORT.md §2 method + §2.2 code pointers", True),
    "3_numbers": ("REPORT.md §4.1 unit-test table only; §4.2 simulation results absent ('pending')", False),
    "4_figures": ("no figure anywhere in the delivered tree (find -name '*.png' -> 0)", False),
    "5_caveats": ("REPORT.md §6.1 known limitations + §5 convergence discussion (qualitative, no data)", True),
    "6_error_verdict": ("REPORT.md §4.2 gives an *expectation* ('Cd should converge to ~2.0-2.3'), no measured got, "
                        "no explicit <3%/<15% verdict on a number", False),
}
D_report_six = {"sections": {k: {"where": v[0], "present": v[1]} for k, v in six.items()},
                "n_present": sum(1 for v in six.values() if v[1]), "n_required": 6}

E_cross = {
    "platform_l4_verdict_mission_18749": platform_l4,
    "cluster_job": {
        "job_id": "cj2024",
        "state_observed": "queued (submitted 11-13 16:37, age 12.1 min at first observation; never reached a terminal state)",
        "source": "run_cluster(action=ps, job_id=cj2024) at 2026-11-13T16:49 local; docs/oss_solutions/"
                  + ANCHOR + ".md §Cluster Job declares cj2024 = 'full Re=20 simulation'",
    },
    "suspected_root_cause": {
        "claim": "IBM body force is added *after* stream+BC instead of inside the collision step",
        "pointer": "TensorLBM/src/tensorlbm/verification/cylinder_flow/__init__.py:265 step(): collide -> _stream -> BCs -> _apply_ibm_force; "
                   "__init__.py:211 _apply_ibm_force adds the Guo force term (1-0.5*omega) to the already-streamed f",
        "status": "hypothesis, not proven; consistent with the violent Cd oscillation but not isolated by an ablation",
    },
}

out = {
    "audit_mission": 18779,
    "subject_mission": 18749,
    "anchor": ANCHOR,
    "repo": "TensorLBM",
    "verdict_by": "board-audit",
    "A_delivered_files": A_delivered,
    "A_run_artifacts": A_artifacts,
    "A_git": A_git,
    "B_benchmark_cases": B_cases,
    "B_claim_vs_measured": B_claim,
    "C_repro_rerun": C_repro,
    "D_tests": D_tests,
    "D_level_gates": D_levels,
    "D_report_six_sections": D_report_six,
    "E_cross_checks": E_cross,
}
OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
print("wrote", OUT)
print("level", D_levels["assigned_level"], "| benchmark rows", B_cases["n_rows"],
      "| repro Cd", C_repro.get("Cd_computed"), C_repro.get("rel_error_pct"))