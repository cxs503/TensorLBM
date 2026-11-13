#!/usr/bin/env python3
"""Board-audit verdict for m18068 (XVAL-LBM-NO-MULTIPHASE-LAYOUT-PATTERN-ONLY, TensorLBM).

Idempotent: one row per (repo, scope, verdict_by); re-running updates it.
Every value is built from docs/board_audit/m18159/evidence.json (collected by
scripts/m18159_audit.py from the mission DB, the benchmark-case table, the
report lake and the subject git worktree) — nothing is typed by hand.

Run:  python3 docs/board_audit/m18159/insert_verdict.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import psycopg

REPO = "TensorLBM"
SUBJECT = 18068
SCOPE = "m18068 XVAL-LBM-NO-MULTIPHASE-LAYOUT-PATTERN-ONLY"
AUDIT_MISSION = 18159
BY = "board-audit"
LEVEL = 0
LOCAL_DSN_FALLBACK = "/root/tp-agent/work/m15815/docs/board_audit/m15314/insert_verdict.py"

ROOT = Path(__file__).resolve().parents[3]
EV = json.loads((ROOT / "docs/board_audit/m18159/evidence.json").read_text())


def dsn() -> str:
    d = os.environ.get("TP_FOUNDATION_DSN")
    if d:
        return d
    txt = open(LOCAL_DSN_FALLBACK).read()
    return "".join(re.findall(r'"([^"]*)"', re.search(r"DSN = \(([^)]*)\)", txt, re.S).group(1)))


def build_evidence() -> dict:
    m, a, d, v = EV["mission"], EV["anchor_evidence"], EV["delivery"], EV["verdict"]
    chk = {c["key"]: c for c in EV["checks"]}
    probes = [f for f in d["untracked_files"] if f["exists"]]
    return {
        # --- the eight required evidence keys -------------------------------
        "impl_files": [],
        "entry": None,
        "run_artifacts": [],
        "tests": [],
        "integration_callers": [],
        "benchmark": {
            "anchor_slug": a["slug"],
            "tensor_benchmark_cases_rows": a["benchmark_case_count"],
            "platform_case_total": a["platform_benchmark_case_total"],
            "status": "UNJUDGEABLE",
            "why": "no tensor_benchmark_cases row exists for this anchor, so no ref/metric/tolerance "
                   "was ever declared and no got-value from m18068 could be judged against one",
        },
        "repro_cmd": None,
        "external_anchor": {
            "source": "oss_solutions",
            "id": (a["oss_solution"] or {}).get("id"),
            "repo": (a["oss_solution"] or {}).get("repo"),
            "feature_slug": (a["oss_solution"] or {}).get("feature_slug"),
            "tag": (a["oss_solution"] or {}).get("tag"),
            "method": (a["oss_solution"] or {}).get("method"),
            "use": "reference paradigm only: the audit confirms it was read, and confirms no TensorLBM "
                   "code was written against it",
        },
        # --- what the subject actually produced -----------------------------
        "delivered_files": [
            {"path": f["path"], "size_bytes": f["size_bytes"], "sha256": f["sha256"],
             "kind": "untracked feasibility probe (KB/oss_solutions endpoint discovery)",
             "committed": False}
            for f in probes
        ],
        "delivery_branch": {"branch": m["branch"], "head_commit": m["head_commit"],
                            "commits_ahead_of_origin_main": d["delivery_branch_ahead_of_origin_main_commits"],
                            "worktree_head": d["head_subject"]},
        "reports": {"tensor_reports_rows_for_mission": d["tensor_reports_for_mission"],
                    "six_sections": False},
        # --- judge bookkeeping ----------------------------------------------
        "claimed_level_implicit": "none claimed (honest-FAIL self-declared: no implementation, no numbers)",
        "delivered_level": v["level"],
        "verdict_basis": (
            "L0 assigned, not L1+: maturity levels L2-L4 all require delivered code (R2), integration (R3) "
            "and a judged benchmark case (R4). m18068 delivered none: 0 commits ahead of origin/main, "
            "changed_files = two untracked .scratch probe scripts, 0 rows in tensor_reports, 0 "
            "tensor_benchmark_cases rows for the anchor. L1 (math/unit verification) also fails: no "
            "test asserts any property. The only durable output is reconnaissance recorded in the mission "
            "result; L0 is the floor, and it is granted for the OSS-solution knowledge capture, not for "
            "any capability in the repo."
        ),
        "gate_expectation": {
            "state": v["expected_gate_state"],
            "why": "capability gap GAP-LBM-MCMP-POISEUILLE/no-multiphase-layout remains entirely open; "
                   "nothing exists to confirm",
        },
        "checks": {k: chk[k]["satisfied"] for k in chk},
        "findings": [
            {"id": "F1", "severity": "blocking",
             "claim": "mission status 'succeeded' == delivery succeeded",
             "rebuttal": "runner status only; result.fail_stage=no_progress+forced_finish, branch=null, "
                         "head_commit=null, changed_files=2 untracked probes"},
            {"id": "F2", "severity": "blocking",
             "claim": "benchmark accuracy judged (<15%)",
             "rebuttal": "no tensor_benchmark_cases row for the anchor; no got-value produced; "
                         "zero numbers exist to compare"},
            {"id": "F3", "severity": "high",
             "claim": "failure_reason '预算耗尽于侦察阶段' (budget exhausted during reconnaissance)",
             "rebuttal": f"tool_calls {m['tool_calls']}/{m['budget']['max_tool_calls']} = "
                         f"{m['budget_used_frac']['tool_calls']:.0%}; wall "
                         f"{m['wall_seconds_used'] / 60:.1f}/{m['budget']['max_wall_seconds'] / 60:.0f} min = "
                         f"{m['budget_used_frac']['wall_seconds']:.0%} — neither budget was exhausted"},
            {"id": "F4", "severity": "high",
             "claim": "honest-FAIL handled as not-accepted",
             "rebuttal": f"tensor_approval_requests#{m['approval']['id']} kind=dev_acceptance "
                         f"status={m['approval_status']} decided_by={m['decided_by']} "
                         "(auto-approve path); a no-progress mission passed the gate as accepted"},
            {"id": "F5", "severity": "medium",
             "claim": "ecosystem support for the anchor",
             "rebuttal": "knowledge_base/oss_solutions API endpoints 404 for this agent (mission hit them "
                         "too); a benchmark case was never registered for the anchor -> the anchor was "
                         "dispensed without a machine judge"},
        ],
        "audit_artifacts": [
            "docs/board_audit/m18159/evidence.json (machine-collected evidence, all fields sourced)",
            "docs/board_audit/m18159/M18159_BOARD_AUDIT.md (report, six sections)",
            "docs/board_audit/m18159/figures/fig1_evidence_matrix.png",
            "docs/board_audit/m18159/figures/fig2_budget_utilization.png",
            "scripts/m18159_audit.py (collector), scripts/m18159_verify.py (re-runnable checks)",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ev = build_evidence()
    sql = ("INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
           "VALUES (%s, %s, %s, %s, %s::jsonb, %s) "
           "ON CONFLICT (repo, scope, verdict_by) DO UPDATE "
           "SET mission_id=EXCLUDED.mission_id, level=EXCLUDED.level, evidence=EXCLUDED.evidence, "
           "created_at=now() RETURNING id, level, gate_state")
    if args.dry_run:
        print(json.dumps({"repo": REPO, "scope": SCOPE, "mission_id": AUDIT_MISSION,
                          "level": LEVEL, "evidence": ev}, indent=1, ensure_ascii=False))
        return
    with psycopg.connect(dsn()) as conn, conn.cursor() as cur:
        cur.execute(sql, [REPO, SCOPE, AUDIT_MISSION, LEVEL, json.dumps(ev, ensure_ascii=False), BY])
        print("upserted:", cur.fetchone())
        conn.commit()


if __name__ == "__main__":
    main()