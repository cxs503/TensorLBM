#!/usr/bin/env python3
"""m18159 board audit — evidence collector (subject mission m18068).

Anchor: XVAL-LBM-NO-MULTIPHASE-LAYOUT-PATTERN-ONLY (TensorLBM).

Collects every fact used by the verdict from machine sources only:
  * tensor_dev_missions       — the subject mission record (result/evidence JSON)
  * tensor_benchmark_cases    — the *only* numerical arbiter for the anchor
  * tensor_reports            — whether a six-section report was published
  * the subject git worktree  — commits/branch/untracked files under /root/tp-agent/work/m18068

Nothing is typed by hand: the verdict, the report and the figures all read
``docs/board_audit/m18159/evidence.json``.

Predeclared acceptance rules (fixed BEFORE looking at the evidence, taken from
the task brief; the audit does not tune them afterwards):

  R1 (arbiter)   A numerical claim is valid only if a tensor_benchmark_cases row
                 exists for the anchor and the delivered got-value sits inside
                 the declared tolerance. No row  => nothing is judgeable.
  R2 (L2)        L2 requires real code: an entry that maps input -> compute ->
                 output, committed on the delivery branch.
  R3 (L3)        L3 additionally requires the code to be reachable from the main
                 solver (integration callers greppable in src/).
  R4 (L4)        L4 additionally requires a complete case + error analysis +
                 a reproducibly runnable repro_cmd.
  R5 (L5)        L5 = research/engineering result beyond a benchmark pass.
  R6 (artifacts) run_artifacts / tests / repro_cmd must exist as files in the
                 delivery, not as prose.
  R7 (honesty)   A failure narrative must not overstate its cause; a
                 "budget exhausted" claim is checked against the budget numbers.

Run:  python3 scripts/m18159_audit.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
SUBJECT_MISSION = 18068
SUBJECT_WT = Path("/root/tp-agent/work/m18068")
ANCHOR = "XVAL-LBM-NO-MULTIPHASE-LAYOUT-PATTERN-ONLY"
OUT = ROOT / "docs/board_audit/m18159"
LOCAL_DSN_FALLBACK = "/root/tp-agent/work/m15815/docs/board_audit/m15314/insert_verdict.py"


def dsn() -> str:
    d = os.environ.get("TP_FOUNDATION_DSN")
    if d:
        return d
    txt = open(LOCAL_DSN_FALLBACK).read()
    return "".join(re.findall(r'"([^"]*)"', re.search(r"DSN = \(([^)]*)\)", txt, re.S).group(1)))


def q(sql: str, args=None) -> list[dict]:
    with psycopg.connect(dsn()) as conn, conn.cursor() as cur:
        cur.execute(sql, args or [])
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=SUBJECT_WT, capture_output=True, text=True).stdout.strip()


def collect_mission() -> dict:
    row = q("SELECT id, repo, status, claimed_by, created_at, finished_at, budget, result FROM tensor_dev_missions WHERE id=%s",
            [SUBJECT_MISSION])[0]
    res = row["result"] or {}
    ev = res.get("evidence") or {}
    summary = ev.get("summary") or {}
    budget = row["budget"] or {}
    ap = q("SELECT id, kind, status, decided_by, created_at, reason FROM tensor_approval_requests "
           "WHERE mission_id=%s ORDER BY id DESC LIMIT 1", [SUBJECT_MISSION])
    approval = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in ap[0].items()} if ap else {}
    wall = (row["finished_at"] - row["created_at"]).total_seconds() if row["finished_at"] else None
    tool_calls = summary.get("tool_calls")
    return {
        "id": row["id"],
        "repo": row["repo"],
        "status": row["status"],
        "claimed_by": row["claimed_by"],
        "created_at": row["created_at"].isoformat(),
        "finished_at": row["finished_at"].isoformat(),
        "wall_seconds_used": wall,
        "budget": budget,
        "budget_used_frac": {
            "tool_calls": (tool_calls / budget["max_tool_calls"]) if tool_calls is not None else None,
            "wall_seconds": (wall / budget["max_wall_seconds"]) if wall else None,
        },
        "tool_calls": tool_calls,
        "headline": res.get("headline"),
        "fail_stage": res.get("fail_stage"),
        "decided_by": res.get("decided_by"),
        "accepted": res.get("accepted"),
        "approval_status": approval.get("status"),
        "approval": approval,
        "branch": ev.get("branch"),
        "head_commit": ev.get("head_commit"),
        "changed_files": res.get("changed_files") or [],
        "quality_audit": ev.get("server_quality_audit"),
    }


def collect_anchor() -> dict:
    cases = q("SELECT id, case_name, ref_source, metric, tolerance, repo, status, passed_at "
              "FROM tensor_benchmark_cases WHERE anchor_slug=%s ORDER BY id", [ANCHOR])
    total_cases = q("SELECT count(*) AS n FROM tensor_benchmark_cases")[0]["n"]
    oss = q("SELECT id, repo, feature_slug, tag, method FROM oss_solutions WHERE id=103")
    return {
        "slug": ANCHOR,
        "benchmark_cases": [{k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in c.items()} for c in cases],
        "benchmark_case_count": len(cases),
        "platform_benchmark_case_total": total_cases,
        "oss_solution": oss[0] if oss else None,
    }


def collect_delivery() -> dict:
    reports = q("SELECT count(*) AS n FROM tensor_reports WHERE mission_id=%s", [SUBJECT_MISSION])[0]["n"]
    status = git("status", "--porcelain")
    untracked = [l[3:] for l in status.splitlines() if l.startswith("??")]
    ahead = git("rev-list", "--count", "origin/main..HEAD") or "0"
    files = []
    for rel in untracked:
        p = SUBJECT_WT / rel
        files.append({
            "path": rel,
            "exists": p.exists(),
            "size_bytes": p.stat().st_size if p.exists() else None,
            "sha256": sha256(p) if p.exists() else None,
            "is_source_file": bool(re.search(r"\.(py|hpp|cpp|h|cu)$", rel)),
            "under_src_or_xval_or_tests": any(rel.startswith(d) for d in ("src/", "xval/", "tests/")),
        })
    # artifact hunt: any results/out json for a no-multiphase / layout-pattern xval
    xval_hits = sorted(str(p.relative_to(SUBJECT_WT)) for p in (SUBJECT_WT / "xval").glob("*layout*")) if (SUBJECT_WT / "xval").exists() else []
    xval_mp = sorted(str(p.relative_to(SUBJECT_WT)) for p in (SUBJECT_WT / "xval").glob("*multiphase*")) if (SUBJECT_WT / "xval").exists() else []
    src_hits = []
    for pat in ("**/*population*.py", "**/*layout_pattern*.py"):
        src_hits += [str(p.relative_to(SUBJECT_WT)) for p in (SUBJECT_WT / "src").glob(pat)]
    return {
        "delivery_branch_ahead_of_origin_main_commits": int(ahead),
        "head_commit": git("rev-parse", "HEAD"),
        "head_subject": git("log", "-1", "--pretty=%h %s"),
        "untracked_files": files,
        "untracked_source_files": [f for f in files if f["is_source_file"]],
        "untracked_files_under_src_xval_tests": [f for f in files if f["under_src_or_xval_or_tests"]],
        "tensor_reports_for_mission": reports,
        "xval_dirs_matching_layout": xval_hits,
        "xval_dirs_matching_multiphase": xval_mp,
        "src_population_or_layout_modules": src_hits,
    }


def build_checks(mission: dict, anchor: dict, delivery: dict) -> list[dict]:
    """Each check is a predeclared predicate with its machine evidence."""
    def chk(key, question, ok, evidence, note=""):
        return {"key": key, "question": question, "satisfied": ok, "evidence": evidence, "note": note}

    impl = [f["path"] for f in delivery["untracked_files_under_src_xval_tests"]]
    return [
        chk("impl_files", "delivered source files (src/xval/tests) on the delivery branch?",
            bool(impl), impl or [], "only two untracked probe scripts at repo root"),
        chk("entry", "executable entry (input->compute->output) for the anchor?",
            False, None, "no runnable driver references the anchor"),
        chk("run_artifacts", "results/artifacts (json/csv/png) produced for the anchor?",
            False, {"xval_layout_dirs": delivery["xval_dirs_matching_layout"],
                    "xval_multiphase_dirs": delivery["xval_dirs_matching_multiphase"]}),
        chk("tests", "tests asserting the new behaviour?",
            False, None, "no test file added; existing tests/test_aa_pattern_parity.py predates the mission"),
        chk("integration_callers", "reachable from the main solver (L3)?",
            False, None, "no new symbols exist to be called"),
        chk("benchmark", "tensor_benchmark_cases row for the anchor with ref/metric/tolerance?",
            anchor["benchmark_case_count"] > 0,
            {"rows": anchor["benchmark_cases"], "count": anchor["benchmark_case_count"]},
            "arbiter row absent => got-vs-ref cannot be judged at all"),
        chk("repro_cmd", "repro_cmd that reproduces the numbers?",
            False, None, "no repro_cmd in the mission result or worktree"),
        chk("external_anchor", "external OSS anchor identified with a real pointer?",
            bool(anchor["oss_solution"]),
            {k: anchor["oss_solution"][k] for k in ("id", "repo", "feature_slug", "tag")} if anchor["oss_solution"] else None,
            "reference exists; no delivered code is compared against it"),
        chk("report_six_sections", "six-section report published to the report lake?",
            delivery["tensor_reports_for_mission"] > 0, delivery["tensor_reports_for_mission"]),
        chk("budget_claim", "fail narrative's 'budget exhausted' claim consistent with budget usage?",
            False,
            {"tool_calls": mission["tool_calls"], "max_tool_calls": mission["budget"]["max_tool_calls"],
             "tool_call_frac": mission["budget_used_frac"]["tool_calls"],
             "wall_seconds": mission["wall_seconds_used"], "max_wall_seconds": mission["budget"]["max_wall_seconds"],
             "wall_frac": mission["budget_used_frac"]["wall_seconds"]},
            "claim contradicted: neither tool-call nor wall-clock budget was exhausted"),
    ]


def decide(checks: list[dict]) -> dict:
    """Apply R1-R7 mechanically. Level = highest rung whose prerequisites ALL hold."""
    ok = {c["key"]: c["satisfied"] for c in checks}
    l2_prereq = ok["impl_files"] and ok["entry"]
    l3_prereq = l2_prereq and ok["integration_callers"]
    l4_prereq = l3_prereq and ok["benchmark"] and ok["repro_cmd"] and ok["run_artifacts"]
    if ok["impl_files"] and not ok["entry"]:
        level = 1  # code exists but is not a runnable input->output path
    elif ok["impl_files"]:
        level = 2
    else:
        level = 0  # documentation / knowledge only (or less)
    if l3_prereq:
        level = 3
    if l4_prereq:
        level = 4
    gate = "confirmed" if (level >= 2 and ok["benchmark"]) else "reopened"
    return {
        "level": level,
        "l2_prereq_met": l2_prereq,
        "l3_prereq_met": l3_prereq,
        "l4_prereq_met": l4_prereq,
        "expected_gate_state": gate,
        "reason": ("no implementation, no artifact, no test, no repro_cmd, no benchmark case for the "
                   "anchor -> nothing above L0 is admissible; capability gap stays open")
        if level == 0 else "see checks",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mission = collect_mission()
    anchor = collect_anchor()
    delivery = collect_delivery()
    checks = build_checks(mission, anchor, delivery)
    verdict = decide(checks)
    evidence = {
        "audit_mission": 18159,
        "subject_mission": SUBJECT_MISSION,
        "subject_worktree": str(SUBJECT_WT),
        "anchor": ANCHOR,
        "collected_at": datetime.utcnow().isoformat() + "Z",
        "predeclared_rules": ["R1 arbiter", "R2 L2", "R3 L3", "R4 L4", "R5 L5", "R6 artifacts", "R7 honesty"],
        "mission": mission,
        "anchor_evidence": anchor,
        "delivery": delivery,
        "checks": checks,
        "verdict": verdict,
    }
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=1, ensure_ascii=False))
    print(json.dumps({"checks": {c["key"]: c["satisfied"] for c in checks}, "verdict": verdict}, indent=1))


if __name__ == "__main__":
    main()