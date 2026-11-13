#!/usr/bin/env python3
"""Re-runnable assertions for the m18159 verdict (self-check, not a physics test).

Fails loudly if the evidence collected by scripts/m18159_audit.py no longer
supports the verdict that was written into tensor_maturity_verdicts. Run it
after any re-audit:

    python3 scripts/m18159_audit.py && python3 scripts/m18159_verify.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/board_audit/m18159"
EV = json.loads((OUT / "evidence.json").read_text())
LOCAL_DSN_FALLBACK = "/root/tp-agent/work/m15815/docs/board_audit/m15314/insert_verdict.py"


def dsn() -> str:
    import os
    d = os.environ.get("TP_FOUNDATION_DSN")
    if d:
        return d
    txt = open(LOCAL_DSN_FALLBACK).read()
    return "".join(re.findall(r'"([^"]*)"', re.search(r"DSN = \(([^)]*)\)", txt, re.S).group(1)))


def check_missing_evidence():
    """Verdict must not claim code that the worktree does not have."""
    d, m = EV["delivery"], EV["mission"]
    assert d["delivery_branch_ahead_of_origin_main_commits"] == 0, "delivery branch gained commits — re-audit"
    assert d["untracked_files_under_src_xval_tests"] == [], "source/xval/tests files appeared — re-audit"
    assert d["tensor_reports_for_mission"] == 0, "a report was published later — re-audit"
    assert all(f["sha256"] and not f["under_src_or_xval_or_tests"] for f in d["untracked_files"])


def check_arbiter_absent():
    a = EV["anchor_evidence"]
    assert a["benchmark_case_count"] == 0, "benchmark case now exists — the anchor became judgeable, re-audit"


def check_level_rules():
    v = EV["verdict"]
    assert v["level"] == 0
    assert v["expected_gate_state"] == "reopened"
    assert not v["l2_prereq_met"] and not v["l3_prereq_met"] and not v["l4_prereq_met"]
    met = [c["key"] for c in EV["checks"] if c["satisfied"]]
    assert met == ["external_anchor"], f"unexpected met checks: {met}"


def check_budget_claim():
    m = EV["mission"]
    assert m["budget_used_frac"]["tool_calls"] < 0.9, "tool-call budget really was exhausted — claim would hold"
    assert m["budget_used_frac"]["wall_seconds"] < 0.9


def check_verdict_row():
    with psycopg.connect(dsn()) as conn, conn.cursor() as cur:
        cur.execute("SELECT level, mission_id, evidence, verdict_by, gate_state FROM tensor_maturity_verdicts "
                    "WHERE repo='TensorLBM' AND scope=%s AND verdict_by='board-audit'",
                    [f"m18068 {EV['anchor']}"])
        rows = cur.fetchall()
    assert len(rows) == 1, f"expected exactly one board-audit verdict row, got {len(rows)}"
    level, mission_id, evidence, by, gate_state = rows[0]
    assert level == EV["verdict"]["level"] and mission_id == 18159
    for key in ("impl_files", "entry", "run_artifacts", "tests", "integration_callers",
                "benchmark", "repro_cmd", "external_anchor"):
        assert key in evidence, f"evidence key missing: {key}"
    print(f"verdict row ok: level={level} gate_state={gate_state} by={by} keys={len(evidence)}")


def check_artifacts_present():
    for p in ("evidence.json", "M18159_BOARD_AUDIT.md",
              "figures/fig1_evidence_matrix.png", "figures/fig2_budget_utilization.png"):
        assert (OUT / p).exists(), f"missing artifact {p}"


if __name__ == "__main__":
    checks = [("missing_evidence", check_missing_evidence), ("arbiter_absent", check_arbiter_absent),
              ("level_rules", check_level_rules), ("budget_claim", check_budget_claim),
              ("artifacts_present", check_artifacts_present), ("verdict_row", check_verdict_row)]
    bad = 0
    for name, fn in checks:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            bad += 1
            print(f"FAIL {name}: {e}")
    sys.exit(1 if bad else 0)