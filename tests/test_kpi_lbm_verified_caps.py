"""KPI-LBM-VERIFIED-CAPS 计量器回归测试（口径/自检/只读护栏/双路对拍/产物自洽）。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.kpi_lbm_verified_caps import collect, compute, oracle, selftest, spec

OUT = Path("runs/kpi_lbm_verified_caps")
PG_OK = os.path.exists("/usr/bin/psql") and os.path.exists("/var/run/postgresql")


def test_selftest_all_framed_cases_pass():
    r = selftest.run()
    assert r["n_fail"] == 0, r["fails"]
    assert r["n_cases"] == 12


def test_spec_is_frozen_and_self_consistent():
    d = spec.spec_dict()
    assert d["credit_gate_state"] == "confirmed"
    assert d["credit_min_level"] == 3
    assert d["target"] is None and d["target_status"] == "NOT_CALIBRATED"
    assert len(spec.spec_sha256()) == 64


def test_readonly_guard_rejects_writes():
    for sql in ("INSERT INTO x VALUES (1)", "UPDATE feature_ledger SET status='done'", "delete from x"):
        with pytest.raises(collect.ReadOnlyViolation):
            collect._assert_readonly(sql)
    collect._assert_readonly("select 1")
    collect._assert_readonly("WITH a AS (SELECT 1) SELECT * FROM a")


def test_compute_dedupes_and_respects_floor():
    rows = [
        {"id": "1", "mission_id": "1", "gate_state": "confirmed", "gate_level": "3", "scope": "m1 CASE-A",
         "created_at": "2026-11-10T00:00:00", "evidence_anchor": "CASE-A", "audited_mission": "1"},
        {"id": "2", "mission_id": "1", "gate_state": "confirmed", "gate_level": "4", "scope": "m1 CASE-A",
         "created_at": "2026-11-10T01:00:00", "evidence_anchor": "CASE-A", "audited_mission": "1"},
        {"id": "3", "mission_id": "1", "gate_state": "confirmed", "gate_level": "2", "scope": "m1 CASE-B",
         "created_at": "2026-11-10T01:00:00", "evidence_anchor": "CASE-B", "audited_mission": "1"},
    ]
    roster = compute.anchor_roster(rows)
    from datetime import datetime, timezone
    st = datetime(2026, 11, 9, tzinfo=timezone.utc)
    en = datetime(2026, 11, 14, tzinfo=timezone.utc)
    m = compute.window_metrics(roster, {}, st, en)
    assert m["current_primary"] == 1 and m["anchors"][0]["max_level"] == 4


def test_later_reopened_invalidates_credit():
    from datetime import datetime, timezone
    rows = [
        {"id": "1", "mission_id": "1", "gate_state": "confirmed", "gate_level": "4", "scope": "m1 CASE-A",
         "created_at": "2026-11-10T00:00:00", "evidence_anchor": "CASE-A", "audited_mission": "1"},
        {"id": "2", "mission_id": "2", "gate_state": "reopened", "gate_level": "1", "scope": "m2 CASE-A",
         "created_at": "2026-11-11T00:00:00", "evidence_anchor": "CASE-A", "audited_mission": "1"},
    ]
    m = compute.window_metrics(compute.anchor_roster(rows), {}, datetime(2026, 11, 9, tzinfo=timezone.utc),
                               datetime(2026, 11, 14, tzinfo=timezone.utc))
    assert m["current_primary"] == 0
    assert m["invalidated_by_later_verdict"] == ["CASE-A"]


@pytest.mark.skipif(not PG_OK, reason="postgres/psql unavailable")
def test_oracle_matches_python_on_live_snapshot():
    if not (OUT / "snapshot.json").exists():
        pytest.skip("run measure first")
    snap = json.loads((OUT / "snapshot.json").read_text())
    s = compute.summarize(snap)
    cmp = oracle.compare(s)
    assert cmp["match"], cmp["diffs"]
    collect.assert_all_readonly(snap)


def test_result_artifact_self_consistent():
    if not (OUT / "result.json").exists():
        pytest.skip("run measure first")
    res = json.loads((OUT / "result.json").read_text())
    assert res["spec_sha256"] == spec.spec_sha256()
    assert isinstance(res["current"], int) and res["current"] >= 0
    assert res["target"] is None and res["target_status"] == "NOT_CALIBRATED"
    assert res["current"] <= res["current_all_anchors"]
    assert res["current_strict"] <= res["current"]
    assert res["verification"]["oracle_match"] is True
    assert any(g["id"] == "GAP-KPI-TARGET-UNCALIBRATED" for g in res["top_gaps"])
