"""SDAA 工件台账 v2 契约测试（tools/sdaa_ledger_v2.py）。

覆盖三组不变量：
1) evidence_level 定级规则（REAL/MOCK/STUB/WITHHELD/NOT_SUPPORTED）逐条可复现；
2) 缺失字段纪律：null + missing_reason，default_fill_count 恒为 0；
3) 冲突重裁纪律：与历史终态冲突的条目必有 conflict_reason，需重跑者进 needs_rerun；
4) 落盘台账的机器判据（entries>=3000 / parse_ok>=0.9 / 每条有 evidence_level / HTML 可打开）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import sdaa_ledger_v2 as ledger  # noqa: E402

LEDGER_JSON = REPO / "artifacts" / "sdaa_ledger_v2.json"
LEDGER_HTML = REPO / "artifacts" / "sdaa_ledger_v2.html"


def admit(doc, rel="sdaa_case.json", history=None):
    text = json.dumps(doc, ensure_ascii=False)
    return ledger.adjudicate_entry(doc, text, True, None, rel, history or {})


# --------------------------------------------------------------- 定级规则
def test_real_needs_device_evidence_and_metrics():
    doc = {
        "case": "couette_2d",
        "device_evidence": {"device": "sdaa:0", "sdaa_device_name": "/dev/tcaicard10"},
        "results": [{"l2_rel_err": 1.2e-4, "max_abs_err": 5.8e-6, "max_rel_err_masked_pct": 0.02}],
        "verdict": "PASS",
        "gates": [{"name": "l2", "pass": True}],
    }
    e = admit(doc)
    assert e["evidence_level"] == "REAL"
    assert e["device_kind"] == "sdaa"
    assert e["conflict"] is None


def test_mock_marker_downgrades_to_mock():
    e = admit({"case": "x", "note": "values are synthetic placeholders", "value_err": 0.1})
    assert e["evidence_level"] == "MOCK"
    assert "硬编码" in e["evidence_reason"] or "伪造" in e["evidence_reason"]


def test_stub_when_only_structure():
    e = admit({"benchmark": "empty_case", "status": "in_progress", "notes": "todo"})
    assert e["evidence_level"] == "STUB"
    assert e["needs_rerun"] is True


def test_withheld_when_sdaa_has_no_valid_numbers():
    doc = {"case": "sphere_re100", "device_evidence": {"device_name": "/dev/tcaicard5"},
           "gate": "no-evidence", "overall": {"pass": False},
           "reason": "SDAA cannot produce any valid Cd value"}
    e = admit(doc)
    assert e["evidence_level"] == "WITHHELD"
    assert e["device_kind"] == "sdaa"


def test_withheld_when_device_is_cuda():
    doc = {"case": "annulus", "device": "cuda:0 (NVIDIA GPU)", "verdict": "PASS",
           "l2_err": 0.01, "max_err": 0.02, "rel_err": 0.03}
    e = admit(doc)
    assert e["evidence_level"] == "WITHHELD"
    assert e["device_kind"] == "cuda"
    assert e["conflict"]["type"] == "claim_unsupported"
    assert e["needs_rerun"] is True
    assert "CUDA" in e["conflict_reason"]


def test_not_supported_marker():
    e = admit({"case": "y", "note": "backend is not supported on this device"})
    assert e["evidence_level"] == "NOT_SUPPORTED"


def test_report_artifact_is_withheld_and_case_null():
    e = admit({"title": "t", "summary": "s", "scenario_key": "k", "markdown": "# x"},
              rel="sdaa_x_report.json")
    assert e["evidence_level"] == "WITHHELD"
    assert e["evidence_kind"] == "report"
    assert e["case"] is None and "case" in e["missing"]


# --------------------------------------------------------------- 缺失字段纪律
def test_missing_fields_are_null_with_reason_never_defaulted():
    e = admit({"case": "bare"})
    for field in ("device", "numeric_metrics", "verdict_claimed", "wall_clock_s",
                  "mission_id", "timestamp", "reference"):
        assert e[field] is None
        assert e["missing"].get(field), f"{field} 缺失但没写 missing_reason"


# --------------------------------------------------------------- 冲突重裁纪律
def test_self_inconsistent_claim():
    doc = {
        "case": "c", "device_evidence": {"device_name": "/dev/tcaicard1"},
        "verdict": "PASS", "l2_err": 0.01, "max_rel_err": 0.02, "cd_err": 0.03,
        "gates": [{"name": "a", "pass": True}, {"name": "b", "pass": False}],
    }
    e = admit(doc)
    assert e["evidence_level"] == "REAL"
    assert e["conflict"]["type"] == "self_inconsistent"
    assert e["gate_pass"] is False
    assert "FAIL" in e["conflict_reason"]


def test_history_drift_conflict_gets_reason():
    hist = {"c": {"terminal": "FAIL", "sources": ["docs/sdaa_sweep/c.md"], "verdict_histogram": {"FAIL": 1}}}
    doc = {"case": "c", "device_evidence": {"device_name": "/dev/tcaicard1"},
           "verdict": "PASS", "l2_err": 0.01, "max_rel_err": 0.02, "cd_err": 0.03,
           "gates": [{"name": "a", "pass": True}]}
    e = admit(doc, history=hist)
    assert e["conflict"]["type"] == "history_drift"
    assert e["conflict_reason"] and "docs/sdaa_sweep/c.md" in e["conflict_reason"]
    assert e["needs_rerun"] is False  # 漂移只重裁，不重跑


def test_history_conflict_when_history_pass_but_no_evidence():
    hist = {"c": {"terminal": "PASS", "sources": ["docs/sdaa_sweep/c.md"], "verdict_histogram": {"PASS": 1}}}
    doc = {"case": "c", "config": {"dx": 1.0}, "lattice": "D2Q9", "elapsed_s": 1.0}
    e = admit(doc, history=hist)
    assert e["evidence_level"] in ("WITHHELD", "STUB")
    assert e["conflict"] is not None and e["conflict_reason"]
    assert e["needs_rerun"] is True


def test_conflict_entries_never_lose_reason():
    docs = [
        {"case": "a", "verdict": "PASS"},
        {"case": "b", "overall_pass": True, "gates": [{"pass": False}],
         "device_evidence": {"device_name": "/dev/tcaicard2"},
         "l2_err": 1e-3, "max_err": 2e-3, "rel_err": 3e-3},
    ]
    for doc in docs:
        e = admit(doc)
        if e["conflict"] is not None:
            assert e["conflict_reason"], "冲突条目必须带 conflict_reason"


# --------------------------------------------------------------- 渲染
def test_html_render_contains_levels_and_is_standalone():
    built = {
        "stats": {
            "entries": 1, "unique_artifacts": 1, "parse_ok": 1.0, "cases": 1,
            "conflict_entries": 0, "conflict_entries_canonical": 0, "needs_rerun_entries": 0,
            "needs_rerun_canonical": 0, "default_fill_count": 0, "duplicate_instances": 0,
            "duplication_factor": 1.0, "parse_fail_count": 0,
            "evidence_level_instances": {"REAL": 1, "MOCK": 0, "STUB": 0, "WITHHELD": 0,
                                         "NOT_SUPPORTED": 0},
            "evidence_level_canonical": {"REAL": 1, "MOCK": 0, "STUB": 0, "WITHHELD": 0,
                                         "NOT_SUPPORTED": 0},
            "entries_without_evidence_level": 0, "conflict_entries_missing_reason": 0,
            "evidence_kind_instances": {"run_evidence": 1}, "device_kind_instances": {"sdaa": 1},
            "claimed_terminal_instances": {"PASS": 1}, "conflict_type_instances": {},
            "cases_with_real_evidence": 1, "missing_field_histogram": {"mission_id": 1},
            "generated_utc": "2026-11-06T00:00:00+00:00", "tool": "tools/sdaa_ledger_v2.py",
        },
        "corpus": {"roots": [{"worktree": "self", "path": "/tmp"}], "worktree_count": 1,
                   "outside_scope_instances": 0, "outside_scope_note": "n/a",
                   "pattern": "artifacts/**/sdaa*.json"},
        "cases": [{"case": "couette_2d", "instances": 1, "dominant_level": "REAL",
                   "levels": {"REAL": 1}, "needs_rerun": 0}],
        "needs_rerun": [],
        "entries": [{
            "entry_id": "e0", "case": "couette_2d", "worktree": "self",
            "artifact_rel": "sdaa_couette_2d.json", "artifact_sha256": "0" * 64,
            "evidence_kind": "run_evidence", "evidence_level": "REAL",
            "evidence_reason": "ok", "evidence_signals": {"result_metrics": 3, "gate_pass": 1,
                                                         "gate_fail": 0},
            "verdict_claimed": "PASS", "device_kind": "sdaa", "gate_pass": True,
            "numeric_metrics": {"l2_rel_err": 1e-4}, "conflict": None,
            "conflict_reason": None, "needs_rerun": False, "missing": {},
        }],
        "entries_canonical": ["e0"],
    }
    out = ledger.render_html(built)
    assert out.lstrip().lower().startswith("<!doctype html>")
    assert "couette_2d" in out and "REAL" in out
    assert "http" not in out.split("<style>")[0]  # 无外链依赖


# --------------------------------------------------------------- 落盘台账判据
@pytest.mark.skipif(not LEDGER_JSON.is_file(), reason="台账未生成")
def test_ledger_artifact_contract():
    data = json.loads(LEDGER_JSON.read_text(encoding="utf-8"))
    entries = data["entries"]
    st = data["stats"]
    assert st["entries"] == len(entries) >= 3000
    assert st["parse_ok"] >= 0.90
    assert all(e["evidence_level"] in ledger.LEVELS for e in entries)
    assert st["default_fill_count"] == 0
    for e in entries:
        for field in ("case", "device", "numeric_metrics", "verdict_claimed"):
            if e.get(field) is None:
                assert field in e["missing"], f"{e['entry_id']} 的 {field} 缺 missing_reason"
    conflicts = [e for e in entries if e["conflict"]]
    assert conflicts and all(e["conflict_reason"] for e in conflicts)
    assert st["needs_rerun_entries"] == len(data["needs_rerun"]) > 0
    assert LEDGER_HTML.is_file()
    head = LEDGER_HTML.read_text(encoding="utf-8", errors="replace").lstrip().lower()
    assert head.startswith("<!doctype html>")
    # 唯一内容视图与实例视图自洽
    assert st["unique_artifacts"] + st["duplicate_instances"] == st["entries"]