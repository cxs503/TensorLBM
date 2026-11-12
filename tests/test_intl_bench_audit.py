"""Evidence-ladder tests for ``tensorlbm.audit.intl_bench`` (GAP-LBM-INTL-BENCH).

These tests assert *invariants of the grading machinery*, not the current audit
numbers: a grade may only move up the ladder 1 module-exists < 2 API-test <
3 conservation/gate case < 4 standard case < 5 grid convergence < 6 experiment,
and ``verified`` is unattainable without a benchmark result on disk that states a
truth verdict.  The point is that no module can be self-declared verified by an
auditing agent (owner 2026-11-12 ruling).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tensorlbm.audit import intl_bench as ib

REPO = Path(__file__).resolve().parents[1]


# ── fixture: a throw-away "repo" whose evidence level is exactly controllable ──
def _write(p: Path, text: str = "") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.strip() + "\n")


def make_repo(tmp_path: Path) -> Path:
    root = tmp_path / "fakerepo"
    _write(root / "src/solver_alpha.py", "def step(u):\n    return u\n")
    _write(root / "tests/test_solver_alpha.py", "def test_x():\n    assert True\n")
    _write(root / "benchmarks/pending/case_pending/result.json", json.dumps({"status": "pending"}))
    _write(root / "benchmarks/verified/case_ok/result.json",
           json.dumps({"cd_ref": 1.0, "cd_got": 1.002, "rel_err_pct": 0.2,
                       "verified": True, "grid_convergence_order": 1.98}))
    _write(root / "benchmarks/verified/case_noverdict/result.json",
           json.dumps({"cd_got": 3.1, "grid_convergence_order": 1.5}))
    _write(root / "benchmarks/STATUS.md",
           "| case | note |\n|---|---|\n| `case_ok` | +0.20% vs ref |\n")
    return root


def module(key, impl, tests=(), cases=(), docs=(), note="", family="基础LBM内核"):
    return (key, family, list(impl), list(tests), list(cases), list(docs), note)


def test_ladder_is_monotone_in_evidence(tmp_path):
    root = make_repo(tmp_path)
    status = ib.load_status(root)
    lv = {}
    lv["impl_only"] = ib.grade(root, status, module("impl_only", ["src/solver_alpha.py"]))
    lv["with_test"] = ib.grade(root, status, module("with_test", ["src/solver_alpha.py"],
                                                    ["tests/test_solver_alpha.py"]))
    lv["with_pending"] = ib.grade(root, status, module("with_pending", ["src/solver_alpha.py"],
                                                      ["tests/test_solver_alpha.py"],
                                                      ["case_pending"]))
    lv["with_verified"] = ib.grade(root, status, module("with_verified", ["src/solver_alpha.py"],
                                                       ["tests/test_solver_alpha.py"],
                                                       ["case_ok"]))
    tiers = [lv[k]["evidence_tier"] for k in ("impl_only", "with_test", "with_pending", "with_verified")]
    assert tiers == sorted(tiers), tiers
    assert tiers == [1, 2, 3, 5], tiers
    assert lv["impl_only"]["level"] == "implemented"
    assert lv["with_pending"]["level"] == "partial"
    # tier 5 + a truth verdict on disk is the only route to "verified"
    assert lv["with_verified"]["level"] == "verified"
    assert lv["with_verified"]["verdict_fields"]


def test_verified_requires_truth_verdict_on_disk(tmp_path):
    root = make_repo(tmp_path)
    status = ib.load_status(root)
    r = ib.grade(root, status, module("no_verdict", ["src/solver_alpha.py"], [],
                                      ["case_noverdict"]))
    assert r["evidence_tier"] == 5
    assert r["level"] == "partial"          # convergence alone does not verify
    assert "downgrade_reason" in r


def test_known_defect_without_verified_case_is_failed(tmp_path, monkeypatch):
    root = make_repo(tmp_path)
    status = ib.load_status(root)
    monkeypatch.setitem(ib.KNOWN_DEFECTS, "impl_only", "结果文件记录结构性 FAIL")
    r = ib.grade(root, status, module("impl_only", ["src/solver_alpha.py"]))
    assert r["level"] == "failed"
    assert r["defect"].startswith("结果文件记录")


def test_missing_impl_is_not_promoted(tmp_path):
    root = make_repo(tmp_path)
    status = ib.load_status(root)
    r = ib.grade(root, status, module("ghost", ["src/does_not_exist.py"]))
    assert r["impl_paths"] == []
    assert r["level"] == "implemented"


def test_compete_aggregation_bounds_and_gap(monkeypatch):
    rows = [
        {"key": "a", "family": "基础LBM内核", "evidence_tier": 6, "level": "verified"},
        {"key": "b", "family": "基础LBM内核", "evidence_tier": 2, "level": "implemented"},
    ]
    monkeypatch.setattr(ib, "FAMILIES", ["基础LBM内核"])
    monkeypatch.setitem(ib.DIM_TOPIC, "基础LBM内核", "collision")
    monkeypatch.setitem(ib.UPSTREAM, "Dummy", {"root": "/nonexistent", "tier": {"collision": 4}, "snapshot": "x"})
    out = ib.compete(rows)
    assert len(out) == 1
    s = out[0]
    assert s["n_modules"] == 2
    assert s["mean_evidence_tier"] == pytest.approx(4.0)
    assert 0.0 <= s["maturity_index_0_10"] <= 10.0
    assert sum(s["counts"].values()) == 2
    assert s["gap_grade"][0] in "ABCD"


def test_citation_check_reports_missing_checkout_without_crashing(monkeypatch):
    monkeypatch.setattr(ib, "UPSTREAM", {"Ghost": {"root": "/definitely/not/here", "tier": {}, "cites": {}}})
    rep = {}
    bad = ib.check_citations(rep)
    assert bad and "Ghost" in bad[0]
    assert rep["citation_stats"]["Ghost"] == {"ok": 0, "total": 0}


def test_cli_end_to_end_writes_artifacts(tmp_path):
    root = make_repo(tmp_path)
    out = tmp_path / "out"
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    proc = subprocess.run([sys.executable, "-m", "tensorlbm.audit", "--repo-root", str(root),
                           "--out", str(out)], capture_output=True, text=True, env=env, timeout=300)
    assert proc.returncode in (0, 3), proc.stderr[-2000:]
    for name in ("module_maturity.json", "module_maturity.csv", "compete_score.csv", "maturity.sql"):
        assert (out / name).is_file(), name
    rep = json.loads((out / "module_maturity.json").read_text())
    assert rep["provenance"]["n_modules"] == len(ib.MODULES)
    assert rep["provenance"]["input_sha256"]          # numbers are bound to hashed inputs
    assert set(rep["level_counts"]) == {"verified", "partial", "implemented", "failed"}


def test_real_repo_has_no_unbacked_verified(tmp_path):
    """The load-bearing invariant: every 'verified' row points at evidence on disk."""
    out = tmp_path / "real"
    rep = ib.run_audit(REPO, out)
    for r in rep["modules"]:
        if r["level"] == "verified":
            case = r["case_verified"]
            assert case, r["key"]
            case_dir = REPO / "benchmarks" / "verified" / case
            assert case_dir.is_dir(), case_dir
            assert r["verdict_fields"], f"{r['key']} verified without a truth verdict"
            assert r["has_convergence_field"], f"{r['key']} verified without convergence evidence"
    assert rep["level_counts"]["verified"] >= 1       # sphere_re200-style anchors exist


def test_audit_is_deterministic(tmp_path):
    a = ib.run_audit(REPO, tmp_path / "a")
    b = ib.run_audit(REPO, tmp_path / "b")
    ka = [(r["key"], r["level"], r["evidence_tier"]) for r in a["modules"]]
    kb = [(r["key"], r["level"], r["evidence_tier"]) for r in b["modules"]]
    assert ka == kb
    assert [s["maturity_index_0_10"] for s in a["scores"]] == [s["maturity_index_0_10"] for s in b["scores"]]

def test_public_api_caller_returns_scorecard(tmp_path):
    """Platform-side integration: ``tensorlbm.api.intl_bench_scorecard`` is a real call."""
    from tensorlbm.api import intl_bench_scorecard

    sc = intl_bench_scorecard(str(REPO), str(tmp_path / "sc"))
    assert sc["n_modules"] == len(ib.MODULES)
    assert len(sc["scores"]) == len({m[1] for m in ib.MODULES})
    for s in sc["scores"]:
        assert 0.0 <= s["maturity_index_0_10"] <= 10.0
        assert s["gap_grade"][0] in "ABCD"
    assert sc["level_counts"]["verified"] == sum(c["verified"] for c in [s["counts"] for s in sc["scores"]])
