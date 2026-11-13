"""Evidence-ladder tests for ``tensorlbm.audit.intl_bench`` (GAP-LBM-INTL-BENCH).

These tests assert *invariants of the grading machinery*, not the current audit
numbers: a grade may only move up the ladder 1 module-exists < 2 API-test <
3 conservation/gate case < 4 standard case < 5 grid convergence < 6 experiment,
and ``verified`` is unattainable without a benchmark result on disk that states a
truth verdict.  The point is that no module can be self-declared verified by an
auditing agent (owner 2026-11-12 ruling).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
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


# ── docs/ SQL *content* seal (m15617 round-3 / finding F4) ────────────────────
# The mission deliverable is one generated SQL snapshot shipped under two docs/
# names.  On main before this guard *both* copies were stale w.r.t. the generator
# (m15912-named: 27/45 modules carried a superseded payload -- status_row 24,
# test_paths 7, impl_paths 1, pending_status_row 1; m15617-named: 45/45 payloads
# stale in the superseded $json$ layout, and its headline level distribution said
# verified=7 where the current grading says 4, because the three evidence-reuse
# downgrades carried in ``downgrade_reason`` were absent).  A DDL/schema-only
# check cannot see any of that, so these tests compare *values* against a fresh
# in-memory audit.  When they fail the repair is one command:
#     python -m tensorlbm.audit --repo-root . --out artifacts/m15617
EVIDENCE_KEYS = ("evidence_tier", "tier_why", "impl_paths", "test_paths", "doc_paths",
                 "case_verified", "case_pending", "verdict_fields",
                 "has_convergence_field", "error_samples", "status_row",
                 "pending_status_row", "defect", "note", "downgrade_reason",
                 "evidence_shared_with", "audit_id")


def _sql_inserts(sql_text: str, table: str) -> list[str]:
    """Every ``INSERT INTO <table> ... VALUES (...)`` payload, in file order."""
    rows = []
    for line in sql_text.splitlines():
        if not line.startswith(f"INSERT INTO {table} "):
            continue
        rows.append(line.split("VALUES (", 1)[1].rstrip(";").rstrip(")"))
    return rows


def _sql_quoted(payload: str) -> list[str]:
    """The single-quoted literals of a VALUES payload, with '' unescaped."""
    return [f.replace("''", "'") for f in re.findall(r"'((?:[^']|'')*)'", payload)]


def test_committed_docs_sql_evidence_matches_current_audit(tmp_path):
    """Anti-drift on *values*, not just DDL: a reader of docs/ must see today's grades."""
    sc = ib.run_audit(REPO, tmp_path / "now")
    fresh = {r["key"]: r for r in sc["modules"]}
    for name in ib.DOC_SQL_FILES:
        path = REPO / "docs" / name
        assert path.is_file(), path
        rows = _sql_inserts(path.read_text(), "tensorlbm_module_maturity")
        assert len(rows) == len(ib.MODULES), name
        seen = set()
        for payload in rows:
            key, area, tier, evidence = _sql_quoted(payload)[:4]
            seen.add(key)
            assert key in fresh, f"{name}: unknown module {key}"
            want = {k: fresh[key].get(k) for k in EVIDENCE_KEYS if k != "audit_id"}
            want["audit_id"] = ib.AUDIT_ID
            got = json.loads(evidence)
            assert set(got) == set(want), \
                f"{name}:{key} evidence keys {sorted(set(got) ^ set(want))}"
            diff = {k: (got[k], want[k]) for k in EVIDENCE_KEYS if got[k] != want[k]}
            assert not diff, f"{name}:{key} stale evidence field(s) {sorted(diff)}: {diff}"
            assert tier == fresh[key]["level"], f"{name}:{key} tier"
            assert area == fresh[key]["family"], f"{name}:{key} area"
        assert seen == set(fresh), f"{name}: modules missing from the deliverable"


def test_committed_docs_sql_compete_rows_match_current_audit(tmp_path):
    """The ten-dimension weekly score table in docs/ must match the current scoring."""
    sc = ib.run_audit(REPO, tmp_path / "now")
    fresh = {s["dimension"]: s for s in sc["scores"]}
    rows = _sql_inserts((REPO / "docs" / ib.DOC_SQL_FILES[0]).read_text(),
                        "tensorlbm_compete_score")
    assert len(rows) == len(fresh)
    for payload in rows:
        week, dim, maturity, gap, notes = _sql_quoted(payload)[:5]
        score = float(re.search(r",\s*(-?[0-9.]+),\s*'\{", payload).group(1))
        assert dim in fresh, dim
        assert float(maturity) == pytest.approx(fresh[dim]["maturity_index_0_10"])
        assert score == pytest.approx(fresh[dim]["maturity_index_0_10"])
        assert gap == fresh[dim]["gap_grade"].split("(")[0]
        assert json.loads(notes)["n_modules"] == fresh[dim]["n_modules"]


def test_docs_sql_deliverables_are_one_snapshot(tmp_path):
    """Every docs/ name is written from one string -- byte-identical, generator-equal."""
    sc = ib.run_audit(REPO, tmp_path / "now")
    texts = [(REPO / "docs" / n).read_text() for n in ib.DOC_SQL_FILES]
    assert len(set(texts)) == 1, f"docs names diverge: {ib.DOC_SQL_FILES}"
    assert sc["sql_deliverable"]["files"] == list(ib.DOC_SQL_FILES)
    assert sc["sql_deliverable"]["sha256"] == hashlib.sha256(texts[0].encode()).hexdigest()
    assert texts[0] == (tmp_path / "now" / "maturity.sql").read_text()
    assert "\\$json\\$" not in texts[0]       # superseded jsonb layout must not come back


def test_run_audit_does_not_mutate_the_docs_deliverables(tmp_path):
    """A library/API caller (and this suite) must not rewrite the tree it audits.

    Only the CLI refreshes docs/: otherwise importing the audit and running it on
    the real repo silently rewrites the deliverable, which is how the staleness in
    finding F4 stayed invisible (the bytes changed but nobody diffs a refresh).
    Artifacts go where the caller asks (``run_audit`` defaults to the in-repo
    ``artifacts/m15617`` for the CLI's convenience), so a caller that wants no tree
    writes at all passes ``out_dir``.
    """
    doc_sql = sorted((REPO / "docs").glob("tensorlbm_maturity_m*.sql"))
    assert doc_sql
    before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in doc_sql}
    rep = ib.run_audit(REPO, tmp_path / "out")
    after = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in doc_sql}
    assert before.keys() == after.keys()
    assert before == after                      # bytes and mtimes untouched
    assert Path(rep["sql_deliverable"]["files"][0]).name in before
    assert (tmp_path / "out" / "maturity.sql").is_file()


# ── the drift tool behind the §7 write-up must itself be checked ──────────────
def _drift_tool():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "docs_sql_drift", REPO / "tools" / "audit" / "docs_sql_drift.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def test_docs_sql_drift_tool_detects_value_drift():
    """``tools/audit/docs_sql_drift.py`` flags stale values and stays quiet when clean."""
    tool = _drift_tool()

    def dump(ev, level="partial"):
        return ("INSERT INTO tensorlbm_module_maturity (module_path,area,tier,evidence,intl_ref,"
                "action,gap_task) VALUES ('m1','fam','" + level + "','" + json.dumps(ev)
                + "','ref','keep','');\n")

    old = tool.evidence_rows(dump({"evidence_tier": 4, "status_row": "analytic / standard case"}))
    new = tool.evidence_rows(dump({"evidence_tier": 4, "status_row": "yes"}))
    rep = tool.diff(old, new)
    assert rep["identical"] is False
    assert rep["n_modules_drift"] == 1
    assert rep["field_counts"] == {"status_row": 1}
    assert rep["modules"]["m1"]["new"] == {"status_row": "yes"}
    assert tool.diff(new, new)["identical"] is True


def test_docs_sql_drift_tool_reads_the_superseded_jsonb_layout():
    """The superseded m15617 dump ($json$ block, different column order) is also readable.

    Reading it matters: the two layouts are not interchangeable SQL, so "the copies
    agree" has to be decided on parsed values, and a layout difference alone is a
    finding (45/45 stale rows when the guard was introduced).
    """
    tool = _drift_tool()
    jsonb = ("INSERT INTO tensorlbm_module_maturity (audit_id,module_key,family,level,evidence_tier,"
             "tier_why,evidence,mission_id) VALUES ('m15617-intl-bench-v1','m1','fam','verified',5,"
             "'why',$json${\"evidence_tier\": 5, \"status_row\": \"yes\"}$json$::jsonb,15617);\n")
    assert tool.evidence_rows(jsonb) == {"m1": {"evidence_tier": 5, "status_row": "yes"}}
    assert tool.level_rows(jsonb) == {"m1": "verified"}
    rep = tool.diff(tool.evidence_rows(jsonb), tool.evidence_rows(jsonb.replace(
        "'verified'", "'partial'")), tool.level_rows(jsonb),
        tool.level_rows(jsonb.replace("'verified'", "'partial'")))
    assert rep["field_counts"] == {"level": 1}
    assert rep["identical"] is False
