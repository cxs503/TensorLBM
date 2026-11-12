"""Regression tests for the independent square-cylinder audit (mission m16608).

The point of the audit is that it must be able to *disagree*.  These tests
therefore probe it with tampered inputs instead of only running it green:

* it must not import the case's own judges (``refspec``/``verify``/``run``);
* a measurement whose error exceeds the bar must be reported FAIL;
* the reference-centre robustness check must fire when a band is wide enough
  for the verdict to flip, and must stay quiet on a narrow band;
* a stored error column that reproduces no named convention must be blocking,
  while one that reproduces the declared legacy convention must only be
  advisory.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "verified" / "square_cylinder"
AUDIT = CASE_DIR / "independent_audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sq_independent_audit", AUDIT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_audit_exists_and_does_not_import_the_case_judges():
    src = AUDIT.read_text()
    assert AUDIT.is_file()
    for forbidden in ("import refspec", "from refspec", "import verify", "from verify"):
        assert forbidden not in src, f"audit must be a second opinion, found: {forbidden}"


def test_audit_reproduces_the_archived_numbers_on_the_live_tree():
    res = subprocess.run([sys.executable, str(AUDIT), "--json", "/tmp/sq_audit_test.json"],
                         capture_output=True, text=True)
    data = json.loads(Path("/tmp/sq_audit_test.json").read_text())
    # own arithmetic on the archived measurements must land on the published errors
    assert abs(data["levels"]["32"]["cd_err_pct"] - 0.11) < 0.02
    assert abs(data["levels"]["32"]["st_err_pct"] - 0.82) < 0.02
    assert data["levels"]["64"]["column"] == "pair"
    # unresolved checks are reported, never absorbed into a pass
    assert data["counts"]["not_run"] >= 1
    assert data["disposition"] in ("approve_with_notes", "request_changes")
    assert res.returncode == (1 if data["counts"]["blocking"] else 0)


def _stage(tmp_path: Path, *, cd: float, st: float, band: str) -> Path:
    stage = tmp_path / "stage"
    stage.mkdir()
    for D, scale in ((32, 1.0), (48, 1.0)):
        (stage / f"case_D{D}.json").write_text(json.dumps(
            {"D": D, "re": 100.0, "nu_lb": 0.016, "steps": 60000,
             "cd": cd * scale, "st": st * scale, "cd_ref": 1.48, "st_ref": 0.147,
             "err_pct": (cd * scale - 1.48) / 1.48 * 100.0,
             "st_err_pct": (st * scale - 0.147) / 0.147 * 100.0}))
    (stage / "result.json").write_text(json.dumps({"reference": band}))
    return stage


def test_out_of_bar_measurement_is_flagged_not_passed(tmp_path):
    mod = _load_module()
    stage = _stage(tmp_path, cd=1.60, st=0.147, band="Sohankar 1.46/0.147, Present 1.48/0.147")
    res = mod.audit(stage, "")
    assert res["levels"]["32"]["passes_3pct"] is False
    assert res["disposition"] == "request_changes"


def test_wide_band_makes_the_verdict_conditional(tmp_path):
    mod = _load_module()
    stage = _stage(tmp_path, cd=1.487, st=0.1479,
                   band="Yoon 1.44/0.145, Present 1.48/0.147, Sen 1.53/0.149")
    res = mod.audit(stage, "")
    kinds = {f["kind"] for f in res["findings"]}
    assert "verdict-conditional-on-centre" in kinds
    assert res["scope"]["band_robustness"]["cd"]["max_abs_pct_over_band"] > mod.BAR_PCT


def test_narrow_band_keeps_the_verdict_unconditional(tmp_path):
    mod = _load_module()
    stage = _stage(tmp_path, cd=1.487, st=0.1479,
                   band="Present 1.481/0.147, Sahu 1.488/0.149")
    res = mod.audit(stage, "")
    kinds = {f["kind"] for f in res["findings"]}
    assert "verdict-conditional-on-centre" not in kinds


def test_legacy_error_column_is_advisory_but_unknown_drift_is_blocking(tmp_path):
    mod = _load_module()
    stage = _stage(tmp_path, cd=1.487, st=0.1479, band="Present 1.48/0.147")
    p = stage / "case_D64.json"
    cd, st = 1.487, 0.1479
    p.write_text(json.dumps({"D": 64, "re": 100.0, "force_columns": {"n_solid_cells": 4096},
                             "pair": {"cd": cd, "st": st,
                                      "err_cd_pct": (cd - mod.SUPERSEDED[0]) / mod.SUPERSEDED[0] * 100.0,
                                      "err_st_pct": (st - mod.SUPERSEDED[1]) / mod.SUPERSEDED[1] * 100.0}}))
    res = mod.audit(stage, "")
    kinds = {(f["severity"], f["kind"]) for f in res["findings"]}
    assert ("advisory", "stored-error-legacy") in kinds

    p.write_text(json.dumps({"D": 64, "re": 100.0, "force_columns": {"n_solid_cells": 4096},
                             "pair": {"cd": cd, "st": st, "err_cd_pct": 9.99, "err_st_pct": 9.99}}))
    res = mod.audit(stage, "")
    kinds = {(f["severity"], f["kind"]) for f in res["findings"]}
    assert ("blocking", "stored-error-drift") in kinds


def test_ledger_scope_mismatch_fires_on_registered_re():
    mod = _load_module()
    res = mod.audit(CASE_DIR, "指标: St=fD/U 与 Cd 均值 vs Norberg1993 (Re=200, 阻塞比 1/20)")
    kinds = {f["kind"] for f in res["findings"]}
    assert "anchor-scope-mismatch" in kinds
    # ...and stays quiet when no acceptance text is supplied (nothing to compare)
    res2 = mod.audit(CASE_DIR, "")
    assert "anchor-scope-mismatch" not in {f["kind"] for f in res2["findings"]}