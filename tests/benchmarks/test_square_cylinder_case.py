"""Regression tests for CASE-LBM-SQUARE-CYL-NORBERG (audit m16183 remediation, item 5).

Audit m16183 (verdict: reopened / request changes) found that this case had
**no test at all** (``grep -rl square_cylinder tests/`` was empty) and that the
in-repo verifier could not even run on the current tree.  These tests are the
machine referee for the five remediation items:

* item 2 (verifier runnable on the current tree)  -> ``test_verify_passes_on_committed_archive``
* item 3 (one reference convention, no silent fork) -> ``test_refspec_is_the_only_constant_source``,
  ``test_reference_drift_is_a_hard_failure``, ``test_undeclared_error_convention_fails``
* item 3b (acceptance metric names its force column) -> ``test_force_column_policy_is_material``
* item 4 (fresh re-run wiring) -> ``test_repro_gate_fires_when_supplied``
* item 1 (registry row) -> ``test_registry_sql_matches_refspec``

They are stdlib + subprocess only (no torch), and every failure mode is exercised
by tampering a temporary copy of the archive, so a passing run cannot be vacuous.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CASE_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "verified" / "square_cylinder"
sys.path.insert(0, str(CASE_DIR))

import refspec as R  # noqa: E402

PRODUCER_AND_VERIFIERS = ["run.py", "verify.py", "verify_traces.py"]


def _run_verify(stage: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CASE_DIR / "verify.py"), str(stage), *extra],
        capture_output=True,
        text=True,
        timeout=300,
    )


def _copy_archive(tmp_path: Path) -> Path:
    stage = tmp_path / "archive"
    stage.mkdir()
    for p in CASE_DIR.glob("case_D*.json"):
        shutil.copy(p, stage / p.name)
    shutil.copy(CASE_DIR / "result.json", stage / "result.json")
    return stage


# --------------------------------------------------------------------------- #
# item 3: one reference convention
# --------------------------------------------------------------------------- #
def test_refspec_is_the_only_constant_source() -> None:
    """No producer/verifier may define REF_CD/REF_ST itself (audit m16183 item 3)."""
    for name in PRODUCER_AND_VERIFIERS:
        src = (CASE_DIR / name).read_text()
        assert re.search(r"^\s*(REF_CD|REF_ST)\s*=\s*[0-9]", src, flags=re.M) is None, (
            f"{name} re-defines a reference constant; import it from refspec.py instead"
        )
        assert ("import refspec" in src) or ("from refspec import" in src), f"{name} does not import refspec"


def test_locked_constants_and_bands_are_consistent() -> None:
    assert R.rel_err_pct(R.REF_CD, R.REF_CD) == 0.0
    assert R.within_band(R.REF_CD, R.CD_BAND), "locked Cd must lie inside its own band"
    assert R.within_band(R.REF_ST, R.ST_BAND), "locked St must lie inside its own band"
    # the superseded pair must be distinguishable from the locked pair
    assert abs(R.SUPERSEDED_REF["cd"] - R.REF_CD) > 0 and abs(R.SUPERSEDED_REF["st"] - R.REF_ST) > 0
    # the locked pair is the centre of the published cluster
    lo, hi = R.CD_BAND
    assert abs(R.REF_CD - 0.5 * (lo + hi)) < 0.05
    lo, hi = R.ST_BAND
    assert abs(R.REF_ST - 0.5 * (lo + hi)) < 5e-3


def test_force_column_policy_is_material() -> None:
    """The locked primary column must be the one that passes; ``wet`` must not be scored."""
    rec = R.normalize_case(CASE_DIR / "case_D64.json")
    assert R.PRIMARY_COLUMN == rec["primary"]
    wet_cd = rec["columns"]["wet"]["cd"]
    pair_cd = rec["columns"]["pair"]["cd"]
    wet_err = abs(R.rel_err_pct(wet_cd, R.REF_CD))
    pair_err = abs(R.rel_err_pct(pair_cd, R.REF_CD))
    assert pair_err <= R.TOL_PCT < wet_err, (
        "force-column policy is not material: "
        f"pair err {pair_err:.2f}% vs wet err {wet_err:.2f}% (tol {R.TOL_PCT}%)"
    )


# --------------------------------------------------------------------------- #
# item 2: the bundled verifier runs and passes on the committed archive
# --------------------------------------------------------------------------- #
def test_archive_normalizes_both_schemas() -> None:
    cases = R.load_archive(CASE_DIR)
    assert {32, 48, 64} <= set(cases)
    assert cases[32]["schema"] == "summary"
    assert cases[64]["schema"] == "triple"
    assert cases[64]["surface_links"] == 12 * 64 - 4, "n_links_wet spelling must be normalised"
    for d, rec in cases.items():
        assert rec["primary_cd"] > 0 and rec["primary_st"] > 0, d


def test_verify_passes_on_committed_archive() -> None:
    r = _run_verify(CASE_DIR)
    assert r.returncode == 0, f"verify.py failed on the committed archive:\n{r.stdout}\n{r.stderr}"
    assert "VERIFY RESULT: ALL OK" in r.stdout
    assert "FAIL 0" in r.stdout


def test_committed_measurements_meet_the_bar() -> None:
    """The headline claim, asserted directly (independent of verify.py's own output)."""
    cases = R.load_archive(CASE_DIR)
    for d in (32, 48, 64):
        ecd = abs(R.rel_err_pct(cases[d]["primary_cd"], R.REF_CD))
        est = abs(R.rel_err_pct(cases[d]["primary_st"], R.REF_ST))
        assert ecd <= R.TOL_PCT, f"D{d} Cd error {ecd:.2f}% > {R.TOL_PCT}%"
        assert est <= R.TOL_PCT, f"D{d} St error {est:.2f}% > {R.TOL_PCT}%"


# --------------------------------------------------------------------------- #
# item 3: convention drift must FAIL, not pass silently
# --------------------------------------------------------------------------- #
def test_reference_drift_is_a_hard_failure(tmp_path: Path) -> None:
    """Rewriting a stored cd_ref to the superseded constant must be caught."""
    stage = _copy_archive(tmp_path)
    p = stage / "case_D32.json"
    d = json.loads(p.read_text())
    d["cd_ref"] = R.SUPERSEDED_REF["cd"]
    p.write_text(json.dumps(d, indent=2))
    r = _run_verify(stage)
    assert r.returncode == 1, "verify.py accepted a stored cd_ref from the superseded convention"
    assert "FAIL" in r.stdout


def test_undeclared_error_convention_fails(tmp_path: Path) -> None:
    stage = _copy_archive(tmp_path)
    p = stage / "case_D48.json"
    d = json.loads(p.read_text())
    d["err_pct"] = float(d["err_pct"]) + 2.5  # matches neither convention
    p.write_text(json.dumps(d, indent=2))
    r = _run_verify(stage)
    assert r.returncode == 1
    assert "attributable to a named convention" in r.stdout


def test_legacy_error_convention_is_declared(tmp_path: Path) -> None:
    """case_D64.json's stored errors are the superseded convention — declared, not scored."""
    rec = R.normalize_case(CASE_DIR / "case_D64.json")
    stored = rec["stored_err_pct"]
    assert R.convention_for(rec["primary_cd"], rec["primary_st"], stored["cd"], stored["st"]) == "superseded"
    r = _run_verify(CASE_DIR)
    assert "DECLARED LEGACY" in r.stdout


# --------------------------------------------------------------------------- #
# item 4: reproduction gate wiring (fires only when a fresh run is supplied)
# --------------------------------------------------------------------------- #
def test_repro_gate_fires_when_supplied(tmp_path: Path) -> None:
    repro = tmp_path / "repro"
    repro.mkdir()
    src = json.loads((CASE_DIR / "case_D32.json").read_text())
    src["cd"] = round(src["cd"] * 1.02, 4)  # +2% > 0.5% repro tolerance
    (repro / "case_D32.json").write_text(json.dumps(src, indent=2))
    r = _run_verify(CASE_DIR, "--repro-dir", str(repro))
    assert r.returncode == 1
    assert "fresh re-run vs replaced archive" in r.stdout

    # a matching re-run must pass the same gate
    repro2 = tmp_path / "repro_ok"
    repro2.mkdir()
    (repro2 / "case_D32.json").write_text(json.dumps(json.loads((CASE_DIR / "case_D32.json").read_text()), indent=2))
    r2 = _run_verify(CASE_DIR, "--repro-dir", str(repro2))
    assert "fresh re-run vs replaced archive" in r2.stdout
    assert "[OK  ] D32: fresh re-run vs replaced archive" in r2.stdout


# --------------------------------------------------------------------------- #
# item 1: machine referee row
# --------------------------------------------------------------------------- #
def test_registry_sql_matches_refspec() -> None:
    """The benchmark-case registration SQL must carry the locked constants and columns."""
    sql = (CASE_DIR / "register_case.sql").read_text()
    assert "CASE-LBM-SQUARE-CYL-NORBERG" in sql
    assert f"{R.REF_CD}" in sql and f"{R.REF_ST}" in sql
    assert f"{R.TOL_PCT}" in sql
    assert R.PRIMARY_COLUMN in sql
    assert "tensor_benchmark_cases" in sql
    assert ("ON CONFLICT" in sql) or ("DELETE FROM tensor_benchmark_cases" in sql), (
        "registration must be idempotent (ON CONFLICT or delete-then-insert)"
    )