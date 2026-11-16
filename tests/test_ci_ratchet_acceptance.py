"""Acceptance tests for the rung-1 ratchet ledger (XVAL-LBM-GITLAB-CI-STAGED-GATES).

Rung 1 of the ALT-2 ladder (``lint`` / ``format-check``) is a *ratchet*: the
aggregate rule-debt may never grow, and the frozen baseline may only be raised
through a signed, attributable acceptance record.  These tests pin that
discipline:

1. the committed ledger chain is arithmetically intact (floor + signed drift ==
   baseline, no chain break, no empty reason);
2. every accepted increase is confined to files that did *not* exist at the
   baseline commit, and every such file names the commit that introduced it
   (so the debt has an owner instead of being anonymous);
3. the gate itself is *not* weakened -- ``evaluate`` still fails on any growth
   the ledger does not explain, and ``audit_ledger`` reports tampering.

Run: ``pytest tests/test_ci_ratchet_acceptance.py -q``
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "tools" / "ci_ratchet_baseline.json"


@pytest.fixture(scope="module")
def ratchet():
    """Import ``tools/ci_ratchet.py`` without requiring ``tools`` to be a package."""
    spec = importlib.util.spec_from_file_location(
        "ci_ratchet_under_test", REPO_ROOT / "tools" / "ci_ratchet.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text())


def _entry(mode: str = "ruff", **overrides) -> dict:
    entry = {
        "mode": mode,
        "ref_from": "a" * 40,
        "ref_to": "b" * 9,
        "recorded_at": "2026-11-16T00:00:00Z",
        "reason": "later merged work added files after the baseline commit",
        "from_total": 100,
        "to_total": 110,
        "delta": 10,
        "new_debt_files": {"x.py": 3},
        "introduced_by": {"x.py": "abcdef  m1: deliverable"},
    }
    entry.update(overrides)
    return entry


# --------------------------------------------------------------------------- #
# 1. the committed ledger explains the committed baseline
# --------------------------------------------------------------------------- #
def test_committed_ledger_chain_is_intact(ratchet, baseline: dict) -> None:
    ledger = ratchet.audit_ledger(baseline)
    assert ledger["problems"] == []
    assert ledger["ok"] is True
    assert ledger["n_acceptances"] >= 1, "the rung-1 restore must be recorded, not silent"


def test_committed_baseline_matches_signed_drift(ratchet, baseline: dict) -> None:
    ledger = ratchet.audit_ledger(baseline)
    for mode, payload in ledger["modes"].items():
        floor = baseline["floor"][mode]
        assert payload["chain"][0][0] == floor
        assert baseline[mode]["total"] - floor == payload["delta"]
        assert payload["delta"] == sum(
            e["delta"] for e in baseline["acceptances"] if e["mode"] == mode
        )


def test_acceptances_are_attributable(baseline: dict) -> None:
    for entry in baseline["acceptances"]:
        assert entry["reason"].strip(), "an accepted increase must state why"
        assert entry["delta"] > 0
        assert entry["to_total"] - entry["from_total"] == entry["delta"]
        files = entry["new_debt_files"]
        assert files, "an acceptance must name the files that carry the new debt"
        assert set(files) == set(entry["introduced_by"]), "every debt file needs an owner"
        assert all(v and v != "?" for v in entry["introduced_by"].values())
        if entry.get("by_code_delta"):
            assert sum(entry["by_code_delta"].values()) == entry["delta"]


@pytest.mark.parametrize("mode", ["ruff", "format"])
def test_accepted_files_did_not_exist_at_the_baseline_commit(baseline: dict, mode: str) -> None:
    """The policy that makes the increase honest: new files only, never old code."""
    entries = [e for e in baseline["acceptances"] if e["mode"] == mode]
    assert entries, f"no acceptance recorded for {mode}"
    if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
        pytest.skip("git unavailable; provenance cannot be re-derived here")
    for entry in entries:
        for name in entry["new_debt_files"]:
            probe = subprocess.run(
                ["git", "cat-file", "-e", f"{entry['ref_from']}:{name}"],
                cwd=REPO_ROOT,
                capture_output=True,
            )
            assert probe.returncode != 0, (
                f"{name} already existed at {entry['ref_from']}: the increase should have been "
                "fixed, not accepted"
            )


# --------------------------------------------------------------------------- #
# 2. the gate is not weakened by the ledger
# --------------------------------------------------------------------------- #
def test_gate_still_refuses_unexplained_growth(ratchet) -> None:
    base = {"kind": "ruff", "total": 100, "by_code": {"F401": 60}, "ruff_version": "ruff 0.15.0"}
    flat = {"kind": "ruff", "total": 100, "by_code": {"F401": 60}, "ruff_version": "ruff 0.15.0"}
    grown = {"kind": "ruff", "total": 101, "by_code": {"F401": 61}, "ruff_version": "ruff 0.15.0"}
    assert ratchet.evaluate(flat, base)["ok"] is True
    red = ratchet.evaluate(grown, base)
    assert red["ok"] is False
    assert any("grew" in v for v in red["violations"])


def test_audit_detects_a_tampered_baseline(ratchet, baseline: dict) -> None:
    inflated = json.loads(json.dumps(baseline))
    inflated[inflated["acceptances"][0]["mode"]]["total"] += 7
    assert ratchet.audit_ledger(inflated)["problems"], "raising the total must break the chain"


def test_audit_detects_a_chain_break_and_an_empty_reason(ratchet, baseline: dict) -> None:
    broken = json.loads(json.dumps(baseline))
    broken["acceptances"].append(_entry("ruff", from_total=999, to_total=1000))
    assert any("chain break" in p for p in ratchet.audit_ledger(broken)["problems"])

    silent = {"acceptances": [_entry(reason="   ")], "floor": {"ruff": 100}, "ruff": {"total": 110}}
    assert any("empty reason" in p for p in ratchet.audit_ledger(silent)["problems"])


# --------------------------------------------------------------------------- #
# 3. growth attribution (pure function)
# --------------------------------------------------------------------------- #
def test_classify_growth_separates_drift_from_regression(ratchet) -> None:
    current = {"total": 6, "files": {"old.py": 3, "new.py": 2, "fixed.py": 1}}
    reference = {"total": 3, "files": {"old.py": 2, "fixed.py": 2}}
    known = {"old.py", "fixed.py"}
    out = ratchet.classify_growth(current, reference, known)
    assert out["new_debt_files"] == {"new.py": 2}
    assert out["pre_existing_regressions"] == {"old.py": {"at_ref": 2, "now": 3}}
    assert out["fixed_files"] == {"fixed.py": {"at_ref": 2, "now": 1}}
    assert out["ok"] is False, "a pre-existing file that grew must block acceptance"


def test_classify_growth_accepts_purely_new_files(ratchet) -> None:
    current = {"total": 4, "files": {"new_a.py": 3, "new_b.py": 1}}
    reference = {"total": 0, "files": {}}
    out = ratchet.classify_growth(current, reference, known_paths=set())
    assert out["ok"] is True
    assert out["new_debt_files"] == {"new_a.py": 3, "new_b.py": 1}
    assert out["pre_existing_regressions"] == {}


def test_baseline_payload_drops_per_file_detail(ratchet) -> None:
    measurement = {
        "kind": "ruff",
        "total": 1,
        "by_code": {"F401": 1},
        "files": {"a.py": 1},
        "paths": ["src"],
        "ruff_version": "ruff 0.15.0",
    }
    stripped = ratchet.strip_for_baseline(measurement)
    assert "files" not in stripped
    assert stripped["total"] == 1


# --------------------------------------------------------------------------- #
# 4. CLI wiring
# --------------------------------------------------------------------------- #
def test_cli_audit_passes_on_the_committed_baseline() -> None:
    proc = subprocess.run(
        [sys.executable, "tools/ci_ratchet.py", "ruff", "--audit"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout


def test_cli_accept_drift_requires_a_reason() -> None:
    proc = subprocess.run(
        [sys.executable, "tools/ci_ratchet.py", "ruff", "--accept-drift"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "--reason" in proc.stdout + proc.stderr
