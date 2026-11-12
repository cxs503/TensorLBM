"""Guards for the machine-checked benchmark status ledger (m16798).

The ledger exists because the two human indexes
(`benchmarks/pending/README.md`, `benchmarks/STATUS.md`) and the per-case
machine verdicts on disk (`result.json` / `validation_status.json`) were never
cross-checked. These tests pin the detection logic on synthetic fixtures (so a
regression cannot hide behind the current state of the repo) and then assert
the live invariants of this repository.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import status_ledger as L  # noqa: E402

PENDING_MD = """# pending

| case | status | blocker |
|---|---|---|
| [alpha](alpha/) | ❌ | fails |
| [beta](beta/) | ⏸️ | arched |
| [beta](beta/) | ⏸️ superseded | dup row |
| [gone](gone/) | ❌ | moved away |
"""

STATUS_MD = """# Benchmark status

verified: **99** | pending: **99**  (counts refreshed 1970-01-01)

## verified/

| case | note |
|---|---|
| `v1` | ok |
| `v1` | duplicate |
"""


def _mk(root: Path, bucket: str, name: str, payload=None, readme=True) -> Path:
    d = root / "benchmarks" / bucket / name
    d.mkdir(parents=True, exist_ok=True)
    if payload is not None:
        (d / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    if readme:
        (d / "README.md").write_text("# case\n", encoding="utf-8")
    return d


@pytest.fixture()
def fixture(tmp_path: Path) -> Path:
    (tmp_path / "benchmarks" / "pending").mkdir(parents=True)
    (tmp_path / "benchmarks" / "verified").mkdir(parents=True)
    (tmp_path / "benchmarks" / "pending" / "README.md").write_text(PENDING_MD, encoding="utf-8")
    (tmp_path / "benchmarks" / "STATUS.md").write_text(STATUS_MD, encoding="utf-8")
    _mk(tmp_path, "pending", "alpha", {"verdict": "pass"})  # claim ❌ vs machine pass
    _mk(tmp_path, "pending", "beta", {"verdict": "not_verified"})
    _mk(tmp_path, "pending", "gamma", None)  # unregistered, no verdict
    _mk(tmp_path, "verified", "v1", {"passed": True})
    _mk(tmp_path, "verified", "v2", {"status": "verified"})  # unlisted
    return tmp_path


def test_unregistered_pending_case_detected(fixture: Path):
    led = L.build_ledger(fixture)
    ids = {(v["id"], v["case"]) for v in led["violations"]["hard"]}
    assert ("unregistered_pending_case", "gamma") in ids
    assert ("unregistered_pending_case", "alpha") not in ids


def test_unlisted_verified_case_detected(fixture: Path):
    led = L.build_ledger(fixture)
    ids = {(v["id"], v["case"]) for v in led["violations"]["hard"]}
    assert ("unlisted_verified_case", "v2") in ids
    assert ("unlisted_verified_case", "v1") not in ids


def test_stale_counts_detected_and_refreshed(fixture: Path):
    led = L.build_ledger(fixture)
    assert any(v["id"] == "stale_counts" for v in led["violations"]["hard"])
    L.write_artifacts(led, fixture)
    led2 = L.build_ledger(fixture)
    assert not any(v["id"] == "stale_counts" for v in led2["violations"]["hard"])
    assert (led2["counts"]["advertised_verified"], led2["counts"]["advertised_pending"]) == (2, 3)


def test_soft_debt_duplicate_and_stale_rows(fixture: Path):
    led = L.build_ledger(fixture)
    soft = {(v["id"], v["case"]) for v in led["violations"]["soft"]}
    assert ("duplicate_index_row", "beta") in soft
    assert ("stale_index_row", "gone") in soft


def test_status_claim_conflict_is_reported_not_ignored(fixture: Path):
    """Index says ❌ (fail) but result.json says pass -> soft, explicit."""
    led = L.build_ledger(fixture)
    soft = {(v["id"], v["case"]) for v in led["violations"]["soft"]}
    assert ("status_claim_conflict", "alpha") in soft
    assert ("status_claim_conflict", "beta") not in soft


def test_verdict_extraction_never_guesses(fixture: Path):
    assert L.machine_verdict(fixture / "benchmarks" / "pending" / "alpha")["state"] == "pass"
    assert L.machine_verdict(fixture / "benchmarks" / "pending" / "beta")["state"] == "fail"
    assert L.machine_verdict(fixture / "benchmarks" / "pending" / "gamma")["state"] == "unknown"
    bad = _mk(fixture, "pending", "broken", {"verdict": "some-unheard-of-token"})
    assert L.machine_verdict(bad)["state"] == "unknown"


def test_write_is_idempotent_and_registration_converges(fixture: Path):
    L.write_artifacts(L.build_ledger(fixture), fixture)
    first = (fixture / "benchmarks" / "STATUS.md").read_text(encoding="utf-8")
    pend_first = (fixture / "benchmarks" / "pending" / "README.md").read_text(encoding="utf-8")
    state = L.write_artifacts(L.build_ledger(fixture), fixture)
    assert state["benchmarks/STATUS.md"] == "unchanged"
    assert state["benchmarks/pending/README.md"] == "unchanged"
    assert (fixture / "benchmarks" / "STATUS.md").read_text(encoding="utf-8") == first
    assert (fixture / "benchmarks" / "pending" / "README.md").read_text(encoding="utf-8") == pend_first
    # gamma is registered by the generated block, so the second pass is clean.
    led = L.build_ledger(fixture)
    assert not any(v["id"] == "unregistered_pending_case" for v in led["violations"]["hard"])


def test_live_repo_has_no_hard_violations():
    """Anti-rot gate: the checked-in archives must agree with their indexes."""
    led = L.build_ledger(ROOT)
    assert led["violations"]["hard"] == [], led["violations"]["hard"]
    cnt = led["counts"]
    assert cnt["advertised_verified"] == cnt["verified_dirs"]
    assert cnt["advertised_pending"] == cnt["pending_dirs"]


def test_live_repo_counts_match_directory_listing():
    led = L.build_ledger(ROOT)
    assert led["counts"]["pending_dirs"] == len(L.case_dirs(ROOT, "pending"))
    assert led["counts"]["verified_dirs"] == len(L.case_dirs(ROOT, "verified"))
    # machine verdicts are on disk, not invented
    for c in led["cases"]:
        if c["machine"] != "unknown":
            assert c["machine_source"] and Path(ROOT / c["machine_source"]).is_file()


def test_ledger_json_artifact_is_current():
    """STATUS_LEDGER.json must describe the current tree (regenerate with --write)."""
    path = ROOT / "benchmarks" / "STATUS_LEDGER.json"
    assert path.is_file(), "run python benchmarks/status_ledger.py --write"
    committed = json.loads(path.read_text(encoding="utf-8"))
    live = L.build_ledger(ROOT)
    assert committed["counts"] == live["counts"]
    assert [c["case"] for c in committed["cases"]] == [c["case"] for c in live["cases"]]