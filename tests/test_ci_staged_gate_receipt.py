"""Acceptance tests for the committed staged-gate receipt (anti-staleness).

The receipt in ``docs/ci/staged_gate_semantics_receipt.json`` is evidence: it
claims that the GitLab ladder in ``.gitlab-ci.yml`` satisfies C1-C4 and 67
structural/artefact contracts.  Evidence that nobody re-checks is folklore, so
these tests pin the two properties that make the receipt usable at all:

* **identity** -- the receipt names the exact source bytes (repo-relative path
  + sha256), so any checkout can tell *which* pipeline it describes;
* **freshness** -- ``tools/verify_ci_staged_gates.py --check`` re-derives the
  receipt from today's ``.gitlab-ci.yml`` and fails on any disagreement.

Every freshness test is paired with a *negative* one (mutate the pipeline or
the document, watch the gate turn red), so the checks cannot pass vacuously.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "verify_ci_staged_gates.py"
CI = REPO_ROOT / ".gitlab-ci.yml"
RECEIPT = REPO_ROOT / "docs" / "ci" / "staged_gate_semantics_receipt.json"
RECEIPT_MD = REPO_ROOT / "docs" / "ci" / "staged_gate_semantics_receipt.md"
DESIGN_DOC = REPO_ROOT / "docs" / "ci" / "gitlab_ci_staged_gates.md"


def run_tool(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the verifier CLI from the repo root (same call CI makes)."""
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.fixture(scope="module")
def receipt() -> dict:
    return json.loads(RECEIPT.read_text())


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #
def test_receipt_records_the_exact_source_bytes(receipt: dict) -> None:
    """The receipt must name the bytes it was derived from."""
    assert receipt["source"] == ".gitlab-ci.yml", receipt["source"]
    assert receipt["source_sha256"] == hashlib.sha256(CI.read_bytes()).hexdigest()


def test_receipt_leaks_no_absolute_path(receipt: dict) -> None:
    """Regression: the receipt once recorded a foreign agent worktree path.

    A committed artefact with ``/root/...`` in it is neither reproducible nor
    reviewable -- the previous revision carried
    ``/srv/tensor/agent/work/m22498/.gitlab-ci.yml``, so the receipt could not be
    compared on any other checkout.
    """
    assert not receipt["source"].startswith("/")
    for leaked in ("/root/", "tp-agent", "m22498", "\\"):
        assert leaked not in RECEIPT.read_text()
    # the markdown twin carries the abbreviated (16 hex char) fingerprint
    assert receipt["source_sha256"][:16] in RECEIPT_MD.read_text()


# --------------------------------------------------------------------------- #
# freshness (positive)
# --------------------------------------------------------------------------- #
def test_committed_receipt_is_fresh() -> None:
    proc = run_tool("--check", str(RECEIPT))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "STALE" not in proc.stdout


def test_design_doc_ladder_table_is_fresh() -> None:
    proc = run_tool("--check-doc", str(DESIGN_DOC))
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_receipt_inventory_matches_the_pipeline(receipt: dict) -> None:
    """The generated ladder inventory must cover every declared stage."""
    ladder = receipt["ladder"]
    assert [stage["stage"] for stage in ladder] == receipt["stages"]
    named = [job["name"] for stage in ladder for job in stage["jobs"]]
    assert sorted(named) == sorted(receipt["gates"] + receipt["advisory"])
    for stage in ladder:
        for job in stage["jobs"]:
            assert job["tier"] in {"blocking", "advisory", "manual"}
            assert job["events"], f"{job['name']} runs on no event"


# --------------------------------------------------------------------------- #
# freshness (negative -- the gate must be able to fail)
# --------------------------------------------------------------------------- #
def test_check_fails_when_the_pipeline_changes(tmp_path: Path) -> None:
    """Mutating the ladder without regenerating the receipt must be caught."""
    mutated = tmp_path / "ci.yml"
    text = CI.read_text()
    # `nightly-full` is the only schedule-only gate; drop its rules so it also
    # runs on merge requests.  One byte-level edit, one semantic change.
    marker = (
        'nightly-full:\n  stage: nightly\n  rules:\n    - if: $CI_PIPELINE_SOURCE == "schedule"\n'
    )
    assert marker in text, "nightly-full rules block moved; update this mutation"
    mutated.write_text(text.replace(marker, "nightly-full:\n  stage: nightly\n", 1))

    proc = run_tool("--ci", str(mutated), "--check", str(RECEIPT))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "STALE" in proc.stdout
    assert "events" in proc.stdout or "ladder" in proc.stdout


def test_check_fails_when_the_tier_changes(tmp_path: Path) -> None:
    """A blocking gate demoted to advisory is a policy change, not a detail."""
    mutated = tmp_path / "ci.yml"
    text = CI.read_text()
    marker = "typecheck-core:\n  extends: .quality-base\n"
    assert marker in text
    body_start = text.index(marker)
    body_end = text.index("allow_failure: false", body_start)
    mutated.write_text(
        text[:body_end] + "allow_failure: true" + text[body_end + len("allow_failure: false") :]
    )

    proc = run_tool("--ci", str(mutated), "--check", str(RECEIPT))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "STALE" in proc.stdout


def test_check_fails_when_the_doc_block_is_edited(tmp_path: Path) -> None:
    """A hand-edited ladder table must be reported, not trusted."""
    doc = tmp_path / "design.md"
    text = DESIGN_DOC.read_text()
    assert "| `test` | 5 |" in text
    doc.write_text(text.replace("| `test` | 5 |", "| `test` | 4 |", 1))

    proc = run_tool("--check-doc", str(doc))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "STALE" in proc.stderr


def test_write_doc_repairs_a_stale_table(tmp_path: Path) -> None:
    """`--write-doc` restores the block, and the repair is idempotent."""
    doc = tmp_path / "design.md"
    doc.write_text(DESIGN_DOC.read_text().replace("| `test` | 5 |", "| `test` | 9 |", 1))

    assert run_tool("--check-doc", str(doc)).returncode == 1
    repair = run_tool("--write-doc", str(doc))
    assert repair.returncode == 0, repair.stdout + repair.stderr
    assert "rewrote" in repair.stdout
    assert run_tool("--check-doc", str(doc)).returncode == 0
    again = run_tool("--write-doc", str(doc))
    assert "unchanged" in again.stdout


def test_environment_dependent_probe_data_is_not_drift(tmp_path: Path) -> None:
    """`greenness` depends on the host's tool set, so it must not fail the check.

    Committed-vs-live equality is only meaningful for the deterministic part of
    the receipt; a receipt with local probe results attached is still fresh.
    """
    committed = json.loads(RECEIPT.read_text())
    committed["greenness"] = {"counts": {"pass": 0, "fail": 9, "not_run": 4}, "per_gate": {}}
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(committed))

    proc = run_tool("--check", str(path))
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_check_reports_a_missing_receipt(tmp_path: Path) -> None:
    proc = run_tool("--check", str(tmp_path / "absent.json"))
    assert proc.returncode == 1
    assert "STALE" in proc.stderr
