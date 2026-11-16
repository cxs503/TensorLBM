"""Regression pins for the lint/format debt ratchet (``tools/lint_ratchet.py``).

The ratchet is the *only* thing standing between the repository and re-rot of
the debt it paid down (ledger item ``LBM-P0-RUFF``), so it is tested as an
invariant, not as a script that merely executes:

* the measurement really counts ruff findings / unformatted files,
* a regression is a non-zero exit, an improvement is a zero exit,
* ``--update`` refuses to raise a baseline,
* the checked-in baseline is respected by the tree as it stands,
* unparsable committed sources stay at zero.
"""

from __future__ import annotations

import ast
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import lint_ratchet  # noqa: E402  (path set above: tools/ is not a package)

RUFF = shutil.which("ruff")
needs_ruff = pytest.mark.skipif(RUFF is None, reason="ruff not installed (dev extra)")

#: Scripts repaired under LBM-P0-RUFF; each was unparsable in the tree and is
#: pinned here so the same breakage cannot silently return.
REPAIRED_SCRIPTS = (
    "benchmarks/contact_line/make_doc.py",
    "benchmarks/in45_channel_turb/plot_campaign.py",
    "benchmarks/in45_hb_turb/run_turbulence_campaign.py",
    "benchmarks/in99_n03_channel_turb/run_validation.py",
)


# --------------------------------------------------------------------------- #
# pure logic
# --------------------------------------------------------------------------- #
def test_regression_is_detected_and_improvement_is_not() -> None:
    baseline = {"ruff_check_findings": 100, "invalid_syntax_findings": 0}
    assert [
        m.name
        for m in lint_ratchet.evaluate(
            {"ruff_check_findings": 101, "invalid_syntax_findings": 0}, baseline
        )
        if m.regressed
    ] == ["ruff_check_findings"]

    improved = lint_ratchet.evaluate(
        {"ruff_check_findings": 97, "invalid_syntax_findings": 0}, baseline
    )
    assert not any(m.regressed for m in improved)
    assert {m.name: m.improvement for m in improved} == {
        "invalid_syntax_findings": 0,
        "ruff_check_findings": 3,
    }


def test_missing_measurement_defaults_to_zero_not_to_a_pass() -> None:
    """A metric absent from the measurement must surface as 0 vs baseline."""
    metrics = lint_ratchet.evaluate({}, {"ruff_check_findings": 7})
    assert metrics[0].current == 0 and metrics[0].baseline == 7
    assert metrics[0].improvement == 7


# --------------------------------------------------------------------------- #
# measurement really measures
# --------------------------------------------------------------------------- #
@needs_ruff
def test_count_lint_counts_actual_ruff_findings(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    (tmp_path / "sample.py").write_text("import os\nimport sys\nx = 1\n", encoding="utf-8")
    monkeypatch.setattr(lint_ratchet, "REPO_ROOT", tmp_path)
    # E401 (multiple imports on one line) + F401 (unused imports x2) + I001.
    assert lint_ratchet.count_lint((str(tmp_path),)) >= 2


@needs_ruff
def test_count_unformatted_detects_a_file_ruff_would_rewrite(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    (tmp_path / "sample.py").write_text("x = 1\ny = 2\nz  =  3\n", encoding="utf-8")
    monkeypatch.setattr(lint_ratchet, "REPO_ROOT", tmp_path)
    assert lint_ratchet.count_unformatted((str(tmp_path),)) == 1


@needs_ruff
def test_count_syntax_errors_flags_an_unparsable_file(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    (tmp_path / "broken.py").write_text('x = "unterminated\n', encoding="utf-8")
    monkeypatch.setattr(lint_ratchet, "REPO_ROOT", tmp_path)
    assert lint_ratchet.count_syntax_errors((str(tmp_path),)) >= 1


# --------------------------------------------------------------------------- #
# CLI behaviour
# --------------------------------------------------------------------------- #
def _write_baseline(path: Path, findings: int) -> None:
    path.write_text(
        json.dumps({"baseline": {"ruff_check_findings": findings}, "measured_with": "test"}),
        encoding="utf-8",
    )


def test_cli_update_refuses_to_raise_baseline(tmp_path: Path, monkeypatch, capsys) -> None:  # noqa: ANN001
    baseline_path = tmp_path / "lint_baseline.json"
    _write_baseline(baseline_path, 10)
    monkeypatch.setattr(lint_ratchet, "BASELINE_PATH", baseline_path)
    monkeypatch.setattr(lint_ratchet, "measure", lambda scope=None: {"ruff_check_findings": 11})
    monkeypatch.setattr(lint_ratchet, "ruff_version", lambda: "test")

    assert lint_ratchet.main(["--update"]) == 1
    assert json.loads(baseline_path.read_text())["baseline"]["ruff_check_findings"] == 10
    assert "refusing to raise baseline" in capsys.readouterr().out


def test_cli_update_locks_in_an_improvement(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    baseline_path = tmp_path / "lint_baseline.json"
    _write_baseline(baseline_path, 10)
    monkeypatch.setattr(lint_ratchet, "BASELINE_PATH", baseline_path)
    monkeypatch.setattr(lint_ratchet, "measure", lambda scope=None: {"ruff_check_findings": 8})
    monkeypatch.setattr(lint_ratchet, "ruff_version", lambda: "test")

    assert lint_ratchet.main(["--update"]) == 0
    assert json.loads(baseline_path.read_text())["baseline"]["ruff_check_findings"] == 8


def test_cli_check_fails_on_regression_and_passes_at_baseline(
    tmp_path: Path, monkeypatch, capsys
) -> None:  # noqa: ANN001
    baseline_path = tmp_path / "lint_baseline.json"
    _write_baseline(baseline_path, 10)
    monkeypatch.setattr(lint_ratchet, "BASELINE_PATH", baseline_path)
    monkeypatch.setattr(lint_ratchet, "ruff_version", lambda: "test")
    monkeypatch.setattr(lint_ratchet, "measure", lambda scope=None: {"ruff_check_findings": 12})
    assert lint_ratchet.main(["--check"]) == 1
    assert "REGRESSED" in capsys.readouterr().out

    monkeypatch.setattr(lint_ratchet, "measure", lambda scope=None: {"ruff_check_findings": 10})
    assert lint_ratchet.main(["--check"]) == 0


# --------------------------------------------------------------------------- #
# invariants on the checked-in tree
# --------------------------------------------------------------------------- #
def test_baseline_file_is_well_formed_and_meets_ledger_target() -> None:
    doc = json.loads(lint_ratchet.BASELINE_PATH.read_text(encoding="utf-8"))
    baseline = doc["baseline"]
    assert set(baseline) == {
        "ruff_check_findings",
        "ruff_format_unformatted_files",
        "invalid_syntax_findings",
    }
    assert doc["ledger_slug"] == "LBM-P0-RUFF"
    # Acceptance for the ledger item: ruff_count <= 3049.
    assert doc["ledger_target"]["ruff_check_findings"] == 3049
    assert baseline["ruff_check_findings"] <= doc["ledger_target"]["ruff_check_findings"]
    assert baseline["invalid_syntax_findings"] == 0


@needs_ruff
def test_repository_respects_its_baseline() -> None:
    doc = json.loads(lint_ratchet.BASELINE_PATH.read_text(encoding="utf-8"))
    metrics = lint_ratchet.evaluate(lint_ratchet.measure(), doc["baseline"])
    regressed = {m.name: (m.current, m.baseline) for m in metrics if m.regressed}
    assert regressed == {}, f"lint/format debt regressed: {regressed}"


@needs_ruff
def test_no_unparsable_committed_sources() -> None:
    assert lint_ratchet.count_syntax_errors() == 0


@pytest.mark.parametrize("relpath", REPAIRED_SCRIPTS)
def test_repaired_benchmark_scripts_parse(relpath: str) -> None:
    path = REPO_ROOT / relpath
    assert path.is_file(), relpath
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
