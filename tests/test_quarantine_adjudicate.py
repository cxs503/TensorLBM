"""Unit contracts for the CI-quarantine adjudication harness.

These tests cover the pure parts of ``tools/quarantine_adjudicate.py``
(parsing, classification, aggregation, rendering).  The pytest-in-pytest part is
deliberately *not* exercised here: the harness runs one node id per subprocess so
a recursive full-suite run cannot happen by accident.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import quarantine_adjudicate as qa  # noqa: E402


SAMPLE = """\
# --- batch 4 (2026-08-19): order-dependent under full-suite in-process run. ---
tests/test_lbm_step.py::TestPerformance::test_mlups_wall_function[D3Q19-19]

# --- batch 5 (2026-08-20): verified failing on stock main. ---
tests/test_free_surface_conversion_density_audit.py::test_actual_b_step
tests/test_free_surface_i_to_g_exact_ledger.py::test_candidates[B-3]

# duplicated id must collapse
tests/test_free_surface_i_to_g_exact_ledger.py::test_candidates[B-3]

# non-node-id lines are ignored
some prose that is not a node id
"""


def test_parse_quarantine_tags_batch_and_dedupes(tmp_path: Path) -> None:
    quarantine = tmp_path / "ci_quarantine.txt"
    quarantine.write_text(SAMPLE, encoding="utf-8")

    entries = qa.parse_quarantine(quarantine)

    assert [entry.nodeid for entry in entries] == [
        "tests/test_lbm_step.py::TestPerformance::test_mlups_wall_function[D3Q19-19]",
        "tests/test_free_surface_conversion_density_audit.py::test_actual_b_step",
        "tests/test_free_surface_i_to_g_exact_ledger.py::test_candidates[B-3]",
    ]
    assert [entry.batch for entry in entries] == [
        "batch 4 (2026-08-19): order-dependent under full-suite in-process run.",
        "batch 5 (2026-08-20): verified failing on stock main.",
        "batch 5 (2026-08-20): verified failing on stock main.",
    ]
    assert entries[1].file == "tests/test_free_surface_conversion_density_audit.py"


def test_classify_pass_now() -> None:
    verdict, signature = qa.classify(0, "1 passed in 0.20s")
    assert verdict == qa.PASS_NOW
    assert "passes" in signature


def test_classify_empty_evidence_from_zero_count_assertion() -> None:
    stdout = "\n".join(
        [
            "    def test_x():",
            ">       assert audit.i_to_g_count > 0",
            "E       assert 0 > 0",
            "E        +  where 0 = audit.i_to_g_count",
        ]
    )
    verdict, signature = qa.classify(1, stdout)
    assert verdict == qa.EMPTY_EVIDENCE
    assert "assert 0 > 0" in signature


def test_classify_numeric_tolerance_assertion() -> None:
    stdout = "\n".join(
        [
            ">       assert measured == pytest.approx(reference, rel=0.05)",
            "E       assert 1.37 == 1.0 ± 5.0e-02 * 1.0",
            "E        +  where 1.0 ± 5.0e-02 * 1.0 = approx(1.0, rel=0.05)",
        ]
    )
    verdict, signature = qa.classify(1, stdout)
    assert verdict == qa.NUMERIC
    # the ledger must keep the measured value, not just "assertion failed"
    assert "1.37" in signature


def test_classify_api_drift() -> None:
    stdout = "\n".join(
        [
            ">       flags = record.conversion_evidence['cells']",
            "E       AttributeError: 'NoneType' object has no attribute 'conversion_evidence'",
        ]
    )
    verdict, signature = qa.classify(1, stdout)
    assert verdict == qa.API_DRIFT
    assert "AttributeError" in signature


def test_classify_collection_error() -> None:
    verdict, signature = qa.classify(2, "ERROR tests/x.py - ImportError: no module named y")
    assert verdict == qa.API_DRIFT
    assert "ImportError" in signature

    verdict, signature = qa.classify(2, "errors during collection\n1 error")
    assert verdict == qa.ERROR


def test_classify_timeout_never_masks_as_failure() -> None:
    verdict, signature = qa.classify(None, "", timed_out=True)
    assert verdict == qa.TIMEOUT
    assert "timeout" in signature


def test_classify_threshold_assertion() -> None:
    stdout = "\n".join(
        [
            ">       assert len(cells) > 0",
            "E       assert 0 > 0",
        ]
    )
    verdict, _ = qa.classify(1, stdout)
    assert verdict in {qa.EMPTY_EVIDENCE, qa.THRESHOLD}


def test_summarise_counts_per_batch_and_lists_candidates() -> None:
    rows = [
        qa.Verdict("tests/a.py::t1", "tests/a.py", "batch 5", qa.PASS_NOW, 0, "ok", 1.0, False),
        qa.Verdict("tests/a.py::t2", "tests/a.py", "batch 5", qa.NUMERIC, 1, "bad", 2.0, False),
        qa.Verdict("tests/b.py::t3", "tests/b.py", "batch 6", qa.TIMEOUT, None, "slow", 300.0, True),
    ]
    summary = qa.summarise(rows)

    assert summary["total"] == 3
    assert summary["counts"][qa.PASS_NOW] == 1
    assert summary["counts"][qa.NUMERIC] == 1
    assert summary["counts"][qa.TIMEOUT] == 1
    assert summary["counts"][qa.ASSERT] == 0
    assert summary["per_batch"]["batch 5"] == {qa.PASS_NOW: 1, qa.NUMERIC: 1}
    assert summary["unquarantine_candidates"] == ["tests/a.py::t1"]
    assert summary["total_seconds"] == pytest.approx(303.0)


def test_select_filters_by_file_substring_then_limits(tmp_path: Path) -> None:
    quarantine = tmp_path / "ci_quarantine.txt"
    quarantine.write_text(SAMPLE, encoding="utf-8")
    entries = qa.parse_quarantine(quarantine)

    free_surface = qa.select(entries, files=["free_surface"], limit=None)
    assert len(free_surface) == 2
    assert all("free_surface" in entry.file for entry in free_surface)

    assert len(qa.select(entries, files=[], limit=2)) == 2
    assert qa.select(entries, files=["nothing_matches"], limit=None) == []


def test_render_markdown_escapes_pipes_and_lists_every_verdict() -> None:
    rows = [
        asdict(
            qa.Verdict(
                "tests/a.py::t1", "tests/a.py", "batch 5", qa.THRESHOLD, 1, "assert a|b", 1.5, False
            )
        )
    ]
    payload = {
        "verdicts": rows,
        "summary": qa.summarise(
            [
                qa.Verdict(
                    "tests/a.py::t1", "tests/a.py", "batch 5", qa.THRESHOLD, 1, "x", 1.5, False
                )
            ]
        ),
    }

    markdown = qa.render_markdown(payload)

    assert "| `THRESHOLD` | 1 |" in markdown
    assert "`tests/a.py::t1`" in markdown
    assert "assert a\\|b" in markdown
    assert json.dumps({"nodeid": rows[0]["nodeid"]})  # row stays json-serialisable