"""Tests for the fail-closed ladder assembly and the shipped ladder artifacts.

The artifact tests are skipped when ``artifacts/in12_s6`` is absent (it is
produced by ``benchmarks/in12_s6_wall_ladder/build_ladder.py``), but when present
every criterion is recomputed from its own recorded value and threshold, so a
tampered or hand-edited verdict cannot pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorlbm.validation import Criterion, assemble_ladder, criterion, verdict

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "in12_s6"
LADDER_JSON = ARTIFACT_DIR / "ladder.json"
DOC_PATH = REPO_ROOT / "docs" / "IN12_S6_WALL_LADDER.md"


def test_criterion_evaluation_directions() -> None:
    assert criterion("a", "d", 2.0, 3.0, "<=").passed
    assert not criterion("b", "d", 4.0, 3.0, "<=").passed
    assert criterion("c", "d", -5.0, 3.0, "abs>=").passed
    assert criterion("e", "d", 50.0, (30.0, 150.0), "in_band").passed
    assert not criterion("f", "d", 5000.0, (30.0, 150.0), "in_band").passed
    assert criterion("g", "d", 2.5e4, 100.0, ">=").passed


def test_criterion_is_fail_closed_for_missing_or_non_finite_values() -> None:
    assert not criterion("a", "d", None, 3.0, "<=").passed
    assert not criterion("b", "d", float("nan"), 3.0, "<=").passed
    with pytest.raises(ValueError):
        criterion("c", "d", 1.0, 2.0, "approximately")


def test_ladder_verdict_is_fail_closed() -> None:
    passing = verdict([criterion("a", "d", 1.0, 2.0, "<=")])
    assert passing["verdict"] == "PASS"
    failing = verdict(
        [criterion("a", "d", 1.0, 2.0, "<="), criterion("b", "d", 3.0, 2.0, "<=")]
    )
    assert failing["verdict"] == "FAIL"
    assert failing["failed_criteria"] == ["b"]
    assert verdict([])["verdict"] == "FAIL"


def test_assemble_ladder_propagates_case_failure() -> None:
    payload = assemble_ladder(
        {
            "case_ok": {"criteria": [criterion("a", "d", 1.0, 2.0, "<=")]},
            "case_bad": {"criteria": [criterion("b", "d", 5.0, 2.0, "<=")]},
        },
        metadata={"node": "test"},
    )
    assert payload["overall_verdict"] == "FAIL"
    assert payload["failed_cases"] == ["case_bad"]
    assert payload["cases"]["case_ok"]["verdict"] == "PASS"
    assert payload["cases"]["case_bad"]["verdict"] == "FAIL"


def _recompute(item: dict) -> bool:
    value = item["value"]
    threshold = item["threshold"]
    comparison = item["comparison"]
    if value is None or threshold is None:
        return False
    if comparison == "<=":
        return value <= threshold
    if comparison == "<":
        return value < threshold
    if comparison == ">=":
        return value >= threshold
    if comparison == ">":
        return value > threshold
    if comparison == "abs<=":
        return abs(value) <= threshold
    if comparison == "abs>=":
        return abs(value) >= threshold
    if comparison == "in_band":
        return threshold[0] <= value <= threshold[1]
    raise AssertionError(f"unexpected comparison {comparison}")


@pytest.mark.skipif(not LADDER_JSON.exists(), reason="ladder artifact not built")
def test_shipped_ladder_recomputes_and_is_fail_closed() -> None:
    payload = json.loads(LADDER_JSON.read_text(encoding="utf-8"))
    assert payload["schema"] == "tensorlbm.wall-ladder/v1"
    all_passed = True
    for case_name, case in payload["cases"].items():
        assert case["criteria"], f"{case_name} has no criteria"
        case_failures = []
        for item in case["criteria"]:
            assert _recompute(item) == item["passed"], (
                f"{case_name}/{item['name']} recorded pass={item['passed']} but "
                f"recomputes to {_recompute(item)}"
            )
            if not item["passed"]:
                case_failures.append(item["name"])
        expected = "PASS" if not case_failures else "FAIL"
        assert case["verdict"] == expected
        assert sorted(case_failures) == sorted(case["failed_criteria"])
        all_passed &= expected == "PASS"
    assert payload["overall_verdict"] == ("PASS" if all_passed else "FAIL")


@pytest.mark.skipif(not ARTIFACT_DIR.exists(), reason="ladder artifacts not built")
def test_channel_artifacts_record_provenance_and_gci() -> None:
    channel = json.loads((ARTIFACT_DIR / "channel_turb_3d.json").read_text(encoding="utf-8"))
    assert len(channel["levels"]) == 3
    for level in channel["levels"]:
        assert len(level["source_sha256"]) == 64
        assert level["rms_y30_100_pct"] > 0.0
    # monotonically improving profile error
    rms = [level["rms_y30_100_pct"] for level in channel["levels"]]
    assert rms == sorted(rms, reverse=True)
    for name in ("u_tau", "u_plus_at_y50", "bulk_velocity"):
        report = channel["gci"][name]
        assert report["apparent_order"] is not None
        assert report["gci_fine_pct"] is not None
    assert channel["rms_decay_fit"]["exponent"] > 0.0


@pytest.mark.skipif(not ARTIFACT_DIR.exists(), reason="ladder artifacts not built")
def test_wall_function_calibration_band_is_physical() -> None:
    calibration = json.loads(
        (ARTIFACT_DIR / "wall_function_calibration.json").read_text(encoding="utf-8")
    )
    band_low, band_high = calibration["band_checked"]
    assert band_low == pytest.approx(30.0)
    assert band_high > band_low
    # every reference case must contain the three wall laws and a y+ curve
    for case_name, case in calibration["cases"].items():
        assert set(case["curves"]) == {"spalding", "log", "reichardt"}
        for rows in case["curves"].values():
            assert len(rows) >= 20
            y_plus_values = [row["y_plus"] for row in rows]
            assert y_plus_values == sorted(y_plus_values)
    # Spalding and Reichardt must disagree inside the buffer layer but converge
    # in the log layer: the error is not monotone in y+.
    dns = calibration["cases"]["dns_180"]["curves"]
    spalding_errors = [abs(row["cf_error_pct"]) for row in dns["spalding"]]
    assert max(spalding_errors) > 5.0


@pytest.mark.skipif(not ARTIFACT_DIR.exists(), reason="ladder artifacts not built")
def test_negative_control_supports_the_wall_function_claim() -> None:
    control = json.loads((ARTIFACT_DIR / "negative_control.json").read_text(encoding="utf-8"))
    ratios = [row["cell_ratio"] for row in control["rows"]]
    assert ratios == sorted(ratios)
    assert ratios[-1] > 100.0
    for row in control["rows"]:
        assert row["wall_resolved"]["total_cells"] > row["wall_modeled"]["total_cells"]


@pytest.mark.skipif(not LADDER_JSON.exists(), reason="ladder artifact not built")
def test_document_quotes_the_artifact_numbers() -> None:
    assert DOC_PATH.exists()
    document = DOC_PATH.read_text(encoding="utf-8")
    payload = json.loads(LADDER_JSON.read_text(encoding="utf-8"))
    # the six report elements must be present
    for heading in (
        "## 1. 案例介绍",
        "## 2. 怎么做的",
        "## 3. 计算结果",
        "## 4. 结果图",
        "## 5. 注意点",
        "## 6. 误差判定",
    ):
        assert heading in document
    # every figure recorded in the ladder metadata is embedded
    for figure in payload["metadata"]["figures"]:
        assert figure["path"] in document
        assert (REPO_ROOT / figure["path"]).exists()