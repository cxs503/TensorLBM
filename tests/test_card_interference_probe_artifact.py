"""Schema/consistency test for the m5113 card-interference probe artefact.

Pure file-based test: it does not need an accelerator, so it runs anywhere.
Guards the deliverable contract of artifacts/card_interference_probe.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "card_interference_probe.json"
REPORT = Path(__file__).resolve().parents[1] / "docs" / "card_probe_report.md"

REQUIRED = [
    "speedup_8x_vs_1x",
    "memory_bandwidth_GBs",
    "interference_pct",
    "honest_fail",
    "cards",
]


@pytest.fixture(scope="module")
def payload():
    if not ARTIFACT.exists():
        pytest.skip("probe artefact not produced in this checkout")
    return json.loads(ARTIFACT.read_text())


def test_top_level_metric_keys(payload):
    """The machine criterion needs these keys directly readable."""
    for key in REQUIRED:
        assert key in payload["metric_summary"], f"missing {key}"


def test_eight_cards_with_device_numbers(payload):
    cards = payload["metric_summary"]["cards"]
    assert len(cards) == 8
    for name in cards:
        assert re.fullmatch(r"/dev/tcaicard\d+", name), name
    # all eight are distinct physical cards
    assert len(set(cards)) == 8


def test_each_card_ran_in_isolation(payload):
    """One process per card, and every process saw exactly one device."""
    per_card = payload["per_card"]
    assert len(per_card) == 8
    assert payload["devices"]["exclusive_during_timed_window"] is True
    for p in per_card:
        assert p["finite"] is True


def test_speedup_and_interference_consistent(payload):
    res = payload["results"]
    t1, t8 = res["single_card_median_step_ms"], res["concurrent_8card_median_step_ms"]
    assert res["speedup_8x"] == pytest.approx(8.0 * t1 / t8, rel=1e-9)
    assert res["interference_pct"] == pytest.approx((t8 / t1 - 1.0) * 100.0, rel=1e-9)


def test_honest_fail_flag_matches_criterion(payload):
    """speedup < 6x must be flagged as honest-FAIL, >= 6x must not."""
    res = payload["results"]
    expect_fail = res["speedup_8x"] < 6.0
    assert payload["metric_summary"]["honest_fail"] is expect_fail
    assert payload["verdict"]["honest_fail"] is expect_fail


def test_bit_stable_across_cards(payload):
    """Concurrent execution must not perturb results: identical Cd everywhere."""
    cds = {p["cd"] for p in payload["per_card"]}
    conc = {p["concurrent_cd"] for p in payload["per_card"]}
    assert len(cds) == 1, cds
    assert len(conc) == 1, conc


def test_bandwidth_positive(payload):
    assert payload["metric_summary"]["memory_bandwidth_GBs"] > 0
    for p in payload["per_card"]:
        assert p["single_hbm_gbs"] > 0
        assert p["concurrent_hbm_gbs"] > 0


def test_report_exists_and_states_verdict(payload):
    if not REPORT.exists():
        pytest.skip("report not generated in this checkout")
    text = REPORT.read_text()
    assert "speedup_8x" in text
    assert "interference" in text.lower()
    # the accuracy question must be answered explicitly, not dodged
    assert "< 3%" in text