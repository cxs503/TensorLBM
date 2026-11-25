"""Soundness of the calibrated contact-angle inversion (Huang-2007 xval, hold-out round).

Two things are asserted on numbers, not merely executed:

1. ``inversion_branches`` / ``is_monotone`` expose a **non-monotone** calibration
   instead of letting ``invert`` silently return one of several solutions.
2. The committed hold-out evidence
   (``benchmarks/xval_huang2007_contact_angle_inversion/holdout/``) really satisfies
   the owner 15 % gate on every measured row.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorlbm.contact_angle_inversion import (
    CalibrationResult,
    forward,
    invert,
    inversion_branches,
    is_monotone,
    young_prediction,
)

REPO = Path(__file__).resolve().parents[1]
XVAL = REPO / "benchmarks" / "xval_huang2007_contact_angle_inversion" / "holdout"


def _cal(fw, cos, psi_l=0.8587, psi_v=0.1475) -> CalibrationResult:
    cal = CalibrationResult(nx=64, ny=32, radius=8.0, steps=1, wall_rows=2,
                            psi_l=psi_l, psi_v=psi_v)
    cal.fw_values = list(fw)
    cal.cos_values = list(cos)
    cal.valid_range = (min(fw), max(fw))
    return cal


def test_monotone_table_is_single_valued():
    fw = [0.3, 0.4, 0.5, 0.6, 0.7]
    cos = [0.8, 0.5, 0.0, -0.5, -0.9]  # strictly decreasing
    cal = _cal(fw, cos)
    assert is_monotone(cal) is True
    for theta in (40.0, 70.0, 90.0, 120.0):
        br = inversion_branches(cal, theta)
        assert len(br) == 1
        # round trip through the same interpolation it inverted
        assert math.isclose(forward(cal, br[0]), theta, abs_tol=1e-6)


def test_non_monotone_table_yields_several_branches():
    # the measured coarse sweep of m18090 has cos rising between f_w 0.60 and 0.65
    fw = [0.55, 0.60, 0.65, 0.70]
    cos = [-0.4075, -0.0899, -0.1021, +0.2451]
    cal = _cal(fw, cos)
    assert is_monotone(cal) is False

    theta = 95.5  # cos = -0.0958 sits inside the dip -> three consistency branches
    br = inversion_branches(cal, theta)
    assert len(br) == 3, br
    assert br[0] < br[1] < br[2]

    assert invert(cal, theta) == pytest.approx(br[0])                 # historical default
    assert invert(cal, theta, on_ambiguous="last") == pytest.approx(br[-1])
    assert invert(cal, theta, on_ambiguous="mean") == pytest.approx(0.5 * (br[0] + br[-1]))
    assert math.isnan(invert(cal, theta, on_ambiguous="nan"))         # refuse to guess


def test_ambiguity_selector_rejects_unknown_value():
    cal = _cal([0.55, 0.60, 0.65, 0.70], [-0.4075, -0.0899, -0.1021, 0.2451])
    with pytest.raises(ValueError):
        invert(cal, 95.5, on_ambiguous="second")


def test_target_outside_measured_range_falls_back_or_refuses():
    cal = _cal([0.5, 0.6, 0.7], [-0.2, -0.1, 0.1])
    theta = 160.0  # cos = -0.94, far outside the measured window
    assert inversion_branches(cal, theta) == []
    assert math.isnan(invert(cal, theta))
    fw_young = invert(cal, theta, fallback_to_young=True)
    cos_t = math.cos(math.radians(theta))
    expected = (cal.psi_v + cos_t * (cal.psi_l - cal.psi_v)) / cal.psi_l
    assert math.isclose(fw_young, expected, rel_tol=1e-12)
    # ... and the Young prediction of that f_w reproduces the requested angle
    assert math.isclose(young_prediction(fw_young, cal.psi_l, cal.psi_v), theta, abs_tol=1e-6)


def test_committed_coarse_sweep_is_non_monotone():
    path = XVAL / "evidence" / "cal_coarse.json"
    if not path.exists():
        pytest.skip("committed coarse sweep not present")
    payload = json.loads(path.read_text())
    d = payload["calibration"]
    cal = _cal(d["fw_values"], d["cos_values"], d["psi_l"], d["psi_v"])
    assert is_monotone(cal) is False
    assert payload["monotone"] is False
    # the dip: cos at f_w=0.65 is *above* cos at f_w=0.60 -> the inversion of
    # theta in the dip has more than one solution
    i60 = d["fw_values"].index(0.60)
    i65 = d["fw_values"].index(0.65)
    assert d["cos_values"][i65] < d["cos_values"][i60]
    theta_dip = math.degrees(math.acos(d["cos_values"][i60])) + 0.3
    assert len(inversion_branches(cal, theta_dip)) >= 2

def test_committed_evidence_holds_the_gate():
    summary_path = XVAL / "summary.json"
    if not summary_path.exists():
        pytest.skip("hold-out summary.json not present")
    summary = json.loads(summary_path.read_text())
    rows = summary["rows"]
    assert rows, "no rows in committed summary"
    measured = [r for r in rows if r.get("status") == "ok"]
    assert len(measured) >= 8
    bad = [(r["row_id"], r["err_pct"]) for r in measured if r["err_pct"] >= summary["gate_pct"]]
    assert not bad, f"rows outside the {summary['gate_pct']}% gate: {bad}"
    # every arm must actually be measured (an unmeasured arm is not a pass)
    for arm in ("A_inband_coarse", "B_fine_transfer", "C_tau_transfer"):
        arm_rows = [r for r in rows if r["arm"] == arm]
        assert arm_rows and all(r.get("status") == "ok" for r in arm_rows), arm