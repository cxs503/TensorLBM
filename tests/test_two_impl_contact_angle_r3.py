"""Round-3 verifier tests: the third estimator and the level-sensitivity probe.

These tests bind the audit to ground truth that is known analytically, so a
failure means the *verifier* is wrong, not the subject under audit.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "development" / "verification" / "contact-angle" / "round-3"))

from run_r3 import (  # noqa: E402
    calibrate_synthetic,
    geodesic_angle,
    gas_level,
    level_convention_angles,
    offset_circle_prediction,
    right_crossing,
    symmetry_error,
    synthetic_cap,
)


@pytest.mark.parametrize("theta_deg", [45.0, 70.0, 90.0, 110.0, 135.0])
def test_geodesic_estimator_recovers_synthetic_cap(theta_deg):
    psi = synthetic_cap(theta_deg)
    out = geodesic_angle(psi, xi=2.0)
    assert math.isfinite(out["theta_geodesic_deg"])
    # xi/R = 2/14 sits in the declared coarse tier (0.12 < xi/R <= 0.25)
    assert abs(out["theta_geodesic_deg"] - theta_deg) < 2.0, out


def test_estimator_calibration_over_the_declared_synthetic_matrix():
    """Two declared tiers: 0.5 deg for xi/R <= 0.12, 2.0 deg up to xi/R = 0.25."""
    cal = calibrate_synthetic()
    assert cal["n"] == 30
    assert cal["max_dev_fine_deg"] < 0.5, cal["max_dev_fine_deg"]
    assert cal["max_dev_coarse_deg"] < 2.0, cal["max_dev_coarse_deg"]
    assert cal["n_tier_violations"] == 0, cal["tier_violations"]
    assert cal["median_dev_deg"] < 0.2, cal["median_dev_deg"]


@pytest.mark.parametrize("theta_deg", [70.0, 110.0])
def test_offset_contour_follows_the_closed_form_offset_circle_law(theta_deg):
    """Offset level sets share the centre, so their wall angle is the closed form."""
    psi = synthetic_cap(theta_deg, radius=18.0, xi=2.0, ny=66, nx=77)
    out = offset_circle_prediction(psi, xi=2.0)
    assert math.isfinite(out["offset_dev_deg"])
    assert out["offset_dev_deg"] < 1.0, out


def test_offset_circle_law_is_not_the_naive_level_independence():
    """The naive expectation (angle independent of level) is geometrically wrong."""
    psi = synthetic_cap(45.0, radius=18.0, xi=2.0, ny=66, nx=77)
    ang = [geodesic_angle(psi, xi=2.0, frac=f)["theta_geodesic_deg"] for f in (0.35, 0.5, 0.65)]
    assert max(ang) - min(ang) > 1.0, ang
    assert abs(ang[1] - 45.0) < 0.5, ang


def test_level_convention_spread_is_small_on_a_symmetric_cap():
    psi = synthetic_cap(90.0, radius=18.0, xi=2.0, ny=66, nx=77)
    out = level_convention_angles(psi, xi=2.0)
    assert out["spread_deg"] < 0.5, out


def test_geodesic_estimator_does_not_read_the_prescribed_angle():
    """Same field, repeated call -> bitwise identical, no hidden angle input."""
    psi = synthetic_cap(110.0)
    a = geodesic_angle(psi, xi=2.0)["theta_geodesic_deg"]
    b = geodesic_angle(psi, xi=2.0)["theta_geodesic_deg"]
    assert a == b
    assert abs(a - 110.0) < 0.5


def test_gas_level_uses_far_field_not_the_drop():
    psi = synthetic_cap(90.0, radius=10.0)
    gas, liquid, level = gas_level(psi, 0.5)
    assert gas < -0.9 and liquid > 0.9
    assert abs(level - 0.5 * (gas + liquid)) < 1e-12


def test_right_crossing_finds_the_outward_interface():
    row = np.array([1.0, 1.0, 1.0, 0.5, 0.0, -1.0, -1.0])
    x = right_crossing(row, 0.25, jc=1)
    assert 2.0 < x < 5.0


def test_symmetry_error_is_zero_for_a_symmetric_field():
    psi = synthetic_cap(90.0)
    assert symmetry_error(psi) < 1e-12


def test_symmetry_error_detects_asymmetry():
    psi = synthetic_cap(90.0)
    psi[:, 5] += 0.5
    assert symmetry_error(psi) > 0.1