"""Regression tests for the p127w2-N04 compressible verification face.

Two layers:

* the exact oblique-shock solver (``tensorlbm.compressible.solver``) -- its
  theta-beta-M inversion is now root-found on the monotone branch; the old
  Newton version had the derivative sign reversed and returned angles on the
  strong branch (Cp twice the linear limit at small deflection), so the
  weak-shock continuity test below is the regression guard for that defect;
* the boom/Ackeret module (``tensorlbm.compressible.boom``) -- analytic anchors
  for the scalar Burgers solver (Rankine-Hugoniot shock speed ``A/2`` and the
  expansion-fan slope ``1/t``) and the criterion-(d) audit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.compressible.ackeret import ackeret_pressure_coefficient
from tensorlbm.compressible.boom import (
    ackeret_validity_scan,
    criterion_d_audit,
    expansion_fan_slope_check,
    n_wave_metrics,
    shock_speed_check,
    wedge_signature,
)
from tensorlbm.compressible.solver import (
    oblique_shock_angle,
    oblique_shock_relations,
    oblique_shock_theta,
)


# ---------------------------------------------------------------------------
# oblique shock: theta-beta-M inversion
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mach", [1.5, 2.0, 3.0, 5.0])
@pytest.mark.parametrize("theta_deg", [0.5, 2.0, 5.0, 10.0, 20.0])
def test_oblique_shock_angle_roundtrip(mach: float, theta_deg: float) -> None:
    """theta(beta(M, theta)) must return the requested deflection."""
    theta = math.radians(theta_deg)
    try:
        beta, theta_max = oblique_shock_angle(mach, theta)
    except ValueError:
        pytest.skip("deflection beyond the detachment angle")
    if theta_deg > 0.9 * math.degrees(theta_max):
        # near detachment dtheta/dbeta -> 0, so a roundtrip in eta is ill posed
        pytest.skip("close to the detachment angle")
    assert oblique_shock_theta(beta, mach) == pytest.approx(theta, rel=0, abs=1e-9)


def test_oblique_shock_weak_limit_matches_linear_theory() -> None:
    """Regression guard for the reversed-derivative defect.

    In the weak-shock limit the exact oblique-shock Cp tends to the Ackeret
    value ``2*theta/sqrt(Ma^2-1)``.  The buggy inversion returned ~2x that value
    at every deflection angle, so a 2 % band at theta = 0.5 deg pins it.
    """
    mach, theta_deg = 2.0, 0.5
    res = oblique_shock_relations(mach, math.radians(theta_deg))
    cp_exact = 2.0 / (1.4 * mach * mach) * (res.p_ratio - 1.0)
    cp_linear = ackeret_pressure_coefficient(mach, theta_deg)
    assert abs(cp_exact - cp_linear) / cp_linear < 0.02


def test_oblique_shock_reference_values() -> None:
    """Anderson's table: Ma=2, theta=10 deg -> beta=39.31 deg, p2/p1=1.7066."""
    res = oblique_shock_relations(2.0, math.radians(10.0))
    assert math.degrees(res.beta) == pytest.approx(39.31, abs=0.02)
    assert res.p_ratio == pytest.approx(1.7066, abs=2e-3)
    assert res.M1n == pytest.approx(2.0 * math.sin(res.beta), rel=1e-12)


def test_oblique_shock_detachment_raises() -> None:
    with pytest.raises(ValueError):
        oblique_shock_relations(2.0, math.radians(45.0))


# ---------------------------------------------------------------------------
# Ackeret validity audit
# ---------------------------------------------------------------------------
def test_ackeret_error_grows_with_deflection() -> None:
    scan = ackeret_validity_scan(mach=2.0)
    err = scan["rel_error"]
    assert np.all(np.diff(err) > 0.0)
    assert err[0] < 0.02                      # 0.5 deg is inside linear theory
    assert err[-1] > 0.25                     # 15 deg is far outside


def test_ackeret_band_boundary_at_mach2() -> None:
    scan = ackeret_validity_scan(mach=2.0, tol=0.10)
    assert scan["theta_max_deg"] == pytest.approx(3.0, abs=1e-9)


def test_criterion_d_verdict_is_fail_with_reason() -> None:
    audit = criterion_d_audit(mach=2.0, theta_deg=10.0, tol=0.10)
    assert audit["verdict"] == "FAIL"
    assert audit["near_field_rel_error_vs_ackeret"] > 0.10
    assert audit["theta_max_within_tol_deg"] == pytest.approx(3.0, abs=1e-9)
    assert "Ackeret" in audit["reason"]


# ---------------------------------------------------------------------------
# scalar Burgers solver: analytic anchors
# ---------------------------------------------------------------------------
def test_burgers_shock_speed_matches_rankine_hugoniot() -> None:
    res = shock_speed_check()
    assert res["rel_error"] < 1e-6


def test_burgers_expansion_fan_slope() -> None:
    res = expansion_fan_slope_check()
    assert res["rel_error"] < 0.01


# ---------------------------------------------------------------------------
# wedge signature
# ---------------------------------------------------------------------------
def test_wedge_signature_shape_and_metrics() -> None:
    xi, u, dp = wedge_signature(2.0, 10.0, chord=1.0, n=801)
    met = n_wave_metrics(xi, u)
    # symmetric compression/expansion pair, unit-chord wavelength
    assert met.peak_pos == pytest.approx(dp, rel=2e-3)
    assert met.peak_neg == pytest.approx(-dp, rel=2e-3)
    assert met.duration == pytest.approx(1.0, abs=0.02)
    # symmetric signature carries (numerically) zero impulse
    assert abs(met.impulse) < 0.05 * dp


def test_wedge_signature_requires_supersonic() -> None:
    with pytest.raises(ValueError):
        wedge_signature(0.8, 5.0)