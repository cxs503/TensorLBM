"""Unit tests for the Cox-Voinov / Tanner scaling analysis (IN100-N02).

Every test builds a *synthetic* dataset from the law it is supposed to recover, so
the fits and the machine gates are checked against known-truth input rather than
against the campaign's own output.
"""

from __future__ import annotations

import math

import pytest

from tensorlbm.contact_line.scaling import (
    ScalingPoint,
    cox_voinov_fit,
    decades_covered,
    evaluate_c1,
    evaluate_c2,
    evaluate_c3,
    microscopic_length,
    power_law_fit,
    resolvability_gate,
)

DEG = math.pi / 180.0


def _point(ca: float, theta_a: float, theta_r: float, theta_e: float, unc: float = 0.2,
           drift: float = 10.0, macro: float = 30.0) -> ScalingPoint:
    return ScalingPoint(
        gx=ca * 1e-4,
        capillary=ca,
        velocity=ca * 1e-1,
        velocity_stderr=1e-6,
        theta_a_deg=theta_a,
        theta_a_unc_deg=unc,
        theta_r_deg=theta_r,
        theta_r_unc_deg=unc,
        theta_e_deg=theta_e,
        theta_e_unc_deg=unc,
        wall_fraction=0.8,
        extra={"drift": drift, "macro_length": macro},
    )


def _tanner_points(theta_e_deg: float = 0.0, prefactor_deg: float = 60.0, n: int = 8,
                   ca_range: tuple[float, float] = (1e-3, 1e-1)) -> list[ScalingPoint]:
    """Synthetic Tanner law ``theta_d = A Ca^(1/3)`` (plus a static offset)."""
    pts = []
    for k in range(n):
        ca = ca_range[0] * (ca_range[1] / ca_range[0]) ** (k / (n - 1))
        theta = theta_e_deg + prefactor_deg * ca ** (1.0 / 3.0)
        pts.append(_point(ca, theta, theta_e_deg - 0.5 * (theta - theta_e_deg), theta_e_deg))
    return pts


class TestDecades:
    def test_simple(self):
        assert decades_covered([1.0, 10.0]) == pytest.approx(1.0)
        assert decades_covered([1e-3, 1e-1]) == pytest.approx(2.0)

    def test_degenerate(self):
        assert decades_covered([1.0]) == 0.0
        assert decades_covered([0.0, -1.0]) == 0.0


class TestPowerLaw:
    def test_recovers_exponent(self):
        x = [1e-4 * 1.7**k for k in range(8)]
        y = [3.5 * v ** (1.0 / 3.0) for v in x]
        fit = power_law_fit(x, y, bootstrap=200)
        assert fit["exponent"] == pytest.approx(1.0 / 3.0, abs=1e-6)
        assert fit["prefactor"] == pytest.approx(3.5, rel=1e-6)
        assert fit["r2"] == pytest.approx(1.0, abs=1e-9)
        assert fit["ci_low"] <= fit["exponent"] <= fit["ci_high"]

    def test_needs_three_points(self):
        with pytest.raises(ValueError):
            power_law_fit([1.0, 2.0], [1.0, 2.0])

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            power_law_fit([1.0, 2.0, 3.0], [1.0, 2.0])


class TestGate:
    def test_marks_unresolved_points(self):
        pts = [_point(1e-3, 10.0, 10.0, 10.0, unc=2.0, drift=10.0),   # no change
               _point(1e-2, 20.0, 12.0, 10.0, unc=0.2, drift=10.0),   # resolved
               _point(1e-1, 30.0, 15.0, 10.0, unc=0.2, drift=0.05)]   # no drift
        out = resolvability_gate(pts, k=2.0, min_decorrelation=0.5)
        assert out[0].resolvable is False
        assert "angle" in out[0].gate_reason
        assert out[1].resolvable is True
        assert out[2].resolvable is False
        assert "drift" in out[2].gate_reason


class TestTanner:
    def test_passes_on_tanner_data(self):
        pts = resolvability_gate(_tanner_points())
        fit = evaluate_c2(pts, tolerance=0.10, min_decades=1.5, bootstrap=100)
        assert fit.passes
        assert fit.exponent == pytest.approx(1.0 / 3.0, abs=0.02)
        assert fit.decades >= 1.5

    def test_fails_on_wrong_exponent(self):
        pts = []
        for k in range(8):
            ca = 1e-3 * (100.0 ** (k / 7.0))
            theta = 30.0 * ca**0.6  # pure power law, exponent 0.6 != 1/3
            pts.append(_point(ca, theta, 0.5 * theta, 0.0))
        pts = resolvability_gate(pts)
        fit = evaluate_c2(pts, tolerance=0.10, min_decades=1.5, bootstrap=100)
        assert not fit.passes
        assert fit.exponent == pytest.approx(0.6, abs=0.02)

    def test_fails_on_short_range(self):
        pts = resolvability_gate(_tanner_points(ca_range=(1e-3, 3e-3)))
        fit = evaluate_c2(pts, tolerance=0.10, min_decades=1.5, bootstrap=50)
        assert not fit.passes
        assert "decades" in fit.note


class TestCoxVoinov:
    def test_recovers_log_term(self):
        theta_e = 40.0
        macro = 25.0
        l_micro = 2.5
        K = 9.0 * math.log(macro / l_micro)
        pts = []
        for k in range(8):
            ca = 1e-3 * (50.0 ** (k / 7.0))
            th = ((theta_e * DEG) ** 3 + K * ca) ** (1.0 / 3.0)
            pts.append(_point(ca, math.degrees(th), theta_e, theta_e, macro=macro))
        pts = resolvability_gate(pts)
        fit = cox_voinov_fit(pts, macro_length=macro)
        assert fit.slope == pytest.approx(K, rel=1e-3)
        assert fit.r2 > 0.999
        assert fit.passes
        assert fit.micro_length == pytest.approx(l_micro, rel=1e-3)

    def test_negative_slope_is_rejected(self):
        pts = []
        for k in range(6):
            ca = 1e-3 * (20.0 ** (k / 5.0))
            pts.append(_point(ca, 60.0 - 5.0 * math.log10(ca + 1.0) * 0.1, 60.0, 60.0))
        pts = resolvability_gate(pts, k=0.1)
        fit = cox_voinov_fit(pts, macro_length=30.0)
        assert not fit.passes or fit.slope < 0.0

    def test_microscopic_length_roundtrip(self):
        K = 9.0 * math.log(20.0 / 4.0)
        assert microscopic_length(K, 20.0) == pytest.approx(4.0, rel=1e-9)

    def test_c3_flags_indistinguishable_scales(self):
        # slope = 0 -> l_m = L -> the logarithm is not a ratio of distinct scales
        fit = cox_voinov_fit(
            resolvability_gate(_tanner_points()), macro_length=30.0
        )
        verdict = evaluate_c3(fit, micro_measured=3.0)
        assert "distinct_scales" in verdict
        assert verdict["passes"] is False

    def test_c3_consistent_when_truth_matches(self):
        theta_e, macro, l_micro = 30.0, 20.0, 3.0
        K = 9.0 * math.log(macro / l_micro)
        pts = []
        for k in range(8):
            ca = 1e-2 * (30.0 ** (k / 7.0))
            th = ((theta_e * DEG) ** 3 + K * ca) ** (1.0 / 3.0)
            pts.append(_point(ca, math.degrees(th), theta_e, theta_e, macro=macro))
        pts = resolvability_gate(pts)
        fit = cox_voinov_fit(pts, macro_length=macro)
        verdict = evaluate_c3(fit, micro_measured=l_micro, tolerance=0.20)
        assert verdict["passes"] is True
        assert verdict["ratio_fit_over_measured"] == pytest.approx(1.0, rel=1e-3)


class TestC1:
    def test_extrapolates_to_static_angle(self):
        theta_e = 50.0
        K = 30.0
        pts = []
        for k in range(6):
            ca = 1e-4 * (20.0 ** (k / 5.0))
            th = ((theta_e * DEG) ** 3 + K * ca) ** (1.0 / 3.0)
            pts.append(_point(ca, math.degrees(th), theta_e, theta_e))
        pts = resolvability_gate(pts, k=0.1)
        verdict = evaluate_c1(pts, tolerance_deg=3.0)
        assert verdict["passes"]
        assert abs(verdict["error_deg"]) < 1.0

    def test_insufficient_points(self):
        verdict = evaluate_c1([_point(1e-3, 60.0, 60.0, 60.0)])
        assert verdict["passes"] is False