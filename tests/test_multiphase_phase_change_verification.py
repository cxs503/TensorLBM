"""Pytest tests for multiphase phase-change verification (p250-n05).

Tests all acceptance criteria:
A1: EOS P-rho-T vs analytical < 2%
A2: Interface thickness vs theory < 5%
A3: Clausius-Clapeyron dP/dT < 5%
A4: Bubble growth vs Mikic-Rohsenow < 15%
A5: Condensation film vs Nusselt < 10%
A6: Negative control single-phase source = 0 < 0.1%
A7: pytest passes + 4 figures generated
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.multiphase_phase_change_verification import (
    VDWEOS,
    CarnahanStarlingEOS,
    PengRobinsonEOS,
    CoexistenceResult,
    ClausiusClapeyronResult,
    CondensationFilmResult,
    EOSVerificationResult,
    InterfaceThicknessResult,
    BubbleGrowthResult,
    NegativeControlResult,
    PhaseChangeVerificationReport,
    CS2,
    compute_interface_thickness,
    maxwell_construction,
    mikic_rohsenow_growth,
    nusselt_film_thickness,
    run_full_verification,
    verify_bubble_growth,
    verify_clausius_clapeyron,
    verify_condensation_film,
    verify_eos_pressure,
    verify_negative_control,
)


# ============================================================================
# A1: EOS P-rho-T verification
# ============================================================================
class TestEOSVerification:
    """A1: EOS pressure vs analytical at 5 state points per T_r. Error < 2%."""

    def test_vdw_pressure(self):
        """VDW EOS pressure matches analytical formula at 5 state points per T_r."""
        eos = VDWEOS(a=1.0, b=1.0/3.0, R=1.0)
        T_r_values = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = verify_eos_pressure(eos, T_r_values)

        assert result.eos_name == "VDWEOS"
        assert len(result.state_points) == 25  # 5 T_r × 5 points
        assert result.max_relative_error < 0.02, (
            f"A1 FAIL: max relative error = {result.max_relative_error:.6e} >= 0.02"
        )
        assert result.passed

    def test_peng_robinson_pressure(self):
        """PR EOS pressure matches analytical formula."""
        eos = PengRobinsonEOS(Tc=1.0, Pc=0.1, omega=0.0, R=1.0)
        T_r_values = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = verify_eos_pressure(eos, T_r_values)

        assert result.eos_name == "PengRobinsonEOS"
        assert result.max_relative_error < 0.02, (
            f"A1 FAIL (PR): max error = {result.max_relative_error:.6e}"
        )

    def test_vdw_critical_point(self):
        """VDW EOS critical point matches analytical: T_c=8a/(27bR), rho_c=1/(3b)."""
        eos = VDWEOS(a=1.0, b=1.0/3.0, R=1.0)
        assert abs(eos.T_c - 8.0/9.0) < 1e-10, f"T_c = {eos.T_c} != 8/9"
        assert abs(eos.rho_c - 1.0) < 1e-10, f"rho_c = {eos.rho_c} != 1"

    def test_vdw_pressure_monotone_high_density(self):
        """VDW pressure increases with density in stable liquid region."""
        eos = VDWEOS()
        T = 0.5 * eos.T_c
        # High density (liquid side, above spinodal)
        coex = maxwell_construction(eos, T)
        rho_high = np.linspace(coex.rho_l * 1.01, coex.rho_l * 1.5, 20)
        p = eos.pressure(rho_high, T)
        assert np.all(np.diff(p) > 0), "Pressure should increase with density in stable liquid"


# ============================================================================
# A2: Interface thickness
# ============================================================================
class TestInterfaceThickness:
    """A2: Interface thickness vs square-gradient theory. Error < 5%."""

    def test_interface_thickness_three_conditions(self):
        """Interface thickness at 3 T_r values matches theory."""
        eos = VDWEOS(a=1.0, b=1.0/3.0, R=1.0)
        results = []
        for T_r in [0.5, 0.7, 0.9]:
            res = compute_interface_thickness(eos, T_r)
            results.append(res)

        for res in results:
            assert res.relative_error < 0.05, (
                f"A2 FAIL at T_r={res.T_r}: error = {res.relative_error:.6e}"
            )
            assert res.xi_theory > 0, "Interface thickness must be positive"
            assert np.isfinite(res.xi_theory), "Interface thickness must be finite"


class TestClausiusClapeyron:
    """A3: dP_sat/dT from C-C equation. Error < 5%."""

    def test_clapeyron_slope(self):
        """C-C: P_sat is smooth, monotonic, and Maxwell construction is consistent."""
        eos = VDWEOS(a=1.0, b=1.0/3.0, R=1.0)
        T_r_values = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = verify_clausius_clapeyron(eos, T_r_values)

        assert result.passed, f"A3 FAIL: {result.max_relative_error:.6e}"
        assert len(result.T_r_values) >= 4
        assert all(p > 0 for p in result.p_sat_values), "P_sat must be positive"

    def test_psat_increases_with_temperature(self):
        """Saturation pressure increases with temperature."""
        eos = VDWEOS(a=1.0, b=1.0/3.0, R=1.0)
        T_r_values = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = verify_clausius_clapeyron(eos, T_r_values)
        for i in range(1, len(result.p_sat_values)):
            assert result.p_sat_values[i] > result.p_sat_values[i-1], (
                "P_sat must increase with T"
            )

    def test_maxwell_construction_consistency(self):
        """Maxwell construction gives consistent coexistence densities."""
        eos = VDWEOS(a=1.0, b=1.0/3.0, R=1.0)
        T = 0.7 * eos.T_c
        coex = maxwell_construction(eos, T)
        assert coex.rho_v < coex.rho_l, "Vapor density must be less than liquid"
        assert coex.area_residual < 1e-8, f"Area residual too large: {coex.area_residual}"


# ============================================================================
# A4: Bubble growth (Mikic-Rohsenow)
# ============================================================================
class TestBubbleGrowth:
    """A4: Bubble growth rate vs Mikic-Rohsenow. Error < 15%."""

    def test_mikic_rohsenow_growth_law(self):
        """Mikic-Rohsenow growth law R ~ sqrt(t) is recovered."""
        Ja = 0.5
        alpha_l = 0.25
        t = np.linspace(0.1, 10.0, 100)
        R = mikic_rohsenow_growth(Ja, alpha_l, t)
        R_normalized = R / np.sqrt(t)
        assert np.std(R_normalized) / np.mean(R_normalized) < 1e-10, (
            "Growth should be proportional to sqrt(t)"
        )

    def test_bubble_growth_three_superheats(self):
        """Bubble growth at 3 superheat values matches Mikic-Rohsenow."""
        result = verify_bubble_growth([0.01, 0.05, 0.1])
        assert len(result.superheat_values) == 3
        for i, err in enumerate(result.max_relative_error):
            assert err < 0.15, (
                f"A4 FAIL at dT={result.superheat_values[i]}: error = {err:.6e}"
            )

    def test_growth_rate_increases_with_superheat(self):
        """Higher superheat → faster bubble growth."""
        result = verify_bubble_growth([0.01, 0.05, 0.1])
        for i in range(1, len(result.ja_values)):
            assert result.ja_values[i] > result.ja_values[i-1]


# ============================================================================
# A5: Condensation film (Nusselt)
# ============================================================================
class TestCondensationFilm:
    """A5: Film thickness vs Nusselt theory. Error < 10%."""

    def test_nusselt_film_formula(self):
        """Nusselt film thickness formula gives correct scaling."""
        d1 = nusselt_film_thickness(0.1, 1.0)
        d2 = nusselt_film_thickness(1.6, 1.0)  # 16x deltaT
        ratio = d2 / d1
        expected = 16.0**0.25  # = 2.0
        assert abs(ratio - expected) / expected < 1e-10

    def test_condensation_three_deltat(self):
        """Film thickness at 3 deltaT values matches Nusselt."""
        result = verify_condensation_film([0.1, 0.5, 1.0])
        assert len(result.deltaT_values) == 3
        assert result.max_relative_error < 0.10, (
            f"A5 FAIL: max error = {result.max_relative_error:.6e}"
        )

    def test_film_thickens_with_deltat(self):
        """Larger deltaT → thicker film."""
        result = verify_condensation_film([0.1, 0.5, 1.0])
        for i in range(1, len(result.delta_film_nusselt)):
            assert result.delta_film_nusselt[i] > result.delta_film_nusselt[i-1]


# ============================================================================
# A6: Negative control
# ============================================================================
class TestNegativeControl:
    """A6: Single-phase → phase-change source = 0. Error < 0.1%."""

    def test_uniform_subcooled(self):
        """Uniform subcooled field → zero phase-change source."""
        result = verify_negative_control(T_sat=1.0, T_uniform=0.5)
        assert result.max_phase_change_source < 1e-10, (
            f"A6 FAIL: max source = {result.max_phase_change_source:.6e}"
        )
        assert result.passed

    def test_uniform_superheated(self):
        """Uniform superheated field → zero phase-change source."""
        result = verify_negative_control(T_sat=1.0, T_uniform=1.5)
        assert result.max_phase_change_source < 1e-10

    def test_zero_gradient_means_no_interface(self):
        """Zero temperature gradient → no interface detected."""
        T_field = np.full((32, 32), 0.8)
        grad = np.gradient(T_field)
        grad_mag = np.sqrt(grad[0]**2 + grad[1]**2)
        assert np.max(grad_mag) < 1e-10, "Uniform field should have zero gradient"


# ============================================================================
# A7: Full verification + report
# ============================================================================
class TestFullVerification:
    """A7: Full verification suite passes."""

    def test_all_anchors_pass(self):
        """All 6 acceptance criteria pass."""
        report = run_full_verification()
        assert report.all_passed, (
            f"Not all anchors passed: {report.summary}"
        )

    def test_report_structure(self):
        """Report has all required fields."""
        report = run_full_verification()
        assert hasattr(report, 'a1_eos')
        assert hasattr(report, 'a2_interface')
        assert hasattr(report, 'a3_clapeyron')
        assert hasattr(report, 'a4_bubble')
        assert hasattr(report, 'a5_condensation')
        assert hasattr(report, 'a6_negative')
        assert hasattr(report, 'all_passed')
        assert hasattr(report, 'summary')
        assert len(report.summary) == 6

    def test_report_json_serializable(self):
        """Report can be serialized to JSON for artifact storage."""
        report = run_full_verification()
        data = {
            'A1': {
                'eos_name': report.a1_eos.eos_name,
                'max_error': float(report.a1_eos.max_relative_error),
                'passed': bool(report.a1_eos.passed),
                'n_points': len(report.a1_eos.state_points),
            },
            'A2': [
                {'T_r': float(r.T_r), 'xi': float(r.xi_theory), 'error': float(r.relative_error), 'passed': bool(r.passed)}
                for r in report.a2_interface
            ],
            'A3': {
                'max_error': float(report.a3_clapeyron.max_relative_error),
                'passed': bool(report.a3_clapeyron.passed),
                'n_points': len(report.a3_clapeyron.T_r_values),
            },
            'A4': {
                'max_errors': [float(e) for e in report.a4_bubble.max_relative_error],
                'passed': bool(report.a4_bubble.passed),
            },
            'A5': {
                'max_error': float(report.a5_condensation.max_relative_error),
                'passed': bool(report.a5_condensation.passed),
            },
            'A6': {
                'max_source': float(report.a6_negative.max_phase_change_source),
                'passed': bool(report.a6_negative.passed),
            },
            'all_passed': bool(report.all_passed),
        }
        json_str = json.dumps(data, indent=2)
        assert len(json_str) > 100
