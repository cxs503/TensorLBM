"""N01 wetting dynamics: comprehensive test suite.

Tests cover:
- Unit tests for utility functions (20 tests)
- Integration tests for LBM simulation (marked slow)
- Data class serialization tests
- Edge case tests
"""

from __future__ import annotations

import json
import math
import pytest
import torch

from tensorlbm.wetting_n01 import (
    N01CaseResult,
    N01ConvergenceResult,
    N01NegativeControlResult,
    N01SweepResult,
    SIGMA_LU,
    RHO_L_LU,
    RHO_V_LU,
    G_LU,
    LU_PER_MM,
    THETA_TARGETS,
    RADIUS_MM,
    _CALIBRATION_POINTS,
    _interp_fw_to_angle,
    _interp_angle_to_fw,
    angle_to_wall_fraction,
    wall_fraction_to_angle_approx,
    capillary_rise_analytical,
    _mm_to_lu,
    run_single_case,
    run_parameter_sweep,
    run_grid_convergence,
    run_negative_controls,
)


# ---------------------------------------------------------------------------
# Unit tests: utility functions
# ---------------------------------------------------------------------------

class TestCalibrationMapping:
    def test_fw_to_angle_monotonic(self):
        angles = [_interp_fw_to_angle(fw) for fw, _ in _CALIBRATION_POINTS]
        for i in range(len(angles) - 1):
            assert angles[i] > angles[i + 1], f"Not monotonic at index {i}"

    def test_fw_to_angle_bounds(self):
        assert _interp_fw_to_angle(0.0) == pytest.approx(180.0, abs=1.0)
        assert _interp_fw_to_angle(0.98) == pytest.approx(20.0, abs=1.0)

    def test_angle_to_fw_roundtrip(self):
        for theta in [30, 60, 90, 120, 150]:
            fw = angle_to_wall_fraction(theta)
            theta_back = _interp_fw_to_angle(fw)
            assert abs(theta_back - theta) < 3.0, f"Roundtrip error for {theta}deg: {theta_back}"

    def test_angle_to_fw_out_of_range(self):
        with pytest.raises(ValueError):
            angle_to_wall_fraction(0.0)
        with pytest.raises(ValueError):
            angle_to_wall_fraction(180.0)

    def test_wall_fraction_to_angle_approx(self):
        for fw, theta in _CALIBRATION_POINTS:
            assert wall_fraction_to_angle_approx(fw) == pytest.approx(theta, abs=1.0)


class TestCapillaryRise:
    def test_hydrophilic_rise_positive(self):
        h = capillary_rise_analytical(SIGMA_LU, RHO_L_LU, RHO_V_LU, 30.0, 20.0, G_LU)
        assert h > 0

    def test_hydrophobic_rise_negative(self):
        h = capillary_rise_analytical(SIGMA_LU, RHO_L_LU, RHO_V_LU, 150.0, 20.0, G_LU)
        assert h < 0

    def test_neutral_zero_rise(self):
        h = capillary_rise_analytical(SIGMA_LU, RHO_L_LU, RHO_V_LU, 90.0, 20.0, G_LU)
        assert abs(h) < 1e-10

    def test_invalid_inputs_nan(self):
        assert math.isnan(capillary_rise_analytical(SIGMA_LU, RHO_L_LU, RHO_V_LU, 30.0, -1.0, G_LU))
        assert math.isnan(capillary_rise_analytical(SIGMA_LU, RHO_L_LU, RHO_V_LU, 30.0, 20.0, 0.0))

    def test_radius_scaling(self):
        h1 = capillary_rise_analytical(SIGMA_LU, RHO_L_LU, RHO_V_LU, 60.0, 10.0, G_LU)
        h2 = capillary_rise_analytical(SIGMA_LU, RHO_L_LU, RHO_V_LU, 60.0, 20.0, G_LU)
        assert abs(h1 / h2 - 2.0) < 0.01


class TestUnitConversion:
    def test_mm_to_lu(self):
        assert _mm_to_lu(1.0) == pytest.approx(20.0)
        assert _mm_to_lu(0.25) == pytest.approx(5.0)
        assert _mm_to_lu(4.0) == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestDataClasses:
    def test_case_result_to_dict(self):
        r = N01CaseResult(
            theta_target_deg=90.0, wall_fraction=0.66, R_lu=20.0, R_mm=1.0,
            theta_measured_deg=88.0, theta_unc_deg=1.0, theta_error_deg=2.0,
            capillary_rise_lu=10.0, capillary_rise_analytical_lu=10.5,
            capillary_rise_error_pct=4.76, bond_number=0.1, spurious_current_max=1e-4,
            n_points=100, rms_fit=0.5, relax_steps=8000, elapsed_s=1.0,
        )
        d = r.to_dict()
        json.dumps(d)

    def test_convergence_result_to_dict(self):
        r = N01ConvergenceResult(
            theta_target_deg=90.0, R_lu=20.0,
            resolutions=[{"factor": 10, "dx_lu": 2.0, "nx": 64, "ny": 48,
                          "theta_deg": 88.0, "unc_deg": 1.0, "n_points": 50, "rms_fit": 0.3}],
            convergence_order=1.9, theoretical_order=2.0, order_error_pct=5.0, passes=True,
        )
        json.dumps(r.to_dict())

    def test_negative_control_result_to_dict(self):
        r = N01NegativeControlResult(
            test_name="test", description="desc", expected="exp",
            actual="act", passed=True, detail="ok",
        )
        json.dumps(r.to_dict())

    def test_sweep_result_to_dict(self):
        r = N01SweepResult(
            cases=[], convergence=[], negative_controls=[],
            sigma_measured=SIGMA_LU, timestamp="2026-01-01T00:00:00Z",
            metadata={"test": True},
        )
        json.dumps(r.to_dict())


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_extreme_hydrophilic(self):
        fw = angle_to_wall_fraction(30.0)
        assert fw > 0.9

    def test_extreme_hydrophobic(self):
        fw = angle_to_wall_fraction(150.0)
        assert fw < 0.2

    def test_superhydrophobic(self):
        fw = angle_to_wall_fraction(170.0)
        assert fw < 0.1

    def test_smallest_radius(self):
        assert _mm_to_lu(0.25) == pytest.approx(5.0)

    def test_largest_radius(self):
        assert _mm_to_lu(4.0) == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# Integration tests (ALL marked slow due to LBM computational cost)
# ---------------------------------------------------------------------------

class TestSingleCase:
    @pytest.mark.slow
    def test_single_case_runs(self):
        result = run_single_case(90.0, 1.0, relax_steps=200, device="cpu")
        assert isinstance(result, N01CaseResult)
        assert not math.isnan(result.theta_measured_deg)

    @pytest.mark.slow
    def test_single_case_bond_number_positive(self):
        result = run_single_case(90.0, 1.0, relax_steps=200, device="cpu")
        assert result.bond_number > 0

    @pytest.mark.slow
    def test_single_case_wall_fraction_in_range(self):
        result = run_single_case(90.0, 1.0, relax_steps=200, device="cpu")
        assert 0.0 <= result.wall_fraction <= 1.0


class TestParameterSweep:
    @pytest.mark.slow
    def test_sweep_returns_25_cases(self):
        result = run_parameter_sweep(quick=True, relax_steps=200, device="cpu")
        assert len(result.cases) == 25

    @pytest.mark.slow
    def test_sweep_metadata(self):
        result = run_parameter_sweep(quick=True, relax_steps=200, device="cpu")
        assert result.metadata["n_cases"] == 25
        assert result.metadata["quick"] is True


class TestGridConvergence:
    @pytest.mark.slow
    def test_convergence_runs(self):
        result = run_grid_convergence(90.0, 1.0, relax_steps=200, device="cpu")
        assert isinstance(result, N01ConvergenceResult)
        assert len(result.resolutions) == 4

    @pytest.mark.slow
    def test_convergence_resolution_count(self):
        result = run_grid_convergence(90.0, 1.0, relax_steps=200, device="cpu")
        assert len(result.resolutions) == 4


class TestNegativeControls:
    @pytest.mark.slow
    def test_nc_count(self):
        results = run_negative_controls(device="cpu", relax_steps=200)
        assert len(results) == 3

    @pytest.mark.slow
    def test_nc_names(self):
        results = run_negative_controls(device="cpu", relax_steps=200)
        names = {r.test_name for r in results}
        assert "negative_surface_tension" in names
        assert "superhydrophobic_limit" in names
        assert "contact_line_pinning" in names
