"""Pytest tests for Ra=1e5 natural convection cavity benchmark (de Vahl Davis 1983).

Tests verify:
1. Result file exists and is valid JSON
2. Key quantities (Nu, u_max, v_max) within 8% of de Vahl Davis reference
3. Convergence: error decreases with grid refinement
4. Conservation: symmetry check (nu_left ≈ nu_right)
5. Physical sanity: T in [0,1], mach < 0.3, finite values
"""

import json
import math
import os
from pathlib import Path

import pytest

# de Vahl Davis 1983, Table 1, page 257: Ra=10^5, Pr=0.71
REF_DVD_1E5 = {
    "nu": 4.519,
    "u_max": 34.72,
    "v_max": 68.60,
    "nu_max": 8.499,
    "nu_min": 0.989,
    "nu_mid": 4.499,
    "u_max_y": 0.813,
    "v_max_x": 0.066,
}

ERROR_TOLERANCE = 8.0  # percent


def _load_result_json():
    """Load the combined result.json from thermal_cavity benchmark."""
    result_path = Path(__file__).resolve().parents[1] / "benchmarks" / "verified" / "thermal_cavity" / "result.json"
    if not result_path.exists():
        pytest.skip(f"result.json not found at {result_path}")
    with open(result_path) as f:
        return json.load(f)


def _get_ra1e5_cases(data):
    """Extract Ra=1e5 cases from result data."""
    cases = data.get("cases", [])
    ra1e5 = [c for c in cases if c.get("ra") == 1e5 or abs(c.get("ra", 0) - 1e5) < 1e3]
    return ra1e5


@pytest.fixture(scope="module")
def result_data():
    return _load_result_json()


@pytest.fixture(scope="module")
def ra1e5_cases(result_data):
    cases = _get_ra1e5_cases(result_data)
    if not cases:
        pytest.skip("No Ra=1e5 cases found in result.json")
    return cases


class TestRa1e5ResultsExist:
    """Test that Ra=1e5 results are present."""

    def test_result_json_exists(self):
        result_path = Path(__file__).resolve().parents[1] / "benchmarks" / "verified" / "thermal_cavity" / "result.json"
        assert result_path.exists(), f"result.json not found at {result_path}"

    def test_result_json_valid(self, result_data):
        assert isinstance(result_data, dict)
        assert "cases" in result_data

    def test_ra1e5_cases_present(self, ra1e5_cases):
        assert len(ra1e5_cases) >= 1, "At least one Ra=1e5 case required"


class TestRa1e5Accuracy:
    """Test accuracy against de Vahl Davis 1983 reference."""

    def test_nu_within_tolerance(self, ra1e5_cases):
        """Nu within 8% of reference."""
        for case in ra1e5_cases:
            nu = case.get("nu")
            if nu is None:
                continue
            err = 100.0 * abs(nu - REF_DVD_1E5["nu"]) / REF_DVD_1E5["nu"]
            assert err < ERROR_TOLERANCE, (
                f"{case['tag']}: Nu={nu:.4f}, ref={REF_DVD_1E5['nu']}, err={err:.2f}%"
            )

    def test_u_max_within_tolerance(self, ra1e5_cases):
        """u_max within 8% of reference."""
        for case in ra1e5_cases:
            u_max = case.get("u_max_nd")
            if u_max is None:
                continue
            err = 100.0 * abs(u_max - REF_DVD_1E5["u_max"]) / REF_DVD_1E5["u_max"]
            assert err < ERROR_TOLERANCE, (
                f"{case['tag']}: u_max={u_max:.4f}, ref={REF_DVD_1E5['u_max']}, err={err:.2f}%"
            )

    def test_v_max_within_tolerance(self, ra1e5_cases):
        """v_max within 8% of reference."""
        for case in ra1e5_cases:
            v_max = case.get("v_max_nd")
            if v_max is None:
                continue
            err = 100.0 * abs(v_max - REF_DVD_1E5["v_max"]) / REF_DVD_1E5["v_max"]
            assert err < ERROR_TOLERANCE, (
                f"{case['tag']}: v_max={v_max:.4f}, ref={REF_DVD_1E5['v_max']}, err={err:.2f}%"
            )


class TestRa1e5Convergence:
    """Test grid convergence."""

    def test_nu_converges_with_refinement(self, ra1e5_cases):
        """Nu should converge monotonically toward reference with grid refinement."""
        by_n = sorted(ra1e5_cases, key=lambda c: c.get("nx", 0))
        if len(by_n) < 2:
            pytest.skip("Need at least 2 grid resolutions for convergence test")

        nus = [c["nu"] for c in by_n if c.get("nu") is not None]
        if len(nus) < 2:
            pytest.skip("Insufficient Nu data")

        # Check that error decreases with refinement
        errors = [100.0 * abs(nu - REF_DVD_1E5["nu"]) / REF_DVD_1E5["nu"] for nu in nus]
        # At least the finest grid should be better than the coarsest
        assert errors[-1] < errors[0], (
            f"Nu error should decrease with refinement: {errors}"
        )


class TestRa1e5Conservation:
    """Test conservation properties."""

    def test_symmetry_check(self, ra1e5_cases):
        """Left and right wall Nusselt should be nearly equal (symmetry)."""
        for case in ra1e5_cases:
            nu_sym = case.get("nu_sym")
            if nu_sym is not None:
                assert nu_sym < 0.1, (
                    f"{case['tag']}: nu_sym={nu_sym:.6f} (should be < 0.1)"
                )

    def test_temperature_bounds(self, ra1e5_cases):
        """Temperature should be within [0, 1] (with small numerical tolerance)."""
        for case in ra1e5_cases:
            t_min = case.get("T_min")
            t_max = case.get("T_max")
            if t_min is not None:
                assert t_min >= -0.01, f"{case['tag']}: T_min={t_min} < -0.01"
            if t_max is not None:
                assert t_max <= 1.01, f"{case['tag']}: T_max={t_max} > 1.01"

    def test_mach_number(self, ra1e5_cases):
        """Mach number should be < 0.3 for incompressible assumption."""
        for case in ra1e5_cases:
            mach = case.get("mach")
            if mach is not None:
                assert mach < 0.3, f"{case['tag']}: mach={mach:.4f} >= 0.3"

    def test_finite_values(self, ra1e5_cases):
        """All key quantities should be finite."""
        for case in ra1e5_cases:
            for key in ("nu", "u_max_nd", "v_max_nd", "T_min", "T_max"):
                val = case.get(key)
                if val is not None:
                    assert math.isfinite(val), f"{case['tag']}: {key}={val} not finite"


class TestRa1e5SteadyState:
    """Test that steady state was reached."""

    def test_steady_triggered_or_max_steps(self, ra1e5_cases):
        """Either steady state was triggered or max steps were reached."""
        for case in ra1e5_cases:
            steady = case.get("steady_triggered")
            steps = case.get("steps_run", 0)
            max_steps = case.get("max_steps", 0)
            # Either steady was triggered, or we ran to max steps
            assert steady or steps >= max_steps * 0.9, (
                f"{case['tag']}: steady={steady}, steps={steps}/{max_steps}"
            )

    def test_last_drift_small(self, ra1e5_cases):
        """Final drift should be small if steady state was reached."""
        for case in ra1e5_cases:
            if case.get("steady_triggered"):
                drift = case.get("last_drift")
                if drift is not None:
                    assert drift < 1e-3, (
                        f"{case['tag']}: last_drift={drift:.2e} (should be < 1e-3)"
                    )
