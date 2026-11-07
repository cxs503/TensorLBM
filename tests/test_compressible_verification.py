"""Pytest tests for compressible LBM shock wave verification module."""

import math
import numpy as np
import pytest

from tensorlbm.compressible.solver import (
    GodunovSolver1D,
    D2Q16Lattice,
    normal_shock_relations,
    oblique_shock_relations,
)


class TestRankineHugoniot:
    """Test Rankine-Hugoniot exact shock relations."""

    @pytest.mark.parametrize("M1,expected_rho,expected_p,expected_T", [
        (2.0, 2.666667, 4.5, 1.6875),
        (3.0, 3.857143, 10.333333, 2.679012),
        (5.0, 5.0, 29.0, 5.8),
        (8.0, 5.565217, 74.5, 13.386719),
        (10.0, 5.714286, 116.5, 20.3875),
    ])
    def test_normal_shock_ratios(self, M1, expected_rho, expected_p, expected_T):
        """Verify normal shock relations against known values."""
        rh = normal_shock_relations(M1, gamma=1.4)
        assert abs(rh.rho_ratio - expected_rho) / expected_rho < 1e-4
        assert abs(rh.p_ratio - expected_p) / expected_p < 1e-4
        assert abs(rh.T_ratio - expected_T) / expected_T < 1e-4

    def test_normal_shock_M2_subsonic(self):
        """Post-shock Mach number must be subsonic."""
        for M1 in [2.0, 3.0, 5.0, 8.0, 10.0]:
            rh = normal_shock_relations(M1)
            assert rh.M2 < 1.0, f"M2={rh.M2} should be < 1 for M1={M1}"

    def test_normal_shock_density_limit(self):
        """Density ratio must approach (gamma+1)/(gamma-1) = 6 for gamma=1.4."""
        rh = normal_shock_relations(100.0)  # Very high Mach
        assert rh.rho_ratio < 6.0  # Limit is 6 for gamma=1.4
        assert rh.rho_ratio > 5.9  # Close to limit

    def test_normal_shock_invalid_M1(self):
        """Should raise ValueError for M1 <= 1."""
        with pytest.raises(ValueError):
            normal_shock_relations(1.0)
        with pytest.raises(ValueError):
            normal_shock_relations(0.5)

    def test_oblique_shock_weak_solution(self):
        """Oblique shock should return weak shock solution."""
        os_result = oblique_shock_relations(3.0, math.radians(15.0))
        assert os_result.M2 < 3.0  # Mach decreases across shock
        assert os_result.beta > math.asin(1.0 / 3.0)  # Beta > Mach angle


class TestGodunovSolver:
    """Test Godunov solver for 1D compressible flow."""

    def test_normal_shock_stationary(self):
        """Stationary normal shock should be preserved exactly."""
        for M1 in [2.0, 3.0, 5.0]:
            solver = GodunovSolver1D(nx=128, gamma=1.4, cfl=0.4)
            solver.initialize_normal_shock(M1=M1, gamma=1.4)
            solver.run(200, report_interval=200)

            rh = normal_shock_relations(M1, 1.4)
            rho1, p1 = 1.0, 1.0
            rho2 = rho1 * rh.rho_ratio
            p2 = p1 * rh.p_ratio

            nx_mid = 128 // 2
            rho_post = solver.rho[nx_mid:].mean()
            p_post = solver.p[nx_mid:].mean()

            assert abs(rho_post - rho2) / rho2 < 1e-10
            assert abs(p_post - p2) / p2 < 1e-10

    def test_no_nan_after_steps(self):
        """Solver should not produce NaN values."""
        solver = GodunovSolver1D(nx=64, gamma=1.4, cfl=0.4)
        solver.initialize_normal_shock(M1=5.0, gamma=1.4)
        solver.run(100, report_interval=100)

        assert not np.any(np.isnan(solver.rho))
        assert not np.any(np.isnan(solver.u))
        assert not np.any(np.isnan(solver.p))

    def test_positive_pressure(self):
        """Pressure should remain positive."""
        solver = GodunovSolver1D(nx=64, gamma=1.4, cfl=0.4)
        solver.initialize_normal_shock(M1=10.0, gamma=1.4)
        solver.run(100, report_interval=100)

        assert np.all(solver.p > 0)
        assert np.all(solver.rho > 0)


class TestD2Q16Lattice:
    """Test D2Q16 lattice definition."""

    def test_lattice_properties(self):
        lattice = D2Q16Lattice()
        assert lattice.Q == 16
        assert len(lattice.cx) == 16
        assert len(lattice.cy) == 16
        assert len(lattice.w) == 16
        assert abs(sum(lattice.w) - 1.0) < 1e-10

    def test_weights_positive(self):
        lattice = D2Q16Lattice()
        assert all(w > 0 for w in lattice.w)
