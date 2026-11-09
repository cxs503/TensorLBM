"""Tests for shock stand-off distance resolvability analysis.

Tests the pipeline logic using both synthetic data (fast) and actual
solver runs (on small grids). The full four-grid study is designed
to run on the cluster; these tests verify the analysis framework.
"""

import math
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorlbm.compressible_shock.standoff import (
    richardson_extrapolation_three_grid,
    cross_triplet_richardson,
    resolvability_law,
    classify_standoff,
    corner_mass_balance,
    run_forward_step,
    FOUR_GRID_SPEC,
    TRIPLETS,
)
from tensorlbm.compressible_shock.flux import primitive


# ============================================================================
# Richardson extrapolation tests (synthetic data)
# ============================================================================

class TestRichardsonExtrapolation:
    """Test Richardson extrapolation with known analytical sequences."""

    def test_exact_second_order_convergence(self):
        """f(h) = f_exact + C*h^2 with r=2 should give exact extrapolation."""
        f_exact = 1.0
        C = 0.1
        # h = 1, 0.5, 0.25 → f = 1.1, 1.025, 1.00625
        f1 = f_exact + C * 1.0**2   # coarse
        f2 = f_exact + C * 0.5**2   # medium
        f3 = f_exact + C * 0.25**2  # fine

        result = richardson_extrapolation_three_grid(f1, f2, f3, r=2.0, order=2.0)

        assert abs(result["extrapolated"] - f_exact) < 1e-10
        assert result["apparent_order"] == pytest.approx(2.0, abs=0.01)
        assert result["gci_fine"] < 0.01  # < 1%

    def test_first_order_convergence(self):
        """f(h) = f_exact + C*h with r=2."""
        f_exact = 2.0
        C = 0.2
        f1 = f_exact + C * 1.0
        f2 = f_exact + C * 0.5
        f3 = f_exact + C * 0.25

        result = richardson_extrapolation_three_grid(f1, f2, f3, r=2.0, order=1.0)

        assert abs(result["extrapolated"] - f_exact) < 1e-10
        assert result["gci_fine"] < 0.05

    def test_non_converging_sequence(self):
        """Oscillating sequence should still produce a result."""
        f1, f2, f3 = 1.0, 1.5, 0.8
        result = richardson_extrapolation_three_grid(f1, f2, f3, r=2.0, order=2.0)
        assert math.isfinite(result["extrapolated"])
        assert math.isfinite(result["gci_fine"])

    def test_zero_difference(self):
        """All same values → zero GCI."""
        result = richardson_extrapolation_three_grid(1.0, 1.0, 1.0, r=2.0, order=2.0)
        assert result["gci_fine"] == 0.0
        assert result["extrapolated"] == 1.0


# ============================================================================
# Classification tests
# ============================================================================

class TestClassification:
    """Test the PASS/UNRESOLVED/WITHHELD classification logic."""

    def test_pass_classification(self):
        """Well-resolved case should be PASS."""
        assert classify_standoff(R=10.0, shock_thickness_cells=2.0, gci_fine=0.01) == "PASS"

    def test_unresolved_low_R(self):
        """Low R should be UNRESOLVED."""
        assert classify_standoff(R=1.5, shock_thickness_cells=3.0, gci_fine=0.01) == "UNRESOLVED"

    def test_unresolved_high_gci(self):
        """High GCI should be UNRESOLVED."""
        assert classify_standoff(R=10.0, shock_thickness_cells=2.0, gci_fine=0.05) == "UNRESOLVED"

    def test_unresolved_thick_shock(self):
        """Shock thicker than R/2 should be UNRESOLVED."""
        assert classify_standoff(R=4.0, shock_thickness_cells=3.0, gci_fine=0.01) == "UNRESOLVED"

    def test_withheld_nan_R(self):
        """Non-finite R should be WITHHELD."""
        assert classify_standoff(R=float("nan"), shock_thickness_cells=2.0, gci_fine=0.01) == "WITHHELD"

    def test_withheld_very_low_R(self):
        """R < 1 should be WITHHELD."""
        assert classify_standoff(R=0.5, shock_thickness_cells=2.0, gci_fine=0.01) == "WITHHELD"


# ============================================================================
# Cross-triplet Richardson tests
# ============================================================================

class TestCrossTriplet:
    """Test cross-triplet Richardson extrapolation."""

    def test_consistent_triplets(self):
        """Two overlapping triplets should give similar extrapolated values."""
        # Synthetic data: f(h) = 1.0 + 0.1*h^2 with r=2
        # h=4,2,1,0.5 → f = 2.6, 1.4, 1.1, 1.025
        grid_results = {
            "G1_coarse": {"standoff": 2.6},
            "G2_medium": {"standoff": 1.4},
            "G3_fine": {"standoff": 1.1},
            "G4_finest": {"standoff": 1.025},
        }
        rich = cross_triplet_richardson(grid_results)

        # Both triplets should extrapolate close to 1.0
        for label, r in rich.items():
            assert abs(r["extrapolated"] - 1.0) < 0.1


# ============================================================================
# Solver integration tests (small grids, fast)
# ============================================================================

class TestSolverIntegration:
    """Integration tests with actual solver on small grids."""

    def test_forward_step_runs(self):
        """Forward step solver runs without error."""
        solver, meta = run_forward_step(nx=60, ny=20, mach=3.0, t_end=0.5)
        assert solver.t > 0
        rho, u, v, p = primitive(solver.U, solver.gamma)
        assert rho.min() > 0
        assert p.min() > 0

    def test_corner_mass_balance_runs(self):
        """Corner mass balance computation runs."""
        solver, meta = run_forward_step(nx=60, ny=20, mach=3.0, t_end=0.5)
        corner = corner_mass_balance(solver, step_x=0.6, step_h=0.2)
        assert "mass_residual_rel" in corner
        assert math.isfinite(corner["mass_residual_rel"])
        assert corner["n_cells_in_cv"] > 0

    def test_standoff_measurement_on_coarse_grid(self):
        """Stand-off measurement produces a finite value on coarse grid."""
        from tensorlbm.compressible_shock.standoff import run_four_grid_study
        small_grids = [
            {"nx": 60, "ny": 20, "label": "G1_coarse"},
            {"nx": 120, "ny": 40, "label": "G2_medium"},
        ]
        results = run_four_grid_study(mach=3.0, t_end=0.5, grid_spec=small_grids)
        assert len(results) == 2
        for label, res in results.items():
            assert math.isfinite(res["standoff"])
            assert math.isfinite(res["R"])


# ============================================================================
# Resolvability law tests
# ============================================================================

class TestResolvabilityLaw:
    """Test the resolvability law determination."""

    def test_law_with_synthetic_data(self):
        """Resolvability law should classify grids correctly."""
        # Synthetic: stand-off converges to 0.1, grids have different R
        grid_results = {
            "G1_coarse": {"standoff": 0.12, "R": 1.2, "shock_thickness_cells": 4},
            "G2_medium": {"standoff": 0.105, "R": 4.2, "shock_thickness_cells": 2},
            "G3_fine": {"standoff": 0.101, "R": 8.4, "shock_thickness_cells": 1},
            "G4_finest": {"standoff": 0.1001, "R": 16.8, "shock_thickness_cells": 1},
        }
        law = resolvability_law(grid_results, mach=3.0)
        assert math.isfinite(law["R_min"])
        # G1 should be UNRESOLVED or WITHHELD, G3/G4 should be PASS
        assert law["classifications"]["G1_coarse"]["classification"] in ("UNRESOLVED", "WITHHELD")
        assert law["classifications"]["G4_finest"]["classification"] == "PASS"


# ============================================================================
# Acceptance criteria tests
# ============================================================================

class TestAcceptanceCriteria:
    """Verify acceptance criteria can be checked."""

    def test_monotonic_convergence_check(self):
        """Monotonic sequence should pass."""
        from tensorlbm.compressible_shock.standoff import _check_monotonic
        grid_results = {
            "G1_coarse": {"standoff": 0.15},
            "G2_medium": {"standoff": 0.12},
            "G3_fine": {"standoff": 0.105},
            "G4_finest": {"standoff": 0.101},
        }
        assert _check_monotonic(grid_results) == True

    def test_non_monotonic_check(self):
        """Non-monotonic sequence should fail."""
        from tensorlbm.compressible_shock.standoff import _check_monotonic
        grid_results = {
            "G1_coarse": {"standoff": 0.15},
            "G2_medium": {"standoff": 0.08},  # jump
            "G3_fine": {"standoff": 0.105},
            "G4_finest": {"standoff": 0.101},
        }
        assert _check_monotonic(grid_results) == False

    def test_gci_check(self):
        """GCI check should work."""
        from tensorlbm.compressible_shock.standoff import _check_gci
        richardson = {
            "G2_medium_to_G4_finest": {"gci_fine": 0.02},
        }
        assert _check_gci(richardson) == True

        richardson_bad = {
            "G2_medium_to_G4_finest": {"gci_fine": 0.05},
        }
        assert _check_gci(richardson_bad) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
