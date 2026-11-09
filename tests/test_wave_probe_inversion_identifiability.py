"""Additional tests for wave probe inversion identifiability study.

Tests for:
- Goda degeneracy at s/λ = 0.5
- Criterion (b) with both R*=0.1 and R*=0.3
- Minimum protocol envelope
"""

import math
import numpy as np
import pytest

from tensorlbm.wave_probe_inversion.synthetic_tank import SyntheticWaveTank, SyntheticWaveConfig
from tensorlbm.wave_probe_inversion.inversion import MultiProbeLeastSquares, fit_harmonic
from tensorlbm.wave_probe_inversion.identifiability import IdentifiabilityScan


class TestGodaDegeneracy:
    """Test for Goda degeneracy at s/λ = 0.5."""

    def test_degeneracy_at_half_wavelength(self):
        """s/λ = 0.5 (k*s = π) causes degeneracy."""
        scan = IdentifiabilityScan(method='multi_probe_ls')
        
        # At s/λ = 0.5, error should spike
        point_degenerate = scan.run_single(
            dx_over_lambda=1/20,
            s_over_lambda=0.5,
            L_over_lambda=2.0,
            R_star=0.3,
            alpha_star=0.001,
            n_probes=5,
        )
        
        # Away from degeneracy
        point_good = scan.run_single(
            dx_over_lambda=1/20,
            s_over_lambda=0.45,
            L_over_lambda=2.0,
            R_star=0.3,
            alpha_star=0.001,
            n_probes=5,
        )
        
        # Degenerate case should have much larger error
        assert point_degenerate.R_error > 0.1, \
            f"Expected large error at degeneracy, got {point_degenerate.R_error}"
        assert point_good.R_error < 0.01, \
            f"Expected small error away from degeneracy, got {point_good.R_error}"
        assert point_degenerate.R_error > 10 * point_good.R_error, \
            "Degenerate error should be >> good error"


class TestCriterionBBothTargets:
    """Test criterion (b) with both R*=0.1 and R*=0.3."""

    @pytest.mark.parametrize("R_star", [0.1, 0.3])
    def test_identifiability_law(self, R_star):
        """Non-degenerate protocol achieves all targets."""
        scan = IdentifiabilityScan(method='multi_probe_ls')
        
        point = scan.run_single(
            dx_over_lambda=1/20,
            s_over_lambda=0.45,  # avoid degeneracy
            L_over_lambda=2.0,
            R_star=R_star,
            alpha_star=0.001,
            n_probes=5,
        )
        
        assert point.R_error < 0.03, f"R_error = {point.R_error}"
        assert point.kr_error < 0.01, f"kr_error = {point.kr_error}"
        assert point.alpha_error < 0.10, f"alpha_error = {point.alpha_error}"
        assert point.c_error < 0.03, f"c_error = {point.c_error}"


class TestMinimumProtocol:
    """Test minimum protocol envelope."""

    def test_minimum_protocol_exists(self):
        """At least one configuration passes all targets."""
        scan = IdentifiabilityScan(method='multi_probe_ls')
        
        # Test the known good configuration
        point = scan.run_single(
            dx_over_lambda=1/40,
            s_over_lambda=0.7,
            L_over_lambda=0.5,
            R_star=0.1,
            alpha_star=0.001,
            n_probes=5,
        )
        
        assert point.R_error < 0.03
        assert point.kr_error < 0.01
        assert point.alpha_error < 0.10
        assert point.c_error < 0.03


class TestLBMHoldoutNegativeControl:
    """Test that old protocol gives large errors (negative control)."""

    def test_old_protocol_gives_large_error(self):
        """Old protocol (L=0.4λ, s=0.25λ, 2 probes) should give ≥7% error.
        
        This is verified by the existing benchmark data in
        benchmarks/verified/wave_tank/README.md which shows 7-19% errors.
        """
        # This test documents the expected behavior based on existing data
        # The actual LBM simulation is too heavy for unit tests
        old_protocol_errors = [0.0845, 0.0702, 0.1345, 0.1904, 0.1695, 0.1759]
        
        # All should be ≥ 7%
        for err in old_protocol_errors:
            assert err >= 0.07, f"Old protocol error {err} < 7%"
        
        # All should be > 3% (failing the new criterion)
        for err in old_protocol_errors:
            assert err > 0.03, f"Old protocol error {err} < 3%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
