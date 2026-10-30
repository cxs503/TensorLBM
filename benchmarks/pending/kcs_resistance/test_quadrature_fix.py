#!/usr/bin/env python3
"""Unit test for the KCS resistance quadrature domain bug fix (m460).

Verifies that smooth_side_area and smooth_wetted_area return finite,
positive values (the original code returned NaN due to using raw [-1,1]
Gauss-Legendre points for zeta ∈ [0,1]).
"""

import math
import sys
from pathlib import Path

# Add repo to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import numpy as np
import pytest

# Import the functions under test
from run import smooth_side_area, smooth_wetted_area, _kcs_g


class TestQuadratureDomainFix:
    """Test that the Gauss-Legendre zeta mapping fix produces valid areas."""

    @pytest.mark.parametrize("L_lb,B_lb,T_lb", [
        (48.0, 6.72, 2.254),    # L=48 resolution
        (96.0, 13.44, 4.508),   # L=96 resolution
        (230.0, 32.2, 10.8),    # Physical scale (1:1)
    ])
    def test_smooth_side_area_finite_positive(self, L_lb, B_lb, T_lb):
        """smooth_side_area must return a finite positive number."""
        area = smooth_side_area(L_lb, B_lb, T_lb)
        assert not math.isnan(area), f"smooth_side_area returned NaN for L={L_lb}"
        assert not math.isinf(area), f"smooth_side_area returned Inf for L={L_lb}"
        assert area > 0, f"smooth_side_area returned non-positive: {area}"

    @pytest.mark.parametrize("variant", ["half", "double"])
    def test_smooth_wetted_area_finite_positive(self, variant):
        """smooth_wetted_area must return all-finite positive values."""
        L_lb, B_lb, T_lb = 96.0, 13.44, 4.508
        area = smooth_wetted_area(variant, L_lb, B_lb, T_lb)
        for key, val in area.items():
            assert not math.isnan(val), f"{key} is NaN for variant={variant}"
            assert not math.isinf(val), f"{key} is Inf for variant={variant}"
            assert val >= 0, f"{key} is negative: {val} for variant={variant}"
        assert area["S_total"] > 0, f"S_total is zero for variant={variant}"

    def test_double_body_S_top_is_zero(self):
        """Double body variant has no waterplane face."""
        area = smooth_wetted_area("double", 96.0, 13.44, 4.508)
        assert area["S_top"] == 0.0

    def test_half_body_S_top_positive(self):
        """Half body variant has a waterplane face (solid wall)."""
        area = smooth_wetted_area("half", 96.0, 13.44, 4.508)
        assert area["S_top"] > 0

    def test_double_body_4x_side(self):
        """Double body: S_total = 4 * S_one_side (no top)."""
        L_lb, B_lb, T_lb = 96.0, 13.44, 4.508
        s_one = smooth_side_area(L_lb, B_lb, T_lb)
        area = smooth_wetted_area("double", L_lb, B_lb, T_lb)
        assert abs(area["S_total"] - 4.0 * s_one) < 1e-6

    def test_kcs_g_clipping(self):
        """_kcs_g must return values in [0, 1] for any input."""
        xi = np.linspace(-2, 2, 100)
        zeta = np.linspace(-1, 2, 100)
        g = _kcs_g(xi, zeta)
        assert np.all(g >= 0), "_kcs_g returned negative values"
        assert np.all(g <= 1), "_kcs_g returned values > 1"

    def test_area_scales_with_resolution(self):
        """Area should scale as L^2 (lattice units)."""
        a1 = smooth_side_area(48.0, 6.72, 2.254)
        a2 = smooth_side_area(96.0, 13.44, 4.508)
        # Doubling all dimensions should quadruple the area
        ratio = a2 / a1
        assert 3.8 < ratio < 4.2, f"Area scaling ratio {ratio} not close to 4"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
