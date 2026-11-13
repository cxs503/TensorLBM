"""Tests for voxel image-processing porosity validation.

XVAL-LBM-VOXEL-IMGPROC-POROSITY-REPORT: validates that porosity computed
from voxelized masks matches analytical/reference values within 15%.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.preprocess_geo import random_porosity_mask_2d, random_porosity_mask_3d


# ---------------------------------------------------------------------------
# A1: Random 2-D mask porosity
# ---------------------------------------------------------------------------
class TestRandom2DPorosity:
    """Random 2-D mask: actual porosity should match requested porosity."""

    @pytest.mark.parametrize("phi_target", [0.2, 0.4, 0.6, 0.8])
    def test_uncorrelated_porosity_accuracy(self, phi_target):
        mask = random_porosity_mask_2d(128, 128, porosity=phi_target,
                                       device=torch.device("cpu"), seed=0, sigma=0.0)
        phi_actual = float((~mask).sum()) / float(mask.numel())
        rel_err = abs(phi_actual - phi_target) / phi_target
        assert rel_err < 0.15, f"φ_target={phi_target}, φ_actual={phi_actual:.6f}, rel_err={rel_err:.4f}"

    @pytest.mark.parametrize("sigma", [1.0, 3.0])
    def test_correlated_porosity_accuracy(self, sigma):
        phi_target = 0.5
        mask = random_porosity_mask_2d(128, 128, porosity=phi_target,
                                       device=torch.device("cpu"), seed=0, sigma=sigma)
        phi_actual = float((~mask).sum()) / float(mask.numel())
        rel_err = abs(phi_actual - phi_target) / phi_target
        assert rel_err < 0.15, f"σ={sigma}, φ_actual={phi_actual:.6f}, rel_err={rel_err:.4f}"


# ---------------------------------------------------------------------------
# A2: Random 3-D mask porosity
# ---------------------------------------------------------------------------
class TestRandom3DPorosity:
    """Random 3-D mask: actual porosity should match requested porosity."""

    @pytest.mark.parametrize("phi_target", [0.2, 0.4, 0.6, 0.8])
    def test_3d_porosity_accuracy(self, phi_target):
        mask = random_porosity_mask_3d(64, 64, 64, porosity=phi_target,
                                       device=torch.device("cpu"), seed=0, sigma=0.0)
        phi_actual = float((~mask).sum()) / float(mask.numel())
        rel_err = abs(phi_actual - phi_target) / phi_target
        assert rel_err < 0.15, f"φ_target={phi_target}, φ_actual={phi_actual:.6f}, rel_err={rel_err:.4f}"


# ---------------------------------------------------------------------------
# A3: Analytical geometry porosity
# ---------------------------------------------------------------------------
class TestAnalyticalPorosity:
    """Voxelized analytical geometry porosity should match analytical value."""

    def test_sphere_porosity(self):
        n = 80
        radius = 0.4 * n / 2.0
        z, y, x = np.ogrid[-n//2:n//2, -n//2:n//2, -n//2:n//2]
        r2 = (x + 0.5)**2 + (y + 0.5)**2 + (z + 0.5)**2
        solid = r2 < radius**2
        phi_computed = 1.0 - float(solid.sum()) / float(solid.size)
        v_sphere = (4.0 / 3.0) * math.pi * radius**3
        phi_analytical = 1.0 - v_sphere / float(n)**3
        rel_err = abs(phi_computed - phi_analytical) / phi_analytical
        assert rel_err < 0.15, f"Sphere: φ_comp={phi_computed:.6f}, φ_anal={phi_analytical:.6f}, err={rel_err:.4f}"

    def test_cylinder_porosity(self):
        n = 80
        radius = 0.3 * n / 2.0
        height = 0.6 * n
        z, y, x = np.ogrid[-n//2:n//2, -n//2:n//2, -n//2:n//2]
        r2 = (x + 0.5)**2 + (y + 0.5)**2
        z_phys = (z + 0.5)
        solid = (r2 < radius**2) & (np.abs(z_phys) < height / 2.0)
        phi_computed = 1.0 - float(solid.sum()) / float(solid.size)
        v_cyl = math.pi * radius**2 * height
        phi_analytical = 1.0 - v_cyl / float(n)**3
        rel_err = abs(phi_computed - phi_analytical) / phi_analytical
        assert rel_err < 0.15, f"Cylinder: φ_comp={phi_computed:.6f}, φ_anal={phi_analytical:.6f}, err={rel_err:.4f}"

    def test_ellipsoid_porosity(self):
        n = 80
        a, b, c = 0.4 * n / 2.0, 0.25 * n / 2.0, 0.15 * n / 2.0
        z, y, x = np.ogrid[-n//2:n//2, -n//2:n//2, -n//2:n//2]
        r2 = ((x + 0.5) / a)**2 + ((y + 0.5) / b)**2 + ((z + 0.5) / c)**2
        solid = r2 < 1.0
        phi_computed = 1.0 - float(solid.sum()) / float(solid.size)
        v_ell = (4.0 / 3.0) * math.pi * a * b * c
        phi_analytical = 1.0 - v_ell / float(n)**3
        rel_err = abs(phi_computed - phi_analytical) / phi_analytical
        assert rel_err < 0.15, f"Ellipsoid: φ_comp={phi_computed:.6f}, φ_anal={phi_analytical:.6f}, err={rel_err:.4f}"


# ---------------------------------------------------------------------------
# A4: Convergence
# ---------------------------------------------------------------------------
class TestConvergence:
    """Porosity should converge with increasing resolution."""

    def test_sphere_convergence_decreasing_error(self):
        errs = []
        for n in [20, 40, 80, 160]:
            radius = 0.4 * n / 2.0
            z, y, x = np.ogrid[-n//2:n//2, -n//2:n//2, -n//2:n//2]
            r2 = (x + 0.5)**2 + (y + 0.5)**2 + (z + 0.5)**2
            solid = r2 < radius**2
            phi_computed = 1.0 - float(solid.sum()) / float(solid.size)
            v_sphere = (4.0 / 3.0) * math.pi * radius**3
            phi_analytical = 1.0 - v_sphere / float(n)**3
            rel_err = abs(phi_computed - phi_analytical) / phi_analytical
            errs.append(rel_err)
        # Errors should decrease monotonically
        for i in range(1, len(errs)):
            assert errs[i] < errs[i - 1], f"Error not decreasing: {errs}"

    def test_sphere_convergence_order(self):
        """Convergence order should be >= 1."""
        ns = [20, 40, 80, 160]
        errs = []
        for n in ns:
            radius = 0.4 * n / 2.0
            z, y, x = np.ogrid[-n//2:n//2, -n//2:n//2, -n//2:n//2]
            r2 = (x + 0.5)**2 + (y + 0.5)**2 + (z + 0.5)**2
            solid = r2 < radius**2
            phi_computed = 1.0 - float(solid.sum()) / float(solid.size)
            v_sphere = (4.0 / 3.0) * math.pi * radius**3
            phi_analytical = 1.0 - v_sphere / float(n)**3
            rel_err = abs(phi_computed - phi_analytical) / phi_analytical
            errs.append(rel_err)
        log_h = np.log([1.0 / n for n in ns])
        log_err = np.log(errs)
        p = float(np.polyfit(log_h, log_err, 1)[0])
        assert p >= 1.0, f"Convergence order p={p:.2f} < 1.0"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
class TestDeterminism:
    """Same inputs should give identical outputs."""

    def test_2d_determinism(self):
        m1 = random_porosity_mask_2d(64, 64, porosity=0.5, device=torch.device("cpu"), seed=42)
        m2 = random_porosity_mask_2d(64, 64, porosity=0.5, device=torch.device("cpu"), seed=42)
        assert torch.equal(m1, m2)

    def test_3d_determinism(self):
        m1 = random_porosity_mask_3d(32, 32, 32, porosity=0.5, device=torch.device("cpu"), seed=42)
        m2 = random_porosity_mask_3d(32, 32, 32, porosity=0.5, device=torch.device("cpu"), seed=42)
        assert torch.equal(m1, m2)
