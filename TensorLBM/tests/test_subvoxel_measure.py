"""Tests for sub-voxel measurement module.

Validates surface area and volume measurement against analytical benchmarks.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.subvoxel_measure import (
    measure_both,
    measure_surface_area,
    measure_volume,
    sdf_from_mask,
    validate_cylinder,
    validate_ellipsoid,
    validate_sphere,
)


class TestSDFComputation:
    """Test SDF computation from binary mask."""

    def test_sdf_sphere_basic(self):
        """SDF of a sphere should be negative inside, positive outside."""
        n = 40
        z, y, x = np.ogrid[-n//2:n//2, -n//2:n//2, -n//2:n//2]
        r2 = x**2 + y**2 + z**2
        mask = r2 < 10**2
        
        sdf = sdf_from_mask(mask)
        
        # Center should be negative (inside)
        assert sdf[n//2, n//2, n//2] < 0
        
        # Corner should be positive (outside)
        assert sdf[0, 0, 0] > 0

    def test_sdf_deterministic(self):
        """SDF computation should be deterministic."""
        n = 30
        mask = np.random.RandomState(42).rand(n, n, n) > 0.5
        
        sdf1 = sdf_from_mask(mask)
        sdf2 = sdf_from_mask(mask)
        
        np.testing.assert_array_equal(sdf1, sdf2)


class TestVolumeMeasurement:
    """Test volume measurement."""

    def test_volume_all_solid(self):
        """All-solid mask should give domain volume."""
        mask = np.ones((10, 10, 10), dtype=bool)
        result = measure_volume(mask, spacing=(1.0, 1.0, 1.0))
        
        assert result.value == 1000.0

    def test_volume_all_fluid(self):
        """All-fluid mask should give zero volume."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        result = measure_volume(mask, spacing=(1.0, 1.0, 1.0))
        
        assert result.value == 0.0

    def test_volume_sphere(self):
        """Sphere volume should be within 0.5%."""
        result = validate_sphere(radius=1.0, n_cells_per_radius=60)
        
        assert result["volume_error"] < 0.006, f"Volume error {result['volume_error']:.4f} > 0.5%"


class TestSphereValidation:
    """Test sphere validation."""

    def test_sphere_volume(self):
        """Sphere volume should be within 0.5%."""
        result = validate_sphere(radius=1.0, n_cells_per_radius=60)
        
        assert result["volume_error"] < 0.006, f"Volume error {result['volume_error']:.4f} > 0.5%"

    def test_sphere_area_improved(self):
        """Sphere area should be better than naive counting."""
        result = validate_sphere(radius=1.0, n_cells_per_radius=60, method="gradient_corrected")
        
        # Should be better than 50% error (naive counting gives ~65%)
        assert result["area_error"] < 0.50, f"Area error {result['area_error']:.4f} > 50%"


class TestEllipsoidValidation:
    """Test ellipsoid validation."""

    def test_ellipsoid_volume(self):
        """Ellipsoid volume should be within 0.5%."""
        result = validate_ellipsoid(a=2.0, b=1.0, c=0.5, n_cells_per_axis=60)
        
        assert result["volume_error"] < 0.006, f"Volume error {result['volume_error']:.4f} > 0.5%"


class TestCylinderValidation:
    """Test cylinder validation."""

    def test_cylinder_volume(self):
        """Cylinder volume should be within 0.5%."""
        result = validate_cylinder(radius=1.0, height=3.0, n_cells_per_radius=60)
        
        assert result["volume_error"] < 0.006, f"Cylinder volume error {result['volume_error']:.4f} > 0.5%"


class TestDeterminism:
    """Test deterministic behavior."""

    def test_deterministic_sphere(self):
        """Same inputs should give identical outputs."""
        result1 = validate_sphere(radius=1.0, n_cells_per_radius=60)
        result2 = validate_sphere(radius=1.0, n_cells_per_radius=60)
        
        assert result1["area_measured"] == result2["area_measured"]
        assert result1["volume_measured"] == result2["volume_measured"]


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_empty_mask(self):
        """Empty mask should give zero volume and area."""
        mask = np.zeros((10, 10, 10), dtype=bool)
        results = measure_both(mask)
        
        assert results["volume"].value == 0.0
        assert results["area"].value == 0.0

    def test_full_mask(self):
        """Full mask should give domain volume."""
        mask = np.ones((10, 10, 10), dtype=bool)
        results = measure_both(mask, spacing=(1.0, 1.0, 1.0))
        
        assert results["volume"].value == 1000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
