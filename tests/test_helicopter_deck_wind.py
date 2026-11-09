"""Tests for helicopter deck wind environment module (IN99-N06).

Validates:
* SFS2 geometry generation
* Cube benchmark evaluation
* Deck wind statistics extraction
* CAP 437 safety grading
* Grid convergence assessment
"""

import numpy as np
import pytest

from tensorlbm.helicopter_deck_wind import (
    SFS2Superstructure,
    sfs2_geometry,
    CubeBenchmark,
    DeckWindStatistics,
    CAP437Grading,
    extract_deck_statistics,
    grade_deck_location,
    safety_margin_matrix,
    assess_grid_convergence,
)


class TestSFS2Geometry:
    """Test SFS2 simplified frigate superstructure geometry."""

    def test_sfs2_full_scale(self):
        """Test full-scale SFS2 geometry."""
        sfs2 = sfs2_geometry(scale=1.0)
        assert isinstance(sfs2, SFS2Superstructure)
        assert len(sfs2.boxes) > 0
        assert sfs2.height > 0
        assert sfs2.breadth > 0

    def test_sfs2_model_scale(self):
        """Test model-scale SFS2 geometry (1:16)."""
        sfs2_full = sfs2_geometry(scale=1.0)
        sfs2_model = sfs2_geometry(scale=1.0/16.0)
        
        # Model scale should be 1/16 of full scale
        assert abs(sfs2_model.height - sfs2_full.height / 16.0) < 1e-6
        assert abs(sfs2_model.breadth - sfs2_full.breadth / 16.0) < 1e-6

    def test_sfs2_deck_location(self):
        """Test helicopter deck location is defined."""
        sfs2 = sfs2_geometry(scale=1.0)
        assert len(sfs2.deck_location) == 4
        x0, x1, y0, y1 = sfs2.deck_location
        assert x1 > x0
        assert y1 > y0

    def test_sfs2_to_dict(self):
        """Test geometry serialization."""
        sfs2 = sfs2_geometry(scale=1.0)
        d = sfs2.to_dict()
        assert "boxes" in d
        assert "deck_location" in d
        assert "extent_m" in d


class TestCubeBenchmark:
    """Test cube benchmark verification."""

    def test_cube_benchmark_creation(self):
        """Test cube benchmark initialization."""
        cube = CubeBenchmark(
            side_length_m=1.0,
            reynolds_number=20000.0,
            u_infinity_ms=15.0,
        )
        assert cube.side_length_m == 1.0
        assert cube.reynolds_number == 20000.0
        assert cube.cd_reference == 1.05

    def test_cube_benchmark_evaluate_pass(self):
        """Test cube benchmark evaluation with passing values."""
        cube = CubeBenchmark(
            side_length_m=1.0,
            reynolds_number=20000.0,
            u_infinity_ms=15.0,
        )
        result = cube.evaluate(
            cd_measured=1.03,  # within 10% of 1.05
            sep_bubble_measured=1.70,  # within 10% of 1.75
            reattach_measured=2.70,  # within 10% of 2.75
        )
        assert result["overall_pass"]
        assert result["cd_pass"]
        assert result["separation_bubble_pass"]
        assert result["reattachment_point_pass"]

    def test_cube_benchmark_evaluate_fail(self):
        """Test cube benchmark evaluation with failing values."""
        cube = CubeBenchmark(
            side_length_m=1.0,
            reynolds_number=20000.0,
            u_infinity_ms=15.0,
        )
        result = cube.evaluate(
            cd_measured=1.50,  # >10% error
            sep_bubble_measured=1.70,
            reattach_measured=2.70,
        )
        assert not result["overall_pass"]
        assert not result["cd_pass"]


class TestDeckWindStatistics:
    """Test deck wind statistics extraction."""

    def test_extract_statistics(self):
        """Test wind statistics extraction from synthetic field."""
        nx, ny = 10, 8
        x_coords = np.linspace(0, 6, nx)
        y_coords = np.linspace(-1.5, 1.5, ny)
        u_field = np.random.uniform(10, 20, (ny, nx))
        u_rms_field = 0.1 * u_field

        stats = extract_deck_statistics(
            u_field=u_field,
            x_coords=x_coords,
            y_coords=y_coords,
            wind_heading_deg=0.0,
            u_rms_field=u_rms_field,
        )

        assert isinstance(stats, DeckWindStatistics)
        assert stats.grid_shape == (ny, nx)
        assert stats.max_wind_speed() > 0
        assert stats.max_turbulence_intensity() > 0

    def test_extract_statistics_default_rms(self):
        """Test wind statistics with default RMS estimation."""
        nx, ny = 10, 8
        x_coords = np.linspace(0, 6, nx)
        y_coords = np.linspace(-1.5, 1.5, ny)
        u_field = np.ones((ny, nx)) * 15.0

        stats = extract_deck_statistics(
            u_field=u_field,
            x_coords=x_coords,
            y_coords=y_coords,
            wind_heading_deg=0.0,
            u_rms_field=None,  # Should estimate as 10% of mean
        )

        # TI should be approximately 10%
        assert abs(stats.mean_turbulence_intensity() - 10.0) < 1.0


class TestCAP437Grading:
    """Test CAP 437 safety grading."""

    def test_grade_deck_location_green(self):
        """Test Zone 1 (Green) grading."""
        zone = grade_deck_location(u_mean_ms=12.0, ti_percent=8.0)
        assert zone == 1

    def test_grade_deck_location_yellow(self):
        """Test Zone 2 (Yellow) grading."""
        zone = grade_deck_location(u_mean_ms=17.0, ti_percent=12.0)
        assert zone == 2

    def test_grade_deck_location_red(self):
        """Test Zone 3 (Red) grading."""
        zone = grade_deck_location(u_mean_ms=22.0, ti_percent=18.0)
        assert zone == 3

    def test_safety_margin_matrix(self):
        """Test safety margin matrix generation."""
        nx, ny = 10, 8
        x_coords = np.linspace(0, 6, nx)
        y_coords = np.linspace(-1.5, 1.5, ny)
        
        # Create synthetic field with mixed zones
        u_field = np.ones((ny, nx)) * 15.0
        u_field[0:3, :] = 22.0  # Red zone
        u_field[3:6, :] = 17.0  # Yellow zone
        u_field[6:8, :] = 12.0  # Green zone
        
        u_rms_field = 0.1 * u_field

        stats = extract_deck_statistics(
            u_field=u_field,
            x_coords=x_coords,
            y_coords=y_coords,
            wind_heading_deg=0.0,
            u_rms_field=u_rms_field,
        )

        grading = safety_margin_matrix(stats)

        assert isinstance(grading, CAP437Grading)
        assert grading.zone_matrix.shape == (ny, nx)
        assert 1 in grading.zone_counts or 2 in grading.zone_counts or 3 in grading.zone_counts

    def test_safety_margin_matrix_fail_closed(self):
        """Test fail-closed policy for invalid data."""
        nx, ny = 5, 5
        x_coords = np.linspace(0, 6, nx)
        y_coords = np.linspace(-1.5, 1.5, ny)
        
        u_field = np.ones((ny, nx)) * 15.0
        u_field[0, 0] = np.nan  # Invalid data
        
        u_rms_field = 0.1 * u_field

        stats = extract_deck_statistics(
            u_field=u_field,
            x_coords=x_coords,
            y_coords=y_coords,
            wind_heading_deg=0.0,
            u_rms_field=u_rms_field,
        )

        grading = safety_margin_matrix(stats, fail_closed=True)

        # Invalid point should be Zone 3 (Red) due to fail-closed
        assert grading.zone_matrix[0, 0] == 3


class TestGridConvergence:
    """Test grid convergence assessment."""

    def test_assess_grid_convergence(self):
        """Test GCI grid convergence assessment."""
        # Synthetic convergence study
        values = [1.10, 1.06, 1.05]  # coarse, medium, fine
        grid_spacings = [0.20, 0.10, 0.05]  # dx

        result = assess_grid_convergence(values, grid_spacings)

        assert "p_observed" in result
        assert "gci_fine_percent" in result
        assert "convergence_achieved" in result
        assert result["gci_fine_percent"] >= 0

    def test_assess_grid_convergence_insufficient(self):
        """Test GCI with insufficient grid levels."""
        with pytest.raises(ValueError, match="exactly three grid levels"):
            assess_grid_convergence([1.0, 1.1], [0.1, 0.05])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
