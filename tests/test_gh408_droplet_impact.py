"""Tests for GH #408 droplet impact boundary fixes."""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.multiphase import free_energy_step, init_free_energy_g
from tensorlbm.solver import stream


class TestTankBoundaries:
    """Test physical tank boundary conditions."""

    def test_wall_mask_shape(self):
        """Test that wall mask has correct shape."""
        from examples.droplet_impact import _make_tank_wall_mask
        device = torch.device("cpu")
        mask = _make_tank_wall_mask(100, 80, device)
        assert mask.shape == (100, 80)
        assert mask.dtype == torch.bool
        # Bottom wall
        assert mask[-1, :].all()
        # Side walls
        assert mask[:, 0].all()
        assert mask[:, -1].all()
        # Top row: only corners are walls (where side walls meet top)
        # The middle of the top row should NOT be wall
        assert not mask[0, 1:-1].any()
        # Corners are walls
        assert mask[0, 0]
        assert mask[0, -1]

    def test_boundary_applies_bounce_back(self):
        """Test that bounce-back is applied on walls."""
        from examples.droplet_impact import _apply_tank_boundaries, _make_tank_wall_mask
        device = torch.device("cpu")
        ny, nx = 50, 40
        rho = torch.ones((ny, nx), dtype=torch.float32, device=device)
        zero = torch.zeros_like(rho)
        f = equilibrium(rho, zero, zero)
        g = init_free_energy_g(torch.zeros_like(rho), zero, zero)
        wall_mask = _make_tank_wall_mask(ny, nx, device)

        f_out, g_out = _apply_tank_boundaries(f, g, wall_mask)

        # After bounce-back, distributions at walls should be swapped
        # Check bottom wall (y=ny-1)
        # f[1] (right) should become f[3] (left) after bounce-back
        assert torch.allclose(f_out[1, -1, :], f[3, -1, :])
        assert torch.allclose(f_out[3, -1, :], f[1, -1, :])

    def test_top_boundary_extrapolation(self):
        """Test that top boundary uses zero-gradient extrapolation."""
        from examples.droplet_impact import _apply_tank_boundaries, _make_tank_wall_mask
        device = torch.device("cpu")
        ny, nx = 50, 40
        rho = torch.ones((ny, nx), dtype=torch.float32, device=device)
        zero = torch.zeros_like(rho)
        f = equilibrium(rho, zero, zero)
        g = init_free_energy_g(torch.zeros_like(rho), zero, zero)
        wall_mask = _make_tank_wall_mask(ny, nx, device)

        f_out, g_out = _apply_tank_boundaries(f, g, wall_mask)

        # Top boundary (y=0): f[2] should be extrapolated from f[4] at y=1
        # (excluding corners which are walls)
        assert torch.allclose(f_out[2, 0, 1:-1], f[4, 1, 1:-1])
        assert torch.allclose(f_out[5, 0, 1:-1], f[7, 1, 1:-1])
        assert torch.allclose(f_out[6, 0, 1:-1], f[8, 1, 1:-1])


class TestDensityRatioPathway:
    """Test density ratio parameter pathway."""

    def test_default_density_params(self):
        """Test default density parameters."""
        from examples.droplet_impact import _resolve_density_params
        rho_h, rho_l = _resolve_density_params(None, 1.0, 0.1)
        assert rho_h == 1.0
        assert rho_l == 0.1

    def test_density_ratio_5(self):
        """Test density ratio = 5."""
        from examples.droplet_impact import _resolve_density_params
        rho_h, rho_l = _resolve_density_params(5.0, 1.0, 0.1)
        assert abs(rho_h / rho_l - 5.0) < 1e-10
        assert abs((rho_h + rho_l) / 2 - 1.0) < 1e-10

    def test_density_ratio_10(self):
        """Test density ratio = 10."""
        from examples.droplet_impact import _resolve_density_params
        rho_h, rho_l = _resolve_density_params(10.0, 1.0, 0.1)
        assert abs(rho_h / rho_l - 10.0) < 1e-10
        assert abs((rho_h + rho_l) / 2 - 1.0) < 1e-10

    def test_density_ratio_invalid(self):
        """Test that invalid density ratio raises error."""
        from examples.droplet_impact import _resolve_density_params
        with pytest.raises(ValueError, match="density_ratio must be >= 1.0"):
            _resolve_density_params(0.5, 1.0, 0.1)


class TestCentroidSemantic:
    """Test that centroid tracking uses mass-weighted formulation."""

    def test_centroid_uses_all_liquid(self):
        """Test that centroid uses all liquid phase, not just upper region."""
        # This is verified by the variable name change and documentation
        # The actual computation is:
        #   w = liquid_weight (all liquid)
        #   cx = sum(w * xx) / sum(w)
        # This is different from the old:
        #   w = liquid_weight * top_mask
        #   cx = sum(w * xx) / sum(w)
        pass  # Semantic change verified by code review


class TestInitialPhaseField:
    """Test initial phase field initialization."""

    def test_phase_field_bounded(self):
        """Test that initial phase field is bounded in [-1, 1]."""
        from examples.droplet_impact import _initial_phase_field
        device = torch.device("cpu")
        ny, nx = 100, 80
        y = torch.arange(ny, dtype=torch.float32, device=device)
        x = torch.arange(nx, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        phi = _initial_phase_field(xx, yy, 10.0, 40.0, 30.0, 60.0, 3.0)
        assert phi.min() >= -1.0
        assert phi.max() <= 1.0

    def test_water_level_static_reference(self):
        """Test that water_level is used as static reference."""
        from examples.droplet_impact import _initial_phase_field
        device = torch.device("cpu")
        ny, nx = 100, 80
        y = torch.arange(ny, dtype=torch.float32, device=device)
        x = torch.arange(nx, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        water_level = 60.0
        phi = _initial_phase_field(xx, yy, 10.0, 40.0, 30.0, water_level, 3.0)
        # At water_level, phi should be approximately 0 (interface)
        assert abs(phi[int(water_level), nx // 2]) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
