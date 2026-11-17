"""Tests for GH #408 droplet impact boundary fixes."""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.multiphase import init_free_energy_g


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

    def test_nonuniform_populations_are_conserved_without_wrap(self):
        from examples.droplet_impact import _make_tank_wall_mask
        from tensorlbm.boundary.tank import stream_tank
        from tensorlbm.d2q9 import OPPOSITE

        mask = _make_tank_wall_mask(9, 10, torch.device("cpu"))
        rng = torch.Generator().manual_seed(42)
        f = torch.rand((9, 9, 10), generator=rng, dtype=torch.float64)
        g = torch.randn((9, 9, 10), generator=rng, dtype=torch.float64)
        a, b = stream_tank(f, g, mask)
        torch.testing.assert_close(a[:, ~mask].sum(), f[:, ~mask].sum(), atol=1e-12, rtol=0)
        torch.testing.assert_close(b[:, ~mask].sum(), g[:, ~mask].sum(), atol=1e-12, rtol=0)
        # Tagged outgoing pulse at the bottom must reflect locally, not appear at top.
        pulse = torch.zeros_like(f)
        pulse[2, -2, 4] = 7.0
        streamed, _ = stream_tank(pulse, pulse, mask)
        assert streamed[int(OPPOSITE[2]), -2, 4] == 7
        assert streamed[:, 0].count_nonzero() == 0
        assert streamed.sum() == 7
        # A top outgoing pulse also reflects at the top.
        pulse.zero_()
        pulse[4, 0, 4] = 3.0
        streamed, _ = stream_tank(pulse, pulse, mask)
        assert streamed[2, 0, 4] == 3
        assert streamed[:, -2].count_nonzero() == 0

    @pytest.mark.parametrize("rho_top", [0.97, 1.03])
    def test_top_pressure_prescribes_density_and_zero_tangential_velocity(self, rho_top):
        from examples.droplet_impact import _make_tank_wall_mask
        from tensorlbm.boundary.tank import stream_tank

        rho = torch.ones((12, 14), dtype=torch.float64)
        f = equilibrium(rho, 0.02 * rho, -0.01 * rho)
        g = torch.randn_like(f)
        mask = _make_tank_wall_mask(12, 14, torch.device("cpu"))
        out, phase = stream_tank(f, g, mask, top_bc="pressure", rho_top=rho_top)
        actual_rho, ux, _ = macroscopic(out)
        torch.testing.assert_close(actual_rho[0, 1:-1], torch.full((12,), rho_top, dtype=rho.dtype))
        torch.testing.assert_close(
            ux[0, 1:-1], torch.zeros(12, dtype=rho.dtype), atol=1e-14, rtol=0
        )
        torch.testing.assert_close(phase[:, ~mask].sum(), g[:, ~mask].sum(), atol=1e-12, rtol=0)


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
        from examples.droplet_impact import _diagnostics

        rho = torch.ones((6, 8), dtype=torch.float64)
        yy, xx = torch.meshgrid(torch.arange(6), torch.arange(8), indexing="ij")
        phi = -rho.clone()
        phi[1, 2] = 1
        phi[4, 6] = 1
        z = torch.zeros_like(rho)
        f = equilibrium(rho, z, z)
        g = init_free_energy_g(phi, z, z)
        result = _diagnostics(
            f, g, torch.zeros_like(rho, dtype=torch.bool), xx, yy, float(g.sum()), 48, 0
        )
        assert result["liquid_centroid_x"] == pytest.approx(4, abs=1e-7)
        assert result["liquid_centroid_y"] == pytest.approx(2.5, abs=1e-7)


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
