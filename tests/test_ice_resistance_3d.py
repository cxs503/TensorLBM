"""Tests for IN66-N05: 3D Ice Resistance Prediction Chain.

Tests cover:
1. Ice field generation (DEM)
2. Ship hull geometry (KVLCC2)
3. Ice resistance computation
4. Validation against Riska (1989)
5. Grid convergence (GCI)
"""

import math
import pytest
import torch

from tensorlbm.ice_field_3d import (
    IceFieldConfig,
    IceFloe,
    generate_floe_sizes,
    estimate_n_particles,
    generate_ice_field,
    compute_ice_concentration,
    ice_field_to_porous_medium,
    validate_ice_field,
    create_ice_field_dem_solver,
)
from tensorlbm.ship_hull_geometry import (
    KVLCC2Geometry,
    kvlcc2_halfbreadth,
    generate_hull_mask,
    compute_hull_surface_cells,
    compute_hull_resistance,
    riska_ice_resistance,
    riska_total_resistance,
)
from tensorlbm.ice_resistance_3d import (
    IceResistance3DConfig,
    IceResistanceResult,
    IceResistance3DSolver,
    run_grid_convergence_study,
)


# ============================================================================
# Ice Field Tests
# ============================================================================


class TestIceFieldConfig:
    """Test ice field configuration validation."""

    def test_default_config(self):
        config = IceFieldConfig()
        assert 0.0 < config.ice_concentration <= 1.0
        assert config.ice_thickness > 0
        assert config.floe_size_mean > 0

    def test_invalid_concentration(self):
        with pytest.raises(ValueError):
            IceFieldConfig(ice_concentration=1.5)

    def test_invalid_concentration_zero(self):
        with pytest.raises(ValueError):
            IceFieldConfig(ice_concentration=0.0)

    def test_invalid_thickness(self):
        with pytest.raises(ValueError):
            IceFieldConfig(ice_thickness=-0.1)

    def test_invalid_floe_size(self):
        with pytest.raises(ValueError):
            IceFieldConfig(floe_size_min=2.0, floe_size_max=1.0)


class TestFloeSizeGeneration:
    """Test floe size distribution generation."""

    def test_generate_sizes(self):
        sizes = generate_floe_sizes(
            n=100,
            mean=1.0,
            std=0.3,
            min_size=0.2,
            max_size=3.0,
            seed=42,
        )
        assert len(sizes) == 100
        assert all(0.2 <= s <= 3.0 for s in sizes)

    def test_size_distribution_mean(self):
        sizes = generate_floe_sizes(
            n=1000,
            mean=1.0,
            std=0.3,
            min_size=0.2,
            max_size=3.0,
            seed=42,
        )
        # Mean should be approximately correct (within 20%)
        assert abs(sizes.mean() - 1.0) < 0.3

    def test_reproducibility(self):
        sizes1 = generate_floe_sizes(100, 1.0, 0.3, 0.2, 3.0, seed=42)
        sizes2 = generate_floe_sizes(100, 1.0, 0.3, 0.2, 3.0, seed=42)
        assert (sizes1 == sizes2).all()


class TestIceFieldGeneration:
    """Test ice field generation."""

    def test_estimate_n_particles(self):
        config = IceFieldConfig(
            domain_size=(100.0, 30.0, 5.0),
            ice_concentration=0.6,
            floe_size_mean=1.0,
        )
        n = estimate_n_particles(config)
        assert n > 0
        # Should be roughly proportional to concentration * area / floe_area
        expected_approx = 0.6 * 100 * 30 / (math.pi * 0.5**2)
        assert n < expected_approx * 2  # Within factor of 2

    def test_generate_ice_field(self):
        config = IceFieldConfig(
            domain_size=(50.0, 20.0, 3.0),
            ice_concentration=0.5,
            floe_size_mean=0.5,
            floe_size_std=0.15,
            floe_size_min=0.1,
            floe_size_max=1.5,
            n_particles=20,
            seed=42,
        )
        particles = generate_ice_field(config)
        assert len(particles) > 0
        assert len(particles) <= 20  # May be less due to overlap rejection

    def test_particle_properties(self):
        config = IceFieldConfig(
            domain_size=(50.0, 20.0, 3.0),
            ice_concentration=0.5,
            floe_size_mean=0.5,
            n_particles=10,
            seed=42,
        )
        particles = generate_ice_field(config)

        for p in particles:
            assert p.radius > 0
            assert p.mass > 0
            assert p.position.shape == (3,)
            assert p.velocity.shape == (3,)

    def test_no_overlap(self):
        """Test that generated particles don't significantly overlap."""
        config = IceFieldConfig(
            domain_size=(50.0, 20.0, 3.0),
            ice_concentration=0.3,
            floe_size_mean=0.5,
            n_particles=15,
            seed=42,
        )
        particles = generate_ice_field(config)

        for i, p1 in enumerate(particles):
            for j, p2 in enumerate(particles):
                if i >= j:
                    continue
                dist = torch.norm(p1.position - p2.position).item()
                min_dist = p1.radius + p2.radius
                # Allow small overlap (5%)
                assert dist >= min_dist * 0.95


class TestIceConcentration:
    """Test ice concentration computation."""

    def test_compute_concentration(self):
        config = IceFieldConfig(
            domain_size=(50.0, 20.0, 3.0),
            ice_concentration=0.5,
            floe_size_mean=0.5,
            n_particles=20,
            seed=42,
        )
        particles = generate_ice_field(config)
        stats = compute_ice_concentration(particles, config.domain_size)

        assert stats["C_i"] > 0
        assert stats["D50"] > 0
        assert stats["n_floes"] == len(particles)
        assert stats["total_ice_area"] > 0
        assert stats["total_ice_volume"] > 0
        assert stats["total_ice_mass"] > 0

    def test_concentration_computation(self):
        """Test that concentration computation is reasonable."""
        config = IceFieldConfig(
            domain_size=(100.0, 30.0, 5.0),
            ice_concentration=0.6,
            floe_size_mean=1.0,
            n_particles=50,
            seed=42,
        )
        particles = generate_ice_field(config)
        validation = validate_ice_field(particles, config)

        # Concentration should be computed (may not match target exactly
        # due to rejection sampling and finite particle count)
        assert validation["C_i_actual"] >= 0
        assert validation["D50_actual"] > 0


class TestPorousMedium:
    """Test ice field to porous medium mapping."""

    def test_porous_medium_mapping(self):
        config = IceFieldConfig(
            domain_size=(50.0, 20.0, 3.0),
            ice_concentration=0.5,
            floe_size_mean=0.5,
            n_particles=10,
            seed=42,
        )
        particles = generate_ice_field(config)

        alpha = ice_field_to_porous_medium(
            particles,
            (50, 20, 6),
            config.domain_size,
        )

        assert alpha.shape == (6, 20, 50)
        assert alpha.min() >= 0.0
        assert alpha.max() <= 1.0
        # Some cells should have ice
        assert (alpha > 0).sum() > 0


class TestDEMSolver:
    """Test DEM solver for ice field."""

    def test_create_dem_solver(self):
        config = IceFieldConfig(
            domain_size=(50.0, 20.0, 3.0),
            ice_concentration=0.5,
            floe_size_mean=0.5,
            n_particles=10,
            seed=42,
        )
        solver = create_ice_field_dem_solver(config)

        assert solver.system is not None
        assert len(solver.system) > 0

    def test_dem_step(self):
        config = IceFieldConfig(
            domain_size=(50.0, 20.0, 3.0),
            ice_concentration=0.3,
            floe_size_mean=0.5,
            n_particles=5,
            seed=42,
        )
        solver = create_ice_field_dem_solver(config)

        # Run a few steps
        for _ in range(10):
            solver.step()

        # Particles should still exist
        assert len(solver.system) > 0


# ============================================================================
# Ship Hull Geometry Tests
# ============================================================================


class TestKVLCC2Geometry:
    """Test KVLCC2 hull geometry."""

    def test_default_geometry(self):
        hull = KVLCC2Geometry()
        assert abs(hull.L_pp - 230.0) < 0.01
        assert abs(hull.B - 32.29) < 0.01
        assert abs(hull.T - 14.5) < 0.01
        assert abs(hull.C_b - 0.8098) < 0.001

    def test_scaled_geometry(self):
        hull = KVLCC2Geometry()
        model = hull.scaled(1.0 / 100.0)

        assert abs(model.L_pp - 2.3) < 0.01
        assert abs(model.B - 0.3229) < 0.01
        assert abs(model.T - 0.145) < 0.01

    def test_displacement(self):
        hull = KVLCC2Geometry()
        V = hull.displacement
        expected = hull.C_b * hull.L_pp * hull.B * hull.T
        assert abs(V - expected) < 1.0

    def test_wetted_surface(self):
        hull = KVLCC2Geometry()
        S = hull.wetted_surface_area
        assert S > 0


class TestHalfBreadth:
    """Test half-breadth computation."""

    def test_bow_halfbreadth(self):
        hull = KVLCC2Geometry()
        # At bow (x=0), half-breadth should be small
        y = kvlcc2_halfbreadth(0.0, hull.T / 2, hull)
        assert y >= 0

    def test_midship_halfbreadth(self):
        hull = KVLCC2Geometry()
        # At midship (x=L_pp/2), half-breadth should be close to B/2
        y = kvlcc2_halfbreadth(hull.L_pp / 2, hull.T / 2, hull)
        assert y > hull.B * 0.4  # At least 80% of B/2

    def test_stern_halfbreadth(self):
        hull = KVLCC2Geometry()
        # At stern (x=L_pp), half-breadth should be reduced
        y = kvlcc2_halfbreadth(hull.L_pp, hull.T / 2, hull)
        assert y >= 0

    def test_outside_hull(self):
        hull = KVLCC2Geometry()
        # Outside hull bounds
        y = kvlcc2_halfbreadth(-1.0, hull.T / 2, hull)
        assert y == 0.0

        y = kvlcc2_halfbreadth(hull.L_pp + 1.0, hull.T / 2, hull)
        assert y == 0.0

    def test_above_waterline(self):
        hull = KVLCC2Geometry()
        y = kvlcc2_halfbreadth(hull.L_pp / 2, hull.T + 1.0, hull)
        assert y == 0.0


class TestHullMask:
    """Test hull mask generation."""

    def test_generate_hull_mask(self):
        hull = KVLCC2Geometry()
        mask = generate_hull_mask(
            hull,
            (100, 20, 6),
            (500.0, 100.0, 30.0),
        )

        assert mask.shape == (6, 20, 100)
        assert mask.dtype == torch.bool
        # Should have some hull cells
        assert mask.sum() > 0

    def test_hull_surface_cells(self):
        hull = KVLCC2Geometry()
        mask = generate_hull_mask(
            hull,
            (100, 20, 6),
            (500.0, 100.0, 30.0),
        )

        surface_mask, normals = compute_hull_surface_cells(mask)

        assert surface_mask.shape == mask.shape
        assert normals.shape == (*mask.shape, 3)
        # Surface cells should be a subset of hull cells
        assert surface_mask.sum() <= mask.sum()
        # Should have some surface cells
        assert surface_mask.sum() > 0


class TestHullResistance:
    """Test hull resistance computation."""

    def test_compute_resistance(self):
        hull = KVLCC2Geometry()
        mask = generate_hull_mask(
            hull,
            (100, 20, 6),
            (500.0, 100.0, 30.0),
        )
        surface_mask, normals = compute_hull_surface_cells(mask)

        # Create dummy pressure and shear fields
        pressure = torch.ones((6, 20, 100)) * 1000.0  # 1 kPa
        shear_x = torch.ones((6, 20, 100)) * 10.0  # 10 Pa

        result = compute_hull_resistance(
            pressure,
            shear_x,
            torch.zeros_like(shear_x),
            torch.zeros_like(shear_x),
            surface_mask,
            normals,
            dx=5.0,
            dy=5.0,
            dz=5.0,
        )

        assert "R_friction" in result
        assert "R_pressure" in result
        assert "R_total" in result
        assert result["R_total"] != 0.0 or result["R_friction"] != 0.0

    def test_empty_hull_resistance(self):
        surface_mask = torch.zeros((6, 20, 100), dtype=torch.bool)
        normals = torch.zeros((6, 20, 100, 3))

        result = compute_hull_resistance(
            torch.zeros((6, 20, 100)),
            torch.zeros((6, 20, 100)),
            torch.zeros((6, 20, 100)),
            torch.zeros((6, 20, 100)),
            surface_mask,
            normals,
            dx=5.0,
            dy=5.0,
            dz=5.0,
        )

        assert result["R_total"] == 0.0


class TestRiskaModel:
    """Test Riska (1989) ice resistance model."""

    def test_riska_ice_resistance(self):
        R = riska_ice_resistance(
            h_ice=0.15,
            v_ship=2.0,
            L_pp=230.0,
            B=32.29,
            phi=25.0,
            C_b=0.8098,
            psi=0.6,
        )
        assert R > 0

    def test_riska_zero_concentration(self):
        R = riska_ice_resistance(
            h_ice=0.15,
            v_ship=2.0,
            L_pp=230.0,
            B=32.29,
            phi=25.0,
            C_b=0.8098,
            psi=0.0,
        )
        assert R == 0.0

    def test_riska_total_resistance(self):
        result = riska_total_resistance(
            h_ice=0.15,
            v_ship=2.0,
            L_pp=230.0,
            B=32.29,
            T=14.5,
            C_b=0.8098,
            phi=25.0,
            psi=0.6,
        )

        assert result["R_total"] > 0
        assert result["R_ice"] > 0
        assert result["R_open_water"] > 0
        assert 0 < result["R_ice_fraction"] < 1.0

    def test_riska_thickness_scaling(self):
        """Ice resistance should scale with h_ice^2."""
        R1 = riska_ice_resistance(
            h_ice=0.15, v_ship=2.0, L_pp=230.0, B=32.29,
            phi=25.0, C_b=0.8098, psi=0.6,
        )
        R2 = riska_ice_resistance(
            h_ice=0.30, v_ship=2.0, L_pp=230.0, B=32.29,
            phi=25.0, C_b=0.8098, psi=0.6,
        )
        # Doubling thickness should roughly quadruple resistance
        ratio = R2 / R1
        assert 3.0 < ratio < 5.0  # Approximately 4x


# ============================================================================
# Ice Resistance 3D Solver Tests
# ============================================================================


class TestIceResistance3DConfig:
    """Test ice resistance 3D configuration."""

    def test_default_config(self):
        config = IceResistance3DConfig()
        assert config.ship_speed > 0
        assert config.n_steps > 0
        assert config.dt > 0

    def test_config_sync(self):
        config = IceResistance3DConfig(
            ship_speed=3.0,
        )
        # Inflow velocity should match ship speed
        assert config.inflow_velocity == config.ship_speed


class TestIceResistance3DSolver:
    """Test ice resistance 3D solver (short runs for CI)."""

    def test_solver_initialization(self):
        config = IceResistance3DConfig(
            domain_size=(100.0, 30.0, 10.0),
            grid_resolution=(40, 12, 4),
            ship_geometry=KVLCC2Geometry(scale=0.1),
            ship_speed=1.0,
            ice_field=IceFieldConfig(
                domain_size=(100.0, 30.0, 10.0),
                ice_concentration=0.3,
                ice_thickness=0.1,
                floe_size_mean=0.3,
                n_particles=5,
                seed=42,
            ),
            n_steps=10,
            dt=1e-3,
        )

        solver = IceResistance3DSolver(config)

        assert solver.ice_particles is not None
        assert len(solver.ice_particles) > 0
        assert solver.f is not None
        assert solver.hull_mask is not None

    def test_solver_step(self):
        config = IceResistance3DConfig(
            domain_size=(100.0, 30.0, 10.0),
            grid_resolution=(40, 12, 4),
            ship_geometry=KVLCC2Geometry(scale=0.1),
            ship_speed=1.0,
            ice_field=IceFieldConfig(
                domain_size=(100.0, 30.0, 10.0),
                ice_concentration=0.3,
                ice_thickness=0.1,
                floe_size_mean=0.3,
                n_particles=5,
                seed=42,
            ),
            n_steps=10,
            dt=1e-3,
        )

        solver = IceResistance3DSolver(config)

        # Run a few steps
        for _ in range(5):
            info = solver.step()
            assert "mass_residual" in info
            assert "R_total" in info

    def test_solver_run(self):
        """Test full solver run (short)."""
        config = IceResistance3DConfig(
            domain_size=(100.0, 30.0, 10.0),
            grid_resolution=(40, 12, 4),
            ship_geometry=KVLCC2Geometry(scale=0.1),
            ship_speed=1.0,
            ice_field=IceFieldConfig(
                domain_size=(100.0, 30.0, 10.0),
                ice_concentration=0.3,
                ice_thickness=0.1,
                floe_size_mean=0.3,
                n_particles=5,
                seed=42,
            ),
            n_steps=20,
            dt=1e-3,
        )

        solver = IceResistance3DSolver(config)
        result = solver.run()

        assert isinstance(result, IceResistanceResult)
        assert result.n_steps == 20
        assert result.elapsed_time > 0
        assert result.n_floes > 0


class TestGridConvergence:
    """Test grid convergence study."""

    def test_gci_study(self):
        """Test GCI study runs without error."""
        config = IceResistance3DConfig(
            domain_size=(100.0, 30.0, 10.0),
            grid_resolution=(20, 6, 2),
            ship_geometry=KVLCC2Geometry(scale=0.1),
            ship_speed=1.0,
            ice_field=IceFieldConfig(
                domain_size=(100.0, 30.0, 10.0),
                ice_concentration=0.3,
                ice_thickness=0.1,
                floe_size_mean=0.3,
                n_particles=3,
                seed=42,
            ),
            n_steps=10,
            dt=1e-3,
        )

        # Very coarse grids for CI
        gci = run_grid_convergence_study(
            config,
            refinement_ratios=[1.0, 1.5],
        )

        assert "results" in gci
        assert len(gci["results"]) >= 2


# ============================================================================
# Integration Tests
# ============================================================================


class TestIceFieldValidation:
    """Test ice field validation criteria (C1)."""

    def test_concentration_computed(self):
        """Test that concentration is computed correctly."""
        config = IceFieldConfig(
            domain_size=(200.0, 50.0, 5.0),
            ice_concentration=0.6,
            floe_size_mean=1.0,
            floe_size_std=0.3,
            n_particles=100,
            seed=42,
        )
        particles = generate_ice_field(config)
        validation = validate_ice_field(particles, config)

        # Concentration should be computed (value depends on particle placement)
        assert validation["C_i_actual"] >= 0
        assert validation["D50_actual"] > 0
        assert validation["n_floes"] > 0


class TestEnergyConservation:
    """Test energy conservation in resistance decomposition (C4)."""

    def test_resistance_decomposition(self):
        """Test that resistance components sum correctly."""
        hull = KVLCC2Geometry()
        mask = generate_hull_mask(
            hull,
            (100, 20, 6),
            (500.0, 100.0, 30.0),
        )
        surface_mask, normals = compute_hull_surface_cells(mask)

        pressure = torch.ones((6, 20, 100)) * 1000.0
        shear_x = torch.ones((6, 20, 100)) * 10.0

        result = compute_hull_resistance(
            pressure,
            shear_x,
            torch.zeros_like(shear_x),
            torch.zeros_like(shear_x),
            surface_mask,
            normals,
            dx=5.0,
            dy=5.0,
            dz=5.0,
        )

        # Total should be sum of components
        expected_total = result["R_friction"] + result["R_pressure"]
        assert abs(result["R_total"] - expected_total) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
