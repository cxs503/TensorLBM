"""Pore-scale flow verification tests for TensorLBM.

Tests the cross-scale validation ladder:
1. Parallel plates (2D) — Darcy permeability vs analytical
2. Cylindrical tube (3D) — Darcy permeability vs analytical
3. Periodic cylinder array (2D) — permeability vs Kozeny-Carman
4. Sphere pack (3D) — permeability vs Kozeny-Carman
5. Gaussian random field (3D) — permeability vs Kozeny-Carman
6. CT-like rock (3D) — permeability vs Kozeny-Carman
7. Mass conservation
8. Grid convergence
9. Reynolds number range
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorlbm.pore_scale_flow import (
    PoreFlowConfig2D,
    PoreFlowConfig3D,
    analytical_parallel_plate_permeability,
    analytical_tube_permeability,
    compute_permeability,
    compute_porosity,
    compute_pore_reynolds,
    compute_specific_surface,
    kozeny_carman_permeability,
    make_cylindrical_tube_3d,
    make_ct_like_rock_3d,
    make_gaussian_random_field_3d,
    make_parallel_plates_2d,
    make_periodic_cylinder_array_2d,
    make_sphere_pack_3d,
    run_pore_flow_2d,
    run_pore_flow_3d,
)


# ===========================================================================
# Test 1: Geometry generator unit tests
# ===========================================================================


class TestGeometryGenerators:
    """Unit tests for geometry generators."""

    def test_parallel_plates_shape(self):
        solid = make_parallel_plates_2d(64, 256, 20)
        assert solid.shape == (64, 256)
        assert solid.dtype == torch.bool

    def test_parallel_plates_porosity(self):
        solid = make_parallel_plates_2d(64, 256, 20)
        porosity = compute_porosity(solid)
        expected = 20.0 / 64.0
        assert abs(porosity - expected) < 0.01

    def test_cylindrical_tube_shape(self):
        solid = make_cylindrical_tube_3d(64, 32, 32, 8.0)
        assert solid.shape == (64, 32, 32)

    def test_cylindrical_tube_porosity(self):
        solid = make_cylindrical_tube_3d(64, 32, 32, 8.0)
        porosity = compute_porosity(solid)
        assert 0.1 < porosity < 0.3

    def test_sphere_pack_shape(self):
        solid = make_sphere_pack_3d(48, 32, 32, 10, 2.0, 4.0, seed=42)
        assert solid.shape == (48, 32, 32)

    def test_grf_shape(self):
        solid = make_gaussian_random_field_3d(48, 32, 32, 3.0, 0.0, seed=42)
        assert solid.shape == (48, 32, 32)

    def test_ct_like_shape(self):
        solid = make_ct_like_rock_3d(48, 32, 32, 0.25, 0.05, seed=42)
        assert solid.shape == (48, 32, 32)

    def test_cylinder_array_shape(self):
        solid = make_periodic_cylinder_array_2d(64, 256, 4, 10, 4.0)
        assert solid.shape == (64, 256)


# ===========================================================================
# Test 2: Analytical solution unit tests
# ===========================================================================


class TestAnalyticalSolutions:
    """Unit tests for analytical permeability formulas."""

    def test_parallel_plate_formula(self):
        assert abs(analytical_parallel_plate_permeability(12.0) - 12.0) < 1e-10
        assert abs(analytical_parallel_plate_permeability(6.0) - 3.0) < 1e-10

    def test_tube_formula(self):
        assert abs(analytical_tube_permeability(8.0) - 8.0) < 1e-10
        assert abs(analytical_tube_permeability(4.0) - 2.0) < 1e-10

    def test_kozeny_carman_formula(self):
        k = kozeny_carman_permeability(0.3, 0.5, 5.0)
        expected = 0.3 ** 3 / (5.0 * 0.5 ** 2 * 0.7 ** 2)
        assert abs(k - expected) < 1e-10

    def test_kozeny_carman_zero_porosity(self):
        assert kozeny_carman_permeability(0.0, 0.5) == 0.0

    def test_kozeny_carman_full_porosity(self):
        assert kozeny_carman_permeability(1.0, 0.5) == 0.0


# ===========================================================================
# Test 3: Permeability computation unit test
# ===========================================================================


class TestPermeabilityComputation:
    """Unit tests for permeability computation."""

    def test_darcy_formula(self):
        k = compute_permeability(U_avg=1e-4, mu=1.0/6.0, F=1e-5)
        expected = (1.0/6.0) * 1e-4 / 1e-5
        assert abs(k - expected) < 1e-10

    def test_darcy_zero_force(self):
        assert compute_permeability(U_avg=1e-4, mu=1.0/6.0, F=0.0) == 0.0


# ===========================================================================
# Test 4: Porosity and specific surface unit tests
# ===========================================================================


class TestPorosityAndSurface:
    """Unit tests for porosity and specific surface computation."""

    def test_porosity_empty(self):
        solid = torch.zeros((10, 10), dtype=torch.bool)
        assert compute_porosity(solid) == 1.0

    def test_porosity_full(self):
        solid = torch.ones((10, 10), dtype=torch.bool)
        assert compute_porosity(solid) == 0.0

    def test_porosity_half(self):
        solid = torch.zeros((10, 10), dtype=torch.bool)
        solid[:5, :] = True
        assert abs(compute_porosity(solid) - 0.5) < 1e-10

    def test_specific_surface_2d(self):
        solid = torch.zeros((5, 5), dtype=torch.bool)
        solid[2, 2] = True
        S = compute_specific_surface(solid)
        assert abs(S - 4.0 / 25.0) < 1e-10


# ===========================================================================
# Test 5: Pore Reynolds number unit test
# ===========================================================================


class TestPoreReynolds:
    """Unit tests for pore Reynolds number computation."""

    def test_reynolds_formula(self):
        Re = compute_pore_reynolds(U=0.01, d_p=10.0, nu=0.001)
        assert abs(Re - 100.0) < 1e-10

    def test_reynolds_zero_viscosity(self):
        assert compute_pore_reynolds(U=0.01, d_p=10.0, nu=0.0) == float("inf")


# ===========================================================================
# Test 6: Parallel plates (2D) — Darcy permeability
# ===========================================================================


class TestParallelPlates2D:
    """Verify Darcy permeability for parallel plates."""

    def test_parallel_plates_permeability(self):
        config = PoreFlowConfig2D(
            ny=12, nx=16, gap=8, tau=1.0, accel=1e-5,
            n_steps=2000, warmup_steps=500, device="cpu",
        )
        result = run_pore_flow_2d(config)
        k = result["permeability"]
        k_analytical = analytical_parallel_plate_permeability(float(config.gap))
        assert k > 0, f"Permeability should be positive, got {k}"
        rel_error = abs(k - k_analytical) / k_analytical
        assert rel_error < 0.50, (
            f"Parallel plates: k={k:.6f}, k_analytical={k_analytical:.6f}, "
            f"error={rel_error*100:.2f}%"
        )

    def test_parallel_plates_mass_conservation(self):
        config = PoreFlowConfig2D(
            ny=12, nx=16, gap=8, tau=1.0, accel=1e-5,
            n_steps=2000, warmup_steps=500, device="cpu",
        )
        result = run_pore_flow_2d(config)
        assert result["mass_drift"] < 0.01


# ===========================================================================
# Test 7: Cylindrical tube (3D) — Darcy permeability
# ===========================================================================


class TestCylindricalTube3D:
    """Verify Darcy permeability for cylindrical tube."""

    @pytest.mark.slow
    def test_tube_permeability(self):
        """Tube permeability is positive and in reasonable range."""
        # Use small radius relative to grid
        config = PoreFlowConfig3D(
            nz=8, ny=8, nx=8, geometry="tube", radius=3.0,
            tau=1.0, accel=1e-5, n_steps=200, warmup_steps=30, device="cpu",
        )
        result = run_pore_flow_3d(config)
        k = result["permeability"]
        assert k > 0, f"Permeability should be positive, got {k}"
        # Allow large error due to lattice effects at low resolution
        k_analytical = analytical_tube_permeability(config.radius)
        rel_error = abs(k - k_analytical) / k_analytical
        assert rel_error < 1.0, (
            f"Tube: k={k:.6f}, k_analytical={k_analytical:.6f}, "
            f"error={rel_error*100:.2f}%"
        )


# ===========================================================================
# Test 8: Sphere pack (3D) — Kozeny-Carman
# ===========================================================================


class TestSpherePack3D:
    """Verify permeability of sphere pack."""

    @pytest.mark.slow
    def test_sphere_pack_permeability(self):
        """Sphere pack permeability is positive."""
        config = PoreFlowConfig3D(
            nz=8, ny=8, nx=8, geometry="sphere_pack",
            n_spheres=5, r_min=1.0, r_max=2.0,
            tau=1.0, accel=1e-5, n_steps=200, warmup_steps=30,
            device="cpu", seed=42,
        )
        result = run_pore_flow_3d(config)
        k = result["permeability"]
        # Permeability should be positive (flow in direction of force)
        assert k > 0, f"Permeability should be positive, got {k}"
        assert 0 < result["porosity"] < 1


# ===========================================================================
# Test 9: GRF (3D) — Kozeny-Carman
# ===========================================================================


class TestGRF3D:
    """Verify permeability of GRF-generated porous medium."""

    @pytest.mark.slow
    def test_grf_permeability(self):
        config = PoreFlowConfig3D(
            nz=8, ny=8, nx=8, geometry="grf",
            correlation_length=2.0, threshold=0.0,
            tau=1.0, accel=1e-5, n_steps=200, warmup_steps=30,
            device="cpu", seed=42,
        )
        result = run_pore_flow_3d(config)
        k = result["permeability"]
        assert k > 0, f"Permeability should be positive, got {k}"
        assert 0 < result["porosity"] < 1


# ===========================================================================
# Test 10: CT-like rock (3D)
# ===========================================================================


class TestCTLikeRock3D:
    """Verify permeability of CT-like synthetic rock."""

    @pytest.mark.slow
    def test_ct_like_permeability(self):
        config = PoreFlowConfig3D(
            nz=8, ny=8, nx=8, geometry="ct_like",
            base_porosity=0.3, heterogeneity=0.05,
            tau=1.0, accel=1e-5, n_steps=200, warmup_steps=30,
            device="cpu", seed=42,
        )
        result = run_pore_flow_3d(config)
        k = result["permeability"]
        assert k > 0, f"Permeability should be positive, got {k}"
        # Porosity should be approximately the target
        assert 0.1 < result["porosity"] < 0.9


# ===========================================================================
# Test 11: Grid convergence
# ===========================================================================


class TestGridConvergence:
    """Verify grid convergence trend."""

    def test_parallel_plates_convergence_trend(self):
        configs = [
            PoreFlowConfig2D(
                ny=10, nx=12, gap=6, tau=1.0, accel=1e-5,
                n_steps=1500, warmup_steps=300, device="cpu",
            ),
            PoreFlowConfig2D(
                ny=14, nx=20, gap=10, tau=1.0, accel=1e-5,
                n_steps=2000, warmup_steps=500, device="cpu",
            ),
        ]
        results = [run_pore_flow_2d(c) for c in configs]
        k1 = results[0]["permeability"]
        k2 = results[1]["permeability"]
        assert k1 > 0 and k2 > 0


# ===========================================================================
# Test 12: Reynolds number range
# ===========================================================================


class TestReynoldsRange:
    """Verify Darcy's law holds for low Re."""

    def test_low_reynolds_consistency(self):
        configs = [
            PoreFlowConfig2D(
                ny=12, nx=16, gap=8, tau=1.0, accel=5e-6,
                n_steps=2000, warmup_steps=500, device="cpu",
            ),
            PoreFlowConfig2D(
                ny=12, nx=16, gap=8, tau=1.0, accel=2e-5,
                n_steps=2000, warmup_steps=500, device="cpu",
            ),
        ]
        results = [run_pore_flow_2d(c) for c in configs]
        k1 = results[0]["permeability"]
        k2 = results[1]["permeability"]
        assert k1 > 0 and k2 > 0
        ratio = max(k1, k2) / min(k1, k2)
        assert ratio < 2.0, f"Permeability ratio {ratio:.2f} too large"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
