"""TDD tests for far-field boundary mode in cross-validation runners.

Verifies that:
1. ``far_field_bc_27`` exists and produces correct D3Q27 far-field BC
2. ``SphereCrossValidationConfig`` defaults to ``boundary_mode="farfield"``
3. Both ``"farfield"`` and ``"channel"`` modes produce finite results
4. The result artifact records which boundary mode was used
5. Channel mode is still available (backward compatibility)
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.boundaries import far_field_bc_2d
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.boundaries_d3q27 import far_field_bc_27
from tensorlbm.d2q9 import equilibrium
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.d3q27 import equilibrium27
from tensorlbm.cylinder_cross_validation import run_single_combination as run_cylinder
from tensorlbm.sphere_cross_validation import (
    SphereCrossValidationConfig,
    _run_single_combination as run_sphere_single,
)


# ---------------------------------------------------------------------------
# far_field_bc_27 unit tests
# ---------------------------------------------------------------------------

class TestFarFieldBC27:
    """Verify the D3Q27 far-field boundary condition."""

    def test_far_field_bc_27_exists_and_callable(self) -> None:
        """far_field_bc_27 must be importable and callable."""
        assert callable(far_field_bc_27)

    def test_far_field_bc_27_preserves_shape(self) -> None:
        """Output shape must match input shape (27, nz, ny, nx)."""
        nz, ny, nx = 8, 8, 12
        rho = torch.ones(nz, ny, nx)
        ux = torch.full_like(rho, 0.05)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)
        f = equilibrium27(rho, ux, uy, uz)
        f_out = far_field_bc_27(f, u_in=0.05)
        assert f_out.shape == f.shape == (27, nz, ny, nx)

    def test_far_field_bc_27_sets_inlet_to_equilibrium(self) -> None:
        """Inlet face (x=0) must be free-stream equilibrium."""
        nz, ny, nx = 6, 6, 10
        u_in = 0.05
        rho = torch.ones(nz, ny, nx)
        f = equilibrium27(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        f_out = far_field_bc_27(f, u_in=u_in)
        rho1 = torch.ones((nz, ny, nx))
        feq_expected = equilibrium27(
            rho1, torch.full_like(rho1, u_in), torch.zeros_like(rho1), torch.zeros_like(rho1)
        )
        assert torch.allclose(f_out[:, :, :, 0], feq_expected[:, :, :, 0])

    def test_far_field_bc_27_sets_lateral_faces(self) -> None:
        """All four lateral faces must be free-stream equilibrium."""
        nz, ny, nx = 6, 6, 10
        u_in = 0.05
        rho = torch.ones(nz, ny, nx)
        f = equilibrium27(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        f_out = far_field_bc_27(f, u_in=u_in)
        rho1 = torch.ones((nz, ny, nx))
        feq_expected = equilibrium27(
            rho1, torch.full_like(rho1, u_in), torch.zeros_like(rho1), torch.zeros_like(rho1)
        )
        # y- and y+
        assert torch.allclose(f_out[:, 0, :, :], feq_expected[:, 0, :, :])
        assert torch.allclose(f_out[:, -1, :, :], feq_expected[:, -1, :, :])
        # z- and z+
        assert torch.allclose(f_out[:, :, 0, :], feq_expected[:, :, 0, :])
        assert torch.allclose(f_out[:, :, -1, :], feq_expected[:, :, -1, :])

    def test_far_field_bc_27_outlet_zero_gradient(self) -> None:
        """Outlet (x=nx-1) must equal x=nx-2 (zero gradient)."""
        nz, ny, nx = 6, 6, 10
        u_in = 0.05
        rho = torch.ones(nz, ny, nx)
        f = equilibrium27(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        f_out = far_field_bc_27(f, u_in=u_in)
        assert torch.allclose(f_out[:, :, :, -1], f_out[:, :, :, -2])

    def test_far_field_bc_27_finite_output(self) -> None:
        """Output must be finite."""
        nz, ny, nx = 6, 6, 10
        rho = torch.ones(nz, ny, nx)
        f = equilibrium27(rho, torch.full_like(rho, 0.05), torch.zeros_like(rho), torch.zeros_like(rho))
        f_out = far_field_bc_27(f, u_in=0.05)
        assert torch.isfinite(f_out).all().item()


# ---------------------------------------------------------------------------
# far_field_bc_2d unit tests
# ---------------------------------------------------------------------------

class TestFarFieldBC2d:
    """Verify the D2Q9 far-field boundary condition."""

    def test_far_field_bc_2d_preserves_shape(self) -> None:
        ny, nx = 10, 20
        rho = torch.ones(ny, nx)
        f = equilibrium(rho, torch.full_like(rho, 0.05), torch.zeros_like(rho))
        f_out = far_field_bc_2d(f, u_in=0.05)
        assert f_out.shape == f.shape == (9, ny, nx)

    def test_far_field_bc_2d_sets_inlet(self) -> None:
        ny, nx = 10, 20
        u_in = 0.05
        rho = torch.ones(ny, nx)
        f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
        f_out = far_field_bc_2d(f, u_in=u_in)
        feq_expected = equilibrium(torch.ones(ny, nx), torch.full_like(rho, u_in), torch.zeros_like(rho))
        assert torch.allclose(f_out[:, :, 0], feq_expected[:, :, 0])


# ---------------------------------------------------------------------------
# far_field_bc_3d unit tests
# ---------------------------------------------------------------------------

class TestFarFieldBC3d:
    """Verify the D3Q19 far-field boundary condition."""

    def test_far_field_bc_3d_preserves_shape(self) -> None:
        nz, ny, nx = 8, 8, 12
        rho = torch.ones(nz, ny, nx)
        f = equilibrium3d(rho, torch.full_like(rho, 0.05), torch.zeros_like(rho), torch.zeros_like(rho))
        f_out = far_field_bc_3d(f, u_in=0.05)
        assert f_out.shape == f.shape == (19, nz, ny, nx)


# ---------------------------------------------------------------------------
# Sphere cross-validation boundary_mode tests
# ---------------------------------------------------------------------------

class TestSphereBoundaryMode:
    """Verify boundary_mode parameter in sphere cross-validation."""

    def test_config_defaults_to_farfield(self) -> None:
        config = SphereCrossValidationConfig()
        assert config.boundary_mode == "farfield"

    def test_farfield_mode_produces_finite_result(self) -> None:
        config = SphereCrossValidationConfig(
            nx=12, ny=12, nz=12, steps=5, boundary_mode="farfield"
        )
        result = run_sphere_single(config, "D3Q19", "BGK", "none")
        assert result.finite
        assert result.Cd is not None
        assert math.isfinite(result.Cd)
        assert result.boundary_mode == "farfield"

    def test_channel_mode_produces_finite_result(self) -> None:
        config = SphereCrossValidationConfig(
            nx=12, ny=12, nz=12, steps=5, boundary_mode="channel"
        )
        result = run_sphere_single(config, "D3Q19", "BGK", "none")
        assert result.finite
        assert result.Cd is not None
        assert math.isfinite(result.Cd)
        assert result.boundary_mode == "channel"

    def test_farfield_d3q27_produces_finite_result(self) -> None:
        config = SphereCrossValidationConfig(
            nx=12, ny=12, nz=12, steps=5, boundary_mode="farfield"
        )
        result = run_sphere_single(config, "D3Q27", "BGK", "none")
        assert result.finite
        assert result.Cd is not None
        assert math.isfinite(result.Cd)
        assert result.boundary_mode == "farfield"

    def test_channel_d3q27_produces_finite_result(self) -> None:
        config = SphereCrossValidationConfig(
            nx=12, ny=12, nz=12, steps=5, boundary_mode="channel"
        )
        result = run_sphere_single(config, "D3Q27", "BGK", "none")
        assert result.finite
        assert result.Cd is not None
        assert math.isfinite(result.Cd)
        assert result.boundary_mode == "channel"

    def test_farfield_and_channel_produce_different_cd(self) -> None:
        """Far-field and channel should produce different Cd (different physics)."""
        config_ff = SphereCrossValidationConfig(
            nx=20, ny=20, nz=20, steps=30, boundary_mode="farfield"
        )
        config_ch = SphereCrossValidationConfig(
            nx=20, ny=20, nz=20, steps=30, boundary_mode="channel"
        )
        result_ff = run_sphere_single(config_ff, "D3Q19", "BGK", "none")
        result_ch = run_sphere_single(config_ch, "D3Q19", "BGK", "none")
        assert result_ff.Cd != result_ch.Cd


# ---------------------------------------------------------------------------
# Cylinder cross-validation boundary_mode tests
# ---------------------------------------------------------------------------

class TestCylinderBoundaryMode:
    """Verify boundary_mode parameter in cylinder cross-validation."""

    def test_farfield_mode_produces_finite_result(self) -> None:
        result = run_cylinder(
            "BGK", "none", re=100, nx=50, ny=30, steps=30, boundary_mode="farfield"
        )
        assert result["finite"] is True
        assert math.isfinite(result["Cd"])
        assert result["boundary_mode"] == "farfield"

    def test_channel_mode_produces_finite_result(self) -> None:
        result = run_cylinder(
            "BGK", "none", re=100, nx=50, ny=30, steps=30, boundary_mode="channel"
        )
        assert result["finite"] is True
        assert math.isfinite(result["Cd"])
        assert result["boundary_mode"] == "channel"

    def test_farfield_and_channel_produce_different_cd(self) -> None:
        """Far-field and channel should produce different Cd."""
        result_ff = run_cylinder(
            "BGK", "none", re=100, nx=50, ny=30, steps=30, boundary_mode="farfield"
        )
        result_ch = run_cylinder(
            "BGK", "none", re=100, nx=50, ny=30, steps=30, boundary_mode="channel"
        )
        assert result_ff["Cd"] != result_ch["Cd"]
