"""Tests for the Immersed Boundary Method (IBM) module."""
from __future__ import annotations

import pytest
import torch

from tensorlbm import equilibrium
from tensorlbm.ibm import (
    ibm_apply_body_force_2d,
    ibm_delta_4pt,
    ibm_delta_hat,
    ibm_direct_forcing,
    ibm_force_spread,
    ibm_velocity_interpolate,
)

# ---------------------------------------------------------------------------
# Delta kernel tests
# ---------------------------------------------------------------------------


class TestDeltaHat:
    def test_zero_at_boundary(self) -> None:
        r = torch.tensor([-1.0, 1.0])
        phi = ibm_delta_hat(r)
        assert torch.allclose(phi, torch.zeros(2), atol=1e-6)

    def test_one_at_zero(self) -> None:
        r = torch.tensor([0.0])
        assert float(ibm_delta_hat(r).item()) == pytest.approx(1.0)

    def test_non_negative(self) -> None:
        r = torch.linspace(-2.0, 2.0, 100)
        assert (ibm_delta_hat(r) >= 0).all()

    def test_zero_outside_support(self) -> None:
        r = torch.tensor([-1.5, 1.5, 2.0, -2.0])
        phi = ibm_delta_hat(r)
        assert torch.allclose(phi, torch.zeros(4), atol=1e-6)


class TestDelta4pt:
    def test_zero_outside_support(self) -> None:
        r = torch.tensor([-2.5, 2.5, 3.0])
        phi = ibm_delta_4pt(r)
        assert torch.allclose(phi, torch.zeros(3), atol=1e-6)

    def test_non_negative(self) -> None:
        r = torch.linspace(-3.0, 3.0, 200)
        assert (ibm_delta_4pt(r) >= -1e-6).all()

    def test_finite(self) -> None:
        r = torch.linspace(-3.0, 3.0, 200)
        assert torch.isfinite(ibm_delta_4pt(r)).all()


# ---------------------------------------------------------------------------
# Velocity interpolation
# ---------------------------------------------------------------------------


class TestIBMVelocityInterpolate:
    def test_uniform_flow_returns_uniform(self) -> None:
        """Interpolating into a uniform flow should return that velocity."""
        ny, nx = 16, 16
        ux = torch.full((ny, nx), 0.05)
        uy = torch.full((ny, nx), 0.02)
        # Single marker at a nice grid-aligned position
        marker_x = torch.tensor([7.5])
        marker_y = torch.tensor([7.5])
        u_mx, u_my = ibm_velocity_interpolate(ux, uy, marker_x, marker_y, kernel="hat")
        assert float(u_mx[0].item()) == pytest.approx(0.05, abs=1e-5)
        assert float(u_my[0].item()) == pytest.approx(0.02, abs=1e-5)

    def test_zero_flow_returns_zero(self) -> None:
        ny, nx = 16, 16
        ux = torch.zeros((ny, nx))
        uy = torch.zeros((ny, nx))
        marker_x = torch.tensor([5.0, 10.0])
        marker_y = torch.tensor([5.0, 8.0])
        u_mx, u_my = ibm_velocity_interpolate(ux, uy, marker_x, marker_y)
        assert torch.allclose(u_mx, torch.zeros(2), atol=1e-7)
        assert torch.allclose(u_my, torch.zeros(2), atol=1e-7)

    def test_4pt_kernel_uniform_flow(self) -> None:
        ny, nx = 20, 20
        ux = torch.full((ny, nx), 0.04)
        uy = torch.zeros((ny, nx))
        marker_x = torch.tensor([9.5])
        marker_y = torch.tensor([9.5])
        u_mx, u_my = ibm_velocity_interpolate(ux, uy, marker_x, marker_y, kernel="4pt")
        assert float(u_mx[0].item()) == pytest.approx(0.04, abs=1e-4)

    def test_output_shape(self) -> None:
        ny, nx = 16, 16
        ux = torch.rand((ny, nx))
        uy = torch.rand((ny, nx))
        n = 5
        marker_x = torch.rand(n) * nx
        marker_y = torch.rand(n) * ny
        u_mx, u_my = ibm_velocity_interpolate(ux, uy, marker_x, marker_y)
        assert u_mx.shape == (n,)
        assert u_my.shape == (n,)


# ---------------------------------------------------------------------------
# Force spreading
# ---------------------------------------------------------------------------


class TestIBMForceSpread:
    def test_force_conservation(self) -> None:
        """Total Eulerian force should equal total Lagrangian force."""
        ny, nx = 16, 16
        marker_fx = torch.tensor([0.01, -0.005, 0.008])
        marker_fy = torch.tensor([0.0, 0.003, -0.002])
        marker_x = torch.tensor([4.0, 8.0, 12.0])
        marker_y = torch.tensor([4.0, 8.0, 12.0])
        fx_grid, fy_grid = ibm_force_spread(
            marker_fx, marker_fy, marker_x, marker_y, ny, nx, kernel="hat"
        )
        assert float(fx_grid.sum().item()) == pytest.approx(
            float(marker_fx.sum().item()), abs=1e-5
        )
        assert float(fy_grid.sum().item()) == pytest.approx(
            float(marker_fy.sum().item()), abs=1e-5
        )

    def test_output_shape(self) -> None:
        ny, nx = 16, 16
        fx_grid, fy_grid = ibm_force_spread(
            torch.tensor([0.01]),
            torch.tensor([0.0]),
            torch.tensor([8.0]),
            torch.tensor([8.0]),
            ny, nx,
        )
        assert fx_grid.shape == (ny, nx)
        assert fy_grid.shape == (ny, nx)

    def test_zero_force_gives_zero_grid(self) -> None:
        ny, nx = 16, 16
        fx_grid, fy_grid = ibm_force_spread(
            torch.zeros(3),
            torch.zeros(3),
            torch.tensor([4.0, 8.0, 12.0]),
            torch.tensor([4.0, 8.0, 12.0]),
            ny, nx,
        )
        assert torch.allclose(fx_grid, torch.zeros((ny, nx)), atol=1e-7)
        assert torch.allclose(fy_grid, torch.zeros((ny, nx)), atol=1e-7)


# ---------------------------------------------------------------------------
# Direct forcing
# ---------------------------------------------------------------------------


class TestIBMDirectForcing:
    def test_stationary_target_in_zero_flow(self) -> None:
        """Markers in zero flow with zero target should produce zero force."""
        ny, nx = 16, 16
        ux = torch.zeros((ny, nx))
        uy = torch.zeros((ny, nx))
        marker_x = torch.tensor([8.0])
        marker_y = torch.tensor([8.0])
        u_target_x = torch.zeros(1)
        u_target_y = torch.zeros(1)
        fx, fy = ibm_direct_forcing(ux, uy, marker_x, marker_y, u_target_x, u_target_y)
        assert torch.allclose(fx, torch.zeros((ny, nx)), atol=1e-7)
        assert torch.allclose(fy, torch.zeros((ny, nx)), atol=1e-7)

    def test_output_shape(self) -> None:
        ny, nx = 20, 20
        ux = torch.rand((ny, nx)) * 0.05
        uy = torch.zeros((ny, nx))
        n = 4
        marker_x = torch.rand(n) * nx
        marker_y = torch.rand(n) * ny
        u_target_x = torch.zeros(n)
        u_target_y = torch.zeros(n)
        fx, fy = ibm_direct_forcing(ux, uy, marker_x, marker_y, u_target_x, u_target_y)
        assert fx.shape == (ny, nx)
        assert fy.shape == (ny, nx)

    def test_finite_output(self) -> None:
        ny, nx = 16, 16
        ux = torch.rand((ny, nx)) * 0.05
        uy = torch.rand((ny, nx)) * 0.03
        marker_x = torch.tensor([4.0, 8.0, 12.0])
        marker_y = torch.tensor([4.0, 8.0, 12.0])
        u_target_x = torch.zeros(3)
        u_target_y = torch.zeros(3)
        fx, fy = ibm_direct_forcing(ux, uy, marker_x, marker_y, u_target_x, u_target_y)
        assert torch.isfinite(fx).all()
        assert torch.isfinite(fy).all()


# ---------------------------------------------------------------------------
# Body-force application
# ---------------------------------------------------------------------------


class TestIBMApplyBodyForce2D:
    def test_preserves_shape(self) -> None:
        ny, nx = 8, 12
        rho = torch.ones((ny, nx))
        ux = torch.full_like(rho, 0.05)
        uy = torch.zeros_like(rho)
        f = equilibrium(rho, ux, uy)
        fx = torch.zeros((ny, nx))
        fy = torch.zeros((ny, nx))
        f_out = ibm_apply_body_force_2d(f, fx, fy)
        assert f_out.shape == f.shape

    def test_zero_force_is_identity(self) -> None:
        ny, nx = 8, 12
        rho = torch.ones((ny, nx))
        ux = torch.full_like(rho, 0.05)
        uy = torch.zeros_like(rho)
        f = equilibrium(rho, ux, uy)
        fx = torch.zeros((ny, nx))
        fy = torch.zeros((ny, nx))
        f_out = ibm_apply_body_force_2d(f, fx, fy)
        assert torch.allclose(f_out, f, atol=1e-6)

    def test_finite_output(self) -> None:
        ny, nx = 8, 12
        rho = torch.ones((ny, nx))
        ux = torch.full_like(rho, 0.04)
        uy = torch.zeros_like(rho)
        f = equilibrium(rho, ux, uy)
        fx = torch.full((ny, nx), 1e-4)
        fy = torch.zeros((ny, nx))
        f_out = ibm_apply_body_force_2d(f, fx, fy)
        assert torch.isfinite(f_out).all()
