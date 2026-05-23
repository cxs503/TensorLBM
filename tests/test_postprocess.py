"""Tests for postprocess.py: extract_velocity_profile, compute_pressure_coefficient, q_criterion."""
from __future__ import annotations

import pytest
import torch

from tensorlbm import (
    compute_pressure_coefficient,
    compute_q_criterion,
    extract_velocity_profile,
)


class TestExtractVelocityProfile:
    def test_x_axis_shape(self) -> None:
        ux = torch.rand((10, 12))
        uy = torch.rand((10, 12))
        ux_p, uy_p = extract_velocity_profile(ux, uy, axis="x", index=3)
        assert ux_p.shape == (10,)
        assert uy_p.shape == (10,)

    def test_y_axis_shape(self) -> None:
        ux = torch.rand((10, 12))
        uy = torch.rand((10, 12))
        ux_p, uy_p = extract_velocity_profile(ux, uy, axis="y", index=5)
        assert ux_p.shape == (12,)
        assert uy_p.shape == (12,)

    def test_x_axis_values(self) -> None:
        ux = torch.arange(120, dtype=torch.float32).reshape(10, 12)
        uy = torch.zeros(10, 12)
        ux_p, uy_p = extract_velocity_profile(ux, uy, axis="x", index=4)
        expected = ux[:, 4]
        assert torch.equal(ux_p, expected)

    def test_y_axis_values(self) -> None:
        ux = torch.arange(120, dtype=torch.float32).reshape(10, 12)
        uy = torch.zeros(10, 12)
        ux_p, uy_p = extract_velocity_profile(ux, uy, axis="y", index=2)
        expected = ux[2, :]
        assert torch.equal(ux_p, expected)

    def test_invalid_axis_raises(self) -> None:
        ux = torch.rand((10, 12))
        uy = torch.rand((10, 12))
        with pytest.raises(ValueError, match="axis"):
            extract_velocity_profile(ux, uy, axis="z", index=0)

    def test_default_axis_is_x(self) -> None:
        ux = torch.rand((10, 12))
        uy = torch.rand((10, 12))
        ux_p, _ = extract_velocity_profile(ux, uy, index=0)
        assert ux_p.shape == (10,)


class TestComputePressureCoefficient:
    def test_output_shape_matches_input(self) -> None:
        rho = torch.ones((8, 10))
        cp = compute_pressure_coefficient(rho, u_in=0.1)
        assert cp.shape == rho.shape

    def test_uniform_rho_gives_zero_cp(self) -> None:
        rho = torch.ones((8, 10))
        cp = compute_pressure_coefficient(rho, u_in=0.1, rho_ref=1.0)
        assert torch.allclose(cp, torch.zeros_like(cp), atol=1e-6)

    def test_nonzero_rho_gives_nonzero_cp(self) -> None:
        rho = torch.full((8, 10), 1.05)
        cp = compute_pressure_coefficient(rho, u_in=0.1, rho_ref=1.0)
        assert (cp > 0).all()

    def test_zero_u_in_returns_zeros(self) -> None:
        rho = torch.rand((8, 10)) + 0.5
        cp = compute_pressure_coefficient(rho, u_in=0.0)
        assert torch.allclose(cp, torch.zeros_like(cp))

    def test_3d_input(self) -> None:
        rho = torch.ones((4, 6, 8))
        cp = compute_pressure_coefficient(rho, u_in=0.1)
        assert cp.shape == (4, 6, 8)

    def test_formula_correctness(self) -> None:
        rho = torch.tensor([[1.1]])
        u_in = 0.2
        rho_ref = 1.0
        cs2 = 1.0 / 3.0
        expected = cs2 * (1.1 - 1.0) / (0.5 * rho_ref * u_in**2)
        cp = compute_pressure_coefficient(rho, u_in=u_in, rho_ref=rho_ref, cs2=cs2)
        assert cp.item() == pytest.approx(float(expected), rel=1e-5)


class TestComputeQCriterion:
    def test_output_shape(self) -> None:
        nz, ny, nx = 6, 8, 10
        ux = torch.rand((nz, ny, nx))
        uy = torch.rand((nz, ny, nx))
        uz = torch.rand((nz, ny, nx))
        q = compute_q_criterion(ux, uy, uz)
        assert q.shape == (nz, ny, nx)

    def test_finite_values(self) -> None:
        nz, ny, nx = 6, 8, 10
        ux = torch.rand((nz, ny, nx))
        uy = torch.rand((nz, ny, nx))
        uz = torch.rand((nz, ny, nx))
        q = compute_q_criterion(ux, uy, uz)
        assert torch.isfinite(q).all()

    def test_uniform_flow_has_zero_q(self) -> None:
        """Uniform flow has no velocity gradients, so Q = 0 everywhere."""
        nz, ny, nx = 6, 8, 10
        ux = torch.full((nz, ny, nx), 0.05)
        uy = torch.zeros(nz, ny, nx)
        uz = torch.zeros(nz, ny, nx)
        q = compute_q_criterion(ux, uy, uz)
        assert torch.allclose(q, torch.zeros_like(q), atol=1e-6)

    def test_pure_strain_gives_negative_q(self) -> None:
        """Pure extensional flow: dudx > 0, dvdy < 0, Q < 0 in interior."""
        nz, ny, nx = 6, 10, 12
        xx = torch.arange(nx, dtype=torch.float32).view(1, 1, nx).expand(nz, ny, nx)
        yy = torch.arange(ny, dtype=torch.float32).view(1, ny, 1).expand(nz, ny, nx)
        ux = 0.1 * xx
        uy = -0.1 * yy
        uz = torch.zeros_like(ux)
        q = compute_q_criterion(ux, uy, uz)
        # Interior cells should have Q ≤ 0 (strain dominant)
        assert (q[1:-1, 1:-1, 1:-1] <= 1e-6).all()

    def test_pure_rotation_gives_positive_q(self) -> None:
        """Solid-body rotation has omega > S, so Q > 0 in interior."""
        nz, ny, nx = 6, 20, 20
        yy = torch.arange(ny, dtype=torch.float32).view(1, ny, 1).expand(nz, ny, nx)
        xx = torch.arange(nx, dtype=torch.float32).view(1, 1, nx).expand(nz, ny, nx)
        omega = 0.1
        ux = -omega * yy
        uy = omega * xx
        uz = torch.zeros_like(ux)
        q = compute_q_criterion(ux, uy, uz)
        assert (q[1:-1, 1:-1, 1:-1] > 0).all()
