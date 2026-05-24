"""Tests for postprocess.py: extract_velocity_profile, compute_pressure_coefficient, q_criterion."""
from __future__ import annotations

import pytest
import torch

from tensorlbm import (
    RunningStats,
    compute_divergence,
    compute_drag_lift_coefficients,
    compute_enstrophy_2d,
    compute_kinetic_energy,
    compute_lambda2_criterion,
    compute_pressure_coefficient,
    compute_q_criterion,
    compute_velocity_magnitude,
    compute_vorticity_2d,
    compute_vorticity_3d,
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


class TestComputeLambda2Criterion:
    def test_output_shape(self) -> None:
        nz, ny, nx = 6, 8, 10
        ux = torch.rand(nz, ny, nx)
        uy = torch.rand(nz, ny, nx)
        uz = torch.rand(nz, ny, nx)
        l2 = compute_lambda2_criterion(ux, uy, uz)
        assert l2.shape == (nz, ny, nx)

    def test_finite_values(self) -> None:
        nz, ny, nx = 6, 8, 10
        ux = torch.rand(nz, ny, nx)
        uy = torch.rand(nz, ny, nx)
        uz = torch.rand(nz, ny, nx)
        l2 = compute_lambda2_criterion(ux, uy, uz)
        assert torch.isfinite(l2).all()

    def test_uniform_flow_zero(self) -> None:
        """Uniform flow: no gradients → S = Ω = 0 → λ₂ = 0."""
        nz, ny, nx = 6, 8, 10
        ux = torch.full((nz, ny, nx), 0.05)
        uy = torch.zeros(nz, ny, nx)
        uz = torch.zeros(nz, ny, nx)
        l2 = compute_lambda2_criterion(ux, uy, uz)
        assert torch.allclose(l2, torch.zeros_like(l2), atol=1e-5)

    def test_solid_body_rotation_negative_lambda2(self) -> None:
        """Solid-body rotation should give λ₂ < 0 in interior."""
        nz, ny, nx = 6, 20, 20
        yy = torch.arange(ny, dtype=torch.float32).view(1, ny, 1).expand(nz, ny, nx)
        xx = torch.arange(nx, dtype=torch.float32).view(1, 1, nx).expand(nz, ny, nx)
        omega = 0.1
        ux = -omega * yy
        uy = omega * xx
        uz = torch.zeros_like(ux)
        l2 = compute_lambda2_criterion(ux, uy, uz)
        assert (l2[1:-1, 1:-1, 1:-1] < 0).all()


class TestComputeVorticity2d:
    def test_output_shape(self) -> None:
        ux = torch.rand(10, 12)
        uy = torch.rand(10, 12)
        omega = compute_vorticity_2d(ux, uy)
        assert omega.shape == (10, 12)

    def test_uniform_flow_zero(self) -> None:
        ux = torch.full((10, 12), 0.05)
        uy = torch.zeros(10, 12)
        omega = compute_vorticity_2d(ux, uy)
        assert torch.allclose(omega, torch.zeros_like(omega), atol=1e-6)

    def test_solid_body_rotation(self) -> None:
        """Solid-body rotation ux=-ωy, uy=ωx gives ωz = 2ω in interior."""
        ny, nx = 20, 20
        yy = torch.arange(ny, dtype=torch.float32).view(ny, 1).expand(ny, nx)
        xx = torch.arange(nx, dtype=torch.float32).view(1, nx).expand(ny, nx)
        omega_val = 0.1
        ux = -omega_val * yy
        uy = omega_val * xx
        omega = compute_vorticity_2d(ux, uy)
        # interior cells: ωz = ∂uy/∂x - ∂ux/∂y = ω - (-ω) = 2ω
        assert torch.allclose(omega[1:-1, 1:-1], torch.full((18, 18), 2 * omega_val), atol=1e-5)

    def test_3d_input_raises(self) -> None:
        ux = torch.rand(4, 6, 8)
        uy = torch.rand(4, 6, 8)
        with pytest.raises(ValueError):
            compute_vorticity_2d(ux, uy)


class TestComputeVorticity3d:
    def test_output_shapes(self) -> None:
        nz, ny, nx = 6, 8, 10
        ux = torch.rand(nz, ny, nx)
        uy = torch.rand(nz, ny, nx)
        uz = torch.rand(nz, ny, nx)
        ox, oy, oz = compute_vorticity_3d(ux, uy, uz)
        assert ox.shape == (nz, ny, nx)
        assert oy.shape == (nz, ny, nx)
        assert oz.shape == (nz, ny, nx)

    def test_uniform_flow_zero(self) -> None:
        nz, ny, nx = 6, 8, 10
        ux = torch.full((nz, ny, nx), 0.05)
        uy = torch.zeros(nz, ny, nx)
        uz = torch.zeros(nz, ny, nx)
        ox, oy, oz = compute_vorticity_3d(ux, uy, uz)
        assert torch.allclose(ox, torch.zeros_like(ox), atol=1e-6)
        assert torch.allclose(oy, torch.zeros_like(oy), atol=1e-6)
        assert torch.allclose(oz, torch.zeros_like(oz), atol=1e-6)


class TestComputeVelocityMagnitude:
    def test_2d_shape(self) -> None:
        ux = torch.rand(8, 10)
        uy = torch.rand(8, 10)
        mag = compute_velocity_magnitude(ux, uy)
        assert mag.shape == (8, 10)

    def test_3d_shape(self) -> None:
        ux = torch.rand(4, 6, 8)
        uy = torch.rand(4, 6, 8)
        uz = torch.rand(4, 6, 8)
        mag = compute_velocity_magnitude(ux, uy, uz)
        assert mag.shape == (4, 6, 8)

    def test_nonnegative(self) -> None:
        ux = torch.randn(6, 8)
        uy = torch.randn(6, 8)
        mag = compute_velocity_magnitude(ux, uy)
        assert (mag >= 0).all()

    def test_known_values(self) -> None:
        ux = torch.tensor([[3.0]])
        uy = torch.tensor([[4.0]])
        mag = compute_velocity_magnitude(ux, uy)
        assert mag.item() == pytest.approx(5.0, rel=1e-5)


class TestComputeKineticEnergy:
    def test_2d_shape(self) -> None:
        ux = torch.rand(8, 10)
        uy = torch.rand(8, 10)
        ke = compute_kinetic_energy(ux, uy)
        assert ke.shape == (8, 10)

    def test_3d_shape(self) -> None:
        ux = torch.rand(4, 6, 8)
        uy = torch.rand(4, 6, 8)
        uz = torch.rand(4, 6, 8)
        ke = compute_kinetic_energy(ux, uy, uz)
        assert ke.shape == (4, 6, 8)

    def test_nonnegative(self) -> None:
        ux = torch.randn(6, 8)
        uy = torch.randn(6, 8)
        ke = compute_kinetic_energy(ux, uy)
        assert (ke >= 0).all()

    def test_known_value(self) -> None:
        ux = torch.tensor([[1.0]])
        uy = torch.tensor([[0.0]])
        ke = compute_kinetic_energy(ux, uy)
        assert ke.item() == pytest.approx(0.5, rel=1e-5)


class TestComputeEnstrophy2d:
    def test_shape(self) -> None:
        ux = torch.rand(8, 10)
        uy = torch.rand(8, 10)
        e = compute_enstrophy_2d(ux, uy)
        assert e.shape == (8, 10)

    def test_nonnegative(self) -> None:
        ux = torch.randn(8, 10)
        uy = torch.randn(8, 10)
        e = compute_enstrophy_2d(ux, uy)
        assert (e >= 0).all()

    def test_uniform_flow_zero(self) -> None:
        ux = torch.full((8, 10), 0.05)
        uy = torch.zeros(8, 10)
        e = compute_enstrophy_2d(ux, uy)
        assert torch.allclose(e, torch.zeros_like(e), atol=1e-6)


class TestComputeDivergence:
    def test_2d_shape(self) -> None:
        ux = torch.rand(8, 10)
        uy = torch.rand(8, 10)
        div = compute_divergence(ux, uy)
        assert div.shape == (8, 10)

    def test_3d_shape(self) -> None:
        ux = torch.rand(4, 6, 8)
        uy = torch.rand(4, 6, 8)
        uz = torch.rand(4, 6, 8)
        div = compute_divergence(ux, uy, uz)
        assert div.shape == (4, 6, 8)

    def test_2d_uz_raises(self) -> None:
        ux = torch.rand(8, 10)
        uy = torch.rand(8, 10)
        uz = torch.rand(8, 10)
        with pytest.raises(ValueError):
            compute_divergence(ux, uy, uz)

    def test_uniform_flow_zero(self) -> None:
        ux = torch.full((8, 10), 0.05)
        uy = torch.zeros(8, 10)
        div = compute_divergence(ux, uy)
        assert torch.allclose(div, torch.zeros_like(div), atol=1e-6)

    def test_invalid_ndim_raises(self) -> None:
        ux = torch.rand(4)
        uy = torch.rand(4)
        with pytest.raises(ValueError):
            compute_divergence(ux, uy)


class TestComputeDragLiftCoefficients:
    def test_known_values(self) -> None:
        # Cd = Fx / (0.5 * rho * U^2 * A)
        # = 1.0 / (0.5 * 1.0 * 1.0 * 1.0) = 2.0
        cd, cl = compute_drag_lift_coefficients(1.0, 0.5, u_in=1.0, rho_ref=1.0, area=1.0)
        assert cd == pytest.approx(2.0, rel=1e-5)
        assert cl == pytest.approx(1.0, rel=1e-5)

    def test_zero_u_in_returns_zeros(self) -> None:
        cd, cl = compute_drag_lift_coefficients(1.0, 1.0, u_in=0.0)
        assert cd == 0.0
        assert cl == 0.0

    def test_tensor_input(self) -> None:
        fx = torch.tensor(1.0)
        fy = torch.tensor(0.5)
        cd, cl = compute_drag_lift_coefficients(fx, fy, u_in=1.0, rho_ref=1.0, area=1.0)
        assert isinstance(cd, float)
        assert isinstance(cl, float)
        assert cd == pytest.approx(2.0, rel=1e-5)

    def test_symmetry(self) -> None:
        cd, cl = compute_drag_lift_coefficients(2.0, 0.0, u_in=2.0, rho_ref=1.0, area=1.0)
        assert cl == pytest.approx(0.0, abs=1e-10)


class TestRunningStats:
    def test_no_data_raises(self) -> None:
        stats = RunningStats()
        with pytest.raises(RuntimeError):
            _ = stats.mean

    def test_count(self) -> None:
        stats = RunningStats()
        for _ in range(5):
            stats.update(torch.rand(4, 4))
        assert stats.count == 5

    def test_single_sample_mean(self) -> None:
        stats = RunningStats()
        field = torch.ones(4, 4)
        stats.update(field)
        assert torch.allclose(stats.mean, field)

    def test_single_sample_variance_zero(self) -> None:
        stats = RunningStats()
        stats.update(torch.rand(4, 4))
        assert torch.allclose(stats.variance, torch.zeros(4, 4), atol=1e-6)

    def test_mean_converges(self) -> None:
        torch.manual_seed(0)
        stats = RunningStats()
        samples = []
        for _ in range(1000):
            f = torch.randn(4, 4)
            samples.append(f)
            stats.update(f)
        true_mean = torch.stack(samples).mean(0)
        assert torch.allclose(stats.mean, true_mean, atol=1e-4)

    def test_variance_converges(self) -> None:
        torch.manual_seed(0)
        stats = RunningStats()
        samples = []
        for _ in range(1000):
            f = torch.randn(4, 4)
            samples.append(f)
            stats.update(f)
        true_var = torch.stack(samples).var(0, unbiased=True)
        assert torch.allclose(stats.variance, true_var, atol=1e-3)

    def test_fluctuation(self) -> None:
        stats = RunningStats()
        for _ in range(10):
            stats.update(torch.ones(4, 4))
        fluct = stats.fluctuation(torch.full((4, 4), 1.5))
        assert torch.allclose(fluct, torch.full((4, 4), 0.5), atol=1e-5)

    def test_reset(self) -> None:
        stats = RunningStats()
        stats.update(torch.rand(4, 4))
        stats.reset()
        assert stats.count == 0
        with pytest.raises(RuntimeError):
            _ = stats.mean

