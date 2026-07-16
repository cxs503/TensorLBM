"""Tests for the full cascaded central-moment collision operators (D3Q19/D3Q27).

These tests verify the complete CM hierarchy: forward central-moment transform,
cascaded relaxation with trace/deviatoric split, inverse transform, conservation,
equilibrium fixed-point, and stability under streaming.
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.d3q27 import equilibrium27, macroscopic27


# ---------------------------------------------------------------------------
# Test states
# ---------------------------------------------------------------------------

def _d3q19_state(shape=(2, 3, 4), dtype=torch.float64, seed=42):
    torch.manual_seed(seed)
    rho = 0.9 + 0.2 * torch.rand(shape, dtype=dtype)
    ux = 0.04 * (2.0 * torch.rand(shape, dtype=dtype) - 1.0)
    uy = 0.04 * (2.0 * torch.rand(shape, dtype=dtype) - 1.0)
    uz = 0.04 * (2.0 * torch.rand(shape, dtype=dtype) - 1.0)
    feq = equilibrium3d(rho, ux, uy, uz)
    return feq + 1.0e-3 * torch.randn_like(feq)


def _d3q27_state(shape=(2, 3, 4), dtype=torch.float64, seed=43):
    torch.manual_seed(seed)
    rho = 0.9 + 0.2 * torch.rand(shape, dtype=dtype)
    ux = 0.04 * (2.0 * torch.rand(shape, dtype=dtype) - 1.0)
    uy = 0.04 * (2.0 * torch.rand(shape, dtype=dtype) - 1.0)
    uz = 0.04 * (2.0 * torch.rand(shape, dtype=dtype) - 1.0)
    feq = equilibrium27(rho, ux, uy, uz)
    return feq + 1.0e-3 * torch.randn_like(feq)


def _conserved_raw_moments(f, c_tensor, q):
    """Return m000, m100, m010, m001 directly from populations."""
    directions = c_tensor.to(device=f.device, dtype=f.dtype)
    flattened = f.reshape(q, -1)
    mass = flattened.sum(dim=0)
    momentum = directions.T @ flattened
    return torch.cat((mass.unsqueeze(0), momentum), dim=0)


# ---------------------------------------------------------------------------
# D3Q19: moment matrix and central-moment transform
# ---------------------------------------------------------------------------

class TestD3Q19MatrixAndTransform:
    def test_moment_matrix_is_invertible(self):
        from tensorlbm.cascaded_collision import _get_d3q19_matrices
        M, M_inv = _get_d3q19_matrices(torch.device("cpu"), torch.float64)
        eye = M @ M_inv
        torch.testing.assert_close(
            eye, torch.eye(19, dtype=torch.float64), rtol=1e-10, atol=1e-11,
        )

    def test_central_moment_round_trip_zero_velocity(self):
        """At u=0, central moments equal raw moments; round-trip is identity."""
        from tensorlbm.cascaded_collision import (
            _get_d3q19_matrices, _to_central_d3q19, _to_raw_d3q19,
        )
        f = _d3q19_state()
        nz, ny, nx = f.shape[1:]
        M, _ = _get_d3q19_matrices(f.device, f.dtype)
        m = (M @ f.reshape(19, -1)).reshape(19, nz, ny, nx)
        zero = torch.zeros_like(m[0])
        k = _to_central_d3q19(m, zero, zero, zero)
        torch.testing.assert_close(k, m, rtol=1e-12, atol=1e-13)
        m_back = _to_raw_d3q19(k, zero, zero, zero)
        torch.testing.assert_close(m_back, m, rtol=1e-12, atol=1e-13)

    def test_central_moment_round_trip_nonzero_velocity(self):
        """Shift then unshift must recover the original raw moments."""
        from tensorlbm.cascaded_collision import (
            _get_d3q19_matrices, _to_central_d3q19, _to_raw_d3q19,
        )
        f = _d3q19_state()
        rho, ux, uy, uz = macroscopic3d(f)
        feq = equilibrium3d(rho, ux, uy, uz)
        f_neq = f - feq
        nz, ny, nx = f.shape[1:]
        M, _ = _get_d3q19_matrices(f.device, f.dtype)
        m = (M @ f_neq.reshape(19, -1)).reshape(19, nz, ny, nx)
        k = _to_central_d3q19(m, ux, uy, uz)
        m_back = _to_raw_d3q19(k, ux, uy, uz)
        torch.testing.assert_close(m_back, m, rtol=1e-10, atol=1e-13)

    def test_second_order_central_moments_equal_raw_at_neq(self):
        """For f_neq (zero mass/momentum), 2nd-order central = raw moments."""
        from tensorlbm.cascaded_collision import (
            _get_d3q19_matrices, _to_central_d3q19,
        )
        f = _d3q19_state()
        rho, ux, uy, uz = macroscopic3d(f)
        feq = equilibrium3d(rho, ux, uy, uz)
        f_neq = f - feq
        nz, ny, nx = f.shape[1:]
        M, _ = _get_d3q19_matrices(f.device, f.dtype)
        m = (M @ f_neq.reshape(19, -1)).reshape(19, nz, ny, nx)
        k = _to_central_d3q19(m, ux, uy, uz)
        # 2nd order: indices 4-9 should be unchanged
        for i in range(4, 10):
            torch.testing.assert_close(k[i], m[i], rtol=1e-7, atol=1e-9)


# ---------------------------------------------------------------------------
# D3Q19: collision properties
# ---------------------------------------------------------------------------

class TestCascadedD3Q19:
    @pytest.mark.parametrize("tau", [0.55, 0.83, 1.7])
    def test_equilibrium_is_fixed_point(self, tau):
        from tensorlbm.cascaded_collision import collide_cascaded_d3q19
        torch.manual_seed(101)
        rho = 0.9 + 0.2 * torch.rand((2, 3, 4), dtype=torch.float64)
        ux = 0.05 * (2.0 * torch.rand((2, 3, 4), dtype=torch.float64) - 1.0)
        uy = 0.05 * (2.0 * torch.rand((2, 3, 4), dtype=torch.float64) - 1.0)
        uz = 0.05 * (2.0 * torch.rand((2, 3, 4), dtype=torch.float64) - 1.0)
        feq = equilibrium3d(rho, ux, uy, uz)
        out = collide_cascaded_d3q19(feq, tau=tau, s_bulk=1.13, s_3=0.71, s_4=1.37)
        torch.testing.assert_close(out, feq, rtol=2e-6, atol=2e-7)

    @pytest.mark.parametrize("tau", [0.55, 0.83, 1.7])
    def test_conservation(self, tau):
        from tensorlbm.cascaded_collision import collide_cascaded_d3q19
        from tensorlbm.d3q19 import C as C19
        f = _d3q19_state()
        before = _conserved_raw_moments(f, C19, 19)
        out = collide_cascaded_d3q19(f, tau=tau, s_bulk=1.13, s_3=0.71, s_4=1.37)
        after = _conserved_raw_moments(out, C19, 19)
        torch.testing.assert_close(after, before, rtol=2e-6, atol=2e-7)

    def test_non_equilibrium_decays(self):
        from tensorlbm.cascaded_collision import collide_cascaded_d3q19
        f = _d3q19_state()
        feq = equilibrium3d(*macroscopic3d(f))
        before = (f - feq).pow(2).sum().sqrt()
        out = collide_cascaded_d3q19(f, tau=0.83, s_bulk=1.0, s_3=1.0, s_4=1.0)
        after = (out - feq).pow(2).sum().sqrt()
        assert after < before

    def test_full_relaxation_gives_equilibrium(self):
        """With all rates = 1 (tau=1, s_bulk=s_3=s_4=1), result is feq."""
        from tensorlbm.cascaded_collision import collide_cascaded_d3q19
        f = _d3q19_state()
        feq = equilibrium3d(*macroscopic3d(f))
        out = collide_cascaded_d3q19(f, tau=1.0, s_bulk=1.0, s_3=1.0, s_4=1.0)
        torch.testing.assert_close(out, feq, rtol=2e-6, atol=2e-7)

    def test_shear_relaxation_rate_matches_tau(self):
        """The shear stress decays at exactly 1/tau."""
        from tensorlbm.cascaded_collision import collide_cascaded_d3q19
        from tensorlbm.d3q19 import C as C19
        f = _d3q19_state()
        rho, ux, uy, uz = macroscopic3d(f)
        feq = equilibrium3d(rho, ux, uy, uz)
        f_neq = f - feq

        # Compute raw 2nd-order stress (central = raw for f_neq)
        c = C19.to(device=f.device, dtype=f.dtype)
        cx = c[:, 0].view(19, 1, 1, 1)
        cy = c[:, 1].view(19, 1, 1, 1)
        cz = c[:, 2].view(19, 1, 1, 1)
        pi_xy_before = (cx * cy * f_neq).sum(0)

        tau = 0.83
        out = collide_cascaded_d3q19(f, tau=tau, s_bulk=1.0, s_3=1.0, s_4=1.0)
        f_neq_out = out - feq
        pi_xy_after = (cx * cy * f_neq_out).sum(0)

        expected = (1.0 - 1.0 / tau) * pi_xy_before
        torch.testing.assert_close(pi_xy_after, expected, rtol=1e-6, atol=1e-8)

    def test_bulk_relaxation_rate_independent(self):
        """The trace (bulk) mode relaxes at s_bulk, not 1/tau."""
        from tensorlbm.cascaded_collision import collide_cascaded_d3q19
        from tensorlbm.d3q19 import C as C19
        f = _d3q19_state()
        rho, ux, uy, uz = macroscopic3d(f)
        feq = equilibrium3d(rho, ux, uy, uz)
        f_neq = f - feq

        c = C19.to(device=f.device, dtype=f.dtype)
        cx = c[:, 0].view(19, 1, 1, 1)
        cy = c[:, 1].view(19, 1, 1, 1)
        cz = c[:, 2].view(19, 1, 1, 1)
        trace_before = (
            (cx * cx * f_neq).sum(0)
            + (cy * cy * f_neq).sum(0)
            + (cz * cz * f_neq).sum(0)
        )

        tau = 0.83
        s_bulk = 1.31
        out = collide_cascaded_d3q19(f, tau=tau, s_bulk=s_bulk, s_3=1.0, s_4=1.0)
        f_neq_out = out - feq
        trace_after = (
            (cx * cx * f_neq_out).sum(0)
            + (cy * cy * f_neq_out).sum(0)
            + (cz * cz * f_neq_out).sum(0)
        )
        expected = (1.0 - s_bulk) * trace_before
        torch.testing.assert_close(trace_after, expected, rtol=1e-6, atol=1e-8)

    def test_float32_consistency(self):
        """float32 populations should also conserve and fix equilibrium."""
        from tensorlbm.cascaded_collision import collide_cascaded_d3q19
        from tensorlbm.d3q19 import C as C19
        f = _d3q19_state(dtype=torch.float32)
        before = _conserved_raw_moments(f, C19, 19)
        out = collide_cascaded_d3q19(f, tau=0.83, s_bulk=1.13, s_3=0.71, s_4=1.37)
        after = _conserved_raw_moments(out, C19, 19)
        torch.testing.assert_close(after, before, rtol=5e-4, atol=5e-5)


# ---------------------------------------------------------------------------
# D3Q27: moment matrix and central-moment transform
# ---------------------------------------------------------------------------

class TestD3Q27MatrixAndTransform:
    def test_moment_matrix_is_invertible(self):
        from tensorlbm.cascaded_collision import _get_d3q27_matrices
        M, M_inv = _get_d3q27_matrices(torch.device("cpu"), torch.float64)
        eye = M @ M_inv
        torch.testing.assert_close(
            eye, torch.eye(27, dtype=torch.float64), rtol=1e-10, atol=1e-11,
        )

    def test_central_moment_round_trip_nonzero_velocity(self):
        from tensorlbm.cascaded_collision import (
            _get_d3q27_matrices, _to_central_d3q27, _to_raw_d3q27,
        )
        f = _d3q27_state()
        rho, ux, uy, uz = macroscopic27(f)
        feq = equilibrium27(rho, ux, uy, uz)
        f_neq = f - feq
        nz, ny, nx = f.shape[1:]
        M, _ = _get_d3q27_matrices(f.device, f.dtype)
        m = (M @ f_neq.reshape(27, -1)).reshape(27, nz, ny, nx)
        k = _to_central_d3q27(m, ux, uy, uz)
        m_back = _to_raw_d3q27(k, ux, uy, uz)
        torch.testing.assert_close(m_back, m, rtol=1e-10, atol=1e-13)

    def test_second_order_central_moments_equal_raw_at_neq(self):
        from tensorlbm.cascaded_collision import (
            _get_d3q27_matrices, _to_central_d3q27,
        )
        f = _d3q27_state()
        rho, ux, uy, uz = macroscopic27(f)
        feq = equilibrium27(rho, ux, uy, uz)
        f_neq = f - feq
        nz, ny, nx = f.shape[1:]
        M, _ = _get_d3q27_matrices(f.device, f.dtype)
        m = (M @ f_neq.reshape(27, -1)).reshape(27, nz, ny, nx)
        k = _to_central_d3q27(m, ux, uy, uz)
        for i in range(4, 10):
            torch.testing.assert_close(k[i], m[i], rtol=1e-7, atol=1e-9)


# ---------------------------------------------------------------------------
# D3Q27: collision properties
# ---------------------------------------------------------------------------

class TestCascadedD3Q27:
    @pytest.mark.parametrize("tau", [0.55, 0.83, 1.7])
    def test_equilibrium_is_fixed_point(self, tau):
        from tensorlbm.cascaded_collision import collide_cascaded_d3q27
        torch.manual_seed(201)
        rho = 0.9 + 0.2 * torch.rand((2, 3, 4), dtype=torch.float64)
        ux = 0.05 * (2.0 * torch.rand((2, 3, 4), dtype=torch.float64) - 1.0)
        uy = 0.05 * (2.0 * torch.rand((2, 3, 4), dtype=torch.float64) - 1.0)
        uz = 0.05 * (2.0 * torch.rand((2, 3, 4), dtype=torch.float64) - 1.0)
        feq = equilibrium27(rho, ux, uy, uz)
        out = collide_cascaded_d3q27(
            feq, tau=tau, s_bulk=1.13, s_3=0.71, s_4=1.37, s_5=0.9, s_6=1.1,
        )
        torch.testing.assert_close(out, feq, rtol=2e-6, atol=2e-7)

    @pytest.mark.parametrize("tau", [0.55, 0.83, 1.7])
    def test_conservation(self, tau):
        from tensorlbm.cascaded_collision import collide_cascaded_d3q27
        from tensorlbm.d3q27 import C as C27
        f = _d3q27_state()
        before = _conserved_raw_moments(f, C27, 27)
        out = collide_cascaded_d3q27(
            f, tau=tau, s_bulk=1.13, s_3=0.71, s_4=1.37, s_5=0.9, s_6=1.1,
        )
        after = _conserved_raw_moments(out, C27, 27)
        torch.testing.assert_close(after, before, rtol=2e-6, atol=2e-7)

    def test_non_equilibrium_decays(self):
        from tensorlbm.cascaded_collision import collide_cascaded_d3q27
        f = _d3q27_state()
        feq = equilibrium27(*macroscopic27(f))
        before = (f - feq).pow(2).sum().sqrt()
        out = collide_cascaded_d3q27(f, tau=0.83, s_bulk=1.0, s_3=1.0, s_4=1.0)
        after = (out - feq).pow(2).sum().sqrt()
        assert after < before

    def test_full_relaxation_gives_equilibrium(self):
        from tensorlbm.cascaded_collision import collide_cascaded_d3q27
        f = _d3q27_state()
        feq = equilibrium27(*macroscopic27(f))
        out = collide_cascaded_d3q27(f, tau=1.0, s_bulk=1.0, s_3=1.0, s_4=1.0)
        torch.testing.assert_close(out, feq, rtol=2e-6, atol=2e-7)

    def test_shear_relaxation_rate_matches_tau(self):
        from tensorlbm.cascaded_collision import collide_cascaded_d3q27
        from tensorlbm.d3q27 import C as C27
        f = _d3q27_state()
        rho, ux, uy, uz = macroscopic27(f)
        feq = equilibrium27(rho, ux, uy, uz)
        f_neq = f - feq

        c = C27.to(device=f.device, dtype=f.dtype)
        cx = c[:, 0].view(27, 1, 1, 1)
        cy = c[:, 1].view(27, 1, 1, 1)
        cz = c[:, 2].view(27, 1, 1, 1)
        pi_xy_before = (cx * cy * f_neq).sum(0)

        tau = 0.83
        out = collide_cascaded_d3q27(f, tau=tau, s_bulk=1.0, s_3=1.0, s_4=1.0)
        f_neq_out = out - feq
        pi_xy_after = (cx * cy * f_neq_out).sum(0)

        expected = (1.0 - 1.0 / tau) * pi_xy_before
        torch.testing.assert_close(pi_xy_after, expected, rtol=1e-6, atol=1e-8)

    def test_bulk_relaxation_rate_independent(self):
        from tensorlbm.cascaded_collision import collide_cascaded_d3q27
        from tensorlbm.d3q27 import C as C27
        f = _d3q27_state()
        rho, ux, uy, uz = macroscopic27(f)
        feq = equilibrium27(rho, ux, uy, uz)
        f_neq = f - feq

        c = C27.to(device=f.device, dtype=f.dtype)
        cx = c[:, 0].view(27, 1, 1, 1)
        cy = c[:, 1].view(27, 1, 1, 1)
        cz = c[:, 2].view(27, 1, 1, 1)
        trace_before = (
            (cx * cx * f_neq).sum(0)
            + (cy * cy * f_neq).sum(0)
            + (cz * cz * f_neq).sum(0)
        )

        tau = 0.83
        s_bulk = 1.31
        out = collide_cascaded_d3q27(f, tau=tau, s_bulk=s_bulk, s_3=1.0, s_4=1.0)
        f_neq_out = out - feq
        trace_after = (
            (cx * cx * f_neq_out).sum(0)
            + (cy * cy * f_neq_out).sum(0)
            + (cz * cz * f_neq_out).sum(0)
        )
        expected = (1.0 - s_bulk) * trace_before
        torch.testing.assert_close(trace_after, expected, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# Contract registration
# ---------------------------------------------------------------------------

class TestContractRegistration:
    def test_d3q19_cm_is_available(self):
        from tensorlbm.advanced_collision_contract import collision_capability_matrix
        matrix = collision_capability_matrix()
        cap = matrix["D3Q19"]["CM"]
        assert cap.available
        assert cap.status == "AVAILABLE"
        assert cap.entrypoint == "tensorlbm.cascaded_collision.collide_cascaded_d3q19"

    def test_d3q27_cm_is_available(self):
        from tensorlbm.advanced_collision_contract import collision_capability_matrix
        matrix = collision_capability_matrix()
        cap = matrix["D3Q27"]["CM"]
        assert cap.available
        assert cap.status == "AVAILABLE"
        assert cap.entrypoint == "tensorlbm.cascaded_collision.collide_cascaded_d3q27"

    @pytest.mark.parametrize("lattice,q,equilibrium", [
        ("D3Q19", 19, equilibrium3d),
        ("D3Q27", 27, equilibrium27),
    ])
    def test_cm_dispatch_equilibrium_fixed_point(self, lattice, q, equilibrium):
        from tensorlbm.advanced_collision_contract import collide_advanced_3d
        rho = torch.ones((2, 3, 4))
        zero = torch.zeros_like(rho)
        f = equilibrium(rho, zero, zero, zero)
        out = collide_advanced_3d(lattice, "CM", f, tau=0.8)
        assert out.shape == (q, 2, 3, 4)
        torch.testing.assert_close(out, f, rtol=2e-5, atol=2e-5)

    @pytest.mark.parametrize("lattice,q,equilibrium", [
        ("D3Q19", 19, equilibrium3d),
        ("D3Q27", 27, equilibrium27),
    ])
    def test_cm_dispatch_conservation(self, lattice, q, equilibrium):
        from tensorlbm.advanced_collision_contract import collide_advanced_3d
        torch.manual_seed(301)
        rho = 0.9 + 0.2 * torch.rand((2, 3, 4))
        ux = 0.03 * torch.randn_like(rho)
        uy = 0.03 * torch.randn_like(rho)
        uz = 0.03 * torch.randn_like(rho)
        f = equilibrium(rho, ux, uy, uz) + 1e-3 * torch.randn(q, 2, 3, 4)
        out = collide_advanced_3d(
            lattice, "CM", f, tau=0.83, s_bulk=1.13, s_3=0.71, s_4=1.37,
        )
        # Check mass conservation
        assert torch.allclose(out.sum(0), f.sum(0), atol=1e-5)


# ---------------------------------------------------------------------------
# Stability: short collision-streaming run
# ---------------------------------------------------------------------------

class TestStability:
    def test_d3q19_periodic_stability(self):
        """A few collision+streaming steps must remain bounded."""
        from tensorlbm.cascaded_collision import collide_cascaded_d3q19
        from tensorlbm.solver3d import stream3d

        torch.manual_seed(401)
        nz, ny, nx = 8, 8, 8
        rho = torch.ones((nz, ny, nx), dtype=torch.float64)
        ux = 0.02 * torch.randn(nz, ny, nx, dtype=torch.float64)
        uy = 0.02 * torch.randn(nz, ny, nx, dtype=torch.float64)
        uz = 0.02 * torch.randn(nz, ny, nx, dtype=torch.float64)
        f = equilibrium3d(rho, ux, uy, uz)
        f += 1e-4 * torch.randn_like(f)

        tau = 0.6
        for _ in range(20):
            f = collide_cascaded_d3q19(f, tau=tau, s_bulk=1.0, s_3=1.0, s_4=1.0)
            f = stream3d(f)

        assert torch.isfinite(f).all(), "populations became non-finite"
        rho_out = f.sum(0)
        torch.testing.assert_close(rho_out, rho, rtol=1e-3, atol=1e-3)

    def test_d3q27_periodic_stability(self):
        """A few collision+streaming steps must remain bounded."""
        from tensorlbm.cascaded_collision import collide_cascaded_d3q27
        from tensorlbm.d3q27 import stream27

        torch.manual_seed(402)
        nz, ny, nx = 8, 8, 8
        rho = torch.ones((nz, ny, nx), dtype=torch.float64)
        ux = 0.02 * torch.randn(nz, ny, nx, dtype=torch.float64)
        uy = 0.02 * torch.randn(nz, ny, nx, dtype=torch.float64)
        uz = 0.02 * torch.randn(nz, ny, nx, dtype=torch.float64)
        f = equilibrium27(rho, ux, uy, uz)
        f += 1e-4 * torch.randn_like(f)

        tau = 0.6
        for _ in range(20):
            f = collide_cascaded_d3q27(f, tau=tau, s_bulk=1.0, s_3=1.0, s_4=1.0)
            f = stream27(f)

        assert torch.isfinite(f).all(), "populations became non-finite"
        rho_out = f.sum(0)
        torch.testing.assert_close(rho_out, rho, rtol=1e-3, atol=1e-3)

    def test_d3q19_sphere_flow_stability(self):
        """Simulate a simple uniform flow past an obstacle (stability check)."""
        from tensorlbm.cascaded_collision import collide_cascaded_d3q19
        from tensorlbm.solver3d import stream3d

        torch.manual_seed(403)
        nz, ny, nx = 12, 12, 12
        rho = torch.ones((nz, ny, nx), dtype=torch.float64)
        ux = 0.05 * torch.ones(nz, ny, nx, dtype=torch.float64)
        uy = torch.zeros_like(ux)
        uz = torch.zeros_like(ux)
        f = equilibrium3d(rho, ux, uy, uz)

        # Place a simple obstacle mask (sphere-like)
        cz, cy, cx = torch.meshgrid(
            torch.arange(nz), torch.arange(ny), torch.arange(nx),
            indexing="ij",
        )
        center = torch.tensor([nz // 2, ny // 2, nx // 2], dtype=torch.float64)
        dist = ((cz - center[0]) ** 2 + (cy - center[1]) ** 2 + (cx - center[2]) ** 2).sqrt()
        obstacle = dist < 2.0

        tau = 0.7
        for step in range(30):
            f = collide_cascaded_d3q19(f, tau=tau, s_bulk=1.0, s_3=1.0, s_4=1.0)
            f = stream3d(f)
            # Simple bounce-back on obstacle
            for q in range(19):
                f[q, obstacle] = f[q, obstacle].clamp(min=0)

        assert torch.isfinite(f).all(), "populations became non-finite"
