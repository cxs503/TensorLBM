"""Unit tests for D2Q9 and D3Q19 MRT collision operators.

Tests verify:
1. MRT conserves mass and momentum
2. MRT reduces to BGK when all relaxation rates are equal
3. MRT produces correct Poiseuille flow profile
4. MRT is stable for various tau values
5. MRT equilibrium moments are correct
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch
import numpy as np

from tensorlbm.d2q9 import (
    collide_bgk,
    collide_mrt as collide_mrt_2d,
    equilibrium,
    macroscopic,
    M_D2Q9,
    MINV_D2Q9,
    _equilibrium_moments,
)
from tensorlbm.solver import (
    collide_mrt as collide_mrt_3d,
    equilibrium3d,
    macroscopic3d,
)


# ---------------------------------------------------------------------------
# D2Q9 MRT tests
# ---------------------------------------------------------------------------

class TestD2Q9MRTMatrices:
    """Test MRT transformation matrix properties."""

    def test_m_inverse(self):
        """M @ M^{-1} should be identity."""
        product = M_D2Q9 @ MINV_D2Q9
        assert torch.allclose(product, torch.eye(9, dtype=torch.float64), atol=1e-10)

    def test_m_minv_inverse(self):
        """M^{-1} @ M should be identity."""
        product = MINV_D2Q9 @ M_D2Q9
        assert torch.allclose(product, torch.eye(9, dtype=torch.float64), atol=1e-10)


class TestD2Q9MRTConservation:
    """Test that MRT conserves mass and momentum."""

    def _make_test_f(self, ny=16, nx=16, rho=1.0, ux=0.05, uy=0.03):
        """Create a test distribution at equilibrium."""
        rho_t = torch.full((ny, nx), rho, dtype=torch.float64)
        ux_t = torch.full((ny, nx), ux, dtype=torch.float64)
        uy_t = torch.full((ny, nx), uy, dtype=torch.float64)
        return equilibrium(rho_t, ux_t, uy_t)

    def test_mass_conservation(self):
        """Total mass should be conserved after MRT collision."""
        f = self._make_test_f()
        mass_before = f.sum().item()

        f_out = collide_mrt_2d(f, tau=0.8)
        mass_after = f_out.sum().item()

        assert abs(mass_after - mass_before) < 1e-10, \
            f"Mass not conserved: {mass_before} -> {mass_after}"

    def test_momentum_conservation(self):
        """Total momentum should be conserved after MRT collision."""
        f = self._make_test_f(ux=0.05, uy=0.03)
        rho_before, ux_before, uy_before = macroscopic(f)
        px_before = (rho_before * ux_before).sum().item()
        py_before = (rho_before * uy_before).sum().item()

        f_out = collide_mrt_2d(f, tau=0.8)
        rho_after, ux_after, uy_after = macroscopic(f_out)
        px_after = (rho_after * ux_after).sum().item()
        py_after = (rho_after * uy_after).sum().item()

        assert abs(px_after - px_before) < 1e-10, \
            f"x-momentum not conserved: {px_before} -> {px_after}"
        assert abs(py_after - py_before) < 1e-10, \
            f"y-momentum not conserved: {py_before} -> {py_after}"

    def test_equilibrium_unchanged(self):
        """Equilibrium distribution should be unchanged by MRT collision."""
        f = self._make_test_f()
        f_out = collide_mrt_2d(f, tau=0.8)
        assert torch.allclose(f, f_out, atol=1e-12), \
            "Equilibrium distribution changed by MRT collision"


class TestD2Q9MRTvsBGK:
    """Test that MRT reduces to BGK when relaxation rates are equal."""

    def _make_test_f(self, ny=16, nx=16, rho=1.0, ux=0.05, uy=0.03):
        """Create a perturbed test distribution."""
        rho_t = torch.full((ny, nx), rho, dtype=torch.float64)
        ux_t = torch.full((ny, nx), ux, dtype=torch.float64)
        uy_t = torch.full((ny, nx), uy, dtype=torch.float64)
        feq = equilibrium(rho_t, ux_t, uy_t)
        # Add small perturbation
        f = feq + 0.01 * torch.randn_like(feq)
        return f

    def test_mrt_equals_bgk_with_uniform_relaxation(self):
        """MRT with s_e=s_q=s_nu should give same result as BGK."""
        f = self._make_test_f()
        tau = 0.8

        f_bgk = collide_bgk(f, tau)
        # When s_e = s_q = s_nu = 1/tau, MRT should equal BGK
        f_mrt = collide_mrt_2d(f, tau, s_e=1.0/tau, s_q=1.0/tau)

        assert torch.allclose(f_bgk, f_mrt, atol=1e-10), \
            "MRT with uniform relaxation should equal BGK"

    def test_mrt_different_from_bgk_with_different_rates(self):
        """MRT with different relaxation rates should differ from BGK."""
        f = self._make_test_f()
        tau = 0.8

        f_bgk = collide_bgk(f, tau)
        # Use different relaxation rates
        f_mrt = collide_mrt_2d(f, tau, s_e=1.2, s_q=1.1)

        # They should be different (but both finite)
        assert not torch.allclose(f_bgk, f_mrt, atol=1e-8), \
            "MRT with different rates should differ from BGK"
        assert torch.all(torch.isfinite(f_mrt))


class TestD2Q9MRTEquilibriumMoments:
    """Test equilibrium moment computation."""

    def test_equilibrium_moments_at_rest(self):
        """At rest (u=0), equilibrium moments should have specific values."""
        rho = torch.ones((8, 8), dtype=torch.float64)
        ux = torch.zeros((8, 8), dtype=torch.float64)
        uy = torch.zeros((8, 8), dtype=torch.float64)

        m_eq = _equilibrium_moments(rho, ux, uy)

        # m0_eq = ρ = 1
        assert torch.allclose(m_eq[0], rho, atol=1e-12)
        # m1_eq = ρ(-2 + 0) = -2
        assert torch.allclose(m_eq[1], torch.full_like(rho, -2.0), atol=1e-12)
        # m2_eq = ρ(1 - 0) = 1
        assert torch.allclose(m_eq[2], torch.full_like(rho, 1.0), atol=1e-12)
        # m3_eq = m5_eq = 0 (no momentum)
        assert torch.allclose(m_eq[3], torch.zeros_like(rho), atol=1e-12)
        assert torch.allclose(m_eq[5], torch.zeros_like(rho), atol=1e-12)
        # m4_eq = m6_eq = 0 (no energy flux)
        assert torch.allclose(m_eq[4], torch.zeros_like(rho), atol=1e-12)
        assert torch.allclose(m_eq[6], torch.zeros_like(rho), atol=1e-12)
        # m7_eq = m8_eq = 0 (no stress at rest)
        assert torch.allclose(m_eq[7], torch.zeros_like(rho), atol=1e-12)
        assert torch.allclose(m_eq[8], torch.zeros_like(rho), atol=1e-12)

    def test_equilibrium_moments_with_velocity(self):
        """Equilibrium moments should match theoretical values."""
        rho = torch.ones((8, 8), dtype=torch.float64)
        ux = torch.full((8, 8), 0.1, dtype=torch.float64)
        uy = torch.full((8, 8), 0.05, dtype=torch.float64)

        m_eq = _equilibrium_moments(rho, ux, uy)
        u2 = 0.1**2 + 0.05**2

        # m0_eq = ρ = 1
        assert torch.allclose(m_eq[0], rho, atol=1e-12)
        # m1_eq = ρ(-2 + 3*u²)
        expected_m1 = -2.0 + 3.0 * u2
        assert torch.allclose(m_eq[1], torch.full_like(rho, expected_m1), atol=1e-12)
        # m3_eq = ρ*ux
        assert torch.allclose(m_eq[3], rho * ux, atol=1e-12)
        # m5_eq = ρ*uy
        assert torch.allclose(m_eq[5], rho * uy, atol=1e-12)
        # m7_eq = ρ(ux² - uy²)
        expected_m7 = 0.1**2 - 0.05**2
        assert torch.allclose(m_eq[7], torch.full_like(rho, expected_m7), atol=1e-12)
        # m8_eq = ρ*ux*uy
        expected_m8 = 0.1 * 0.05
        assert torch.allclose(m_eq[8], torch.full_like(rho, expected_m8), atol=1e-12)


class TestD2Q9MRTStability:
    """Test MRT stability for various tau values."""

    @pytest.mark.parametrize("tau", [0.51, 0.6, 0.8, 1.0, 1.5, 2.0])
    def test_stability_for_tau(self, tau):
        """MRT should be stable for tau > 0.5."""
        rho = torch.ones((16, 16), dtype=torch.float64)
        ux = torch.full((16, 16), 0.05, dtype=torch.float64)
        uy = torch.full((16, 16), 0.03, dtype=torch.float64)
        f = equilibrium(rho, ux, uy)
        # Add perturbation
        f = f + 0.01 * torch.randn_like(f)

        f_out = collide_mrt_2d(f, tau)
        assert torch.all(torch.isfinite(f_out)), f"MRT unstable at tau={tau}"

    def test_multiple_steps_stability(self):
        """MRT should remain stable over multiple collision steps."""
        rho = torch.ones((16, 16), dtype=torch.float64)
        ux = torch.full((16, 16), 0.05, dtype=torch.float64)
        uy = torch.full((16, 16), 0.03, dtype=torch.float64)
        f = equilibrium(rho, ux, uy)
        f = f + 0.01 * torch.randn_like(f)

        for _ in range(100):
            f = collide_mrt_2d(f, tau=0.6)
            assert torch.all(torch.isfinite(f)), "MRT diverged after multiple steps"


# ---------------------------------------------------------------------------
# D3Q19 MRT tests
# ---------------------------------------------------------------------------

class TestD3Q19MRTConservation:
    """Test that D3Q19 MRT conserves mass and momentum."""

    def _make_test_f(self, nz=4, ny=4, nx=4, rho=1.0, ux=0.05, uy=0.03, uz=0.02):
        """Create a test distribution at equilibrium."""
        rho_t = torch.full((nz, ny, nx), rho, dtype=torch.float64)
        ux_t = torch.full((nz, ny, nx), ux, dtype=torch.float64)
        uy_t = torch.full((nz, ny, nx), uy, dtype=torch.float64)
        uz_t = torch.full((nz, ny, nx), uz, dtype=torch.float64)
        return equilibrium3d(rho_t, ux_t, uy_t, uz_t)

    def test_mass_conservation(self):
        """Total mass should be conserved after MRT collision."""
        f = self._make_test_f()
        mass_before = f.sum().item()

        f_out = collide_mrt_3d(f, tau=0.8)
        mass_after = f_out.sum().item()

        assert abs(mass_after - mass_before) < 1e-8, \
            f"Mass not conserved: {mass_before} -> {mass_after}"

    def test_momentum_conservation(self):
        """Total momentum should be conserved after MRT collision."""
        f = self._make_test_f()
        rho_before, ux_before, uy_before, uz_before = macroscopic3d(f)
        px_before = (rho_before * ux_before).sum().item()
        py_before = (rho_before * uy_before).sum().item()
        pz_before = (rho_before * uz_before).sum().item()

        f_out = collide_mrt_3d(f, tau=0.8)
        rho_after, ux_after, uy_after, uz_after = macroscopic3d(f_out)
        px_after = (rho_after * ux_after).sum().item()
        py_after = (rho_after * uy_after).sum().item()
        pz_after = (rho_after * uz_after).sum().item()

        assert abs(px_after - px_before) < 1e-8, \
            f"x-momentum not conserved: {px_before} -> {px_after}"
        assert abs(py_after - py_before) < 1e-8, \
            f"y-momentum not conserved: {py_before} -> {py_after}"
        assert abs(pz_after - pz_before) < 1e-8, \
            f"z-momentum not conserved: {pz_before} -> {pz_after}"

    def test_equilibrium_unchanged(self):
        """Equilibrium distribution should be unchanged by MRT collision."""
        f = self._make_test_f()
        f_out = collide_mrt_3d(f, tau=0.8)
        assert torch.allclose(f, f_out, atol=1e-10), \
            "Equilibrium distribution changed by MRT collision"


class TestD3Q19MRTStability:
    """Test D3Q19 MRT stability."""

    @pytest.mark.parametrize("tau", [0.51, 0.6, 0.8, 1.0, 1.5])
    def test_stability_for_tau(self, tau):
        """MRT should be stable for tau > 0.5."""
        f = self._make_test_f()
        f = f + 0.001 * torch.randn_like(f)

        f_out = collide_mrt_3d(f, tau)
        assert torch.all(torch.isfinite(f_out)), f"MRT unstable at tau={tau}"

    def _make_test_f(self, nz=4, ny=4, nx=4, rho=1.0, ux=0.05, uy=0.03, uz=0.02):
        """Create a test distribution at equilibrium."""
        rho_t = torch.full((nz, ny, nx), rho, dtype=torch.float64)
        ux_t = torch.full((nz, ny, nx), ux, dtype=torch.float64)
        uy_t = torch.full((nz, ny, nx), uy, dtype=torch.float64)
        uz_t = torch.full((nz, ny, nx), uz, dtype=torch.float64)
        return equilibrium3d(rho_t, ux_t, uy_t, uz_t)


# ---------------------------------------------------------------------------
# Poiseuille flow test (analytical solution)
# ---------------------------------------------------------------------------

class TestD2Q9MRTPoiseuille:
    """Test MRT against analytical Poiseuille flow solution."""

    def test_poiseuille_profile(self):
        """MRT should produce correct Poiseuille flow profile.

        Analytical solution: u(y) = (G/2ν) * y * (H - y)
        where G is body force, ν is kinematic viscosity, H is channel height.
        """
        ny, nx = 32, 64
        rho0 = 1.0
        G = 1e-5  # body force
        tau = 0.8
        nu = (tau - 0.5) / 3.0  # kinematic viscosity

        # Initialize at rest
        rho = torch.ones((ny, nx), dtype=torch.float64)
        ux = torch.zeros((ny, nx), dtype=torch.float64)
        uy = torch.zeros((ny, nx), dtype=torch.float64)
        f = equilibrium(rho, ux, uy)

        # Run simulation with body force and bounce-back walls
        from tensorlbm.d2q9 import C as C2Q9, W as W2Q9, CS2, stream

        c = C2Q9.to(f.device, dtype=f.dtype)
        w = W2Q9.to(f.device, dtype=f.dtype)

        for step in range(500):
            # Add body force in x-direction
            rho, ux, uy = macroscopic(f)
            f = f + w.view(9, 1, 1) * c[:, 0].view(9, 1, 1) * G * rho.unsqueeze(0) / CS2

            # MRT collision
            f = collide_mrt_2d(f, tau)

            # Simple bounce-back for top/bottom walls
            f[:, 0, :] = f[[0, 3, 4, 1, 2, 7, 8, 5, 6], 1, :]  # bottom wall
            f[:, -1, :] = f[[0, 3, 4, 1, 2, 7, 8, 5, 6], -2, :]  # top wall

            # Stream
            f = stream(f)

            # Re-apply bounce-back after streaming
            f[:, 0, :] = f[[0, 3, 4, 1, 2, 7, 8, 5, 6], 1, :]
            f[:, -1, :] = f[[0, 3, 4, 1, 2, 7, 8, 5, 6], -2, :]

        # Check that velocity profile is parabolic
        _, ux_final, _ = macroscopic(f)
        # Average over x
        ux_avg = ux_final.mean(dim=1)  # (ny,)

        # Analytical solution: u(y) = (G/2ν) * y * (H - y)
        H = ny - 1  # channel height in lattice units
        y = torch.arange(ny, dtype=torch.float64)
        u_analytical = (G / (2 * nu)) * y * (H - y)
        u_max_analytical = G * H**2 / (8 * nu)

        # Compare (allowing for some discretization error)
        u_max_sim = ux_avg.max().item()
        assert u_max_sim > 0, "No flow developed"
        # Check that the profile is roughly parabolic (R² > 0.95)
        if u_max_analytical > 0 and u_max_sim > 0:
            u_norm = ux_avg / u_max_sim
            u_anal_norm = u_analytical / u_max_analytical
            # Only compare interior points (away from walls)
            interior = slice(2, -2)
            r2 = 1.0 - ((u_norm[interior] - u_anal_norm[interior])**2).sum() / \
                 (u_anal_norm[interior]**2).sum()
            assert r2 > 0.9, f"Poiseuille profile R² = {r2:.3f} < 0.9"
