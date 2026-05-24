"""Tests for the Regularized BGK (RLBM) collision operators.

Reference: Latt & Chopard, *Math. Comput. Simul.* (2006).

Covers:
- Mass and momentum conservation (D2Q9 and D3Q19).
- Identity at equilibrium (f_eq is a fixed point of RLBM).
- Agreement with BGK when starting from a low-order (purely hydrodynamic)
  non-equilibrium state.
- Finite, stable output at near-marginal τ.
"""
from __future__ import annotations

import torch

from tensorlbm import (
    collide_bgk,
    collide_rlbm,
    collide_rlbm3d,
    equilibrium,
    equilibrium3d,
    macroscopic,
    macroscopic3d,
)

# ---------------------------------------------------------------------------
# D2Q9 RLBM tests
# ---------------------------------------------------------------------------


class TestCollideRLBM2D:
    def test_preserves_shape(self) -> None:
        ny, nx = 8, 12
        rho = torch.ones((ny, nx))
        ux = torch.full_like(rho, 0.05)
        uy = torch.zeros_like(rho)
        f = equilibrium(rho, ux, uy)
        f_out = collide_rlbm(f, tau=0.7)
        assert f_out.shape == f.shape

    def test_conserves_mass(self) -> None:
        ny, nx = 8, 12
        torch.manual_seed(0)
        rho = torch.rand((ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.05
        uy = torch.rand_like(rho) * 0.03
        f = equilibrium(rho, ux, uy)
        # Add small random perturbation to test non-equilibrium projection
        f = f + 1e-4 * torch.randn_like(f)
        rho_in = f.sum(dim=0)
        f_out = collide_rlbm(f, tau=0.7)
        rho_out = f_out.sum(dim=0)
        assert torch.allclose(rho_out, rho_in, atol=1e-5)

    def test_conserves_momentum(self) -> None:
        ny, nx = 8, 12
        torch.manual_seed(1)
        rho = torch.rand((ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.05
        uy = torch.rand_like(rho) * 0.03
        f = equilibrium(rho, ux, uy)
        f = f + 1e-4 * torch.randn_like(f)
        _, ux_in, uy_in = macroscopic(f)
        f_out = collide_rlbm(f, tau=0.7)
        _, ux_out, uy_out = macroscopic(f_out)
        assert torch.allclose(ux_out, ux_in, atol=1e-5)
        assert torch.allclose(uy_out, uy_in, atol=1e-5)

    def test_identity_at_equilibrium(self) -> None:
        """f_eq must be a fixed point of RLBM (Π_neq vanishes)."""
        ny, nx = 8, 12
        rho = torch.ones((ny, nx))
        ux = torch.full_like(rho, 0.04)
        uy = torch.full_like(rho, -0.02)
        feq = equilibrium(rho, ux, uy)
        f_out = collide_rlbm(feq, tau=0.7)
        assert torch.allclose(f_out, feq, atol=1e-6)

    def test_finite_output_low_tau(self) -> None:
        """RLBM should remain finite even at near-marginal τ (low viscosity)."""
        ny, nx = 8, 12
        torch.manual_seed(2)
        rho = torch.ones((ny, nx))
        ux = torch.full_like(rho, 0.05)
        uy = torch.zeros_like(rho)
        f = equilibrium(rho, ux, uy) + 1e-3 * torch.randn(9, ny, nx)
        f_out = collide_rlbm(f, tau=0.505)
        assert torch.isfinite(f_out).all()

    def test_matches_bgk_for_hermite_neq(self) -> None:
        """When fneq is purely a 2nd-order Hermite mode, RLBM and BGK coincide.

        We construct fneq by giving the distribution a small velocity-derivative
        signature consistent with a Newtonian shear. For such a state the
        projection onto the Hermite-2 space is the identity, so RLBM and BGK
        should yield the same post-collision distribution.
        """
        ny, nx = 16, 16
        rho = torch.ones((ny, nx))
        # A shear flow: ux varies linearly in y
        ys = torch.arange(ny, dtype=torch.float32).view(ny, 1).expand(ny, nx)
        ux = 0.001 * (ys - (ny - 1) / 2.0)
        uy = torch.zeros_like(ux)
        # Build f as feq evaluated *with* the velocity field, then take one BGK
        # step on the equilibrium plus a small Hermite-2 perturbation built from
        # the strain rate: this is the so-called "f1" assumption used by Latt.
        f_eq = equilibrium(rho, ux, uy)
        # Apply Chapman-Enskog-consistent f1: f1_i = -τ * w_i / c_s^2 * Q_i:S
        # where S is the strain-rate tensor. For our linear shear S_xy = du/dy/2.
        from tensorlbm.d2q9 import C, W

        c = C.float()
        w = W.float()
        cx = c[:, 0].view(9, 1, 1)
        cy = c[:, 1].view(9, 1, 1)
        w_v = w.view(9, 1, 1)
        du_dy = 0.001  # constant shear
        # Π_neq_xy ~ -2 ν ρ S_xy = -ν ρ du/dy
        nu = 0.2
        pi_xy = -nu * rho * du_dy
        # f1_i = (w_i / (2 c_s^4)) * 2 Q_i,xy * Π_xy
        f1 = (9.0 / 2.0) * w_v * 2.0 * (cx * cy) * pi_xy.unsqueeze(0)
        f = f_eq + f1

        f_bgk = collide_bgk(f, tau=0.6)
        f_rlbm = collide_rlbm(f, tau=0.6)
        # Allow small numerical noise from macroscopic recomputation
        assert torch.allclose(f_bgk, f_rlbm, atol=1e-5)


# ---------------------------------------------------------------------------
# D3Q19 RLBM tests
# ---------------------------------------------------------------------------


class TestCollideRLBM3D:
    def test_preserves_shape(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.04)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)
        f = equilibrium3d(rho, ux, uy, uz)
        f_out = collide_rlbm3d(f, tau=0.7)
        assert f_out.shape == f.shape

    def test_conserves_mass(self) -> None:
        nz, ny, nx = 4, 6, 8
        torch.manual_seed(3)
        rho = torch.rand((nz, ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.04
        uy = torch.rand_like(rho) * 0.02
        uz = torch.rand_like(rho) * 0.01
        f = equilibrium3d(rho, ux, uy, uz)
        f = f + 1e-4 * torch.randn_like(f)
        rho_in = f.sum(dim=0)
        f_out = collide_rlbm3d(f, tau=0.7)
        rho_out = f_out.sum(dim=0)
        assert torch.allclose(rho_out, rho_in, atol=1e-4)

    def test_conserves_momentum(self) -> None:
        nz, ny, nx = 4, 6, 8
        torch.manual_seed(4)
        rho = torch.rand((nz, ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.04
        uy = torch.rand_like(rho) * 0.02
        uz = torch.rand_like(rho) * 0.01
        f = equilibrium3d(rho, ux, uy, uz)
        f = f + 1e-4 * torch.randn_like(f)
        _, ux_in, uy_in, uz_in = macroscopic3d(f)
        f_out = collide_rlbm3d(f, tau=0.7)
        _, ux_out, uy_out, uz_out = macroscopic3d(f_out)
        assert torch.allclose(ux_out, ux_in, atol=1e-4)
        assert torch.allclose(uy_out, uy_in, atol=1e-4)
        assert torch.allclose(uz_out, uz_in, atol=1e-4)

    def test_identity_at_equilibrium(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.03)
        uy = torch.full_like(rho, 0.01)
        uz = torch.full_like(rho, -0.01)
        feq = equilibrium3d(rho, ux, uy, uz)
        f_out = collide_rlbm3d(feq, tau=0.7)
        assert torch.allclose(f_out, feq, atol=1e-5)

    def test_finite_output_low_tau(self) -> None:
        nz, ny, nx = 4, 6, 8
        torch.manual_seed(5)
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.04)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)
        f = equilibrium3d(rho, ux, uy, uz) + 1e-3 * torch.randn(19, nz, ny, nx)
        f_out = collide_rlbm3d(f, tau=0.505)
        assert torch.isfinite(f_out).all()

    def test_pi_neq_relaxed_correctly(self) -> None:
        """After RLBM, Π_neq must be exactly (1 − 1/τ) times its pre-collision value.

        This is the defining property of the regularized scheme: the second-order
        non-equilibrium moments are the only modes kept, and they relax at rate
        ω = 1/τ.
        """
        nz, ny, nx = 4, 6, 8
        torch.manual_seed(11)
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.02)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)
        f = equilibrium3d(rho, ux, uy, uz) + 1e-3 * torch.randn(19, nz, ny, nx)

        tau = 0.8
        # Pre-collision macroscopic + Π
        rho_in, ux_in, uy_in, uz_in = macroscopic3d(f)
        feq_in = equilibrium3d(rho_in, ux_in, uy_in, uz_in)
        fneq_in = f - feq_in

        from tensorlbm.d3q19 import C as C3

        c = C3.float()
        cx, cy = c[:, 0], c[:, 1]
        pi_xy_in = (cx.view(19, 1, 1, 1) * cy.view(19, 1, 1, 1) * fneq_in).sum(dim=0)

        f_out = collide_rlbm3d(f, tau=tau)
        # Post-collision: mass and momentum unchanged, so feq is the same
        fneq_out = f_out - feq_in
        pi_xy_out = (cx.view(19, 1, 1, 1) * cy.view(19, 1, 1, 1) * fneq_out).sum(dim=0)

        expected = (1.0 - 1.0 / tau) * pi_xy_in
        assert torch.allclose(pi_xy_out, expected, atol=1e-6)

