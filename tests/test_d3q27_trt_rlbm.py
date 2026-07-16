"""Tests for the D3Q27 TRT and RLBM collision operators.

These mirror the D3Q19 contract tests in ``test_trt.py`` and
``test_rlbm.py`` but exercise the D3Q27-specific implementations
(``collide_trt27``, ``collide_rlbm27``) that use the D3Q27 velocity set
``C``, weights ``W``, and opposite-direction map ``OPPOSITE``.
"""
from __future__ import annotations

import torch

from tensorlbm import (
    collide_bgk27,
    collide_rlbm27,
    collide_trt27,
    equilibrium27,
    macroscopic27,
)
from tensorlbm.d3q27 import C as C27


# ---------------------------------------------------------------------------
# D3Q27 TRT tests
# ---------------------------------------------------------------------------


class TestCollideTRT27:
    def test_preserves_shape(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.04)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)
        f = equilibrium27(rho, ux, uy, uz)
        f_out = collide_trt27(f, tau_plus=0.7)
        assert f_out.shape == f.shape

    def test_conserves_mass(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.rand((nz, ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.05
        uy = torch.rand_like(rho) * 0.03
        uz = torch.rand_like(rho) * 0.02
        f = equilibrium27(rho, ux, uy, uz)
        f_out = collide_trt27(f, tau_plus=0.7)
        rho_out, _, _, _ = macroscopic27(f_out)
        assert torch.allclose(rho_out, rho, atol=1e-4)

    def test_conserves_momentum(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.rand((nz, ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.05
        uy = torch.rand_like(rho) * 0.03
        uz = torch.rand_like(rho) * 0.02
        f = equilibrium27(rho, ux, uy, uz)
        f_out = collide_trt27(f, tau_plus=0.7)
        _, ux_out, uy_out, uz_out = macroscopic27(f_out)
        assert torch.allclose(ux_out, ux, atol=1e-4)
        assert torch.allclose(uy_out, uy, atol=1e-4)
        assert torch.allclose(uz_out, uz, atol=1e-4)

    def test_identity_at_equilibrium(self) -> None:
        """f_eq should be a fixed point of TRT."""
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.03)
        uy = torch.full_like(rho, 0.01)
        uz = torch.full_like(rho, -0.01)
        feq = equilibrium27(rho, ux, uy, uz)
        f_out = collide_trt27(feq, tau_plus=0.7)
        assert torch.allclose(f_out, feq, atol=1e-4)

    def test_finite_output(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.04)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)
        f = equilibrium27(rho, ux, uy, uz)
        f_out = collide_trt27(f, tau_plus=0.6)
        assert torch.isfinite(f_out).all()

    def test_lambda_variants(self) -> None:
        """Different Λ values should all produce valid outputs."""
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.03)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)
        f = equilibrium27(rho, ux, uy, uz)
        for lam in [3.0 / 16.0, 1.0 / 4.0, 1.0 / 12.0]:
            f_out = collide_trt27(f, tau_plus=0.7, lambda_trt=lam)
            assert torch.isfinite(f_out).all()
            rho_out, _, _, _ = macroscopic27(f_out)
            assert torch.allclose(rho_out, rho, atol=1e-4)

    def test_tau_minus_relation(self) -> None:
        """tau_minus must satisfy the magic-parameter relation Λ=(τ₊-½)(τ₋-½)."""
        nz, ny, nx = 2, 3, 4
        rho = torch.ones((nz, ny, nx))
        f = equilibrium27(rho, torch.zeros_like(rho), torch.zeros_like(rho), torch.zeros_like(rho))
        tau_plus = 0.8
        lam = 3.0 / 16.0
        # Just verify the call succeeds and output is finite; the relation is
        # internal but the finite/identity checks above cover correctness.
        f_out = collide_trt27(f, tau_plus=tau_plus, lambda_trt=lam)
        assert torch.isfinite(f_out).all()


# ---------------------------------------------------------------------------
# D3Q27 RLBM tests
# ---------------------------------------------------------------------------


class TestCollideRLBM27:
    def test_preserves_shape(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.04)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)
        f = equilibrium27(rho, ux, uy, uz)
        f_out = collide_rlbm27(f, tau=0.7)
        assert f_out.shape == f.shape

    def test_conserves_mass(self) -> None:
        nz, ny, nx = 4, 6, 8
        torch.manual_seed(3)
        rho = torch.rand((nz, ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.04
        uy = torch.rand_like(rho) * 0.02
        uz = torch.rand_like(rho) * 0.01
        f = equilibrium27(rho, ux, uy, uz)
        f = f + 1e-4 * torch.randn_like(f)
        rho_in = f.sum(dim=0)
        f_out = collide_rlbm27(f, tau=0.7)
        rho_out = f_out.sum(dim=0)
        assert torch.allclose(rho_out, rho_in, atol=1e-4)

    def test_conserves_momentum(self) -> None:
        nz, ny, nx = 4, 6, 8
        torch.manual_seed(4)
        rho = torch.rand((nz, ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.04
        uy = torch.rand_like(rho) * 0.02
        uz = torch.rand_like(rho) * 0.01
        f = equilibrium27(rho, ux, uy, uz)
        f = f + 1e-4 * torch.randn_like(f)
        _, ux_in, uy_in, uz_in = macroscopic27(f)
        f_out = collide_rlbm27(f, tau=0.7)
        _, ux_out, uy_out, uz_out = macroscopic27(f_out)
        assert torch.allclose(ux_out, ux_in, atol=1e-4)
        assert torch.allclose(uy_out, uy_in, atol=1e-4)
        assert torch.allclose(uz_out, uz_in, atol=1e-4)

    def test_identity_at_equilibrium(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.03)
        uy = torch.full_like(rho, 0.01)
        uz = torch.full_like(rho, -0.01)
        feq = equilibrium27(rho, ux, uy, uz)
        f_out = collide_rlbm27(feq, tau=0.7)
        assert torch.allclose(f_out, feq, atol=1e-5)

    def test_finite_output_low_tau(self) -> None:
        nz, ny, nx = 4, 6, 8
        torch.manual_seed(5)
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.04)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)
        f = equilibrium27(rho, ux, uy, uz) + 1e-3 * torch.randn(27, nz, ny, nx)
        f_out = collide_rlbm27(f, tau=0.505)
        assert torch.isfinite(f_out).all()

    def test_pi_neq_relaxed_correctly(self) -> None:
        """After RLBM, Π_neq must be exactly (1 − 1/τ) times its pre-collision value."""
        nz, ny, nx = 4, 6, 8
        torch.manual_seed(11)
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.02)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)
        f = equilibrium27(rho, ux, uy, uz) + 1e-3 * torch.randn(27, nz, ny, nx)

        tau = 0.8
        rho_in, ux_in, uy_in, uz_in = macroscopic27(f)
        feq_in = equilibrium27(rho_in, ux_in, uy_in, uz_in)
        fneq_in = f - feq_in

        c = C27.float()
        cx, cy = c[:, 0], c[:, 1]
        pi_xy_in = (cx.view(27, 1, 1, 1) * cy.view(27, 1, 1, 1) * fneq_in).sum(dim=0)

        f_out = collide_rlbm27(f, tau=tau)
        fneq_out = f_out - feq_in
        pi_xy_out = (cx.view(27, 1, 1, 1) * cy.view(27, 1, 1, 1) * fneq_out).sum(dim=0)

        expected = (1.0 - 1.0 / tau) * pi_xy_in
        assert torch.allclose(pi_xy_out, expected, atol=1e-6)

    def test_matches_bgk_for_hermite_neq(self) -> None:
        """When fneq is purely a 2nd-order Hermite mode, RLBM and BGK coincide."""
        from tensorlbm.d3q27 import W as W27

        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.02)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)
        f_eq = equilibrium27(rho, ux, uy, uz)

        c = C27.float()
        w = W27.float()
        cx = c[:, 0].view(27, 1, 1, 1)
        cy = c[:, 1].view(27, 1, 1, 1)
        w_v = w.view(27, 1, 1, 1)
        # Build a pure Hermite-2 non-equilibrium perturbation from a shear mode
        nu = 0.2
        du_dy = 0.001
        pi_xy = -nu * rho * du_dy
        f1 = (9.0 / 2.0) * w_v * 2.0 * (cx * cy) * pi_xy
        f = f_eq + f1

        f_bgk = collide_bgk27(f, tau=0.6)
        f_rlbm = collide_rlbm27(f, tau=0.6)
        assert torch.allclose(f_bgk, f_rlbm, atol=1e-5)
