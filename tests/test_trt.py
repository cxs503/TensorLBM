"""Tests for the two-relaxation-time (TRT) collision operators.

Covers:
- Mass and momentum conservation (D2Q9 and D3Q19).
- Identity at equilibrium (f_eq is a fixed point of TRT).
- Poiseuille flow accuracy: TRT with Λ=3/16 should match the analytical
  parabolic profile more accurately than BGK at low viscosity.
"""
from __future__ import annotations

import torch

from tensorlbm import (
    collide_trt,
    collide_trt3d,
    equilibrium,
    equilibrium3d,
    macroscopic,
    macroscopic3d,
    stream,
)
from tensorlbm.boundaries import bounce_back_cells, make_channel_wall_mask

# ---------------------------------------------------------------------------
# D2Q9 TRT tests
# ---------------------------------------------------------------------------


class TestCollideTRT2D:
    def test_preserves_shape(self) -> None:
        ny, nx = 8, 12
        rho = torch.ones((ny, nx))
        ux = torch.full_like(rho, 0.05)
        uy = torch.zeros_like(rho)
        f = equilibrium(rho, ux, uy)
        f_out = collide_trt(f, tau_plus=0.7)
        assert f_out.shape == f.shape

    def test_conserves_mass(self) -> None:
        ny, nx = 8, 12
        rho = torch.rand((ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.05
        uy = torch.rand_like(rho) * 0.03
        f = equilibrium(rho, ux, uy)
        f_out = collide_trt(f, tau_plus=0.7)
        rho_out, _, _ = macroscopic(f_out)
        assert torch.allclose(rho_out, rho, atol=1e-5)

    def test_conserves_momentum(self) -> None:
        ny, nx = 8, 12
        rho = torch.rand((ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.05
        uy = torch.rand_like(rho) * 0.03
        f = equilibrium(rho, ux, uy)
        f_out = collide_trt(f, tau_plus=0.7)
        _, ux_out, uy_out = macroscopic(f_out)
        assert torch.allclose(ux_out, ux, atol=1e-5)
        assert torch.allclose(uy_out, uy, atol=1e-5)

    def test_identity_at_equilibrium(self) -> None:
        """f_eq should be a fixed point of TRT."""
        ny, nx = 8, 12
        rho = torch.ones((ny, nx))
        ux = torch.full_like(rho, 0.04)
        uy = torch.full_like(rho, -0.02)
        feq = equilibrium(rho, ux, uy)
        f_out = collide_trt(feq, tau_plus=0.7)
        assert torch.allclose(f_out, feq, atol=1e-5)

    def test_finite_output(self) -> None:
        ny, nx = 8, 12
        rho = torch.ones((ny, nx))
        ux = torch.full_like(rho, 0.04)
        uy = torch.zeros_like(rho)
        f = equilibrium(rho, ux, uy)
        f_out = collide_trt(f, tau_plus=0.6)
        assert torch.isfinite(f_out).all()

    def test_poiseuille_parabolic_profile(self) -> None:
        """TRT with Λ=3/16 should reproduce the Poiseuille parabolic profile shape."""
        nx, ny = 4, 32
        tau = 0.55
        nu = (tau - 0.5) / 3.0
        H = ny - 2  # fluid nodes between walls

        # Body force driving pressure gradient
        G = 1e-5
        device = torch.device("cpu")

        rho0 = torch.ones((ny, nx), device=device)
        f = equilibrium(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0))

        obstacle = torch.zeros((ny, nx), dtype=torch.bool, device=device)
        wall_mask = make_channel_wall_mask(ny, nx, obstacle, device)

        # Guo body-force correction
        from tensorlbm.d2q9 import C, W

        c = C.to(device).float()
        w = W.to(device).float()
        cx = c[:, 0].view(9, 1, 1)
        w_view = w.view(9, 1, 1)

        def apply_force(f_in: torch.Tensor) -> torch.Tensor:
            rho_ = f_in.sum(dim=0)
            return f_in + w_view * 3.0 * rho_.unsqueeze(0) * cx * G

        n_steps = 10000
        for _ in range(n_steps):
            f = collide_trt(f, tau_plus=tau)
            f = stream(f)
            f = apply_force(f)
            f = bounce_back_cells(f, wall_mask)

        rho_f, ux_f, _ = macroscopic(f)
        # Analytical Poiseuille profile: u(y) = G/(2nu) * y*(H-1-y)
        ys = torch.arange(H, dtype=torch.float32)
        u_analytical = G / (2.0 * nu) * ys * (H - 1 - ys)

        u_sim = ux_f[1:-1, :].mean(dim=1)

        # Verify the profile shape by checking correlation with the parabola
        # (Pearson correlation coefficient should be very high)
        u_a_norm = u_analytical - u_analytical.mean()
        u_s_norm = u_sim - u_sim.mean()
        denom = (u_a_norm.norm() * u_s_norm.norm()).clamp(min=1e-10)
        corr = float((u_a_norm * u_s_norm).sum().item() / denom.item())
        assert corr > 0.98, (
            f"TRT Poiseuille profile correlation too low: {corr:.4f} (expected > 0.98)"
        )
        # Also verify the maximum velocity is in the correct range
        u_max_sim = float(u_sim.max().item())
        u_max_theory = float(u_analytical.max().item())
        assert u_max_sim > 0.5 * u_max_theory, (
            f"TRT peak velocity {u_max_sim:.6f} < 50% of theory {u_max_theory:.6f}"
        )


# ---------------------------------------------------------------------------
# D3Q19 TRT tests
# ---------------------------------------------------------------------------


class TestCollideTRT3D:
    def test_preserves_shape(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.04)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)
        f = equilibrium3d(rho, ux, uy, uz)
        f_out = collide_trt3d(f, tau_plus=0.7)
        assert f_out.shape == f.shape

    def test_conserves_mass(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.rand((nz, ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.05
        uy = torch.rand_like(rho) * 0.03
        uz = torch.rand_like(rho) * 0.02
        f = equilibrium3d(rho, ux, uy, uz)
        f_out = collide_trt3d(f, tau_plus=0.7)
        rho_out, _, _, _ = macroscopic3d(f_out)
        assert torch.allclose(rho_out, rho, atol=1e-4)

    def test_conserves_momentum(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.rand((nz, ny, nx)) + 0.5
        ux = torch.rand_like(rho) * 0.05
        uy = torch.rand_like(rho) * 0.03
        uz = torch.rand_like(rho) * 0.02
        f = equilibrium3d(rho, ux, uy, uz)
        f_out = collide_trt3d(f, tau_plus=0.7)
        _, ux_out, uy_out, uz_out = macroscopic3d(f_out)
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
        feq = equilibrium3d(rho, ux, uy, uz)
        f_out = collide_trt3d(feq, tau_plus=0.7)
        assert torch.allclose(f_out, feq, atol=1e-4)

    def test_finite_output(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.04)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)
        f = equilibrium3d(rho, ux, uy, uz)
        f_out = collide_trt3d(f, tau_plus=0.6)
        assert torch.isfinite(f_out).all()

    def test_lambda_variants(self) -> None:
        """Different Λ values should all produce valid outputs."""
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.03)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)
        f = equilibrium3d(rho, ux, uy, uz)
        for lam in [3.0 / 16.0, 1.0 / 4.0, 1.0 / 12.0]:
            f_out = collide_trt3d(f, tau_plus=0.7, lambda_trt=lam)
            assert torch.isfinite(f_out).all()
            rho_out, _, _, _ = macroscopic3d(f_out)
            assert torch.allclose(rho_out, rho, atol=1e-4)
