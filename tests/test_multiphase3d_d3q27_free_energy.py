"""Tests for D3Q27 free-energy / Cahn-Hilliard phase-field collision operators.

Verifies:
    - init_free_energy_g_3d_27: output shape, zero-velocity equilibrium, mass
    - free_energy_step_3d_27: output shapes, mass conservation, finiteness,
      reference-formula match, uniform-phase stability, SGS option
    - Static-droplet stability: a tanh-profile droplet remains finite and
      bounded over many collision+stream steps
    - D3Q27 vs D3Q19 structural parity: same algorithmic pattern, 27 directions
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm import (
    equilibrium27,
    free_energy_step_3d_27,
    init_free_energy_g_3d_27,
    macroscopic27,
    stream27,
)
from tensorlbm.d3q27 import C as C27
from tensorlbm.d3q27 import W as W27

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fe_state_27(
    nz: int = 5, ny: int = 6, nx: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equilibrium f and a random-phase g for D3Q27 free-energy tests."""
    rho = torch.ones((nz, ny, nx), device=DEVICE)
    zero = torch.zeros_like(rho)
    f = equilibrium27(rho, zero, zero, zero)
    phi = torch.rand_like(rho) * 2.0 - 1.0
    g = init_free_energy_g_3d_27(phi)
    return f, g


# ---------------------------------------------------------------------------
# init_free_energy_g_3d_27
# ---------------------------------------------------------------------------

class TestInitFreeEnergyG27:
    def test_output_shape(self) -> None:
        nz, ny, nx = 5, 6, 8
        phi = torch.ones((nz, ny, nx), device=DEVICE)
        g = init_free_energy_g_3d_27(phi)
        assert g.shape == (27, nz, ny, nx)

    def test_zero_velocity_equilibrium(self) -> None:
        """At rest, g_i = w_i * phi (equilibrium with u=0)."""
        nz, ny, nx = 4, 5, 7
        phi = torch.rand((nz, ny, nx), device=DEVICE) * 2.0 - 1.0
        g = init_free_energy_g_3d_27(phi)
        w = W27.to(DEVICE).view(27, 1, 1, 1)
        expected = w * phi.unsqueeze(0)
        assert torch.allclose(g, expected, atol=1e-6)

    def test_mass_is_phi(self) -> None:
        """Zeroth moment of g equals phi."""
        nz, ny, nx = 4, 5, 7
        phi = torch.rand((nz, ny, nx), device=DEVICE) * 2.0 - 1.0
        g = init_free_energy_g_3d_27(phi)
        assert torch.allclose(g.sum(dim=0), phi, atol=1e-6)

    def test_with_velocity(self) -> None:
        """Non-zero velocity shifts the equilibrium."""
        nz, ny, nx = 4, 5, 7
        phi = torch.ones((nz, ny, nx), device=DEVICE)
        ux = torch.full((nz, ny, nx), 0.01, device=DEVICE)
        uy = torch.zeros_like(ux)
        uz = torch.zeros_like(ux)
        g = init_free_energy_g_3d_27(phi, ux, uy, uz)
        c = C27.to(DEVICE).float()
        w = W27.to(DEVICE).view(27, 1, 1, 1)
        cx = c[:, 0].view(27, 1, 1, 1)
        cu = cx * ux.unsqueeze(0)
        u_sq = (ux ** 2).unsqueeze(0)
        expected = w * phi.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * u_sq)
        assert torch.allclose(g, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# free_energy_step_3d_27 — basic properties
# ---------------------------------------------------------------------------

class TestFreeEnergyStep27Basic:
    def test_output_shapes(self) -> None:
        f, g = _make_fe_state_27()
        f_out, g_out = free_energy_step_3d_27(f, g)
        assert f_out.shape == f.shape
        assert g_out.shape == g.shape
        assert f_out.shape[0] == 27
        assert g_out.shape[0] == 27

    def test_finite_values(self) -> None:
        f, g = _make_fe_state_27()
        f_out, g_out = free_energy_step_3d_27(f, g)
        assert torch.isfinite(f_out).all()
        assert torch.isfinite(g_out).all()

    def test_f_mass_conservation(self) -> None:
        """Total mass of f is conserved by collision (no streaming)."""
        f, g = _make_fe_state_27()
        mass_before = f.sum()
        f_out, _ = free_energy_step_3d_27(f, g)
        assert torch.allclose(f_out.sum(), mass_before, atol=1e-3)

    def test_g_mass_conservation(self) -> None:
        """Sum of g (order-parameter integral) is conserved by collision."""
        f, g = _make_fe_state_27()
        g_sum_before = g.sum()
        _, g_out = free_energy_step_3d_27(f, g)
        assert torch.allclose(g_out.sum(), g_sum_before, atol=1e-3)

    def test_streaming_preserves_mass(self) -> None:
        f, g = _make_fe_state_27()
        mass_before = f.sum()
        f, g = free_energy_step_3d_27(f, g)
        f = stream27(f)
        g = stream27(g)
        assert torch.allclose(f.sum(), mass_before, atol=1e-3)


# ---------------------------------------------------------------------------
# free_energy_step_3d_27 — reference formula match
# ---------------------------------------------------------------------------

class TestFreeEnergyStep27ReferenceFormula:
    def test_matches_reference_formula(self) -> None:
        """The D3Q27 step must match the explicit reference formula."""
        torch.manual_seed(7)
        nz, ny, nx = 3, 4, 5
        rho = torch.ones((nz, ny, nx), dtype=torch.float32)
        zero = torch.zeros_like(rho)
        f = equilibrium27(rho, zero, zero, zero)
        rho, ux, uy, uz = macroscopic27(f)
        phi = torch.rand_like(rho) * 2.0 - 1.0
        g = init_free_energy_g_3d_27(phi)
        # The production step obtains the order parameter from g's zeroth moment.
        phi = g.sum(dim=0)
        kwargs = dict(
            tau_f=0.9, tau_g=0.8, A=0.11, B=0.13, kappa=0.02,
            Gamma=0.4, gx=0.001, gy=-0.002, gz=0.003,
        )
        got_f, got_g = free_energy_step_3d_27(f, g, **kwargs)

        # Chemical potential (periodic 7-point Laplacian)
        mu = -kwargs["A"] * phi + kwargs["B"] * phi ** 3 - kwargs["kappa"] * (
            torch.roll(phi, 1, 2) + torch.roll(phi, -1, 2)
            + torch.roll(phi, 1, 1) + torch.roll(phi, -1, 1)
            + torch.roll(phi, 1, 0) + torch.roll(phi, -1, 0)
            - 6.0 * phi
        )
        grad_x = 0.5 * (torch.roll(mu, -1, 2) - torch.roll(mu, 1, 2))
        grad_y = 0.5 * (torch.roll(mu, -1, 1) - torch.roll(mu, 1, 1))
        grad_z = 0.5 * (torch.roll(mu, -1, 0) - torch.roll(mu, 1, 0))

        rho_s = rho.clamp(min=1e-12)
        expected_feq = equilibrium27(
            rho,
            ux + kwargs["tau_f"] * (-phi * grad_x + rho * kwargs["gx"]) / rho_s,
            uy + kwargs["tau_f"] * (-phi * grad_y + rho * kwargs["gy"]) / rho_s,
            uz + kwargs["tau_f"] * (-phi * grad_z + rho * kwargs["gz"]) / rho_s,
        )
        expected_f = f - (f - expected_feq) / kwargs["tau_f"]

        cv = C27.to(phi.device).float()
        csq = (cv[:, 0] ** 2 + cv[:, 1] ** 2 + cv[:, 2] ** 2).view(27, 1, 1, 1)
        w = W27.to(phi.device).view(27, 1, 1, 1)
        cu = (
            cv[:, 0].view(27, 1, 1, 1) * ux
            + cv[:, 1].view(27, 1, 1, 1) * uy
            + cv[:, 2].view(27, 1, 1, 1) * uz
        )
        u_sq = (ux ** 2 + uy ** 2 + uz ** 2).unsqueeze(0)
        expected_g_eq = (
            w * phi.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * u_sq)
            + w * kwargs["Gamma"] * (csq / (1.0 / 3.0) - 3.0) * mu.unsqueeze(0)
        )
        expected_g = g - (g - expected_g_eq) / kwargs["tau_g"]

        assert torch.equal(got_f, expected_f)
        assert torch.equal(got_g, expected_g)


# ---------------------------------------------------------------------------
# free_energy_step_3d_27 — uniform phase stability
# ---------------------------------------------------------------------------

class TestFreeEnergyStep27UniformPhase:
    def test_uniform_phi_stable(self) -> None:
        """Uniform phi → zero Korteweg force → no blow-up over 20 steps."""
        nz, ny, nx = 6, 6, 6
        rho = torch.ones((nz, ny, nx), device=DEVICE)
        zero = torch.zeros_like(rho)
        f = equilibrium27(rho, zero, zero, zero)
        phi = torch.ones_like(rho)  # uniform +1 phase
        g = init_free_energy_g_3d_27(phi)
        for _ in range(20):
            f, g = free_energy_step_3d_27(f, g)
            f = stream27(f)
            g = stream27(g)
        assert torch.isfinite(f).all()
        assert torch.isfinite(g).all()
        rho_out, _, _, _ = macroscopic27(f)
        assert torch.isfinite(rho_out).all()

    def test_uniform_phi_no_drift(self) -> None:
        """Uniform phi should not drift from +1 over 20 steps."""
        nz, ny, nx = 6, 6, 6
        rho = torch.ones((nz, ny, nx), device=DEVICE)
        zero = torch.zeros_like(rho)
        f = equilibrium27(rho, zero, zero, zero)
        phi = torch.ones_like(rho)
        g = init_free_energy_g_3d_27(phi)
        for _ in range(20):
            f, g = free_energy_step_3d_27(f, g)
            f = stream27(f)
            g = stream27(g)
        phi_out = g.sum(dim=0)
        assert torch.allclose(phi_out, torch.ones_like(phi_out), atol=0.05)


# ---------------------------------------------------------------------------
# free_energy_step_3d_27 — SGS model option
# ---------------------------------------------------------------------------

class TestFreeEnergyStep27SGS:
    def test_smagorinsky_finite(self) -> None:
        f, g = _make_fe_state_27()
        f_out, g_out = free_energy_step_3d_27(f, g, C_s=0.1, sgs_model="smagorinsky")
        assert torch.isfinite(f_out).all()
        assert torch.isfinite(g_out).all()

    def test_wale_finite(self) -> None:
        f, g = _make_fe_state_27()
        f_out, g_out = free_energy_step_3d_27(f, g, C_s=0.1, sgs_model="wale")
        assert torch.isfinite(f_out).all()
        assert torch.isfinite(g_out).all()

    def test_vreman_finite(self) -> None:
        f, g = _make_fe_state_27()
        f_out, g_out = free_energy_step_3d_27(f, g, C_s=0.1, sgs_model="vreman")
        assert torch.isfinite(f_out).all()
        assert torch.isfinite(g_out).all()

    def test_invalid_sgs_model_raises(self) -> None:
        f, g = _make_fe_state_27()
        with pytest.raises(ValueError, match="sgs_model"):
            free_energy_step_3d_27(f, g, C_s=0.1, sgs_model="bad_model")


# ---------------------------------------------------------------------------
# free_energy_step_3d_27 — buoyancy option
# ---------------------------------------------------------------------------

class TestFreeEnergyStep27Buoyancy:
    def test_buoyancy_finite(self) -> None:
        """Boussinesq buoyancy with rho_heavy/rho_light should be finite."""
        f, g = _make_fe_state_27()
        f_out, g_out = free_energy_step_3d_27(
            f, g, gz=-1e-4, rho_heavy=1.0, rho_light=0.5,
        )
        assert torch.isfinite(f_out).all()
        assert torch.isfinite(g_out).all()


# ---------------------------------------------------------------------------
# Static droplet stability (D3Q27 free-energy)
# ---------------------------------------------------------------------------

class TestStaticDroplet27FreeEnergy:
    def test_droplet_remains_finite(self) -> None:
        """A tanh-profile droplet should remain finite for 50 steps."""
        from tensorlbm.phasefield.static_droplet import initialize_static_droplet

        nz, ny, nx = 24, 24, 24
        # Match interface_width to the free-energy parameters:
        #   width = sqrt(2*kappa/A) = sqrt(2*0.02/0.04) = 1.0
        phi = initialize_static_droplet(
            (nz, ny, nx), radius=6.0, interface_width=1.0,
        )

        rho = torch.ones_like(phi)
        zero = torch.zeros_like(phi)
        f = equilibrium27(rho, zero, zero, zero)
        g = init_free_energy_g_3d_27(phi)

        for _ in range(50):
            f, g = free_energy_step_3d_27(
                f, g, tau_f=1.0, tau_g=0.7, A=0.04, B=0.04, kappa=0.02, Gamma=0.5,
            )
            f = stream27(f)
            g = stream27(g)

        assert torch.isfinite(f).all()
        assert torch.isfinite(g).all()
        rho_out, _, _, _ = macroscopic27(f)
        phi_out = g.sum(dim=0)
        assert torch.isfinite(rho_out).all()
        assert torch.isfinite(phi_out).all()
        assert phi_out.max() > 0.5  # droplet core still present
        assert phi_out.min() < -0.5  # bulk phase still present

    def test_droplet_mass_drift_bounded(self) -> None:
        """Total phi integral should not drift more than 10% over 50 steps."""
        from tensorlbm.phasefield.static_droplet import initialize_static_droplet

        nz, ny, nx = 20, 20, 20
        phi = initialize_static_droplet(
            (nz, ny, nx), radius=5.0, interface_width=1.0,
        )

        rho = torch.ones_like(phi)
        zero = torch.zeros_like(phi)
        f = equilibrium27(rho, zero, zero, zero)
        g = init_free_energy_g_3d_27(phi)

        phi0 = g.sum()
        for _ in range(50):
            f, g = free_energy_step_3d_27(
                f, g, tau_f=1.0, tau_g=0.7, A=0.04, B=0.04, kappa=0.02, Gamma=0.5,
            )
            f = stream27(f)
            g = stream27(g)

        phi1 = g.sum()
        drift = abs(phi1 - phi0) / abs(phi0)
        assert drift < 0.1, f"phi drift {drift:.4f} exceeds 10%"


# ---------------------------------------------------------------------------
# D3Q27 vs D3Q19 structural parity
# ---------------------------------------------------------------------------

class TestD3Q27VsD3Q19Parity:
    def test_both_lattices_zero_force_for_uniform_phi(self) -> None:
        """Both D3Q19 and D3Q27 produce zero Korteweg force for uniform phi."""
        from tensorlbm.phasefield.free_energy import (
            DoubleWellFreeEnergy,
            force_minus_phi_grad_mu,
        )

        phi = torch.full((5, 6, 7), 0.5, dtype=torch.float32)
        model = DoubleWellFreeEnergy(A=0.1, B=0.1, kappa=0.02)
        mu = model.chemical_potential(phi, boundary="periodic")
        fx, fy, fz = force_minus_phi_grad_mu(phi, mu, boundary="periodic")
        # The force is lattice-agnostic, so both lattices use the same operators
        assert torch.allclose(fx, torch.zeros_like(fx), atol=1e-6)
        assert torch.allclose(fy, torch.zeros_like(fy), atol=1e-6)
        assert torch.allclose(fz, torch.zeros_like(fz), atol=1e-6)

    def test_diff_factor_sums_to_zero(self) -> None:
        """The diffusion factor Σ w_i (|c|²/cs² - 3) = 0 for both lattices."""
        cs2 = 1.0 / 3.0
        # D3Q27
        c27 = C27.float()
        w27 = W27.float()
        csq27 = c27[:, 0] ** 2 + c27[:, 1] ** 2 + c27[:, 2] ** 2
        diff27 = w27 * (csq27 / cs2 - 3.0)
        assert abs(diff27.sum().item()) < 1e-6

        # D3Q19
        from tensorlbm.d3q19 import C as C19
        from tensorlbm.d3q19 import W as W19
        c19 = C19.float()
        w19 = W19.float()
        csq19 = c19[:, 0] ** 2 + c19[:, 1] ** 2 + c19[:, 2] ** 2
        diff19 = w19 * (csq19 / cs2 - 3.0)
        assert abs(diff19.sum().item()) < 1e-6
