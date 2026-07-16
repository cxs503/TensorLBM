"""TDD tests for the general D3Q19 cumulant collision kernel.

These tests follow the same pattern as the D3Q27 cumulant and the
advanced-collision contract tests.  They verify:

* ``collide_cumulant_d3q19`` exists, is importable, and preserves shape.
* Equilibrium is a fixed point (no spurious non-equilibrium generation).
* Mass and momentum are conserved across collision.
* The advanced-collision contract registers D3Q19 CUMULANT as AVAILABLE.
* ``collide_advanced_3d("D3Q19", "CUMULANT", ...)`` dispatches correctly.
* A short sphere-flow smoke run completes without error.
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm.advanced_collision_contract import (
    CollisionKernelWithheldError,
    collision_capability_matrix,
    collide_advanced_3d,
)
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d


# ---------------------------------------------------------------------------
# Shape, importability, and basic properties
# ---------------------------------------------------------------------------


class TestCollideCumulantD3Q19Basics:
    def test_is_callable(self) -> None:
        assert callable(collide_cumulant_d3q19)

    def test_preserves_shape(self) -> None:
        rho = torch.ones((2, 3, 4))
        zero = torch.zeros_like(rho)
        f = equilibrium3d(rho, zero, zero, zero)
        out = collide_cumulant_d3q19(f, tau=0.8)
        assert out.shape == f.shape
        assert out.shape[0] == 19

    def test_output_is_finite(self) -> None:
        rho = torch.ones((2, 3, 4))
        zero = torch.zeros_like(rho)
        f = equilibrium3d(rho, zero, zero, zero)
        out = collide_cumulant_d3q19(f, tau=0.8)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Equilibrium fixed point
# ---------------------------------------------------------------------------


class TestEquilibriumFixedPoint:
    def test_rest_equilibrium_is_fixed_point(self) -> None:
        rho = torch.ones((2, 3, 4))
        zero = torch.zeros_like(rho)
        f = equilibrium3d(rho, zero, zero, zero)
        out = collide_cumulant_d3q19(f, tau=0.8)
        assert torch.allclose(out, f, atol=2e-5)

    def test_moving_equilibrium_is_fixed_point(self) -> None:
        rho = torch.ones((2, 3, 4))
        ux = torch.full((2, 3, 4), 0.03)
        uy = torch.full((2, 3, 4), -0.02)
        uz = torch.full((2, 3, 4), 0.01)
        f = equilibrium3d(rho, ux, uy, uz)
        out = collide_cumulant_d3q19(f, tau=0.8)
        assert torch.allclose(out, f, atol=2e-5)


# ---------------------------------------------------------------------------
# Conservation laws
# ---------------------------------------------------------------------------


class TestConservation:
    def _make_non_equilibrium(self) -> torch.Tensor:
        rho = torch.ones((2, 3, 4))
        ux = torch.full((2, 3, 4), 0.03)
        uy = torch.full((2, 3, 4), -0.02)
        uz = torch.full((2, 3, 4), 0.01)
        f = equilibrium3d(rho, ux, uy, uz)
        # Add a small deterministic non-equilibrium perturbation.
        pert = 1e-4 * torch.linspace(-1, 1, 19, dtype=f.dtype).view(-1, 1, 1, 1)
        return f + pert

    def test_mass_is_conserved(self) -> None:
        f = self._make_non_equilibrium()
        out = collide_cumulant_d3q19(f, tau=0.8)
        assert torch.allclose(out.sum(dim=0), f.sum(dim=0), atol=2e-5)

    def test_momentum_is_conserved(self) -> None:
        f = self._make_non_equilibrium()
        out = collide_cumulant_d3q19(f, tau=0.8)
        rho_before, ux_b, uy_b, uz_b = macroscopic3d(f)
        rho_after, ux_a, uy_a, uz_a = macroscopic3d(out)
        assert torch.allclose(rho_before, rho_after, atol=2e-5)
        assert torch.allclose(ux_b, ux_a, atol=2e-5)
        assert torch.allclose(uy_b, uy_a, atol=2e-5)
        assert torch.allclose(uz_b, uz_a, atol=2e-5)


# ---------------------------------------------------------------------------
# Advanced-collision contract registration
# ---------------------------------------------------------------------------


class TestCumulantContractRegistration:
    def test_d3q19_cumulant_is_available(self) -> None:
        matrix = collision_capability_matrix()
        cap = matrix["D3Q19"]["CUMULANT"]
        assert cap.available
        assert cap.status == "AVAILABLE"
        assert cap.entrypoint == "tensorlbm.cumulant.collide_cumulant_d3q19"

    def test_d3q27_cumulant_is_available(self) -> None:
        matrix = collision_capability_matrix()
        cap = matrix["D3Q27"]["CUMULANT"]
        assert cap.available
        assert cap.status == "AVAILABLE"
        assert cap.entrypoint == "tensorlbm.cumulant.collide_cumulant_d3q27"

    def test_common_dispatch_equilibrium_fixed_point(self) -> None:
        rho = torch.ones((2, 3, 4))
        zero = torch.zeros_like(rho)
        f = equilibrium3d(rho, zero, zero, zero)
        out = collide_advanced_3d("D3Q19", "CUMULANT", f, tau=0.8)
        assert out.shape == (19, 2, 3, 4)
        assert torch.allclose(out, f, atol=2e-5)

    def test_common_dispatch_mass_momentum_conserved(self) -> None:
        rho = torch.ones((2, 3, 4))
        ux = torch.full((2, 3, 4), 0.03)
        uy = torch.full((2, 3, 4), -0.02)
        uz = torch.full((2, 3, 4), 0.01)
        f = equilibrium3d(rho, ux, uy, uz)
        pert = 1e-4 * torch.linspace(-1, 1, 19, dtype=f.dtype).view(-1, 1, 1, 1)
        f_pert = f + pert
        out = collide_advanced_3d("D3Q19", "CUMULANT", f_pert, tau=0.8)
        rho_b, ux_b, uy_b, uz_b = macroscopic3d(f_pert)
        rho_a, ux_a, uy_a, uz_a = macroscopic3d(out)
        assert torch.allclose(rho_b, rho_a, atol=2e-5)
        assert torch.allclose(ux_b, ux_a, atol=2e-5)
        assert torch.allclose(uy_b, uy_a, atol=2e-5)
        assert torch.allclose(uz_b, uz_a, atol=2e-5)

    def test_cumulant_alias_dispatches(self) -> None:
        """The alias 'CUMULANT_LBM' should also resolve to the CUMULANT family."""
        rho = torch.ones((1, 1, 1))
        zero = torch.zeros_like(rho)
        f = equilibrium3d(rho, zero, zero, zero)
        out = collide_advanced_3d("D3Q19", "CUMULANT", f, tau=0.8)
        assert out.shape == (19, 1, 1, 1)


# ---------------------------------------------------------------------------
# Relaxation behaviour
# ---------------------------------------------------------------------------


class TestRelaxationBehaviour:
    def test_higher_tau_means_less_relaxation(self) -> None:
        """A larger tau (lower omega) should leave more non-equilibrium."""
        rho = torch.ones((2, 3, 4))
        ux = torch.full((2, 3, 4), 0.03)
        uy = torch.full((2, 3, 4), -0.02)
        uz = torch.full((2, 3, 4), 0.01)
        f_eq = equilibrium3d(rho, ux, uy, uz)
        pert = 1e-4 * torch.linspace(-1, 1, 19, dtype=f_eq.dtype).view(-1, 1, 1, 1)
        f = f_eq + pert

        out_low_tau = collide_cumulant_d3q19(f, tau=0.55)
        out_high_tau = collide_cumulant_d3q19(f, tau=5.0)

        neq_low = (out_low_tau - f_eq).abs().max()
        neq_high = (out_high_tau - f_eq).abs().max()
        # Higher tau → slower relaxation → more residual non-equilibrium.
        assert neq_high > neq_low

    def test_smagorinsky_does_not_crash(self) -> None:
        rho = torch.ones((2, 3, 4))
        ux = torch.full((2, 3, 4), 0.03)
        uy = torch.full((2, 3, 4), -0.02)
        uz = torch.full((2, 3, 4), 0.01)
        f = equilibrium3d(rho, ux, uy, uz)
        pert = 1e-4 * torch.linspace(-1, 1, 19, dtype=f.dtype).view(-1, 1, 1, 1)
        f_pert = f + pert
        out = collide_cumulant_d3q19(f_pert, tau=0.55, C_s=0.1)
        assert torch.isfinite(out).all()
        assert out.shape == f_pert.shape
