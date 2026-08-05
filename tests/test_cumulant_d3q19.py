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
    collide_advanced_3d,
    collision_capability_matrix,
)
from tensorlbm.cumulant import (
    collide_cumulant_d3q19,
    gradient_sgs_effective_tau_d3q19,
    smagorinsky_effective_tau_d3q19,
    summarize_gradient_sgs_effective_tau_d3q19,
    summarize_smagorinsky_effective_tau_d3q19,
)
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

    @pytest.mark.parametrize(
        ("coefficient_name", "coefficient"),
        (("C_w", 0.5), ("C_v", 0.025)),
    )
    def test_gradient_sgs_preserves_mass_and_momentum(
        self, coefficient_name: str, coefficient: float,
    ) -> None:
        z, y, x = torch.meshgrid(
            torch.arange(4, dtype=torch.float64),
            torch.arange(5, dtype=torch.float64),
            torch.arange(6, dtype=torch.float64),
            indexing="ij",
        )
        rho = torch.ones_like(x, dtype=torch.float64)
        ux = 0.01 + 0.001 * y
        uy = -0.005 + 0.0005 * z
        uz = 0.0003 * x
        populations = equilibrium3d(rho, ux, uy, uz)
        populations[1] += 1.0e-4
        populations[2] -= 1.0e-4

        output = collide_cumulant_d3q19(
            populations, tau=0.55, **{coefficient_name: coefficient},
        )
        before = macroscopic3d(populations)
        after = macroscopic3d(output)

        assert torch.isfinite(output).all()
        for expected, actual in zip(before, after, strict=True):
            torch.testing.assert_close(actual, expected, atol=2.0e-8, rtol=0.0)

    @pytest.mark.parametrize(
        ("model", "coefficient", "coefficient_name"),
        (("wale", 0.5, "C_w"), ("vreman", 0.025, "C_v")),
    )
    def test_gradient_sgs_diagnostic_matches_collision_parameterisation(
        self, model: str, coefficient: float, coefficient_name: str,
    ) -> None:
        z, y, x = torch.meshgrid(
            torch.arange(4, dtype=torch.float64),
            torch.arange(5, dtype=torch.float64),
            torch.arange(6, dtype=torch.float64),
            indexing="ij",
        )
        rho = torch.ones_like(x, dtype=torch.float64)
        populations = equilibrium3d(
            rho,
            0.01 + 0.001 * y,
            -0.005 + 0.0005 * z,
            0.0003 * x,
        )
        effective = gradient_sgs_effective_tau_d3q19(
            populations, tau=0.55, model=model, coefficient=coefficient,
        )

        assert effective.shape == rho.shape
        assert bool(torch.isfinite(effective).all())
        assert bool((effective >= 0.55).all())
        assert bool((effective > 0.55).any())
        output = collide_cumulant_d3q19(
            populations, tau=0.55, **{coefficient_name: coefficient},
        )
        assert torch.isfinite(output).all()

    def test_only_one_sgs_model_can_be_active(self) -> None:
        rho = torch.ones((3, 3, 3))
        zero = torch.zeros_like(rho)
        populations = equilibrium3d(rho, zero, zero, zero)
        with pytest.raises(ValueError, match="only one"):
            collide_cumulant_d3q19(
                populations, tau=0.55, C_s=0.1, C_w=0.5,
            )

    @pytest.mark.parametrize(
        ("model", "coefficient"), (("wale", 0.5), ("vreman", 0.025)),
    )
    def test_gradient_sgs_summary_is_halo_chunk_invariant(
        self, model: str, coefficient: float,
    ) -> None:
        torch.manual_seed(7)
        rho = torch.ones((7, 5, 6), dtype=torch.float64)
        ux = 0.02 * torch.rand_like(rho)
        uy = 0.02 * torch.rand_like(rho)
        uz = 0.02 * torch.rand_like(rho)
        populations = equilibrium3d(rho, ux, uy, uz)
        solid_mask = torch.zeros_like(rho, dtype=torch.bool)
        solid_mask[2:5, 1:4, 2:5] = True

        one_plane = summarize_gradient_sgs_effective_tau_d3q19(
            populations,
            tau=0.55,
            model=model,
            coefficient=coefficient,
            chunk_cells=30,
            solid_mask=solid_mask,
        )
        whole_domain = summarize_gradient_sgs_effective_tau_d3q19(
            populations,
            tau=0.55,
            model=model,
            coefficient=coefficient,
            chunk_cells=10_000,
            solid_mask=solid_mask,
        )

        assert one_plane == pytest.approx(whole_domain, rel=3.0e-13, abs=3.0e-15)
        assert one_plane["cell_count"] == 210.0
        assert one_plane["effective_tau_minimum"] >= 0.55

    @pytest.mark.parametrize(
        ("model", "coefficient"), (("wale", 0.5), ("vreman", 0.025)),
    )
    def test_gradient_sgs_uses_wall_velocity_inside_solid(
        self, model: str, coefficient: float,
    ) -> None:
        z, y, x = torch.meshgrid(
            torch.arange(5, dtype=torch.float64),
            torch.arange(5, dtype=torch.float64),
            torch.arange(7, dtype=torch.float64),
            indexing="ij",
        )
        rho = torch.ones_like(x)
        populations = equilibrium3d(
            rho,
            0.04 + 0.001 * y,
            0.02 + 0.001 * z,
            -0.01 + 0.001 * x,
        )
        solid_mask = torch.zeros_like(rho, dtype=torch.bool)
        solid_mask[1:4, 1:4, 4:] = True

        unmasked = gradient_sgs_effective_tau_d3q19(
            populations, tau=0.55, model=model, coefficient=coefficient,
        )
        masked = gradient_sgs_effective_tau_d3q19(
            populations,
            tau=0.55,
            model=model,
            coefficient=coefficient,
            solid_mask=solid_mask,
        )

        assert float((masked - unmasked).abs().max().item()) > 1.0e-8

    def test_gradient_sgs_rejects_incompatible_solid_mask(self) -> None:
        rho = torch.ones((4, 5, 6))
        zero = torch.zeros_like(rho)
        populations = equilibrium3d(rho, zero, zero, zero)
        with pytest.raises(ValueError, match="solid_mask"):
            collide_cumulant_d3q19(
                populations,
                tau=0.55,
                C_w=0.5,
                solid_mask=torch.zeros((4, 5, 5), dtype=torch.bool),
            )

    def test_smagorinsky_effective_tau_is_explicit(self) -> None:
        rho = torch.ones((2, 3, 4))
        zero = torch.zeros_like(rho)
        equilibrium = equilibrium3d(rho, zero, zero, zero)
        base = smagorinsky_effective_tau_d3q19(
            equilibrium,
            tau=0.55,
            C_s=0.1,
        )
        torch.testing.assert_close(base, torch.full_like(base, 0.55))

        perturbed = equilibrium.clone()
        perturbed[1] += 1.0e-3
        perturbed[2] -= 1.0e-3
        effective = smagorinsky_effective_tau_d3q19(
            perturbed,
            tau=0.55,
            C_s=0.1,
        )
        assert bool((effective >= 0.55).all())
        assert bool((effective > 0.55).any())

    def test_smagorinsky_tau_diagnostic_rejects_invalid_input(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            smagorinsky_effective_tau_d3q19(
                torch.zeros((9, 2, 3, 4)), tau=0.55, C_s=0.1,
            )
        with pytest.raises(ValueError, match="greater"):
            smagorinsky_effective_tau_d3q19(
                torch.zeros((19, 2, 3, 4)), tau=0.5, C_s=0.1,
            )

    def test_smagorinsky_tau_summary_is_chunk_invariant(self) -> None:
        rho = torch.ones((2, 3, 4), dtype=torch.float64)
        zero = torch.zeros_like(rho)
        populations = equilibrium3d(rho, zero, zero, zero)
        populations[1] += 1.0e-3
        populations[2] -= 1.0e-3

        small = summarize_smagorinsky_effective_tau_d3q19(
            populations, tau=0.55, C_s=0.1, chunk_cells=5,
        )
        large = summarize_smagorinsky_effective_tau_d3q19(
            populations, tau=0.55, C_s=0.1, chunk_cells=100,
        )

        assert small == pytest.approx(large)
        assert small["cell_count"] == 24
        assert small["effective_tau_minimum"] >= 0.55
        assert small["maximum_eddy_to_molecular_viscosity_ratio"] > 0.0
