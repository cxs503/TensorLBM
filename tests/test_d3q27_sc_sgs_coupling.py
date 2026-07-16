"""Tests for D3Q27 Shan-Chen + SGS (Smagorinsky/WALE/Vreman) coupling.

Verifies that the optional ``sgs_model`` parameter added to
``collide_sc_single_component_27`` and ``collide_sc_two_component_27``
follows the same selection pattern as ``free_energy_step_3d`` in
:mod:`tensorlbm.multiphase3d`.

Test matrix
-----------
- Default (C_s=0):  output identical to the original no-SGS collision.
- Smagorinsky:      tau_eff >= tau, output differs from no-SGS.
- WALE:             zero eddy viscosity for uniform flow → matches no-SGS;
                    non-zero for sheared flow → differs from no-SGS.
- Vreman:           same uniform/sheared contract as WALE.
- Invalid model:    raises ValueError.
- Conservation:     mass preserved (single-component); shape/finite checks.
"""
from __future__ import annotations

import copy

import pytest
import torch

from tensorlbm import collide_sc_single_component_27, collide_sc_two_component_27
from tensorlbm.d3q27 import equilibrium27, macroscopic27
from tensorlbm.turbulence import (
    _neq_stress_norm_27,
    _nu_t_to_tau_eff,
    _smagorinsky_tau,
    _vreman_nu_t_3d,
    _wale_nu_t_3d,
)

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f27_uniform(nz: int = 4, ny: int = 6, nx: int = 8, u_mag: float = 0.04) -> torch.Tensor:
    """Equilibrium D3Q27 distribution with random but *uniform* velocity."""
    rho = torch.rand((nz, ny, nx)) + 0.5
    ux = torch.full_like(rho, u_mag)
    uy = torch.full_like(rho, 0.0)
    uz = torch.full_like(rho, 0.0)
    return equilibrium27(rho, ux, uy, uz)


def _f27_sheared(nz: int = 6, ny: int = 8, nx: int = 10, u_mag: float = 0.1) -> torch.Tensor:
    """Equilibrium D3Q27 distribution with a *turbulent-like* 3-D velocity field.

    A random velocity field with spatial variation in all three directions
    produces non-zero velocity-gradient invariants so that both WALE and
    Vreman yield a non-zero eddy viscosity (unlike pure 2-D shear, for which
    both models correctly give ν_t = 0).
    """
    torch.manual_seed(42)
    rho = torch.rand((nz, ny, nx)) + 0.5
    ux = torch.rand((nz, ny, nx)) * u_mag * 2 - u_mag
    uy = torch.rand((nz, ny, nx)) * u_mag * 2 - u_mag
    uz = torch.rand((nz, ny, nx)) * u_mag * 2 - u_mag
    return equilibrium27(rho, ux, uy, uz)


def _f27_random(nz: int = 4, ny: int = 6, nx: int = 8, u_mag: float = 0.04) -> torch.Tensor:
    """Equilibrium D3Q27 distribution with random velocity (non-uniform)."""
    rho = torch.rand((nz, ny, nx)) + 0.5
    ux = torch.rand_like(rho) * u_mag
    uy = torch.rand_like(rho) * u_mag
    uz = torch.rand_like(rho) * u_mag
    return equilibrium27(rho, ux, uy, uz)


# ===========================================================================
# Single-component: collide_sc_single_component_27
# ===========================================================================

class TestSingleComponentDefault:
    """Default behaviour (C_s=0) must be identical to the original collision."""

    def test_default_no_sgs_matches_explicit_zero(self) -> None:
        f = _f27_random()
        out_default = collide_sc_single_component_27(f, G=-4.0, tau=1.0)
        out_explicit = collide_sc_single_component_27(
            f, G=-4.0, tau=1.0, C_s=0.0, sgs_model="smagorinsky",
        )
        assert torch.allclose(out_default, out_explicit, atol=0, rtol=0)

    def test_default_preserves_shape(self) -> None:
        f = _f27_random()
        out = collide_sc_single_component_27(f, G=-4.0, tau=1.0)
        assert out.shape == f.shape

    def test_default_finite(self) -> None:
        f = _f27_random()
        out = collide_sc_single_component_27(f, G=-4.0, tau=1.0)
        assert torch.isfinite(out).all()

    def test_default_mass_conserved(self) -> None:
        f = _f27_random()
        out = collide_sc_single_component_27(f, G=-4.0, tau=1.0)
        assert torch.allclose(f.sum(), out.sum(), rtol=1e-4, atol=1e-4)


# ===========================================================================
# Single-component: Smagorinsky SGS
# ===========================================================================

class TestSingleComponentSmagorinsky:
    """Smagorinsky SGS coupling for D3Q27 single-component SC."""

    def test_sgs_differs_from_no_sgs(self) -> None:
        """With C_s > 0 the collision must differ from the no-SGS baseline."""
        f = _f27_random()
        tau = 1.0
        out_no_sgs = collide_sc_single_component_27(f, G=-4.0, tau=tau, C_s=0.0)
        out_sgs = collide_sc_single_component_27(
            f, G=-4.0, tau=tau, C_s=0.1, sgs_model="smagorinsky",
        )
        assert not torch.allclose(out_no_sgs, out_sgs, atol=1e-8)

    def test_sgs_tau_eff_ge_tau(self) -> None:
        """Effective relaxation time must be >= baseline tau (more dissipation)."""
        f = _f27_random()
        tau = 1.0
        rho, ux, uy, uz = macroscopic27(f)
        psi = torch.exp(rho)  # psi_exp default
        # Recompute feq the same way the collision does
        from tensorlbm.multiphase3d_d3q27 import _sc_neighbor_weighted_sum_27
        sx, sy, sz = _sc_neighbor_weighted_sum_27(psi, None)
        rho_s = torch.clamp(rho, min=1e-12)
        Fx = -(-4.0) * psi * sx
        Fy = -(-4.0) * psi * sy
        Fz = -(-4.0) * psi * sz
        feq = equilibrium27(
            rho,
            ux + tau * Fx / rho_s,
            uy + tau * Fy / rho_s,
            uz + tau * Fz / rho_s,
        )
        pi_norm = _neq_stress_norm_27(f - feq)
        tau_eff = _smagorinsky_tau(tau, pi_norm, rho_s, 0.1)
        assert (tau_eff >= tau - 1e-12).all()

    def test_shape_and_finite(self) -> None:
        f = _f27_random()
        out = collide_sc_single_component_27(
            f, G=-4.0, tau=1.0, C_s=0.1, sgs_model="smagorinsky",
        )
        assert out.shape == f.shape
        assert torch.isfinite(out).all()

    def test_mass_conserved(self) -> None:
        f = _f27_random()
        out = collide_sc_single_component_27(
            f, G=-4.0, tau=1.0, C_s=0.1, sgs_model="smagorinsky",
        )
        assert torch.allclose(f.sum(), out.sum(), rtol=1e-4, atol=1e-4)


# ===========================================================================
# Single-component: WALE SGS
# ===========================================================================

class TestSingleComponentWALE:
    """WALE SGS coupling for D3Q27 single-component SC."""

    def test_uniform_flow_matches_no_sgs(self) -> None:
        """Uniform velocity → zero velocity gradients → ν_t = 0 → matches no-SGS."""
        f = _f27_uniform()
        tau = 1.0
        out_no_sgs = collide_sc_single_component_27(f, G=-4.0, tau=tau, C_s=0.0)
        out_wale = collide_sc_single_component_27(
            f, G=-4.0, tau=tau, C_s=0.5, sgs_model="wale",
        )
        assert torch.allclose(out_no_sgs, out_wale, atol=1e-6, rtol=1e-6)

    def test_sheared_flow_differs_from_no_sgs(self) -> None:
        """Sheared velocity → non-zero ν_t → differs from no-SGS."""
        f = _f27_sheared()
        tau = 1.0
        out_no_sgs = collide_sc_single_component_27(f, G=-4.0, tau=tau, C_s=0.0)
        out_wale = collide_sc_single_component_27(
            f, G=-4.0, tau=tau, C_s=0.5, sgs_model="wale",
        )
        assert not torch.allclose(out_no_sgs, out_wale, atol=1e-8)

    def test_shape_and_finite(self) -> None:
        f = _f27_sheared()
        out = collide_sc_single_component_27(
            f, G=-4.0, tau=1.0, C_s=0.5, sgs_model="wale",
        )
        assert out.shape == f.shape
        assert torch.isfinite(out).all()

    def test_mass_conserved(self) -> None:
        f = _f27_sheared()
        out = collide_sc_single_component_27(
            f, G=-4.0, tau=1.0, C_s=0.5, sgs_model="wale",
        )
        assert torch.allclose(f.sum(), out.sum(), rtol=1e-4, atol=1e-4)


# ===========================================================================
# Single-component: Vreman SGS
# ===========================================================================

class TestSingleComponentVreman:
    """Vreman SGS coupling for D3Q27 single-component SC."""

    def test_uniform_flow_matches_no_sgs(self) -> None:
        """Uniform velocity → zero velocity gradients → ν_t = 0 → matches no-SGS."""
        f = _f27_uniform()
        tau = 1.0
        out_no_sgs = collide_sc_single_component_27(f, G=-4.0, tau=tau, C_s=0.0)
        out_vreman = collide_sc_single_component_27(
            f, G=-4.0, tau=tau, C_s=0.025, sgs_model="vreman",
        )
        assert torch.allclose(out_no_sgs, out_vreman, atol=1e-6, rtol=1e-6)

    def test_sheared_flow_differs_from_no_sgs(self) -> None:
        """Sheared velocity → non-zero ν_t → differs from no-SGS."""
        f = _f27_sheared()
        tau = 1.0
        out_no_sgs = collide_sc_single_component_27(f, G=-4.0, tau=tau, C_s=0.0)
        out_vreman = collide_sc_single_component_27(
            f, G=-4.0, tau=tau, C_s=0.025, sgs_model="vreman",
        )
        assert not torch.allclose(out_no_sgs, out_vreman, atol=1e-8)

    def test_shape_and_finite(self) -> None:
        f = _f27_sheared()
        out = collide_sc_single_component_27(
            f, G=-4.0, tau=1.0, C_s=0.025, sgs_model="vreman",
        )
        assert out.shape == f.shape
        assert torch.isfinite(out).all()

    def test_mass_conserved(self) -> None:
        f = _f27_sheared()
        out = collide_sc_single_component_27(
            f, G=-4.0, tau=1.0, C_s=0.025, sgs_model="vreman",
        )
        assert torch.allclose(f.sum(), out.sum(), rtol=1e-4, atol=1e-4)


# ===========================================================================
# Single-component: invalid sgs_model
# ===========================================================================

class TestSingleComponentInvalidModel:
    def test_invalid_model_raises_value_error(self) -> None:
        f = _f27_random()
        with pytest.raises(ValueError, match="sgs_model"):
            collide_sc_single_component_27(
                f, G=-4.0, tau=1.0, C_s=0.1, sgs_model="k_epsilon",
            )

    def test_invalid_model_raises_even_with_zero_cs(self) -> None:
        """Validation should happen regardless of C_s value."""
        f = _f27_random()
        with pytest.raises(ValueError, match="sgs_model"):
            collide_sc_single_component_27(
                f, G=-4.0, tau=1.0, C_s=0.0, sgs_model="invalid",
            )


# ===========================================================================
# Two-component: collide_sc_two_component_27
# ===========================================================================

class TestTwoComponentDefault:
    """Default behaviour (C_s=0) must be identical to the original collision."""

    def test_default_no_sgs_matches_explicit_zero(self) -> None:
        f1 = _f27_random()
        f2 = _f27_random()
        o1_def, o2_def = collide_sc_two_component_27(f1, f2, G_12=0.9, tau1=1.0, tau2=1.0)
        o1_exp, o2_exp = collide_sc_two_component_27(
            f1, f2, G_12=0.9, tau1=1.0, tau2=1.0, C_s=0.0, sgs_model="smagorinsky",
        )
        assert torch.allclose(o1_def, o1_exp, atol=0, rtol=0)
        assert torch.allclose(o2_def, o2_exp, atol=0, rtol=0)

    def test_default_preserves_shape(self) -> None:
        f1 = _f27_random()
        f2 = _f27_random()
        o1, o2 = collide_sc_two_component_27(f1, f2, G_12=0.9, tau1=1.0, tau2=1.0)
        assert o1.shape == f1.shape
        assert o2.shape == f2.shape

    def test_default_finite(self) -> None:
        f1 = _f27_random()
        f2 = _f27_random()
        o1, o2 = collide_sc_two_component_27(f1, f2, G_12=0.9, tau1=1.0, tau2=1.0)
        assert torch.isfinite(o1).all()
        assert torch.isfinite(o2).all()


class TestTwoComponentSmagorinsky:
    def test_sgs_differs_from_no_sgs(self) -> None:
        f1 = _f27_random()
        f2 = _f27_random()
        o1_n, o2_n = collide_sc_two_component_27(f1, f2, G_12=0.9, tau1=1.0, tau2=1.0, C_s=0.0)
        o1_s, o2_s = collide_sc_two_component_27(
            f1, f2, G_12=0.9, tau1=1.0, tau2=1.0, C_s=0.1, sgs_model="smagorinsky",
        )
        assert not torch.allclose(o1_n, o1_s, atol=1e-8)
        assert not torch.allclose(o2_n, o2_s, atol=1e-8)

    def test_shape_and_finite(self) -> None:
        f1 = _f27_random()
        f2 = _f27_random()
        o1, o2 = collide_sc_two_component_27(
            f1, f2, G_12=0.9, tau1=1.0, tau2=1.0, C_s=0.1, sgs_model="smagorinsky",
        )
        assert o1.shape == f1.shape
        assert o2.shape == f2.shape
        assert torch.isfinite(o1).all()
        assert torch.isfinite(o2).all()


class TestTwoComponentWALE:
    def test_uniform_flow_matches_no_sgs(self) -> None:
        f1 = _f27_uniform()
        f2 = _f27_uniform()
        tau = 1.0
        o1_n, o2_n = collide_sc_two_component_27(f1, f2, G_12=0.9, tau1=tau, tau2=tau, C_s=0.0)
        o1_w, o2_w = collide_sc_two_component_27(
            f1, f2, G_12=0.9, tau1=tau, tau2=tau, C_s=0.5, sgs_model="wale",
        )
        assert torch.allclose(o1_n, o1_w, atol=1e-6, rtol=1e-6)
        assert torch.allclose(o2_n, o2_w, atol=1e-6, rtol=1e-6)

    def test_sheared_flow_differs_from_no_sgs(self) -> None:
        f1 = _f27_sheared()
        f2 = _f27_sheared()
        tau = 1.0
        o1_n, o2_n = collide_sc_two_component_27(f1, f2, G_12=0.9, tau1=tau, tau2=tau, C_s=0.0)
        o1_w, o2_w = collide_sc_two_component_27(
            f1, f2, G_12=0.9, tau1=tau, tau2=tau, C_s=0.5, sgs_model="wale",
        )
        assert not torch.allclose(o1_n, o1_w, atol=1e-8)
        assert not torch.allclose(o2_n, o2_w, atol=1e-8)

    def test_shape_and_finite(self) -> None:
        f1 = _f27_sheared()
        f2 = _f27_sheared()
        o1, o2 = collide_sc_two_component_27(
            f1, f2, G_12=0.9, tau1=1.0, tau2=1.0, C_s=0.5, sgs_model="wale",
        )
        assert o1.shape == f1.shape
        assert o2.shape == f2.shape
        assert torch.isfinite(o1).all()
        assert torch.isfinite(o2).all()


class TestTwoComponentVreman:
    def test_uniform_flow_matches_no_sgs(self) -> None:
        f1 = _f27_uniform()
        f2 = _f27_uniform()
        tau = 1.0
        o1_n, o2_n = collide_sc_two_component_27(f1, f2, G_12=0.9, tau1=tau, tau2=tau, C_s=0.0)
        o1_v, o2_v = collide_sc_two_component_27(
            f1, f2, G_12=0.9, tau1=tau, tau2=tau, C_s=0.025, sgs_model="vreman",
        )
        assert torch.allclose(o1_n, o1_v, atol=1e-6, rtol=1e-6)
        assert torch.allclose(o2_n, o2_v, atol=1e-6, rtol=1e-6)

    def test_sheared_flow_differs_from_no_sgs(self) -> None:
        f1 = _f27_sheared()
        f2 = _f27_sheared()
        tau = 1.0
        o1_n, o2_n = collide_sc_two_component_27(f1, f2, G_12=0.9, tau1=tau, tau2=tau, C_s=0.0)
        o1_v, o2_v = collide_sc_two_component_27(
            f1, f2, G_12=0.9, tau1=tau, tau2=tau, C_s=0.025, sgs_model="vreman",
        )
        assert not torch.allclose(o1_n, o1_v, atol=1e-8)
        assert not torch.allclose(o2_n, o2_v, atol=1e-8)

    def test_shape_and_finite(self) -> None:
        f1 = _f27_sheared()
        f2 = _f27_sheared()
        o1, o2 = collide_sc_two_component_27(
            f1, f2, G_12=0.9, tau1=1.0, tau2=1.0, C_s=0.025, sgs_model="vreman",
        )
        assert o1.shape == f1.shape
        assert o2.shape == f2.shape
        assert torch.isfinite(o1).all()
        assert torch.isfinite(o2).all()


class TestTwoComponentInvalidModel:
    def test_invalid_model_raises_value_error(self) -> None:
        f1 = _f27_random()
        f2 = _f27_random()
        with pytest.raises(ValueError, match="sgs_model"):
            collide_sc_two_component_27(
                f1, f2, G_12=0.9, tau1=1.0, tau2=1.0, C_s=0.1, sgs_model="k_omega",
            )

    def test_invalid_model_raises_even_with_zero_cs(self) -> None:
        f1 = _f27_random()
        f2 = _f27_random()
        with pytest.raises(ValueError, match="sgs_model"):
            collide_sc_two_component_27(
                f1, f2, G_12=0.9, tau1=1.0, tau2=1.0, C_s=0.0, sgs_model="bad",
            )


# ===========================================================================
# Cross-model consistency: WALE and Vreman give different results
# ===========================================================================

class TestCrossModelConsistency:
    def test_wale_and_vreman_differ_for_sheared_flow(self) -> None:
        """WALE and Vreman use different formulas → different outputs."""
        f = _f27_sheared()
        tau = 1.0
        out_wale = collide_sc_single_component_27(
            f, G=-4.0, tau=tau, C_s=0.5, sgs_model="wale",
        )
        out_vreman = collide_sc_single_component_27(
            f, G=-4.0, tau=tau, C_s=0.025, sgs_model="vreman",
        )
        assert not torch.allclose(out_wale, out_vreman, atol=1e-8)

    def test_smagorinsky_and_wale_differ(self) -> None:
        f = _f27_sheared()
        tau = 1.0
        out_smag = collide_sc_single_component_27(
            f, G=-4.0, tau=tau, C_s=0.1, sgs_model="smagorinsky",
        )
        out_wale = collide_sc_single_component_27(
            f, G=-4.0, tau=tau, C_s=0.5, sgs_model="wale",
        )
        assert not torch.allclose(out_smag, out_wale, atol=1e-8)


# ===========================================================================
# Solid mask interaction
# ===========================================================================

class TestSolidMaskInteraction:
    def test_sgs_with_solid_mask_preserves_solid_cells(self) -> None:
        """Solid cells must be untouched even with SGS enabled."""
        f = _f27_sheared()
        nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
        solid_mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        solid_mask[0, 0, 0] = True
        solid_mask[1, 2, 3] = True

        out = collide_sc_single_component_27(
            f, G=-4.0, tau=1.0, C_s=0.1, sgs_model="smagorinsky",
            solid_mask=solid_mask,
        )
        # Solid cells should be unchanged
        assert torch.allclose(out[:, 0, 0, 0], f[:, 0, 0, 0])
        assert torch.allclose(out[:, 1, 2, 3], f[:, 1, 2, 3])

    def test_sgs_with_solid_mask_finite(self) -> None:
        f = _f27_sheared()
        nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
        solid_mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        solid_mask[0, 0, 0] = True

        for model in ("smagorinsky", "wale", "vreman"):
            cs = 0.1 if model == "smagorinsky" else (0.5 if model == "wale" else 0.025)
            out = collide_sc_single_component_27(
                f, G=-4.0, tau=1.0, C_s=cs, sgs_model=model,
                solid_mask=solid_mask,
            )
            assert torch.isfinite(out).all(), f"NaN/Inf for {model}"
