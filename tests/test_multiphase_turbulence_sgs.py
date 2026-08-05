"""TDD specification for optional WALE/Vreman SGS models in multiphase couplings.

Verifies that ``free_surface_step`` and ``free_energy_step_3d`` accept an
``sgs_model`` selector (``'smagorinsky'`` | ``'wale'`` | ``'vreman'``) and
dispatch to the corresponding eddy-viscosity / tau_eff calculation when
``C_s > 0``.

Design contract:
  - Default ``sgs_model='smagorinsky'`` preserves existing behaviour exactly.
  - ``C_s == 0`` disables SGS regardless of ``sgs_model`` (no-op path).
  - WALE and Vreman use velocity-gradient-based eddy viscosity →
    ``_nu_t_to_tau_eff``, while Smagorinsky uses the non-equilibrium stress
    norm → ``_smagorinsky_tau``.
  - Invalid ``sgs_model`` strings raise ``ValueError`` before any computation.
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.free_surface_lbm import (
    GAS,
    INTERFACE,
    LIQUID,
    free_surface_step,
)
from tensorlbm.multiphase3d import free_energy_step_3d, init_free_energy_g_3d

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Free-surface helpers
# ---------------------------------------------------------------------------

def _fs_state(
    nz: int = 6, ny: int = 6, nx: int = 8, u_mag: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A small GAS/INTERFACE/LIQUID domain with strong shear for SGS differentiation."""
    flags = torch.full((nz, ny, nx), GAS, dtype=torch.int8)
    flags[:, :, 0] = INTERFACE
    flags[:, :, 1] = INTERFACE
    flags[:, :, 2:] = LIQUID

    fill = torch.zeros((nz, ny, nx))
    fill[flags == INTERFACE] = 0.5
    fill[flags == LIQUID] = 1.0

    solid = torch.zeros_like(flags, dtype=torch.bool)

    rho = torch.where(
        flags == GAS,
        torch.full_like(fill, 0.001),
        torch.ones_like(fill),
    )
    xs = torch.arange(nx, dtype=torch.float32).view(1, 1, nx)
    ys = torch.arange(ny, dtype=torch.float32).view(1, ny, 1)
    zs = torch.arange(nz, dtype=torch.float32).view(nz, 1, 1)
    ux = u_mag * torch.sin(2.0 * torch.pi * xs / nx).expand_as(fill)
    uy = u_mag * 0.5 * torch.cos(2.0 * torch.pi * ys / ny).expand_as(fill)
    uz = u_mag * 0.3 * torch.sin(2.0 * torch.pi * zs / nz).expand_as(fill)
    f = equilibrium3d(rho, ux, uy, uz)
    return f, fill, flags, solid


# ---------------------------------------------------------------------------
# Free-energy helpers
# ---------------------------------------------------------------------------

def _fe_state(
    nz: int = 4, ny: int = 6, nx: int = 8, u_mag: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A small 3-D free-energy domain with non-trivial velocity."""
    rho = torch.ones((nz, ny, nx))
    xs = torch.arange(nx, dtype=torch.float32).view(1, 1, nx)
    ux = u_mag * torch.sin(2.0 * torch.pi * xs / nx).expand_as(rho)
    zero = torch.zeros_like(rho)
    f = equilibrium3d(rho, ux, zero, zero)
    phi = torch.tanh(torch.linspace(-3, 3, nx).view(1, 1, nx).expand(nz, ny, -1))
    g = init_free_energy_g_3d(phi, ux, zero, zero)
    return f, g


# ===========================================================================
# FREE-SURFACE: sgs_model parameter acceptance
# ===========================================================================

class TestFreeSurfaceSGSAcceptance:
    def test_accepts_sgs_model_wale(self) -> None:
        f, fill, flags, solid = _fs_state()
        free_surface_step(
            f, fill, flags, solid, tau=0.8, C_s=0.1, sgs_model='wale',
            freeze_topology=True,
        )

    def test_accepts_sgs_model_vreman(self) -> None:
        f, fill, flags, solid = _fs_state()
        free_surface_step(
            f, fill, flags, solid, tau=0.8, C_s=0.1, sgs_model='vreman',
            freeze_topology=True,
        )

    def test_invalid_sgs_model_raises_value_error(self) -> None:
        f, fill, flags, solid = _fs_state()
        with pytest.raises(ValueError, match="sgs_model"):
            free_surface_step(
                f, fill, flags, solid, tau=0.8, C_s=0.1, sgs_model='invalid',
                freeze_topology=True,
            )


# ===========================================================================
# FREE-SURFACE: default behaviour unchanged
# ===========================================================================

class TestFreeSurfaceDefaultUnchanged:
    def test_default_equals_smagorinsky(self) -> None:
        """Omitting sgs_model must be identical to sgs_model='smagorinsky'."""
        f, fill, flags, solid = _fs_state()
        kwargs = dict(tau=0.8, C_s=0.1, freeze_topology=True)
        out_default = free_surface_step(f.clone(), fill, flags, solid, **kwargs)
        out_smg = free_surface_step(
            f.clone(), fill, flags, solid, sgs_model='smagorinsky', **kwargs,
        )
        assert torch.equal(out_default[0], out_smg[0])

    def test_cs_zero_disables_sgs_regardless_of_model(self) -> None:
        """C_s=0 must produce identical output for all sgs_model choices."""
        f, fill, flags, solid = _fs_state()
        kwargs = dict(tau=0.8, C_s=0.0, freeze_topology=True)
        out_smg = free_surface_step(f.clone(), fill, flags, solid,
                                    sgs_model='smagorinsky', **kwargs)
        out_wale = free_surface_step(f.clone(), fill, flags, solid,
                                     sgs_model='wale', **kwargs)
        out_vreman = free_surface_step(f.clone(), fill, flags, solid,
                                       sgs_model='vreman', **kwargs)
        assert torch.equal(out_smg[0], out_wale[0])
        assert torch.equal(out_smg[0], out_vreman[0])


# ===========================================================================
# FREE-SURFACE: WALE / Vreman produce different results than Smagorinsky
# ===========================================================================

class TestFreeSurfaceSGSDifferentiation:
    def test_wale_differs_from_smagorinsky(self) -> None:
        f, fill, flags, solid = _fs_state()
        kwargs = dict(tau=0.8, C_s=0.5, freeze_topology=True)
        out_smg = free_surface_step(f.clone(), fill, flags, solid,
                                     sgs_model='smagorinsky', **kwargs)
        out_wale = free_surface_step(f.clone(), fill, flags, solid,
                                     sgs_model='wale', **kwargs)
        assert not torch.equal(out_smg[0], out_wale[0])

    def test_vreman_differs_from_smagorinsky(self) -> None:
        f, fill, flags, solid = _fs_state()
        kwargs = dict(tau=0.8, C_s=0.5, freeze_topology=True)
        out_smg = free_surface_step(f.clone(), fill, flags, solid,
                                     sgs_model='smagorinsky', **kwargs)
        out_vreman = free_surface_step(f.clone(), fill, flags, solid,
                                       sgs_model='vreman', **kwargs)
        assert not torch.equal(out_smg[0], out_vreman[0])

    def test_wale_differs_from_vreman(self) -> None:
        f, fill, flags, solid = _fs_state()
        kwargs = dict(tau=0.8, C_s=0.5, freeze_topology=True)
        out_wale = free_surface_step(f.clone(), fill, flags, solid,
                                     sgs_model='wale', **kwargs)
        out_vreman = free_surface_step(f.clone(), fill, flags, solid,
                                       sgs_model='vreman', **kwargs)
        assert not torch.equal(out_wale[0], out_vreman[0])

    def test_sgs_output_is_finite(self) -> None:
        f, fill, flags, solid = _fs_state()
        for model in ('smagorinsky', 'wale', 'vreman'):
            out = free_surface_step(
                f.clone(), fill, flags, solid,
                tau=0.8, C_s=0.1, sgs_model=model, freeze_topology=True,
            )
            assert torch.isfinite(out[0]).all(), model


# ===========================================================================
# FREE-ENERGY 3-D: sgs_model parameter acceptance
# ===========================================================================

class TestFreeEnergySGSAcceptance:
    def test_accepts_cs_and_sgs_model_wale(self) -> None:
        f, g = _fe_state()
        free_energy_step_3d(f, g, tau_f=0.8, C_s=0.1, sgs_model='wale')

    def test_accepts_cs_and_sgs_model_vreman(self) -> None:
        f, g = _fe_state()
        free_energy_step_3d(f, g, tau_f=0.8, C_s=0.1, sgs_model='vreman')

    def test_invalid_sgs_model_raises_value_error(self) -> None:
        f, g = _fe_state()
        with pytest.raises(ValueError, match="sgs_model"):
            free_energy_step_3d(f, g, tau_f=0.8, C_s=0.1, sgs_model='invalid')


# ===========================================================================
# FREE-ENERGY 3-D: default behaviour unchanged
# ===========================================================================

class TestFreeEnergyDefaultUnchanged:
    def test_default_no_sgs(self) -> None:
        """Without C_s/sgs_model, the step must behave as before (plain BGK)."""
        f, g = _fe_state()
        out_default = free_energy_step_3d(f.clone(), g, tau_f=0.8)
        out_explicit = free_energy_step_3d(
            f.clone(), g, tau_f=0.8, C_s=0.0, sgs_model='smagorinsky',
        )
        assert torch.equal(out_default[0], out_explicit[0])

    def test_cs_zero_disables_sgs_regardless_of_model(self) -> None:
        f, g = _fe_state()
        kw = dict(tau_f=0.8, C_s=0.0)
        out_smg = free_energy_step_3d(f.clone(), g, sgs_model='smagorinsky', **kw)
        out_wale = free_energy_step_3d(f.clone(), g, sgs_model='wale', **kw)
        out_vreman = free_energy_step_3d(f.clone(), g, sgs_model='vreman', **kw)
        assert torch.equal(out_smg[0], out_wale[0])
        assert torch.equal(out_smg[0], out_vreman[0])


# ===========================================================================
# FREE-ENERGY 3-D: WALE / Vreman differentiation
# ===========================================================================

class TestFreeEnergySGSDifferentiation:
    def test_wale_differs_from_smagorinsky(self) -> None:
        f, g = _fe_state()
        kw = dict(tau_f=0.8, C_s=0.1)
        out_smg = free_energy_step_3d(f.clone(), g, sgs_model='smagorinsky', **kw)
        out_wale = free_energy_step_3d(f.clone(), g, sgs_model='wale', **kw)
        assert not torch.allclose(out_smg[0], out_wale[0], atol=1e-7)

    def test_vreman_differs_from_smagorinsky(self) -> None:
        f, g = _fe_state()
        kw = dict(tau_f=0.8, C_s=0.1)
        out_smg = free_energy_step_3d(f.clone(), g, sgs_model='smagorinsky', **kw)
        out_vreman = free_energy_step_3d(f.clone(), g, sgs_model='vreman', **kw)
        assert not torch.allclose(out_smg[0], out_vreman[0], atol=1e-7)

    def test_wale_differs_from_vreman(self) -> None:
        f, g = _fe_state()
        kw = dict(tau_f=0.8, C_s=0.1)
        out_wale = free_energy_step_3d(f.clone(), g, sgs_model='wale', **kw)
        out_vreman = free_energy_step_3d(f.clone(), g, sgs_model='vreman', **kw)
        assert not torch.allclose(out_wale[0], out_vreman[0], atol=1e-7)

    def test_sgs_output_is_finite(self) -> None:
        f, g = _fe_state()
        for model in ('smagorinsky', 'wale', 'vreman'):
            out = free_energy_step_3d(
                f.clone(), g, tau_f=0.8, C_s=0.1, sgs_model=model,
            )
            assert torch.isfinite(out[0]).all(), model

    def test_sgs_does_not_break_g_evolution(self) -> None:
        """The order-parameter distribution g must remain finite and shaped."""
        f, g = _fe_state()
        for model in ('smagorinsky', 'wale', 'vreman'):
            _, g_out = free_energy_step_3d(
                f.clone(), g, tau_f=0.8, C_s=0.1, sgs_model=model,
            )
            assert g_out.shape == g.shape, model
            assert torch.isfinite(g_out).all(), model


# ===========================================================================
# Capability contract: multiphase SGS coupling reflected in matrix
# ===========================================================================

class TestCapabilityContractMultiphaseSGS:
    def test_wale_d3q19_bgk_notes_multiphase_coupling(self) -> None:
        from tensorlbm.turbulence_capability_contract import (
            turbulence_capability_matrix,
        )
        cap = turbulence_capability_matrix()["wale"]["D3Q19"]["BGK"]
        assert cap.implementation_status == "IMPLEMENTED"
        note = cap.note.lower()
        assert "free_surface" in note or "multiphase" in note or "free-energy" in note

    def test_vreman_d3q19_bgk_notes_multiphase_coupling(self) -> None:
        from tensorlbm.turbulence_capability_contract import (
            turbulence_capability_matrix,
        )
        cap = turbulence_capability_matrix()["vreman"]["D3Q19"]["BGK"]
        assert cap.implementation_status == "IMPLEMENTED"
        note = cap.note.lower()
        assert "free_surface" in note or "multiphase" in note or "free-energy" in note
