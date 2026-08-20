"""Regression tests for the interface-limited Gallium Stefan closure."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from benchmark_gallium_pf import (  # noqa: E402
    GALLIUM_SOLID_TO_LIQUID_CONDUCTIVITY_RATIO,
    interface_stefan_phase_source,
    momentum_solid_mask,
    stefan_nondimensional_diagnostic,
)


def test_gallium_conductivity_ratio_is_a_material_property_default():
    """The default two-material flux uses k_s/k_l, not a tuned unity value."""
    assert abs(GALLIUM_SOLID_TO_LIQUID_CONDUCTIVITY_RATIO - 40.6 / 28.0) < 1e-14


def test_interface_stefan_source_is_local_and_enthalpy_conservative():
    """Superheat far from the front must not melt bulk solid cells.

    The source is supported only in the diffuse/sharp interface band and its
    paired sensible-temperature update preserves discrete local enthalpy.
    """
    phi = -torch.ones((1, 5, 9), dtype=torch.float64)
    phi[:, :, :3] = 1.0
    temperature = torch.full_like(phi, 0.15)
    temperature[:, :, 0] = 1.0
    temperature[:, :, 1] = 0.80
    temperature[:, :, 2] = 0.50
    temperature[:, :, 5:] = 0.95  # deliberately superheated bulk solid
    cp, latent_heat, tm, alpha = 1.0, 8.0, 0.15, 0.1

    delta_phi, delta_temperature = interface_stefan_phase_source(
        phi,
        temperature,
        cp=cp,
        latent_heat=latent_heat,
        melting_temperature=tm,
        thermal_diffusivity=alpha,
    )

    # Only the solid cell adjacent to the liquid may melt.  In particular,
    # remote hot solid cannot be a volumetric phase source.
    # Boundary rows are excluded because the closure requires a complete
    # two-sided normal stencil; all interior front cells must advance.
    assert torch.all(delta_phi[:, 1:-1, 3] > 0.0)
    assert torch.count_nonzero(delta_phi[:, :, 4:]) == 0
    assert torch.count_nonzero(delta_phi[:, :, :3]) == 0

    enthalpy_increment = cp * delta_temperature + latent_heat * delta_phi / 2.0
    assert torch.allclose(enthalpy_increment, torch.zeros_like(enthalpy_increment), atol=1e-12)


def test_partially_melted_stefan_front_is_hydrodynamically_liquid():
    """A fractional front cell must receive buoyancy and carry liquid flow.

    ``phi < 0`` denotes less than half liquid, not an impermeable solid.  If
    it is bounce-backed until it crosses zero, the interface-limited Stefan
    front never develops the liquid convection that shapes this benchmark.
    Only exactly unmelted cells are solid for momentum.
    """
    wall = torch.zeros((1, 1, 4), dtype=torch.bool)
    phi = torch.tensor([[[-1.0, -0.9, 0.0, 1.0]]])
    mask = momentum_solid_mask(wall, phi)
    assert torch.equal(mask, torch.tensor([[[True, False, False, False]]]))


def test_interface_equilibrium_consumes_only_front_superheat_conservatively():
    """Thermal excess is latent heat only at a liquid/solid face, never bulk."""
    from benchmark_gallium_pf import interface_equilibrium_phase_source

    phi = -torch.ones((1, 5, 8), dtype=torch.float64)
    phi[:, :, :2] = 1.0
    temperature = torch.full_like(phi, 0.15)
    temperature[:, :, 2] = 0.55  # superheated interface solid
    temperature[:, :, 5:] = 0.95  # forbidden remote superheated bulk
    delta_phi, delta_temperature = interface_equilibrium_phase_source(
        phi, temperature, cp=1.0, latent_heat=8.0, melting_temperature=0.15
    )

    assert torch.all(delta_phi[:, 1:-1, 2] > 0.0)
    assert torch.count_nonzero(delta_phi[:, :, 3:]) == 0
    assert torch.allclose(
        temperature + delta_temperature,
        torch.where(delta_phi != 0, torch.full_like(temperature, 0.15), temperature),
    )
    assert torch.allclose(
        delta_temperature + 4.0 * delta_phi, torch.zeros_like(delta_temperature), atol=1e-12
    )


def test_interface_stefan_source_uses_isothermal_front_not_cell_average_superheat():
    """A fractional front's stored sensible heat cannot alter flux twice."""
    phi = -torch.ones((1, 5, 7), dtype=torch.float64)
    phi[:, :, :3] = 1.0
    base = torch.full_like(phi, 0.10)
    base[:, :, 1] = 0.65  # liquid one-sided stencil
    base[:, :, 4] = 0.05  # solid one-sided stencil
    hot_front = base.clone()
    hot_front[:, :, 3] = 0.90  # only the volume-average front-cell value differs
    common = dict(cp=1.0, latent_heat=8.0, melting_temperature=0.15, thermal_diffusivity=0.1)
    d_base, _ = interface_stefan_phase_source(phi, base, **common)
    d_hot, _ = interface_stefan_phase_source(phi, hot_front, **common)
    assert torch.allclose(d_base, d_hot, atol=1e-14)


def test_interface_stefan_source_uses_conductivity_weighted_flux_jump():
    """The solid-side flux must use its own conductivity, not liquid α.

    For a cold solid-side stencil the solid flux opposes melting.  Increasing
    its conductivity must therefore reduce the local Stefan increment by the
    analytically known amount; this guards the discrete two-material flux
    jump against silently treating gallium solid and liquid as identical.
    """
    phi = -torch.ones((1, 5, 7), dtype=torch.float64)
    phi[:, :, :3] = 1.0
    temperature = torch.full_like(phi, 0.15)
    temperature[:, :, 2] = 0.35
    temperature[:, :, 4] = 0.05
    common = dict(cp=1.0, latent_heat=8.0, melting_temperature=0.15, thermal_diffusivity=0.1)
    base, _ = interface_stefan_phase_source(phi, temperature, **common)
    weighted, _ = interface_stefan_phase_source(
        phi, temperature, solid_conductivity_ratio=1.5, **common
    )

    # The solid stencil is 0.10 below Tm, so weighting it more reduces Δφ by
    # 2*(α cp/L)*(0.5*0.10).
    expected = -2.0 * 0.1 / 8.0 * 0.5 * 0.10
    assert torch.allclose(
        weighted[:, 1:-1, 3] - base[:, 1:-1, 3],
        torch.full_like(base[:, 1:-1, 3], expected),
        atol=1e-14,
    )


def test_stefan_nondimensional_diagnostic_reports_lattice_and_gau_viskanta_scales():
    """The reported Stefan/Fourier scales must be the actual code scales.

    This is measurement-only: it guards against silently mixing physical and
    lattice time in the source before comparison with Gau--Viskanta data.
    """
    d = stefan_nondimensional_diagnostic(
        nx=40,
        tau_T=0.8,
        steps=8000,
        cp=1.0,
        latent_heat=18.52,
        T_hot=1.0,
        T_melt=0.148,
    )
    assert abs(d["alpha"] - 0.1) < 1e-14
    assert abs(d["Ste"] - 0.046004319654427646) < 1e-14
    assert abs(d["Fo_final"] - 0.5) < 1e-14
    assert abs(d["stefan_gradient_coefficient"] - 0.1 / 18.52) < 1e-14
