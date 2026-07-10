"""Energy and phase-volume regression tests for the Gallium PF Stefan closure."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from benchmark_gallium_pf import (  # noqa: E402
    conservative_phase_field_update,
    phase_field_update_with_energy_closure,
    phase_increment_to_temperature,
    stefan_phase_source,
)


def test_stefan_source_survives_conservative_pf_update_and_preserves_enthalpy():
    """PF sharpening may redistribute phase, but must not cancel Stefan melting.

    With insulated faces and zero flow, the only global liquid-volume change is
    the Stefan source.  The matching latent-temperature increment must leave
    discrete sensible-plus-latent enthalpy unchanged.
    """
    phi = -torch.ones((1, 9, 11), dtype=torch.float64)
    phi[:, :, 0] = 1.0  # imposed hot-wall liquid, excluded from source
    temperature = torch.full_like(phi, 0.35)
    temperature[:, :, 0] = 1.0
    cp, latent_heat, melting_temperature = 1.0, 8.0, 0.15

    delta_phi, latent_temperature_increment = stefan_phase_source(
        phi, temperature, cp=cp, latent_heat=latent_heat,
        melting_temperature=melting_temperature, rate=1.0,
    )
    phi_after_source = phi + delta_phi
    phi_after_pf = conservative_phase_field_update(
        phi_after_source, ux=torch.zeros_like(phi), uy=torch.zeros_like(phi),
        mobility=0.1, interface_mobility=0.02, interface_width=4.0,
    )

    # The conservative transport/sharpening closure cannot alter total phase.
    assert torch.allclose(phi_after_pf.sum(), phi_after_source.sum(), atol=1e-11)
    # Thus the positive Stefan source cannot be erased by anti-diffusion.
    assert phi_after_pf.sum() > phi.sum()

    enthalpy_before = (cp * temperature + latent_heat * (phi + 1.0) / 2.0).sum()
    enthalpy_after = (
        cp * (temperature + latent_temperature_increment)
        + latent_heat * (phi_after_source + 1.0) / 2.0
    ).sum()
    assert torch.allclose(enthalpy_after, enthalpy_before, atol=1e-11)


def test_pf_flux_energy_closure_preserves_cellwise_enthalpy():
    """A conservative PF flux still requires a local sensible-energy update.

    Global phase volume conservation alone is insufficient: moving latent
    enthalpy from one cell to the next without the corresponding temperature
    change produces a spurious local heat source/sink.  The coupled update
    must preserve cp*T + L*(phi+1)/2 in every active cell before transport.
    """
    phi = -torch.ones((1, 7, 9), dtype=torch.float64)
    phi[:, :, :3] = 1.0
    temperature = torch.linspace(0.1, 0.7, 9, dtype=torch.float64).view(1, 1, 9)
    temperature = temperature.expand_as(phi).clone()
    cp, latent_heat = 1.0, 8.0

    phi_next, temperature_increment = phase_field_update_with_energy_closure(
        phi, ux=torch.zeros_like(phi), uy=torch.zeros_like(phi),
        mobility=0.1, interface_mobility=0.02, interface_width=4.0,
        cp=cp, latent_heat=latent_heat,
    )

    enthalpy_before = cp * temperature + latent_heat * (phi + 1.0) / 2.0
    enthalpy_after = (cp * (temperature + temperature_increment)
                      + latent_heat * (phi_next + 1.0) / 2.0)
    assert torch.allclose(enthalpy_after, enthalpy_before, atol=1e-11)
