"""Linear irregular-wave, RAO, and Cummins 6DOF analytical regression checks.

These tests use prescribed synthetic hydrodynamic coefficients and a local
JONSWAP spectrum.  They do not instantiate an LBM solver or couple a hull.
"""

import math

import pytest
import torch

from tensorlbm.rao_analysis import (
    compute_natural_frequencies,
    compute_rao_from_hydrodynamics,
    compute_rao_from_timeseries,
    spectral_response_analysis,
)
from tensorlbm.rigid_body_6dof import (
    BodyProperties6DOF,
    HydrostaticMatrix,
    RadiationData,
    cummins_time_integration,
    generate_jonswap_excitation,
)


def _jonswap_spectrum(omega: torch.Tensor, omega_peak: float = 1.0, gamma: float = 3.3) -> torch.Tensor:
    alpha, gravity = 0.0081, 9.81
    positive_omega = torch.clamp(omega, min=1e-6)
    sigma = torch.where(omega <= omega_peak, 0.07, 0.09)
    peak = torch.exp(-((omega - omega_peak) ** 2) / (2.0 * sigma**2 * omega_peak**2))
    return alpha * gravity**2 * positive_omega**-5 * torch.exp(
        -1.25 * (omega_peak / positive_omega) ** 4
    ) * gamma**peak


def _body_and_radiation() -> tuple[BodyProperties6DOF, HydrostaticMatrix, RadiationData]:
    mass = 1_000.0
    body = BodyProperties6DOF(
        mass=mass,
        inertia_matrix=torch.diag(torch.tensor([400.0, 900.0, 900.0])),
        displacement_volume=mass / 1_000.0,
        waterplane_area=1.0,
        water_density=1_000.0,
    )
    hydro = HydrostaticMatrix.from_body(body, gm_transverse=0.5, gm_longitudinal=1.0)
    omega = torch.linspace(0.2, 2.0, 32)
    added_mass = torch.zeros(32, 6, 6)
    damping = torch.zeros(32, 6, 6)
    for dof in (2, 3, 4):
        added_mass[:, dof, dof] = 100.0
        damping[:, dof, dof] = 80.0
    return body, hydro, RadiationData(omega, added_mass, damping, added_mass[0].clone())


def test_hydrodynamic_rao_and_natural_frequency_are_finite() -> None:
    body, hydro, radiation = _body_and_radiation()
    excitation = torch.zeros(32, 6)
    excitation[:, 2] = body.water_density * body.gravity * body.waterplane_area
    rao = compute_rao_from_hydrodynamics(
        radiation.omega, radiation.added_mass, radiation.damping, hydro.c_matrix,
        body.build_mass_matrix(), excitation,
    )
    assert rao.rao_amplitude.shape == (32, 6)
    assert torch.isfinite(rao.rao_amplitude).all()
    natural = compute_natural_frequencies(body.build_mass_matrix(), radiation.added_mass[0], hydro.c_matrix)
    assert torch.isfinite(natural).all()
    assert natural.max() > 0.0


def test_irregular_wave_generation_is_seed_reproducible_and_sra_is_positive() -> None:
    _, _, radiation = _body_and_radiation()
    spectrum = _jonswap_spectrum(radiation.omega)
    unit_transfer = torch.ones(32, 6, dtype=torch.complex64)
    first = generate_jonswap_excitation(radiation.omega, spectrum, unit_transfer, 10.0, 16, 20.0, 0.1, seed=12)
    second = generate_jonswap_excitation(radiation.omega, spectrum, unit_transfer, 10.0, 16, 20.0, 0.1, seed=12)
    assert torch.allclose(first[1], second[1])
    response = spectral_response_analysis(radiation.omega, spectrum, torch.ones_like(spectrum))
    assert response["m0"] > 0.0
    assert torch.allclose(response["response_spectrum"], spectrum)


def test_timeseries_rao_and_prescribed_cummins_response_are_bounded() -> None:
    n, dt, omega = 2_048, 0.05, 1.0
    time = torch.arange(n) * dt
    elevation = 0.05 * torch.cos(omega * time)
    response = 0.10 * torch.cos(omega * time + 0.3)
    frequencies, amplitude, _ = compute_rao_from_timeseries(time, elevation, response)
    index = torch.argmin(torch.abs(frequencies - omega))
    assert amplitude[index].item() == pytest.approx(2.0, rel=0.2)

    body, hydro, radiation = _body_and_radiation()
    excitation = torch.zeros(100, 6)
    excitation[:, 2] = 50.0 * torch.cos(torch.arange(100) * 0.02)
    state = cummins_time_integration(body, hydro, radiation, excitation, 0.02, 100)
    assert torch.isfinite(state.position).all()
    assert torch.isfinite(state.velocity).all()
