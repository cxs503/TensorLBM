"""Tests for isolated D3Q27 2:1 temporal interface reflux bookkeeping."""
from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q27 import C, equilibrium27
from tensorlbm.d3q27_temporal_reflux import (
    D3Q27InterfaceFluxPacket,
    reflux_d3q27_2to1,
)


def _moments(packet: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return packet.sum(), (packet[:, None] * C.to(packet)).sum(dim=0)


def _uniform_equilibrium_packet() -> torch.Tensor:
    rho = torch.full((1, 1, 1), 1.03, dtype=torch.float64)
    equilibrium = equilibrium27(
        rho,
        torch.full_like(rho, 0.021),
        torch.full_like(rho, -0.013),
        torch.full_like(rho, 0.007),
    )
    return equilibrium[:, 0, 0, 0]


def test_uniform_equilibrium_has_zero_temporal_mismatch_and_corrections() -> None:
    # The two fine packets are half-coarse-step integrals in a common orientation.
    coarse = D3Q27InterfaceFluxPacket(_uniform_equilibrium_packet(), substep=None)
    fine_0 = D3Q27InterfaceFluxPacket(_uniform_equilibrium_packet() / 2.0, substep=0)
    fine_1 = D3Q27InterfaceFluxPacket(_uniform_equilibrium_packet() / 2.0, substep=1)

    result = reflux_d3q27_2to1(coarse, (fine_0, fine_1))

    assert torch.equal(result.mismatch, torch.zeros_like(coarse.flux))
    assert torch.equal(result.coarse_correction, torch.zeros_like(coarse.flux))
    assert all(torch.equal(correction, torch.zeros_like(coarse.flux)) for correction in result.fine_corrections)


def test_arbitrary_packets_are_conservative_in_mass_and_all_momentum_components() -> None:
    coarse_flux = torch.tensor(
        [0.11, -0.37, 0.22, 0.08, -0.09, 0.31, -0.15, 0.42, -0.29] + [0.0] * 18,
        dtype=torch.float64,
    )
    fine_0_flux = torch.tensor(
        [-0.05, 0.14, 0.28, -0.41, 0.17, -0.07, 0.33, 0.06, 0.19] + [0.0] * 18,
        dtype=torch.float64,
    )
    fine_1_flux = torch.tensor(
        [0.23, -0.18, 0.04, 0.25, -0.32, 0.09, -0.12, 0.37, -0.21] + [0.0] * 18,
        dtype=torch.float64,
    )
    result = reflux_d3q27_2to1(
        D3Q27InterfaceFluxPacket(coarse_flux, substep=None),
        (
            D3Q27InterfaceFluxPacket(fine_0_flux, substep=0),
            D3Q27InterfaceFluxPacket(fine_1_flux, substep=1),
        ),
    )

    raw_coarse_mass, raw_coarse_momentum = _moments(coarse_flux)
    raw_fine_mass, raw_fine_momentum = _moments(fine_0_flux + fine_1_flux)
    coarse_mass, coarse_momentum = _moments(result.corrected_coarse_flux)
    fine_mass, fine_momentum = _moments(result.corrected_fine_fluxes[0] + result.corrected_fine_fluxes[1])
    mismatch_mass, mismatch_momentum = _moments(result.mismatch)

    assert torch.allclose(mismatch_mass, raw_coarse_mass - raw_fine_mass, rtol=0.0, atol=1e-14)
    assert torch.allclose(mismatch_momentum, raw_coarse_momentum - raw_fine_momentum, rtol=0.0, atol=1e-14)
    assert torch.allclose(coarse_mass, fine_mass, rtol=0.0, atol=1e-14)
    assert torch.allclose(coarse_momentum, fine_momentum, rtol=0.0, atol=1e-14)
    assert torch.equal(result.coarse_correction + sum(result.fine_corrections), torch.zeros_like(coarse_flux))


def test_missing_or_invalid_fine_substeps_are_rejected() -> None:
    packet = torch.ones(27, dtype=torch.float64)
    coarse = D3Q27InterfaceFluxPacket(packet, substep=None)

    with pytest.raises(ValueError, match="exactly two"):
        reflux_d3q27_2to1(coarse, (D3Q27InterfaceFluxPacket(packet, substep=0),))
    with pytest.raises(ValueError, match="substeps 0 and 1"):
        reflux_d3q27_2to1(
            coarse,
            (D3Q27InterfaceFluxPacket(packet, substep=0), D3Q27InterfaceFluxPacket(packet, substep=0)),
        )
    with pytest.raises(ValueError, match="finite"):
        reflux_d3q27_2to1(
            coarse,
            (
                D3Q27InterfaceFluxPacket(packet, substep=0),
                D3Q27InterfaceFluxPacket(torch.full((27,), float("nan")), substep=1),
            ),
        )
