"""Tests for the isolated fixed-nested D3Q27 planar interface primitive."""
from __future__ import annotations

import torch

from tensorlbm.d3q27 import C, equilibrium27
from tensorlbm.fixed_nested_interface import (
    reconstruct_coarse_incoming_from_fine_d3q27,
    reconstruct_fine_incoming_from_coarse_d3q27,
)


def _face_equilibrium(rho: float, velocity: tuple[float, float, float], shape: tuple[int, int]) -> torch.Tensor:
    """Return one uniform D3Q27 face in the helper's ``(Q, t0, t1)`` layout."""
    rho_field = torch.full(shape, rho, dtype=torch.float64)
    # equilibrium27 expects a volumetric ``(nz, ny, nx)`` field; take nz=1.
    return equilibrium27(
        rho_field.unsqueeze(0),
        torch.full((1, *shape), velocity[0], dtype=torch.float64),
        torch.full((1, *shape), velocity[1], dtype=torch.float64),
        torch.full((1, *shape), velocity[2], dtype=torch.float64),
    ).squeeze(1)


def _moments(populations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mass = populations.sum()
    momentum = (populations.sum(dim=(1, 2), keepdim=True) * C.to(populations)).sum(dim=0).squeeze()
    return mass, momentum


def test_uniform_equilibrium_is_unchanged_in_both_interface_directions() -> None:
    normal = (1, 0, 0)  # Points from the coarse patch into the fine patch.
    coarse = _face_equilibrium(1.03, (0.021, -0.013, 0.007), (3, 2))
    fine = _face_equilibrium(1.03, (0.021, -0.013, 0.007), (6, 4))

    reconstructed_fine = reconstruct_fine_incoming_from_coarse_d3q27(coarse, fine, normal)
    reconstructed_coarse = reconstruct_coarse_incoming_from_fine_d3q27(
        fine.unsqueeze(0).repeat(2, 1, 1, 1), coarse, normal
    )

    assert torch.equal(reconstructed_fine, fine)
    assert torch.equal(reconstructed_coarse, coarse)


def test_planar_packet_conserves_volume_weighted_mass_and_momentum() -> None:
    """A coarse step equals two fine substeps; coarse/fine volumes are 8/1."""
    normal = (1, 0, 0)
    coarse_outgoing = torch.zeros(27, 1, 1, dtype=torch.float64)
    # Analytic D3Q27 packet: directions +x, (+x,+y,-z), and (-x,+y,0).
    coarse_outgoing[1, 0, 0] = 0.37
    coarse_outgoing[23, 0, 0] = 0.11
    coarse_outgoing[8, 0, 0] = 9.0  # Travels fine -> coarse; must not be transferred here.
    fine_incoming = torch.zeros(27, 2, 2, dtype=torch.float64)

    fine_step = reconstruct_fine_incoming_from_coarse_d3q27(
        coarse_outgoing, fine_incoming, normal
    )
    crossing = (C @ torch.tensor(normal)) > 0
    coarse_packet = torch.where(crossing[:, None, None], coarse_outgoing, torch.zeros_like(coarse_outgoing))
    coarse_mass, coarse_momentum = _moments(coarse_packet)
    fine_mass, fine_momentum = _moments(fine_step)

    # Vc/Vf = 8 and dtc/dtf = 2. Four fine face cells are crossed each fine step.
    assert torch.allclose(8.0 * coarse_mass, 2.0 * fine_mass, rtol=0.0, atol=1e-14)
    assert torch.allclose(8.0 * coarse_momentum, 2.0 * fine_momentum, rtol=0.0, atol=1e-14)
    assert torch.count_nonzero(fine_step[8]) == 0

    fine_outgoing_substeps = torch.stack((fine_step, fine_step))
    coarse_incoming = torch.zeros(27, 1, 1, dtype=torch.float64)
    reconstructed_coarse = reconstruct_coarse_incoming_from_fine_d3q27(
        fine_outgoing_substeps, coarse_incoming, normal
    )
    reverse = (C @ torch.tensor(normal)) < 0
    assert torch.count_nonzero(reconstructed_coarse[reverse]) == 0
