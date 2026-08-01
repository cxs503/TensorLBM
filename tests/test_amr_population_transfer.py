from __future__ import annotations

import pytest
import torch

from tensorlbm.amr_population_transfer import (
    regularize_nonequilibrium_second_order,
    rescale_nonequilibrium,
)
from tensorlbm.d3q19 import C, equilibrium3d


def _moments(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    by_q = f.sum(dim=(1, 2, 3))
    return by_q.sum(), (by_q[:, None] * C.to(by_q)).sum(dim=0)


def test_second_order_projection_removes_ghost_mode_and_conserves_moments() -> None:
    perturbation = torch.zeros((19, 2, 3, 4), dtype=torch.float64)
    # This opposite-pair/rest perturbation has zero mass and momentum, while
    # containing both stress and higher-order kinetic content.
    perturbation[0] = -0.06
    perturbation[1] = 0.03
    perturbation[2] = 0.03
    projected = regularize_nonequilibrium_second_order(perturbation)

    mass, momentum = _moments(projected)
    assert mass.item() == pytest.approx(0.0, abs=1e-13)
    assert torch.allclose(momentum, torch.zeros_like(momentum), atol=1e-13, rtol=0.0)
    assert not torch.equal(projected, perturbation)
    cx = C[:, 0, None, None, None].to(projected)
    assert torch.allclose(
        (cx.square() * projected).sum(dim=0),
        (cx.square() * perturbation).sum(dim=0),
        atol=2e-8,
        rtol=0.0,
    )


@pytest.mark.parametrize("regularize", [False, True])
def test_rescaling_keeps_uniform_moving_equilibrium(regularize: bool) -> None:
    rho = torch.ones((3, 4, 5), dtype=torch.float64)
    ux = torch.full_like(rho, 0.04)
    zero = torch.zeros_like(rho)
    equilibrium = equilibrium3d(rho, ux, zero, zero)
    transferred = rescale_nonequilibrium(
        equilibrium,
        tau_source=0.5004,
        tau_target=0.5002,
        spatial_ratio=2.0,
        regularize=regularize,
    )
    assert torch.allclose(transferred, equilibrium, atol=2e-8, rtol=0.0)


def test_regularized_rescaling_preserves_density_and_momentum() -> None:
    torch.manual_seed(42)
    rho = torch.ones((3, 4, 5), dtype=torch.float64)
    ux = torch.full_like(rho, 0.03)
    zero = torch.zeros_like(rho)
    state = equilibrium3d(rho, ux, zero, zero)
    perturbation = torch.randn_like(state) * 1e-5
    perturbation -= perturbation.mean(dim=0, keepdim=True)
    state += perturbation
    before = _moments(state)
    transferred = rescale_nonequilibrium(
        state,
        tau_source=0.5008,
        tau_target=0.5004,
        spatial_ratio=2.0,
        regularize=True,
    )
    after = _moments(transferred)
    assert after[0].item() == pytest.approx(before[0].item(), abs=2e-6)
    assert torch.allclose(after[1], before[1], atol=2e-6, rtol=0.0)


def test_rescaling_rejects_invalid_parameters() -> None:
    state = torch.ones((19, 2, 2, 2))
    with pytest.raises(ValueError, match="positive"):
        rescale_nonequilibrium(
            state, tau_source=0.0, tau_target=0.6, spatial_ratio=2.0,
        )
