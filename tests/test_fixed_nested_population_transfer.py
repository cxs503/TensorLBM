"""Conservation tests for the isolated fixed 2:1 LBM population transfer."""
from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import C as C19
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.d3q27 import C as C27
from tensorlbm.d3q27 import equilibrium27
from tensorlbm.fixed_nested_transfer import (
    prolongate_populations_2to1,
    restrict_populations_2to1,
)


def _moments(f: torch.Tensor, directions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return integrated mass and momentum for unit-volume cells."""
    mass = f.sum()
    momentum = (f.reshape(f.shape[0], -1).sum(dim=1, keepdim=True) * directions.to(f)).sum(dim=0)
    return mass, momentum


@pytest.mark.parametrize("q", [19, 27])
def test_prolongation_then_restriction_is_exact_for_arbitrary_populations(q: int) -> None:
    coarse = torch.randn(q, 2, 3, 4, dtype=torch.float64)

    fine = prolongate_populations_2to1(coarse)
    restored = restrict_populations_2to1(fine)

    assert fine.shape == (q, 4, 6, 8)
    assert torch.equal(restored, coarse)


@pytest.mark.parametrize(("q", "directions"), [(19, C19), (27, C27)])
def test_restriction_preserves_volume_weighted_mass_and_momentum(
    q: int, directions: torch.Tensor
) -> None:
    fine = torch.randn(q, 4, 6, 8, dtype=torch.float64)

    coarse = restrict_populations_2to1(fine)
    fine_mass, fine_momentum = _moments(fine, directions)
    coarse_mass, coarse_momentum = _moments(coarse, directions)

    # A coarse cell represents 2**3 fine-cell volumes.
    assert torch.allclose(8.0 * coarse_mass, fine_mass, rtol=0.0, atol=1e-12)
    assert torch.allclose(8.0 * coarse_momentum, fine_momentum, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(("equilibrium_fn", "q", "directions"), [
    (equilibrium3d, 19, C19),
    (equilibrium27, 27, C27),
])
def test_prolongation_preserves_uniform_moving_equilibrium_and_integrated_momentum(
    equilibrium_fn, q: int, directions: torch.Tensor
) -> None:
    shape = (2, 3, 4)
    rho = torch.full(shape, 1.07, dtype=torch.float64)
    ux = torch.full(shape, 0.031, dtype=torch.float64)
    uy = torch.full(shape, -0.017, dtype=torch.float64)
    uz = torch.full(shape, 0.009, dtype=torch.float64)
    coarse = equilibrium_fn(rho, ux, uy, uz)

    fine = prolongate_populations_2to1(coarse)
    expected = equilibrium_fn(
        rho.repeat_interleave(2, 0).repeat_interleave(2, 1).repeat_interleave(2, 2),
        ux.repeat_interleave(2, 0).repeat_interleave(2, 1).repeat_interleave(2, 2),
        uy.repeat_interleave(2, 0).repeat_interleave(2, 1).repeat_interleave(2, 2),
        uz.repeat_interleave(2, 0).repeat_interleave(2, 1).repeat_interleave(2, 2),
    )
    coarse_mass, coarse_momentum = _moments(coarse, directions)
    fine_mass, fine_momentum = _moments(fine, directions)

    assert fine.shape[0] == q
    assert torch.allclose(fine, expected, rtol=0.0, atol=1e-12)
    assert torch.allclose(fine_mass, 8.0 * coarse_mass, rtol=0.0, atol=1e-12)
    assert torch.allclose(fine_momentum, 8.0 * coarse_momentum, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("shape", [(18, 4, 4, 4), (19, 3, 4, 4), (27, 4, 5, 6)])
def test_transfer_rejects_non_d3q19_d3q27_or_non_2to1_shapes(shape: tuple[int, ...]) -> None:
    f = torch.zeros(shape)
    with pytest.raises(ValueError):
        restrict_populations_2to1(f)


@pytest.mark.parametrize("q", [18, 20, 26])
def test_prolongation_rejects_other_stencils(q: int) -> None:
    with pytest.raises(ValueError):
        prolongate_populations_2to1(torch.zeros(q, 2, 2, 2))
