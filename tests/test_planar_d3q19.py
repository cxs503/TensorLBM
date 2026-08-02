from __future__ import annotations

import pytest
import torch

from tensorlbm.cumulant import collide_cumulant_d2q9
from tensorlbm.d2q9 import C as C2, equilibrium
from tensorlbm.d3q19 import C as C3
from tensorlbm.planar_d3q19 import (
    collide_planar_cumulant_d3q19,
    lift_d2q9_to_d3q19,
    marginalize_d3q19_to_d2q9,
    maximum_planar_plane_spread,
)


def _state() -> torch.Tensor:
    torch.manual_seed(20260802)
    rho = torch.ones((3, 5, 7), dtype=torch.float64)
    ux = 0.03 + 0.002 * torch.randn_like(rho)
    uy = 0.002 * torch.randn_like(rho)
    planes = [equilibrium(rho[z], ux[z], uy[z]) for z in range(3)]
    return torch.stack(planes, dim=1)


def test_planar_lift_round_trip_and_conserved_moments() -> None:
    d2 = _state()
    d3 = lift_d2q9_to_d3q19(d2)
    torch.testing.assert_close(marginalize_d3q19_to_d2q9(d3), d2)
    c2 = C2.to(dtype=d2.dtype)
    c3 = C3.to(dtype=d3.dtype)
    d2_momentum = torch.einsum("qa,qzyx->azyx", c2, d2)
    d3_momentum = torch.einsum("qa,qzyx->azyx", c3, d3)
    torch.testing.assert_close(d3.sum(dim=0), d2.sum(dim=0))
    torch.testing.assert_close(d3_momentum[:2], d2_momentum)
    assert torch.count_nonzero(d3_momentum[2]) == 0


@pytest.mark.parametrize("tau", (0.53, 0.68, 1.1))
def test_planar_d3q19_collision_exactly_matches_d2q9(tau: float) -> None:
    d2 = _state()
    d3 = lift_d2q9_to_d3q19(d2)
    actual = marginalize_d3q19_to_d2q9(
        collide_planar_cumulant_d3q19(d3, tau),
    )
    expected = torch.stack([
        collide_cumulant_d2q9(d2[:, z], tau) for z in range(d2.shape[1])
    ], dim=1)
    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-15)


def test_planar_plane_spread_detects_broken_extrusion() -> None:
    d3 = lift_d2q9_to_d3q19(_state()[:, :1].expand(-1, 3, -1, -1).clone())
    assert maximum_planar_plane_spread(d3) < 1e-15
    d3[1, 1, 2, 3] += 1e-4
    assert maximum_planar_plane_spread(d3) > 5e-5


def test_planar_collision_rejects_invalid_tau() -> None:
    with pytest.raises(ValueError, match="tau"):
        collide_planar_cumulant_d3q19(lift_d2q9_to_d3q19(_state()), 0.5)
