from __future__ import annotations

import pytest
import torch

from tensorlbm.cumulant import collide_cumulant_d2q9
from tensorlbm.bfl_d3q19 import (
    bouzidi_bounce_back_d3q19,
    compute_q_cylinder_d3q19,
)
from tensorlbm.d2q9 import C as C2, equilibrium
from tensorlbm.d3q19 import C as C3
from tensorlbm.interpolated_bc import bouzidi_bounce_back, compute_q_circle
from tensorlbm.planar_d3q19 import (
    collide_planar_cumulant_d3q19,
    lift_d2q9_to_d3q19,
    marginalize_d3q19_to_d2q9,
    maximum_planar_plane_spread,
)
from tensorlbm.solver import stream
from tensorlbm.solver3d import stream3d


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


def test_planar_collision_stream_and_curved_wall_match_d2q9() -> None:
    ny = nx = 13
    rho = torch.ones((ny, nx), dtype=torch.float64)
    ux = torch.full_like(rho, 0.03)
    uy = torch.zeros_like(rho)
    d2 = equilibrium(rho, ux, uy)
    d3 = lift_d2q9_to_d3q19(
        d2[:, None].expand(-1, 3, -1, -1).clone(),
    )
    tau = 0.68
    d2_post = collide_cumulant_d2q9(d2, tau)
    d3_post = collide_planar_cumulant_d3q19(d3, tau)
    d2_streamed = stream(d2_post)
    d3_streamed = stream3d(d3_post)

    d2_mask, d2_q = compute_q_circle(
        nx, ny, 6.0, 6.0, 2.0, torch.device("cpu"),
    )
    for direction in range(1, 9):
        d2_streamed = bouzidi_bounce_back(
            d2_streamed,
            d2_post,
            d2_mask[direction],
            d2_q[direction],
            direction,
        )
    d3_mask, d3_q = compute_q_cylinder_d3q19(
        nx, ny, 3, 6.0, 6.0, 2.0, torch.device("cpu"),
    )
    d3_streamed = bouzidi_bounce_back_d3q19(
        d3_streamed, d3_post, d3_mask, d3_q,
    )

    marginal = marginalize_d3q19_to_d2q9(d3_streamed)
    torch.testing.assert_close(
        marginal,
        d2_streamed[:, None].expand_as(marginal),
        rtol=2e-14,
        atol=2e-15,
    )
