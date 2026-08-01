"""Moment identities for the wall-model body-force discretisation."""
from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import C as C19
from tensorlbm.d3q27 import C as C27
from tensorlbm.wall_model import guo_body_force_d3q19, guo_body_force_d3q27


@pytest.mark.parametrize(
    ("q", "velocities", "apply_force"),
    (
        (19, C19, guo_body_force_d3q19),
        (27, C27, guo_body_force_d3q27),
    ),
)
def test_guo_wall_source_has_zero_mass_and_exact_force_moment(
    q: int,
    velocities: torch.Tensor,
    apply_force,
) -> None:
    torch.manual_seed(731)
    shape = (2, 3, 4)
    dtype = torch.float64
    state = torch.zeros((q, *shape), dtype=dtype)
    ux, uy, uz = (
        0.08 * torch.randn(shape, dtype=dtype) for _ in range(3)
    )
    fx, fy, fz = (
        2.0e-4 * torch.randn(shape, dtype=dtype) for _ in range(3)
    )

    updated = apply_force(state, fx, fy, fz, ux, uy, uz)
    source = updated - state
    c = velocities.to(dtype=dtype)

    torch.testing.assert_close(
        source.sum(dim=0), torch.zeros(shape, dtype=dtype),
        rtol=0.0, atol=2.0e-18,
    )
    torch.testing.assert_close(
        torch.einsum("ia,izyx->azyx", c, source),
        torch.stack((fx, fy, fz)),
        rtol=2.0e-14, atol=2.0e-18,
    )
    assert updated.dtype is dtype


def test_mass_identity_covers_nonorthogonal_velocity_and_force() -> None:
    shape = (1, 1, 1)
    dtype = torch.float64
    state = torch.zeros((19, *shape), dtype=dtype)
    ux = torch.full(shape, 0.07, dtype=dtype)
    zero = torch.zeros(shape, dtype=dtype)
    fx = torch.full(shape, -3.0e-4, dtype=dtype)

    source = guo_body_force_d3q19(
        state, fx, zero, zero, ux, zero, zero,
    )

    assert float(source.sum()) == pytest.approx(0.0, abs=1.0e-19)

