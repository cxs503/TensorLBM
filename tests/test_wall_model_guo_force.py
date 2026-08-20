"""Moment identities for the wall-model body-force discretisation."""

from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import C as C19
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.d3q27 import C as C27
from tensorlbm.wall_function_common import (
    _apply_body_force,
    _near_wall_mask,
    wall_function,
)
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
    ux, uy, uz = (0.08 * torch.randn(shape, dtype=dtype) for _ in range(3))
    fx, fy, fz = (2.0e-4 * torch.randn(shape, dtype=dtype) for _ in range(3))

    updated = apply_force(state, fx, fy, fz, ux, uy, uz)
    source = updated - state
    c = velocities.to(dtype=dtype)

    torch.testing.assert_close(
        source.sum(dim=0),
        torch.zeros(shape, dtype=dtype),
        rtol=0.0,
        atol=2.0e-18,
    )
    torch.testing.assert_close(
        torch.einsum("ia,izyx->azyx", c, source),
        torch.stack((fx, fy, fz)),
        rtol=2.0e-14,
        atol=2.0e-18,
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
        state,
        fx,
        zero,
        zero,
        ux,
        zero,
        zero,
    )

    assert float(source.sum()) == pytest.approx(0.0, abs=1.0e-19)


@pytest.mark.parametrize("chunk_size", (1, 4, 7, 19))
def test_direction_chunked_guo_force_matches_whole_lattice(
    chunk_size: int,
) -> None:
    torch.manual_seed(20260802)
    shape = (3, 4, 5)
    state = torch.randn((19, *shape), dtype=torch.float64)
    fields = [torch.randn(shape, dtype=torch.float64) for _ in range(6)]
    expected = guo_body_force_d3q19(state, *fields)

    actual = guo_body_force_d3q19(
        state,
        *fields,
        direction_chunk_size=chunk_size,
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("chunk_size", (0, 20, True))
def test_direction_chunked_guo_force_rejects_invalid_size(
    chunk_size: int,
) -> None:
    shape = (1, 1, 1)
    state = torch.zeros((19, *shape))
    zero = torch.zeros(shape)
    with pytest.raises(ValueError, match="direction_chunk_size"):
        guo_body_force_d3q19(
            state,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            direction_chunk_size=chunk_size,
        )


@pytest.mark.parametrize(("q", "lattice"), ((19, "D3Q19"), (27, "D3Q27")))
def test_solver_agnostic_wall_source_uses_the_same_moment_contract(
    q: int,
    lattice: str,
) -> None:
    shape = (2, 2, 3)
    dtype = torch.float64
    state = torch.zeros((q, *shape), dtype=dtype)
    ux = torch.full(shape, 0.06, dtype=dtype)
    uy = torch.full(shape, -0.01, dtype=dtype)
    uz = torch.full(shape, 0.02, dtype=dtype)
    fx = torch.full(shape, -2.0e-4, dtype=dtype)
    fy = torch.full(shape, 3.0e-5, dtype=dtype)
    fz = torch.full(shape, -1.0e-5, dtype=dtype)

    source = _apply_body_force(
        state,
        fx,
        fy,
        fz,
        lattice,
        ux=ux,
        uy=uy,
        uz=uz,
    )

    torch.testing.assert_close(
        source.sum(dim=0),
        torch.zeros(shape, dtype=dtype),
        rtol=0.0,
        atol=2.0e-18,
    )


def test_precomputed_wall_traction_is_not_divided_by_wall_distance_again() -> None:
    shape = (2, 4, 5)
    dtype = torch.float64
    rho = torch.ones(shape, dtype=dtype)
    ux = torch.full(shape, 0.04, dtype=dtype)
    zero = torch.zeros(shape, dtype=dtype)
    state = equilibrium3d(rho, ux, zero, zero)
    solid = torch.zeros(shape, dtype=torch.bool)
    solid[:, 0, :] = True
    near = _near_wall_mask(solid)
    u_tau = torch.full(shape, 0.01, dtype=dtype)
    y_plus = torch.full(shape, 50.0, dtype=dtype)

    corrected = wall_function(
        state.clone(),
        solid,
        u_tau,
        y_plus,
        lattice="D3Q19",
        y_val=0.5,
        rho=rho,
        ux=ux,
        uy=zero,
        uz=zero,
    )
    corrected_other_y = wall_function(
        state.clone(),
        solid,
        u_tau,
        y_plus,
        lattice="D3Q19",
        y_val=2.0,
        rho=rho,
        ux=ux,
        uy=zero,
        uz=zero,
    )
    source = corrected - state
    injected_x_momentum = torch.einsum(
        "i,izyx->",
        C19[:, 0].to(dtype=dtype),
        source,
    )

    assert float(injected_x_momentum) == pytest.approx(
        -(0.01**2) * int(near.sum()),
        abs=2.0e-15,
    )
    torch.testing.assert_close(corrected, corrected_other_y)
