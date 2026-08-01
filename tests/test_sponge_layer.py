from __future__ import annotations

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.sponge_layer import (
    apply_equilibrium_difference_sponge,
    build_sponge_sigma_3d,
    smoothstep5,
)


def test_fifth_order_ramp_has_exact_endpoints() -> None:
    x = torch.tensor([0.0, 0.5, 1.0])
    y = smoothstep5(x)
    assert torch.equal(y[[0, 2]], torch.tensor([0.0, 1.0]))
    assert y[1].item() == 0.5


def test_sponge_is_zero_interior_and_strongest_at_selected_face() -> None:
    sigma = build_sponge_sigma_3d(
        (9, 11, 13), width=3, max_strength=0.2, faces=("x+",),
    )
    assert sigma[4, 5, 6].item() == 0.0
    assert torch.isclose(sigma[4, 5, -1], torch.tensor(0.2))
    assert sigma[4, 5, -2].item() < sigma[4, 5, -1].item()


def test_target_equilibrium_is_fixed_point() -> None:
    shape = (5, 7, 9)
    rho = torch.ones(shape)
    ux = torch.full(shape, 0.04)
    zero = torch.zeros(shape)
    f = equilibrium3d(rho, ux, zero, zero)
    sigma = torch.rand(shape) * 0.2
    out = apply_equilibrium_difference_sponge(
        f, sigma, velocity_target=(0.04, 0.0, 0.0),
    )
    assert torch.allclose(out, f, atol=1e-7, rtol=0.0)


def test_sponge_reduces_macroscopic_velocity_perturbation() -> None:
    shape = (5, 7, 9)
    rho = torch.ones(shape)
    ux = torch.full(shape, 0.08)
    zero = torch.zeros(shape)
    f = equilibrium3d(rho, ux, zero, zero)
    sigma = torch.full(shape, 0.25)
    out = apply_equilibrium_difference_sponge(
        f, sigma, velocity_target=(0.04, 0.0, 0.0),
    )
    _, ux_out, _, _ = macroscopic3d(out)
    assert torch.allclose(ux_out, torch.full_like(ux_out, 0.07), atol=2e-7)
