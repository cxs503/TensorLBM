from __future__ import annotations

import pytest
import torch

from tensorlbm import (
    apparent_viscosity_power_law,
    collide_power_law_bgk,
    equilibrium,
    strain_rate_magnitude_2d,
)


def test_strain_rate_magnitude_linear_shear() -> None:
    ny, nx = 32, 8
    y = torch.arange(ny, dtype=torch.float32).unsqueeze(1).expand(ny, nx)
    ux = 0.02 * y
    uy = torch.zeros_like(ux)

    gamma = strain_rate_magnitude_2d(ux, uy)
    interior = gamma[1:-1, 1:-1]
    assert torch.allclose(interior, torch.full_like(interior, 0.02), atol=2e-4)


def test_apparent_viscosity_power_law_shear_thinning() -> None:
    shear_rate = torch.tensor([0.01, 0.1, 1.0], dtype=torch.float32)
    nu = apparent_viscosity_power_law(shear_rate, consistency_index=0.05, flow_index=0.6)
    assert nu[0] > nu[1] > nu[2]


def test_apparent_viscosity_power_law_bounds() -> None:
    shear_rate = torch.tensor([0.0, 1e-8, 1e2], dtype=torch.float32)
    nu = apparent_viscosity_power_law(
        shear_rate,
        consistency_index=0.05,
        flow_index=0.4,
        nu_min=1e-3,
        nu_max=1e-1,
    )
    assert torch.all(nu >= 1e-3)
    assert torch.all(nu <= 1e-1)


def test_collide_power_law_bgk_identity_at_equilibrium() -> None:
    ny, nx = 16, 20
    rho = torch.ones((ny, nx), dtype=torch.float32)
    ux = torch.full_like(rho, 0.03)
    uy = torch.full_like(rho, -0.01)
    feq = equilibrium(rho, ux, uy)

    f_out = collide_power_law_bgk(feq, consistency_index=0.01, flow_index=0.8)
    assert torch.allclose(f_out, feq, atol=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"consistency_index": 0.0, "flow_index": 1.0}, "consistency_index"),
        ({"consistency_index": 0.1, "flow_index": 0.0}, "flow_index"),
    ],
)
def test_apparent_viscosity_power_law_invalid_params(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _ = apparent_viscosity_power_law(torch.tensor([1.0]), **kwargs)
