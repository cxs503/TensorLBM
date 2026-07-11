"""Reference invariants for the current D3Q27 cumulant collision.

The oracle here is the D3Q27 discrete-velocity definition itself: collision
must preserve the raw moments m000, m100, m010 and m001, and the discrete
second-order equilibrium must be a collision fixed point.  Raw moments are
formed independently of ``macroscopic27`` so a shared recovery helper cannot
mask a conservation regression.
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm.cumulant import collide_cumulant_d3q27
from tensorlbm.d3q27 import equilibrium27


# D3Q27 velocities in the documented population ordering.  Keep this local to
# the reference test rather than calling the collision implementation's moment
# recovery path.
_D3Q27_C = torch.tensor(
    [
        [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
        [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, 1],
        [1, 0, -1], [-1, 0, -1], [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
        [1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1], [1, 1, -1], [-1, 1, -1],
        [1, -1, -1], [-1, -1, -1],
    ],
    dtype=torch.float64,
)


def _conserved_raw_moments(f: torch.Tensor) -> torch.Tensor:
    """Return m000, m100, m010, m001 directly from D3Q27 populations."""
    directions = _D3Q27_C.to(device=f.device, dtype=f.dtype)
    flattened = f.reshape(27, -1)
    mass = flattened.sum(dim=0)
    momentum = directions.T @ flattened
    return torch.cat((mass.unsqueeze(0), momentum), dim=0)


def _non_equilibrium_state() -> torch.Tensor:
    torch.manual_seed(20260711)
    shape = (2, 3, 4)
    rho = 0.9 + 0.2 * torch.rand(shape, dtype=torch.float64)
    ux = 0.04 * (2.0 * torch.rand(shape, dtype=torch.float64) - 1.0)
    uy = 0.04 * (2.0 * torch.rand(shape, dtype=torch.float64) - 1.0)
    uz = 0.04 * (2.0 * torch.rand(shape, dtype=torch.float64) - 1.0)
    feq = equilibrium27(rho, ux, uy, uz)

    # A bounded, deliberately non-equilibrium perturbation. Conservation is
    # measured before/after rather than imposed by the perturbation.
    return feq + 1.0e-3 * torch.randn_like(feq)


@pytest.mark.parametrize("tau", [0.55, 0.83, 1.7])
@pytest.mark.parametrize("C_s", [0.0, 0.12])
def test_d3q27_cumulant_preserves_independent_conserved_raw_moments(tau: float, C_s: float):
    before = _non_equilibrium_state()
    after = collide_cumulant_d3q27(
        before, tau=tau, omega_b=1.13, omega_odd=0.71, omega_even=1.37, C_s=C_s
    )

    torch.testing.assert_close(
        _conserved_raw_moments(after), _conserved_raw_moments(before), rtol=2e-6, atol=2e-7
    )


@pytest.mark.parametrize("tau", [0.55, 0.83, 1.7])
@pytest.mark.parametrize("C_s", [0.0, 0.12])
def test_d3q27_cumulant_keeps_discrete_equilibrium_fixed(tau: float, C_s: float):
    torch.manual_seed(20260712)
    shape = (2, 3, 4)
    rho = 0.9 + 0.2 * torch.rand(shape, dtype=torch.float64)
    ux = 0.06 * (2.0 * torch.rand(shape, dtype=torch.float64) - 1.0)
    uy = 0.06 * (2.0 * torch.rand(shape, dtype=torch.float64) - 1.0)
    uz = 0.06 * (2.0 * torch.rand(shape, dtype=torch.float64) - 1.0)
    equilibrium = equilibrium27(rho, ux, uy, uz)

    after = collide_cumulant_d3q27(
        equilibrium, tau=tau, omega_b=1.13, omega_odd=0.71, omega_even=1.37, C_s=C_s
    )

    torch.testing.assert_close(after, equilibrium, rtol=2e-6, atol=2e-7)
