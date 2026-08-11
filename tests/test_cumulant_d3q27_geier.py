"""Correctness contract for the Geier 2015 D3Q27 cumulant operator.

The oracle here is the *definition* of a cumulant rather than a stored
baseline: ``C = log K`` with ``K(0) = ρ``.  Two consequences are checked
directly, because they are exactly what the legacy
``collide_cumulant_d3q27`` path fails to satisfy:

1. For a product-form equilibrium every cumulant of order ≥ 4 vanishes.
2. The transform is exactly invertible, so the discrete equilibrium is a
   fixed point of the collision for *any* set of relaxation rates.

The legacy transform yields ``C^eq_222 = 1/9`` instead of 0, which is why it
is documented as a central-moment collision rather than a cumulant one; the
last test in this module pins that discrepancy so the distinction cannot be
silently erased.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.cumulant import (
    _central_to_cumulant,
    _central_to_cumulant_geier,
    _cumulant_to_central_geier,
    _get_matrices,
    _to_central,
    collide_cumulant_geier_d3q27,
)
from tensorlbm.d3q27 import equilibrium27, macroscopic27

# D3Q27 velocities, kept local so a regression in the collision module's own
# moment helpers cannot mask a conservation failure.
_D3Q27_C = torch.tensor(
    [
        [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1], [1, 1, 0], [-1, 1, 0], [1, -1, 0],
        [-1, -1, 0], [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1], [1, 1, 1],
        [-1, 1, 1], [1, -1, 1], [-1, -1, 1], [1, 1, -1], [-1, 1, -1],
        [1, -1, -1], [-1, -1, -1],
    ],
    dtype=torch.float64,
)

_FOURTH = range(17, 23)
_FIFTH = range(23, 26)
_SIXTH = (26,)


def _conserved_raw_moments(f: torch.Tensor) -> torch.Tensor:
    directions = _D3Q27_C.to(device=f.device, dtype=f.dtype)
    flat = f.reshape(27, -1)
    return torch.cat((flat.sum(dim=0).unsqueeze(0), directions.T @ flat), dim=0)


def _central_moments(f: torch.Tensor) -> torch.Tensor:
    rho, ux, uy, uz = macroscopic27(f)
    matrix, _ = _get_matrices(f.device, f.dtype)
    m = (matrix @ f.reshape(27, -1)).reshape(27, *f.shape[1:])
    return _to_central(m, ux, uy, uz)


def _equilibrium(shape=(2, 3, 4), *, seed=20260811, amplitude=0.12):
    torch.manual_seed(seed)
    rho = 0.9 + 0.2 * torch.rand(shape, dtype=torch.float64)
    ux = amplitude * (2.0 * torch.rand(shape, dtype=torch.float64) - 1.0)
    uy = amplitude * (2.0 * torch.rand(shape, dtype=torch.float64) - 1.0)
    uz = amplitude * (2.0 * torch.rand(shape, dtype=torch.float64) - 1.0)
    return equilibrium27(rho, ux, uy, uz)


def _perturbed(shape=(2, 3, 4), *, seed=20260811, scale=1.0e-3):
    feq = _equilibrium(shape, seed=seed)
    torch.manual_seed(seed + 1)
    return feq + scale * torch.randn_like(feq)


def test_transform_round_trip_is_exact():
    """κ → C → κ must recover the input; the collision relies on this."""
    k = _central_moments(_perturbed())
    recovered = _cumulant_to_central_geier(_central_to_cumulant_geier(k))
    assert torch.allclose(recovered, k, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("speed", [0.0, 0.1, 0.2])
def test_equilibrium_cumulants_above_third_order_vanish(speed):
    """The defining property of cumulants for a product-form equilibrium."""
    shape = (2, 3, 4)
    rho = torch.ones(shape, dtype=torch.float64)
    ux = torch.full(shape, speed, dtype=torch.float64)
    zero = torch.zeros_like(ux)
    keq = _central_moments(equilibrium27(rho, ux, zero, zero))
    ceq = _central_to_cumulant_geier(keq)
    for index in (*_FOURTH, *_FIFTH, *_SIXTH):
        assert ceq[index].abs().max() < 1e-12, f"C^eq[{index}] should vanish"


@pytest.mark.parametrize("tau", [0.55, 0.83, 1.7])
@pytest.mark.parametrize("omega_even", [0.4, 1.0, 1.6])
def test_discrete_equilibrium_is_a_fixed_point(tau, omega_even):
    """Holds for any relaxation rates because C^eq is built from equilibrium27."""
    feq = _equilibrium()
    out = collide_cumulant_geier_d3q27(
        feq, tau=tau, omega_b=1.3, omega_odd=0.7, omega_even=omega_even
    )
    assert torch.allclose(out, feq, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("tau", [0.55, 0.83, 1.7])
def test_collision_conserves_mass_and_momentum(tau):
    f = _perturbed()
    before = _conserved_raw_moments(f)
    after = _conserved_raw_moments(
        collide_cumulant_geier_d3q27(
            f, tau=tau, omega_b=1.2, omega_odd=1.1, omega_even=0.9
        )
    )
    assert torch.allclose(after, before, rtol=0.0, atol=1e-12)


def test_collision_is_finite_and_shape_preserving():
    f = _perturbed()
    out = collide_cumulant_geier_d3q27(f, tau=0.5005)
    assert out.shape == f.shape
    assert torch.isfinite(out).all()


def test_relaxation_actually_damps_the_non_equilibrium_part():
    """A no-op transform pair would silently pass every test above."""
    f = _perturbed(scale=1.0e-3)
    rho, ux, uy, uz = macroscopic27(f)
    before = (f - equilibrium27(rho, ux, uy, uz)).abs().max()
    out = collide_cumulant_geier_d3q27(f, tau=0.55, omega_b=1.0,
                                       omega_odd=1.0, omega_even=1.0)
    rho, ux, uy, uz = macroscopic27(out)
    after = (out - equilibrium27(rho, ux, uy, uz)).abs().max()
    assert after < before


def test_legacy_transform_is_not_a_cumulant_transform():
    """Pins the documented discrepancy between the legacy and Geier paths.

    Applying the legacy formulas to a *full* equilibrium leaves a 6th-order
    residual of ρ/9, whereas a genuine cumulant transform must return zero.
    This guards the docstring claim in ``_central_to_cumulant``.
    """
    shape = (2, 2, 2)
    rho = torch.ones(shape, dtype=torch.float64)
    ux = torch.full(shape, 0.1, dtype=torch.float64)
    zero = torch.zeros_like(ux)
    keq = _central_moments(equilibrium27(rho, ux, zero, zero))

    legacy = _central_to_cumulant(keq)
    geier = _central_to_cumulant_geier(keq)

    assert geier[26].abs().max() < 1e-12
    assert legacy[26].abs().max() > 0.1
    assert torch.allclose(
        legacy[26], torch.full(shape, 1.0 / 9.0, dtype=torch.float64),
        rtol=0.0, atol=1e-6,
    )
