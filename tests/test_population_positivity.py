from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d3q19 import C, equilibrium3d
from tensorlbm.population_positivity import limit_nonequilibrium_for_positivity


def test_positive_equilibrium_is_unchanged() -> None:
    shape = (4, 5, 6)
    rho = torch.ones(shape, dtype=torch.float64)
    ux = torch.full(shape, 0.04, dtype=torch.float64)
    zero = torch.zeros(shape, dtype=torch.float64)
    f = equilibrium3d(rho, ux, zero, zero)
    out, diagnostics = limit_nonequilibrium_for_positivity(f)
    assert out is f
    assert torch.equal(out, f)
    assert diagnostics.limited_cells == 0
    assert diagnostics.minimum_alpha == 1.0


def test_negative_nonequilibrium_is_limited_while_moments_are_preserved() -> None:
    shape = (3, 4, 5)
    rho = torch.ones(shape, dtype=torch.float64)
    zero = torch.zeros(shape, dtype=torch.float64)
    f = equilibrium3d(rho, zero, zero, zero)
    cell = (1, 2, 3)
    # A zero-mass, zero-momentum perturbation: opposite x directions gain
    # equally while the rest population loses their combined amount.
    delta = 0.2
    f[(0,) + cell] -= 2.0 * delta
    f[(1,) + cell] += delta
    f[(2,) + cell] += delta
    before_by_q = f.sum(dim=(1, 2, 3))
    before_mass = before_by_q.sum()
    before_momentum = (before_by_q[:, None] * C.to(f)).sum(dim=0)
    out, diagnostics = limit_nonequilibrium_for_positivity(f, floor=1e-10)
    after_by_q = out.sum(dim=(1, 2, 3))
    after_mass = after_by_q.sum()
    after_momentum = (after_by_q[:, None] * C.to(f)).sum(dim=0)
    assert float(out.min()) >= 1e-10
    assert diagnostics.limited_cells == 1
    assert 0.0 <= diagnostics.minimum_alpha < 1.0
    # D3Q19 weights are stored as float32 constants.
    assert after_mass.item() == pytest.approx(before_mass.item(), abs=5e-9)
    assert torch.allclose(after_momentum, before_momentum, atol=5e-9, rtol=0.0)


def test_invalid_floor_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        limit_nonequilibrium_for_positivity(torch.ones((19, 2, 2, 2)), floor=-1.0)
    with pytest.raises(ValueError, match="finite"):
        limit_nonequilibrium_for_positivity(
            torch.ones((19, 2, 2, 2)), floor=math.nan,
        )


def test_sparse_d3q27_limiter_preserves_mass_and_momentum() -> None:
    from tensorlbm.d3q27 import C as C27
    from tensorlbm.d3q27 import equilibrium27

    shape = (3, 4, 5)
    rho = torch.ones(shape, dtype=torch.float64)
    zero = torch.zeros(shape, dtype=torch.float64)
    f = equilibrium27(rho, zero, zero, zero)
    cell = (1, 2, 3)
    f[(0,) + cell] -= 0.4
    f[(1,) + cell] += 0.2
    f[(2,) + cell] += 0.2
    before = f.sum(dim=(1, 2, 3))
    out, diagnostics = limit_nonequilibrium_for_positivity(f, floor=1e-10)
    after = out.sum(dim=(1, 2, 3))
    assert diagnostics.limited_cells == 1
    assert after.sum().item() == pytest.approx(before.sum().item(), abs=5e-9)
    assert torch.allclose(
        (after[:, None] * C27.to(after)).sum(dim=0),
        (before[:, None] * C27.to(before)).sum(dim=0),
        atol=5e-9, rtol=0.0,
    )


def test_nonfinite_state_is_passed_to_caller_for_fail_closed_abort() -> None:
    f = torch.ones((19, 2, 2, 2))
    f[1, 0, 0, 0] = torch.nan
    out, diagnostics = limit_nonequilibrium_for_positivity(f)
    assert out is f
    assert math.isnan(diagnostics.minimum_population_before)
    assert math.isnan(diagnostics.minimum_alpha)
