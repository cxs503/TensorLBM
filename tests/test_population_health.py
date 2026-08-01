from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.population_health import inspect_population_health


def test_uniform_equilibrium_health_is_reduced_to_scalars() -> None:
    rho = torch.full((3, 4, 5), 1.02, dtype=torch.float64)
    ux = torch.full_like(rho, 0.04)
    zero = torch.zeros_like(rho)
    health = inspect_population_health(equilibrium3d(rho, ux, zero, zero))

    assert health.finite is True
    assert health.minimum_population > 0.0
    assert health.maximum_density == pytest.approx(1.02, abs=2e-8)
    assert health.minimum_density == pytest.approx(1.02, abs=2e-8)
    assert health.maximum_speed == pytest.approx(0.04, abs=2e-8)
    assert health.maximum_speed_index_zyx is not None
    assert health.to_dict()["finite"] is True


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_population_is_reported_without_macroscopic_claims(bad: float) -> None:
    f = torch.ones((19, 2, 2, 2))
    f[3, 1, 1, 1] = bad
    health = inspect_population_health(f)

    assert health.finite is False
    assert health.minimum_density is None
    assert health.maximum_density is None
    assert health.maximum_speed is None
    assert health.maximum_speed_index_zyx is None


def test_population_health_rejects_wrong_layout() -> None:
    with pytest.raises(ValueError, match="shape"):
        inspect_population_health(torch.ones((9, 3, 4)))
    with pytest.raises(TypeError, match="floating"):
        inspect_population_health(torch.ones((19, 2, 2, 2), dtype=torch.int64))
