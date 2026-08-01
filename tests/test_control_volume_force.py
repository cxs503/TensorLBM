from __future__ import annotations

import pytest
import torch

from tensorlbm.control_volume_force import (
    box_control_volume,
    observe_control_volume_force,
    streaming_momentum_import,
)
from tensorlbm.d3q19 import C, equilibrium3d
from tensorlbm.solver3d import stream3d


def _state(shape=(8, 9, 11), velocity=(0.03, -0.01, 0.007)):
    rho = torch.ones(shape, dtype=torch.float64)
    ux = torch.full(shape, velocity[0], dtype=torch.float64)
    uy = torch.full(shape, velocity[1], dtype=torch.float64)
    uz = torch.full(shape, velocity[2], dtype=torch.float64)
    return equilibrium3d(rho, ux, uy, uz)


def _cv(shape=(8, 9, 11)):
    return box_control_volume(shape, x0=2, x1=8, y0=2, y1=7, z0=2, z1=6)


def test_uniform_stream_has_zero_net_import_and_zero_observed_force() -> None:
    f = _state()
    cv = _cv()
    streamed = stream3d(f)
    result = observe_control_volume_force(f, streamed, f, cv)
    assert torch.allclose(result.momentum_import, torch.zeros(3, dtype=f.dtype), atol=1e-14)
    assert torch.allclose(result.fluid_momentum_change, torch.zeros(3, dtype=f.dtype), atol=1e-14)
    assert torch.allclose(result.force_on_body, torch.zeros(3, dtype=f.dtype), atol=1e-14)


def test_internal_momentum_source_is_reported_as_opposite_body_force() -> None:
    f = _state(velocity=(0.0, 0.0, 0.0))
    cv = _cv()
    f_new = f.clone()
    cell = (3, 4, 5)
    impulse = 0.0125
    # Transfer population from -x to +x at one interior cell: ΔP_x=2*impulse.
    plus_x, minus_x = 1, 2
    f_new[(plus_x,) + cell] += impulse
    f_new[(minus_x,) + cell] -= impulse
    result = observe_control_volume_force(f, f_new, f, cv)
    assert result.momentum_import[0].item() == pytest.approx(0.0, abs=1e-14)
    assert result.fluid_momentum_change[0].item() == pytest.approx(2 * impulse)
    assert result.force_on_body[0].item() == pytest.approx(-2 * impulse)


def test_single_population_crossing_has_exact_kinetic_flux() -> None:
    f = torch.zeros((19, 8, 9, 11), dtype=torch.float64)
    cv = _cv()
    # Direction +x source just outside x0 enters the CV.
    f[1, 3, 4, 1] = 0.2
    imported = streaming_momentum_import(f, cv)
    assert torch.equal(imported, 0.2 * C[1].to(dtype=f.dtype))


def test_control_volume_must_be_interior() -> None:
    f = _state()
    bad = torch.zeros(f.shape[1:], dtype=torch.bool)
    bad[:, 2:5, 2:5] = True
    with pytest.raises(ValueError, match="strictly interior"):
        streaming_momentum_import(f, bad)


def test_periodic_axis_control_volume_may_span_complete_axis() -> None:
    shape = (3, 9, 11)
    f = _state(shape=shape)
    cv = box_control_volume(
        shape, x0=2, x1=8, y0=2, y1=7, z0=0, z1=3,
        periodic_axes=("z",),
    )
    streamed = stream3d(f)
    result = observe_control_volume_force(
        f, streamed, f, cv, periodic_axes=("z",),
    )
    assert torch.allclose(result.force_on_body, torch.zeros(3, dtype=f.dtype), atol=1e-14)
