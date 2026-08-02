from __future__ import annotations

import pytest
import torch

from tensorlbm.control_volume_force import (
    assess_nested_control_volume_invariance,
    box_control_volume,
    fluid_momentum,
    fluid_momentum_change,
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


def test_large_float32_cv_accumulates_local_change_before_reduction() -> None:
    shape = (3, 130, 130)
    old = torch.full((19, *shape), 0.055, dtype=torch.float32)
    new = old.clone()
    cv = box_control_volume(
        shape, x0=2, x1=128, y0=2, y1=128, z0=0, z1=3,
        periodic_axes=("z",),
    )
    new[1, cv] += 1.0e-7
    expected = (new[1, cv] - old[1, cv]).sum(dtype=torch.float64)

    stable = fluid_momentum_change(
        old, new, cv, periodic_axes=("z",),
    )[0]
    assert stable.item() == pytest.approx(float(expected), abs=1e-14)


def test_directionwise_observers_match_q_wide_reference() -> None:
    torch.manual_seed(20260802)
    shape = (6, 7, 9)
    old = 0.02 + torch.rand((19, *shape), dtype=torch.float64)
    new = old + 1.0e-6 * torch.randn_like(old)
    cv = box_control_volume(
        shape, x0=2, x1=7, y0=2, y1=5, z0=2, z1=4,
    )
    solid = torch.zeros(shape, dtype=torch.bool)
    solid[2, 3, 4] = True
    owned = cv & ~solid
    c = C.to(dtype=torch.float64)
    inventory = old[:, owned].sum(dim=1)
    change = (new[:, owned] - old[:, owned]).sum(dim=1)
    expected_momentum = (inventory[:, None] * c).sum(dim=0)
    expected_change = (change[:, None] * c).sum(dim=0)

    actual_momentum = fluid_momentum(old, cv, solid=solid)
    actual_change = fluid_momentum_change(old, new, cv, solid=solid)

    torch.testing.assert_close(
        actual_momentum, expected_momentum, rtol=0.0, atol=2.0e-14,
    )
    torch.testing.assert_close(
        actual_change, expected_change, rtol=0.0, atol=2.0e-20,
    )


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


def test_nested_control_volume_assessment_requires_two_consistent_observers() -> None:
    assessment = assess_nested_control_volume_invariance(100.0, [100.4, 99.6])
    assert assessment.maximum_difference_pct == pytest.approx(0.4)
    assert assessment.meets(1.0)
    assert not assess_nested_control_volume_invariance(100.0, [100.1]).meets(1.0)
    assert not assess_nested_control_volume_invariance(
        100.0, [float("nan"), 100.0],
    ).meets(1.0)


def test_curved_moving_slip_impulse_is_invariant_across_nested_control_volumes() -> None:
    """The first off-lattice slip impulse must close before wake evolution."""
    from tensorlbm.bfl_d3q19 import bouzidi_bounce_back_d3q19
    from tensorlbm.boundaries3d import sphere_mask
    from tensorlbm.interpolated_bc import compute_q_sphere
    from tensorlbm.wall_model import compute_bfl_link_normal

    shape = (32, 32, 48)
    nx, ny, nz = 48, 32, 32
    cx, cy, cz, radius = 18.0, 16.0, 16.0, 6.0
    rho = torch.ones(shape, dtype=torch.float64)
    ux = torch.full(shape, 0.06, dtype=torch.float64)
    zero = torch.zeros(shape, dtype=torch.float64)
    old = equilibrium3d(rho, ux, zero, zero)
    solid = sphere_mask(nx, ny, nz, cx, cy, cz, radius, device=torch.device("cpu"))
    masks, q = compute_q_sphere(
        nx, ny, nz, cx, cy, cz, radius, device=torch.device("cpu"),
    )
    nx_n, ny_n, nz_n = compute_bfl_link_normal(masks)
    normal_speed = ux * nx_n
    wall_velocity = (
        ux - normal_speed * nx_n,
        -normal_speed * ny_n,
        -normal_speed * nz_n,
    )
    new, bfl_force = bouzidi_bounce_back_d3q19(
        old.clone(), old, masks, q,
        wall_velocity=wall_velocity, wall_density=rho, return_force=True,
    )
    tight = box_control_volume(shape, x0=9, x1=28, y0=7, y1=26, z0=7, z1=26)
    wide = box_control_volume(shape, x0=6, x1=31, y0=4, y1=29, z0=4, z1=29)
    tight_force = observe_control_volume_force(old, new, old, tight, solid=solid).force_on_body
    wide_force = observe_control_volume_force(old, new, old, wide, solid=solid).force_on_body
    assert torch.allclose(tight_force, wide_force, atol=1e-11, rtol=0.0)
    assert tight_force[0].item() == pytest.approx(bfl_force[0], abs=1e-8)
