"""Unit tests for the union (block-ring) control volume.

Directive 2026-10-11 module item 5: the static-block CV observer and the
shell force ledger must share one CV face definition.  The union of
strictly interior boxes is that definition for surface-shell layouts; the
discrete observer applies unchanged and is additive over disjoint members.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.control_volume_force import (
    box_control_volume,
    observe_control_volume_force,
    union_box_control_volume,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import stream3d

_SHAPE = (18, 18, 22)  # (nz, ny, nx)


def test_single_box_union_equals_box_control_volume() -> None:
    box = (4, 10, 3, 9, 5, 12)  # (x0, x1, y0, y1, z0, z1) half-open
    union = union_box_control_volume(_SHAPE, [box])
    reference = box_control_volume(_SHAPE, x0=4, x1=10, y0=3, y1=9, z0=5, z1=12)
    assert torch.equal(union, reference)


def test_two_tiling_boxes_equal_envelope_box() -> None:
    left = (2, 8, 3, 9, 5, 12)
    right = (8, 14, 3, 9, 5, 12)
    union = union_box_control_volume(_SHAPE, [left, right])
    envelope = box_control_volume(_SHAPE, x0=2, x1=14, y0=3, y1=9, z0=5, z1=12)
    assert torch.equal(union, envelope)
    assert int(union.sum().item()) == int(envelope.sum().item())


def test_overlapping_boxes_are_or_merged() -> None:
    first = (2, 8, 2, 8, 4, 10)
    second = (6, 12, 5, 11, 7, 13)
    union = union_box_control_volume(_SHAPE, [first, second])
    expected = box_control_volume(_SHAPE, x0=2, x1=8, y0=2, y1=8, z0=4, z1=10) | box_control_volume(
        _SHAPE, x0=6, x1=12, y0=5, y1=11, z0=7, z1=13
    )
    assert torch.equal(union, expected)


def test_union_validates_each_member() -> None:
    with pytest.raises(ValueError, match="at least one"):
        union_box_control_volume(_SHAPE, [])
    with pytest.raises(ValueError, match="strictly interior"):
        union_box_control_volume(_SHAPE, [(0, 5, 3, 9, 5, 12)])
    with pytest.raises(ValueError, match="strictly interior"):
        union_box_control_volume(_SHAPE, [(4, 10, 3, 9, 5, 22)])


def _one_collide_stream_step(shape, tau=0.62):
    rho = torch.full(shape, 1.02)
    ux = torch.full(shape, 0.05)
    uy = torch.full(shape, -0.01)
    uz = torch.full(shape, 0.02)
    f_old = equilibrium3d(rho, ux, uy, uz)
    r, u_x, u_y, u_z = macroscopic3d(f_old)
    feq = equilibrium3d(r, u_x, u_y, u_z)
    post = f_old - (f_old - feq) / tau
    f_new = stream3d(post)
    return f_old, f_new, post


def test_observer_additive_over_disjoint_union_members() -> None:
    shape = _SHAPE
    solid = torch.zeros(shape, dtype=torch.bool)
    solid[7:10, 7:10, 7:10] = True  # body inside the first member box
    f_old, f_new, post = _one_collide_stream_step(shape)
    box_a = (4, 11, 4, 11, 4, 11)
    box_b = (14, 20, 12, 16, 6, 13)
    result_a = observe_control_volume_force(
        f_old,
        f_new,
        post,
        box_control_volume(shape, x0=4, x1=11, y0=4, y1=11, z0=4, z1=11),
        solid=solid,
        periodic_axes=(),
    )
    result_b = observe_control_volume_force(
        f_old,
        f_new,
        post,
        box_control_volume(shape, x0=14, x1=20, y0=12, y1=16, z0=6, z1=13),
        solid=solid,
        periodic_axes=(),
    )
    result_union = observe_control_volume_force(
        f_old,
        f_new,
        post,
        union_box_control_volume(shape, [box_a, box_b]),
        solid=solid,
        periodic_axes=(),
    )
    for component in range(3):
        total = float(result_a.force_on_body[component]) + float(result_b.force_on_body[component])
        assert float(result_union.force_on_body[component]) == pytest.approx(
            total, rel=1e-3, abs=1e-3
        )
    # the union's discrete flux only sees true outer faces: the import term
    # is additive as well, which is the face-definition alignment claim
    for component in range(3):
        total = float(result_a.momentum_import[component]) + float(
            result_b.momentum_import[component]
        )
        assert float(result_union.momentum_import[component]) == pytest.approx(
            total, rel=1e-3, abs=1e-3
        )
