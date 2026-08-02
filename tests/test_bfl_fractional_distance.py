from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.bfl_common import (
    bfl_bounce_back_common,
    compute_q_cylinder_common,
    compute_q_sphere_common,
)
from tensorlbm.bfl_d3q19 import (
    bouzidi_bounce_back_d3q19,
    compute_q_cylinder_d3q19,
)
from tensorlbm.interpolated_bc import compute_q_circle, compute_q_sphere
from tensorlbm.interpolated_bc_common import compute_q_sphere_27
from tensorlbm.interpolated_bc_ellipsoid import compute_q_ellipsoid

DEVICE = torch.device("cpu")


def test_d2_circle_diagonal_q_is_ray_fraction() -> None:
    mask, q = compute_q_circle(9, 9, 4.0, 4.0, 2.0, DEVICE)
    expected = 2.0 - math.sqrt(2.0)
    assert mask[7, 6, 6]
    assert float(q[7, 6, 6]) == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("common", (False, True))
def test_d3q19_sphere_face_diagonal_q_is_ray_fraction(common: bool) -> None:
    function = compute_q_sphere_common if common else compute_q_sphere
    mask, q = function(9, 9, 9, 4.0, 4.0, 4.0, 2.0, DEVICE)
    expected = 2.0 - math.sqrt(2.0)
    assert mask[8, 4, 6, 6]
    assert float(q[8, 4, 6, 6]) == pytest.approx(expected, abs=1e-7)


def test_common_extruded_cylinder_diagonal_q_is_ray_fraction() -> None:
    mask, q = compute_q_cylinder_common(
        9, 9, 3, 4.0, 4.0, 2.0, DEVICE, lattice="D3Q19",
    )
    expected = 2.0 - math.sqrt(2.0)
    assert mask[8, 1, 6, 6]
    assert float(q[8, 1, 6, 6]) == pytest.approx(expected, abs=1e-7)


def test_vector_common_matches_admitted_d3q19_cylinder_bfl() -> None:
    reference_mask, reference_q = compute_q_cylinder_d3q19(
        9, 9, 3, 4.0, 4.0, 2.0, DEVICE,
    )
    common_mask, common_q = compute_q_cylinder_common(
        9, 9, 3, 4.0, 4.0, 2.0, DEVICE, lattice="D3Q19",
    )
    assert torch.equal(common_mask, reference_mask)
    torch.testing.assert_close(common_q, reference_q)

    torch.manual_seed(20260802)
    previous = 0.02 + 0.05 * torch.rand(19, 3, 9, 9)
    streamed = 0.02 + 0.05 * torch.rand(19, 3, 9, 9)
    reference = bouzidi_bounce_back_d3q19(
        streamed, previous, reference_mask, reference_q,
    )
    common = bfl_bounce_back_common(
        streamed, previous, common_mask, common_q, lattice="D3Q19",
    )
    torch.testing.assert_close(common, reference, rtol=0.0, atol=0.0)


def test_d3q27_body_diagonal_q_is_ray_fraction() -> None:
    mask, q = compute_q_sphere_27(9, 9, 9, 4.0, 4.0, 4.0, 2.0, DEVICE)
    expected = 2.0 - 2.0 / math.sqrt(3.0)
    assert mask[26, 6, 6, 6]
    assert float(q[26, 6, 6, 6]) == pytest.approx(expected, abs=1e-7)


def test_ellipsoid_spherical_limit_diagonal_q_is_ray_fraction() -> None:
    mask, q = compute_q_ellipsoid(
        9, 9, 9, 4.0, 4.0, 4.0, 2.0, 2.0, 0.0, DEVICE,
    )
    expected = 2.0 - math.sqrt(2.0)
    assert mask[8, 4, 6, 6]
    assert float(q[8, 4, 6, 6]) == pytest.approx(expected, abs=1e-7)
