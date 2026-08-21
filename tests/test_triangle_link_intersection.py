from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import C
from tensorlbm.triangle_link_intersection import refine_bfl_q_with_triangles


def _square_at_x(x: float) -> tuple[torch.Tensor, torch.Tensor]:
    vertices = torch.tensor(
        [
            [x, 0.0, 0.0],
            [x, 4.0, 0.0],
            [x, 4.0, 4.0],
            [x, 0.0, 4.0],
        ]
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
    return vertices, faces


def test_exact_axis_link_fraction_replaces_halfway_fallback() -> None:
    shape = (5, 5, 5)
    mask = torch.zeros((19, *shape), dtype=torch.bool)
    q = torch.full(mask.shape, 0.5)
    target = torch.zeros(shape, dtype=torch.bool)
    direction = int(
        torch.nonzero(
            (C[:, 0] == 1) & (C[:, 1] == 0) & (C[:, 2] == 0),
        )[0]
    )
    mask[direction, 2, 2, 1] = True
    target[2, 2, 2] = True
    vertices, faces = _square_at_x(1.25)

    refined, diagnostics = refine_bfl_q_with_triangles(
        mask,
        q,
        vertices,
        faces,
        target_solid=target,
    )

    assert refined[direction, 2, 2, 1] == pytest.approx(0.25)
    assert diagnostics.target_links == 1
    assert diagnostics.resolved_links == 1
    assert diagnostics.missing_links == 0
    assert diagnostics.minimum_q == pytest.approx(0.25)
    assert diagnostics.maximum_q == pytest.approx(0.25)
    assert torch.equal(refined[~mask], q[~mask])


def test_diagonal_link_q_is_parameter_fraction_not_euclidean_distance() -> None:
    shape = (5, 5, 5)
    mask = torch.zeros((19, *shape), dtype=torch.bool)
    q = torch.full(mask.shape, 0.5)
    direction = int(
        torch.nonzero(
            (C[:, 0] == 1) & (C[:, 1] == 1) & (C[:, 2] == 0),
        )[0]
    )
    mask[direction, 2, 1, 1] = True
    vertices, faces = _square_at_x(1.75)

    refined, diagnostics = refine_bfl_q_with_triangles(
        mask,
        q,
        vertices,
        faces,
    )

    assert diagnostics.missing_links == 0
    assert refined[direction, 2, 1, 1] == pytest.approx(0.75)


def test_missing_selected_cad_intersection_fails_closed() -> None:
    shape = (5, 5, 5)
    mask = torch.zeros((19, *shape), dtype=torch.bool)
    q = torch.full(mask.shape, 0.5)
    direction = int(torch.nonzero(C[:, 0] == 1)[0])
    mask[direction, 2, 2, 1] = True
    vertices, faces = _square_at_x(3.5)

    with pytest.raises(ValueError, match="did not intersect 1 of 1"):
        refine_bfl_q_with_triangles(mask, q, vertices, faces)


def test_target_component_limits_which_links_are_refined() -> None:
    shape = (5, 5, 6)
    mask = torch.zeros((19, *shape), dtype=torch.bool)
    q = torch.full(mask.shape, 0.5)
    target = torch.zeros(shape, dtype=torch.bool)
    direction = int(
        torch.nonzero(
            (C[:, 0] == 1) & (C[:, 1] == 0) & (C[:, 2] == 0),
        )[0]
    )
    mask[direction, 2, 2, 1] = True
    mask[direction, 2, 2, 3] = True
    target[2, 2, 2] = True
    vertices, faces = _square_at_x(1.25)

    refined, diagnostics = refine_bfl_q_with_triangles(
        mask,
        q,
        vertices,
        faces,
        target_solid=target,
    )

    assert diagnostics.target_links == 1
    assert refined[direction, 2, 2, 1] == pytest.approx(0.25)
    assert refined[direction, 2, 2, 3] == pytest.approx(0.5)
