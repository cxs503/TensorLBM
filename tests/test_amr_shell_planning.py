"""Contract tests for the shared shell+wake refinement planner (amr_shell_planning.py).

The planner is used verbatim by the single-layer shell runner and the nested
L1-shell runner, so the returned box must:

* be strictly inside the coarse grid and at least 3 cells on every axis
  (StaticBlockAMR3D requires a non-degenerate 2:1 block);
* cover every refined cell after padding (the coarse-fine interface must stay
  off the solid shell surface);
* grow monotonically with the shell margin / wake extension.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.boundaries3d import sphere_mask


def _sphere(nx: int = 96, ny: int = 64, nz: int = 64, radius: float = 6.0) -> torch.Tensor:
    return sphere_mask(
        nx,
        ny,
        nz,
        nx * 0.5,
        ny / 2.0,
        nz / 2.0,
        radius,
        device=torch.device("cpu"),
    )


def test_plan_box_is_valid_and_covers_the_shell() -> None:
    solid = _sphere()
    plan = plan_body_shell_box(solid, shell_margin=8, wake_cells=25, pad=2)
    box = plan.box
    nz, ny, nx = solid.shape
    # strictly interior + non-degenerate
    assert 1 <= box.x0 < box.x1 <= nx - 1
    assert 1 <= box.y0 < box.y1 <= ny - 1
    assert 1 <= box.z0 < box.z1 <= nz - 1
    assert min(box.x1 - box.x0, box.y1 - box.y0, box.z1 - box.z0) >= 3
    # the padded box contains every refined coarse cell (it is a superset
    # of the refined region, padded so the interface stays off the shell)
    assert int(plan.refine_cells) > 0
    refined = plan.shell_mask | plan.wake_mask
    assert int(refined.sum().item()) == plan.refine_cells
    assert bool((refined & ~solid).any())  # the shell is fluid-only by construction
    box_slice = refined[box.z0 : box.z1, box.y0 : box.y1, box.x0 : box.x1]
    assert int(box_slice.sum().item()) == int(refined.sum().item())


def test_shell_mask_is_monotone_in_margin() -> None:
    solid = _sphere()
    thin = plan_body_shell_box(solid, shell_margin=4, wake_cells=10, pad=2)
    thick = plan_body_shell_box(solid, shell_margin=8, wake_cells=10, pad=2)
    # a larger shell margin must not shrink the refined region or its box
    assert thin.refine_cells <= thick.refine_cells
    assert thin.box.x0 >= thick.box.x0 and thin.box.x1 <= thick.box.x1
    assert thin.box.y0 >= thick.box.y0 and thin.box.y1 <= thick.box.y1
    assert thin.box.z0 >= thick.box.z0 and thin.box.z1 <= thick.box.z1


def test_plan_box_grows_with_wake_extension() -> None:
    solid = _sphere()
    short = plan_body_shell_box(solid, shell_margin=8, wake_cells=10, pad=2)
    long = plan_body_shell_box(solid, shell_margin=8, wake_cells=40, pad=2)
    # the wake extends in +x only
    assert long.box.x1 >= short.box.x1
    assert long.box.x0 == short.box.x0
    assert long.box.y0 == short.box.y0 and long.box.y1 == short.box.y1
    assert long.box.z0 == short.box.z0 and long.box.z1 == short.box.z1
    assert long.refine_cells >= short.refine_cells


def test_plan_rejects_degenerate_configurations() -> None:
    # an empty solid mask leaves the hull-proximity shell empty
    empty = torch.zeros(16, 16, 16, dtype=torch.bool)
    with pytest.raises(ValueError, match="empty shell"):
        plan_body_shell_box(empty, shell_margin=4, wake_cells=4, pad=2)
