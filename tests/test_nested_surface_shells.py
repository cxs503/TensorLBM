"""Unit tests for the body-agnostic nested surface-shell planner.

Directive 2026-10-11 module item 1: exact nesting L3 ⊂ L2 ⊂ L1, thickness
counted in each level's own cells, machine-checked cell counts.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.amr_shell_planning import (
    NestedShellPlan,
    dilate_chebyshev,
    plan_nested_surface_shells,
    plan_shells_from_root_mask,
)


def _plane(nz: int, ny: int, nx: int, thickness: int) -> torch.Tensor:
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool)
    solid[:thickness] = True
    return solid


def _upsample(mask: torch.Tensor, factor: int) -> torch.Tensor:
    return (
        mask.repeat_interleave(factor, 0).repeat_interleave(factor, 1).repeat_interleave(factor, 2)
    )


def test_plane_shell_exact_count_and_thickness() -> None:
    solid = _plane(16, 16, 64, 4)
    plan = plan_nested_surface_shells((solid,), (2,), (3,))
    assert isinstance(plan, NestedShellPlan)
    assert plan.shell_cells_by_level[0] == 3 * 16 * 64
    assert int(plan.shells[0].sum().item()) == 3 * 16 * 64
    thin = plan_nested_surface_shells((solid,), (2,), (1,))
    assert thin.shell_cells_by_level[0] == 1 * 16 * 64
    # shell is fluid only, adjacent to the solid
    assert not bool((plan.shells[0] & solid).any())
    assert bool((plan.shells[0] & dilate_chebyshev(solid, 1)).any())


def test_dilate_chebyshev_matches_iterated_single_pass() -> None:
    solid = _plane(8, 8, 8, 3)
    one_by_one = dilate_chebyshev(dilate_chebyshev(solid, 1), 1)
    assert torch.equal(dilate_chebyshev(solid, 2), one_by_one)
    assert torch.equal(dilate_chebyshev(solid, 0), solid)


def test_nesting_holds_for_consistent_thickness() -> None:
    root = _plane(8, 16, 64, 4)
    solid_l1 = _upsample(root, 2)  # z < 8 on the 2:1 grid
    solid_l2 = _upsample(solid_l1, 2)  # z < 16 on the 4:1 grid
    plan = plan_nested_surface_shells((solid_l1, solid_l2), (2, 4), (2, 2), enforce_nesting=False)
    assert plan.nesting_violation_cells_before_enforcement == (0, 0)
    enforced = plan_nested_surface_shells((solid_l1, solid_l2), (2, 4), (2, 2))
    for level in range(2):
        assert torch.equal(enforced.shells[level], plan.shells[level])
    assert enforced.shell_cells_by_level[0] == 2 * 32 * 128
    assert enforced.shell_cells_by_level[1] == 2 * 64 * 256


def test_thick_child_shell_is_clipped_to_parent() -> None:
    root = _plane(8, 16, 64, 4)
    solid_l1 = _upsample(root, 2)
    solid_l2 = _upsample(solid_l1, 2)
    unenforced = plan_nested_surface_shells(
        (solid_l1, solid_l2), (2, 4), (2, 8), enforce_nesting=False
    )
    # child band z in [16, 23] sticks out of the parent band z in [16, 19]
    assert unenforced.nesting_violation_cells_before_enforcement[1] == 4 * 64 * 256
    enforced = plan_nested_surface_shells((solid_l1, solid_l2), (2, 4), (2, 8))
    parent = _upsample(enforced.shells[0], 2)
    assert not bool((enforced.shells[1] & ~parent).any())
    assert enforced.shell_cells_by_level[1] == 4 * 64 * 256


def test_plan_from_root_mask_wrapper_matches_explicit_solids() -> None:
    root = _plane(8, 16, 64, 4)
    via_wrapper = plan_shells_from_root_mask(root, (2, 4), (2, 2))
    explicit = plan_nested_surface_shells((_upsample(root, 2), _upsample(root, 4)), (2, 4), (2, 2))
    for wrapper_shell, explicit_shell in zip(via_wrapper.shells, explicit.shells):
        assert torch.equal(wrapper_shell, explicit_shell)


def test_cylinder_shell_bounded_by_euclidean_ideal() -> None:
    nz, ny, nx, radius = 40, 40, 40, 4
    zz, yy, xx = torch.meshgrid(torch.arange(nz), torch.arange(ny), torch.arange(nx), indexing="ij")
    solid = ((yy - 20.0) ** 2 + (zz - 20.0) ** 2) <= radius**2
    thickness = 5
    plan = plan_nested_surface_shells((solid,), (1,), (thickness,))
    measured = plan.shell_cells_by_level[0]
    ideal = 2 * torch.pi * radius * nx * thickness  # S_wetted * t
    assert float(ideal) * 0.8 <= measured <= float(ideal) * 2.5


def test_error_cases() -> None:
    solid = _plane(8, 8, 8, 3)
    with pytest.raises(ValueError, match="equal length"):
        plan_nested_surface_shells((solid,), (2, 4), (2,))
    with pytest.raises(ValueError, match="at least one level"):
        plan_nested_surface_shells((), (), ())
    with pytest.raises(ValueError, match="positive"):
        plan_nested_surface_shells((solid,), (2,), (0,))
    with pytest.raises(ValueError, match="positive"):
        plan_nested_surface_shells((solid,), (0,), (2,))
    with pytest.raises(ValueError, match="boolean"):
        plan_nested_surface_shells((solid.float(),), (2,), (2,))
    full = torch.ones((8, 8, 8), dtype=torch.bool)
    with pytest.raises(ValueError, match="empty shell"):
        plan_nested_surface_shells((full,), (2,), (2,))
    # incompatible level grids
    other = _plane(9, 8, 8, 3)
    with pytest.raises(ValueError, match="incompatible"):
        plan_nested_surface_shells((solid, other), (2, 4), (2, 2))
    with pytest.raises(ValueError, match="empty"):
        plan_shells_from_root_mask(torch.zeros((8, 8, 8), dtype=torch.bool), (2,), (2,))


def test_counts_and_ratios_reported() -> None:
    root = _plane(8, 16, 64, 4)
    plan = plan_shells_from_root_mask(root, (2, 4), (2, 2))
    assert plan.ratios == (2, 4)
    assert plan.thickness_level_cells == (2, 2)
    assert plan.solid_cells_by_level == (
        int(_upsample(root, 2).sum()),
        int(_upsample(root, 4).sum()),
    )
    assert plan.shell_cells_by_level == tuple(int(shell.sum().item()) for shell in plan.shells)
