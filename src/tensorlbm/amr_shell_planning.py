"""Body-fitted shell + wake refinement planning for the AMR sphere runners.

Shared by the AMR sphere validation examples.  The coarse-grid refinement
mask is the union of a hull-proximity surface shell
(:class:`~tensorlbm.refinement.HullProximityRegion`) and a downstream wake
slab (:class:`~tensorlbm.refinement.WakeRegion`) whose lateral extent is
clipped to the shell's own z/y range so the fine block stays body-fitted
instead of spanning the whole coarse domain height/depth.  The padded
bounding box of that mask is returned as a
:class:`~tensorlbm.refinement.BoxRegion`.

The clipping and padding logic (including the box size validation) follows
``examples/amr_sphere_shell_validate.py`` so all runners produce identical
boxes.

This module also hosts the body-agnostic **nested surface-shell planner**
(:func:`plan_nested_surface_shells`) introduced by the high-Re SUBOFF
surface-refinement directive: concentric shells L1 (2:1) ⊃ L2 (4:1) ⊃
L3 (8:1) around an arbitrary solid, each a fixed number of that level's own
cells thick, with nesting enforced exactly.  Any body works — pass the
solid mask at each level's own resolution (exact CAD rebuild where the
body is analytic, upsampled voxels otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tensorlbm.refinement import BoxRegion, HullProximityRegion, WakeRegion

__all__ = [
    "NestedShellPlan",
    "ShellPlan",
    "dilate_chebyshev",
    "plan_body_shell_box",
    "plan_nested_surface_shells",
    "plan_shells_from_root_mask",
]


@dataclass(frozen=True)
class ShellPlan:
    """Result of :func:`plan_body_shell_box`."""

    box: BoxRegion
    refine_cells: int
    shell_mask: torch.Tensor
    wake_mask: torch.Tensor


def plan_body_shell_box(
    solid_mask: torch.Tensor,
    shell_margin: int,
    wake_cells: int,
    pad: int = 2,
) -> ShellPlan:
    """Plan the body-fitted shell + wake refinement region and its bbox.

    Args:
        solid_mask: Coarse-grid boolean solid mask ``(nz, ny, nx)``.
        shell_margin: Hull-proximity shell thickness in coarse cells
            (``HullProximityRegion`` margin).
        wake_cells: Downstream wake extension in coarse cells.
        pad: Coarse-cell padding added around the refinement-mask bounding
            box so the coarse-fine interface stays off the solid shell
            surface.  The single-layer shell runner uses ``2``; the nested
            L1-shell runner passes its ``--wall-margin``.

    Returns:
        :class:`ShellPlan` with the padded bounding box, the number of
        refined coarse cells and the two component masks.

    Raises:
        ValueError: if the shell is empty, the combined shell+wake region is
            empty, or the padded box degenerates below the minimum size.
    """
    shell_mask = HullProximityRegion(
        solid_mask,
        margin=shell_margin,
    ).expand_mask()
    wake_mask = WakeRegion(
        solid_mask,
        extend_x=wake_cells,
    ).expand_mask()
    # WakeRegion fills the full downstream cross-section plane; clip it
    # laterally to the body's shell extent so the fine block stays
    # body-fitted instead of spanning the entire coarse domain height/depth.
    shell_idx = shell_mask.nonzero(as_tuple=False)
    if shell_idx.numel() == 0:
        raise ValueError(
            "empty shell refinement region; adjust --shell-margin",
        )
    sz0, sy0 = int(shell_idx[:, 0].min().item()), int(shell_idx[:, 1].min().item())
    sz1, sy1 = int(shell_idx[:, 0].max().item()), int(shell_idx[:, 1].max().item())
    wake_mask[:sz0, :, :] = False
    wake_mask[sz1 + 1 :, :, :] = False
    wake_mask[:, :sy0, :] = False
    wake_mask[:, sy1 + 1 :, :] = False
    refine_mask = shell_mask | wake_mask
    indices = refine_mask.nonzero(as_tuple=False)
    if indices.numel() == 0:
        raise ValueError(
            "empty shell+wake refinement region; adjust --shell-margin/--wake-cells",
        )
    z_min, y_min, x_min = (int(indices[:, a].min().item()) for a in range(3))
    z_max, y_max, x_max = (int(indices[:, a].max().item()) + 1 for a in range(3))
    nz, ny, nx = solid_mask.shape
    x0 = max(1, x_min - pad)
    x1 = min(nx - 1, x_max + pad)
    y0 = max(1, y_min - pad)
    y1 = min(ny - 1, y_max + pad)
    z0 = max(1, z_min - pad)
    z1 = min(nz - 1, z_max + pad)
    if min(x1 - x0, y1 - y0, z1 - z0) < 3:
        raise ValueError(
            "refinement box too small: try a larger --wake-cells or a smaller --shell-margin",
        )
    box = BoxRegion(x0, x1, y0, y1, z0, z1)
    return ShellPlan(box, int(refine_mask.sum().item()), shell_mask, wake_mask)


# --------------------------------------------------------------------------- #
# Nested surface-shell planning (body-agnostic, additive)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NestedShellPlan:
    """Result of :func:`plan_nested_surface_shells`.

    ``shells[i]`` is the refinement region of level ``i`` (0 = first fine
    level) on that level's own resolution, as a boolean mask shaped like
    ``solid_by_level[i]``.  Nesting ``shells[i] ⊆ upsample(shells[i-1])``
    holds exactly when ``enforce_nesting`` was requested (default).
    """

    shells: tuple[torch.Tensor, ...]
    ratios: tuple[int, ...]
    thickness_level_cells: tuple[int, ...]
    solid_cells_by_level: tuple[int, ...]
    shell_cells_by_level: tuple[int, ...]
    nesting_violation_cells_before_enforcement: tuple[int, ...]


def dilate_chebyshev(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Chebyshev dilation of a boolean mask by ``radius`` cells.

    This is the shell-band criterion both refinement routes implement
    (static-block planning and ``octree_boundary.geometry.build_shell_cell_mask``
    both dilate the solid), exposed so layout scripts measure the same
    region the runtime would refine.
    """
    if radius <= 0:
        return mask.clone()
    x = mask.float().unsqueeze(0).unsqueeze(0)
    for _ in range(radius):
        x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
    return x.squeeze(0).squeeze(0).bool()


def _upsample_any(mask: torch.Tensor, ratio: int) -> torch.Tensor:
    """Boolean exact-cover upsample by an integer factor (any-semantics)."""
    if ratio == 1:
        return mask
    return mask.repeat_interleave(ratio, 0).repeat_interleave(ratio, 1).repeat_interleave(ratio, 2)


def plan_nested_surface_shells(
    solid_by_level: tuple[torch.Tensor, ...],
    ratios: tuple[int, ...],
    thickness_level_cells: tuple[int, ...],
    *,
    enforce_nesting: bool = True,
) -> NestedShellPlan:
    """Plan concentric surface shells around an arbitrary body.

    Level ``i`` has refinement ratio ``ratios[i]`` against the root grid and
    a shell of ``thickness_level_cells[i]`` cells of its own grid measured
    from the body surface outward (Chebyshev dilation of the solid, fluid
    side only — the same implementable criterion both refinement routes
    use).

    Args:
        solid_by_level: boolean solid mask per level, each at that level's
            own resolution and covering at least the shell of its level
            (exact CAD rebuild for analytic bodies; upsampled voxels
            otherwise).  Level 0 is the first *fine* level, not the root.
        ratios: refinement ratio vs the root grid per level (e.g.
            ``(2, 4, 8)``).
        thickness_level_cells: shell thickness per level, in that level's
            own cells (e.g. ``(8, 8, 8)``).
        enforce_nesting: intersect each child shell with the upsampled
            parent shell so containment is exact (default).  The
            pre-enforcement violation counts are always reported.

    Raises:
        ValueError: on length mismatches, non-positive inputs or an empty
            shell at any level.
    """
    if not (len(solid_by_level) == len(ratios) == len(thickness_level_cells)):
        raise ValueError("solid_by_level, ratios and thickness must have equal length")
    if len(solid_by_level) == 0:
        raise ValueError("at least one level is required")
    if any(int(r) <= 0 for r in ratios):
        raise ValueError("ratios must be positive")
    if any(int(t) <= 0 for t in thickness_level_cells):
        raise ValueError("thickness must be positive")

    shells: list[torch.Tensor] = []
    violations: list[int] = []
    solids_cells: list[int] = []
    shell_cells: list[int] = []
    for index, (solid, ratio, thickness) in enumerate(
        zip(solid_by_level, ratios, thickness_level_cells, strict=True)
    ):
        if solid.dtype != torch.bool:
            raise ValueError("solid masks must be boolean")
        shell = dilate_chebyshev(solid, int(thickness)) & ~solid
        if not bool(shell.any()):
            raise ValueError(f"empty shell at level index {index}")
        if index > 0:
            parent = _upsample_any(shells[index - 1], int(ratio) // int(ratios[index - 1]))
            if parent.shape != shell.shape:
                raise ValueError(
                    f"level {index} grid {tuple(shell.shape)} incompatible with "
                    f"upsampled parent {tuple(parent.shape)}"
                )
            violations.append(int((shell & ~parent).sum().item()))
            if enforce_nesting:
                shell = shell & parent
        else:
            violations.append(0)
        shells.append(shell)
        solids_cells.append(int(solid.sum().item()))
        shell_cells.append(int(shell.sum().item()))

    return NestedShellPlan(
        shells=tuple(shells),
        ratios=tuple(int(r) for r in ratios),
        thickness_level_cells=tuple(int(t) for t in thickness_level_cells),
        solid_cells_by_level=tuple(solids_cells),
        shell_cells_by_level=tuple(shell_cells),
        nesting_violation_cells_before_enforcement=tuple(violations),
    )


def plan_shells_from_root_mask(
    root_solid: torch.Tensor,
    ratios: tuple[int, ...],
    thickness_level_cells: tuple[int, ...],
    *,
    enforce_nesting: bool = True,
) -> NestedShellPlan:
    """Convenience wrapper planning shells from a single root-resolution mask.

    The per-level solids are voxel upsamples of the root mask (exact-cover,
    any-semantics).  This is the drop-in API for bodies without an analytic
    rebuild; for analytic bodies (e.g. SUBOFF) pass CAD-rebuilt solids to
    :func:`plan_nested_surface_shells` instead so the fine shells follow
    the true surface rather than the root rasterisation.
    """
    if not bool(root_solid.any()):
        raise ValueError("root solid mask is empty")
    solid_by_level = []
    current = root_solid.bool()
    previous_ratio = 1
    for ratio in ratios:
        factor = int(ratio) // previous_ratio
        current = _upsample_any(current, factor)
        solid_by_level.append(current)
        previous_ratio = int(ratio)
    return plan_nested_surface_shells(
        tuple(solid_by_level),
        ratios,
        thickness_level_cells,
        enforce_nesting=enforce_nesting,
    )
