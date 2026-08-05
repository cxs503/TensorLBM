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
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from tensorlbm.refinement import BoxRegion, HullProximityRegion, WakeRegion


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
        solid_mask, margin=shell_margin,
    ).expand_mask()
    wake_mask = WakeRegion(
        solid_mask, extend_x=wake_cells,
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
    wake_mask[sz1 + 1:, :, :] = False
    wake_mask[:, :sy0, :] = False
    wake_mask[:, sy1 + 1:, :] = False
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
            "refinement box too small: try a larger --wake-cells or a smaller "
            "--shell-margin",
        )
    box = BoxRegion(x0, x1, y0, y1, z0, z1)
    return ShellPlan(box, int(refine_mask.sum().item()), shell_mask, wake_mask)


__all__ = ["ShellPlan", "plan_body_shell_box"]
