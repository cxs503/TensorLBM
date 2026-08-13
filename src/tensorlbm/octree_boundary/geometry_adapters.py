"""Geometry adapters for the octree boundary shell.

The octree shell builder (``build_octree_shell``) classifies every candidate
leaf centre as inside/outside the body.  The default implementation is a
sphere test.  These adapters let any body defined by an analytic signed
distance or by a voxel solid mask drive the same shell builder, so the shell
solver (P2 stepping, P3 BFL/force) is reused unchanged across geometries.

Adapters are pure functions returning ``inside_fn(centers)`` where
``centers`` is a ``(N, 3)`` float64 tensor of leaf centres in *world* (L1
physical) cell coordinates and the return is a bool tensor of length N.

Common geometries
-----------------
* :func:`sphere_inside_fn`  — analytic sphere (default; kept for symmetry).
* :func:`solid_mask_inside_fn` — arbitrary voxel solid mask (e.g. SUBOFF):
  leaf centres are inside iff the nearest voxel is solid.  The mask is
  ``(nz, ny, nx)`` bool and ``offset`` maps world coordinates to mask
  indices (usually the L1 block origin in the mask frame).
* :func:`solid_mask_shell_fn` — body surface shell band around a voxel body:
  a leaf is *in the shell band* when it is fluid but within
  ``bl_thickness`` of the solid (used to mask which cells become leaves).
"""

from __future__ import annotations

import torch

__all__ = [
    "sphere_inside_fn",
    "solid_mask_inside_fn",
    "solid_mask_shell_fn",
]


def sphere_inside_fn(
    center: tuple[float, float, float],
    radius: float,
) -> callable:
    """Return an inside test for a sphere at ``center`` with ``radius``.

    The returned function carries ``analytic_q=True`` so the octree builder
    knows the analytic sphere q-field (``compute_q_sphere_at_points``) is
    valid for BFL.
    """

    def inside_fn(centers: torch.Tensor) -> torch.Tensor:
        dist2 = (
            (centers[:, 0] - center[0]) ** 2
            + (centers[:, 1] - center[1]) ** 2
            + (centers[:, 2] - center[2]) ** 2
        )
        return dist2 <= radius ** 2

    inside_fn.analytic_q = True  # type: ignore[attr-defined]
    return inside_fn


def solid_mask_inside_fn(
    solid_mask: torch.Tensor,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    device: torch.device | None = None,
) -> callable:
    """Return an inside test for a voxel solid mask.

    Args:
        solid_mask: bool tensor ``(nz, ny, nx)``; True = solid body cell.
        offset: world (L1 physical) coordinate of mask index 0, i.e.
            ``mask_index = world - offset``.
        device: device for the returned test's intermediate tensors.
            Defaults to the mask device.
    """
    mask = solid_mask
    dev = mask.device if device is None else device
    off = torch.tensor(offset, dtype=torch.float64, device=dev)
    if mask.device != dev:
        mask = mask.to(dev)

    def inside_fn(centers: torch.Tensor) -> torch.Tensor:
        # centres: (N, 3) float64 world coords (x, y, z) -> mask index
        idx = (centers.to(device=dev) - off).round().to(torch.int64)
        nz, ny, nx = mask.shape
        x = idx[:, 0].clamp(0, nx - 1)
        y = idx[:, 1].clamp(0, ny - 1)
        z = idx[:, 2].clamp(0, nz - 1)
        return mask[z, y, x]

    return inside_fn


def solid_mask_shell_fn(
    solid_mask: torch.Tensor,
    bl_thickness: float,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    device: torch.device | None = None,
) -> callable:
    """Return a shell-band test for a voxel body.

    A point is in the shell band when it is fluid (nearest voxel not solid)
    and lies within ``bl_thickness`` of the solid.  The band is computed by
    a fixed number of dilation passes: with unit voxels, ``bl_thickness``
    cells of dilation ≈ ``ceil(bl_thickness)`` passes.
    """
    mask = solid_mask
    dev = mask.device if device is None else device
    off = torch.tensor(offset, dtype=torch.float64, device=dev)

    # Dilate the solid mask by the band thickness (in mask-cell units).
    import torch.nn.functional as F

    dilated = mask.float().unsqueeze(0).unsqueeze(0)  # (1,1,nz,ny,nx)
    n_passes = max(1, int(round(bl_thickness)))
    for _ in range(n_passes):
        dilated = F.max_pool3d(
            dilated, kernel_size=3, stride=1, padding=1,
        )
    dilated = dilated.squeeze(0).squeeze(0) > 0.5
    shell_band = dilated & ~mask  # fluid but near solid

    def inside_fn(centers: torch.Tensor) -> torch.Tensor:
        idx = (centers.to(device=dev) - off).round().to(torch.int64)
        nz, ny, nx = mask.shape
        z = idx[:, 0].clamp(0, nz - 1)
        y = idx[:, 1].clamp(0, ny - 1)
        x = idx[:, 2].clamp(0, nx - 1)
        return shell_band[z, y, x]

    return inside_fn
