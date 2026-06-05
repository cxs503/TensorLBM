"""Multi-patch local refinement solver with proper corner/edge injection.

Extends the 2-level MultiGridSolver to support:
1. Full 3D injection (faces, edges, corners) — not just 6 faces
2. Multiple patches per level (sliding window for slender bodies)
3. Automatic patch partitioning along the hull length

Architecture:
    L0 (coarse): full domain
    L1 patches: N overlapping boxes along the hull at 2× refinement
    L2 patches (optional): surface shell at 4×

Each patch is a Level3 instance that can be solved independently within
a sub-step, communicating via L0 (restrict-update-inject cycle).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F

from .refinement import BoxRegion, _coarse_to_fine_3d, _fine_to_coarse_3d
from .surface_refinement import (
    surface_shell_mask, refined_bounding_box, _make_wall_3d
)


# ---------------------------------------------------------------------------
# Proper 3D injection (faces + edges + corners)
# ---------------------------------------------------------------------------

def inject_full_3d(coarse: torch.Tensor, fine: torch.Tensor,
                   cz0: int, cz1: int, cy0: int, cy1: int,
                   cx0: int, cx1: int, ratio: int = 2) -> None:
    """Inject coarse border into fine boundary with full corner/edge support.

    The coarse inner region [cz0+1:cz1-1, ...] is upsampled and its
    border cells are copied to the fine grid's border.  Dimensions are
    validated and adapted if needed.
    """
    # Extract inner region from coarse (excluding 1-cell border)
    inner = coarse[:, cz0 + 1:cz1 - 1, cy0 + 1:cy1 - 1, cx0 + 1:cx1 - 1]
    up = _coarse_to_fine_3d(inner, ratio)

    R = ratio
    fine_nz, fine_ny, fine_nx = fine.shape[1:]
    
    # Validate: up should be R cells smaller than fine on each side
    expected = (fine_nz - 2 * R, fine_ny - 2 * R, fine_nx - 2 * R)
    actual = (up.shape[1], up.shape[2], up.shape[3])
    
    # If dimensions don't match, pad/trim up to fit
    if expected != actual:
        # Trim up to fit fine grid
        dz = min(up.shape[1], fine_nz - 2 * R)
        dy = min(up.shape[2], fine_ny - 2 * R)
        dx = min(up.shape[3], fine_nx - 2 * R)
        up = up[:, :dz, :dy, :dx]
        # Also adjust R to match actual border width
        R_eff_z = (fine_nz - up.shape[1]) // 2
        R_eff_y = (fine_ny - up.shape[2]) // 2
        R_eff_x = (fine_nx - up.shape[3]) // 2
        Rz = max(1, R_eff_z); Ry = max(1, R_eff_y); Rx = max(1, R_eff_x)
    else:
        Rz = Ry = Rx = R

    # Face injections (use effective border width per dimension)
    fine[:, :Rz, Ry:-Ry, Rx:-Rx] = up[:, :Rz, :, :]
    fine[:, -Rz:, Ry:-Ry, Rx:-Rx] = up[:, -Rz:, :, :]
    fine[:, Rz:-Rz, :Ry, Rx:-Rx] = up[:, :, :Ry, :]
    fine[:, Rz:-Rz, -Ry:, Rx:-Rx] = up[:, :, -Ry:, :]
    fine[:, Rz:-Rz, Ry:-Ry, :Rx] = up[:, :, :, :Rx]
    fine[:, Rz:-Rz, Ry:-Ry, -Rx:] = up[:, :, :, -Rx:]

    # Edge injections
    fine[:, :Rz, :Ry, Rx:-Rx] = up[:, :Rz, :Ry, :]
    fine[:, :Rz, -Ry:, Rx:-Rx] = up[:, :Rz, -Ry:, :]
    fine[:, -Rz:, :Ry, Rx:-Rx] = up[:, -Rz:, :Ry, :]
    fine[:, -Rz:, -Ry:, Rx:-Rx] = up[:, -Rz:, -Ry:, :]
    fine[:, :Rz, Ry:-Ry, :Rx] = up[:, :Rz, :, :Rx]
    fine[:, :Rz, Ry:-Ry, -Rx:] = up[:, :Rz, :, -Rx:]
    fine[:, -Rz:, Ry:-Ry, :Rx] = up[:, -Rz:, :, :Rx]
    fine[:, -Rz:, Ry:-Ry, -Rx:] = up[:, -Rz:, :, -Rx:]
    fine[:, Rz:-Rz, :Ry, :Rx] = up[:, :, :Ry, :Rx]
    fine[:, Rz:-Rz, :Ry, -Rx:] = up[:, :, :Ry, -Rx:]
    fine[:, Rz:-Rz, -Ry:, :Rx] = up[:, :, -Ry:, :Rx]
    fine[:, Rz:-Rz, -Ry:, -Rx:] = up[:, :, -Ry:, -Rx:]

    # Corner injections
    fine[:, :Rz, :Ry, :Rx] = up[:, :Rz, :Ry, :Rx]
    fine[:, :Rz, :Ry, -Rx:] = up[:, :Rz, :Ry, -Rx:]
    fine[:, :Rz, -Ry:, :Rx] = up[:, :Rz, -Ry:, :Rx]
    fine[:, :Rz, -Ry:, -Rx:] = up[:, :Rz, -Ry:, -Rx:]
    fine[:, -Rz:, :Ry, :Rx] = up[:, -Rz:, :Ry, :Rx]
    fine[:, -Rz:, :Ry, -Rx:] = up[:, -Rz:, :Ry, -Rx:]
    fine[:, -Rz:, -Ry:, :Rx] = up[:, -Rz:, -Ry:, :Rx]
    fine[:, -Rz:, -Ry:, -Rx:] = up[:, -Rz:, -Ry:, -Rx:]


def restrict_full_3d(child: torch.Tensor, parent: torch.Tensor,
                     cz0: int, cz1: int, cy0: int, cy1: int,
                     cx0: int, cx1: int, ratio: int = 2) -> None:
    """Restrict child core back to parent, excluding border cells."""
    avg = _fine_to_coarse_3d(child, ratio)
    parent[:, cz0 + 1:cz1 - 1, cy0 + 1:cy1 - 1, cx0 + 1:cx1 - 1] = \
        avg[:, 1:-1, 1:-1, 1:-1]


# ---------------------------------------------------------------------------
# Multi-patch partitioning
# ---------------------------------------------------------------------------

def partition_hull_boxes(
    mask_L0: torch.Tensor,
    n_patches: int = 3,
    overlap: int = 6,
    yz_pad: int = 4,
) -> list[BoxRegion]:
    """Split the hull into *n_patches* overlapping boxes along x.

    Each box covers the full y/z extent of the hull at that x-position,
    plus padding.  Overlapping ensures smooth transitions between patches.

    Args:
        mask_L0: Coarse obstacle mask (nz, ny, nx).
        n_patches: Number of patches along x.
        overlap: Overlap between adjacent patches (coarse cells).
        yz_pad: Extra padding in y and z.

    Returns:
        List of BoxRegion (in coarse coordinates).
    """
    nz, ny, nx = mask_L0.shape
    # Find hull x-extent
    x_any = mask_L0.float().sum(dim=(0, 1))
    x_idx = torch.nonzero(x_any).squeeze(1)
    if x_idx.numel() == 0:
        return [BoxRegion(0, nx, 0, ny, 0, nz)]

    x_min, x_max = int(x_idx.min().item()), int(x_idx.max().item())
    hull_len = x_max - x_min + 1
    patch_len = hull_len // n_patches + overlap

    boxes = []
    for i in range(n_patches):
        x0 = x_min + i * (hull_len // n_patches) - overlap // 2
        x0 = max(0, x0)
        x1 = min(nx, x0 + patch_len)
        # Full y/z extent with padding
        y_any = mask_L0[:, :, x0:x1].float().sum(dim=(0, 2))
        y_idx = torch.nonzero(y_any).squeeze(1)
        y0 = max(0, int(y_idx.min().item()) - yz_pad) if y_idx.numel() > 0 else 0
        y1 = min(ny, int(y_idx.max().item()) + 1 + yz_pad) if y_idx.numel() > 0 else ny

        z_any = mask_L0[:, y0:y1, x0:x1].float().sum(dim=(1, 2))
        z_idx = torch.nonzero(z_any).squeeze(1)
        z0 = max(0, int(z_idx.min().item()) - yz_pad) if z_idx.numel() > 0 else 0
        z1 = min(nz, int(z_idx.max().item()) + 1 + yz_pad) if z_idx.numel() > 0 else nz

        boxes.append(BoxRegion(x0, x1, y0, y1, z0, z1))

    return boxes


# ---------------------------------------------------------------------------
# Multi-patch 2-level solver
# ---------------------------------------------------------------------------

@dataclass
class PatchLevel:
    """One patch at a refinement level."""
    f: torch.Tensor
    mask: torch.Tensor
    wall_mask: torch.Tensor
    box: BoxRegion  # in parent coordinates


@dataclass
class MultiPatchSolver:
    """2-level solver with multiple fine patches along the hull.

    Each patch is solved independently within a sub-step, then restricted
    back to the coarse grid.  Patches communicate through the coarse grid
    (overlap ensures smooth transitions).
    """
    coarse_f: torch.Tensor
    coarse_mask: torch.Tensor
    coarse_wall: torch.Tensor
    patches: list[PatchLevel] = field(default_factory=list)
    ratio: int = 2

    @classmethod
    def from_mask(
        cls,
        mask_L0: torch.Tensor,
        wall_L0: torch.Tensor,
        f_L0: torch.Tensor,
        n_patches: int = 3,
        overlap: int = 8,
        yz_pad: int = 4,
        ratio: int = 2,
    ) -> MultiPatchSolver:
        """Build a multi-patch solver from a coarse mask.

        Args:
            mask_L0: Coarse obstacle mask (nz, ny, nx).
            wall_L0: Coarse wall mask.
            f_L0: Initial coarse distributions.
            n_patches: Number of patches along x.
            overlap: Overlap between patches (coarse cells).
            yz_pad: Padding in y and z.
            ratio: Refinement ratio (default 2).
        """
        boxes = partition_hull_boxes(mask_L0, n_patches, overlap, yz_pad)
        device = mask_L0.device

        patches = []
        for box in boxes:
            # Extract coarse patch
            mask_c = mask_L0[box.z0:box.z1, box.y0:box.y1, box.x0:box.x1]
            # Upsample mask for fine grid
            mask_f = mask_c.repeat_interleave(ratio, 0).repeat_interleave(ratio, 1).repeat_interleave(ratio, 2)
            nz_f, ny_f, nx_f = mask_f.shape
            # Fine wall mask
            wall_f = _make_wall_3d(nz_f, ny_f, nx_f, mask_f, device=device)
            # Initialize fine distributions
            f_fine = _coarse_to_fine_3d(
                f_L0[:, box.z0:box.z1, box.y0:box.y1, box.x0:box.x1], ratio
            )
            patches.append(PatchLevel(f=f_fine, mask=mask_f, wall_mask=wall_f, box=box))

        solver = cls(coarse_f=f_L0, coarse_mask=mask_L0, coarse_wall=wall_L0,
                     patches=patches, ratio=ratio)
        solver._compute_cells()
        return solver

    def _compute_cells(self):
        total = self.coarse_f.shape[1] * self.coarse_f.shape[2] * self.coarse_f.shape[3]
        for p in self.patches:
            total += p.f.shape[1] * p.f.shape[2] * p.f.shape[3]
        self.total_cells = total

    def step(self, collide_fn, stream_fn, boundary_fn) -> None:
        """One coarse step with multi-patch sub-stepping."""
        R = self.ratio

        # 1. Coarse step
        self.coarse_f = collide_fn(self.coarse_f)
        self.coarse_f = stream_fn(self.coarse_f)
        boundary_fn(self.coarse_f)

        # 2. Inject coarse → all patches
        for p in self.patches:
            inject_full_3d(
                self.coarse_f, p.f,
                p.box.z0, p.box.z1, p.box.y0, p.box.y1, p.box.x0, p.box.x1,
                R
            )

        # 3. Fine sub-steps
        for _ in range(R):
            for p in self.patches:
                p.f = collide_fn(p.f)
                p.f = stream_fn(p.f)
                boundary_fn(p.f)
            # Re-inject after each sub-step
            for p in self.patches:
                inject_full_3d(
                    self.coarse_f, p.f,
                    p.box.z0, p.box.z1, p.box.y0, p.box.y1, p.box.x0, p.box.x1,
                    R
                )

        # 4. Restrict patches → coarse
        for p in self.patches:
            restrict_full_3d(
                p.f, self.coarse_f,
                p.box.z0, p.box.z1, p.box.y0, p.box.y1, p.box.x0, p.box.x1,
                R
            )


__all__ = [
    "inject_full_3d",
    "restrict_full_3d",
    "partition_hull_boxes",
    "MultiPatchSolver",
    "PatchLevel",
]
