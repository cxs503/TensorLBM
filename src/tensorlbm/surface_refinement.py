"""Surface-shell local refinement for LBM — practical 3-level solver.

Implements a 3-level nested grid with surface-aware refinement:
  L0: Coarse background (full domain)
  L1: Box refinement around object (proximity + wake, 2×)
  L2: Surface shell refinement (thin layer around surface, 4×)

The surface-shell approach dramatically reduces cells for slender bodies
(SUBOFF, ship hulls) where the surface is a small fraction of the volume.

Usage:
    from tensorlbm.surface_refinement import build_3level_solver, surface_shell_mask
    solver = build_3level_solver(mask, ...)
    for step in range(n_steps):
        solver.step(collide_fn, stream_fn, boundary_fn)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass

from .refinement import BoxRegion, _coarse_to_fine_3d, _fine_to_coarse_3d


# ---------------------------------------------------------------------------
# Surface extraction
# ---------------------------------------------------------------------------

def surface_mask(mask: torch.Tensor) -> torch.Tensor:
    """Extract surface cells (mask cells with at least one fluid neighbor).

    Uses 6-neighbor (face-connected) connectivity.
    """
    surf = torch.zeros_like(mask)
    for dim, shift in [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]:
        rolled = torch.roll(mask.float(), shift, dim)
        surf |= mask & (~rolled.bool())
    return surf


def surface_shell_mask(mask: torch.Tensor, margin: int = 3) -> torch.Tensor:
    """Dilate surface cells by *margin* cells to create a refinement band.

    Returns a boolean mask of cells within *margin* of the object surface.
    """
    surf = surface_mask(mask)
    if margin == 0:
        return surf
    # 3D dilation via convolution
    ksize = 2 * margin + 1
    kernel = torch.ones((1, 1, ksize, ksize, ksize), device=mask.device) / (ksize ** 3)
    padded = surf.float().unsqueeze(0).unsqueeze(0)
    dilated = (F.conv3d(padded, kernel, padding=margin) > 0.01).squeeze(0).squeeze(0)
    return dilated.bool()


def refined_bounding_box(refined_mask: torch.Tensor, pad: int = 2) -> BoxRegion:
    """Compute tight bounding box of a refinement mask, with padding."""
    idx = torch.nonzero(refined_mask)
    if idx.numel() == 0:
        return BoxRegion(0, 1, 0, 1, 0, 1)
    nz, ny, nx = refined_mask.shape
    z0 = max(0, int(idx[:, 0].min()) - pad)
    z1 = min(nz, int(idx[:, 0].max()) + 1 + pad)
    y0 = max(0, int(idx[:, 1].min()) - pad)
    y1 = min(ny, int(idx[:, 1].max()) + 1 + pad)
    x0 = max(0, int(idx[:, 2].min()) - pad)
    x1 = min(nx, int(idx[:, 2].max()) + 1 + pad)
    return BoxRegion(x0, x1, y0, y1, z0, z1)


# ---------------------------------------------------------------------------
# 3-Level solver (L0 → L1 2× → L2 4×)
# ---------------------------------------------------------------------------

@dataclass
class Level3:
    """One level of the 3-level solver."""
    f: torch.Tensor          # (19, nz, ny, nx)
    mask: torch.Tensor       # obstacle mask at this level
    wall_mask: torch.Tensor  # wall mask at this level
    offset: tuple[int, int, int, int, int, int]  # (z0,z1,y0,y1,x0,x1) in parent coords
    parent: Level3 | None = None
    ratio_to_parent: int = 2


class SurfaceRefinementSolver:
    """3-level LBM solver with surface-shell refinement.

    L0: coarse background (full domain)
    L1: box around object (2× refinement)
    L2: surface shell only (4× refinement)

    The L2 level only covers cells within a thin band around the object
    surface, dramatically reducing cell count for slender bodies.
    """

    def __init__(self, levels: list[Level3]):
        self.levels = levels
        self._b = 2  # border width in fine cells

    @classmethod
    def from_mask(
        cls,
        mask_L0: torch.Tensor,
        wall_L0: torch.Tensor,
        f_L0: torch.Tensor,
        *,
        L1_pad: int = 4,
        L2_margin: int = 1,
        L2_pad: int = 2,
    ) -> SurfaceRefinementSolver:
        """Build a 3-level surface-refinement solver from a coarse mask.

        Args:
            mask_L0: Coarse obstacle mask (nz, ny, nx).
            wall_L0: Coarse wall mask.
            f_L0: Initial distributions at L0 (19, nz, ny, nx).
            L1_pad: Padding around hull bounding box for L1.
            L2_margin: Surface dilation margin for L2 shell.
            L2_pad: Extra padding for L2 bounding box.
        """
        nz, ny, nx = mask_L0.shape
        nz_c, ny_c, nx_c = nz, ny, nx  # L0 dimensions

        # --- L1: box around hull + wake ---
        # Dilate hull to get proximity region
        prox = surface_shell_mask(mask_L0, margin=L1_pad)
        # Also include wake (downstream of trailing edge)
        x_any = mask_L0.float().sum(dim=(0, 1))
        te_cells = torch.nonzero(x_any)
        te = int(te_cells.max().item()) if te_cells.numel() > 0 else nx // 2
        wake_end = min(nx, te + nx // 4)
        prox[:, :, te:wake_end] = True
        L1_box = refined_bounding_box(prox, pad=2)

        # L1 fine grid dimensions (2× refinement)
        nf1_x = (L1_box.x1 - L1_box.x0) * 2
        nf1_y = (L1_box.y1 - L1_box.y0) * 2
        nf1_z = (L1_box.z1 - L1_box.z0) * 2

        # L1 mask at 2× (upsample from coarse)
        mask_L1_c = mask_L0[L1_box.z0:L1_box.z1, L1_box.y0:L1_box.y1, L1_box.x0:L1_box.x1]
        mask_L1 = mask_L1_c.repeat_interleave(2, 0).repeat_interleave(2, 1).repeat_interleave(2, 2)
        wall_L1 = _make_wall_3d(nf1_z, nf1_y, nf1_x, mask_L1, device=mask_L0.device)
        f_L1 = _coarse_to_fine_3d(
            f_L0[:, L1_box.z0:L1_box.z1, L1_box.y0:L1_box.y1, L1_box.x0:L1_box.x1], 2
        )

        # --- L2: surface shell only ---
        # Compute surface shell on L1's coarse mask
        surf_L1 = surface_shell_mask(mask_L1_c, margin=L2_margin)
        L2_box = refined_bounding_box(surf_L1, pad=L2_pad)

        nf2_x = (L2_box.x1 - L2_box.x0) * 4  # ×4 from L0 (L1 already 2×, L2 2× again)
        nf2_y = (L2_box.y1 - L2_box.y0) * 4
        nf2_z = (L2_box.z1 - L2_box.z0) * 4

        mask_L2_c = mask_L1_c[L2_box.z0:L2_box.z1, L2_box.y0:L2_box.y1, L2_box.x0:L2_box.x1]
        # Upsample to L2 resolution: L1 coarse → 2× (L1 fine) → 2× again (L2 fine) = 4×
        mask_L2 = mask_L2_c.repeat_interleave(4, 0).repeat_interleave(4, 1).repeat_interleave(4, 2)
        wall_L2 = _make_wall_3d(nf2_z, nf2_y, nf2_x, mask_L2, device=mask_L0.device)
        f_L2 = _coarse_to_fine_3d(
            f_L1[:, L2_box.z0 * 2:L2_box.z1 * 2, L2_box.y0 * 2:L2_box.y1 * 2, L2_box.x0 * 2:L2_box.x1 * 2], 2
        )

        # Build levels
        l0 = Level3(f=f_L0, mask=mask_L0, wall_mask=wall_L0,
                     offset=(0, nz_c, 0, ny_c, 0, nx_c))
        l1 = Level3(f=f_L1, mask=mask_L1, wall_mask=wall_L1,
                     offset=(L1_box.z0, L1_box.z1, L1_box.y0, L1_box.y1, L1_box.x0, L1_box.x1),
                     parent=l0)
        l2 = Level3(f=f_L2, mask=mask_L2, wall_mask=wall_L2,
                     offset=(L2_box.z0, L2_box.z1, L2_box.y0, L2_box.y1, L2_box.x0, L2_box.x1),
                     parent=l1)
        l1.parent = l0
        l2.parent = l1

        solver = cls([l0, l1, l2])
        solver._cell_count = nx_c * ny_c * nz_c + nf1_x * nf1_y * nf1_z + nf2_x * nf2_y * nf2_z
        return solver

    @property
    def total_cells(self) -> int:
        return self._cell_count

    def _inject_up(self, coarse: Level3, fine: Level3) -> None:
        """Inject coarse border into fine boundary with full corner/edge support."""
        cz0, cz1, cy0, cy1, cx0, cx1 = fine.offset
        # Offsets are in parent's COARSE coordinates — convert to parent's FINE coordinates
        r = fine.ratio_to_parent
        from .multipatch import inject_full_3d
        inject_full_3d(coarse.f, fine.f, cz0 * r, cz1 * r, cy0 * r, cy1 * r, cx0 * r, cx1 * r, r)

    def _restrict_down(self, child: Level3) -> None:
        """Restrict child core back to parent interior."""
        parent = child.parent
        cz0, cz1, cy0, cy1, cx0, cx1 = child.offset
        r = child.ratio_to_parent
        avg = _fine_to_coarse_3d(child.f, r)
        # Convert to parent fine coordinates
        parent.f[:, cz0 * r + 1:cz1 * r - 1, cy0 * r + 1:cy1 * r - 1, cx0 * r + 1:cx1 * r - 1] = \
            avg[:, 1:-1, 1:-1, 1:-1]

    def step(self, collide_fn, stream_fn, boundary_fn) -> None:
        """One coarse time step with all sub-steps."""
        l0, l1, l2 = self.levels

        # L0: one step
        l0.f = collide_fn(l0.f)
        l0.f = stream_fn(l0.f)
        boundary_fn(l0)
        self._inject_up(l0, l1)

        # L1 + L2: two sub-steps
        for sub in range(2):
            l1.f = collide_fn(l1.f)
            l1.f = stream_fn(l1.f)
            if sub == 0:
                self._inject_up(l1, l2)
            # L2: two sub-sub-steps
            for _ in range(2):
                l2.f = collide_fn(l2.f)
                l2.f = stream_fn(l2.f)
                boundary_fn(l2)
            self._restrict_down(l2)
            boundary_fn(l1)

        self._restrict_down(l1)


def _make_wall_3d(nz: int, ny: int, nx: int, obstacle: torch.Tensor, device) -> torch.Tensor:
    """Create wall mask excluding obstacle cells."""
    w = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    w[0, :, :] = True
    w[-1, :, :] = True
    w[:, 0, :] = True
    w[:, -1, :] = True
    w[:, :, 0] = True
    w[:, :, -1] = True
    w[obstacle] = False
    return w


__all__ = [
    "surface_mask",
    "surface_shell_mask",
    "refined_bounding_box",
    "SurfaceRefinementSolver",
    "Level3",
]
