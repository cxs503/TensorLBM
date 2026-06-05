"""Local mesh refinement for LBM — multi-block patch-based approach.

Provides a multi-level grid system where fine-resolution patches overlay a
coarse background grid.  Typical use: refine near hull boundaries and in
wake regions while keeping a coarser far-field resolution.

Architecture
------------
``MultiGridSolver`` manages *N* ``GridLevel`` instances, each with:
- its own ``(nx, ny, nz)`` domain and cell size ``dx``
- an f-distribution tensor ``(19, nz, ny, nx)``
- overlap (ghost) regions for inter-level exchange

Time stepping follows the typical LBM multi-grid schedule: the fine level
runs 2 collision-stream cycles per 1 coarse-level cycle.

Refinement Strategy
-------------------
The ``RefinementRegion`` defines *where* to refine.  Built-in strategies:
- ``BoxRegion``              – axis-aligned box
- ``HullProximityRegion``    – cells within N*distance of hull mask
- ``WakeRegion``             – downstream of hull

References
----------
Filippova & Hänel (1998) J. Comput. Phys. 147 219
Dupuis & Chopard (2003) Int. J. Mod. Phys. B 17 169
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoxRegion:
    """Axis-aligned 3-D box in coarse-grid index space."""
    x0: int; x1: int
    y0: int; y1: int
    z0: int; z1: int

    def to_fine(self, ratio: int = 2) -> BoxRegion:
        return BoxRegion(
            self.x0 * ratio, self.x1 * ratio,
            self.y0 * ratio, self.y1 * ratio,
            self.z0 * ratio, self.z1 * ratio,
        )


@dataclass
class HullProximityRegion:
    """Refine all cells within *margin* cells of the hull surface.

    Computed at construction from the hull boolean mask.
    """
    mask: torch.Tensor  # (nz, ny, nx) boolean
    margin: int = 3

    def expand_mask(self) -> torch.Tensor:
        """Return boolean mask of cells to refine (coarse grid)."""
        m = self.mask.float()
        kernel = torch.ones((1, 1, 3, 3, 3), device=m.device) / 27.0
        # Convolve to find cells near hull
        padded = F.pad(m.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1), mode='replicate')
        blurred = F.conv3d(padded, kernel).squeeze()
        return blurred > 0.01


@dataclass
class WakeRegion:
    """Refine downstream of hull (x > hull trailing edge)."""
    hull_mask: torch.Tensor  # (nz, ny, nx)
    extend_x: int = 40

    def expand_mask(self) -> torch.Tensor:
        m = self.hull_mask.float()
        # Find trailing edge (rightmost hull cell)
        x_any = m.sum(dim=(0, 1))  # (nx,)
        te = int(torch.nonzero(x_any).max().item()) if x_any.any() else 0
        wake = torch.zeros_like(m, dtype=torch.bool)
        if te > 0:
            wake[:, :, te:te + self.extend_x] = True
        return wake


# ---------------------------------------------------------------------------
# Interpolation (coarse ↔ fine)
# ---------------------------------------------------------------------------

def _coarse_to_fine_3d(f_coarse: torch.Tensor, ratio: int = 2) -> torch.Tensor:
    """Upsample distributions from coarse to fine grid.

    Uses trilinear interpolation.  Input shape: (19, nz, ny, nx).
    """
    # Use torch.nn.functional.interpolate for efficient upsampling
    b19, nz, ny, nx = f_coarse.shape
    f_4d = f_coarse.unsqueeze(0)  # (1, 19, nz, ny, nx) → treat 19 as channels
    f_up = F.interpolate(f_4d, size=(nz * ratio, ny * ratio, nx * ratio),
                         mode='trilinear', align_corners=True)
    return f_up.squeeze(0)


def _fine_to_coarse_3d(f_fine: torch.Tensor, ratio: int = 2) -> torch.Tensor:
    """Restrict distributions from fine to coarse grid by averaging.

    Input shape: (19, nz, ny, nx) where nx, ny, nz are multiples of ratio.
    """
    b19, nz_f, ny_f, nx_f = f_fine.shape
    # Reshape and average over ratio-sized blocks
    nz_c = nz_f // ratio; ny_c = ny_f // ratio; nx_c = nx_f // ratio
    f_reshaped = f_fine.view(b19, nz_c, ratio, ny_c, ratio, nx_c, ratio)
    f_coarse = f_reshaped.mean(dim=(2, 4, 6))  # average over fine cells
    return f_coarse


# ---------------------------------------------------------------------------
# Grid Level
# ---------------------------------------------------------------------------

@dataclass
class GridLevel:
    """One level of the multi-grid hierarchy.

    Attributes
    ----------
    f: Distribution tensor (19, nz, ny, nx).
    dx: Cell size relative to the finest level (= 1 / ratio).
    region: Bounding box in COARSE-grid index space (defines where this
            patch lives within the parent grid).
    mask: Optional boolean solid mask for this level.
    """
    f: torch.Tensor
    dx: float
    region: BoxRegion
    mask: torch.Tensor | None = None

    @property
    def nz(self) -> int: return self.f.shape[1]
    @property
    def ny(self) -> int: return self.f.shape[2]
    @property
    def nx(self) -> int: return self.f.shape[3]


# ---------------------------------------------------------------------------
# Multi-Grid Solver
# ---------------------------------------------------------------------------

@dataclass
class MultiGridSolver:
    """Multi-level LBM solver with local refinement.

    Level 0 = coarsest background grid.
    Level 1, 2, ... = successively finer patches.

    Usage::

        solver = MultiGridSolver(coarse_f, fine_regions=[...])
        for step in range(n_steps):
            solver.step(collision_fn, stream_fn, boundary_fn)
    """
    levels: list[GridLevel] = field(default_factory=list)
    ratio: int = 2  # refinement ratio between levels

    @property
    def coarse(self) -> GridLevel:
        return self.levels[0]

    @property
    def finest(self) -> GridLevel:
        return self.levels[-1]

    def add_patch(self, f: torch.Tensor, region: BoxRegion, mask: torch.Tensor | None = None) -> None:
        """Add a finer patch.  *region* is in parent (coarse) coordinates."""
        level_idx = len(self.levels)
        dx = 1.0 / (self.ratio ** level_idx)
        self.levels.append(GridLevel(f=f, dx=dx, region=region, mask=mask))

    def _overlap_region_fine(self, idx: int) -> BoxRegion:
        """Return the fine-level index range that overlaps coarse cell boundaries."""
        r = self.levels[idx].region
        inner = BoxRegion(
            x0=r.x0 * self.ratio + 1,
            x1=r.x1 * self.ratio - 1,
            y0=r.y0 * self.ratio + 1,
            y1=r.y1 * self.ratio - 1,
            z0=r.z0 * self.ratio + 1,
            z1=r.z1 * self.ratio - 1,
        )
        return inner

    def step(
        self,
        collide_fn,
        stream_fn,
        boundary_fn,
        collision_kwargs: dict | None = None,
    ) -> None:
        """Advance all grid levels one coarse time step.

        Fine levels run ``ratio`` sub-steps, exchanging boundary data with
        the coarse level at each sub-step.
        """
        if collision_kwargs is None:
            collision_kwargs = {}

        # 1. Coarse level: one full step
        l0 = self.levels[0]
        l0.f = collide_fn(l0.f, **collision_kwargs)
        l0.f = stream_fn(l0.f)
        boundary_fn(l0)

        # 2. Inject coarse solution into fine-level boundaries
        for li in range(1, len(self.levels)):
            self._inject_boundary(li)

        # 3. Fine levels: ratio sub-steps
        for _sub in range(self.ratio):
            for li in range(1, len(self.levels)):
                lf = self.levels[li]
                lf.f = collide_fn(lf.f, **collision_kwargs)
                lf.f = stream_fn(lf.f)
                boundary_fn(lf)

            # Re-inject boundaries between sub-steps
            for li in range(1, len(self.levels)):
                self._inject_boundary(li)

        # 4. Restrict fine solution back to coarse (overwrite overlap zone)
        for li in range(1, len(self.levels)):
            self._restrict_to_coarse(li)

    def _inject_boundary(self, fine_idx: int) -> None:
        """Interpolate coarse distributions to fine-level boundary cells."""
        lf = self.levels[fine_idx]
        lc = self.levels[0]  # always from coarsest
        r = self.ratio

        # Extract coarse patch corresponding to fine region (with margin)
        r_c = lf.region
        f_coarse_patch = lc.f[:, r_c.z0:r_c.z1, r_c.y0:r_c.y1, r_c.x0:r_c.x1]
        f_fine_interp = _coarse_to_fine_3d(f_coarse_patch, r)

        # Only overwrite boundary shell (exterior 2 cells)
        nz_f, ny_f, nx_f = f_fine_interp.shape[1:]
        boundary_mask = torch.ones((nz_f, ny_f, nx_f), dtype=torch.bool, device=lf.f.device)
        boundary_mask[2:-2, 2:-2, 2:-2] = False

        lf.f[:, boundary_mask] = f_fine_interp[:, boundary_mask]

    def _restrict_to_coarse(self, fine_idx: int) -> None:
        """Average fine solution back onto coarse grid overlap region."""
        lf = self.levels[fine_idx]
        lc = self.levels[0]
        r = self.ratio
        rc = lf.region

        f_fine_avg = _fine_to_coarse_3d(lf.f, r)
        # Write back (skip 1-cell border to avoid boundary pollution)
        lc.f[:, rc.z0 + 1:rc.z1 - 1, rc.y0 + 1:rc.y1 - 1, rc.x0 + 1:rc.x1 - 1] = \
            f_fine_avg[:, 1:-1, 1:-1, 1:-1]


# ---------------------------------------------------------------------------
# Convenience: build refinement region from hull proximity + wake
# ---------------------------------------------------------------------------

def build_refinement_region(
    hull_mask: torch.Tensor,
    margin: int = 4,
    wake_extend: int = 30,
) -> tuple[torch.Tensor, BoxRegion]:
    """Build a combined refinement mask (hull proximity + wake).

    Returns (mask, bounding_box) where mask is on the COARSE grid.
    """
    prox = HullProximityRegion(hull_mask, margin=margin).expand_mask()
    wake = WakeRegion(hull_mask, extend_x=wake_extend).expand_mask()
    combined = prox | wake

    # Find bounding box
    nz, ny, nx = combined.shape
    idx = torch.nonzero(combined)
    if idx.numel() == 0:
        return combined, BoxRegion(0, 1, 0, 1, 0, 1)

    z0, z1 = int(idx[:, 0].min()), int(idx[:, 0].max()) + 1
    y0, y1 = int(idx[:, 1].min()), int(idx[:, 1].max()) + 1
    x0, x1 = int(idx[:, 2].min()), int(idx[:, 2].max()) + 1

    # Pad for overlap
    pad = 4
    z0 = max(0, z0 - pad); z1 = min(nz, z1 + pad)
    y0 = max(0, y0 - pad); y1 = min(ny, y1 + pad)
    x0 = max(0, x0 - pad); x1 = min(nx, x1 + pad)

    return combined, BoxRegion(x0, x1, y0, y1, z0, z1)


__all__ = [
    "BoxRegion", "HullProximityRegion", "WakeRegion",
    "GridLevel", "MultiGridSolver",
    "build_refinement_region",
    "_coarse_to_fine_3d", "_fine_to_coarse_3d",
]
