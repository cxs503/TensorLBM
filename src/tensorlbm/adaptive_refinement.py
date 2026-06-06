"""Adaptive Mesh Refinement (AMR) for LBM — dynamic patch management.

Implements error-indicator–driven refinement where fine patches are added or
removed at runtime based on local flow features.  This contrasts with the
static multi-block approach in ``refinement.py`` / ``multipatch.py``.

Supported lattices
------------------
* D2Q9  — 2-D flows  (``AdaptiveSolver2D``)
* D3Q19 — 3-D flows  (``AdaptiveSolver3D``)

Error indicators
----------------
Three local refinement indicators are provided:

* ``nonequilibrium_indicator_2d`` / ``nonequilibrium_indicator_3d``
  Norm of the non-equilibrium part of the distribution: ``|f - f_eq|``.
  This is the classic LBM refinement criterion (Lagrava et al. 2012).

* ``vorticity_indicator_2d`` / ``vorticity_indicator_3d``
  Vorticity magnitude ``|∇×u|``.  Useful for wake-driven refinement.

* ``gradient_indicator`` (general 2-D / 3-D scalar)
  Gradient magnitude of any scalar field (density, pressure, …).

Workflow
--------
::

    solver = AdaptiveSolver2D(coarse_f, coarse_mask)
    for step in range(n_steps):
        solver.step(collide_fn, stream_fn, boundary_fn)
        if solver.should_adapt(step):
            rho, ux, uy = macroscopic(solver.coarse_f)
            indicator = nonequilibrium_indicator_2d(solver.coarse_f, rho, ux, uy)
            solver.adapt(indicator)

References
----------
Lagrava D., Malaspinas O., Latt J., Chopard B. (2012)
    Advances in multi-domain lattice Boltzmann grid refinement.
    J. Comput. Phys. 231, 4808–4822.
Filippova O., Hänel D. (1998)
    Grid refinement for lattice-BGK models.
    J. Comput. Phys. 147, 219–228.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .refinement import BoxRegion, _coarse_to_fine_3d, _fine_to_coarse_3d

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Utility: 2-D coarse/fine exchange
# ---------------------------------------------------------------------------

def _coarse_to_fine_2d(f_coarse: torch.Tensor, ratio: int = 2) -> torch.Tensor:
    """Bilinear upsample D2Q9 distributions.  Shape: (9, ny, nx) → (9, ny*r, nx*r)."""
    q, ny, nx = f_coarse.shape
    out = F.interpolate(
        f_coarse.unsqueeze(0),          # (1, 9, ny, nx)
        size=(ny * ratio, nx * ratio),
        mode="bilinear",
        align_corners=True,
    )
    return out.squeeze(0)               # (9, ny*r, nx*r)


def _fine_to_coarse_2d(f_fine: torch.Tensor, ratio: int = 2) -> torch.Tensor:
    """Restrict D2Q9 distributions by block-averaging.  Shape: (9, ny*r, nx*r) → (9, ny, nx)."""
    q, ny_f, nx_f = f_fine.shape
    ny_c = ny_f // ratio
    nx_c = nx_f // ratio
    f_r = f_fine.view(q, ny_c, ratio, nx_c, ratio)
    return f_r.mean(dim=(2, 4))


# ---------------------------------------------------------------------------
# Error indicators — 2-D
# ---------------------------------------------------------------------------

def nonequilibrium_indicator_2d(
    f: torch.Tensor,
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
) -> torch.Tensor:
    """Non-equilibrium norm ``‖f − f_eq‖₂`` per cell (D2Q9).

    Args:
        f:   Distribution tensor (9, ny, nx).
        rho: Density field (ny, nx).
        ux:  x-velocity field (ny, nx).
        uy:  y-velocity field (ny, nx).

    Returns:
        Indicator field (ny, nx), values ≥ 0.
    """
    from .d2q9 import equilibrium
    f_eq = equilibrium(rho, ux, uy, device=f.device)
    return (f - f_eq).norm(dim=0)


def vorticity_indicator_2d(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """Vorticity magnitude ``|∂uy/∂x − ∂ux/∂y|`` (D2Q9).

    Args:
        ux: x-velocity (ny, nx).
        uy: y-velocity (ny, nx).

    Returns:
        Indicator field (ny, nx).
    """
    duy_dx = torch.gradient(uy, dim=1)[0]
    dux_dy = torch.gradient(ux, dim=0)[0]
    return (duy_dx - dux_dy).abs()


def gradient_indicator_2d(phi: torch.Tensor) -> torch.Tensor:
    """Gradient magnitude ``‖∇φ‖`` of a scalar field (2-D).

    Args:
        phi: Scalar field (ny, nx).

    Returns:
        Indicator field (ny, nx).
    """
    gx = torch.gradient(phi, dim=1)[0]
    gy = torch.gradient(phi, dim=0)[0]
    return (gx * gx + gy * gy).sqrt()


# ---------------------------------------------------------------------------
# Error indicators — 3-D
# ---------------------------------------------------------------------------

def nonequilibrium_indicator_3d(
    f: torch.Tensor,
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> torch.Tensor:
    """Non-equilibrium norm per cell (D3Q19).

    Args:
        f:   Distribution tensor (19, nz, ny, nx).
        rho: Density (nz, ny, nx).
        ux:  x-velocity (nz, ny, nx).
        uy:  y-velocity (nz, ny, nx).
        uz:  z-velocity (nz, ny, nx).

    Returns:
        Indicator field (nz, ny, nx).
    """
    from .d3q19 import equilibrium3d
    f_eq = equilibrium3d(rho, ux, uy, uz, device=f.device)
    return (f - f_eq).norm(dim=0)


def vorticity_indicator_3d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> torch.Tensor:
    """Vorticity magnitude ``‖∇×u‖`` (D3Q19).

    Args:
        ux, uy, uz: Velocity components (nz, ny, nx).

    Returns:
        Indicator field (nz, ny, nx).
    """
    duz_dy = torch.gradient(uz, dim=1)[0]
    duy_dz = torch.gradient(uy, dim=0)[0]
    dux_dz = torch.gradient(ux, dim=0)[0]
    duz_dx = torch.gradient(uz, dim=2)[0]
    duy_dx = torch.gradient(uy, dim=2)[0]
    dux_dy = torch.gradient(ux, dim=1)[0]
    wx = duz_dy - duy_dz
    wy = dux_dz - duz_dx
    wz = duy_dx - dux_dy
    return (wx * wx + wy * wy + wz * wz).sqrt()


def gradient_indicator_3d(phi: torch.Tensor) -> torch.Tensor:
    """Gradient magnitude ``‖∇φ‖`` of a scalar field (3-D).

    Args:
        phi: Scalar field (nz, ny, nx).

    Returns:
        Indicator field (nz, ny, nx).
    """
    gx = torch.gradient(phi, dim=2)[0]
    gy = torch.gradient(phi, dim=1)[0]
    gz = torch.gradient(phi, dim=0)[0]
    return (gx * gx + gy * gy + gz * gz).sqrt()


# ---------------------------------------------------------------------------
# Cell marking
# ---------------------------------------------------------------------------

def mark_cells_for_refinement(
    indicator: torch.Tensor,
    refine_threshold: float,
    coarsen_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return binary masks identifying cells to refine or coarsen.

    A cell is flagged for **refinement** when ``indicator > refine_threshold``
    and for **coarsening** when ``indicator < coarsen_threshold``.

    Args:
        indicator:         Error indicator field (ny, nx) or (nz, ny, nx).
        refine_threshold:  Upper threshold — cells above this value are refined.
        coarsen_threshold: Lower threshold — cells below this value are coarsened.
                           Must satisfy ``coarsen_threshold < refine_threshold``.

    Returns:
        (refine_mask, coarsen_mask) — boolean tensors of the same shape as
        *indicator*.
    """
    if coarsen_threshold >= refine_threshold:
        raise ValueError(
            f"coarsen_threshold ({coarsen_threshold}) must be strictly less than "
            f"refine_threshold ({refine_threshold})"
        )
    refine_mask = indicator > refine_threshold
    coarsen_mask = indicator < coarsen_threshold
    return refine_mask, coarsen_mask


def _bounding_box_2d(mask: torch.Tensor, pad: int = 2) -> BoxRegion | None:
    """Tight 2-D bounding box (returned as BoxRegion with z0=z1=0)."""
    idx = torch.nonzero(mask)
    if idx.numel() == 0:
        return None
    ny, nx = mask.shape
    y0 = max(0, int(idx[:, 0].min()) - pad)
    y1 = min(ny, int(idx[:, 0].max()) + 1 + pad)
    x0 = max(0, int(idx[:, 1].min()) - pad)
    x1 = min(nx, int(idx[:, 1].max()) + 1 + pad)
    return BoxRegion(x0, x1, y0, y1, 0, 0)


def _group_refine_boxes_2d(
    refine_mask: torch.Tensor,
    pad: int = 2,
    max_patches: int = 8,
) -> list[BoxRegion]:
    """Convert a refinement mask into a list of non-overlapping bounding boxes.

    Uses a simple connected-component–style splitting: the mask is divided
    into at most *max_patches* column strips, each converted to a tight box.
    This keeps implementation simple while covering typical LBM use cases
    (boundary layers, wake, shear layers).

    Args:
        refine_mask: Boolean field (ny, nx).
        pad:         Cell padding added to each box.
        max_patches: Maximum number of patches to generate.

    Returns:
        List of :class:`BoxRegion` objects (in coarse-grid coordinates).
    """
    if not refine_mask.any():
        return []
    ny, nx = refine_mask.shape
    # Find column ranges that contain any flagged cell
    col_active = refine_mask.any(dim=0)  # (nx,)
    active_cols = torch.nonzero(col_active).squeeze(1)
    if active_cols.numel() == 0:
        return []
    # Split into strips
    x_min = int(active_cols.min())
    x_max = int(active_cols.max()) + 1
    strip_len = max(1, (x_max - x_min + max_patches - 1) // max_patches)
    boxes: list[BoxRegion] = []
    x = x_min
    while x < x_max and len(boxes) < max_patches:
        x0 = max(0, x - pad)
        x1 = min(nx, x + strip_len + pad)
        strip_mask = refine_mask[:, x0:x1]
        if strip_mask.any():
            row_active = strip_mask.any(dim=1)
            row_idx = torch.nonzero(row_active).squeeze(1)
            y0 = max(0, int(row_idx.min()) - pad)
            y1 = min(ny, int(row_idx.max()) + 1 + pad)
            boxes.append(BoxRegion(x0, x1, y0, y1, 0, 0))
        x += strip_len
    return boxes


# ---------------------------------------------------------------------------
# Adaptation schedule
# ---------------------------------------------------------------------------

@dataclass
class AdaptationSchedule:
    """Controls when the AMR solver adapts its patch structure.

    Attributes:
        interval:    Adapt every *interval* coarse time steps.
        warmup:      Do not adapt for the first *warmup* steps (allow flow to
                     develop).
        max_patches: Maximum number of fine patches active at one time.
        refine_threshold:  Indicator value above which cells are refined.
        coarsen_threshold: Indicator value below which patches may be removed.
    """
    interval: int = 20
    warmup: int = 0
    max_patches: int = 8
    refine_threshold: float = 1e-3
    coarsen_threshold: float = 1e-5

    def should_adapt(self, step: int) -> bool:
        """Return True if the solver should adapt at *step*."""
        return step >= self.warmup and (step - self.warmup) % self.interval == 0


# ---------------------------------------------------------------------------
# 2-D adaptive patch
# ---------------------------------------------------------------------------

@dataclass
class AMRPatch2D:
    """A dynamically managed fine-resolution D2Q9 patch.

    Attributes:
        f:     Distribution tensor (9, ny_f, nx_f).
        box:   Bounding box in coarse-grid coordinates (z fields unused).
        ratio: Refinement ratio relative to coarse grid (default 2).
    """
    f: torch.Tensor
    box: BoxRegion
    ratio: int = 2

    @property
    def ny(self) -> int:
        return self.f.shape[1]

    @property
    def nx(self) -> int:
        return self.f.shape[2]


# ---------------------------------------------------------------------------
# 2-D adaptive solver (D2Q9)
# ---------------------------------------------------------------------------

class AdaptiveSolver2D:
    """Adaptive-mesh LBM solver for 2-D flows (D2Q9).

    The solver maintains a coarse background grid and a dynamic list of
    fine patches.  At each adaptation step it:

    1. Evaluates an error indicator on the coarse field.
    2. Marks cells for refinement / coarsening.
    3. Adds new fine patches over flagged regions.
    4. Removes patches whose indicator has dropped below the coarsen
       threshold.

    Usage::

        from tensorlbm.adaptive_refinement import (
            AdaptiveSolver2D, AdaptationSchedule,
            nonequilibrium_indicator_2d,
        )
        from tensorlbm.d2q9 import macroscopic

        schedule = AdaptationSchedule(interval=50, refine_threshold=1e-3)
        solver = AdaptiveSolver2D(f_coarse, schedule=schedule)

        for step in range(n_steps):
            solver.step(collide_fn, stream_fn, boundary_fn)
            if solver.should_adapt(step):
                rho, ux, uy = macroscopic(solver.coarse_f)
                indicator = nonequilibrium_indicator_2d(solver.coarse_f, rho, ux, uy)
                solver.adapt(indicator)

    Args:
        coarse_f:  Initial coarse-grid distributions (9, ny, nx).
        schedule:  Adaptation schedule.  Defaults to
                   ``AdaptationSchedule()``.
        mask:      Optional coarse solid mask (ny, nx).
    """

    def __init__(
        self,
        coarse_f: torch.Tensor,
        schedule: AdaptationSchedule | None = None,
        mask: torch.Tensor | None = None,
    ) -> None:
        self.coarse_f: torch.Tensor = coarse_f
        self.schedule: AdaptationSchedule = schedule or AdaptationSchedule()
        self.mask: torch.Tensor | None = mask
        self.patches: list[AMRPatch2D] = []
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self.coarse_f.device

    def should_adapt(self, step: int) -> bool:
        """Convenience wrapper around the schedule."""
        return self.schedule.should_adapt(step)

    def adapt(self, indicator: torch.Tensor) -> None:
        """Update the patch list based on *indicator*.

        Existing patches whose maximum indicator has fallen below
        ``schedule.coarsen_threshold`` are removed.  New patches are
        created for regions exceeding ``schedule.refine_threshold``.

        Args:
            indicator: Error indicator (ny, nx), same spatial size as the
                       coarse grid.
        """
        refine_mask, coarsen_mask = mark_cells_for_refinement(
            indicator,
            self.schedule.refine_threshold,
            self.schedule.coarsen_threshold,
        )

        # --- coarsening: restrict and remove expired patches ------------
        surviving: list[AMRPatch2D] = []
        for patch in self.patches:
            b = patch.box
            local_indicator = indicator[b.y0:b.y1, b.x0:b.x1]
            if local_indicator.max().item() < self.schedule.coarsen_threshold:
                # Restrict fine solution back to coarse before discarding
                self._restrict_patch_to_coarse(patch)
            else:
                surviving.append(patch)
        self.patches = surviving

        # --- refinement: add new patches --------------------------------
        if refine_mask.any():
            new_boxes = _group_refine_boxes_2d(
                refine_mask,
                pad=2,
                max_patches=self.schedule.max_patches,
            )
            for box in new_boxes:
                if len(self.patches) >= self.schedule.max_patches:
                    break
                if not self._patch_exists(box):
                    self._add_patch(box, ratio=2)

    def step(
        self,
        collide_fn: Callable,
        stream_fn: Callable,
        boundary_fn: Callable,
    ) -> None:
        """Advance one coarse time step.

        The coarse grid takes one full step.  Each fine patch takes
        ``ratio`` sub-steps (fine time = coarse time / ratio).

        Args:
            collide_fn:   Collision operator ``f → f'``.
            stream_fn:    Streaming operator ``f → f'``.
            boundary_fn:  Boundary-condition operator ``f → f'``.
        """
        # --- 1. Coarse step --------------------------------------------
        self.coarse_f = collide_fn(self.coarse_f)
        self.coarse_f = stream_fn(self.coarse_f)
        self.coarse_f = boundary_fn(self.coarse_f)

        # --- 2. Inject coarse → fine boundaries -----------------------
        for patch in self.patches:
            self._inject_to_patch(patch)

        # --- 3. Fine sub-steps ----------------------------------------
        for _ in range(patch.ratio if self.patches else 0):
            for patch in self.patches:
                patch.f = collide_fn(patch.f)
                patch.f = stream_fn(patch.f)
                patch.f = boundary_fn(patch.f)
            # Re-inject boundaries after each sub-step
            for patch in self.patches:
                self._inject_to_patch(patch)

        # --- 4. Restrict fine → coarse --------------------------------
        for patch in self.patches:
            self._restrict_patch_to_coarse(patch)

        self._step_count += 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_patch(self, box: BoxRegion, ratio: int = 2) -> None:
        """Initialise a new fine patch from the current coarse field."""
        f_coarse_patch = self.coarse_f[
            :, box.y0:box.y1, box.x0:box.x1
        ]
        f_fine = _coarse_to_fine_2d(f_coarse_patch, ratio)
        self.patches.append(AMRPatch2D(f=f_fine, box=box, ratio=ratio))

    def _inject_to_patch(self, patch: AMRPatch2D) -> None:
        """Overwrite fine-level boundary cells with upsampled coarse values."""
        b = patch.box
        r = patch.ratio
        f_coarse_patch = self.coarse_f[:, b.y0:b.y1, b.x0:b.x1]
        f_up = _coarse_to_fine_2d(f_coarse_patch, r)

        ny_f, nx_f = patch.f.shape[1], patch.f.shape[2]
        border = torch.ones((ny_f, nx_f), dtype=torch.bool, device=self.device)
        border[r:-r, r:-r] = False

        # Clamp upsampled size to patch size in case of rounding
        fy = min(f_up.shape[1], ny_f)
        fx = min(f_up.shape[2], nx_f)
        patch.f[:, border[:fy, :fx]] = f_up[:, border[:fy, :fx]]

    def _restrict_patch_to_coarse(self, patch: AMRPatch2D) -> None:
        """Average fine interior back to the coarse grid."""
        b = patch.box
        r = patch.ratio
        f_avg = _fine_to_coarse_2d(patch.f, r)
        ny_c = b.y1 - b.y0
        nx_c = b.x1 - b.x0
        avg_y = min(f_avg.shape[1], ny_c)
        avg_x = min(f_avg.shape[2], nx_c)
        self.coarse_f[:, b.y0:b.y0 + avg_y, b.x0:b.x0 + avg_x] = (
            f_avg[:, :avg_y, :avg_x]
        )

    def _patch_exists(self, box: BoxRegion) -> bool:
        """Return True if a patch overlapping *box* already exists."""
        for p in self.patches:
            b = p.box
            x_overlap = b.x0 < box.x1 and b.x1 > box.x0
            y_overlap = b.y0 < box.y1 and b.y1 > box.y0
            if x_overlap and y_overlap:
                return True
        return False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def n_patches(self) -> int:
        """Number of active fine patches."""
        return len(self.patches)

    @property
    def total_cells(self) -> int:
        """Total cell count (coarse + all fine patches)."""
        q, ny, nx = self.coarse_f.shape
        total = ny * nx
        for p in self.patches:
            total += p.ny * p.nx
        return total

    def patch_info(self) -> list[dict]:
        """Return a list of dicts describing active patches."""
        return [
            {
                "box": p.box,
                "ratio": p.ratio,
                "ny": p.ny,
                "nx": p.nx,
                "cells": p.ny * p.nx,
            }
            for p in self.patches
        ]


# ---------------------------------------------------------------------------
# 3-D adaptive patch
# ---------------------------------------------------------------------------

@dataclass
class AMRPatch3D:
    """A dynamically managed fine-resolution D3Q19 patch.

    Attributes:
        f:     Distribution tensor (19, nz_f, ny_f, nx_f).
        box:   Bounding box in coarse-grid coordinates.
        ratio: Refinement ratio relative to coarse grid (default 2).
    """
    f: torch.Tensor
    box: BoxRegion
    ratio: int = 2

    @property
    def nz(self) -> int:
        return self.f.shape[1]

    @property
    def ny(self) -> int:
        return self.f.shape[2]

    @property
    def nx(self) -> int:
        return self.f.shape[3]


# ---------------------------------------------------------------------------
# 3-D adaptive solver (D3Q19)
# ---------------------------------------------------------------------------

class AdaptiveSolver3D:
    """Adaptive-mesh LBM solver for 3-D flows (D3Q19).

    Mirrors :class:`AdaptiveSolver2D` for three-dimensional simulations.
    Patch injection and restriction use trilinear interpolation / block
    averaging provided by ``refinement._coarse_to_fine_3d`` and
    ``_fine_to_coarse_3d``.

    Usage::

        from tensorlbm.adaptive_refinement import (
            AdaptiveSolver3D, AdaptationSchedule,
            nonequilibrium_indicator_3d,
        )
        from tensorlbm.d3q19 import macroscopic3d

        schedule = AdaptationSchedule(interval=10, refine_threshold=5e-4)
        solver = AdaptiveSolver3D(f_coarse, schedule=schedule)

        for step in range(n_steps):
            solver.step(collide_fn, stream_fn, boundary_fn)
            if solver.should_adapt(step):
                rho, ux, uy, uz = macroscopic3d(solver.coarse_f)
                indicator = nonequilibrium_indicator_3d(
                    solver.coarse_f, rho, ux, uy, uz
                )
                solver.adapt(indicator)

    Args:
        coarse_f:  Initial coarse-grid distributions (19, nz, ny, nx).
        schedule:  Adaptation schedule.
        mask:      Optional coarse solid mask (nz, ny, nx).
    """

    def __init__(
        self,
        coarse_f: torch.Tensor,
        schedule: AdaptationSchedule | None = None,
        mask: torch.Tensor | None = None,
    ) -> None:
        self.coarse_f: torch.Tensor = coarse_f
        self.schedule: AdaptationSchedule = schedule or AdaptationSchedule()
        self.mask: torch.Tensor | None = mask
        self.patches: list[AMRPatch3D] = []
        self._step_count: int = 0

    @property
    def device(self) -> torch.device:
        return self.coarse_f.device

    def should_adapt(self, step: int) -> bool:
        return self.schedule.should_adapt(step)

    def adapt(self, indicator: torch.Tensor) -> None:
        """Update patches based on *indicator* (nz, ny, nx)."""
        refine_mask, coarsen_mask = mark_cells_for_refinement(
            indicator,
            self.schedule.refine_threshold,
            self.schedule.coarsen_threshold,
        )

        # Coarsening
        surviving: list[AMRPatch3D] = []
        for patch in self.patches:
            b = patch.box
            local = indicator[b.z0:b.z1, b.y0:b.y1, b.x0:b.x1]
            if local.max().item() < self.schedule.coarsen_threshold:
                self._restrict_patch_to_coarse(patch)
            else:
                surviving.append(patch)
        self.patches = surviving

        # Refinement
        if refine_mask.any():
            new_boxes = _group_refine_boxes_3d(
                refine_mask, pad=2, max_patches=self.schedule.max_patches
            )
            for box in new_boxes:
                if len(self.patches) >= self.schedule.max_patches:
                    break
                if not self._patch_exists(box):
                    self._add_patch(box, ratio=2)

    def step(
        self,
        collide_fn: Callable,
        stream_fn: Callable,
        boundary_fn: Callable,
    ) -> None:
        """Advance one coarse time step (fine patches sub-step *ratio* times)."""
        # 1. Coarse step
        self.coarse_f = collide_fn(self.coarse_f)
        self.coarse_f = stream_fn(self.coarse_f)
        self.coarse_f = boundary_fn(self.coarse_f)

        # 2. Inject coarse → fine boundaries
        for patch in self.patches:
            self._inject_to_patch(patch)

        # 3. Fine sub-steps
        ratio = self.patches[0].ratio if self.patches else 0
        for _ in range(ratio):
            for patch in self.patches:
                patch.f = collide_fn(patch.f)
                patch.f = stream_fn(patch.f)
                patch.f = boundary_fn(patch.f)
            for patch in self.patches:
                self._inject_to_patch(patch)

        # 4. Restrict fine → coarse
        for patch in self.patches:
            self._restrict_patch_to_coarse(patch)

        self._step_count += 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_patch(self, box: BoxRegion, ratio: int = 2) -> None:
        f_c = self.coarse_f[:, box.z0:box.z1, box.y0:box.y1, box.x0:box.x1]
        f_f = _coarse_to_fine_3d(f_c, ratio)
        self.patches.append(AMRPatch3D(f=f_f, box=box, ratio=ratio))

    def _inject_to_patch(self, patch: AMRPatch3D) -> None:
        b = patch.box
        r = patch.ratio
        f_c = self.coarse_f[:, b.z0:b.z1, b.y0:b.y1, b.x0:b.x1]
        f_up = _coarse_to_fine_3d(f_c, r)

        nz_f, ny_f, nx_f = patch.f.shape[1:]
        border = torch.ones((nz_f, ny_f, nx_f), dtype=torch.bool, device=self.device)
        border[r:-r, r:-r, r:-r] = False

        fz = min(f_up.shape[1], nz_f)
        fy = min(f_up.shape[2], ny_f)
        fx = min(f_up.shape[3], nx_f)
        patch.f[:, border[:fz, :fy, :fx]] = f_up[:, border[:fz, :fy, :fx]]

    def _restrict_patch_to_coarse(self, patch: AMRPatch3D) -> None:
        b = patch.box
        r = patch.ratio
        f_avg = _fine_to_coarse_3d(patch.f, r)
        nz_c = b.z1 - b.z0
        ny_c = b.y1 - b.y0
        nx_c = b.x1 - b.x0
        az = min(f_avg.shape[1], nz_c)
        ay = min(f_avg.shape[2], ny_c)
        ax = min(f_avg.shape[3], nx_c)
        self.coarse_f[
            :,
            b.z0:b.z0 + az,
            b.y0:b.y0 + ay,
            b.x0:b.x0 + ax,
        ] = f_avg[:, :az, :ay, :ax]

    def _patch_exists(self, box: BoxRegion) -> bool:
        for p in self.patches:
            b = p.box
            if (b.x0 < box.x1 and b.x1 > box.x0
                    and b.y0 < box.y1 and b.y1 > box.y0
                    and b.z0 < box.z1 and b.z1 > box.z0):
                return True
        return False

    @property
    def n_patches(self) -> int:
        return len(self.patches)

    @property
    def total_cells(self) -> int:
        q, nz, ny, nx = self.coarse_f.shape
        total = nz * ny * nx
        for p in self.patches:
            total += p.nz * p.ny * p.nx
        return total

    def patch_info(self) -> list[dict]:
        return [
            {
                "box": p.box,
                "ratio": p.ratio,
                "nz": p.nz,
                "ny": p.ny,
                "nx": p.nx,
                "cells": p.nz * p.ny * p.nx,
            }
            for p in self.patches
        ]


# ---------------------------------------------------------------------------
# 3-D box grouping helper
# ---------------------------------------------------------------------------

def _group_refine_boxes_3d(
    refine_mask: torch.Tensor,
    pad: int = 2,
    max_patches: int = 8,
) -> list[BoxRegion]:
    """Convert a 3-D refinement mask to a list of bounding box patches.

    Splits the mask into at most *max_patches* strips along x.

    Args:
        refine_mask: Boolean field (nz, ny, nx).
        pad:         Cell padding per axis.
        max_patches: Maximum number of patches.

    Returns:
        List of :class:`BoxRegion` objects in coarse coordinates.
    """
    if not refine_mask.any():
        return []
    nz, ny, nx = refine_mask.shape
    col_active = refine_mask.any(dim=(0, 1))  # (nx,)
    active_cols = torch.nonzero(col_active).squeeze(1)
    if active_cols.numel() == 0:
        return []
    x_min = int(active_cols.min())
    x_max = int(active_cols.max()) + 1
    strip_len = max(1, (x_max - x_min + max_patches - 1) // max_patches)
    boxes: list[BoxRegion] = []
    x = x_min
    while x < x_max and len(boxes) < max_patches:
        x0 = max(0, x - pad)
        x1 = min(nx, x + strip_len + pad)
        strip = refine_mask[:, :, x0:x1]
        if strip.any():
            row_active = strip.any(dim=(0, 2))
            row_idx = torch.nonzero(row_active).squeeze(1)
            y0 = max(0, int(row_idx.min()) - pad)
            y1 = min(ny, int(row_idx.max()) + 1 + pad)

            slab_active = strip.any(dim=(1, 2))
            slab_idx = torch.nonzero(slab_active).squeeze(1)
            z0 = max(0, int(slab_idx.min()) - pad)
            z1 = min(nz, int(slab_idx.max()) + 1 + pad)

            boxes.append(BoxRegion(x0, x1, y0, y1, z0, z1))
        x += strip_len
    return boxes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Indicators — 2-D
    "nonequilibrium_indicator_2d",
    "vorticity_indicator_2d",
    "gradient_indicator_2d",
    # Indicators — 3-D
    "nonequilibrium_indicator_3d",
    "vorticity_indicator_3d",
    "gradient_indicator_3d",
    # Cell marking
    "mark_cells_for_refinement",
    # Schedule
    "AdaptationSchedule",
    # 2-D AMR
    "AMRPatch2D",
    "AdaptiveSolver2D",
    # 3-D AMR
    "AMRPatch3D",
    "AdaptiveSolver3D",
    # Internal (exported for testing)
    "_coarse_to_fine_2d",
    "_fine_to_coarse_2d",
    "_group_refine_boxes_2d",
    "_group_refine_boxes_3d",
]
