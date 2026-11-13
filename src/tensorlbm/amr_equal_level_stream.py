"""Equal-level border stream correction for 2:1 block-structured LBM (D2Q9).

Cross-validation target
-----------------------
``lssfau/walberla`` ``src/lbm/refinement/EqualLevelBorderStreamCorrection.h``
(see ``oss_solutions`` id 283, tag ``block-amr-equal-level-stream-correction``):
*2:1 block refinement in which the coarse cells covered by a finer block are
not maintained; a coarse cell that borders the finer block therefore has no
same-level source population for the stencil directions that point into the
refined region.  Before streaming the missing populations are replaced by
values copied from the equally-sized (same-level) neighbour cell; after
streaming the substituted values are restored so the correction never
persists in the state.*

This module implements that mechanism for the 2-D D2Q9 lattice together with a
minimal two-level composite runtime that can run the three coupling choices the
correction is about:

``coupling="blocked"``
    waLBerla practice: coarse cells under the fine patch are frozen (never
    advanced).  Border cells either take the stale value (``stream_correction
    =False``, the negative control) or the equal-level copy (``True``).
``coupling="restricted"``
    TensorLBM practice (``tensorlbm.static_block_amr`` /
    ``tensorlbm.amr_cavity2d``): the coarse cells under the fine patch are
    refreshed every step by restricting the fine solution, so a same-level
    source always exists and no substitution is required.

Everything is NumPy/float64 by default so accuracy studies are not polluted by
single-precision round-off.  The D2Q9 conventions and the regularised BGK
collision are imported from :mod:`tensorlbm.amr_cavity2d` (same lattice order,
already audited in this repository) instead of being re-derived.

Units
-----
Coarse grid: ``dx = 1``, ``dt = 1``.  Fine grid: ``dx = 1/2``, ``dt = 1/2``
(convective 2:1 refinement), hence ``nu_f = 2 * nu_c`` and
``tau_f = 0.5 + 2 * (tau_c - 0.5)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .amr_cavity2d import (
    _C,
    _OPP,
    _collide_rlbm,
    _equilibrium,
    _macroscopic,
    _pre_bounce,
    _stream,
)

__all__ = [
    "EqualLevelBorderStreamCorrection",
    "analytic_fields_2d",
    "calibrate_prolongation_scale",
    "ghost_ring_from_parent",
    "prolongation_scale",
    "restriction_scale",
]

_TGEN = _equilibrium


def prolongation_scale(
    tau_coarse: float,
    tau_fine: float,
    ratio: int = 2,
    convention: str | float = "fh_spacing",
) -> float:
    """Non-equilibrium amplification used when coarse data feed a fine ghost.

    Conventions kept side by side on purpose -- the repository currently holds
    two different ones and the cross-validation benchmark measures which one is
    consistent with the analytic solution:

    ``"fh_spacing"``
        Filippova-Haenel with explicit grid-spacing ratio,
        ``(tau_f - 1/2) / (tau_c - 1/2) * ratio``.  Scales the resolved
        non-equilibrium stress when ``du/dx`` doubles while ``dx`` halves.
    ``"fh_plain"``
        ``(tau_f - 1/2) / (tau_c - 1/2)`` (no spacing factor).
    ``"repo_torch"``
        :func:`tensorlbm.amr_population_transfer.rescale_nonequilibrium`
        with ``tau_target=tau_f``, ``spatial_ratio=1/ratio``.
    ``"repo_cavity"``
        :func:`tensorlbm.amr_cavity2d._fh_rescale`, i.e. ``tau_f/(ratio*tau_c)``.
    """
    if isinstance(convention, (int, float)) and not isinstance(convention, bool):
        scale = float(convention)
    elif convention == "fh_spacing":
        scale = (tau_fine - 0.5) / (tau_coarse - 0.5) * ratio
    elif convention == "fh_plain":
        scale = (tau_fine - 0.5) / (tau_coarse - 0.5)
    elif convention == "repo_torch":
        scale = tau_fine / ((1.0 / ratio) * tau_coarse)
    elif convention == "repo_cavity":
        scale = tau_fine / (ratio * tau_coarse)
    else:  # pragma: no cover - guarded by the config validator
        raise ValueError(f"unknown prolongation convention {convention!r}")
    if not (math.isfinite(scale) and scale > 0.0):
        raise ValueError("prolongation scale must be finite and positive")
    return scale


def restriction_scale(prolong_scale: float) -> float:
    """Exact inverse of :func:`prolongation_scale` (round-trip consistency)."""
    return 1.0 / prolong_scale


class EqualLevelBorderStreamCorrection:
    """Population-level ``prepare``/``apply`` correction of equal-level borders.

    Built from the boolean mask of coarse cells that are *not* advanced at
    level 0 (``blocked``; the cells covered by the fine patch in the waLBerla
    coupling).  For every active coarse cell ``dst`` and stencil direction
    ``q`` whose pull source ``dst + c_q`` is blocked, the population
    ``f_q(dst)`` is unavailable at the same level.  ``prepare`` saves it and
    substitutes the population of the equally-sized neighbour
    ``dst - c_q``; ``apply`` restores the saved value.

    The substitution is therefore *streaming-time only*: after
    ``restore(prepare(f))`` the state is bit-identical to the input, which is
    asserted by ``tests/test_amr_equal_level_stream.py``.
    """

    def __init__(
        self,
        blocked: np.ndarray,
        *,
        periodic: bool = True,
        donor: str = "mirror",
    ) -> None:
        blocked = np.asarray(blocked, dtype=bool)
        if blocked.ndim != 2:
            raise ValueError("blocked must be a 2-D boolean mask")
        if donor not in ("mirror", "self"):
            raise ValueError("donor must be 'mirror' or 'self'")
        ny, nx = blocked.shape
        rows, cols = np.nonzero(~blocked)
        dirs: list[int] = []
        dst: list[int] = []   # blocked source cell whose slot is substituted
        src: list[int] = []   # equally-sized donor cell the value is copied from
        for q in range(1, 9):
            cx, cy = int(_C[q, 0]), int(_C[q, 1])
            sr = rows + cy
            sc = cols + cx
            if periodic:
                sr = sr % ny
                sc = sc % nx
            else:
                inside = (sr >= 0) & (sr < ny) & (sc >= 0) & (sc < nx)
                rows_q, cols_q, sr, sc = rows[inside], cols[inside], sr[inside], sc[inside]
            hit = blocked[sr, sc]
            if not hit.any():
                continue
            # donor: 'mirror' -> cell of the destination mirrored through the
            # border (one further away from the patch); 'self' -> the
            # destination itself (zero-order extrapolation).
            back = 2 if donor == "mirror" else 1
            mr = rows[hit if not periodic else slice(None)][hit] if False else None
            base_rows = rows if periodic else rows_q
            base_cols = cols if periodic else cols_q
            mr = (base_rows - back * cy)[hit]
            mc = (base_cols - back * cx)[hit]
            if periodic:
                mr = mr % ny
                mc = mc % nx
            else:
                keep = (mr >= 0) & (mr < ny) & (mc >= 0) & (mc < nx)
                if not keep.any():
                    continue
                dirs.extend([q] * int(keep.sum()))
                dst.extend((sr[hit][keep] * nx + sc[hit][keep]).tolist())
                src.extend((mr[keep] * nx + mc[keep]).tolist())
                continue
            dirs.extend([q] * int(hit.sum()))
            dst.extend((sr[hit] * nx + sc[hit]).tolist())
            src.extend((mr * nx + mc).tolist())
        self.blocked = blocked
        self.periodic = periodic
        self.donor = donor
        self.dirs = np.asarray(dirs, dtype=np.int64)
        self.dst = np.asarray(dst, dtype=np.int64)
        self.src = np.asarray(src, dtype=np.int64)
        self._nx = nx
        del mr

    def __len__(self) -> int:
        return int(self.dirs.size)

    @property
    def stencil_directions(self) -> tuple[int, ...]:
        """Distinct stencil directions the correction touches (sorted)."""
        return tuple(sorted(int(q) for q in np.unique(self.dirs)))

    def _view(self, f: np.ndarray) -> np.ndarray:
        if f.ndim != 3 or f.shape[0] != 9:
            raise ValueError("f must have shape (9, ny, nx)")
        if f.shape[2] != self._nx or f.shape[1] != self.blocked.shape[0]:
            raise ValueError("f shape does not match the mask the plan was built from")
        if not f.flags["C_CONTIGUOUS"]:
            raise ValueError("f must be C-contiguous for flat-index correction")
        return f.reshape(9, -1)

    def prepare(self, f: np.ndarray) -> np.ndarray:
        """Substitute blocked-direction populations; returns the saved values."""
        g = self._view(f)
        saved = g[self.dirs, self.dst].copy()
        g[self.dirs, self.dst] = g[self.dirs, self.src]
        return saved

    def restore(self, f: np.ndarray, saved: np.ndarray) -> None:
        """Undo :meth:`prepare` (state must be bit-identical afterwards)."""
        g = self._view(f)
        g[self.dirs, self.dst] = saved

    def stream_with_correction(self, post: np.ndarray) -> np.ndarray:
        """``prepare -> stream -> restore`` in one call (pull, double periodic)."""
        saved = self.prepare(post)
        try:
            streamed = _stream(post)
        finally:
            self.restore(post, saved)
        return streamed

    def summary(self) -> dict[str, object]:
        return {
            "links": len(self),
            "directions": list(self.stencil_directions),
            "blocked_cells": int(self.blocked.sum()),
            "periodic": self.periodic,
        }




def _analytic_field_error(
    coarse: np.ndarray,
    fine: np.ndarray,
    scale: float,
    *,
    ratio: int = 2,
) -> float:
    """Relative L2 mismatch of the *non-equilibrium* part after prolongation.

    ``coarse``/``fine`` are D2Q9 fields of the same analytic solution sampled on
    the coarse lattice and on the fine patch (piecewise-constant coarse
    prolongation reproduces the same physical points).  Comparing only the
    non-equilibrium part removes the O(h^2) *sampling* offset that is present
    for every scale and would otherwise mask the scale-dependent error.
    """
    rho_c, ux_c, uy_c = _macroscopic(coarse)
    feq_c = _equilibrium(rho_c, ux_c, uy_c)
    neq_c = coarse - feq_c
    neq_c_fine = np.repeat(np.repeat(neq_c, ratio, axis=1), ratio, axis=2)
    rho_f, ux_f, uy_f = _macroscopic(fine)
    neq_f = fine - _equilibrium(rho_f, ux_f, uy_f)
    num = float(np.linalg.norm((scale * neq_c_fine - neq_f).ravel()))
    den = float(np.linalg.norm(neq_f.ravel()))
    return num / den


def analytic_fields_2d(
    n_coarse: int,
    patch: int,
    velocity,
    *,
    ratio: int = 2,
    dtype=np.float64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample one analytic solution on a coarse grid and on its 2:1 fine patch.

    Returns ``(coarse, fine, upsampled_coarse, patch_mask)`` where
    ``upsampled_coarse`` is the coarse field repeated onto the fine lattice
    (piecewise-constant prolongation without rescaling).
    """
    p0 = (n_coarse - patch) // 2
    xc = (np.arange(n_coarse, dtype=dtype) + 0.5)[None, :] * np.ones((n_coarse, 1), dtype=dtype)
    yc = (np.arange(n_coarse, dtype=dtype) + 0.5)[:, None] * np.ones((1, n_coarse), dtype=dtype)
    uxc, uyc = velocity(xc, yc)
    coarse = _equilibrium(
        np.ones_like(uxc, dtype=dtype), np.asarray(uxc, dtype=dtype), np.asarray(uyc, dtype=dtype),
    ).astype(dtype)
    nf = ratio * patch
    xf = (p0 + (np.arange(nf, dtype=dtype) + 0.5) / ratio)[None, :] * np.ones((nf, 1), dtype=dtype)
    yf = (p0 + (np.arange(nf, dtype=dtype) + 0.5) / ratio)[:, None] * np.ones((1, nf), dtype=dtype)
    uxf, uyf = velocity(xf, yf)
    fine = _equilibrium(
        np.ones_like(uxf, dtype=dtype), np.asarray(uxf, dtype=dtype), np.asarray(uyf, dtype=dtype),
    ).astype(dtype)
    upsampled = np.repeat(np.repeat(coarse[:, p0 : p0 + patch, p0 : p0 + patch], ratio, axis=1), ratio, axis=2)
    mask = np.zeros((n_coarse, n_coarse), dtype=bool)
    mask[p0 : p0 + patch, p0 : p0 + patch] = True
    return coarse, fine, upsampled, mask


def calibrate_prolongation_scale(
    coarse: np.ndarray,
    fine: np.ndarray,
    scales: np.ndarray,
) -> dict[str, object]:
    """Scan prolongation scales on an exact analytic field pair."""
    errors = [_analytic_field_error(coarse, fine, float(s)) for s in scales]
    best = int(np.argmin(errors))
    order = np.argsort(errors)
    return {
        "scales": [float(s) for s in scales],
        "errors": errors,
        "best_scale": float(scales[best]),
        "best_error": float(errors[best]),
        "runner_up_scale": float(scales[order[1]]) if len(order) > 1 else None,
        "objective_gap": float(errors[order[1]] - errors[best]) if len(order) > 1 else 0.0,
    }


def ghost_ring_from_parent(
    parent: np.ndarray,
    fine_shape: tuple[int, int],
    patch_origin: tuple[int, int],
    scale: float,
    *,
    ratio: int = 2,
    periodic: bool = True,
) -> np.ndarray:
    """Filippova-Haenel ghost ring (one cell wide) for an interior fine patch."""
    p0, q0 = patch_origin
    npy, npx = parent.shape[1:]
    fi = np.arange(-1, fine_shape[0] + 1, dtype=np.int64)
    fj = np.arange(-1, fine_shape[1] + 1, dtype=np.int64)
    ci = p0 + np.floor((fi + 0.5) / ratio).astype(np.int64)
    cj = q0 + np.floor((fj + 0.5) / ratio).astype(np.int64)
    if periodic:
        ci, cj = ci % npy, cj % npx
    else:
        ci, cj = np.clip(ci, 0, npy - 1), np.clip(cj, 0, npx - 1)
    sub = parent[:, ci[:, None], cj[None, :]]
    rho, ux, uy = _macroscopic(sub)
    feq = _equilibrium(rho, ux, uy)
    recon = feq + scale * (sub - feq)
    ring = np.empty((9, fine_shape[0] + 2, fine_shape[1] + 2), dtype=parent.dtype)
    ring[:, 1:-1, 1:-1] = reconstruct_from_parent(
        parent, fine_shape, patch_origin, scale, ratio=ratio, periodic=periodic,
    )
    ring[:, 0, :] = recon[:, 0, :]
    ring[:, -1, :] = recon[:, -1, :]
    ring[:, :, 0] = recon[:, :, 0]
    ring[:, :, -1] = recon[:, :, -1]
    return ring


def reconstruct_from_parent(
    parent: np.ndarray,
    fine_shape: tuple[int, int],
    patch_origin: tuple[int, int],
    scale: float,
    *,
    ratio: int = 2,
    periodic: bool = True,
) -> np.ndarray:
    """F-H prolongation of a coarse field onto a fine patch (no ghost ring)."""
    p0, q0 = patch_origin
    npy, npx = parent.shape[1:]
    fi = np.arange(fine_shape[0], dtype=np.int64)
    ci = p0 + np.floor((fi + 0.5) / ratio).astype(np.int64)
    cj = q0 + np.floor((np.arange(fine_shape[1], dtype=np.int64) + 0.5) / ratio).astype(np.int64)
    if periodic:
        ci, cj = ci % npy, cj % npx
    else:
        ci, cj = np.clip(ci, 0, npy - 1), np.clip(cj, 0, npx - 1)
    sub = parent[:, ci[:, None], cj[None, :]]
    rho, ux, uy = _macroscopic(sub)
    feq = _equilibrium(rho, ux, uy)
    return feq + scale * (sub - feq)
