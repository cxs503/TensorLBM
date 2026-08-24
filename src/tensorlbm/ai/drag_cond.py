"""Physics-geometry conditioning for the SUBOFF drag surrogate (B4 v3).

The v1/v2 surrogate conditioned the FNO on four log-parameters plus a
hull-identity one-hot block.  In every leave-one-hull-out (LOHO) fold the
held hull's one-hot channel is constant across the training set, so the
weights feeding it never train and fire through random init at test time
(B4 v2 diagnostic, ``/nfs/wangxi/runs/b4_v2_20260824/diag_nooh.py``).  This
module replaces identity with *continuous mask-derived geometry features*
so hull discrimination is carried by channels that vary inside every fold:

- ``log_aproj_ratio`` — log10 of the x-projected lattice area normalised by
  the bare-hull area (the same ``A_proj`` the C_D label uses);
- ``sail_frac`` / ``fin_frac`` — net sail / fin voxel counts as fractions of
  the bare-hull volume (disjoint decomposition: hull ⊕ sail ⊕ fin);
- ``solid_frac`` — total solid volume over the bare-hull volume
  (== ``1 + sail_frac + fin_frac`` by construction).

All four are deterministic functions of the *design parameters*
``(hull_type, sail_scale, fin_scale)`` evaluated with the same CAD
predicates the scan chain voxelises with (:mod:`tensorlbm.suboff_cad`), so
the encoding is computable for new designs without a stored simulation
mask.  The physical guarantees (bit-level bare-hull scale invariance,
appendage no-op when absent, monotone fractions, exact agreement with
``build_suboff_mask``) are pinned by ``tests/test_drag_cond.py``.

Also provides the two v3 training companions:

- :class:`CondFNODrag` — the FiLM-conditioned FNO drag regressor of the
  B4 v1/v2 protocol (same body plan, parameter-creation order and default
  init as the run-script model, so runs are seed-comparable), extended with
  an optional force-time-series auxiliary head reading the same pooled
  features as the main head;
- :class:`QuotaSampler` — per-dataset equal-quota batch sampling for
  small-N multi-campaign corpora;
- :func:`force_tail_bins` — the auxiliary head's target: log10 mean force
  in ``n_bins`` uniform bins over the tail of the drag history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import cast

import numpy as np
import torch
from torch import nn

from ..suboff_cad import (
    SuboffHullType,
    suboff_fins_contain_points,
    suboff_hull_mask,
    suboff_sail_contains_points,
)
from .fno import SpectralConv2d

__all__ = [
    "COND_V3_CHANNEL_NAMES",
    "CondFNODrag",
    "GEOMETRY_CHANNEL_NAMES",
    "PRODUCTION_GRID",
    "QuotaSampler",
    "SuboffGeometryFeatures",
    "SuboffGrid",
    "condition_v3",
    "force_tail_bins",
    "geometry_channels",
    "suboff_geometry_features",
]

# Production placement of the suboff_n128 case (tensorlbm.cases.suboff):
# (nz, ny, nx) = (64, 64, 128), hull centred at cx = 0.35 nx, length 0.6 nx.
_PRODUCTION_RESOLUTION = 128


@dataclass(frozen=True)
class SuboffGrid:
    """Voxel grid + hull placement shared by the mask builders."""

    nx: int
    ny: int
    nz: int
    cx: float
    cy: float
    cz: float
    length: float

    @classmethod
    def from_resolution(cls, resolution: int) -> SuboffGrid:
        """Mirror ``SuboffChannelCase.make_resolution`` / ``build_solid``."""
        nz = ny = resolution // 2
        nx = resolution
        return cls(nx=nx, ny=ny, nz=nz, cx=nx * 0.35, cy=ny / 2.0, cz=nz / 2.0, length=0.6 * nx)


PRODUCTION_GRID = SuboffGrid.from_resolution(_PRODUCTION_RESOLUTION)


@dataclass(frozen=True)
class SuboffGeometryFeatures:
    """Mask-derived geometry features of one design point.

    ``v_sail`` / ``v_fin`` are *net* contributions: voxels added by the sail
    (resp. fins) on top of everything before it, so the decomposition is
    disjoint and ``v_solid == v_bare + v_sail + v_fin`` exactly.
    """

    hull_type: str
    sail_scale: float
    fin_scale: float
    v_bare: int
    v_sail: int
    v_fin: int
    v_solid: int
    aproj: int
    aproj_bare: int

    @property
    def log_aproj_ratio(self) -> float:
        """log10(A_proj / A_proj_bare) — x-projected area vs the bare hull."""
        return math.log10(self.aproj / self.aproj_bare)

    @property
    def sail_frac(self) -> float:
        """Net sail volume as a fraction of the bare-hull volume."""
        return self.v_sail / self.v_bare

    @property
    def fin_frac(self) -> float:
        """Net fin volume as a fraction of the bare-hull volume."""
        return self.v_fin / self.v_bare

    @property
    def solid_frac(self) -> float:
        """Total solid volume as a fraction of the bare-hull volume."""
        return self.v_solid / self.v_bare


GEOMETRY_CHANNEL_NAMES = ("log_aproj_ratio", "sail_frac", "fin_frac", "solid_frac")

#: Full v3 condition vector: four log-parameters + the geometry block.
COND_V3_CHANNEL_NAMES = (
    "log10_re",
    "log10_u_in",
    "log10_sail_scale",
    "log10_fin_scale",
) + GEOMETRY_CHANNEL_NAMES


def _component_counts(
    hull_key: str, sail_key: float, fin_key: float, grid: SuboffGrid, device: str
) -> tuple[int, int, int, int, int, int]:
    """Bare/sail/fin/solid voxel counts + projected areas (cache worker)."""
    hull_type = SuboffHullType(hull_key)
    dev = torch.device(device)
    zz, yy, xx = torch.meshgrid(
        torch.arange(grid.nz, device=dev, dtype=torch.float32),
        torch.arange(grid.ny, device=dev, dtype=torch.float32),
        torch.arange(grid.nx, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    center = (grid.cx, grid.cy, grid.cz)
    hull = suboff_hull_mask(
        grid.nx, grid.ny, grid.nz, grid.cx, grid.cy, grid.cz, grid.length, 0.0, dev, None
    )
    v_bare = int(hull.sum().item())
    aproj_bare = int((hull.max(dim=2).values > 0).sum().item())

    solid = hull
    v_sail = v_fin = 0
    if hull_type in (SuboffHullType.WITH_SAIL, SuboffHullType.FULL):
        sail = suboff_sail_contains_points(
            xx, yy, zz, center=center, length=grid.length, scale=sail_key
        )
        add = sail & ~solid
        v_sail = int(add.sum().item())
        solid = solid | add
    if hull_type == SuboffHullType.FULL:
        fins = suboff_fins_contain_points(
            xx, yy, zz, center=center, length=grid.length, scale=fin_key
        )
        add = fins & ~solid
        v_fin = int(add.sum().item())
        solid = solid | add
    v_solid = int(solid.sum().item())
    aproj = int((solid.max(dim=2).values > 0).sum().item())
    return v_bare, v_sail, v_fin, v_solid, aproj, aproj_bare


def suboff_geometry_features(
    hull_type: str,
    sail_scale: float = 1.0,
    fin_scale: float = 1.0,
    *,
    grid: SuboffGrid | None = None,
    device: str = "cpu",
) -> SuboffGeometryFeatures:
    """Geometry features of one design point from the CAD predicates.

    Deterministic function of ``(hull_type, sail_scale, fin_scale)`` and the
    grid; identical inputs give bit-identical output (memoised internally —
    multi-point corpora share few unique design keys).
    """
    if grid is None:
        grid = PRODUCTION_GRID
    # Float scales become exact cache keys (mirrors plan.json round-trips).
    s = float(sail_scale)
    f = float(fin_scale)
    if not (s > 0.0 and math.isfinite(s)) or not (f > 0.0 and math.isfinite(f)):
        raise ValueError(f"scales must be finite and positive, got {sail_scale}, {fin_scale}")
    v_bare, v_sail, v_fin, v_solid, aproj, aproj_bare = _feature_counts(
        SuboffHullType(hull_type).value, round(s, 9), round(f, 9), grid, device
    )
    return SuboffGeometryFeatures(
        hull_type=SuboffHullType(hull_type).value,
        sail_scale=s,
        fin_scale=f,
        v_bare=v_bare,
        v_sail=v_sail,
        v_fin=v_fin,
        v_solid=v_solid,
        aproj=aproj,
        aproj_bare=aproj_bare,
    )


@lru_cache(maxsize=4096)
def _feature_counts(
    hull_key: str, sail_key: float, fin_key: float, grid: SuboffGrid, device: str
) -> tuple[int, int, int, int, int, int]:
    return _component_counts(hull_key, sail_key, fin_key, grid, device)


def geometry_channels(features: SuboffGeometryFeatures) -> np.ndarray:
    """The 4-dim geometry block of the condition vector."""
    return np.array(
        [
            features.log_aproj_ratio,
            features.sail_frac,
            features.fin_frac,
            features.solid_frac,
        ],
        dtype=np.float64,
    )


def condition_v3(
    re: np.ndarray,
    u_in: np.ndarray,
    sail_scale: np.ndarray,
    fin_scale: np.ndarray,
    geometry: np.ndarray,
) -> np.ndarray:
    """Assemble the (N, 8) v3 condition vector.

    ``geometry`` is the (N, 4) block from :func:`geometry_channels` (one row
    per point, e.g. precomputed over the corpus).  No identity/one-hot
    column is used anywhere.
    """
    re = np.asarray(re, dtype=np.float64)
    u_in = np.asarray(u_in, dtype=np.float64)
    sail_scale = np.asarray(sail_scale, dtype=np.float64)
    fin_scale = np.asarray(fin_scale, dtype=np.float64)
    geometry = np.asarray(geometry, dtype=np.float64)
    n = len(re)
    if geometry.shape != (n, len(GEOMETRY_CHANNEL_NAMES)):
        raise ValueError(f"geometry block must be ({n}, 4), got {geometry.shape}")
    logs = np.stack(
        [np.log10(re), np.log10(u_in), np.log10(sail_scale), np.log10(fin_scale)], axis=1
    )
    return np.concatenate([logs, geometry], axis=1)


class CondFNODrag(nn.Module):
    """FiLM-conditioned FNO2d drag regressor (B4 v1/v2 protocol body).

    Body plan identical to the v1/v2 run-script model: lift 1x1 conv,
    ``n_layers`` x [SpectralConv2d + pointwise 1x1 + per-layer FiLM +
    GELU], global mean pool, MLP head on ``concat(pooled, cond)`` — see
    :class:`tensorlbm.ai.drag_surrogate.FNODragRegressor` for the
    unconditional ancestor.  Parameter-creation order is unchanged, so for
    a fixed torch seed the shared modules initialise identically whether or
    not ``aux_dim > 0`` (the auxiliary head is created last).

    ``aux_dim > 0`` adds a force-time-series head reading the same pooled
    features as the main head (used with a small loss weight as an
    auxiliary task; the main output is unchanged).
    """

    def __init__(
        self,
        in_ch: int = 5,
        width: int = 32,
        n_layers: int = 4,
        modes: tuple[int, int] = (16, 32),
        cond_dim: int = 8,
        mlp_hidden: int = 128,
        film_hidden: int = 64,
        aux_dim: int = 0,
    ) -> None:
        super().__init__()
        my, mx = modes
        self.aux_dim = int(aux_dim)
        self.lift = nn.Conv2d(in_ch, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, my, mx) for _ in range(n_layers)]
        )
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, film_hidden), nn.GELU(), nn.Linear(film_hidden, film_hidden)
        )
        self.film = nn.ModuleList([nn.Linear(film_hidden, 2 * width) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Linear(width + cond_dim, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, 1)
        )
        self.aux_head: nn.Sequential | None = None
        if self.aux_dim > 0:
            self.aux_head = nn.Sequential(
                nn.Linear(width + cond_dim, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, self.aux_dim),
            )

    def forward(
        self, x: torch.Tensor, p: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        e = self.cond_embed(p)
        x = self.lift(x)
        for spec, pw, fl in zip(self.spectral, self.pointwise, self.film):
            h = spec(x) + pw(x)
            g, b = fl(e).chunk(2, dim=-1)
            x = nn.functional.gelu(g[..., None, None] * h + b[..., None, None])
        feat = torch.cat([x.mean(dim=(2, 3)), p], dim=1)
        y = cast(torch.Tensor, self.head(feat).squeeze(-1))
        if not return_aux:
            return y
        aux_head = self.aux_head
        if aux_head is None:
            raise RuntimeError("return_aux=True requires aux_dim > 0 at construction")
        return y, cast(torch.Tensor, aux_head(feat))


class QuotaSampler:
    """Per-dataset equal-quota index sampler for multi-campaign training.

    Motivation: the joint corpus mixes campaigns of very different size
    (6 to 78 points); plain uniform batching under-weights the small, hard
    campaigns.  Each epoch draw gives every dataset present in ``fit_idx``
    **exactly** ``quota`` index slots, where

        quota = max_k n_k,   n_k = #fit points of dataset k,

    implemented by tiling each dataset's (sorted) fit indices and
    truncating to ``quota`` — index repetition, no smoothing, trivially
    auditable.  ``epoch_indices`` returns the concatenated slots in a fresh
    random order; the epoch length is ``K * quota`` (K = #datasets present).
    """

    def __init__(self, labels: np.ndarray, fit_idx: np.ndarray | list[int]) -> None:
        labels = np.asarray(labels)
        fit = np.asarray(sorted(fit_idx), dtype=np.int64)
        if fit.size == 0:
            raise ValueError("fit_idx must be non-empty")
        if labels.ndim != 1 or len(labels) <= fit.max():
            raise ValueError(
                f"labels shape {labels.shape} inconsistent with fit index max {fit.max()}"
            )
        self._fit = fit
        self._labels = labels
        self._per_dataset: dict[int, np.ndarray] = {}
        for k in np.unique(labels[fit]):
            self._per_dataset[int(k)] = fit[labels[fit] == k]
        self._quota = max(int(len(v)) for v in self._per_dataset.values())

    @property
    def quota(self) -> int:
        """Slots per dataset per epoch (= size of the largest dataset)."""
        return self._quota

    @property
    def per_dataset_fit_counts(self) -> dict[int, int]:
        """Fit-point count per dataset label."""
        return {k: int(len(v)) for k, v in self._per_dataset.items()}

    def epoch_indices(self, rng: np.random.Generator) -> np.ndarray:
        """One epoch of indices: every dataset contributes exactly ``quota``."""
        parts = [
            np.tile(idx, self._quota // len(idx) + 1)[: self._quota]
            for idx in self._per_dataset.values()
        ]
        return rng.permutation(np.concatenate(parts))


def force_tail_bins(force: np.ndarray, *, tail_frac: float = 0.25, n_bins: int = 8) -> np.ndarray:
    """Auxiliary-head target: log10 mean force in ``n_bins`` uniform tail bins.

    The tail window matches the C_D label convention of the B4 cache
    builder exactly (``force[int(len(force) * (1 - tail_frac)):]``), then is
    split into ``n_bins`` near-uniform contiguous bins
    (``numpy.array_split``); each bin contributes its arithmetic mean.
    Raises on non-positive / non-finite bin means (log10 input).
    """
    force = np.asarray(force, dtype=np.float64)
    if force.ndim != 1 or force.size == 0:
        raise ValueError(f"force must be a non-empty 1-D array, got {force.shape}")
    if not (0.0 < tail_frac < 1.0):
        raise ValueError(f"tail_frac must be in (0, 1), got {tail_frac}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    tail = force[int(force.size * (1.0 - tail_frac)) :]
    if tail.size < n_bins:
        raise ValueError(
            f"tail has {tail.size} samples < n_bins={n_bins}; decrease n_bins or tail_frac"
        )
    means = np.array([np.mean(b) for b in np.array_split(tail, n_bins)])
    if not np.isfinite(means).all() or not (means > 0.0).all():
        raise ValueError(f"tail bin means must be finite and positive, got {means}")
    return np.log10(means)
