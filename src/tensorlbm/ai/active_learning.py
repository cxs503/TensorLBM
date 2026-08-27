"""B4-P3b — active-learning loop for the SUBOFF drag surrogate (2026-08-25).

Closes the loop the ship-design roadmap promised:

    guardrail-flagged queries
        -> acquisition of new design points (3 strategies)
        -> labels (oracle cache lookup here; scan_runner in production)
        -> corpus augmentation + incremental retrain (verbatim serving
           protocol of ``b4_serve_20260824/train_ensemble.py``)
        -> verdicts improve (flagged queries un-flag *because the corpus
           now covers them*, and the C_D trend they carry flips to agree
           with the labels).

Motivating failure (measured 2026-08-25, unmerged PR #241 ``exp/b4-echo``
@ 98c617f7): the serving 5-seed ensemble (v3 hand channels) answers ``ok``
in channel space on hull-form variants while its ``l_over_d`` C_D trend
runs OPPOSITE to the B4-fam family cache.  This module is the loop that
fixes that class of failure: add labeled family points -> retrain ->
trend flips + guard un-flags appropriately.

Provenance / attribution
------------------------
- The hull-form-aware geometry front-end (component decomposition under a
  deformed :class:`~tensorlbm.suboff_cad.SuboffConfig`, v3 channel block)
  duplicates ~50 lines of PR #241 ``geometry_pipeline.suboff_component_counts``
  (``exp/b4-echo`` @ 98c617f7, unmerged) so this module builds on ``main``
  without importing an unmerged branch; ``hullform_geo_block`` reproduces
  the fit-time construction of :func:`.drag_cond.geometry_channels`.
- The honest hull-form verdict duplicates the *contract* of #241's
  ``_downgrade_hullform`` (never answer confident ``ok`` for a design the
  corpus never contained) generalised from "is mother" to "is inside the
  corpus hull-form-axis envelope" — which is exactly what changes when the
  loop retrains on acquired family points.
- The split / fit-stats / training loop are copied VERBATIM from
  ``/nfs/wangxi/runs/b4_serve_20260824/train_ensemble.py`` (which pins its
  own worktree and cannot be imported); only the loop additionally returns
  the fitted model + normalisation.

Everything /nfs-related (checkpoints, caches, scan datasets) enters as an
explicit path argument from the caller; the module itself is import-safe
and testable on CPU-only machines with synthetic data.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from ..suboff_cad import (
    SuboffConfig,
    SuboffHullType,
    suboff_fins_contain_points,
    suboff_hull_mask,
    suboff_sail_contains_points,
)
from .drag_cond import (
    PRODUCTION_GRID,
    CondFNODrag,
    QuotaSampler,
    SuboffGrid,
    condition_v3,
)
from .inference_service import (
    FLAG_REJECT,
    FLAG_REVIEW,
    CondDragCheckpoint,
    EnvelopeMahalanobisGuardrail,
    GuardVerdict,
    ModelEnsembleBackend,
    ensemble_stats,
    load_checkpoint,
    save_checkpoint,
)

__all__ = [
    "AcquisitionLabel",
    "AcquisitionPoint",
    "ARM_CFG",
    "ARCH_BASE",
    "DEFAULT_AXIS_RANGES",
    "FlaggedQuery",
    "HP",
    "HULLFORM_AXES",
    "LoopReport",
    "MotherEval",
    "STRATEGY_NAMES",
    "ServiceSpec",
    "TrendSpec",
    "TrendStat",
    "augment_corpus",
    "axes_envelope",
    "corpus_cond_v3",
    "corpus_design_keys",
    "corpus_param_keys",
    "eval_loop",
    "fit_stats",
    "honest_verdict",
    "hullform_component_counts",
    "hullform_condition_rows",
    "hullform_geo_block",
    "labels_from_cache",
    "load_corpus_index",
    "point_param_key",
    "predict_design",
    "propose_acquisition",
    "retrain_ensemble",
    "spearman_rho",
    "split_random",
    "trend_stat",
    "write_loop_report",
]

#: The hull-form (lines-plan) axes of ``SuboffConfig`` (PR #241 naming).
HULLFORM_AXES = ("l_over_d_mult", "nose_len_mult", "stern_len_mult", "sail_x_mult")

#: Default design box of the hull-form axes (the 2026-08-24 hullform campaign
#: plan ranges: ``scan_suboff_hullform_20260824/plan.json`` variables).
DEFAULT_AXIS_RANGES: dict[str, tuple[float, float]] = {
    "l_over_d_mult": (0.75, 1.30),
    "nose_len_mult": (1.00, 1.30),
    "stern_len_mult": (1.00, 1.30),
    "sail_x_mult": (1.00, 1.30),
}

#: Acquisition strategy names (all three ablated in the 2026-08-25 demo).
STRATEGY_NAMES = ("envelope_shell", "max_disagreement", "coverage")

_AXIS_EPS = 1e-9

#: Hull identity order of the B4 caches (``cache['hull']`` ints).
HULL_ORDER = ("bare_hull", "with_sail", "full")


# ---------------------------------------------------------------------------
# Geometry front-end for hull-form variants (duplicated from PR #241 with
# attribution; the fit-time drag_cond builder cannot express these axes)
# ---------------------------------------------------------------------------


def hullform_component_counts(
    hull_type: str,
    sail_scale: float,
    fin_scale: float,
    grid: SuboffGrid,
    config: SuboffConfig | None = None,
) -> tuple[int, int, int, int, int, int]:
    """Disjoint hull/sail/fin voxel decomposition under a variant config.

    Evaluates the CAD point predicates over ``grid`` exactly the way the
    fit-time builder (:func:`.drag_cond.suboff_geometry_features` via
    ``_component_counts``) does — same operations, same order, CPU — so for
    mother designs (``config`` all-1.0 or ``None``) the counts are
    bit-identical to the training-cache values.  ``config`` additionally
    enables the hull-form axes.

    Duplicate of PR #241 ``geometry_pipeline.suboff_component_counts``
    (``exp/b4-echo`` @ 98c617f7, unmerged — do not import).

    Returns ``(v_bare, v_sail, v_fin, v_solid, aproj, aproj_bare)``.
    """
    ht = SuboffHullType(hull_type)
    cfg = SuboffConfig() if config is None else config
    dev = torch.device("cpu")
    zz, yy, xx = torch.meshgrid(
        torch.arange(grid.nz, device=dev, dtype=torch.float32),
        torch.arange(grid.ny, device=dev, dtype=torch.float32),
        torch.arange(grid.nx, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    center = (grid.cx, grid.cy, grid.cz)
    hull = suboff_hull_mask(
        grid.nx, grid.ny, grid.nz, grid.cx, grid.cy, grid.cz, grid.length, 0.0, dev, cfg
    )
    v_bare = int(hull.sum().item())
    aproj_bare = int((hull.max(dim=2).values > 0).sum().item())
    if v_bare <= 0 or aproj_bare <= 0:
        raise ValueError(f"degenerate geometry at grid {grid}: v_bare={v_bare}")
    solid = hull
    v_sail = 0
    v_fin = 0
    if ht in (SuboffHullType.WITH_SAIL, SuboffHullType.FULL):
        sail = suboff_sail_contains_points(
            xx, yy, zz, center=center, length=grid.length, scale=sail_scale, config=cfg
        )
        add = sail & ~solid
        v_sail = int(add.sum().item())
        solid = solid | add
    if ht == SuboffHullType.FULL:
        fins = suboff_fins_contain_points(
            xx, yy, zz, center=center, length=grid.length, scale=fin_scale, config=cfg
        )
        add = fins & ~solid
        v_fin = int(add.sum().item())
        solid = solid | add
    v_solid = int(solid.sum().item())
    aproj = int((solid.max(dim=2).values > 0).sum().item())
    return v_bare, v_sail, v_fin, v_solid, aproj, aproj_bare


def hullform_geo_block(
    hull_type: str,
    sail_scale: float,
    fin_scale: float,
    grid: SuboffGrid = PRODUCTION_GRID,
    config: SuboffConfig | None = None,
) -> np.ndarray:
    """The (4,) v3 geometry-channel block for a (possibly variant) design.

    Identical formula to :func:`.drag_cond.geometry_channels` —
    ``[log10(aproj/aproj_bare), v_sail/v_bare, v_fin/v_bare, v_solid/v_bare]``
    — but computed under a deformed ``config`` (hull-form axes), which the
    fit-time builder cannot express.  Mother configs reproduce the cached
    ``geo`` rows bitwise.
    """
    v_bare, v_sail, v_fin, v_solid, aproj, aproj_bare = hullform_component_counts(
        hull_type, sail_scale, fin_scale, grid, config
    )
    return np.array(
        [
            math.log10(aproj / aproj_bare),
            v_sail / v_bare,
            v_fin / v_bare,
            v_solid / v_bare,
        ],
        dtype=np.float64,
    )


def _params_config(params: dict[str, Any]) -> SuboffConfig:
    """CAD config from a params dict (mother defaults for absent axes)."""
    known = set(HULLFORM_AXES) | {"hull_type", "sail_scale", "fin_scale", "u_in"}
    unknown = sorted(set(params) - known)
    if unknown:
        raise ValueError(f"unknown design params: {unknown}; supported {sorted(known)}")
    return SuboffConfig(
        sail_scale=float(params.get("sail_scale", 1.0)),
        fin_scale=float(params.get("fin_scale", 1.0)),
        **{axis: float(params.get(axis, 1.0)) for axis in HULLFORM_AXES},
    )


@lru_cache(maxsize=64)
def _cached_geo_block(
    hull_type: str,
    sail_key: float,
    fin_key: float,
    axis_key: tuple[float, ...],
    grid: SuboffGrid,
) -> np.ndarray:
    cfg = SuboffConfig(
        sail_scale=sail_key,
        fin_scale=fin_key,
        **{axis: float(val) for axis, val in zip(HULLFORM_AXES, axis_key)},
    )
    return hullform_geo_block(hull_type, sail_key, fin_key, grid, cfg)


def _geo_of(params: dict[str, Any], grid: SuboffGrid) -> np.ndarray:
    return _cached_geo_block(
        str(params.get("hull_type", "with_sail")),
        round(float(params.get("sail_scale", 1.0)), 9),
        round(float(params.get("fin_scale", 1.0)), 9),
        tuple(round(float(params.get(a, 1.0)), 9) for a in HULLFORM_AXES),
        grid,
    )


def hullform_condition_rows(
    params: dict[str, Any],
    re_list: Sequence[float] | np.ndarray,
    grid: SuboffGrid = PRODUCTION_GRID,
    u_in: float | None = None,
) -> np.ndarray:
    """(N, 8) condition_v3 rows of a (possibly variant) design over ``re_list``.

    Fit-time construction: the geo block from :func:`hullform_geo_block`
    appended to the four log-parameters by :func:`.drag_cond.condition_v3`.
    ``u_in`` defaults to the point's own ``params['u_in']`` or 0.1.
    """
    re = np.asarray(re_list, dtype=np.float64).ravel()
    if re.size == 0 or not np.isfinite(re).all() or not (re > 0.0).all():
        raise ValueError("re_list entries must be finite and positive")
    u = 0.1 if u_in is None else float(u_in)
    geo = _geo_of(params, grid)
    return condition_v3(
        re,
        np.full(re.shape, u),
        np.full(re.shape, float(params.get("sail_scale", 1.0))),
        np.full(re.shape, float(params.get("fin_scale", 1.0))),
        np.broadcast_to(geo, (re.size, 4)),
    )


# ---------------------------------------------------------------------------
# Query / acquisition data model
# ---------------------------------------------------------------------------


def point_param_key(params: dict[str, Any], re: float) -> str:
    """Canonical key of a design point (hull + scales + hull-form axes + Re).

    ``u_in`` is not part of the key (the whole B4 corpus is ``u_in=0.1``)."""
    bits = [
        str(params.get("hull_type", "with_sail")),
        f"{float(params.get('sail_scale', 1.0)):.9g}",
        f"{float(params.get('fin_scale', 1.0)):.9g}",
    ]
    bits += [f"{float(params.get(a, 1.0)):.9g}" for a in HULLFORM_AXES]
    bits.append(f"{float(re):.9g}")
    return "|".join(bits)


@dataclass(frozen=True)
class FlaggedQuery:
    """One guardrail-flagged served query feeding the acquisition step."""

    params: dict[str, Any]
    re: float
    verdict: str
    score: float
    member_std: float

    @property
    def key(self) -> str:
        return point_param_key(self.params, self.re)


@dataclass(frozen=True)
class AcquisitionPoint:
    """One proposed design point to label (the loop's ask)."""

    params: dict[str, Any]
    re: float
    strategy: str

    @property
    def key(self) -> str:
        return point_param_key(self.params, self.re)


@dataclass(frozen=True)
class AcquisitionLabel:
    """Oracle label of one acquisition point (honest about mismatches)."""

    point_key: str
    matched: bool
    cd: float
    source_row: int | None = None
    source_fam: str | None = None
    re_delta: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)


def _flagged_axis_ranges(queries: Sequence[FlaggedQuery]) -> dict[str, tuple[float, float]]:
    """Axis design box of the flagged region (fallback: campaign defaults)."""
    out: dict[str, tuple[float, float]] = {}
    for axis in HULLFORM_AXES:
        vals = [float(q.params.get(axis, 1.0)) for q in queries if axis in q.params]
        out[axis] = (min(vals), max(vals)) if vals else DEFAULT_AXIS_RANGES[axis]
    return out


def _flagged_re_window(queries: Sequence[FlaggedQuery]) -> tuple[float, float]:
    res = np.asarray([q.re for q in queries], dtype=np.float64)
    return float(res.min()), float(res.max())


def _spread_levels(res_sorted: np.ndarray, n_levels: int) -> list[float]:
    """Deterministic spread of Re levels over the flagged Re values."""
    n = min(n_levels, len(res_sorted))
    pick = np.unique(np.linspace(0, len(res_sorted) - 1, n).round().astype(int))
    return [float(res_sorted[i]) for i in pick]


def _eligible_re_levels(
    shape: dict[str, Any], queries: Sequence[FlaggedQuery], n_re_levels: int
) -> list[float]:
    """Re levels a candidate shape may pair with.

    The labeled families share no Re values with each other (every cache
    row carries a distinct Re), so a candidate is only *labelable* when
    its Re comes from flagged queries of the same shape family: queries
    that agree with ``shape`` on every axis it moves.  Mother shapes and
    mixed shapes fall back to the full flagged Re set.
    """
    moved = [a for a in HULLFORM_AXES if abs(float(shape.get(a, 1.0)) - 1.0) > _AXIS_EPS]
    res = sorted(
        {
            float(q.re)
            for q in queries
            if all(abs(float(q.params.get(a, 1.0)) - float(shape[a])) <= _AXIS_EPS for a in moved)
        }
    )
    if not res:
        res = sorted({float(q.re) for q in queries})
    return _spread_levels(np.asarray(res, dtype=np.float64), n_re_levels)


def _exclusion_floor(existing_cond: np.ndarray, guard: EnvelopeMahalanobisGuardrail) -> float:
    """Score floor: candidates below it are indistinguishable from the corpus.

    The floor is the 10th percentile of the existing cloud's own scores —
    a candidate scoring lower than 10 % of the corpus itself sits inside
    the dense core and adds nothing (and typically duplicates a row).
    """
    own = guard.row_scores(np.asarray(existing_cond, dtype=np.float64))
    return float(np.quantile(own, 0.10))


def _product(levels: list[list[float]]) -> list[tuple[float, ...]]:
    """Cartesian product of axis anchor levels (deterministic order)."""
    out: list[tuple[float, ...]] = [()]
    for lev in levels:
        out = [prefix + (v,) for prefix in out for v in lev]
    return out


def propose_acquisition(
    queries: Sequence[FlaggedQuery],
    *,
    strategy: str,
    budget: int,
    existing_cond: np.ndarray,
    axis_ranges: dict[str, tuple[float, float]] | None = None,
    member_std_fn: Callable[[list[AcquisitionPoint]], np.ndarray] | None = None,
    grid: SuboffGrid = PRODUCTION_GRID,
    seed: int = 0,
    n_candidates: int | None = None,
    n_re_levels: int = 8,
) -> list[AcquisitionPoint]:
    """Propose ``budget`` new design points given flagged queries.

    Strategies (ablated in the 2026-08-25 demo):

    - ``envelope_shell`` — Latin hypercube on the hull-form axes + log-Re
      window of the flagged region, kept on the Mahalanobis *shell* of the
      existing condition cloud (the score band the flagged queries
      occupy — near the cloud, where the guard starts to care, not deep
      inside it and not far away);
    - ``max_disagreement`` — grid over the axis anchor levels
      ({low, mother 1.0, high} per axis) x flagged Re levels, maximising
      the ensemble member std via batched ``member_std_fn``;
    - ``coverage`` — family corners (one axis at a design-box extreme, the
      others at mother 1.0) x the flagged Re levels of that corner's OWN
      family, round-robin so a small budget covers every family (the
      labeled families share no Re values — cross-family pairings are
      unlabelable, so ``max_disagreement`` corners use the same rule).

    All strategies are seeded and bitwise reproducible; every candidate is
    excluded when its channel-space Mahalanobis score falls below the
    existing cloud's 10th-percentile score (duplicates add nothing), and
    candidate keys are unique.  Returns exactly ``budget`` points when the
    candidate pool suffices (fewer otherwise — never silently padded).
    """
    if strategy not in STRATEGY_NAMES:
        raise ValueError(f"strategy must be one of {STRATEGY_NAMES}, got {strategy!r}")
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    if len(queries) < 2:
        raise ValueError("need at least 2 flagged queries to locate the flagged region")
    existing_cond = np.asarray(existing_cond, dtype=np.float64)
    if existing_cond.ndim != 2 or existing_cond.shape[0] < 2:
        raise ValueError(f"existing_cond must be (N>=2, D), got {existing_cond.shape}")
    guard = EnvelopeMahalanobisGuardrail(existing_cond)
    floor = _exclusion_floor(existing_cond, guard)
    ranges = dict(axis_ranges) if axis_ranges else _flagged_axis_ranges(queries)
    re_lo, re_hi = _flagged_re_window(queries)
    u_in = float(queries[0].params.get("u_in", 0.1))
    rng = np.random.default_rng(seed)

    def accept(cands: list[AcquisitionPoint]) -> list[AcquisitionPoint]:
        seen: set[str] = set()
        out: list[AcquisitionPoint] = []
        for p in cands:
            if p.key in seen:
                continue
            cond = hullform_condition_rows(p.params, [p.re], grid=grid, u_in=u_in)
            if float(guard.row_scores(cond)[0]) < floor:
                continue
            seen.add(p.key)
            out.append(p)
        return out

    if strategy == "envelope_shell":
        n_cand = int(n_candidates) if n_candidates else max(8 * budget, 64)
        dim = len(HULLFORM_AXES) + 1
        u = (rng.permutation(n_cand)[:, None] + rng.random((n_cand, dim))) / n_cand
        # axis multipliers are sampled LINEARLY, Re in log space
        ax_lo = np.array([ranges[a][0] for a in HULLFORM_AXES])
        ax_hi = np.array([ranges[a][1] for a in HULLFORM_AXES])
        pts = np.column_stack(
            [
                ax_lo + u[:, : dim - 1] * (ax_hi - ax_lo),
                10.0 ** (math.log10(re_lo) + u[:, -1] * (math.log10(re_hi) - math.log10(re_lo))),
            ]
        )
        band_lo = min(q.score for q in queries)
        band_hi = max(q.score for q in queries)
        cands: list[AcquisitionPoint] = []
        for row in pts:
            params = {
                "hull_type": "with_sail",
                "sail_scale": 1.0,
                "fin_scale": 1.0,
                **{a: float(v) for a, v in zip(HULLFORM_AXES, row[:4])},
            }
            re = float(row[4])
            cond = hullform_condition_rows(params, [re], grid=grid, u_in=u_in)
            score = float(guard.row_scores(cond)[0])
            if band_lo <= score <= band_hi:
                cands.append(AcquisitionPoint(params=params, re=re, strategy=strategy))
        return accept(cands)[:budget]

    if strategy == "max_disagreement":
        if member_std_fn is None:
            raise ValueError("max_disagreement requires member_std_fn")
        levels: list[list[float]] = []
        for axis in HULLFORM_AXES:
            lo_a, hi_a = ranges[axis]
            lev = sorted({round(lo_a, 9), 1.0, round(hi_a, 9)})
            levels.append([float(v) for v in lev])
        combos = _product(levels)
        shape_by_combo = {
            combo: {
                "hull_type": "with_sail",
                "sail_scale": 1.0,
                "fin_scale": 1.0,
                **{a: float(v) for a, v in zip(HULLFORM_AXES, combo)},
            }
            for combo in combos
        }
        cands = [
            AcquisitionPoint(
                params=shape_by_combo[combo],
                re=re,
                strategy=strategy,
            )
            for combo in combos
            for re in _eligible_re_levels(shape_by_combo[combo], queries, n_re_levels)
        ]
        cands = accept(cands)
        if not cands:
            return []
        stds = np.asarray(member_std_fn(cands), dtype=np.float64)
        if stds.shape != (len(cands),):
            raise ValueError(f"member_std_fn must return ({len(cands)},), got {stds.shape}")
        order = sorted(range(len(cands)), key=lambda i: (-float(stds[i]), cands[i].key))
        return [cands[i] for i in order[:budget]]

    # coverage: family corners x flagged Re levels, round-robin over corners
    corners: list[dict[str, float]] = []
    for axis in HULLFORM_AXES:
        for extreme in ranges[axis]:
            if abs(float(extreme) - 1.0) <= _AXIS_EPS:
                continue
            shape = {a: 1.0 for a in HULLFORM_AXES}
            shape[axis] = float(extreme)
            if shape not in corners:
                corners.append(shape)
    corners.sort(key=lambda s: tuple(s[a] for a in HULLFORM_AXES))
    per_corner = [_eligible_re_levels(shape, queries, n_re_levels) for shape in corners]
    cands = []
    for j in range(max(len(lev) for lev in per_corner)):
        for shape, lev in zip(corners, per_corner):
            if j < len(lev):
                cands.append(
                    AcquisitionPoint(
                        params={
                            "hull_type": "with_sail",
                            "sail_scale": 1.0,
                            "fin_scale": 1.0,
                            **shape,
                        },
                        re=lev[j],
                        strategy=strategy,
                    )
                )
    return accept(cands)[:budget]


# ---------------------------------------------------------------------------
# Labeling oracle: B4-fam family cache lookup
# ---------------------------------------------------------------------------


def labels_from_cache(
    points: Sequence[AcquisitionPoint],
    cache_path: str | Path,
    *,
    re_tol: float = 0.0,
) -> list[AcquisitionLabel]:
    """Label acquisition points from the B4-fam family cache (the oracle).

    Matches a point to a cache row on the FULL design key — hull type,
    sail/fin scale, u_in, all four hull-form axes (exact, rounded to 9
    decimals) — plus Reynolds within relative tolerance ``re_tol``
    (0 = exact; the demo reports the per-label ``re_delta`` whenever a
    tolerance is used).  Among rows inside tolerance the nearest in Re
    wins.  Only family rows (``dsi >= 6``) carry hull-form axes in the
    sibling ``*_meta.json``; mother-corpus points cannot be matched and
    are reported honestly as ``matched=False`` (they duplicate the
    training corpus anyway).
    """
    cache_path = Path(cache_path)
    meta_path = cache_path.parent / (cache_path.stem + "_meta.json")
    if not cache_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"cache/meta pair not found: {cache_path}, {meta_path}")
    z = np.load(cache_path)
    meta = json.loads(meta_path.read_text())
    fam_rows = np.nonzero(np.asarray(z["dsi"]) >= 6)[0]
    if fam_rows.size != len(meta):
        raise ValueError(
            f"family rows {fam_rows.size} != meta entries {len(meta)}; not a B4-fam cache?"
        )
    index: dict[tuple[Any, ...], list[tuple[int, float]]] = {}
    for i, m in enumerate(meta):
        row = int(fam_rows[i])
        key = (
            str(m["hull"]),
            round(float(m["sail"]), 9),
            round(float(m["fin"]), 9),
            round(float(m["u_in"]), 9),
        ) + tuple(round(float(m[a]), 9) for a in HULLFORM_AXES)
        index.setdefault(key, []).append((row, float(z["re"][row])))

    out: list[AcquisitionLabel] = []
    for p in points:
        key = (
            str(p.params.get("hull_type", "with_sail")),
            round(float(p.params.get("sail_scale", 1.0)), 9),
            round(float(p.params.get("fin_scale", 1.0)), 9),
            round(float(p.params.get("u_in", 0.1)), 9),
        ) + tuple(round(float(p.params.get(a, 1.0)), 9) for a in HULLFORM_AXES)
        rows = index.get(key, [])
        best: tuple[float, int, float] | None = None
        for row, re_row in rows:
            delta = abs(p.re - re_row) / re_row
            if delta <= max(re_tol, 0.0) and (best is None or delta < best[0]):
                best = (delta, row, re_row)
        if best is None:
            out.append(AcquisitionLabel(point_key=p.key, matched=False, cd=float("nan")))
            continue
        delta, row, _re_row = best
        i = int(np.nonzero(fam_rows == row)[0][0])
        m = meta[i]
        out.append(
            AcquisitionLabel(
                point_key=p.key,
                matched=True,
                cd=float(z["cd"][row]),
                source_row=row,
                source_fam=str(m.get("fam", "")),
                re_delta=float(delta),
                payload={
                    "x": np.asarray(z["x"][row], dtype=np.float32),
                    "re": float(z["re"][row]),
                    "uin": float(z["uin"][row]),
                    "sail": float(z["sail"][row]),
                    "fin": float(z["fin"][row]),
                    "hull": int(z["hull"][row]),
                    "step": int(z["step"][row]),
                    "aproj": int(z["aproj"][row]),
                    "cd": float(z["cd"][row]),
                    "aux": np.asarray(z["aux"][row], dtype=np.float64),
                    "mask_bit_eq": bool(z["mask_bit_eq"][row]),
                    "fam": int(z["fam"][row]),
                    "fam_name": str(m.get("fam", "")),
                },
            )
        )
    return out


# ---------------------------------------------------------------------------
# Corpus model + augmentation
# ---------------------------------------------------------------------------


def load_corpus_index(cache_path: str | Path) -> dict[str, np.ndarray]:
    """Load a B4 corpus cache (``cache_v4.npz`` layout) as a column dict."""
    z = np.load(Path(cache_path))
    d = {k: np.asarray(z[k]) for k in z.files}
    for req in ("x", "dsi", "re", "uin", "sail", "fin", "hull", "cd", "geo", "aux"):
        if req not in d:
            raise ValueError(f"corpus cache {cache_path} missing key {req!r}")
    return d


def corpus_cond_v3(index: dict[str, np.ndarray]) -> np.ndarray:
    """(N, 8) condition_v3 matrix of a corpus (``geo`` is the v3 block)."""
    return condition_v3(index["re"], index["uin"], index["sail"], index["fin"], index["geo"])


def corpus_param_keys(index: dict[str, np.ndarray]) -> list[str]:
    """Row keys ``re|uin|sail|fin|hull`` (the serving split group key)."""
    return [
        f"{r:.6g}|{u:.6g}|{s:.6g}|{f:.6g}|{HULL_ORDER[int(h)]}"
        for r, u, s, f, h in zip(
            index["re"], index["uin"], index["sail"], index["fin"], index["hull"]
        )
    ]


def corpus_design_keys(index: dict[str, np.ndarray]) -> list[tuple[str, float, float, float]]:
    """Row-aligned ``(hull, sail, fin, u_in)`` keys for field resolution."""
    return [
        (HULL_ORDER[int(h)], float(s), float(f), float(u))
        for h, s, f, u in zip(index["hull"], index["sail"], index["fin"], index["uin"])
    ]


def augment_corpus(
    index: dict[str, np.ndarray],
    points: Sequence[AcquisitionPoint],
    labels: Sequence[AcquisitionLabel],
    *,
    grid: SuboffGrid = PRODUCTION_GRID,
) -> dict[str, np.ndarray]:
    """Append oracle-labeled acquisition points to a corpus (v4 schema).

    Every label must be matched (unmatched labels raise — padding the
    training corpus with proxy values would be dishonest).  New rows carry
    the cache_v4 schema exactly: simulation payload (``x``/``aux``/
    ``aproj``/``step``/``mask_bit_eq``/``cd``) from the oracle row, the v3
    geometry block recomputed by the fit-time construction
    (:func:`hullform_geo_block` — bit-identical to the cached value for
    mother designs and to the #241-validated variant construction), and a
    FRESH dataset id per family so the quota sampler weights the new
    campaigns (family ``dsi`` 6..9 of the B4-fam cache are remapped above
    the base corpus's max ``dsi``).
    """
    if len(points) != len(labels):
        raise ValueError(f"{len(points)} points but {len(labels)} labels")
    unmatched = [lab.point_key for lab in labels if not lab.matched]
    if unmatched:
        raise ValueError(f"cannot augment with unmatched labels: {unmatched}")
    n0 = len(index["cd"])
    base_dsi = int(np.max(index["dsi"]))
    fam_dsis = sorted({int(lab.payload["fam"]) for lab in labels})
    offset = base_dsi + 1 - fam_dsis[0]
    new_rows: dict[str, list[Any]] = {k: [] for k in index}
    for p, lab in zip(points, labels):
        pay = lab.payload
        hull = str(p.params.get("hull_type", "with_sail"))
        sail = float(p.params.get("sail_scale", 1.0))
        fin = float(p.params.get("fin_scale", 1.0))
        cfg = _params_config(p.params)
        v_bare, v_sail, v_fin, v_solid, aproj, aproj_bare = hullform_component_counts(
            hull, sail, fin, grid, cfg
        )
        vals: dict[str, Any] = {
            "x": pay["x"],
            "dsi": int(pay["fam"]) + offset,
            "re": float(pay["re"]),
            "uin": float(pay["uin"]),
            "sail": float(pay["sail"]),
            "fin": float(pay["fin"]),
            "hull": int(pay["hull"]),
            "step": int(pay["step"]),
            "aproj": int(pay["aproj"]),
            "cd": float(pay["cd"]),
            "aux": np.asarray(pay["aux"], dtype=np.float64),
            "mask_bit_eq": bool(pay["mask_bit_eq"]),
            "geo": hullform_geo_block(hull, sail, fin, grid, cfg),
            "v_sail": int(v_sail),
            "v_fin": int(v_fin),
            "v_solid": int(v_solid),
            "aproj_cad": int(aproj),
            "aproj_bare": int(aproj_bare),
        }
        for k in index:
            if k not in vals:
                raise ValueError(
                    f"corpus key {k!r} has no value source for acquired rows; "
                    "extend the label payload or drop the key"
                )
            new_rows[k].append(vals[k])
    out: dict[str, np.ndarray] = {}
    for k, base in index.items():
        stacked = np.stack([np.asarray(v) for v in new_rows[k]])
        out[k] = np.concatenate([base, stacked.astype(base.dtype, copy=False)], axis=0)
    if len(out["cd"]) != n0 + len(points):
        raise AssertionError("row count mismatch after augmentation")
    return out


def axes_envelope(params_rows: Sequence[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Hull-form-axis envelope ``(min, max)`` of a corpus's design rows."""
    env: dict[str, tuple[float, float]] = {}
    for axis in HULLFORM_AXES:
        vals = [float(p.get(axis, 1.0)) for p in params_rows]
        env[axis] = (min(vals), max(vals))
    return env


# ---------------------------------------------------------------------------
# Honest hull-form verdict (contract duplicated from PR #241)
# ---------------------------------------------------------------------------


def honest_verdict(
    channel_verdict: GuardVerdict,
    params: dict[str, Any],
    env: dict[str, tuple[float, float]],
) -> GuardVerdict:
    """Guard verdict that never answers confident ``ok`` out-of-corpus.

    Duplicates the honesty contract of PR #241
    ``geometry_pipeline._downgrade_hullform`` (``exp/b4-echo`` @ 98c617f7,
    unmerged): a design whose hull-form axes leave the corpus envelope is
    downgraded to at least ``review`` with the underlying channel-space
    verdict preserved in the reasons; ``reject`` is never weakened.  When
    the loop retrains on acquired family points the envelope widens and
    the SAME query legitimately answers ``ok`` — that is the un-flagging
    this module demonstrates (and it is only honest because the retrained
    model actually covers the design).
    """
    outside = [
        axis
        for axis in HULLFORM_AXES
        if axis in env
        and not (
            env[axis][0] - _AXIS_EPS <= float(params.get(axis, 1.0)) <= env[axis][1] + _AXIS_EPS
        )
    ]
    if not outside or channel_verdict.flag in (FLAG_REJECT, FLAG_REVIEW):
        return channel_verdict
    reason = (
        "hull-form axes ("
        + ", ".join(outside)
        + ") outside the served corpus envelope; channel-space guard said "
        + f"flag={channel_verdict.flag} score={channel_verdict.score:.2f} "
        + "(downgraded to review; active-learning acquisition of labeled family "
        "points widens this envelope)"
    )
    return GuardVerdict(
        flag=FLAG_REVIEW,
        score=channel_verdict.score,
        reasons=(reason,) + tuple(channel_verdict.reasons),
    )


# ---------------------------------------------------------------------------
# Trend statistics (the #241 disagreement, quantified)
# ---------------------------------------------------------------------------


def _rankdata_avg(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean of their rank block)."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=np.float64)
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_rho(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    """Spearman rank correlation (tie-safe, numpy-only)."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.size != ya.size or xa.size < 2:
        raise ValueError("spearman_rho needs two same-length arrays (N>=2)")
    ra = _rankdata_avg(xa)
    rb = _rankdata_avg(ya)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float(ra @ ra) * float(rb @ rb))
    if denom == 0.0:
        return 0.0
    return float(ra @ rb / denom)


@dataclass(frozen=True)
class TrendStat:
    """Direction of a one-axis C_D sweep, per Re point and pooled."""

    rho_per_re: tuple[float, ...]
    mean_rho: float
    sign: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "rho_per_re": list(self.rho_per_re),
            "mean_rho": self.mean_rho,
            "sign": self.sign,
        }


def trend_stat(values: Sequence[float], cd_rows: np.ndarray) -> TrendStat:
    """Spearman rho of C_D vs the swept axis, per Re column + pooled mean.

    ``cd_rows`` is ``(n_values, n_re)`` — one C_D curve per swept value.
    """
    cd = np.asarray(cd_rows, dtype=np.float64)
    vals = [float(v) for v in values]
    if cd.ndim != 2 or cd.shape[0] != len(vals) or cd.shape[1] < 1:
        raise ValueError(f"cd_rows must be (n_values={len(vals)}, n_re>=1), got {cd.shape}")
    rhos = [spearman_rho(vals, cd[:, j]) for j in range(cd.shape[1])]
    mean_rho = float(np.mean(rhos))
    return TrendStat(
        rho_per_re=tuple(float(r) for r in rhos),
        mean_rho=mean_rho,
        sign=int(np.sign(mean_rho)),
    )


# ---------------------------------------------------------------------------
# Serving bundle + design prediction
# ---------------------------------------------------------------------------


@dataclass
class ServiceSpec:
    """Everything needed to serve one ensemble snapshot of the loop."""

    ckpt_dir: str | Path
    guard_features: np.ndarray  # (N, 8) corpus condition matrix
    axes_env: dict[str, tuple[float, float]]
    corpus_cache: np.ndarray | None = None  # (N, 5, ny, nx)
    cache_re: np.ndarray | None = None
    cache_designs: list[tuple[str, float, float, float]] | None = None
    device: str = "cpu"
    ckpt_glob: str = "*.pt"

    def backend(self) -> ModelEnsembleBackend:
        paths = sorted(Path(self.ckpt_dir).glob(self.ckpt_glob))
        if not paths:
            raise FileNotFoundError(f"no {self.ckpt_glob} checkpoints in {self.ckpt_dir}")
        return ModelEnsembleBackend([load_checkpoint(p) for p in paths], device=self.device)

    def guard(self) -> EnvelopeMahalanobisGuardrail:
        return EnvelopeMahalanobisGuardrail(np.asarray(self.guard_features, dtype=np.float64))

    def nearest_field(
        self, hull_type: str, sail_scale: float, fin_scale: float, re_hint: float, u_in: float
    ) -> tuple[np.ndarray, int]:
        """Corpus field row of the exact design nearest in log-Re.

        Mirrors ``inference_service.DragSurrogateService._resolve_field``
        (exact ``(hull, sail, fin, u_in)`` match, nearest log-Re); the
        hull-form variants resolve to their mother design's field — the
        variant enters through its condition channels, the field is a
        documented approximation (same contract as PR #241).
        """
        if self.corpus_cache is None or self.cache_re is None or self.cache_designs is None:
            raise ValueError("ServiceSpec has no corpus field cache attached")
        best: tuple[float, int] | None = None
        for row, key in enumerate(self.cache_designs):
            if (
                key[0] == hull_type
                and abs(key[1] - float(sail_scale)) <= 1e-12
                and abs(key[2] - float(fin_scale)) <= 1e-12
                and abs(key[3] - float(u_in)) <= 1e-12
            ):
                d = abs(
                    math.log10(max(self.cache_re[row], 1e-12)) - math.log10(max(re_hint, 1e-12))
                )
                if best is None or d < best[0]:
                    best = (d, row)
        if best is None:
            raise ValueError(
                f"design ({hull_type}, {sail_scale}, {fin_scale}, u_in={u_in}) not in the "
                "attached field cache"
            )
        row = int(best[1])
        return np.asarray(self.corpus_cache[row], dtype=np.float32), row


def predict_design(
    backend: ModelEnsembleBackend,
    spec: ServiceSpec,
    params: dict[str, Any],
    re_list: Sequence[float] | np.ndarray,
    *,
    grid: SuboffGrid = PRODUCTION_GRID,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Serve one (possibly variant) design over ``re_list``.

    Returns ``(member_cd (M, N), mean (N,), std (N,))`` — the condition
    rows use the hull-form-aware construction, the reference field is the
    mother-design corpus field nearest in log-Re.
    """
    re = np.asarray(re_list, dtype=np.float64).ravel()
    u_in = float(params.get("u_in", 0.1))
    cond = hullform_condition_rows(params, re, grid=grid, u_in=u_in)
    field, _row = spec.nearest_field(
        str(params.get("hull_type", "with_sail")),
        float(params.get("sail_scale", 1.0)),
        float(params.get("fin_scale", 1.0)),
        float(re[0]),
        u_in,
    )
    member_cd = backend.predict(field, cond)
    mean, std, _lo, _hi = ensemble_stats(member_cd)
    return member_cd, mean, std


# ---------------------------------------------------------------------------
# Retrain protocol — VERBATIM from b4_serve_20260824/train_ensemble.py
# (which copies train_fno_v4.py verbatim and cannot be imported)
# ---------------------------------------------------------------------------

ARCH_BASE = dict(in_ch=5, width=32, n_layers=4, modes=(16, 32), mlp_hidden=128, film_hidden=64)
HP = dict(epochs=500, batch=32, lr=1e-3, wd=1e-4, patience=60, seed=0)
SPLIT_SEED = 0
VAL_SEED = 1
VAL_FRAC = 0.15
TEST_FRAC = 0.20
AUX_DIM = 8
AUX_LAMBDA = 0.1
ARM_CFG: dict[str, Any] = dict(cond="v3", sampling="quota", aux_lam=AUX_LAMBDA)  # C_full


def corpus_with_cond(index: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Attach ``cond_v3`` and ``ylog`` columns (the training view)."""
    d = dict(index)
    d["cond_v3"] = corpus_cond_v3(index)
    d["ylog"] = np.log10(index["cd"])
    return d


def _split_groups(d: dict[str, np.ndarray], idx: list[int]) -> dict[str, tuple[list[int], int]]:
    keys = corpus_param_keys(d)
    g: dict[str, list[int]] = {}
    for i in idx:
        g.setdefault(keys[i], []).append(i)
    return {k: (v, int(d["dsi"][min(v)])) for k, v in g.items()}


def _carve_val(d: dict[str, np.ndarray], train: list[int]) -> tuple[list[int], list[int]]:
    rng = np.random.RandomState(VAL_SEED)
    fit, val = [], []
    groups = _split_groups(d, train)
    for s in sorted({ss for _, ss in groups.values()}):
        g = {k: v for k, (v, ss) in groups.items() if ss == s}
        ks = sorted(g)
        rng.shuffle(ks)
        n_val = max(1, int(round(VAL_FRAC * len(ks))))
        for k in ks[:n_val]:
            val += g[k]
        for k in ks[n_val:]:
            fit += g[k]
    return sorted(fit), sorted(val)


def split_random(d: dict[str, np.ndarray]) -> dict[str, list[int]]:
    """Random split of a corpus (verbatim discipline of train_ensemble.py).

    Per-dataset group carve with ``SPLIT_SEED`` (test) and ``VAL_SEED``
    (val) — the same seeds and fractions as the serving run, so the
    ORIGINAL rows keep identical fit/val/test membership after
    augmentation: each dataset is shuffled with its own draws in dataset
    order, and appended datasets (higher ``dsi``) only add draws after
    the originals.
    """
    rng = np.random.RandomState(SPLIT_SEED)
    idx = list(range(len(d["cd"])))
    train: list[int] = []
    test: list[int] = []
    groups = _split_groups(d, idx)
    for ds in sorted({ss for _, ss in groups.values()}):
        g = {k: v for k, (v, s) in groups.items() if s == ds}
        ks = sorted(g)
        rng.shuffle(ks)
        n_test = max(1, int(round(TEST_FRAC * len(ks))))
        for k in ks[:n_test]:
            test += g[k]
        for k in ks[n_test:]:
            train += g[k]
    fit, val = _carve_val(d, sorted(train))
    return {"train": sorted(train), "fit": fit, "val": val, "test": sorted(test)}


def fit_stats(
    x: np.ndarray,
    cond: np.ndarray,
    ylog: np.ndarray,
    aux: np.ndarray,
    idx: list[int],
) -> dict[str, Any]:
    """Z-score statistics from the FIT split only (serving protocol)."""
    cx = x[idx]
    cm = cx.transpose(1, 0, 2, 3).reshape(cx.shape[1], -1)
    p = cond[idx]
    y = ylog[idx]
    a = aux[idx]
    return dict(
        ch_mean=cm.mean(axis=1),
        ch_std=np.maximum(cm.std(axis=1), 1e-8),
        p_mean=p.mean(axis=0),
        p_std=np.where(p.std(axis=0) < 1e-6, 1.0, p.std(axis=0)),
        y_mean=float(y.mean()),
        y_std=float(max(y.std(), 1e-8)),
        a_mean=a.mean(axis=0),
        a_std=np.where(a.std(axis=0) < 1e-6, 1.0, a.std(axis=0)),
    )


def train_member(
    d: dict[str, np.ndarray],
    split: dict[str, list[int]],
    device: torch.device,
    seed: int,
    hp_base: dict[str, Any] | None = None,
) -> tuple[CondFNODrag, dict[str, Any], dict[str, Any]]:
    """One ensemble member (verbatim loop of train_ensemble.train_member)."""
    cfg = ARM_CFG
    x, cd, ylog, aux = d["x"], d["cd"], d["ylog"], d["aux"]
    cond_arr = d["cond_v3"]
    fit_idx, val_idx = split["fit"], split["val"]
    hp = dict(HP if hp_base is None else hp_base, seed=seed)
    epochs = int(hp["epochs"])
    batch = int(hp["batch"])
    patience = int(hp["patience"])
    aux_lam = float(cfg["aux_lam"])
    arch = dict(
        ARCH_BASE,
        cond_dim=cond_arr.shape[1],
        aux_dim=AUX_DIM if float(cfg["aux_lam"]) > 0 else 0,
    )
    torch.manual_seed(hp["seed"])
    st = fit_stats(x, cond_arr, ylog, aux, fit_idx)
    fit_arr = np.asarray(fit_idx)
    sampler = (
        QuotaSampler(d["dsi"][fit_arr], np.arange(len(fit_arr)))
        if cfg["sampling"] == "quota"
        else None
    )
    rng = np.random.default_rng(1000 + hp["seed"])

    xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    pt = torch.from_numpy(cond_arr.astype(np.float32)).to(device)
    yt = torch.from_numpy(ylog.astype(np.float32)).to(device)
    at = torch.from_numpy(np.asarray(aux, dtype=np.float32)).to(device)
    ch_m = torch.as_tensor(st["ch_mean"], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    ch_s = torch.as_tensor(st["ch_std"], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    p_m = torch.as_tensor(st["p_mean"], dtype=torch.float32, device=device)
    p_s = torch.as_tensor(st["p_std"], dtype=torch.float32, device=device)
    a_m = torch.as_tensor(st["a_mean"], dtype=torch.float32, device=device)
    a_s = torch.as_tensor(st["a_std"], dtype=torch.float32, device=device)

    def norm(idxs: list[int]) -> tuple[torch.Tensor, ...]:
        sel = torch.as_tensor(np.asarray(idxs), dtype=torch.long, device=device)
        xn = (xt[sel] - ch_m) / ch_s
        pn = (pt[sel] - p_m) / p_s
        yn = (yt[sel] - float(st["y_mean"])) / float(st["y_std"])
        an = (at[sel] - a_m) / a_s
        return xn, pn, yn, an

    xf, pf, yf, af = norm(fit_idx)
    xv, pv, _, _ = norm(val_idx)

    model = CondFNODrag(**arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["wd"]))
    n = len(fit_idx)
    best, best_state, best_ep = np.inf, None, -1
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        if sampler is None:
            perm = torch.randperm(n, device=xf.device)
        else:
            perm = torch.as_tensor(sampler.epoch_indices(rng), device=xf.device)
        for b in range(0, len(perm), batch):
            sel = perm[b : b + batch]
            opt.zero_grad()
            if aux_lam > 0:
                y_hat, a_hat = cast(
                    "tuple[torch.Tensor, torch.Tensor]",
                    model(xf[sel], pf[sel], return_aux=True),
                )
                loss = torch.nn.functional.mse_loss(y_hat, yf[sel]) + aux_lam * (
                    torch.nn.functional.mse_loss(a_hat, af[sel])
                )
            else:
                loss = torch.nn.functional.mse_loss(
                    cast(torch.Tensor, model(xf[sel], pf[sel])), yf[sel]
                )
            # torch ships Tensor.backward untyped (every trainer in this repo
            # carries the same no-untyped-call); silenced to keep the gate literal
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
        model.eval()
        with torch.no_grad():
            zv = model(xv, pv)
        cd_pred = 10.0 ** (zv.double().cpu().numpy() * float(st["y_std"]) + float(st["y_mean"]))
        vm = float(np.mean(np.abs(cd_pred - cd[val_idx]) / cd[val_idx]) * 100)
        if vm < best - 1e-9:
            best, best_ep = vm, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return (
        model,
        st,
        dict(
            best_epoch=best_ep,
            best_val_mape=best,
            train_secs=round(time.time() - t0, 1),
            seed=seed,
            arch=arch,
            hp=hp,
        ),
    )


def predict_rows(
    model: CondFNODrag,
    st: dict[str, Any],
    x: np.ndarray,
    cond: np.ndarray,
    idxs: Sequence[int] | np.ndarray,
    device: torch.device,
    batch: int = 64,
) -> np.ndarray:
    """Batched C_D prediction on corpus rows (verbatim predict)."""
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    pt = torch.from_numpy(np.asarray(cond, dtype=np.float32)).to(device)
    ch_m = torch.as_tensor(st["ch_mean"], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    ch_s = torch.as_tensor(st["ch_std"], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    p_m = torch.as_tensor(st["p_mean"], dtype=torch.float32, device=device)
    p_s = torch.as_tensor(st["p_std"], dtype=torch.float32, device=device)
    out = []
    model.eval()
    with torch.no_grad():
        for b in range(0, len(idxs), batch):
            sel = np.asarray(idxs[b : b + batch])
            xn = (xt[sel] - ch_m) / ch_s
            pn = (pt[sel] - p_m) / p_s
            z = model(xn, pn).double().cpu().numpy()
            out.append(10.0 ** (z * float(st["y_std"]) + float(st["y_mean"])))
    return np.concatenate(out)


def retrain_ensemble(
    corpus: dict[str, np.ndarray],
    out_dir: str | Path,
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    device: torch.device | str = "cpu",
    corpus_tag: str = "b4_al_augmented",
    hp_overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Incremental retrain on the augmented corpus (serving protocol).

    Same arch / split machinery / quota sampling / aux loss / early stop
    as ``train_ensemble.py``; fit statistics are recomputed from the
    AUGMENTED fit split (the protocol computes z-scores from whatever the
    fit split is — keeping the pre-augmentation stats would silently
    mismatch the new condition rows).  Members are saved as
    ``CondDragCheckpoint`` bundles named ``al_aug_s{k}.pt``.
    ``hp_overrides`` exists for CPU tests (the demo uses the serving
    defaults).
    """
    hp_base = dict(HP, **(hp_overrides or {}))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    d = corpus_with_cond(corpus)
    split = split_random(d)
    te = np.asarray(split["test"])
    records: list[dict[str, Any]] = []
    for seed in seeds:
        model, st, info = train_member(d, split, dev, seed, hp_base=hp_base)
        pred = predict_rows(model, st, d["x"], d["cond_v3"], te, dev)
        true = d["cd"][te]
        mape = float(np.mean(np.abs(pred - true) / true) * 100)
        ckpt = CondDragCheckpoint(
            arch=info["arch"],
            state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
            norm=dict(
                ch_mean=st["ch_mean"],
                ch_std=st["ch_std"],
                p_mean=st["p_mean"],
                p_std=st["p_std"],
                y_mean=st["y_mean"],
                y_std=st["y_std"],
            ),
            meta=dict(
                arm="C_full",
                split="random",
                seed=seed,
                corpus=corpus_tag,
                protocol="active_learning.retrain_ensemble (train_ensemble.py verbatim)",
                best_val_mape=info["best_val_mape"],
                test_mape=mape,
            ),
        )
        path = out_dir / f"al_aug_s{seed}.pt"
        save_checkpoint(ckpt, path)
        records.append(dict(info, mape=mape, ckpt=str(path), n_fit=len(split["fit"])))
    return records


# ---------------------------------------------------------------------------
# Loop evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrendSpec:
    """One trend-sign test (the #241 disagreement, quantified)."""

    axis: str
    values: tuple[float, ...]
    re_grid: tuple[float, ...]
    base_params: dict[str, Any]
    truth_cd: np.ndarray | None = None  # (n_values, n_re) cache-interpolated truth

    def as_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "values": list(self.values),
            "re_grid": list(self.re_grid),
            "truth_cd": None if self.truth_cd is None else np.asarray(self.truth_cd).tolist(),
        }


@dataclass(frozen=True)
class MotherEval:
    """Fixed in-corpus eval rows (mother-corpus regression guard)."""

    x: np.ndarray  # (N, 5, ny, nx)
    cond: np.ndarray  # (N, 8)
    cd: np.ndarray  # (N,)


@dataclass
class LoopReport:
    """Every number the loop demonstration reports (JSON via as_dict)."""

    n_eval_points: int
    family_mape_before: float
    family_mape_after: float
    verdicts_before: dict[str, int]
    verdicts_after: dict[str, int]
    verdict_flips: list[dict[str, Any]]
    member_std_mean_before: float
    member_std_mean_after: float
    trend: dict[str, Any]
    mother_mape_before: float | None
    mother_mape_after: float | None
    per_point: list[dict[str, Any]]
    ckpt_dirs: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_eval_points": self.n_eval_points,
            "family_mape_before": self.family_mape_before,
            "family_mape_after": self.family_mape_after,
            "verdicts_before": dict(self.verdicts_before),
            "verdicts_after": dict(self.verdicts_after),
            "verdict_flips": self.verdict_flips,
            "member_std_mean_before": self.member_std_mean_before,
            "member_std_mean_after": self.member_std_mean_after,
            "trend": self.trend,
            "mother_mape_before": self.mother_mape_before,
            "mother_mape_after": self.mother_mape_after,
            "per_point": self.per_point,
            "ckpt_dirs": dict(self.ckpt_dirs),
        }


def eval_loop(
    before_ckpt_dir: str | Path,
    after_ckpt_dir: str | Path,
    eval_points: Sequence[AcquisitionPoint],
    *,
    guard_features_before: np.ndarray,
    guard_features_after: np.ndarray,
    axes_env_before: dict[str, tuple[float, float]],
    axes_env_after: dict[str, tuple[float, float]],
    corpus_cache: np.ndarray,
    cache_re: np.ndarray,
    cache_designs: list[tuple[str, float, float, float]],
    trend: TrendSpec | None = None,
    mother_eval: MotherEval | None = None,
    device: str = "cpu",
    grid: SuboffGrid = PRODUCTION_GRID,
    labels_cd: dict[str, float] | None = None,
) -> LoopReport:
    """Before/after verdict of the active-learning loop.

    - **family MAPE** on ``eval_points`` (ground truth from ``labels_cd``
      keyed by :func:`point_param_key`; points without a label are skipped);
    - **verdict flips**: honest verdicts (channel guard + corpus axes
      envelope) before vs after retrain;
    - **trend-sign test** (the headline): the swept-axis C_D direction vs
      ``trend.truth_cd`` before and after;
    - **mother MAPE**: the fixed in-corpus rows must not regress beyond
      noise.

    Both snapshots resolve reference fields from the SAME mother corpus
    cache (apples-to-apples: the after-improvement must come from
    conditioning and training, not from a luckier field source).
    """
    specs = {
        "before": ServiceSpec(
            ckpt_dir=before_ckpt_dir,
            guard_features=guard_features_before,
            axes_env=axes_env_before,
            corpus_cache=corpus_cache,
            cache_re=cache_re,
            cache_designs=cache_designs,
            device=device,
        ),
        "after": ServiceSpec(
            ckpt_dir=after_ckpt_dir,
            guard_features=guard_features_after,
            axes_env=axes_env_after,
            corpus_cache=corpus_cache,
            cache_re=cache_re,
            cache_designs=cache_designs,
            device=device,
        ),
    }
    built = {k: (s.backend(), s.guard()) for k, s in specs.items()}

    labels = labels_cd or {}
    per_point: list[dict[str, Any]] = []
    flips: list[dict[str, Any]] = []
    apes: dict[str, list[float]] = {"before": [], "after": []}
    stds: dict[str, list[float]] = {"before": [], "after": []}
    verdicts: dict[str, dict[str, int]] = {"before": {}, "after": {}}
    for p in eval_points:
        true = labels.get(p.key)
        if true is None or not np.isfinite(true):
            continue
        row: dict[str, Any] = {
            "key": p.key,
            "params": dict(p.params),
            "re": p.re,
            "cd_true": float(true),
        }
        flags: dict[str, str] = {}
        for tag in ("before", "after"):
            backend, guard = built[tag]
            _mat, mean, std = predict_design(backend, specs[tag], p.params, [p.re], grid=grid)
            cond = hullform_condition_rows(p.params, [p.re], grid=grid)
            chan = guard.check(cond)
            verdict = honest_verdict(chan, p.params, specs[tag].axes_env)
            flags[tag] = verdict.flag
            ape = abs(float(mean[0]) - float(true)) / abs(float(true))
            apes[tag].append(ape)
            stds[tag].append(float(std[0]))
            verdicts[tag][verdict.flag] = verdicts[tag].get(verdict.flag, 0) + 1
            row[f"cd_{tag}"] = float(mean[0])
            row[f"ape_{tag}"] = ape
            row[f"chan_flag_{tag}"] = chan.flag
            row[f"chan_score_{tag}"] = float(chan.score)
            row[f"flag_{tag}"] = verdict.flag
        row["flip"] = f"{flags['before']}->{flags['after']}"
        flips.append({"key": p.key, "before": flags["before"], "after": flags["after"]})
        per_point.append(row)

    if not apes["before"]:
        raise ValueError("no eval points had labels; pass labels_cd for every point")

    trend_out: dict[str, Any] = {}
    if trend is not None:
        trend_out["spec"] = trend.as_dict()
        if trend.truth_cd is not None:
            truth = trend_stat(trend.values, np.asarray(trend.truth_cd, dtype=np.float64))
            trend_out["truth"] = truth.as_dict()
        for tag in ("before", "after"):
            backend = built[tag][0]
            rows = []
            for val in trend.values:
                params = dict(trend.base_params)
                params[trend.axis] = float(val)
                _mat, mean, _std = predict_design(
                    backend, specs[tag], params, trend.re_grid, grid=grid
                )
                rows.append(mean)
            st = trend_stat(trend.values, np.stack(rows))
            trend_out[tag] = st.as_dict()
            trend_out[tag]["cd"] = np.stack(rows).tolist()
        if "truth" in trend_out:
            trend_out["sign_agree_before"] = (
                trend_out["before"]["sign"] == trend_out["truth"]["sign"]
            )
            trend_out["sign_agree_after"] = trend_out["after"]["sign"] == trend_out["truth"]["sign"]
            trend_out["flipped_to_agree"] = bool(
                not trend_out["sign_agree_before"] and trend_out["sign_agree_after"]
            )

    mother: dict[str, float | None] = {"before": None, "after": None}
    if mother_eval is not None:
        for tag in ("before", "after"):
            backend = built[tag][0]
            preds = []
            for i in range(len(mother_eval.cd)):
                mat = backend.predict(
                    np.asarray(mother_eval.x[i], dtype=np.float32),
                    np.asarray(mother_eval.cond[i], dtype=np.float64)[None, :],
                )
                mean, _s, _lo, _hi = ensemble_stats(mat)
                preds.append(float(mean[0]))
            ape = np.abs(np.asarray(preds) - mother_eval.cd) / mother_eval.cd
            mother[tag] = float(np.mean(ape) * 100)

    return LoopReport(
        n_eval_points=len(apes["before"]),
        family_mape_before=float(np.mean(apes["before"]) * 100),
        family_mape_after=float(np.mean(apes["after"]) * 100),
        verdicts_before=verdicts["before"],
        verdicts_after=verdicts["after"],
        verdict_flips=flips,
        member_std_mean_before=float(np.mean(stds["before"])),
        member_std_mean_after=float(np.mean(stds["after"])),
        trend=trend_out,
        mother_mape_before=mother["before"],
        mother_mape_after=mother["after"],
        per_point=per_point,
        ckpt_dirs={"before": str(before_ckpt_dir), "after": str(after_ckpt_dir)},
    )


def write_loop_report(report: LoopReport, path: str | Path) -> str:
    """Serialise a LoopReport as JSON (numpy-aware)."""

    def _default(o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        raise TypeError(f"not JSON serializable: {type(o)}")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.as_dict(), indent=1, default=_default))
    return str(p)
