"""Retrieval-based reference fields for new-geometry two-stage serving (v1).

Serving a NEW geometry through the SDF two-stage drag surrogate previously
required an LBM run to produce the 5-channel reference mid-plane field.
The 2026-09-04 field-sensitivity study
(``/nfs/wangxi/runs/l2_field_sensitivity_20260904/`` — machine truth
``study.json``, verdict §7 of ``report.md``) showed that requirement is
artificial: the frozen SDF encoder carries ALL geometry signal and the
reference field only has to stay inside the corpus manifold.

Headline numbers (held-25 ensemble MAPE, ts2 arm):

- field swaps of ANY strategy are invisible: cond-nearest donor **x0.998**
  of baseline, corpus-MEAN fields x0.998, random donor x1.004 — all below
  batch-composition fp noise;
- feeding a DIFFERENT geometry's SDF is catastrophic: **x57.9** (ts2,
  12.5605 % vs 0.2171 % baseline; x59.3 on ts4) — hence the hard contract
  that this provider NEVER supplies a donor SDF;
- the only observed failure mode is a grossly out-of-manifold field
  (all-zeros probe: x7.7 ts2 / x17.9 ts4), guarded here by a relative-L2
  in-manifold check against the pool-mean fields.

v1 is deliberately dumb — nearest-neighbour retrieval, a mean fallback,
and a guard.  The study verdict leaves no accuracy budget for a learned
field generator: anything in-manifold scores like anything else.

2026-09-08 follow-ups (``/nfs/wangxi/runs/borrow_cache_20260908/``):
this provider memoizes borrow results per query CONTENT (an LRU keyed on
a blake2b digest of the query arrays — never object identity), so
repeated queries of the same geometry pay the ``sdf_near`` scan once
(57.7 ms retrieval -> cache hit on the 378-row production pool), and it
carries the calibrated out-of-family ``distance_advisory_threshold``
consumed by :mod:`tensorlbm.ai.field_borrow` (see
:data:`DISTANCE_ADVISORY_THRESHOLD` for the calibration).
"""

from __future__ import annotations

import hashlib
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["BorrowedField", "FieldProvider"]

#: Reference-field channel count expected by the two-stage drag surrogate.
FIELD_CHANNELS = 5

#: Retrieval strategies understood by :meth:`FieldProvider.borrow`.
STRATEGIES: tuple[str, ...] = ("sdf_near", "cond_near", "mean")

#: Machine-readable evidence this module implements (read-only).
STUDY_PATH = "/nfs/wangxi/runs/l2_field_sensitivity_20260904"

#: Default in-manifold guard threshold (see :class:`FieldProvider`).
GUARD_THRESHOLD = 0.15

#: Default borrow-result cache size in LRU entries (0 disables the cache).
DEFAULT_BORROW_CACHE_SIZE = 16

#: Default out-of-family distance-advisory threshold (see
#: :class:`FieldProvider`).  Calibrated 2026-09-08 as the MAX of the
#: design-level leave-one-out nearest-neighbour SDF-L2 distance
#: distribution of the 406-row production corpus itself (122 unique
#: designs; each design's SDF to its nearest OTHER-design SDF, exact
#: float64 chunked L2 — the runtime ``sdf_near`` math):
#: min 0.000 / median 1.124 / p90 2.366 / **max 8.875**.  The corpus is
#: a frozen finite population, so its LOO max IS the exact in-family
#: envelope — every corpus design sits at <= 8.875 by construction and
#: the rule has zero false positives on the corpus — while a p90
#: multiple (e.g. 2 x p90 = 4.73) would mislabel the legitimate slender
#: tail (designs at 2.94-8.88) as out-of-family.  Walkthrough anchors:
#: design 106 in-family at 2.981 (ratio 0.34), the sphere probe at
#: 32.58 (ratio 3.67, ``suspect``).  Machine truth:
#: ``/nfs/wangxi/runs/borrow_cache_20260908/calibration.json``.
DISTANCE_ADVISORY_THRESHOLD = 8.875449208906158

#: Pool rows scanned per distance block (bounds peak memory on big pools).
_CHUNK_ROWS = 64

#: Keys every production cache file must carry for :meth:`FieldProvider.from_corpus`.
_CORPUS_KEYS = ("x", "dsi", "re", "uin", "sail", "fin")


@dataclass
class BorrowedField:
    """One borrowed reference field — deliberately carries NO SDF.

    Attributes:
        fields: the borrowed 5-channel reference field (a private copy,
            ``(5, ...)`` with the pool's dtype; for ``strategy="mean"`` the
            float64 pool mean cast back to the pool dtype, mirroring the
            study's float64 -> float32 API boundary).  READ-ONLY by
            contract: on a provider-cache hit the SAME array object is
            returned again (see :meth:`FieldProvider.borrow`), so callers
            must not mutate it in place — every downstream consumer
            (the two-stage backends, the guard) only reads it.
        donor_index: pool row the field came from, or ``None`` for the
            mean fallback.
        strategy: ``"sdf_near"`` | ``"cond_near"`` | ``"mean"``.
        distance: the retrieval distance to the chosen donor (full-array
            SDF-L2 or plain cond-L2, float64), or ``None`` for ``"mean"``.
        guard_ok: whether ``guard_rel_l2 <= guard_threshold``.
        guard_rel_l2: ``||fields - pool_mean|| / ||pool_mean||`` (float64);
            exactly ``0.0`` for the mean strategy (the anchor is itself).
        guard_threshold: the threshold ``guard_ok`` was judged against.
        provenance: pool size, donor key (when the loader knows one), the
            study citation, and the strategy-resolution audit trail.

    There is intentionally no ``sdf`` attribute: the donor's SDF is the one
    input the study measured as catastrophic (x57.9 / x59.3 held MAPE), so
    the API makes returning it unrepresentable.  The caller always feeds
    the TARGET's own SDF to the surrogate.
    """

    fields: np.ndarray
    donor_index: int | None
    strategy: str
    distance: float | None
    guard_ok: bool
    guard_rel_l2: float
    guard_threshold: float
    provenance: dict[str, Any]


def _content_key(arr: np.ndarray | None) -> tuple[Any, ...] | None:
    """Content fingerprint of a query array: ``(dtype, shape, blake2b)``.

    The digest covers the raw bytes, so equal-content queries hit the
    same cache entry regardless of which array object carries them (the
    borrow cache is keyed on CONTENT, never on object identity); shape
    and dtype ride along so a reinterpreted buffer can never collide
    with a genuine repeat.  Returns ``None`` for ``None`` (an absent
    input is its own key part).
    """
    if arr is None:
        return None
    a = np.asarray(arr)
    digest = hashlib.blake2b(a.tobytes(), digest_size=16).hexdigest()
    return (a.dtype.str, a.shape, digest)


def _nearest_row(pool: np.ndarray, target: np.ndarray) -> tuple[int, float]:
    """argmin_i ``||pool[i] - target||_2`` over flattened rows.

    Chunked float64 accumulation (bounded peak memory on the 406 x 32 x 32
    x 64 production SDF pool); ties break to the LOWEST row index, so the
    result is deterministic for identical inputs.
    """
    flat = np.asarray(target, dtype=np.float64).ravel()
    best_idx = 0
    best_d2 = math.inf
    n = int(pool.shape[0])
    for start in range(0, n, _CHUNK_ROWS):
        block = pool[start : start + _CHUNK_ROWS]
        diff = block.reshape(block.shape[0], -1).astype(np.float64, copy=False) - flat
        d2 = np.einsum("ij,ij->i", diff, diff)
        j = int(np.argmin(d2))
        if float(d2[j]) < best_d2:
            best_d2 = float(d2[j])
            best_idx = start + j
    return best_idx, math.sqrt(best_d2)


class FieldProvider:
    """Borrow in-manifold reference fields from a corpus pool.

    The pool is supplied BY THE CALLER as plain arrays — no paths are
    hardcoded.  ``pool_fields`` is ``(N, 5, ...)``; ``pool_sdfs`` and
    ``pool_cond`` are optional and enable the matching retrieval strategy
    (``(N, ...)`` voxel SDFs and ``(N, C)`` condition rows respectively).

    Guard semantics: after retrieval, the borrowed field is checked for
    in-manifold-ness as ``||f - mean(pool_fields)|| / ||mean(pool_fields)||``
    (float64).  ``guard_ok`` iff that ratio is ``<= guard_threshold``
    (default ``0.15``, the study's operating point; measured on the
    production 406-row pool: corpus rows sit at 0.024-0.159 with median
    ~0.054, so a strict 0.15 flags a five-row dsi=3/4/5 tail and 0.16
    covers every corpus row, while the all-zeros probe — the only measured
    failure mode, x7.7 ts2 / x17.9 ts4 held MAPE — sits at exactly 1.0,
    a >6x margin).  The threshold is a constructor argument so operators
    can tighten or loosen it without editing code.

    Scope caveat: the invariance numbers were measured with IN-CORPUS
    donors; a borrowed field that passes the guard but is unlike any
    corpus row is out of the study's support, hence the guard.

    Per-geometry result cache (2026-09-08 follow-up #1):
    :meth:`borrow` memoizes its :class:`BorrowedField` in a bounded LRU
    (``cache_size`` entries, default :data:`DEFAULT_BORROW_CACHE_SIZE`,
    ``0`` disables) keyed on the CONTENT of the query arrays —
    ``(strategy, sdf digest, cond digest)`` via blake2b over the raw
    bytes plus shape/dtype, never object identity.  Pure memoization:
    same inputs, same outputs — the pool is treated as immutable after
    construction, so a hit replays exactly what a miss computes (the
    sdf_near scan over the 378-row production pool costs ~58 ms per
    query; repeated same-geometry queries — a Re sweep is one retrieval,
    a dashboard re-query is zero — pay it once).  Default ON because the
    borrow path itself is opt-in and the flag-off service composition is
    untouched either way.  Read-only contract: a hit returns the SAME
    ``BorrowedField`` (same ``fields`` array object, same ``provenance``
    dict) as the miss that populated it; callers must treat both as
    read-only — every downstream consumer only reads them.  On a miss
    ``fields`` is still a private copy of the pool row, so a caller that
    DOES mutate it can corrupt later cache hits but never the pool.

    Out-of-family distance advisory (2026-09-08 follow-up #2): the
    constructor also records ``distance_advisory_threshold`` (default
    :data:`DISTANCE_ADVISORY_THRESHOLD`, the corpus LOO-NN envelope —
    see the constant for the calibration).  The provider itself makes no
    decision with it; :func:`tensorlbm.ai.field_borrow.borrow_serving_field`
    reads it to attach a NON-BLOCKING ``distance_advisory`` to the
    response ``info`` (the retrieval distance is the one geometry-side
    out-of-family signal for STL-sourced shapes — the field guard
    measures the borrowed field, in-manifold by construction, and the
    cond guard sees only CAD descriptors).  Operators calibrating a
    different pool pass their own threshold at construction.
    """

    def __init__(
        self,
        pool_fields: np.ndarray,
        pool_sdfs: np.ndarray | None = None,
        pool_cond: np.ndarray | None = None,
        guard_threshold: float = GUARD_THRESHOLD,
        cache_size: int = DEFAULT_BORROW_CACHE_SIZE,
        distance_advisory_threshold: float = DISTANCE_ADVISORY_THRESHOLD,
    ) -> None:
        fields = np.asarray(pool_fields)
        if fields.ndim < 2 or fields.shape[1] != FIELD_CHANNELS:
            raise ValueError(
                f"pool_fields must be (N, {FIELD_CHANNELS}, ...) stacked "
                f"{FIELD_CHANNELS}-channel reference fields, got shape {fields.shape}"
            )
        if fields.shape[0] == 0:
            raise ValueError("pool_fields must contain at least one row, got shape (0, ...)")
        self.pool_fields = fields

        if pool_sdfs is None:
            self.pool_sdfs: np.ndarray | None = None
        else:
            sdfs = np.asarray(pool_sdfs)
            if sdfs.ndim < 2 or sdfs.shape[0] != fields.shape[0]:
                raise ValueError(
                    f"pool_sdfs must be (N, ...) with N={fields.shape[0]} rows matching "
                    f"pool_fields, got shape {sdfs.shape}"
                )
            self.pool_sdfs = sdfs

        if pool_cond is None:
            self.pool_cond: np.ndarray | None = None
        else:
            cond = np.asarray(pool_cond)
            if cond.ndim != 2 or cond.shape[0] != fields.shape[0] or cond.shape[1] == 0:
                raise ValueError(
                    f"pool_cond must be (N, C) with N={fields.shape[0]} rows matching "
                    f"pool_fields and C >= 1 columns, got shape {cond.shape}"
                )
            self.pool_cond = cond

        if not guard_threshold > 0.0:
            raise ValueError(f"guard_threshold must be > 0, got {guard_threshold!r}")
        self.guard_threshold = float(guard_threshold)

        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 0:
            raise ValueError(f"cache_size must be a non-negative int, got {cache_size!r}")
        self.cache_size = int(cache_size)
        self._borrow_cache: OrderedDict[tuple[Any, ...], BorrowedField] | None = (
            OrderedDict() if self.cache_size > 0 else None
        )

        if not distance_advisory_threshold > 0.0 or not math.isfinite(distance_advisory_threshold):
            raise ValueError(
                f"distance_advisory_threshold must be a finite positive number, "
                f"got {distance_advisory_threshold!r}"
            )
        self.distance_advisory_threshold = float(distance_advisory_threshold)

        self._mean_fields: np.ndarray | None = None
        self._pool_keys: list[str] | None = None
        self._source: str | None = None

    # ------------------------------------------------------------------ api

    def borrow(
        self,
        target_sdf: np.ndarray | None = None,
        target_cond: np.ndarray | None = None,
        strategy: str | None = None,
    ) -> BorrowedField:
        """Borrow one reference field for a NEW geometry.

        CONTRACT — the target owns the SDF.  The returned
        :class:`BorrowedField` carries ONLY the 5-channel reference field;
        the caller MUST pass the TARGET's own SDF (from the standard
        voxelisation + SDF path) to the two-stage surrogate.  Feeding a
        donor's SDF instead is the one catastrophic input mistake measured
        by the study — x57.9 (ts2) / x59.3 (ts4) held MAPE
        (``sdfswap_cond_near`` in ``study.json``) — which is why this API
        never returns an SDF at all.

        Strategy resolution (``strategy=None``): ``"sdf_near"`` when the
        pool has SDFs and ``target_sdf`` is given, else ``"cond_near"``
        when the pool has cond and ``target_cond`` is given, else
        ``"mean"``.  An explicit strategy whose inputs are missing raises
        :class:`ValueError` with a precise message instead of silently
        falling back.

        Distances: ``sdf_near`` uses full-array SDF-L2 (float64
        accumulation); ``cond_near`` uses plain Euclidean L2 over the
        supplied columns — normalise/z-score ``pool_cond`` and
        ``target_cond`` first if you want the study's z-space metric.

        The guard is evaluated on the BORROWED field (see class docstring);
        a ``guard_ok=False`` result is still returned, for the caller to
        refuse or escalate.

        Result cache: with ``cache_size > 0`` (the default) the computed
        :class:`BorrowedField` is memoized per query CONTENT (see the
        class docstring) and a repeat query returns the SAME object —
        byte-identical fields, identical provenance, no re-scan.  The
        cache sits AFTER strategy resolution and shape validation, so a
        malformed query still raises exactly as without it.  Treat the
        returned ``fields`` / ``provenance`` as read-only: they are
        shared with every other caller of the same query.
        """
        chosen = self._choose_strategy(target_sdf, target_cond, strategy)
        cache = self._borrow_cache
        key: tuple[Any, ...] | None = None
        if cache is not None:
            key = (chosen, _content_key(target_sdf), _content_key(target_cond))
            cached = cache.get(key)
            if cached is not None:
                cache.move_to_end(key)
                return cached
        provenance: dict[str, Any] = {
            "pool_size": int(self.pool_fields.shape[0]),
            "strategy": chosen,
            "requested": {
                "target_sdf": target_sdf is not None,
                "target_cond": target_cond is not None,
            },
            "study": STUDY_PATH,
            "source": self._source,
            "contract": (
                "caller passes the TARGET's own SDF to the surrogate; "
                "this provider never returns a donor SDF"
            ),
        }

        idx: int | None
        dist: float | None
        if chosen == "sdf_near":
            idx, dist = self._nearest_sdf(target_sdf)
        elif chosen == "cond_near":
            idx, dist = self._nearest_cond(target_cond)
        else:
            idx, dist = None, None

        if idx is None:
            fields = self._pool_mean_fields().astype(self.pool_fields.dtype, copy=True)
            guard_rel_l2 = 0.0
        else:
            fields = self.pool_fields[idx].copy()
            guard_rel_l2 = self._rel_l2_to_mean(fields)

        provenance["donor_key"] = (
            None if idx is None or self._pool_keys is None else self._pool_keys[idx]
        )
        if idx is not None:
            provenance["donor_index"] = idx

        result = BorrowedField(
            fields=fields,
            donor_index=idx,
            strategy=chosen,
            distance=dist,
            guard_ok=guard_rel_l2 <= self.guard_threshold,
            guard_rel_l2=guard_rel_l2,
            guard_threshold=self.guard_threshold,
            provenance=provenance,
        )
        if cache is not None and key is not None:
            cache[key] = result
            if len(cache) > self.cache_size:
                cache.popitem(last=False)  # evict the least-recently-used
        return result

    @classmethod
    def from_corpus(
        cls,
        path: str | os.PathLike[str],
        *,
        guard_threshold: float = GUARD_THRESHOLD,
        cache_size: int = DEFAULT_BORROW_CACHE_SIZE,
        distance_advisory_threshold: float = DISTANCE_ADVISORY_THRESHOLD,
    ) -> FieldProvider:
        """Build a provider from the production corpus artifact (READ-ONLY).

        Two accepted layouts for ``path``:

        1. A DIRECTORY holding the four production files, mirroring
           ``/nfs/wangxi/runs/ckpt_bundle_rehearsal_20260831/rehearsal.py``
           ``load_fam`` exactly (row order included)::

               cache_fam.npz    keys x/dsi/re/uin/sail/fin  (350 rows)
               cache_ext56.npz  same keys                    (56 slender rows)
               sdf_fam350.npz   key sdf                      (SDFs of rows 0-349)
               sdf_ext2.npz     keys d{dsi}                  (one SDF per ext row)

           The production copies live in four different run directories;
           assemble a VIEW directory of symlinks rather than copying data.
           ``cond`` is derived per row as
           ``[log10(re), log10(uin), log10(sail), log10(fin)]`` (the
           corpus convention: ts2 uses columns 0:2, ts4 uses 0:4), and each
           row gets a ``donor_key`` ``"dsi=.. re=.. uin=.. sail=.. fin=.."``
           for provenance.

        2. An ``.npz`` SNAPSHOT with key ``x`` (or ``fields``) plus optional
           ``sdf`` / ``cond`` / ``dsi`` / ``re`` (the latter two only feed
           ``donor_key`` provenance).

        Memory: the directory layout materialises the pool (~0.21 GB
        float32 for the 406-row production corpus: 105 MB fields + 107 MB
        SDFs).  For lazy access, load the members with
        ``np.load(..., mmap_mode="r")`` and hand the arrays to the ordinary
        constructor — it keeps them array-backed without copying.

        The study's donor pool was the 381-row in-support subset of this
        corpus (406 rows, 25 held out); this loader returns ALL rows and
        leaves subsetting to the caller.
        """
        p = os.fspath(path)
        if os.path.isfile(p) and p.endswith(".npz"):
            return cls._from_snapshot_npz(
                p,
                guard_threshold=guard_threshold,
                cache_size=cache_size,
                distance_advisory_threshold=distance_advisory_threshold,
            )
        if not os.path.isdir(p):
            raise FileNotFoundError(
                f"corpus path {p!r} is neither an .npz snapshot nor a directory"
            )
        return cls._from_production_dir(
            p,
            guard_threshold=guard_threshold,
            cache_size=cache_size,
            distance_advisory_threshold=distance_advisory_threshold,
        )

    # ------------------------------------------------------------- internals

    def _choose_strategy(
        self,
        target_sdf: np.ndarray | None,
        target_cond: np.ndarray | None,
        strategy: str | None,
    ) -> str:
        if strategy is None:
            if self.pool_sdfs is not None and target_sdf is not None:
                return "sdf_near"
            if self.pool_cond is not None and target_cond is not None:
                return "cond_near"
            return "mean"
        chosen = str(strategy)
        if chosen not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
        if chosen == "sdf_near":
            if self.pool_sdfs is None:
                raise ValueError(
                    "strategy 'sdf_near' requires pool_sdfs: this FieldProvider "
                    "was constructed without pool_sdfs"
                )
            if target_sdf is None:
                raise ValueError("strategy 'sdf_near' requires target_sdf: borrow(target_sdf=None)")
        if chosen == "cond_near":
            if self.pool_cond is None:
                raise ValueError(
                    "strategy 'cond_near' requires pool_cond: this FieldProvider "
                    "was constructed without pool_cond"
                )
            if target_cond is None:
                raise ValueError(
                    "strategy 'cond_near' requires target_cond: borrow(target_cond=None)"
                )
        return chosen

    def _nearest_sdf(self, target_sdf: np.ndarray | None) -> tuple[int, float]:
        pool = self.pool_sdfs
        if pool is None or target_sdf is None:  # guarded by _choose_strategy
            raise RuntimeError("sdf_near resolved without pool_sdfs/target_sdf")
        target = np.asarray(target_sdf)
        row_shape = pool.shape[1:]
        if target.shape != row_shape:
            raise ValueError(
                f"target_sdf must have shape {row_shape} to match the pool SDF "
                f"rows, got {target.shape}"
            )
        return _nearest_row(pool, target)

    def _nearest_cond(self, target_cond: np.ndarray | None) -> tuple[int, float]:
        pool = self.pool_cond
        if pool is None or target_cond is None:  # guarded by _choose_strategy
            raise RuntimeError("cond_near resolved without pool_cond/target_cond")
        target = np.asarray(target_cond)
        if target.shape != pool.shape[1:]:
            raise ValueError(
                f"target_cond must have shape {pool.shape[1:]} to match the "
                f"pool_cond columns, got {target.shape}"
            )
        return _nearest_row(pool, target)

    def _pool_mean_fields(self) -> np.ndarray:
        """Float64 per-pixel pool mean (computed lazily, cached)."""
        if self._mean_fields is None:
            self._mean_fields = np.mean(self.pool_fields, axis=0, dtype=np.float64)
        return self._mean_fields

    def _rel_l2_to_mean(self, fields: np.ndarray) -> float:
        mean = self._pool_mean_fields()
        diff = np.asarray(fields, dtype=np.float64) - mean
        num = float(np.linalg.norm(diff.ravel()))
        den = float(np.linalg.norm(mean.ravel()))
        if num == 0.0:
            return 0.0
        if den == 0.0:  # degenerate all-zero pool mean
            return math.inf
        return num / den

    @classmethod
    def _from_production_dir(
        cls,
        directory: str,
        *,
        guard_threshold: float,
        cache_size: int,
        distance_advisory_threshold: float,
    ) -> FieldProvider:
        def load_npz(name: str, required: tuple[str, ...]) -> dict[str, np.ndarray]:
            fp = os.path.join(directory, name)
            if not os.path.isfile(fp):
                raise FileNotFoundError(f"corpus file {fp} not found under {directory!r}")
            with np.load(fp) as z:
                missing = [k for k in required if k not in z.files]
                if missing:
                    raise ValueError(f"{name} is missing required keys {missing}")
                return {k: np.asarray(z[k]) for k in required}

        fam = load_npz("cache_fam.npz", _CORPUS_KEYS)
        ext = load_npz("cache_ext56.npz", _CORPUS_KEYS)
        fields = np.concatenate([fam["x"], ext["x"]])

        sdf_fam = load_npz("sdf_fam350.npz", ("sdf",))["sdf"]
        ext_ds = [int(v) for v in ext["dsi"]]
        with np.load(os.path.join(directory, "sdf_ext2.npz")) as zse:
            missing = [f"d{i}" for i in ext_ds if f"d{i}" not in zse.files]
            if missing:
                raise ValueError(f"sdf_ext2.npz is missing SDF keys {missing}")
            ext_sdf = np.stack([np.asarray(zse[f"d{i}"]) for i in ext_ds])
        sdfs = np.concatenate([sdf_fam, ext_sdf])
        if sdfs.shape[0] != fields.shape[0]:
            raise ValueError(
                f"SDF count {sdfs.shape[0]} does not match field rows {fields.shape[0]}"
            )

        # cond convention: [log10 re, log10 uin, log10 sail, log10 fin] per row.
        params = {k: np.concatenate([fam[k], ext[k]]) for k in ("dsi", "re", "uin", "sail", "fin")}
        for k in ("re", "uin", "sail", "fin"):
            if np.any(params[k] <= 0):
                raise ValueError(f"corpus column {k!r} has non-positive values; cannot take log10")
        cond = np.stack([np.log10(params[k]) for k in ("re", "uin", "sail", "fin")], axis=1)
        keys = [
            f"dsi={int(d)} re={float(r):.10g} uin={float(u):.10g} "
            f"sail={float(s):.10g} fin={float(f):.10g}"
            for d, r, u, s, f in zip(
                params["dsi"],
                params["re"],
                params["uin"],
                params["sail"],
                params["fin"],
                strict=True,
            )
        ]

        provider = cls(
            fields,
            pool_sdfs=sdfs,
            pool_cond=cond,
            guard_threshold=guard_threshold,
            cache_size=cache_size,
            distance_advisory_threshold=distance_advisory_threshold,
        )
        provider._pool_keys = keys
        provider._source = f"production-dir:{directory}"
        return provider

    @classmethod
    def _from_snapshot_npz(
        cls,
        path: str,
        *,
        guard_threshold: float,
        cache_size: int,
        distance_advisory_threshold: float,
    ) -> FieldProvider:
        with np.load(path) as z:
            files = list(z.files)
            fkey = "x" if "x" in files else ("fields" if "fields" in files else None)
            if fkey is None:
                raise ValueError(f"snapshot {path!r} has neither 'x' nor 'fields'; found {files}")
            fields = np.asarray(z[fkey])
            sdfs = np.asarray(z["sdf"]) if "sdf" in files else None
            cond = np.asarray(z["cond"]) if "cond" in files else None
            dsi = np.asarray(z["dsi"]) if "dsi" in files else None
            re_arr = np.asarray(z["re"]) if "re" in files else None
        keys = None
        if dsi is not None and re_arr is not None:
            keys = [f"dsi={int(a)} re={float(b):.10g}" for a, b in zip(dsi, re_arr, strict=True)]
        provider = cls(
            fields,
            pool_sdfs=sdfs,
            pool_cond=cond,
            guard_threshold=guard_threshold,
            cache_size=cache_size,
            distance_advisory_threshold=distance_advisory_threshold,
        )
        provider._pool_keys = keys
        provider._source = f"snapshot-npz:{path}"
        return provider
