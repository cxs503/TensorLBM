"""B4-W16 — anchor-row selection for bringing a new axis value into support.

Campaign verdict (wave-15 W2, ``/nfs/wangxi/runs/anchor_min_20260829/``,
recipe written down in ``docs/serving_v6qx_20260828.md``): the SDF
two-stage path loses a constant multiplicative offset only when an axis
value has *no* training support, and the cheapest verified repair is to
add a handful of scanned anchor rows at the new axis value to the
training corpus.  The measured law at slender ``l_over_d`` 1.30:

* ``k = 3`` anchors at the **min / geometric-mid / max** of the intended
  query range in log10(Re) collapse held-out MAPE from 10.78 % to
  0.41 % (ts2) / 0.21 % (ts4), with the per-decade Re-slope error gone
  (``-0.8`` to ``+0.1`` %/decade vs ``+3`` to ``+13`` for ``k = 1``);
* the controlling variable is the anchors' **log10(Re) span, not their
  count**: two adjacent anchors (0.03-decade span) stay at 3.6 %, the
  one failing random ``k = 3`` draw (0.32-decade span) lands at 1.15 %,
  and every draw with at least ~0.6-decade span lands at 0.4-0.5 %;
* returns saturate from ``k >= 2`` (``k = 22`` only reaches 0.27 %).

This module is the pure selection layer: given the query range a caller
wants to serve at the new axis value, it emits the anchor Reynolds
numbers to scan, checks the span rule, and — when the axis value already
has archived rows (the slender-1.30 case: 28 held-out rows exist and
three of them promote the 350-row hole corpus to 353 rows) — picks the
rows nearest the targets.  It contains no I/O and no training; scan
drivers live in the run directories, and corpus refresh goes through
the existing corpus/retrain entries (see the serving doc recipe).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "MIN_SPAN_DECADES",
    "anchor_targets",
    "match_anchor_rows",
    "promote_anchor_rows",
    "span_decades",
    "validate_span",
]

#: Below this log10(Re) span the slope cliff is not cleared, regardless
#: of how many anchors are added (measured: 0.03 -> 3.6 %, 0.32 -> 1.15 %,
#: >= 0.64 -> 0.4-0.5 %; the 0.4 cut is the conservative in-between).
MIN_SPAN_DECADES = 0.4


def anchor_targets(re_min: float, re_max: float, k: int = 3) -> np.ndarray:
    """Geometric-spaced anchor Reynolds numbers covering ``[re_min, re_max]``.

    The campaign recipe is ``k = 3`` — the two range ends plus the
    geometric midpoint, i.e. equally spaced in log10(Re).  ``k = 2``
    (the ends only) is the budget variant that still clears the slope
    cliff whenever the span rule holds; ``k = 1`` never does and is
    rejected here.

    Parameters
    ----------
    re_min:
        Smallest Reynolds number the new axis value should serve.
    re_max:
        Largest Reynolds number the new axis value should serve.
    k:
        Number of anchors, ``>= 2``.

    Returns
    -------
    numpy.ndarray
        Shape ``(k,)`` float64 array, ascending, within
        ``[re_min, re_max]`` (the endpoints are exact).
    """
    if k < 2:
        raise ValueError(f"k = {k} anchors never clear the slope cliff; use k >= 2")
    if not 0.0 < re_min < re_max:
        raise ValueError(f"need 0 < re_min < re_max, got re_min={re_min}, re_max={re_max}")
    return np.geomspace(re_min, re_max, num=k)


def span_decades(re_values: np.ndarray | list[float] | tuple[float, ...]) -> float:
    """log10 span of a set of Reynolds numbers (max - min, in decades)."""
    arr = np.asarray(re_values, dtype=np.float64)
    if arr.size < 2:
        raise ValueError("span needs at least two values")
    if np.any(arr <= 0.0):
        raise ValueError("Reynolds numbers must be positive")
    return float(np.log10(arr.max()) - np.log10(arr.min()))


def validate_span(
    re_values: np.ndarray | list[float] | tuple[float, ...],
    min_span_decades: float = MIN_SPAN_DECADES,
) -> tuple[bool, float]:
    """Check the span rule: anchors must straddle ``min_span_decades``.

    Returns ``(ok, span)``; ``span`` is :func:`span_decades` of the
    inputs so callers can log how much margin the set has.
    """
    span = span_decades(re_values)
    return (span >= min_span_decades, span)


def match_anchor_rows(
    row_re: np.ndarray | list[float],
    targets: np.ndarray | list[float],
    max_log10_distance: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick, for each anchor target, the archived row nearest in log10(Re).

    For the promote-from-held-out case (the axis value already has
    scanned rows): given the Reynolds numbers of the archived rows and
    the :func:`anchor_targets` output, return the row indices that best
    realise the anchor set.

    **Warning — order semantics (the 2026-08-30 blunt-anchor footgun):**
    ``row_indices`` follows **target order** (ascending anchor Re), *not*
    the order of any cached corpus.  Corpus arrays loaded from a training
    cache / npz ride cache order, so the ONLY correct consumption is
    positional indexing — ``row_arrays[i][row_indices]`` — never
    ``zip(row_arrays, row_indices)``: a zip silently pairs each anchor
    with the wrong row (two separate consumers hit exactly this).

    >>> corpus_re = np.array([300.0, 50.0, 185.0, 700.0, 90.0])  # cache order
    >>> targets = anchor_targets(50.0, 700.0, k=3)
    >>> row_indices, achieved = match_anchor_rows(corpus_re, targets)
    >>> assert np.array_equal(corpus_re[row_indices], achieved)  # positional, never zip

    Parameters
    ----------
    row_re:
        Reynolds numbers of the archived rows (any order).
    targets:
        Anchor targets from :func:`anchor_targets`.
    max_log10_distance:
        Tolerance, in decades, for a row to count as realising a target
        (0.05 ~ 12 %).  A target with no row within tolerance raises —
        the axis value needs a scan, not a promotion.

    Returns
    -------
    row_indices:
        Shape ``(len(targets),)`` int64 array of indices into ``row_re``,
        in **target order** (NOT cache order).
    achieved:
        Shape ``(len(targets),)`` float64 array of the matched rows'
        Reynolds numbers (ascending, like the targets).
    """
    rows = np.asarray(row_re, dtype=np.float64)
    tgt = np.asarray(targets, dtype=np.float64)
    if rows.size == 0:
        raise ValueError("no archived rows to match against")
    if np.any(rows <= 0.0) or np.any(tgt <= 0.0):
        raise ValueError("Reynolds numbers must be positive")
    log_rows = np.log10(rows)
    log_tgt = np.log10(tgt)
    idx = np.argmin(np.abs(log_tgt[:, None] - log_rows[None, :]), axis=1)
    dist = np.abs(log_tgt - log_rows[idx])
    for t, d, i in zip(tgt, dist, idx, strict=True):
        if d > max_log10_distance:
            raise ValueError(
                f"no archived row within {max_log10_distance} decades of target "
                f"Re = {t:.4g} (nearest row Re = {rows[i]:.4g}, {d:.3f} decades "
                "away) — scan this anchor instead of promoting a held-out row"
            )
    if len(set(idx.tolist())) != len(idx):
        raise ValueError("one archived row matched two anchor targets; span the targets")
    return idx.astype(np.int64), rows[idx]


def promote_anchor_rows(
    row_re: np.ndarray | list[float],
    targets: np.ndarray | list[float],
    max_log10_distance: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Promote archived rows to anchors, returning ``(achieved, row_indices)``.

    A thin alias over :func:`match_anchor_rows` for the
    promote-from-held-out call sites: identical matching and identical
    rejections (it delegates), with the returns swapped so the reading
    is positional-first — the achieved Reynolds numbers, then the indices
    to apply to every corpus array.

    **Warning — same order semantics as :func:`match_anchor_rows`:**
    ``row_indices`` follows **target order**, *not* the order of any
    cached corpus.  Consume positionally, ``row_arrays[i][row_indices]``
    — never ``zip(row_arrays, row_indices)`` (see the warning and
    usage example on :func:`match_anchor_rows`).

    Parameters
    ----------
    row_re:
        Reynolds numbers of the archived rows (any order).
    targets:
        Anchor targets from :func:`anchor_targets`.
    max_log10_distance:
        Tolerance, in decades, for a row to count as realising a target
        (0.05 ~ 12 %).  A target with no row within tolerance raises —
        the axis value needs a scan, not a promotion.

    Returns
    -------
    achieved:
        Shape ``(len(targets),)`` float64 array of the matched rows'
        Reynolds numbers (ascending, like the targets).
    row_indices:
        Shape ``(len(targets),)`` int64 array of indices into ``row_re``,
        in **target order** (NOT cache order).
    """
    row_indices, achieved = match_anchor_rows(row_re, targets, max_log10_distance)
    return achieved, row_indices
