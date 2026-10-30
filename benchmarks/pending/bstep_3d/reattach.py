"""Robust reattachment-length measurement for the 3D backward-facing step (bstep_3d).

Background
----------
The pending bstep_3d records (``result_H40.json`` / ``result_re200_H49.json`` /
``result_re400_H49.json``) were produced by an estimator that took the *first*
negative->positive zero crossing of the near-wall tangential velocity::

    pos = int(np.argmax(row > 0.0))
    if pos == 0 or row[pos] <= 0.0:
        return None            # -> caller stores xr_h = 0.0

That single line has three defects, all of which fire on the recorded data:

1. ``row[0] > 0`` (the first downstream node next to the step corner is
   positive - normal when the corner cell is not resolved as a hard corner, and
   what happens at the higher-Re records) makes ``argmax`` return 0, so the
   function returns ``None`` and the caller silently writes ``xr_h = 0.0``.
   ``result_re400_H49.json`` is self-inconsistent proof: it reports
   ``separation_bubble.max_backflow_over_umax = -0.175`` at
   ``backflow_x_rel_h = 4.35`` *and* ``xr_h = 0.0``.  A separated flow with a
   reattachment length of zero cannot exist.
2. Taking the first crossing ignores a secondary/corner recirculation cell:
   if the wall shear goes negative -> positive -> negative -> positive (corner
   bubble plus main bubble), the first crossing is not the main reattachment.
3. ``None`` (no crossing found) is overloaded onto three physically different
   outcomes: no backflow at all (already attached), backflow that never
   reattaches inside the domain (X_r larger than the outlet), and corrupted
   data.  Downstream convergence logic then treats all three as ``xr_h = 0.0``
   and reports "converged" on garbage (``result_re400_H49.json``:
   ``converged: true``, ``xr_h: 0.0``).

This module replaces that estimator.  It is measurement-only: no change to the
flow solver, boundary conditions or geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# status codes ---------------------------------------------------------------
REATTACHED = "reattached"  # main recirculation closes inside the sampled window
NO_BACKFLOW = "no_backflow"  # wall shear never goes negative -> X_r = 0 by physics
NOT_REATTACHED = "not_reattached_in_domain"  # backflow still present at last node
NO_DATA = "no_data"  # empty / all-NaN input


@dataclass
class Reattachment:
    """Result of :func:`locate_reattachment`.

    ``x`` is the reattachment coordinate in the same units as the profile index
    (i.e. lattice columns, offset by ``x0``); ``None`` whenever the status is
    not :data:`REATTACHED`.
    """

    x: float | None
    status: str
    n_negative_runs: int = 0
    corner_attached: bool = False
    min_rel: float | None = None  # strongest backflow / reference scale
    backflow_start: float | None = None
    backflow_end: float | None = None
    run_lengths: list[int] = field(default_factory=list)
    run_areas: list[float] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == REATTACHED

    def rel(self, x_origin: float, scale: float) -> float | None:
        """Normalised length ``(x - x_origin) / scale`` (``None`` if not located)."""
        if self.x is None:
            return None
        return float((self.x - x_origin) / scale)


def _smooth(profile: np.ndarray, width: int) -> np.ndarray:
    """Centred moving average with edge-replicated padding; ``width <= 1`` is a no-op.

    Edge replication matters: zero padding (``np.convolve(..., mode="same")``)
    would shrink the first and last samples by up to ``(width-1)/width`` and
    could hide the backflow right at the step corner or shift the closure node.
    """
    if width is None or width <= 1:
        return profile.astype(np.float64, copy=False)
    if width % 2 == 0:
        width += 1
    if width > profile.size:
        return profile.astype(np.float64, copy=False)
    half = width // 2
    padded = np.pad(profile.astype(np.float64), (half, half), mode="edge")
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(padded, kernel, mode="valid")


def _negative_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive ``[start, end]`` index ranges of the True runs of ``mask``."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def locate_reattachment(
    profile,
    x0: float = 0.0,
    *,
    smooth: int = 3,
    rel_thresh: float = 1e-3,
    u_scale: float | None = None,
) -> Reattachment:
    """Locate the main recirculation closure from a near-wall tangential velocity row.

    Parameters
    ----------
    profile
        1-D array of the wall-tangential velocity at successive streamwise nodes
        starting at the step face (index 0 == ``x0``).  For D3Q19 bounce-back the
        first off-wall node ``y = 1`` carries ``u_tau ∝ tau_w`` up to a constant,
        so a sign change of this row is the standard reattachment marker.
    x0
        Streamwise coordinate of ``profile[0]`` (lattice columns).
    smooth
        Moving-average width (odd, in cells) used before sign detection.
        Default 3 suppresses one-cell lattice noise; set 1 to disable.
    rel_thresh
        Backflow detection threshold as a fraction of ``u_scale``.  Prevents
        round-off-level negative noise from being counted as separation.
    u_scale
        Reference velocity for ``rel_thresh`` (typically the inlet ``U_max``).
        Defaults to ``max|profile|``.

    Returns
    -------
    :class:`Reattachment`
        The reattachment point is the *terminating* crossing of the run with the
        largest backflow area (length x strength), i.e. the main bubble - not the
        first crossing, so a corner recirculation cell cannot truncate the
        measurement, and a one-cell noise blip cannot win on length alone.  ``status``
        distinguishes "never separated" from "not reattached in the window",
        which the pre-fix estimator collapsed onto ``0.0``.
    """
    arr = np.asarray(profile, dtype=np.float64).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr).any():
        return Reattachment(x=None, status=NO_DATA)
    finite = np.isfinite(arr)
    if not finite.all():
        arr = np.where(finite, arr, 0.0)

    scale = float(np.max(np.abs(arr))) if u_scale is None else float(u_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        return Reattachment(x=None, status=NO_DATA, min_rel=0.0)

    sm = _smooth(arr, smooth)
    thresh = rel_thresh * scale
    min_rel = float(sm.min()) / scale

    runs = _negative_runs(sm < -thresh)
    if not runs:
        return Reattachment(x=None, status=NO_BACKFLOW, min_rel=min_rel)

    lengths = [e - s + 1 for s, e in runs]
    # main bubble = largest negative area, i.e. length x strength (ties -> upstream)
    areas = [float(np.sum(-sm[a : b + 1])) for a, b in runs]
    k = int(np.argmax(areas))
    s, e = runs[k]
    corner_attached = s == 0

    base = dict(
        n_negative_runs=len(runs),
        corner_attached=corner_attached,
        min_rel=min_rel,
        backflow_start=float(x0 + s),
        backflow_end=float(x0 + e),
        run_lengths=lengths,
        run_areas=areas,
    )

    if e >= arr.size - 1:
        return Reattachment(x=None, status=NOT_REATTACHED, **base)

    a, b = sm[e], sm[e + 1]
    if b - a == 0.0:
        return Reattachment(x=None, status=NO_DATA, **base)
    x_cross = x0 + e + (0.0 - a) / (b - a)
    return Reattachment(x=float(x_cross), status=REATTACHED, **base)


def legacy_first_crossing(profile, x0: float = 0.0) -> float | None:
    """Exact copy of the pre-fix estimator from ``run.py`` / ``run_highre.py``.

    Kept **only** to reproduce and document the defect (mission m469): it returns
    ``None`` whenever the first downstream node is already positive, which the
    callers turned into ``xr_h = 0.0``.
    """
    row = np.asarray(profile, dtype=np.float64).reshape(-1)
    if row.size == 0:
        return None
    pos = int(np.argmax(row > 0.0))
    if pos == 0 or row[pos] <= 0.0:
        return None
    a, b = row[pos - 1], row[pos]
    if b - a == 0.0:
        return None
    return x0 + (pos - 1) + (0.0 - a) / (b - a)


def measure_planes(
    ux,
    x_step: int,
    step_h: int,
    *,
    y: int = 1,
    smooth: int = 3,
    rel_thresh: float = 1e-3,
    u_scale: float | None = None,
) -> dict:
    """Spanwise-resolved reattachment measurement for a 3D field ``ux[z, y, x]``.

    Coordinate convention: the step face downstream of which the bottom wall
    exists sits at column ``x_step`` (fluid columns are ``x >= x_step``), so the
    physical step face is at ``x = x_step - 0.5`` and a closure at column
    ``c = x_step + i`` has ``X_r/h = (i + 0.5) / step_h`` -- the same convention
    as the pre-fix estimator.

    Returns the spanwise-averaged main value plus plane statistics.  Invalid
    planes are excluded from the statistics and counted in ``n_planes_invalid``
    instead of being folded in as ``0.0`` (the pre-fix behaviour).
    """
    arr = np.asarray(ux, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"expected ux[z, y, x], got shape {arr.shape}")
    nz = arr.shape[0]
    half = 0.5  # step face offset from the first fluid column

    plane_vals: list[float] = []
    statuses: list[str] = []
    for z in range(nz):
        r = locate_reattachment(
            arr[z, y, x_step:],
            x0=0.0,
            smooth=smooth,
            rel_thresh=rel_thresh,
            u_scale=u_scale,
        )
        statuses.append(r.status)
        if r.x is not None:
            plane_vals.append((r.x + half) / step_h)

    main = locate_reattachment(
        arr[:, y, x_step:].mean(axis=0),
        x0=0.0,
        smooth=smooth,
        rel_thresh=rel_thresh,
        u_scale=u_scale,
    )
    valid = np.array(plane_vals, dtype=float)

    def _h(coord: float | None) -> float | None:
        return None if coord is None else (coord + half) / step_h

    return {
        "xr_h": None if main.x is None else (main.x + half) / step_h,
        "xr_status": main.status,
        "xr_n_negative_runs": main.n_negative_runs,
        "xr_corner_attached": bool(main.corner_attached),
        "xr_min_rel": main.min_rel,
        "xr_backflow_start_h": _h(main.backflow_start),
        "xr_backflow_end_h": _h(main.backflow_end),
        "xr_planes_mean": float(valid.mean()) if valid.size else None,
        "xr_planes_std": float(valid.std()) if valid.size else None,
        "xr_planes_min": float(valid.min()) if valid.size else None,
        "xr_planes_max": float(valid.max()) if valid.size else None,
        "xr_planes_valid": int(valid.size),
        "xr_planes_invalid": int(nz - valid.size),
        "xr_plane_statuses": {s: statuses.count(s) for s in sorted(set(statuses))},
    }
