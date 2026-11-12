"""Bluff-body (square-cylinder) benchmark support: masks, force/Strouhal
reduction and reference-anchored scoring.

Motivation
----------
The vortex-shedding benchmarks of this repository (``cylinder``,
``square_cylinder``) each re-implemented their own obstacle mask, force
normalisation, Strouhal extraction and reference comparison.  That produced
two failure modes seen in audits:

* two different reference constants living in two files, so the *same* raw
  data yielded two different error percentages (m16383/m16205 audit finding);
* ad-hoc Strouhal estimators whose window/taper choices were undocumented.

This module keeps the physics-facing pieces in one importable place so that a
benchmark script only has to describe its *case* (geometry, Re, resolution)
and then hand the measured series to :func:`assess` for scoring.

Units
-----
Everything is in lattice units: ``D`` is the obstacle side in lattice nodes,
``u_in`` is the inflow velocity in lattice units, ``dt = 1``.  With
``rho = 1`` the dynamic pressure over the frontal area per unit depth is
``q = 0.5 * rho * u_in**2 * D``.

Nothing in this module depends on a particular domain size, Reynolds number or
reference paper: reference values are passed in (typically loaded from a JSON
file next to the case, see :func:`load_reference`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

__all__ = [
    "rect_mask_2d",
    "blockage_ratio",
    "force_coefficients",
    "strouhal_fft",
    "strouhal_zero_crossings",
    "strouhal_from_series",
    "drag_statistics",
    "load_reference",
    "reference_band",
    "assess",
    "assess_case",
]


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def rect_mask_2d(
    nx: int,
    ny: int,
    cx: float,
    cy: float,
    side: float,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Axis-aligned solid square mask with a *half-way bounce-back* wall.

    The solid block covers ``round(side)`` nodes in each direction; half-way
    bounce-back places the physical wall half a link outside the outermost
    solid node, so the effective obstacle size is exactly ``side`` lattice
    units (not ``side + 1``).  This is what makes ``Cd`` and ``St`` comparable
    with body-length-normalised references.

    Args:
        nx, ny: grid dimensions (``ny`` = cross-stream, ``nx`` = streamwise).
        cx, cy: obstacle centre in lattice nodes.
        side: obstacle side length in lattice units (``D``).
        device: torch device (default: CPU).

    Returns:
        Boolean tensor of shape ``(ny, nx)``, ``True`` inside the solid.
    """
    cand = [
        int(round(cx - side / 2.0)),
        int(round(cy - side / 2.0)),
        int(round(side)),
    ]
    x0, y0, n = cand
    if n < 1:
        raise ValueError(f"side={side} rounds to <1 node")
    if x0 < 0 or y0 < 0 or x0 + n > nx or y0 + n > ny:
        raise ValueError(
            f"obstacle [{y0}:{y0 + n}, {x0}:{x0 + n}] does not fit in grid ({ny}, {nx})"
        )
    mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    mask[y0 : y0 + n, x0 : x0 + n] = True
    return mask


def blockage_ratio(mask: torch.Tensor) -> float:
    """Cross-stream blockage ``B = body height / channel height``.

    Measured from the mask itself (not from the requested parameters), so a
    report can state the *achieved* blockage of the grid that actually ran.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2-D (ny, nx)")
    solid_rows = int(mask.any(dim=1).sum().item())
    if solid_rows == 0:
        raise ValueError("mask is empty")
    return solid_rows / mask.shape[0]


# --------------------------------------------------------------------------
# force reduction
# --------------------------------------------------------------------------
def force_coefficients(
    fx: Sequence[float] | np.ndarray,
    fy: Sequence[float] | np.ndarray,
    u_in: float,
    side: float,
    rho: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalise raw momentum-exchange forces to ``(Cd, Cl)``.

    ``Cd = Fx / (0.5 rho u_in^2 D)`` and likewise for ``Cl`` with ``Fy``.
    Per unit depth; ``D`` is the frontal height, i.e. the obstacle side.
    """
    q = 0.5 * rho * u_in * u_in * float(side)
    if q <= 0.0:
        raise ValueError("dynamic-pressure scale must be positive")
    fx = np.asarray(fx, dtype=np.float64)
    fy = np.asarray(fy, dtype=np.float64)
    if fx.shape != fy.shape:
        raise ValueError(f"fx/fy length mismatch: {fx.shape} vs {fy.shape}")
    return fx / q, fy / q


def drag_statistics(cd: Sequence[float] | np.ndarray, start: int = 0) -> dict[str, float | int]:
    """Time statistics of a drag series over the analysis window ``cd[start:]``."""
    arr = np.asarray(cd, dtype=np.float64)[max(0, int(start)) :]
    if arr.size == 0:
        raise ValueError("empty analysis window")
    return {
        "n_samples": int(arr.size),
        "mean": float(arr.mean()),
        "rms": float(np.sqrt(np.mean(arr * arr))),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std(ddof=0)),
    }


# --------------------------------------------------------------------------
# Strouhal extraction
# --------------------------------------------------------------------------
def _detrend_taper(sig: np.ndarray) -> np.ndarray:
    s = np.asarray(sig, dtype=np.float64)
    if s.size < 8:
        raise ValueError(f"lift series too short for spectral analysis: {s.size}")
    s = s - s.mean()
    return s * np.hanning(s.size)


def strouhal_fft(cl: Sequence[float] | np.ndarray, u_in: float, side: float) -> dict[str, float]:
    """Shedding frequency from the lift spectrum: ``St = f D / U``.

    Returns the peak frequency, ``St``, the peak bin's half-width frequency
    resolution ``1/(N dt)`` and the spectral energy fraction inside the
    Hanning main lobe ``f_peak +/- 2/T`` (a shed-quality indicator: a clean,
    single-frequency wake puts ~all of the fluctuating energy in the
    fundamental, a broadband/beating wake puts much less).
    """
    s = _detrend_taper(np.asarray(cl, dtype=np.float64))
    n = s.size
    spec = np.abs(np.fft.rfft(s))
    freqs = np.fft.rfftfreq(n, d=1.0)
    k = int(np.argmax(spec[1:])) + 1  # skip DC
    f_peak = float(freqs[k])
    # parabolic (3-point) refinement of the peak location
    if 0 < k < spec.size - 1:
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        denom = a - 2.0 * b + c
        if abs(denom) > 1e-300:
            delta = 0.5 * (a - c) / denom
            if abs(delta) <= 1.0:
                f_peak += delta * float(freqs[1] - freqs[0])
    df = float(freqs[1] - freqs[0])
    # Hanning main lobe spans +/-2 bins; the fraction inside that lobe is the
    # "single-frequency wake" indicator (a pure tone gives ~1.0).
    band = np.abs(freqs - f_peak) <= 2.0 * df
    energy = float((spec[band] ** 2).sum() / max((spec[1:] ** 2).sum(), 1e-300))
    return {
        "f_peak": f_peak,
        "st": f_peak * float(side) / float(u_in),
        "df": df,
        "bin": k,
        "energy_frac": energy,
    }


def strouhal_zero_crossings(
    cl: Sequence[float] | np.ndarray, u_in: float, side: float
) -> dict[str, float]:
    """Shedding frequency from mean-crossing intervals (time-domain check).

    Counts upward crossings of the mean lift in the analysis window and uses
    the mean interval between them as the period.  This estimator makes no
    stationarity/spectral assumption and is therefore the natural companion /
    cross-check of :func:`strouhal_fft`; it is noisier when the lift signal
    itself carries a slow modulation.
    """
    s = np.asarray(cl, dtype=np.float64)
    if s.size < 16:
        raise ValueError(f"lift series too short for crossing analysis: {s.size}")
    s = s - s.mean()
    up = np.nonzero((s[:-1] <= 0.0) & (s[1:] > 0.0))[0]
    if up.size < 3:
        return {"st": float("nan"), "n_crossings": int(up.size), "period": float("nan")}
    period = float(np.diff(up).mean())
    return {
        "st": float(side) / (period * float(u_in)),
        "n_crossings": int(up.size),
        "period": period,
    }


def strouhal_from_series(
    cl: Sequence[float] | np.ndarray, u_in: float, side: float
) -> dict[str, Any]:
    """Primary (FFT) Strouhal estimate plus its time-domain cross-check."""
    fft = strouhal_fft(cl, u_in, side)
    cross = strouhal_zero_crossings(cl, u_in, side)
    return {"fft": fft, "crossing": cross, "st": fft["st"]}


# --------------------------------------------------------------------------
# reference-anchored scoring
# --------------------------------------------------------------------------
def load_reference(path: str | Path) -> dict[str, Any]:
    """Load a case reference file (JSON).

    Expected shape::

        {"case": ..., "rows": [{"re":200, "metric":"cd", "ref":1.42,
          "band":[1.34,1.50], "tol_pct":10.0, "source":"...", "grade":"..."}]}

    ``ref`` may be ``null`` when the reference is *withheld* (original table
    not obtainable); such rows are carried through scoring as ``WITHHELD``
    instead of being silently dropped or, worse, invented.
    """
    data = json.loads(Path(path).read_text())
    if "rows" not in data or not isinstance(data["rows"], list):
        raise ValueError(f"{path}: reference file needs a 'rows' list")
    for row in data["rows"]:
        for key in ("re", "metric"):
            if key not in row:
                raise KeyError(f"{path}: reference row missing {key!r}: {row}")
        if row.get("ref") is None and not row.get("withheld_reason"):
            raise KeyError(
                f"{path}: row {row} has no ref and no withheld_reason — "
                "a withheld reference must say why"
            )
    return data


def reference_band(rows: Iterable[Mapping[str, Any]], re: float, metric: str) -> dict[str, Any]:
    """Select the reference row active for ``(re, metric)``.

    Exact ``re`` match first; otherwise the row with the largest ``re`` not
    exceeding the requested one (a "last verified rung"), flagged as
    ``exact=False`` so callers can report an extrapolated comparison honestly.
    """
    cand = [r for r in rows if r["metric"] == metric and r.get("re") is not None]
    if not cand:
        raise KeyError(f"no reference row for metric={metric!r}")
    exact = [r for r in cand if abs(float(r["re"]) - float(re)) < 1e-9]
    if exact:
        return {**exact[0], "exact": True}
    lower = [r for r in cand if float(r["re"]) < float(re)]
    if not lower:
        raise KeyError(f"no reference row for metric={metric!r} at or below re={re}")
    best = max(lower, key=lambda r: float(r["re"]))
    return {**best, "exact": False}


def assess(got: float | None, row: Mapping[str, Any]) -> dict[str, Any]:
    """Score one measured value against one reference row.

    Returns a row containing ``ref``, ``got``, ``err_pct``, ``tol_pct`` and a
    ``status`` of ``PASS`` / ``FAIL`` / ``WITHHELD`` / ``NO_DATA``.  The
    comparison is against the reference *centre*; ``band`` (if present) is
    echoed so a report can also show the band-relative position.
    """
    ref = row.get("ref")
    tol = row.get("tol_pct")
    out: dict[str, Any] = {
        "metric": row["metric"],
        "re": row.get("re"),
        "ref": ref,
        "band": row.get("band"),
        "tol_pct": tol,
        "got": None if got is None else float(got),
        "source": row.get("source"),
        "grade": row.get("grade"),
    }
    if ref is None:
        out.update(status="WITHHELD", err_pct=None, withheld_reason=row.get("withheld_reason"))
        return out
    if got is None or not math.isfinite(float(got)):
        out.update(status="NO_DATA", err_pct=None)
        return out
    err = (float(got) - float(ref)) / float(ref) * 100.0
    out["err_pct"] = err
    if tol is None:
        out["status"] = "NO_TOLERANCE"
        return out
    out["status"] = "PASS" if abs(err) <= float(tol) else "FAIL"
    if row.get("exact") is False:
        out["note"] = (
            "reference rung below the case Re (no exact-match row): comparison is "
            "trend-indicative, not an exact-referee pass/fail"
        )
    return out


def assess_case(
    got: Mapping[str, float | None],
    reference: Mapping[str, Any],
    re: float,
) -> list[dict[str, Any]]:
    """Score every metric in ``got`` against the loaded reference file."""
    rows = reference["rows"]
    out = []
    for metric, value in got.items():
        try:
            row = reference_band(rows, re, metric)
        except KeyError:
            out.append({"metric": metric, "status": "NO_REFERENCE", "got": value})
            continue
        out.append(assess(value, row))
    return out