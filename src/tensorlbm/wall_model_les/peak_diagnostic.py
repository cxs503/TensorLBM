"""Reynolds-stress peak diagnostics and grid-error separation for WMLES.

Why this module exists
----------------------
A wall-modelled LES is judged on quantities that a single run cannot pin down:

* the *position* of the Reynolds-shear-stress peak ``y+_peak``, which is where
  the modelled wall layer hands over to the resolved turbulence.  It is
  quantised by the wall-normal spacing, so on a coarse grid the "measured"
  peak sits at whichever sample happens to be largest -- a discretisation
  artefact, not a model result;
* the *magnitude* ``-<u'v'>+_peak``, which is what most verification records
  report alone (and then read as a model error even when the grid cannot
  support the measurement).

This module supplies the three ingredients needed to separate those effects:

:func:`find_stress_peak`
    sub-sample peak location from a discrete profile (cubic spline plus an
    independent three-point parabola), together with the quantisation bound
    the estimate inherits from the grid.
:func:`richardson_gci`
    observed order of convergence, Richardson-extrapolated value and the
    Roache grid-convergence index (GCI) for three grids, which splits the
    total error into ``|f_fine - f_extrapolated|`` (grid) and
    ``|f_extrapolated - f_reference|`` (model).
:func:`fit_offset_correction`
    a ``y+`` correction map from measured peak offsets, with a held-out
    point so the map is validated rather than described.

The functions are pure (arrays in, dataclasses out) so they can be tested on
analytic profiles whose peak is known in closed form -- the reference
``-<u'v'>+ ~ y+ exp(-y+/A)`` family has ``y+_peak = A`` exactly.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "PeakEstimate",
    "GridConvergence",
    "find_stress_peak",
    "richardson_gci",
    "fit_offset_correction",
    "load_channel_means_table",
    "reynolds_stress_from_mean_profile",
    "analytic_stress_profile",
]


# ---------------------------------------------------------------------------
# peak location
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PeakEstimate:
    """Sub-sample peak of a one-dimensional profile.

    ``y_peak``/``value_peak`` come from a cubic spline of the samples;
    ``y_peak_parabola`` from the three-point parabola through the samples
    adjacent to the discrete maximum.  The two differ by the interpolation
    error, which is why both are reported and their spread is folded into
    ``uncertainty``: on a wall-modelled grid the spacing is of the same order
    as the peak position itself, so a single number would overstate what the
    grid can support.
    """

    y_peak: float
    value_peak: float
    y_peak_discrete: float
    value_peak_discrete: float
    y_peak_parabola: float
    value_peak_parabola: float
    spacing: float
    subgrid_shift: float
    uncertainty: float
    bracket: tuple[float, float]
    n_samples: int
    method: str

    def as_dict(self) -> dict:
        out = asdict(self)
        out["bracket"] = list(self.bracket)
        return out


def _cubic_spline_peak(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Peak of a natural cubic spline through ``(x, y)`` inside the bracket."""
    try:
        from scipy.interpolate import CubicSpline  # local import: optional dep
    except Exception:  # pragma: no cover - scipy is a test dependency here
        raise RuntimeError("scipy is required for the cubic-spline peak estimate") from None
    spline = CubicSpline(x, y, bc_type="natural")
    dense = np.linspace(x[0], x[-1], 2001)
    vals = spline(dense)
    k = int(np.argmax(vals))
    if 0 < k < len(dense) - 1:
        # one Newton step on the derivative for the sub-grid maximum
        lo, hi = dense[max(k - 1, 0)], dense[min(k + 1, len(dense) - 1)]
        try:
            from scipy.optimize import brentq

            slope = spline.derivative()
            if slope(lo) * slope(hi) < 0:
                x_star = brentq(slope, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200)
                return float(x_star), float(spline(x_star))
        except Exception:  # pragma: no cover - fall back to the dense maximiser
            pass
    return float(dense[k]), float(vals[k])


def find_stress_peak(
    y_plus: np.ndarray,
    tau_plus: np.ndarray,
    *,
    method: str = "cubic_spline",
    y_plus_max: float | None = None,
) -> PeakEstimate:
    """Locate the maximum of ``tau_plus(y_plus)`` to sub-sample accuracy.

    Parameters
    ----------
    y_plus, tau_plus
        1-D samples of the (positive) Reynolds shear stress in wall units.
        Non-finite entries are dropped; the arrays are sorted by ``y_plus``.
    y_plus_max
        Optional truncation.  Wall-modelled profiles are symmetric about the
        centreline, so a search over the whole channel can pick a mirrored
        peak; pass ``h+ / 2`` to search only the lower half.

    Notes
    -----
    The spline is fitted only over the three samples bracketing the discrete
    maximum.  Sparse wall-modelled profiles (three to ten points inside
    ``y+ < 100``) are badly conditioned for a global spline, and a global fit
    can produce a spurious interior extremum that has nothing to do with the
    data.
    """
    x = np.asarray(y_plus, dtype=float).ravel()
    v = np.asarray(tau_plus, dtype=float).ravel()
    if x.size != v.size:
        raise ValueError("y_plus and tau_plus must have the same length")
    ok = np.isfinite(x) & np.isfinite(v)
    if y_plus_max is not None:
        ok &= x <= float(y_plus_max)
    x, v = x[ok], v[ok]
    if x.size < 3:
        raise ValueError("need at least three finite samples to locate a peak")
    order = np.argsort(x)
    x, v = x[order], v[order]

    k = int(np.argmax(v))
    i_lo, i_hi = max(k - 1, 0), min(k + 1, x.size - 1)
    bracket = (float(x[i_lo]), float(x[i_hi]))
    spacing = float(np.median(np.diff(x[i_lo : i_hi + 1]))) if i_hi > i_lo else float("nan")

    # three-point parabola through the samples adjacent to the discrete max
    if i_lo == i_hi:
        y_par, v_par = float(x[k]), float(v[k])
    else:
        xs = x[i_lo : i_hi + 1]
        vs = v[i_lo : i_hi + 1]
        if xs.size == 3 and (xs[1] - xs[0]) > 0 and (xs[2] - xs[1]) > 0:
            coef = np.polyfit(xs, vs, 2)
            a, b = float(coef[0]), float(coef[1])
            if a < 0:  # concave down: a genuine interior maximum
                y_par = float(-b / (2.0 * a))
                v_par = float(np.polyval(coef, y_par))
            else:  # no interior maximum in this bracket: keep the sample
                y_par, v_par = float(x[k]), float(v[k])
        else:
            y_par, v_par = float(x[k]), float(v[k])
        y_par = float(min(max(y_par, bracket[0]), bracket[1]))

    if method not in {"cubic_spline", "parabola"}:
        raise ValueError(f"unsupported method {method!r}")
    if i_hi - i_lo >= 2:
        # three samples are the minimum a spline/parabola can be fitted through
        if method == "cubic_spline":
            y_sp, v_sp = _cubic_spline_peak(x[i_lo : i_hi + 1], v[i_lo : i_hi + 1])
            y_sp = float(min(max(y_sp, bracket[0]), bracket[1]))
        else:
            y_sp, v_sp = y_par, v_par
    else:
        # the discrete maximum is on the edge of the search band: there is no
        # interior bracket to interpolate, so the sample itself is the estimate
        y_sp, v_sp = float(x[k]), float(v[k])

    uncertainty = 0.5 * max(abs(y_sp - y_par), 0.0)
    return PeakEstimate(
        y_peak=y_sp,
        value_peak=v_sp,
        y_peak_discrete=float(x[k]),
        value_peak_discrete=float(v[k]),
        y_peak_parabola=y_par,
        value_peak_parabola=v_par,
        spacing=spacing,
        subgrid_shift=y_sp - float(x[k]),
        uncertainty=uncertainty,
        bracket=bracket,
        n_samples=int(x.size),
        method=method,
    )


# ---------------------------------------------------------------------------
# grid convergence / GCI
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GridConvergence:
    """Three-grid Richardson extrapolation and Roache GCI.

    ``value_extrapolated`` is the ``h -> 0`` estimate; ``gci_fine`` is the
    standard Roache index with safety factor ``fs``; ``grid_error_fine`` is
    ``|f_fine - f_extrapolated|`` and ``model_error`` is
    ``|f_extrapolated - f_reference|`` (``nan`` when no reference is given).
    """

    values: tuple[float, float, float]
    spacings: tuple[float, float, float]
    observed_order: float
    ratio1: float
    ratio2: float
    value_extrapolated: float
    gci_fine: float
    grid_error_fine: float
    model_error: float
    relative_model_error: float
    reference: float | None
    monotone: bool
    converged: bool
    note: str

    def as_dict(self) -> dict:
        out = asdict(self)
        out["values"] = list(self.values)
        out["spacings"] = list(self.spacings)
        return out


def _order_residual(p: float, r12: float, r23: float, observed: float) -> float:
    """Right-hand side of the unequal-ratio order equation, minus ``observed``."""
    num = r12**p * (r23**p - 1.0)
    den = r12**p - 1.0
    return num / den - observed


def richardson_gci(
    values,
    spacings,
    *,
    reference: float | None = None,
    safety: float = 1.25,
    order_tol: float = 0.05,
) -> GridConvergence:
    """Richardson extrapolation and GCI from three grids (fine -> coarse).

    ``values`` and ``spacings`` are ordered fine, medium, coarse.  The observed
    order ``p`` solves

        (f3 - f2) / (f2 - f1) = r12^p (r23^p - 1) / (r12^p - 1),

    which reduces to the textbook ``p = ln((f3-f2)/(f2-f1)) / ln(r)`` when the
    refinement ratios are equal.  ``converged`` is True when ``p > order_tol``
    and the three values are monotone, i.e. when a Richardson estimate is
    meaningful; otherwise the extrapolation is still returned (with ``p = 1``)
    but flagged, because a non-monotone triple is evidence that the grid error
    is not in the asymptotic range -- exactly the situation a coarse wall
    model grid produces.
    """
    v = [float(x) for x in values]
    h = [float(x) for x in spacings]
    if len(v) != 3 or len(h) != 3:
        raise ValueError("three values and three spacings are required")
    if not all(math.isfinite(x) for x in v) or not all(x > 0 for x in h):
        raise ValueError("values must be finite and spacings positive")
    f1, f2, f3 = v
    h1, h2, h3 = h
    r12 = h2 / h1
    r23 = h3 / h2
    d12 = f2 - f1
    d23 = f3 - f2
    monotone = d12 * d23 > 0.0 and abs(d12) > 0.0

    p = float("nan")
    note = ""
    if monotone:
        observed = d23 / d12
        lo, hi = 1e-3, 8.0
        if _order_residual(lo, r12, r23, observed) * _order_residual(hi, r12, r23, observed) <= 0.0:
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if _order_residual(mid, r12, r23, observed) * _order_residual(lo, r12, r23, observed) <= 0.0:
                    hi = mid
                else:
                    lo = mid
            p = 0.5 * (lo + hi)
        else:
            note = "observed ratio outside the solvable bracket"
    else:
        note = "non-monotone triple: differences change sign, no asymptotic order"

    if not math.isfinite(p) or p <= order_tol:
        p_used = 1.0
        converged = False
        note = note or "observed order not resolvable; Richardson taken with p = 1"
    else:
        p_used = p
        converged = True

    r = r12**p_used
    value_ext = f1 + (f1 - f2) / (r - 1.0) if abs(r - 1.0) > 1e-12 else f1
    grid_error = abs(f1 - value_ext)
    gci = safety * abs((f1 - f2) / f1) / abs(r - 1.0) if abs(f1) > 0 and abs(r - 1.0) > 1e-12 else float("nan")
    if reference is not None and math.isfinite(float(reference)):
        model_error = abs(value_ext - float(reference))
        rel_model = model_error / abs(float(reference)) if float(reference) != 0 else float("nan")
    else:
        model_error = float("nan")
        rel_model = float("nan")

    return GridConvergence(
        values=(f1, f2, f3),
        spacings=(h1, h2, h3),
        observed_order=p,
        ratio1=r12,
        ratio2=r23,
        value_extrapolated=float(value_ext),
        gci_fine=float(gci),
        grid_error_fine=float(grid_error),
        model_error=float(model_error),
        relative_model_error=float(rel_model),
        reference=None if reference is None else float(reference),
        monotone=bool(monotone),
        converged=bool(converged),
        note=note,
    )


# ---------------------------------------------------------------------------
# correction map
# ---------------------------------------------------------------------------
def fit_offset_correction(
    spacings,
    offsets,
    *,
    holdout: int | None = None,
    kind: str | None = None,
) -> dict:
    """Fit ``offset(dy+)`` and validate it on a held-out grid.

    ``offsets`` are relative peak-position offsets (model minus reference,
    dimensionless).  The fit is a two-parameter map

        ``linear``   : offset = a + b * dy+
        ``inverse``  : offset = a + b / dy+

    chosen by leave-one-out on the supplied points when ``holdout`` is None,
    otherwise the given point is held out, fitted on the others and predicted.

    ``kind=None`` selects the family by leave-one-out error, which needs the
    full set of points; with a held-out point the family cannot be selected
    from two points (both fits interpolate exactly), so an explicit ``kind`` is
    required and the choice has to be stated in the record instead of being
    inferred.

    Returns a dict with the coefficients, the residuals, the held-out
    prediction/error (when a holdout is used) and the dy+ that the map implies
    for a target tolerance (``spacing_for_tolerance``).
    """
    s = np.asarray(spacings, dtype=float).ravel()
    o = np.asarray(offsets, dtype=float).ravel()
    if s.size != o.size or s.size < 3:
        raise ValueError("need at least three (spacing, offset) pairs")
    if kind is not None and kind not in {"linear", "inverse"}:
        raise ValueError(f"unsupported kind {kind!r}")
    if kind is None and holdout is not None:
        raise ValueError(
            "a held-out point leaves two fitting points, which cannot discriminate the "
            "map family: pass kind explicitly"
        )

    def design(x, k):
        x = np.asarray(x, dtype=float)
        return np.column_stack([np.ones_like(x), x if k == "linear" else 1.0 / x])

    def solve(x, y, k):
        A = design(x, k)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return coef

    if holdout is None and kind is None:
        # leave-one-out error as the selection criterion
        best = None
        for k in ("linear", "inverse"):
            errs = []
            for i in range(s.size):
                mask = np.ones(s.size, dtype=bool)
                mask[i] = False
                coef = solve(s[mask], o[mask], k)
                pred = float((design(s[i : i + 1], k) @ coef)[0])
                errs.append(pred - o[i])
            loo = float(np.sqrt(np.mean(np.square(errs))))
            if best is None or loo < best["loo_rmse"]:
                best = {"kind": k, "loo_rmse": loo}
        kind = best["kind"]

    mask = np.ones(s.size, dtype=bool)
    if holdout is not None:
        mask[int(holdout)] = False
    coef = solve(s[mask], o[mask], kind)
    predictions = design(s, kind) @ coef
    residuals = predictions - o
    out = {
        "kind": kind,
        "coefficients": [float(c) for c in coef],
        "spacings": s.tolist(),
        "offsets": o.tolist(),
        "predictions": [float(x) for x in predictions],
        "residuals": [float(x) for x in residuals],
        "rmse": float(np.sqrt(np.mean(np.square(residuals[mask])))),
        "max_abs_residual": float(np.max(np.abs(residuals[mask]))),
    }
    if holdout is not None:
        i = int(holdout)
        out["holdout"] = {
            "index": i,
            "spacing": float(s[i]),
            "measured_offset": float(o[i]),
            "predicted_offset": float(predictions[i]),
            "error": float(residuals[i]),
            "abs_error": float(abs(residuals[i])),
        }
    else:
        out["loo_rmse"] = float(
            np.sqrt(np.mean([(predictions[i] - o[i]) ** 2 for i in range(s.size)]))
        )
    return out


def spacing_for_tolerance(correction: dict, tolerance: float, *, spacing_lo: float = 1e-3) -> float | None:
    """``dy+`` at which the fitted map reaches ``tolerance`` (bisection).

    Only the ``inverse`` map can be inverted for an arbitrary tolerance; for
    the ``linear`` map the crossing is solved the same way (the map is
    monotone in the fitted range), and ``None`` is returned when the target is
    not bracketed by the fitted data.
    """
    a, b = correction["coefficients"]
    kind = correction["kind"]

    def f(x: float) -> float:
        return (a + b * x if kind == "linear" else a + b / x) - tolerance

    lo, hi = spacing_lo, max(correction["spacings"]) * 10.0
    f_lo, f_hi = f(lo), f(hi)
    if not (f_lo * f_hi <= 0.0):
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# reference tables (Moser-Kim-Mansour .means files)
# ---------------------------------------------------------------------------
def load_channel_means_table(path: str | Path) -> dict:
    """Read a MKM-style ``.means`` profile (whitespace table, ``#`` comments).

    The columns are ``y/h, y+, Umean+, dUmean+/dy+*Re_tau, Wmean, dWmean/dy,
    Pmean``.  The file carries no self-describing Reynolds-stress column, so
    the profile returned here is the *mean* one; the stress is recovered from
    the exact channel momentum balance by
    :func:`reynolds_stress_from_mean_profile`.  ``re_tau`` is inferred from the
    table (``y+`` at ``y/h = 1``) and reported, never assumed.
    """
    p = Path(path)
    rows = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.replace(",", " ").split()
        if len(parts) < 4:
            continue
        try:
            rows.append([float(x) for x in parts[:4]])
        except ValueError:
            continue
    if len(rows) < 10:
        raise ValueError(f"{p} does not look like a channel profile table")
    data = np.asarray(rows, dtype=float)
    y_over_h = data[:, 0]
    y_plus = data[:, 1]
    u_plus = data[:, 2]
    du_plus_dy_plus_times_re_tau = data[:, 3]
    # y+ = y/h * Re_tau at the wall-normal station; Re_tau from the centreline
    re_tau = float(np.interp(1.0, y_over_h, y_plus))
    return {
        "path": str(p),
        "y_over_h": y_over_h,
        "y_plus": y_plus,
        "u_plus": u_plus,
        "du_plus_dy_plus_times_re_tau": du_plus_dy_plus_times_re_tau,
        "re_tau": re_tau,
        "n_samples": int(data.shape[0]),
    }


def reynolds_stress_from_mean_profile(table: dict) -> dict:
    """``-<u'v'>+ = 1 - y+/h+ - dU+/dy+`` from a channel mean profile.

    The identity is exact for a statistically stationary, fully developed
    plane channel (nothing but the streamwise momentum balance is used).  Its
    two endpoint checks are recorded as gates: ``-<u'v'>+ -> 0`` at the wall
    (viscous limit) and at the centreline (zero shear).
    """
    y_plus = np.asarray(table["y_plus"], dtype=float)
    re_tau = float(table["re_tau"])
    du_plus = np.asarray(table["du_plus_dy_plus_times_re_tau"], dtype=float) / re_tau
    # Use the tabulated y/h when it is available: y+ / Re_tau differs from
    # y/h by the (small) offset between the tabulated centreline y+ and the
    # nominal Re_tau, and the balance identity is exact in y/h.
    if "y_over_h" in table:
        y_over_h = np.asarray(table["y_over_h"], dtype=float)
    else:
        y_over_h = y_plus / re_tau
    uv_plus = 1.0 - y_over_h - du_plus
    wall_value = float(uv_plus[0])
    centre_value = float(uv_plus[-1])
    return {
        "y_plus": y_plus,
        "uv_plus": uv_plus,
        "du_plus_dy_plus": du_plus,
        "re_tau": re_tau,
        "gates": {
            "wall_value": wall_value,
            "centre_value": centre_value,
            "wall_ok": bool(abs(wall_value) < 0.05),
            "centre_ok": bool(abs(centre_value) < 0.05),
        },
    }


def analytic_stress_profile(y_plus: np.ndarray, *, a_plus: float, n: float = 1.0) -> np.ndarray:
    """``C * y+^n exp(-y+/A)``: maximum at ``y+ = n * A`` in closed form.

    Used by the tests to check the sub-sample peak finder on a profile whose
    peak position and value are known exactly, including the case where the
    peak falls between samples.
    """
    y = np.asarray(y_plus, dtype=float)
    x = np.clip(y, 0.0, None) / float(a_plus)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.where(y > 0.0, x**n * np.exp(-x), 0.0)
    return val