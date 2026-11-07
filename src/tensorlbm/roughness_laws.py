"""Closed-form rough-wall friction laws used to adjudicate the LBM results.

Three independent reference systems are implemented, because the campaign has
to answer three different questions:

1. **Is the computed roughness function right?**  :func:`nikuradse_delta_u_plus`
   is the fully rough asymptote of Nikuradse's (1933) sand-grain data,
   ``dU+ = (1/kappa) ln(ks+) - 3.5`` (slope ``1/kappa`` — acceptance criterion
   (c)), and :func:`cebeci_delta_u_plus` is the smooth-to-rough transition
   closure ``dU+ = (1/kappa) ln(1 + 0.3 ks+)`` used in k-omega/k-eps wall
   treatments (Cebeci & Bradshaw 1977; Knopp et al. 2006).  The two bracket the
   "literature band" of :func:`literature_band`; in the fully rough limit they
   differ by the additive constant ``(1/kappa) ln(0.3) = -2.94`` versus
   ``-3.5``, i.e. ~0.56 u_tau, which is the physically honest width of the
   band and matches the scatter of the classical sand-grain experiments.

2. **What is the friction velocity / Cf convention?**  :func:`ittc57_cf` is the
   1957 ITTC model-ship correlation line (flat-plate friction at the same Re),
   ``Cf = 2/U+^2`` is the exact LBM convention (momentum balance), and
   :func:`delta_cf_from_delta_u_plus` converts a measured roughness function
   into a friction increment **without** any additional empirical constant:
   ``dCf = 2/(U+_sm - dU+)^2 - 2/U+_sm^2``.

3. **How does the ship-scale roughness allowance compare?**  :func:`ittc78_allowance`
   is the ITTC-1978 correlation-allowance increment
   ``dCf = [105 (ks/L)^(1/3) - 0.64] x 1e-3`` for a hull of length ``L`` with
   equivalent sand-grain height ``ks`` (ITTC 1978 Performance Committee,
   reproduced in ITTC 7.5-02-03-01.4 and in Molland et al. 2017).  Its ``ks``
   is an *effective* (hydraulic) sand-grain height, which is exactly why the
   comparison at fixed geometry must report the ``ks_eff`` calibration too.

Everything here is pure scalar math so it can be unit-tested independently of
the solver.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = [
    "KAPPA",
    "B_SMOOTH",
    "SMOOTH_REGIME_KS_PLUS",
    "FULLY_ROUGH_KS_PLUS",
    "regime",
    "cebeci_delta_u_plus",
    "nikuradse_delta_u_plus",
    "literature_band",
    "equivalent_ks_plus",
    "u_plus_from_velocity",
    "cf_from_u_plus",
    "delta_cf_from_delta_u_plus",
    "ittc57_cf",
    "ittc78_allowance",
    "fit_log_law",
    "log_law_u_plus",
]

KAPPA: float = 0.41
B_SMOOTH: float = 5.0
#: Adams & Johnston (1984) regime boundaries used throughout the platform.
SMOOTH_REGIME_KS_PLUS: float = 2.25
FULLY_ROUGH_KS_PLUS: float = 90.0

#: Coefficient of the transition closure ``dU+ = (1/kappa) ln(1 + a ks+)``.
CEBECI_A: float = 0.3
#: Fully rough intercept: ``dU+ = (1/kappa) ln(ks+) + NIKURADSE_C``.
NIKURADSE_C: float = -3.5


def regime(ks_plus: float) -> str:
    """Return ``"smooth"``, ``"transitional"`` or ``"fully_rough"``."""
    ks_plus = float(ks_plus)
    if ks_plus < 0.0:
        raise ValueError(f"ks_plus must be non-negative, got {ks_plus!r}")
    if ks_plus < SMOOTH_REGIME_KS_PLUS:
        return "smooth"
    if ks_plus <= FULLY_ROUGH_KS_PLUS:
        return "transitional"
    return "fully_rough"


def cebeci_delta_u_plus(ks_plus: Sequence[float] | np.ndarray | float) -> np.ndarray:
    """Transition closure ``dU+ = (1/kappa) ln(1 + 0.3 ks+)``.

    Vanishes at ``ks+ = 0`` and approaches the Nikuradse slope ``1/kappa`` for
    large ``ks+`` with an intercept that is ``(1/kappa) ln(0.3) = -2.94``.
    """
    ks = np.asarray(ks_plus, dtype=float)
    if np.any(ks < 0.0):
        raise ValueError("ks_plus must be non-negative")
    return (1.0 / KAPPA) * np.log1p(CEBECI_A * ks)


def nikuradse_delta_u_plus(ks_plus: Sequence[float] | np.ndarray | float) -> np.ndarray:
    """Fully rough asymptote ``dU+ = (1/kappa) ln(ks+) - 3.5``.

    Defined for ``ks+ > 0``; for ``ks+ <= 0`` the function returns 0 (a smooth
    wall has no roughness shift).  Values below zero are clipped at zero,
    because a negative shift is unphysical (it would mean roughness *reduces*
    drag): inside the hydraulically smooth regime the shift is ``0``.
    """
    ks = np.asarray(ks_plus, dtype=float)
    if np.any(ks < 0.0):
        raise ValueError("ks_plus must be non-negative")
    out = np.zeros_like(ks)
    pos = ks > 0.0
    out[pos] = np.maximum((1.0 / KAPPA) * np.log(ks[pos]) + NIKURADSE_C, 0.0)
    return out


def literature_band(ks_plus: Sequence[float] | np.ndarray | float, tol_u: float = 0.5) -> dict[str, np.ndarray]:
    """Envelope of the two classical roughness-function closures.

    Returns a dict with ``lo``/``hi`` (band edges) and ``ref`` (their mean),
    widened by ``tol_u`` (default ``0.5 u_tau``, the acceptance criterion (a)
    tolerance).  The band is **not** a free fudge: ``lo`` is whichever of the
    two closures is smaller, ``hi`` the larger, both evaluated from their
    published constants.
    """
    ks = np.asarray(ks_plus, dtype=float)
    cb = cebeci_delta_u_plus(ks)
    nk = nikuradse_delta_u_plus(ks)
    lo = np.minimum(cb, nk) - float(tol_u)
    hi = np.maximum(cb, nk) + float(tol_u)
    return {"lo": lo, "hi": hi, "ref": 0.5 * (cb + nk), "cebeci": cb, "nikuradse": nk}


def equivalent_ks_plus(delta_u_plus: float) -> float:
    """Invert the Nikuradse fully-rough asymptote for the hydraulic ``ks+``.

    ``ks+_eff = exp(kappa (dU+ + 3.5))``.  This is the *definition* of the
    equivalent sand-grain height: the sand-grain size that would produce the
    measured roughness function.  It is the honest way to compare a resolved
    rib/gouge geometry against a sand-grain correlation, and the ratio
    ``ks_eff/ks_geom`` is reported as the morphology factor.
    """
    d = float(delta_u_plus)
    return float(math.exp(KAPPA * (d - NIKURADSE_C)))


def u_plus_from_velocity(velocity: float, u_tau: float) -> float:
    """``U+ = U/u_tau`` with input validation (both must be finite, u_tau > 0)."""
    if not (math.isfinite(velocity) and math.isfinite(u_tau)):
        raise ValueError("velocity and u_tau must be finite")
    if u_tau <= 0.0:
        raise ValueError(f"u_tau must be positive, got {u_tau!r}")
    return float(velocity) / float(u_tau)


def cf_from_u_plus(u_plus: float) -> float:
    """Skin-friction coefficient ``Cf = 2/U+^2`` (momentum-balance convention)."""
    u = float(u_plus)
    if not math.isfinite(u) or u <= 0.0:
        raise ValueError(f"U+ must be positive, got {u_plus!r}")
    return 2.0 / (u * u)


def delta_cf_from_delta_u_plus(u_plus_smooth: float, delta_u_plus: float) -> float:
    """``dCf = Cf_rough - Cf_smooth`` from a roughness function (Hama/Clauser).

    Both cases share the same friction velocity (matched ``Re_tau``), hence
    ``Cf = 2/U+^2`` gives ``dCf = 2/(U+_sm - dU+)^2 - 2/U+_sm^2``.  Raises if
    the shift would drive ``U+`` non-positive (a physically impossible input).
    """
    u_sm = float(u_plus_smooth)
    d = float(delta_u_plus)
    if u_sm <= 0.0:
        raise ValueError("U+_smooth must be positive")
    if d >= u_sm:
        raise ValueError("delta_u_plus must be smaller than U+_smooth")
    return cf_from_u_plus(u_sm - d) - cf_from_u_plus(u_sm)


def ittc57_cf(reynolds: float) -> float:
    """1957 ITTC model-ship correlation line ``0.075/(log10 Re - 2)^2``."""
    re = float(reynolds)
    if not math.isfinite(re) or re <= 1.0e2:
        raise ValueError(f"ITTC-57 needs Re > 100, got {reynolds!r}")
    return 0.075 / (math.log10(re) - 2.0) ** 2


def ittc78_allowance(ks_over_l: float | Sequence[float] | np.ndarray) -> np.ndarray:
    """ITTC-1978 roughness allowance ``dCf = [105 (ks/L)^(1/3) - 0.64] x 1e-3``.

    Negative values (which the raw polynomial returns for ``ks/L < 5.3e-7``)
    are clipped at zero: the allowance is defined as an increment and cannot be
    a drag reduction.
    """
    x = np.asarray(ks_over_l, dtype=float)
    if np.any(x < 0.0):
        raise ValueError("ks/L must be non-negative")
    return np.maximum((105.0 * np.cbrt(x) - 0.64) * 1e-3, 0.0)


def log_law_u_plus(y_plus: Sequence[float] | np.ndarray | float, b: float = B_SMOOTH, kappa: float = KAPPA) -> np.ndarray:
    """``U+ = (1/kappa) ln(y+) + b`` (valid for ``y+ > 0``)."""
    y = np.asarray(y_plus, dtype=float)
    if np.any(y <= 0.0):
        raise ValueError("y+ must be positive")
    return (1.0 / float(kappa)) * np.log(y) + float(b)


def fit_log_law(
    y_plus: Sequence[float] | np.ndarray,
    u_plus: Sequence[float] | np.ndarray,
    *,
    kappa: float = KAPPA,
    fit_kappa: bool = True,
) -> dict[str, float]:
    """Least-squares fit of the log law to a measured mean profile.

    With ``fit_kappa`` both ``(1/kappa, b)`` are fitted (useful as a solver
    health check); with ``fit_kappa=False`` the universal ``kappa`` is imposed
    and only the intercept ``b`` is measured, which is the definition used for
    the roughness function ``dU+ = B_smooth - b``.

    Returns ``{"b", "kappa", "slope", "rms", "n_points"}``.  Raises if fewer
    than two points are supplied or the inputs contain non-finite values.
    """
    y = np.asarray(y_plus, dtype=float).ravel()
    u = np.asarray(u_plus, dtype=float).ravel()
    if y.shape != u.shape:
        raise ValueError("y_plus and u_plus must have the same length")
    good = np.isfinite(y) & np.isfinite(u) & (y > 0.0)
    y, u = y[good], u[good]
    if y.size < 2:
        raise ValueError("need at least two valid points to fit the log law")
    ln_y = np.log(y)
    if fit_kappa:
        slope, intercept = np.polyfit(ln_y, u, 1)
        kappa_out = 1.0 / float(slope) if slope != 0.0 else float("nan")
        resid = u - (slope * ln_y + intercept)
    else:
        slope = 1.0 / float(kappa)
        intercept = float(np.mean(u - slope * ln_y))
        kappa_out = float(kappa)
        resid = u - (slope * ln_y + intercept)
    return {
        "b": float(intercept),
        "kappa": kappa_out,
        "slope": float(slope),
        "rms": float(np.sqrt(np.mean(resid**2))),
        "n_points": int(y.size),
    }