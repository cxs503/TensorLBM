"""Virk drag-reduction asymptotes for polymer solutions in turbulent pipe/channel flow.

Virk (1975, *AIChE J.* 21:625) reduced the drag-reduction phenomenology to two
friction-factor asymptotes in Prandtl-Karman coordinates
``x = log10(Re sqrt(f))``, ``y = 1/sqrt(f)``:

    Newtonian (Prandtl-von Karman):   y = 4.0  x - 0.4                  (N)
    maximum drag reduction (MDR):     y = 19.0 x - 32.4                 (MDR)

Polymer solutions lie *between* the two lines; Virk parametrised the family as
``y = (4 + delta) x - 0.4 - B`` and called ``Delta B`` the *intercept increment*
(drag-reduction increment), with the empirical maximum ``Delta B_max ~ 11.7``
reached on the MDR asymptote around ``Re ~ 10^4``.

Everything here is an *analytic envelope* evaluated in closed form: no closure
coefficient is fitted.  The Weissenberg dependence ``Delta B(Wi)`` is a
**declared model** (:func:`delta_B_polymer_model`), a saturating interpolation
between the two exact asymptotes -- it is *not* a DNS measurement.  See
``docs/viscoelastic_verification.md`` for the honest status of the turbulent
branch.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "NEWTONIAN_SLOPE",
    "NEWTONIAN_INTERCEPT",
    "MDR_SLOPE",
    "MDR_INTERCEPT",
    "DELTA_B_MDR",
    "WI_MDR",
    "friction_newtonian",
    "friction_mdr",
    "re_sqrt_f",
    "delta_B_from_friction",
    "friction_from_delta_B",
    "drag_reduction",
    "delta_B_polymer_model",
]

NEWTONIAN_SLOPE = 4.0
NEWTONIAN_INTERCEPT = 0.4
MDR_SLOPE = 19.0
MDR_INTERCEPT = 32.4
#: Virk's MDR intercept increment at the reference condition Re ~ 1e4.
DELTA_B_MDR = 11.7
#: Weissenberg number at which the declared Delta B(Wi) model reaches 0.99 MDR.
WI_MDR = 10.0


def _newton_solve(fun, y0=10.0, tol=1e-14, itmax=300):
    """Scalar Newton iteration with numerical Jacobian; vectorised by caller."""
    y = float(y0)
    for _ in range(itmax):
        f0 = fun(y)
        if abs(f0) < tol:
            return y
        h = 1e-7 * max(1.0, abs(y))
        d = (fun(y + h) - fun(y - h)) / (2.0 * h)
        if not np.isfinite(d) or d == 0.0:
            return y
        step = f0 / d
        y_new = y - step
        if y_new <= 1e-8:
            return 1e-8
        y = y_new
        if abs(step) < 1e-15 * max(1.0, abs(y)):
            return y
    return y


def friction_newtonian(re):
    """Prandtl-von Karman friction factor: solves 1/sqrt(f) = 4 log10(Re sqrt f) - 0.4."""
    re = np.asarray(re, dtype=float)
    out = np.empty(re.shape, dtype=float)
    for k, r in enumerate(re.ravel()):
        if not np.isfinite(r) or r <= 0:
            out.ravel()[k] = np.nan
            continue
        y = _newton_solve(lambda y, _r=r: NEWTONIAN_SLOPE * np.log10(_r / y)
                          - NEWTONIAN_INTERCEPT - y, y0=8.0)
        out.ravel()[k] = 1.0 / y ** 2
    return out


def friction_mdr(re):
    """Virk MDR friction factor: solves 1/sqrt(f) = 19 log10(Re sqrt f) - 32.4."""
    re = np.asarray(re, dtype=float)
    out = np.empty(re.shape, dtype=float)
    for k, r in enumerate(re.ravel()):
        if not np.isfinite(r) or r <= 0:
            out.ravel()[k] = np.nan
            continue
        y = _newton_solve(lambda y, _r=r: MDR_SLOPE * np.log10(_r / y)
                          - MDR_INTERCEPT - y, y0=25.0)
        out.ravel()[k] = 1.0 / y ** 2
    return out


def re_sqrt_f(re, f):
    return np.asarray(re, float) * np.sqrt(np.asarray(f, float))


def delta_B_from_friction(re, f):
    """Virk intercept increment of a measured ``f(Re)``: ``y - (4x - 0.4)``."""
    y = 1.0 / np.sqrt(np.asarray(f, float))
    x = np.log10(re_sqrt_f(re, f))
    return y - (NEWTONIAN_SLOPE * x - NEWTONIAN_INTERCEPT)


def friction_from_delta_B(re, delta_B, y0=10.0):
    """Invert :func:`delta_B_from_friction`: friction factor for a given Delta B."""
    rr, dd = np.broadcast_arrays(np.asarray(re, float), np.asarray(delta_B, float))
    out = np.empty(rr.shape, dtype=float)
    for k, (r, d) in enumerate(zip(rr.ravel(), dd.ravel())):
        y = _newton_solve(lambda y, _r=r, _d=d: y
                          - (NEWTONIAN_SLOPE * np.log10(_r / y) - NEWTONIAN_INTERCEPT)
                          - _d, y0=y0)
        out.ravel()[k] = 1.0 / y ** 2
    return out


def drag_reduction(re, f_polymer):
    """Drag reduction percentage ``DR = 100 (1 - f_polymer / f_newtonian)``."""
    f_n = friction_newtonian(re)
    return 100.0 * (1.0 - np.asarray(f_polymer, float) / f_n)


def delta_B_polymer_model(wi, delta_B_max=DELTA_B_MDR, wi_mdr=WI_MDR):
    """**Declared model** of the drag-reduction increment ``Delta B(Wi)``.

    Saturating law between the two exact asymptotes::

        Delta B(Wi) = Delta_B_max * (1 - exp(-Wi / Wi_MDR))

    Properties (each one tested): ``Delta B(0) = 0`` (Newtonian limit),
    monotone increasing in ``Wi``, ``Delta B < Delta_B_max`` for all finite
    ``Wi``, and ``-> Delta_B_max`` as ``Wi -> inf`` (Virk MDR bound).  It is a
    phenomenological envelope *consistent with* Virk's asymptotes, not a
    simulation result.
    """
    wi = np.asarray(wi, dtype=float)
    return delta_B_max * (1.0 - np.exp(-np.maximum(wi, 0.0) / wi_mdr))