"""Modified log-law closure for microbubble drag reduction (IN62-N04).

Two independent, *derived* mechanisms are combined here.  No constant in this
module is fitted to a target drag-reduction value; the only numbers that enter
are textbook ones (``KAPPA=0.41``, ``B=5.2``, sublayer edge ``y0+=11``).

Mechanism M2 -- density-weighted mixing length (turbulence modulation)
---------------------------------------------------------------------
Let ``alpha`` be the void fraction in the buffer layer.  Microbubbles carry
negligible momentum compared with water (``rho_g/rho_l ~ 1e-3``), so the
Reynolds shear stress is carried by the liquid mass flux only:

    tau_t = rho_m <u'v'> = -rho_l (1-alpha) l^2 (dU/dy)^2 ,   l = kappa y .

In the wall layer ``tau ~ tau_w``; with ``u_tau = sqrt(tau_w/rho_l)`` this gives

    dU+/dy+ = 1 / (kappa_eff y+) ,      kappa_eff = kappa sqrt(1 - alpha) .

So the bubbly log law keeps the Prandtl form but with a smaller slope
coefficient.  The intercept follows from *continuity of the profile at the
viscous-sublayer edge* ``y0+ = 11`` (bubbles of ``d+ >~ 5`` do not enter the
sublayer, so the sublayer is unaffected):

    (1/kappa_eff) ln(y0+) + B_eff = (1/kappa) ln(y0+) + B .

Mechanism M1 -- wall area coverage (slip at the plate)
-----------------------------------------------------
A wall void fraction ``alpha_w`` means a fraction ``alpha_w`` of the plate area
is in contact with gas rather than liquid.  With mobile interfaces those
patches transmit negligible shear, so

    DR_cov = alpha_w .

The equivalent slip length ``b`` of such patches is obtained from the resolved
laminar two-layer LBM run in :mod:`tensorlbm.microbubble.lbm` and enters the log
law as ``u+ = (1/kappa_eff) ln(y+ + b+) + B_eff``.

Drag law (matched outer velocity)
---------------------------------
For a plate at the same free-stream speed and the same boundary-layer
thickness, the outer velocity at a wall distance ``y_ref`` is the same in both
runs:

    clean :  A  = (1/kappa) ln(y_ref+) + B ,
    bubbly:  A/r = (1/kappa_eff) ln(r (y_ref+ + b+_clean)) + B_eff ,

with ``r = u_tau'/u_tau``.  ``A/r`` is strictly decreasing in ``r`` and the
right-hand side is increasing, hence the root is unique (bisection below).  The
friction coefficient and drag reduction follow from the constant-pressure-
gradient / matched-speed identity ``Cf'/Cf = r^2``:

    DR = 1 - (1 - alpha_w) r^2 ,    Cf(alpha_w) = Cf_clean (1 - alpha_w) r^2 .

The two mechanisms compose: the covered wall area is switched off, and the
uncovered area carries the reduced-slope log law.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Textbook closure constants (Prandtl / von Karman), not fitted here.
KAPPA = 0.41
B_INTERCEPT = 5.2
Y0_PLUS = 11.0
A_PLUS = 26.0

#: White's turbulent flat-plate skin-friction correlation ``Cf(Re_theta)``,
#: used only as an *external* anchor for the alpha = 0 limiting case.
WHITE_CF_COEFF = 0.246
WHITE_CF_EXP = -0.268


@dataclass(frozen=True)
class LogLawParams:
    """Closure parameters (all dimensionless)."""

    kappa: float = KAPPA
    b_intercept: float = B_INTERCEPT
    y0_plus: float = Y0_PLUS
    #: wall distance (clean wall units) at which the outer velocity is matched
    y_ref_plus: float = 100.0

    def kappa_eff(self, alpha: float) -> float:
        """``kappa sqrt(1-alpha)`` (density-weighted mixing length)."""
        a = min(max(float(alpha), 0.0), 1.0)
        return self.kappa * math.sqrt(1.0 - a)

    def b_eff(self, alpha: float) -> float:
        """Intercept from sublayer continuity (see module docstring)."""
        ke = self.kappa_eff(alpha)
        return self.b_intercept + (1.0 / self.kappa - 1.0 / ke) * math.log(self.y0_plus)


_DEFAULT = LogLawParams()


def kappa_eff(alpha: float, p: LogLawParams | None = None) -> float:
    """Effective von Karman constant at void fraction ``alpha``."""
    return (p or _DEFAULT).kappa_eff(alpha)


def b_eff(alpha: float, p: LogLawParams | None = None) -> float:
    """Effective log-law intercept at void fraction ``alpha``."""
    return (p or _DEFAULT).b_eff(alpha)


def u_plus(y_plus, alpha: float = 0.0, slip_plus: float = 0.0,
           p: LogLawParams | None = None):
    """Modified log law ``u+ = (1/kappa_eff) ln(y+ + b+) + B_eff``."""
    p = p or _DEFAULT
    ke = p.kappa_eff(alpha)
    y = np.asarray(y_plus, dtype=float) + float(slip_plus)
    y = np.maximum(y, 1e-12)
    return np.log(y) / ke + p.b_eff(alpha)


def solve_r(ke: float, be: float, slip_plus: float = 0.0,
            y_ref_plus: float = 100.0, tol: float = 1e-12) -> float:
    """Root ``r = u_tau'/u_tau`` of the matched-outer-velocity relation.

    ``ke`` and ``be`` are the effective log-law constants of the bubbly layer;
    ``slip_plus`` is the slip length in clean wall units.
    """
    A = math.log(y_ref_plus) / KAPPA + B_INTERCEPT

    def residual(r: float) -> float:
        arg = r * (y_ref_plus + float(slip_plus))
        if arg <= 0.0 or r <= 0.0:
            return 1e30
        return A / r - (math.log(arg) / ke + be)

    lo, hi = 1e-6, 50.0
    if residual(lo) <= 0.0:
        return lo
    if residual(hi) >= 0.0:
        return hi
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, abs(hi)):
            break
    return 0.5 * (lo + hi)


def u_tau_ratio(alpha: float, slip_plus: float = 0.0,
                p: LogLawParams | None = None, tol: float = 1e-12) -> float:
    """Friction-velocity ratio ``r = u_tau'/u_tau`` at matched outer velocity.

    ``slip_plus`` is the slip length of the bubbly wall in *clean* wall units,
    ``b+ = b u_tau/nu``.  The bubbly wall units are ``r`` times larger, so the
    bubbly argument of the logarithm is ``r (y_ref+ + slip_plus)``.
    """
    p = p or _DEFAULT
    return solve_r(p.kappa_eff(alpha), p.b_eff(alpha), slip_plus,
                   p.y_ref_plus, tol)


def drag_reduction_loglaw(alpha: float, slip_plus: float = 0.0,
                          p: LogLawParams | None = None) -> float:
    """M2 only: ``DR = 1 - r^2`` (turbulence modulation, no area coverage)."""
    r = u_tau_ratio(alpha, slip_plus, p)
    return max(0.0, min(1.0, 1.0 - r * r))


def coverage_drag_reduction(alpha_w: float) -> float:
    """M1 only: gas-covered wall area ``DR_cov = alpha_w``."""
    return max(0.0, min(1.0, float(alpha_w)))


def drag_reduction(alpha_w: float, alpha_buffer: float | None = None,
                   slip_plus: float = 0.0,
                   p: LogLawParams | None = None) -> float:
    """Total drag reduction (M1 and M2 composed multiplicatively).

    ``alpha_buffer`` is the void fraction used for the mixing-length reduction
    (defaults to ``alpha_w``, i.e. the conservative uniform-layer assumption).
    """
    p = p or _DEFAULT
    a_buf = alpha_w if alpha_buffer is None else float(alpha_buffer)
    r = u_tau_ratio(a_buf, slip_plus, p)
    cov = coverage_drag_reduction(alpha_w)
    return max(0.0, min(1.0, 1.0 - (1.0 - cov) * r * r))


def cf_clean_correlation(re_theta: float) -> float:
    """White's turbulent flat-plate correlation ``Cf = 0.246 Re_theta^-0.268``."""
    return WHITE_CF_COEFF * float(re_theta) ** WHITE_CF_EXP


def cf_analytic(alpha_w: float, re_theta: float = 2000.0,
                alpha_buffer: float | None = None, slip_plus: float = 0.0,
                p: LogLawParams | None = None) -> float:
    """``Cf(alpha_w)`` from the modified log law and White's clean anchor."""
    dr = drag_reduction(alpha_w, alpha_buffer, slip_plus, p)
    return cf_clean_correlation(re_theta) * (1.0 - dr)


def dr_table(alpha_grid, re_theta: float = 2000.0,
             alpha_buffer_of=None, slip_plus: float = 0.0) -> dict:
    """Tabulate ``r``, ``DR`` and ``Cf`` over a wall-void-fraction grid."""
    alphas = np.asarray(list(alpha_grid), dtype=float)
    rows = []
    for a in alphas:
        a_buf = a if alpha_buffer_of is None else float(alpha_buffer_of(float(a)))
        r = u_tau_ratio(a_buf, slip_plus)
        dr = drag_reduction(float(a), a_buf, slip_plus)
        rows.append({
            "alpha_w": float(a),
            "alpha_buffer": float(a_buf),
            "kappa_eff": kappa_eff(a_buf),
            "u_tau_ratio": r,
            "DR": dr,
            "DR_loglaw_only": max(0.0, 1.0 - r * r),
            "Cf": cf_clean_correlation(re_theta) * (1.0 - dr),
        })
    return {"re_theta": float(re_theta), "rows": rows}

# ---------------------------------------------------------------------------
# profile-aware (log-layer averaged) route
# ---------------------------------------------------------------------------


def log_layer_average(alpha_of_y, u_tau: float, nu: float, delta: float,
                      p: LogLawParams | None = None,
                      y1_plus: float = 30.0, y2_frac: float = 0.15,
                      n: int = 96) -> dict:
    """Effective ``(kappa_eff, B_eff)`` of the whole log region.

    The log law of a layer whose void fraction varies with ``y`` is matched in
    the sense of the logarithmic average of the slope:

        1/kappa_eff = (1/ln(y2/y1)) int_{y1}^{y2} (1/kappa_eff(alpha(y))) dln y ,

    ``y1+ = 30`` (bottom of the log region) and ``y2 = 0.15 delta`` (top, where
    the wake starts).  ``B_eff`` then follows from sublayer continuity exactly
    as in the uniform case.
    """
    p = p or _DEFAULT
    y1 = y1_plus * nu / u_tau
    y2 = max(y2_frac * delta, 2.0 * y1)
    ys = np.geomspace(y1, y2, n)
    inv_ke = np.array([1.0 / p.kappa_eff(alpha_of_y(float(y))) for y in ys])
    w = 1.0 / ys
    inv_ke_eff = float(np.trapezoid(inv_ke * w, ys) / np.trapezoid(w, ys))
    ke_eff = 1.0 / inv_ke_eff
    be_eff = p.b_intercept + (1.0 / p.kappa - 1.0 / ke_eff) * math.log(p.y0_plus)
    return {"kappa_eff": ke_eff, "B_eff": be_eff,
            "y1": y1, "y2": y2, "n": n}


def drag_reduction_layer(alpha_w: float, alpha_of_y, u_tau: float, nu: float,
                         delta: float, slip_plus: float = 0.0,
                         p: LogLawParams | None = None) -> dict:
    """``DR`` and ``Cf`` ratio from the profile-aware log-law route."""
    p = p or _DEFAULT
    avg = log_layer_average(alpha_of_y, u_tau, nu, delta, p)
    r = solve_r(avg["kappa_eff"], avg["B_eff"], slip_plus, p.y_ref_plus)
    dr_m2 = max(0.0, 1.0 - r * r)
    dr = 1.0 - (1.0 - coverage_drag_reduction(alpha_w)) * r * r
    return {"u_tau_ratio": r, "DR_loglaw": dr_m2, "DR": dr,
            "Cf_ratio": (1.0 - coverage_drag_reduction(alpha_w)) * r * r,
            **avg}
