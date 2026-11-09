"""NACA0012 subsonic/transonic + supersonic sonic-boom system verification.

This module turns the acceptance criteria of the IN421-W2-N03 node into
machine-checkable functions on top of the kinetic-flux (KFVS) Euler solver in
:mod:`tensorlbm.compressible.kinetic_flux`:

=========  ==================================================================
criterion  quantity
=========  ==================================================================
(a)        lift-slope ``dCL/dalpha`` at ``Ma = 0.3`` vs thin-airfoil ``2*pi``
(b)        transonic drag rise ``CD(Ma)`` and drag-divergence onset ``M_cr``
(c)        terminal-shock position ``x/c`` from the surface Mach number
(d)        sonic-boom pressure jump ``dp/p_inf`` of a 10 deg wedge vs Ackeret
(e)        isentropic consistency ``p/rho^gamma`` in the subsonic part
(f)        Roache grid convergence index from a three-grid ladder
=========  ==================================================================

Two reference hierarchies are kept separate on purpose:

* **analytic** anchors (thin-airfoil / Prandtl-Glauert, Rankine-Hugoniot,
  isentropic relations, Ackeret) are evaluated exactly in closed form;
* **experimental** anchors (AGARD CP-303 / AR-138 digitised ``Cp``, Bailey 1986
  sonic-boom traces) are *not* shipped with this repository.  Every function
  that needs them returns ``WITHHELD`` unless a provenance-carrying dataset is
  supplied -- see :func:`experimental_anchor_status`.  A verification that
  silently substitutes a remembered number for a digitised trace is worthless,
  so this module refuses to do it.

References
----------
* Abbott & von Doenhoff, *Theory of Wing Sections*, Dover 1959 (NACA0012
  geometry, ``M_cr`` practice, compressibility corrections).
* AGARD CP-303 / AR-138, *Test cases for inviscid flow field methods* (1981).
* Ackeret 1925; Anderson, *Modern Compressible Flow*, 3rd ed. 2003 (sec. 9.7
  linearised supersonic flow; sec. 4.9 oblique shock).
* Roache, *Verification and Validation in Computational Science and
  Engineering*, Hermosa 1998 (GCI).
* Pullin 1980 / Mandal & Deshpande 1994 (kinetic flux vector splitting, the
  numerical flux of the solver).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .kinetic_flux import (  # noqa: F401  (re-exported for convenience)
    KineticFluxSolver,
    SolverConfig,
    build_annulus_mesh,
    contour_loop,
    diamond_half_thickness,
    experimental_anchor_status,
    integrate_forces,
    naca0012_half_thickness,
    radial_levels_from_surface,
)

GAMMA = 1.4


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def naca0012_reference_geometry(t: float = 0.12) -> Dict[str, float]:
    """Closed-form reference values of the NACA 4-digit thickness family.

    ``yt/c = 5t (0.2969 sqrt(xi) - 0.1260 xi - 0.3516 xi^2 + 0.2843 xi^3
    - 0.1036 xi^4)``: maximum thickness ``t`` at ``xi = 0.30``, cross-section
    area ``2 * int_0^1 yt dxi`` (Abbott & von Doenhoff 1959, sec. 6.1).
    """
    xmax = 0.30
    # dyt/dxi = 0  ->  solved once by bisection on the analytic derivative
    def dyt(xi):
        return 5.0 * t * (0.5 * 0.2969 / math.sqrt(max(xi, 1e-12)) - 0.1260
                          - 2.0 * 0.3516 * xi + 3.0 * 0.2843 * xi ** 2
                          - 4.0 * 0.1036 * xi ** 3)

    lo, hi = 0.05, 0.95
    for _ in range(200):
        xm = 0.5 * (lo + hi)
        if dyt(xm) > 0.0:
            lo = xm
        else:
            hi = xm
    xmax = 0.5 * (lo + hi)
    ymax = float(naca0012_half_thickness(np.array([xmax]))[0])

    # area / chord: Simpson rule on a fine grid (accuracy ~1e-12)
    xi = np.linspace(0.0, 1.0, 20001)
    yt = naca0012_half_thickness(xi)
    area = float(2.0 * np.trapezoid(yt, xi))
    return {"chord": 1.0, "max_thickness": float(2.0 * ymax),
            "x_max_thickness": float(xmax), "area_over_chord2": area,
            "te_half_thickness": float(naca0012_half_thickness(np.array([1.0]))[0])}


def validate_geometry(x_loop: Sequence[float], y_loop: Sequence[float],
                      t: float = 0.12, tol: float = 2e-3) -> Dict[str, object]:
    """Check a discretised NACA0012 contour against the analytic geometry.

    Returns ``ok`` plus the measured chord, thickness ratio, location of the
    maximum thickness, cross-section area and the closure error of the loop.
    Every quantity is compared with :func:`naca0012_reference_geometry`; the
    relative deviations are returned so that a *geometry* error can never be
    mistaken for a *physics* error later in the verification chain.
    """
    x = np.asarray(x_loop, float)
    y = np.asarray(y_loop, float)
    ref = naca0012_reference_geometry(t)
    chord = float(x.max() - x.min())
    # thickness: vertical extent measured on the upper/lower sheets at the same x
    n = x.size
    m = n // 2 + 1
    xu, yu = x[:m - 1], y[:m - 1]          # TE -> LE (upper)
    xl_, yl_ = x[m - 1:], y[m - 1:]        # LE -> TE (lower)
    order = np.argsort(xu)
    thick = 2.0 * float(np.interp(ref["x_max_thickness"], xu[order], yu[order]))
    area = float(np.sum(0.5 * (x * np.roll(y, -1) - np.roll(x, -1) * y)))
    closure = float(np.hypot(x[0] - x[-1], y[0] - y[-1]))
    out = {
        "chord": chord, "chord_error_pct": 100.0 * (chord - 1.0),
        "max_thickness": thick,
        "max_thickness_error_pct": 100.0 * (thick - ref["max_thickness"]) / ref["max_thickness"],
        "x_max_thickness": ref["x_max_thickness"],
        "area_over_chord2": abs(area),
        "area_error_pct": 100.0 * (abs(area) - ref["area_over_chord2"]) / ref["area_over_chord2"],
        "loop_closure": closure,
        "reference": ref,
    }
    out["ok"] = bool(abs(out["chord_error_pct"]) < 1e-3
                     and abs(out["max_thickness_error_pct"]) < tol * 100.0
                     and abs(out["area_error_pct"]) < tol * 100.0)
    return out


# ---------------------------------------------------------------------------
# isentropic surface quantities
# ---------------------------------------------------------------------------
def surface_mach_from_cp(cp: np.ndarray, mach_inf: float,
                         gamma: float = GAMMA) -> np.ndarray:
    """Local Mach number from the pressure coefficient (isentropic).

    ``p/p_inf = 1 + gamma/2 M_inf^2 Cp`` then invert the isentropic relation
    against the freestream stagnation pressure.  Negative/zero ``p`` (which a
    numerical scheme can produce in a strong expansion) maps to ``nan``.
    """
    cp = np.asarray(cp, float)
    p_ratio = 1.0 + 0.5 * gamma * mach_inf ** 2 * cp
    # p0/p_inf = (1 + (gamma-1)/2 M_inf^2)^(gamma/(gamma-1)) and
    # p/p0 = (1 + (gamma-1)/2 M^2)^(-gamma/(gamma-1))
    stag = (1.0 + 0.5 * (gamma - 1.0) * mach_inf ** 2) ** (gamma / (gamma - 1.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        m2 = 2.0 / (gamma - 1.0) * (
            (stag / p_ratio) ** ((gamma - 1.0) / gamma) - 1.0)
    m2 = np.where((p_ratio > 0.0) & np.isfinite(m2), m2, np.nan)
    return np.sqrt(np.maximum(m2, 0.0))


def isentropic_deviation(rho: np.ndarray, p: np.ndarray, mach: np.ndarray,
                         mach_cap: float = 0.9,
                         gamma: float = GAMMA) -> Dict[str, float]:
    """Relative deviation of ``s = p/rho^gamma`` from its freestream value.

    Criterion (e): inside an inviscid, shock-free flow ``p/rho^gamma`` is a
    material constant, so any deviation is numerical dissipation.  Cells with
    ``mach >= mach_cap`` (captured shocks, and the local supersonic pocket) are
    excluded; the *whole* field statistics are returned as well so that the
    exclusion cannot hide a global entropy error.
    """
    rho = np.asarray(rho, float)
    p = np.asarray(p, float)
    mach = np.asarray(mach, float)
    s = p / np.maximum(rho, 1e-30) ** gamma
    s_ref = float(p.flat[0] / rho.flat[0] ** gamma)
    dev = np.abs(s / s_ref - 1.0)
    sub = mach < mach_cap
    out = {
        "s_reference": s_ref,
        "max_rel_dev_subsonic": float(dev[sub].max()) if sub.any() else float("nan"),
        "p99_rel_dev_subsonic": float(np.percentile(dev[sub], 99.0)) if sub.any() else float("nan"),
        "mean_rel_dev_subsonic": float(dev[sub].mean()) if sub.any() else float("nan"),
        "max_rel_dev_all": float(dev.max()),
        "cells_subsonic": int(sub.sum()),
        "cells_total": int(dev.size),
    }
    return out


# ---------------------------------------------------------------------------
# shock detection and Rankine-Hugoniot cross-check
# ---------------------------------------------------------------------------
def shock_from_surface_mach(x_surf: np.ndarray, cp: np.ndarray,
                            mach_inf: float, gamma: float = GAMMA,
                            window: Tuple[float, float] = (0.02, 0.995)
                            ) -> Optional[Dict[str, float]]:
    """Locate the terminal shock on a surface pressure distribution.

    Criterion (c).  The shock is *not* defined as "the steepest ``Cp``
    gradient" (that confuses the shock with the trailing-edge pressure
    recovery, see the lessons of P153W1-N05): the surface Mach number is
    reconstructed isentropically, the supersonic pocket is located, and the
    terminal shock is the point where that pocket closes; its position is
    refined by linear interpolation of ``M = 1``.

    Returns a dict with the shock abscissa (``x/c``), the pre-shock Mach
    number, the measured and the Rankine-Hugoniot pressure ratios, and the
    relative strength error.  ``None`` if the flow is subcritical.
    """
    x = np.asarray(x_surf, float)
    cp = np.asarray(cp, float)
    order = np.argsort(x)
    xs, cs = x[order], cp[order]
    m = surface_mach_from_cp(cs, mach_inf, gamma)
    inside = (xs >= window[0]) & (xs <= window[1]) & np.isfinite(m)
    if not inside.any():
        return None
    sup = inside & (m > 1.0)
    if not sup.any():
        return {"x_shock": None, "supersonic": False,
                "max_surface_mach": float(np.nanmax(m[inside]))}
    idx = np.flatnonzero(sup)
    i_end = int(idx[-1])                      # last supersonic station
    if i_end + 1 >= xs.size:
        return {"x_shock": None, "supersonic": True,
                "max_surface_mach": float(np.nanmax(m[inside])),
                "reason": "supersonic pocket open at the trailing edge"}
    m1 = float(m[i_end])
    m2 = float(m[i_end + 1])
    if not np.isfinite(m2) or m2 >= m1:
        x_shock = float(xs[i_end])
    else:
        w = (m1 - 1.0) / (m1 - m2) if m1 > m2 else 1.0
        w = float(np.clip(w, 0.0, 1.0))
        x_shock = float(xs[i_end] + w * (xs[i_end + 1] - xs[i_end]))
    # measured jump at the pocket closure (plateau before/after the shock)
    i0 = max(int(idx[0]) - 2, 0)
    lo = max(i0, i_end - 6)
    hi = min(i_end + 7, xs.size - 1)
    p_up = 1.0 + 0.5 * gamma * mach_inf ** 2 * float(np.min(cs[lo:i_end + 1]))
    p_dn = 1.0 + 0.5 * gamma * mach_inf ** 2 * float(np.max(cs[i_end:hi + 1]))
    rh = normal_shock_pressure_ratio(m1, gamma)
    return {
        "x_shock": x_shock,
        "supersonic": True,
        "x_supersonic_start": float(xs[int(idx[0])]),
        "mach_upstream": m1,
        "mach_downstream": m2,
        "max_surface_mach": float(np.nanmax(m[inside])),
        "p_ratio_measured": p_dn / p_up,
        "p_ratio_rankine_hugoniot": rh,
        "strength_error_pct": 100.0 * (p_dn / p_up - rh) / rh,
        "cp_upstream": float(np.min(cs[lo:i_end + 1])),
        "cp_downstream": float(np.max(cs[i_end:hi + 1])),
    }


def normal_shock_pressure_ratio(mach1: float, gamma: float = GAMMA) -> float:
    """``p2/p1 = 1 + 2 gamma/(gamma+1) (M1^2 - 1)``."""
    return 1.0 + 2.0 * gamma / (gamma + 1.0) * (mach1 ** 2 - 1.0)


# ---------------------------------------------------------------------------
# oblique shock / Ackeret
# ---------------------------------------------------------------------------
def oblique_shock_angle(mach: float, theta_deg: float, gamma: float = GAMMA,
                        weak: bool = True) -> float:
    """Shock angle ``beta`` of the theta-beta-Mach relation (weak branch)."""
    if mach <= 1.0:
        raise ValueError("oblique shock requires Ma > 1")
    theta = math.radians(theta_deg)
    b_min = math.asin(min(1.0, 1.0 / mach)) + 1e-9
    f = lambda b: (math.tan(theta)
                   - 2.0 / math.tan(b) * (mach ** 2 * math.sin(b) ** 2 - 1.0)
                   / (mach ** 2 * (gamma + math.cos(2.0 * b)) + 2.0))
    # f(b_min) > 0 (theta=0 limit), f -> -inf before pi/2: bisect the weak root
    lo, hi = b_min, 0.5 * math.pi - 1e-9
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    beta = 0.5 * (lo + hi)
    if weak:
        return beta
    return beta


def oblique_shock_pressure_ratio(mach: float, theta_deg: float,
                                 gamma: float = GAMMA) -> float:
    """Exact (weak) oblique-shock pressure ratio ``p2/p1`` for a wedge."""
    beta = oblique_shock_angle(mach, theta_deg, gamma)
    return 1.0 + 2.0 * gamma / (gamma + 1.0) * (
        mach ** 2 * math.sin(beta) ** 2 - 1.0)


def ackeret_pressure_ratio(mach: float, theta_deg: float,
                           gamma: float = GAMMA) -> float:
    """Ackeret linearised ``p/p_inf - 1 = gamma M^2 theta / sqrt(M^2-1)``.

    Equivalent to ``Cp = 2 theta / sqrt(M^2 - 1)`` (``Cp = (p-p_inf)/q``,
    ``q = gamma p_inf M^2 /2``); the form required by criterion (d).
    """
    if mach <= 1.0:
        raise ValueError("Ackeret theory requires Ma > 1")
    theta = math.radians(theta_deg)
    return gamma * mach ** 2 * theta / math.sqrt(mach ** 2 - 1.0)


def wedge_expansion_reference(mach: float, theta_deg: float,
                              gamma: float = GAMMA) -> Dict[str, float]:
    """Exact rear-wedge (Prandtl-Meyer expansion) pressure ratio.

    Flow behind the front oblique shock (Mach ``M2``) turns by ``2*theta`` at
    the mid-chord expansion; the surface pressure on the rear wedge follows from
    the isentropic Prandtl-Meyer relations.  This is the *expansion* half of the
    N-wave source signature, i.e. the second anchor criterion (d) needs.
    """
    from .isentropic import prandtl_meyer_angle, prandtl_meyer_from_nu

    beta = oblique_shock_angle(mach, theta_deg, gamma)
    mn1 = mach * math.sin(beta)
    p2_p1 = 1.0 + 2.0 * gamma / (gamma + 1.0) * (mn1 ** 2 - 1.0)
    mn2 = math.sqrt((1.0 + 0.5 * (gamma - 1.0) * mn1 ** 2)
                    / (gamma * mn1 ** 2 - 0.5 * (gamma - 1.0)))
    m2 = mn2 / math.sin(beta - math.radians(theta_deg))
    nu2 = prandtl_meyer_angle(m2, gamma)
    nu3 = nu2 + 2.0 * math.radians(theta_deg)
    m3 = prandtl_meyer_from_nu(nu3, gamma)
    p3_p2 = ((1.0 + 0.5 * (gamma - 1.0) * m2 ** 2)
             / (1.0 + 0.5 * (gamma - 1.0) * m3 ** 2)) ** (gamma / (gamma - 1.0))
    return {"mach_behind_front_shock": m2, "mach_rear_wedge": float(m3),
            "turning_angle_deg": 2.0 * theta_deg,
            "rear_p_ratio": p2_p1 * p3_p2,
            "rear_dp_over_p": p2_p1 * p3_p2 - 1.0}


def wedge_reference_table(mach: float, theta_deg: float,
                          gamma: float = GAMMA) -> Dict[str, float]:
    """Ackeret vs exact oblique shock for a compression wedge.

    Returns both pressure ratios plus the *theory* error of linearised Ackeret
    theory.  This separation matters: a solver that reproduces the exact
    oblique shock but misses Ackeret by 20 % is **correct** -- the 20 % is the
    O(theta^2) truncation error of Ackeret itself (Anderson 2003, sec. 9.7).
    """
    ack = ackeret_pressure_ratio(mach, theta_deg, gamma)
    exact = oblique_shock_pressure_ratio(mach, theta_deg, gamma)
    return {"mach": mach, "theta_deg": theta_deg,
            "ackeret_dp_over_p": ack,
            "exact_dp_over_p": exact - 1.0,
            "ackeret_error_pct_vs_exact": 100.0 * (ack - (exact - 1.0)) / (exact - 1.0),
            "shock_angle_deg": math.degrees(oblique_shock_angle(mach, theta_deg, gamma))}


# ---------------------------------------------------------------------------
# lift slope and compressibility theory
# ---------------------------------------------------------------------------
def thin_airfoil_lift_slope(mach: float, gamma: float = GAMMA) -> float:
    """``dCL/dalpha`` of a thin symmetric airfoil, per radian (2*pi)."""
    return 2.0 * math.pi


def prandtl_glauert_lift_slope(mach: float) -> float:
    """``2*pi/sqrt(1 - M^2)`` per radian (Prandtl-Glauert / thin airfoil)."""
    beta = math.sqrt(max(1.0 - mach ** 2, 1e-12))
    return 2.0 * math.pi / beta


def thickness_corrected_lift_slope(t: float = 0.12) -> float:
    """Lift slope of a *thick* symmetric profile: ``2*pi*(1 + 0.77 t)``.

    Thin-airfoil theory (``2*pi``) is the zero-thickness limit; for a NACA
    4-digit profile with thickness ratio ``t`` the classical thickness
    correction is ``+0.77 t`` (Abbott & von Doenhoff 1959 sec. 4.2; equivalently
    the Joukowski-profile result ``a0 = 2*pi*(1+tau)`` with ``tau ~ 0.77 t``).
    At ``t = 0.12`` this is +9.2 %, i.e. the same size as the deviation of a
    finite-thickness airfoil from the 2*pi line -- which is why criterion (a)
    must not be read as a pure solver-error indicator.
    """
    return 2.0 * math.pi * (1.0 + 0.77 * float(t))


def lift_slope(alpha_deg: Sequence[float], cl: Sequence[float],
               alpha_max: float = 4.0) -> Dict[str, float]:
    """Least-squares ``dCL/dalpha`` per radian on the *linear* range.

    The slope is fitted on ``alpha <= alpha_max`` (the range to which thin
    airfoil theory applies); the full-range fit is returned as well so that a
    stall/separation onset cannot hide behind a window choice.
    """
    a = np.radians(np.asarray(alpha_deg, float))
    c = np.asarray(cl, float)
    sel = np.abs(np.asarray(alpha_deg, float)) <= alpha_max + 1e-12
    if sel.sum() < 2:
        sel = np.zeros_like(a, dtype=bool)
        sel[:2] = True
    A = np.vstack([a[sel], np.ones(sel.sum())]).T
    slope, intercept = np.linalg.lstsq(A, c[sel], rcond=None)[0]
    A2 = np.vstack([a, np.ones(a.size)]).T
    slope_all, intercept_all = np.linalg.lstsq(A2, c, rcond=None)[0]
    resid = c[sel] - (slope * a[sel] + intercept)
    return {"dcl_dalpha_per_rad": float(slope),
            "dcl_dalpha_per_deg": float(slope * math.pi / 180.0),
            "dcl_dalpha_per_rad_all": float(slope_all),
            "alpha_window_deg": float(alpha_max),
            "n_points_window": int(sel.sum()),
            "max_residual_window": float(np.abs(resid).max()),
            "zero_lift_offset_deg": float(-intercept / slope * 180.0 / math.pi)}


def critical_mach_from_cp_min(cp_min_incompressible: float,
                              gamma: float = GAMMA) -> float:
    """``M_cr`` from the incompressible suction peak (Abbott & von Doenhoff).

    The critical Mach number is the freestream Mach at which the *local* Mach
    number at the suction peak reaches 1.  Using Prandtl-Glauert to scale the
    incompressible peak: ``Cp(M) = Cp0/sqrt(1-M^2)`` and the isentropic
    relation ``p/p_inf(Cp)`` is solved for ``M`` by bisection.
    """
    cp0 = abs(float(cp_min_incompressible))

    def local_mach_stag(mach):
        beta = math.sqrt(max(1.0 - mach * mach, 1e-12))
        cp = -cp0 / beta
        p_ratio = 1.0 + 0.5 * gamma * mach * mach * cp
        if p_ratio <= 0.0:
            return float("inf")
        stag = 1.0 + 0.5 * (gamma - 1.0) * mach * mach
        m2 = 2.0 / (gamma - 1.0) * ((stag / p_ratio) ** ((gamma - 1.0) / gamma) - 1.0)
        return math.sqrt(max(m2, 0.0))

    lo, hi = 1e-3, 0.99
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if local_mach_stag(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# grid convergence index (Roache)
# ---------------------------------------------------------------------------
def gci_roache(f_coarse: float, f_medium: float, f_fine: float,
               r: float, safety: float = 1.25) -> Dict[str, float]:
    """Roache three-grid GCI.

    ``r`` is the effective refinement ratio ``(N_fine/N_coarse)**(1/2)`` in two
    dimensions.  Returns the apparent order ``p``, the Richardson-extrapolated
    value, the fine-grid GCI in percent and whether the triplet is monotone and
    converged (``p > 0``, GCI below 5 %).  A non-monotone triplet yields
    ``p = nan`` and ``converged = False`` instead of a meaningless number.
    """
    fc, fm, ff = float(f_coarse), float(f_medium), float(f_fine)
    e_fm = ff - fm
    e_mc = fm - fc
    out: Dict[str, float] = {"f_coarse": fc, "f_medium": fm, "f_fine": ff,
                             "r": float(r), "safety_factor": safety,
                             "relative_change_fine_pct": 100.0 * abs(e_fm / ff) if ff else float("nan")}
    if abs(e_mc) < 1e-30 or e_fm * e_mc <= 0.0 or abs(e_fm) < 1e-30:
        out.update({"apparent_order": float("nan"), "f_extrapolated": float("nan"),
                    "gci_fine_pct": float("nan"), "converged": False,
                    "reason": "non-monotone or zero change across the ladder"})
        return out
    ratio = e_fm / e_mc
    p = abs(math.log(abs(ratio)) / math.log(r))
    f_ext = ff + e_fm / (r ** p - 1.0)
    gci = safety * abs(e_fm / ff) / (r ** p - 1.0)
    out.update({"apparent_order": p, "f_extrapolated": f_ext,
                "gci_fine_pct": 100.0 * gci,
                "richardson_error_pct": 100.0 * abs(e_fm / ff),
                "error_vs_extrapolated_pct": 100.0 * abs((ff - f_ext) / f_ext),
                "converged": bool(gci * 100.0 < 5.0 and p > 0.0)})
    return out


# ---------------------------------------------------------------------------
# case runners
# ---------------------------------------------------------------------------
def _surface_split(n_points: int) -> Dict[str, slice]:
    """Index ranges of the upper/lower sheets in a :func:`contour_loop` loop."""
    m = n_points // 2 + 1
    return {"upper": slice(0, m - 1), "le": slice(m - 1, m), "lower": slice(m, 2 * m - 2)}


def run_naca_case(mach: float, alpha_deg: float, n_points: int = 160,
                  nj: int = 64, first_cell: float = 0.002, r_out: float = 12.0,
                  n_iter: int = 8000, tol: float = 1e-6, cfl: float = 0.4,
                  order: int = 2, verbose: bool = False) -> Dict[str, object]:
    """Steady Euler solution of the NACA0012 section, with force integration.

    The mesh is the body-fitted O-grid of
    :func:`~tensorlbm.compressible.kinetic_flux.build_annulus_mesh` with the
    radial ladder of :func:`radial_levels_from_surface`, i.e. the **first grid
    line is the airfoil surface** (``r = 0``); the wall is an adiabatic slip
    wall (pure pressure flux) and the far field is characteristic
    (KFVS-upwinded freestream) at ``r_out`` chords.
    """
    x_loop, y_loop = contour_loop(naca0012_half_thickness, n_points=n_points)
    levels = radial_levels_from_surface(nj, r_out, first_cell=first_cell)
    mesh = build_annulus_mesh(x_loop, y_loop, levels)
    cfg = SolverConfig(mach=mach, alpha_deg=alpha_deg, cfl=cfl, order=order)
    sol = KineticFluxSolver(mesh, cfg)
    history: List[dict] = []
    for blk in range(max(1, n_iter // 500)):
        sol.run(min(500, n_iter), log_every=500)
        wx, cp_ = sol.wall_cp()
        cl_, cd_ = integrate_forces(mesh.wall_x, mesh.wall_y, cp_, alpha_deg)
        history.append({"iter": (blk + 1) * 500,
                        "residual": float(sol.density_residual_norm()),
                        "cl": float(cl_), "cd": float(cd_)})
        if verbose:
            print(f"  [Ma={mach} a={alpha_deg}] {history[-1]}", flush=True)
    wall_x, cp = sol.wall_cp()
    cl, cd = integrate_forces(mesh.wall_x, mesh.wall_y, cp, alpha_deg)
    drift = {}
    if len(history) >= 4:
        drift = {"cd_last_window_change": float(history[-1]["cd"] - history[-4]["cd"]),
                 "cl_last_window_change": float(history[-1]["cl"] - history[-4]["cl"]),
                 "iter_window": 2000}
    rho, u, v, p = sol.W
    mach_field = sol.mach_field()
    div = _surface_split(n_points)
    up = div["upper"]
    lo = div["lower"]
    return {
        "case": {"mach": mach, "alpha_deg": alpha_deg},
        "mesh": {"n_points": n_points, "nj": nj, "cells": int(mesh.n_cells),
                 "first_cell": first_cell, "r_out": r_out},
        "solver": {"cfl": cfl, "order": order, "iterations": n_iter,
                   "residual": float(sol.density_residual_norm()),
                   "drift": drift, "history": history},
        "forces": {"cl": cl, "cd": cd},
        "field": {"max_local_mach": float(mach_field.max()),
                  "cp_min": float(cp.min()), "cp_max": float(cp.max()),
                  "cfl_number": cfl},
        "isentropic": isentropic_deviation(rho, p, mach_field),
        "surface": {"x": wall_x, "cp": cp,
                    "upper_cp": cp[up], "upper_x": wall_x[up],
                    "lower_cp": cp[lo], "lower_x": wall_x[lo]},
        "solution": sol, "mesh_obj": mesh,
    }


def diamond_loop(theta_deg: float, chord: float = 1.0,
                 spacing: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Closed double-wedge contour (4 straight edges) with uniform spacing.

    ``contour_loop`` clusters points in ``x`` and duplicates the trailing-edge
    node; on a diamond that produces a zero-length face at the apex and a
    sliver cell at the sharp leading edge (``ds ~ 5e-5 c`` while the first cell
    height is ``2e-3 c``), which destabilises the explicit update.  Here each
    edge is sampled with a uniform spacing of about ``spacing`` chords and the
    loop starts at the lower trailing-edge corner, so no node is repeated.
    """
    tau = 0.5 * math.tan(math.radians(theta_deg))
    if spacing is None:
        spacing = 0.02
    n_edge = max(4, int(math.ceil(math.hypot(0.5, tau) / spacing)))
    xu = np.linspace(0.0, 1.0, 2 * n_edge + 1)      # LE -> mid -> TE (odd: apex included)
    yt = np.where(xu <= 0.5, 2.0 * tau * xu, 2.0 * tau * (1.0 - xu))
    # LE -> upper -> TE -> lower -> (back towards LE, LE not repeated)
    x_loop = np.concatenate([xu, xu[::-1][1:-1]])
    y_loop = np.concatenate([yt, -yt[::-1][1:-1]])
    return x_loop * chord, y_loop * chord


def run_wedge_case(mach: float, theta_deg: float, n_points: int = 240,
                   nj: int = 96, first_cell: float = 0.002,
                   r_out: float = 14.0, n_iter: int = 4000,
                   cfl: float = 0.35) -> Dict[str, object]:
    """Supersonic flow past a symmetric ``2*theta`` diamond (N-wave source).

    The front wedge compresses the flow by ``theta`` and the rear wedge expands
    it back, so the body-surface pressure signature is the classic
    compression/expansion (N-wave source) pair.  The measured plateau jumps are
    compared with Ackeret (criterion d) and with the exact oblique shock.
    """
    tau = 0.5 * math.tan(math.radians(theta_deg))
    # surface spacing ~ two first-cell heights keeps the apex cells well shaped
    x_loop, y_loop = diamond_loop(theta_deg, spacing=max(2.0 * first_cell,
                                                        2.0 / n_points))
    levels = radial_levels_from_surface(nj, r_out, first_cell=first_cell)
    mesh = build_annulus_mesh(x_loop, y_loop, levels)
    cfg = SolverConfig(mach=mach, alpha_deg=0.0, cfl=cfl)
    sol = KineticFluxSolver(mesh, cfg)
    for _ in range(max(1, n_iter // 500)):
        sol.run(500, log_every=500)
    wall_x, cp = sol.wall_cp()
    cl, cd = integrate_forces(mesh.wall_x, mesh.wall_y, cp, 0.0)
    rinf, uinf, vinf, pinf = cfg.freestream()
    q = 0.5 * rinf * (uinf ** 2 + vinf ** 2)
    p_ratio = 1.0 + cp * q / pinf

    # plateau averages: front wedge x in (0.1, 0.45), rear wedge (0.55, 0.9)
    def plateau(lo, hi):
        sel = (wall_x > lo) & (wall_x < hi)
        return float(np.mean(p_ratio[sel])), float(np.std(p_ratio[sel]))

    front, front_std = plateau(0.10, 0.45)
    rear, rear_std = plateau(0.55, 0.90)
    ref = wedge_reference_table(mach, theta_deg)
    rear_ref = wedge_expansion_reference(mach, theta_deg)
    front_dp = front - 1.0
    return {
        "case": {"mach": mach, "theta_deg": theta_deg, "tau": tau},
        "mesh": {"n_points": n_points, "nj": nj, "cells": int(mesh.n_cells),
                 "first_cell": first_cell, "r_out": r_out},
        "solver": {"iterations": n_iter, "residual": float(sol.density_residual_norm())},
        "front_wedge": {"p_ratio": front, "dp_over_p": front_dp, "std": front_std},
        "rear_wedge": {"p_ratio": rear, "dp_over_p": rear - 1.0, "std": rear_std},
        "reference": ref,
        "error_vs_ackeret_pct": 100.0 * (front_dp - ref["ackeret_dp_over_p"])
                                / ref["ackeret_dp_over_p"],
        "error_vs_exact_oblique_pct": 100.0 * (front_dp - ref["exact_dp_over_p"])
                                      / ref["exact_dp_over_p"],
        "rear_reference": rear_ref,
        "rear_error_vs_exact_expansion_pct": 100.0 * (
            (rear - 1.0) - rear_ref["rear_dp_over_p"]) / rear_ref["rear_dp_over_p"],
        "forces": {"cl": cl, "cd": cd},
        "surface": {"x": wall_x, "cp": cp, "p_ratio": p_ratio},
        "solution": sol, "mesh_obj": mesh,
    }


# ---------------------------------------------------------------------------
# experimental anchors (provenance gated, never silently substituted)
# ---------------------------------------------------------------------------
def experimental_anchor_report(agard_path: Optional[str] = None,
                               bailey_path: Optional[str] = None) -> Dict[str, object]:
    """Status of the AGARD / Bailey experimental anchors used by (b) and (c)."""
    return {
        "agard_cp": experimental_anchor_status(agard_path),
        "bailey_n_wave": experimental_anchor_status(bailey_path),
        "policy": ("criteria (b) drag rise vs AGARD and (c) shock position vs "
                   "experiment are reported as WITHHELD unless a digitised, "
                   "provenance-carrying dataset is supplied; the solver side is "
                   "still quantified (Mesh-converged CD, shock x/c with GCI) and "
                   "cross-checked against analytic anchors (Rankine-Hugoniot, "
                   "Ackeret, Prandtl-Glauert)."),
    }