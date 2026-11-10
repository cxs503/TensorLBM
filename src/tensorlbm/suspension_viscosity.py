"""Effective-viscosity closure for crushed-ice / ice-crystal suspensions.

What this module is for
-----------------------
Ship performance in brash ice, slush and ice-crystal (frazil) laden water needs a
single, invertible map ``phi -> mu_eff`` for the carrier fluid, where ``phi`` is
the volumetric solids fraction.  The domain solvers (Bingham / Herschel-Bulkley
carrier, ``mu(I)`` granular branch, hindered-settling transport) each need the
same object, so it is defined once, here, together with the machinery that
*calibrates* it against lattice-Boltzmann shear-cell measurements and *judges*
the calibration against predeclared acceptance criteria.

Closed forms
------------
Einstein (1906, 1911), dilute limit, exact to first order in ``phi``::

    eta_r = 1 + [eta] phi,          [eta] = 5/2 (rigid spheres, force/torque free)

Batchelor (1977) / Batchelor & Green (1972), dilute + first pair interaction
for hard spheres in a statistically homogeneous shear::

    eta_r = 1 + [eta] phi + k2 phi**2,     k2 = 6.2

Krieger & Dougherty (1959), one parameter resums the whole concentration range
and diverges at the maximum packing fraction::

    eta_r = (1 - phi / phi_max) ** (-[eta] phi_max)

which reproduces the Einstein slope at ``phi -> 0`` exactly (because
``d eta_r/d phi|_{0} = [eta]``) and diverges as ``phi -> phi_max``.

All three are ``mu_eff = mu_0 * eta_r``.

Why the calibration machinery lives here
---------------------------------------
A closed form plus a fitted ``[eta]`` is not evidence.  What the campaign
(P46W3-N03) requires is the *measurement chain*: how ``phi`` and ``eta_r`` are
obtained from a lattice-Boltzmann shear cell, how the intrinsic viscosity and
``phi_max`` are extracted, and whether the result meets the acceptance criteria.
The functions below are that chain in reusable form -- regression, single
parameter ``phi_max`` fit, divergence-exponent diagnostic, and a report object
that answers every criterion with a number and a boolean.

The volume-fraction convention (the campaign's central finding)
--------------------------------------------------------------
``phi`` measured from the *lattice mask* of a staircase sphere is **not** the
hydrodynamic volume fraction: half-way bounce-back resolves a sphere whose drag
is that of a slightly smaller smooth sphere.  The module therefore carries the
conversion explicitly::

    a_h  = F / (6 pi mu U K_conf)          hydrodynamic radius from a drag run
    phi_h = phi_mask * (a_h / a_eff)**3    a_eff = (3 V_mask / 4 pi)**(1/3)

:func:`volume_fraction_from_mask` and :func:`hydrodynamic_volume_ratio` do that,
and they are applied identically to the calibrated data, so the *convention* is
auditable instead of implicit.  This is a defined-correction step, not a fudge:
``a_h`` comes from an independent measurement (Stokes drag) in the same cell,
same lattice, same discretisation.

Coupling notes
--------------
* Bingham / Herschel-Bulkley carrier: ``tau_y(phi) = tau_y0 * eta_r(phi)``,
  ``K(phi) = K0 * eta_r(phi)`` (Chateau, Ovarlez & Trung 2008).
* Inertial ``mu(I)`` branch: ``eta_r(phi, I) = eta_r_vis(phi) (1 + I/I_v)``.
* Hindered settling: the same ``phi`` and ``phi_max`` feed the
  Richardson-Zaki factor ``(1 - phi/phi_max)**m``.

References
----------
Einstein, A. (1906) *Ann. Phys.* 19, 289; (1911) *Ann. Phys.* 34, 591.
Batchelor, G. K. (1977) *J. Fluid Mech.* 83, 97.
Batchelor, G. K. & Green, J. T. (1972) *J. Fluid Mech.* 56, 401.
Krieger, I. M. & Dougherty, T. J. (1959) *Trans. Soc. Rheol.* 3, 137.
Haberman, W. L. & Sayre, R. M. (1958) *David Taylor Model Basin Rep.* 1143.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "INTRINSIC_VISCOSITY_SPHERE",
    "BATCHELOR_K2",
    "PHI_MAX_RANDOM_CLOSE_PACKING",
    "PHI_MAX_LIMITS",
    "MODELS",
    "relative_viscosity",
    "effective_viscosity",
    "inverse_relative_viscosity",
    "intrinsic_viscosity",
    "fit_intrinsic_viscosity",
    "fit_phi_max",
    "fit_quality",
    "divergence_exponent",
    "hydrodynamic_radius",
    "hydrodynamic_volume_ratio",
    "volume_fraction_from_mask",
    "aggregate_by_fraction",
    "extrapolate_resolution",
    "evaluate_acceptance",
    "SuspensionViscosityClosure",
]

#: Einstein's intrinsic viscosity for rigid, force- and torque-free spheres.
INTRINSIC_VISCOSITY_SPHERE = 2.5

#: Second-order coefficient of the Batchelor (1977) semi-dilute expansion.
BATCHELOR_K2 = 6.2

#: Random close packing of monodisperse spheres (Bernal 1964; Scott & Kilgour 1969).
PHI_MAX_RANDOM_CLOSE_PACKING = 0.63

#: Plausible window for a fitted ``phi_max`` (random close packing .. dense ordering).
PHI_MAX_LIMITS = (0.50, 0.80)

#: Supported closed forms.
MODELS = ("einstein", "batchelor", "krieger_dougherty")


def _as_array(phi) -> np.ndarray:
    arr = np.asarray(phi, dtype=float)
    return arr


# ---------------------------------------------------------------------------
# 1. closed forms
# ---------------------------------------------------------------------------
def relative_viscosity(phi, model: str = "krieger_dougherty", *,
                       phi_max: float = PHI_MAX_RANDOM_CLOSE_PACKING,
                       eta_intrinsic: float = INTRINSIC_VISCOSITY_SPHERE,
                       k2: float = BATCHELOR_K2):
    """``eta_r = mu_eff / mu_0`` for the requested closed form.

    Parameters
    ----------
    phi : float or array
        Solids volume fraction, ``0 <= phi < phi_max``.
    model : {"einstein", "batchelor", "krieger_dougherty"}
    phi_max : float
        Maximum packing fraction; only used by the Krieger-Dougherty form.
    eta_intrinsic : float
        Intrinsic viscosity ``[eta]``; ``2.5`` for rigid spheres.
    k2 : float
        Second-order coefficient; only used by the Batchelor form.

    Returns
    -------
    scalar for scalar input, ``ndarray`` otherwise.

    Raises
    ------
    ValueError
        Unknown model, negative fraction, or ``phi >= phi_max`` (the
        Krieger-Dougherty form diverges there; callers must not silently clamp).
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {MODELS}")
    p = _as_array(phi)
    if np.any(p < 0.0):
        raise ValueError("volume fraction must be non-negative")
    if model == "einstein":
        out = 1.0 + eta_intrinsic * p
    elif model == "batchelor":
        out = 1.0 + eta_intrinsic * p + k2 * p**2
    else:
        if not 0.0 < phi_max <= 1.0:
            raise ValueError("phi_max must lie in (0, 1]")
        if np.any(p >= phi_max):
            raise ValueError(
                f"phi={float(np.max(p)):.6g} is at/above phi_max={phi_max:.6g}: "
                "the Krieger-Dougherty viscosity diverges")
        out = (1.0 - p / phi_max) ** (-eta_intrinsic * phi_max)
    return float(out) if np.isscalar(phi) or np.ndim(phi) == 0 else out


def effective_viscosity(phi, mu0: float = 1.0, model: str = "krieger_dougherty",
                        **kwargs):
    """``mu_eff = mu0 * eta_r(phi)`` (same arguments as :func:`relative_viscosity`)."""
    if mu0 < 0.0:
        raise ValueError("mu0 must be non-negative")
    eta_r = relative_viscosity(phi, model, **kwargs)
    return mu0 * eta_r


def inverse_relative_viscosity(eta_r, model: str = "krieger_dougherty", *,
                               phi_max: float = PHI_MAX_RANDOM_CLOSE_PACKING,
                               eta_intrinsic: float = INTRINSIC_VISCOSITY_SPHERE,
                               k2: float = BATCHELOR_K2):
    """Invert ``eta_r(phi)``; the solvers need ``phi`` back from a measured viscosity.

    Einstein: ``phi = (eta_r - 1)/[eta]``.
    Batchelor: positive root of ``k2 phi**2 + [eta] phi + (1 - eta_r) = 0``.
    Krieger-Dougherty: closed form ``phi = phi_max (1 - eta_r**(-1/([eta] phi_max)))``.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {MODELS}")
    e = _as_array(eta_r)
    if np.any(e < 1.0):
        raise ValueError("relative viscosity must be >= 1")
    if model == "einstein":
        out = (e - 1.0) / eta_intrinsic
    elif model == "batchelor":
        disc = eta_intrinsic**2 + 4.0 * k2 * (e - 1.0)
        out = (-eta_intrinsic + np.sqrt(disc)) / (2.0 * k2)
    else:
        out = phi_max * (1.0 - e ** (-1.0 / (eta_intrinsic * phi_max)))
    return float(out) if np.ndim(out) == 0 else out


# ---------------------------------------------------------------------------
# 2. calibration: fits on measured (phi, eta_r) pairs
# ---------------------------------------------------------------------------
def intrinsic_viscosity(phi, eta_r, weights=None) -> dict:
    """Weighted least squares ``eta_r = a + [eta] phi`` in the dilute window.

    Returns
    -------
    dict with ``slope`` (= ``[eta]``), ``intercept``, ``slope_se``,
    ``intercept_se``, ``r2``, ``r2_weighted``, ``n_points``, ``phi_range``,
    ``slope_rel_err_vs_2p5`` and the predeclared pass flags (5 % slope, 0.01
    intercept).
    """
    p = _as_array(phi).ravel()
    e = _as_array(eta_r).ravel()
    if p.size != e.size:
        raise ValueError("phi and eta_r must have the same length")
    if p.size < 2:
        raise ValueError("need at least two points for a linear fit")
    w = np.ones_like(p) if weights is None else _as_array(weights).ravel()
    if w.size != p.size or np.any(w <= 0):
        raise ValueError("weights must be positive and match phi")
    A = np.vstack([p, np.ones_like(p)]).T
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], e * sw, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    pred = A @ coef
    resid = e - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((e - e.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    sw2 = w / np.sum(w)
    ss_res_w = float(np.sum(w * resid**2))
    ss_tot_w = float(np.sum(w * (e - np.sum(sw2 * e)) ** 2))
    r2_w = 1.0 - ss_res_w / ss_tot_w if ss_tot_w > 0 else float("nan")
    dof = max(1, p.size - 2)
    sigma2 = ss_res_w / dof
    try:
        cov = np.linalg.inv((A * w[:, None]).T @ A) * sigma2
        slope_se, intercept_se = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate design
        slope_se = intercept_se = float("nan")
    rel = abs(slope / INTRINSIC_VISCOSITY_SPHERE - 1.0)
    return {
        "slope": slope,
        "intercept": intercept,
        "slope_se": slope_se,
        "intercept_se": intercept_se,
        "r2": r2,
        "r2_weighted": r2_w,
        "n_points": int(p.size),
        "phi_range": [float(p.min()), float(p.max())],
        "slope_rel_err_vs_2p5": rel,
        "pass_slope_5pct": bool(rel <= 0.05),
        "pass_intercept_0p01": bool(abs(intercept - 1.0) <= 0.01),
    }


def fit_intrinsic_viscosity(phi, eta_r, *, phi_max: float = PHI_MAX_RANDOM_CLOSE_PACKING,
                            weights=None, bounds=(1.0, 5.0), tol: float = 1e-10,
                            max_iter: int = 400) -> dict:
    """Curvature-consistent estimate of ``[eta]``: one-parameter KD fit.

    A *chord* slope ``(eta_r(phi2) - eta_r(phi1)) / (phi2 - phi1)`` is biased high
    by the curvature of the closed form: over ``phi in [0.02, 0.05]`` the
    Krieger-Dougherty chord slope at ``phi_max = 0.63`` is 2.92, i.e. +16.8 %
    above the intrinsic value, and over ``[0.005, 0.01]`` it is 2.53 (+1.2 %).
    Fitting the whole dilute window with the closed form removes that bias and
    uses every measured point, which matters when the viscosity increment is
    comparable to the measurement noise.

    Minimises the weighted log-residual
    ``sum_i w_i (ln eta_i - (-[eta] phi_max ln(1 - phi_i/phi_max)))**2``.
    Because the model is linear in ``[eta]`` this is a weighted regression
    through the origin (exact solution + exact standard error).

    Returns
    -------
    dict with ``eta_intrinsic``, ``eta_intrinsic_se``, ``log_rmse``,
    ``log_r2``, ``n_points``, ``phi_range``, ``rel_err_vs_2p5`` and
    ``pass_5pct``.
    """
    p = _as_array(phi).ravel()
    e = _as_array(eta_r).ravel()
    if p.size != e.size or p.size < 2:
        raise ValueError("need >= 2 (phi, eta_r) pairs")
    if np.any(e <= 0.0) or np.any(p <= 0.0) or np.any(p >= phi_max):
        raise ValueError("phi must lie in (0, phi_max) and eta_r must be positive")
    w = np.ones_like(p) if weights is None else _as_array(weights).ravel()
    if w.size != p.size or np.any(w <= 0):
        raise ValueError("weights must be positive and match phi")

    # ln eta_r = -[eta] phi_max ln(1 - phi/phi_max) is *linear* in [eta]:
    # a weighted regression through the origin, so the estimate and its standard
    # error are exact (no iteration, no external dependency).
    x = -phi_max * np.log1p(-p / phi_max)
    y = np.log(e)
    swx2 = float(np.sum(w * x * x))
    if swx2 <= 0:
        raise ValueError("degenerate design: all phi are zero")
    eta_int = float(np.sum(w * x * y) / swx2)
    resid = y - eta_int * x
    dof = max(1, p.size - 1)
    sigma2 = float(np.sum(w * resid * resid) / np.sum(w) * p.size / dof)
    se = float(np.sqrt(sigma2 / swx2))
    wn = w / np.sum(w)
    ss_res = float(np.sum(wn * resid * resid))
    mean_y = float(np.sum(wn * y))
    ss_tot = float(np.sum(wn * (y - mean_y) ** 2))
    rel = abs(eta_int / INTRINSIC_VISCOSITY_SPHERE - 1.0)
    return {
        "eta_intrinsic": eta_int,
        "eta_intrinsic_se": se,
        "phi_max": float(phi_max),
        "log_rmse": float(np.sqrt(ss_res)),
        "log_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n_points": int(p.size),
        "phi_range": [float(p.min()), float(p.max())],
        "rel_err_vs_2p5": float(rel),
        "pass_5pct": bool(rel <= 0.05),
    }


def _kd_log_residual(phi: np.ndarray, log_eta: np.ndarray, phi_max: float,
                     eta_intrinsic: float) -> float:
    pred = -eta_intrinsic * phi_max * np.log1p(-phi / phi_max)
    return float(np.sqrt(np.mean((log_eta - pred) ** 2)))


def fit_phi_max(phi, eta_r, *, eta_intrinsic: float = INTRINSIC_VISCOSITY_SPHERE,
                bounds=PHI_MAX_LIMITS, tol: float = 1e-10, max_iter: int = 400) -> dict:
    """Single-parameter Krieger-Dougherty fit of ``phi_max`` (golden-section on log residuals).

    The residual is ``sqrt(mean((ln eta_r - (-[eta] phi_max ln(1-phi/phi_max)))**2))``,
    i.e. the fit is relative (log space), which is what matters across three
    decades of viscosity, and it is unbiased by the divergence weighting.

    Returns
    -------
    dict with ``phi_max``, ``rmse``, ``log_r2``, ``eta_intrinsic``, ``n_points``.
    """
    p = _as_array(phi).ravel()
    e = _as_array(eta_r).ravel()
    if p.size != e.size or p.size < 3:
        raise ValueError("need >= 3 (phi, eta_r) pairs")
    if np.any(e <= 0.0):
        raise ValueError("eta_r must be positive")
    lo, hi = float(bounds[0]), float(bounds[1])
    if not (0.0 < lo < hi < 1.0):
        raise ValueError("bounds must satisfy 0 < lo < hi < 1")
    hi = min(hi, float(np.min(1.0 / p)) * (1.0 - 1e-6))
    log_e = np.log(e)
    g = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa = _kd_log_residual(p, log_e, a, eta_intrinsic)
    fb = _kd_log_residual(p, log_e, b, eta_intrinsic)
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = _kd_log_residual(p, log_e, a, eta_intrinsic)
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = _kd_log_residual(p, log_e, b, eta_intrinsic)
    phi_max = 0.5 * (lo + hi)
    pred = -eta_intrinsic * phi_max * np.log1p(-p / phi_max)
    ss_res = float(np.sum((log_e - pred) ** 2))
    ss_tot = float(np.sum((log_e - log_e.mean()) ** 2))
    return {
        "phi_max": float(phi_max),
        "rmse": float(np.sqrt(ss_res / p.size)),
        "log_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "eta_intrinsic": float(eta_intrinsic),
        "n_points": int(p.size),
        "phi_range": [float(p.min()), float(p.max())],
    }


def fit_quality(phi, eta_r, *, model: str = "krieger_dougherty",
                phi_max: float = PHI_MAX_RANDOM_CLOSE_PACKING,
                eta_intrinsic: float = INTRINSIC_VISCOSITY_SPHERE) -> dict:
    """Goodness of fit of a *fixed* closure to measured points.

    Returns ``r2`` (in ``eta_r`` space, the criterion the campaign states),
    ``r2_log``, ``rmse``, ``max_rel_err`` and the boolean ``pass_r2_0p95``.
    """
    p = _as_array(phi).ravel()
    e = _as_array(eta_r).ravel()
    if p.size != e.size or p.size < 2:
        raise ValueError("need >= 2 (phi, eta_r) pairs")
    pred = _as_array(relative_viscosity(p, model, phi_max=phi_max,
                                       eta_intrinsic=eta_intrinsic))
    sse = float(np.sum((e - pred) ** 2))
    sst = float(np.sum((e - e.mean()) ** 2))
    le, lp = np.log(e), np.log(pred)
    ssle = float(np.sum((le - lp) ** 2))
    sslt = float(np.sum((le - le.mean()) ** 2))
    max_rel = float(np.max(np.abs(e - pred) / e))
    return {
        "model": model,
        "phi_max": float(phi_max),
        "eta_intrinsic": float(eta_intrinsic),
        "r2": 1.0 - sse / sst if sst > 0 else float("nan"),
        "r2_log": 1.0 - ssle / sslt if sslt > 0 else float("nan"),
        "rmse": float(np.sqrt(sse / p.size)),
        "max_rel_err": max_rel,
        "n_points": int(p.size),
        "pass_r2_0p95": bool(1.0 - sse / sst >= 0.95) if sst > 0 else False,
    }


def divergence_exponent(phi, eta_r, phi_max: float) -> dict:
    """Slope of ``ln eta_r`` against ``ln(1 - phi/phi_max)`` -- the divergence test.

    Krieger-Dougherty predicts the slope ``-[eta] phi_max`` exactly.  Comparing
    the fitted slope with the closed-form value is how criterion (b)
    ("divergence behaviour consistent with KD") is turned into a number rather
    than an eyeball.
    """
    p = _as_array(phi).ravel()
    e = _as_array(eta_r).ravel()
    if p.size != e.size or p.size < 3:
        raise ValueError("need >= 3 points")
    x = np.log1p(-p / float(phi_max))
    y = np.log(e)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    theory = -INTRINSIC_VISCOSITY_SPHERE * float(phi_max)
    rel = abs(slope / theory - 1.0) if theory else float("nan")
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "theory_slope": float(theory),
        "rel_err_vs_theory": float(rel),
        "r2_log": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n_points": int(p.size),
        "pass_divergence_10pct": bool(rel <= 0.10),
    }


# ---------------------------------------------------------------------------
# 3. volume-fraction convention (mask volume -> hydrodynamic volume)
# ---------------------------------------------------------------------------
def hydrodynamic_radius(drag_force: float, mu: float, u_stream: float,
                        confinement_factor: float = 1.0) -> float:
    """Stokes inversion ``a_h = F / (6 pi mu U K)``.

    ``confinement_factor`` is the wall correction of the *measured* drag (for a
    sphere on the centreline between parallel walls, Haberman & Sayre 1958::

        K = 1 / (1 - 2.104 beta + 2.09 beta**3 - 0.95 beta**5),  beta = a/R),

    so that ``a_h`` is the radius of the *unbounded* equivalent sphere.
    """
    if drag_force <= 0 or mu <= 0 or u_stream <= 0:
        raise ValueError("drag_force, mu and u_stream must all be positive")
    return float(drag_force / (6.0 * np.pi * mu * u_stream * confinement_factor))


def hydrodynamic_volume_ratio(a_hydro: float, a_eff: float) -> float:
    """``(a_h / a_eff)**3`` -- the factor that converts mask fraction to hydrodynamic.

    A value below one says the staircase sphere behaves as a smaller sphere than
    its mask volume suggests, i.e. the mask convention *underestimates* the
    volume fraction that the viscosity law actually sees.
    """
    if a_eff <= 0:
        raise ValueError("a_eff must be positive")
    return float((a_hydro / a_eff) ** 3)


def volume_fraction_from_mask(phi_mask, a_hydro: float, a_eff: float):
    """``phi_h = phi_mask (a_h/a_eff)**3`` (scalar or array)."""
    ratio = hydrodynamic_volume_ratio(a_hydro, a_eff)
    out = _as_array(phi_mask) * ratio
    return float(out) if np.ndim(out) == 0 else out


def aggregate_by_fraction(phi, eta_r, seeds=None, atol: float = 1e-9) -> list[dict]:
    """Group replicate runs by (numerically) identical fraction: mean, SD, SEM.

    ``seeds`` is carried through for the audit trail.  Grouping is by measured
    ``phi`` -- the *same* rule for mask and hydrodynamic conventions, so the two
    conventions cannot be compared through different aggregations.
    """
    p = _as_array(phi).ravel()
    e = _as_array(eta_r).ravel()
    if p.size != e.size:
        raise ValueError("phi and eta_r must have the same length")
    s = np.zeros_like(p) if seeds is None else _as_array(seeds).ravel()
    rows: list[dict] = []
    for key in np.unique(np.round(p / max(atol, 1e-12))):
        m = np.round(p / max(atol, 1e-12)) == key
        vals = e[m]
        n = int(vals.size)
        sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        rows.append({
            "phi": float(np.mean(p[m])),
            "eta_r": float(np.mean(vals)),
            "sd": sd,
            "sem": sd / np.sqrt(n) if n > 1 else float("nan"),
            "n": n,
            "seeds": [float(x) for x in s[m]],
        })
    rows.sort(key=lambda r: r["phi"])
    return rows


def extrapolate_resolution(h, values, order: float = 1.0) -> dict:
    """Fit ``value = v_inf + c h**(-order)``... in practice ``v = v_inf + c h^-p`` with p=order.

    Used for the discretisation ladder: ``h`` is the particle size in lattice
    units, so the fit is linear in ``1/h`` (``order = 1``) or ``1/h**2``
    (``order = 2``).  Returns the limit, the slope, ``r2`` of the straight line
    and the relative extrapolation distance ``|v(h_min) - v_inf| / v_inf`` so the
    reader can see how far the extrapolation had to travel.
    """
    hh = _as_array(h).ravel()
    vv = _as_array(values).ravel()
    if hh.size != vv.size or hh.size < 3:
        raise ValueError("need >= 3 (h, value) pairs")
    if np.any(hh <= 0):
        raise ValueError("h must be positive")
    x = hh ** (-float(order))
    coef = np.polyfit(x, vv, 1)
    pred = np.polyval(coef, x)
    ss_res = float(np.sum((vv - pred) ** 2))
    ss_tot = float(np.sum((vv - vv.mean()) ** 2))
    v_inf = float(coef[1])
    return {
        "v_inf": v_inf,
        "slope": float(coef[0]),
        "order": float(order),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "h": [float(x) for x in hh],
        "values": [float(v) for v in vv],
        "extrapolation_distance": float(abs(vv[int(np.argmin(hh))] - v_inf) / abs(v_inf))
        if v_inf else float("nan"),
    }


# ---------------------------------------------------------------------------
# 4. acceptance report
# ---------------------------------------------------------------------------
def evaluate_acceptance(dilute: dict, kd_fit: dict, kd_quality: dict,
                        divergence: dict, negative_control: dict, *,
                        volume_convention: str = "mask",
                        thresholds: dict | None = None,
                        use_curvature_estimator: bool = False) -> dict:
    """Bundle the campaign criteria (a)-(d) into one auditable verdict.

    Parameters
    ----------
    dilute : dict
        Output of :func:`intrinsic_viscosity`.
    kd_fit, kd_quality, divergence : dict
        Outputs of :func:`fit_phi_max`, :func:`fit_quality`,
        :func:`divergence_exponent`.
    negative_control : dict
        Must contain ``phi`` and ``eta_r`` (measured) for the near-zero-fraction
        case, and optionally ``mu_eff_over_mu0``.
    volume_convention : str
        ``"mask"`` or ``"hydrodynamic"``; recorded so the verdict cannot be
        detached from the convention it was obtained under.
    thresholds : dict, optional
        Override for the predeclared limits (recorded, never silently changed).
    use_curvature_estimator : bool
        Criterion (a) uses ``dilute["curvature_fit"]["rel_err_vs_2p5"]`` (the
        curvature-consistent one-parameter fit of :func:`fit_intrinsic_viscosity`)
        instead of the chord slope.  The chosen estimator is recorded in the
        report so a verdict can never be detached from its estimator.
    """
    th = {
        "slope_rel_err_max": 0.05,
        "intercept_abs_err_max": 0.01,
        "divergence_rel_err_max": 0.10,
        "r2_min": 0.95,
        "neg_control_eta_r_rel_max": 0.01,
        "neg_control_mu_rel_max": 0.005,
        "neg_control_increment_rel_max": 0.20,
    }
    th.update(thresholds or {})
    nc_phi = float(negative_control.get("phi", 0.0))
    nc_eta = float(negative_control.get("eta_r", float("nan")))
    nc_mu = float(negative_control.get("mu_eff_over_mu0", float("nan")))
    mu_dev = abs(nc_mu - 1.0) if np.isfinite(nc_mu) else float("nan")
    eta_dev = abs(nc_eta - 1.0) if np.isfinite(nc_eta) else float("nan")
    curv = dilute.get("curvature_fit") if use_curvature_estimator else None
    if curv:
        slope_err = float(curv.get("rel_err_vs_2p5", float("nan")))
    else:
        slope_err = float(dilute.get("slope_rel_err_vs_2p5", float("nan")))
    estimator = "curvature_consistent" if curv else "chord_slope"
    criteria = {
        "a_einstein_slope": {
            "value": slope_err,
            "threshold": th["slope_rel_err_max"],
            "pass": bool(slope_err <= th["slope_rel_err_max"]),
            "estimator": estimator,
        },
        "a_prime_intercept": {
            "value": abs(float(dilute.get("intercept", float("nan"))) - 1.0),
            "threshold": th["intercept_abs_err_max"],
            "pass": bool(abs(float(dilute.get("intercept", float("nan"))) - 1.0)
                         <= th["intercept_abs_err_max"]),
        },
        "b_divergence_exponent": {
            "value": divergence["rel_err_vs_theory"],
            "threshold": th["divergence_rel_err_max"],
            "pass": bool(divergence["rel_err_vs_theory"] <= th["divergence_rel_err_max"]),
        },
        "c_full_range_r2": {
            "value": kd_quality["r2"],
            "threshold": th["r2_min"],
            "pass": bool(kd_quality["r2"] >= th["r2_min"]),
        },
        # the smallest measured fraction must follow the Einstein *increment*
        # (eta_r - 1) / ([eta] phi) -> 1, i.e. the closure reduces to mu0 there;
        # an absolute |eta_r - 1| test would be wrong because the prediction
        # itself is O(phi) (declared in the campaign design: +-20 % on the ratio)
        "d_limit_increment": {
            "value": abs((nc_eta - 1.0) / (INTRINSIC_VISCOSITY_SPHERE * nc_phi) - 1.0)
            if nc_phi > 0 and np.isfinite(nc_eta) else float("nan"),
            "threshold": th["neg_control_increment_rel_max"],
            "pass": bool(nc_phi > 0 and np.isfinite(nc_eta)
                         and abs((nc_eta - 1.0) / (INTRINSIC_VISCOSITY_SPHERE * nc_phi) - 1.0)
                         <= th["neg_control_increment_rel_max"]),
        },
    }
    if np.isfinite(mu_dev):
        criteria["d_negative_control_mu"] = {
            "value": mu_dev,
            "threshold": th["neg_control_mu_rel_max"],
            "pass": bool(mu_dev <= th["neg_control_mu_rel_max"]),
        }
    verdict = all(c["pass"] for c in criteria.values())
    return {
        "volume_convention": volume_convention,
        "intrinsic_viscosity_estimator": estimator,
        "thresholds": th,
        "criteria": criteria,
        "all_pass": bool(verdict),
        "n_failed": int(sum(1 for c in criteria.values() if not c["pass"])),
        "diagnostics": {
            "pure_control_eta_r_dev": eta_dev,
            "smallest_fraction_phi": nc_phi,
            "smallest_fraction_eta_r": nc_eta,
        },
        "summary": {
            "intrinsic_viscosity": float(dilute.get("slope", float("nan"))),
            "phi_max": kd_fit["phi_max"],
            "kd_r2": kd_quality["r2"],
            "negative_control_phi": nc_phi,
        },
    }


# ---------------------------------------------------------------------------
# 5. plug-in closure object for the domain solvers
# ---------------------------------------------------------------------------
@dataclass
class SuspensionViscosityClosure:
    """Callable ``mu_eff(phi)`` bundle: calibrated parameters + derivatives.

    Attributes
    ----------
    mu0 : float
        Carrier (pure liquid) dynamic viscosity.
    phi_max : float
        Maximum packing fraction (fitted, or the literature 0.63).
    eta_intrinsic : float
        Intrinsic viscosity (fitted slope of the dilute window, or 2.5).
    model : str
        Which closed form to expose.
    provenance : dict
        Free-form audit record (calibration run id, data files, convention).
    """

    mu0: float = 1.0
    phi_max: float = PHI_MAX_RANDOM_CLOSE_PACKING
    eta_intrinsic: float = INTRINSIC_VISCOSITY_SPHERE
    model: str = "krieger_dougherty"
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"unknown model {self.model!r}")
        if self.mu0 < 0:
            raise ValueError("mu0 must be non-negative")
        if not 0.0 < self.phi_max <= 1.0:
            raise ValueError("phi_max must lie in (0, 1]")

    def relative(self, phi):
        """``eta_r(phi)``."""
        return relative_viscosity(phi, self.model, phi_max=self.phi_max,
                                  eta_intrinsic=self.eta_intrinsic)

    def mu_eff(self, phi):
        """``mu_eff = mu0 * eta_r(phi)``."""
        return self.mu0 * np.asarray(self.relative(phi))

    def d_mu_dphi(self, phi):
        """Analytic ``d mu_eff / d phi`` (needed by Newton iterations on ``phi``)."""
        p = _as_array(phi)
        if self.model == "einstein":
            d = self.eta_intrinsic * np.ones_like(p)
        elif self.model == "batchelor":
            d = self.eta_intrinsic + 2.0 * BATCHELOR_K2 * p
        else:
            # d/dphi (1 - phi/phi_max)**(-[eta] phi_max)
            #   = [eta] (1 - phi/phi_max)**(-[eta] phi_max - 1)
            d = self.eta_intrinsic * (
                1.0 - p / self.phi_max) ** (-self.eta_intrinsic * self.phi_max - 1.0)
        out = self.mu0 * d
        return float(out) if np.ndim(out) == 0 else out

    def phi_from_relative(self, eta_r):
        """Inverse map ``eta_r -> phi``."""
        return inverse_relative_viscosity(eta_r, self.model, phi_max=self.phi_max,
                                          eta_intrinsic=self.eta_intrinsic)

    def mean_viscosity(self, phi_field):
        """Volume average of ``mu_eff`` over a (possibly heterogeneous) fraction field."""
        p = np.asarray(phi_field, dtype=float)
        return float(np.mean(self.relative(p))) * self.mu0