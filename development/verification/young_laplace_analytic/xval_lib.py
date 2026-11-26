"""Reference/measurement library for XVAL-LBM-YOUNG-LAPLACE-ANALYTIC-PLUS-DROPLET-CALI.

Two independent objects live here, kept deliberately disjoint:

``analytic_*``  **closed-form** surface tension of the free-energy (Swift-Oono-
                Yeomans / Cahn-Hilliard square-gradient) functional used by the
                kernel under test.  It is a function of the free-energy
                coefficients ``(kappa, A, B)`` only - no simulation enters it.

``measure_*``   **measurements on a field**: the mechanical pressure jump
                ``dp = p_in - p_out`` read from the solver's pressure field, the
                equivalent radius ``R_eq`` from the phase-field area, the
                gradient-energy route ``sigma_cap = kappa * integral
                |grad psi|^2 dA / (2 pi R)`` and rest/mass diagnostics.

Neither object uses the other: ``dp`` never sees ``sigma``, the masks never see
the target radius, and ``sigma_cap`` never sees ``dp``.  That separation is what
makes the Young-Laplace comparison a cross-validation instead of algebra.

Conventions (2D, D2Q9 free-energy binary model, matched density rho0):

  free energy   F[psi] = integral [ -A/2 psi^2 + B/4 psi^4 + kappa/2 |grad psi|^2 ]
  chemical pot. mu = -A psi + B psi^3 - kappa lap(psi)
  profile       psi(r) = psi0 tanh((r - R) / xi),  psi0 = sqrt(A/B),  xi = sqrt(2 kappa / A)
  tension       sigma = sqrt(8 kappa A^3 / (9 B^2))          [coefficient route]
                      = (4/3) kappa psi0^2 / xi             [gradient-energy route]
  Laplace (2D)  dp = sigma / R                              [cylinder, not sphere]

The two sigma forms are algebraically identical whenever ``xi = sqrt(2 kappa/A)``;
:func:`soy_constants` returns both and their residual so the identity is *pinned*
instead of hand-copied.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Analytic block
# ---------------------------------------------------------------------------


def analytic_sigma_from_coefficients(kappa: float, A: float, B: float) -> float:
    """``sigma = sqrt(8 kappa A^3 / (9 B^2))`` (square-gradient/CH tension)."""
    if kappa <= 0.0 or A <= 0.0 or B <= 0.0:
        raise ValueError("kappa, A, B must be positive")
    return math.sqrt(8.0 * kappa * A**3 / (9.0 * B * B))


def analytic_sigma_from_gradient_energy(kappa: float, xi: float, psi0: float = 1.0) -> float:
    """``sigma = (4/3) kappa psi0^2 / xi`` from the tanh profile gradient energy.

    ``integral kappa (dpsi/dx)^2 dx`` over one interface with
    ``psi = psi0 tanh(x/xi)`` is ``kappa psi0^2 * (4/3) / xi`` because
    ``integral sech^4 = 4/3``.
    """
    if kappa <= 0.0 or xi <= 0.0:
        raise ValueError("kappa and xi must be positive")
    return (4.0 / 3.0) * kappa * psi0**2 / xi


def soy_constants(sigma_target: float, xi: float) -> dict[str, float]:
    """Free-energy coefficients for a SOY run and the analytic tension they imply.

    Mirrors the kernel convention ``kappa = 0.75 sigma xi``, ``a = -2 kappa/xi^2``
    (``tensorlbm.multiphase_soy_dropin.SOYConfig.kappa`` / ``.a``) with the
    double-well written as ``-A/2 psi^2 + B/4 psi^4`` (``A = B = -a``), so
    ``psi0 = 1``.  ``sigma_analytic`` is then *recomputed from* ``(kappa, A, B)``
    - it is not the input ``sigma_target`` copied over; the residual field shows
    the two agree only because the OSS convention is self-consistent.
    """
    kappa = 0.75 * sigma_target * xi
    a = -2.0 * kappa / (xi**2)
    A = -a
    B = -a
    psi0 = math.sqrt(A / B)
    s_coef = analytic_sigma_from_coefficients(kappa, A, B)
    s_grad = analytic_sigma_from_gradient_energy(kappa, xi, psi0)
    return {
        "sigma_target": sigma_target,
        "xi": xi,
        "kappa": kappa,
        "a": a,
        "A": A,
        "B": B,
        "psi0": psi0,
        "sigma_analytic": s_coef,
        "sigma_analytic_grad": s_grad,
        "sigma_identity_residual": abs(s_coef - s_grad) / s_coef,
        "sigma_target_rel_error": (s_coef - sigma_target) / sigma_target,
        "xi_from_identity": math.sqrt(2.0 * kappa / A),
    }


def laplace_reference_2d(sigma: float, radius: float, dims: int = 2) -> float:
    """Analytic Young-Laplace jump: ``sigma/R`` (cylinder) or ``2 sigma/R`` (sphere)."""
    if dims == 2:
        return sigma / radius
    if dims == 3:
        return 2.0 * sigma / radius
    raise ValueError("dims must be 2 or 3")


def analytic_tanh_profile(r: np.ndarray, radius: float, xi: float, psi0: float = 1.0) -> np.ndarray:
    """``psi = -psi0 tanh((r - R)/xi)`` - +inside, -outside (kernel sign convention)."""
    return -psi0 * np.tanh((np.asarray(r, dtype=np.float64) - radius) / xi)


# ---------------------------------------------------------------------------
# Measurement block (fields only)
# ---------------------------------------------------------------------------


def _grid(shape: tuple[int, int]) -> tuple[np.ndarray, float]:
    ny, nx = shape
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    return np.hypot(xx - nx / 2.0, yy - ny / 2.0), 0.5 * float(min(nx, ny))


def equivalent_radius(psi: np.ndarray) -> tuple[float, tuple[float, float]]:
    """``R_eq = sqrt(area/pi)`` of the ``psi > 0`` set, plus its centroid.

    The centroid is returned so a drifting droplet cannot be mistaken for a
    wrong radius: ``R_eq`` is measured about the *measured* centre, never about
    the box centre that the initial condition used.
    """
    mask = psi > 0.0
    area = float(mask.sum())
    if area <= 0.0:
        raise ValueError("no droplet phase (psi > 0) found")
    ny, nx = psi.shape
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    return math.sqrt(area / math.pi), (float(xx[mask].mean()), float(yy[mask].mean()))


def radial_distance(shape: tuple[int, int], centre: tuple[float, float]) -> np.ndarray:
    ny, nx = shape
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    return np.hypot(xx - centre[0], yy - centre[1])


def pressure_jump(
    p: np.ndarray,
    r: np.ndarray,
    radius: float,
    xi: float,
    shell: float = 1.5,
) -> tuple[float, int, int]:
    """``dp = mean(p)`` inside ``r < R - shell xi`` minus outside ``r > R + shell xi``.

    ``R`` is passed in explicitly so the caller decides *which* radius convention
    is used (measured ``R_eq`` for the primary metric, the initial radius for the
    reproduction of the library convention).
    """
    inside = r < radius - shell * xi
    outside = r > radius + shell * xi
    if not bool(inside.any()) or not bool(outside.any()):
        raise ValueError("masks empty: box too small for this radius/interface width")
    return (
        float(np.mean(p[inside]) - np.mean(p[outside])),
        int(inside.sum()),
        int(outside.sum()),
    )


def gradient_energy_sigma(grad_x: np.ndarray, grad_y: np.ndarray, kappa: float, radius: float) -> float:
    """``sigma_cap = kappa * integral |grad psi|^2 dA / (2 pi R)`` (perimeter form).

    Uses the *measured* gradients of the field and the coefficient ``kappa``; it
    never touches ``dp``, so agreement with ``dp * R`` is a real two-route check
    of the Laplace curvature relation.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    return float(kappa * np.sum(grad_x * grad_x + grad_y * grad_y) / (2.0 * math.pi * radius))


def spurious_velocity(ux: np.ndarray, uy: np.ndarray) -> dict[str, float]:
    """Static droplets must be at rest: report max and RMS speed."""
    speed = np.hypot(ux, uy)
    return {
        "u_max": float(speed.max()),
        "u_rms": float(np.sqrt(np.mean(speed**2))),
        "u_max_over_cs": float(speed.max() / (1.0 / math.sqrt(3.0))),
    }


def interface_width_fit(psi: np.ndarray, radius: float, centre: tuple[float, float]) -> dict[str, float]:
    """Fit ``psi = tanh((r - R)/xi)`` to the measured radial profile.

    The fit is over the annulus ``|r - R| < 2.5`` lattice units and returns the
    measured width ``xi_meas`` and radius ``R_fit``; a systematic ``xi_meas/xi``
    away from 1 is the discretisation signature that would bias the tension.
    """
    r = radial_distance(psi.shape, centre)
    sel = np.abs(r - radius) < 2.5
    if sel.sum() < 8:
        return {"xi_meas": float("nan"), "R_fit": float("nan"), "n_fit": int(sel.sum())}
    rr = r[sel]
    pp = -psi[sel]  # inside positive, like tanh((R - r)/xi)
    # tanh((R - r)/xi) = tanh(u) -> u = artanh(p); linear in r
    pp = np.clip(pp, -0.999, 0.999)
    u = np.arctanh(pp)
    A = np.stack([rr, np.ones_like(rr)], axis=1)
    slope, intercept = np.linalg.lstsq(A, u, rcond=None)[0]
    xi_fit = 1.0 / slope if slope != 0.0 else float("nan")
    R_fit = -intercept * xi_fit
    return {"xi_meas": float(xi_fit), "R_fit": float(R_fit), "n_fit": int(sel.sum())}


# ---------------------------------------------------------------------------
# Calibration / cross-validation
# ---------------------------------------------------------------------------


def fit_sigma_through_origin(dp: np.ndarray, inv_radius: np.ndarray) -> float:
    """Single-unknown least squares ``dp_i = sigma / R_i`` -> ``sigma = sum(x y)/sum(x^2)``.

    Identical definition to ``tensorlbm.mcmp_verification._fit_laplace_level``, so
    the droplet-calibration numbers of the two anchors are comparable.
    """
    x = np.asarray(inv_radius, dtype=np.float64)
    y = np.asarray(dp, dtype=np.float64)
    return float(np.sum(x * y) / np.sum(x * x))


def free_line_fit(dp: np.ndarray, inv_radius: np.ndarray) -> dict[str, float]:
    """Diagnostic-only free line ``dp = a/R + b``; the intercept is the curvature bias."""
    x = np.asarray(inv_radius, dtype=np.float64)
    y = np.asarray(dp, dtype=np.float64)
    M = np.stack([x, np.ones_like(x)], axis=1)
    (a, b), *_ = np.linalg.lstsq(M, y, rcond=None)
    yhat = M @ np.array([a, b])
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "slope": float(a),
        "intercept": float(b),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "intercept_over_slope_times_Rmax": abs(b) * float(np.max(1.0 / x)) / abs(a),
    }


def leave_one_radius_out(radii: list[float], dp: list[float]) -> list[dict[str, float]]:
    """Fit ``sigma`` on all radii but one, predict ``dp`` at the held-out radius.

    This is the *cross-validation* half of the anchor: the calibration radii are
    not allowed to explain themselves.
    """
    out = []
    radii = list(map(float, radii))
    dp = list(map(float, dp))
    for i in range(len(radii)):
        tr_r = [radii[j] for j in range(len(radii)) if j != i]
        tr_dp = [dp[j] for j in range(len(radii)) if j != i]
        if len(tr_r) < 2:
            continue
        sig = fit_sigma_through_origin(np.array(tr_dp), 1.0 / np.array(tr_r))
        pred = sig / radii[i]
        out.append(
            {
                "held_out_radius": radii[i],
                "sigma_train": sig,
                "dp_pred": pred,
                "dp_meas": dp[i],
                "err_pct": 100.0 * (pred - dp[i]) / dp[i],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Planted defects (falsifiability of the analysis chain itself)
# ---------------------------------------------------------------------------

DEFECT_DESCRIPTIONS = {
    "sphere_factor_in_2d": "use the 3D sphere factor 2 sigma/R on the 2D cylinder data",
    "sigma_coef_half_kappa": "analytic sigma from kappa/2 (drops the OSS 0.75 factor chain)",
    "double_count_interface": "gradient-energy integral over a full diameter (two interfaces) -> 2 sigma",
    "mask_leak_into_interface": "inside mask reaches R + xi, i.e. averages over the interface shell",
    "radius_from_initial": "use the initial radius instead of the measured R_eq in the fit",
}


def planted_defects(
    cases: list[dict[str, Any]],
    *,
    shell: float = 1.5,
) -> dict[str, dict[str, float]]:
    """Apply five planted defects to the analysis of *measured* numbers.

    Each defect is a plausible implementation mistake; the falsifiability gate
    asks how many of them the 15 % tolerance actually rejects.  Every entry is
    computed from the same measured fields, so a defect that stays below the
    tolerance is reported as *not caught* instead of being quietly dropped.
    """
    valid = [c for c in cases if c["measures"]["dp_eq"] == c["measures"]["dp_eq"]]
    out: dict[str, dict[str, float]] = {}

    # 1. wrong dimensionality factor
    errs = [
        100.0 * (2.0 * c["measures"]["dp_eq"] - c["ref"]["dp_laplace"]) / c["ref"]["dp_laplace"]
        for c in valid
    ]
    out["sphere_factor_in_2d"] = {"max_abs_err_pct": max(abs(e) for e in errs)}

    # 2. halved kappa in the analytic tension -> sigma_coef/sqrt(2)... use exact formula
    errs = []
    for c in valid:
        s2 = analytic_sigma_from_coefficients(
            c["analytic"]["kappa"] / 2.0, c["analytic"]["A"], c["analytic"]["B"]
        )
        dp2 = laplace_reference_2d(s2, c["measures"]["R_eq"])
        errs.append(100.0 * (c["measures"]["dp_eq"] - dp2) / dp2)
    out["sigma_coef_half_kappa"] = {"max_abs_err_pct": max(abs(e) for e in errs)}

    # 3. sigma_cap doubled (full diameter cut crosses two interfaces)
    errs = [
        100.0 * (2.0 * c["measures"]["sigma_cap"] - c["analytic"]["sigma_analytic"])
        / c["analytic"]["sigma_analytic"]
        for c in valid
    ]
    out["double_count_interface"] = {"max_abs_err_pct": max(abs(e) for e in errs)}

    # 4. mask leak: recompute dp with the inside mask reaching into the interface
    errs = []
    for c in valid:
        leaked = c["_fields"]
        r = leaked["r"]
        p = leaked["p"]
        w = c["analytic"]["xi"]
        inside = r < c["measures"]["R_eq"] + w
        outside = r > c["measures"]["R_eq"] + shell * w
        if not bool(inside.any()):
            continue
        dp_leak = float(np.mean(p[inside]) - np.mean(p[outside]))
        errs.append(100.0 * (dp_leak - c["ref"]["dp_laplace"]) / c["ref"]["dp_laplace"])
    out["mask_leak_into_interface"] = {"max_abs_err_pct": max(abs(e) for e in errs) if errs else float("nan")}

    # 5. radius convention: initial radius instead of measured R_eq
    errs = []
    for c in valid:
        dp_init = c["measures"]["dp_fixed"]
        ref_init = laplace_reference_2d(c["analytic"]["sigma_analytic"], c["params"]["radius"])
        errs.append(100.0 * (dp_init - ref_init) / ref_init)
    out["radius_from_initial"] = {"max_abs_err_pct": max(abs(e) for e in errs)}

    for k, v in out.items():
        v["description"] = DEFECT_DESCRIPTIONS[k]
    return out


def strip_fields(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop heavy field arrays before JSON serialisation."""
    return [{k: v for k, v in c.items() if not k.startswith("_")} for c in cases]