"""Reduction utilities for the Re_tau = 180/395/590 channel-flow verification face.

Everything here is a *pure function* of arrays so the acceptance gates can be
unit-tested without running the solver (``tests/test_p223w3_n02_channel_dns.py``).

Conventions
-----------
* The native kernel (``src/tensorlbm/turbulence/les_chan.c``) writes profiles on
  rows ``y = 1 .. ny-2`` of a lattice whose walls are the half-way bounce-back
  planes ``y = 0.5`` and ``y = H + 0.5``.  The wall distance of row ``y`` is
  therefore ``y - 0.5`` lattice units, which is what :func:`wall_units` uses.
* Wall units use the friction velocity of the **body-force route**
  ``u_tau = sqrt(a * h)``, which is exact for the steady momentum balance of a
  force-driven channel and does not rely on any fit.  The independent
  ``stress_fit`` route is kept as a cross-check (gate f).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

KAPPA = 0.41
B_LOG = 5.2


# ---------------------------------------------------------------------------
# law of the wall
# ---------------------------------------------------------------------------

def log_law(y_plus, kappa: float = KAPPA, b: float = B_LOG) -> np.ndarray:
    """von Karman log law ``U+ = ln(y+)/kappa + B``."""
    y_plus = np.asarray(y_plus, dtype=float)
    return np.log(np.clip(y_plus, 1e-12, None)) / kappa + b


def reichardt(y_plus, kappa: float = KAPPA) -> np.ndarray:
    """Reichardt (1951) composite law of the wall (whole-channel closure)."""
    y_plus = np.asarray(y_plus, dtype=float)
    return (
        np.log1p(kappa * y_plus) / kappa
        + 7.8 * (1.0 - np.exp(-y_plus / 11.0) - (y_plus / 11.0) * np.exp(-y_plus / 3.0))
    )


def wall_units(y_index, u_tau: float, nu: float) -> np.ndarray:
    """Wall distance in wall units for the half-way bounce-back rows."""
    y_index = np.asarray(y_index, dtype=float)
    return (y_index - 0.5) * u_tau / nu


@dataclass(frozen=True)
class ProfileSet:
    """Mean/statistics profiles expressed in wall units."""

    y_plus: np.ndarray
    u_plus: np.ndarray
    uu_plus: np.ndarray
    vv_plus: np.ndarray
    ww_plus: np.ndarray
    uv_plus: np.ndarray
    y_over_h: np.ndarray
    u_mean: np.ndarray


def build_profiles(profiles: dict, derived: dict) -> ProfileSet:
    """Convert a raw kernel ``profiles`` block into wall units."""
    nu = float(derived["nu"])
    h = float(derived["half_height"])
    u_tau = float(derived["u_tau"])  # echoed from the configuration (body force)
    y = np.asarray(profiles["y"], dtype=float)
    yp = wall_units(y, u_tau, nu)
    return ProfileSet(
        y_plus=yp,
        u_plus=np.asarray(profiles["mean_ux"], dtype=float) / u_tau,
        uu_plus=np.asarray(profiles["r_uu"], dtype=float) / u_tau**2,
        vv_plus=np.asarray(profiles["r_vv"], dtype=float) / u_tau**2,
        ww_plus=np.asarray(profiles["r_ww"], dtype=float) / u_tau**2,
        uv_plus=np.asarray(profiles["r_uv"], dtype=float) / u_tau**2,
        y_over_h=(y - 0.5) / h,
        u_mean=np.asarray(profiles["mean_ux"], dtype=float),
    )


# ---------------------------------------------------------------------------
# independent friction-velocity routes
# ---------------------------------------------------------------------------

def stress_fit_u_tau(
    y_index, u_mean, uv, nu: float, h: float, lo: float = 0.10, hi: float = 0.70
):
    """Least-squares friction velocity from the total-stress budget.

    ``nu dU/dy - <u'v'> = u_tau^2 (1 - y/h)`` holds exactly for the steady
    force-driven channel, so a linear fit of the left-hand side against
    ``y/h`` over ``lo <= y/h <= hi`` returns ``u_tau`` from the intercept.
    """
    y_index = np.asarray(y_index, dtype=float)
    yw = y_index - 0.5
    u = np.asarray(u_mean, dtype=float)
    uv = np.asarray(uv, dtype=float)
    du = np.gradient(u, yw)
    total = nu * du - uv
    sel = (yw / h >= lo) & (yw / h <= hi)
    if sel.sum() < 3:
        raise ValueError("not enough rows for the stress fit")
    a = np.stack([np.ones(int(sel.sum())), yw[sel] / h], axis=1)
    coef, *_ = np.linalg.lstsq(a, total[sel], rcond=None)
    return math.sqrt(max(float(coef[0]), 1e-30)), total


def friction_routes(profiles: dict, derived: dict) -> dict:
    """u_tau from two independent routes plus the imposed value."""
    nu = float(derived["nu"])
    h = float(derived["half_height"])
    u_tau_bf = math.sqrt(float(derived["accel"]) * h)
    u_tau_sf, _ = stress_fit_u_tau(
        profiles["y"], profiles["mean_ux"], profiles["r_uv"], nu, h
    )
    return {
        "body_force": u_tau_bf,
        "stress_fit": u_tau_sf,
        "imposed": float(derived["u_tau"]),
        "spread_rel": abs(u_tau_sf - u_tau_bf) / u_tau_bf,
    }


# ---------------------------------------------------------------------------
# log-law / profile errors
# ---------------------------------------------------------------------------

def log_law_error(prof, y_lo: float = 30.0, y_hi: float = 200.0, kappa=KAPPA, b=B_LOG) -> dict:
    """Relative error of ``U+`` against the log law inside a y+ window."""
    yp, up = np.asarray(prof.y_plus), np.asarray(prof.u_plus)
    sel = (yp >= y_lo) & (yp <= y_hi)
    if sel.sum() < 2:
        return {"n_points": int(sel.sum()), "max_rel_error": float("nan"),
                "rms_rel_error": float("nan"), "k_fit": float("nan"),
                "b_fit": float("nan")}
    ref = log_law(yp[sel], kappa, b)
    rel = np.abs(up[sel] - ref) / np.abs(ref)
    # independent fit of the same window: U+ = ln(y+)/k + b
    design = np.stack([np.log(yp[sel]), np.ones(int(sel.sum()))], axis=1)
    (slope, intercept), *_ = np.linalg.lstsq(design, up[sel], rcond=None)
    return {
        "n_points": int(sel.sum()),
        "max_rel_error": float(rel.max()),
        "rms_rel_error": float(np.sqrt(np.mean(rel**2))),
        "k_fit": float(1.0 / slope),
        "b_fit": float(intercept),
    }


def reichardt_error(prof, y_lo: float = 5.0, y_hi: float = 200.0) -> dict:
    yp, up = np.asarray(prof.y_plus), np.asarray(prof.u_plus)
    sel = (yp >= y_lo) & (yp <= y_hi)
    if sel.sum() < 2:
        return {"n_points": int(sel.sum()), "max_rel_error": float("nan")}
    ref = reichardt(yp[sel])
    rel = np.abs(up[sel] - ref) / np.abs(ref)
    return {"n_points": int(sel.sum()), "max_rel_error": float(rel.max()),
            "rms_rel_error": float(np.sqrt(np.mean(rel**2)))}


def peak(values, y_plus) -> dict:
    """Peak magnitude and its wall-normal location."""
    v = np.asarray(values, dtype=float)
    yp = np.asarray(y_plus, dtype=float)
    i = int(np.argmax(v))
    return {"value": float(v[i]), "y_plus": float(yp[i])}


# ---------------------------------------------------------------------------
# friction coefficient
# ---------------------------------------------------------------------------

def cf_from_bulk(u_tau: float, u_bulk: float) -> float:
    """Skin-friction coefficient ``Cf = 2 tau_w / (rho U_b^2) = 2 (u_tau/U_b)^2``."""
    return 2.0 * (u_tau / u_bulk) ** 2


def dean_cf(re_bulk):
    """Dean (1978) turbulent duct correlation ``Cf = 0.073 Re_b^-0.25``."""
    return 0.073 * np.asarray(re_bulk, dtype=float) ** -0.25


def bulk_reynolds(u_bulk: float, height: float, nu: float) -> float:
    """Bulk Reynolds number on the full channel height ``Re_b = U_b 2h / nu``."""
    return u_bulk * height / nu


# ---------------------------------------------------------------------------
# streamwise energy spectrum
# ---------------------------------------------------------------------------

def streamwise_spectrum(plane_ux, lx: float) -> tuple[np.ndarray, np.ndarray]:
    """One-dimensional streamwise spectrum of ``u_x`` on an x-z plane.

    ``plane_ux`` has shape ``(nz, nx)``.  The mean is removed in ``x``, a
    Hann window is applied, the power spectral density is averaged over the
    span, and the result is normalised so that ``sum(E) * dk = <u'u'>``.
    """
    u = np.asarray(plane_ux, dtype=float)
    if u.ndim != 2:
        raise ValueError("plane_ux must be 2-D (nz, nx)")
    nz, nx = u.shape
    u = u - u.mean(axis=1, keepdims=True)
    win = np.hanning(nx)
    win /= np.sqrt(np.mean(win**2))
    spec = np.fft.rfft(u * win, axis=1)
    psd = (np.abs(spec) ** 2) / nx**2
    e = psd.mean(axis=0)
    n = np.arange(e.size)
    dk = 2.0 * np.pi / lx
    k = n * dk
    # one-sided: double the interior bins so sum(E)*dk == <u'u'>
    e = e.copy()
    e[1:] *= 2.0
    sel = n >= 1
    return k[sel], e[sel] / dk


def power_law_slope(k, e, k_min: float, k_max: float) -> dict:
    """Least-squares slope of ``log E`` against ``log k`` inside a band."""
    k = np.asarray(k, dtype=float)
    e = np.asarray(e, dtype=float)
    sel = (k >= k_min) & (k <= k_max) & (e > 0.0)
    if sel.sum() < 3:
        return {"slope": float("nan"), "n": int(sel.sum()), "k_min": k_min, "k_max": k_max}
    slope, intercept = np.polyfit(np.log(k[sel]), np.log(e[sel]), 1)
    return {"slope": float(slope), "intercept": float(intercept),
            "n": int(sel.sum()), "k_min": k_min, "k_max": k_max}


def spectrum_from_csv(csv_path, lx: float) -> tuple[np.ndarray, np.ndarray]:
    """Load a mid-plane ``*_xzm.csv`` slice and return its streamwise spectrum."""
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    x = data["x"].astype(int)
    z = data["z"].astype(int)
    ux = data["ux"].astype(float)
    nx = int(x.max()) + 1
    nz = int(z.max()) + 1
    plane = np.zeros((nz, nx))
    plane[z, x] = ux
    return streamwise_spectrum(plane, lx)


# ---------------------------------------------------------------------------
# laminar negative control
# ---------------------------------------------------------------------------

def poiseuille_exact(y_index, height: float, accel: float, nu: float) -> np.ndarray:
    """Exact plane-Poiseuille profile of a force-driven laminar channel."""
    yw = np.asarray(y_index, dtype=float) - 0.5
    return accel * yw * (height - yw) / (2.0 * nu)


def relative_l2(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def relative_max(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30)))