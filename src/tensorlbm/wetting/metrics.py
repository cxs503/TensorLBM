"""Measurement operators for the wetting verification ladder (IN32-L01).

Every routine here is a *measurement* on a simulated phase field; no physics
constants are hard-coded except the analytic relations they are checked
against (all such relations are documented in the docstring and evaluated from
the simulated field, not from the input parameters).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .. import cac_lbm2d as cac


# ---------------------------------------------------------------------------
# Interface tension / Young-Laplace
# ---------------------------------------------------------------------------


def flat_interface_tension(sim: "cac.CAC2D", pad: int = 8) -> dict:
    """Interfacial tension of a *flat* interface from the simulated profile.

    ``sigma = int (f(phi) + (k/2) |dphi/dy|^2) dy`` with
    ``f = beta phi^2 (phi-1)^2``.  At equilibrium the two terms are equal
    (first integral), so their separate integrals are also reported as a
    self-consistency check.
    """
    m = cac.macro(sim)
    phi = m["phi"].mean(dim=1)  # column average -> (ny,)
    gx, gy = m["grad"][0].mean(dim=1), m["grad"][1].mean(dim=1)
    f = sim.beta * phi * phi * (phi - 1.0) ** 2
    grad2 = 0.5 * sim.k * (gx * gx + gy * gy)
    sl = slice(pad, len(phi) - pad)
    return {
        "sigma_integral": float((f[sl] + grad2[sl]).sum().item()),
        "sigma_from_well": float(f[sl].sum().item()),
        "sigma_from_gradient": float(grad2[sl].sum().item()),
        "phi_min": float(phi.min().item()),
        "phi_max": float(phi.max().item()),
        "width_fit": _width_from_profile(phi, sim.width),
    }


def _width_from_profile(phi: np.ndarray, nominal_width: float) -> float:
    """Interface width implied by the tanh ansatz ``phi = 0.5(1+tanh(2(y0-y)/W))``.

    ``phi=0.9`` and ``phi=0.1`` are separated by ``W * atanh(0.8)``, so the
    measured 10-90 thickness divided by ``atanh(0.8)`` gives ``W``.
    """
    phi = np.asarray(phi, dtype=float)
    y = np.arange(len(phi), dtype=float)

    def crossing(level):
        for i in range(len(phi) - 1):
            a, b = phi[i], phi[i + 1]
            if (a - level) * (b - level) < 0:
                return y[i] + (a - level) / (a - b)
        return float("nan")

    y90, y10 = crossing(0.9), crossing(0.1)
    if not np.isfinite(y90) or not np.isfinite(y10):
        return float("nan")
    return float(abs(y10 - y90) / np.arctanh(0.8))


def laplace_drop(sim: "cac.CAC2D", lo: float = 0.9, hi: float = 0.1) -> dict:
    """Young-Laplace measurement on a circular (2D) liquid drop.

    ``Req = sqrt(A/pi)`` from the ``phi > 0.5`` area, ``dp = <p>_in - <p>_out``
    with bulk bands ``phi > lo`` / ``phi < hi``, and ``sigma_rec = dp * Req``
    (2D: ``dp = sigma / R``).
    """
    m = cac.macro(sim)
    phi, p = m["phi"], m["p"]
    inside = phi > 0.5
    area = float(inside.sum().item())
    req = math.sqrt(area / math.pi)
    pin = p[phi > lo]
    pout = p[phi < hi]
    dp = float(pin.mean().item() - pout.mean().item())
    speed = (m["u"][0] ** 2 + m["u"][1] ** 2).sqrt()
    return {
        "area": area,
        "R_eq": req,
        "p_in": float(pin.mean().item()),
        "p_out": float(pout.mean().item()),
        "p_in_std": float(pin.std().item()),
        "dp": dp,
        "sigma_rec": dp * req,
        "u_max": float(speed.max().item()),
        "phi_min": float(phi.min().item()),
        "phi_max": float(phi.max().item()),
    }


# ---------------------------------------------------------------------------
# Static contact angle
# ---------------------------------------------------------------------------


def interface_contour(phi: np.ndarray, y_wall: float, level: float = 0.5) -> np.ndarray:
    """Upper branch of the ``phi = level`` contour, as (x, y) pairs.

    Scans every column upward from the wall and takes the *topmost* crossing
    (for a drop resting on a plate the interface above the wall is
    single-valued in x for contact angles below ~120 deg; the fit mask in
    :func:`contact_angle_circle_fit` further trims the contact-line region).
    """
    ny, nx = phi.shape
    pts = []
    for j in range(nx):
        col = phi[:, j]
        # highest index where col crosses `level` from below
        above = col > level
        if not above.any():
            continue
        idx = np.where(above)[0]
        i0, i1 = idx[0], idx[-1]
        if i1 + 1 >= ny:
            continue
        a, b = col[i1], col[i1 + 1]
        if a == b:
            continue
        y = i1 + (a - level) / (a - b)
        if y > y_wall:
            pts.append((float(j), float(y)))
    return np.array(pts)


def fit_circle(pts: np.ndarray) -> tuple[float, float, float, float]:
    """Algebraic (Kasa) circle fit.  Returns (xc, yc, R, rms_residual)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = sol[0], sol[1]
    R = math.sqrt(max(sol[2] + xc * xc + yc * yc, 1e-30))
    rms = float(np.sqrt(np.mean((np.hypot(x - xc, y - yc) - R) ** 2)))
    return float(xc), float(yc), float(R), rms


def contact_angle_circle_fit(
    phi: np.ndarray,
    y_wall: float,
    trim: float = 0.05,
    level: float = 0.5,
) -> dict:
    """Apparent contact angle from a circle fit of the drop profile.

    ``trim`` removes the fraction of the contour closest to the wall on each
    side (where the diffuse interface is bent by the wall condition).

    A sessile lens is a *small* circular cap above the wall, so its centre lies
    ``R cos(theta)`` **below** the wall and the angle through the liquid is

        theta = arccos((y_wall - y_c) / R)

    (the radius to the contact point is perpendicular to the interface tangent;
    ``y_c = y_wall`` is the neutral 90 deg case, ``y_c -> -inf`` the fully
    wetting limit).
    """
    pts = interface_contour(phi, y_wall, level)
    if len(pts) < 8:
        raise ValueError("too few contour points")
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    keep = (pts[:, 0] > x_min + trim * (x_max - x_min)) & (pts[:, 0] < x_max - trim * (x_max - x_min))
    use = pts[keep] if keep.sum() >= 8 else pts
    xc, yc, R, rms = fit_circle(use)
    arg = (y_wall - yc) / R
    theta = math.degrees(math.acos(max(-1.0, min(1.0, arg))))
    return {
        "theta_deg": theta,
        "x_center": xc,
        "y_center": yc,
        "R_fit": R,
        "rms": rms,
        "n_points": int(len(use)),
        "contact_width": float(x_max - x_min),
        "y_wall": y_wall,
    }


def spherical_cap_inversion(phi: np.ndarray, y_wall: float, level: float = 0.5) -> dict:
    """Independent angle estimator: invert the 2D circular-segment geometry.

    The sessile lens is a small circular cap: with the centre ``R cos(theta)``
    below the wall the liquid area and the contact width are

        ``A = R^2 (theta - sin theta cos theta)``,  ``w = 2 R sin theta``

    (``theta = 90`` deg gives a half disc, ``A = pi R^2 / 2``).  ``A`` and ``w``
    are measured from the phase field, so ``theta`` follows from a 1-D root
    solve and is an estimator independent of the circle fit.

    Both quantities must come from the *same* sharp-cap geometry:

    * ``w`` from the two rows adjacent to the wall: the chord of the
      ``phi = level`` isocontour at rows ``i0`` and ``i0+1`` is linearly
      extrapolated to the wall plane ``y = y_wall`` (the isocontour inside the
      first fluid row already sits half a cell above the wall).  Fallback, when
      the cap is thinner than one lattice cell above ``i0``: the single-row
      chord, flagged ``extrapolation="single_row"``.
    * ``A`` from the column integral ``int (y_if(x) - y_wall) dx`` of the same
      isocontour (``interface_contour``), i.e. the area of the cap above the
      wall plane.

    .. note:: **R5 correction** (2026-11, anchor
       ``XVAL-LBM-AKAI2018-GEOMETRICAL-WETTING``).  The previous version measured
       ``A`` as the plain sum of ``phi`` over every row below the wall and ``w``
       as the number of liquid columns plus one.  For a colour-gradient field
       (``phi in [-1, 1]``, vapour at ``-1``) that sum is ``2 A_liquid -
       A_domain``: it mixes the vapour into the "cap area", and the ``+1`` cell
       inflates the width.  On exact diffused caps of known angle the old
       estimator was off by a median 122 deg and failed its root bracket in 156
       of 162 cases (``benchmarks/xval_akai2018_wetting/estimator_analytic_study.py``).
       The form below returns the prescribed angle to within 2 deg over
       ``theta in [30, 150] deg``, ``R in [10, 40]``, ``xi in [1, 3]`` and to
       within 0.5 deg for ``R >= 20``.
    """
    ny, nx = phi.shape
    i0 = int(math.floor(y_wall)) + 1
    if i0 + 1 >= ny:
        raise ValueError(f"wall plane y={y_wall} leaves no fluid rows (ny={ny})")
    x_center = (nx - 1) / 2.0

    def _chord(row: np.ndarray) -> tuple[float, float] | None:
        xs = [
            j + (level - row[j]) / (row[j + 1] - row[j])
            for j in range(nx - 1)
            if (row[j] - level) * (row[j + 1] - level) < 0.0
        ]
        if len(xs) < 2:
            return None
        # leftmost/rightmost crossing: NOT ``min/max(key=|x - x_center|)``,
        # which ties on a symmetric cap (both sides equally far) and returns
        # the same crossing twice, giving a zero-width chord.
        return min(xs), max(xs)

    c0 = _chord(phi[i0, :])
    if c0 is None:
        raise ValueError("no interface crossing in the first fluid row")
    c1 = _chord(phi[i0 + 1, :])
    if c1 is None:
        xl, xr = c0
        extrapolation = "single_row"
    else:
        f = (y_wall - i0) / ((i0 + 1) - i0)  # <= 0, linear extrapolation to the wall
        xl = c0[0] + (c1[0] - c0[0]) * f
        xr = c0[1] + (c1[1] - c0[1]) * f
        extrapolation = "two_row"
    w = xr - xl
    if not math.isfinite(w) or w <= 0.0:
        raise ValueError(f"degenerate contact width (w={w})")

    pts = interface_contour(phi, y_wall, level)
    if len(pts) < 8:
        raise ValueError(f"too few contour points ({len(pts)}) for the cap area")
    pts = pts[np.argsort(pts[:, 0])]
    xs_c = np.concatenate([[xl], pts[:, 0], [xr]])
    hs_c = np.concatenate([[0.0], np.clip(pts[:, 1] - y_wall, 0.0, None), [0.0]])
    order = np.argsort(xs_c)
    if hasattr(np, "trapezoid"):
        area = float(np.trapezoid(hs_c[order], xs_c[order]))
    else:  # pragma: no cover - numpy < 2.0
        area = float(np.trapz(hs_c[order], xs_c[order]))
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError(f"degenerate cap area (A={area})")

    target = area / (w * w)

    def f(th: float) -> float:
        return (th - math.sin(th) * math.cos(th)) / (4.0 * math.sin(th) ** 2) - target

    lo, hi = 1e-6, math.pi - 1e-6
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0.0:
        raise ValueError(
            f"cap inversion bracket failed: f(lo)={flo:.3e}, f(hi)={fhi:.3e}, target={target:.3e}"
        )
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if flo * fm <= 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
        if hi - lo < 1e-13:
            break
    th = 0.5 * (lo + hi)
    R = w / (2.0 * math.sin(th))
    return {
        "theta_deg": math.degrees(th),
        "R_cap": R,
        "area": area,
        "contact_width": w,
        "x_left": float(xl),
        "x_right": float(xr),
        "extrapolation": extrapolation,
        "n_points": int(pts.shape[0]),
    }


def meniscus_height(phi: np.ndarray, x_lo: int, x_hi: int, level: float = 0.5) -> tuple[float, float]:
    """Height of the ``phi = level`` crossing in a channel column band.

    Returns (mean crossing height, column-to-column std) over ``x in [x_lo,
    x_hi)``.  The crossing is interpolated between the last liquid and the
    first gas node of each column.
    """
    hs = []
    for j in range(x_lo, x_hi):
        col = phi[:, j]
        above = col > level
        if not above.any():
            continue
        i1 = np.where(above)[0][-1]
        if i1 + 1 >= len(col):
            continue
        a, b = col[i1], col[i1 + 1]
        hs.append(i1 + (a - level) / (a - b) if a != b else float(i1))
    if not hs:
        return float("nan"), float("nan")
    return float(np.mean(hs)), float(np.std(hs))


# ---------------------------------------------------------------------------
# Oscillating drop (Lamb / Rayleigh)
# ---------------------------------------------------------------------------


def radial_extents(phi: np.ndarray, level: float = 0.5) -> tuple[float, float, float]:
    """``Rx``, ``Ry`` and equivalent radius from the ``phi > level`` mask."""
    lum = phi > level
    ys, xs = np.nonzero(lum)
    if len(xs) == 0:
        return float("nan"), float("nan"), float("nan")
    rx = 0.5 * (xs.max() - xs.min() + 1)
    ry = 0.5 * (ys.max() - ys.min() + 1)
    r_eq = math.sqrt(float(lum.sum()) / math.pi)
    return float(rx), float(ry), r_eq


def damped_sine_fit(t: np.ndarray, signal: np.ndarray, omega_guess: float) -> dict:
    """Least-squares fit of ``A e^{-gamma t} cos(omega t + phi) + c``.

    The non-linear parameters (gamma, omega) are scanned around the guess and
    the linear ones (A cos, A sin, c) are solved exactly for each candidate
    (variable projection); the reported frequency is the *damped* one and the
    Rayleigh/Lamb comparison uses it directly.
    """
    t = np.asarray(t, dtype=float)
    s = np.asarray(signal, dtype=float)
    best = None
    for gamma in np.linspace(0.0, 4.0 * max(omega_guess, 1e-9) / 4.0, 41):
        env = np.exp(-gamma * t)
        for om in np.linspace(0.6 * omega_guess, 1.4 * omega_guess, 121):
            c, sn = np.cos(om * t), np.sin(om * t)
            A = np.column_stack([env * c, env * sn, np.ones_like(t)])
            coef, *_ = np.linalg.lstsq(A, s, rcond=None)
            resid = A @ coef - s
            rss = float(resid @ resid)
            if best is None or rss < best["rss"]:
                amp = math.hypot(coef[0], coef[1])
                ph = math.atan2(-coef[1], coef[0])
                best = {"rss": rss, "gamma": float(gamma), "omega": float(om), "amp": amp, "phase": ph, "offset": float(coef[2])}
    ss_tot = float(((s - s.mean()) ** 2).sum())
    best["r2"] = 1.0 - best["rss"] / ss_tot if ss_tot > 0 else float("nan")
    best["omega0"] = math.sqrt(best["omega"] ** 2 + best["gamma"] ** 2)
    return best


# ---------------------------------------------------------------------------
# Cox-Voinov (dynamic contact angle)
# ---------------------------------------------------------------------------


def cox_voinov_fit(ca: np.ndarray, theta_deg: np.ndarray, theta_e_deg: float) -> dict:
    """Fit ``theta_d^3 = theta_e^3 + 9 Ca ln(L/Ls)`` to measured pairs.

    The model predicts a *linear* relation between ``theta_d^3`` and ``Ca``
    with slope ``9 ln(L/Ls) > 0`` and intercept ``theta_e^3``.  The fit returns
    the slope, the implied ``ln(L/Ls)``, the intercept, ``R^2`` and the mean
    absolute error of ``theta_d`` predicted back from ``Ca``.
    """
    ca = np.asarray(ca, dtype=float)
    th = np.radians(np.asarray(theta_deg, dtype=float))
    y = th**3
    A = np.column_stack([ca, np.ones_like(ca)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    rss = float(((pred - y) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    slope, icept = float(coef[0]), float(coef[1])
    th_pred = np.degrees(np.cbrt(np.clip(pred, 0.0, None)))
    return {
        "slope": slope,
        "ln_L_over_Ls": slope / 9.0 if slope > 0 else float("nan"),
        "intercept": icept,
        "theta_e_from_intercept_deg": math.degrees(math.cbrt(max(icept, 0.0))),
        "theta_e_input_deg": float(theta_e_deg),
        "r2": 1.0 - rss / ss_tot if ss_tot > 0 else float("nan"),
        "mae_theta_deg": float(np.mean(np.abs(th_pred - np.degrees(th)))),
        "n_points": int(len(ca)),
    }