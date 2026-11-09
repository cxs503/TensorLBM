"""Contact-angle estimators used by the static wetting benchmark, with self-validation.

Three independent estimators of ``theta`` (the angle measured *inside the
liquid* between the ``rho = (rho_l + rho_v)/2`` level set and the wall plane):

``cap``
    Integral (spherical-cap inversion, the "centroid/area" family): the liquid
    content area above the wall ``A`` and the *apex* height ``h`` of the level
    set above the wall plane fix the angle of the equivalent circular segment

        A = R^2 (theta - sin(theta) cos(theta)),   h = R (1 - cos theta)
        =>  theta = g^-1(A / h^2),  g(theta) = (theta - sin theta cos theta)/(1 - cos theta)^2

    ``g`` decreases monotonically from +inf (theta -> 0) to 0 (theta -> pi), so
    the inversion is unique.  Both observables are integrals away from the wall:
    ``h`` is measured at the apex (far from the one-cell wall depletion layer
    the pseudopotential wall leaves behind) and ``A`` is a bulk integral, so
    this estimator is the one the benchmark accepts/rejects on.  It is also
    what makes small angles measurable: the level set of a strong wetting case
    sits *above* the wall plane, so a contact-width observable taken on the
    first fluid row does not exist there (measured: it returns NaN for the whole
    0.30-0.40 wall-fraction range), while ``A`` and ``h`` stay well defined.

``circle``
    Whole-cap algebraic circle fit (:func:`tensorlbm.contact_line.theta_fit.circle_fit`)
    extrapolated to the wall.  The classical estimator; it is exact for a clean
    circular cap but drifts once the drop develops a flat foot or a wetting film
    (both are reported, not hidden, via its ``rms`` and the estimator spread).

``band``
    Near-wall local circle fit over a *scale-aware* set of fit windows
    (:func:`tensorlbm.contact_line.theta_fit.angle_estimate` with windows tied to
    the initial radius ``R`` instead of absolute lattice heights).  Reads the
    slope of the interface where it actually meets the wall, at the price of a
    larger statistical error.

Convention trap (measured, see ``README.md``)
---------------------------------------------
Every contour estimator shipped in :mod:`tensorlbm.contact_line.theta_fit`
(:func:`~tensorlbm.contact_line.theta_fit.contact_angle_from_circle`,
:func:`~tensorlbm.contact_line.theta_fit.angle_estimate`,
:func:`~tensorlbm.contact_line.theta_fit.local_tangent_angle`) evaluates
``theta = atan2(a, y_c - y_wall)`` / ``atan2(1, dx/dy)``, which returns the
**supplement** of the physical angle measured inside the liquid: a wetting drop
whose interface leaves the wall at 30 deg is reported as 150 deg (for a
semicircle the two conventions coincide, which is why the difference is easy to
miss).  This module reports the *physical* angle -- the one the Young relation
``cos(theta) = (psi_w - psi_v)/(psi_l - psi_v)`` and the wall fraction knob are
defined against -- and converts the library estimators by ``180 - theta_lib``
(``LIB_IS_SUPPLEMENT``).  ``test_wetting_contact_angle.py`` pins both facts.

Agreement of the three (``spread``) is the instrument's own uncertainty floor: on
an analytic cap all three reproduce the true angle to <1 deg (asserted in
``test_wetting_contact_angle.py``), so on a simulated field a larger spread
flags a shape the estimators were not designed for (foot/film/neck) instead of
silently returning a wrong number.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from tensorlbm.contact_line.theta_fit import (
    angle_estimate,
    circle_fit,
    contact_angle_from_circle,
    interface_points,
)

#: The library's contour estimators return ``180 - theta_physical``.
LIB_IS_SUPPLEMENT = True

__all__ = [
    "LIB_IS_SUPPLEMENT",
    "cap_inversion",
    "measure_contact_angle",
    "synthetic_cap_density",
]


# ---------------------------------------------------------------------------
# integral (cap) inversion
# ---------------------------------------------------------------------------


def _cap_shape_factor(theta: float) -> float:
    """``(theta - sin theta cos theta) / sin^2 theta`` — monotone on (0, pi)."""
    s = math.sin(theta)
    return (theta - s * math.cos(theta)) / (s * s)


def _cap_shape_factor_area_height(theta: float) -> float:
    """``(theta - sin theta cos theta) / (1 - cos theta)^2`` — monotone on (0, pi)."""
    c = math.cos(theta)
    return (theta - math.sin(theta) * c) / ((1.0 - c) ** 2)


def _apex_height(frac: np.ndarray, i_row: int, level: float) -> float:
    """Sub-pixel height of the ``frac = level`` contour above the first fluid row."""
    best = float("nan")
    for j in range(frac.shape[1]):
        col = frac[i_row:, j]
        above = col >= level
        if not above.any():
            continue
        i_hi = int(np.nonzero(above)[0][-1])          # topmost liquid cell of this column
        if i_hi + 1 >= col.size:                       # touches the top wall: not a drop apex
            continue
        f_in, f_out = col[i_hi], col[i_hi + 1]
        y = float(i_row + i_hi) + ((f_in - level) / (f_in - f_out) if f_in != f_out else 0.0)
        if not math.isfinite(best) or y > best:
            best = y
    return best


def cap_inversion(
    rho: np.ndarray | torch.Tensor,
    *,
    rho_l: float,
    rho_v: float,
    y_wall: float,
    solid: np.ndarray | torch.Tensor | None = None,
    level: float = 0.5,
) -> dict[str, float]:
    """Angle from liquid-content area + apex height (equivalent circular segment).

    Args:
        rho:    Density field ``(ny, nx)`` (row index = y, increasing upward).
        rho_l:  Liquid density of the coexistence pair.
        rho_v:  Vapour density.
        y_wall: Wall plane height in cell-centre coordinates (``wall_rows - 0.5``).
        solid:  Optional boolean wall mask ``(ny, nx)``.
        level:  Apex level in *normalised liquid fraction* (0.5 = mid-density).

    Returns:
        ``{"theta_deg", "h", "area", "a_row"}``; ``theta_deg`` is NaN when no
        liquid is found above the first fluid row (fully dewetted wall).
    """
    rho_np = np.asarray(rho, dtype=np.float64)
    if solid is None:
        solid_np = np.zeros_like(rho_np, dtype=bool)
    else:
        solid_np = np.asarray(solid, dtype=bool)
    frac = np.clip((rho_np - rho_v) / (rho_l - rho_v), 0.0, 1.0)
    frac = np.where(solid_np, 0.0, frac)
    area = float(frac.sum())
    i_row = int(math.floor(float(y_wall))) + 1  # first fluid row above the wall
    y_apex = _apex_height(frac, i_row, level)
    h = y_apex - float(y_wall)
    if not math.isfinite(h) or h <= 0.5 or area <= 0.0:
        return {"theta_deg": float("nan"), "h": float(h), "area": area, "a_row": float("nan")}
    ratio = area / (h * h)
    lo, hi = 1e-6, math.pi - 1e-6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _cap_shape_factor_area_height(mid) > ratio:
            lo = mid
        else:
            hi = mid
    theta = 0.5 * (lo + hi)
    # contact half-width at the first fluid row (diagnostic only, NaN for films)
    row = frac[i_row]
    above = row >= level
    a_row = float("nan")
    if above.any():
        idx = np.nonzero(above)[0]
        a_row = 0.5 * float(idx[-1] - idx[0])
    return {"theta_deg": math.degrees(theta), "h": float(h), "area": area, "a_row": a_row}


# ---------------------------------------------------------------------------
# contour estimators
# ---------------------------------------------------------------------------


def _band_windows(radius: float) -> tuple[float, ...]:
    """Fit-band heights scaled to the drop radius (absolute windows misfire for R < 15)."""
    r = float(radius)
    return tuple(max(2.5, f * r) for f in (0.30, 0.40, 0.50, 0.60))


def measure_contact_angle(
    rho: np.ndarray | torch.Tensor,
    *,
    rho_l: float,
    rho_v: float,
    y_wall: float,
    solid: np.ndarray | torch.Tensor | None = None,
    radius: float = 13.0,
    keep_points: bool = False,
) -> dict[str, object]:
    """All three estimators plus their spread on one density field."""
    rho_t = torch.as_tensor(np.asarray(rho, dtype=np.float32))
    solid_t = None if solid is None else torch.as_tensor(np.asarray(solid, dtype=bool))
    level_rho = 0.5 * (rho_l + rho_v)
    out: dict[str, object] = {}

    cap = cap_inversion(rho, rho_l=rho_l, rho_v=rho_v, y_wall=y_wall, solid=solid, level=0.5)
    out.update(theta_cap=cap["theta_deg"], apex_height=cap["h"], area=cap["area"],
               a_row=cap["a_row"])

    pts = interface_points(rho_t, level=level_rho, solid=solid_t)
    out["n_points"] = int(pts.shape[0])
    theta_c = theta_b = rms = float("nan")
    if pts.shape[0] >= 8:
        fit = circle_fit(pts)
        theta_c, _a = contact_angle_from_circle(fit, y_wall)
        theta_c = 180.0 - theta_c if LIB_IS_SUPPLEMENT else theta_c
        rms = float(fit.rms)
        try:
            est = angle_estimate(pts, y_wall, windows=_band_windows(radius), offset=1.5)
            theta_b = float(est.theta_deg)
            theta_b = 180.0 - theta_b if LIB_IS_SUPPLEMENT else theta_b
        except ValueError:
            theta_b = float("nan")
        if keep_points:
            stride = max(1, int(pts.shape[0]) // 240)
            out["points"] = [[round(float(x), 4), round(float(y), 4)]
                             for x, y in pts[::stride].tolist()]
    out.update(theta_circle=float(theta_c), theta_band=float(theta_b), rms_fit=float(rms))

    vals = [v for v in (out["theta_cap"], theta_c, theta_b) if isinstance(v, float) and math.isfinite(v)]
    out["spread"] = float(max(vals) - min(vals)) if len(vals) > 1 else float("nan")
    out["theta"] = float(out["theta_cap"]) if isinstance(out["theta_cap"], float) else float("nan")
    return out


# ---------------------------------------------------------------------------
# analytic reference (for estimator self-validation)
# ---------------------------------------------------------------------------


def synthetic_cap_density(
    *,
    theta_deg: float,
    radius: float,
    nx: int,
    ny: int,
    wall_rows: int = 2,
    rho_l: float = 1.957,
    rho_v: float = 0.1596,
    width: float = 1.5,
    centre_x: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Diffuse (tanh) density field of the *exact* circular cap of ``theta_deg``.

    Returns ``(rho, solid, y_wall)``.  The sharp cap is a circle of radius
    ``radius`` whose centre sits at ``y_wall - radius cos(theta)``: for
    ``theta < 90`` the centre is *below* the wall, which is what makes the drop a
    wide, squat lens rather than a bead, and the interface meets the wall plane
    at exactly ``theta`` (angle measured inside the liquid).
    """
    yy, xx = np.meshgrid(np.arange(ny, dtype=np.float64), np.arange(nx, dtype=np.float64),
                         indexing="ij")
    y_wall = wall_rows - 0.5
    cx = 0.5 * nx if centre_x is None else float(centre_x)
    cy = y_wall - radius * math.cos(math.radians(theta_deg))
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - radius
    rho = 0.5 * (rho_l + rho_v) - 0.5 * (rho_l - rho_v) * np.tanh(dist / width)
    solid = np.zeros((ny, nx), dtype=bool)
    solid[:wall_rows, :] = True
    solid[ny - wall_rows:, :] = True
    rho[solid] = rho_l
    return rho, solid, y_wall