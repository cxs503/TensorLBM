"""Quantitative metrics for the compressible-shock *family* verification surface.

The single-point Sod / normal-shock checks live in ``analysis`` and
``rankine_hugoniot``.  The family-level verification (Mach 2-10 scan, double
Mach reflection, forward-facing step) needs quantities that have an
*independent* reference rather than another number from the same code, so this
module supplies two things:

Theory (closed form, or solved from the exact shock relations)
--------------------------------------------------------------
``theta_max``                maximum attached deflection and the detachment
                             (sonic) angle ``beta*``
``sonic_shock_angle``        ``beta*`` by its definition ``M2(beta*) = 1``
``three_shock_slip_angle``   von Neumann three-shock theory: slip-line angle and
                             Mach-stem normal Mach number at the triple point of
                             a Mach reflection (pressure-matched shock polars)
``regular_reflection_angle`` two-shock theory: reflected-shock angle for a shock
                             of given angle hitting a plane wall

Field detectors (turn a 2-D field into one number with a documented rule)
-------------------------------------------------------------------------
``front_angle``              inclination of a fitted front segment
``slip_line_angle``          density-jump / pressure-continuity ray search from
                             the triple point
``vortex_core``              primary roll-up centre from the vorticity extremum
``sonic_point_on_front``     where the flow just behind the front is sonic, with
                             the local shock angle (detached bow shock)

Angles are in degrees.  "Front" angles are measured from the wall-parallel
x axis (0 deg = parallel to the wall, 90 deg = normal to it); shock angles
``beta`` follow the usual convention (angle between shock and upstream flow).
"""

from __future__ import annotations

import math

import numpy as np

from .rankine_hugoniot import GAMMA, normal_shock, oblique_shock, oblique_shock_beta

__all__ = [
    "theta_max",
    "sonic_shock_angle",
    "three_shock_slip_angle",
    "regular_reflection_angle",
    "front_angle",
    "slip_line_angle",
    "slip_line_angle_velocity",
    "vortex_core",
    "sonic_point_on_front",
    "bilinear",
]


# --------------------------------------------------------------------- theory
def _deflection(M1: float, beta: float, gamma: float = GAMMA) -> float:
    """Flow deflection [rad] across an oblique shock (theta-beta-M relation)."""
    s2 = math.sin(beta) ** 2
    t = (
        2.0
        / math.tan(beta)
        * (M1 * M1 * s2 - 1.0)
        / (M1 * M1 * (gamma + math.cos(2.0 * beta)) + 2.0)
    )
    return math.atan(t)


def theta_max(M1: float, gamma: float = GAMMA, n: int = 2000) -> dict:
    """Maximum attached deflection and the shock angle where it occurs.

    At ``beta = beta*`` the flow behind the shock is exactly sonic, so ``beta*``
    is also the *detachment* angle of a detached bow shock.  The maximum of
    theta(beta) on ``beta in (mu, pi/2)`` is located on a coarse grid and then
    refined by golden-section search.
    """
    if M1 <= 1.0:
        raise ValueError("theta_max requires supersonic M1 > 1")
    mu = math.asin(1.0 / M1)
    grid = np.linspace(mu + 1e-9, 0.5 * math.pi, n + 1)
    vals = np.array([_deflection(M1, b, gamma) for b in grid])
    k = int(np.argmax(vals))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, n)]
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(120):
        x1 = hi - gr * (hi - lo)
        x2 = lo + gr * (hi - lo)
        if _deflection(M1, x1, gamma) < _deflection(M1, x2, gamma):
            lo = x1
        else:
            hi = x2
    beta_star = 0.5 * (lo + hi)
    th_star = _deflection(M1, beta_star, gamma)
    m2n = normal_shock(M1 * math.sin(beta_star), gamma)["M2"]
    den = math.sin(beta_star - th_star)
    m2_full = m2n / den if den > 0 else float("nan")
    return {
        "M1": M1,
        "theta_max_deg": math.degrees(th_star),
        "beta_star_deg": math.degrees(beta_star),
        "mu_deg": math.degrees(mu),
        "M2_full_at_star": m2_full,
    }


def sonic_shock_angle(M1: float, gamma: float = GAMMA) -> dict:
    """Shock angle at which the flow behind the shock is sonic (``M2 = 1``).

    This is the *sonic point* of a detached bow shock; it is found by bisection
    on ``M2_full(beta) - 1`` and is therefore defined by the condition itself
    (``M2_full`` is returned and must be 1 to rounding).  It lies slightly below
    the maximum-deflection (detachment) angle of :func:`theta_max` -- the two
    are distinct conditions and both are reported.
    """
    mu = math.asin(1.0 / M1)
    lo, hi = mu + 1e-12, 0.5 * math.pi - 1e-12

    def m2_full(beta: float) -> tuple[float, float]:
        theta = _deflection(M1, beta, gamma)
        m2n = normal_shock(M1 * math.sin(beta), gamma)["M2"]
        den = math.sin(beta - theta)
        return (m2n / den if den > 0 else float("inf")), theta

    for _ in range(200):
        m = 0.5 * (lo + hi)
        v, _ = m2_full(m)
        if v > 1.0:
            lo = m
        else:
            hi = m
        if hi - lo < 1e-13:
            break
    beta = 0.5 * (lo + hi)
    v, theta = m2_full(beta)
    return {
        "M1": M1,
        "beta_sonic_deg": math.degrees(beta),
        "theta_at_sonic_deg": math.degrees(theta),
        "M2_full_at_sonic": v,
        "mu_deg": math.degrees(mu),
    }


def three_shock_slip_angle(
    shock_normal_mach: float,
    deflection_deg: float,
    gamma: float = GAMMA,
    tol: float = 1e-13,
) -> dict:
    """von Neumann three-shock theory for a Mach-reflection triple point.

    Inputs are the *measured* incident-shock quantities: its normal Mach number
    and the flow deflection it imposes.  Those two fix the upstream Mach number
    through ``M1 sin(beta_i) = shock_normal_mach`` together with the theta-beta-M
    relation, so the theory consumes an experimental/simulation input instead of
    reproducing the code's own output.

    Closure: region 2 (incident + reflected shock) and region 3 (Mach stem) must
    share the same pressure across the slip line, while both flows are parallel
    to it.  With the slip-line inclination ``chi`` measured from the wall
    (positive towards the wall side) the reflected shock turns the flow by
    ``theta_w - chi`` and the root of ``p2(chi) - p3(chi)`` is the solution.
    """
    th_w = math.radians(deflection_deg)
    ms = shock_normal_mach

    def resid(beta: float) -> float:
        M1 = ms / math.sin(beta)
        if M1 <= 1.0:
            return -1.0
        return _deflection(M1, beta, gamma) - th_w

    mu = math.asin(min(1.0, 1.0 / ms))
    betas = np.linspace(mu + 1e-10, 0.5 * math.pi - 1e-10, 4001)
    f = np.array([resid(b) for b in betas])
    s = np.sign(f)
    cross = np.flatnonzero(s[:-1] * s[1:] < 0.0)
    if cross.size == 0:
        raise ValueError("no incident-shock solution for this normal Mach / deflection")
    i = int(cross[0])
    a, b = betas[i], betas[i + 1]
    fa = f[i]
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = resid(m)
        if fa * fm <= 0.0:
            b = m
        else:
            a, fa = m, fm
        if abs(b - a) < tol:
            break
    beta_i = 0.5 * (a + b)
    M1 = ms / math.sin(beta_i)

    inc = oblique_shock(M1, th_w, gamma)
    M2i = inc["M2"]
    p2i = inc["p_ratio"]

    def gap(chi: float) -> tuple[float, dict]:
        d_r = th_w - chi
        if d_r <= 1e-12:
            return math.inf, {}
        try:
            rs = oblique_shock(M2i, d_r, gamma)
            stem = oblique_shock(M1, chi, gamma)
        except ValueError:
            return math.nan, {}
        p2 = p2i * rs["p_ratio"]
        p3 = stem["p_ratio"]
        return p2 - p3, {"reflected": rs, "stem": stem, "p2": p2, "p3": p3}

    chis = np.linspace(1e-4, th_w - 1e-3, 2000)
    gaps = np.array([gap(c)[0] for c in chis], dtype=float)
    ok = np.isfinite(gaps)
    idx = np.flatnonzero(ok[:-1] & ok[1:] & (np.sign(gaps[:-1]) * np.sign(gaps[1:]) < 0))
    if idx.size == 0:
        raise ValueError("three-shock closure failed: shock polars do not cross")
    i = int(idx[0])
    a, b = chis[i], chis[i + 1]
    fa = gaps[i]
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = gap(m)[0]
        if fa * fm <= 0.0:
            b = m
        else:
            a, fa = m, fm
        if abs(b - a) < 1e-15:
            break
    chi = 0.5 * (a + b)
    g, aux = gap(chi)
    stem, rs = aux["stem"], aux["reflected"]
    return {
        "shock_normal_mach": ms,
        "deflection_deg": deflection_deg,
        "M1": M1,
        "incident_beta_deg": math.degrees(beta_i),
        "incident_M2": M2i,
        "incident_p_ratio": p2i,
        "slip_line_angle_deg": math.degrees(chi),
        "mach_stem_beta_deg": stem["beta_deg"],
        "mach_stem_normal_mach": M1 * math.sin(stem["beta"]),
        "stem_p_ratio": stem["p_ratio"],
        "reflected_shock_beta_deg": rs["beta_deg"],
        "reflected_shock_turning_deg": math.degrees(th_w - chi),
        "pressure_residual": float(g),
    }


def regular_reflection_angle(
    M1: float, incident_beta_deg: float, gamma: float = GAMMA
) -> dict:
    """Two-shock (regular reflection) theory for a shock reflecting off a wall.

    A shock at angle ``beta_i`` in a Mach-``M1`` stream hits a plane wall; the
    wall forces the downstream flow to be parallel to it again, so the reflected
    shock removes exactly the incident deflection ``theta_i``.  ``M2`` and
    ``theta_i`` come from the exact oblique-shock solution, hence the reflected
    angle has no free parameter and can be compared to a measured front.
    """
    if incident_beta_deg <= 0.0:
        raise ValueError("incident_beta_deg must be positive")
    beta_i = math.radians(incident_beta_deg)
    theta_i = _deflection(M1, beta_i, gamma)
    jump = normal_shock(M1 * math.sin(beta_i), gamma)
    den = math.sin(beta_i - theta_i)
    M2 = jump["M2"] / den if den > 0 else float("inf")
    out = {
        "M1": M1,
        "incident_beta_deg": float(incident_beta_deg),
        "incident_theta_deg": math.degrees(theta_i),
        "M2": M2,
        "incident_p_ratio": jump["p_ratio"],
        "attached": False,
    }
    try:
        beta_r = oblique_shock_beta(M2, theta_i, gamma=gamma, weak=True)
    except ValueError as exc:
        return {**out, "reason": str(exc)}
    out.update(
        {
            "attached": True,
            "reflected_beta_deg": math.degrees(beta_r),
            "reflected_beta_from_wall_deg": math.degrees(beta_r - theta_i),
            "reflected_beta_wall_from_shock_deg": math.degrees(beta_r - theta_i),
        }
    )
    return out


# ------------------------------------------------------------------- detectors
def bilinear(field: np.ndarray, xc: np.ndarray, yc: np.ndarray, x, y):
    """Bilinear sample of ``field[j, i]`` at physical points ``(x, y)`` (nan outside)."""
    F = np.asarray(field, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = np.broadcast_arrays(x, y)
    ny, nx = F.shape
    dx = xc[1] - xc[0]
    dy = yc[1] - yc[0]
    fi = (x - xc[0]) / dx
    fj = (y - yc[0]) / dy
    i0 = np.floor(fi).astype(int)
    j0 = np.floor(fj).astype(int)
    inside = (i0 >= 0) & (i0 <= nx - 2) & (j0 >= 0) & (j0 <= ny - 2)
    i0c = np.clip(i0, 0, nx - 2)
    j0c = np.clip(j0, 0, ny - 2)
    tx = fi - i0c
    ty = fj - j0c
    val = (
        F[j0c, i0c] * (1 - tx) * (1 - ty)
        + F[j0c, i0c + 1] * tx * (1 - ty)
        + F[j0c + 1, i0c] * (1 - tx) * ty
        + F[j0c + 1, i0c + 1] * tx * ty
    )
    return np.where(inside, val, np.nan)


def front_angle(
    x_front: np.ndarray, yc: np.ndarray, y_lo: float, y_hi: float, min_points: int = 6
) -> dict:
    """Inclination of the leading front fitted on ``y in [y_lo, y_hi]``.

    ``x_front`` is the per-row front position (``analysis.front_profile``).
    Reports degrees from the x axis, the raw slope and the fit rms so the
    quality of the angle is auditable.
    """
    x = np.asarray(x_front, dtype=float)
    y = np.asarray(yc, dtype=float)
    sel = np.isfinite(x) & (y >= y_lo) & (y <= y_hi)
    if sel.sum() < max(min_points, 3):
        return {"angle_deg": float("nan"), "n_points": int(sel.sum()), "rms": float("nan")}
    slope, intercept = np.polyfit(y[sel], x[sel], 1)
    resid = x[sel] - (slope * y[sel] + intercept)
    return {
        "angle_deg": float(math.degrees(math.atan2(1.0, slope))),
        "dx_dy": float(slope),
        "intercept": float(intercept),
        "n_points": int(sel.sum()),
        "rms": float(np.sqrt(np.mean(resid**2))),
        "y_range": [float(y_lo), float(y_hi)],
    }


def slip_line_angle(
    rho: np.ndarray,
    p: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    origin: tuple[float, float],
    length: float,
    angles_deg=None,
    offset_cells: float = 1.5,
    n_samples: int = 80,
) -> dict:
    """Slip-line inclination from a density jump with pressure continuity.

    A slip line is a density discontinuity with (to leading order) continuous
    pressure.  Rays are cast downstream from the triple point; along each ray the
    normal derivatives ``|d rho/dn| / rho`` and ``|d p/dn| / p`` are sampled and
    scored by

        score(chi) = mean(|d rho/dn|/rho) - mean(|d p/dn|/p)

    so a density jump scores high while a shock (which has both) scores ~0.
    The maximiser is the slip-line angle; the whole curve is returned for audit.
    """
    x0, y0 = origin
    if angles_deg is None:
        angles_deg = np.arange(-25.0, 25.0001, 0.25)
    angles_deg = np.asarray(angles_deg, dtype=float)
    s = np.linspace(0.02 * length, length, n_samples)
    h = offset_cells * (xc[1] - xc[0])
    scores = np.full(angles_deg.shape, np.nan)
    drho_out = np.full(angles_deg.shape, np.nan)
    dp_out = np.full(angles_deg.shape, np.nan)
    for k, a in enumerate(angles_deg):
        th = math.radians(a)
        cx, cy = math.cos(th), math.sin(th)
        px = x0 + s * cx
        py = y0 + s * cy
        nx, ny = -cy, cx
        r_m = bilinear(rho, xc, yc, px + h * nx, py + h * ny)
        r_p = bilinear(rho, xc, yc, px - h * nx, py - h * ny)
        p_m = bilinear(p, xc, yc, px + h * nx, py + h * ny)
        p_p = bilinear(p, xc, yc, px - h * nx, py - h * ny)
        good = np.isfinite(r_m) & np.isfinite(r_p) & np.isfinite(p_m) & np.isfinite(p_p)
        if good.sum() < 5:
            continue
        r_loc = np.maximum(0.5 * (np.abs(r_m) + np.abs(r_p)), 1e-12)
        p_loc = np.maximum(0.5 * (np.abs(p_m) + np.abs(p_p)), 1e-12)
        drho = np.abs(r_m - r_p) / r_loc
        dp = np.abs(p_m - p_p) / p_loc
        scores[k] = float(np.mean(drho[good] - dp[good]))
        drho_out[k] = float(np.mean(drho[good]))
        dp_out[k] = float(np.mean(dp[good]))
    if not np.isfinite(scores).any():
        return {"angle_deg": float("nan"), "score": float("nan")}
    k = int(np.nanargmax(scores))
    return {
        "angle_deg": float(angles_deg[k]),
        "score": float(scores[k]),
        "density_jump": float(drho_out[k]),
        "pressure_jump": float(dp_out[k]),
        "angles_deg": angles_deg,
        "scores": scores,
        "origin": [float(x0), float(y0)],
        "length": float(length),
    }


def slip_line_angle_velocity(
    u: np.ndarray,
    v: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    origin: tuple[float, float],
    length: float,
    angles_deg=None,
    offset_cells: float = 1.5,
    n_samples: int = 80,
) -> dict:
    """Slip-line inclination from the *velocity* jump (vortex-sheet detector).

    A contact is a tangential discontinuity: across it the tangential velocity
    jumps while the normal velocity is continuous.  Rays from the triple point
    are scored by

        score(chi) = mean(|du_tangential|) - mean(|du_normal|)

    which uses the momentum field only, so it is independent of the density /
    entropy detector in :func:`slip_line_angle` (a genuine cross-check of the
    same geometric object, not a re-parameterisation of it).
    """
    x0, y0 = origin
    if angles_deg is None:
        angles_deg = np.arange(-25.0, 25.0001, 0.25)
    angles_deg = np.asarray(angles_deg, dtype=float)
    s = np.linspace(0.02 * length, length, n_samples)
    h = offset_cells * (xc[1] - xc[0])
    scores = np.full(angles_deg.shape, np.nan)
    for k, a in enumerate(angles_deg):
        th = math.radians(a)
        cx, cy = math.cos(th), math.sin(th)
        nx, ny = -cy, cx
        px = x0 + s * cx
        py = y0 + s * cy
        um = bilinear(u, xc, yc, px + h * nx, py + h * ny)
        up = bilinear(u, xc, yc, px - h * nx, py - h * ny)
        vm = bilinear(v, xc, yc, px + h * nx, py + h * ny)
        vp = bilinear(v, xc, yc, px - h * nx, py - h * ny)
        good = np.isfinite(um) & np.isfinite(up) & np.isfinite(vm) & np.isfinite(vp)
        if good.sum() < 5:
            continue
        dut = np.abs((um - up) * cx + (vm - vp) * cy)
        dun = np.abs((um - up) * nx + (vm - vp) * ny)
        scores[k] = float(np.mean(dut[good] - dun[good]))
    if not np.isfinite(scores).any():
        return {"angle_deg": float("nan"), "score": float("nan")}
    k = int(np.nanargmax(scores))
    return {
        "angle_deg": float(angles_deg[k]),
        "score": float(scores[k]),
        "angles_deg": angles_deg,
        "scores": scores,
        "origin": [float(x0), float(y0)],
        "length": float(length),
    }


def vortex_core(
    u: np.ndarray,
    v: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    box: tuple[float, float, float, float] | None = None,
    refine: int = 3,
) -> dict:
    """Primary vortex centre from the extremum of ``|omega|`` inside ``box``.

    ``omega = dv/dx - du/dy`` with central differences.  ``box = (x0, x1, y0,
    y1)`` restricts the search so the boundary layers and corner singularities
    cannot win; the centre is refined by the |omega|-weighted centroid of the
    ``(2*refine+1)^2`` cells around the discrete peak.  The peak circulation
    ``omega * dx * dy`` and the sign are returned with it.
    """
    dvdx = np.gradient(v, xc, axis=1)
    dudy = np.gradient(u, yc, axis=0)
    omega = dvdx - dudy
    mask = np.ones_like(omega, dtype=bool)
    if box is not None:
        x0, x1, y0, y1 = box
        X, Y = np.meshgrid(xc, yc)
        mask = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    masked = np.where(mask, np.abs(omega), -np.inf)
    j, i = np.unravel_index(int(np.argmax(masked)), masked.shape)
    j_lo, j_hi = max(j - refine, 0), min(j + refine + 1, omega.shape[0])
    i_lo, i_hi = max(i - refine, 0), min(i + refine + 1, omega.shape[1])
    sub = np.abs(omega[j_lo:j_hi, i_lo:i_hi])
    w = sub / max(sub.sum(), 1e-300)
    ys = yc[j_lo:j_hi]
    xs = xc[i_lo:i_hi]
    return {
        "x": float((w.sum(axis=1) * xs).sum()),
        "y": float((w.sum(axis=0) * ys).sum()),
        "i": int(i),
        "j": int(j),
        "omega_peak": float(omega[j, i]),
        "omega_abs_peak": float(abs(omega[j, i])),
        "box": list(box) if box is not None else None,
    }


def sonic_point_on_front(
    x_front: np.ndarray,
    yc: np.ndarray,
    xc: np.ndarray,
    U: np.ndarray,
    gamma: float = GAMMA,
    back_cells: int = 3,
    y_lo: float | None = None,
    y_hi: float | None = None,
) -> dict:
    """Locate where the flow behind the leading front becomes sonic.

    For a detached bow shock the downstream Mach number rises continuously from
    the (subsonic) normal part to the supersonic far field; the crossing
    ``M2 = 1`` is the sonic point, and the local shock angle there is the
    *detachment angle* ``beta*`` of oblique-shock theory -- an exact reference.
    The local shock angle is the inclination of the front normal:

        beta = angle between the shock normal and the upstream flow (x axis)
             = atan2(1, dx/dy) + 90 deg  (mod 180)

    which, for a front inclined at ``alpha`` from the x axis, is ``90 - alpha``.
    """
    from .flux import primitive

    rho, u, v, p = primitive(U, gamma)
    a = np.sqrt(gamma * p / rho)
    M = np.hypot(u, v) / a
    prof_angle = front_angle(x_front, yc, y_lo if y_lo is not None else yc[0], y_hi if y_hi is not None else yc[-1])
    rows = []
    for j, y in enumerate(yc):
        xf = x_front[j]
        if not np.isfinite(xf):
            continue
        i = int(np.argmin(np.abs(xc - xf)))
        k = min(i + back_cells, len(xc) - 1)
        rows.append((float(y), float(xf), float(M[j, k]), float(a[j, k])))
    if len(rows) < 6:
        return {"found": False, "reason": "front too short", "front_angle": prof_angle}
    arr = np.array(rows)
    ys, xs, Ms = arr[:, 0], arr[:, 1], arr[:, 2]
    order = np.argsort(ys)
    ys, xs, Ms = ys[order], xs[order], Ms[order]
    # local shock angle from the smoothed front slope
    xs_s = np.convolve(np.pad(xs, (2, 2), mode="edge"), np.ones(5) / 5.0, mode="same")[2:-2]
    slope = np.gradient(xs_s, ys)
    beta = np.degrees(np.arctan2(1.0, slope))  # angle of front from x axis
    beta_shock = 90.0 - beta  # shock angle from the upstream flow direction
    cross = np.flatnonzero((Ms[:-1] - 1.0) * (Ms[1:] - 1.0) <= 0.0)
    out = {
        "found": bool(cross.size > 0),
        "front_angle": prof_angle,
        "y": ys.tolist(),
        "M2_behind": Ms.tolist(),
        "shock_angle_deg": beta_shock.tolist(),
    }
    if cross.size == 0:
        out["reason"] = "no M2 = 1 crossing on the front"
        return out
    k = int(cross[0])
    frac = (1.0 - Ms[k]) / (Ms[k + 1] - Ms[k]) if Ms[k + 1] != Ms[k] else 0.0
    out.update(
        {
            "sonic_point_y": float(ys[k] + frac * (ys[k + 1] - ys[k])),
            "sonic_point_x": float(xs[k] + frac * (xs[k + 1] - xs[k])),
            "sonic_shock_angle_deg": float(beta_shock[k] + frac * (beta_shock[k + 1] - beta_shock[k])),
        }
    )
    return out

def shock_polar_angle_from_ratio(
    M1: float, ratio: float, gamma: float = GAMMA, kind: str = "rho", weak: bool = True
) -> dict:
    """Invert the shock jump for the shock angle: ``beta`` from a measured ratio.

    The oblique-shock jump depends on ``beta`` only through ``Mn1 = M1 sin beta``,
    so a measured density (or pressure) ratio can be inverted for ``beta`` on the
    weak branch by bisection on ``Mn1``.  Comparing that thermodynamic angle with
    the geometric inclination of the same front is a consistency test that uses
    no information from the solver: geometry on one side, Rankine-Hugoniot on
    the other.  ``ratio`` must be > 1.
    """
    if M1 <= 1.0:
        raise ValueError("M1 must be supersonic")
    if not (ratio > 1.0):
        raise ValueError("ratio must exceed 1")

    def jump(mn: float) -> float:
        j = normal_shock(mn, gamma)
        return j["rho_ratio"] if kind == "rho" else j["p_ratio"]

    mu = math.asin(1.0 / M1)
    lo, hi = mu + 1e-12, 0.5 * math.pi - 1e-12
    # weak branch: the ratio increases monotonically with Mn1 from the Mach angle
    target = ratio
    if jump(M1 * math.sin(hi)) < target:
        return {"attached": False, "reason": "ratio above the normal-shock limit"}
    a, b = lo, hi
    for _ in range(200):
        m = 0.5 * (a + b)
        if jump(M1 * math.sin(m)) < target:
            a = m
        else:
            b = m
        if b - a < 1e-13:
            break
    beta = 0.5 * (a + b)
    return {
        "attached": True,
        "M1": M1,
        "ratio": ratio,
        "kind": kind,
        "beta_deg": math.degrees(beta),
        "beta_strong_deg": 180.0 - math.degrees(beta),
        "Mn1": M1 * math.sin(beta),
        "theta_deg": math.degrees(_deflection(M1, beta, gamma)),
    }


def contact_ridge_angle(
    rho: np.ndarray,
    p: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    origin: tuple[float, float],
    r_min: float,
    r_max: float,
    gamma: float = GAMMA,
    entropy_frac: float = 0.5,
    pressure_tol: float = 0.5,
) -> dict:
    """Entropy-ridge line fit: an independent estimator of a contact angle.

    A slip line is an *entropy* jump with continuous pressure, so among the cells
    in the annulus ``r_min < |x - origin| < r_max`` downwind of the triple point
    the contact is the ridge of ``|grad s|`` (``s = ln(p / rho^gamma)``) that also
    has a weak pressure gradient (``|grad p|/p < pressure_tol * max``).  A
    weighted principal-axis fit of those cells gives the line direction and its
    rms width -- a detector that is independent of the ray-cast estimator in
    :func:`slip_line_angle` (gradient ridge vs jump integration).
    """
    x0, y0 = origin
    s = np.log(p) - gamma * np.log(rho)
    gs = np.sqrt(np.gradient(s, xc, axis=1) ** 2 + np.gradient(s, yc, axis=0) ** 2)
    gp = np.sqrt(np.gradient(p, xc, axis=1) ** 2 + np.gradient(p, yc, axis=0) ** 2) / p
    X, Y = np.meshgrid(xc, yc)
    r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    ann = (r > r_min) & (r < r_max) & (Y < y0) & (X < x0 + 0.02)
    if ann.sum() < 10:
        return {"found": False, "reason": "annulus too small"}
    peak = float(np.max(gs[ann]))
    sel = ann & (gs > entropy_frac * peak) & (gp < pressure_tol * float(np.max(gp[ann])))
    n = int(sel.sum())
    if n < 6:
        return {"found": False, "reason": f"only {n} ridge cells", "annulus_cells": int(ann.sum())}
    xs, ys, w = X[sel], Y[sel], gs[sel]
    cx = float((w * xs).sum() / w.sum())
    cy = float((w * ys).sum() / w.sum())
    cov = np.cov(np.vstack([xs - cx, ys - cy]), aweights=w)
    ev, evec = np.linalg.eigh(cov)
    v = evec[:, int(np.argmax(ev))]
    ang = float(math.degrees(math.atan2(v[1], v[0])) % 180.0)
    return {
        "found": True,
        "angle_deg": ang,
        "centroid": [cx, cy],
        "n_cells": n,
        "annulus_cells": int(ann.sum()),
        "rms_width": float(math.sqrt(max(ev.min(), 0.0))),
        "range": [float(r_min), float(r_max)],
        "pressure_grad_tol": float(pressure_tol),
    }


def three_shock_closure(
    upstream_mach: float,
    incident_beta_deg: float,
    gamma: float = GAMMA,
    n: int = 400,
) -> dict:
    """Diagnose whether von Neumann three-shock theory closes for a triple point.

    Given the *steady* triple-point frame (upstream Mach along the wall and the
    incident-shock angle) the reflected shock has to turn the flow back by
    ``theta_i - chi`` while the incident shock turned it by ``theta_i``.  The
    turning capability of an oblique shock is limited by ``theta_max(M2i)``, so
    for strong incident shocks the admissible slip-line angle is pushed towards
    ``theta_i`` and the pressure polars may not cross at all -- the classical
    failure behind the von Neumann paradox.  This routine returns the numbers
    that decide it instead of raising, so a verification report can state *why*
    theory cannot supply a reference value.
    """
    beta_i = math.radians(incident_beta_deg)
    theta_i = _deflection(upstream_mach, beta_i, gamma)
    inc = oblique_shock(upstream_mach, theta_i, gamma)
    m2i = inc["M2"]
    p2i = inc["p_ratio"]
    tm = theta_max(m2i, gamma)
    d_max = math.radians(tm["theta_max_deg"])
    chi_min = theta_i - d_max
    gaps = []
    chis = np.linspace(max(chi_min, 1e-6), theta_i - 1e-9, n)
    for c in chis:
        try:
            rs = oblique_shock(m2i, theta_i - c, gamma)
            stem = oblique_shock(upstream_mach, c, gamma)
            gaps.append(p2i * rs["p_ratio"] - stem["p_ratio"])
        except ValueError:
            gaps.append(float("nan"))
    gaps = np.array(gaps)
    ok = np.isfinite(gaps)
    crossing = bool(ok.any() and np.any(np.sign(gaps[ok])[:-1] * np.sign(gaps[ok])[1:] < 0))
    return {
        "upstream_mach": upstream_mach,
        "incident_beta_deg": incident_beta_deg,
        "incident_theta_deg": math.degrees(theta_i),
        "M2_behind_incident": m2i,
        "theta_max_downstream_deg": tm["theta_max_deg"],
        "chi_min_feasible_deg": math.degrees(chi_min),
        "pressure_gap_at_chi_min": float(gaps[0]) if gaps.size else float("nan"),
        "pressure_gap_range": [float(np.nanmin(gaps)), float(np.nanmax(gaps))] if ok.any() else [float("nan")] * 2,
        "polylines_cross": crossing,
        "chi_scan_deg": np.degrees(chis),
        "pressure_gap_scan": gaps,
    }
