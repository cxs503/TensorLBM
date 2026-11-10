"""Self-contained aerodynamic reference solutions for airfoil verification.

This module provides the *reference side* of the NACA0012 external-aerodynamics
verification surface of TensorLBM:

1. :func:`naca0012_half_thickness` / :func:`naca0012_surface` -- analytic
   NACA 4-digit thickness family, cosine-spaced closed contour.
2. :class:`VortexPanelSolution` / :func:`solve_vortex_panel` -- classical
   Hess-Smith constant-strength vortex panel method with Kutta condition
   (Katz & Plotkin, *Low-Speed Aerodynamics*, ch. 11).  Gives the inviscid
   limit of ``C_p`` on the surface, the lift coefficient, the pressure
   distribution and the surface edge-velocity distribution ``V_e(s)``.
3. :func:`thin_airfoil_cl` -- thin-airfoil theory (``C_L = 2 pi alpha``).
4. :func:`squire_young_polar` -- laminar integral boundary-layer polar:
   Thwaites' method for the momentum thickness plus Head's shape-factor
   correlation, closed with the Squire-Young far-wake drag formula.
   This is the standard low-order *viscous* surrogate for XFOIL used when
   XFOIL itself is unavailable; it is validated internally on the flat plate
   against the Blasius solution ``C_D = 2 * 1.328 / sqrt(Re)``.

All routines are pure NumPy, deterministic and unit-testable.  Every number
that a report quotes as "reference" is produced here (zero hand-copied data).

References
----------
Squire & Young (1939) ARC R&M 1838; Thwaites (1949) JAS 16(2);
Head (1958) ARC R&M 3152; White, *Viscous Fluid Flow*, 3rd ed., ch. 4;
Katz & Plotkin, *Low-Speed Aerodynamics*, 2nd ed., ch. 11;
Abbott & von Doenhoff, *Theory of Wing Sections* (1949) ch. 4 & 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

THIN_AIRFOIL_CL_SLOPE = 2.0 * math.pi  # per radian


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def naca0012_half_thickness(xi: np.ndarray | float, t: float = 0.12) -> np.ndarray:
    """NACA 4-digit half-thickness distribution y_t(x/c) with closed TE.

    ``y_t = 5 t (0.2969 sqrt(xi) - 0.1260 xi - 0.3516 xi^2
                 + 0.2843 xi^3 - 0.1036 xi^4)``

    The ``-0.1036`` coefficient (instead of the classical ``-0.1015``) yields
    a closed trailing edge, which is what a finite-resolution lattice
    simulation needs (an open TE would be one cell thick anyway).
    """
    xi = np.asarray(xi, dtype=np.float64)
    return 5.0 * t * (
        0.2969 * np.sqrt(np.clip(xi, 0.0, None))
        - 0.1260 * xi
        - 0.3516 * xi**2
        + 0.2843 * xi**3
        - 0.1036 * xi**4
    )


def naca0012_surface(
    n_panels: int = 200, t: float = 0.12, spacing: str = "quadratic"
) -> np.ndarray:
    """Closed NACA0012 contour, LE-first on the lower surface, clockwise.

    Args:
        n_panels: number of panels (returned array has ``n_panels + 1`` nodes).
        t: thickness ratio.
        spacing: ``"quadratic"`` (LE-clustered, TE panel ~2/n chord -- default,
            chosen for panel-method conditioning) or ``"cosine"`` (clusters at
            both LE and TE).

    Returns:
        ``(n_panels + 1, 2)`` contour, ``P[-1] == P[0]``.
    """
    if n_panels < 8:
        raise ValueError("n_panels must be >= 8")
    n_side = n_panels // 2
    if spacing == "cosine":
        beta = np.linspace(0.0, math.pi, n_side + 1)
        xi = 0.5 * (1.0 - np.cos(beta))
    elif spacing == "quadratic":
        xi = (np.linspace(0.0, 1.0, n_side + 1)) ** 2
    else:
        raise ValueError(f"unknown spacing {spacing!r}")
    yt = naca0012_half_thickness(xi, t)
    x_lo, y_lo = xi[::-1], -yt[::-1]      # TE -> LE along the lower surface
    x_up, y_up = xi[1:], yt[1:]           # LE -> TE along the upper surface
    pts = np.column_stack([np.concatenate([x_lo, x_up]), np.concatenate([y_lo, y_up])])
    pts[-1] = pts[0]
    return pts


def airfoil_frame(
    x: np.ndarray,
    y: np.ndarray,
    chord: float,
    x_le: float,
    y_c: float,
    alpha_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map lattice coordinates to chord-aligned airfoil coordinates.

    The lattice body is axis-aligned with chord along ``+x``; the freestream
    is inclined by ``alpha``.  ``X`` is measured from the leading edge along
    the chord and ``Y`` normal to it (positive towards the suction side for
    positive incidence).
    """
    dx = x - x_le
    dy = y - y_c
    a = math.radians(alpha_deg)
    ca, sa = math.cos(a), math.sin(a)
    X = dx * ca + dy * sa
    Y = -dx * sa + dy * ca
    return X / chord, Y / chord


def thickness_slope(xi: np.ndarray | float, t: float = 0.12) -> np.ndarray:
    """d y_t / d xi for the (closed-TE) NACA 4-digit thickness law."""
    xi = np.asarray(xi, dtype=np.float64)
    xs = np.clip(xi, 1e-12, None)
    return 5.0 * t * (
        0.2969 * 0.5 / np.sqrt(xs)
        - 0.1260
        - 2.0 * 0.3516 * xi
        + 3.0 * 0.2843 * xi**2
        - 4.0 * 0.1036 * xi**3
    )


def analytic_signed_distance(
    X: np.ndarray, Y: np.ndarray, t: float = 0.12
) -> np.ndarray:
    """Approximate signed distance to the NACA0012 surface (chord units).

    Uses the body-normal projection of the thickness function:
    ``phi = (|Y| - y_t(X)) / sqrt(1 + (dy_t/dX)^2)``, positive in the fluid.
    This is first-order accurate in the surface curvature and is used only to
    obtain the wall-intersection fraction ``q`` of the boundary links of the
    lattice grid, where the remaining distance to the wall is a fraction of a
    cell.  Away from the surface the field is a monotone surrogate, which is
    all that is required.
    """
    Xc = np.clip(X, 0.0, 1.0)
    yt = naca0012_half_thickness(Xc, t)
    dyt = thickness_slope(Xc, t)
    phi = (np.abs(Y) - yt) / np.sqrt(1.0 + dyt**2)
    # outside the chord range the nearest surface features are the LE/TE
    # points; blend in the distance to those end points.
    out_up = X > 1.0
    if np.any(out_up):
        r_te = np.sqrt((X - 1.0) ** 2 + Y**2)
        phi = np.where(out_up, np.minimum(phi, r_te), phi)
    out_le = X < 0.0
    if np.any(out_le):
        r_le = np.sqrt(X**2 + Y**2)
        phi = np.where(out_le, np.minimum(phi, r_le), phi)
    return phi


def inside_naca0012(X: np.ndarray, Y: np.ndarray, t: float = 0.12) -> np.ndarray:
    """Exact analytic point-in-body test for the NACA 4-digit thickness law."""
    return (X >= 0.0) & (X <= 1.0) & (np.abs(Y) <= naca0012_half_thickness(np.clip(X, 0, 1), t))


# ---------------------------------------------------------------------------
# Panel method
# ---------------------------------------------------------------------------

@dataclass
class VortexPanelSolution:
    """Inviscid panel-method solution for a closed airfoil contour."""

    alpha_deg: float
    nodes: np.ndarray            # (N+1, 2) contour nodes
    gamma: np.ndarray            # (N,) vortex strength per panel
    cp: np.ndarray               # (N,) Cp at panel control points
    vt: np.ndarray               # (N,) tangential velocity / U_inf
    xc: np.ndarray               # (N,) control-point x/c
    yc: np.ndarray               # (N,) control-point y/c
    s: np.ndarray                # (N,) arc length from LE
    cl: float                    # 2 * circulation / (U c)
    cl_pressure: float           # lift from the pressure integral
    cm: float                    # quarter-chord moment (about c/4, LE-up)
    cd_pressure: float = 0.0     # inviscid (form) drag from the pressure integral

    @property
    def cl_slope_per_rad(self) -> float:
        return self.cl / math.radians(self.alpha_deg) if self.alpha_deg else float("nan")

    def cp_at(self, xc_query: np.ndarray, upper: bool) -> np.ndarray:
        """Interpolate the panel Cp onto chordwise stations of one side."""
        # split the contour into upper (increasing x) and lower (decreasing x)
        xs, cps = self._surface_branch(upper)
        return np.interp(xc_query, xs, cps, left=np.nan, right=np.nan)

    def _surface_branch(self, upper: bool) -> tuple[np.ndarray, np.ndarray]:
        xc = self.xc
        # contour starts at TE, goes along lower surface (decreasing x) to LE,
        # then along upper surface (increasing x) back to TE.
        i_le = int(np.argmin(xc))
        if upper:
            sl = slice(i_le + 1, None)
            xs, cps = xc[sl], self.cp[sl]
            order = np.argsort(xs)
            return xs[order], cps[order]
        sl = slice(0, i_le + 1)
        xs, cps = xc[sl][::-1], self.cp[sl][::-1]
        order = np.argsort(xs)
        return xs[order], cps[order]


def _vortex_panel_induced(
    xp: np.ndarray, yp: np.ndarray, x1: np.ndarray, y1: np.ndarray,
    x2: np.ndarray, y2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Velocity induced at ``(xp, yp)`` by unit-strength constant vortex panels.

    Classical log/angle form (Katz & Plotkin eqs. 10.20-10.21, rotated 90
    degrees for a vortex rather than a source sheet).  The returned field
    corresponds to a counter-clockwise-positive circulation ``Gamma = +1``
    per unit panel strength (verified against brute-force quadrature of the
    Biot-Savart integral in ``tests/test_naca0012_aero.py``).
    """
    dx1 = xp - x1
    dy1 = yp - y1
    dx2 = xp - x2
    dy2 = yp - y2
    r1 = np.maximum(np.hypot(dx1, dy1), 1e-12)
    r2 = np.maximum(np.hypot(dx2, dy2), 1e-12)
    ln = np.log(r1 / r2)
    beta = np.arctan2(dx1 * dy2 - dy1 * dx2, dx1 * dx2 + dy1 * dy2)
    theta = np.arctan2(y2 - y1, x2 - x1)
    u_loc = beta
    w_loc = -ln
    ca = np.cos(theta)
    sa = np.sin(theta)
    u = ca * u_loc - sa * w_loc
    w = sa * u_loc + ca * w_loc
    return u / (2.0 * math.pi), w / (2.0 * math.pi)


def _panel_geometry(nodes: np.ndarray):
    """Panel geometry helper: control points, lengths, tangents, outward normals.

    ``nodes`` must be a closed contour.  The winding is detected from the
    signed area so that the outward normal is always correct.
    """
    p1 = nodes[:-1]
    p2 = nodes[1:]
    xm = 0.5 * (p1[:, 0] + p2[:, 0])
    ym = 0.5 * (p1[:, 1] + p2[:, 1])
    length = np.hypot(p2[:, 0] - p1[:, 0], p2[:, 1] - p1[:, 1])
    theta = np.arctan2(p2[:, 1] - p1[:, 1], p2[:, 0] - p1[:, 0])
    tx, ty = np.cos(theta), np.sin(theta)
    signed_area = 0.5 * np.sum(p1[:, 0] * p2[:, 1] - p2[:, 0] * p1[:, 1])
    # outward normal = tangent rotated +90 deg for a clockwise contour
    sgn = 1.0 if signed_area < 0.0 else -1.0
    nx, ny = -sgn * ty, sgn * tx
    return p1, p2, xm, ym, length, tx, ty, nx, ny


def _source_panel_induced(
    xp: np.ndarray, yp: np.ndarray, x1: np.ndarray, y1: np.ndarray,
    x2: np.ndarray, y2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Velocity induced at ``(xp, yp)`` by unit-strength constant source panels.

    ``u_loc = ln(r1/r2)/2pi``, ``w_loc = beta/2pi`` (Katz & Plotkin 10.20-10.21).
    """
    dx1 = xp - x1
    dy1 = yp - y1
    dx2 = xp - x2
    dy2 = yp - y2
    r1 = np.maximum(np.hypot(dx1, dy1), 1e-12)
    r2 = np.maximum(np.hypot(dx2, dy2), 1e-12)
    ln = np.log(r1 / r2)
    beta = np.arctan2(dx1 * dy2 - dy1 * dx2, dx1 * dx2 + dy1 * dy2)
    theta = np.arctan2(y2 - y1, x2 - x1)
    u_loc = ln / (2.0 * math.pi)
    w_loc = beta / (2.0 * math.pi)
    ca = np.cos(theta)
    sa = np.sin(theta)
    return ca * u_loc - sa * w_loc, sa * u_loc + ca * w_loc


def solve_vortex_panel(
    n_panels: int = 200,
    alpha_deg: float = 0.0,
    thickness: float = 0.12,
    nodes: np.ndarray | None = None,
    spacing: str = "quadratic",
    ridge: float = 1e-6,
) -> VortexPanelSolution:
    """Constant-strength vortex panel method with a Kutta condition.

    Formulation: one unknown vortex strength ``gamma_j`` per panel; the
    Neumann condition ``V . n = 0`` is collocated at the panel mid-points;
    the Kutta condition is ``gamma_0 = -gamma_{N-1}`` on the two panels that
    meet at the trailing edge.  The resulting ``(N+1) x N`` system is solved in
    the least-squares sense.  For this formulation the solved ``gamma_j`` *is*
    the total tangential surface velocity ``V_t/U_inf`` at the control point
    (vortex-sheet identity), hence ``C_p = 1 - gamma_j^2``.

    A Hess-Smith source+vortex variant was tried first and rejected: the normal
    velocity induced by a *uniform* vortex sheet on a closed contour vanishes
    identically (verified to 1e-16 on a circular contour), so the single
    vortex unknown is decoupled from the Neumann equations and the system
    becomes numerically singular.  That experience is recorded in
    ``docs/reports/naca0012_aero_polar.md``.

    Args:
        n_panels: panels on the contour.
        alpha_deg: incidence (deg).
        thickness: thickness ratio.
        nodes: optional explicit closed contour (analytic cylinder test).
        spacing: panel spacing law for the NACA contour (see
            :func:`naca0012_surface`).

    Returns:
        :class:`VortexPanelSolution`; ``cl`` from the circulation
        (``C_L = 2 Gamma``), ``cl_pressure`` from the pressure integral, so the
        two independent routes can be cross-checked.
    """
    if nodes is None:
        nodes = naca0012_surface(n_panels, thickness, spacing)
    p1, p2, xm, ym, length, tx, ty, nx, ny = _panel_geometry(nodes)
    n = len(p1)

    A = np.zeros((n, n))
    for j in range(n):
        du, dw = _vortex_panel_induced(xm, ym, p1[j, 0], p1[j, 1], p2[j, 0], p2[j, 1])
        A[:, j] = du * nx + dw * ny
    # self-induced velocity of a straight constant-strength vortex panel at its
    # own mid-point is zero (odd integrand) -> explicit zero diagonal
    np.fill_diagonal(A, 0.0)

    a = math.radians(alpha_deg)
    u_inf, v_inf = math.cos(a), math.sin(a)
    rhs = -(u_inf * nx + v_inf * ny)

    row = np.zeros((1, n))
    row[0, 0] = 1.0
    row[0, -1] = 1.0
    # Tikhonov regularisation on the second difference of gamma: the discrete
    # vortex-panel system has a near-null "sawtooth" mode (condition number
    # ~3e6 for 200 panels) which otherwise produces |gamma| ~ 1e3 spikes on the
    # LE panels without affecting the circulation.  lam is scaled by the mean
    # squared influence coefficient so it is geometry independent.
    d2 = np.zeros((n - 2, n))
    for i in range(n - 2):
        d2[i, i], d2[i, i + 1], d2[i, i + 2] = 1.0, -2.0, 1.0
    lam = ridge * float(np.mean(A**2))
    A_reg = np.vstack([A, row, math.sqrt(lam) * d2])
    rhs_reg = np.concatenate([rhs, [0.0], np.zeros(n - 2)])
    gamma, *_ = np.linalg.lstsq(A_reg, rhs_reg, rcond=None)

    vt = gamma.copy()
    cp = 1.0 - vt**2
    circ = float(np.sum(gamma * length))
    cl = 2.0 * circ
    cp_cl = float(-np.sum(cp * ny * length))           # lift  = -oint Cp n_y ds
    cp_cd = float(-np.sum(cp * nx * length))           # pressure drag
    cm = float(np.sum(cp * ((xm - 0.25) * ny - ym * nx) * length))
    s_mid = np.concatenate([[0.0], np.cumsum(length)])
    s_mid = 0.5 * (s_mid[:-1] + s_mid[1:])
    return VortexPanelSolution(
        alpha_deg=alpha_deg,
        nodes=nodes,
        gamma=gamma,
        cp=cp,
        vt=vt,
        xc=xm,
        yc=ym,
        s=s_mid,
        cl=cl,
        cl_pressure=cp_cl,
        cm=cm,
        cd_pressure=cp_cd,
    )


def cylinder_nodes(n_panels: int = 200, radius: float = 0.5, n_points: int = 400) -> np.ndarray:
    """Closed circular contour (analytic panel-method validation case)."""
    beta = np.linspace(0.0, 2.0 * math.pi, n_points + 1)
    pts = np.column_stack([radius * np.cos(beta), radius * np.sin(beta)])
    pts[-1] = pts[0]
    return pts


# ---------------------------------------------------------------------------
# Thin airfoil and boundary-layer polar surrogate
# ---------------------------------------------------------------------------

def thin_airfoil_cl(alpha_deg: float) -> float:
    """Thin-airfoil theory lift coefficient of a symmetric section."""
    return THIN_AIRFOIL_CL_SLOPE * math.radians(alpha_deg)


def flat_plate_cd_blasius(re: float, turbulent: bool = False) -> float:
    """Flat-plate (both sides) skin-friction drag coefficient.

    Laminar: ``2 * 1.328 / sqrt(Re)``; turbulent: ``2 * 0.074 / Re**0.2``
    (Prandtl-Schlichting, valid 5e5 .. 1e7).
    """
    if turbulent:
        return 2.0 * 0.074 / re**0.2
    return 2.0 * 1.328 / math.sqrt(re)


def head_shape_factor(lam: float) -> float:
    """Head's shape factor H(lambda) correlation (White, eq. 4-54)."""
    if lam <= 0.0:
        return 2.088 + 0.0731 / max(lam + 0.14, 1e-6)
    if lam < 0.1:
        return 2.61 - 3.75 * lam + 5.24 * lam**2
    return 2.39


@dataclass
class IntegralBLResult:
    """Laminar integral boundary-layer polar from Thwaites + Squire-Young."""

    alpha_deg: float
    re: float
    cd: float
    cl_inviscid: float
    separated: bool
    s_sep: float | None
    theta_te: float
    h_te: float
    detail: dict = field(default_factory=dict)


def squire_young_polar(
    panel: VortexPanelSolution,
    re: float,
    t: float = 0.12,
) -> IntegralBLResult:
    """Laminar viscous polar of the section via Thwaites + Head + Squire-Young.

    The edge velocity ``V_e(s)`` comes from the inviscid panel solution.
    Momentum thickness is integrated from the leading-edge stagnation point
    along each surface to the trailing edge; the far-wake Squire-Young formula
    converts the TE momentum thickness into a profile drag coefficient

        ``C_D = 2 (theta_TE/c) (V_e,TE/U)^((5+H_TE)/2)``

    For a flat plate both surfaces contribute ``theta = sqrt(0.45/Re)``, so
    ``C_D = 4 sqrt(0.45)/sqrt(Re) = 2.683/sqrt(Re)``, within 1.0 % of the
    Blasius total ``2 * 1.328/sqrt(Re) = 2.656/sqrt(Re)`` -- this is the
    internal validation used by the test-suite and it anchors the surrogate
    polar to an exact analytic result.

    Only *laminar* closure is implemented: transition is not modelled, so the
    result is a low-Reynolds-number surrogate for XFOIL, not a replacement.
    """
    xc = panel.xc
    vt = panel.vt
    i_le = int(np.argmin(xc))
    nu = 1.0  # velocities are normalised by U_inf and lengths by chord
    out = {}
    cd_total = 0.0
    separated_any = False
    s_sep_first = None
    theta_te_avg = 0.0
    h_te_avg = 0.0
    n_branches = 0

    branches = {
        "upper": np.arange(i_le, len(xc)),
        "lower": np.arange(i_le, -1, -1),
    }
    for name, idx in branches.items():
        idx = np.asarray(idx)
        if len(idx) < 5:
            continue
        xe = xc[idx]
        ve_raw = np.abs(vt[idx])
        # start the boundary layer at the stagnation point: skip the region
        # where the edge velocity is still (numerically) zero, otherwise
        # dV/ds at the stagnation point is unbounded and Thwaites' lambda
        # blows up through the pole of Head's H(lambda) at lambda = -0.14.
        vmax = float(ve_raw.max()) if len(ve_raw) else 0.0
        keep = np.where(ve_raw > 0.05 * vmax)[0]
        if len(keep) < 5:
            continue
        start = int(keep[0])
        xe = xe[start:]
        ve = ve_raw[start:]
        dx = np.diff(xe)
        dy = np.diff(panel.yc[idx][start:])
        ds = np.hypot(dx, dy)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        if s[-1] <= 1e-6:
            continue
        ve_safe = np.maximum(ve, 1e-3)
        integrand = ve_safe**5
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(s))])
        theta2 = 0.45 * nu / ve_safe**6 * cum
        dve = np.gradient(ve_safe, s)
        lam = theta2 / nu * dve
        lam_h = np.clip(lam, -0.1, 0.1)          # Head correlation domain
        theta = np.sqrt(np.maximum(theta2, 0.0) / re)
        sep = np.where(lam < -0.09)[0]
        if len(sep):
            separated_any = True
            if s_sep_first is None:
                s_sep_first = float(s[sep[0]])
            k = int(sep[0])                      # freeze at separation
        else:
            k = len(s) - 1
        theta_te = float(theta[k])
        h_te = float(head_shape_factor(float(lam_h[k])))
        ve_te = float(ve_safe[k])
        cd_branch = 2.0 * theta_te * ve_te ** ((5.0 + h_te) / 2.0)
        cd_total += cd_branch
        theta_te_avg += theta_te
        h_te_avg += h_te
        n_branches += 1
        out[name] = {
            "s_te": float(s[-1]),
            "x_sep": float(xe[k]),
            "theta_te": theta_te,
            "h_te": h_te,
            "ve_te": ve_te,
            "cd_branch": cd_branch,
            "lambda_min": float(np.min(lam)),
            "separated": bool(len(sep) > 0),
        }
    if n_branches:
        theta_te_avg /= n_branches
        h_te_avg /= n_branches
    return IntegralBLResult(
        alpha_deg=panel.alpha_deg,
        re=re,
        cd=cd_total,
        cl_inviscid=panel.cl,
        separated=separated_any,
        s_sep=s_sep_first,
        theta_te=theta_te_avg,
        h_te=h_te_avg,
        detail=out,
    )


__all__ = [
    "THIN_AIRFOIL_CL_SLOPE",
    "naca0012_half_thickness",
    "naca0012_surface",
    "airfoil_frame",
    "thickness_slope",
    "analytic_signed_distance",
    "inside_naca0012",
    "VortexPanelSolution",
    "solve_vortex_panel",
    "cylinder_nodes",
    "thin_airfoil_cl",
    "flat_plate_cd_blasius",
    "head_shape_factor",
    "IntegralBLResult",
    "squire_young_polar",
]