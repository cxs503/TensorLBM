"""Free-streamline (constant-pressure cavity) hydrofoil solver -- 2-D.

Mission P40W3B-N02 (sheet / cloud cavitation validation face).  The companion
module :mod:`tensorlbm.cavitation_airfoil` holds the homogeneous-mixture
Navier-Stokes route; this module implements the *classical* route that the
cavitation community uses to predict attached (sheet) cavity lengths, and that
Brennen's textbook curves are actually drawn against: a **closed,
constant-pressure cavity on the suction side of a 2-D section**, solved as a
free-boundary potential-flow problem with a boundary-element (panel) method.

Why this formulation
--------------------
A sheet cavity is, to leading order, a region of constant pressure ``p_v``
bounded by a streamline.  With ``p_inf = 0`` and ``U_inf = 1`` the cavity
number ``sigma = (p_inf - p_v) / (0.5 rho U^2)`` fixes the cavity-surface speed

.. math::   |q| = q_c = \\sqrt{1 + \\sigma}      (Bernoulli, p = p_v)

so the cavity surface carries a **Dirichlet condition on the speed** and its
*shape* is the unknown of a free-boundary problem.  This is the Riabouchinsky /
Tulin closed-cavity family of solutions; it is the reference inviscid model for
``L_cav/c`` versus ``sigma``.

Numerics
--------
* Flow engine: Hess-Smith constant-source panels + one uniform vortex strength,
  a linear system of ``N + 1`` unknowns per flow solve::

      sum_j (n_i . S_ij) sigma_j + (n_i . sum_j V_ij) gamma = -n_i . U_inf   (N rows)
      q_t(TE-upper) + q_t(TE-lower)                              = 0         (Kutta)

  ``S_ij``/``V_ij`` are the source/vortex influence coefficients obtained by
  8-point Gauss-Legendre quadrature (no closed-form kernel to get wrong); the
  only analytic term is the source self-influence ``n_i . S_ii = +1/2`` (the
  outer limit of a source sheet).  The engine is verified against the analytic
  circle solution (``q = 2 U sin(theta)``, ``Cp_min = -3``) and against
  thin-aerofoil theory (``Cl = 2 pi alpha``) in
  ``tests/test_cavitation_free_streamline.py``.
* Free boundary: on cavity panels the source/vortex strengths satisfy
  ``q_t = q_c`` (dynamic condition) and the panel geometry is updated by a
  transpiration step until the normal velocity on the cavity surface vanishes
  (kinematic condition).  The cavity length ``L`` is the extra unknown; it is
  driven by the normal-velocity residual of the last cavity station, so the
  system is neither over- nor under-determined.

Declared approximations (see the report)
----------------------------------------
* inviscid (no boundary layer, no re-entrant jet); the detachment point is a
  model parameter (default: leading edge);
* closed cavity (Riabouchinsky) -- no wake, so the closure region loses the
  vapour transport that drives *cloud* shedding; this module therefore does not
  predict ``f_cav`` (reported honestly as NOT_SUPPORTED here);
* the cavity thickness is measured along the chord normal (thin-cavity
  approximation for the geometry update only, the flow solve is exact for the
  panel geometry).

References
----------
Brennen, C. E. (1995) *Cavitation and Bubble Dynamics*, Oxford Univ. Press,
    ch. 7 (attached cavities, L/c vs sigma).
Wu, T. Y. (1956) "A wake model for free-streamline flow theory", *J. Fluid
    Mech.* 13, 161-181.
Tulin, M. P. (1956) "Supercavitating flows -- small perturbation theory",
    *J. Ship Res.* 7, 16-37.
Kinnas, S. A. & Fine, N. E. (1993) "A numerical nonlinear analysis of the flow
    around two- and three-dimensional partially cavitating hydrofoils",
    *J. Fluid Mech.* 254, 151-181.
Katz, J. & Plotkin, A. (2001) *Low-Speed Aerodynamics*, 2nd ed., CUP
    (Hess-Smith panel method, source self-influence).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .cavitation_airfoil import airfoil_surface

__all__ = [
    "CavitySolution",
    "flow_solve",
    "foil_contour",
    "panel_geometry",
    "solve_cavity",
]


# ---------------------------------------------------------------------------
# quadrature / panel geometry
# ---------------------------------------------------------------------------


def _gauss_legendre(n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights on ``[-1, 1]`` (``numpy`` only)."""
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


_GL_X, _GL_W = _gauss_legendre(8)


def panel_geometry(nodes: np.ndarray) -> dict:
    """Panel data from a closed polyline ``nodes`` (``(M+1, 2)``).

    The outward normal is derived from the polygon orientation (shoelace sign)
    so that callers never have to remember the winding direction.
    """
    a = nodes[:-1]
    b = nodes[1:]
    d = b - a
    length = np.hypot(d[:, 0], d[:, 1])
    if np.any(length <= 0.0):
        msg = "degenerate panel (repeated node)"
        raise ValueError(msg)
    t = d / length[:, None]
    n = np.stack([t[:, 1], -t[:, 0]], axis=1)
    area2 = float(np.sum(a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]))
    # shoelace > 0 => counter-clockwise => (t_y, -t_x) is already outward
    if area2 <= 0.0:
        n = -n
    return {
        "a": a,
        "b": b,
        "t": t,
        "n": n,
        "length": length,
        "mid": 0.5 * (a + b),
        "signed_area": 0.5 * area2,
    }


def _influence_raw(targets: np.ndarray, geom: dict) -> tuple[np.ndarray, ...]:
    """Induced velocity of unit source/vortex panels at ``targets``.

    Returns ``(Sx, Sy, Vx, Vy)``, each of shape ``(N_targets, N_panels)``: the
    ``x``/``y`` velocity induced at every target point by a unit-strength
    constant source (``S``) and a unit-strength constant vortex sheet (``V``) on
    every panel, integrated with 8-point Gauss-Legendre quadrature.  Self terms
    are zero here; the caller adds the analytic source self-influence ``+1/2``
    (outer limit of a source sheet).
    """
    n_src = geom["a"].shape[0]
    pts = targets[:, None, None, :]  # (Nt, 1, 1, 2)
    pa = geom["a"][None, :, None, :]
    pb = geom["b"][None, :, None, :]
    s = _GL_X[None, None, :]
    dl = 0.5 * geom["length"][None, :, None] * _GL_W[None, None, :]
    px = 0.5 * (pa[..., 0] + pb[..., 0]) + s * 0.5 * (pb[..., 0] - pa[..., 0])
    py = 0.5 * (pa[..., 1] + pb[..., 1]) + s * 0.5 * (pb[..., 1] - pa[..., 1])
    rx = pts[..., 0] - px
    ry = pts[..., 1] - py
    r2 = rx**2 + ry**2
    r2 = np.where(r2 < 1e-24, 1e-24, r2)
    inv = 1.0 / (2.0 * math.pi * r2)
    sx = np.sum(rx * inv * dl, axis=2)
    sy = np.sum(ry * inv * dl, axis=2)
    # vortex (Gamma > 0 counter-clockwise): u = (-(y-y'), x-x') / (2 pi r^2) * dl
    vx = np.sum(-ry * inv * dl, axis=2)
    vy = np.sum(rx * inv * dl, axis=2)
    diag = np.arange(min(n_src, targets.shape[0]))
    sx[diag, diag] = 0.0
    sy[diag, diag] = 0.0
    vx[diag, diag] = 0.0
    vy[diag, diag] = 0.0
    assert sx.shape == (targets.shape[0], n_src)
    return sx, sy, vx, vy


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def _cosine_stations(n: int, a: float = 0.0, b: float = 1.0) -> np.ndarray:
    """``n`` cosine-clustered stations on ``[a, b]`` (dense at both ends)."""
    th = np.linspace(0.0, math.pi, n)
    return a + (b - a) * 0.5 * (1.0 - np.cos(th))


def _surface_y(kind: str, x: np.ndarray, upper: bool) -> np.ndarray:
    """Airfoil surface ordinate at chordwise stations ``x`` (camber +- thickness)."""
    surf = airfoil_surface(kind, n=801)
    xc = surf.xc.numpy()
    yc = surf.camber.numpy()
    yt = surf.half_thickness.numpy()
    cam = np.interp(np.clip(x, 0.0, 1.0), xc, yc)
    th = np.interp(np.clip(x, 0.0, 1.0), xc, yt)
    return cam + th if upper else cam - th


def foil_contour(
    kind: str,
    alpha_deg: float,
    n_foil: int = 120,
    n_cav: int = 48,
    l_cav: float = 0.2,
    thickness: np.ndarray | None = None,
    x_detach: float = 0.0,
) -> np.ndarray:
    """Closed contour of the foil + attached cavity (outward normal automatic).

    The contour runs: trailing edge -> lower surface -> leading edge -> cavity
    surface (from ``x_detach`` to the closure ``x_detach + l_cav``) -> wetted
    suction surface -> trailing edge.  ``thickness`` is the cavity thickness at
    ``n_cav`` cosine-spaced stations normalised to ``[0, 1]`` (last value is
    forced to zero: closure); ``None`` gives a parabolic initial guess.
    """
    if l_cav <= 0.0:
        msg = "l_cav must be positive"
        raise ValueError(msg)
    # lower surface, TE -> LE
    xl = _cosine_stations(n_foil // 2, 1.0, 0.0)
    lower = np.stack([xl, _surface_y(kind, xl, upper=False)], axis=1)
    # wetted suction surface, closure -> TE (leading edge handled by the cavity)
    xd = float(np.clip(x_detach, 0.0, 0.9))
    xe = min(xd + l_cav, 1.0)
    xu = _cosine_stations(n_foil // 2, xe, 1.0)
    upper = np.stack([xu, _surface_y(kind, xu, upper=True)], axis=1)
    # cavity surface, LE/detachment -> closure
    xi = np.linspace(0.0, 1.0, n_cav + 1)
    xs = xd + (xe - xd) * 0.5 * (1.0 - np.cos(math.pi * xi))
    if thickness is None:
        thick = 0.5 * l_cav * 0.12 * np.sin(math.pi * xi) ** 0.8
    else:
        thick = np.asarray(thickness, dtype=float)
        if thick.shape != xs.shape:
            msg = f"thickness must have shape {xs.shape}, got {thick.shape}"
            raise ValueError(msg)
    thick = thick.copy()
    thick[0] = 0.0
    thick[-1] = 0.0
    y_base = _surface_y(kind, xs, upper=True)
    cav = np.stack([xs, y_base + np.clip(thick, 0.0, None)], axis=1)
    # assemble: TE -> lower -> (LE via cavity start) -> cavity -> upper -> TE
    # the cavity starts at the leading edge: its first station coincides with
    # the last lower-surface node, so it is dropped (shared LE node).
    first_cav = 1 if xd <= 1e-12 else 0
    nodes = np.vstack([lower, cav[first_cav:], upper[1:]])
    if np.hypot(*(nodes[0] - nodes[-1])) > 1e-9:
        nodes = np.vstack([nodes, nodes[0]])
    return nodes


# ---------------------------------------------------------------------------
# flow solve
# ---------------------------------------------------------------------------


@dataclass
class FlowState:
    """Result of one panel-method flow solve at fixed geometry."""

    sigma_src: np.ndarray
    gamma: float
    qt: np.ndarray  # tangential surface speed in the contour direction
    cp: np.ndarray  # pressure coefficient on each panel
    n_residual: np.ndarray  # normal velocity on each panel (0 on wetted panels)
    cl: float
    cd: float
    circulation: float = 0.0


def flow_solve(
    geom: dict,
    alpha_deg: float,
    q_c: float | None = None,
    cavity_mask: np.ndarray | None = None,
    kutta_panels: tuple[int, int] | None = None,
    kutta: str = "extrapolated",
) -> FlowState:
    """Hess-Smith panel solve with either wall or constant-pressure panels.

    ``cavity_mask`` selects the panels whose boundary condition is the cavity
    dynamic condition ``q_t = q_c`` (instead of zero normal velocity).
    ``kutta_panels`` defaults to the two panels adjacent to the trailing edge
    (first and last panel of the contour, which both meet at the TE).

    ``kutta="none"`` closes the system with ``gamma = 0`` instead, i.e. the
    non-lifting solution.  That branch is the **verified** one: for a circular
    section it reproduces the analytic ``q = 2 U sin(theta)`` to better than
    0.1 % (and ``Cp_min = -3``), see
    ``tests/test_cavitation_free_streamline.py``.  The lifting (Kutta) branch is
    *not* validated yet -- it delivers about half of thin-aerofoil theory for a
    symmetric section, so any cavity result built on it is reported WITHHELD
    (see the P40W3B-N02 report).
    """
    a = math.radians(alpha_deg)
    u_inf = np.array([math.cos(a), math.sin(a)])
    n_pan = geom["mid"].shape[0]
    if cavity_mask is None:
        cavity_mask = np.zeros(n_pan, dtype=bool)
    if kutta_panels is None:
        kutta_panels = (0, n_pan - 1)
    sx, sy, vx, vy = _influence_raw(geom["mid"], geom)
    n = geom["n"]
    t = geom["t"]
    # project onto the *target* panel frame (target i, source j)
    Sn_wall = sx * n[:, 0:1] + sy * n[:, 1:2]
    St_wall = sx * t[:, 0:1] + sy * t[:, 1:2]
    Vn_tot = vx * n[:, 0:1] + vy * n[:, 1:2]
    Vt_tot = vx * t[:, 0:1] + vy * t[:, 1:2]
    # source self-influence: outer limit of a constant source sheet (+1/2)
    diag = np.arange(n_pan)
    Sn_wall[diag, diag] = 0.5
    Vn_tot[diag, diag] = 0.0
    Vt_tot[diag, diag] = 0.0
    # RHS / LHS rows: wall panels use the normal condition, cavity panels the
    # tangential (dynamic) condition.
    A = np.empty((n_pan + 1, n_pan + 1))
    b = np.empty(n_pan + 1)
    wall = ~cavity_mask
    A[:n_pan][wall, :n_pan] = Sn_wall[wall, :]
    A[:n_pan][wall, n_pan] = np.sum(Vn_tot, axis=1)[wall]
    b[:n_pan][wall] = -np.sum(n[wall] * u_inf, axis=1)
    if cavity_mask.any():
        i_cav = np.flatnonzero(cavity_mask)
        A[i_cav, :n_pan] = St_wall[i_cav, :]
        A[i_cav, n_pan] = np.sum(Vt_tot, axis=1)[i_cav]
        q_target = float(q_c) if q_c is not None else 0.0
        b[i_cav] = q_target - np.sum(t[i_cav] * u_inf, axis=1)
    # Kutta condition.  The two panels adjacent to the trailing edge carry the
    # surface speed at their midpoints; the strong (equal-and-opposite) form is
    # imposed on the values *extrapolated to the TE point*, because for a nearly
    # closed TE the two midpoints are only a fraction of a panel apart and the
    # mid-point form barely constrains the circulation.  The extrapolation is
    # linear in the panel velocities, so the condition stays a single linear row.
    iu, il = kutta_panels
    iu2 = iu - 1 if iu > 0 else iu + 1
    il2 = il + 1 if il < n_pan - 1 else il - 1
    te = geom["mid"][iu]
    s1u = float(np.hypot(*(geom["mid"][iu] - te)))
    s2u = float(np.hypot(*(geom["mid"][iu2] - te)))
    s1l = float(np.hypot(*(geom["mid"][il] - te)))
    s2l = float(np.hypot(*(geom["mid"][il2] - te)))
    # linear extrapolation of q(s) to s = 0 (the TE): weight (s2, -s1)/(s2-s1)
    def _extrap(s1: float, s2: float) -> tuple[float, float]:
        if abs(s2 - s1) < 1e-12:
            return 1.0, 0.0
        return s2 / (s2 - s1), -s1 / (s2 - s1)

    w1u, w2u = _extrap(s1u, s2u)
    w1l, w2l = _extrap(s1l, s2l)
    comb = np.zeros(n_pan)
    comb[:] = 0.0
    comb[iu] += w1u
    comb[iu2] += w2u
    comb[il] += w1l
    comb[il2] += w2l
    if kutta == "none":
        A[n_pan, :n_pan] = 0.0
        A[n_pan, n_pan] = 1.0
        b[n_pan] = 0.0
    else:
        A[n_pan, :n_pan] = comb @ St_wall
        A[n_pan, n_pan] = float(comb @ np.sum(Vt_tot, axis=1))
        b[n_pan] = -float(comb @ np.sum(t * u_inf, axis=1))
    sol = np.linalg.solve(A, b)
    sigma_src = sol[:n_pan]
    gamma = float(sol[n_pan])
    # surface kinematics: tangential speed and normal velocity on every panel
    qt = np.sum(t * u_inf, axis=1) + St_wall @ sigma_src + Vt_tot @ np.full(n_pan, gamma)
    vn = np.sum(n * u_inf, axis=1) + Sn_wall @ sigma_src + Vn_tot @ np.full(n_pan, gamma)
    vn[wall] = 0.0
    cp = 1.0 - qt**2
    # force coefficients by pressure integration (2-D, chord = 1, unit depth):
    # F = -oint p n ds with p = 0.5 rho U^2 Cp  =>  Cl = -oint Cp n_y ds,
    # Cd = -oint Cp n_x ds (n = verified outward normal, ds = panel length).
    n_ds = geom["n"] * geom["length"][:, None]
    cd = -float(np.sum(cp * n_ds[:, 0]))
    cl_force = -float(np.sum(cp * n_ds[:, 1]))
    circulation = float(np.sum(qt * geom["length"]))  # Kutta-Joukowski cross-check
    return FlowState(
        sigma_src=sigma_src,
        gamma=gamma,
        qt=qt,
        cp=cp,
        n_residual=vn,
        cl=float(cl_force),
        cd=float(cd),
        circulation=circulation,
    )


# ---------------------------------------------------------------------------
# free-boundary (cavity shape + length) iteration
# ---------------------------------------------------------------------------


@dataclass
class CavitySolution:
    """Converged (or best-effort) sheet-cavity solution."""

    kind: str
    alpha_deg: float
    sigma: float
    l_cav: float
    delta_cav: float
    area: float
    q_c: float
    cl: float
    cd: float
    iterations: int
    residual: float
    residual_history: list[float] = field(default_factory=list)
    converged: bool = False
    thickness: list[float] = field(default_factory=list)
    x_cavity: list[float] = field(default_factory=list)
    cp: list[float] = field(default_factory=list)
    x_cp: list[float] = field(default_factory=list)
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "alpha_deg": self.alpha_deg,
            "sigma": self.sigma,
            "l_cav": self.l_cav,
            "delta_cav": self.delta_cav,
            "area": self.area,
            "q_c": self.q_c,
            "cl": self.cl,
            "cd": self.cd,
            "iterations": self.iterations,
            "residual": self.residual,
            "converged": self.converged,
            "residual_history": self.residual_history,
            "thickness": self.thickness,
            "x_cavity": self.x_cavity,
            "cp": self.cp,
            "x_cp": self.x_cp,
            "notes": self.notes,
        }


def solve_cavity(
    kind: str = "NACA0015",
    alpha_deg: float = 5.0,
    sigma: float = 1.0,
    l_init: float | None = None,
    n_foil: int = 120,
    n_cav: int = 48,
    max_iter: int = 300,
    omega: float = 0.6,
    tol: float = 2.0e-4,
    x_detach: float = 0.0,
) -> CavitySolution:
    """Solve the closed constant-pressure cavity on the suction side.

    Iteration: (1) panel flow solve with ``q_t = q_c`` on the cavity panels;
    (2) transpiration update of the cavity thickness from the normal-velocity
    residual; (3) the residual of the last cavity station drives the cavity
    length.  Under-relaxation is halved automatically when the residual grows.
    """
    q_c = math.sqrt(1.0 + sigma)
    # start from a modest cavity: the residual-driven growth is robust but a
    # starting point far from the answer costs iterations.
    L = float(l_init) if l_init is not None else 0.5 * sigma
    L = float(np.clip(L, 0.02, 0.95))
    xi = np.linspace(0.0, 1.0, n_cav + 1)
    thick = 0.5 * L * 0.10 * np.sin(math.pi * xi) ** 0.8
    hist: list[float] = []
    best = None
    resid_prev = math.inf
    lam = omega
    cp = []
    x_cp = []
    x_cav = []
    for it in range(max_iter):
        nodes = foil_contour(
            kind, alpha_deg, n_foil, n_cav, L, thick, x_detach=x_detach
        )
        geom = panel_geometry(nodes)
        n_pan = geom["mid"].shape[0]
        # cavity panel indices: the block right after the lower surface
        i0 = n_foil // 2  # first cavity panel
        i1 = i0 + n_cav - 1  # last cavity panel
        cav = np.zeros(n_pan, dtype=bool)
        cav[i0:i1] = True  # end station panels belong to the wetted surface
        st = flow_solve(geom, alpha_deg, q_c=q_c, cavity_mask=cav)
        res = np.abs(st.n_residual[cav])
        rmax = float(res.max()) if res.size else 0.0
        hist.append(rmax)
        x_cav = (nodes[i0 : i1 + 1, 0]).tolist()
        x_cp = geom["mid"][:, 0].tolist()
        cp = st.cp.tolist()
        if best is None or rmax < best[0]:
            best = (rmax, L, thick.copy(), st)
        if rmax < tol:
            break
        if rmax > resid_prev:
            lam = max(0.05, 0.5 * lam)
        else:
            lam = min(omega, 1.15 * lam)
        resid_prev = rmax
        # transpiration: move the surface along the normal by (v_n * ds)
        ds = np.diff(nodes[i0 : i1 + 2, 0])
        vn = st.n_residual[i0 : i1 + 1]
        dthick = lam * vn * np.clip(ds, 1e-3, None)
        # smooth the correction (removes the panel-to-panel sawtooth)
        kern = np.array([0.25, 0.5, 0.25])
        dthick = np.convolve(np.pad(dthick, 1, mode="edge"), kern, mode="valid")
        thick = np.clip(thick + dthick, 0.0, 0.45)
        thick[0] = 0.0
        thick[-1] = 0.0
        # length: last interior station residual (positive => still too short)
        r_last = float(st.n_residual[i1 - 1])
        L = float(np.clip(L + lam * r_last * 0.5, 0.02, 0.95))
    rmax, L, thick, st = best
    d_max = float(np.max(thick))
    area = float(np.trapezoid(thick, xi) * L) if hasattr(np, "trapezoid") else float(
        np.trapz(thick, xi) * L
    )
    return CavitySolution(
        kind=kind,
        alpha_deg=alpha_deg,
        sigma=sigma,
        l_cav=L,
        delta_cav=d_max,
        area=area,
        q_c=q_c,
        cl=st.cl,
        cd=st.cd,
        iterations=it + 1,
        residual=rmax,
        residual_history=hist,
        converged=bool(rmax < tol),
        thickness=thick.tolist(),
        x_cavity=x_cav,
        cp=cp,
        x_cp=x_cp,
        notes="closed (Riabouchinsky-type) constant-pressure cavity; inviscid",
    )