"""Finite-wing induced-drag verification surface: elliptic loading, tip-vortex
roll-up and the Trefftz-plane energy ledger.

Scope
-----
Induced drag of a finite-span wing is an *inviscid* momentum defect produced by
the trailing vortex sheet.  This module therefore builds a self-contained chain
of classical but fully numerical methods (no look-up tables) so that every
acceptance quantity can be checked against an analytic anchor or against an
independent formulation:

1. ``solve_lifting_line`` -- Prandtl/Glauert lifting-line equation solved by
   Fourier collocation for an arbitrary spanwise chord / twist distribution.
   Analytic anchor: elliptic planform -> pure A_1 loading, span efficiency
   e = 1, C_Di = C_L^2/(pi*AR).
2. ``VortexLattice`` -- horseshoe vortex lattice with a desingularised
   (finite-core) Biot-Savart kernel; independent method for the same wing:
   spanwise loading and near-field induced drag
   D_i = rho * sum_j Gamma_j w_j dy_j.
3. ``trefftz_energy_kirchhoff`` -- far-field energy ledger.  The cross-flow
   kinetic energy per unit length of the trailing vortex sheet is evaluated in
   the Trefftz plane in Kirchhoff/Green function form
   E = (rho/2) iint (v_y^2 + v_z^2) dy dz = -(rho/4 pi) sum_ij G_i G_j ln r_ij,
   and Munk's theorem gives the induced drag as D_i = 2 E (the cross-flow
   kinetic energy and the equal Bernoulli pressure-work part).  This is a
   far-field *energy* route, independent of the near-field force work
   D_i = -rho sum_j G_j w_j dy_j; their difference is the ledger residual.
4. ``tip_vortex_rollup`` -- 2D discrete-vortex method (point vortices with
   Krasny-type core spreading) for the roll-up of the trailing sheet in the
   cross-flow plane; the converged tip-vortex core position is compared with the
   lifting-line prediction y_v = pi*b/8 (Betz roll-up; rolled-up vortex
   separation b0 = pi*b/4 for elliptic loading).

All quantities are non-dimensionalised with U = rho = 1 and wing area S = 1,
hence b = sqrt(AR).

References
----------
* Glauert, H. (1926) The Elements of Aerofoil and Airscrew Theory.
* Prandtl, L. (1918) Tragfluegeltheorie I, Nachr. Ges. Wiss. Goettingen.
* Munk, M. M. (1921) The minimum induced drag of aerofoils, NACA TR 121.
* Betz, A. (1933) Behavior of vortex systems, NACA TM 713.
* Katz, J. & Plotkin, A. (2001) Low-Speed Aerodynamics, 2nd ed., ch. 12-13.
* Rossow, V. J. (1999) Lift-generated vortex wakes of subsonic transport
  aircraft, Prog. Aerosp. Sci. 35, 507-660 (b0 = pi*b/4 rolled-up spacing).
* Krasny, R. (1987) Computation of vortex sheet roll-up in the Trefftz plane,
  J. Fluid Mech. 184, 123-155.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "NACA0012_SECTION",
    "WingGeometry",
    "wing_geometry",
    "chord_at",
    "solve_lifting_line",
    "LiftingLineResult",
    "VortexLattice",
    "VLMWake",
    "wake_from_loading",
    "elliptic_twist",
    "tip_vortex_position_prediction",
    "trefftz_energy_kirchhoff",
    "tip_vortex_rollup",
    "RollupResult",
    "elliptic_spanload",
    "prandtl_induced_drag",
    "elliptic_loading_deviation",
    "aspect_ratio_scaling",
    "run_induced_drag_study",
]

# ---------------------------------------------------------------------------
# Section data: NACA 0012 (symmetric, zero design lift)
# ---------------------------------------------------------------------------
# Thin-aerofoil lift-curve slope 2*pi/rad; the NACA 0012 mean line is straight
# (alpha_L0 = 0), which is what a thin lifting-surface model needs.
NACA0012_SECTION = {
    "a0": 2.0 * math.pi,       # section lift slope [1/rad]
    "alpha_L0": 0.0,           # zero-lift angle [rad]
    "thickness_ratio": 0.12,
}


# ---------------------------------------------------------------------------
# Planform geometry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WingGeometry:
    """Non-dimensional finite wing (U = rho = 1, S = 1)."""

    aspect_ratio: float
    kind: str = "elliptic"          # "elliptic" | "rectangular"
    area: float = 1.0
    span: float = field(init=False)
    root_chord: float = field(init=False)

    def __post_init__(self) -> None:
        kind = self.kind.lower()
        if kind not in ("elliptic", "rectangular"):
            raise ValueError(f"unknown planform kind {self.kind!r}")
        if self.aspect_ratio <= 0:
            raise ValueError("aspect ratio must be positive")
        b = math.sqrt(self.aspect_ratio * self.area)
        c0 = 4.0 * self.area / (math.pi * b) if kind == "elliptic" else self.area / b
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "span", b)
        object.__setattr__(self, "root_chord", c0)

    @property
    def semi_span(self) -> float:
        return 0.5 * self.span

    def chord(self, y):
        """Local chord at spanwise station(s) y (clipped at |y| <= b/2)."""
        y = np.asarray(y, dtype=float)
        eta = np.clip(np.abs(y) / self.semi_span, 0.0, 1.0)
        if self.kind == "elliptic":
            return self.root_chord * np.sqrt(np.maximum(0.0, 1.0 - eta**2))
        return np.full_like(y, self.root_chord)

    def trailing_edge_x(self, y):
        """Trailing-edge streamwise position (leading edge at x = 0)."""
        return self.chord(y)


def wing_geometry(aspect_ratio: float, kind: str = "elliptic",
                  area: float = 1.0) -> WingGeometry:
    return WingGeometry(aspect_ratio=aspect_ratio, kind=kind, area=area)


def chord_at(geo: WingGeometry, y):
    return geo.chord(y)


# ---------------------------------------------------------------------------
# Lifting line (Glauert Fourier collocation)
# ---------------------------------------------------------------------------
@dataclass
class LiftingLineResult:
    """Container for a converged Prandtl lifting-line solution."""

    geometry: WingGeometry
    alpha_deg: float
    a_n: np.ndarray              # Fourier coefficients A_1..A_N
    theta: np.ndarray            # collocation stations
    y: np.ndarray
    chord: np.ndarray
    gamma: np.ndarray            # circulation Gamma(y)
    cl: np.ndarray               # section lift coefficient cl(y)
    circulation_root: float
    cl_total: float
    cd_i: float
    span_efficiency: float       # Oswald e = 1/(1+delta)
    delta: float

    def circulation(self, y):
        """Evaluate Gamma(y) on an arbitrary spanwise grid (Fourier series)."""
        y = np.asarray(y, dtype=float)
        b = self.geometry.span
        eta = np.clip(np.abs(y) / (0.5 * b), 0.0, 1.0)
        theta = np.arccos(eta)
        n = np.arange(1, self.a_n.size + 1)
        s = np.sin(np.outer(theta, n))
        return 2.0 * b * (s @ self.a_n)


def solve_lifting_line(aspect_ratio: float, kind: str = "elliptic",
                       alpha_deg: float = 4.0, n_terms: int = 40,
                       section: dict | None = None, area: float = 1.0,
                       twist_deg=None) -> LiftingLineResult:
    """Solve Prandtl's monoplane equation with Glauert's Fourier series.

    Gamma(theta) = 2 b sum_n A_n sin(n theta), y = -(b/2) cos theta, collocation
    at theta_m = m pi/(N+1).  The linear system is

        sum_n A_n sin(n theta_m) [sin(theta_m) + n mu_m]
            = mu_m alpha_m sin(theta_m),      mu = a0 c(theta)/(4 b).
    """
    geo = WingGeometry(aspect_ratio=aspect_ratio, kind=kind, area=area)
    sec = dict(NACA0012_SECTION if section is None else section)
    a0 = float(sec["a0"])
    alpha_l0 = float(sec.get("alpha_L0", 0.0))
    b = geo.span
    n = np.arange(1, n_terms + 1)
    theta = np.arange(1, n_terms + 1) * math.pi / (n_terms + 1)
    y = -0.5 * b * np.cos(theta)
    c = np.asarray(geo.chord(y), dtype=float)
    mu = a0 * c / (4.0 * b)
    alpha = np.full(theta.shape, math.radians(alpha_deg) - alpha_l0)
    if twist_deg is not None:
        # the twist callable is a function of the spanwise station y [deg]
        tw = (twist_deg(y) if callable(twist_deg) else np.asarray(twist_deg, float))
        alpha = alpha + np.radians(np.broadcast_to(tw, theta.shape))
    s = np.sin(np.outer(theta, n))                       # (N, N)
    mat = s * (np.sin(theta)[:, None] + n[None, :] * mu[:, None])
    rhs = mu * alpha * np.sin(theta)
    a_n = np.linalg.solve(mat, rhs)

    cl_total = math.pi * b**2 * a_n[0] / geo.area
    if abs(a_n[0]) > 1e-14:
        delta = float(np.sum(np.arange(2, n_terms + 1) * (a_n[1:] / a_n[0]) ** 2))
    else:                        # no lift (alpha = alpha_L0): no induced drag
        delta = 0.0
    cd_i = cl_total**2 * (1.0 + delta) / (math.pi * geo.aspect_ratio)
    e = 1.0 / (1.0 + delta) if math.isfinite(delta) else 0.0

    th_f = np.linspace(1e-9, math.pi - 1e-9, 401)
    y_f = -0.5 * b * np.cos(th_f)
    c_f = np.asarray(geo.chord(y_f), dtype=float)
    gam_f = 2.0 * b * (np.sin(np.outer(th_f, n)) @ a_n)
    with np.errstate(divide="ignore", invalid="ignore"):
        cl_f = np.where(c_f > 1e-12, 2.0 * gam_f / c_f, 0.0)
    return LiftingLineResult(
        geometry=geo, alpha_deg=alpha_deg, a_n=a_n, theta=theta, y=y_f,
        chord=c_f, gamma=gam_f, cl=cl_f,
        circulation_root=float(2.0 * b * np.sum(a_n * np.sin(n * math.pi / 2.0))),
        cl_total=float(cl_total), cd_i=float(cd_i),
        span_efficiency=float(e), delta=float(delta),
    )


# ---------------------------------------------------------------------------
# Analytic anchors
# ---------------------------------------------------------------------------
def elliptic_spanload(y, span: float, cl_root: float):
    """Elliptic section-lift distribution cl(y) = cl0 sqrt(1-(2y/b)^2)."""
    y = np.asarray(y, dtype=float)
    eta = np.clip(np.abs(y) / (0.5 * span), 0.0, 1.0)
    return cl_root * np.sqrt(np.maximum(0.0, 1.0 - eta**2))


def prandtl_induced_drag(cl: float, aspect_ratio: float, e: float = 1.0) -> float:
    """Prandtl induced-drag formula C_Di = C_L^2 / (pi AR e)."""
    return cl**2 / (math.pi * aspect_ratio * e)


def elliptic_loading_deviation(y, cl, span: float, cl_root: float | None = None) -> dict:
    """Normalised L2 deviation of cl(y) from the elliptic law (shape error)."""
    y = np.asarray(y, dtype=float)
    cl = np.asarray(cl, dtype=float)
    if cl_root is None:
        cl_root = float(np.max(np.abs(cl)))
    ref = elliptic_spanload(y, span, cl_root)
    num = float(np.trapezoid((cl - ref) ** 2, y))
    den = float(np.trapezoid(ref**2, y))
    return {
        "cl_root": cl_root,
        "l2_abs": math.sqrt(num),
        "l2_rel": math.sqrt(num / den) if den > 0 else float("inf"),
        "max_abs_dev": float(np.max(np.abs(cl - ref))),
        "linf_rel": float(np.max(np.abs(cl - ref)) / cl_root) if cl_root > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Biot-Savart primitives (finite segment, desingularised core)
# ---------------------------------------------------------------------------
def _segment_velocity(points, a, b, gamma, rc=0.0):
    """Velocity at `points (M,3)` induced by a straight segment a->b.

    Exact finite-segment Biot-Savart form

        v = Gamma/(4 pi) * (r1 x r2)(|r1|+|r2|)
            / [|r1||r2| (|r1||r2| + r1.r2 + rc^2)]

    which reduces to Gamma/(2 pi h) for an infinite line.  rc regularises the
    kernel near the filament (rc = 0 recovers the singular kernel).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    r1 = points - a
    r2 = points - b
    n1 = np.linalg.norm(r1, axis=-1)
    n2 = np.linalg.norm(r2, axis=-1)
    dot12 = np.einsum("...i,...i->...", r1, r2)
    cross = np.cross(r1, r2)
    denom = n1 * n2 * (n1 * n2 + dot12 + rc**2)
    safe = np.abs(denom) > 1e-300
    scale = np.where(safe, (n1 + n2) / np.where(safe, denom, 1.0), 0.0)
    return (gamma / (4.0 * math.pi)) * cross * scale[..., None]


def _semi_infinite_velocity(points, start, direction, gamma, rc=0.0,
                            length=1.0e6):
    """Velocity of a semi-infinite filament from `start` along `direction`."""
    start = np.asarray(start, dtype=float)
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    end = start + length * d
    return _segment_velocity(points, start, end, gamma, rc)


# ---------------------------------------------------------------------------
# Vortex lattice (horseshoe) solver
# ---------------------------------------------------------------------------
@dataclass
class VLMWake:
    """Trailing-vortex sheet extracted from a VLM solution."""

    y_edges: np.ndarray     # spanwise boundary of each wake filament
    strength: np.ndarray    # streamwise vorticity strength at each boundary
    x_start: np.ndarray     # streamwise origin of each filament (TE or 1/4c)
    span: float

    def velocity(self, points, rc=0.0):
        """Cross-flow velocity induced at `points (M,3)` by the wake sheet."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        v = np.zeros_like(points)
        for y_e, g, xs in zip(self.y_edges, self.strength, self.x_start):
            if g == 0.0:
                continue
            v += _semi_infinite_velocity(points, (xs, y_e, 0.0), (1.0, 0.0, 0.0),
                                         g, rc=rc)
        return v


class VortexLattice:
    """Horseshoe vortex-lattice method for a planar finite wing.

    n_span spanwise strips (cosine spacing) x n_chord chordwise panels.  The
    bound vortex sits on the quarter-chord line of each panel and the control
    point on the three-quarter-chord line; no-penetration is imposed on the flat
    camber surface (NACA 0012 mean line).
    """

    def __init__(self, aspect_ratio: float, kind: str = "elliptic",
                 alpha_deg: float = 4.0, n_span: int = 40, n_chord: int = 6,
                 area: float = 1.0, core: float = 0.0,
                 wake_length: float = 1.0e5, twist_fn=None):
        self.geometry = WingGeometry(aspect_ratio=aspect_ratio, kind=kind,
                                    area=area)
        self.alpha_deg = float(alpha_deg)
        self.n_span = int(n_span)
        self.n_chord = int(n_chord)
        self.core = float(core)
        self.wake_length = float(wake_length)
        # optional geometric twist: local incidence alpha + twist(y) [deg]
        self.twist_fn = twist_fn
        self._build()
        self._solve()
        self._postprocess()

    # -- geometry ----------------------------------------------------------
    def _build(self):
        geo = self.geometry
        b = geo.span
        th = np.arange(self.n_span + 1) * math.pi / self.n_span
        self.y_edges = -0.5 * b * np.cos(th)                 # (n_span+1,)
        self.y_mid = 0.5 * (self.y_edges[:-1] + self.y_edges[1:])
        self.dy = np.diff(self.y_edges)
        self.c_mid = np.asarray(geo.chord(self.y_mid), dtype=float)
        self.x_bound = 0.25 * self.c_mid
        self.x_te = np.asarray(geo.trailing_edge_x(self.y_mid), dtype=float)
        # chordwise bound-vortex / control-point fractions
        self.eta_b = (np.arange(self.n_chord) + 0.25) / self.n_chord
        self.eta_cp = (np.arange(self.n_chord) + 0.75) / self.n_chord
        if self.twist_fn is None:
            self.twist_strip = np.zeros(self.n_span)
        else:
            self.twist_strip = np.asarray(self.twist_fn(self.y_mid), dtype=float)

    def _horseshoe_velocity(self, points, j, m):
        """Unit-strength horseshoe (strip j, chordwise m) velocity at points."""
        c = self.c_mid[j]
        xb = self.eta_b[m] * c
        ya, yb = self.y_edges[j], self.y_edges[j + 1]
        rc = self.core
        v = _segment_velocity(points, (xb, ya, 0.0), (xb, yb, 0.0), 1.0, rc)
        v = v + _semi_infinite_velocity(points, (xb, yb, 0.0), (1.0, 0.0, 0.0),
                                        1.0, rc=rc, length=self.wake_length)
        v = v - _semi_infinite_velocity(points, (xb, ya, 0.0), (1.0, 0.0, 0.0),
                                        1.0, rc=rc, length=self.wake_length)
        return v

    # -- solution ----------------------------------------------------------
    def _solve(self):
        ns, nc = self.n_span, self.n_chord
        n_pan = ns * nc
        cps = np.empty((n_pan, 3))
        for j in range(ns):
            for m in range(nc):
                cps[j * nc + m] = (self.eta_cp[m] * self.c_mid[j],
                                   self.y_mid[j], 0.0)
        self.cp = cps
        aic = np.empty((n_pan, n_pan))
        for j in range(ns):
            for m in range(nc):
                aic[:, j * nc + m] = self._horseshoe_velocity(cps, j, m)[:, 2]
        alpha_local = self.alpha_deg + np.repeat(self.twist_strip, self.n_chord)
        rhs = -np.sin(np.radians(alpha_local))
        gamma = np.linalg.solve(aic, rhs)
        self.gamma_panels = gamma.reshape(ns, nc)
        self.gamma_strip = self.gamma_panels.sum(axis=1)

    def _postprocess(self):
        geo = self.geometry
        u = 1.0
        self.cl_strip = 2.0 * self.gamma_strip / (u * self.c_mid)
        lift = float(np.sum(self.gamma_strip * self.dy))
        self.cl_total = lift / (0.5 * u**2 * geo.area)
        # Near-field induced drag: D_i = rho * sum_j Gamma_j w_j dy_j with the
        # downwash w_j of the aggregated trailing-vortex sheet (leg strength at
        # a strip boundary = Gamma_{k-1} - Gamma_k, which telescopes the
        # lattice into the physical sheet vorticity) evaluated on the
        # bound-vortex line.
        wake_qc = self.wake("quarter_chord")
        pts = np.stack([self.x_bound, self.y_mid, np.zeros_like(self.y_mid)],
                       axis=1)
        self.downwash_strip = wake_qc.velocity(pts, rc=self.core)[:, 2]
        # lift is along +z, so the induced drag is the *work done against the
        # downwash*: D_i = -rho sum_j Gamma_j w_j dy_j  (w_j < 0 for positive lift)
        self.drag_near = float(-np.sum(self.gamma_strip * self.downwash_strip
                                       * self.dy))
        self.cd_i_near = self.drag_near / (0.5 * u**2 * geo.area)

    # -- wake sheet --------------------------------------------------------
    def wake(self, origin: str = "quarter_chord") -> VLMWake:
        """Trailing-vortex sheet as filaments at the strip boundaries."""
        ns = self.n_span
        strength = np.zeros(ns + 1)
        for k in range(ns + 1):
            right = self.gamma_strip[k] if k < ns else 0.0
            left = self.gamma_strip[k - 1] if k > 0 else 0.0
            strength[k] = left - right
        if origin == "trailing_edge":
            ys = np.concatenate(([self.y_edges[0]], self.y_mid,
                                 [self.y_edges[-1]]))
            xs = np.concatenate(([self.x_te[0]], self.x_te, [self.x_te[-1]]))
            x_start = np.interp(self.y_edges, ys, xs)
        elif origin == "quarter_chord":
            x_start = np.full(ns + 1, 0.25 * float(np.mean(self.c_mid)))
        else:
            raise ValueError(f"unknown wake origin {origin!r}")
        return VLMWake(y_edges=self.y_edges, strength=strength,
                       x_start=x_start, span=self.geometry.span)

    def loading(self):
        return {"y": self.y_mid.copy(), "cl": self.cl_strip.copy(),
                "chord": self.c_mid.copy(), "dy": self.dy.copy(),
                "gamma": self.gamma_strip.copy()}


# ---------------------------------------------------------------------------
# Trefftz-plane far-field energy ledger
# ---------------------------------------------------------------------------
def wake_from_loading(circulation_fn, span: float, x_te_fn=None,
                      n_wake: int = 200) -> VLMWake:
    """Trailing-vortex sheet sampled from a spanwise loading Gamma(y).

    The lattice or analytic loading is resampled at ``n_wake`` cosine stations so
    that the sheet discretisation used by the far-field energy ledger can be
    refined independently of the VLM panel count.  ``n_wake`` is the refinement
    parameter of the Trefftz-plane energy integral.
    """
    th = (np.arange(n_wake) + 0.5) * math.pi / n_wake
    y_far = -0.5 * span * np.cos(th)
    g_far = np.asarray(circulation_fn(y_far), dtype=float)
    order = np.argsort(y_far)
    y_far, g_far = y_far[order], g_far[order]
    y_edges = y_far[:-1]
    strength = g_far[:-1] - g_far[1:]
    if x_te_fn is None:
        x_start = np.zeros_like(y_edges)
    else:
        x_start = np.asarray(x_te_fn(y_edges), dtype=float)
    return VLMWake(y_edges=y_edges, strength=strength, x_start=x_start,
                   span=span)


def trefftz_energy_kirchhoff(wake: VLMWake, blob: bool = True,
                             n_quad: int | None = None) -> dict:
    """Cross-flow kinetic energy of the wake sheet in the Trefftz plane.

    The trailing vortex sheet is a set of streamwise vorticity filaments of
    strength ``G_i`` at spanwise positions ``y_i`` (Trefftz plane, x-directed
    lines).  Its cross-flow kinetic energy per unit length is the 2-D Kirchhoff
    energy of that line-vortex system,

        E = (rho/2) iint (v_y^2 + v_z^2) dy dz
          = -(rho/4 pi) sum_i sum_j G_i G_j ln r_ij,

    where the diagonal terms are bounded by the local sheet spacing
    (``r_ii^2 = 0.5 (d_i^2 + d_i^2)`` with ``d_i`` the local filament spacing,
    the standard vortex-blob regularisation of a vortex sheet).  Munk's theorem
    then gives the induced drag as ``D_i = 2 E`` (half kinetic energy, half the
    equal Bernoulli pressure-work term of the cross-flow), which is checked
    against the near-field force work in :func:`run_induced_drag_study`.
    """
    y = np.asarray(wake.y_edges, dtype=float)
    g = np.asarray(wake.strength, dtype=float)
    spacing = np.empty_like(y)
    spacing[1:-1] = 0.5 * (y[2:] - y[:-2])
    spacing[0] = y[1] - y[0]
    spacing[-1] = y[-1] - y[-2]
    r = np.abs(y[:, None] - y[None, :])
    if blob:
        sig2 = 0.5 * (spacing[:, None] ** 2 + spacing[None, :] ** 2)
        kernel = 0.5 * np.log(r**2 + sig2)
    else:                                              # pragma: no cover
        r = np.where(r > 0, r, np.min(spacing))
        kernel = np.log(r)
    energy = -float(np.sum(g[:, None] * g[None, :] * kernel)) / (4.0 * math.pi)
    return {
        "energy_cross": energy,
        "d_i_far": 2.0 * energy,
        "n_filaments": int(y.size),
        "spacing_min": float(np.min(spacing)),
        "spacing_max": float(np.max(spacing)),
    }


def tip_vortex_position_prediction(span: float) -> float:
    """Betz roll-up position of the tip vortex for elliptic loading, pi*b/8.

    The fully rolled-up trailing vortex pair of an elliptically loaded wing has
    the spanwise separation ``b0 = pi*b/4`` (Betz 1933; Rossow 1999), i.e. each
    vortex sits at ``y_v = pi*b/8 = 0.3927 b``.  Equivalent statement: the
    spanwise position of the vorticity centroid of the (elliptic) loading.
    """
    return math.pi * span / 8.0


def elliptic_twist(aspect_ratio: float, alpha_deg: float,
                   section: dict | None = None, kind: str = "rectangular",
                   area: float = 1.0):
    """Washout distribution that produces elliptic loading on a wing.

    Requiring ``cl(y) = cl0 sqrt(1 - (2y/b)^2)`` with a uniform induced angle
    ``alpha_i = cl0/(4 AR)`` (elliptic loading of a constant-chord wing) gives

        alpha(y) = alpha_i + cl0/a0 sqrt(1-(2y/b)^2),
        cl0      = a0 alpha / (1 + a0/(4 AR)),

    so the geometric twist relative to the root is a pure washout
    ``t(y) = -(cl0/a0)(1 - sqrt(1-(2y/b)^2))`` [rad].  Returns a function of the
    spanwise station ``y`` in degrees.
    """
    geo = WingGeometry(aspect_ratio=aspect_ratio, kind=kind, area=area)
    sec = dict(NACA0012_SECTION if section is None else section)
    a0 = float(sec["a0"])
    alpha = math.radians(alpha_deg) - float(sec.get("alpha_L0", 0.0))
    cl0 = a0 * alpha / (1.0 + a0 / (4.0 * geo.aspect_ratio))
    twist0 = -cl0 / a0

    def twist(y):
        y = np.asarray(y, dtype=float)
        eta = np.clip(np.abs(y) / geo.semi_span, 0.0, 1.0)
        return np.degrees(twist0 * (1.0 - np.sqrt(np.maximum(0.0, 1.0 - eta**2))))

    return twist


# ---------------------------------------------------------------------------
# Tip-vortex roll-up (2D discrete vortex method, Trefftz plane)
# ---------------------------------------------------------------------------
@dataclass
class RollupResult:
    """Post-roll-up state of the trailing vortex sheet in the Trefftz plane."""

    y: np.ndarray
    z: np.ndarray
    gamma: np.ndarray
    t_final: float
    n_step: int
    core_sigma: np.ndarray
    tip_y: float                 # vorticity-maximum core position (y)
    tip_z: float
    tip_y_left: float
    history_y: np.ndarray        # core y-position history
    history_t: np.ndarray
    tip_circulation: float       # |Gamma| contained in the tip cluster

    def cluster_centroid(self, side: int = +1, y_cut_rel: float = 0.55,
                         span: float | None = None):
        """Vorticity-weighted centroid of the outer (tip) region."""
        b = span if span is not None else 2.0 * float(np.max(np.abs(self.y)))
        sel = side * self.y > y_cut_rel * 0.5 * b
        if not np.any(sel):
            return float("nan"), float("nan"), 0.0
        w = np.abs(self.gamma[sel])
        return (float((self.y[sel] * w).sum() / w.sum()),
                float((self.z[sel] * w).sum() / w.sum()),
                float(w.sum()))


def _sheet_filaments(circulation_fn, span: float, n_sheet: int = 160):
    """Discretise the streamwise vorticity of the wake sheet.

    Returns filament positions y_k (at half-panel boundaries), strengths
    nu_k = Gamma_out(y_k) - Gamma_in(y_k) and the total circulation.
    """
    th = (np.arange(n_sheet) + 0.5) * math.pi / n_sheet
    y_far = -0.5 * span * np.cos(th)                       # half-panel centres
    g_far = np.asarray(circulation_fn(y_far), dtype=float)
    order = np.argsort(y_far)
    y_far, g_far = y_far[order], g_far[order]
    # filament at each interior boundary between half panels
    y_k = y_far[:-1]
    nu_k = g_far[:-1] - g_far[1:]
    return y_k, nu_k


def _crossflow_velocity(y, z, gamma, sigma, exclude_self=True):
    """2D Biot-Savart of line vortices (x-directed) with Rankine cores."""
    dy = y[None, :] - y[:, None]
    dz = z[None, :] - z[:, None]
    s2 = sigma[None, :] ** 2
    r2 = dy**2 + dz**2 + s2
    if exclude_self:
        np.fill_diagonal(r2, np.inf)
    k = gamma[None, :] / (2.0 * math.pi * r2)
    vy = -np.einsum("ij,ij->i", k, dz)
    vz = np.einsum("ij,ij->i", k, dy)
    return vy, vz


def tip_vortex_rollup(circulation_fn, span: float, n_sheet: int = 160,
                      t_end: float | None = None, gamma_ref: float | None = None,
                      core_ratio: float = 0.02, alpha_core: float = 0.002,
                      cfl: float = 0.4, max_steps: int = 40000,
                      n_hist: int = 60, remove_drift: bool = True) -> RollupResult:
    """Roll up the trailing vortex sheet in the Trefftz plane.

    Point vortices with Krasny-style core spreading
    ``sigma^2(t) = sigma0^2 + 4 alpha_core (Gamma_ref/2pi) t``; classical
    second-order (midpoint) time integration with a CFL condition on the
    maximum induced velocity.
    """
    y, nu = _sheet_filaments(circulation_fn, span, n_sheet=n_sheet)
    z = np.zeros_like(y)
    gamma = nu
    if gamma_ref is None:
        gamma_ref = float(np.max(np.abs(circulation_fn(np.linspace(-0.5*span,0.5*span,201)))))
    sigma0 = np.full_like(y, core_ratio * span)
    t = 0.0
    if t_end is None:
        t_end = 4.0 * span**2 / max(gamma_ref, 1e-12)

    hist_t, hist_y = [], []
    next_hist = 0.0
    n_step = 0
    dt_min = 1e-9 * span**2 / max(gamma_ref, 1e-12)

    def core(tv):
        return np.sqrt(sigma0**2 + 4.0 * alpha_core * (gamma_ref / (2.0 * math.pi)) * tv)

    while t < t_end and n_step < max_steps:
        sig = core(t)
        vy, vz = _crossflow_velocity(y, z, gamma, sig)
        if remove_drift:
            # the counter-rotating pair self-propels in the cross-flow plane
            # (physical wake descent); remove that rigid translation so that the
            # roll-up geometry is computed in the pair frame
            wgt = np.abs(gamma)
            vy = vy - float(np.sum(wgt * vy) / np.sum(wgt))
            vz = vz - float(np.sum(wgt * vz) / np.sum(wgt))
        vmax = float(np.max(np.hypot(vy, vz))) + 1e-12
        dt = cfl * float(np.min(sig)) / vmax
        dt = min(dt, t_end - t)
        if dt < dt_min:
            break
        # midpoint (RK2)
        ym = y + 0.5 * dt * vy
        zm = z + 0.5 * dt * vz
        sig_m = core(t + 0.5 * dt)
        vym, vzm = _crossflow_velocity(ym, zm, gamma, sig_m)
        if remove_drift:
            wgt = np.abs(gamma)
            vym = vym - float(np.sum(wgt * vym) / np.sum(wgt))
            vzm = vzm - float(np.sum(wgt * vzm) / np.sum(wgt))
        y = y + dt * vym
        z = z + dt * vzm
        t += dt
        n_step += 1
        if t >= next_hist:
            hist_t.append(t)
            hist_y.append(float(np.average(np.abs(y), weights=np.abs(gamma) + 1e-30)))
            next_hist += t_end / n_hist

    sig = core(t)
    res = RollupResult(y=y, z=z, gamma=gamma, t_final=t, n_step=n_step,
                       core_sigma=sig, tip_y=float("nan"), tip_z=float("nan"),
                       tip_y_left=float("nan"),
                       history_y=np.asarray(hist_y), history_t=np.asarray(hist_t),
                       tip_circulation=float(np.sum(np.abs(gamma))) / 2.0)
    res.tip_y, res.tip_z, _ = res.cluster_centroid(+1, 0.55, span)
    res.tip_y_left, _, _ = res.cluster_centroid(-1, 0.55, span)
    return res


def _core_position(res: RollupResult, span: float, side: int = +1,
                   n_grid: int = 401, sigma_kde: float | None = None):
    """Vortex-core location = maximum of the smoothed vorticity magnitude."""
    if sigma_kde is None:
        sigma_kde = 0.02 * span
    sel = side * res.y > 0.0
    if not np.any(sel):
        return float("nan"), float("nan")
    ys = res.y[sel]
    zs = res.z[sel]
    gs = np.abs(res.gamma[sel])
    lo, hi = 0.0, 0.55 * span
    yg = np.linspace(lo, hi, n_grid)
    zg = np.linspace(-0.25 * span, 0.25 * span, n_grid // 2)
    yy = yg[:, None]
    zz = zg[None, :]
    dens = np.zeros((n_grid, n_grid // 2))
    for yi, zi, gi in zip(ys, zs, gs):
        dens += gi * np.exp(-((yy - yi) ** 2 + (zz - zi) ** 2) / (2 * sigma_kde**2))
    i, j = np.unravel_index(np.argmax(dens), dens.shape)
    return float(yg[i]), float(zg[j])


# ---------------------------------------------------------------------------
# Aspect-ratio scaling law and full study driver
# ---------------------------------------------------------------------------
def aspect_ratio_scaling(points) -> dict:
    """Fit C_Di*AR = slope * C_L^2 (through the origin) and compare 1/(pi e)."""
    cl = np.asarray([p["cl"] for p in points], dtype=float)
    cdi = np.asarray([p["cd_i"] for p in points], dtype=float)
    ar = np.asarray([p["ar"] for p in points], dtype=float)
    x, yv = cl**2, cdi * ar
    slope = float(np.sum(x * yv) / np.sum(x * x))
    pred = slope * x
    ss_res = float(np.sum((yv - pred) ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    e_implied = 1.0 / (math.pi * slope)
    return {
        "slope": slope,
        "intercept_free_slope_1_over_pi_e": slope,
        "r2": r2,
        "e_implied": e_implied,
        "points": [{"ar": float(a), "cl": float(c), "cd_i": float(d),
                    "cl2": float(c * c), "cdi_ar": float(d * a)}
                   for a, c, d in zip(ar, cl, cdi)],
    }


def run_induced_drag_study(ar_list=(6.0, 8.0, 10.0), alpha_deg: float = 4.0,
                           kinds=("elliptic", "rectangular", "twisted_rectangular"),
                           n_span: int = 40, n_chord: int = 6,
                           n_wake: int = 200, n_sheet: int = 160,
                           do_rollup: bool = True, rollup_kwargs=None) -> dict:
    """Run the finite-wing induced-drag verification sweep.

    For every (planform, aspect ratio) the following are produced:

    * lifting-line solution (analytic anchor: elliptic wing -> e = 1),
    * vortex-lattice solution: spanwise loading, C_L, near-field D_i,
    * Trefftz-plane cross-flow energy ledger (far-field D_i),
    * tip-vortex roll-up (DVM) position versus the Betz prediction,
    * the four acceptance checks C1..C4.
    """
    out = {"alpha_deg": alpha_deg, "cases": [], "scaling": {}}
    rollup_kwargs = dict(rollup_kwargs or {})
    for kind in kinds:
        planform = "rectangular" if kind.startswith("twisted") else kind
        twist_fn = (elliptic_twist(ar, alpha_deg, kind="rectangular")
                    if kind.startswith("twisted") else None)
        points = []
        for ar in ar_list:
            tw = elliptic_twist(ar, alpha_deg, kind=planform) if kind.startswith("twisted") else None
            ll = solve_lifting_line(ar, kind=planform, alpha_deg=alpha_deg,
                                    twist_deg=tw)
            vlm = VortexLattice(ar, kind=planform, alpha_deg=alpha_deg,
                                n_span=n_span, n_chord=n_chord, twist_fn=tw)
            span = ll.geometry.span
            chord = ll.geometry.area / span
            # --- spanwise loading law -------------------------------------
            # normalized lift per unit span (circulation) and section lift
            gam_norm = ll.circulation(vlm.y_mid) / max(abs(ll.circulation_root), 1e-30)
            dev_circ = elliptic_loading_deviation(vlm.y_mid, gam_norm, span)
            dev_cl = elliptic_loading_deviation(vlm.y_mid, vlm.cl_strip, span)
            # --- far-field energy ledger ----------------------------------
            wake = wake_from_loading(ll.circulation, span, n_wake=n_wake)
            ledger = trefftz_energy_kirchhoff(wake)
            cdi_prandtl = prandtl_induced_drag(vlm.cl_total, ar,
                                              e=ll.span_efficiency)
            case = {
                "kind": kind,
                "planform": planform,
                "ar": float(ar),
                "span": float(span),
                "root_chord": float(ll.geometry.root_chord),
                "chord_mean": float(chord),
                "twist_tip_deg": (float(tw(vlm.y_mid[-1])) if tw is not None else 0.0),
                "ll": {"cl": ll.cl_total, "cd_i": ll.cd_i,
                       "e": ll.span_efficiency, "delta": ll.delta,
                       "a1": float(ll.a_n[0]),
                       "a3_over_a1": float(abs(ll.a_n[2] / ll.a_n[0])),
                       "circulation_root": ll.circulation_root,
                       "lift_slope_per_rad": ll.cl_total / math.radians(alpha_deg)},
                "vlm": {"cl": vlm.cl_total, "cd_i_near": vlm.cd_i_near,
                        "n_span": n_span, "n_chord": n_chord},
                "loading": {
                    "y": vlm.y_mid.tolist(),
                    "cl_vlm": vlm.cl_strip.tolist(),
                    "cl_ll": [float(x) for x in
                              (np.interp(vlm.y_mid, ll.y, ll.cl))],
                    "gamma_norm": gam_norm.tolist(),
                    "elliptic_circ": elliptic_spanload(vlm.y_mid, span, 1.0).tolist(),
                    "elliptic_cl": elliptic_spanload(vlm.y_mid, span,
                                                     float(np.max(vlm.cl_strip))).tolist(),
                },
                "deviation": {"circ_l2_rel": dev_circ["l2_rel"],
                              "circ_linf_rel": dev_circ["linf_rel"],
                              "cl_l2_rel": dev_cl["l2_rel"],
                              "cl_linf_rel": dev_cl["linf_rel"]},
                "prandtl": {"cd_i": cdi_prandtl, "e": ll.span_efficiency},
                "trefftz": {"d_i_far": ledger["d_i_far"],
                            "energy_cross": ledger["energy_cross"],
                            "n_filaments": ledger["n_filaments"]},
                "betz_tip_position": tip_vortex_position_prediction(span),
            }
            case["checks"] = {
                "c2_prandtl_rel": abs(vlm.cd_i_near - cdi_prandtl) / cdi_prandtl,
                "c3_ledger_residual": abs(ledger["d_i_far"] - vlm.cd_i_near)
                / vlm.cd_i_near,
                "c3_far_over_near": ledger["d_i_far"] / vlm.cd_i_near,
            }
            if do_rollup:
                res = tip_vortex_rollup(lambda yy: ll.circulation(yy), span,
                                        n_sheet=n_sheet, **rollup_kwargs)
                y_pred = case["betz_tip_position"]
                case["rollup"] = {
                    "tip_y": res.tip_y, "tip_z": res.tip_z,
                    "tip_y_left": res.tip_y_left,
                    "y_pred_betz": y_pred,
                    "abs_err": abs(abs(res.tip_y) - y_pred),
                    "rel_err": abs(abs(res.tip_y) - y_pred) / y_pred,
                    "err_over_chord": abs(abs(res.tip_y) - y_pred) / chord,
                    "t_final": res.t_final, "n_step": res.n_step,
                    "n_sheet": n_sheet,
                    "core_sigma_max": float(np.max(res.core_sigma)),
                    "trajectory_t": res.history_t.tolist(),
                    "trajectory_y": res.history_y.tolist(),
                    "final_y": res.y.tolist(), "final_z": res.z.tolist(),
                    "final_gamma": res.gamma.tolist(),
                }
                case["checks"]["c4_tip_over_chord"] = case["rollup"]["err_over_chord"]
                case["checks"]["c4_tip_rel"] = case["rollup"]["rel_err"]
            points.append({"ar": ar, "cl": vlm.cl_total,
                           "cd_i": vlm.cd_i_near})
            out["cases"].append(case)
        out["scaling"][kind] = aspect_ratio_scaling(points)
    return out
