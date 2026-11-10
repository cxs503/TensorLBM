"""Ship-deck thermal field: solar / sky / sea boundary -> temperature field.

Scope
-----
``tensorlbm.deck_thermal_load`` answers "what is the equilibrium temperature of
a deck *panel*" (0-D lumped node).  This module answers the next question:
**what temperature field does a deck actually carry** once the absorbed solar
load (direct + diffuse + ground/sea-reflected), the long-wave exchange with sky
and sea and the convective exchange with the air drive a conducting slab whose
underside is cooled by the interior space.

Two solvers, deliberately independent
-------------------------------------
1. :func:`solve_deck_field` -- a **thermal lattice-Boltzmann (D2Q5 ADE)**
   conduction solver on the physical grid (``x`` along the deck, ``z`` through
   the thickness) with a nonlinear surface flux in the top row, a convective
   (Robin) flux in the bottom row, reflecting (adiabatic) lateral ends and an
   optional frozen linearisation of the radiative coefficient (used to make the
   comparison with (2) a pure discretisation test).
2. :func:`analytic_steady_field_1d` and :func:`analytic_steady_2d_fourier` --
   two **independent analytic** steady-state solutions:

   * 1-D through-thickness: the nonlinear surface balance
     ``q_net(T_s) = (T_s - T_in)/(L/k + 1/h_in)`` is solved by bisection
     (bracket-independent, 200 halvings), then the interior profile is linear:
     ``T(z) = T_in + q (1/h_in + z/k)``;
   * 2-D shadow edge: the linearised problem (Robin top with
     ``h_eff = 4 eps sigma T_m^3 + h_c``, Robin bottom, adiabatic ends) is solved
     exactly by a cosine series ``T = sum_n theta_n(z) cos(n pi x / W)`` with
     ``theta_n = a_n cosh(lambda_n z) + b_n sinh(lambda_n z)``.

Solar boundary, three explicit components
-----------------------------------------
:func:`solar_flux_components` returns the absorbed direct-beam, isotropic
sky-diffuse and ground(sea)-reflected terms separately.  Its ``total`` is the
same floating-point expression as
:func:`tensorlbm.deck_thermal_load.absorbed_solar_flux`; a unit test pins the
two together so the split can never drift from the established aggregate.

Discretisation notes (honest limits)
------------------------------------
* Surface fluxes are applied as **volume sources in the boundary row**
  (``S = q/dx``), i.e. at the row centre, a half cell inside the face.  The
  induced temperature offset is ``O(q dx / k)`` -- for the ship-deck cases here
  ~1e-2 K, three decades below the 3 % gate -- but it *is* a first-order
  boundary treatment; the grid-convergence study in the benchmark quantifies it.
* D2Q5 needs a square lattice, so ``dx = L/ny`` fixes ``nx`` for a given deck
  width.  A real deck aspect ratio (12 m / 20 mm = 600) is therefore not
  resolved in plane; the 2-D case here is a 0.2 m shadow-edge patch.  This is
  physics, not a shortcut: the in-plane diffusion time of a steel deck
  (~1e7 s) is seven decades longer than the through-thickness time (~35 s), so
  a deck sits close to local radiative equilibrium with only weak lateral
  smoothing.

References
----------
Berdahl, P., Martin, M. (1984). *Solar Energy* 32, 663 (clear-sky emissivity).
ASHRAE Handbook of Fundamentals (2009), ch. 14-15.
Incropera et al. (2007). *Fundamentals of Heat and Mass Transfer*, 6th ed.,
    ch. 5 (2-D steady conduction by separation of variables).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .deck_thermal_load import (
    DeckPanel,
    SkySeaEnvironment,
    SunState,
    absorbed_solar_flux,
    cosine_incidence,
    deck_energy_balance,
    deck_view_factors,
)
from .gray_enclosure_radiosity import STEFAN_BOLTZMANN

__all__ = [
    "DeckMaterial",
    "DeckFieldSolution",
    "solar_flux_components",
    "surface_net_flux",
    "analytic_steady_field_1d",
    "analytic_steady_2d_fourier",
    "solve_deck_field",
    "cosine_coefficients",
]


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------


@dataclass
class DeckMaterial:
    """Homogeneous deck slab properties."""

    k: float = 45.0  # W/(m K), shipbuilding steel ~45-50
    rho: float = 7800.0  # kg/m3
    cp: float = 500.0  # J/(kg K)
    thickness: float = 0.020  # m

    def __post_init__(self) -> None:
        for label in ("k", "rho", "cp", "thickness"):
            value = getattr(self, label)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be finite and > 0, got {value!r}")

    @property
    def alpha(self) -> float:
        """Thermal diffusivity [m2/s]."""
        return float(self.k / (self.rho * self.cp))


@dataclass
class DeckFieldSolution:
    """Steady deck temperature field plus the audits that make it citable."""

    T: np.ndarray  # (ny, nx) [K]; row 0 = underside, row ny-1 = top face
    x: np.ndarray  # (nx,) in-plane coordinates [m]
    z: np.ndarray  # (ny,) through-thickness coordinates [m]
    q_top: np.ndarray  # (nx,) net flux into the top face [W/m2]
    q_bottom: np.ndarray  # (nx,) net flux into the bottom face [W/m2]
    steps: int
    dt: float
    dx: float
    tau: float
    converged: bool
    max_step_change: float  # max|dT| over the last check window [K]
    energy_residual: float  # |sum q_top + sum q_bottom| / sum|q_top|
    solar: dict
    linearised: bool

    def as_dict(self) -> dict:
        return {
            "steps": int(self.steps),
            "dt": float(self.dt),
            "dx": float(self.dx),
            "tau": float(self.tau),
            "converged": bool(self.converged),
            "max_step_change": float(self.max_step_change),
            "energy_residual": float(self.energy_residual),
            "T_min": float(self.T.min()),
            "T_max": float(self.T.max()),
            "T_top_mean": float(self.T[-1].mean()),
            "T_bottom_mean": float(self.T[0].mean()),
            "q_top_mean": float(self.q_top.mean()),
            "q_bottom_mean": float(self.q_bottom.mean()),
            "solar": {k: float(v) for k, v in self.solar.items()},
            "linearised": bool(self.linearised),
        }


# ---------------------------------------------------------------------------
# solar boundary: three explicit components
# ---------------------------------------------------------------------------


def solar_flux_components(panel: DeckPanel, sun: SunState) -> dict:
    r"""Absorbed solar flux split into its three physical components [W/m2].

    .. math::
        q_b = \alpha_s G_b\cos\theta, \qquad
        q_d = \alpha_s G_d F_{sky}, \qquad
        q_r = \alpha_s G_b\cos\theta_h \rho_{sea} F_{sea}

    with :math:`F_{sky} = (1+\cos\beta)/2` and :math:`F_{sea} = 1 - F_{sky}`
    (closed-form sky-dome family), :math:`\theta` the incidence angle on the
    tilted panel and :math:`\theta_h` the incidence angle on a horizontal
    surface (what the sea actually sees).
    """
    F_sky, F_sea = deck_view_factors(panel.tilt_deg)
    direct_inc = sun.beam_irradiance * cosine_incidence(panel, sun)
    diffuse_inc = sun.diffuse_horizontal * F_sky
    cos_h = math.sin(math.radians(max(0.0, sun.elevation_deg)))
    ground_inc = sun.beam_irradiance * cos_h * sun.sea_albedo * F_sea
    alpha = panel.solar_absorptance
    return {
        "direct_incident": float(direct_inc),
        "diffuse_incident": float(diffuse_inc),
        "ground_incident": float(ground_inc),
        "incident_total": float(direct_inc + diffuse_inc + ground_inc),
        "direct": float(alpha * direct_inc),
        "diffuse": float(alpha * diffuse_inc),
        "ground_reflected": float(alpha * ground_inc),
        "total": float(alpha * (direct_inc + diffuse_inc + ground_inc)),
        "F_sky": float(F_sky),
        "F_sea": float(F_sea),
    }


def surface_net_flux(
    T_surf: float | np.ndarray,
    panel: DeckPanel,
    env: SkySeaEnvironment,
    q_solar_abs: float | np.ndarray,
    longwave: str = "analytic",
) -> np.ndarray:
    r"""Net heat flux *into* the deck at its upper face [W/m2].

    ``q = q_solar + q_lw(T) + h_c (T_air - T)`` with ``q_lw`` from
    :func:`tensorlbm.deck_thermal_load.deck_longwave_analytic` (closed form,
    black surroundings) or from the gray enclosure.  Scalar or array input.
    """
    T = np.atleast_1d(np.asarray(T_surf, dtype=np.float64))
    q_solar = np.broadcast_to(np.asarray(q_solar_abs, dtype=np.float64), T.shape)
    if longwave == "analytic":
        F_sky, F_sea = deck_view_factors(panel.tilt_deg)
        T_eff = (F_sky * env.sky_temperature() ** 4 + F_sea * env.T_sea**4) ** 0.25
        q_lw = panel.emissivity * STEFAN_BOLTZMANN * (T_eff**4 - T**4)
    elif longwave == "enclosure":
        q_lw = np.array(
            [
                deck_energy_balance(float(t), panel, env, longwave="enclosure")["q_longwave"]
                for t in T
            ]
        )
    else:
        raise ValueError(f"longwave must be 'analytic' or 'enclosure', got {longwave!r}")
    return q_solar + q_lw + panel.h_conv * (env.T_air - T)


# ---------------------------------------------------------------------------
# analytic anchor 1: through-thickness steady field
# ---------------------------------------------------------------------------


def analytic_steady_field_1d(
    panel: DeckPanel,
    env: SkySeaEnvironment,
    material: DeckMaterial,
    T_interior: float,
    h_interior: float,
    q_solar_abs: float | None = None,
    bracket: tuple[float, float] = (150.0, 800.0),
    n_bisect: int = 200,
    n_points: int = 5,
    longwave: str = "analytic",
) -> dict:
    r"""Analytic steady 1-D deck field (bisection + linear interior profile).

    At steady state the flux crossing the slab equals the surface net load:

    .. math::
        q_{net}(T_s) = \frac{T_s - T_{in}}{R}, \qquad
        R = \frac{L}{k} + \frac{1}{h_{in}},

    solved for :math:`T_s` by plain bisection (a different algorithm from the
    Brent solve used in ``deck_thermal_load``).  The interior profile follows
    from the constant flux:

    .. math::
        T(z) = T_{in} + q\left(\frac{1}{h_{in}} + \frac{z}{k}\right),
        \quad z \in [0, L].

    ``h_interior = 0`` degenerates to an adiabatic underside: then ``q = 0``
    and the steady state is the isothermal 0-D equilibrium.
    """
    if not math.isfinite(T_interior) or T_interior <= 0.0:
        raise ValueError(f"T_interior must be finite and > 0, got {T_interior!r}")
    if not math.isfinite(h_interior) or h_interior < 0.0:
        raise ValueError(f"h_interior must be finite and >= 0, got {h_interior!r}")
    if q_solar_abs is None:
        q_solar_abs = absorbed_solar_flux(panel, env.sun)
    L, k = material.thickness, material.k
    lo, hi = bracket
    if h_interior == 0.0:
        f_lo = float(surface_net_flux(lo, panel, env, q_solar_abs, longwave)[0])
        f_hi = float(surface_net_flux(hi, panel, env, q_solar_abs, longwave)[0])
        if f_lo * f_hi > 0.0:
            raise ValueError(f"no sign change of q_net on bracket {bracket}")
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            f_mid = float(surface_net_flux(mid, panel, env, q_solar_abs, longwave)[0])
            if f_lo * f_mid <= 0.0:
                hi = mid
            else:
                lo, f_lo = mid, f_mid
        T_s = 0.5 * (lo + hi)
        q = 0.0
    else:
        R = L / k + 1.0 / h_interior

        def residual(T_s: float) -> float:
            q_surf = float(surface_net_flux(T_s, panel, env, q_solar_abs, longwave)[0])
            return q_surf - (T_s - T_interior) / R

        f_lo, f_hi = residual(lo), residual(hi)
        if f_lo * f_hi > 0.0:
            raise ValueError(
                f"no sign change of the steady-state residual on {bracket}: "
                f"r({lo})={f_lo:.4g}, r({hi})={f_hi:.4g}"
            )
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            f_mid = residual(mid)
            if f_lo * f_mid <= 0.0:
                hi = mid
            else:
                lo, f_lo = mid, f_mid
        T_s = 0.5 * (lo + hi)
        q = (T_s - T_interior) / R
    z = np.linspace(0.0, L, n_points)
    if h_interior == 0.0:
        profile = np.full_like(z, T_s)
    else:
        profile = T_interior + q * (1.0 / h_interior + z / k)
    return {
        "T_surface": float(T_s),
        "T_interior": float(T_interior),
        "q": float(q),
        "z": z.tolist(),
        "T_profile": profile.tolist(),
        "resistance": float(L / k + (1.0 / h_interior if h_interior > 0 else math.inf)),
        "bisections": int(n_bisect),
        "residual": float(
            float(surface_net_flux(T_s, panel, env, q_solar_abs, longwave)[0]) - q
        ),
        "q_solar_abs": float(q_solar_abs),
        "h_interior": float(h_interior),
        "longwave": longwave,
    }


# ---------------------------------------------------------------------------
# analytic anchor 2: 2-D (shadow-edge) cosine-series solution
# ---------------------------------------------------------------------------


def cosine_coefficients(
    edges: np.ndarray | list[float],
    values: np.ndarray | list[float],
    width: float,
    n_modes: int,
) -> np.ndarray:
    r"""Exact cosine coefficients of a piecewise-constant profile on ``[0, W]``.

    :math:`A_0 = W^{-1}\int_0^W A\,dx`,
    :math:`A_n = 2W^{-1}\int_0^W A\cos(n\pi x/W)\,dx`, each integral in closed
    form per panel (no quadrature error).
    """
    edges = np.asarray(edges, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if edges.ndim != 1 or values.ndim != 1 or values.size != edges.size - 1:
        raise ValueError("edges must have len(values) + 1 entries")
    if not (math.isfinite(width) and width > 0.0):
        raise ValueError(f"width must be finite and > 0, got {width!r}")
    if edges[0] < -1e-12 or edges[-1] > width + 1e-12:
        raise ValueError("edges must lie inside [0, width]")
    coef = np.zeros(n_modes + 1)
    for a, b, v in zip(edges[:-1], edges[1:], values):
        if b <= a:
            continue
        coef[0] += v * (b - a) / width
        for n in range(1, n_modes + 1):
            lam = n * math.pi / width
            coef[n] += 2.0 * v * (math.sin(lam * b) - math.sin(lam * a)) / (width * lam)
    return coef


def analytic_steady_2d_fourier(
    panel: DeckPanel,
    env: SkySeaEnvironment,
    material: DeckMaterial,
    T_interior: float,
    h_interior: float,
    solar_edges: np.ndarray | list[float],
    solar_values: np.ndarray | list[float],
    width: float,
    T_linearisation: float,
    n_modes: int = 400,
    longwave: str = "analytic",
) -> dict:
    r"""Exact steady solution of the *linearised* 2-D deck problem.

    Top face (radiation linearised about :math:`T_m`, plus convection; the
    conductive flux in :math:`+z` is :math:`-k\partial_z T`):

    .. math::
        k\,\partial_z T\big|_{L} = A(x) - h_{eff}T(L), \qquad
        A(x) = q_s(x) + \varepsilon\sigma(T_{eff}^4 - T_m^4) + h_rT_m + h_cT_{air},

    with :math:`h_r = 4\varepsilon\sigma T_m^3` and
    :math:`h_{eff} = h_r + h_c`.  Bottom face:
    :math:`k\,\partial_z T|_0 = h_{in}(T_{in} - T(0))`.  Lateral ends are
    adiabatic, hence the cosine basis.  Every mode obeys
    :math:`\theta_n'' = \lambda_n^2\theta_n` with :math:`\lambda_n = n\pi/W`;
    the two Robin conditions give, for each mode, a 2x2 linear system for
    ``(a_n, b_n)``.
    """
    L, k = material.thickness, material.k
    F_sky, F_sea = deck_view_factors(panel.tilt_deg)
    T_eff = (F_sky * env.sky_temperature() ** 4 + F_sea * env.T_sea**4) ** 0.25
    Tm = float(T_linearisation)
    if not (math.isfinite(Tm) and Tm > 0.0):
        raise ValueError(f"T_linearisation must be finite and > 0, got {Tm!r}")
    h_r = 4.0 * panel.emissivity * STEFAN_BOLTZMANN * Tm**3
    h_eff = h_r + panel.h_conv
    const = (
        panel.emissivity * STEFAN_BOLTZMANN * (T_eff**4 - Tm**4)
        + h_r * Tm
        + panel.h_conv * env.T_air
    )

    coef_solar = cosine_coefficients(solar_edges, solar_values, width, n_modes)
    A = coef_solar.copy()
    A[0] += const
    a = np.zeros(n_modes + 1)
    b = np.zeros(n_modes + 1)
    # Sign convention: z is measured upward from the underside, so the
    # conductive flux in +z is -k dT/dz.  Heat *entering* the slab at the
    # bottom (interior convection) therefore obeys  k theta'(0) = h_in (theta(0)
    # - T_in), and the surface load entering at the top obeys
    # k theta'(L) = A - h_eff theta(L).  Both give Robin conditions with the
    # *plus* sign on k theta':
    #   mode 0 (theta = a + b z):  -h_in a + k b = -h_in T_in
    #                              h_eff a + (h_eff L + k) b = A_0
    M0 = np.array([[-h_interior, k], [h_eff, h_eff * L + k]], dtype=np.float64)
    rhs0 = np.array([-h_interior * T_interior, A[0]], dtype=np.float64)
    if abs(np.linalg.det(M0)) < 1e-300:
        raise ValueError("degenerate 2-D problem: h_interior = h_eff = 0 (no anchor)")
    a[0], b[0] = np.linalg.solve(M0, rhs0)
    for n in range(1, n_modes + 1):
        lam = n * math.pi / width
        ch, sh = math.cosh(lam * L), math.sinh(lam * L)
        # (a_n, b_n):  -h_in a + k lam b = 0
        #   (h_eff ch + k lam sh) a + (h_eff sh + k lam ch) b = A_n
        M = np.array(
            [[-h_interior, k * lam], [h_eff * ch + k * lam * sh, h_eff * sh + k * lam * ch]],
            dtype=np.float64,
        )
        a[n], b[n] = np.linalg.solve(M, np.array([0.0, A[n]], dtype=np.float64))

    def evaluate(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        z = np.atleast_1d(np.asarray(z, dtype=np.float64))
        X, Z = np.meshgrid(x, z, indexing="xy")
        T = a[0] + b[0] * Z
        for n in range(1, n_modes + 1):
            lam = n * math.pi / width
            T = T + (a[n] * np.cosh(lam * Z) + b[n] * np.sinh(lam * Z)) * np.cos(lam * X)
        return T

    return {
        "evaluate": evaluate,
        "a": a,
        "b": b,
        "A_coefficients": A,
        "h_eff": float(h_eff),
        "h_r": float(h_r),
        "T_eff": float(T_eff),
        "T_linearisation": Tm,
        "n_modes": int(n_modes),
        "const": float(const),
    }


# ---------------------------------------------------------------------------
# LBM field solver (D2Q5 thermal ADE)
# ---------------------------------------------------------------------------

_W5 = np.array([1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
_CS2 = 1.0 / 3.0


def _equilibrium(T: np.ndarray) -> np.ndarray:
    return _W5[:, None, None] * T[None, :, :]


def _stream(g: np.ndarray) -> np.ndarray:
    """Streaming with reflecting (zero-flux) walls on all four sides.

    Index convention: ``g[k]`` with ``k = 0`` rest, 1 = +x, 2 = -x, 3 = +z
    (towards the top face, row index increases), 4 = -z.  Reflection at a wall
    is the adiabatic boundary condition of the ADE.
    """
    out = np.empty_like(g)
    out[0] = g[0]
    out[1, :, 1:] = g[1, :, :-1]
    out[1, :, 0] = g[2, :, 0]
    out[2, :, :-1] = g[2, :, 1:]
    out[2, :, -1] = g[1, :, -1]
    out[3, 1:, :] = g[3, :-1, :]
    out[3, 0, :] = g[4, 0, :]
    out[4, :-1, :] = g[4, 1:, :]
    out[4, -1, :] = g[3, -1, :]
    return out


def solve_deck_field(
    panel: DeckPanel,
    env: SkySeaEnvironment,
    material: DeckMaterial,
    T_interior: float,
    h_interior: float,
    solar_profile: np.ndarray | float | None = None,
    nx: int | None = None,
    ny: int = 8,
    width: float | None = None,
    tau: float = 1.0,
    longwave: str = "analytic",
    linearise_at: float | None = None,
    max_steps: int = 400_000,
    tol: float = 1e-8,
    check_every: int = 100,
) -> DeckFieldSolution:
    r"""Steady deck temperature field from a D2Q5 thermal lattice Boltzmann (ADE).

    Parameters
    ----------
    panel, env :
        Deck panel and atmospheric boundary (see
        :mod:`tensorlbm.deck_thermal_load`).
    material :
        Slab properties; the lattice spacing is ``dx = thickness/ny``.
    T_interior, h_interior :
        Interior-space temperature [K] and underside convective coefficient
        [W/(m2 K)]; ``h_interior = 0`` gives an adiabatic underside.
    solar_profile :
        Absorbed solar flux per column [W/m2] (scalar = uniform; use
        ``solar_flux_components(...)["total"]`` for a uniform deck).
    nx, width :
        In-plane resolution and physical width; ``nx`` defaults to
        ``round(width/dx)`` (or 1 column when neither is given).
    tau :
        BGK relaxation time of the ADE; it sets the time step
        ``dt = cs^2 (tau - 1/2) dx^2 / alpha``.  Large ``tau`` means big steps
        and a fast march, but the *steady* discrete solution is **not** exactly
        tau-invariant: the naive ``w_i S`` source term leaves an O(tau)
        consistency error (measured field error vs the analytic profile:
        7.4e-05 at tau=1, 2.4e-03 at tau=5, 2.9e-02 at tau=20 — see the
        P93W3-N02 benchmark).  Hence the default ``tau = 1``; raise it only for
        transient estimates, not for acceptance numbers.
    longwave :
        ``"analytic"`` (closed-form black surroundings).  The gray-enclosure
        long-wave form is temperature-implicit and therefore only offered by
        :func:`surface_net_flux` / :func:`analytic_steady_field_1d`; passing it
        here raises ``NotImplementedError`` instead of being silently ignored.
    linearise_at :
        If given, the long-wave term is frozen as
        :math:`\varepsilon\sigma(T_{eff}^4-T_m^4) + 4\varepsilon\sigma T_m^3(T_m-T)`
        with :math:`T_m` = this value, so the run is the discrete counterpart of
        :func:`analytic_steady_2d_fourier`.
    max_steps, tol, check_every :
        Time-marching controls.  Convergence is declared when the enclosure
        energy balance closes,
        ``|sum_i q_top,i + sum_i q_bottom,i| / sum_i |q_top,i| < tol``, checked
        every ``check_every`` steps.  A per-step temperature change is *not*
        used: it measures ``dT/dt`` and stays small while the field still
        drifts (see the report's trap list).

    Returns
    -------
    DeckFieldSolution
        Field, fluxes and audits (see the dataclass).
    """
    L = material.thickness
    ny = int(ny)
    if ny < 2:
        raise ValueError(f"ny must be >= 2, got {ny!r}")
    dx = L / ny
    if nx is None:
        nx = max(1, int(round((width if width is not None else dx) / dx)))
    nx = int(nx)
    if nx < 1:
        raise ValueError(f"nx must be >= 1, got {nx!r}")
    if not (tau > 0.5):
        raise ValueError(f"tau must be > 0.5, got {tau!r}")

    if longwave != "analytic":
        raise NotImplementedError(
            f"solve_deck_field only carries the closed-form long-wave model, got {longwave!r}; "
            "use surface_net_flux(..., longwave='enclosure') or analytic_steady_field_1d for the "
            "gray-enclosure form"
        )
    alpha = material.alpha
    dt = _CS2 * (tau - 0.5) * dx * dx / alpha

    if solar_profile is None:
        solar_profile = absorbed_solar_flux(panel, env.sun)
    if np.isscalar(solar_profile):
        q_solar = np.full(nx, float(solar_profile))
    else:
        q_solar = np.array(solar_profile, dtype=np.float64)
        if q_solar.shape != (nx,):
            raise ValueError(f"solar_profile shape {q_solar.shape} does not match ({nx},)")

    F_sky, F_sea = deck_view_factors(panel.tilt_deg)
    T_eff = (F_sky * env.sky_temperature() ** 4 + F_sea * env.T_sea**4) ** 0.25
    eps_sigma = panel.emissivity * STEFAN_BOLTZMANN
    h_c = panel.h_conv
    linearised = linearise_at is not None
    if linearised:
        Tm = float(linearise_at)
        h_r = 4.0 * eps_sigma * Tm**3
        const_top = eps_sigma * (T_eff**4 - Tm**4) + h_r * Tm + h_c * env.T_air
        h_eff = h_r + h_c
    else:
        const_top = eps_sigma * T_eff**4 + h_c * env.T_air
        h_eff = math.nan

    def top_flux(T_top: np.ndarray) -> np.ndarray:
        """Net flux into the top face [W/m2] at the given surface temperature."""
        if linearised:
            return q_solar + const_top - h_eff * T_top
        return (
            q_solar
            + const_top
            - h_c * T_top
            - eps_sigma * T_top**4
        )

    c_src = dt / (material.rho * material.cp * dx)
    w3 = _W5[:, None, None]
    aq = 1.0 - 1.0 / tau
    bq = 1.0 / tau

    T = np.full((ny, nx), float(T_interior), dtype=np.float64)
    g = _equilibrium(T)
    out = np.empty_like(g)
    S = np.zeros((ny, nx), dtype=np.float64)

    converged = False
    steps = 0
    change = math.inf
    check_every = max(1, int(check_every))
    for steps in range(1, max_steps + 1):
        T = g.sum(axis=0)
        # sources: volume form of the two surface fluxes (S = q/dx)
        S[-1] = top_flux(T[-1])
        S[0] = h_interior * (T_interior - T[0])
        S *= c_src
        # BGK collision, in place
        g *= aq
        g += (bq * w3) * T[None, :, :]
        g += w3 * S[None, :, :]
        # streaming with reflecting (adiabatic) walls
        out[0] = g[0]
        out[1, :, 1:] = g[1, :, :-1]
        out[1, :, 0] = g[2, :, 0]
        out[2, :, :-1] = g[2, :, 1:]
        out[2, :, -1] = g[1, :, -1]
        out[3, 1:, :] = g[3, :-1, :]
        out[3, 0, :] = g[4, 0, :]
        out[4, :-1, :] = g[4, 1:, :]
        out[4, -1, :] = g[3, -1, :]
        g, out = out, g
        if steps % check_every == 0:
            # Steady-state certificate: the enclosure energy balance must close.
            # (A per-step temperature change is NOT a valid criterion: it only
            # measures dT/dt, which stays small while the field still drifts.)
            T_new = g.sum(axis=0)
            change = float(np.max(np.abs(T_new - T))) / float(np.max(np.abs(T_new)))
            q_t = top_flux(T_new[-1])
            q_b = h_interior * (T_interior - T_new[0])
            scale_flux = float(np.sum(np.abs(q_t)))
            imbalance = abs(float(np.sum(q_t)) + float(np.sum(q_b)))
            imbalance = imbalance / (scale_flux if scale_flux > 0 else 1.0)
            if imbalance < tol:
                converged = True
                break

    T = g.sum(axis=0)
    q_top = top_flux(T[-1])
    q_bot = h_interior * (T_interior - T[0])
    denom = float(np.sum(np.abs(q_top)))
    energy_residual = abs(float(np.sum(q_top)) + float(np.sum(q_bot))) / (denom if denom > 0 else 1.0)
    return DeckFieldSolution(
        T=T,
        x=(np.arange(nx) + 0.5) * dx,
        z=(np.arange(ny) + 0.5) * dx,
        q_top=q_top,
        q_bottom=q_bot,
        steps=int(steps),
        dt=float(dt),
        dx=float(dx),
        tau=float(tau),
        converged=bool(converged),
        max_step_change=float(change),
        energy_residual=float(energy_residual),
        solar=solar_flux_components(panel, env.sun),
        linearised=bool(linearised),
    )
