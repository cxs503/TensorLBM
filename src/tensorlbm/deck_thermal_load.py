"""Solar / sky long-wave boundary condition -> ship deck thermal load.

Purpose
-------
Convert an atmospheric radiant environment (direct sun, sky dome, sea
surface) into a **deck wall heat flux** that can be injected into a thermal
lattice-Boltzmann / conduction boundary, and anchor the result on analytic
solutions.

Model
-----
A deck panel of area :math:`A`, tilt :math:`\\beta` from horizontal
(0 = up-facing, 90 deg = vertical) and azimuth :math:`\\psi`, emissivity
:math:`\\varepsilon`, solar absorptance :math:`\\alpha_s`, receiving

* direct sun :math:`G_b` at incidence :math:`\\theta`,
* isotropic sky diffuse :math:`G_d`,
* sea-reflected beam (albedo :math:`\\rho_{sea}`),
* long-wave exchange with the sky dome at :math:`T_{sky}` and the sea surface
  at :math:`T_{sea}`,

balances at steady state

.. math::
    q_{net}(T) = \\alpha_s G_{abs} + q_{lw}(T) + h (T_{air} - T) = 0

where :math:`q_{lw}` is obtained either from

* the **analytic** lumped form (black surroundings)
  :math:`q_{lw} = \\varepsilon\\sigma(T_{eff}^4 - T^4)` with
  :math:`T_{eff}^4 = F_{sky}T_{sky}^4 + F_{sea}T_{sea}^4`, or
* the **gray enclosure** form: deck + sky patch + sea patch solved with
  :class:`tensorlbm.gray_enclosure_radiosity.GrayEnclosure` (captures the
  multiple reflections that the black-surroundings form neglects),

with the sky view factor from the closed-form family
:math:`F_{sky} = (1+\\cos\\beta)/2` and :math:`F_{sea} = 1 - F_{sky}`
(summation rule, exact).

Reference solution
------------------
:func:`reference_deck_temperature` solves the scalar energy balance with a
*different* algorithm (300 bisection steps on the black-surroundings form,
bracket-independent) than :func:`solve_deck_temperature` (Brent on the
selected flux model), so the two agree only if the load path is correct.
For ``h = 0`` the balance degrades to the exact closed form
:math:`T = (T_{eff}^4 + \\alpha_s G_{abs}/(\\varepsilon\\sigma))^{1/4}`,
which :func:`closed_form_equilibrium_temperature` returns for a third,
fully independent check.

References
----------
Berdahl, P., Martin, M. (1984). Emissivity of clear skies. *Solar Energy* 32,
    663-664.
ASHRAE Handbook of Fundamentals (2009), ch. 14 (clear-sky model,
    ``A exp(-B/sin beta)``) and ch. 15 (Cooper declination correlation).
Incropera et al. (2007), ch. 12-13 (long-wave exchange with the sky dome).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .gray_enclosure_radiosity import (
    STEFAN_BOLTZMANN,
    GrayEnclosure,
    GraySurface,
)
from .radiative_view_factors import vf_sky_dome_to_ground, vf_sky_dome_to_tilted_plane

__all__ = [
    "DeckPanel",
    "SunState",
    "SkySeaEnvironment",
    "deck_view_factors",
    "cosine_incidence",
    "absorbed_solar_flux",
    "effective_surroundings_temperature",
    "deck_longwave_analytic",
    "deck_longwave_enclosure",
    "deck_energy_balance",
    "solve_deck_temperature",
    "reference_deck_temperature",
    "closed_form_equilibrium_temperature",
    "deck_wall_heat_flux",
    "inject_deck_boundary",
    "solar_position",
    "clear_sky_irradiance",
    "deck_longwave_raytraced",
    "sky_view_factor_shaded_deck",
]


# ---------------------------------------------------------------------------
# data containers
# ---------------------------------------------------------------------------


@dataclass
class DeckPanel:
    """A radiating deck panel."""

    area: float = 24.0
    tilt_deg: float = 0.0
    azimuth_deg: float = 0.0  # 0 = facing north, 90 = east
    emissivity: float = 0.9
    solar_absorptance: float = 0.6
    h_conv: float = 0.0  # W/(m2 K); 0 => pure radiative equilibrium

    def __post_init__(self) -> None:
        if not math.isfinite(self.area) or self.area <= 0.0:
            raise ValueError(f"area must be finite and > 0, got {self.area!r}")
        if not (0.0 < self.emissivity <= 1.0):
            raise ValueError(f"emissivity must be in (0, 1], got {self.emissivity!r}")
        if not (0.0 <= self.solar_absorptance <= 1.0):
            raise ValueError(
                f"solar_absorptance must be in [0, 1], got {self.solar_absorptance!r}"
            )
        if not math.isfinite(self.h_conv) or self.h_conv < 0.0:
            raise ValueError(f"h_conv must be finite and >= 0, got {self.h_conv!r}")


@dataclass
class SunState:
    """Direct-beam state at the deck."""

    beam_irradiance: float = 0.0  # G_b, normal to beam [W/m2]
    elevation_deg: float = 0.0  # above horizon
    azimuth_deg: float = 0.0  # from north, clockwise
    diffuse_horizontal: float = 0.0  # G_d on a horizontal surface [W/m2]
    sea_albedo: float = 0.06

    def is_day(self) -> bool:
        return self.beam_irradiance > 0.0 and self.elevation_deg > 0.0

    def unit_vector(self) -> np.ndarray:
        """Unit vector pointing from the deck towards the sun (z = up)."""
        e = math.radians(self.elevation_deg)
        a = math.radians(self.azimuth_deg)
        return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


@dataclass
class SkySeaEnvironment:
    """Atmosphere + sea boundary condition."""

    T_air: float = 300.0
    T_dewpoint: float = 285.0
    T_sea: float = 293.0
    sky_emissivity: float = 1.0  # 1.0 = black sky (analytic limit)
    sea_emissivity: float = 0.96
    sun: SunState = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        for label in ("T_air", "T_dewpoint", "T_sea"):
            value = getattr(self, label)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be finite and > 0, got {value!r}")
        if self.sun is None:
            self.sun = SunState()

    def sky_temperature(self) -> float:
        """Berdahl-Martin effective sky temperature [K]."""
        tdp = (self.T_dewpoint - 273.15) / 100.0
        eps_sky = max(0.0, min(1.0, 0.711 + 0.56 * tdp + 0.73 * tdp * tdp))
        return self.T_air * eps_sky**0.25


# ---------------------------------------------------------------------------
# geometry / solar
# ---------------------------------------------------------------------------


def deck_view_factors(tilt_deg: float) -> tuple[float, float]:
    """(F_deck->sky, F_deck->sea) from the closed-form sky-dome family."""
    F_sky = vf_sky_dome_to_tilted_plane(tilt_deg)
    F_sea = vf_sky_dome_to_ground(tilt_deg)
    return F_sky, F_sea


def cosine_incidence(panel: DeckPanel, sun: SunState) -> float:
    """Cosine of the sun incidence angle on the deck (0 when shadowed)."""
    beta = math.radians(panel.tilt_deg)
    psi = math.radians(panel.azimuth_deg)
    n = np.array([math.sin(beta) * math.cos(psi), math.sin(beta) * math.sin(psi), math.cos(beta)])
    return float(max(0.0, float(n @ sun.unit_vector())))


def absorbed_solar_flux(panel: DeckPanel, sun: SunState) -> float:
    """Solar flux absorbed per unit deck area [W/m2].

    ``alpha_s * (G_b cos(theta) + G_d F_sky + G_b cos(theta_h) rho_sea F_sea)``
    where ``theta_h = 90 deg - elevation`` is the incidence on a horizontal
    surface for the sea-reflected component.
    """
    F_sky, F_sea = deck_view_factors(panel.tilt_deg)
    cos_inc = cosine_incidence(panel, sun)
    direct = sun.beam_irradiance * cos_inc
    diffuse = sun.diffuse_horizontal * F_sky
    cos_h = math.sin(math.radians(max(0.0, sun.elevation_deg)))
    reflected = sun.beam_irradiance * cos_h * sun.sea_albedo * F_sea
    return float(panel.solar_absorptance * (direct + diffuse + reflected))


def solar_position(
    latitude_deg: float, day_of_year: int, hour_solar: float
) -> tuple[float, float]:
    """Sun elevation and azimuth [deg] (Cooper declination + hour angle).

    Parameters
    ----------
    latitude_deg : float
        Site latitude, positive north.
    day_of_year : int
        Day of year (1-365).
    hour_solar : float
        Apparent solar time [h], 12 = solar noon.
    """
    delta = math.radians(23.45) * math.sin(2.0 * math.pi * (284 + day_of_year) / 365.0)
    phi = math.radians(latitude_deg)
    H = math.radians(15.0 * (hour_solar - 12.0))
    sin_elev = math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.cos(H)
    sin_elev = max(-1.0, min(1.0, sin_elev))
    elev = math.asin(sin_elev)
    cos_az = (
        math.sin(delta) * math.cos(phi) - math.cos(delta) * math.sin(phi) * math.cos(H)
    ) / max(math.cos(elev), 1e-12)
    cos_az = max(-1.0, min(1.0, cos_az))
    az = math.acos(cos_az)  # measured from north
    if H > 0.0:  # afternoon: sun in the west
        az = 2.0 * math.pi - az
    return math.degrees(elev), math.degrees(az)


def clear_sky_irradiance(
    elevation_deg: float, A: float = 1085.0, B: float = 0.205, C: float = 0.134
) -> tuple[float, float]:
    """ASHRAE clear-sky beam and diffuse irradiance [W/m2].

    ``G_b = A exp(-B / sin(beta))``, ``G_d = C G_b`` (summer, 21 July
    coefficients).  Returns ``(0, 0)`` below the horizon.
    """
    sin_b = math.sin(math.radians(max(0.0, elevation_deg)))
    if sin_b <= 1e-9:
        return 0.0, 0.0
    G_b = A * math.exp(-B / sin_b)
    return float(G_b), float(C * G_b)


# ---------------------------------------------------------------------------
# long-wave exchange
# ---------------------------------------------------------------------------


def effective_surroundings_temperature(F_sky: float, T_sky: float, F_sea: float, T_sea: float) -> float:
    """View-factor-weighted surroundings temperature [K] (black surroundings)."""
    return float((F_sky * T_sky**4 + F_sea * T_sea**4) ** 0.25)


def deck_longwave_analytic(panel: DeckPanel, env: SkySeaEnvironment, T_deck: float) -> float:
    """Net long-wave flux to the deck, black-surroundings form [W/m2]."""
    F_sky, F_sea = deck_view_factors(panel.tilt_deg)
    T_eff = effective_surroundings_temperature(F_sky, env.sky_temperature(), F_sea, env.T_sea)
    return float(panel.emissivity * STEFAN_BOLTZMANN * (T_eff**4 - T_deck**4))


def deck_longwave_enclosure(
    panel: DeckPanel,
    env: SkySeaEnvironment,
    T_deck: float,
    dome_area_ratio: float = 1000.0,
) -> tuple[float, dict]:
    """Net long-wave flux to the deck from a gray deck/sky/sea enclosure.

    Sky and sea are represented as patches of area
    ``dome_area_ratio * A_deck`` so that their view factors back to the deck
    are the reciprocity-consistent ``A_deck F / A_patch``; the deck's own row
    sums to one identically (``F_sky + F_sea = 1``).  The returned flux is
    positive *into* the deck, matching :func:`deck_longwave_analytic`.
    """
    F_sky, F_sea = deck_view_factors(panel.tilt_deg)
    A = panel.area
    A_p = dome_area_ratio * A
    F = np.array(
        [
            [0.0, F_sky, F_sea],
            [A * F_sky / A_p, 1.0 - A * F_sky / A_p, 0.0],
            [A * F_sea / A_p, 0.0, 1.0 - A * F_sea / A_p],
        ]
    )
    surfaces = [
        GraySurface(area=A, emissivity=panel.emissivity, temperature=T_deck, name="deck"),
        GraySurface(
            area=A_p, emissivity=env.sky_emissivity, temperature=env.sky_temperature(), name="sky"
        ),
        GraySurface(area=A_p, emissivity=env.sea_emissivity, temperature=env.T_sea, name="sea"),
    ]
    sol = GrayEnclosure(surfaces, F).solve()
    # enclosure convention: q = J - G is positive when *leaving* the surface;
    # the deck model reports positive = into the deck, so flip the sign.
    return float(-sol.net_flux[0]), sol.as_dict()


# ---------------------------------------------------------------------------
# energy balance
# ---------------------------------------------------------------------------


def deck_energy_balance(
    T_deck: float,
    panel: DeckPanel,
    env: SkySeaEnvironment,
    longwave: str = "analytic",
) -> dict:
    """Surface energy balance for the deck at a given temperature.

    Returns a dict with ``q_solar``, ``q_longwave`` (positive = into the deck),
    ``q_conv``, ``q_net`` (sum; zero at equilibrium) and ``q_required`` (the
    cooling/heating duty needed to hold ``T_deck``, positive = must be
    removed).
    """
    if not math.isfinite(T_deck) or T_deck <= 0.0:
        raise ValueError(f"T_deck must be finite and > 0, got {T_deck!r}")
    q_solar = absorbed_solar_flux(panel, env.sun)
    if longwave == "analytic":
        q_lw = deck_longwave_analytic(panel, env, T_deck)
    elif longwave == "enclosure":
        q_lw, _ = deck_longwave_enclosure(panel, env, T_deck)
    else:
        raise ValueError(f"longwave must be 'analytic' or 'enclosure', got {longwave!r}")
    q_conv = panel.h_conv * (env.T_air - T_deck)
    q_net = q_solar + q_lw + q_conv
    return {
        "T_deck": float(T_deck),
        "q_solar": float(q_solar),
        "q_longwave": float(q_lw),
        "q_conv": float(q_conv),
        "q_net": float(q_net),
        "q_required": float(-q_net),
    }


def solve_deck_temperature(
    panel: DeckPanel,
    env: SkySeaEnvironment,
    longwave: str = "analytic",
    bracket: tuple[float, float] = (150.0, 500.0),
    tol: float = 1e-10,
) -> dict:
    """Brent solve of ``q_net(T_deck) = 0`` for the deck equilibrium.

    Returns the balance dict plus ``iterations``-free diagnostics:
    ``residual`` (W/m2) and the constituent fluxes.
    """
    from scipy.optimize import brentq

    def f(T: float) -> float:
        return deck_energy_balance(T, panel, env, longwave=longwave)["q_net"]

    lo, hi = bracket
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"no sign change of q_net on bracket {bracket}: q({lo})={f_lo:.3f}, q({hi})={f_hi:.3f}"
        )
    T_eq = brentq(f, lo, hi, xtol=tol, rtol=4.0 * np.finfo(float).eps, maxiter=200)
    out = deck_energy_balance(T_eq, panel, env, longwave=longwave)
    out["residual"] = out["q_net"]
    out["bracket"] = [lo, hi]
    return out


def reference_deck_temperature(
    panel: DeckPanel,
    env: SkySeaEnvironment,
    n_bisect: int = 300,
    bracket: tuple[float, float] = (50.0, 1000.0),
) -> dict:
    """Independent reference solution (bisection, black-surroundings form).

    Deliberately different from :func:`solve_deck_temperature`: a fixed-count
    bisection on the closed-form lumped balance, with a wide bracket that does
    not depend on the caller's guess.  300 bisections over [50, 1000] K leave
    a bracket width below 1e-87 K, i.e. the returned root is exact to double
    precision.
    """
    F_sky, F_sea = deck_view_factors(panel.tilt_deg)
    T_sky = env.sky_temperature()
    T_eff4 = F_sky * T_sky**4 + F_sea * env.T_sea**4
    q_solar = absorbed_solar_flux(panel, env.sun)
    eps_sigma = panel.emissivity * STEFAN_BOLTZMANN
    h = panel.h_conv

    def f(T: float) -> float:
        return q_solar + eps_sigma * (T_eff4 - T**4) + h * (env.T_air - T)

    lo, hi = bracket
    if f(lo) * f(hi) > 0.0:
        raise ValueError("reference solver: bracket does not contain a root")
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    T_eq = 0.5 * (lo + hi)
    return {
        "T_deck": float(T_eq),
        "q_solar": float(q_solar),
        "q_longwave": float(eps_sigma * (T_eff4 - T_eq**4)),
        "q_conv": float(h * (env.T_air - T_eq)),
        "q_net": float(f(T_eq)),
        "T_eff": float(T_eff4**0.25),
        "q_required": float(-f(T_eq)),
    }


def closed_form_equilibrium_temperature(
    panel: DeckPanel, env: SkySeaEnvironment
) -> float:
    """Exact closed form for ``h = 0``: ``T = (T_eff^4 + alpha G/(eps sigma))^{1/4}``."""
    if panel.h_conv != 0.0:
        raise ValueError("closed form requires h_conv = 0")
    F_sky, F_sea = deck_view_factors(panel.tilt_deg)
    T_eff4 = F_sky * env.sky_temperature() ** 4 + F_sea * env.T_sea**4
    q_solar = absorbed_solar_flux(panel, env.sun)
    return float((T_eff4 + q_solar / (panel.emissivity * STEFAN_BOLTZMANN)) ** 0.25)


# ---------------------------------------------------------------------------
# LBM / conduction boundary injection
# ---------------------------------------------------------------------------


def deck_wall_heat_flux(panel: DeckPanel, env: SkySeaEnvironment, T_deck: float) -> float:
    """Net radiant + convective flux entering the deck surface [W/m2].

    Positive = energy the deck must absorb (it is colder than its radiant
    environment); this is the quantity to inject as a Neumann wall condition.
    """
    return float(deck_energy_balance(T_deck, panel, env)["q_net"])


def inject_deck_boundary(
    T: np.ndarray,
    mask: np.ndarray,
    q_flux_W_m2: float,
    dt_phys: float,
    rho_phys: float,
    cp_phys: float,
    dx_phys: float,
    dA_cell: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Apply a deck heat flux as a temperature increment on boundary cells.

    Each boundary cell of volume ``dA_cell * dx_phys`` receives
    ``q_flux_W_m2 * dA_cell``; the update is
    ``dT = q dA dt / (rho cp dA dx) = q dt / (rho cp dx)``.

    Returns the updated temperature array and an audit dict comparing the
    injected enthalpy with ``q A dt`` (``A = n_cells * dA_cell``).
    """
    T = np.asarray(T, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if T.shape != mask.shape:
        raise ValueError(f"T shape {T.shape} != mask shape {mask.shape}")
    n = int(np.count_nonzero(mask))
    if n == 0:
        raise ValueError("mask selects no boundary cells")
    dA = 1.0 if dA_cell is None else float(dA_cell)
    dT = q_flux_W_m2 * dt_phys / (rho_phys * cp_phys * dx_phys)
    T_new = T.copy()
    T_new[mask] += dT
    injected = float(np.sum(rho_phys * cp_phys * (T_new[mask] - T[mask]) * dA * dx_phys))
    expected = q_flux_W_m2 * n * dA * dt_phys
    denom = abs(expected) if abs(expected) > 1e-300 else 1.0
    return T_new, {
        "n_cells": n,
        "dT": float(dT),
        "injected_energy_J": injected,
        "expected_energy_J": float(expected),
        "relative_error": float(abs(injected - expected) / denom),
        "total_area_m2": float(n * dA),
    }

def deck_longwave_raytraced(
    panel: DeckPanel,
    env: SkySeaEnvironment,
    T_deck: float,
    n_rays: int = 1_000_000,
    seed: int = 0,
    wall_height: float = 0.0,
    wall_distance: float = 1.0,
) -> dict:
    r"""Monte-Carlo estimator of the deck long-wave exchange [W/m2].

    Cosine-weighted rays are launched from the deck normal and binned into sky
    (direction above the horizon) or sea (below).  The incoming irradiance is
    :math:`\varepsilon\sigma\langle T_{env}^4\rangle`, the outgoing emission
    :math:`\varepsilon\sigma T_{deck}^4`.  This is a *sampling* estimator of
    the same integral that :func:`deck_longwave_analytic` evaluates in closed
    form, so agreement between the two is an independent implementation check.

    An optional vertical wall of height ``wall_height`` at distance
    ``wall_distance`` along the panel's azimuth blocks directions with
    ``d_z/d_x < h/wall_distance`` (infinite wall, no side edges): the classic
    ship superstructure shading of a deck.
    """
    if n_rays <= 0:
        raise ValueError(f"n_rays must be > 0, got {n_rays}")
    rng = np.random.default_rng(seed)
    beta = math.radians(panel.tilt_deg)
    psi = math.radians(panel.azimuth_deg)
    n = np.array([math.sin(beta) * math.cos(psi), math.sin(beta) * math.sin(psi), math.cos(beta)])
    # local basis (u, v, n)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    r_a = rng.random(n_rays)
    r_b = rng.random(n_rays)
    cos_t = np.sqrt(1.0 - r_a)
    sin_t = np.sqrt(r_a)
    phi = 2.0 * math.pi * r_b
    d = (
        (sin_t * np.cos(phi))[:, None] * u[None, :]
        + (sin_t * np.sin(phi))[:, None] * v[None, :]
        + cos_t[:, None] * n[None, :]
    )
    to_sky = d[:, 2] > 0.0
    if wall_height > 0.0:
        blocked = (d @ u > 0.0) & (d[:, 2] / np.maximum(np.abs(d @ u), 1e-300) < wall_height / wall_distance)
        to_sky = to_sky & ~blocked
    T_sky = env.sky_temperature()
    T_env = np.where(to_sky, T_sky, env.T_sea)
    q_in = panel.emissivity * STEFAN_BOLTZMANN * float(np.mean(T_env**4))
    q_out = panel.emissivity * STEFAN_BOLTZMANN * T_deck**4
    F_sky = float(np.count_nonzero(to_sky)) / n_rays
    stderr = math.sqrt(max(F_sky * (1.0 - F_sky), 0.0) / n_rays)
    return {
        "q_longwave": q_in - q_out,
        "q_in": q_in,
        "q_out": q_out,
        "F_sky_mc": F_sky,
        "F_sky_stderr": stderr,
        "T_eff": float(np.mean(T_env**4)) ** 0.25,
    }


def sky_view_factor_shaded_deck(
    wall_height: float,
    wall_distance: float,
    n_theta: int = 400,
    n_phi: int = 800,
) -> float:
    r"""Sky view factor of a horizontal deck shaded by an infinite vertical wall.

    Quadrature form of the integral evaluated by
    :func:`deck_longwave_raytraced` for ``tilt = 0``.  Limits:
    ``F -> 1`` as ``wall_height -> 0`` and ``F -> 1/2`` as
    ``wall_height/wall_distance -> inf`` (a full half-hemisphere is blocked).

    Used to certify the MC wall-shading branch of the deck model.
    """
    if wall_height < 0.0 or wall_distance <= 0.0:
        raise ValueError("require wall_height >= 0 and wall_distance > 0")
    # Gauss-Legendre in theta on [0, pi/2], phi on [0, 2*pi]
    xt, wt = np.polynomial.legendre.leggauss(n_theta)
    xp, wp = np.polynomial.legendre.leggauss(n_phi)
    theta = 0.25 * math.pi * (xt + 1.0)
    wt = 0.5 * math.pi * 0.5 * wt
    phi = math.pi * (xp + 1.0)
    wp = math.pi * wp  # [0,1] -> [0, 2*pi] => 2*pi*0.5*w
    st = np.sin(theta)[:, None]
    ct = np.cos(theta)[:, None]
    dx = st * np.cos(phi)[None, :]
    dz = np.broadcast_to(ct, dx.shape)
    visible = dz > 0.0
    if wall_height > 0.0:
        slope = np.where(dx > 1e-300, dz / np.maximum(np.abs(dx), 1e-300), np.inf)
        visible = visible & ~((dx > 0.0) & (slope < wall_height / wall_distance))
    integrand = np.where(visible, dz * st, 0.0)
    w2 = wt[:, None] * wp[None, :]
    return float(np.sum(integrand * w2) / math.pi)
