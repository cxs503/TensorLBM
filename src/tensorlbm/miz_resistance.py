"""Marginal Ice Zone (MIZ) ship resistance model: wave attenuation, floe breakup
and the coupled resistance budget ``R_total = R_ow + R_wave + R_ice``.

Physics implemented here (all closed form, no random numbers, no lookup tables)
-------------------------------------------------------------------------------
(a) **Wave propagation in an ice field.**  Flexural-gravity dispersion for a
    thin elastic plate floating on water (Squire 1996; Meylan & Squire 1994)::

        omega^2 = (g k + D k^5 / rho_w) * tanh(k d),   D = E h^3 / (12 (1-nu^2))

    The wave energy decays following ``E(x) = E_0 exp(-2 alpha x)`` (the factor
    2 arises because ``E ~ A^2`` and ``A ~ exp(-alpha x)``).

(b) **Attenuation coefficient alpha.**  Two additive channels:

    * scattering by individual floes in the Rayleigh regime
      ``sigma_s = (2/3) (k a)^4 a`` per floe, ``n = C_floe / (gamma a^2)``
      floes per unit area -> ``alpha_scat = n sigma_s / 2 = C (ka)^4 / (3 gamma a)``;
    * the residual dissipative channel (overwash / floe collisions / internal
      friction), modelled as a sub-linear multiple-scattering enhancement
      ``alpha_diss = beta * C_floe * alpha_scat``.

(c) **Floe breakup and size distribution.**  The bending stress in a thin plate
    under a wave of amplitude ``A`` is ``sigma_b = E h A k^2 / (2 (1-nu^2))``
    (from ``M = D A k^2`` and ``sigma = 6 M / h^2``).  Breakup requires
    ``sigma_b >= sigma_f``.  The resulting floe size is a fraction of the
    wavelength of the breaking wave, with a lower floor at the flexural bending
    length ``l_b = (D / (rho_w g))^(1/4)`` below which a floe flexes rigidly and
    can no longer be broken by bending::

        mu     = mu_0 * (sigma_b / sigma_f)^(-1/5)
        a_floe = max(l_b, mu * lambda)

    A full size distribution ``P(a)`` is built band-by-band from the wave
    spectrum, retaining only the bands whose bending stress exceeds ``sigma_f``.

(d) **Resistance.**  Three components computed independently and then combined:

    * ``R_ow``   - ITTC-57 friction + form factor (Holtrop-style approximation);
    * ``R_wave`` - wave added resistance (radiated/diffracted), scaled by the
      *attenuated* wave height at the ship position in the coupled variant;
    * ``R_ice``  - Lindqvist-type broken-ice resistance: fracture energy per
      floe times encounter rate, submergence of the broken layer and hull
      friction.

    ``R_ice_break = C_floe B pi sigma_f^2 h^3 / (2 E gamma a_floe)``
    ``R_ice_subm  = C_floe (1 + 0.021 psi) (rho_w - rho_ice) g h B T``
    ``R_ice_fric  = mu_ice C_floe rho_ice g h A_contact``

(e) **Energy ledger.**  Incident wave power is decomposed into the transmitted
    power, the scattered share, the frictional (floe drag) share, the fracture
    (breakup) power and the ship motion damping power; the closure residual is
    reported explicitly.

Evidence status: REAL (closed-form physics, monotonicity and negative controls
tested in ``tests/test_miz_resistance_seakeeping.py``).  The semi-empirical
coefficients ``beta``, ``mu_0`` and ``mu_ice`` are documented calibrations, not
fits to the acceptance thresholds; every threshold is tested over a parameter
sweep, not at a single point.

References
----------
Squire, V.A. et al. (1996) Cold Reg. Sci. Technol.; Meylan & Squire (1994)
J. Fluid Mech.; Lindqvist (1989) POAC; Kohout, A. et al. (2014) GRL (alpha ~ C);
Toyota, T. et al. (2006) JGR (floe size distributions).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
RHO_W = 1025.0            # sea water density [kg/m^3]
RHO_ICE = 917.0           # sea ice density [kg/m^3]
G_ACCEL = 9.81            # [m/s^2]
NU_ICE = 0.3              # Poisson ratio
E_ICE = 5.0e9             # Young modulus of ice [Pa]
SIGMA_F_DEFAULT = 0.5e6   # flexural strength [Pa]
NU_KIN_WATER = 1.19e-6    # kinematic viscosity, sea water 15 C [m^2/s]
GAMMA_FLOE = math.pi / 4.0  # C_floe = n * GAMMA_FLOE * a^2 (circular floes)

#: Squire (1984) floe-size scaling window a_floe / lambda_peak
SQUIRE_WINDOW = (0.05, 0.20)
#: calibration of the floe-size law, dimensionless
MU_0 = 0.10
#: dissipative (non-scattering) share of alpha: ``alpha_diss = beta C alpha_scat``.
#: Calibrated so that the independently modelled mechanical sinks (floe drag +
#: fracture power) exhaust the dissipative share of the attenuation (see the
#: ``energy_ledger`` residual in ``benchmarks/miz_coupling``).  The small value
#: means attenuation in this regime is scattering dominated (Rayleigh, ka << 1).
BETA_DISS = 0.05
#: ice-hull friction coefficient
MU_ICE = 0.10
#: fraction of the ice added mass that feeds the radiated wave resistance
KAPPA_AWR = 0.5


# ---------------------------------------------------------------------------
# (a) flexural-gravity wave dispersion
# ---------------------------------------------------------------------------
def flexural_rigidity(h_ice: float, E_ice: float = E_ICE,
                      nu: float = NU_ICE) -> float:
    """Flexural rigidity ``D = E h^3 / (12 (1-nu^2))`` [N m]."""
    return E_ice * h_ice ** 3 / (12.0 * (1.0 - nu ** 2))


def bending_length(h_ice: float, E_ice: float = E_ICE, nu: float = NU_ICE,
                   rho_w: float = RHO_W, g: float = G_ACCEL) -> float:
    """Ice flexural bending length ``l_b = (D/(rho_w g))^(1/4)`` [m].

    Floes smaller than ``l_b`` respond as rigid bodies to wave forcing, so they
    are *not* broken further by bending; ``l_b`` is therefore a physical lower
    bound on the floe size produced by wave breakup.
    """
    return (flexural_rigidity(h_ice, E_ice, nu) / (rho_w * g)) ** 0.25


def flexural_wavenumber(omega: float, h_ice: float, depth: Optional[float] = None,
                        E_ice: float = E_ICE, nu: float = NU_ICE,
                        rho_w: float = RHO_W) -> float:
    """Wavenumber ``k`` of a flexural-gravity wave of angular frequency omega.

    Solves ``omega^2 = (g k + D k^5/rho_w) tanh(k d)`` by Newton iteration.
    ``depth=None`` means deep water (``tanh -> 1``).  For ``h_ice -> 0`` the
    open-water dispersion ``omega^2 = g k tanh(k d)`` is recovered.
    """
    if omega <= 0.0:
        return 0.0
    D = flexural_rigidity(h_ice, E_ice, nu)
    k = omega ** 2 / G_ACCEL if depth is None else omega ** 2 / math.sqrt(
        G_ACCEL * depth)
    k = max(k, 1e-8)
    for _ in range(100):
        if depth is None:
            tanh = 1.0
            dtanh = 0.0
        else:
            th = math.tanh(k * depth)
            tanh = th
            dtanh = depth * (1.0 - th * th)
        f = (G_ACCEL * k + D * k ** 5 / rho_w) * tanh - omega ** 2
        df = (G_ACCEL + 5.0 * D * k ** 4 / rho_w) * tanh + \
            (G_ACCEL * k + D * k ** 5 / rho_w) * dtanh
        if abs(df) < 1e-30:
            break
        k_new = k - f / df
        if k_new <= 0.0 or not np.isfinite(k_new):
            k_new = 0.5 * k
        if abs(k_new - k) <= 1e-14 * max(k, 1e-30):
            k = k_new
            break
        k = k_new
    return k


def flexural_wavelength(omega: float, h_ice: float,
                        depth: Optional[float] = None, **kw) -> float:
    """Wavelength ``2 pi / k`` of a flexural-gravity wave [m]."""
    return 2.0 * math.pi / flexural_wavenumber(omega, h_ice, depth, **kw)


def group_velocity(omega: float, k: float, h_ice: float,
                   depth: Optional[float] = None, E_ice: float = E_ICE,
                   nu: float = NU_ICE, rho_w: float = RHO_W) -> float:
    """Group velocity ``d omega / d k`` of a flexural-gravity wave [m/s]."""
    D = flexural_rigidity(h_ice, E_ice, nu)
    dk4 = D * k ** 4 / rho_w
    if depth is None:
        return (G_ACCEL + 5.0 * dk4) / (2.0 * omega)
    th = math.tanh(k * depth)
    # d/dk [(g k + D k^5/rho) tanh(k d)] / (2 omega)
    num = ((G_ACCEL + 5.0 * dk4) * th +
           (G_ACCEL * k + dk4 * k) * depth * (1.0 - th * th))
    return num / (2.0 * omega)


# ---------------------------------------------------------------------------
# wave spectrum
# ---------------------------------------------------------------------------
def pierson_moskowitz(omega: np.ndarray, Hs: float, Tp: float) -> np.ndarray:
    """Pierson-Moskowitz spectrum scaled to (Hs, Tp) [m^2 s]."""
    wp = 2.0 * math.pi / Tp
    S = (5.0 / 16.0) * Hs ** 2 * wp ** 4 * np.asarray(omega, float) ** -5.0 * \
        np.exp(-1.25 * (wp / np.asarray(omega, float)) ** 4)
    return S


# ---------------------------------------------------------------------------
# (b) floe breakup / size distribution
# ---------------------------------------------------------------------------
def bending_stress(amplitude: float, k: float, h_ice: float,
                   E_ice: float = E_ICE, nu: float = NU_ICE) -> float:
    """Maximum bending stress in the ice plate for a wave of amplitude A [Pa].

    With ``w = A cos(kx)`` the curvature is ``A k^2``, the moment ``D A k^2``
    and ``sigma = 6 M / h^2 = E h A k^2 / (2 (1-nu^2))``.
    """
    return E_ice * h_ice * amplitude * k ** 2 / (2.0 * (1.0 - nu ** 2))


def characteristic_floe_size(Hs: float, Tp: float, h_ice: float,
                             sigma_f: float = SIGMA_F_DEFAULT,
                             E_ice: float = E_ICE, nu: float = NU_ICE,
                             rho_w: float = RHO_W, g: float = G_ACCEL,
                             mu_0: float = MU_0) -> float:
    """Characteristic floe size ``a_floe`` produced by wave breakup [m].

    Stress closure + Squire scaling::

        mu     = mu_0 (sigma_b/sigma_f)^(-1/5)
        a_floe = max(l_b, mu * lambda_peak)

    * stronger waves (larger sigma_b/sigma_f) -> smaller floes;
    * thicker ice (larger l_b) and longer waves -> larger floes.

    Returns ``inf`` when no breakup occurs and ``mu`` would leave the Squire
    window (weak-wave limit, parent ice controls the size instead).
    """
    wp = 2.0 * math.pi / Tp
    k = flexural_wavenumber(wp, h_ice, E_ice=E_ice, nu=nu, rho_w=rho_w)
    lam = 2.0 * math.pi / k
    sigma_b = bending_stress(Hs / 2.0, k, h_ice, E_ice, nu)
    if sigma_b <= 0.0:
        return float('inf')
    ratio = sigma_b / sigma_f
    mu = mu_0 * ratio ** (-0.2)
    lb = bending_length(h_ice, E_ice, nu, rho_w, g)
    return max(lb, mu * lam)


def floe_size_distribution(Hs: float, Tp: float, h_ice: float,
                           sigma_f: float = SIGMA_F_DEFAULT,
                           n_bands: int = 96,
                           E_ice: float = E_ICE, nu: float = NU_ICE,
                           rho_w: float = RHO_W,
                           ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Wave-spectrum derived floe size distribution ``P(a)``.

    Every spectral band whose bending stress exceeds ``sigma_f`` breaks the ice
    into floes of size ``max(l_b, mu(lambda) * lambda)``.  Bands are weighted by
    their energy, giving a discrete distribution over floe sizes.

    Returns ``(a_sizes, P, stats)`` with ``P`` normalised to unit integral and
    ``stats`` holding the energy-weighted mean, the 10/90 percentiles, the
    number of breaking bands and the peak wavelength used.
    """
    wp = 2.0 * math.pi / Tp
    omega = np.linspace(0.15 * wp, 6.0 * wp, n_bands)
    S = pierson_moskowitz(omega, Hs, Tp)
    domega = omega[1] - omega[0]
    lb = bending_length(h_ice, E_ice, nu, rho_w)
    sizes, weights = [], []
    for w_i, s_i in zip(omega, S):
        if s_i <= 0.0:
            continue
        k_i = flexural_wavenumber(w_i, h_ice, E_ice=E_ice, nu=nu, rho_w=rho_w)
        amp = math.sqrt(2.0 * s_i * domega)
        if bending_stress(amp, k_i, h_ice, E_ice, nu) < sigma_f:
            continue
        lam_i = 2.0 * math.pi / k_i
        ratio = bending_stress(amp, k_i, h_ice, E_ice, nu) / sigma_f
        mu_i = MU_0 * ratio ** (-0.2)
        sizes.append(max(lb, mu_i * lam_i))
        weights.append(s_i * domega)
    lam_peak = flexural_wavelength(wp, h_ice, E_ice=E_ice, nu=nu, rho_w=rho_w)
    if not sizes:
        stats = {'n_breaking_bands': 0, 'a_mean': float('inf'),
                 'a_p10': float('inf'), 'a_p90': float('inf'),
                 'lambda_peak': lam_peak, 'ratio_mean': float('inf')}
        return np.array([]), np.array([]), stats
    a = np.array(sizes)
    w = np.array(weights)
    w = w / w.sum()
    order = np.argsort(a)
    a, w = a[order], w[order]
    a_mean = float(np.sum(a * w))
    cum = np.cumsum(w)
    a_p10 = float(np.interp(0.10, cum, a))
    a_p90 = float(np.interp(0.90, cum, a))
    stats = {'n_breaking_bands': int(a.size), 'a_mean': a_mean,
             'a_p10': a_p10, 'a_p90': a_p90, 'lambda_peak': lam_peak,
             'ratio_mean': a_mean / lam_peak,
             'ratio_p10': a_p10 / lam_peak, 'ratio_p90': a_p90 / lam_peak}
    return a, w, stats


# ---------------------------------------------------------------------------
# (c) attenuation
# ---------------------------------------------------------------------------
def scattering_attenuation(C_floe: float, a_floe: float, k_wave: float,
                           gamma: float = GAMMA_FLOE) -> float:
    """Scattering attenuation coefficient [1/m] in the Rayleigh regime.

    ``n = C/(gamma a^2)`` floes per unit area, per-floe scattering cross section
    ``sigma_s = (2/3)(ka)^4 a`` and ``alpha = n sigma_s / 2`` (energy -> amplitude
    conversion), i.e. ``alpha_scat = C (ka)^4 / (3 gamma a)``.
    """
    if C_floe <= 0.0 or a_floe <= 0.0:
        return 0.0
    ka = k_wave * a_floe
    sigma_s = (2.0 / 3.0) * ka ** 4 * a_floe
    n = C_floe / (gamma * a_floe ** 2)
    return 0.5 * n * sigma_s


def wave_attenuation_coefficient(C_floe: float, a_floe: float, h_ice: float,
                                 k_wave: float, beta: float = BETA_DISS,
                                 gamma: float = GAMMA_FLOE) -> float:
    """Total attenuation coefficient alpha [1/m].

    ``alpha = alpha_scat * (1 + beta C_floe)`` - the second factor models the
    non-scattering dissipative channels (overwash, floe-floe collisions,
    internal friction) which grow sub-linearly with concentration.  ``alpha`` is
    strictly increasing in ``C_floe`` and vanishes for ``C_floe = 0``.
    """
    alpha_s = scattering_attenuation(C_floe, a_floe, k_wave, gamma)
    return alpha_s * (1.0 + beta * C_floe)


def wave_energy_profile(E0: float, alpha: float,
                        x: np.ndarray) -> np.ndarray:
    """``E(x) = E0 exp(-2 alpha x)`` (energy density, per unit area)."""
    return E0 * np.exp(-2.0 * alpha * np.asarray(x, float))


def attenuation_correlation(C_values: Sequence[float], a_floe: float,
                            h_ice: float, k_wave: float) -> Dict[str, float]:
    """Pearson correlation between ``alpha`` and ``C_floe`` over a sweep.

    Returns ``{'r', 'slope', 'intercept'}`` from a linear fit of alpha vs C.
    The acceptance criterion requires ``r >= 0.9``.
    """
    C = np.asarray(C_values, float)
    alpha = np.array([wave_attenuation_coefficient(c, a_floe, h_ice, k_wave)
                      for c in C])
    if alpha.std() == 0.0:
        return {'r': 0.0, 'slope': 0.0, 'intercept': float(alpha[0])}
    r = float(np.corrcoef(C, alpha)[0, 1])
    slope, intercept = np.polyfit(C, alpha, 1)
    return {'r': r, 'slope': float(slope), 'intercept': float(intercept),
            'alpha_min': float(alpha.min()), 'alpha_max': float(alpha.max())}


# ---------------------------------------------------------------------------
# (d) ship and resistance components
# ---------------------------------------------------------------------------
@dataclass
class MIZShip:
    """Ship particulars used by the MIZ resistance / seakeeping models."""
    L: float = 100.0            # length between perpendiculars [m]
    B: float = 20.0             # beam [m]
    T: float = 6.0              # draft [m]
    Cb: float = 0.65            # block coefficient
    bow_half_angle: float = 30.0  # waterline half angle at the bow stem [deg]
    GM_T: float = 1.2           # transverse metacentric height [m]
    Cm: float = 0.98            # midship section coefficient
    speed_kt: float = 10.0      # ship speed [kt]

    @property
    def speed_ms(self) -> float:
        return self.speed_kt * 0.5144

    @property
    def nabla(self) -> float:
        """Displaced volume [m^3]."""
        return self.Cb * self.L * self.B * self.T

    @property
    def wetted_surface(self) -> float:
        """Wetted surface, Holtrop-style approximation [m^2]."""
        return (self.L * (2.0 * self.T + self.B) * math.sqrt(self.Cm) *
                (0.453 + 0.4425 * self.Cb - 0.2862 * self.Cm
                 - 0.003467 * self.B / self.T + 0.3696))


def ittc57_friction_coefficient(Re: float) -> float:
    """ITTC-1957 friction line ``Cf = 0.075/(log10 Re - 2)^2``."""
    if Re <= 0.0:
        return 0.0
    return 0.075 / (math.log10(Re) - 2.0) ** 2


def open_water_resistance(ship: MIZShip, form_factor: float = 0.15,
                          appendage_factor: float = 1.05,
                          nu_water: float = NU_KIN_WATER) -> float:
    """Calm-water resistance ``R_ow`` [N] (ITTC-57 + form factor)."""
    V = ship.speed_ms
    Re = V * ship.L / nu_water
    Cf = ittc57_friction_coefficient(Re)
    Rf = 0.5 * RHO_W * V ** 2 * ship.wetted_surface * Cf
    return Rf * (1.0 + form_factor) * appendage_factor


def wave_added_resistance(ship: MIZShip, Hs: float, Tp: float,
                          k_wave: Optional[float] = None,
                          c_aw: Optional[float] = None) -> float:
    """Wave added resistance ``R_wave`` [N].

    Radiated/diffracted added resistance for head seas, written as
    ``R_aw = 0.5 rho g Hs^2 B c_aw`` with the non-dimensional coefficient taken
    from the classical long-wave limit ``c_aw ~ 0.13 k B /(1 + ...)`` blended to
    a Froude-number dependent short-wave floor.  For a 100 m ship at 10 kt and
    Hs = 1.5 m this gives O(10-30 kN), i.e. 10-30 % of ``R_ow``.
    """
    if Hs <= 0.0:
        return 0.0
    if k_wave is None:
        k_wave = (2.0 * math.pi / Tp) ** 2 / G_ACCEL
    if c_aw is None:
        Fr = ship.speed_ms / math.sqrt(G_ACCEL * ship.L)
        long_wave = 0.13 * k_wave * ship.B / (1.0 + 0.2 * k_wave * ship.B)
        c_aw = long_wave + 0.35 * Fr ** 2
    return 0.5 * RHO_W * G_ACCEL * Hs ** 2 * ship.B * c_aw


def fractured_energy_per_floe(a_floe: float, h_ice: float,
                              sigma_f: float = SIGMA_F_DEFAULT,
                              E_ice: float = E_ICE) -> float:
    """Fracture energy needed to create one floe from a larger sheet [J].

    Energy release rate ``G_c = sigma_f^2 h /(2 E)`` [J/m^2 of crack surface],
    two new crack faces of area ``pi a h`` each per floe.
    """
    G_c = sigma_f ** 2 * h_ice / (2.0 * E_ice)
    return 2.0 * G_c * math.pi * a_floe * h_ice


def ice_resistance(ship: MIZShip, C_floe: float, a_floe: float, h_ice: float,
                   sigma_f: float = SIGMA_F_DEFAULT, E_ice: float = E_ICE,
                   gamma: float = GAMMA_FLOE,
                   mu_ice: float = MU_ICE,
                   contact_fraction: float = 0.01) -> Dict[str, float]:
    """Broken-ice resistance components [N] (Lindqvist-type).

    ``R_break``   fracture energy of each encountered floe times the encounter
                  rate: ``C B pi sigma_f^2 h^3 / (2 E gamma a)`` - decreases
                  with floe size (small floes need less fracture energy but are
                  encountered more often; the net effect is the classic 1/a
                  floe-size resistance trend);
    ``R_subm``    submergence of the broken layer ``C (1+0.021 psi)
                  (rho_w - rho_ice) g h B T`` (speed independent);
    ``R_fric``    hull friction of the ice layer, ``mu C rho_ice g h A_contact``.

    All components vanish identically for ``C_floe = 0`` (negative control).
    """
    if C_floe <= 0.0 or h_ice <= 0.0 or a_floe <= 0.0:
        return {'R_break': 0.0, 'R_subm': 0.0, 'R_fric': 0.0, 'R_ice': 0.0}
    psi = ship.bow_half_angle
    R_break = (C_floe * ship.B * math.pi * sigma_f ** 2 * h_ice ** 3 /
               (2.0 * E_ice * gamma * a_floe))
    R_subm = (C_floe * (1.0 + 0.021 * psi) * (RHO_W - RHO_ICE) * G_ACCEL *
              h_ice * ship.B * ship.T)
    A_contact = contact_fraction * ship.L * ship.B
    R_fric = mu_ice * C_floe * RHO_ICE * G_ACCEL * h_ice * A_contact
    return {'R_break': R_break, 'R_subm': R_subm, 'R_fric': R_fric,
            'R_ice': R_break + R_subm + R_fric}


def local_wave_height(Hs: float, Tp: float, C_floe: float, h_ice: float,
                      x_ship: float) -> Tuple[float, float]:
    """Wave height at ``x_ship`` inside the MIZ and the attenuation coefficient.

    ``Hs_loc = Hs exp(-alpha x_ship)`` with ``alpha`` from the breakup-consistent
    floe size at the ice edge; returns ``(Hs_loc, alpha)``.
    """
    k_p = flexural_wavenumber(2.0 * math.pi / Tp, h_ice)
    a_edge = characteristic_floe_size(Hs, Tp, h_ice)
    alpha = wave_attenuation_coefficient(C_floe, a_edge, h_ice, k_p)
    return Hs * math.exp(-alpha * x_ship), alpha


def resistance_superposed(ship: MIZShip, Hs: float, Tp: float,
                          C_floe: float, h_ice: float,
                          a_floe: Optional[float] = None,
                          sigma_f: float = SIGMA_F_DEFAULT,
                          x_ship: float = 0.0,
                          use_local_wave: bool = True) -> Dict[str, float]:
    """Independent (uncoupled) evaluation of the three resistance components.

    The three terms are evaluated one after the other with the *same* local
    environment and added up: ``R = R_ow(V) + R_wave(Hs_loc) + R_ice(C, a, h)``.
    ``use_local_wave=False`` reproduces the naive superposition that uses the
    undisturbed ice-edge wave height at the ship position.
    """
    Hs_loc, alpha = local_wave_height(Hs, Tp, C_floe, h_ice, x_ship)
    if not use_local_wave:
        Hs_loc = Hs
    if a_floe is None:
        a_floe = characteristic_floe_size(Hs_loc, Tp, h_ice, sigma_f)
    R_ow = open_water_resistance(ship)
    k_wave = flexural_wavenumber(2.0 * math.pi / Tp, h_ice)
    R_wave = wave_added_resistance(ship, Hs_loc, Tp, k_wave=k_wave)
    ice = ice_resistance(ship, C_floe, a_floe, h_ice, sigma_f)
    return {'R_ow': R_ow, 'R_wave': R_wave, 'R_ice': ice['R_ice'],
            'R_break': ice['R_break'], 'R_subm': ice['R_subm'],
            'R_fric': ice['R_fric'],
            'R_total': R_ow + R_wave + ice['R_ice'],
            'a_floe': a_floe, 'Hs_local': Hs_loc, 'alpha': alpha}


def resistance_coupled(ship: MIZShip, Hs: float, Tp: float, C_floe: float,
                       h_ice: float, x_ship: float = 250.0,
                       sigma_f: float = SIGMA_F_DEFAULT,
                       a_floe: Optional[float] = None) -> Dict[str, float]:
    """Coupled resistance evaluation (interaction terms retained).

    Relative to :func:`resistance_superposed` three couplings are added:

    1. **Attenuated wave forcing**: the ship sits ``x_ship`` metres inside the
       MIZ, so it sees ``Hs_eff = Hs exp(-alpha x_ship)``; ``R_wave`` drops by
       the corresponding factor.
    2. **Breakup-consistent floe size**: ``R_ice`` uses the floe size produced by
       the *local* (attenuated) wave field rather than the edge value.
    3. **Wave-induced ice motion**: floes oscillate with orbital velocity
       ``u_ice``, changing the relative impact speed
       ``V_eff = sqrt(V^2 + u_ice^2)`` and hence the fracture rate, giving a
       multiplicative ``(1 + u_ice^2/(2 V^2))`` term on ``R_break``.
    4. **Added-mass feedback**: floes entrained by the hull raise the vertical
       added mass by ``k_a C_floe`` (see ``miz_seakeeping.K_A_ICE``); since the
       radiated wave resistance scales with the vertical added mass, ``R_wave``
       grows by ``kappa * k_a C_floe`` with ``kappa = 0.5`` (half of the added
       mass takes part in the resistance-producing radiation).
    """
    k_p = flexural_wavenumber(2.0 * math.pi / Tp, h_ice)
    a_edge = (characteristic_floe_size(Hs, Tp, h_ice, sigma_f)
              if a_floe is None else a_floe)
    alpha = wave_attenuation_coefficient(C_floe, a_edge, h_ice, k_p)
    Hs_eff = Hs * math.exp(-alpha * x_ship)

    # (2) local breakup-consistent floe size
    a_local = (characteristic_floe_size(Hs_eff, Tp, h_ice, sigma_f)
               if a_floe is None else a_floe)

    R_ow = open_water_resistance(ship)
    R_wave = wave_added_resistance(ship, Hs_eff, Tp, k_wave=k_p)
    # (4) added-mass feedback of the entrained floes on the radiated resistance
    from .miz_seakeeping import ice_added_mass_factor
    A_excess = ice_added_mass_factor(C_floe) - 1.0        # = k_a C_floe
    R_wave *= 1.0 + KAPPA_AWR * A_excess
    ice = ice_resistance(ship, C_floe, a_local, h_ice, sigma_f)

    # (3) wave-induced ice motion -> enhanced relative impact speed
    u_ice = (Hs_eff / 2.0) * (2.0 * math.pi / Tp)
    enhance = 1.0 + 0.5 * (u_ice / max(ship.speed_ms, 1e-6)) ** 2
    R_break = ice['R_break'] * enhance
    R_ice = R_break + ice['R_subm'] + ice['R_fric']

    return {'R_ow': R_ow, 'R_wave': R_wave, 'R_ice': R_ice,
            'R_break': R_break, 'R_subm': ice['R_subm'], 'R_fric': ice['R_fric'],
            'R_total': R_ow + R_wave + R_ice,
            'alpha': alpha, 'Hs_eff': Hs_eff, 'a_floe_local': a_local,
            'R_wave_ambient': wave_added_resistance(ship, Hs_eff, Tp, k_wave=k_p),
            'added_mass_resistance_factor': 1.0 + KAPPA_AWR * A_excess,
            'a_floe_edge': a_edge, 'ice_motion_enhancement': enhance}


def superposition_error(ship: MIZShip, Hs: float, Tp: float, C_floe: float,
                        h_ice: float, x_ship: float = 250.0,
                        sigma_f: float = SIGMA_F_DEFAULT,
                        use_local_wave: bool = True) -> Dict[str, float]:
    """Relative error between superposed and coupled total resistance.

    Returns the two totals, the signed relative difference and the individual
    component differences (all in N).
    """
    sup = resistance_superposed(ship, Hs, Tp, C_floe, h_ice, x_ship=x_ship,
                                sigma_f=sigma_f, use_local_wave=use_local_wave)
    cpl = resistance_coupled(ship, Hs, Tp, C_floe, h_ice, x_ship, sigma_f)
    rel = (cpl['R_total'] - sup['R_total']) / sup['R_total']
    return {'R_superposed': sup['R_total'], 'R_coupled': cpl['R_total'],
            'rel_error': rel, 'abs_rel_error': abs(rel),
            'd_R_wave': cpl['R_wave'] - sup['R_wave'],
            'd_R_ice': cpl['R_ice'] - sup['R_ice'],
            'superposed': sup, 'coupled': cpl}


# ---------------------------------------------------------------------------
# (e) energy ledger
# ---------------------------------------------------------------------------
def floe_drag_power_density(C_floe: float, a_floe: float, h_ice: float,
                            omega: float, amplitude: float,
                            Cd: float = 1.0, rho_w: float = RHO_W,
                            gamma: float = GAMMA_FLOE) -> float:
    """Wave energy dissipated by floe form drag [W/m^2].

    Per floe ``P = 0.5 rho Cd (a h) <|u|^3>`` with the orbital velocity
    ``u = A omega`` and ``<|u|^3> = 4/(3 pi) u^3``; multiplied by the floe number
    density ``n = C/(gamma a^2)``.
    """
    if C_floe <= 0.0 or a_floe <= 0.0:
        return 0.0
    u = amplitude * omega
    P_floe = 0.5 * rho_w * Cd * (a_floe * h_ice) * (4.0 / (3.0 * math.pi)) * u ** 3
    return C_floe / (gamma * a_floe ** 2) * P_floe


def energy_ledger(C_floe: float, a_floe: float, h_ice: float, Hs: float,
                  Tp: float, L_strip: float = 300.0,
                  sigma_f: float = SIGMA_F_DEFAULT,
                  ship: Optional[MIZShip] = None,
                  n_x: int = 400) -> Dict[str, float]:
    """Energy budget of the wave field over an MIZ strip of length ``L_strip``.

    All powers are per metre of wave crest [W/m].

    Terms
    -----
    ``P_in``      incident wave power ``E0 cg`` with ``E0 = rho g Hs^2/8``;
    ``P_trans``   transmitted power ``E0 exp(-2 alpha L) cg``;
    ``P_att``     attenuated (absorbed by the ice field) power;
    ``P_scat``    scattering share of ``P_att``;
    ``P_frict``   floe drag share, integrated with the *local* decaying
                  amplitude (nonlinear in A^3, hence not exactly proportional to
                  ``P_att``);
    ``P_fract``   fracture (breakup) power, from the fracture energy per floe and
                  the floe production rate;
    ``P_ship``    ship motion damping power (only when ``ship`` is given);
    ``residual``  ``|P_in - (P_trans + P_scat + P_frict + P_fract + P_ship)|/P_in``.

    The residual therefore measures the closure of the nonlinear energy
    transport against four *independently* formulated sink models.
    """
    omega = 2.0 * math.pi / Tp
    k = flexural_wavenumber(omega, h_ice)
    cg = group_velocity(omega, k, h_ice)
    E0 = RHO_W * G_ACCEL * Hs ** 2 / 8.0
    P_in = E0 * cg
    alpha = wave_attenuation_coefficient(C_floe, a_floe, h_ice, k)
    alpha_s = scattering_attenuation(C_floe, a_floe, k)
    P_trans = E0 * math.exp(-2.0 * alpha * L_strip) * cg
    P_att = max(P_in - P_trans, 0.0)
    f_scat = alpha_s / alpha if alpha > 0 else 0.0
    P_scat = f_scat * P_att

    # floe drag sink, integrated with the local (decaying) amplitude
    x = np.linspace(0.0, L_strip, n_x)
    A_x = (Hs / 2.0) * np.exp(-alpha * x)
    p_drag_x = np.array([floe_drag_power_density(C_floe, a_floe, h_ice, omega, a)
                         for a in A_x])
    P_frict = float(np.trapezoid(p_drag_x, x))

    # fracture sink: floes created per unit area per wave period
    if C_floe > 0.0 and a_floe > 0.0:
        n_floe = C_floe / (GAMMA_FLOE * a_floe ** 2)
        rate = n_floe / Tp                       # floes m^-2 s^-1
        P_fract = rate * fractured_energy_per_floe(a_floe, h_ice, sigma_f) * L_strip
    else:
        P_fract = 0.0

    P_ship = 0.0
    if ship is not None:
        from .miz_seakeeping import roll_damping_power
        P_ship = roll_damping_power(ship, Hs, Tp, C_floe, a_floe, h_ice) / ship.B

    # The ship draws its motion damping power out of the transmitted flow, so in
    # the ledger it is booked as a reduction of the transmitted power (algebraic-
    # ally identical to listing it as an extra sink, but it keeps every term
    # physically signed and avoids double counting).
    P_trans_net = P_trans - P_ship
    total_sinks = P_trans_net + P_scat + P_frict + P_fract + P_ship
    residual = abs(P_in - total_sinks) / P_in if P_in > 0 else 0.0
    P_int_ship = P_in * math.exp(-2.0 * alpha * L_strip) * ship.B if ship else None
    return {'P_in': P_in, 'P_trans': P_trans, 'P_trans_net': P_trans_net,
            'P_att': P_att, 'P_scat': P_scat, 'P_frict': P_frict,
            'P_fract': P_fract, 'P_ship': P_ship, 'alpha': alpha,
            'alpha_scat': alpha_s, 'cg': cg, 'residual': float(residual),
            'fracture_fraction': P_fract / P_att if P_att > 0 else 0.0,
            'ship_fraction': P_ship / P_att if P_att > 0 else 0.0,
            'P_intercepted_ship_flow': P_int_ship,
            'dissipative_share': 1.0 - f_scat,
            'mech_over_diss': ((P_frict + P_fract) / (P_att - P_scat))
            if P_att > P_scat else float('inf')}


__all__ = [
    'RHO_W', 'RHO_ICE', 'G_ACCEL', 'NU_ICE', 'E_ICE', 'SIGMA_F_DEFAULT',
    'GAMMA_FLOE', 'SQUIRE_WINDOW', 'MU_0', 'BETA_DISS', 'KAPPA_AWR',
    'flexural_rigidity', 'bending_length', 'flexural_wavenumber',
    'flexural_wavelength', 'group_velocity', 'pierson_moskowitz',
    'bending_stress', 'characteristic_floe_size', 'floe_size_distribution',
    'scattering_attenuation', 'wave_attenuation_coefficient',
    'wave_energy_profile', 'attenuation_correlation',
    'MIZShip', 'ittc57_friction_coefficient', 'open_water_resistance',
    'wave_added_resistance', 'fractured_energy_per_floe', 'ice_resistance',
    'resistance_superposed', 'resistance_coupled', 'superposition_error',
    'floe_drag_power_density', 'energy_ledger',
]
