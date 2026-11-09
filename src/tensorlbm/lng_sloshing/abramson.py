"""Abramson (1966) rectangular-tank sloshing anchors and the matching linear theory.

Analytic (REAL, closed-form) reference used by the LNG-tank part-fill sloshing
campaign.  Two independent groups of formulae are provided:

1. **Eigen-frequencies** — the classical potential-flow result for a rectangular
   tank of length ``L`` and still-water depth ``h`` (Abramson 1966, NASA SP-106,
   *The Dynamic Behavior of Liquids in Moving Containers*, §2; repeated in
   Faltinsen & Timokha 2009, *Sloshing*, eq. 4.13):

       omega_n^2 = g * k_n * tanh(k_n h),      k_n = n*pi/L,
       f_n       = omega_n / (2*pi).

   The limit cases ``k h -> 0`` (``omega_n -> k_n sqrt(g h)``, shallow water) and
   ``k h -> inf`` (``omega_n -> sqrt(g k_n)``, deep water) are reproduced
   exactly and are covered by the unit tests.

2. **Linear forced response and wall pressure** — the first-mode (odd-mode)
   response of a tank in harmonic sway ``X(t) = X0 sin(omega t)``.  Derivation
   (tank-fixed frame, ``x`` in ``[0, L]``, ``y`` measured up from the free
   surface, bottom at ``y = -h``):

       phi   = sum_n a_n(t) cos(k_n x) cosh(k_n (y+h)) / cosh(k_n h)
       eta   = sum_n eta_n(t) cos(k_n x)
       d eta_n/dt = k_n tanh(k_n h) a_n                (kinematic BC)
       d a_n/dt + g eta_n = C_n a_x(t),
           C_n = 4 / (L k_n^2)                          (Bernoulli, projected)

   which collapses to a forced oscillator per mode

       d^2 eta_n/dt^2 + omega_n^2 eta_n = beta_n a_x(t),
           beta_n = 4 tanh(k_n h) / (n pi).

   Harmonic sway of amplitude ``X0`` at frequency ``omega`` gives
   ``a_x = X0 omega^2`` (acceleration amplitude) and the steady-state first-mode
   elevation amplitude

       |eta_1| = X0 * beta_1 * (omega^2/omega_1^2)
                 / sqrt((1 - omega^2/omega_1^2)^2 + (2 zeta omega/omega_1)^2).

   The dynamic wall pressure (relative to the *rest* hydrostatic field, i.e.
   what an LBM gauge referenced to the still-water profile measures) is

       p_dyn(x, y, t) = -rho [ dphi/dt + a_x(t) x ],

   so at the tank wall (``x = 0``, the term ``a_x x`` vanishes) the amplitude is

       |p_wall(y)| = rho g |eta_1| (omega^2/omega_1^2)
                     * cosh(k_1 (y+h)) / cosh(k_1 h),

   i.e. the pressure decays with depth as a cosh profile and peaks at the still
   water line.  This is the linear (potential-flow) counterpart of the impact
   pressure extracted from the free-surface LBM run; the nonlinear bore impact is
   *not* described by it (documented limitation).

The static limit is used as an internal consistency test of the derivation: for
``omega -> 0`` the modal amplitude tends to the modal projection of the rigid
tilt ``eta = -(a_x/g)(x - L/2)``, that is ``|eta_1| -> (a_x/g) * 4/pi^2``.

References
----------
Abramson, H. N. (1966). *The Dynamic Behavior of Liquids in Moving Containers*,
    NASA SP-106, ch. 2 (natural frequencies of a rectangular tank).
Faltinsen, O. M. & Timokha, A. N. (2009). *Sloshing*, Cambridge University
    Press, ch. 4 (linear modal theory), ch. 6 (second-order / nonlinear).
Faltinsen, O. M. (1978). A nonlinear theory of sloshing in rectangular tanks.
    J. Ship Research 22(4), 241-249.
Wagner, H. (1932). Ueber Stoss- und Gleitvorgaenge an der Oberflaeche von
    Fluessigkeiten. ZAMM 12, 193-215 (stagnation-pressure bound used for the
    bore-impact cross-check).
"""

from __future__ import annotations

import math

__all__ = [
    "abramson_wave_number",
    "abramson_angular_frequency",
    "abramson_natural_frequency",
    "abramson_mode_shape",
    "abramson_mode_ratio",
    "abramson_hydrostatic_wave_speed",
    "linear_forced_modal_amplitude",
    "linear_wall_pressure_amplitude",
    "linear_response_ratio",
    "stagnation_impact_pressure",
    "SP106_REFERENCE",
]

SP106_REFERENCE = (
    "Abramson, H.N. (1966) NASA SP-106 'The Dynamic Behavior of Liquids in "
    "Moving Containers', ch.2: omega_n^2 = g k_n tanh(k_n h), k_n = n pi / L"
)


def abramson_wave_number(mode: int, length: float) -> float:
    """``k_n = n*pi/L`` (1D standing wave in a rectangular tank)."""
    if mode < 1:
        raise ValueError(f"mode must be >= 1, got {mode}")
    if length <= 0.0:
        raise ValueError(f"length must be > 0, got {length}")
    return mode * math.pi / length


def abramson_angular_frequency(
    mode: int, length: float, depth: float, g: float = 9.81
) -> float:
    """Abramson (1966) angular natural frequency ``omega_n`` [rad/s]."""
    k = abramson_wave_number(mode, length)
    return math.sqrt(g * k * math.tanh(k * depth))


def abramson_natural_frequency(
    mode: int, length: float, depth: float, g: float = 9.81
) -> float:
    """Abramson (1966) natural frequency ``f_n = omega_n / (2 pi)`` [Hz]."""
    return abramson_angular_frequency(mode, length, depth, g) / (2.0 * math.pi)


def abramson_mode_shape(mode: int, x: float, length: float) -> float:
    """First-order standing-wave mode shape ``cos(k_n x)`` (x measured from wall)."""
    return math.cos(abramson_wave_number(mode, length) * x)


def abramson_mode_ratio(fundamental: float, depth: float, length: float) -> float:
    """``omega_2/omega_1`` for the second sloshing mode (deep-water limit 2.0)."""
    if fundamental <= 0.0:
        raise ValueError("fundamental must be > 0")
    w2 = abramson_angular_frequency(2, length, depth)
    w1 = fundamental
    return w2 / w1


def abramson_hydrostatic_wave_speed(depth: float, g: float = 9.81) -> float:
    """Long-wave (shallow water) limit speed ``sqrt(g h)``."""
    return math.sqrt(g * max(depth, 0.0))


def _damping_denominator(omega: float, omega_n: float, zeta: float) -> float:
    r = omega / omega_n
    return math.sqrt((1.0 - r * r) ** 2 + (2.0 * zeta * r) ** 2)


def linear_forced_modal_amplitude(
    mode: int,
    length: float,
    depth: float,
    sway_amplitude: float,
    omega: float,
    g: float = 9.81,
    zeta: float = 0.0,
) -> float:
    """Steady-state first-order elevation amplitude at the antinode (mode ``n``).

    Returns ``|eta_n|`` in the same units as ``sway_amplitude``.  Only odd modes
    are driven by horizontal sway (``beta_n = 4 tanh(k_n h)/(n pi)``), so even
    modes return the (mathematically correct) zero forcing for ``n`` even.
    """
    if sway_amplitude < 0.0:
        raise ValueError("sway_amplitude must be >= 0")
    wn = abramson_angular_frequency(mode, length, depth, g)
    k = abramson_wave_number(mode, length)
    beta = 4.0 * math.tanh(k * depth) / float(mode) / math.pi
    if mode % 2 == 0:
        beta = 0.0
    denom = _damping_denominator(omega, wn, zeta)
    if denom <= 0.0:  # pragma: no cover - guarded by construction
        return float("inf")
    return abs(sway_amplitude * omega * omega * beta) / (wn * wn * denom)


def linear_response_ratio(
    length: float, depth: float, omega: float, g: float = 9.81, zeta: float = 0.0
) -> float:
    """``|eta_1|/X0`` for the first mode (dimensionless linear response)."""
    return linear_forced_modal_amplitude(
        1, length, depth, 1.0, omega, g=g, zeta=zeta
    )


def linear_wall_pressure_amplitude(
    y: float,
    length: float,
    depth: float,
    eta_amplitude: float,
    omega: float,
    g: float = 9.81,
    sway_amplitude: float = 0.0,
    zeta: float = 0.0,
) -> float:
    """Linear dynamic wall pressure amplitude at height ``y`` (0 = still water).

    Pressure head ``p/(rho g)`` in units of length, directly comparable with a
    ``p/(rho g)`` LBM density gauge.  Two terms contribute, and **their relative
    phase matters**:

    * modal (potential) term, in phase with the surface elevation
      ``eta_1(t)``:  ``|eta_1| (omega^2/omega_1^2) cosh(k(y+h))/cosh(k h)``;
    * inertial / quasi-static tilt term, in phase with the tank acceleration:
      ``X0 omega^2 L / (2 g)``, independent of ``y``.

    The modal term lags the tank acceleration by the driven-oscillator phase
    ``phi = atan2(2 zeta r, 1 - r^2)``, ``r = omega/omega_1``, so the two terms
    add as phasors::

        head = |tilt + modal * exp(-i phi)|
             = hypot(tilt + modal cos(phi), modal sin(phi)).

    The ``x - L/2`` origin in the Bernoulli equation (fixed by volume
    conservation, ``p/rho = -(dphi/dt + g y + a_x (x - L/2))``) is what produces
    the tilt term.  Ignoring either the tilt term or the phase gives 2-4x errors
    in the wall pressure at off-resonance frequency ratios - both mistakes were
    found by comparing against the free-surface LBM (see the campaign report and
    the unit tests).

    Limits asserted by the tests: ``omega -> 0`` gives the static tilt at the
    wall ``a_x L/(2 g)``; ``zeta = 0`` reduces to the in-phase/anti-phase sum.
    """
    k = abramson_wave_number(1, length)
    wn = abramson_angular_frequency(1, length, depth, g)
    r = omega / wn
    decay = math.cosh(k * (y + depth)) / math.cosh(k * depth)
    modal = abs(eta_amplitude) * r * r * decay
    tilt = abs(sway_amplitude) * omega * omega * length / (2.0 * g)
    phi = math.atan2(2.0 * zeta * r, 1.0 - r * r)
    return math.hypot(tilt + modal * math.cos(phi), modal * math.sin(phi))


def stagnation_impact_pressure(velocity: float, rho: float = 1.0, cp: float = 1.0) -> float:
    """Stagnation (Wagner-type) bound for a bore front hitting a wall.

    ``p = cp * 0.5 * rho * V^2`` with ``cp = 1`` (classical incompressible
    stagnation value, Wagner 1932; used as an order-of-magnitude physics
    cross-check, not as an impact-law calibration).
    """
    return cp * 0.5 * rho * velocity * velocity