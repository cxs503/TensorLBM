"""Analytic anchors for rectangular-tank sloshing: eigen-frequencies, linear
free-decay kinematics, linear wall pressure and impact-pressure bounds.

This module is deliberately free of any numerical solver: it holds the
*closed-form* (REAL) reference against which the free-surface LBM of
:mod:`tensorlbm.sloshing_resonance.tank2d` is judged.

1. Eigen-frequencies (Abramson 1966, NASA SP-106 §2; Faltinsen & Timokha 2009,
   *Sloshing*, eq. 4.13).  For a rectangular tank of length ``L`` and
   still-water depth ``h`` the ``n``-th antisymmetric (sloshing) mode obeys

       k_n     = n*pi / L
       omega_n = sqrt(g k_n tanh(k_n h))
       f_n     = omega_n / (2*pi)                                     (1)

   .. note::
      The campaign brief quotes ``f_n = (n/2L) sqrt(g tanh(n pi h / L))``,
      which is **not dimensionally a frequency** (it yields 1/sqrt(length)/s).
      The dimensionally consistent form that the quoted expression is aiming at
      is ``f_n = (n / 2L) * sqrt(g * (L / (n pi)) * tanh(n pi h / L))``, which is
      algebraically identical to (1).  Criterion (a)/(b) are therefore evaluated
      against (1); the equivalence of the two forms is asserted in the tests.

2. Linear free-decay kinematics.  A tank released from rest with the pure
   ``n``-th mode surface elevation ``eta(x,0) = A cos(k_n x)`` and zero velocity
   oscillates as

       eta(x,t) = A cos(k_n x) cos(omega_n t),        u(x,y<eta,t=0) = 0   (2)

   i.e. the surface passes through the equilibrium with maximal speed at
   ``t = T_n/4`` (``eta = 0``), and returns to ``+A`` at ``t = T_n``.  The
   quarter-period crossing time is the phase anchor used by criterion (c).

3. Linear wall pressure (potential flow, ``p_dyn`` relative to the *still water*
   hydrostatic field, which is what an LBM gauge referenced to the rest state
   measures).  With ``eta = a cos(k_n x) cos(omega_n t)`` the velocity potential
   is ``phi = -(a omega_n / (k_n sinh(k_n h))) cosh(k_n (y + h)) cos(k_n x)
   sin(omega_n t)`` (``y`` measured up from the still-water line), so

       p_dyn(x,y,t) = -rho dphi/dt
                    = rho g a cosh(k_n (y+h)) / cosh(k_n h)
                      * cos(k_n x) cos(omega_n t)                   (3)

   At the wall (``x = 0``) this peaks at the still-water line with
   ``rho g a`` and decays as ``1/cosh(k_n h)`` at the bottom.  (3) is the
   linear counterpart of the LBM wall gauge; wave breaking / run-up impacts are
   outside its range of validity.

4. Impact-pressure bounds (used where the linear model is useless):

   * stagnation (Wagner, inviscid free-surface impact):
     ``p_stag = 0.5 rho V^2`` with ``V`` the characteristic wall run-up speed;
   * surge / bore reflection (Ritter 1892 shallow-water dam break, reflected
     bore): ``p_bore = rho g h_r`` with the reflected bore depth; for a bore
     travelling on depth ``h`` at speed ``v`` the simple momentum balance on a
     wall gives ``p_bore = rho g h (1 + v/(2 sqrt(g h)))^2`` ;
   * hydrostatic upper bound from the maximum steady run-up depth
     ``p_hyd = rho g (h + eta_max)``.

   These are bounds, not experimental data.  No digitised sloshing-impact
   *experiment* is shipped with this repository, so criterion (d) is reported as
   WITHHELD against experiment and evaluated against (3)/(4) instead — see the
   campaign JSON ``criterion_d`` block.

References
----------
Abramson, H. N. (1966). *The Dynamic Behavior of Liquids in Moving Containers*,
    NASA SP-106, §2.
Faltinsen, O. M. & Timokha, A. N. (2009). *Sloshing*. Cambridge University Press.
Ritter, A. (1892). "Die Fortpflanzung der Wasserwellen." Z. Ver. Deutsch. Ing. 36.
Wagner, H. (1932). "Über Stoß- und Gleitvorgänge an der Oberfläche von
    Flüssigkeiten." ZAMM 12, 193-215.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "DEEP_WATER_TOL",
    "AbramsonAnchor",
    "abramson_wave_number",
    "abramson_frequency",
    "abramson_brief_formula",
    "abramson_period",
    "abramson_angular_frequency",
    "deep_water_frequency",
    "shallow_water_frequency",
    "linear_decay_elevation",
    "linear_decay_velocity_amplitude",
    "linear_wall_pressure_amplitude",
    "linear_wall_pressure_bottom_ratio",
    "stagnation_impact_pressure",
    "bore_reflection_pressure",
    "hydrostatic_runup_pressure",
]

DEEP_WATER_TOL = 1e-12


def abramson_wave_number(mode: int, length: float) -> float:
    """``k_n = n pi / L`` (rad/m)."""
    if mode < 1:
        raise ValueError("mode must be >= 1")
    if length <= 0.0:
        raise ValueError("length must be positive")
    return mode * math.pi / length


def abramson_angular_frequency(mode: int, depth: float, length: float, g: float) -> float:
    """``omega_n = sqrt(g k_n tanh(k_n h))`` (rad/s) — Abramson (1966) eq. (1)."""
    if depth <= 0.0:
        raise ValueError("depth must be positive")
    k = abramson_wave_number(mode, length)
    return math.sqrt(g * k * math.tanh(k * depth))


def abramson_frequency(mode: int, depth: float, length: float, g: float) -> float:
    """``f_n = omega_n / (2 pi)`` (Hz) — the reference of criteria (a) and (b)."""
    return abramson_angular_frequency(mode, depth, length, g) / (2.0 * math.pi)


def abramson_brief_formula(mode: int, depth: float, length: float, g: float) -> float:
    """The campaign-brief expression, made dimensionally consistent.

    ``f_n = (n/2L) sqrt(g * L/(n pi) * tanh(n pi h / L))`` — algebraically the
    same as :func:`abramson_frequency`; kept as an executable proof that the
    brief's (mis-transcribed) formula reduces to (1).
    """
    k = abramson_wave_number(mode, length)
    return (mode / (2.0 * length)) * math.sqrt(g * length / (mode * math.pi) * math.tanh(k * depth))


def abramson_period(mode: int, depth: float, length: float, g: float) -> float:
    """Natural period ``T_n = 1/f_n`` (s)."""
    return 1.0 / abramson_frequency(mode, depth, length, g)


def shallow_water_frequency(mode: int, depth: float, length: float, g: float) -> float:
    """Shallow-water asymptote ``f_n -> k_n sqrt(g h) / (2 pi)`` (``k h -> 0``)."""
    k = abramson_wave_number(mode, length)
    return k * math.sqrt(g * depth) / (2.0 * math.pi)


def deep_water_frequency(mode: int, length: float, g: float) -> float:
    """Deep-water asymptote ``f_n -> sqrt(g k_n) / (2 pi)`` (``k h -> inf``)."""
    k = abramson_wave_number(mode, length)
    return math.sqrt(g * k) / (2.0 * math.pi)


def linear_decay_elevation(
    mode: int, amplitude: float, x: float, t: float, depth: float, length: float, g: float
) -> float:
    """``eta(x,t) = A cos(k_n x) cos(omega_n t)`` — equation (2)."""
    k = abramson_wave_number(mode, length)
    w = abramson_angular_frequency(mode, depth, length, g)
    return amplitude * math.cos(k * x) * math.cos(w * t)


def linear_decay_velocity_amplitude(
    mode: int, amplitude: float, depth: float, length: float, g: float
) -> float:
    """Peak horizontal velocity on the still-water line, ``a omega coth(k h)``."""
    k = abramson_wave_number(mode, length)
    w = abramson_angular_frequency(mode, depth, length, g)
    return amplitude * w / math.tanh(k * depth)


def linear_wall_pressure_amplitude(
    mode: int, amplitude: float, y: float, depth: float, length: float, g: float, rho: float
) -> float:
    """Amplitude of ``p_dyn`` at the wall, equation (3), ``y`` up from the SWL."""
    k = abramson_wave_number(mode, length)
    if y + depth < 0.0:
        raise ValueError("gauge below the tank bottom")
    return rho * g * amplitude * math.cosh(k * (y + depth)) / math.cosh(k * depth)


def linear_wall_pressure_bottom_ratio(mode: int, depth: float, length: float) -> float:
    """``p_dyn(bottom) / p_dyn(SWL) = 1 / cosh(k_n h)`` — depth-decay factor of (3)."""
    k = abramson_wave_number(mode, length)
    return 1.0 / math.cosh(k * depth)


def stagnation_impact_pressure(rho: float, velocity: float) -> float:
    """Wagner stagnation pressure ``0.5 rho V^2`` (Pa) — bound (4a)."""
    return 0.5 * rho * velocity * velocity


def bore_reflection_pressure(rho: float, g: float, depth: float, velocity: float) -> float:
    """Bore-on-wall pressure ``rho g h (1 + v/(2 sqrt(g h)))^2`` (Pa) — bound (4b).

    Momentum balance for a bore of speed ``v`` running into a vertical wall on a
    still-water depth ``h``; the wall run-up depth is ``h_r = h (1 + v/(2 sqrt(g
    h)))^2`` and the pressure at the wall foot is hydrostatic in ``h_r``.
    """
    if depth <= 0.0:
        raise ValueError("depth must be positive")
    fr = velocity / (2.0 * math.sqrt(g * depth))
    return rho * g * depth * (1.0 + fr) ** 2


def hydrostatic_runup_pressure(rho: float, g: float, depth: float, runup: float) -> float:
    """Hydrostatic pressure of the maximum run-up ``rho g (h + eta_max)`` (Pa)."""
    return rho * g * (depth + max(runup, 0.0))


@dataclass(frozen=True)
class AbramsonAnchor:
    """One row of the Abramson frequency ladder (all quantities in SI)."""

    mode: int
    depth: float
    length: float
    g: float

    @property
    def wave_number(self) -> float:
        return abramson_wave_number(self.mode, self.length)

    @property
    def frequency(self) -> float:
        return abramson_frequency(self.mode, self.depth, self.length, self.g)

    @property
    def angular_frequency(self) -> float:
        return abramson_angular_frequency(self.mode, self.depth, self.length, self.g)

    @property
    def period(self) -> float:
        return 1.0 / self.frequency

    @property
    def kh(self) -> float:
        return self.wave_number * self.depth

    @property
    def regime(self) -> str:
        if self.kh < 0.5:
            return "shallow"
        if self.kh > 3.0:
            return "deep"
        return "intermediate"
