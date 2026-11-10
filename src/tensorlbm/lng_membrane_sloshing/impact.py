"""Wagner / MLM impact-pressure anchors for LNG membrane-tank sloshing.

Two closed-form impact models bracket the wall pressure of a liquid front that
runs into a vertical tank wall, and both are derived here (rather than quoted)
so that every step is checkable inside the repository:

1. **Wagner (1932) flat-plate model.**  For a rigid 2-D body of deadrise angle
   ``beta`` entering an inviscid liquid with normal velocity ``V``, the wetted
   half-width obeys the Wagner condition and grows as

       c(t) = (pi/2) V t / tan(beta).                                    (1)

   With the fluid below the plate and the plate moving in ``+y`` at speed
   ``V``, the plate boundary condition ``v_y = V`` on ``|x| < c`` is satisfied by
   the complex potential

       w(z) = -i V (z - sqrt(z^2 - c^2)),   phi = Re w,                 (2)

   which gives ``phi = -V sqrt(c^2 - x^2)`` on the wetted part.  The linearised
   unsteady Bernoulli equation ``p = -rho phi_t`` then yields the classical
   Wagner spray-root pressure

       p(x,t) = rho V c cdot / sqrt(c^2 - x^2)
              = rho V^2 (pi / (2 tan(beta))) / sqrt(1 - xi^2),  xi = x/c,  (3)

   which is *integrable* but singular at the spray root (``xi -> 1``), and whose
   integrated value is the Wagner slam force per unit span

       F(t) = int_{-c}^{c} p dx = rho V^2 pi^2 c(t) / (2 tan(beta)).      (4)

   In the degenerate flat-impact limit ``beta -> 0`` (a vertical wall hit
   normally by a liquid front, which is the sloshing configuration) the
   wagnerian peak pressure diverges and the physical peak is set by the liquid
   compressibility instead: the acoustic water-hammer value

       p_WH = rho c_s V,                                                 (5)

   with ``c_s`` the sound speed of the liquid (in the lattice: ``c_s = 1/sqrt3``;
   in LNG: about 1250 m/s).  This is *Wagner's beta -> 0 regularisation* and it
   is the anchor the campaign brief quotes as ``p_max = rho c V``.

2. **MLM (modified Logvinovich model, Korobkin 2004).**  MLM keeps the
   *nonlinear* Bernoulli equation while using the Wagner flow field:

       p = -rho (phi_t + |grad phi|^2 / 2).                              (6)

   Evaluating (6) with (2) at the stagnation point of the wetted body
   (``x = 0``, where ``u = 0`` and ``v = V``) gives the closed form used here,

       p_MLM = rho V^2 [ pi / (2 tan(beta)) - 1/2 ],                      (7)
       p_MLM = rho V^2 (1/2) * [ pi / tan(beta) - 1 ],

   i.e. the incompressible stagnation pressure ``1/2 rho V^2`` plus the
   unsteady (added-mass) term that grows as the deadrise flattens.  (7) is
   positive only for ``tan(beta) < pi``, i.e. ``beta < 72.3 deg``; beyond that
   the linearised kinematics behind (2) is no longer compatible with a
   non-negative pressure and MLM is declared out of validity (Korobkin 2004
   reports the same restriction; the *force* form of MLM stays valid further).

   For ``beta = 90 deg`` (a body sliding parallel to the free surface, i.e. no
   impact at all in the water-entry sense) (7) degenerates to a *negative*
   value, so the physically meaningful low-amplitude limit used for the
   bracket is the pure stagnation value ``1/2 rho V^2``.

Dimensionless form used by the campaign
--------------------------------------
All comparisons are made on ``p* = p_max / (rho V^2)`` with ``V`` the measured
normal impact velocity, so the anchors reduce to

    p*_stag = 0.5                          (incompressible stagnation)
    p*_MLM(beta) = pi/(2 tan beta) - 0.5   (MLM, eq. 7)
    p*_WH = 1/Ma,  Ma = V/c_s              (water hammer, eq. 5)

and are independent of the tank scale.  The campaign reports the LBM peak
against the MLM anchor (criterion (2), 3 fill levels) and quotes the stagnation
and water-hammer values as the lower/upper bracket of the same impact event.

References
----------
Wagner, H. (1932). ZAMM 12, 193-215.
Korobkin, A. A. (2004). "Analytical models of water impact." Euro. J. Appl.
    Math. 15, 821-838.
Zhao, R. & Faltinsen, O. M. (1993). J. Fluid Mech. 246, 593-612.
Bagnold, R. A. (1939). "Interim report on wave-pressure research." J. Inst.
    Civ. Eng. 12, 202-226.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "WagnerImpact",
    "wagner_wetted_halfwidth",
    "wagner_contact_time",
    "wagner_complex_potential",
    "wagner_plate_potential",
    "wagner_pressure_profile",
    "wagner_slam_force_per_length",
    "mlm_stagnation_pressure",
    "mlm_validity_limit_deg",
    "water_hammer_pressure",
    "stagnation_pressure",
    "impact_pressure_anchors",
    "dimensionless_anchors",
    "MLM_BETA_LIMIT_DEG",
    "LNG_SOUND_SPEED",
]

#: Sound speed in liquid methane / LNG near its boiling point [m/s]
#: (about 1250 m/s at 111 K, 1 bar; used only for the physical-unit bracket).
LNG_SOUND_SPEED = 1250.0

#: ``tan(beta) = pi`` -> MLM pressure (7) changes sign; above this deadrise the
#: linearised Wagner kinematics and a non-negative pressure are incompatible.
MLM_BETA_LIMIT_DEG = math.degrees(math.atan(math.pi))


# ---------------------------------------------------------------------------
# Wagner (1932)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WagnerImpact:
    """A constant-velocity 2-D impact of deadrise ``beta`` at speed ``V``."""

    v_impact: float
    beta_deg: float
    rho: float = 1.0

    def __post_init__(self) -> None:
        if self.v_impact <= 0.0:
            raise ValueError("v_impact must be > 0")
        if not (0.0 < self.beta_deg < 90.0):
            raise ValueError("beta_deg must be in (0, 90)")
        if self.rho <= 0.0:
            raise ValueError("rho must be > 0")

    @property
    def beta_rad(self) -> float:
        return math.radians(self.beta_deg)

    @property
    def tan_beta(self) -> float:
        return math.tan(self.beta_rad)


def wagner_wetted_halfwidth(t: float, impact: WagnerImpact) -> float:
    """Wagner wetted half-width ``c(t) = (pi/2) V t / tan(beta)`` [m]."""
    if t <= 0.0:
        return 0.0
    return 0.5 * math.pi * impact.v_impact * t / impact.tan_beta


def wagner_contact_time(halfwidth: float, impact: WagnerImpact) -> float:
    """Time at which the wetted half-width reaches ``halfwidth`` [s]."""
    if halfwidth < 0.0:
        raise ValueError("halfwidth must be >= 0")
    return 2.0 * halfwidth * impact.tan_beta / (math.pi * impact.v_impact)


def wagner_complex_potential(
    x: np.ndarray | float, y: np.ndarray | float, c: float, v: float
) -> np.ndarray | complex:
    """Complex potential (2) ``w = -i V (z - sqrt(z^2 - c^2))``, ``z = x + i y``.

    Principal-branch convention: the *fluid* is the upper half plane ``y >= 0``
    and the plate at ``y = 0`` is approached from above, which gives
    ``phi = -V sqrt(c^2 - x^2)`` on the wetted part and ``v_y = +V`` there (see
    ``test_wagner_potential_satisfies_plate_boundary_condition``).  The physical
    picture is recovered by flipping ``y -> -y`` (the fluid below the plate).
    """
    z = np.asarray(x, dtype=complex) + 1j * np.asarray(y, dtype=complex)
    # sqrt(z^2 - c^2) built as the product of the two principal roots: this is
    # the branch that is analytic in the cut plane (cut = [-c, c]) and *continuous*
    # when the plate is approached from either half plane, which the single
    # np.sqrt(z**2 - c**2) is not.
    root = np.sqrt(z - c + 0j) * np.sqrt(z + c + 0j)
    return -1j * v * (z - root)


def wagner_plate_potential(x: np.ndarray | float, c: float, v: float) -> np.ndarray:
    """Potential on the wetted plate ``phi = -V sqrt(c^2 - x^2)`` (|x| < c)."""
    xa = np.asarray(x, dtype=float)
    return -v * np.sqrt(np.clip(c * c - xa * xa, 0.0, None))


def wagner_pressure_profile(
    x: np.ndarray | float,
    t: float,
    impact: WagnerImpact,
    *,
    xi_clip: float = 1.0 - 1e-6,
) -> np.ndarray:
    """Linearised Wagner pressure (3) [Pa] on the wetted part of the body.

    ``p = rho V^2 pi/(2 tan beta) / sqrt(1 - xi^2)`` for ``|xi| < 1`` and ``0``
    outside.  The spray-root value is clipped at ``|xi| = xi_clip`` (the
    singularity is physical in the inviscid model and is regularised by
    compressibility, eq. 5).
    """
    c = wagner_wetted_halfwidth(t, impact)
    xa = np.asarray(x, dtype=float)
    if c <= 0.0:
        return np.zeros_like(xa)
    xi = np.clip(xa / c, -xi_clip, xi_clip)
    inside = np.abs(xa) < c
    p = impact.rho * impact.v_impact**2 * (0.5 * math.pi / impact.tan_beta) / np.sqrt(
        1.0 - xi * xi
    )
    return np.where(inside, p, 0.0)


def wagner_slam_force_per_length(t: float, impact: WagnerImpact) -> float:
    """Wagner slam force per unit span, eq. (4) [N/m]."""
    c = wagner_wetted_halfwidth(t, impact)
    return impact.rho * impact.v_impact**2 * math.pi**2 * c / (2.0 * impact.tan_beta)


# ---------------------------------------------------------------------------
# MLM and the bracket
# ---------------------------------------------------------------------------


def mlm_stagnation_pressure(impact: WagnerImpact) -> float:
    """MLM peak pressure at the stagnation point, eq. (7) [Pa].

    Returns a *negative* number for ``beta > 72.3 deg``; callers that need a
    physical bound use :func:`mlm_bounded_pressure`.
    """
    return impact.rho * impact.v_impact**2 * (
        0.5 * math.pi / impact.tan_beta - 0.5
    )


def mlm_validity_limit_deg() -> float:
    """Deadrise above which eq. (7) becomes negative [deg] (``atan(pi)``)."""
    return MLM_BETA_LIMIT_DEG


def mlm_bounded_pressure(impact: WagnerImpact) -> float:
    """MLM peak pressure clipped to the stagnation value for large deadrise.

    For ``beta >= atan(pi)`` eq. (7) is out of its validity window; the
    physically meaningful bound is then the incompressible stagnation pressure
    ``1/2 rho V^2`` (the ``beta -> 90 deg`` limit of the bracket).
    """
    raw = mlm_stagnation_pressure(impact)
    return max(raw, stagnation_pressure(impact.rho, impact.v_impact))


def stagnation_pressure(rho: float, v: float) -> float:
    """Incompressible stagnation pressure ``1/2 rho V^2`` [Pa]."""
    return 0.5 * rho * v * v


def water_hammer_pressure(rho: float, c_sound: float, v: float) -> float:
    """Acoustic water-hammer pressure ``rho c V`` [Pa] (Wagner beta=0 limit)."""
    for name, value in (("rho", rho), ("c_sound", c_sound)):
        if value <= 0.0:
            raise ValueError(f"{name} must be > 0")
    if v < 0.0:
        raise ValueError("v must be >= 0")
    return rho * c_sound * abs(v)


def impact_pressure_anchors(
    v_impact: float,
    beta_deg: float,
    *,
    rho: float,
    c_sound: float,
) -> dict[str, float]:
    """All three anchors for one impact event [Pa] plus their dimensionless form."""
    impact = WagnerImpact(v_impact=v_impact, beta_deg=beta_deg, rho=rho)
    return dimensionless_anchors(impact, c_sound=c_sound)


def dimensionless_anchors(
    impact: WagnerImpact, *, c_sound: float
) -> dict[str, float]:
    """Anchors normalised by ``rho V^2`` (and the Mach number)."""
    q = impact.rho * impact.v_impact**2
    mach = impact.v_impact / c_sound
    return {
        "rho_v2": q,
        "mach": mach,
        "p_star_stagnation": 0.5,
        "p_star_mlm": impact.v_impact**2 * (0.5 * math.pi / impact.tan_beta - 0.5)
        / impact.v_impact**2,
        "p_star_mlm_physical_bound": mlm_bounded_pressure(impact) / q,
        "p_star_water_hammer": 1.0 / mach,
        "beta_deg": impact.beta_deg,
        "mlm_valid": impact.beta_deg < MLM_BETA_LIMIT_DEG,
    }