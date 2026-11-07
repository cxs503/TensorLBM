"""Exact jump relations for a perfect gas (Rankine-Hugoniot family).

Everything here is closed-form physics -- these are the *reference solutions*
against which the kinetic solver is judged.  All functions are pure and take
``gamma`` so that callers can audit the perfect-gas assumption explicitly.

Conventions
-----------
* Shock-stationary frame: ``M1`` is the upstream Mach number normal to the
  front.
* Lab frame (moving shock): the upstream gas is at rest and the shock runs into
  it with shock Mach number ``Ms = W / a1``.
* Oblique shocks: ``beta`` is the shock angle measured from the upstream flow
  direction, ``theta`` the flow deflection.

All thermodynamic quantities are nondimensional (``p``, ``rho``, ``T`` with
``R = 1``), i.e. ``T = p / rho`` and ``a = sqrt(gamma p / rho)``.
"""

from __future__ import annotations

import math

GAMMA = 1.4

__all__ = [
    "GAMMA",
    "sound_speed",
    "normal_shock",
    "moving_shock",
    "oblique_shock_beta",
    "oblique_shock",
    "prandtl_meyer",
    "shock_tube_driver",
    "stagnation_pressure_ratio",
]


def sound_speed(rho: float, p: float, gamma: float = GAMMA) -> float:
    """Speed of sound ``a = sqrt(gamma p / rho)``."""
    return math.sqrt(gamma * p / rho)


def normal_shock(M1: float, gamma: float = GAMMA) -> dict:
    """Exact normal-shock jump for upstream Mach ``M1 >= 1`` (shock frame).

    Returns density / pressure / temperature ratios, the downstream Mach number
    and the nondimensional entropy rise.  Raises for ``M1 < 1`` because an
    expansion cannot be represented by these relations.
    """
    if M1 < 1.0:
        raise ValueError(f"normal_shock requires M1 >= 1 (got {M1})")
    m2 = M1 * M1
    rho_ratio = (gamma + 1.0) * m2 / ((gamma - 1.0) * m2 + 2.0)
    p_ratio = 1.0 + 2.0 * gamma / (gamma + 1.0) * (m2 - 1.0)
    t_ratio = p_ratio / rho_ratio
    M2 = math.sqrt(((gamma - 1.0) * m2 + 2.0) / (2.0 * gamma * m2 - (gamma - 1.0)))
    ds = math.log(p_ratio) - gamma * math.log(rho_ratio)  # >0 by the 2nd law
    return {
        "M1": M1,
        "M2": M2,
        "rho_ratio": rho_ratio,
        "p_ratio": p_ratio,
        "T_ratio": t_ratio,
        "entropy_rise": ds,
        # stagnation pressure ratio p02/p01 (<1: total-pressure loss)
        "p0_ratio": stagnation_pressure_ratio(M1, gamma),
    }


def stagnation_pressure_ratio(M: float, gamma: float = GAMMA) -> float:
    """Isentropic total pressure ratio ``p0/p`` for Mach ``M``."""
    return (1.0 + 0.5 * (gamma - 1.0) * M * M) ** (gamma / (gamma - 1.0))


def moving_shock(
    Ms: float,
    rho1: float = 1.0,
    p1: float = 1.0,
    gamma: float = GAMMA,
    u1: float = 0.0,
) -> dict:
    """Lab-frame moving shock into gas with state ``(rho1, p1, u1=0)``.

    ``Ms = W / a1`` is the shock Mach number.  Returns the shock speed ``W``,
    the post-shock state in the lab frame and the jump ratios.  This is the
    reference for the ``prescribed-drop`` normal-shock verification cases.
    """
    if Ms < 1.0:
        raise ValueError(f"moving_shock requires Ms >= 1 (got {Ms})")
    a1 = sound_speed(rho1, p1, gamma)
    W = Ms * a1
    jump = normal_shock(Ms, gamma)
    rho2 = rho1 * jump["rho_ratio"]
    p2 = p1 * jump["p_ratio"]
    T2 = p2 / rho2
    a2 = sound_speed(rho2, p2, gamma)
    u2 = W * (1.0 - rho1 / rho2)  # mass conservation in the shock frame
    return {
        "Ms": Ms,
        "W": W,
        "rho1": rho1,
        "p1": p1,
        "T1": p1 / rho1,
        "a1": a1,
        "u1": u1,
        "rho2": rho2,
        "p2": p2,
        "T2": T2,
        "u2": u2,
        "a2": a2,
        "M2_lab": u2 / a2,
        "rho_ratio": rho2 / rho1,
        "p_ratio": p2 / p1,
        "T_ratio": T2 / (p1 / rho1),
    }


def oblique_shock_beta(
    M1: float, theta: float, gamma: float = GAMMA, weak: bool = True, tol: float = 1e-12
) -> float:
    """Solve the theta-beta-M relation for the shock angle ``beta`` [rad].

    ``theta`` is the flow deflection [rad].  Returns the weak (``weak=True``) or
    strong solution; raises if the deflection exceeds the detachment angle.
    Bisection is used on the monotone branch so the result is reproducible.
    """
    if theta <= 0.0:
        raise ValueError("theta must be positive")

    def tan_theta(beta: float) -> float:
        s2 = math.sin(beta) ** 2
        return (
            2.0
            / math.tan(beta)
            * (M1 * M1 * s2 - 1.0)
            / (M1 * M1 * (gamma + math.cos(2.0 * beta)) + 2.0)
        )

    mu = math.asin(min(1.0, 1.0 / M1))
    target = math.tan(theta)
    # ascending crossing of tan(theta(beta)) measured from the Mach angle: that
    # is the weak solution.  The descending crossing (near beta = pi/2) is the
    # strong solution and is handled by _strong_beta.
    n = 4000
    hi = 0.5 * math.pi
    prev_b, prev_f = mu, tan_theta(mu)
    bracket = None
    for k in range(1, n + 1):
        b = mu + (hi - mu) * k / n
        f = tan_theta(b)
        if prev_f <= target <= f:
            bracket = (prev_b, b)
            break
        prev_b, prev_f = b, f
    if bracket is None:
        raise ValueError(
            f"no attached weak oblique shock for M1={M1}, theta={math.degrees(theta):.3f} deg"
        )
    a, b = bracket
    fa = tan_theta(a) - target
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = tan_theta(m) - target
        if fa * fm <= 0.0:
            b = m
        else:
            a, fa = m, fm
        if abs(b - a) < tol:
            break
    beta_weak = 0.5 * (a + b)
    return beta_weak if weak else _strong_beta(M1, theta, gamma)


def _strong_beta(M1: float, theta: float, gamma: float = GAMMA) -> float:
    """Strong-shock branch: bisection between the detachment angle and pi/2."""
    # detachment angle: maximise theta(beta)
    n = 2000
    mu = math.asin(min(1.0, 1.0 / M1))
    best = (0.0, mu)
    for k in range(n + 1):
        b = mu + (0.5 * math.pi - mu) * k / n
        s2 = math.sin(b) ** 2
        tt = (
            2.0 / math.tan(b) * (M1 * M1 * s2 - 1.0) / (M1 * M1 * (gamma + math.cos(2.0 * b)) + 2.0)
        )
        if tt > best[0]:
            best = (tt, b)
    theta_max, beta_det = best
    if theta > theta_max:
        raise ValueError("deflection beyond detachment: no attached shock")
    target = math.tan(theta)
    a, b = beta_det, 0.5 * math.pi - 1e-12

    def g(beta):
        s2 = math.sin(beta) ** 2
        return (
            2.0
            / math.tan(beta)
            * (M1 * M1 * s2 - 1.0)
            / (M1 * M1 * (gamma + math.cos(2.0 * beta)) + 2.0)
            - target
        )

    fa = g(a)
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = g(m)
        if fa * fm <= 0.0:
            b = m
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def oblique_shock(M1: float, theta: float, gamma: float = GAMMA, weak: bool = True) -> dict:
    """Oblique-shock solution: shock angle, normal Mach number and ratios.

    Ratios refer to the *normal* component (the tangential velocity and the
    total temperature are conserved); ``p_ratio`` and ``rho_ratio`` are the same
    as the corresponding normal shock at ``Mn1 = M1 sin(beta)``.
    """
    beta = oblique_shock_beta(M1, theta, gamma=gamma, weak=weak)
    Mn1 = M1 * math.sin(beta)
    jump = normal_shock(Mn1, gamma)
    return {
        "M1": M1,
        "theta": theta,
        "beta": beta,
        "beta_deg": math.degrees(beta),
        "Mn1": Mn1,
        "rho_ratio": jump["rho_ratio"],
        "p_ratio": jump["p_ratio"],
        "T_ratio": jump["T_ratio"],
        "M2": jump["M2"] / math.sin(beta - theta) if math.sin(beta - theta) > 0 else float("inf"),
        "Mn2": jump["M2"],
    }


def prandtl_meyer(M: float, gamma: float = GAMMA) -> float:
    """Prandtl-Meyer function ``nu(M)`` [rad]."""
    if M < 1.0:
        raise ValueError("Prandtl-Meyer function requires M >= 1")
    b = math.sqrt((gamma - 1.0) / (gamma + 1.0))
    return math.atan(b * math.sqrt(M * M - 1.0)) / b - math.atan(math.sqrt(M * M - 1.0))


def shock_tube_driver(
    Ms: float,
    rho1: float = 1.0,
    p1: float = 1.0,
    gamma: float = GAMMA,
    a4_over_a1: float = 4.0,
) -> dict:
    """Invert the shock-tube relations: driver state that yields shock Mach ``Ms``.

    The driven section is ``(rho1, p1)`` at rest.  The driver sound speed is
    fixed by ``a4_over_a1`` (a *choice*, not physics: it pins the one remaining
    degree of freedom of the inversion).  Returns the driver state
    ``(rho4, p4)``, the exact contact state and the post-shock state, so the
    resulting Riemann problem is a genuine, non-circular verification case:
    the shock Mach number is an *output* of the solver, never an input.
    """
    if Ms <= 1.0:
        raise ValueError("Ms must exceed 1")
    a1 = sound_speed(rho1, p1, gamma)
    a4 = a4_over_a1 * a1
    # induced velocity behind the shock (nondimensionalised with a1)
    u2 = 2.0 / (gamma + 1.0) * (Ms - 1.0 / Ms) * a1
    bracket = 1.0 - 0.5 * (gamma - 1.0) * u2 / a4
    if bracket <= 1e-6:
        raise ValueError(f"target Ms={Ms} unreachable with a4/a1={a4_over_a1}; raise a4_over_a1")
    p2 = p1 * (1.0 + 2.0 * gamma / (gamma + 1.0) * (Ms * Ms - 1.0))
    rho2 = rho1 * (gamma + 1.0) * Ms * Ms / ((gamma - 1.0) * Ms * Ms + 2.0)
    # state 3 (contact, contact side): p3 = p2, isentropic from 4
    p3_over_p4 = bracket ** (2.0 * gamma / (gamma - 1.0))
    T4_over_T1 = (a4 / a1) ** 2
    p4 = p2 / p3_over_p4
    rho4 = rho1 * (p4 / p1) / T4_over_T1
    rho3 = rho4 * p3_over_p4 ** (1.0 / gamma)
    return {
        "Ms": Ms,
        "W": Ms * a1,
        "a1": a1,
        "a4": a4,
        "rho4": rho4,
        "p4": p4,
        "u4": 0.0,
        "rho1": rho1,
        "p1": p1,
        "u1": 0.0,
        "rho2": rho2,
        "p2": p2,
        "u2": u2,
        "rho3": rho3,
        "p3": p2,
        "u3": u2,
        "p4_over_p1": p4 / p1,
        "rho4_over_rho1": rho4 / rho1,
    }
