"""Wall laws and friction-velocity inversion for wall-modelled LES.

The two laws used here are the classical equilibrium closure (von Karman log
law) and Spalding's unified law, which is valid from the viscous sublayer
through the buffer layer into the log region and is therefore the safe choice
for coarse wall-modelled LES grids where the first off-wall node may sit at
``y+ ~ 20-40``.

Conventions
-----------
``kappa`` and ``B`` follow the mission/predeclared pair ``kappa = 0.41``,
``B = 5.2``.  Spalding's damping constant ``A`` is by default derived from the
log-law intercept, ``A = exp(-kappa * B)``, so that the composite law has the
same log-region asymptote as the equilibrium law.  The historical Spalding
constant ``A = 0.1108`` (kappa = 0.4, B = 5.5) is available explicitly.

All functions are vectorised over torch tensors and are pure (no state).
"""

from __future__ import annotations

import math

import torch

KAPPA_DEFAULT = 0.41
B_DEFAULT = 5.2
VAN_DRIEST_A_PLUS = 26.0  # calibrated: gives B_eff = 5.24 for kappa = 0.41

__all__ = [
    "law_u_plus_scalar",
    "solve_u_tau_scalar",
    "KAPPA_DEFAULT",
    "B_DEFAULT",
    "VAN_DRIEST_A_PLUS",
    "log_law_u_plus",
    "spalding_A_from_B",
    "spalding_y_plus",
    "spalding_u_plus",
    "reichardt_u_plus",
    "law_u_plus",
    "solve_u_tau",
]


def log_law_u_plus(y_plus, kappa: float = KAPPA_DEFAULT, b: float = B_DEFAULT):
    """``u+ = (1/kappa) ln(y+) + B`` (undefined for ``y+ <= 0``)."""
    y_plus = torch.as_tensor(y_plus, dtype=torch.float64)
    return torch.log(y_plus.clamp_min(1.0e-12)) / kappa + b


def spalding_A_from_B(kappa: float = KAPPA_DEFAULT, b: float = B_DEFAULT) -> float:
    """Spalding damping constant consistent with the log-law intercept ``B``."""
    return math.exp(-kappa * b)


def spalding_y_plus(u_plus, kappa: float = KAPPA_DEFAULT, a: float | None = None):
    """Spalding (1961) unified law written as ``y+ = f(u+)``.

    ``y+ = u+ + A [ exp(kappa u+) - 1 - kappa u+ - (kappa u+)^2/2
    - (kappa u+)^3/6 ]``
    """
    u_plus = torch.as_tensor(u_plus, dtype=torch.float64).clamp(min=0.0, max=40.0)
    if a is None:
        a = spalding_A_from_B(kappa, B_DEFAULT)
    ku = kappa * u_plus
    return u_plus + a * (torch.exp(ku) - 1.0 - ku - 0.5 * ku.square() - ku.pow(3) / 6.0)


def spalding_u_plus(y_plus, kappa: float = KAPPA_DEFAULT, a: float | None = None):
    """Invert Spalding's monotone law ``y+ -> u+`` by vectorised bisection."""
    target = torch.as_tensor(y_plus, dtype=torch.float64).clamp_min(0.0)
    low = torch.zeros_like(target)
    high = torch.full_like(target, 60.0)
    for _ in range(60):
        mid = 0.5 * (low + high)
        below = spalding_y_plus(mid, kappa=kappa, a=a) < target
        low = torch.where(below, mid, low)
        high = torch.where(below, high, mid)
    return 0.5 * (low + high)


def reichardt_u_plus(y_plus, kappa: float = KAPPA_DEFAULT):
    """Reichardt (1951) composite law (analytic, no iteration).

    ``u+ = ln(1 + kappa y+)/kappa
    + 7.8 (1 - exp(-y+/11) - (y+/11) exp(-y+/3))``
    """
    y_plus = torch.as_tensor(y_plus, dtype=torch.float64).clamp_min(0.0)
    return torch.log1p(kappa * y_plus) / kappa + 7.8 * (
        1.0 - torch.exp(-y_plus / 11.0) - (y_plus / 11.0) * torch.exp(-y_plus / 3.0)
    )


def law_u_plus(y_plus, law: str = "spalding", kappa: float = KAPPA_DEFAULT, b: float = B_DEFAULT):
    """Dispatch to a named wall law."""
    key = law.lower()
    if key == "log":
        return log_law_u_plus(y_plus, kappa=kappa, b=b)
    if key == "spalding":
        return spalding_u_plus(y_plus, kappa=kappa, a=spalding_A_from_B(kappa, b))
    if key == "reichardt":
        return reichardt_u_plus(y_plus, kappa=kappa)
    msg = f"unknown wall law {law!r}; expected 'log', 'spalding' or 'reichardt'"
    raise ValueError(msg)


def solve_u_tau(
    u_tangential,
    y: float,
    nu: float,
    *,
    law: str = "spalding",
    kappa: float = KAPPA_DEFAULT,
    b: float = B_DEFAULT,
    iterations: int = 80,
):
    """Invert ``u(y) = u_tau * f(y u_tau / nu)`` for the friction velocity.

    Vectorised bisection on ``u_tau`` in ``[u_visc, u_high]`` where the two
    brackets are the viscous-sublayer estimate ``sqrt(u nu / y)`` and twice
    that value, which straddle the solution for every ``y+`` in
    ``[1e-3, 1e7]`` because ``u_tau f(y u_tau/nu)`` is monotone increasing in
    ``u_tau``.

    Parameters
    ----------
    u_tangential:
        Tangential velocity magnitude at the sampling location (lattice units).
    y:
        Wall-normal distance of the sampling location (lattice units).
    nu:
        Kinematic viscosity used by the wall model (lattice units).
    """
    u = torch.as_tensor(u_tangential, dtype=torch.float64).abs()
    if y <= 0.0 or nu <= 0.0:
        msg = "y and nu must be positive"
        raise ValueError(msg)
    # Viscous-sublayer bracket: u+ = y+  =>  u_tau = sqrt(u nu / y).
    low = torch.sqrt(torch.clamp(u * nu / y, min=1.0e-24))
    high = 8.0 * low + 1.0e-12

    def residual(u_tau: torch.Tensor) -> torch.Tensor:
        y_plus = y * u_tau / nu
        return u_tau * law_u_plus(y_plus, law=law, kappa=kappa, b=b) - u

    # Expand the upper bracket until it brackets the (monotone) root.
    for _ in range(40):
        need = residual(high) < 0.0
        if not bool(need.any()):
            break
        high = torch.where(need, high * 4.0, high)
    lo = low
    hi = high
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        below = residual(mid) < 0.0
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    return 0.5 * (lo + hi)

# ---------------------------------------------------------------------------
# pure-Python (scalar) fast path
# ---------------------------------------------------------------------------
def law_u_plus_scalar(
    y_plus: float, law: str = "spalding", kappa: float = KAPPA_DEFAULT, b: float = B_DEFAULT
) -> float:
    """Scalar wall law; Spalding is inverted by Newton (no nested bisection)."""
    if y_plus <= 0.0:
        return 0.0
    key = law.lower()
    if key == "log":
        return math.log(y_plus) / kappa + b
    if key == "reichardt":
        return math.log1p(kappa * y_plus) / kappa + 7.8 * (
            1.0 - math.exp(-y_plus / 11.0) - (y_plus / 11.0) * math.exp(-y_plus / 3.0)
        )
    if key == "spalding":
        a = math.exp(-kappa * b)
        # Newton on y+(u+) = target, started from the log law.
        u = math.log(y_plus) / kappa + b if y_plus > 11.0 else y_plus
        for _ in range(12):
            ku = kappa * u
            f = u + a * (math.exp(ku) - 1.0 - ku - 0.5 * ku * ku - ku**3 / 6.0) - y_plus
            df = 1.0 + a * kappa * (math.exp(ku) - 1.0 - ku - 0.5 * ku * ku)
            if df <= 0.0:
                break
            step = f / df
            u -= step
            if u < 0.0:
                u = 0.5 * y_plus
            if abs(step) < 1.0e-12 * max(1.0, abs(u)):
                break
        return u
    msg = f"unknown wall law {law!r}"
    raise ValueError(msg)


def solve_u_tau_scalar(
    u: float,
    y: float,
    nu: float,
    *,
    law: str = "spalding",
    kappa: float = KAPPA_DEFAULT,
    b: float = B_DEFAULT,
    guess: float | None = None,
    iterations: int = 60,
) -> float:
    """Scalar ``u_tau`` inversion (safeguarded Newton, ~10x faster than torch)."""
    u = abs(float(u))
    if u == 0.0:
        return 0.0
    lo = 0.25 * math.sqrt(u * nu / y)
    hi = 8.0 * lo
    while hi * law_u_plus_scalar(y * hi / nu, law, kappa, b) - u < 0.0:
        hi *= 4.0
    if hi > 1.0e6:
        hi = 1.0e6
    x = guess if (guess is not None and lo < guess < hi) else 0.5 * (lo + hi)
    for _ in range(iterations):
        fx = x * law_u_plus_scalar(y * x / nu, law, kappa, b) - u
        if fx > 0.0:
            hi = x
        else:
            lo = x
        # derivative of u_tau*f(y u_tau/nu)
        h = 1.0e-6 * x + 1.0e-14
        f2 = (x + h) * law_u_plus_scalar(y * (x + h) / nu, law, kappa, b) - u
        df = (f2 - fx) / h
        x_new = x - fx / df if df > 0.0 else 0.5 * (lo + hi)
        if not (lo < x_new < hi):
            x_new = 0.5 * (lo + hi)
        if abs(x_new - x) < 1.0e-13 * max(1.0, x):
            x = x_new
            break
        x = x_new
    return x
