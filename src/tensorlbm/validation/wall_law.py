"""Law-of-the-wall, friction correlations and Blasius similarity solution.

This module is the *reference* half of the high-Reynolds wall-treatment
validation ladder (``tensorlbm.validation``).  It contains only closed-form or
numerically exact reference quantities so that solver output can be judged
against something independent of the solver:

* :func:`blasius_solution` / :func:`blasius_plate_cf` — RK4 shooting solution of
  ``f''' + f f''/2 = 0`` and the resulting ``C_f = 2 f''(0) / sqrt(Re_x)``.
* :func:`log_law_u_plus`, :func:`reichardt_u_plus`, :func:`spalding_y_plus`,
  :func:`spalding_u_plus` — the wall laws used by the LBM wall models.
* :func:`solve_friction_velocity` — the wall-function inverse problem
  (velocity at a wall-normal distance -> friction velocity), which is the
  operation a wall-modelled LES performs at every wall-adjacent cell.
* :func:`dean_channel_cf`, :func:`ittc57_cf`, :func:`schultz_grunow_cf`,
  :func:`schoenherr_cf` — reference skin-friction lines for channel and plate.

All functions are pure Python/``math`` (no torch) so they can be used inside
pytest without a backend and are numerically deterministic across platforms.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable

__all__ = [
    "B_LOG",
    "B_CHANNEL",
    "KAPPA",
    "BlasiusSolution",
    "blasius_plate_cf",
    "blasius_solution",
    "channel_cf_from_u_tau",
    "dean_channel_cf",
    "friction_velocity_from_cf",
    "ittc57_cf",
    "log_law_u_plus",
    "reichardt_u_plus",
    "schoenherr_cf",
    "schultz_grunow_cf",
    "solve_friction_velocity",
    "spalding_u_plus",
    "spalding_y_plus",
    "viscous_u_plus",
    "y_plus",
]

#: von Karman constant used throughout the validation ladder.
KAPPA = 0.41
#: Log-law intercept of the composite wall laws (Reichardt / Spalding fits).
B_LOG = 5.2
#: Intercept obtained by fitting the Re_tau = 180 channel DNS log region.
B_CHANNEL = 5.0


# ---------------------------------------------------------------------------
# Simple wall laws
# ---------------------------------------------------------------------------
def viscous_u_plus(y_plus_value: float) -> float:
    """Viscous sublayer: ``u+ = y+``."""
    return y_plus_value


def log_law_u_plus(
    y_plus_value: float, kappa: float = KAPPA, b: float = B_LOG
) -> float:
    """Classical log law ``u+ = ln(y+)/kappa + B`` (valid for ``y+ > 30``)."""
    if y_plus_value <= 0.0:
        raise ValueError("y+ must be positive for the log law")
    return math.log(y_plus_value) / kappa + b


def reichardt_u_plus(y_plus_value: float, kappa: float = KAPPA) -> float:
    """Reichardt (1951) composite law, valid from the wall to the log layer.

    ``u+ = ln(1 + 0.4 y+)/kappa
            + 7.8 (1 - exp(-y+/11) - (y+/11) exp(-y+/3))``
    """
    if y_plus_value < 0.0:
        raise ValueError("y+ must be non-negative")
    return math.log(1.0 + 0.4 * y_plus_value) / kappa + 7.8 * (
        1.0
        - math.exp(-y_plus_value / 11.0)
        - (y_plus_value / 11.0) * math.exp(-y_plus_value / 3.0)
    )


def spalding_y_plus(u_plus_value: float, kappa: float = KAPPA, b: float = B_LOG) -> float:
    """Spalding (1961) single formula, inverted form ``y+(u+)``."""
    if u_plus_value < 0.0:
        raise ValueError("u+ must be non-negative")
    ku = kappa * u_plus_value
    # exp(kappa u+) overflows near u+ ~ 1700; the law is monotone, so report
    # infinity instead of raising deep inside a bisection bracket.
    leading = math.exp(ku) if ku < 700.0 else math.inf
    return u_plus_value + math.exp(-kappa * b) * (
        leading - 1.0 - ku - ku**2 / 2.0 - ku**3 / 6.0
    )


def spalding_u_plus(
    y_plus_value: float,
    kappa: float = KAPPA,
    b: float = B_LOG,
    *,
    tolerance: float = 1.0e-12,
    maximum_iterations: int = 200,
) -> float:
    """Invert Spalding's law: ``y+ -> u+`` by bisection (monotone in ``u+``)."""
    if y_plus_value < 0.0:
        raise ValueError("y+ must be non-negative")
    if y_plus_value == 0.0:
        return 0.0
    low, high = 0.0, max(2.0, y_plus_value)
    while spalding_y_plus(high, kappa, b) < y_plus_value:
        high *= 2.0
        if high > 1.0e8:  # pragma: no cover - defensive
            raise RuntimeError("Spalding inversion failed to bracket the root")
    for _ in range(maximum_iterations):
        mid = 0.5 * (low + high)
        if spalding_y_plus(mid, kappa, b) < y_plus_value:
            low = mid
        else:
            high = mid
        if high - low < tolerance * max(1.0, high):
            break
    return 0.5 * (low + high)


_WALL_LAWS: dict[str, Callable[[float], float]] = {
    "spalding": spalding_u_plus,
    "reichardt": reichardt_u_plus,
    "log": log_law_u_plus,
}


def y_plus(
    wall_distance: float, friction_velocity: float, nu: float
) -> float:
    """``y+ = y u_tau / nu``."""
    if nu <= 0.0:
        raise ValueError("nu must be positive")
    return wall_distance * friction_velocity / nu


def solve_friction_velocity(
    velocity: float,
    wall_distance: float,
    nu: float,
    *,
    law: str = "spalding",
    kappa: float = KAPPA,
    b: float = B_LOG,
    low: float = 1.0e-12,
    high: float = 1.0,
    tolerance: float = 1.0e-14,
    maximum_iterations: int = 300,
) -> tuple[float, float, float]:
    """Solve the wall-function inverse problem for ``u_tau``.

    Finds ``u_tau`` such that ``velocity = u_tau * law(y u_tau / nu)``.
    Returns ``(u_tau, y_plus, u_plus)``.  This is exactly the closure
    operation a wall-modelled LES applies in the wall-adjacent cell.

    A log-law inversion has no solution for very small ``y+`` (below the log
    layer) — in that case the bracket collapses onto the lower bound and the
    returned ``u_tau`` is flagged by an extremely small ``y_plus``.
    """
    if velocity <= 0.0:
        raise ValueError("velocity must be positive")
    if wall_distance <= 0.0:
        raise ValueError("wall distance must be positive")
    if law not in _WALL_LAWS:
        raise ValueError(f"unknown wall law {law!r}; expected one of {sorted(_WALL_LAWS)}")
    profile = _WALL_LAWS[law]

    def residual(u_tau: float) -> float:
        u_plus_value = profile(max(y_plus(wall_distance, u_tau, nu), 1.0e-300))
        return u_tau * u_plus_value - velocity

    lo, hi = low, high
    f_lo, f_hi = residual(lo), residual(hi)
    if f_lo > 0.0:
        raise ValueError("lower bracket already exceeds the target velocity")
    while f_hi < 0.0:
        hi *= 2.0
        if hi > 1.0e3:  # pragma: no cover - defensive
            raise RuntimeError("failed to bracket u_tau")
        f_hi = residual(hi)
    for _ in range(maximum_iterations):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if f_mid < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tolerance * max(1.0, hi):
            break
    u_tau_value = 0.5 * (lo + hi)
    return (
        u_tau_value,
        y_plus(wall_distance, u_tau_value, nu),
        velocity / u_tau_value,
    )


# ---------------------------------------------------------------------------
# Skin-friction correlations
# ---------------------------------------------------------------------------
def friction_velocity_from_cf(cf: float, reference_velocity: float) -> float:
    """``u_tau = U sqrt(C_f/2)``."""
    if cf <= 0.0:
        raise ValueError("C_f must be positive")
    return reference_velocity * math.sqrt(cf / 2.0)


def channel_cf_from_u_tau(u_tau: float, bulk_velocity: float) -> float:
    """Channel friction coefficient ``C_f = 2 (u_tau/U_b)^2``."""
    if bulk_velocity <= 0.0:
        raise ValueError("bulk velocity must be positive")
    return 2.0 * (u_tau / bulk_velocity) ** 2


def dean_channel_cf(reynolds_bulk: float) -> float:
    """Dean (1978) channel friction correlation ``C_f = 0.073 Re_b^-0.25``."""
    if reynolds_bulk <= 0.0:
        raise ValueError("Reynolds number must be positive")
    return 0.073 * reynolds_bulk**-0.25


def ittc57_cf(reynolds: float) -> float:
    """ITTC-1957 friction line ``C_f = 0.075 / (log10 Re - 2)^2``."""
    if reynolds <= 100.0:
        raise ValueError("ITTC-1957 requires Re > 100")
    return 0.075 / (math.log10(reynolds) - 2.0) ** 2


def schultz_grunow_cf(reynolds: float) -> float:
    """Schultz-Grunow friction line (smooth flat plate)."""
    if reynolds <= 100.0:
        raise ValueError("Schultz-Grunow requires Re > 100")
    return 0.427 / (math.log10(reynolds) - 0.407) ** 2.64


def schoenherr_cf(
    reynolds: float, *, tolerance: float = 1.0e-13, maximum_iterations: int = 100
) -> float:
    """Schoenherr line, solved implicitly from ``0.242/sqrt(Cf) = log10(Re Cf)``."""
    if reynolds <= 0.0:
        raise ValueError("Reynolds number must be positive")
    cf = 0.455 / (math.log10(reynolds)) ** 2.58  # initial guess (Prandtl-Schlichting)
    for _ in range(maximum_iterations):
        residual = 0.242 / math.sqrt(cf) - math.log10(reynolds * cf)
        # d(residual)/d(cf) — Newton with a safe fallback on the monotone branch
        derivative = -0.121 * cf**-1.5 - 1.0 / (cf * math.log(10.0))
        step = residual / derivative
        cf_next = cf - step
        if cf_next <= 0.0:
            cf_next = 0.5 * cf
        if abs(cf_next - cf) < tolerance * cf:
            cf = cf_next
            break
        cf = cf_next
    return cf


# ---------------------------------------------------------------------------
# Blasius similarity solution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BlasiusSolution:
    """Numerical Blasius solution (RK4 shooting)."""

    eta: tuple[float, ...]
    f: tuple[float, ...]
    f_prime: tuple[float, ...]
    f_double_prime: tuple[float, ...]
    wall_derivative: float
    eta_step: float

    @property
    def cf_coefficient(self) -> float:
        """``C_f = 2 f''(0)/sqrt(Re_x)`` coefficient."""
        return 2.0 * self.wall_derivative

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cf_coefficient"] = self.cf_coefficient
        return payload

    def u_over_u_edge(self, eta_value: float) -> float:
        """Interpolate ``f'(eta) = u/U_e`` at an arbitrary similarity coordinate."""
        if eta_value <= self.eta[0]:
            return 0.0
        if eta_value >= self.eta[-1]:
            return 1.0
        for index in range(1, len(self.eta)):
            if self.eta[index] >= eta_value:
                eta0, eta1 = self.eta[index - 1], self.eta[index]
                f0, f1 = self.f_prime[index - 1], self.f_prime[index]
                weight = (eta_value - eta0) / (eta1 - eta0)
                return f0 + weight * (f1 - f0)
        return 1.0


def _blasius_rhs(y1: float, y2: float, y3: float) -> tuple[float, float, float]:
    return y2, y3, -0.5 * y1 * y3


def _integrate_blasius(initial_slope: float, eta_max: float, step: float) -> tuple[float, list]:
    count = int(round(eta_max / step)) + 1
    y1, y2, y3 = 0.0, 0.0, initial_slope
    f_values = [y1]
    fp_values = [y2]
    fpp_values = [y3]
    for _ in range(count - 1):
        k1 = _blasius_rhs(y1, y2, y3)
        k2 = _blasius_rhs(
            y1 + 0.5 * step * k1[0], y2 + 0.5 * step * k1[1], y3 + 0.5 * step * k1[2]
        )
        k3 = _blasius_rhs(
            y1 + 0.5 * step * k2[0], y2 + 0.5 * step * k2[1], y3 + 0.5 * step * k2[2]
        )
        k4 = _blasius_rhs(
            y1 + step * k3[0], y2 + step * k3[1], y3 + step * k3[2]
        )
        y1 += step / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        y2 += step / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        y3 += step / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        f_values.append(y1)
        fp_values.append(y2)
        fpp_values.append(y3)
    return y2, [f_values, fp_values, fpp_values]


def blasius_solution(
    eta_max: float = 10.0,
    step: float = 0.005,
    *,
    tolerance: float = 1.0e-14,
    maximum_iterations: int = 100,
) -> BlasiusSolution:
    """Solve the Blasius equation by RK4 integration with shooting on ``f''(0)``.

    Default ``step = 0.005`` reproduces ``f''(0) = 0.3320573362151963`` to
    better than 1e-10, matching the literature value to ten digits.
    """
    if eta_max <= 0.0 or step <= 0.0:
        raise ValueError("eta_max and step must be positive")
    # Bracket the wall shear: Blasius f''(0) ~ 0.3321.
    low, high = 0.0, 1.0
    low_edge, _ = _integrate_blasius(low, eta_max, step)
    high_edge, _ = _integrate_blasius(high, eta_max, step)
    if low_edge > 1.0 or high_edge < 1.0:
        raise RuntimeError("failed to bracket the Blasius wall shear")
    for _ in range(maximum_iterations):
        mid = 0.5 * (low + high)
        edge, _ = _integrate_blasius(mid, eta_max, step)
        if edge < 1.0:
            low = mid
        else:
            high = mid
        if high - low < tolerance:
            break
    slope = 0.5 * (low + high)
    edge_velocity, series = _integrate_blasius(slope, eta_max, step)
    if abs(edge_velocity - 1.0) > 1.0e-6:  # pragma: no cover - defensive
        raise RuntimeError("Blasius shooting did not converge to f'(inf) = 1")
    count = len(series[0])
    eta = tuple(index * step for index in range(count))
    return BlasiusSolution(
        eta=eta,
        f=tuple(series[0]),
        f_prime=tuple(series[1]),
        f_double_prime=tuple(series[2]),
        wall_derivative=slope,
        eta_step=step,
    )


def blasius_plate_cf(re_x: float, *, wall_derivative: float = 0.3320573362151963) -> float:
    """Local Blasius skin friction ``C_f = 2 f''(0) / sqrt(Re_x)``."""
    if re_x <= 0.0:
        raise ValueError("Re_x must be positive")
    return 2.0 * wall_derivative / math.sqrt(re_x)