"""Rankine-Hugoniot exact relations for compressible shock waves.

All formulas assume a calorically perfect gas with constant specific heat ratio γ.
Reference: Anderson, J.D., "Modern Compressible Flow", 3rd ed., McGraw-Hill, 2003.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalShockResult:
    """Post-shock state from normal shock relations."""
    M2: float          # downstream Mach number
    rho_ratio: float   # rho2/rho1
    p_ratio: float     # p2/p1
    T_ratio: float     # T2/T1
    p0_ratio: float    # p02/p01 (total pressure ratio)
    rho0_ratio: float  # rho02/rho01


@dataclass(frozen=True)
class ObliqueShockResult:
    """Post-shock state from oblique shock relations."""
    beta: float        # shock angle (rad)
    theta: float       # flow deflection angle (rad)
    M1n: float         # normal Mach upstream
    M2n: float         # normal Mach downstream
    M2: float          # downstream Mach number
    rho_ratio: float
    p_ratio: float
    T_ratio: float
    p0_ratio: float


@dataclass(frozen=True)
class IsentropicResult:
    """Isentropic flow relations."""
    T0_ratio: float    # T0/T
    p0_ratio: float    # p0/p
    rho0_ratio: float  # rho0/rho
    A_Astar: float     # area ratio A/A*


def normal_shock_relations(M1: float, gamma: float = 1.4) -> NormalShockResult:
    """Compute normal shock relations for given upstream Mach number.

    Parameters
    ----------
    M1 : float
        Upstream Mach number (must be > 1).
    gamma : float
        Specific heat ratio (default 1.4 for air).

    Returns
    -------
    NormalShockResult with all post-shock ratios.

    References
    ----------
    Anderson (2003), Eq. (3.57)-(3.62).
    """
    if M1 <= 1.0:
        raise ValueError(f"Upstream Mach number must be > 1 for a shock, got M1={M1}")

    M1_sq = M1 * M1

    # Downstream Mach number: Eq. (3.57)
    M2_sq = (1.0 + 0.5 * (gamma - 1.0) * M1_sq) / (gamma * M1_sq - 0.5 * (gamma - 1.0))
    M2 = math.sqrt(M2_sq)

    # Density ratio: Eq. (3.59)
    rho_ratio = ((gamma + 1.0) * M1_sq) / (2.0 + (gamma - 1.0) * M1_sq)

    # Pressure ratio: Eq. (3.58)
    p_ratio = 1.0 + (2.0 * gamma / (gamma + 1.0)) * (M1_sq - 1.0)

    # Temperature ratio: from ideal gas law
    T_ratio = p_ratio / rho_ratio

    # Total pressure ratio: Eq. (3.62)
    term1 = ((gamma + 1.0) * M1_sq) / (2.0 + (gamma - 1.0) * M1_sq)
    term2 = (gamma + 1.0) / (2.0 * gamma * M1_sq - (gamma - 1.0))
    p0_ratio = (term1 ** (gamma / (gamma - 1.0))) * (term2 ** (1.0 / (gamma - 1.0)))

    # Total density ratio (same as total pressure ratio for isentropic relation)
    rho0_ratio = p0_ratio / (T_ratio ** (1.0 / (gamma - 1.0)))
    # Actually: rho02/rho01 = p02/p01 since T02=T01 across shock
    rho0_ratio = p0_ratio

    return NormalShockResult(
        M2=M2,
        rho_ratio=rho_ratio,
        p_ratio=p_ratio,
        T_ratio=T_ratio,
        p0_ratio=p0_ratio,
        rho0_ratio=rho0_ratio,
    )


def oblique_shock_relations(
    M1: float, theta: float, gamma: float = 1.4
) -> ObliqueShockResult:
    """Compute oblique shock relations for given upstream Mach and deflection angle.

    Uses the theta-beta-Mach relation to find the shock angle beta.
    For a given M1 and theta, there may be two solutions (weak/strong shock);
    this function returns the weak shock solution.

    Parameters
    ----------
    M1 : float
        Upstream Mach number.
    theta : float
        Flow deflection angle (radians).
    gamma : float
        Specific heat ratio.

    Returns
    -------
    ObliqueShockResult with shock angle and post-shock state.

    References
    ----------
    Anderson (2003), Eq. (4.17) and subsequent.
    """
    if M1 <= 1.0:
        raise ValueError(f"Upstream Mach number must be > 1, got M1={M1}")

    # Theta-beta-Mach relation: Eq. (4.17)
    # tan(theta) = 2 * cot(beta) * (M1^2 * sin^2(beta) - 1) /
    #              (M1^2 * (gamma + cos(2*beta)) + 2)
    #
    # Solve for beta iteratively (weak shock branch)

    theta_deg = math.degrees(theta)

    # Maximum deflection angle check
    max_theta = _max_deflection_angle(M1, gamma)
    if theta_deg > max_theta:
        raise ValueError(
            f"Deflection angle {theta_deg:.2f}° exceeds maximum "
            f"{max_theta:.2f}° for M1={M1}"
        )

    # Iterative solution for beta (weak shock)
    beta = _solve_theta_beta_mach(M1, theta, gamma)

    # Normal component of upstream Mach
    M1n = M1 * math.sin(beta)

    # Apply normal shock relations to M1n
    ns = normal_shock_relations(M1n, gamma)

    # Normal component of downstream Mach
    M2n = ns.M2

    # Downstream Mach number
    M2 = M2n / math.sin(beta - theta)

    return ObliqueShockResult(
        beta=beta,
        theta=theta,
        M1n=M1n,
        M2n=M2n,
        M2=M2,
        rho_ratio=ns.rho_ratio,
        p_ratio=ns.p_ratio,
        T_ratio=ns.T_ratio,
        p0_ratio=ns.p0_ratio,
    )


def isentropic_relations(M: float, gamma: float = 1.4) -> IsentropicResult:
    """Compute isentropic flow relations for given Mach number.

    Parameters
    ----------
    M : float
        Mach number.
    gamma : float
        Specific heat ratio.

    Returns
    -------
    IsentropicResult with total-to-static ratios and area ratio.

    References
    ----------
    Anderson (2003), Eq. (3.28)-(3.31).
    """
    M_sq = M * M
    factor = 1.0 + 0.5 * (gamma - 1.0) * M_sq

    T0_ratio = factor
    p0_ratio = factor ** (gamma / (gamma - 1.0))
    rho0_ratio = factor ** (1.0 / (gamma - 1.0))

    # Area ratio A/A* (Eq. 3.31)
    if M > 0:
        A_Astar = (
            (1.0 / M)
            * ((2.0 / (gamma + 1.0)) * factor) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
        )
    else:
        A_Astar = float("inf")

    return IsentropicResult(
        T0_ratio=T0_ratio,
        p0_ratio=p0_ratio,
        rho0_ratio=rho0_ratio,
        A_Astar=A_Astar,
    )


def mach_angle(M: float) -> float:
    """Compute Mach angle (radians) for given Mach number.

    mu = arcsin(1/M)

    Parameters
    ----------
    M : float
        Mach number (must be >= 1).

    Returns
    -------
    Mach angle in radians.
    """
    if M < 1.0:
        raise ValueError(f"Mach number must be >= 1 for Mach angle, got M={M}")
    return math.asin(1.0 / M)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _max_deflection_angle(M1: float, gamma: float) -> float:
    """Compute maximum flow deflection angle for oblique shock (degrees)."""
    # Approximate formula; for exact, iterate over beta
    # Use numerical approach
    import numpy as np

    betas = np.linspace(math.asin(1.0 / M1) + 0.001, math.pi / 2 - 0.001, 1000)
    thetas = []
    for beta in betas:
        sin_b = math.sin(beta)
        cos_b = math.cos(beta)
        M1n_sq = M1 * M1 * sin_b * sin_b
        num = M1n_sq - 1.0
        den = M1 * M1 * (gamma + math.cos(2 * beta)) + 2.0
        if den > 0:
            tan_theta = 2.0 * cos_b * num / den
            thetas.append(math.degrees(math.atan(max(0, tan_theta))))
        else:
            thetas.append(0.0)
    return max(thetas) if thetas else 0.0


def _solve_theta_beta_mach(M1: float, theta: float, gamma: float) -> float:
    """Solve theta-beta-Mach relation for beta (weak shock branch).

    Uses Newton-Raphson iteration.
    """
    import numpy as np

    theta_rad = theta if abs(theta) < 0.1 else math.radians(theta) if theta > 1 else theta

    # Initial guess: Mach angle + small offset
    beta = math.asin(1.0 / M1) + 0.1

    for _ in range(100):
        sin_b = math.sin(beta)
        cos_b = math.cos(beta)
        sin_2b = math.sin(2 * beta)
        cos_2b = math.cos(2 * beta)

        M1_sq = M1 * M1
        M1n_sq = M1_sq * sin_b * sin_b

        # f(beta) = tan(theta) - 2*cot(beta)*(M1^2*sin^2(beta)-1)/(M1^2*(gamma+cos(2*beta))+2)
        num = M1n_sq - 1.0
        den = M1_sq * (gamma + cos_2b) + 2.0

        if den <= 0:
            beta += 0.01
            continue

        tan_theta_calc = 2.0 * cos_b * num / den
        f = math.tan(theta_rad) - tan_theta_calc

        if abs(f) < 1e-12:
            break

        # Numerical derivative
        db = 1e-8
        sin_bd = math.sin(beta + db)
        cos_bd = math.cos(beta + db)
        cos_2bd = math.cos(2 * (beta + db))
        M1n_sq_d = M1_sq * sin_bd * sin_bd
        num_d = M1n_sq_d - 1.0
        den_d = M1_sq * (gamma + cos_2bd) + 2.0
        tan_theta_d = 2.0 * cos_bd * num_d / den_d if den_d > 0 else tan_theta_calc
        df = -(tan_theta_d - tan_theta_calc) / db

        if abs(df) < 1e-15:
            beta += 0.01
            continue

        beta -= f / df

        # Clamp to valid range
        beta_min = math.asin(1.0 / M1) + 0.001
        beta_max = math.pi / 2 - 0.001
        beta = max(beta_min, min(beta_max, beta))

    return beta
