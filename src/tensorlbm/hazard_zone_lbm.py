"""Hazard zone analysis: LFL contour for alternative fuel gas dispersion.

Computes the Lower Flammability Limit (LFL) contour for gas cloud hazard
assessment using Gaussian plume and puff models (Pasquill-Gifford).

This is the engineering module imported by the verification engine.

Physical model:
  - Gaussian plume: C(x,y,z) = Q/(2πUσ_yσ_z) exp(-y²/2σ_y²) [exp terms for z]
  - Pasquill-Gifford dispersion coefficients σ_y(x), σ_z(x) by stability class
  - LFL database for common alternative fuels (H₂, CH₄, NH₃, CH₃OH, C₃H₈)

References:
  - Pasquill F. (1974). Atmospheric Diffusion. 2nd ed. Ellis Horwood.
  - API 521 (2020). "Pressure-relieving and depressuring systems."
  - NFPA 497 (2021). "Classification of flammable liquids, gases, or vapors."
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ── LFL database ─────────────────────────────────────────────────────────────

LFL_DATABASE = {
    "H2": {"lfl": 0.04, "name": "Hydrogen", "M": 2.0, "rho_gas": 0.084},
    "CH4": {"lfl": 0.05, "name": "Methane/LNG", "M": 16.0, "rho_gas": 0.657},
    "CH3OH": {"lfl": 0.06, "name": "Methanol", "M": 32.0, "rho_gas": 1.33},
    "NH3": {"lfl": 0.15, "name": "Ammonia", "M": 17.0, "rho_gas": 0.73},
    "C3H8": {"lfl": 0.021, "name": "Propane", "M": 44.0, "rho_gas": 1.88},
}


@dataclass
class ReleaseParams:
    """Gas release parameters for hazard zone calculation."""
    fuel: str = "CH4"
    mass_rate: float = 1.0  # kg/s
    source_height: float = 1.0  # m
    wind_speed: float = 5.0  # m/s
    ambient_temp: float = 298.0  # K
    ambient_pressure: float = 101325.0  # Pa
    stability_class: str = "D"  # Pasquill-Gifford stability


@dataclass
class HazardZoneResult:
    """Result of hazard zone calculation."""
    fuel: str
    lfl: float
    wind_speed: float
    mass_rate: float
    contour_length: float
    contour_width: float
    contour_area: float
    max_downwind_distance: float


# ── Pasquill-Gifford dispersion coefficients ─────────────────────────────────

def pasquill_gifford_dispersion(
    x: float,
    stability: str = "D",
) -> tuple[float, float]:
    """Pasquill-Gifford dispersion coefficients σ_y(x), σ_z(x).

    Empirical power-law fits: σ = a * x^b
    Coefficients from Turner (1970) / Pasquill-Gifford curves.
    """
    coefficients = {
        "A": {"ay": 0.22, "by": 0.95, "az": 0.20, "bz": 1.0},
        "B": {"ay": 0.16, "by": 0.95, "az": 0.12, "bz": 0.95},
        "C": {"ay": 0.11, "by": 0.92, "az": 0.08, "bz": 0.90},
        "D": {"ay": 0.08, "by": 0.90, "az": 0.06, "bz": 0.85},
        "E": {"ay": 0.06, "by": 0.88, "az": 0.04, "bz": 0.82},
        "F": {"ay": 0.04, "by": 0.86, "az": 0.03, "bz": 0.80},
    }

    if stability not in coefficients:
        stability = "D"

    c = coefficients[stability]
    sigma_y = c["ay"] * x ** c["by"]
    sigma_z = c["az"] * x ** c["bz"]

    return max(sigma_y, 1.0), max(sigma_z, 1.0)


def gaussian_plume_concentration(
    x: float,
    y: float,
    z: float,
    Q: float,
    U: float,
    H: float,
    stability: str = "D",
) -> float:
    """Steady-state Gaussian plume concentration (ground-level reflection).

    C(x,y,z) = Q/(2πUσ_yσ_z) × exp(-y²/(2σ_y²))
               × [exp(-(z-H)²/(2σ_z²)) + exp(-(z+H)²/(2σ_z²))]
    """
    if x <= 0:
        return 0.0

    sigma_y, sigma_z = pasquill_gifford_dispersion(x, stability)

    prefactor = Q / (2.0 * math.pi * U * sigma_y * sigma_z)
    lateral = math.exp(-0.5 * (y / sigma_y) ** 2)
    vertical = (
        math.exp(-0.5 * ((z - H) / sigma_z) ** 2) +
        math.exp(-0.5 * ((z + H) / sigma_z) ** 2)
    )

    return prefactor * lateral * vertical


def find_lfl_contour_downwind(
    Q: float,
    U: float,
    H: float,
    LFL: float,
    stability: str = "D",
    x_max: float = 10000.0,
    rho_air: float = 1.2,
) -> float:
    """Find downwind distance where centerline concentration = LFL.

    Uses bisection to find x where C(x, 0, 0) = LFL × ρ_air.
    """
    C_target = LFL * rho_air

    # Check if source concentration exceeds LFL
    C_source = gaussian_plume_concentration(1.0, 0, 0, Q, U, H, stability)
    if C_source < C_target:
        return 0.0

    x_low, x_high = 1.0, x_max

    for _ in range(100):
        x_mid = 0.5 * (x_low + x_high)
        C_mid = gaussian_plume_concentration(x_mid, 0, 0, Q, U, H, stability)

        if C_mid > C_target:
            x_low = x_mid
        else:
            x_high = x_mid

        if abs(x_high - x_low) < 0.1:
            break

    return 0.5 * (x_low + x_high)


def compute_hazard_zone(
    params: ReleaseParams,
    nx: int = 100,
    ny: int = 50,
    x_range: float = 5000.0,
    y_range: float = 500.0,
) -> HazardZoneResult:
    """Compute LFL hazard zone for a continuous release.

    Returns contour dimensions (length, width, area) of the region
    where concentration exceeds LFL.
    """
    fuel_data = LFL_DATABASE.get(params.fuel, LFL_DATABASE["CH4"])
    lfl = fuel_data["lfl"]
    rho_air = 1.2

    max_downwind = find_lfl_contour_downwind(
        params.mass_rate, params.wind_speed, params.source_height,
        lfl, params.stability_class, rho_air=rho_air
    )

    if max_downwind > 0:
        x_half = max_downwind * 0.5
        sigma_y, _ = pasquill_gifford_dispersion(x_half, params.stability_class)
        contour_width = 2.0 * sigma_y * 2.0  # ~2 sigma on each side
    else:
        contour_width = 0.0

    contour_area = max_downwind * contour_width * 0.5  # triangular approximation

    return HazardZoneResult(
        fuel=params.fuel,
        lfl=lfl,
        wind_speed=params.wind_speed,
        mass_rate=params.mass_rate,
        contour_length=max_downwind,
        contour_width=contour_width,
        contour_area=contour_area,
        max_downwind_distance=max_downwind,
    )


__all__ = [
    "LFL_DATABASE",
    "ReleaseParams",
    "HazardZoneResult",
    "pasquill_gifford_dispersion",
    "gaussian_plume_concentration",
    "find_lfl_contour_downwind",
    "compute_hazard_zone",
]
