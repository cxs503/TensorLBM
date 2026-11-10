"""Atmospheric boundary conditions for an ice/water surface.

A single linearised surface energy balance is used everywhere in the package:

    q_s(T_s) = h_c (T_s - T_air) + h_r (T_s - T_sky) - (1 - alpha) Q_sw
             = h_tot (T_s - T_eff)

with

* ``h_c = 4.2 + 4.0 U_10``  (Maykut 1986 bulk sensible transfer, W m^-2 K^-1)
* ``h_r = 4 eps sigma T_air_K^3``  (linearised downward longwave, W m^-2 K^-1)
* ``T_sky = T_air - dT_sky``  (effective radiative sky temperature; the offset
  bundles clear-sky emissivity and cloudiness into one knob -- documented as a
  closure, see the node report, section "limitations")
* ``Q_sw`` shortwave radiation absorbed at the surface (default 0: polar night).

Positive ``q_s`` means heat leaves the water/ice towards the atmosphere.  The
condensed form ``(T_s - T_eff, h_tot)`` is what the column model and the lookup
table consume, so the same boundary condition is used by every fidelity level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import DEFAULT_PROPERTIES, IceWaterProperties


@dataclass(frozen=True)
class AtmosphericForcing:
    """Air-side forcing for one refreezing scenario."""

    t_air: float = -20.0            # degC, 2 m air temperature
    u_wind: float = 5.0             # m/s, 10 m wind speed
    q_shortwave: float = 0.0        # W/m^2 absorbed shortwave (winter: 0)
    sky_temperature_offset: float = 20.0  # K, T_sky = T_air - offset
    albedo: float | None = None     # None -> surface-dependent default
    snow_depth: float = 0.0         # m, thin insulating snow cover (0 = bare ice)
    snow_conductivity: float = 0.31  # W/(m K), Sturm et al. (1997) range 0.1-0.4

    def albedo_for(self, props: IceWaterProperties = DEFAULT_PROPERTIES) -> float:
        return props.albedo_water if self.albedo is None else float(self.albedo)


def sensible_heat_transfer_coefficient(u_wind: float) -> float:
    """``h_c = 4.2 + 4.0 U_10`` W m^-2 K^-1 (Maykut 1986)."""
    return 4.2 + 4.0 * max(float(u_wind), 0.0)


def longwave_linear_coefficient(
    t_air: float, props: IceWaterProperties = DEFAULT_PROPERTIES
) -> float:
    """Linearised longwave coefficient ``4 eps sigma T_air^3``, W m^-2 K^-1."""
    t_k = float(t_air) + 273.15
    return 4.0 * props.emissivity * props.stefan_boltzmann * t_k ** 3


def effective_air_temperature(
    forcing: AtmosphericForcing,
    props: IceWaterProperties = DEFAULT_PROPERTIES,
) -> tuple[float, float]:
    """Return ``(T_eff, h_eff)`` of the condensed surface balance.

    ``q_s = h_eff (T_s - T_eff)``.  Longwave is linearised about ``T_air``,
    the sky is represented by ``T_air - dT_sky`` and an optional snow layer
    adds its conductance in series with the air-side coefficient.
    """
    h_c = sensible_heat_transfer_coefficient(forcing.u_wind)
    h_r = longwave_linear_coefficient(forcing.t_air, props)
    h_tot = h_c + h_r
    t_sky = forcing.t_air - forcing.sky_temperature_offset
    q_sw = max(forcing.q_shortwave, 0.0) * (1.0 - forcing.albedo_for(props))
    t_eff = (h_c * forcing.t_air + h_r * t_sky - q_sw) / h_tot
    if forcing.snow_depth > 0.0:
        # snow conductance in series with the air-side coefficient: the same
        # T_eff, an effective (reduced) transfer coefficient h_eff
        r_snow = forcing.snow_depth / forcing.snow_conductivity
        h_eff = 1.0 / (1.0 / h_tot + r_snow)
        return t_eff, h_eff
    return t_eff, h_tot


def surface_heat_flux(
    t_surface: float,
    forcing: AtmosphericForcing,
    props: IceWaterProperties = DEFAULT_PROPERTIES,
) -> float:
    """Full (unlinearised longwave + linear sensible) upward surface flux."""
    t_eff, h_tot = effective_air_temperature(forcing, props)
    return h_tot * (float(t_surface) - t_eff)


def wind_stress(u_wind: float, props: IceWaterProperties = DEFAULT_PROPERTIES) -> float:
    """Neutral 10 m wind stress ``tau = rho_a C_D U^2``, Pa."""
    return props.rho_air * props.drag_air * max(float(u_wind), 0.0) ** 2


def drag_law_current(
    tau: float, props: IceWaterProperties = DEFAULT_PROPERTIES
) -> float:
    """Steady surface drift from a quadratic drag balance, m/s.

    ``tau = rho_w C_dw U^2`` for an open-water strip whose momentum sink is the
    water-side form drag; for ``C_dw = 1.5e-3`` this reproduces the classical
    wind-drift ratio ``U_s / U_10 ~ 0.03`` at 5-15 m/s.
    """
    return math.sqrt(max(float(tau), 0.0) / (props.rho_w * props.drag_water))


def ocean_heat_flux(
    u_surface: float,
    superheat: float,
    props: IceWaterProperties = DEFAULT_PROPERTIES,
) -> float:
    """Bulk water-side heat flux to the ice, W/m^2.

    ``q_o = rho_w cp_w C_H U_s (T_w - T_f)``.  Positive ``q_o`` warms the ice
    underside (slows refreezing).  ``superheat`` is the mixed-layer temperature
    excess over the freezing point.
    """
    return (
        props.rho_w
        * props.cp_w
        * props.stanton_water
        * max(float(u_surface), 0.0)
        * max(float(superheat), 0.0)
    )


__all__ = [
    "AtmosphericForcing",
    "sensible_heat_transfer_coefficient",
    "longwave_linear_coefficient",
    "effective_air_temperature",
    "surface_heat_flux",
    "wind_stress",
    "drag_law_current",
    "ocean_heat_flux",
]