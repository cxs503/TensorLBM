"""Physical constants and bulk property tables for ice-channel refreezing.

All quantities are SI (m, s, K, W, J, kg); temperatures are degrees Celsius
unless the attribute name ends in ``_K``.

Provenance of the bulk constants
--------------------------------
* Ice / water bulk properties (rho, k, cp, latent heat): Timco & Weeks (2010),
  *A review of the engineering properties of sea ice*, Cold Regions Science and
  Technology 60, 107-129; Weeks & Ackley (1986), *The growth, structure and
  properties of sea ice*, CRREL Monograph 82-1.
* Air-ice sensible heat transfer ``h_c = 4.2 + 4.0 U_10``: Maykut (1986)
  linearisation of the bulk aerodynamic formula (W m^-2 K^-1, U in m/s).
* Air-side drag ``C_D = 1.8e-3`` and water-side Stanton number
  ``C_H = 1.2e-3``: standard bulk formulae (Large & Pond 1981) in the
  neutral-stability limit.

The numerical values are recorded in a single frozen dataclass so that every
derived quantity (volumetric latent heat, diffusivities) is computed once and
cannot drift between modules.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IceWaterProperties:
    """Bulk thermophysical properties of sea ice and sea water."""

    # --- ice -----------------------------------------------------------
    rho_i: float = 917.0          # kg/m^3
    k_i: float = 2.2              # W/(m K)
    cp_i: float = 2100.0          # J/(kg K)
    latent_heat: float = 334.0e3  # J/kg (fusion)
    # --- water ---------------------------------------------------------
    rho_w: float = 1025.0         # kg/m^3
    cp_w: float = 3990.0          # J/(kg K)
    k_w: float = 0.56             # W/(m K)
    nu_w: float = 1.8e-6          # m^2/s (molecular)
    alpha_w: float = 1.4e-7       # m^2/s (molecular)
    # --- phase change --------------------------------------------------
    t_freeze: float = -1.86       # degC, sea water at ~32 psu
    # --- radiation -----------------------------------------------------
    emissivity: float = 0.97
    stefan_boltzmann: float = 5.670374419e-8
    albedo_water: float = 0.07
    albedo_ice: float = 0.55
    # --- air -----------------------------------------------------------
    rho_air: float = 1.3          # kg/m^3
    cp_air: float = 1005.0        # J/(kg K)
    drag_air: float = 1.8e-3      # C_D, 10 m neutral
    # --- air-water / water-ice transfer ---------------------------------
    stanton_water: float = 1.2e-3  # C_H for the water-side heat flux
    drag_water: float = 1.5e-3     # C_dw, surface drift vs wind stress
    gravity: float = 9.81          # m/s^2
    expansion_water: float = 1.7e-4  # 1/K, thermal expansion near freezing

    # ---------------------------------------------------------------- derived
    @property
    def rho_latent(self) -> float:
        """Volumetric latent heat of fusion, J/m^3."""
        return self.rho_i * self.latent_heat

    @property
    def alpha_i(self) -> float:
        """Ice thermal diffusivity, m^2/s."""
        return self.k_i / (self.rho_i * self.cp_i)

    @property
    def volumetric_cp_i(self) -> float:
        """Volumetric heat capacity of ice, J/(m^3 K)."""
        return self.rho_i * self.cp_i

    @property
    def volumetric_cp_w(self) -> float:
        """Volumetric heat capacity of water, J/(m^3 K)."""
        return self.rho_w * self.cp_w


DEFAULT_PROPERTIES = IceWaterProperties()


def freezing_point_C(salinity_ppt: float = 32.0) -> float:
    """Sea-water freezing point (linear UNESCO approximation), degC.

    ``T_f = -0.0575 S + 1.710523e-3 S^1.5 - 2.154996e-4 S^2`` (S in psu).
    Used only to show that the default ``-1.86 degC`` corresponds to ~32 psu.
    """
    s = float(salinity_ppt)
    return -0.0575 * s + 1.710523e-3 * s ** 1.5 - 2.154996e-4 * s ** 2


__all__ = [
    "IceWaterProperties",
    "DEFAULT_PROPERTIES",
    "freezing_point_C",
]