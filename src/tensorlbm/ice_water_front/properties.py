"""Thermophysical properties of the ice/water system and derived groups.

Reference values are the standard engineering data for pure ice and liquid
water at the normal melting point (0 degC, 1 atm):

=========================  =========  ==========  ==================
quantity                    ice        water       unit
=========================  =========  ==========  ==================
density rho                 917        999.8       kg/m^3
thermal conductivity k      2.22       0.5610      W/(m K)
specific heat c_p           2090       4217        J/(kg K)
latent heat of fusion L     --         333.55e3    J/kg
melting temperature T_m     --         273.15      K
=========================  =========  ==========  ==================

Derived: ``alpha = k / (rho c_p)``, ``Ste = c_l dT / L``,
``Ja = rho_l c_l dT / (rho_s L)`` (volumetric Jacob number, = Ste for the
density-equal idealisation).

Sources: Incropera & DeWitt *Fundamentals of Heat and Mass Transfer* table
A.3 (ice) / A.6 (water); ASHRAE Handbook--Fundamentals (2009) ch. 4 for
freezing/melting; Lunardini (1991) *Heat Transfer with Freezing and
Thawing*, table 2.1, for the L value used in phase-change engineering.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PhaseProperties:
    """Homogeneous thermophysical description of one phase."""

    name: str
    rho: float  # kg/m^3
    k: float  # W/(m K)
    cp: float  # J/(kg K)

    @property
    def alpha(self) -> float:
        """Thermal diffusivity ``k/(rho cp)`` in m^2/s."""
        return self.k / (self.rho * self.cp)

    @property
    def volumetric_heat_capacity(self) -> float:
        return self.rho * self.cp


ICE = PhaseProperties(name="ice", rho=917.0, k=2.22, cp=2090.0)
WATER = PhaseProperties(name="water", rho=999.8, k=0.5610, cp=4217.0)

T_MELT = 273.15  # K
L_FUSION = 333.55e3  # J/kg


@dataclass(frozen=True)
class IceWaterCase:
    """A melting or freezing scenario in the ice/water system.

    Parameters
    ----------
    t_wall : float
        Temperature imposed at the driving boundary [K].
    t_initial : float
        Initial (far field / solid) temperature [K].
    length : float
        Characteristic length scale of the problem (domain size or initial
        radius) [m].
    latent_heat : float
        Latent heat of fusion [J/kg].
    t_melt : float
        Melting temperature [K].
    ice, water : PhaseProperties
        Phase data; override for sensitivity studies.
    density_equal : bool
        When True (default) the density-equal idealisation ``rho_s = rho_l``
        is used, which is the assumption under which the Neumann similarity
        solution used as anchor C1 is derived.  Set False to apply the
        volumetric latent heat ``rho_ice * L`` (sensitivity study only --
        the similarity anchor no longer applies exactly).
    """

    t_wall: float
    t_initial: float = T_MELT
    length: float = 0.02
    latent_heat: float = L_FUSION
    t_melt: float = T_MELT
    ice: PhaseProperties = field(default=ICE)
    water: PhaseProperties = field(default=WATER)
    density_equal: bool = True

    # -- geometry of the transition ------------------------------------
    @property
    def melting(self) -> bool:
        """True when the boundary is above the melting point (ice melts)."""
        return self.t_wall > self.t_melt

    @property
    def inner(self) -> PhaseProperties:
        """Phase adjacent to the driving wall (liquid when melting)."""
        return self.water if self.melting else self.ice

    @property
    def outer(self) -> PhaseProperties:
        """Bulk phase ahead of the front (solid when melting)."""
        return self.ice if self.melting else self.water

    # -- driving temperature differences ------------------------------
    @property
    def dT_inner(self) -> float:
        """Signed ``T_wall - T_melt`` (>0 melting, <0 freezing)."""
        return self.t_wall - self.t_melt

    @property
    def dT_outer(self) -> float:
        """Signed ``T_melt - T_initial`` (>=0 when the bulk is subcooled)."""
        return self.t_melt - self.t_initial

    # -- dimensionless groups -----------------------------------------
    @property
    def stefan_inner(self) -> float:
        """``Ste_in = c_in |dT_in| / L`` (always positive)."""
        return self.inner.cp * abs(self.dT_inner) / self.latent_heat

    @property
    def stefan_outer(self) -> float:
        """``Ste_out = c_out dT_out / L`` (0 for an isothermal bulk)."""
        return self.outer.cp * self.dT_outer / self.latent_heat

    @property
    def stefan_number(self) -> float:
        """Primary Stefan number quoted in the campaign brief."""
        return self.stefan_inner

    @property
    def jacob_number(self) -> float:
        """Volumetric Jacob number ``rho_in c_in dT / (rho Lat L)``."""
        lat_rho = self.water.rho if self.density_equal else self.ice.rho
        return self.inner.rho * self.inner.cp * abs(self.dT_inner) / (
            lat_rho * self.latent_heat
        )

    @property
    def diffusivity_ratio(self) -> float:
        """``alpha_outer / alpha_inner``."""
        return self.outer.alpha / self.inner.alpha

    @property
    def conductivity_ratio(self) -> float:
        """``k_outer / k_inner``."""
        return self.outer.k / self.inner.k

    @property
    def latent_volumetric(self) -> float:
        """Latent heat per unit volume [J/m^3].

        Under the density-equal idealisation the reference density is the
        inner (liquid) one, matching the similarity derivation.
        """
        rho_ref = self.water.rho if self.density_equal else self.ice.rho
        return rho_ref * self.latent_heat

    @property
    def alpha_inner(self) -> float:
        return self.inner.alpha

    def summary(self) -> dict:
        """JSON-ready summary of the case (used in the report artefacts)."""
        return {
            "t_wall_K": self.t_wall,
            "t_wall_C": self.t_wall - self.t_melt,
            "t_initial_K": self.t_initial,
            "t_initial_C": self.t_initial - self.t_melt,
            "length_m": self.length,
            "mode": "melting" if self.melting else "freezing",
            "stefan_inner": self.stefan_inner,
            "stefan_outer": self.stefan_outer,
            "jacob_number": self.jacob_number,
            "alpha_ice_m2_s": self.ice.alpha,
            "alpha_water_m2_s": self.water.alpha,
            "diffusivity_ratio": self.diffusivity_ratio,
            "conductivity_ratio": self.conductivity_ratio,
            "latent_volumetric_J_m3": self.latent_volumetric,
            "density_equal": self.density_equal,
        }


__all__ = [
    "PhaseProperties",
    "IceWaterCase",
    "ICE",
    "WATER",
    "T_MELT",
    "L_FUSION",
]