"""LBT-style unit converter mirroring 2b-t/LB-t's Converter API.

The 2b-t/LB-t ``Converter`` (C++) takes three (physical, lbm_equivalent) pairs
for length, velocity and density, then derives the conversion factors for
time, kinematic viscosity and pressure:

    to_lbm_time_            = to_lbm_length_ / to_lbm_velocity_
    to_lbm_kinematic_viscosity_ = to_lbm_velocity_ * to_lbm_length_
    to_lbm_pressure_        = to_lbm_density_ * to_lbm_velocity_**2

This Python reimplementation follows the same interface so that numerical
results can be compared bit-for-bit (within floating-point precision).

Reference: 2b-t/LB-t ``include/lbt/converter.hpp``
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LbtStyleConverter:
    """Unit converter between lattice-Boltzmann and physical units.

    Mirrors the 2b-t/LB-t ``Converter`` class.  Three independent
    (physical, lbm_equivalent) pairs define the conversion; all other
    conversion factors are derived.

    Parameters
    ----------
    physical_length : float
        An arbitrary length in SI units [m].
    equivalent_lbm_length : float
        The equivalent of *physical_length* in LBM units [lattice cells].
    physical_velocity : float
        An arbitrary velocity in SI units [m/s].
    equivalent_lbm_velocity : float
        The equivalent of *physical_velocity* in LBM units [lattice velocity].
    physical_density : float
        An arbitrary density in SI units [kg/m³].
    equivalent_lbm_density : float
        The equivalent of *physical_density* in LBM units [lattice density].

    Derived conversion factors (physical → LBM)
    -------------------------------------------
    to_lbm_length   = physical_length / equivalent_lbm_length
    to_lbm_velocity = physical_velocity / equivalent_lbm_velocity
    to_lbm_density  = physical_density / equivalent_lbm_density
    to_lbm_time     = to_lbm_length / to_lbm_velocity
    to_lbm_nu       = to_lbm_velocity * to_lbm_length
    to_lbm_pressure = to_lbm_density * to_lbm_velocity**2
    """

    physical_length: float
    equivalent_lbm_length: float
    physical_velocity: float
    equivalent_lbm_velocity: float
    physical_density: float
    equivalent_lbm_density: float

    # --- derived conversion factors (physical → LBM) ---
    @property
    def to_lbm_length(self) -> float:
        """Physical length [m] → LBM length [cells]."""
        return self.physical_length / self.equivalent_lbm_length

    @property
    def to_lbm_velocity(self) -> float:
        """Physical velocity [m/s] → LBM velocity."""
        return self.physical_velocity / self.equivalent_lbm_velocity

    @property
    def to_lbm_density(self) -> float:
        """Physical density [kg/m³] → LBM density."""
        return self.physical_density / self.equivalent_lbm_density

    @property
    def to_lbm_time(self) -> float:
        """Physical time [s] → LBM time [steps]."""
        return self.to_lbm_length / self.to_lbm_velocity

    @property
    def to_lbm_nu(self) -> float:
        """Physical kinematic viscosity [m²/s] → LBM viscosity."""
        return self.to_lbm_velocity * self.to_lbm_length

    @property
    def to_lbm_pressure(self) -> float:
        """Physical pressure [Pa] → LBM pressure."""
        return self.to_lbm_density * self.to_lbm_velocity ** 2

    # --- toPhysical: LBM → physical ---
    def length_to_phys(self, lbm_length: float) -> float:
        """Convert LBM length to physical [m]."""
        return self.to_lbm_length * lbm_length

    def velocity_to_phys(self, lbm_velocity: float) -> float:
        """Convert LBM velocity to physical [m/s]."""
        return self.to_lbm_velocity * lbm_velocity

    def density_to_phys(self, lbm_density: float) -> float:
        """Convert LBM density to physical [kg/m³]."""
        return self.to_lbm_density * lbm_density

    def time_to_phys(self, lbm_time: float) -> float:
        """Convert LBM time to physical [s]."""
        return self.to_lbm_time * lbm_time

    def nu_to_phys(self, lbm_nu: float) -> float:
        """Convert LBM kinematic viscosity to physical [m²/s]."""
        return self.to_lbm_nu * lbm_nu

    def pressure_to_phys(self, lbm_pressure: float) -> float:
        """Convert LBM pressure to physical [Pa]."""
        return self.to_lbm_pressure * lbm_pressure

    # --- toLbm: physical → LBM ---
    def length_to_lbm(self, phys_length: float) -> float:
        """Convert physical length [m] to LBM."""
        return phys_length / self.to_lbm_length

    def velocity_to_lbm(self, phys_velocity: float) -> float:
        """Convert physical velocity [m/s] to LBM."""
        return phys_velocity / self.to_lbm_velocity

    def density_to_lbm(self, phys_density: float) -> float:
        """Convert physical density [kg/m³] to LBM."""
        return phys_density / self.to_lbm_density

    def time_to_lbm(self, phys_time: float) -> float:
        """Convert physical time [s] to LBM."""
        return phys_time / self.to_lbm_time

    def nu_to_lbm(self, phys_nu: float) -> float:
        """Convert physical kinematic viscosity [m²/s] to LBM."""
        return phys_nu / self.to_lbm_nu

    def pressure_to_lbm(self, phys_pressure: float) -> float:
        """Convert physical pressure [Pa] to LBM."""
        return phys_pressure / self.to_lbm_pressure

    # --- roundtrip consistency ---
    def roundtrip_error(self, site: str, value: float) -> float:
        """Compute relative roundtrip error for a conversion site.

        Converts ``value`` to LBM and back to physical, then returns
        ``|phys_roundtrip - value| / |value|``.
        """
        to_lbm = getattr(self, f"{site}_to_lbm")
        to_phys = getattr(self, f"{site}_to_phys")
        roundtrip = to_phys(to_lbm(value))
        if value == 0.0:
            return 0.0
        return abs(roundtrip - value) / abs(value)

    # --- Reynolds number ---
    def reynolds(self, velocity: float, length: float, nu: float) -> float:
        """Compute Reynolds number (dimensionless, same in both unit systems)."""
        return velocity * length / nu

    # --- Mach number ---
    def mach(self, u_lb: float) -> float:
        """Compute Mach number from lattice velocity."""
        return abs(u_lb) * math.sqrt(3.0)

    # --- BGK relaxation time ---
    def tau_from_nu_lb(self, nu_lb: float) -> float:
        """BGK relaxation time from lattice viscosity: τ = 0.5 + 3·ν_lb."""
        return 0.5 + 3.0 * nu_lb

    def summary(self) -> str:
        """Human-readable summary of all conversion factors."""
        lines = [
            "LBT-Style Unit Converter",
            "========================",
            f"  physical_length          = {self.physical_length:.6g} m",
            f"  equivalent_lbm_length    = {self.equivalent_lbm_length:.6g}",
            f"  physical_velocity        = {self.physical_velocity:.6g} m/s",
            f"  equivalent_lbm_velocity  = {self.equivalent_lbm_velocity:.6g}",
            f"  physical_density         = {self.physical_density:.6g} kg/m³",
            f"  equivalent_lbm_density   = {self.equivalent_lbm_density:.6g}",
            "  ---- derived factors (phys → LBM) ----",
            f"  to_lbm_length            = {self.to_lbm_length:.6g}",
            f"  to_lbm_velocity          = {self.to_lbm_velocity:.6g}",
            f"  to_lbm_density           = {self.to_lbm_density:.6g}",
            f"  to_lbm_time              = {self.to_lbm_time:.6g}",
            f"  to_lbm_nu                = {self.to_lbm_nu:.6g}",
            f"  to_lbm_pressure          = {self.to_lbm_pressure:.6g}",
        ]
        return "\n".join(lines)
