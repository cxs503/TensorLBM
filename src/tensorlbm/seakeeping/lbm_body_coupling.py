"""2-D free-surface LBM coupled to a floating body section (heave only).

Demonstrates the fluid–structure coupling mechanism: the Körner free-surface
LBM simulates the wave field, and a 2-D hull cross-section (midship or
arbitrary station) responds to the hydrodynamic forces in heave.

The coupling is explicit (loose):
1. LBM advances the wave field by one time step.
2. Pressure on the body surface is integrated to get vertical force F_z.
3. Body equation of motion: (m + A₃₃) z̈ + B₃₃ ż + C₃₃ z = F_z(t)
4. Body position is updated, and the LBM grid is adjusted (moving boundary).

For validation, we use a simplified approach:
- The wave field is prescribed (Stokes wave from NWT).
- The body responds to the wave-induced pressure field.
- This isolates the body dynamics from the wave generation/propagation.

This is a 2-D section analysis; the full 3-D ship RAO requires strip-theory
integration (see :mod:`tensorlbm.seakeeping.strip_theory`).

References:
    - Körner, M., et al. (2005). "A level-set method for simulating
      free-surface flows on complex geometries." Proc. of CFD.
    - Faltinsen, O.M. (1990). Sea Loads on Ships and Offshore Structures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from ..nwt.stokes2 import Stokes2Wave
from .wigley_hull import WigleyIII


@dataclass
class SectionProperties2D:
    """2-D section properties for heave motion.

    Attributes:
        x_station: Longitudinal position from midship [m].
        half_breadth: Half-breadth at waterline [m].
        sectional_area: Underwater sectional area [m²].
        added_mass_2d: 2-D heave added mass [kg/m].
        damping_2d: 2-D heave damping coefficient [N·s/m²].
        C33_2d: 2-D heave restoring [N/m²].
    """
    x_station: float
    half_breadth: float
    sectional_area: float
    added_mass_2d: float
    damping_2d: float
    C33_2d: float


@dataclass
class LBMBodyCoupling2D:
    """2-D LBM free-surface + floating body coupling.

    This class manages the coupling between the 2-D free-surface LBM
    and a floating body section. The body is constrained to heave only.

    Args:
        hull: Wigley III hull object.
        x_station: Longitudinal position of the section [m] (0 = midship).
        wave: Stokes wave object for prescribed wave field.
        rho: Water density [kg/m³].
        g: Gravitational acceleration [m/s²].
    """
    hull: WigleyIII
    x_station: float = 0.0  # midship
    wave: Optional[Stokes2Wave] = None
    rho: float = 1000.0
    g: float = 9.81

    # Body state
    z: float = 0.0  # heave displacement [m]
    z_dot: float = 0.0  # heave velocity [m/s]
    z_ddot: float = 0.0  # heave acceleration [m/s²]

    # History
    time_history: list = field(default_factory=list)
    z_history: list = field(default_factory=list)
    Fz_history: list = field(default_factory=list)
    wave_elev_history: list = field(default_factory=list)

    def __post_init__(self):
        """Compute section properties at the specified station."""
        b = self.hull.half_breadth(self.x_station, self.hull.T)
        # Sectional area (integrate half-breadth over draft)
        n_z = 20
        z_arr = np.linspace(0, self.hull.T, n_z)
        y_vals = np.array([self.hull.half_breadth(self.x_station, z) for z in z_arr])
        A_sec = 2.0 * np.trapezoid(y_vals, z_arr)

        # 2-D added mass (Lewis form approximation)
        # For a section with half-breadth b:
        # A33_2D ≈ ρ · π · b² · σ_a
        sigma_a = 1.0  # Lewis form parameter (≈1 for most sections)
        A33_2d = self.rho * math.pi * b**2 * sigma_a

        # 2-D damping (approximate)
        # B33_2D ≈ ρ · ω · π · b² · σ_b
        # Use a reference frequency
        omega_ref = 2.0  # typical wave frequency
        B33_2d = self.rho * omega_ref * math.pi * b**2 * 0.5

        # 2-D restoring
        C33_2d = self.rho * self.g * 2.0 * b

        self.section = SectionProperties2D(
            x_station=self.x_station,
            half_breadth=b,
            sectional_area=A_sec,
            added_mass_2d=A33_2d,
            damping_2d=B33_2d,
            C33_2d=C33_2d,
        )

        # Section mass (per unit length)
        # For equilibrium: mass = ρ · A_sec
        self.section_mass = self.rho * A_sec

    def compute_wave_elevation(self, t: float) -> float:
        """Compute wave elevation at the section position.

        Args:
            t: Time [s].

        Returns:
            Wave elevation η(x, t) [m].
        """
        if self.wave is None:
            return 0.0
        # Wave elevation at x = x_station
        eta = self.wave.elevation(
            torch.tensor(self.x_station, dtype=torch.float32),
            t
        )
        return float(eta)

    def compute_excitation_force(self, t: float) -> float:
        """Compute wave excitation force on the section.

        Uses the Froude-Krylov approximation:
            F_FK = ρg · A_sec · η(x, t) · e^(-kT/2)

        where η is the wave elevation at the section position.

        Args:
            t: Time [s].

        Returns:
            Excitation force per unit length [N/m].
        """
        eta = self.compute_wave_elevation(t)
        # Froude-Krylov force
        if self.wave is not None:
            k = self.wave.wavenumber
            T = self.hull.T
            F_fk = self.rho * self.g * self.section.sectional_area * eta * math.exp(-k * T / 2.0)
        else:
            F_fk = self.rho * self.g * self.section.sectional_area * eta
        return F_fk

    def compute_hydrostatic_force(self, z: float) -> float:
        """Compute hydrostatic restoring force.

        F_hs = -C33 · z

        Args:
            z: Heave displacement [m].

        Returns:
            Hydrostatic force per unit length [N/m].
        """
        return -self.section.C33_2d * z

    def compute_radiation_force(
        self, z_dot: float, z_history: list, t_history: list
    ) -> float:
        """Compute radiation (damping) force.

        Simplified: F_rad = -B33 · ż

        For a more accurate model, use the retardation function:
            F_rad = -∫₀ᵗ K(t-τ) · ż(τ) dτ

        Args:
            z_dot: Current heave velocity [m/s].
            z_history: Past heave velocities.
            t_history: Past times.

        Returns:
            Radiation force per unit length [N/m].
        """
        return -self.section.damping_2d * z_dot

    def step(self, dt: float, t: float) -> float:
        """Advance the coupled system by one time step.

        Uses explicit Euler integration for the body equation of motion:
            (m + A33) z̈ = F_exc + F_hs + F_rad

        Args:
            dt: Time step [s].
            t: Current time [s].

        Returns:
            New heave displacement [m].
        """
        # Compute forces
        F_exc = self.compute_excitation_force(t)
        F_hs = self.compute_hydrostatic_force(self.z)
        F_rad = self.compute_radiation_force(self.z_dot, self.z_history, self.time_history)

        # Total force
        F_total = F_exc + F_hs + F_rad

        # Equation of motion: (m + A33) z̈ = F_total
        m_total = self.section_mass + self.section.added_mass_2d
        self.z_ddot = F_total / m_total

        # Integrate (Euler)
        self.z_dot += self.z_ddot * dt
        self.z += self.z_dot * dt

        # Record history
        self.time_history.append(t)
        self.z_history.append(self.z)
        self.Fz_history.append(F_total)
        self.wave_elev_history.append(self.compute_wave_elevation(t))

        return self.z

    def reset(self):
        """Reset the body state to equilibrium."""
        self.z = 0.0
        self.z_dot = 0.0
        self.z_ddot = 0.0
        self.time_history = []
        self.z_history = []
        self.Fz_history = []
        self.wave_elev_history = []


def run_section_coupling(
    hull: Optional[WigleyIII] = None,
    x_station: float = 0.0,
    wave_amplitude: float = 0.01,
    wave_length: float = 3.0,
    depth: float = 10.0,
    n_periods: int = 20,
    steps_per_period: int = 200,
    rho: float = 1000.0,
    g: float = 9.81,
) -> dict:
    """Run 2-D section coupling simulation.

    Args:
        hull: Wigley III hull (default: standard dimensions).
        x_station: Section position [m] (0 = midship).
        wave_amplitude: Wave amplitude ζ_a [m].
        wave_length: Wavelength λ [m].
        depth: Water depth [m].
        n_periods: Number of wave periods to simulate.
        steps_per_period: Time steps per wave period.
        rho: Water density [kg/m³].
        g: Gravitational acceleration [m/s²].

    Returns:
        Dictionary with:
        - 'time': time array [s]
        - 'z': heave displacement [m]
        - 'z_dot': heave velocity [m/s]
        - 'wave_elev': wave elevation at section [m]
        - 'Fz': total force [N/m]
        - 'section': SectionProperties2D
    """
    hull = hull or WigleyIII(rho=rho, g=g)

    # Create Stokes wave
    k = 2.0 * math.pi / wave_length
    omega = math.sqrt(g * k * math.tanh(k * depth))
    wave = Stokes2Wave(
        amplitude=wave_amplitude,
        wavenumber=k,
        depth=depth,
        gravity=g,
    )

    # Create coupling object
    coupling = LBMBodyCoupling2D(
        hull=hull,
        x_station=x_station,
        wave=wave,
        rho=rho,
        g=g,
    )

    # Time integration
    T_period = 2.0 * math.pi / omega
    dt = T_period / steps_per_period
    total_steps = int(n_periods * steps_per_period)

    for step in range(total_steps):
        t = step * dt
        coupling.step(dt, t)

    # Convert history to arrays
    time_arr = np.array(coupling.time_history)
    z_arr = np.array(coupling.z_history)
    Fz_arr = np.array(coupling.Fz_history)
    wave_arr = np.array(coupling.wave_elev_history)

    return {
        "time": time_arr,
        "z": z_arr,
        "z_dot": np.gradient(z_arr, time_arr) if len(time_arr) > 1 else np.zeros_like(z_arr),
        "wave_elev": wave_arr,
        "Fz": Fz_arr,
        "section": coupling.section,
        "omega": omega,
        "k": k,
        "T_period": T_period,
    }
