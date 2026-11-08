"""Strip-theory frequency-domain seakeeping solver (BEM reference).

Implements the Salvesen-Tuck-Faltinsen (STF) strip theory for heave and pitch
motions of a ship in head waves.  The theory integrates 2-D section
hydrodynamic coefficients along the ship length.

Theory summary
--------------
For a ship advancing at speed U in head waves of frequency ω_e (encounter
frequency), the equations of motion in heave (3) and pitch (5) are:

    Σⱼ [(M₃ⱼ + A₃ⱼ) ξ̈ⱼ + B₃ⱼ ξ̇ⱼ + C₃ⱼ ξⱼ] = F₃(t)
    Σⱼ [(M₅ⱼ + A₅ⱼ) ξ̈ⱼ + B₅ⱼ ξ̇ⱼ + C₅ⱼ ξⱼ] = F₅(t)

where j ∈ {3, 5} (heave, pitch), and the coefficients are:
    M₃₃ = mass, M₅₅ = Iyy
    A₃₃(ω), A₃₅(ω), A₅₃(ω), A₅₅(ω) — added mass (integrated sections)
    B₃₃(ω), B₃₅(ω), B₅₃(ω), B₅₅(ω) — damping (integrated sections)
    C₃₃ = ρgA_wp, C₅₅ = ρg∇GM_L — hydrostatic restoring
    F₃(ω), F₅(ω) — complex excitation forces (Froude-Krylov + diffraction)

The 2-D section coefficients are computed using:
- Added mass: Lewis-form conformal mapping (Grim, 1957; Frank, 1967)
- Damping: long-wave approximation + Ursell correction
- Excitation: Froude-Krylov + simplified diffraction

References:
    - Salvesen, N., Tuck, E.O., Faltinsen, O. (1970). "Ship motions and
      sea loads." Trans. SNAME, 78, 250-287.
    - Faltinsen, O.M. (1990). Sea Loads on Ships and Offshore Structures.
      Cambridge University Press.
    - Journée, J.M.J. (1992). "Experiments and calculations on 4 Wigley
      hullforms in head waves." Delft UT report 0902.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from .wigley_hull import WigleyIII, sectional_properties


@dataclass
class StripTheoryResult:
    """Result of strip-theory computation at one frequency.

    Attributes:
        omega: Encounter frequency [rad/s].
        added_mass: 2×2 added mass matrix [kg, kg·m, kg·m²].
        damping: 2×2 damping matrix.
        excitation: Complex excitation vector [N, N·m].
        rao_complex: Complex RAO [m, rad].
        rao_amplitude: |RAO| [m/m for heave, rad/m for pitch].
        rao_phase: Phase of RAO [rad].
    """
    omega: float
    added_mass: np.ndarray  # (2, 2)
    damping: np.ndarray  # (2, 2)
    excitation: np.ndarray  # (2,) complex
    rao_complex: np.ndarray  # (2,) complex
    rao_amplitude: np.ndarray  # (2,)
    rao_phase: np.ndarray  # (2,)


class StripTheorySolver:
    """Strip-theory seakeeping solver for Wigley III in head waves.

    Args:
        hull: Wigley III hull object.
        n_stations: Number of strip stations.
        speed: Forward speed U [m/s] (default 0, zero-speed).
        beta: Wave heading angle [rad] (0 = head waves).
    """

    def __init__(
        self,
        hull: Optional[WigleyIII] = None,
        n_stations: int = 21,
        speed: float = 0.0,
        beta: float = 0.0,
    ):
        self.hull = hull or WigleyIII()
        self.n_stations = n_stations
        self.speed = speed
        self.beta = beta

        # Pre-compute sectional properties
        self._sections = sectional_properties(self.hull, n_stations)

    def _encounter_frequency(self, omega: float, k: float) -> float:
        """Compute encounter frequency ω_e = ω - k·U·cos(β)."""
        return omega - k * self.speed * math.cos(self.beta)

    def _section_added_mass_2d(
        self, b: float, T: float, omega: float, sigma_a: float
    ) -> float:
        """2-D heave added mass for a section.

        Uses the Lewis-form result:
            a₃₃ = ρ · π · b² · σ_a · N(ω)

        where N(ω) is a frequency correction factor.
        For high frequency (kb >> 1): a₃₃ → ρ · π · b² · σ_a
        For low frequency (kb → 0): a₃₃ → ρ · π · b² · σ_a · (1 + correction)

        Simplified: use the infinite-frequency value with a smooth transition.
        """
        if b < 1e-10:
            return 0.0
        # Deep-water wavenumber
        k = omega**2 / self.hull.g if omega > 1e-10 else 0.0
        kb = k * b
        # Frequency correction (Grim's approximation)
        if kb > 3.0:
            N_omega = 1.0
        elif kb > 0.1:
            N_omega = 1.0 + 0.3 * math.exp(-kb)
        else:
            N_omega = 1.3
        return self.hull.rho * math.pi * b**2 * sigma_a * N_omega

    def _section_damping_2d(
        self, b: float, T: float, omega: float, sigma_a: float
    ) -> float:
        """2-D heave radiation damping for a section.

        Uses the Ursell/Tuck approximation:
            b₃₃ ≈ ρ · ω · π · b² · σ_b

        where σ_b is the damping Lewis parameter.
        For a section with beam/draft ratio B/T:
            σ_b ≈ σ_a · f(B/T, ω)

        Simplified: use the long-wave limit with frequency correction.
        """
        if b < 1e-10 or omega < 1e-10:
            return 0.0
        k = omega**2 / self.hull.g
        kb = k * b
        # Damping Lewis parameter (approximate)
        # For kb >> 1: σ_b → σ_a (similar to added mass)
        # For kb << 1: σ_b → 0 (no radiation in long waves)
        sigma_b = sigma_a * (1.0 - math.exp(-2.0 * kb))
        return self.hull.rho * omega * math.pi * b**2 * sigma_b

    def _section_excitation_2d(
        self, b: float, T: float, omega: float, A_sec: float, x: float, k: float
    ) -> complex:
        """2-D heave excitation force per unit wave amplitude.

        Froude-Krylov + simplified diffraction:
            f₃ = ρg · [2b · e^(-kT) + diffraction correction]

        The diffraction correction accounts for the scattered wave field.
        For a 2-D section, the total excitation is:
            f₃ = ρg · 2b · H₃(k, b, T)

        where H₃ is the excitation transfer function.
        """
        if b < 1e-10:
            return 0.0
        # Froude-Krylov (pressure integration over section)
        # For a rectangular section of width 2b and draft T:
        # F_FK = ρg · ∫₀ᵀ e^(-kz) · 2b dz = ρg · 2b · (1 - e^(-kT)) / k
        # For the actual section, use the sectional area:
        F_fk = self.hull.rho * self.hull.g * A_sec * math.exp(-k * T / 2.0)

        # Diffraction correction (simplified)
        # For a 2-D cylinder of radius b: correction ≈ 1 + O(kb)²
        # Use the MacCamy-Fuchs approximation for a vertical cylinder
        kb = k * b
        if kb < 0.5:
            F_diff = F_fk * (1.0 + 0.1 * kb**2)
        else:
            F_diff = F_fk * (1.0 + 0.2 * kb)

        # Phase: excitation is in phase with wave elevation at x
        # (for head waves, the phase varies along the ship)
        phase = -k * x  # wave phase at station x
        return F_diff * complex(math.cos(phase), math.sin(phase))

    def _section_pitch_excitation_2d(
        self, b: float, T: float, omega: float, A_sec: float, x: float, k: float
    ) -> complex:
        """2-D pitch excitation moment per unit wave amplitude.

        f₅ = x · f₃ (moment arm from midship)
        """
        f3 = self._section_excitation_2d(b, T, omega, A_sec, x, k)
        return x * f3

    def solve_at_frequency(self, omega: float) -> StripTheoryResult:
        """Solve strip theory at one frequency.

        Args:
            omega: Wave frequency [rad/s].

        Returns:
            StripTheoryResult with added mass, damping, excitation, and RAO.
        """
        hull = self.hull
        sections = self._sections
        stations = sections["stations"]
        y_wp = sections["y_wp"]
        A_sec = sections["A_sec"]
        sigma_a = sections["sigma_a"]
        dx = sections["dx"]

        # Wavenumber from dispersion relation
        k = omega**2 / hull.g  # deep-water approximation
        # For finite depth, iterate: ω² = gk·tanh(kh)
        # Assume deep water for now (h >> λ)

        # Encounter frequency (for zero speed, ω_e = ω)
        omega_e = self._encounter_frequency(omega, k)

        # Integrate section coefficients along the ship
        A33 = 0.0
        A35 = 0.0
        A55 = 0.0
        B33 = 0.0
        B35 = 0.0
        B55 = 0.0
        F3 = complex(0.0, 0.0)
        F5 = complex(0.0, 0.0)

        for i in range(len(stations)):
            b = y_wp[i]
            x = stations[i]
            if b < 1e-10:
                continue

            # 2-D section coefficients
            a33 = self._section_added_mass_2d(b, hull.T, omega_e, sigma_a[i])
            b33 = self._section_damping_2d(b, hull.T, omega_e, sigma_a[i])

            # Excitation forces
            f3 = self._section_excitation_2d(b, hull.T, omega_e, A_sec[i], x, k)
            f5 = self._section_pitch_excitation_2d(b, hull.T, omega_e, A_sec[i], x, k)

            # Integrate (trapezoidal rule)
            weight = dx
            if i == 0 or i == len(stations) - 1:
                weight = dx / 2.0

            A33 += a33 * weight
            A35 += a33 * x * weight  # coupling term (approximate)
            A55 += a33 * x**2 * weight
            B33 += b33 * weight
            B35 += b33 * x * weight
            B55 += b33 * x**2 * weight
            F3 += f3 * weight
            F5 += f5 * weight

        # Mass matrix
        M33 = hull.mass
        M55 = hull.Iyy

        # Hydrostatic restoring
        C33 = hull.C33
        C55 = hull.C55
        C35 = 0.0  # no coupling for symmetric hull at zero speed

        # Solve the 2-DOF equation of motion
        # [-ω²(M + A) + jωB + C] · ξ = F
        Z = np.zeros((2, 2), dtype=complex)
        Z[0, 0] = -(M33 + A33) * omega**2 + 1j * omega * B33 + C33
        Z[0, 1] = -A35 * omega**2 + 1j * omega * B35 + C35
        Z[1, 0] = -A35 * omega**2 + 1j * omega * B35 + C35
        Z[1, 1] = -(M55 + A55) * omega**2 + 1j * omega * B55 + C55

        F = np.array([F3, F5], dtype=complex)
        xi = np.linalg.solve(Z, F)

        # RAO amplitude and phase
        rao_amp = np.abs(xi)
        rao_phase = np.angle(xi)

        return StripTheoryResult(
            omega=omega,
            added_mass=np.array([[A33, A35], [A35, A55]]),
            damping=np.array([[B33, B35], [B35, B55]]),
            excitation=F,
            rao_complex=xi,
            rao_amplitude=rao_amp,
            rao_phase=rao_phase,
        )

    def solve_frequency_range(
        self,
        omega_min: float = 0.5,
        omega_max: float = 5.0,
        n_freq: int = 20,
    ) -> list[StripTheoryResult]:
        """Solve strip theory over a range of frequencies.

        Args:
            omega_min: Minimum frequency [rad/s].
            omega_max: Maximum frequency [rad/s].
            n_freq: Number of frequency points.

        Returns:
            List of StripTheoryResult, one per frequency.
        """
        omegas = np.linspace(omega_min, omega_max, n_freq)
        results = [self.solve_at_frequency(w) for w in omegas]
        return results


def compute_strip_theory_rao(
    hull: Optional[WigleyIII] = None,
    omega_range: Optional[np.ndarray] = None,
    n_stations: int = 21,
) -> dict:
    """Compute heave and pitch RAO using strip theory.

    Args:
        hull: Wigley III hull (default: standard dimensions).
        omega_range: Frequency array [rad/s]. If None, uses 0.5 to 5.0.
        n_stations: Number of strip stations.

    Returns:
        Dictionary with:
        - 'omega': frequency array
        - 'heave_rao': |H3(ω)| [m/m]
        - 'pitch_rao': |H5(ω)| [rad/m]
        - 'heave_phase': phase [rad]
        - 'pitch_phase': phase [rad]
    """
    hull = hull or WigleyIII()
    solver = StripTheorySolver(hull, n_stations=n_stations)

    if omega_range is None:
        omega_range = np.linspace(0.5, 5.0, 20)

    results = [solver.solve_at_frequency(w) for w in omega_range]

    heave_rao = np.array([r.rao_amplitude[0] for r in results])
    pitch_rao = np.array([r.rao_amplitude[1] for r in results])
    heave_phase = np.array([r.rao_phase[0] for r in results])
    pitch_phase = np.array([r.rao_phase[1] for r in results])

    return {
        "omega": omega_range,
        "heave_rao": heave_rao,
        "pitch_rao": pitch_rao,
        "heave_phase": heave_phase,
        "pitch_phase": pitch_phase,
        "results": results,
    }
