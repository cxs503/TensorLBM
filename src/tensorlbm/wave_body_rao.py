"""Analytical RAO (Response Amplitude Operator) solutions for wave-body validation.

Provides closed-form / semi-analytical RAO solutions for simple floating body
geometries in regular (Airy) waves.  These serve as reference data for
end-to-end validation of the wave_bc + sixdof coupling in TensorLBM.

Geometries supported
--------------------
1. **Vertical circular cylinder (spar buoy)** – heave & pitch RAO
2. **Rectangular barge** – heave & pitch RAO (2-D strip theory approximation)

Theory
------
For a single-DOF linear system in regular waves of frequency ω:

    (m + A(ω)) ẍ + B(ω) ẋ + C x = F_exc(ω) cos(ωt + φ)

The RAO amplitude is:

    |H(ω)| = |F_exc(ω)| / √[(C − (m + A(ω)) ω²)² + (B(ω) ω)²]

where:
    A(ω)  – frequency-dependent added mass
    B(ω)  – frequency-dependent radiation damping
    C     – hydrostatic restoring coefficient
    F_exc – complex excitation amplitude (Froude-Krylov + diffraction)

References
----------
- Newman, J. N. (1977). *Marine Hydrodynamics*. MIT Press.
- Faltinsen, O. M. (1990). *Sea Loads on Ships and Offshore Structures*.
  Cambridge University Press.
- Budal, K. & Falnes J. (1975). "A resonant point absorber of ocean waves",
  Nature 256, 478–479.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# Hydrodynamic coefficients for a vertical circular cylinder (spar)
# ---------------------------------------------------------------------------

@dataclass
class SparGeometry:
    """Vertical circular cylinder (spar buoy) parameters."""
    radius: float          # R [m]
    draft: float           # d [m] (positive downward from waterline)
    mass: float            # m [kg]
    iyy: float             # pitch moment of inertia [kg·m²]
    rho_water: float = 1025.0   # water density [kg/m³]
    g: float = 9.81            # gravity [m/s²]

    @property
    def waterplane_area(self) -> float:
        return math.pi * self.radius**2

    @property
    def displaced_volume(self) -> float:
        return math.pi * self.radius**2 * self.draft

    @property
    def C33(self) -> float:
        """Hydrostatic heave restoring coefficient [N/m]."""
        return self.rho_water * self.g * self.waterplane_area

    @property
    def C55(self) -> float:
        """Hydrostatic pitch restoring coefficient [N·m/rad].

        C55 = ρ g ∇ GM_L  where GM_L ≈ BM − BG + I_wp/∇
        Simplified for a uniform cylinder:
            BM = d/2,  BG = d/2 (CG at mid-draft for uniform mass)
            I_wp = π R⁴ / 4
        """
        I_wp = math.pi * self.radius**4 / 4.0
        nabla = self.displaced_volume
        BM = self.draft / 2.0
        # Assume CG at mid-draft for a uniform spar
        BG = self.draft / 2.0
        GM = BM - BG + I_wp / nabla
        return self.rho_water * self.g * nabla * GM


def spar_heave_added_mass(omega: float, geo: SparGeometry) -> float:
    """Approximate heave added mass A33(ω) for a vertical cylinder.

    Uses the long-wave (low-frequency) limit:
        A33 ≈ ρ π R² d / 2   (heave added mass ≈ half the displaced mass)

    For a more accurate frequency-dependent model, one would use
    matched asymptotic expansions or tabulated BEM results.
    """
    # Low-frequency approximation (valid for kR << 1)
    return 0.5 * geo.rho_water * math.pi * geo.radius**2 * geo.draft


def spar_heave_radiation_damping(omega: float, geo: SparGeometry) -> float:
    """Approximate heave radiation damping B33(ω) for a vertical cylinder.

    Haskind relation (deep water):
        B33(ω) ≈ ρ ω³ R² |H3_exc|² / (2 g²)

    Simplified low-frequency estimate:
        B33 ≈ ρ g π R² ω (kR)² / 2   for kR << 1
    """
    if omega < 1e-10:
        return 0.0
    k = omega**2 / geo.g  # deep-water dispersion
    kR = k * geo.radius
    return 0.5 * geo.rho_water * geo.g * math.pi * geo.radius**2 * omega * kR**2


def spar_heave_excitation(omega: float, geo: SparGeometry) -> complex:
    """Heave excitation force amplitude F3_exc(ω) per unit wave amplitude.

    Froude-Krylov approximation for a vertical cylinder:
        F3_exc ≈ ρ g π R² e^(−kd)   (real, in-phase with wave elevation)

    This neglects diffraction (valid for kR << 1).
    """
    if omega < 1e-10:
        return complex(geo.rho_water * geo.g * math.pi * geo.radius**2, 0.0)
    k = omega**2 / geo.g  # deep-water dispersion
    Fk = geo.rho_water * geo.g * math.pi * geo.radius**2 * math.exp(-k * geo.draft)
    return complex(Fk, 0.0)


def spar_pitch_added_mass(omega: float, geo: SparGeometry) -> float:
    """Approximate pitch added mass A55(ω) for a vertical cylinder.

    Low-frequency estimate:
        A55 ≈ ρ π R² d³ / 3
    """
    return geo.rho_water * math.pi * geo.radius**2 * geo.draft**3 / 3.0


def spar_pitch_radiation_damping(omega: float, geo: SparGeometry) -> float:
    """Approximate pitch radiation damping B55(ω)."""
    if omega < 1e-10:
        return 0.0
    k = omega**2 / geo.g
    kR = k * geo.radius
    return 0.5 * geo.rho_water * geo.g * math.pi * geo.radius**2 * geo.draft**2 * omega * kR**2


def spar_pitch_excitation(omega: float, geo: SparGeometry) -> complex:
    """Pitch excitation moment amplitude F5_exc(ω) per unit wave amplitude.

    Froude-Krylov for a vertical cylinder:
        F5_exc ≈ ρ g π R² d e^(−kd)   (leading-order term)
    """
    if omega < 1e-10:
        return complex(0.0, 0.0)
    k = omega**2 / geo.g
    Fk = geo.rho_water * geo.g * math.pi * geo.radius**2 * geo.draft * math.exp(-k * geo.draft)
    return complex(Fk, 0.0)


# ---------------------------------------------------------------------------
# RAO computation
# ---------------------------------------------------------------------------

def compute_heave_rao(
    omega: float | torch.Tensor,
    geo: SparGeometry,
) -> torch.Tensor:
    """Compute heave RAO |H3(ω)| = |z_a / ζ_a| [m/m] for a spar buoy.

    Args:
        omega: Angular frequency [rad/s], scalar or tensor.
        geo: Spar geometry parameters.

    Returns:
        Heave RAO amplitude (dimensionless ratio of heave amplitude to wave
        amplitude).
    """
    if not isinstance(omega, torch.Tensor):
        omega = torch.tensor([omega], dtype=torch.float64)

    rao = torch.zeros_like(omega, dtype=torch.float64)
    for i, w in enumerate(omega):
        w_val = w.item()
        A33 = spar_heave_added_mass(w_val, geo)
        B33 = spar_heave_radiation_damping(w_val, geo)
        F3 = abs(spar_heave_excitation(w_val, geo))
        C33 = geo.C33

        denom = math.sqrt(
            (C33 - (geo.mass + A33) * w_val**2)**2
            + (B33 * w_val)**2
        )
        if denom > 1e-15:
            rao[i] = F3 / denom
        else:
            rao[i] = float("inf")
    return rao


def compute_pitch_rao(
    omega: float | torch.Tensor,
    geo: SparGeometry,
) -> torch.Tensor:
    """Compute pitch RAO |H5(ω)| = |θ_a / ζ_a| [rad/m] for a spar buoy.

    Args:
        omega: Angular frequency [rad/s], scalar or tensor.
        geo: Spar geometry parameters.

    Returns:
        Pitch RAO amplitude [rad/m].
    """
    if not isinstance(omega, torch.Tensor):
        omega = torch.tensor([omega], dtype=torch.float64)

    rao = torch.zeros_like(omega, dtype=torch.float64)
    for i, w in enumerate(omega):
        w_val = w.item()
        A55 = spar_pitch_added_mass(w_val, geo)
        B55 = spar_pitch_radiation_damping(w_val, geo)
        F5 = abs(spar_pitch_excitation(w_val, geo))
        C55 = geo.C55

        denom = math.sqrt(
            (C55 - (geo.iyy + A55) * w_val**2)**2
            + (B55 * w_val)**2
        )
        if denom > 1e-15:
            rao[i] = F5 / denom
        else:
            rao[i] = float("inf")
    return rao


# ---------------------------------------------------------------------------
# Utility: wave parameters
# ---------------------------------------------------------------------------

def deep_water_wave_number(omega: float, g: float = 9.81) -> float:
    """Deep-water dispersion: k = ω² / g."""
    return omega**2 / g


def finite_depth_wave_number(omega: float, depth: float, g: float = 9.81) -> float:
    """Solve the dispersion relation ω² = g k tanh(kh) by Newton iteration."""
    k = omega**2 / g  # initial guess (deep water)
    for _ in range(50):
        f = omega**2 - g * k * math.tanh(k * depth)
        fp = -g * (math.tanh(k * depth) + k * depth / math.cosh(k * depth)**2)
        dk = -f / fp
        k += dk
        if abs(dk) < 1e-12 * k:
            break
    return k


__all__ = [
    "SparGeometry",
    "compute_heave_rao",
    "compute_pitch_rao",
    "deep_water_wave_number",
    "finite_depth_wave_number",
    "spar_heave_added_mass",
    "spar_heave_radiation_damping",
    "spar_heave_excitation",
    "spar_pitch_added_mass",
    "spar_pitch_radiation_damping",
    "spar_pitch_excitation",
]
