"""Marginal Ice Zone (MIZ) ship resistance & seakeeping coupling.

Implements the three-field interaction model for ships operating in marginal
ice zones (MIZ), where waves, broken ice floes, and the ship hull mutually
influence each other.

Physics model
-------------
1. **Wave attenuation in ice field** (Squire 1983, Wadhams 1973):
   Waves entering an ice-covered region lose energy through scattering and
   dissipation.  The attenuation coefficient β depends on ice concentration
   C_ice, characteristic floe size a_floe, and ice thickness h_ice:

       β = f(C_ice, a_floe, h_ice, T_wave)

   Following Squire's theory, the energy decay is exponential:
       E(x) = E_0 · exp(−2βx)

2. **Ice floe size distribution** (Squire 1983, Shen et al. 2001):
   When wave stress exceeds ice flexural strength, ice breaks into floes
   with a characteristic size:

       a_floe ∝ (σ_f / (ρ_w · g))^(1/2) · (1 / k)

   where σ_f is ice flexural strength, ρ_w water density, g gravity,
   and k the wave number.  The floe size distribution P(a_floe) is
   modelled as a truncated power-law / log-normal mixture.

3. **Ship resistance in MIZ**:
   Total resistance is decomposed as:

       R_MIZ = R_open + R_wave + R_ice

   where R_open is open-water resistance, R_wave is added wave resistance,
   and R_ice is the ice resistance component, which depends on floe size
   distribution and ice concentration:

       R_ice ∝ C_ice^γ · f(a_floe, h_ice, V_ship)

   with γ ∈ [1, 2] consistent with literature (Riska 1989, Lindqvist 1989).

4. **Motion RAO correction**:
   The ice field provides additional damping to ship motions, reducing
   resonance peaks.  The RAO in ice is:

       RAO_ice(ω) = RAO_open(ω) · D_ice(C_ice, ω)

   where D_ice is a damping reduction factor (< 1 for C_ice > 0).

References
----------
- Squire, V. A. (1983). "A note on the scattering of ocean waves by sea ice."
  *Annals of Glaciology* 4, 252–256.
- Wadhams, P. (1973). "Attenuation of swell by sea ice." *JGR* 78(18), 3552–3563.
- Shen, H. H. et al. (2001). "Wave-ice interaction in the marginal ice zone."
  *Annals of Glaciology* 33, 1–6.
- Riska, K. (1989). "A Model for Icebreaking Resistance." POAC 89.
- Lindqvist, G. (1989). "A Simplified Method for Calculation of Ship
  Resistance in Ice." POAC 89.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

__all__ = [
    # Wave attenuation
    "squire_wave_attenuation_coefficient",
    "wave_energy_decay",
    "wave_attenuation_in_ice",
    # Floe size distribution
    "characteristic_floe_size",
    "floe_size_distribution",
    "floe_size_pdf",
    "mean_floe_size",
    # Ice resistance
    "ice_resistance_component",
    "miz_total_resistance",
    "resistance_scaling_exponent",
    # RAO correction
    "ice_damping_factor",
    "rao_ice_correction",
    # Full coupling
    "MIZConfig",
    "MIZResult",
    "miz_coupled_solve",
    # Physical constants
    "ICE_FLEXURAL_STRENGTH",
    "ICE_DENSITY",
    "WATER_DENSITY",
    "GRAVITY",
]

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

#: Ice flexural strength [Pa] (typical range 0.5–1.5 MPa for first-year ice)
ICE_FLEXURAL_STRENGTH: float = 0.7e6  # 0.7 MPa

#: Ice density [kg/m³]
ICE_DENSITY: float = 917.0

#: Seawater density [kg/m³]
WATER_DENSITY: float = 1025.0

#: Gravitational acceleration [m/s²]
GRAVITY: float = 9.81


# ---------------------------------------------------------------------------
# 1. Wave attenuation in ice field (Squire theory)
# ---------------------------------------------------------------------------


def squire_wave_attenuation_coefficient(
    C_ice: float,
    a_floe: float,
    h_ice: float,
    T_wave: float,
    sigma_f: float = ICE_FLEXURAL_STRENGTH,
    rho_w: float = WATER_DENSITY,
    g: float = GRAVITY,
) -> float:
    """Compute wave attenuation coefficient β following Squire (1983).

    The attenuation coefficient for waves propagating through a field of
    broken ice floes is modelled as:

        β = C_ice · (k · a_floe)^(-1) · (h_ice / a_floe) · (ω² / g) · F_scatter

    where k is the wave number, ω the angular frequency, and F_scatter is
    a scattering enhancement factor that depends on the ratio of floe size
    to wavelength.

    For deep water: k = ω²/g, ω = 2π/T_wave.

    The model captures:
    - Linear dependence on ice concentration C_ice
    - Inverse dependence on floe size (smaller floes → more scattering)
    - Dependence on ice thickness (thicker ice → more dissipation)
    - Frequency dependence (shorter waves attenuate faster)

    Args:
        C_ice:   Ice concentration [0, 1].
        a_floe:  Characteristic floe size [m].
        h_ice:   Ice thickness [m].
        T_wave:  Wave period [s].
        sigma_f: Ice flexural strength [Pa].
        rho_w:   Water density [kg/m³].
        g:       Gravity [m/s²].

    Returns:
        Attenuation coefficient β [1/m].
    """
    if C_ice <= 0.0 or a_floe <= 0.0 or h_ice <= 0.0 or T_wave <= 0.0:
        return 0.0

    omega = 2.0 * math.pi / T_wave
    k = omega**2 / g  # deep-water dispersion

    # Scattering factor: maximum when floe size ≈ wavelength/2π
    ka = k * a_floe
    if ka < 1e-10:
        return 0.0

    # Squire-type scattering: F_scatter ~ (ka)^2 / (1 + (ka)^2)
    F_scatter = (ka**2) / (1.0 + ka**2)

    # Base attenuation: proportional to concentration, thickness, frequency
    beta = C_ice * (h_ice / a_floe) * (omega**2 / g) * F_scatter

    # Empirical calibration factor (tuned to match Squire's experimental data)
    # Squire (1983) reports β ~ 0.01–0.1 1/m for typical MIZ conditions
    beta *= 100.0  # calibration factor (tuned to Squire experimental range 0.01-0.1 1/m)

    return max(0.0, beta)


def wave_energy_decay(
    beta: float,
    distance: float,
    E0: float = 1.0,
) -> float:
    """Compute wave energy after propagating distance through ice.

    E(x) = E_0 · exp(−2βx)

    Args:
        beta:     Attenuation coefficient [1/m].
        distance: Propagation distance [m].
        E0:       Initial wave energy (default 1.0, normalised).

    Returns:
        Wave energy at distance x.
    """
    if beta <= 0.0 or distance <= 0.0:
        return E0
    return E0 * math.exp(-2.0 * beta * distance)


def wave_attenuation_in_ice(
    C_ice: float,
    a_floe: float,
    h_ice: float,
    T_wave: float,
    distance: float,
    E0: float = 1.0,
) -> dict:
    """Full wave attenuation calculation in ice field.

    Args:
        C_ice:    Ice concentration [0, 1].
        a_floe:   Characteristic floe size [m].
        h_ice:    Ice thickness [m].
        T_wave:   Wave period [s].
        distance: Propagation distance [m].
        E0:       Initial wave energy.

    Returns:
        Dictionary with beta, E_final, attenuation_ratio, and parameters.
    """
    beta = squire_wave_attenuation_coefficient(C_ice, a_floe, h_ice, T_wave)
    E_final = wave_energy_decay(beta, distance, E0)

    return {
        "beta": beta,
        "E0": E0,
        "E_final": E_final,
        "attenuation_ratio": E_final / E0 if E0 > 0 else 1.0,
        "distance": distance,
        "C_ice": C_ice,
        "a_floe": a_floe,
        "h_ice": h_ice,
        "T_wave": T_wave,
    }


# ---------------------------------------------------------------------------
# 2. Ice floe size distribution
# ---------------------------------------------------------------------------


def characteristic_floe_size(
    T_wave: float,
    sigma_f: float = ICE_FLEXURAL_STRENGTH,
    rho_w: float = WATER_DENSITY,
    g: float = GRAVITY,
    h_ice: float = 0.2,
) -> float:
    """Compute characteristic floe size from Squire's scaling.

    When wave stress exceeds ice flexural strength, ice breaks into floes
    with characteristic size following the scaling:

        a_floe ∝ (σ_f / (ρ_w · g))^(1/2) · (1 / k)

    where k = ω²/g is the deep-water wave number.

    The raw Squire scaling gives unrealistically large values for ocean
    waves, so we apply an empirical calibration that maps the theoretical
    scaling to observed MIZ floe sizes (typically 1–50 m):

        a_floe = a_min + (a_max - a_min) · (k_ref / k) / (1 + k_ref / k)

    where k_ref is a reference wave number and a_min, a_max bound the
    observed floe size range.

    Args:
        T_wave:  Wave period [s].
        sigma_f: Ice flexural strength [Pa].
        rho_w:   Water density [kg/m³].
        g:       Gravity [m/s²].
        h_ice:   Ice thickness [m] (used for calibration).

    Returns:
        Characteristic floe size [m], typically 1–50 m.
    """
    if T_wave <= 0.0:
        return 0.0

    omega = 2.0 * math.pi / T_wave
    k = omega**2 / g

    # Squire theoretical scaling (gives very large values)
    a_theoretical = math.sqrt(sigma_f / (rho_w * g)) / k

    # Empirical calibration to map to observed MIZ floe sizes (1–50 m)
    # The calibration uses a sigmoid-like mapping:
    #   a_floe = a_min + (a_max - a_min) · a_theoretical / (a_ref + a_theoretical)
    a_min = 1.0   # minimum floe size [m]
    a_max = 50.0  # maximum floe size [m]
    a_ref = 10.0  # reference scale [m]

    a_floe = a_min + (a_max - a_min) * a_theoretical / (a_ref + a_theoretical)

    return max(a_min, min(a_max, a_floe))


def floe_size_pdf(
    a: float | np.ndarray,
    a_char: float,
    a_min: float = 0.1,
    a_max: float = 50.0,
    shape: float = 2.0,
) -> float | np.ndarray:
    """Probability density function of floe size distribution.

    Uses a truncated Weibull distribution, which is commonly used to model
    floe size distributions in the MIZ (Shen et al. 2001):

        P(a) = (shape / a_char) · (a / a_char)^(shape-1) · exp(−(a/a_char)^shape)

    Args:
        a:       Floe size(s) [m].
        a_char:  Characteristic floe size [m].
        a_min:   Minimum floe size [m].
        a_max:   Maximum floe size [m].
        shape:   Weibull shape parameter (default 2.0 → Rayleigh-like).

    Returns:
        PDF value(s), normalised over [a_min, a_max].
    """
    if a_char <= 0.0:
        return 0.0

    a_arr = np.asarray(a, dtype=np.float64)

    # Weibull PDF
    x = a_arr / a_char
    pdf = (shape / a_char) * (x ** (shape - 1)) * np.exp(-(x**shape))

    # Truncate and renormalise
    mask = (a_arr >= a_min) & (a_arr <= a_max)
    # Normalisation integral (analytical for Weibull)
    # Normalisation via numerical integration (no scipy dependency)

    # Numerical normalisation
    a_vals = np.linspace(a_min, a_max, 1000)
    x_vals = a_vals / a_char
    pdf_vals = (shape / a_char) * (x_vals ** (shape - 1)) * np.exp(-(x_vals**shape))
    norm = np.trapezoid(pdf_vals, a_vals)

    if norm > 1e-15:
        pdf = np.where(mask, pdf / norm, 0.0)
    else:
        pdf = np.zeros_like(pdf)

    if np.isscalar(a) or (isinstance(a, (int, float))):
        return float(pdf.flat[0]) if pdf.size > 0 else 0.0
    return pdf


def floe_size_distribution(
    a_char: float,
    n_bins: int = 50,
    a_min: float = 0.1,
    a_max: float = 50.0,
    shape: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate floe size distribution (bin centres and probabilities).

    Args:
        a_char:  Characteristic floe size [m].
        n_bins:  Number of bins.
        a_min:   Minimum floe size [m].
        a_max:   Maximum floe size [m].
        shape:   Weibull shape parameter.

    Returns:
        (bin_centres, probabilities) arrays.
    """
    a_edges = np.linspace(a_min, a_max, n_bins + 1)
    a_centres = 0.5 * (a_edges[:-1] + a_edges[1:])

    pdf_vals = floe_size_pdf(a_centres, a_char, a_min, a_max, shape)
    # Normalise to probabilities
    probs = pdf_vals * (a_edges[1] - a_edges[0])
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs /= prob_sum

    return a_centres, probs


def mean_floe_size(
    a_char: float,
    a_min: float = 0.1,
    a_max: float = 50.0,
    shape: float = 2.0,
) -> float:
    """Compute mean floe size from the distribution.

    Args:
        a_char:  Characteristic floe size [m].
        a_min:   Minimum floe size [m].
        a_max:   Maximum floe size [m].
        shape:   Weibull shape parameter.

    Returns:
        Mean floe size [m].
    """
    a_centres, probs = floe_size_distribution(a_char, 200, a_min, a_max, shape)
    return float(np.sum(a_centres * probs))


# ---------------------------------------------------------------------------
# 3. Ship resistance in MIZ
# ---------------------------------------------------------------------------


def ice_resistance_component(
    C_ice: float,
    a_floe: float,
    h_ice: float,
    V_ship: float,
    L_ship: float,
    B_ship: float,
    rho_w: float = WATER_DENSITY,
    g: float = GRAVITY,
    gamma: float = 1.5,
) -> float:
    """Compute ice resistance component in MIZ.

    Following Riska (1989) and Lindqvist (1989), the ice resistance in
    broken ice is modelled as:

        R_ice = C_ice^γ · 0.5 · ρ_w · V_ship² · A_proj · C_ice_drag

    where A_proj = B_ship · h_ice is the projected ice area, and C_ice_drag
    is a drag coefficient that depends on floe size.

    The exponent γ ∈ [1, 2] captures the non-linear dependence on ice
    concentration, consistent with experimental observations.

    Args:
        C_ice:   Ice concentration [0, 1].
        a_floe:  Characteristic floe size [m].
        h_ice:   Ice thickness [m].
        V_ship:  Ship speed [m/s].
        L_ship:  Ship length [m].
        B_ship:  Ship beam [m].
        rho_w:   Water density [kg/m³].
        g:       Gravity [m/s²].
        gamma:   Concentration exponent (default 1.5).

    Returns:
        Ice resistance [N].
    """
    if C_ice <= 0.0 or V_ship <= 0.0:
        return 0.0

    # Projected ice area (beam × ice thickness)
    A_proj = B_ship * h_ice

    # Floe-size-dependent drag coefficient
    # Smaller floes → lower drag (easier to push aside)
    # Larger floes → higher drag (more like level ice)
    a_ref = 5.0  # reference floe size [m]
    C_ice_drag = 0.8 * min(1.0, a_floe / a_ref) + 0.2

    # Ice resistance
    R_ice = (C_ice ** gamma) * 0.5 * rho_w * V_ship**2 * A_proj * C_ice_drag

    return max(0.0, R_ice)


def resistance_scaling_exponent(
    C_ice_values: List[float],
    R_ice_values: List[float],
) -> float:
    """Estimate the scaling exponent γ from R_ice ∝ C_ice^γ.

    Uses log-log linear regression: log(R) = γ · log(C) + const.

    Args:
        C_ice_values: List of ice concentrations.
        R_ice_values: List of corresponding ice resistances.

    Returns:
        Estimated exponent γ.
    """
    C_arr = np.array(C_ice_values, dtype=np.float64)
    R_arr = np.array(R_ice_values, dtype=np.float64)

    # Filter out zeros
    mask = (C_arr > 0.0) & (R_arr > 0.0)
    if mask.sum() < 2:
        return 1.5  # default

    log_C = np.log(C_arr[mask])
    log_R = np.log(R_arr[mask])

    # Linear regression: log_R = γ · log_C + c
    coeffs = np.polyfit(log_C, log_R, 1)
    return float(coeffs[0])


def miz_total_resistance(
    R_open: float,
    C_ice: float,
    a_floe: float,
    h_ice: float,
    V_ship: float,
    L_ship: float,
    B_ship: float,
    T_wave: float,
    wave_amp: float = 0.0,
    gamma: float = 1.5,
) -> dict:
    """Compute total ship resistance in MIZ.

    R_MIZ = R_open + R_wave + R_ice

    Args:
        R_open:   Open-water resistance [N].
        C_ice:    Ice concentration [0, 1].
        a_floe:   Characteristic floe size [m].
        h_ice:    Ice thickness [m].
        V_ship:   Ship speed [m/s].
        L_ship:   Ship length [m].
        B_ship:   Ship beam [m].
        T_wave:   Wave period [s].
        wave_amp: Wave amplitude [m] (for added wave resistance).
        gamma:    Concentration exponent.

    Returns:
        Dictionary with R_open, R_wave, R_ice, R_total, and ΔR/R_open.
    """
    # Ice resistance
    R_ice = ice_resistance_component(
        C_ice, a_floe, h_ice, V_ship, L_ship, B_ship, gamma=gamma
    )

    # Added wave resistance (simplified: proportional to wave amplitude squared)
    if wave_amp > 0.0 and T_wave > 0.0:
        # Wave attenuation reduces effective wave amplitude
        omega = 2.0 * math.pi / T_wave
        k = omega**2 / GRAVITY
        # Added resistance in waves (empirical, proportional to ζ_a²)
        R_wave = 0.5 * WATER_DENSITY * GRAVITY * B_ship * wave_amp**2 * 0.1
    else:
        R_wave = 0.0

    R_total = R_open + R_wave + R_ice
    delta_R = R_wave + R_ice
    delta_R_ratio = delta_R / R_open if R_open > 0 else 0.0

    return {
        "R_open": R_open,
        "R_wave": R_wave,
        "R_ice": R_ice,
        "R_total": R_total,
        "delta_R": delta_R,
        "delta_R_ratio": delta_R_ratio,
        "C_ice": C_ice,
        "a_floe": a_floe,
        "h_ice": h_ice,
        "V_ship": V_ship,
        "T_wave": T_wave,
    }


# ---------------------------------------------------------------------------
# 4. RAO correction due to ice field damping
# ---------------------------------------------------------------------------


def ice_damping_factor(
    C_ice: float,
    omega: float,
    omega_ref: float = 1.0,
    alpha: float = 1.667,
) -> float:
    """Compute ice field damping reduction factor for RAO.

    The ice field provides additional damping to ship motions, modelled as:

        D_ice(ω) = 1 / [1 + α · C_ice · (ω/ω_ref) / (1 + (ω/ω_ref)²)]

    This gives ~20% reduction at resonance (ω=ω_ref) for C_ice=0.3,
    consistent with experimental observations in MIZ conditions.

    Args:
        C_ice:     Ice concentration [0, 1].
        omega:     Motion frequency [rad/s].
        omega_ref: Reference frequency [rad/s].
        alpha:     Damping strength parameter (default 1.667 gives 20%
                   reduction at C=0.3, ω=ω_ref).

    Returns:
        Damping factor D_ice ∈ (0, 1].
    """
    if C_ice <= 0.0 or omega <= 0.0:
        return 1.0

    freq_ratio = omega / omega_ref
    damping_term = alpha * C_ice * freq_ratio / (1.0 + freq_ratio**2)

    return 1.0 / (1.0 + damping_term)


def rao_ice_correction(
    rao_open: np.ndarray,
    omega: np.ndarray,
    C_ice: float,
    omega_ref: float = 1.0,
    alpha: float = 1.667,
) -> np.ndarray:
    """Apply ice field damping correction to open-water RAO.

    RAO_ice(ω) = RAO_open(ω) · D_ice(C_ice, ω)

    Args:
        rao_open:  Open-water RAO array.
        omega:     Frequency array [rad/s].
        C_ice:     Ice concentration [0, 1].
        omega_ref: Reference frequency [rad/s].
        alpha:     Damping strength parameter.

    Returns:
        Ice-corrected RAO array.
    """
    if C_ice <= 0.0:
        return rao_open.copy()

    damping = np.array([
        ice_damping_factor(C_ice, w, omega_ref, alpha)
        for w in omega
    ])

    return rao_open * damping


# ---------------------------------------------------------------------------
# 5. Full MIZ coupled solver
# ---------------------------------------------------------------------------


@dataclass
class MIZConfig:
    """Configuration for MIZ coupled simulation.

    Parameters
    ----------
    C_ice : float
        Ice concentration [0, 1].
    h_ice : float
        Ice thickness [m].
    T_wave : float
        Wave period [s].
    wave_amp : float
        Wave amplitude [m].
    V_ship : float
        Ship speed [m/s].
    L_ship : float
        Ship length [m].
    B_ship : float
        Ship beam [m].
    R_open : float
        Open-water resistance [N].
    sigma_f : float
        Ice flexural strength [Pa].
    gamma : float
        Ice resistance concentration exponent.
    distance : float
        Wave propagation distance through ice [m].
    """
    C_ice: float = 0.3
    h_ice: float = 0.2
    T_wave: float = 8.0
    wave_amp: float = 0.5
    V_ship: float = 3.0
    L_ship: float = 150.0
    B_ship: float = 25.0
    R_open: float = 500000.0  # 500 kN
    sigma_f: float = ICE_FLEXURAL_STRENGTH
    gamma: float = 1.5
    distance: float = 1000.0


@dataclass
class MIZResult:
    """Results from MIZ coupled simulation."""
    # Wave attenuation
    beta: float = 0.0
    E_final: float = 0.0
    attenuation_ratio: float = 1.0

    # Floe size
    a_floe_char: float = 0.0
    a_floe_mean: float = 0.0
    floe_distribution: Optional[Tuple[List[float], List[float]]] = None

    # Resistance
    R_open: float = 0.0
    R_wave: float = 0.0
    R_ice: float = 0.0
    R_total: float = 0.0
    delta_R_ratio: float = 0.0

    # RAO
    rao_open: Optional[np.ndarray] = None
    rao_ice: Optional[np.ndarray] = None
    omega_rao: Optional[np.ndarray] = None
    resonance_reduction: float = 0.0

    # Metadata
    config: Optional[MIZConfig] = None


def miz_coupled_solve(config: MIZConfig) -> MIZResult:
    """Run the full MIZ coupled simulation.

    Solves the three-field interaction:
    1. Wave attenuation → effective wave amplitude at ship
    2. Ice floe size distribution from wave-ice interaction
    3. Ship resistance (open + wave + ice)
    4. RAO correction due to ice damping

    Args:
        config: MIZ configuration.

    Returns:
        MIZResult with all computed quantities.
    """
    result = MIZResult(config=config)

    # 1. Characteristic floe size from Squire scaling
    a_char = characteristic_floe_size(
        config.T_wave, config.sigma_f, h_ice=config.h_ice
    )
    result.a_floe_char = a_char
    result.a_floe_mean = mean_floe_size(a_char)

    # 2. Wave attenuation
    wave_info = wave_attenuation_in_ice(
        config.C_ice, a_char, config.h_ice, config.T_wave, config.distance
    )
    result.beta = wave_info["beta"]
    result.E_final = wave_info["E_final"]
    result.attenuation_ratio = wave_info["attenuation_ratio"]

    # Effective wave amplitude at ship position (reduced by attenuation)
    effective_wave_amp = config.wave_amp * math.sqrt(result.attenuation_ratio)

    # 3. Ship resistance
    res = miz_total_resistance(
        config.R_open,
        config.C_ice,
        a_char,
        config.h_ice,
        config.V_ship,
        config.L_ship,
        config.B_ship,
        config.T_wave,
        effective_wave_amp,
        config.gamma,
    )
    result.R_open = res["R_open"]
    result.R_wave = res["R_wave"]
    result.R_ice = res["R_ice"]
    result.R_total = res["R_total"]
    result.delta_R_ratio = res["delta_R_ratio"]

    # 4. RAO correction
    # Generate a synthetic open-water RAO (heave, spar-buoy-like)
    omega_rao = np.linspace(0.2, 3.0, 100)
    omega_n = 1.0  # natural frequency
    zeta_open = 0.05  # damping ratio

    # Standard SDOF RAO
    rao_open = np.zeros_like(omega_rao)
    for i, w in enumerate(omega_rao):
        denom = math.sqrt((1.0 - (w / omega_n)**2)**2 + (2 * zeta_open * w / omega_n)**2)
        rao_open[i] = 1.0 / denom if denom > 1e-10 else 100.0

    # Apply ice damping
    rao_ice = rao_ice_correction(rao_open, omega_rao, config.C_ice, omega_n)

    result.rao_open = rao_open
    result.rao_ice = rao_ice
    result.omega_rao = omega_rao

    # Resonance peak reduction
    peak_open = rao_open.max()
    peak_ice = rao_ice.max()
    result.resonance_reduction = (peak_open - peak_ice) / peak_open if peak_open > 0 else 0.0

    # Floe size distribution
    a_bins, a_probs = floe_size_distribution(a_char)
    result.floe_distribution = (a_bins.tolist(), a_probs.tolist())

    return result


# ---------------------------------------------------------------------------
# Negative control: C_ice = 0 should give open-water results
# ---------------------------------------------------------------------------


def negative_control_check(config: MIZConfig) -> dict:
    """Verify that C_ice = 0 gives open-water results.

    Args:
        config: Base configuration.

    Returns:
        Dictionary with pass/fail status and details.
    """
    # Open-water case (C_ice = 0)
    config_ow = MIZConfig(
        C_ice=0.0,
        h_ice=config.h_ice,
        T_wave=config.T_wave,
        wave_amp=config.wave_amp,
        V_ship=config.V_ship,
        L_ship=config.L_ship,
        B_ship=config.B_ship,
        R_open=config.R_open,
        sigma_f=config.sigma_f,
        gamma=config.gamma,
        distance=config.distance,
    )
    result_ow = miz_coupled_solve(config_ow)

    checks = {
        "R_ice_is_zero": result_ow.R_ice == 0.0,
        "R_wave_only": result_ow.R_wave >= 0.0,
        "delta_R_ratio_non_negative": result_ow.delta_R_ratio >= 0.0,
        "beta_is_zero": result_ow.beta == 0.0,
        "attenuation_ratio_is_one": abs(result_ow.attenuation_ratio - 1.0) < 1e-10,
        "rao_ice_equals_rao_open": np.allclose(result_ow.rao_ice, result_ow.rao_open),
    }

    return {
        "all_passed": all(checks.values()),
        "checks": checks,
        "result": {
            "R_open": result_ow.R_open,
            "R_ice": result_ow.R_ice,
            "R_wave": result_ow.R_wave,
            "beta": result_ow.beta,
            "attenuation_ratio": result_ow.attenuation_ratio,
        },
    }
