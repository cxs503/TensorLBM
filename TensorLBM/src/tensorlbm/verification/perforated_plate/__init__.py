"""Perforated plate oscillatory flow verification face.

LBM solver for forced oscillation through a sharp-edged perforated plate,
producing discharge coefficient C_d(KC, β), added-mass coefficient C_m(KC, β),
and single-cycle energy dissipation Φ.

Physics:
    2-D channel + perforated plate (solid with hole array)
    Sinusoidal body force driving: F(t) = F₀ sin(ωt)
    Bounce-back (half-way) on solid walls and plate
    Momentum-exchange method for wall force measurement

Key non-dimensional groups:
    KC = U_max · T / D   (Keulegan–Carpenter number)
    β  = A_hole / A_total (open area ratio)
    Re = U_max · D / ν    (Reynolds number based on hole diameter)

Outputs:
    C_d(KC, β)  – discharge coefficient from quasi-steady pressure drop
    C_m(KC, β)  – added-mass coefficient from inertial pressure drop
    Φ           – energy dissipation per cycle = ∮ Δp · Q dt
    ζ_eff       – equivalent damping for sloshing: Φ / (2π ω · 2 E_kin)

Reference:
    Idelchik, Handbook of Hydraulic Resistance (quasi-steady orifice)
    Quasi-steady limit: Δp = ½ ρ (1/C_d²) (Q/(Aβ))² · sign(Q)
"""

from __future__ import annotations

from .solver import (
    PerforatedPlateConfig,
    PerforatedPlateSolver,
    make_perforated_plate_mask,
    run_perforated_plate,
)
from .analysis import (
    phase_average,
    extract_cd_cm,
    compute_energy_dissipation,
    compute_zeta_eff,
)
from .reference import (
    idelchik_cd_quasi_steady,
    quasi_steady_pressure_drop,
    stokes_oscillating_wall_shear,
)
from .gates import (
    GATE_THRESHOLDS,
    check_c1_quasi_steady,
    check_c2_phase_monotonic,
    check_c3_cd_vs_beta,
    check_c4_gci,
    check_c5_negative_controls,
    check_c6_energy_budget,
)

__all__ = [
    "PerforatedPlateConfig",
    "PerforatedPlateSolver",
    "make_perforated_plate_mask",
    "run_perforated_plate",
    "phase_average",
    "extract_cd_cm",
    "compute_energy_dissipation",
    "compute_zeta_eff",
    "idelchik_cd_quasi_steady",
    "quasi_steady_pressure_drop",
    "stokes_oscillating_wall_shear",
    "GATE_THRESHOLDS",
    "check_c1_quasi_steady",
    "check_c2_phase_monotonic",
    "check_c3_cd_vs_beta",
    "check_c4_gci",
    "check_c5_negative_controls",
    "check_c6_energy_budget",
]
