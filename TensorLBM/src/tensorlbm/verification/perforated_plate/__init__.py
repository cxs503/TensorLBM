"""Perforated-plate oscillatory-flow verification face (TensorLBM).

Physics target
--------------
Forced oscillation of a viscous fluid through a sharp-edged perforated plate in
a 2-D channel, driven by a sinusoidal body force ``F_x = F0 sin(omega t)``:

* plate geometry: ``n_holes`` rectangular holes of height ``d`` in a plate of
  thickness ``t``, open-area ratio ``beta = n_holes d / (ny - 2)``
* numerical method: D2Q9 BGK, Guo forcing, half-way bounce-back on all solid
  links, momentum-exchange (Ladd) force measurement
* reduced parameters: ``KC = u_h T / d`` (hole-based Keulegan-Carpenter),
  ``Re_h = u_h d / nu``, ``beta``

Deliverables
------------
* ``C_d(KC, beta)`` - discharge coefficient from the unsteady pressure drop
  ``dp = rho K_m du_h/dt + 0.5 rho K_q u_h |u_h|`` with ``K_q = 1/C_d^2``
* ``C_m(KC, beta) = K_m / t`` - added-mass coefficient
* ``Phi`` - energy dissipated per cycle, from two independent paths
  (momentum exchange on the plate and pressure stations in the slab budget)
* ``zeta_eff = Phi / (2 pi omega * 2 E_kin)`` - equivalent sloshing damping and
  its quasi-steady analytic counterpart

Reference: Borda-Carnot sharp-edge orifice model ``C_d = C_c/(1 - beta C_c)``
with ``C_c = 0.61`` (see :mod:`reference`).
"""

from __future__ import annotations

from .solver import (
    PerforatedPlateConfig,
    PerforatedPlateSolver,
    compute_open_area_ratio,
    make_perforated_plate_mask,
    run_perforated_plate,
)
from .analysis import (
    analyze_simulation,
    analytic_dissipation,
    compute_energy_dissipation,
    compute_phase_lag,
    compute_zeta_eff,
    extract_cd_cm,
    first_harmonic,
    kinetic_energy,
    phase_average,
)
from .reference import (
    C_C_SHARP,
    idelchik_cd_quasi_steady,
    loss_coefficient,
    quasi_steady_pressure_drop,
    sharp_edge_cd,
    stokes_layer_thickness,
    stokes_layer_u,
    stokes_layer_wall_shear,
    stokes_oscillating_wall_shear,
    stokes_wall_shear_stress,
    womersley_channel_amplitude,
    womersley_channel_u,
    womersley_wall_shear,
    womersley_wall_shear_amplitude,
)
from .gates import (
    THRESHOLDS,
    evaluate_all,
    gate_A_quasi_steady,
    gate_B_phase_monotonic,
    gate_C_cd_vs_beta,
    gate_D_gci,
    gate_E_negative_controls,
    gate_F_energy_closure,
    spearman,
)

#: backward-compatible alias of :data:`THRESHOLDS`
GATE_THRESHOLDS = THRESHOLDS

__all__ = [
    "PerforatedPlateConfig", "PerforatedPlateSolver", "run_perforated_plate",
    "make_perforated_plate_mask", "compute_open_area_ratio",
    "analyze_simulation", "extract_cd_cm", "compute_energy_dissipation",
    "compute_zeta_eff", "compute_phase_lag", "phase_average", "first_harmonic",
    "kinetic_energy", "analytic_dissipation",
    "sharp_edge_cd", "C_C_SHARP", "idelchik_cd_quasi_steady",
    "loss_coefficient", "quasi_steady_pressure_drop",
    "stokes_layer_thickness", "stokes_layer_u", "stokes_layer_wall_shear",
    "stokes_oscillating_wall_shear", "stokes_wall_shear_stress",
    "womersley_channel_u", "womersley_channel_amplitude", "womersley_wall_shear",
    "womersley_wall_shear_amplitude",
    "THRESHOLDS", "GATE_THRESHOLDS", "spearman", "evaluate_all",
    "gate_A_quasi_steady", "gate_B_phase_monotonic", "gate_C_cd_vs_beta",
    "gate_D_gci", "gate_E_negative_controls", "gate_F_energy_closure",
]