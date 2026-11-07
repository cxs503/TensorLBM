"""Numerical wave tank (NWT) validation surface for TensorLBM.

The package couples three pieces and keeps them separate on purpose:

* :mod:`tensorlbm.nwt.stokes2` — closed-form second-order Stokes wave theory
  (the analytic anchor: dispersion relation, elevation, kinematics, potential);
* :mod:`tensorlbm.nwt.wavemaker` — relaxation-zone generation and absorption on
  the free-surface state (solver-agnostic);
* :mod:`tensorlbm.nwt.reflection` / :mod:`tensorlbm.nwt.analysis` — Goda &
  Suzuki two-point reflection separation, harmonic decomposition, phase-speed
  and mass-budget estimators;
* :mod:`tensorlbm.nwt.tank` — the 2-D flume driver on the Körner free-surface
  LBM (:func:`tensorlbm.free_surface_lbm_2d.koerner_step_2d`).

The acceptance evidence (dispersion, waveform, absorption, mass, grid
convergence, runtime) is produced by
``benchmarks/nwt_stokes2_2d/run_acceptance.py``.
"""

from .stokes2 import Stokes2Wave, dispersion_k, dispersion_omega, group_speed, phase_speed
from .analysis import fit_harmonics, inventory_drift, lag_from_cross_correlation, wavenumber_from_probes
from .reflection import (
    complex_amplitude,
    least_squares_reflection,
    two_point_reflection,
    wavenumber_by_model_fit,
)
from .wavemaker import column_free_surface, ramp_factor, relaxation_weight
from .tank import NWTConfig, NWTResult, build_geometry, run_nwt

__all__ = [
    "Stokes2Wave",
    "dispersion_k",
    "dispersion_omega",
    "group_speed",
    "phase_speed",
    "fit_harmonics",
    "inventory_drift",
    "lag_from_cross_correlation",
    "wavenumber_from_probes",
    "complex_amplitude",
    "least_squares_reflection",
    "two_point_reflection",
    "wavenumber_by_model_fit",
    "column_free_surface",
    "ramp_factor",
    "relaxation_weight",
    "NWTConfig",
    "NWTResult",
    "build_geometry",
    "run_nwt",
]
