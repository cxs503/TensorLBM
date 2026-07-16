"""Common, tensorised three-dimensional phase-field building blocks."""

from .diagnostics import phase_volume_smoothed, phase_volume_threshold
from .ch_validation import (
    FreeEnergyCHDiagnosticResult,
    FreeEnergyCHStepDiagnostic,
    FreeEnergyCHValidationConfig,
    run_closed_periodic_free_energy_diagnostic,
    uniform_phase_capillary_force,
)
from .free_energy import DoubleWellFreeEnergy, force_minus_phi_grad_mu, force_mu_grad_phi
from .operators import central_gradient_3d, laplacian_3d

__all__ = [
    "DoubleWellFreeEnergy",
    "FreeEnergyCHDiagnosticResult",
    "FreeEnergyCHStepDiagnostic",
    "FreeEnergyCHValidationConfig",
    "central_gradient_3d",
    "force_minus_phi_grad_mu",
    "force_mu_grad_phi",
    "laplacian_3d",
    "phase_volume_smoothed",
    "phase_volume_threshold",
    "run_closed_periodic_free_energy_diagnostic",
    "uniform_phase_capillary_force",
]
