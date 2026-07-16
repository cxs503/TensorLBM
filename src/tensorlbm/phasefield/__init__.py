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
from .static_droplet import (
    DropletGeometryDiagnostic,
    KortewegForceDiagnostic,
    LaplaceStyleDiagnostic,
    StaticDropletDiagnosticResult,
    diagnose_static_droplet,
    estimate_droplet_radius,
    initialize_static_droplet,
    periodic_chemical_potential_and_korteweg_force,
)

__all__ = [
    "DoubleWellFreeEnergy",
    "DropletGeometryDiagnostic",
    "FreeEnergyCHDiagnosticResult",
    "FreeEnergyCHStepDiagnostic",
    "FreeEnergyCHValidationConfig",
    "KortewegForceDiagnostic",
    "LaplaceStyleDiagnostic",
    "StaticDropletDiagnosticResult",
    "central_gradient_3d",
    "diagnose_static_droplet",
    "estimate_droplet_radius",
    "force_minus_phi_grad_mu",
    "force_mu_grad_phi",
    "laplacian_3d",
    "initialize_static_droplet",
    "phase_volume_smoothed",
    "phase_volume_threshold",
    "periodic_chemical_potential_and_korteweg_force",
    "run_closed_periodic_free_energy_diagnostic",
    "uniform_phase_capillary_force",
]
