"""Common, tensorised three-dimensional phase-field building blocks."""

from .diagnostics import phase_volume_smoothed, phase_volume_threshold
from .free_energy import DoubleWellFreeEnergy, force_minus_phi_grad_mu, force_mu_grad_phi
from .operators import central_gradient_3d, laplacian_3d

__all__ = [
    "DoubleWellFreeEnergy",
    "central_gradient_3d",
    "force_minus_phi_grad_mu",
    "force_mu_grad_phi",
    "laplacian_3d",
    "phase_volume_smoothed",
    "phase_volume_threshold",
]
