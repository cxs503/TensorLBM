"""Double-well free-energy model and explicitly named capillary-force forms."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .operators import BoundaryPolicy, central_gradient_3d, laplacian_3d


@dataclass(frozen=True)
class DoubleWellFreeEnergy:
    """``f(phi) = -A phi²/2 + B phi⁴/4 + kappa |grad phi|²/2`` model."""

    A: float
    B: float
    kappa: float

    def chemical_potential(
        self,
        phi: torch.Tensor,
        *,
        boundary: BoundaryPolicy,
        solid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``mu = -A phi + B phi³ - kappa laplacian(phi)``."""
        return -self.A * phi + self.B * phi**3 - self.kappa * laplacian_3d(
            phi, boundary=boundary, solid_mask=solid_mask
        )


def _validate_force_fields(phi: torch.Tensor, mu: torch.Tensor) -> None:
    """Reject incompatible phase fields before PyTorch can broadcast them."""
    if phi.ndim != 3 or mu.ndim != 3:
        raise ValueError("phase-field force inputs require 3-D scalar tensors shaped (z, y, x)")
    if phi.shape != mu.shape:
        raise ValueError("phase-field force inputs must have the same shape")
    if phi.device != mu.device:
        raise ValueError("phase-field force inputs must be on the same device")


def force_minus_phi_grad_mu(
    phi: torch.Tensor,
    mu: torch.Tensor,
    *,
    boundary: BoundaryPolicy,
    solid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the Korteweg force convention ``-phi grad(mu)``."""
    _validate_force_fields(phi, mu)
    grad_x, grad_y, grad_z = central_gradient_3d(mu, boundary=boundary, solid_mask=solid_mask)
    return -phi * grad_x, -phi * grad_y, -phi * grad_z


def force_mu_grad_phi(
    phi: torch.Tensor,
    mu: torch.Tensor,
    *,
    boundary: BoundaryPolicy,
    solid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the separate force convention ``mu grad(phi)``."""
    _validate_force_fields(phi, mu)
    grad_x, grad_y, grad_z = central_gradient_3d(phi, boundary=boundary, solid_mask=solid_mask)
    return mu * grad_x, mu * grad_y, mu * grad_z
