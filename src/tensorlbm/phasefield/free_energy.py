"""Double-well free-energy model and explicitly named capillary-force forms.

Two finite-difference operator families are selectable, because the choice is
*not* cosmetic for a curved interface:

``central7``
    seven-point Laplacian + two-point central gradient
    (:mod:`tensorlbm.phasefield.operators`); second-order accurate but with an
    anisotropic leading truncation term.
``isotropic_d3q19``
    D3Q19-stencil isotropic Laplacian + isotropic gradient
    (:mod:`tensorlbm.phasefield.isotropic_fd`), the operators used by the
    lbmpy Allen-Cahn/CSF force chain.  On a static droplet this removes the
    spurious tangential capillary force of the seven-point family; measured in
    ``docs/XVAL_M17410_ALLEN_CAHN_CSF_FORCE_CHAIN.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .isotropic_fd import isotropic_gradient_3d, isotropic_laplacian_3d
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
        operator: FDOperator = "central7",
    ) -> torch.Tensor:
        """Return ``mu = -A phi + B phi³ - kappa laplacian(phi)``."""
        return (
            -self.A * phi
            + self.B * phi**3
            - self.kappa * _laplacian(phi, operator, boundary, solid_mask)
        )


#: Finite-difference operator families accepted by the force forms below.
FDOperator = Literal["central7", "isotropic_d3q19"]


def _laplacian(
    field: torch.Tensor,
    operator: FDOperator,
    boundary: BoundaryPolicy,
    solid_mask: torch.Tensor | None,
) -> torch.Tensor:
    if operator == "central7":
        return laplacian_3d(field, boundary=boundary, solid_mask=solid_mask)
    if operator == "isotropic_d3q19":
        return isotropic_laplacian_3d(field, boundary=boundary, solid_mask=solid_mask)
    raise ValueError(f"unknown fd operator {operator!r}")


def _gradient(
    field: torch.Tensor,
    operator: FDOperator,
    boundary: BoundaryPolicy,
    solid_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if operator == "central7":
        return central_gradient_3d(field, boundary=boundary, solid_mask=solid_mask)
    if operator == "isotropic_d3q19":
        return isotropic_gradient_3d(field, boundary=boundary, solid_mask=solid_mask)
    raise ValueError(f"unknown fd operator {operator!r}")


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
    operator: FDOperator = "central7",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the Korteweg force convention ``-phi grad(mu)``."""
    _validate_force_fields(phi, mu)
    grad_x, grad_y, grad_z = _gradient(mu, operator, boundary, solid_mask)
    return -phi * grad_x, -phi * grad_y, -phi * grad_z


def force_mu_grad_phi(
    phi: torch.Tensor,
    mu: torch.Tensor,
    *,
    boundary: BoundaryPolicy,
    solid_mask: torch.Tensor | None = None,
    operator: FDOperator = "central7",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the separate force convention ``mu grad(phi)``."""
    _validate_force_fields(phi, mu)
    grad_x, grad_y, grad_z = _gradient(phi, operator, boundary, solid_mask)
    return mu * grad_x, mu * grad_y, mu * grad_z
