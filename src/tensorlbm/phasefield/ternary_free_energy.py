"""Ternary (three-component) double-well free-energy model with harmonic lambda.

Cross-validation port of **AminZar69/CudaPhaseField** (Zarareh et al. 2022,
*Computers & Mathematics with Applications* 109, 100–124,
DOI 10.1016/j.camwa.2022.01.005).

Physics / conventions
---------------------
Three phase fields ``phi1, phi2, phi3`` with the constraint ``phi1 + phi2 + phi3 = 1``.
The bulk free energy for each component uses a **double-well** potential::

    f_dw(phi) = phi^2 * (1 - phi)^2

whose derivative is ``phi * (1 - phi) * (1 - 2 * phi)``.

The interfacial tension between components ``i`` and ``j`` is ``sigma(i,j)``.
The **lambda** coefficients are derived from the tension matrix::

    lambda(i) = sigma(j,i) + sigma(k,i) - sigma(j,k)    (cyclic i,j,k)

and the **harmonic mean** lambda is::

    lambda_t = 3 / (1/lambda(1) + 1/lambda(2) + 1/lambda(3))

The chemical potential for component ``i`` (Zarareh et al. 2022, Eq. in
``chemical_potential`` kernel of CudaPhaseField)::

    mu_i = (12/w) * [ lambda(i) * phi_i * (1-phi_i) * (1-2*phi_i)
                     - 2 * lambda_t * phi_1 * phi_2 * (1 - phi_1 - phi_2) ]
           - (3/4) * w * lambda(i) * laplacian(phi_i)

where ``w`` is the interface width (in lattice units).

The Laplacian uses the **D2Q9 isotropic 9-point stencil** (same as CudaPhaseField)::

    laplacian(phi) = [ sum_diag(phi) + 4*sum_cardinal(phi) - 20*phi ] / 6

Gradient uses the **isotropic 8-point stencil**::

    dphi/dx = (phi(x+1,y) - phi(x-1,y))/3 + (phi(x+1,y-1)+phi(x+1,y+1)
              - phi(x-1,y-1) - phi(x-1,y+1))/12
    dphi/dy = (phi(x,y+1) - phi(x,y-1))/3 + (phi(x-1,y+1)+phi(x+1,y+1)
              - phi(x-1,y-1) - phi(x+1,y-1))/12

Reference
---------
Zarareh A, Burnside SB, Khajepor S, Chen B. "Improving the staircase approximation
for wettability implementation of phase-field model: Part 2–Three-component
permeation." *Computers & Mathematics with Applications*, 109, 100-124, 2022.
DOI: https://doi.org/10.1016/j.camwa.2022.01.005

OSS source: https://github.com/AminZar69/CudaPhaseField
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

__all__ = [
    "TernaryDoubleWellFreeEnergy",
    "isotropic_laplacian_2d",
    "isotropic_gradient_2d",
    "lambda_from_sigma",
    "harmonic_lambda",
    "planar_profile_ternary",
]


def isotropic_laplacian_2d(phi: torch.Tensor) -> torch.Tensor:
    """D2Q9 isotropic 9-point Laplacian (periodic).

    Matches CudaPhaseField ``global_subroutines.f90`` line::

        d2phi = ( s(tx-1,ty-1)+s(tx+1,ty-1)+s(tx-1,ty+1)+s(tx+1,ty+1)
                 +4*(s(tx,ty-1)+s(tx-1,ty)+s(tx+1,ty)+s(tx,ty+1))
                 -20*s(tx,ty) ) / 6
    """
    if phi.ndim != 2:
        raise ValueError(f"phi must be 2-D, got {phi.ndim}-D")

    def _roll(dy: int, dx: int) -> torch.Tensor:
        return torch.roll(torch.roll(phi, -dy, dims=-2), -dx, dims=-1)

    diag_sum = _roll(-1, -1) + _roll(-1, 1) + _roll(1, -1) + _roll(1, 1)
    card_sum = _roll(-1, 0) + _roll(1, 0) + _roll(0, -1) + _roll(0, 1)
    return (diag_sum + 4.0 * card_sum - 20.0 * phi) / 6.0


def isotropic_gradient_2d(phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Isotropic 8-point gradient (periodic).

    Matches CudaPhaseField ``global_subroutines.f90`` ``gradient_cal`` kernel::

        dphi/dx = (phi(x+1,y)-phi(x-1,y))/3
                 + (phi(x+1,y-1)+phi(x+1,y+1)-phi(x-1,y-1)-phi(x-1,y+1))/12
        dphi/dy = (phi(x,y+1)-phi(x,y-1))/3
                 + (phi(x-1,y+1)+phi(x+1,y+1)-phi(x-1,y-1)-phi(x+1,y-1))/12

    Note: In PyTorch tensor indexing ``phi[y, x]``, ``_roll(dy, dx)`` at position
    ``(y, x)`` returns ``phi[y+dy, x+dx]``.
    """
    if phi.ndim != 2:
        raise ValueError(f"phi must be 2-D, got {phi.ndim}-D")

    def _roll(dy: int, dx: int) -> torch.Tensor:
        return torch.roll(torch.roll(phi, -dy, dims=-2), -dx, dims=-1)

    # dphi/dx: (phi(x+1,y) - phi(x-1,y))/3 + diagonal terms/12
    # _roll(0, 1) at (y,x) = phi[y, x+1]; _roll(0, -1) at (y,x) = phi[y, x-1]
    dphi_dx = (_roll(0, 1) - _roll(0, -1)) / 3.0 + (
        _roll(-1, 1) + _roll(1, 1) - _roll(-1, -1) - _roll(1, -1)
    ) / 12.0

    # dphi/dy: (phi(x,y+1) - phi(x,y-1))/3 + diagonal terms/12
    # _roll(1, 0) at (y,x) = phi[y+1, x]; _roll(-1, 0) at (y,x) = phi[y-1, x]
    dphi_dy = (_roll(1, 0) - _roll(-1, 0)) / 3.0 + (
        _roll(1, -1) + _roll(1, 1) - _roll(-1, -1) - _roll(-1, 1)
    ) / 12.0

    return dphi_dx, dphi_dy


def lambda_from_sigma(sigma: torch.Tensor) -> torch.Tensor:
    """Compute lambda(i) from the interfacial tension matrix.

    ``lambda(i) = sigma(j,i) + sigma(k,i) - sigma(j,k)`` for cyclic (i,j,k).

    Args:
        sigma: 3x3 symmetric tension matrix ``sigma[i,j] = sigma[j,i]``.

    Returns:
        3-element tensor ``[lambda(1), lambda(2), lambda(3)]``.
    """
    if sigma.shape != (3, 3):
        raise ValueError(f"sigma must be 3x3, got {sigma.shape}")
    lam = torch.empty(3, dtype=sigma.dtype, device=sigma.device)
    # Cyclic: (0,1,2), (1,2,0), (2,0,1)
    lam[0] = sigma[1, 0] + sigma[2, 0] - sigma[1, 2]
    lam[1] = sigma[0, 1] + sigma[2, 1] - sigma[0, 2]
    lam[2] = sigma[0, 2] + sigma[1, 2] - sigma[0, 1]
    return lam


def harmonic_lambda(lam: torch.Tensor) -> torch.Tensor:
    """Harmonic mean of three lambda values.

    ``lambda_t = 3 / (1/lambda(1) + 1/lambda(2) + 1/lambda(3))``
    """
    if lam.shape != (3,):
        raise ValueError(f"lam must be 3-element, got {lam.shape}")
    return 3.0 / (1.0 / lam[0] + 1.0 / lam[1] + 1.0 / lam[2])


@dataclass(frozen=True)
class TernaryDoubleWellFreeEnergy:
    """Ternary double-well free-energy model with harmonic lambda coupling.

    Matches the formulation in AminZar69/CudaPhaseField (Zarareh et al. 2022).

    Args:
        sigma: 3x3 interfacial tension matrix (symmetric, positive).
        w: Interface width in lattice units (default 4, matching OSS).
    """

    sigma: torch.Tensor
    w: float = 4.0

    def __post_init__(self) -> None:
        if self.sigma.shape != (3, 3):
            raise ValueError(f"sigma must be 3x3, got {self.sigma.shape}")
        if self.w <= 0:
            raise ValueError(f"w must be positive, got {self.w}")

    @property
    def lam(self) -> torch.Tensor:
        """Lambda coefficients derived from sigma."""
        return lambda_from_sigma(self.sigma)

    @property
    def lam_t(self) -> torch.Tensor:
        """Harmonic mean lambda."""
        return harmonic_lambda(self.lam)

    def chemical_potential(
        self,
        phi1: torch.Tensor,
        phi2: torch.Tensor,
        phi3: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute chemical potentials for all three components.

        Matches CudaPhaseField ``chemical_potential`` kernel exactly::

            mu_i = (12/w) * [ lambda(i) * phi_i * (1-phi_i) * (1-2*phi_i)
                             - 2 * lambda_t * phi_1 * phi_2 * (1 - phi_1 - phi_2) ]
                   - (3/4) * w * lambda(i) * laplacian(phi_i)

        Note: The coupling term uses ``phi_1 * phi_2 * (1 - phi_1 - phi_2)``
        for ALL three components (as in the OSS code), which equals
        ``phi_1 * phi_2 * phi_3`` due to the constraint.

        Args:
            phi1, phi2, phi3: Phase fields (2-D tensors, same shape).

        Returns:
            ``(mu1, mu2, mu3)`` chemical potential fields.
        """
        if not (phi1.shape == phi2.shape == phi3.shape):
            raise ValueError("All phase fields must have the same shape")
        if phi1.ndim != 2:
            raise ValueError(f"Phase fields must be 2-D, got {phi1.ndim}-D")

        w = float(self.w)
        lam = self.lam.to(dtype=phi1.dtype, device=phi1.device)
        lam_t = float(self.lam_t.to(dtype=phi1.dtype, device=phi1.device))

        # Coupling term: phi1 * phi2 * (1 - phi1 - phi2) = phi1 * phi2 * phi3
        phi3_from_constraint = 1.0 - phi1 - phi2
        coupling = phi1 * phi2 * phi3_from_constraint

        # Double-well derivative: phi * (1-phi) * (1-2*phi)
        def _dw_deriv(phi: torch.Tensor) -> torch.Tensor:
            return phi * (1.0 - phi) * (1.0 - 2.0 * phi)

        # Laplacian (isotropic 9-point)
        d2phi1 = isotropic_laplacian_2d(phi1)
        d2phi2 = isotropic_laplacian_2d(phi2)
        d2phi3 = isotropic_laplacian_2d(phi3)

        mu1 = (12.0 / w) * (lam[0] * _dw_deriv(phi1) - 2.0 * lam_t * coupling) - (
            3.0 / 4.0
        ) * w * lam[0] * d2phi1
        mu2 = (12.0 / w) * (lam[1] * _dw_deriv(phi2) - 2.0 * lam_t * coupling) - (
            3.0 / 4.0
        ) * w * lam[1] * d2phi2
        mu3 = (12.0 / w) * (lam[2] * _dw_deriv(phi3) - 2.0 * lam_t * coupling) - (
            3.0 / 4.0
        ) * w * lam[2] * d2phi3

        return mu1, mu2, mu3


def planar_profile_ternary(
    ny: int,
    nx: int,
    sigma: torch.Tensor | None = None,
    w: float = 4.0,
    interface_row: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a planar interface profile for ternary phase fields.

    Component 1 occupies the top half, component 3 the bottom half,
    component 2 is zero (binary slice of the ternary model).

    The profile uses a hyperbolic tangent transition matching the
    equilibrium profile of the double-well model.

    Args:
        ny, nx: Grid dimensions.
        sigma: 3x3 tension matrix (defaults to all 0.1).
        w: Interface width.
        interface_row: Row index for the interface center (default: ny//2).
        dtype: Tensor dtype.

    Returns:
        ``(phi1, phi2, phi3)`` phase field tensors.
    """
    if sigma is None:
        sigma = torch.full((3, 3), 0.1, dtype=dtype)

    if interface_row is None:
        interface_row = ny // 2

    y = torch.arange(ny, dtype=dtype).view(-1, 1).repeat(1, nx)
    # Hyperbolic tangent profile: phi1 goes from 1 (top) to 0 (bottom)
    # phi3 goes from 0 (top) to 1 (bottom)
    # Interface width w: transition over ~w cells
    xi = w / 2.0  # characteristic width
    phi1 = 0.5 * (1.0 - torch.tanh((y - interface_row) / xi))
    phi3 = 0.5 * (1.0 + torch.tanh((y - interface_row) / xi))
    phi2 = torch.zeros_like(phi1)

    # Enforce sum = 1
    total = phi1 + phi2 + phi3
    phi1 = phi1 / total
    phi2 = phi2 / total
    phi3 = phi3 / total

    return phi1, phi2, phi3
