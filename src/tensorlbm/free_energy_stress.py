"""Free-energy pressure-tensor (Korteweg stress) body force for the binary-fluid
free-energy LBM — cross-validation port of ludwig-cf/ludwig
``FE_FORCE_METHOD_STRESS_DIVERGENCE``.

Physics / conventions (identical to ludwig ``symmetric.c``)
-----------------------------------------------------------
Bulk free energy of the symmetric (double-well) binary fluid::

    f(phi) = a/2 * phi^2 + b/4 * phi^4 + kappa/2 * |grad phi|^2      (a < 0, b > 0)

Chemical potential (ludwig ``fe_symm_mu``)::

    mu = a*phi + b*phi^3 - kappa * laplacian(phi)

Bulk (thermodynamic) pressure ``p0 = phi f'(phi) - f``
(ludwig ``fe_symm_str``: ``0.5*a*phi^2 + 0.75*b*phi^4``), generalised pressure
and pressure tensor::

    d   = p0 - kappa*phi*laplacian(phi) - kappa/2*|grad phi|^2
    P_ab = d*delta_ab + kappa * d_a phi * d_b phi

The capillary body force is the divergence of that tensor,

    F_a = -d_b P_ab

which is analytically equal to the Korteweg /chemical-potential form
``F = -phi*grad(mu)`` that :func:`tensorlbm.multiphase.free_energy_step`
currently hard-codes; the two differ only by the discretisation.

Interfacial tension of this model (ludwig ``fe_symm_interfacial_tension``
evaluates ``sqrt(-8*kappa*a^3/(9*b^2))``)::

    sigma = (2*sqrt(2)/3) * sqrt(kappa * A^3) / b,      A = -a

and the equilibrium interface width ``xi = sqrt(2*kappa/A)``.

All routines are 2-D periodic (the TensorLBM free-energy solver is 2-D); the 3-D
port only changes the gradient stencils and the divergence loop.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "FreeEnergyParams",
    "GRAD_SCHEMES",
    "grad",
    "laplacian",
    "chemical_potential",
    "bulk_pressure",
    "pressure_tensor",
    "stress_divergence_force",
    "grad_mu_force",
    "surface_tension_analytic",
    "interface_width",
    "planar_profile",
    "droplet_profile",
]

GRAD_SCHEMES = ("5pt", "iso9")


class FreeEnergyParams:
    """Free-energy parameters in ludwig's sign convention (``a < 0``)."""

    def __init__(self, A: float = 1.0, B: float = 1.0, kappa: float = 2.0):
        if A <= 0.0 or B <= 0.0 or kappa <= 0.0:
            raise ValueError("A, B, kappa must be positive (a = -A < 0)")
        self.A = float(A)
        self.B = float(B)
        self.kappa = float(kappa)

    @property
    def a(self) -> float:  # ludwig param[0]
        return -self.A

    @property
    def b(self) -> float:  # ludwig param[1]
        return self.B

    def __repr__(self) -> str:  # pragma: no cover
        return f"FreeEnergyParams(A={self.A}, B={self.B}, kappa={self.kappa})"


def _roll(field: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    return torch.roll(torch.roll(field, -dy, dims=-2), -dx, dims=-1)


def grad(field: torch.Tensor, scheme: str = "iso9") -> tuple[torch.Tensor, torch.Tensor]:
    """Periodic gradient of a 2-D field (unit lattice spacing).

    ``"5pt"``   central differences, O(h^2), anisotropic truncation.
    ``"iso9"``  isotropic 9-point (Tomita) gradient, O(h^2) with the leading
                anisotropic term cancelled -- the default fluid gradient in
                ludwig (``gradient_2d_tomita_fluid``).
    """
    if scheme == "5pt":
        gx = 0.5 * (_roll(field, 0, -1) - _roll(field, 0, 1))
        gy = 0.5 * (_roll(field, -1, 0) - _roll(field, 1, 0))
        return gx, gy
    if scheme == "iso9":
        # Tomita / isotropic: 4/12 on axis, 1/12 on diagonals (sum = 1 along x)
        gx = (
            4.0 / 12.0 * (_roll(field, 0, -1) - _roll(field, 0, 1))
            + 1.0 / 12.0 * (_roll(field, -1, -1) - _roll(field, -1, 1))
            + 1.0 / 12.0 * (_roll(field, 1, -1) - _roll(field, 1, 1))
        )
        gy = (
            4.0 / 12.0 * (_roll(field, -1, 0) - _roll(field, 1, 0))
            + 1.0 / 12.0 * (_roll(field, -1, -1) - _roll(field, 1, -1))
            + 1.0 / 12.0 * (_roll(field, -1, 1) - _roll(field, 1, 1))
        )
        return gx, gy
    raise ValueError(f"unknown gradient scheme {scheme!r}; expected one of {GRAD_SCHEMES}")


def laplacian(field: torch.Tensor) -> torch.Tensor:
    """5-point periodic Laplacian (ludwig ``field->delsq`` for the 5-pt stencil)."""
    return (
        _roll(field, 0, -1) + _roll(field, 0, 1)
        + _roll(field, -1, 0) + _roll(field, 1, 0)
        - 4.0 * field
    )


def chemical_potential(phi: torch.Tensor, p: FreeEnergyParams) -> torch.Tensor:
    """``mu = a*phi + b*phi^3 - kappa*laplacian(phi)`` (ludwig fe_symm_mu)."""
    return p.a * phi + p.b * phi ** 3 - p.kappa * laplacian(phi)


def bulk_pressure(phi: torch.Tensor, p: FreeEnergyParams) -> torch.Tensor:
    """``p0 = phi f'(phi) - f = a/2 phi^2 + 3b/4 phi^4`` (ludwig fe_symm_str)."""
    return 0.5 * p.a * phi ** 2 + 0.75 * p.b * phi ** 4


def pressure_tensor(
    phi: torch.Tensor, p: FreeEnergyParams, scheme: str = "iso9"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Free-energy pressure tensor ``P_ab`` (ludwig ``fe_symm_str``).

    Returns ``(Pxx, Pyy, Pxy)``; ``P_ab = d*delta_ab + kappa d_a phi d_b phi``
    with ``d = p0 - kappa*phi*laplacian(phi) - kappa/2*|grad phi|^2``.
    """
    gx, gy = grad(phi, scheme)
    d = (
        bulk_pressure(phi, p)
        - p.kappa * phi * laplacian(phi)
        - 0.5 * p.kappa * (gx * gx + gy * gy)
    )
    Pxx = d + p.kappa * gx * gx
    Pyy = d + p.kappa * gy * gy
    Pxy = p.kappa * gx * gy
    return Pxx, Pyy, Pxy


def stress_divergence_force(
    phi: torch.Tensor, p: FreeEnergyParams, scheme: str = "iso9"
) -> tuple[torch.Tensor, torch.Tensor]:
    """``F = -div P`` with central differences on the discrete tensor.

    This is the ludwig ``FE_FORCE_METHOD_STRESS_DIVERGENCE`` body force: the
    tensor is evaluated at every site, then differenced once (the O(h^2)
    wide-stencil divergence of ``phi_force_stress.c``).
    """
    Pxx, Pyy, Pxy = pressure_tensor(phi, p, scheme)
    Fx = -0.5 * (_roll(Pxx, 0, -1) - _roll(Pxx, 0, 1)) - 0.5 * (
        _roll(Pxy, -1, 0) - _roll(Pxy, 1, 0)
    )
    Fy = -0.5 * (_roll(Pxy, 0, -1) - _roll(Pxy, 0, 1)) - 0.5 * (
        _roll(Pyy, -1, 0) - _roll(Pyy, 1, 0)
    )
    return Fx, Fy


def grad_mu_force(
    phi: torch.Tensor, p: FreeEnergyParams, scheme: str = "iso9"
) -> tuple[torch.Tensor, torch.Tensor]:
    """``F = -phi*grad(mu)`` — TensorLBM's existing free-energy body force."""
    mu = chemical_potential(phi, p)
    gmx, gmy = grad(mu, scheme)
    return -phi * gmx, -phi * gmy


def surface_tension_analytic(p: FreeEnergyParams) -> float:
    """``sigma = sqrt(-8*kappa*a^3/(9*b^2)) = (2 sqrt2/3) sqrt(kappa A^3)/b``."""
    return math.sqrt(-8.0 * p.kappa * p.a ** 3 / (9.0 * p.b ** 2))


def interface_width(p: FreeEnergyParams) -> float:
    """Equilibrium interface thickness parameter ``xi = sqrt(2*kappa/A)``."""
    return math.sqrt(2.0 * p.kappa / p.A)


def planar_profile(ny: int, nx: int, p: FreeEnergyParams, x0: float | None = None) -> torch.Tensor:
    """Equilibrium planar interface ``phi = phi0 tanh((x-x0)/xi)``."""
    x0 = 0.5 * nx if x0 is None else x0
    phi0 = math.sqrt(p.A / p.B)
    xi = interface_width(p)
    x = torch.arange(nx, dtype=torch.float64)
    prof = phi0 * torch.tanh((x - x0) / xi)
    return prof.unsqueeze(0).expand(ny, nx).contiguous()


def droplet_profile(
    ny: int, nx: int, p: FreeEnergyParams, radius: float | None = None
) -> torch.Tensor:
    """Equilibrium droplet ``phi = -phi0 tanh((r-R)/xi)`` (phi>0 outside)."""
    R = (0.25 * min(ny, nx) if radius is None else radius)
    phi0 = math.sqrt(p.A / p.B)
    xi = interface_width(p)
    y = torch.arange(ny, dtype=torch.float64).view(-1, 1)
    x = torch.arange(nx, dtype=torch.float64).view(1, -1)
    r = torch.sqrt((y - 0.5 * ny) ** 2 + (x - 0.5 * nx) ** 2)
    return phi0 * torch.tanh((r - R) / xi)
