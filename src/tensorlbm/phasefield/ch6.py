"""CH6: Lee-Lin consistent Cahn-Hilliard with D3Q19-isotropic operators.

CH6 (``ch6`` geq_convention) implements the Cahn-Hilliard phase-field model
using **D3Q19-isotropic** finite-difference operators for both the chemical
potential and the Korteweg capillary force.  This ensures that the pressure
tensor and the phase-flux are discretised at the same isotropic order, which
eliminates the spurious-tangential capillary force that plagues the
seven-point (``central7``) family on curved interfaces.

Free-energy functional
----------------------
``f(phi) = -A phi²/2 + B phi⁴/4 + kappa |grad phi|²/2``

Chemical potential (isotropic Laplacian):
``mu = -A phi + B phi³ - kappa * lap_iso(phi)``

Korteweg force (isotropic gradient):
``F = -phi * grad_iso(mu)``

Equilibrium (Lee-Lin moment closure):
``g_eq`` satisfies ``sum(g_eq)=phi``, ``sum(c*g_eq)=phi*u``,
``sum(cc*g_eq)=phi*uu + Gamma*mu*I``.

Reference
---------
* Lee, Lin, "A stabilized domain-covering interface-capturing scheme for
  incompressible multiphase flow", J. Comput. Phys. 229 (2010) 5451-5474.
* lbmpy phasefield_allen_cahn isotropic operators (isotropic_fd.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .isotropic_fd import isotropic_gradient_3d, isotropic_laplacian_3d
from .operators import BoundaryPolicy

#: Finite-difference operator family for CH6 (always isotropic).
CH6_OPERATOR: Literal["isotropic_d3q19"] = "isotropic_d3q19"


@dataclass(frozen=True)
class CH6FreeEnergy:
    """CH6 free-energy model using D3Q19-isotropic operators.

    The isotropic Laplacian and gradient are built from the D3Q19 stencil
    (6 axis neighbours at weight 1/3, 12 face-diagonal neighbours at weight 1/6),
    which matches the isotropy of the D3Q19 velocity space exactly.
    """

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
        """Return ``mu = -A phi + B phi³ - kappa * lap_iso(phi)``."""
        lap = isotropic_laplacian_3d(phi, boundary=boundary, solid_mask=solid_mask)
        return -self.A * phi + self.B * phi**3 - self.kappa * lap

    def korteweg_force(
        self,
        phi: torch.Tensor,
        mu: torch.Tensor,
        *,
        boundary: BoundaryPolicy,
        solid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Korteweg force ``F = -phi * grad_iso(mu)``.

        Uses the **same** isotropic gradient operator as the chemical potential,
        guaranteeing that pressure and flux are discretised at the same order.
        """
        grad_x, grad_y, grad_z = isotropic_gradient_3d(
            mu, boundary=boundary, solid_mask=solid_mask
        )
        return -phi * grad_x, -phi * grad_y, -phi * grad_z


def validate_ch6_convention(geq_convention: str) -> None:
    """Validate that the geq_convention is 'ch6'."""
    if geq_convention != "ch6":
        raise ValueError("CH6 requires geq_convention='ch6'")


def ch6_equilibrium(
    phi: torch.Tensor,
    velocity: tuple[torch.Tensor, ...],
    mu: torch.Tensor | None = None,
    Gamma: float = 0.5,
) -> torch.Tensor:
    """Return CH6 equilibrium (Lee-Lin moment closure) for D3Q19.

    Identical to ``ch_equilibrium`` but documented as the CH6-specific
    equilibrium.  Satisfies:
      - sum(g_eq) = phi
      - sum(c * g_eq) = phi * u
      - sum(cc * g_eq) = phi * uu + Gamma * mu * I

    Args:
        phi: Phase field, shape (nz, ny, nx).
        velocity: Velocity components (ux, uy, uz), each shape (nz, ny, nx).
        mu: Chemical potential, shape (nz, ny, nx).  None → zero mu.
        Gamma: Mobility coupling constant.

    Returns:
        Equilibrium distribution g_eq, shape (19, nz, ny, nx).
    """
    if phi.ndim != 3 or len(velocity) != 3:
        raise ValueError("CH6 equilibrium requires 3-D phi and 3 velocity components")
    if not phi.is_floating_point():
        raise ValueError("phi must have a floating dtype")

    from ..d3q19 import W_EXACT64, C

    weights = W_EXACT64.to(device=phi.device, dtype=phi.dtype)
    shape = (-1,) + (1,) * 3
    c = C.to(device=phi.device, dtype=phi.dtype)
    w = weights.view(shape)

    ux, uy, uz = velocity
    cu = (
        c[:, 0].view(shape) * ux
        + c[:, 1].view(shape) * uy
        + c[:, 2].view(shape) * uz
    )
    usq = ux**2 + uy**2 + uz**2

    # Advection part (third-order Hermite)
    geq = w * phi.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * usq.unsqueeze(0))

    # Diffusion part (isotropic, matches D3Q19 second moment)
    if mu is not None:
        lam = 1.5 * (3.0 * (c * c).sum(1) - 3)  # dimension=3, matches ch_equilibrium convention
        geq = geq + w * Gamma * lam.view(shape) * mu.unsqueeze(0)

    return geq


def ch6_step_3d(
    f: torch.Tensor,
    g: torch.Tensor,
    *,
    tau_f: float = 1.0,
    tau_g: float = 0.7,
    A: float = 0.1,
    B: float = 0.1,
    kappa: float = 0.02,
    Gamma: float = 0.5,
    gx: float = 0.0,
    gy: float = 0.0,
    gz: float = 0.0,
    rho_heavy: float | None = None,
    rho_light: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One CH6 collision step for D3Q19.

    Uses D3Q19-isotropic operators for both chemical potential and Korteweg
    force, ensuring pressure and flux are discretised at the same order.

    Args:
        f: Momentum distribution, shape (19, nz, ny, nx).
        g: Order-parameter distribution, shape (19, nz, ny, nx).
        tau_f: Relaxation time for momentum.
        tau_g: Phase relaxation time.
        A: Double-well coefficient.
        B: Quartic coefficient.
        kappa: Interfacial-tension parameter.
        Gamma: Phase-field mobility coupling.
        gx, gy, gz: Body-force accelerations.
        rho_heavy: Effective density for phi=+1 phase (Boussinesq).
        rho_light: Effective density for phi=-1 phase.

    Returns:
        Updated (f, g) after one collision step.
    """
    from ..d3q19 import C, W_EXACT64, equilibrium3d
    from ..multiphase3d import macroscopic3d

    device = f.device
    c = C.to(device=device, dtype=f.dtype)
    w = W_EXACT64.to(device=device, dtype=f.dtype)
    cx = c[:, 0].float().view(19, 1, 1, 1)
    cy = c[:, 1].float().view(19, 1, 1, 1)
    cz = c[:, 2].float().view(19, 1, 1, 1)
    w_v = w.view(19, 1, 1, 1)

    # Macroscopic quantities
    rho, ux, uy, uz = macroscopic3d(f)
    phi = g.sum(dim=0)

    # Effective density for buoyancy
    if rho_heavy is not None and rho_light is not None:
        phi_c = phi.clamp(-1.0, 1.0)
        rho_eff = 0.5 * ((1.0 + phi_c) * rho_heavy + (1.0 - phi_c) * rho_light)
    else:
        rho_eff = rho

    # CH6: isotropic chemical potential and Korteweg force
    model = CH6FreeEnergy(A=A, B=B, kappa=kappa)
    mu = model.chemical_potential(phi, boundary="periodic")
    force_x, force_y, force_z = model.korteweg_force(
        phi, mu, boundary="periodic"
    )
    Fx = force_x + rho_eff * gx
    Fy = force_y + rho_eff * gy
    Fz = force_z + rho_eff * gz

    rho_s = torch.clamp(rho, min=1e-12)
    ux_eq = ux + tau_f * Fx / rho_s
    uy_eq = uy + tau_f * Fy / rho_s
    uz_eq = uz + tau_f * Fz / rho_s

    # Collision for f (BGK with Korteweg + buoyancy force)
    feq = equilibrium3d(rho, ux_eq, uy_eq, uz_eq)
    f_out = f - (f - feq) / tau_f

    # CH6 equilibrium for g (Lee-Lin moment closure)
    velocity = (ux + 0.5 * Fx / rho_s, uy + 0.5 * Fy / rho_s, uz + 0.5 * Fz / rho_s)
    geq = ch6_equilibrium(phi, velocity, mu, Gamma)
    g_out = g - (g - geq) / tau_g

    return f_out, g_out


__all__ = [
    "CH6FreeEnergy",
    "CH6_OPERATOR",
    "ch6_equilibrium",
    "ch6_step_3d",
    "validate_ch6_convention",
]
