"""RANS k-epsilon turbulence model for LBM.

Solves the standard k-ε transport equations coupled with the LBM
solver to provide turbulent eddy viscosity.

Theory
------
Standard k-ε model (Launder & Spalding 1974):

  ∂_t(ρk) + ∂_j(ρk u_j) = ∂_j[(ν + ν_t/σ_k) ∂_j k] + P_k - ρε
  ∂_t(ρε) + ∂_j(ρε u_j) = ∂_j[(ν + ν_t/σ_ε) ∂_j ε] + C_ε1 (ε/k) P_k - C_ε2 ρ ε²/k

---------------------
At each time step:
1. Compute strain-rate S from the LBM velocity field
2. Update k and ε using explicit Euler
3. Compute ν_t = C_μ * k² / ε
4. Use τ_eff = 3 * (ν + ν_t) + 0.5 in MRT collision
5. Clamp ν_t to maintain stability: tau_eff ∈ [0.51, 2.0]

For SUBOFF at Re=10^7 with L_lu=80:
  Expected turbulent effects:
  - ν_t ~ 0.01-0.1 (comparable to laminar ν=0.027 at tau=0.58)
  - Effective Re reduced from 180 to ~80, but correct wall stress
  - Combined with log-law wall function for accurate skin friction

References
----------
- Launder & Spalding (1974) "The numerical computation of turbulent flows"
- Wilcox (2006) "Turbulence Modeling for CFD"
- Succi et al. (2002) "Lattice Boltzmann method for RANS"
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

def _strain_rate_magnitude(
    ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor,
) -> torch.Tensor:
    """Compute strain-rate magnitude S = sqrt(2 S_ij S_ij)."""
    nz, ny, nx = ux.shape
    dx = 1.0

    # Velocity gradients (2nd order central difference)
    dux_dx = (torch.cat([ux[..., 1:], ux[..., -1:]], dim=-1)
              - torch.cat([ux[..., :1], ux[..., :-1]], dim=-1)) / (2 * dx)
    dux_dy = (torch.cat([ux[:, 1:, :], ux[:, -1:, :]], dim=-2)
              - torch.cat([ux[:, :1, :], ux[:, :-1, :]], dim=-2)) / (2 * dx)
    dux_dz = (torch.cat([ux[1:, :, :], ux[-1:, :, :]], dim=-3)
              - torch.cat([ux[:1, :, :], ux[:-1, :, :]], dim=-3)) / (2 * dx)
    duy_dx = (torch.cat([uy[..., 1:], uy[..., -1:]], dim=-1)
              - torch.cat([uy[..., :1], uy[..., :-1]], dim=-1)) / (2 * dx)
    duy_dy = (torch.cat([uy[:, 1:, :], uy[:, -1:, :]], dim=-2)
              - torch.cat([uy[:, :1, :], uy[:, :-1, :]], dim=-2)) / (2 * dx)
    duy_dz = (torch.cat([uy[1:, :, :], uy[-1:, :, :]], dim=-3)
              - torch.cat([uy[:1, :, :], uy[:-1, :, :]], dim=-3)) / (2 * dx)
    duz_dx = (torch.cat([uz[..., 1:], uz[..., -1:]], dim=-1)
              - torch.cat([uz[..., :1], uz[..., :-1]], dim=-1)) / (2 * dx)
    duz_dy = (torch.cat([uz[:, 1:, :], uz[:, -1:, :]], dim=-2)
              - torch.cat([uz[:, :1, :], uz[:, :-1, :]], dim=-2)) / (2 * dx)
    duz_dz = (torch.cat([uz[1:, :, :], uz[-1:, :, :]], dim=-3)
              - torch.cat([uz[:1, :, :], uz[:-1, :, :]], dim=-3)) / (2 * dx)

    # Strain rate tensor S_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
    S11 = dux_dx
    S22 = duy_dy
    S33 = duz_dz
    S12 = 0.5 * (dux_dy + duy_dx)
    S13 = 0.5 * (dux_dz + duz_dx)
    S23 = 0.5 * (duy_dz + duz_dy)

    # S = sqrt(2 * S_ij * S_ij)
    S_mag = torch.sqrt(
        2.0 * (S11**2 + S22**2 + S33**2 + 2*(S12**2 + S13**2 + S23**2))
    )
    return S_mag


# ============================================================================
# k-epsilon constants
# ============================================================================

C_MU = 0.09
C_E1 = 1.44
C_E2 = 1.92
SIGMA_K = 1.0
SIGMA_E = 1.3


@dataclass
class KESolver:
    """k-epsilon turbulence model solver for LBM.

    Parameters
    ----------
    nu : float
        Laminar kinematic viscosity [lu²/step].
    dx : float
        Grid spacing (default 1.0).
    k_min : float
        Minimum TKE to prevent division by zero.
    eps_min : float
        Minimum dissipation.
    nu_t_max : float
        Maximum turbulent viscosity (for stability).
    """

    nu: float = 0.01
    dx: float = 1.0
    k_min: float = 1e-8
    eps_min: float = 1e-12
    nu_t_max: float = 0.5

    def __post_init__(self) -> None:
        self._k: torch.Tensor | None = None
        self._eps: torch.Tensor | None = None

    def initialize(
        self,
        ux: torch.Tensor,
        uy: torch.Tensor,
        uz: torch.Tensor,
        k0: float = 1e-4,
        eps0: float = 1e-6,
    ) -> None:
        """Initialize k and epsilon fields.

        Parameters
        ----------
        ux, uy, uz : torch.Tensor
            Velocity fields, shape (nz, ny, nx).
        k0 : float
            Initial turbulent kinetic energy.
        eps0 : float
            Initial dissipation rate.
        """
        nz, ny, nx = ux.shape
        device = ux.device
        dtype = ux.dtype

        # Estimate from free-stream turbulence intensity
        # TI = sqrt(2/3 * k) / U  ≈ 1-5% for external flows
        u_mag = torch.sqrt(ux**2 + uy**2 + uz**2).mean().item()
        ti = 0.05  # 5% turbulence intensity
        k_init = 1.5 * (ti * u_mag)**2
        k_init = max(k_init, k0)

        # eps = C_mu^{3/4} * k^{3/2} / L_turb
        # L_turb ≈ 0.07 * L_body ≈ 0.07 * nx
        L_turb = 0.07 * nx
        eps_init = C_MU**0.75 * k_init**1.5 / L_turb
        eps_init = max(eps_init, eps0)

        self._k = torch.full((nz, ny, nx), k_init, dtype=dtype, device=device)
        self._eps = torch.full((nz, ny, nx), eps_init, dtype=dtype, device=device)

    def compute_nu_t(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute turbulent eddy viscosity ν_t = C_μ * k² / ε.

        Parameters
        ----------
        mask : torch.Tensor of bool, optional
            Solid cell mask. ν_t = 0 inside solids.

        Returns
        -------
        nu_t : torch.Tensor, shape (nz, ny, nx)
        """
        if self._k is None or self._eps is None:
            raise RuntimeError("k-ε not initialized")
        nu_t = C_MU * self._k**2 / self._eps.clamp(min=self.eps_min)
        nu_t = torch.clamp(nu_t, min=0.0, max=min(self.nu_t_max, self.nu * 10.0))
        nu_t = torch.nan_to_num(nu_t, nan=0.0, posinf=0.0, neginf=0.0)
        if mask is not None:
            nu_t[mask] = 0.0
        return nu_t

    def step(
        self,
        ux: torch.Tensor,
        uy: torch.Tensor,
        uz: torch.Tensor,
        mask: torch.Tensor | None = None,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """Advance k and ε by one time step.

        Uses explicit Euler integration with upwind advection.

        Parameters
        ----------
        ux, uy, uz : torch.Tensor, shape (nz, ny, nx)
            Velocity fields.
        mask : torch.Tensor of bool, optional
            Solid cell mask.
        dt : float
            Time step (default 1.0 for LBM).

        Returns
        -------
        nu_t : torch.Tensor, shape (nz, ny, nx)
            Updated eddy viscosity.
        """
        if self._k is None or self._eps is None:
            raise RuntimeError("Not initialized")

        device = ux.device
        nz, ny, nx = ux.shape

        # Strain rate magnitude
        S_mag = _strain_rate_magnitude(ux, uy, uz)
        # S is shape (6, nz, ny, nx): Sxx, Syy, Szz, Sxy, Sxz, Syz
        S_mag = torch.sqrt(
            2.0 * (S_mag**2 + S_mag**2 + S_mag**2 + 2*(S_mag**2 + S_mag**2 + S_mag**2))
        )

        # Current eddy viscosity
        nu_t = self.compute_nu_t(mask)

        # Production: P_k = ν_t * S²
        P_k = nu_t * S_mag * S_mag

        # Upwind advection of k
        adv_k = self._advect_upwind(self._k, ux, uy, uz)

        # Diffusion of k
        diff_k = self._diffuse_scalar(
            self._k, self.nu + nu_t / SIGMA_K,
        )

        # k equation
        dk_dt = -adv_k + diff_k + P_k - self._eps
        self._k = self._k + dt * dk_dt
        self._k = torch.clamp(self._k, min=self.k_min)

        # ε equation
        adv_e = self._advect_upwind(self._eps, ux, uy, uz)
        diff_e = self._diffuse_scalar(
            self._eps, self.nu + nu_t / SIGMA_E,
        )
        C_e1_term = C_E1 * (self._eps / self._k.clamp(min=self.k_min)) * P_k
        C_e2_term = C_E2 * self._eps**2 / self._k.clamp(min=self.k_min)

        de_dt = -adv_e + diff_e + C_e1_term - C_e2_term
        self._eps = self._eps + dt * de_dt
        self._eps = torch.clamp(self._eps, min=self.eps_min)

        # Zero at solid cells
        if mask is not None:
            self._k[mask] = self.k_min
            self._eps[mask] = self.eps_min

        return self.compute_nu_t(mask)

    def _advect_upwind(
        self,
        phi: torch.Tensor,
        ux: torch.Tensor,
        uy: torch.Tensor,
        uz: torch.Tensor,
    ) -> torch.Tensor:
        """First-order upwind advection of scalar phi."""
        nz, ny, nx = phi.shape
        device = phi.device

        # x-direction
        dphi_dx = torch.zeros_like(phi)
        ux_pos = (ux > 0).float()
        ux_neg = (ux < 0).float()
        phi_xp = torch.cat([phi[..., 1:], phi[..., -1:]], dim=-1)
        phi_xm = torch.cat([phi[..., :1], phi[..., :-1]], dim=-1)
        dphi_dx = ux_pos * (phi - phi_xm) + ux_neg * (phi_xp - phi)

        # y-direction
        dphi_dy = torch.zeros_like(phi)
        uy_pos = (uy > 0).float()
        uy_neg = (uy < 0).float()
        phi_yp = torch.cat([phi[:, 1:, :], phi[:, -1:, :]], dim=-2)
        phi_ym = torch.cat([phi[:, :1, :], phi[:, :-1, :]], dim=-2)
        dphi_dy = uy_pos * (phi - phi_ym) + uy_neg * (phi_yp - phi)

        # z-direction
        dphi_dz = torch.zeros_like(phi)
        uz_pos = (uz > 0).float()
        uz_neg = (uz < 0).float()
        phi_zp = torch.cat([phi[1:, :, :], phi[-1:, :, :]], dim=-3)
        phi_zm = torch.cat([phi[:1, :, :], phi[:-1, :, :]], dim=-3)
        dphi_dz = uz_pos * (phi - phi_zm) + uz_neg * (phi_zp - phi)

        return ux * dphi_dx + uy * dphi_dy + uz * dphi_dz

    def _diffuse_scalar(
        self,
        phi: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """Diffusion of scalar phi with diffusivity gamma.

        Uses second-order central differences.
        """
        nz, ny, nx = phi.shape
        dx = self.dx

        # x-direction
        phi_xp = torch.cat([phi[..., 1:], phi[..., -1:]], dim=-1)
        phi_xm = torch.cat([phi[..., :1], phi[..., :-1]], dim=-1)
        gamma_xp = torch.cat([gamma[..., 1:], gamma[..., -1:]], dim=-1)
        gamma_xm = torch.cat([gamma[..., :1], gamma[..., :-1]], dim=-1)
        d2phi_dx2 = (
            (gamma_xp + gamma) * (phi_xp - phi)
            - (gamma + gamma_xm) * (phi - phi_xm)
        ) / (2.0 * dx**2)

        # y-direction
        phi_yp = torch.cat([phi[:, 1:, :], phi[:, -1:, :]], dim=-2)
        phi_ym = torch.cat([phi[:, :1, :], phi[:, :-1, :]], dim=-2)
        gamma_yp = torch.cat([gamma[:, 1:, :], gamma[:, -1:, :]], dim=-2)
        gamma_ym = torch.cat([gamma[:, :1, :], gamma[:, :-1, :]], dim=-2)
        d2phi_dy2 = (
            (gamma_yp + gamma) * (phi_yp - phi)
            - (gamma + gamma_ym) * (phi - phi_ym)
        ) / (2.0 * dx**2)

        # z-direction
        phi_zp = torch.cat([phi[1:, :, :], phi[-1:, :, :]], dim=-3)
        phi_zm = torch.cat([phi[:1, :, :], phi[:-1, :, :]], dim=-3)
        gamma_zp = torch.cat([gamma[1:, :, :], gamma[-1:, :, :]], dim=-3)
        gamma_zm = torch.cat([gamma[:1, :, :], gamma[:-1, :, :]], dim=-3)
        d2phi_dz2 = (
            (gamma_zp + gamma) * (phi_zp - phi)
            - (gamma + gamma_zm) * (phi - phi_zm)
        ) / (2.0 * dx**2)

        return d2phi_dx2 + d2phi_dy2 + d2phi_dz2


# ============================================================================
# LBM collision with RANS turbulence model
# ============================================================================

def collide_rans_ke(
    f: torch.Tensor,
    tau: float,
    ke_solver: KESolver,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Collision step with k-epsilon RANS model.

    Computes ν_t from k-ε, then uses Smagorinsky MRT with
    effective ν = ν_laminar + ν_turbulent.

    Parameters
    ----------
    f : torch.Tensor, shape (19, nz, ny, nx)
        Distribution function.
    tau : float
        Laminar relaxation time.
    ke_solver : KESolver
        Initialized k-ε solver.
    mask : torch.Tensor, optional
        Solid cell mask.

    Returns
    -------
    f : torch.Tensor
        Post-collision distributions.
    """
    from .d3q19 import macroscopic3d
    from .turbulence import collide_smagorinsky_mrt3d

    if mask is not None:
        mask_3d = mask.bool()
    else:
        mask_3d = None

    # Compute velocity field for k-ε
    _, ux, uy, uz = macroscopic3d(f)

    # Update k-ε and get ν_t
    nu_t = ke_solver.step(ux, uy, uz, mask_3d)

    # Effective relaxation time
    nu_lam = (tau - 0.5) / 3.0
    nu_eff = nu_lam + nu_t.mean().item()  # scalar average for MRT
    tau_eff = min(max(3.0 * nu_eff + 0.5, 0.501), 2.0)

    # Collision with effective tau
    f = collide_smagorinsky_mrt3d(f, tau=tau_eff, C_s=0.0)

    return f


__all__ = [
    "KESolver",
    "collide_rans_ke",
    "C_MU",
    "C_E1",
    "C_E2",
]
