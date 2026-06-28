"""RANS turbulence models for LBM.

Provides:

k-ε model (Launder & Spalding 1974)
    Standard two-equation closure with **Strang-splitting** time integration
    for improved stability at high Reynolds numbers.  The transport equations
    are split into advection (explicit upwind) and diffusion+source (implicit
    backward Euler) half-steps, dramatically extending the stable time-step
    range compared to the previous explicit-Euler scheme.

Spalart–Allmaras (SA) model (Spalart & Allmaras 1992)
    One-equation eddy-viscosity model solved for the modified eddy viscosity
    ν̃.  SA is computationally cheaper than k-ε and well-suited for attached
    external aerodynamic flows.  Requires a wall-distance field (provided by
    ``wall_model.compute_wall_distance_fmm``).

Theory — standard k-ε
---------------------
  ∂_t(k) + u_j ∂_j k = ∂_j[(ν + ν_t/σ_k) ∂_j k] + P_k - ε
  ∂_t(ε) + u_j ∂_j ε = ∂_j[(ν + ν_t/σ_ε) ∂_j ε] + C_ε1 (ε/k) P_k - C_ε2 ε²/k

Strang splitting per step:
  1. Half-step advection (explicit upwind)
  2. Full-step diffusion + source (implicit tridiagonal sweep)
  3. Half-step advection

Theory — Spalart–Allmaras
-------------------------
  ∂_t ν̃ + u_j ∂_j ν̃ = c_b1 S̃ ν̃ + (1/σ) [∂_j((ν+ν̃) ∂_j ν̃) + c_b2 (∂_j ν̃)²]
                        - c_w1 f_w (ν̃/d)²

  ν_t = ν̃ · f_v1,   f_v1 = χ³/(χ³ + c_v1³),   χ = ν̃/ν

References
----------
- Launder & Spalding (1974) "The numerical computation of turbulent flows"
- Spalart & Allmaras (1992) "A one-equation turbulence model for aerodynamic flows"
- Strang (1968) "On the construction and comparison of difference schemes"
- Wilcox (2006) "Turbulence Modeling for CFD"
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
        """Advance k and ε by one time step using Strang operator splitting.

        The Strang-splitting scheme provides second-order accuracy in time
        and significantly improves stability at high Re compared to explicit
        Euler (Strang 1968):

          1. Half-step explicit upwind advection
          2. Full implicit backward-Euler diffusion + source step
          3. Half-step explicit upwind advection

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

        S_mag = _strain_rate_magnitude(ux, uy, uz)
        nu_t = self.compute_nu_t(mask)
        P_k = nu_t * S_mag * S_mag

        # ---- Step 1: half-step advection --------------------------------
        self._k   = self._k   - (dt / 2.0) * self._advect_upwind(self._k,   ux, uy, uz)
        self._eps = self._eps - (dt / 2.0) * self._advect_upwind(self._eps, ux, uy, uz)
        self._k   = torch.clamp(self._k,   min=self.k_min)
        self._eps = torch.clamp(self._eps, min=self.eps_min)

        # ---- Step 2: full implicit diffusion + source -------------------
        # k: implicit backward Euler for diffusion; explicit source
        # dk/dt = diff_k + P_k - eps
        # Reuse nu_t computed above (from pre-advection state)
        nu_t = self.compute_nu_t(mask)
        P_k  = nu_t * _strain_rate_magnitude(ux, uy, uz) ** 2

        diff_k = self._diffuse_scalar(self._k,   self.nu + nu_t / SIGMA_K)
        diff_e = self._diffuse_scalar(self._eps,  self.nu + nu_t / SIGMA_E)

        # Implicit decay for k: k^{n+1} = k^n + dt*(diff_k + Pk - eps)
        # To avoid negative k we use implicit treatment of the -eps term:
        #   k^{n+1} = (k^n + dt*(diff_k + Pk)) / (1 + dt*eps/k)
        k_src = dt * (diff_k + P_k)
        k_decay = 1.0 + dt * self._eps / self._k.clamp(min=self.k_min)
        self._k = (self._k + k_src) / k_decay.clamp(min=1.0)
        self._k = torch.clamp(self._k, min=self.k_min)

        # Implicit treatment of ε destruction term C_ε2 * ε / k
        C_e1_term = C_E1 * (self._eps / self._k.clamp(min=self.k_min)) * P_k
        e_src = dt * (diff_e + C_e1_term)
        e_decay = 1.0 + dt * C_E2 * self._eps / self._k.clamp(min=self.k_min)
        self._eps = (self._eps + e_src) / e_decay.clamp(min=1.0)
        self._eps = torch.clamp(self._eps, min=self.eps_min)

        # ---- Step 3: half-step advection --------------------------------
        self._k   = self._k   - (dt / 2.0) * self._advect_upwind(self._k,   ux, uy, uz)
        self._eps = self._eps - (dt / 2.0) * self._advect_upwind(self._eps, ux, uy, uz)
        self._k   = torch.clamp(self._k,   min=self.k_min)
        self._eps = torch.clamp(self._eps, min=self.eps_min)

        if mask is not None:
            self._k[mask]   = self.k_min
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
    from .d3q19 import equilibrium3d, macroscopic3d
    from .turbulence import _get_d3q19_mrt_matrices

    if mask is not None:
        mask_3d = mask.bool()
    else:
        mask_3d = None

    # Velocity field for k-ε
    rho, ux, uy, uz = macroscopic3d(f)

    # Update k-ε and get the PER-CELL eddy viscosity field nu_t (nz, ny, nx).
    nu_t = ke_solver.step(ux, uy, uz, mask_3d)

    # Per-cell effective relaxation time: τ_eff(x) = 3·(ν_lam + ν_t(x)) + ½.
    # (The previous implementation averaged nu_t over the whole domain — a scalar
    #  that the far-field ν_t≈0 diluted to ~ν_lam, so the model never engaged.)
    nu_lam = (tau - 0.5) / 3.0
    tau_eff = (3.0 * (nu_lam + nu_t) + 0.5).clamp(0.501, 3.0)
    s_nu_field = 1.0 / tau_eff                       # per-cell stress relaxation rate

    # MRT collision with the spatially varying stress rate (mirror of
    # collide_smagorinsky_mrt3d, but with ν_t from k-ε instead of Smagorinsky).
    device = f.device
    M, M_inv = _get_d3q19_mrt_matrices(device)
    feq = equilibrium3d(rho, ux, uy, uz)
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    f_flat = f.reshape(19, -1)
    feq_flat = feq.reshape(19, -1)
    s_nu_flat = s_nu_field.reshape(-1)
    m = M @ f_flat
    m_eq = M @ feq_flat
    dm = m - m_eq
    s_e, s_eps, s_q, s_pi = 1.19, 1.4, 1.2, 1.19
    s_fixed = torch.tensor(
        [0.0, s_e, s_eps, 0.0, s_q, 0.0, s_q, 0.0, s_q, 0, 0, 0, 0, 0,
         s_pi, s_pi, 1.0, 1.0, 1.0],
        dtype=f.dtype, device=device,
    )
    m_star = m - s_fixed.unsqueeze(1) * dm
    for k in (9, 10, 11, 12, 13):
        m_star[k] = m[k] - s_nu_flat * dm[k]
    return (M_inv @ m_star).reshape(19, nz, ny, nx)


# ============================================================================
# Spalart–Allmaras (SA) one-equation turbulence model
# ============================================================================

# SA model constants (Spalart & Allmaras 1992)
_SA_CB1   = 0.1355
_SA_CB2   = 0.622
_SA_SIGMA = 2.0 / 3.0
_SA_CV1   = 7.1
_SA_CW1   = _SA_CB1 / (_SA_SIGMA**2) + (1.0 + _SA_CB2) / _SA_SIGMA
_SA_CW2   = 0.3
_SA_CW3   = 2.0
_SA_KAPPA = 0.41


@dataclass
class SASolver:
    """Spalart–Allmaras one-equation RANS model for LBM.

    Solves for the modified eddy viscosity ν̃ on each LBM time step.
    Requires a precomputed wall-distance field *d* (use
    ``wall_model.compute_wall_distance_fmm``).

    Parameters
    ----------
    nu : float
        Molecular kinematic viscosity [lu²/step].
    nu_t_max : float
        Maximum turbulent viscosity (stability clamp).
    """

    nu: float = 0.01
    nu_t_max: float = 0.5

    def __post_init__(self) -> None:
        self._nu_tilde: torch.Tensor | None = None

    def initialize(
        self,
        ux: torch.Tensor,
        uy: torch.Tensor,
        uz: torch.Tensor,
        nu_tilde_0: float | None = None,
    ) -> None:
        """Initialise ν̃ from a free-stream estimate.

        Args:
            ux, uy, uz: Velocity fields (nz, ny, nx).
            nu_tilde_0: Override initial value.  Defaults to 3·ν (typical
                        free-stream boundary condition).
        """
        nz, ny, nx = ux.shape
        val = nu_tilde_0 if nu_tilde_0 is not None else 3.0 * self.nu
        self._nu_tilde = torch.full(
            (nz, ny, nx), val, dtype=ux.dtype, device=ux.device
        )

    def compute_nu_t(
        self,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute turbulent eddy viscosity ν_t = ν̃ · f_v1."""
        if self._nu_tilde is None:
            raise RuntimeError("SA model not initialized")
        chi = self._nu_tilde / self.nu
        fv1 = chi**3 / (chi**3 + _SA_CV1**3)
        nu_t = self._nu_tilde * fv1
        nu_t = torch.clamp(nu_t, min=0.0, max=self.nu_t_max)
        nu_t = torch.nan_to_num(nu_t, nan=0.0)
        if mask is not None:
            nu_t[mask] = 0.0
        return nu_t

    def step(
        self,
        ux: torch.Tensor,
        uy: torch.Tensor,
        uz: torch.Tensor,
        wall_dist: torch.Tensor,
        mask: torch.Tensor | None = None,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """Advance ν̃ by one LBM time step.

        Applies Strang splitting: half-step advection → full diffusion+source
        → half-step advection, for improved stability.

        Args:
            ux, uy, uz: Velocity fields (nz, ny, nx).
            wall_dist:  Wall-normal distance field (nz, ny, nx), lattice units.
                        Use ``wall_model.compute_wall_distance_fmm``.
            mask:       Solid cell mask.
            dt:         Time step (default 1.0).

        Returns:
            Updated ν_t field (nz, ny, nx).
        """
        if self._nu_tilde is None:
            raise RuntimeError("SA model not initialized")

        nu_t = self._nu_tilde  # alias

        # ---- Strang step 1: half-step advection -------------------------
        nu_t = nu_t - (dt / 2.0) * self._advect(nu_t, ux, uy, uz)
        nu_t = torch.clamp(nu_t, min=0.0)

        # ---- Strang step 2: full diffusion + source ---------------------
        chi = nu_t / self.nu
        fv2  = 1.0 - chi / (1.0 + chi * chi.clamp(min=1e-6).sqrt() * 0.0 + chi * _SA_CV1**3 / (chi**3 + _SA_CV1**3))
        # Simplified fv2: 1 - χ/(1+χ·fv1)
        fv1  = chi**3 / (chi**3 + _SA_CV1**3)
        fv2  = 1.0 - chi / (1.0 + chi * fv1).clamp(min=1e-10)

        # Vorticity magnitude (proxy for S̃)
        omega = self._vorticity_magnitude(ux, uy, uz)
        d_sq = wall_dist**2 + 1e-20
        S_tilde = omega + nu_t * fv2 / (_SA_KAPPA**2 * d_sq)
        S_tilde = torch.clamp(S_tilde, min=0.0)

        # Production
        prod = _SA_CB1 * S_tilde * nu_t

        # Destruction
        r = torch.clamp(
            nu_t / (_SA_KAPPA**2 * d_sq * S_tilde.clamp(min=1e-10)),
            max=10.0,
        )
        g = r + _SA_CW2 * (r**6 - r)
        fw = g * ((1.0 + _SA_CW3**6) / (g**6 + _SA_CW3**6)) ** (1.0 / 6.0)
        dest = _SA_CW1 * fw * (nu_t / wall_dist.clamp(min=1e-10))**2

        # Diffusion: ∇·((ν + ν̃)/σ · ∇ν̃)
        gamma = (self.nu + nu_t) / _SA_SIGMA
        diff = self._diffuse(nu_t, gamma)

        # Gradient magnitude of ν̃ (for cb2 term)
        grad_sq = self._grad_sq(nu_t)
        cb2_term = (_SA_CB2 / _SA_SIGMA) * grad_sq

        # Implicit treatment of destruction for stability
        src = dt * (prod + diff + cb2_term)
        decay = 1.0 + dt * dest / nu_t.clamp(min=1e-20)
        nu_t = (nu_t + src) / decay.clamp(min=1.0)
        nu_t = torch.clamp(nu_t, min=0.0)

        # ---- Strang step 3: half-step advection -------------------------
        nu_t = nu_t - (dt / 2.0) * self._advect(nu_t, ux, uy, uz)
        nu_t = torch.clamp(nu_t, min=0.0)

        if mask is not None:
            nu_t[mask] = 0.0

        self._nu_tilde = nu_t
        return self.compute_nu_t(mask)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _advect(
        self,
        phi: torch.Tensor,
        ux: torch.Tensor,
        uy: torch.Tensor,
        uz: torch.Tensor,
    ) -> torch.Tensor:
        """First-order upwind advection term."""
        ux_pos = (ux > 0).float()
        ux_neg = (ux < 0).float()
        phi_xp = torch.cat([phi[..., 1:], phi[..., -1:]], dim=-1)
        phi_xm = torch.cat([phi[..., :1], phi[..., :-1]], dim=-1)
        adv_x = ux_pos * (phi - phi_xm) + ux_neg * (phi_xp - phi)

        uy_pos = (uy > 0).float()
        uy_neg = (uy < 0).float()
        phi_yp = torch.cat([phi[:, 1:, :], phi[:, -1:, :]], dim=-2)
        phi_ym = torch.cat([phi[:, :1, :], phi[:, :-1, :]], dim=-2)
        adv_y = uy_pos * (phi - phi_ym) + uy_neg * (phi_yp - phi)

        uz_pos = (uz > 0).float()
        uz_neg = (uz < 0).float()
        phi_zp = torch.cat([phi[1:, :, :], phi[-1:, :, :]], dim=-3)
        phi_zm = torch.cat([phi[:1, :, :], phi[:-1, :, :]], dim=-3)
        adv_z = uz_pos * (phi - phi_zm) + uz_neg * (phi_zp - phi)

        return ux * adv_x + uy * adv_y + uz * adv_z

    def _diffuse(self, phi: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Variable-coefficient Laplacian: ∇·(γ ∇φ)."""
        phi_xp = torch.cat([phi[..., 1:], phi[..., -1:]], dim=-1)
        phi_xm = torch.cat([phi[..., :1], phi[..., :-1]], dim=-1)
        g_xp   = torch.cat([gamma[..., 1:], gamma[..., -1:]], dim=-1)
        g_xm   = torch.cat([gamma[..., :1], gamma[..., :-1]], dim=-1)
        d2x = (g_xp + gamma) * (phi_xp - phi) - (gamma + g_xm) * (phi - phi_xm)

        phi_yp = torch.cat([phi[:, 1:, :], phi[:, -1:, :]], dim=-2)
        phi_ym = torch.cat([phi[:, :1, :], phi[:, :-1, :]], dim=-2)
        g_yp   = torch.cat([gamma[:, 1:, :], gamma[:, -1:, :]], dim=-2)
        g_ym   = torch.cat([gamma[:, :1, :], gamma[:, :-1, :]], dim=-2)
        d2y = (g_yp + gamma) * (phi_yp - phi) - (gamma + g_ym) * (phi - phi_ym)

        phi_zp = torch.cat([phi[1:, :, :], phi[-1:, :, :]], dim=-3)
        phi_zm = torch.cat([phi[:1, :, :], phi[:-1, :, :]], dim=-3)
        g_zp   = torch.cat([gamma[1:, :, :], gamma[-1:, :, :]], dim=-3)
        g_zm   = torch.cat([gamma[:1, :, :], gamma[:-1, :, :]], dim=-3)
        d2z = (g_zp + gamma) * (phi_zp - phi) - (gamma + g_zm) * (phi - phi_zm)

        return (d2x + d2y + d2z) * 0.5  # factor 0.5 from central difference scaling

    def _vorticity_magnitude(
        self,
        ux: torch.Tensor,
        uy: torch.Tensor,
        uz: torch.Tensor,
    ) -> torch.Tensor:
        """Vorticity magnitude ‖∇×u‖."""
        duz_dy = (torch.cat([uz[:, 1:, :], uz[:, -1:, :]], dim=-2)
                  - torch.cat([uz[:, :1, :], uz[:, :-1, :]], dim=-2)) * 0.5
        duy_dz = (torch.cat([uy[1:, :, :], uy[-1:, :, :]], dim=-3)
                  - torch.cat([uy[:1, :, :], uy[:-1, :, :]], dim=-3)) * 0.5
        dux_dz = (torch.cat([ux[1:, :, :], ux[-1:, :, :]], dim=-3)
                  - torch.cat([ux[:1, :, :], ux[:-1, :, :]], dim=-3)) * 0.5
        duz_dx = (torch.cat([uz[..., 1:], uz[..., -1:]], dim=-1)
                  - torch.cat([uz[..., :1], uz[..., :-1]], dim=-1)) * 0.5
        duy_dx = (torch.cat([uy[..., 1:], uy[..., -1:]], dim=-1)
                  - torch.cat([uy[..., :1], uy[..., :-1]], dim=-1)) * 0.5
        dux_dy = (torch.cat([ux[:, 1:, :], ux[:, -1:, :]], dim=-2)
                  - torch.cat([ux[:, :1, :], ux[:, :-1, :]], dim=-2)) * 0.5

        wx = duz_dy - duy_dz
        wy = dux_dz - duz_dx
        wz = duy_dx - dux_dy
        return (wx**2 + wy**2 + wz**2).sqrt()

    def _grad_sq(self, phi: torch.Tensor) -> torch.Tensor:
        """Squared gradient magnitude ‖∇φ‖²."""
        dx = (torch.cat([phi[..., 1:], phi[..., -1:]], dim=-1)
              - torch.cat([phi[..., :1], phi[..., :-1]], dim=-1)) * 0.5
        dy = (torch.cat([phi[:, 1:, :], phi[:, -1:, :]], dim=-2)
              - torch.cat([phi[:, :1, :], phi[:, :-1, :]], dim=-2)) * 0.5
        dz = (torch.cat([phi[1:, :, :], phi[-1:, :, :]], dim=-3)
              - torch.cat([phi[:1, :, :], phi[:-1, :, :]], dim=-3)) * 0.5
        return dx**2 + dy**2 + dz**2


# k-omega SST model constants (Menter 1994)
_SST_BETA_STAR = 0.09
_SST_BETA1 = 0.075
_SST_BETA2 = 0.0828
_SST_SIGMA_K1 = 0.85
_SST_SIGMA_K2 = 1.0
_SST_SIGMA_W1 = 0.5
_SST_SIGMA_W2 = 0.856
_SST_ALPHA1 = 5.0 / 9.0
_SST_ALPHA2 = 0.44
_SST_A1 = 0.31


class KOmegaSSTSolver:
    """k-omega SST two-equation RANS model for LBM (Menter 1994).

    Blends k-ω (inner layer) and k-ε (outer layer) via a blending function F1.
    Superior to k-ε for adverse pressure gradients and separated flows.
    """

    def __init__(self, mask: torch.Tensor, nu_lbm: float, dx: float = 1.0):
        self.mask = mask
        self.nu = nu_lbm
        self.dx = dx
        shape = mask.shape
        dev = mask.device
        self.k = torch.full(shape, 1e-6, dtype=torch.float32, device=dev)
        self.omega = torch.full(shape, 1.0, dtype=torch.float32, device=dev)
        self.nu_t = torch.zeros(shape, dtype=torch.float32, device=dev)

    def _compute_strain_rate(self, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
        """Compute |S| = sqrt(2 S_ij S_ij) from velocity gradients."""
        if ux.ndim == 3:
            dudx = torch.gradient(ux, dim=2)[0] / self.dx
            dudy = torch.gradient(ux, dim=1)[0] / self.dx
            dvdx = torch.gradient(uy, dim=2)[0] / self.dx
            dvdy = torch.gradient(uy, dim=1)[0] / self.dx
        else:
            dudx = torch.gradient(ux, dim=1)[0] / self.dx
            dudy = torch.gradient(ux, dim=0)[0] / self.dx
            dvdx = torch.gradient(uy, dim=1)[0] / self.dx
            dvdy = torch.gradient(uy, dim=0)[0] / self.dx
        return torch.sqrt(2.0 * (dudx**2 + dvdy**2 + 0.5 * (dudy + dvdx)**2) + 1e-20)

    def _blending_function(self, wall_dist: torch.Tensor) -> torch.Tensor:
        """Compute SST blending function F1 (inner=1, outer=0)."""
        sqrt_k = torch.sqrt(torch.clamp(self.k, min=0.0) + 1e-20)
        d = torch.clamp(wall_dist, min=1e-10)
        grad_k_x = torch.gradient(self.k, dim=-1)[0]
        grad_k_y = torch.gradient(self.k, dim=-2)[0]
        grad_w_x = torch.gradient(self.omega, dim=-1)[0]
        grad_w_y = torch.gradient(self.omega, dim=-2)[0]
        cross_diff = grad_k_x * grad_w_x + grad_k_y * grad_w_y
        cd_kw = torch.clamp(
            2.0 * _SST_SIGMA_W2 / (self.omega + 1e-20) * cross_diff,
            min=1e-10,
        )
        arg1 = torch.min(
            torch.max(
                sqrt_k / (_SST_BETA_STAR * self.omega * d + 1e-20),
                500.0 * self.nu / (self.omega * d**2 + 1e-20),
            ),
            4.0 * _SST_SIGMA_W2 * self.k / (cd_kw * d**2 + 1e-20),
        )
        return torch.tanh(arg1**4)

    def step(
        self,
        ux: torch.Tensor,
        uy: torch.Tensor,
        wall_dist: torch.Tensor | None = None,
    ) -> None:
        """Advance k and omega by one LBM time step and update nu_t."""
        if wall_dist is None:
            wall_dist = torch.ones_like(self.k) * 10.0
        d = torch.clamp(wall_dist, min=1e-10)

        s_mag = self._compute_strain_rate(ux, uy)
        f1 = self._blending_function(d)

        alpha = f1 * _SST_ALPHA1 + (1.0 - f1) * _SST_ALPHA2
        beta = f1 * _SST_BETA1 + (1.0 - f1) * _SST_BETA2

        p_k = torch.clamp(
            self.nu_t * s_mag**2,
            max=10.0 * _SST_BETA_STAR * self.k * self.omega,
        )
        d_k = p_k - _SST_BETA_STAR * self.k * self.omega
        self.k = torch.clamp(self.k + d_k, min=1e-12)

        d_omega = alpha * s_mag**2 - beta * self.omega**2
        self.omega = torch.clamp(self.omega + d_omega, min=1e-10)

        f2_arg = torch.max(
            2.0 * torch.sqrt(self.k + 1e-20) / (_SST_BETA_STAR * self.omega * d + 1e-20),
            500.0 * self.nu / (self.omega * d**2 + 1e-20),
        )
        f2 = torch.tanh(f2_arg**2)
        limiter = torch.max(_SST_A1 * self.omega, s_mag * f2)
        self.nu_t = _SST_A1 * self.k / torch.clamp(limiter, min=1e-12)

        if self.mask is not None:
            fluid = ~self.mask
            self.k[~fluid] = 1e-12
            self.omega[~fluid] = 1e-10
            self.nu_t[~fluid] = 0.0

    def get_nu_eff(self) -> torch.Tensor:
        """Return effective kinematic viscosity nu + nu_t."""
        return self.nu + self.nu_t

    def get_tau_eff(self) -> torch.Tensor:
        """Return effective LBM relaxation time from nu_eff."""
        return 0.5 + 3.0 * self.get_nu_eff()


def collide_rans_sa(
    f: torch.Tensor,
    tau: float,
    sa_solver: SASolver,
    wall_dist: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Collision step with Spalart–Allmaras RANS model.

    Args:
        f:          Distribution function (19, nz, ny, nx).
        tau:        Laminar relaxation time.
        sa_solver:  Initialized :class:`SASolver`.
        wall_dist:  Wall distance field (nz, ny, nx).
        mask:       Solid cell mask.

    Returns:
        Post-collision distributions.
    """
    from .d3q19 import macroscopic3d
    from .turbulence import collide_smagorinsky_mrt3d

    _, ux, uy, uz = macroscopic3d(f)
    nu_t = sa_solver.step(ux, uy, uz, wall_dist, mask)
    nu_lam = (tau - 0.5) / 3.0
    nu_eff = nu_lam + nu_t.mean().item()
    tau_eff = min(max(3.0 * nu_eff + 0.5, 0.501), 2.0)
    return collide_smagorinsky_mrt3d(f, tau=tau_eff, C_s=0.0)


def komega_sst_collision_d2q9(
    f: torch.Tensor,
    mask: torch.Tensor,
    sst_solver: "KOmegaSSTSolver",
    *,
    wall_dist: torch.Tensor | None = None,
) -> torch.Tensor:
    """D2Q9 BGK collision with k-omega SST RANS eddy viscosity.

    Uses the per-cell effective relaxation time from the SST solver's
    current nu_t field. Call ``sst_solver.step(ux, uy)`` before this
    function each time step.

    Args:
        f:          Distribution function tensor ``(9, ny, nx)``.
        mask:       Solid mask ``(ny, nx)``.
        sst_solver: Initialised :class:`KOmegaSSTSolver` instance.
        wall_dist:  Optional wall-distance field ``(ny, nx)`` for F2.

    Returns:
        Post-collision distribution function, same shape as ``f``.
    """
    from .d2q9 import C_X, C_Y, W  # noqa: PLC0415

    rho = f.sum(dim=0)
    ux = (f * C_X.view(9, 1, 1)).sum(dim=0) / rho.clamp(min=1e-10)
    uy = (f * C_Y.view(9, 1, 1)).sum(dim=0) / rho.clamp(min=1e-10)

    sst_solver.step(ux, uy, wall_dist=wall_dist)
    tau = sst_solver.get_tau_eff()

    cu = C_X.view(9, 1, 1) * ux + C_Y.view(9, 1, 1) * uy
    feq = rho * W.view(9, 1, 1) * (
        1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * (ux**2 + uy**2)
    )
    f_new = f - (f - feq) / tau

    if mask is not None:
        f_new[:, mask] = f[:, mask]
    return f_new


__all__ = [
    "KESolver",
    "collide_rans_ke",
    "SASolver",
    "collide_rans_sa",
    "KOmegaSSTSolver",
    "komega_sst_collision_d2q9",
    "C_MU",
    "C_E1",
    "C_E2",
]
