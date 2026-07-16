"""LES turbulence sub-grid models for the lattice Boltzmann method.

Three families of LES closures are provided, covering D2Q9, D3Q19 and D3Q27
velocity sets:

**Smagorinsky** (Hou *et al.* 1994, Yu *et al.* 2006)
    A self-consistent effective relaxation time is computed from the Frobenius
    norm of the non-equilibrium stress tensor.  BGK and MRT variants are
    available.

**Dynamic Smagorinsky** (Germano *et al.* 1991, Lilly 1992)
    The Smagorinsky constant :math:`C_s` is computed dynamically from the
    Germano identity using a test-filter (box average).  This avoids the need
    for a prescribed constant and naturally yields zero eddy viscosity in
    laminar regions.  BGK and MRT variants are available for D3Q19, and BGK
    and MRT variants for D3Q27.

**WALE** – Wall-Adapting Local Eddy-viscosity (Nicoud & Ducros, 1999)
    The eddy viscosity is derived from the traceless symmetric part of the
    squared velocity-gradient tensor.  The WALE model reproduces the correct
    cubic near-wall behaviour (ν_t ∝ y³) without damping functions, making it
    more accurate than Smagorinsky for wall-bounded flows.

**Vreman** (Vreman, 2004)
    An algebraic model based on the invariants of the velocity-gradient tensor.
    Computationally cheaper than WALE and naturally produces zero eddy viscosity
    in laminar regions and for solid-body rotation.

For WALE and Vreman the velocity-gradient tensor is approximated by second-order
central finite differences of the local macroscopic velocity field (periodic
boundaries).  The eddy viscosity is converted to an effective per-cell relaxation
time via :math:`\\tau_{\\rm eff} = \\tau_0 + 3\\,\\nu_t`.

Exported functions
------------------
Smagorinsky
    - :func:`collide_smagorinsky_bgk`    – D2Q9 BGK + Smagorinsky
    - :func:`collide_smagorinsky_mrt`    – D2Q9 MRT + Smagorinsky
    - :func:`collide_smagorinsky_bgk3d`  – D3Q19 BGK + Smagorinsky
    - :func:`collide_smagorinsky_mrt3d`  – D3Q19 MRT + Smagorinsky
    - :func:`collide_smagorinsky_bgk27`  – D3Q27 BGK + Smagorinsky
    - :func:`collide_smagorinsky_mrt27`  – D3Q27 MRT + Smagorinsky
Dynamic Smagorinsky
    - :func:`collide_dynamic_smagorinsky_bgk`     – D2Q9  BGK + dynamic Smagorinsky
    - :func:`collide_dynamic_smagorinsky_bgk3d`   – D3Q19 BGK + dynamic Smagorinsky
    - :func:`collide_dynamic_smagorinsky_mrt3d`   – D3Q19 MRT + dynamic Smagorinsky
    - :func:`collide_dynamic_smagorinsky_bgk27`   – D3Q27 BGK + dynamic Smagorinsky
    - :func:`collide_dynamic_smagorinsky_mrt27`   – D3Q27 MRT + dynamic Smagorinsky
WALE
    - :func:`collide_wale_bgk`    – D2Q9  BGK + WALE
    - :func:`collide_wale_bgk3d`  – D3Q19 BGK + WALE
    - :func:`collide_wale_bgk27`  – D3Q27 BGK + WALE
Vreman
    - :func:`collide_vreman_bgk`    – D2Q9  BGK + Vreman
    - :func:`collide_vreman_bgk3d`  – D3Q19 BGK + Vreman
    - :func:`collide_vreman_bgk27`  – D3Q27 BGK + Vreman
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .d2q9 import C as C2D
from .d2q9 import equilibrium, macroscopic
from .d3q19 import C as C3D
from .d3q19 import equilibrium3d, macroscopic3d
from .d3q27 import C as C27
from .d3q27 import _get_d3q27_mrt_matrices, equilibrium27, macroscopic27
from .solver import _get_d2q9_mrt_matrices
from .solver3d import _get_d3q19_mrt_matrices

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _neq_stress_norm_2d(f_neq: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of the 2-D non-equilibrium stress tensor per cell.

    For D2Q9 the symmetric stress tensor has three independent components::

        Π_xx, Π_yy, Π_xy

    The Frobenius norm is ``sqrt(Π_xx² + Π_yy² + 2·Π_xy²)``.

    Args:
        f_neq: Non-equilibrium distributions, shape ``(9, ny, nx)``.

    Returns:
        Tensor of shape ``(ny, nx)``.
    """
    device = f_neq.device
    c = C2D.to(device).float()  # (9, 2)
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)

    pi_xx = (cx * cx * f_neq).sum(0)
    pi_yy = (cy * cy * f_neq).sum(0)
    pi_xy = (cx * cy * f_neq).sum(0)

    return torch.sqrt(pi_xx ** 2 + pi_yy ** 2 + 2.0 * pi_xy ** 2)


def _neq_stress_norm_3d(f_neq: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of the 3-D non-equilibrium stress tensor per cell.

    For D3Q19 the symmetric stress tensor has six independent components::

        Π_xx, Π_yy, Π_zz, Π_xy, Π_xz, Π_yz

    The Frobenius norm is
    ``sqrt(Π_xx² + Π_yy² + Π_zz² + 2(Π_xy² + Π_xz² + Π_yz²))``.

    Args:
        f_neq: Non-equilibrium distributions, shape ``(19, nz, ny, nx)``.

    Returns:
        Tensor of shape ``(nz, ny, nx)``.
    """
    device = f_neq.device
    c = C3D.to(device).float()  # (19, 3)
    cx = c[:, 0].view(19, 1, 1, 1)
    cy = c[:, 1].view(19, 1, 1, 1)
    cz = c[:, 2].view(19, 1, 1, 1)

    pi_xx = (cx * cx * f_neq).sum(0)
    pi_yy = (cy * cy * f_neq).sum(0)
    pi_zz = (cz * cz * f_neq).sum(0)
    pi_xy = (cx * cy * f_neq).sum(0)
    pi_xz = (cx * cz * f_neq).sum(0)
    pi_yz = (cy * cz * f_neq).sum(0)

    return torch.sqrt(
        pi_xx ** 2 + pi_yy ** 2 + pi_zz ** 2
        + 2.0 * (pi_xy ** 2 + pi_xz ** 2 + pi_yz ** 2)
    )


def _smagorinsky_tau(
    tau: float,
    pi_norm: torch.Tensor,
    rho: torch.Tensor,
    C_s: float,
) -> torch.Tensor:
    """Per-cell effective relaxation time via the Smagorinsky sub-grid model.

    .. math::

        \\tau_{eff}(x) = \\frac{1}{2}\\left(\\tau_0 +
            \\sqrt{\\tau_0^2 + 18 C_s^2 |\\Pi^{neq}|_F(x) / \\rho(x)}\\right)

    Args:
        tau: Molecular (baseline) relaxation time :math:`\\tau_0`.
        pi_norm: Frobenius norm of the non-equilibrium stress, same shape as *rho*.
        rho: Density field.
        C_s: Smagorinsky constant (lattice units; typically 0.1).

    Returns:
        Effective :math:`\\tau_{eff}` tensor with the same shape as *rho*.
    """
    rho_safe = torch.clamp(rho, min=1e-12)
    discriminant = tau ** 2 + 18.0 * C_s ** 2 * pi_norm / rho_safe
    return 0.5 * (tau + torch.sqrt(torch.clamp(discriminant, min=0.0)))


# ---------------------------------------------------------------------------
# Public collision operators
# ---------------------------------------------------------------------------

def collide_smagorinsky_bgk(
    f: torch.Tensor,
    tau: float,
    C_s: float = 0.1,
) -> torch.Tensor:
    """D2Q9 BGK collision with Smagorinsky LES sub-grid turbulence model.

    Uses a spatially varying effective relaxation time computed from the
    local non-equilibrium stress magnitude.  Suitable for 2-D flows at
    Reynolds numbers where the BGK operator alone would become unstable.

    Args:
        f: Distribution tensor of shape ``(9, ny, nx)``.
        tau: Molecular relaxation time :math:`\\tau_0 > 0.5`.
        C_s: Smagorinsky constant (default 0.1).

    Returns:
        Updated distribution tensor of the same shape.
    """
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    f_neq = f - feq

    pi_norm = _neq_stress_norm_2d(f_neq)
    tau_eff = _smagorinsky_tau(tau, pi_norm, rho, C_s)  # (ny, nx)

    return f - f_neq / tau_eff.unsqueeze(0)


def collide_smagorinsky_bgk3d(
    f: torch.Tensor,
    tau: float,
    C_s: float = 0.1,
) -> torch.Tensor:
    """D3Q19 BGK collision with Smagorinsky LES sub-grid turbulence model.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        tau: Molecular relaxation time :math:`\\tau_0 > 0.5`.
        C_s: Smagorinsky constant (default 0.1).

    Returns:
        Updated distribution tensor of the same shape.
    """
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    f_neq = f - feq

    pi_norm = _neq_stress_norm_3d(f_neq)
    tau_eff = _smagorinsky_tau(tau, pi_norm, rho, C_s)  # (nz, ny, nx)

    return f - f_neq / tau_eff.unsqueeze(0)


def collide_smagorinsky_mrt(
    f: torch.Tensor,
    tau: float,
    C_s: float = 0.1,
    s_e: float = 1.64,
    s_eps: float = 1.54,
    s_q: float = 1.7,
) -> torch.Tensor:
    """D2Q9 MRT collision with Smagorinsky LES sub-grid turbulence model.

    Combines the D2Q9 multi-relaxation-time (MRT) collision operator with a
    spatially varying stress relaxation rate derived from the local
    Smagorinsky effective viscosity.  Recommended for high-Reynolds 2-D flows
    (e.g. cylinder at Re > 500) where BGK alone becomes unstable.

    The MRT relaxation vector is identical to :func:`collide_mrt` except
    that the stress relaxation rate ``1/τ`` (moments 7 and 8) is replaced by
    the local ``1/τ_eff(x)`` computed from the Smagorinsky model.

    Moment ordering (rows of M):
        0: ρ  (conserved, s=0)
        1: e  (energy,          s=s_e)
        2: ε  (energy-square,   s=s_eps)
        3: jx (conserved, s=0)
        4: qx (heat-flux x,     s=s_q)
        5: jy (conserved, s=0)
        6: qy (heat-flux y,     s=s_q)
        7: pxx (stress,         s=1/τ_eff per cell)
        8: pxy (stress,         s=1/τ_eff per cell)

    Args:
        f: Distribution tensor of shape ``(9, ny, nx)``.
        tau: Molecular relaxation time :math:`\\tau_0 > 0.5`.
        C_s: Smagorinsky constant (default 0.1).
        s_e: Relaxation rate for the energy moment.
        s_eps: Relaxation rate for the energy-square moment.
        s_q: Relaxation rate for the heat-flux moments.

    Returns:
        Updated distribution tensor of the same shape.
    """
    device = f.device
    matrix, matrix_inv = _get_d2q9_mrt_matrices(device)

    # Per-cell effective relaxation time from Smagorinsky model
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    f_neq = f - feq
    pi_norm = _neq_stress_norm_2d(f_neq)
    tau_eff = _smagorinsky_tau(tau, pi_norm, rho, C_s)  # (ny, nx)
    s_nu_field = 1.0 / tau_eff  # (ny, nx)

    ny, nx = f.shape[1], f.shape[2]
    f_flat = f.reshape(9, -1)
    feq_flat = feq.reshape(9, -1)
    s_nu_flat = s_nu_field.reshape(-1)  # (N,)

    s_fixed = torch.tensor(
        [0.0, s_e, s_eps, 0.0, s_q, 0.0, s_q, 0.0, 0.0],
        dtype=f.dtype,
        device=device,
    )  # (9,)

    m = matrix @ f_flat
    m_eq = matrix @ feq_flat
    dm = m - m_eq

    m_star = m - s_fixed.unsqueeze(1) * dm
    # Override stress modes 7 and 8 with spatially varying Smagorinsky rate
    for k in (7, 8):
        m_star[k] = m[k] - s_nu_flat * dm[k]

    return (matrix_inv @ m_star).reshape(9, ny, nx)


def collide_smagorinsky_mrt3d(
    f: torch.Tensor,
    tau: float,
    C_s: float = 0.1,
    s_e: float = 1.19,
    s_eps: float = 1.4,
    s_q: float = 1.2,
    s_pi: float | None = None,
) -> torch.Tensor:
    """D3Q19 MRT collision with Smagorinsky LES sub-grid turbulence model.

    Combines the multi-relaxation-time (MRT) collision operator with a
    spatially varying stress relaxation rate derived from the local
    Smagorinsky effective viscosity.  This is the recommended collision
    operator for high-Reynolds ship and ocean engineering simulations.

    The MRT relaxation vector is identical to :func:`collide_mrt3d` except
    that the stress relaxation rate ``1/τ`` is replaced by the local
    ``1/τ_eff(x)`` computed from the Smagorinsky model.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        tau: Molecular relaxation time :math:`\\tau_0 > 0.5`.
        C_s: Smagorinsky constant (default 0.1).
        s_e: Relaxation rate for the energy moment.
        s_eps: Relaxation rate for the energy-square moment.
        s_q: Relaxation rate for heat-flux moments.
        s_pi: Relaxation rate for higher-order stress moments
              (defaults to *s_e* when *None*).

    Returns:
        Updated distribution tensor of the same shape.
    """
    if s_pi is None:
        s_pi = s_e

    device = f.device
    M, M_inv = _get_d3q19_mrt_matrices(device)

    # Compute per-cell effective tau
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    f_neq = f - feq
    pi_norm = _neq_stress_norm_3d(f_neq)
    tau_eff = _smagorinsky_tau(tau, pi_norm, rho, C_s)  # (nz, ny, nx)
    s_nu_field = 1.0 / tau_eff  # (nz, ny, nx)

    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    f_flat = f.reshape(19, -1)      # (19, N)
    feq_flat = feq.reshape(19, -1)  # (19, N)
    s_nu_flat = s_nu_field.reshape(-1)  # (N,)

    m = M @ f_flat               # (19, N)
    m_eq = M @ feq_flat          # (19, N)
    dm = m - m_eq                # (19, N)

    # Build m_star using broadcasting to avoid allocating a full (19, N) s_vec.
    # Fixed-rate modes use s_fixed[:, None] broadcast; stress modes 9-13 use
    # the per-cell Smagorinsky rate.
    s_fixed = torch.tensor(
        [0.0, s_e, s_eps,
         0.0, s_q, 0.0, s_q, 0.0, s_q,
         0.0, 0.0, 0.0, 0.0, 0.0,
         s_pi, s_pi,
         1.0, 1.0, 1.0],
        dtype=f.dtype, device=device,
    )  # (19,)
    m_star = m - s_fixed.unsqueeze(1) * dm  # (19, N) via broadcast
    # Override stress modes 9-13 with the spatially varying Smagorinsky rate
    for k in (9, 10, 11, 12, 13):
        m_star[k] = m[k] - s_nu_flat * dm[k]
    return (M_inv @ m_star).reshape(19, nz, ny, nx)


def _neq_stress_norm_27(f_neq: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of the 3-D non-equilibrium stress tensor for D3Q27.

    Identical formula to :func:`_neq_stress_norm_3d` but operates on the
    27-velocity distribution.

    Args:
        f_neq: Non-equilibrium distributions, shape ``(27, nz, ny, nx)``.

    Returns:
        Tensor of shape ``(nz, ny, nx)``.
    """
    device = f_neq.device
    c = C27.to(device).float()  # (27, 3)
    cx = c[:, 0].view(27, 1, 1, 1)
    cy = c[:, 1].view(27, 1, 1, 1)
    cz = c[:, 2].view(27, 1, 1, 1)

    pi_xx = (cx * cx * f_neq).sum(0)
    pi_yy = (cy * cy * f_neq).sum(0)
    pi_zz = (cz * cz * f_neq).sum(0)
    pi_xy = (cx * cy * f_neq).sum(0)
    pi_xz = (cx * cz * f_neq).sum(0)
    pi_yz = (cy * cz * f_neq).sum(0)

    return torch.sqrt(
        pi_xx ** 2 + pi_yy ** 2 + pi_zz ** 2
        + 2.0 * (pi_xy ** 2 + pi_xz ** 2 + pi_yz ** 2)
    )


def collide_smagorinsky_bgk27(
    f: torch.Tensor,
    tau: float,
    C_s: float = 0.1,
) -> torch.Tensor:
    """D3Q27 BGK collision with Smagorinsky LES sub-grid turbulence model.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        tau: Molecular relaxation time :math:`\\tau_0 > 0.5`.
        C_s: Smagorinsky constant (default 0.1).

    Returns:
        Updated distribution tensor of the same shape.
    """
    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    f_neq = f - feq

    pi_norm = _neq_stress_norm_27(f_neq)
    tau_eff = _smagorinsky_tau(tau, pi_norm, rho, C_s)  # (nz, ny, nx)

    return f - f_neq / tau_eff.unsqueeze(0)


def collide_smagorinsky_mrt27(
    f: torch.Tensor,
    tau: float,
    C_s: float = 0.1,
    s_e: float = 1.19,
    s_eps: float = 1.4,
    s_q: float = 1.2,
    s_pi: float | None = None,
) -> torch.Tensor:
    """D3Q27 MRT collision with Smagorinsky LES sub-grid turbulence model.

    Combines the D3Q27 MRT collision operator with a spatially varying stress
    relaxation rate derived from the local Smagorinsky effective viscosity.

    The MRT relaxation vector follows :func:`~tensorlbm.d3q27.collide_mrt27`
    except that the stress-mode relaxation rates (rows 5–9) are replaced by
    the per-cell ``1/τ_eff(x)`` from the Smagorinsky model.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        tau: Molecular relaxation time :math:`\\tau_0 > 0.5`.
        C_s: Smagorinsky constant (default 0.1).
        s_e: Relaxation rate for the energy moment (row 4).
        s_eps: Relaxation rate for the energy-square moment (row 19).
        s_q: Relaxation rate for 3rd-order heat-flux moments (rows 10–18).
        s_pi: Relaxation rate for 4th-order+ moments (rows 20–26);
              defaults to *s_e* when *None*.

    Returns:
        Updated distribution tensor of the same shape.
    """
    if s_pi is None:
        s_pi = s_e

    device = f.device
    M, M_inv = _get_d3q27_mrt_matrices(device)

    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    f_neq = f - feq
    pi_norm = _neq_stress_norm_27(f_neq)
    tau_eff = _smagorinsky_tau(tau, pi_norm, rho, C_s)  # (nz, ny, nx)
    s_nu_flat = (1.0 / tau_eff).reshape(-1)  # (N,)

    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    f_flat = f.reshape(27, -1)
    feq_flat = feq.reshape(27, -1)

    m = M @ f_flat
    m_eq = M @ feq_flat
    dm = m - m_eq

    # Fixed relaxation rates for non-stress modes
    s_fixed = torch.tensor(
        [
            0.0,   # 0  mass
            0.0,   # 1  jx
            0.0,   # 2  jy
            0.0,   # 3  jz
            s_e,   # 4  energy
            0.0,   # 5  Nxx  – overridden below
            0.0,   # 6  Nyy  – overridden below
            0.0,   # 7  Pxy  – overridden below
            0.0,   # 8  Pxz  – overridden below
            0.0,   # 9  Pyz  – overridden below
            s_q,   # 10
            s_q,   # 11
            s_q,   # 12
            s_q,   # 13
            s_q,   # 14
            s_q,   # 15
            s_q,   # 16
            s_q,   # 17
            s_q,   # 18
            s_eps, # 19
            s_pi,  # 20
            s_pi,  # 21
            s_pi,  # 22
            s_pi,  # 23
            s_pi,  # 24
            s_pi,  # 25
            s_pi,  # 26
        ],
        dtype=f.dtype,
        device=device,
    )
    m_star = m - s_fixed.unsqueeze(1) * dm
    # Override stress modes 5–9 with spatially varying Smagorinsky rate
    for k in (5, 6, 7, 8, 9):
        m_star[k] = m[k] - s_nu_flat * dm[k]
    return (M_inv @ m_star).reshape(27, nz, ny, nx)


# ---------------------------------------------------------------------------
# Velocity-gradient helpers (shared by WALE and Vreman)
# ---------------------------------------------------------------------------

def _velocity_gradients_2d(
    ux: torch.Tensor,
    uy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Velocity-gradient tensor for a 2-D field via central differences.

    For a field of shape ``(ny, nx)``:
        dim -1 ↔ x,  dim -2 ↔ y.

    Returns:
        ``(g11, g12, g21, g22)`` where ``g_ij = ∂u_i/∂x_j``.
    """
    g11 = 0.5 * (torch.roll(ux, -1, dims=-1) - torch.roll(ux, 1, dims=-1))
    g12 = 0.5 * (torch.roll(ux, -1, dims=-2) - torch.roll(ux, 1, dims=-2))
    g21 = 0.5 * (torch.roll(uy, -1, dims=-1) - torch.roll(uy, 1, dims=-1))
    g22 = 0.5 * (torch.roll(uy, -1, dims=-2) - torch.roll(uy, 1, dims=-2))
    return g11, g12, g21, g22


def _velocity_gradients_3d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Velocity-gradient tensor for a 3-D field via central differences.

    For a field of shape ``(nz, ny, nx)``:
        dim 0 ↔ z,  dim 1 ↔ y,  dim 2 ↔ x.

    Returns:
        ``(g11, g12, g13, g21, g22, g23, g31, g32, g33)``
        where ``g_ij = ∂u_i/∂x_j``.
    """
    def _cd(u: torch.Tensor, dim: int) -> torch.Tensor:
        return 0.5 * (torch.roll(u, -1, dims=dim) - torch.roll(u, 1, dims=dim))

    g11 = _cd(ux, 2)  # ∂ux/∂x
    g12 = _cd(ux, 1)  # ∂ux/∂y
    g13 = _cd(ux, 0)  # ∂ux/∂z
    g21 = _cd(uy, 2)  # ∂uy/∂x
    g22 = _cd(uy, 1)  # ∂uy/∂y
    g23 = _cd(uy, 0)  # ∂uy/∂z
    g31 = _cd(uz, 2)  # ∂uz/∂x
    g32 = _cd(uz, 1)  # ∂uz/∂y
    g33 = _cd(uz, 0)  # ∂uz/∂z
    return g11, g12, g13, g21, g22, g23, g31, g32, g33


# ---------------------------------------------------------------------------
# WALE eddy-viscosity helpers
# ---------------------------------------------------------------------------

def _wale_nu_t_2d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    C_w: float,
) -> torch.Tensor:
    """WALE kinematic eddy viscosity for a 2-D velocity field.

    Computes the traceless symmetric part S^d of the squared velocity-gradient
    tensor g² and returns:

    .. math::

        \\nu_t = C_w^2 \\, \\frac{\\|S^d\\|_F^3}{\\|S\\|_F^5 + \\|S^d\\|_F^{5/2}}

    with the lattice filter width :math:`\\Delta = 1`.

    Args:
        ux: x-velocity, shape ``(ny, nx)``.
        uy: y-velocity, shape ``(ny, nx)``.
        C_w: WALE constant (typically 0.5).

    Returns:
        Per-cell eddy viscosity tensor, same shape as *ux*.
    """
    g11, g12, g21, g22 = _velocity_gradients_2d(ux, uy)

    # g² = g @ g  (2×2 matrix product, element-wise per cell)
    g2_11 = g11 * g11 + g12 * g21
    g2_12 = g11 * g12 + g12 * g22
    g2_21 = g21 * g11 + g22 * g21
    g2_22 = g21 * g12 + g22 * g22

    # Traceless symmetric part: S^d_ij = (g²_ij + g²_ji)/2 − δ_ij tr(g²)/D, D=2
    tr_g2 = g2_11 + g2_22
    Sd_11 = g2_11 - tr_g2 * 0.5
    Sd_22 = g2_22 - tr_g2 * 0.5
    Sd_12 = 0.5 * (g2_12 + g2_21)
    Sd_norm2 = Sd_11 ** 2 + Sd_22 ** 2 + 2.0 * Sd_12 ** 2

    # Strain-rate norm: ||S||² where S_ij = (g_ij + g_ji)/2
    S_12 = 0.5 * (g12 + g21)
    S_norm2 = g11 ** 2 + g22 ** 2 + 2.0 * S_12 ** 2

    eps = 1e-30
    numerator = torch.clamp(Sd_norm2, min=0.0) ** 1.5
    denominator = (
        torch.clamp(S_norm2, min=0.0) ** 2.5
        + torch.clamp(Sd_norm2, min=0.0) ** 1.25
        + eps
    )
    return (C_w ** 2) * numerator / denominator


def _wale_nu_t_3d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    C_w: float,
) -> torch.Tensor:
    """WALE kinematic eddy viscosity for a 3-D velocity field."""
    g11, g12, g13, g21, g22, g23, g31, g32, g33 = _velocity_gradients_3d(ux, uy, uz)

    # g² = g @ g  (3×3 matrix product, element-wise per cell)
    g2_11 = g11 * g11 + g12 * g21 + g13 * g31
    g2_12 = g11 * g12 + g12 * g22 + g13 * g32
    g2_13 = g11 * g13 + g12 * g23 + g13 * g33
    g2_21 = g21 * g11 + g22 * g21 + g23 * g31
    g2_22 = g21 * g12 + g22 * g22 + g23 * g32
    g2_23 = g21 * g13 + g22 * g23 + g23 * g33
    g2_31 = g31 * g11 + g32 * g21 + g33 * g31
    g2_32 = g31 * g12 + g32 * g22 + g33 * g32
    g2_33 = g31 * g13 + g32 * g23 + g33 * g33

    # Traceless symmetric part: S^d_ij = (g²_ij + g²_ji)/2 − δ_ij tr(g²)/3
    tr_g2 = g2_11 + g2_22 + g2_33
    inv3 = 1.0 / 3.0
    Sd_11 = g2_11 - tr_g2 * inv3
    Sd_22 = g2_22 - tr_g2 * inv3
    Sd_33 = g2_33 - tr_g2 * inv3
    Sd_12 = 0.5 * (g2_12 + g2_21)
    Sd_13 = 0.5 * (g2_13 + g2_31)
    Sd_23 = 0.5 * (g2_23 + g2_32)
    Sd_norm2 = (
        Sd_11 ** 2 + Sd_22 ** 2 + Sd_33 ** 2
        + 2.0 * (Sd_12 ** 2 + Sd_13 ** 2 + Sd_23 ** 2)
    )

    S_12 = 0.5 * (g12 + g21)
    S_13 = 0.5 * (g13 + g31)
    S_23 = 0.5 * (g23 + g32)
    S_norm2 = (
        g11 ** 2 + g22 ** 2 + g33 ** 2
        + 2.0 * (S_12 ** 2 + S_13 ** 2 + S_23 ** 2)
    )

    eps = 1e-30
    numerator = torch.clamp(Sd_norm2, min=0.0) ** 1.5
    denominator = (
        torch.clamp(S_norm2, min=0.0) ** 2.5
        + torch.clamp(Sd_norm2, min=0.0) ** 1.25
        + eps
    )
    return (C_w ** 2) * numerator / denominator


# ---------------------------------------------------------------------------
# Vreman eddy-viscosity helpers
# ---------------------------------------------------------------------------

def _vreman_nu_t_2d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    C_V: float,
) -> torch.Tensor:
    """Vreman kinematic eddy viscosity for a 2-D velocity field.

    Reference: Vreman (2004) Phys. Fluids 16, 3670.

    With :math:`\\alpha_{ij} = \\partial u_i / \\partial x_j` (lattice Δ = 1)
    and :math:`\\beta_{ij} = \\sum_k \\alpha_{ki}\\,\\alpha_{kj}`:

    .. math::

        \\nu_t = C_V \\sqrt{\\frac{\\max(B_\\beta,0)}{A_\\alpha + \\epsilon}}

    where :math:`A_\\alpha = \\|\\alpha\\|_F^2` and
    :math:`B_\\beta = \\beta_{11}\\beta_{22} - \\beta_{12}^2`.

    Args:
        ux: x-velocity, shape ``(ny, nx)``.
        uy: y-velocity, shape ``(ny, nx)``.
        C_V: Vreman constant (typically 2.5 C_s²; default 0.025 for C_s = 0.1).

    Returns:
        Per-cell eddy viscosity tensor, same shape as *ux*.
    """
    g11, g12, g21, g22 = _velocity_gradients_2d(ux, uy)

    # β = α^T α  (α_ij = g_ij)
    beta_11 = g11 * g11 + g21 * g21
    beta_22 = g12 * g12 + g22 * g22
    beta_12 = g11 * g12 + g21 * g22

    A_alpha = g11 ** 2 + g12 ** 2 + g21 ** 2 + g22 ** 2
    B_beta = beta_11 * beta_22 - beta_12 ** 2

    eps = 1e-30
    return C_V * torch.sqrt(torch.clamp(B_beta, min=0.0) / (A_alpha + eps))


def _vreman_nu_t_3d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    C_V: float,
) -> torch.Tensor:
    """Vreman kinematic eddy viscosity for a 3-D velocity field."""
    g11, g12, g13, g21, g22, g23, g31, g32, g33 = _velocity_gradients_3d(ux, uy, uz)

    # β = α^T α  (α_ij = g_ij)
    beta_11 = g11 * g11 + g21 * g21 + g31 * g31
    beta_22 = g12 * g12 + g22 * g22 + g32 * g32
    beta_33 = g13 * g13 + g23 * g23 + g33 * g33
    beta_12 = g11 * g12 + g21 * g22 + g31 * g32
    beta_13 = g11 * g13 + g21 * g23 + g31 * g33
    beta_23 = g12 * g13 + g22 * g23 + g32 * g33

    A_alpha = (
        g11 ** 2 + g12 ** 2 + g13 ** 2
        + g21 ** 2 + g22 ** 2 + g23 ** 2
        + g31 ** 2 + g32 ** 2 + g33 ** 2
    )
    B_beta = (
        beta_11 * beta_22 - beta_12 ** 2
        + beta_11 * beta_33 - beta_13 ** 2
        + beta_22 * beta_33 - beta_23 ** 2
    )

    eps = 1e-30
    return C_V * torch.sqrt(torch.clamp(B_beta, min=0.0) / (A_alpha + eps))


def _nu_t_to_tau_eff(tau: float, nu_t: torch.Tensor) -> torch.Tensor:
    """Convert per-cell turbulent eddy viscosity to effective relaxation time.

    Uses the lattice relation :math:`\\nu = c_s^2(\\tau - 1/2) = (\\tau-1/2)/3`
    so :math:`\\tau_{\\rm eff} = \\tau_0 + 3\\,\\nu_t` (clamped to stay > 0.5).

    Args:
        tau: Molecular (baseline) relaxation time :math:`\\tau_0`.
        nu_t: Per-cell turbulent eddy viscosity tensor.

    Returns:
        Effective per-cell :math:`\\tau_{\\rm eff}` tensor, same shape as *nu_t*.
    """
    return torch.clamp(tau + 3.0 * nu_t, min=0.5001)


# ---------------------------------------------------------------------------
# WALE collision operators
# ---------------------------------------------------------------------------

def collide_wale_bgk(
    f: torch.Tensor,
    tau: float,
    C_w: float = 0.5,
) -> torch.Tensor:
    """D2Q9 BGK collision with WALE LES sub-grid turbulence model.

    The WALE model (Nicoud & Ducros, 1999) derives the eddy viscosity from the
    traceless symmetric part of the squared velocity-gradient tensor.  Unlike
    Smagorinsky it produces the correct cubic near-wall vanishing of ν_t without
    any explicit damping function.

    Velocity gradients are computed from the local macroscopic velocity field
    via second-order central differences.

    Args:
        f: Distribution tensor of shape ``(9, ny, nx)``.
        tau: Molecular relaxation time :math:`\\tau_0 > 0.5`.
        C_w: WALE constant (default 0.5; typical range 0.3–0.6).

    Returns:
        Updated distribution tensor of the same shape.
    """
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    f_neq = f - feq

    nu_t = _wale_nu_t_2d(ux, uy, C_w)
    tau_eff = _nu_t_to_tau_eff(tau, nu_t)

    return f - f_neq / tau_eff.unsqueeze(0)


def collide_wale_bgk3d(
    f: torch.Tensor,
    tau: float,
    C_w: float = 0.5,
) -> torch.Tensor:
    """D3Q19 BGK collision with WALE LES sub-grid turbulence model.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        tau: Molecular relaxation time :math:`\\tau_0 > 0.5`.
        C_w: WALE constant (default 0.5).

    Returns:
        Updated distribution tensor of the same shape.
    """
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    f_neq = f - feq

    nu_t = _wale_nu_t_3d(ux, uy, uz, C_w)
    tau_eff = _nu_t_to_tau_eff(tau, nu_t)

    return f - f_neq / tau_eff.unsqueeze(0)


def collide_wale_bgk27(
    f: torch.Tensor,
    tau: float,
    C_w: float = 0.5,
) -> torch.Tensor:
    """D3Q27 BGK collision with WALE LES sub-grid turbulence model.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        tau: Molecular relaxation time :math:`\\tau_0 > 0.5`.
        C_w: WALE constant (default 0.5).

    Returns:
        Updated distribution tensor of the same shape.
    """
    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    f_neq = f - feq

    nu_t = _wale_nu_t_3d(ux, uy, uz, C_w)
    tau_eff = _nu_t_to_tau_eff(tau, nu_t)

    return f - f_neq / tau_eff.unsqueeze(0)


# ---------------------------------------------------------------------------
# Vreman collision operators
# ---------------------------------------------------------------------------

def collide_vreman_bgk(
    f: torch.Tensor,
    tau: float,
    C_V: float = 0.025,
) -> torch.Tensor:
    """D2Q9 BGK collision with Vreman LES sub-grid turbulence model.

    The Vreman (2004) model computes the eddy viscosity from the invariants of
    the velocity-gradient tensor.  It is computationally cheap and naturally
    gives zero eddy viscosity in laminar, solid-body-rotation and
    wall-bounded regions.

    Args:
        f: Distribution tensor of shape ``(9, ny, nx)``.
        tau: Molecular relaxation time :math:`\\tau_0 > 0.5`.
        C_V: Vreman constant (default 0.025; corresponds to C_s ≈ 0.1).

    Returns:
        Updated distribution tensor of the same shape.
    """
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    f_neq = f - feq

    nu_t = _vreman_nu_t_2d(ux, uy, C_V)
    tau_eff = _nu_t_to_tau_eff(tau, nu_t)

    return f - f_neq / tau_eff.unsqueeze(0)


def collide_vreman_bgk3d(
    f: torch.Tensor,
    tau: float,
    C_V: float = 0.025,
) -> torch.Tensor:
    """D3Q19 BGK collision with Vreman LES sub-grid turbulence model.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        tau: Molecular relaxation time :math:`\\tau_0 > 0.5`.
        C_V: Vreman constant (default 0.025).

    Returns:
        Updated distribution tensor of the same shape.
    """
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    f_neq = f - feq

    nu_t = _vreman_nu_t_3d(ux, uy, uz, C_V)
    tau_eff = _nu_t_to_tau_eff(tau, nu_t)

    return f - f_neq / tau_eff.unsqueeze(0)


def collide_vreman_bgk27(
    f: torch.Tensor,
    tau: float,
    C_V: float = 0.025,
) -> torch.Tensor:
    """D3Q27 BGK collision with Vreman LES sub-grid turbulence model.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        tau: Molecular relaxation time :math:`\\tau_0 > 0.5`.
        C_V: Vreman constant (default 0.025).

    Returns:
        Updated distribution tensor of the same shape.
    """
    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    f_neq = f - feq

    nu_t = _vreman_nu_t_3d(ux, uy, uz, C_V)
    tau_eff = _nu_t_to_tau_eff(tau, nu_t)

    return f - f_neq / tau_eff.unsqueeze(0)


def _box_filter_2d(field: torch.Tensor, width: int = 2) -> torch.Tensor:
    """Apply a size-preserving 2D box filter using average pooling.

    Args:
        field: Field of shape ``(ny, nx)``.
        width: Filter width.

    Returns:
        Filtered field of shape ``(ny, nx)``.
    """
    pad_left = width // 2
    pad_right = width - 1 - pad_left
    padded = F.pad(
        field.unsqueeze(0).unsqueeze(0),
        (pad_left, pad_right, pad_left, pad_right),
        mode="replicate",
    )
    return F.avg_pool2d(padded, kernel_size=width, stride=1)[0, 0]


def _box_filter_3d(field: torch.Tensor, width: int = 2) -> torch.Tensor:
    """Apply a size-preserving 3D box filter using average pooling.

    Args:
        field: Field of shape ``(nz, ny, nx)``.
        width: Filter width.

    Returns:
        Filtered field of shape ``(nz, ny, nx)``.
    """
    pad_left = width // 2
    pad_right = width - 1 - pad_left
    padded = F.pad(
        field.unsqueeze(0).unsqueeze(0),
        (pad_left, pad_right, pad_left, pad_right, pad_left, pad_right),
        mode="replicate",
    )
    return F.avg_pool3d(padded, kernel_size=width, stride=1)[0, 0]


def _strain_from_fneq_2d(
    f_neq: torch.Tensor,
    rho: torch.Tensor,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate the 2D strain-rate tensor from non-equilibrium stresses."""
    c = C2D.to(f_neq.device).float()
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)
    pi_xx = (cx * cx * f_neq).sum(dim=0)
    pi_yy = (cy * cy * f_neq).sum(dim=0)
    pi_xy = (cx * cy * f_neq).sum(dim=0)
    denom = torch.clamp(2.0 * rho * tau * (1.0 / 3.0), min=1e-12)
    return -pi_xx / denom, -pi_xy / denom, -pi_yy / denom


def _strain_from_fneq_3d(
    f_neq: torch.Tensor,
    rho: torch.Tensor,
    tau: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Estimate the 3D strain-rate tensor from non-equilibrium stresses."""
    c = C3D.to(f_neq.device).float()
    cx = c[:, 0].view(19, 1, 1, 1)
    cy = c[:, 1].view(19, 1, 1, 1)
    cz = c[:, 2].view(19, 1, 1, 1)
    pi_xx = (cx * cx * f_neq).sum(dim=0)
    pi_yy = (cy * cy * f_neq).sum(dim=0)
    pi_zz = (cz * cz * f_neq).sum(dim=0)
    pi_xy = (cx * cy * f_neq).sum(dim=0)
    pi_xz = (cx * cz * f_neq).sum(dim=0)
    pi_yz = (cy * cz * f_neq).sum(dim=0)
    denom = torch.clamp(2.0 * rho * tau * (1.0 / 3.0), min=1e-12)
    return (
        -pi_xx / denom,
        -pi_xy / denom,
        -pi_xz / denom,
        -pi_yy / denom,
        -pi_yz / denom,
        -pi_zz / denom,
    )


def _strain_from_fneq_27(
    f_neq: torch.Tensor,
    rho: torch.Tensor,
    tau: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Estimate the 3D strain-rate tensor from D3Q27 non-equilibrium stresses.

    Identical formula to :func:`_strain_from_fneq_3d` but operates on the
    27-velocity distribution.
    """
    c = C27.to(f_neq.device).float()
    cx = c[:, 0].view(27, 1, 1, 1)
    cy = c[:, 1].view(27, 1, 1, 1)
    cz = c[:, 2].view(27, 1, 1, 1)
    pi_xx = (cx * cx * f_neq).sum(dim=0)
    pi_yy = (cy * cy * f_neq).sum(dim=0)
    pi_zz = (cz * cz * f_neq).sum(dim=0)
    pi_xy = (cx * cy * f_neq).sum(dim=0)
    pi_xz = (cx * cz * f_neq).sum(dim=0)
    pi_yz = (cy * cz * f_neq).sum(dim=0)
    denom = torch.clamp(2.0 * rho * tau * (1.0 / 3.0), min=1e-12)
    return (
        -pi_xx / denom,
        -pi_xy / denom,
        -pi_xz / denom,
        -pi_yy / denom,
        -pi_yz / denom,
        -pi_zz / denom,
    )


def _strain_magnitude_2d(
    s_xx: torch.Tensor,
    s_xy: torch.Tensor,
    s_yy: torch.Tensor,
) -> torch.Tensor:
    """Return ``|S|`` for a 2D symmetric strain tensor."""
    return torch.sqrt(torch.clamp(2.0 * (s_xx**2 + s_yy**2 + 2.0 * s_xy**2), min=0.0))


def _strain_magnitude_3d(
    s_xx: torch.Tensor,
    s_xy: torch.Tensor,
    s_xz: torch.Tensor,
    s_yy: torch.Tensor,
    s_yz: torch.Tensor,
    s_zz: torch.Tensor,
) -> torch.Tensor:
    """Return ``|S|`` for a 3D symmetric strain tensor."""
    return torch.sqrt(
        torch.clamp(
            2.0 * (s_xx**2 + s_yy**2 + s_zz**2 + 2.0 * (s_xy**2 + s_xz**2 + s_yz**2)),
            min=0.0,
        )
    )


def collide_dynamic_smagorinsky_bgk(
    f: torch.Tensor,
    tau: float,
    filter_width: int = 2,
    lambda_clip: float = 0.0,
) -> torch.Tensor:
    """D2Q9 BGK collision with a dynamic Smagorinsky closure.

    Args:
        f: Distribution tensor of shape ``(9, ny, nx)``.
        tau: Molecular relaxation time.
        filter_width: Test-filter width.
        lambda_clip: Minimum allowed value for ``C_s^2``.

    Returns:
        Updated distribution tensor of the same shape.
    """
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    f_neq = f - feq
    s_xx, s_xy, s_yy = _strain_from_fneq_2d(f_neq, rho, tau)
    s_mag = _strain_magnitude_2d(s_xx, s_xy, s_yy)

    uxf = _box_filter_2d(ux, width=filter_width)
    uyf = _box_filter_2d(uy, width=filter_width)
    g11, g12, g21, g22 = _velocity_gradients_2d(uxf, uyf)
    s_tilde_xx = g11
    s_tilde_yy = g22
    s_tilde_xy = 0.5 * (g12 + g21)
    s_tilde_mag = _strain_magnitude_2d(s_tilde_xx, s_tilde_xy, s_tilde_yy)

    l_xx = _box_filter_2d(ux * ux, width=filter_width) - uxf * uxf
    l_xy = _box_filter_2d(ux * uy, width=filter_width) - uxf * uyf
    l_yy = _box_filter_2d(uy * uy, width=filter_width) - uyf * uyf

    m_xx = 2.0 * (4.0 * s_tilde_mag * s_tilde_xx - s_mag * s_xx)
    m_xy = 2.0 * (4.0 * s_tilde_mag * s_tilde_xy - s_mag * s_xy)
    m_yy = 2.0 * (4.0 * s_tilde_mag * s_tilde_yy - s_mag * s_yy)

    num = (l_xx * m_xx + 2.0 * l_xy * m_xy + l_yy * m_yy).mean()
    den = (m_xx**2 + 2.0 * m_xy**2 + m_yy**2).mean()
    cs2 = torch.clamp(num / torch.clamp(den, min=1e-12), min=lambda_clip, max=0.1)
    cs = float(torch.sqrt(torch.clamp(cs2, min=0.0)).item())

    nu = (tau - 0.5) / 3.0
    nu_t = (cs**2) * s_mag
    tau_eff = torch.clamp(0.5 + 3.0 * (nu + nu_t), min=0.5001)
    return f - f_neq / tau_eff.unsqueeze(0)


def collide_dynamic_smagorinsky_bgk3d(
    f: torch.Tensor,
    tau: float,
    filter_width: int = 2,
    lambda_clip: float = 0.0,
) -> torch.Tensor:
    """D3Q19 BGK collision with a dynamic Smagorinsky closure.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        tau: Molecular relaxation time.
        filter_width: Test-filter width.
        lambda_clip: Minimum allowed value for ``C_s^2``.

    Returns:
        Updated distribution tensor of the same shape.
    """
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    f_neq = f - feq
    s_xx, s_xy, s_xz, s_yy, s_yz, s_zz = _strain_from_fneq_3d(f_neq, rho, tau)
    s_mag = _strain_magnitude_3d(s_xx, s_xy, s_xz, s_yy, s_yz, s_zz)

    uxf = _box_filter_3d(ux, width=filter_width)
    uyf = _box_filter_3d(uy, width=filter_width)
    uzf = _box_filter_3d(uz, width=filter_width)
    g11, g12, g13, g21, g22, g23, g31, g32, g33 = _velocity_gradients_3d(uxf, uyf, uzf)
    s_tilde_xx = g11
    s_tilde_yy = g22
    s_tilde_zz = g33
    s_tilde_xy = 0.5 * (g12 + g21)
    s_tilde_xz = 0.5 * (g13 + g31)
    s_tilde_yz = 0.5 * (g23 + g32)
    s_tilde_mag = _strain_magnitude_3d(
        s_tilde_xx, s_tilde_xy, s_tilde_xz, s_tilde_yy, s_tilde_yz, s_tilde_zz
    )

    l_xx = _box_filter_3d(ux * ux, width=filter_width) - uxf * uxf
    l_xy = _box_filter_3d(ux * uy, width=filter_width) - uxf * uyf
    l_xz = _box_filter_3d(ux * uz, width=filter_width) - uxf * uzf
    l_yy = _box_filter_3d(uy * uy, width=filter_width) - uyf * uyf
    l_yz = _box_filter_3d(uy * uz, width=filter_width) - uyf * uzf
    l_zz = _box_filter_3d(uz * uz, width=filter_width) - uzf * uzf

    m_xx = 2.0 * (4.0 * s_tilde_mag * s_tilde_xx - s_mag * s_xx)
    m_xy = 2.0 * (4.0 * s_tilde_mag * s_tilde_xy - s_mag * s_xy)
    m_xz = 2.0 * (4.0 * s_tilde_mag * s_tilde_xz - s_mag * s_xz)
    m_yy = 2.0 * (4.0 * s_tilde_mag * s_tilde_yy - s_mag * s_yy)
    m_yz = 2.0 * (4.0 * s_tilde_mag * s_tilde_yz - s_mag * s_yz)
    m_zz = 2.0 * (4.0 * s_tilde_mag * s_tilde_zz - s_mag * s_zz)

    num = (l_xx * m_xx + l_yy * m_yy + l_zz * m_zz).mean()
    num = num + 2.0 * (l_xy * m_xy + l_xz * m_xz + l_yz * m_yz).mean()
    den = (m_xx**2 + m_yy**2 + m_zz**2).mean()
    den = den + 2.0 * (m_xy**2 + m_xz**2 + m_yz**2).mean()
    cs2 = torch.clamp(num / torch.clamp(den, min=1e-12), min=lambda_clip, max=0.1)
    cs = float(torch.sqrt(torch.clamp(cs2, min=0.0)).item())

    nu = (tau - 0.5) / 3.0
    nu_t = (cs**2) * s_mag
    tau_eff = torch.clamp(0.5 + 3.0 * (nu + nu_t), min=0.5001)
    return f - f_neq / tau_eff.unsqueeze(0)


def collide_dynamic_smagorinsky_mrt3d(
    f: torch.Tensor,
    tau: float,
    filter_width: int = 2,
    lambda_clip: float = 0.0,
    s_e: float = 1.19,
    s_eps: float = 1.4,
    s_q: float = 1.2,
    s_pi: float | None = None,
) -> torch.Tensor:
    """D3Q19 MRT collision with a dynamic Smagorinsky closure.

    Combines the dynamic Smagorinsky procedure (Germano identity with
    test-filtering to compute a global :math:`C_s`) with the D3Q19 MRT
    collision operator.  The per-cell effective relaxation time
    :math:`\\tau_{eff}(x)` replaces the stress-mode relaxation rate
    (modes 9–13), exactly as in :func:`collide_smagorinsky_mrt3d`, while
    the non-stress modes use the fixed MRT rates.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        tau: Molecular relaxation time.
        filter_width: Test-filter width.
        lambda_clip: Minimum allowed value for ``C_s^2``.
        s_e: Relaxation rate for the energy moment.
        s_eps: Relaxation rate for the energy-square moment.
        s_q: Relaxation rate for heat-flux moments.
        s_pi: Relaxation rate for higher-order stress moments
              (defaults to *s_e* when *None*).

    Returns:
        Updated distribution tensor of the same shape.
    """
    if s_pi is None:
        s_pi = s_e

    device = f.device

    # ---- Dynamic Smagorinsky: compute global Cs via Germano identity ----
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    f_neq = f - feq
    s_xx, s_xy, s_xz, s_yy, s_yz, s_zz = _strain_from_fneq_3d(f_neq, rho, tau)
    s_mag = _strain_magnitude_3d(s_xx, s_xy, s_xz, s_yy, s_yz, s_zz)

    uxf = _box_filter_3d(ux, width=filter_width)
    uyf = _box_filter_3d(uy, width=filter_width)
    uzf = _box_filter_3d(uz, width=filter_width)
    g11, g12, g13, g21, g22, g23, g31, g32, g33 = _velocity_gradients_3d(uxf, uyf, uzf)
    s_tilde_xx = g11
    s_tilde_yy = g22
    s_tilde_zz = g33
    s_tilde_xy = 0.5 * (g12 + g21)
    s_tilde_xz = 0.5 * (g13 + g31)
    s_tilde_yz = 0.5 * (g23 + g32)
    s_tilde_mag = _strain_magnitude_3d(
        s_tilde_xx, s_tilde_xy, s_tilde_xz, s_tilde_yy, s_tilde_yz, s_tilde_zz
    )

    l_xx = _box_filter_3d(ux * ux, width=filter_width) - uxf * uxf
    l_xy = _box_filter_3d(ux * uy, width=filter_width) - uxf * uyf
    l_xz = _box_filter_3d(ux * uz, width=filter_width) - uxf * uzf
    l_yy = _box_filter_3d(uy * uy, width=filter_width) - uyf * uyf
    l_yz = _box_filter_3d(uy * uz, width=filter_width) - uyf * uzf
    l_zz = _box_filter_3d(uz * uz, width=filter_width) - uzf * uzf

    m_xx = 2.0 * (4.0 * s_tilde_mag * s_tilde_xx - s_mag * s_xx)
    m_xy = 2.0 * (4.0 * s_tilde_mag * s_tilde_xy - s_mag * s_xy)
    m_xz = 2.0 * (4.0 * s_tilde_mag * s_tilde_xz - s_mag * s_xz)
    m_yy = 2.0 * (4.0 * s_tilde_mag * s_tilde_yy - s_mag * s_yy)
    m_yz = 2.0 * (4.0 * s_tilde_mag * s_tilde_yz - s_mag * s_yz)
    m_zz = 2.0 * (4.0 * s_tilde_mag * s_tilde_zz - s_mag * s_zz)

    num = (l_xx * m_xx + l_yy * m_yy + l_zz * m_zz).mean()
    num = num + 2.0 * (l_xy * m_xy + l_xz * m_xz + l_yz * m_yz).mean()
    den = (m_xx**2 + m_yy**2 + m_zz**2).mean()
    den = den + 2.0 * (m_xy**2 + m_xz**2 + m_yz**2).mean()
    cs2 = torch.clamp(num / torch.clamp(den, min=1e-12), min=lambda_clip, max=0.1)
    cs = float(torch.sqrt(torch.clamp(cs2, min=0.0)).item())

    nu = (tau - 0.5) / 3.0
    nu_t = (cs**2) * s_mag
    tau_eff = torch.clamp(0.5 + 3.0 * (nu + nu_t), min=0.5001)

    # ---- MRT collision with per-cell stress relaxation rate ----
    M, M_inv = _get_d3q19_mrt_matrices(device)
    s_nu_flat = (1.0 / tau_eff).reshape(-1)  # (N,)

    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    f_flat = f.reshape(19, -1)
    feq_flat = feq.reshape(19, -1)

    m = M @ f_flat
    m_eq = M @ feq_flat
    dm = m - m_eq

    s_fixed = torch.tensor(
        [0.0, s_e, s_eps,
         0.0, s_q, 0.0, s_q, 0.0, s_q,
         0.0, 0.0, 0.0, 0.0, 0.0,
         s_pi, s_pi,
         1.0, 1.0, 1.0],
        dtype=f.dtype, device=device,
    )
    m_star = m - s_fixed.unsqueeze(1) * dm
    for k in (9, 10, 11, 12, 13):
        m_star[k] = m[k] - s_nu_flat * dm[k]
    return (M_inv @ m_star).reshape(19, nz, ny, nx)


def collide_dynamic_smagorinsky_bgk27(
    f: torch.Tensor,
    tau: float,
    filter_width: int = 2,
    lambda_clip: float = 0.0,
) -> torch.Tensor:
    """D3Q27 BGK collision with a dynamic Smagorinsky closure.

    Identical dynamic procedure to :func:`collide_dynamic_smagorinsky_bgk3d`
    but operates on the 27-velocity distribution, using the D3Q27 lattice
    constants, equilibrium, and macroscopic recovery.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        tau: Molecular relaxation time.
        filter_width: Test-filter width.
        lambda_clip: Minimum allowed value for ``C_s^2``.

    Returns:
        Updated distribution tensor of the same shape.
    """
    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    f_neq = f - feq
    s_xx, s_xy, s_xz, s_yy, s_yz, s_zz = _strain_from_fneq_27(f_neq, rho, tau)
    s_mag = _strain_magnitude_3d(s_xx, s_xy, s_xz, s_yy, s_yz, s_zz)

    uxf = _box_filter_3d(ux, width=filter_width)
    uyf = _box_filter_3d(uy, width=filter_width)
    uzf = _box_filter_3d(uz, width=filter_width)
    g11, g12, g13, g21, g22, g23, g31, g32, g33 = _velocity_gradients_3d(uxf, uyf, uzf)
    s_tilde_xx = g11
    s_tilde_yy = g22
    s_tilde_zz = g33
    s_tilde_xy = 0.5 * (g12 + g21)
    s_tilde_xz = 0.5 * (g13 + g31)
    s_tilde_yz = 0.5 * (g23 + g32)
    s_tilde_mag = _strain_magnitude_3d(
        s_tilde_xx, s_tilde_xy, s_tilde_xz, s_tilde_yy, s_tilde_yz, s_tilde_zz
    )

    l_xx = _box_filter_3d(ux * ux, width=filter_width) - uxf * uxf
    l_xy = _box_filter_3d(ux * uy, width=filter_width) - uxf * uyf
    l_xz = _box_filter_3d(ux * uz, width=filter_width) - uxf * uzf
    l_yy = _box_filter_3d(uy * uy, width=filter_width) - uyf * uyf
    l_yz = _box_filter_3d(uy * uz, width=filter_width) - uyf * uzf
    l_zz = _box_filter_3d(uz * uz, width=filter_width) - uzf * uzf

    m_xx = 2.0 * (4.0 * s_tilde_mag * s_tilde_xx - s_mag * s_xx)
    m_xy = 2.0 * (4.0 * s_tilde_mag * s_tilde_xy - s_mag * s_xy)
    m_xz = 2.0 * (4.0 * s_tilde_mag * s_tilde_xz - s_mag * s_xz)
    m_yy = 2.0 * (4.0 * s_tilde_mag * s_tilde_yy - s_mag * s_yy)
    m_yz = 2.0 * (4.0 * s_tilde_mag * s_tilde_yz - s_mag * s_yz)
    m_zz = 2.0 * (4.0 * s_tilde_mag * s_tilde_zz - s_mag * s_zz)

    num = (l_xx * m_xx + l_yy * m_yy + l_zz * m_zz).mean()
    num = num + 2.0 * (l_xy * m_xy + l_xz * m_xz + l_yz * m_yz).mean()
    den = (m_xx**2 + m_yy**2 + m_zz**2).mean()
    den = den + 2.0 * (m_xy**2 + m_xz**2 + m_yz**2).mean()
    cs2 = torch.clamp(num / torch.clamp(den, min=1e-12), min=lambda_clip, max=0.1)
    cs = float(torch.sqrt(torch.clamp(cs2, min=0.0)).item())

    nu = (tau - 0.5) / 3.0
    nu_t = (cs**2) * s_mag
    tau_eff = torch.clamp(0.5 + 3.0 * (nu + nu_t), min=0.5001)
    return f - f_neq / tau_eff.unsqueeze(0)


def collide_dynamic_smagorinsky_mrt27(
    f: torch.Tensor,
    tau: float,
    filter_width: int = 2,
    lambda_clip: float = 0.0,
    s_e: float = 1.19,
    s_eps: float = 1.4,
    s_q: float = 1.2,
    s_pi: float | None = None,
) -> torch.Tensor:
    """D3Q27 MRT collision with a dynamic Smagorinsky closure.

    Combines the dynamic Smagorinsky procedure (Germano identity with
    test-filtering to compute a global :math:`C_s`) — identical to
    :func:`collide_dynamic_smagorinsky_bgk27` — with the D3Q27 MRT
    collision operator.  The per-cell effective relaxation time
    :math:`\\tau_{eff}(x)` replaces the stress-mode relaxation rate
    (modes 5–9), exactly as in :func:`collide_smagorinsky_mrt27`, while
    the non-stress modes use the fixed MRT rates.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        tau: Molecular relaxation time.
        filter_width: Test-filter width.
        lambda_clip: Minimum allowed value for ``C_s^2``.
        s_e: Relaxation rate for the energy moment (row 4).
        s_eps: Relaxation rate for the energy-square moment (row 19).
        s_q: Relaxation rate for 3rd-order heat-flux moments (rows 10–18).
        s_pi: Relaxation rate for 4th-order+ moments (rows 20–26);
              defaults to *s_e* when *None*.

    Returns:
        Updated distribution tensor of the same shape.
    """
    if s_pi is None:
        s_pi = s_e

    device = f.device

    # ---- Dynamic Smagorinsky: compute global Cs via Germano identity ----
    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    f_neq = f - feq
    s_xx, s_xy, s_xz, s_yy, s_yz, s_zz = _strain_from_fneq_27(f_neq, rho, tau)
    s_mag = _strain_magnitude_3d(s_xx, s_xy, s_xz, s_yy, s_yz, s_zz)

    uxf = _box_filter_3d(ux, width=filter_width)
    uyf = _box_filter_3d(uy, width=filter_width)
    uzf = _box_filter_3d(uz, width=filter_width)
    g11, g12, g13, g21, g22, g23, g31, g32, g33 = _velocity_gradients_3d(uxf, uyf, uzf)
    s_tilde_xx = g11
    s_tilde_yy = g22
    s_tilde_zz = g33
    s_tilde_xy = 0.5 * (g12 + g21)
    s_tilde_xz = 0.5 * (g13 + g31)
    s_tilde_yz = 0.5 * (g23 + g32)
    s_tilde_mag = _strain_magnitude_3d(
        s_tilde_xx, s_tilde_xy, s_tilde_xz, s_tilde_yy, s_tilde_yz, s_tilde_zz
    )

    l_xx = _box_filter_3d(ux * ux, width=filter_width) - uxf * uxf
    l_xy = _box_filter_3d(ux * uy, width=filter_width) - uxf * uyf
    l_xz = _box_filter_3d(ux * uz, width=filter_width) - uxf * uzf
    l_yy = _box_filter_3d(uy * uy, width=filter_width) - uyf * uyf
    l_yz = _box_filter_3d(uy * uz, width=filter_width) - uyf * uzf
    l_zz = _box_filter_3d(uz * uz, width=filter_width) - uzf * uzf

    m_xx = 2.0 * (4.0 * s_tilde_mag * s_tilde_xx - s_mag * s_xx)
    m_xy = 2.0 * (4.0 * s_tilde_mag * s_tilde_xy - s_mag * s_xy)
    m_xz = 2.0 * (4.0 * s_tilde_mag * s_tilde_xz - s_mag * s_xz)
    m_yy = 2.0 * (4.0 * s_tilde_mag * s_tilde_yy - s_mag * s_yy)
    m_yz = 2.0 * (4.0 * s_tilde_mag * s_tilde_yz - s_mag * s_yz)
    m_zz = 2.0 * (4.0 * s_tilde_mag * s_tilde_zz - s_mag * s_zz)

    num = (l_xx * m_xx + l_yy * m_yy + l_zz * m_zz).mean()
    num = num + 2.0 * (l_xy * m_xy + l_xz * m_xz + l_yz * m_yz).mean()
    den = (m_xx**2 + m_yy**2 + m_zz**2).mean()
    den = den + 2.0 * (m_xy**2 + m_xz**2 + m_yz**2).mean()
    cs2 = torch.clamp(num / torch.clamp(den, min=1e-12), min=lambda_clip, max=0.1)
    cs = float(torch.sqrt(torch.clamp(cs2, min=0.0)).item())

    nu = (tau - 0.5) / 3.0
    nu_t = (cs**2) * s_mag
    tau_eff = torch.clamp(0.5 + 3.0 * (nu + nu_t), min=0.5001)

    # ---- MRT collision with per-cell stress relaxation rate ----
    M, M_inv = _get_d3q27_mrt_matrices(device)
    s_nu_flat = (1.0 / tau_eff).reshape(-1)  # (N,)

    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    f_flat = f.reshape(27, -1)
    feq_flat = feq.reshape(27, -1)

    m = M @ f_flat
    m_eq = M @ feq_flat
    dm = m - m_eq

    # Fixed relaxation rates for non-stress modes
    s_fixed = torch.tensor(
        [
            0.0,   # 0  mass
            0.0,   # 1  jx
            0.0,   # 2  jy
            0.0,   # 3  jz
            s_e,   # 4  energy
            0.0,   # 5  Nxx  – overridden below
            0.0,   # 6  Nyy  – overridden below
            0.0,   # 7  Pxy  – overridden below
            0.0,   # 8  Pxz  – overridden below
            0.0,   # 9  Pyz  – overridden below
            s_q,   # 10
            s_q,   # 11
            s_q,   # 12
            s_q,   # 13
            s_q,   # 14
            s_q,   # 15
            s_q,   # 16
            s_q,   # 17
            s_q,   # 18
            s_eps, # 19
            s_pi,  # 20
            s_pi,  # 21
            s_pi,  # 22
            s_pi,  # 23
            s_pi,  # 24
            s_pi,  # 25
            s_pi,  # 26
        ],
        dtype=f.dtype,
        device=device,
    )
    m_star = m - s_fixed.unsqueeze(1) * dm
    # Override stress modes 5–9 with spatially varying dynamic rate
    for k in (5, 6, 7, 8, 9):
        m_star[k] = m[k] - s_nu_flat * dm[k]
    return (M_inv @ m_star).reshape(27, nz, ny, nx)


__all__ = [
    # Smagorinsky
    "collide_smagorinsky_bgk",
    "collide_smagorinsky_mrt",
    "collide_smagorinsky_bgk3d",
    "collide_smagorinsky_mrt3d",
    "collide_smagorinsky_bgk27",
    "collide_smagorinsky_mrt27",
    "collide_dynamic_smagorinsky_bgk",
    "collide_dynamic_smagorinsky_bgk3d",
    "collide_dynamic_smagorinsky_mrt3d",
    "collide_dynamic_smagorinsky_bgk27",
    "collide_dynamic_smagorinsky_mrt27",
    # WALE
    "collide_wale_bgk",
    "collide_wale_bgk3d",
    "collide_wale_bgk27",
    # Vreman
    "collide_vreman_bgk",
    "collide_vreman_bgk3d",
    "collide_vreman_bgk27",
]
