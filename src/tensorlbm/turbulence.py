"""Smagorinsky large-eddy-simulation (LES) turbulence sub-grid models.

These functions augment the standard BGK and MRT collision operators with a
local effective relaxation time computed from the non-equilibrium stress
magnitude following Hou *et al.* (1994) and Yu *et al.* (2006).

The self-consistent effective relaxation time at each cell is:

.. math::

    \\tau_{eff} = \\frac{1}{2}\\left(\\tau_0 +
        \\sqrt{\\tau_0^2 + 18\\,C_s^2\\,\\frac{|\\Pi^{neq}|_F}{\\rho}}\\right)

where :math:`|\\Pi^{neq}|_F` is the Frobenius norm of the non-equilibrium
stress tensor and :math:`C_s` is the Smagorinsky constant (typically 0.1).

Exported functions
------------------
- :func:`collide_smagorinsky_bgk`   – D2Q9 BGK + Smagorinsky
- :func:`collide_smagorinsky_bgk3d` – D3Q19 BGK + Smagorinsky
- :func:`collide_smagorinsky_mrt3d` – D3Q19 MRT + Smagorinsky (recommended for
  high-Reynolds ship flows)
"""

from __future__ import annotations

import torch

from .d2q9 import C as C2D
from .d2q9 import equilibrium, macroscopic
from .d3q19 import C as C3D
from .d3q19 import equilibrium3d, macroscopic3d
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


__all__ = [
    "collide_smagorinsky_bgk",
    "collide_smagorinsky_bgk3d",
    "collide_smagorinsky_mrt3d",
]
