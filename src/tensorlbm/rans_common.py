"""Common RANS collision module — delegates to public collision kernels.

RANS models (k-epsilon, Spalart-Allmaras, k-omega SST) compute a per-cell
eddy-viscosity field ``nu_t``, convert it to an effective relaxation time via
:func:`~tensorlbm.turbulence._nu_t_to_tau_eff` (the *same* helper used by
Smagorinsky / WALE / Vreman), and delegate to the common BGK / MRT collision.

No collision logic is duplicated: the RANS module only computes ``nu_t`` and
hands it to the shared per-cell-``tau_eff`` collision kernels.

Hot-path invariants
-------------------
* ``nu_t`` is always a per-cell field (``ndim == spatial_dims``), never a
  scalar — no ``.mean().item()`` averaging.
* No GPU→CPU syncs (``.item()``, ``float(tensor)``, ``bool(tensor)``) inside
  the collision path.
* No per-call ``mask.bool()`` allocation — masks are pre-computed by the
  caller.
"""
from __future__ import annotations

import torch

from .d3q19 import equilibrium3d, macroscopic3d
from .d3q27 import _get_d3q27_mrt_matrices, equilibrium27, macroscopic27
from .solver3d import _get_d3q19_mrt_matrices
from .turbulence import _nu_t_to_tau_eff

__all__ = [
    "collide_rans_bgk3d",
    "collide_rans_mrt3d",
    "collide_rans_bgk27",
    "collide_rans_mrt27",
    "collide_rans_3d",
]


# ---------------------------------------------------------------------------
# D3Q19
# ---------------------------------------------------------------------------

def collide_rans_bgk3d(
    f: torch.Tensor,
    tau: float,
    nu_t: torch.Tensor,
) -> torch.Tensor:
    """D3Q19 BGK collision with RANS per-cell eddy viscosity.

    Computes the equilibrium, converts the per-cell ``nu_t`` to an effective
    relaxation time via ``_nu_t_to_tau_eff`` (same as WALE / Vreman BGK), and
    applies the BGK relaxation.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        tau: Molecular (baseline) relaxation time :math:`\\tau_0 > 0.5`.
        nu_t: Per-cell turbulent eddy viscosity, shape ``(nz, ny, nx)``.

    Returns:
        Post-collision distribution of the same shape as *f*.
    """
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    f_neq = f - feq
    tau_eff = _nu_t_to_tau_eff(tau, nu_t)
    return f - f_neq / tau_eff.unsqueeze(0)


def collide_rans_mrt3d(
    f: torch.Tensor,
    tau: float,
    nu_t: torch.Tensor,
    s_e: float = 1.19,
    s_eps: float = 1.4,
    s_q: float = 1.2,
    s_pi: float | None = None,
) -> torch.Tensor:
    """D3Q19 MRT collision with RANS per-cell eddy viscosity.

    The stress-mode relaxation rate (modes 9–13) is replaced by the per-cell
    ``1/τ_eff(x)`` derived from the RANS eddy viscosity, exactly as in
    :func:`~tensorlbm.turbulence.collide_smagorinsky_mrt3d` but with ``nu_t``
    from the RANS model instead of the Smagorinsky formula.

    Args:
        f: Distribution tensor of shape ``(19, nz, ny, nx)``.
        tau: Molecular relaxation time :math:`\\tau_0 > 0.5`.
        nu_t: Per-cell turbulent eddy viscosity, shape ``(nz, ny, nx)``.
        s_e: Relaxation rate for the energy moment.
        s_eps: Relaxation rate for the energy-square moment.
        s_q: Relaxation rate for heat-flux moments.
        s_pi: Relaxation rate for higher-order stress moments
              (defaults to *s_e* when *None*).

    Returns:
        Post-collision distribution of the same shape as *f*.
    """
    if s_pi is None:
        s_pi = s_e

    device = f.device
    M, M_inv = _get_d3q19_mrt_matrices(device)

    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz)
    f_neq = f - feq
    tau_eff = _nu_t_to_tau_eff(tau, nu_t)
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
    # Override stress modes 9–13 with the per-cell RANS rate
    for k in (9, 10, 11, 12, 13):
        m_star[k] = m[k] - s_nu_flat * dm[k]
    return (M_inv @ m_star).reshape(19, nz, ny, nx)


# ---------------------------------------------------------------------------
# D3Q27
# ---------------------------------------------------------------------------

def collide_rans_bgk27(
    f: torch.Tensor,
    tau: float,
    nu_t: torch.Tensor,
) -> torch.Tensor:
    """D3Q27 BGK collision with RANS per-cell eddy viscosity.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        tau: Molecular relaxation time :math:`\\tau_0 > 0.5`.
        nu_t: Per-cell turbulent eddy viscosity, shape ``(nz, ny, nx)``.

    Returns:
        Post-collision distribution of the same shape as *f*.
    """
    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    f_neq = f - feq
    tau_eff = _nu_t_to_tau_eff(tau, nu_t)
    return f - f_neq / tau_eff.unsqueeze(0)


def collide_rans_mrt27(
    f: torch.Tensor,
    tau: float,
    nu_t: torch.Tensor,
    s_e: float = 1.19,
    s_eps: float = 1.4,
    s_q: float = 1.2,
    s_pi: float | None = None,
) -> torch.Tensor:
    """D3Q27 MRT collision with RANS per-cell eddy viscosity.

    The stress-mode relaxation rate (modes 5–9) is replaced by the per-cell
    ``1/τ_eff(x)`` derived from the RANS eddy viscosity, exactly as in
    :func:`~tensorlbm.turbulence.collide_smagorinsky_mrt27` but with ``nu_t``
    from the RANS model.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        tau: Molecular relaxation time :math:`\\tau_0 > 0.5`.
        nu_t: Per-cell turbulent eddy viscosity, shape ``(nz, ny, nx)``.
        s_e: Relaxation rate for the energy moment (row 4).
        s_eps: Relaxation rate for the energy-square moment (row 19).
        s_q: Relaxation rate for 3rd-order heat-flux moments (rows 10–18).
        s_pi: Relaxation rate for 4th-order+ moments (rows 20–26);
              defaults to *s_e* when *None*.

    Returns:
        Post-collision distribution of the same shape as *f*.
    """
    if s_pi is None:
        s_pi = s_e

    device = f.device
    M, M_inv = _get_d3q27_mrt_matrices(device)

    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    f_neq = f - feq
    tau_eff = _nu_t_to_tau_eff(tau, nu_t)
    s_nu_flat = (1.0 / tau_eff).reshape(-1)  # (N,)

    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    f_flat = f.reshape(27, -1)
    feq_flat = feq.reshape(27, -1)

    m = M @ f_flat
    m_eq = M @ feq_flat
    dm = m - m_eq

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
        dtype=f.dtype, device=device,
    )
    m_star = m - s_fixed.unsqueeze(1) * dm
    # Override stress modes 5–9 with the per-cell RANS rate
    for k in (5, 6, 7, 8, 9):
        m_star[k] = m[k] - s_nu_flat * dm[k]
    return (M_inv @ m_star).reshape(27, nz, ny, nx)


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------

def collide_rans_3d(
    lattice: str,
    collision: str,
    f: torch.Tensor,
    *,
    tau: float,
    nu_t: torch.Tensor,
    **rates: float,
) -> torch.Tensor:
    """Unified dispatch for RANS collision with per-cell eddy viscosity.

    Selects the appropriate BGK / MRT kernel for D3Q19 or D3Q27 and delegates.
    The caller supplies a per-cell ``nu_t`` field (computed by the RANS solver)
    and the molecular relaxation time ``tau``; the dispatch converts ``nu_t`` to
    ``tau_eff`` via ``_nu_t_to_tau_eff`` and runs the common collision.

    Args:
        lattice: Lattice name — ``"D3Q19"`` or ``"D3Q27"`` (case-insensitive).
        collision: Collision family — ``"BGK"`` or ``"MRT"`` (case-insensitive).
        f: Distribution tensor ``(19, nz, ny, nx)`` or ``(27, nz, ny, nx)``.
        tau: Molecular relaxation time :math:`\\tau_0 > 0.5`.
        nu_t: Per-cell turbulent eddy viscosity, shape ``(nz, ny, nx)``.
        **rates: Additional MRT relaxation rates (``s_e``, ``s_eps``,
                 ``s_q``, ``s_pi``).

    Returns:
        Post-collision distribution of the same shape as *f*.

    Raises:
        ValueError: If the lattice/collision combination is not supported.
    """
    lattice_u = lattice.upper()
    collision_u = collision.upper()

    if lattice_u == "D3Q19":
        expected_q = 19
    elif lattice_u == "D3Q27":
        expected_q = 27
    else:
        raise ValueError(
            f"lattice must be 'D3Q19' or 'D3Q27', got {lattice!r}"
        )

    if f.ndim != 4 or f.shape[0] != expected_q:
        raise ValueError(
            f"{lattice_u} populations must have shape ({expected_q}, nz, ny, nx), "
            f"got {tuple(f.shape)}"
        )

    if collision_u == "BGK":
        if lattice_u == "D3Q19":
            return collide_rans_bgk3d(f, tau, nu_t)
        return collide_rans_bgk27(f, tau, nu_t)
    if collision_u == "MRT":
        if lattice_u == "D3Q19":
            return collide_rans_mrt3d(f, tau, nu_t, **rates)
        return collide_rans_mrt27(f, tau, nu_t, **rates)

    raise ValueError(
        f"collision must be 'BGK' or 'MRT', got {collision!r}"
    )
