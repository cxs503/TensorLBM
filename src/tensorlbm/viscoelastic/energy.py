"""Finite-volume energy budget of the viscoelastic channel.

For a steady incompressible channel driven by a constant body acceleration ``G``
the mechanical energy identity is

    P_in = eps_s + eps_p + dE_el/dt

with, on the finite-volume face formulation (``Delta u_f = u_{j+1} - u_j`` on
face ``f = j + 1/2``, ``tau_bar_f`` the face-averaged polymer shear stress)::

    P_in  = sum_j G u_j                     (driving power per unit width)
    eps_s = nu_s sum_faces (Delta u_f)^2     (solvent viscous dissipation)
    eps_p = sum_faces tau_bar_f Delta u_f    (polymer working, >= 0 for Oldroyd-B)
    E_el  = sum_j (nu_p / 2 lambda) tr(C - I)    (rate -> 0 at steady state)

Form of the discrete budget
---------------------------
The budget is written in *work form* with the same operators that enter the
momentum balance of the solver::

    eps_s = -nu_s sum_j u_j L2(u)_j ,   eps_p = -sum_j u_j D(tau_xy)_j

where ``L2`` is the second-order discrete Laplacian that honours the half-way
no-slip plane at ``y = 0.5`` (``L2(u)_1 = -4 u_1 + (4/3) u_2``, symmetric at the
top wall) and ``D`` is :func:`ddy_2nd`.  Both are *exact* for the analytic
anchor (quadratic velocity, linear shear stress), hence

    P_in - eps_s - eps_p = sum_j u_j r_j ,   r_j = G + D(tau)_j + nu_s L2(u)_j

is round-off for the anchor field and equals the truncation-limited momentum
residual for the LBM field.  The residual is reported as measured in both cases
(``budget_grid_scaling`` in the study artifact shows its ``O(h^2)`` decay); it is
never asserted to vanish for the LBM field.
"""

from __future__ import annotations

import torch

from .conformation import ddy_2nd, elastic_energy
from .lbm import ViscoelasticParams, _stress_components

__all__ = ["elastic_energy", "energy_budget"]


def _laplacian_halfway(u: torch.Tensor) -> torch.Tensor:
    """Second-order ``d^2/dy^2`` honouring the half-way no-slip planes.

    Interior rows: central difference.  First/last fluid rows: the 3-point
    Lagrange second derivative on the non-uniform stencil ``(0.5, 1, 2)`` with the
    wall value folded in, ``-4 u_1 + (4/3) u_2`` (exact for a quadratic, unlike
    the ``u_2 - 2u_1 + u_0`` form with the wall node value).
    """
    lap = torch.zeros_like(u)
    lap[1:-1] = u[2:] - 2.0 * u[1:-1] + u[:-2]
    lap[1] = -4.0 * u[1] + (4.0 / 3.0) * u[2]
    lap[-2] = -4.0 * u[-2] + (4.0 / 3.0) * u[-3]
    return lap


def energy_budget(
    ux: torch.Tensor,
    C: torch.Tensor,
    params: ViscoelasticParams,
    *,
    G: float | None = None,
    dE_el_dt: float = 0.0,
) -> dict:
    """Evaluate ``P_in - eps_s - eps_p - dE_el/dt`` for a channel field.

    Args:
        ux: streamwise velocity ``(ny, nx)``; the x-mean enters (the channel is
            x-uniform, averaging removes residual x-noise).
        C: conformation tensor ``(4, ny, nx)``.
        params: solver parameters (``nu0, beta, lam, model, L2``).
        G: driving acceleration (defaults to ``params.G``).
        dE_el_dt: elastic free-energy rate (zero at steady state).

    Returns:
        dict with ``p_in``, ``eps_s``, ``eps_p``, ``d_el_dt``, ``residual_rel``
        and the fractions of the input power carried by each channel.
    """
    G_val = float(params.G if G is None else G)
    H = int(ux.shape[0]) - 2
    u = ux.mean(dim=1).clone()
    u[0] = 0.0
    u[-1] = 0.0  # wall rows (unused: the operators above do not read them)

    tau_xy = _stress_components(C, params)[1].mean(dim=1)
    tau_xy[0] = tau_xy[-1] = 0.0
    div_tau = ddy_2nd(tau_xy.reshape(-1, 1)).reshape(-1)
    lap = _laplacian_halfway(u)

    p_in = float(G_val * u[1:-1].sum())
    eps_s = float(-params.nu_s * (u * lap).sum())
    eps_p = float(-(u * div_tau).sum())
    denom = abs(p_in) if abs(p_in) > 0 else 1.0
    residual = (p_in - eps_s - eps_p - float(dE_el_dt)) / denom
    return {
        "p_in": p_in,
        "eps_s": eps_s,
        "eps_p": eps_p,
        "d_el_dt": float(dE_el_dt),
        "residual_rel": residual,
        "solvent_fraction": eps_s / denom,
        "polymer_fraction": eps_p / denom,
        "nu_s": params.nu_s,
        "nu_p": params.nu_p,
        "H": H,
    }
