"""Exact laminar anchors for the viscoelastic channel.

Geometry convention (identical to ``tensorlbm.viscoelastic.lbm``): a D2Q9
channel with ``H`` fluid rows ``j = 1..H`` between walls located half a lattice
spacing outside the first/last fluid row, i.e. at ``y = 0.5`` and ``y = H + 0.5``.
Node ``j`` therefore sits at ``y = j`` and the channel centre is
``y_c = (H + 1) / 2``.

Oldroyd-B plane Poiseuille (body force ``G``, total kinematic viscosity
``nu0 = nu_s + nu_p``)
-------------------------------------------------------------------------
Because the polymeric shear stress of Oldroyd-B in *steady* shear is
``tau_xy = eta_p * gdot`` -- independent of ``lambda`` -- the steady momentum
balance is ``eta_0 u'' = -rho G``, i.e. the velocity profile is exactly the
Newtonian parabola evaluated with the *total* viscosity::

    u(y) = G / (2 nu0) * [ (H/2)^2 - (y - y_c)^2 ]

with ``u_max = G H^2 / (8 nu0)`` and wall shear rate ``gdot_w = G H / (2 nu0)``.
The polymeric contributions are the shear stress ``tau_xy = eta_p u'`` and the
first normal-stress difference

    N1(y) = tau_xx - tau_yy = 2 eta_p lambda gdot(y)^2 ,

so ``N1`` scales with ``Wi^2`` while the *velocity* profile is
``Wi``-independent: this is the sharpest statement of the Oldroyd-B anchor and
the reason a Newtonian-limit test (``lambda -> 0`` with ``eta_p`` fixed) must
recover the same parabola while ``N1 -> 0``.
"""

from __future__ import annotations

import torch

__all__ = [
    "oldroyd_b_poiseuille_conformation",
    "oldroyd_b_poiseuille_fields",
    "oldroyd_b_poiseuille_profile",
    "oldroyd_b_steady_shear_conformation",
    "poiseuille_max_velocity",
    "wall_shear_rate",
]


def poiseuille_max_velocity(G: float, H: int, nu: float) -> float:
    """``u_max = G H^2 / (8 nu)`` (lattice units, ``rho = 1``)."""
    return float(G) * float(H) ** 2 / (8.0 * float(nu))


def wall_shear_rate(G: float, H: int, nu: float) -> float:
    """``gdot_w = G H / (2 nu)`` -- the shear rate at the wall."""
    return float(G) * float(H) / (2.0 * float(nu))


def oldroyd_b_poiseuille_profile(
    H: int, G: float, nu0: float, *, dtype=torch.float64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytic Oldroyd-B velocity profile ``(y, u)`` on all ``H + 2`` rows.

    Wall rows ``0``/``-1`` are set to zero (no slip at ``y = 0.5`` / ``H + 0.5``).
    """
    ny = H + 2
    y = torch.arange(ny, dtype=dtype)
    yc = 0.5 * (H + 1.0)
    u = float(G) / (2.0 * float(nu0)) * (0.25 * float(H) ** 2 - (y - yc) ** 2)
    u[0] = 0.0
    u[-1] = 0.0
    return y, u


def oldroyd_b_poiseuille_fields(
    H: int,
    G: float,
    nu0: float,
    *,
    beta: float = 0.5,
    lam: float = 1.0,
    dtype=torch.float64,
) -> dict:
    """Exact Oldroyd-B plane-Poiseuille fields (velocity, stress, ``N1``).

    Args:
        H: number of fluid rows.
        G: body-force acceleration in ``+x``.
        nu0: total kinematic viscosity ``nu_s + nu_p``.
        beta: solvent fraction ``nu_s / nu0``.
        lam: relaxation time ``lambda``.

    Returns:
        dict with ``y, u, gamma_dot`` (signed ``du/dy``), ``wi``,
        ``C_xx, C_xy, C_yy``, ``tau_xy, n1`` (first normal-stress difference),
        ``nu_s, nu_p, wi_wall`` and the scalars ``u_max``/``gdot_wall``.
    """
    nu_s = float(beta) * float(nu0)
    nu_p = float(nu0) - nu_s
    y, u = oldroyd_b_poiseuille_profile(H, G, nu0, dtype=dtype)
    yc = 0.5 * (H + 1.0)
    gamma_dot = -(float(G) / float(nu0)) * (y - yc)
    gamma_dot[0] = 0.0
    gamma_dot[-1] = 0.0
    wi = float(lam) * gamma_dot
    c_xx = 1.0 + 2.0 * wi**2
    c_xy = wi
    c_yy = torch.ones_like(wi)
    tau_xy = nu_p * gamma_dot
    n1 = 2.0 * nu_p * float(lam) * gamma_dot**2
    return {
        "y": y,
        "u": u,
        "gamma_dot": gamma_dot,
        "wi": wi,
        "C_xx": c_xx,
        "C_xy": c_xy,
        "C_yy": c_yy,
        "tau_xy": tau_xy,
        "n1": n1,
        "nu_s": nu_s,
        "nu_p": nu_p,
        "u_max": poiseuille_max_velocity(G, H, nu0),
        "gdot_wall": wall_shear_rate(G, H, nu0),
        "wi_wall": abs(float(lam) * wall_shear_rate(G, H, nu0)),
    }


def oldroyd_b_steady_shear_conformation(Wi: float) -> tuple[float, float, float, float]:
    """Exact steady simple-shear conformation ``(C_xx, C_xy, C_yy, C_zz)``."""
    return 1.0 + 2.0 * Wi * Wi, Wi, 1.0, 1.0

def oldroyd_b_poiseuille_conformation(
    H: int,
    G: float,
    nu0: float,
    *,
    beta: float = 0.5,
    lam: float = 1.0,
    dtype=torch.float64,
) -> torch.Tensor:
    """Exact steady Oldroyd-B conformation field ``(4, ny)`` for the channel.

    ``C_xx = 1 + 2 Wi(y)^2``, ``C_xy = Wi(y)``, ``C_yy = C_zz = 1`` with
    ``Wi(y) = lambda * gdot(y)``; wall rows are set to ``I``.  Used (a) as an
    exact anchor and (b) as an initial condition that starts the solver on the
    Oldroyd-B fixed point so that verification runs do not pay the full
    ``~5 lambda`` stress transient.
    """
    fields = oldroyd_b_poiseuille_fields(H, G, nu0, beta=beta, lam=lam, dtype=dtype)
    wi = fields["wi"]
    ny = H + 2
    out = torch.zeros((4, ny), dtype=dtype)
    out[0] = 1.0 + 2.0 * wi**2
    out[1] = wi
    out[2] = 1.0
    out[3] = 1.0
    out[0, 0] = out[0, -1] = 1.0
    out[1, 0] = out[1, -1] = 0.0
    return out
