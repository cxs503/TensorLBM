from __future__ import annotations

import torch

from .boundaries import apply_simple_channel_boundaries, bounce_back_cells, cylinder_mask, make_channel_wall_mask
from .d2q9 import C, equilibrium, macroscopic

# ---------------------------------------------------------------------------
# D2Q9 MRT transformation matrix (d'Humières et al. 2002)
# Row order: rho, e, epsilon, jx, qx, jy, qy, pxx, pxy
# Velocity order matches d2q9.C: rest, +x, +y, -x, -y, +x+y, -x+y, -x-y, +x-y
# ---------------------------------------------------------------------------
_M_D2Q9 = torch.tensor(
    [
        [1,  1,  1,  1,  1,  1,  1,  1,  1],   # rho
        [-4, -1, -1, -1, -1,  2,  2,  2,  2],  # e
        [4,  -2, -2, -2, -2,  1,  1,  1,  1],  # epsilon
        [0,   1,  0, -1,  0,  1, -1, -1,  1],  # jx
        [0,  -2,  0,  2,  0,  1, -1, -1,  1],  # qx
        [0,   0,  1,  0, -1,  1,  1, -1, -1],  # jy
        [0,   0, -2,  0,  2,  1,  1, -1, -1],  # qy
        [0,   1, -1,  1, -1,  0,  0,  0,  0],  # pxx
        [0,   0,  0,  0,  0,  1, -1,  1, -1],  # pxy
    ],
    dtype=torch.float32,
)
_M_INV_D2Q9 = torch.linalg.inv(_M_D2Q9)


def collide_bgk(f: torch.Tensor, tau: float) -> torch.Tensor:
    """Single-relaxation-time BGK collision step."""
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    return f - (f - feq) / tau


def collide_mrt(f: torch.Tensor, tau: float, s: torch.Tensor | None = None) -> torch.Tensor:
    """Multiple-relaxation-time (MRT) BGK collision step for D2Q9.

    Uses the d'Humières transformation matrix.  The nine relaxation rates
    correspond to the moment ordering: rho, e, epsilon, jx, qx, jy, qy,
    pxx, pxy.  Conserved modes (rho, jx, jy) are never relaxed regardless
    of the values supplied for those positions in *s*.

    Args:
        f:   Distribution tensor of shape ``(9, ny, nx)``.
        tau: Viscous relaxation time.  Used for the stress modes (pxx, pxy)
             when *s* is ``None``.
        s:   Optional length-9 tensor of per-mode relaxation rates.  When
             ``None`` the standard defaults are used:
             ``[0, 1.4, 1.4, 0, 1.2, 0, 1.2, 1/tau, 1/tau]``.

    Returns:
        Post-collision distribution tensor with the same shape as *f*.
    """
    device = f.device
    M = _M_D2Q9.to(device)
    M_inv = _M_INV_D2Q9.to(device)

    if s is None:
        s_nu = 1.0 / tau
        s = torch.tensor(
            [0.0, 1.4, 1.4, 0.0, 1.2, 0.0, 1.2, s_nu, s_nu],
            dtype=torch.float32,
            device=device,
        )
    else:
        s = s.to(device)

    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)

    ny, nx = f.shape[1], f.shape[2]
    f_flat = f.view(9, -1)
    feq_flat = feq.view(9, -1)

    m = M @ f_flat        # (9, ny*nx)
    meq = M @ feq_flat    # (9, ny*nx)

    m_out = m - s.unsqueeze(-1) * (m - meq)

    return (M_inv @ m_out).view(9, ny, nx)


def stream(f: torch.Tensor) -> torch.Tensor:
    """Streaming by shifting each discrete direction."""
    streamed = torch.empty_like(f)
    for i in range(9):
        cx, cy = int(C[i, 0].item()), int(C[i, 1].item())
        streamed[i] = torch.roll(f[i], shifts=(cy, cx), dims=(0, 1))
    return streamed


__all__ = [
    "cylinder_mask",
    "make_channel_wall_mask",
    "bounce_back_cells",
    "apply_simple_channel_boundaries",
    "collide_bgk",
    "collide_mrt",
    "stream",
]
