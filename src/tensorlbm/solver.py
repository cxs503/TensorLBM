from __future__ import annotations

import functools

import torch

from .boundaries import apply_simple_channel_boundaries, bounce_back_cells, cylinder_mask, make_channel_wall_mask
from .d2q9 import C, equilibrium, macroscopic

# ---------------------------------------------------------------------------
# D2Q9 MRT transformation matrix and its inverse (precomputed from numpy)
# Reference: d'Humières et al. (2002), Phil. Trans. R. Soc. Lond. A.
# Velocity ordering: 0:(0,0), 1:(1,0), 2:(0,1), 3:(-1,0), 4:(0,-1),
#                    5:(1,1), 6:(-1,1), 7:(-1,-1), 8:(1,-1)
# ---------------------------------------------------------------------------
_M_D2Q9_DATA = [
    [ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
    [-4.0, -1.0, -1.0, -1.0, -1.0,  2.0,  2.0,  2.0,  2.0],
    [ 4.0, -2.0, -2.0, -2.0, -2.0,  1.0,  1.0,  1.0,  1.0],
    [ 0.0,  1.0,  0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0],
    [ 0.0, -2.0,  0.0,  2.0,  0.0,  1.0, -1.0, -1.0,  1.0],
    [ 0.0,  0.0,  1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0],
    [ 0.0,  0.0, -2.0,  0.0,  2.0,  1.0,  1.0, -1.0, -1.0],
    [ 0.0,  1.0, -1.0,  1.0, -1.0,  0.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  0.0,  0.0,  1.0, -1.0,  1.0, -1.0],
]

# Invert the float64 matrix with numpy once and store as a Python list
def _invert_d2q9() -> list[list[float]]:
    import numpy as np
    M = np.array(_M_D2Q9_DATA, dtype=np.float64)
    return np.linalg.inv(M).tolist()


_M_D2Q9_INV_DATA = _invert_d2q9()


@functools.lru_cache(maxsize=None)
def _get_d2q9_mrt_matrices(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    M = torch.tensor(_M_D2Q9_DATA, dtype=torch.float32, device=device)
    M_inv = torch.tensor(_M_D2Q9_INV_DATA, dtype=torch.float32, device=device)
    return M, M_inv


# ---------------------------------------------------------------------------
# Collision operators
# ---------------------------------------------------------------------------

def collide_bgk(f: torch.Tensor, tau: float) -> torch.Tensor:
    """Single-relaxation-time BGK collision step."""
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    return f - (f - feq) / tau


def collide_mrt(
    f: torch.Tensor,
    tau: float,
    s_e: float = 1.64,
    s_eps: float = 1.54,
    s_q: float = 1.7,
) -> torch.Tensor:
    """Multi-relaxation-time (MRT) collision step for D2Q9.

    The physical shear viscosity is controlled by *tau* exactly as in BGK:
    ν = (τ − ½)/3.  The extra relaxation rates *s_e*, *s_eps*, *s_q* damp
    the non-hydrodynamic moments and can be tuned independently to improve
    numerical stability at high Reynolds numbers.

    Moment ordering (rows of M):
        0: ρ  (conserved, s=0)
        1: e  (energy,          s=s_e)
        2: ε  (energy-square,   s=s_eps)
        3: jx (conserved, s=0)
        4: qx (heat-flux x,     s=s_q)
        5: jy (conserved, s=0)
        6: qy (heat-flux y,     s=s_q)
        7: pxx (stress,         s=1/tau)
        8: pxy (stress,         s=1/tau)

    Args:
        f: Distribution tensor of shape ``(9, ny, nx)``.
        tau: Relaxation time for shear stress (τ > ½).
        s_e: Relaxation rate for energy moment.
        s_eps: Relaxation rate for energy-square moment.
        s_q: Relaxation rate for heat-flux moments.

    Returns:
        Updated distribution tensor of the same shape.
    """
    device = f.device
    M, M_inv = _get_d2q9_mrt_matrices(device)

    s_nu = 1.0 / tau
    s_vec = torch.tensor([0.0, s_e, s_eps, 0.0, s_q, 0.0, s_q, s_nu, s_nu],
                         dtype=f.dtype, device=device)  # (9,)

    ny, nx = f.shape[1], f.shape[2]
    f_flat = f.reshape(9, -1)                  # (9, N)
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    feq_flat = feq.reshape(9, -1)

    m = M @ f_flat                             # (9, N)
    m_eq = M @ feq_flat                        # (9, N)
    m_star = m - s_vec.unsqueeze(1) * (m - m_eq)  # (9, N)
    return (M_inv @ m_star).reshape(9, ny, nx)


# ---------------------------------------------------------------------------
# Streaming step
# ---------------------------------------------------------------------------

def stream(f: torch.Tensor) -> torch.Tensor:
    """Vectorised streaming by gathering from shifted source indices (periodic).

    Replaces the per-direction ``torch.roll`` loop with a single advanced-index
    gather, which is more GPU-friendly.
    """
    ny, nx = f.shape[1], f.shape[2]
    device = f.device
    c = C.to(device)  # (9, 2) — columns are (cx, cy)

    # Source row and column for each direction (periodic wrap)
    y_src = (torch.arange(ny, device=device).unsqueeze(0) - c[:, 1].unsqueeze(1)) % ny  # (9, ny)
    x_src = (torch.arange(nx, device=device).unsqueeze(0) - c[:, 0].unsqueeze(1)) % nx  # (9, nx)

    # Expand to full (9, ny, nx) index tensors
    q_idx = torch.arange(9, device=device).view(9, 1, 1).expand(9, ny, nx)
    y_idx = y_src.unsqueeze(2).expand(9, ny, nx)
    x_idx = x_src.unsqueeze(1).expand(9, ny, nx)

    return f[q_idx, y_idx, x_idx]


__all__ = [
    "cylinder_mask",
    "make_channel_wall_mask",
    "bounce_back_cells",
    "apply_simple_channel_boundaries",
    "collide_bgk",
    "collide_mrt",
    "stream",
]
