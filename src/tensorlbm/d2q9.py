"""Core D2Q9 utilities for a lightweight PyTorch LBM implementation."""

from __future__ import annotations

import torch

# Direction vectors (cx, cy) following a standard D2Q9 ordering.
C = torch.tensor(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1],
    ],
    dtype=torch.int64,
)

# Lattice weights for each direction in D2Q9.
W = torch.tensor([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])

# Opposite direction index map, used in bounce-back boundary handling.
OPPOSITE = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.int64)


def equilibrium(rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Compute D2Q9 equilibrium distribution f_eq from density and velocity.

    Args:
        rho: Density tensor with shape ``(..., ny, nx)`` or compatible.
        u: Velocity tensor with shape ``(..., ny, nx, 2)``.

    Returns:
        Tensor with shape ``(..., ny, nx, 9)``.
    """
    if u.shape[-1] != 2:
        msg = "u must have a final dimension of size 2"
        raise ValueError(msg)

    c = C.to(device=u.device, dtype=u.dtype)
    w = W.to(device=u.device, dtype=u.dtype)

    cu = torch.einsum("...d,qd->...q", u, c)
    u2 = (u * u).sum(dim=-1, keepdim=True)

    rho_expanded = rho.unsqueeze(-1)
    return w * rho_expanded * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2)


def macroscopic(f: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover macroscopic density and velocity from particle populations.

    Args:
        f: Particle populations with shape ``(..., ny, nx, 9)``.
        eps: Small positive value to avoid division-by-zero in velocity recovery.

    Returns:
        Tuple ``(rho, u)`` with shapes ``(..., ny, nx)`` and ``(..., ny, nx, 2)``.
    """
    rho = f.sum(dim=-1)
    c = C.to(device=f.device, dtype=f.dtype)
    momentum = torch.einsum("...q,qd->...d", f, c)
    u = momentum / rho.clamp_min(eps).unsqueeze(-1)
    return rho, u


def collide_and_stream(
    f: torch.Tensor,
    omega: float,
    obstacle_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one BGK collision + periodic streaming step.

    This function is intentionally lightweight and CPU-friendly for examples/tests.

    Args:
        f: Particle populations with shape ``(..., ny, nx, 9)``.
        omega: Relaxation parameter in ``(0, 2]``.
        obstacle_mask: Optional boolean mask with shape ``(..., ny, nx)``. Cells
            set to ``True`` use simple on-site bounce-back.

    Returns:
        Updated populations with the same shape as ``f``.
    """
    if not (0.0 < omega <= 2.0):
        msg = "omega must be in (0, 2]"
        raise ValueError(msg)

    rho, u = macroscopic(f)
    f_eq = equilibrium(rho, u)
    f_post = f + omega * (f_eq - f)

    streamed = torch.empty_like(f_post)
    for i, (cx, cy) in enumerate(C.tolist()):
        streamed[..., i] = torch.roll(f_post[..., i], shifts=(cy, cx), dims=(-2, -1))

    if obstacle_mask is None:
        return streamed

    mask = obstacle_mask.to(device=f.device, dtype=torch.bool)
    if mask.shape != f.shape[:-1]:
        msg = "obstacle_mask must match f.shape[:-1]"
        raise ValueError(msg)

    out = streamed.clone()
    expanded_mask = mask.unsqueeze(-1)
    opposite = OPPOSITE.to(device=f.device)
    bounced = f_post[..., opposite]
    out = torch.where(expanded_mask, bounced, out)
    return out


def initialize_equilibrium(
    ny: int,
    nx: int,
    rho0: float = 1.0,
    u0: tuple[float, float] = (0.0, 0.0),
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create an equilibrium-initialized D2Q9 state for a rectangular grid."""
    rho = torch.full((ny, nx), rho0, device=device, dtype=dtype)
    u = torch.zeros((ny, nx, 2), device=device, dtype=dtype)
    u[..., 0] = u0[0]
    u[..., 1] = u0[1]
    return equilibrium(rho, u)
