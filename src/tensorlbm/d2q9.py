"""D2Q9 lattice constants and distribution helper functions."""

import torch

# D2Q9 lattice velocities (cx, cy), weights, and opposite-direction mapping
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
W = torch.tensor([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=torch.float32)
OPPOSITE = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.int64)


def equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    """Compute D2Q9 equilibrium distribution with shape ``(9, ny, nx)``."""
    if device is None:
        device = rho.device
    c = C.to(device)
    w = W.to(device).view(9, 1, 1)

    u_sq = ux * ux + uy * uy
    cu = c[:, 0].view(9, 1, 1) * ux + c[:, 1].view(9, 1, 1) * uy
    return w * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(0))


def macroscopic(f: torch.Tensor, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover ``rho``, ``ux``, and ``uy`` from distribution ``f``."""
    if device is None:
        device = f.device
    c = C.to(device)

    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    ux = (f * c[:, 0].view(9, 1, 1)).sum(dim=0) / rho_safe
    uy = (f * c[:, 1].view(9, 1, 1)).sum(dim=0) / rho_safe
    return rho, ux, uy
