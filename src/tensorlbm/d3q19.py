import functools

import torch

# D3Q19 lattice velocities (cx, cy, cz), weights, and opposite-direction mapping
C = torch.tensor(
    [
        # rest
        [0, 0, 0],
        # face-centre (±x, ±y, ±z)
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        # edge-centre (xy plane)
        [1, 1, 0],
        [-1, -1, 0],
        [1, -1, 0],
        [-1, 1, 0],
        # edge-centre (xz plane)
        [1, 0, 1],
        [-1, 0, -1],
        [1, 0, -1],
        [-1, 0, 1],
        # edge-centre (yz plane)
        [0, 1, 1],
        [0, -1, -1],
        [0, 1, -1],
        [0, -1, 1],
    ],
    dtype=torch.int64,
)

W = torch.tensor(
    [
        1 / 3,
        1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18,
        1 / 36, 1 / 36, 1 / 36, 1 / 36,
        1 / 36, 1 / 36, 1 / 36, 1 / 36,
        1 / 36, 1 / 36, 1 / 36, 1 / 36,
    ],
    dtype=torch.float32,
)

OPPOSITE = torch.tensor(
    [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17],
    dtype=torch.int64,
)


@functools.cache
def _c_on(device: torch.device) -> torch.Tensor:
    return C.to(device)


@functools.cache
def _w_on(device: torch.device) -> torch.Tensor:
    return W.to(device)


def equilibrium3d(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    device: torch.device | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute D3Q19 equilibrium distribution f_eq for rho and velocity fields.

    Args:
        rho: density field of shape (nz, ny, nx).
        ux, uy, uz: velocity components of shape (nz, ny, nx).
        out: optional pre-allocated output tensor of shape (19, nz, ny, nx).
            If provided, the result is written into this tensor in-place,
            avoiding a new allocation.

    Returns:
        Tensor of shape (19, nz, ny, nx).
    """
    if not (rho.shape == ux.shape == uy.shape == uz.shape):
        raise ValueError(
            "rho, ux, uy, and uz shapes must match: "
            f"rho={tuple(rho.shape)}, ux={tuple(ux.shape)}, "
            f"uy={tuple(uy.shape)}, uz={tuple(uz.shape)}"
        )
    if device is None:
        device = rho.device
    c = _c_on(device)
    w = _w_on(device).view(19, 1, 1, 1)

    u_sq = ux * ux + uy * uy + uz * uz
    cu = (
        c[:, 0].view(19, 1, 1, 1) * ux
        + c[:, 1].view(19, 1, 1, 1) * uy
        + c[:, 2].view(19, 1, 1, 1) * uz
    )
    result = w * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(0))
    if out is not None:
        out.copy_(result)
        return out
    return result


def macroscopic3d(
    f: torch.Tensor,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover rho, ux, uy, uz from particle distributions.

    Args:
        f: distribution tensor of shape (19, nz, ny, nx).

    Returns:
        Tuple (rho, ux, uy, uz) each of shape (nz, ny, nx).
    """
    if device is None:
        device = f.device
    c = _c_on(device)

    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    ux = (f * c[:, 0].view(19, 1, 1, 1)).sum(dim=0) / rho_safe
    uy = (f * c[:, 1].view(19, 1, 1, 1)).sum(dim=0) / rho_safe
    uz = (f * c[:, 2].view(19, 1, 1, 1)).sum(dim=0) / rho_safe
    return rho, ux, uy, uz
