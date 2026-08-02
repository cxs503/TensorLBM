import functools

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
_W_VALUES = (
    4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9,
    1 / 36, 1 / 36, 1 / 36, 1 / 36,
)
W = torch.tensor(_W_VALUES, dtype=torch.float32)
W_EXACT64 = torch.tensor(_W_VALUES, dtype=torch.float64)
WEIGHT_PRECISION_SCHEME = "rational_binary64_cast_to_runtime_dtype_v1"
OPPOSITE = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.int64)


@functools.cache
def _c_on(device: torch.device) -> torch.Tensor:
    return C.to(device)


@functools.cache
def _w_on(
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return W_EXACT64.to(device=device, dtype=dtype)


def equilibrium(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute D2Q9 equilibrium distribution f_eq for rho and velocity fields."""
    if not (rho.shape == ux.shape == uy.shape):
        raise ValueError(
            "rho, ux, and uy shapes must match: "
            f"rho={tuple(rho.shape)}, ux={tuple(ux.shape)}, uy={tuple(uy.shape)}"
        )
    if device is None:
        device = rho.device
    c = _c_on(device)
    w = _w_on(device, rho.dtype).view(9, 1, 1)

    u_sq = ux * ux + uy * uy
    cu = c[:, 0].view(9, 1, 1).to(rho.dtype) * ux + c[:, 1].view(9, 1, 1).to(rho.dtype) * uy
    return w * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(0))


def macroscopic(
    f: torch.Tensor,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover rho, ux, uy from particle distributions."""
    if device is None:
        device = f.device
    c = _c_on(device)

    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    ux = (f * c[:, 0].view(9, 1, 1)).sum(dim=0) / rho_safe
    uy = (f * c[:, 1].view(9, 1, 1)).sum(dim=0) / rho_safe
    return rho, ux, uy
