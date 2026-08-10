"""Exact D2Q9 collision embedded in D3Q19 storage for planar benchmarks."""

from __future__ import annotations

import math

import torch

from .cumulant import collide_cumulant_d2q9
# D3Q19 directions grouped by their (cx,cy) D2Q9 marginal.  The group order
# is the public D2Q9 direction order.
PLANAR_D3Q19_GROUPS: tuple[tuple[int, ...], ...] = (
    (0, 5, 6),
    (1, 11, 13),
    (3, 15, 17),
    (2, 12, 14),
    (4, 16, 18),
    (7,),
    (10,),
    (8,),
    (9,),
)

_D3_TO_D2 = (
    0, 1, 3, 2, 4, 0, 0, 5, 7, 8, 6, 1, 3, 1, 3, 2, 4, 2, 4,
)

# Conditional D3Q19 weight inside each D2Q9 marginal, written as exact Python
# fractions rather than ratios of the public float32 weight tensors.
_D3_LIFT_RATIOS = (
    3 / 4,
    1 / 2,
    1 / 2,
    1 / 2,
    1 / 2,
    1 / 8,
    1 / 8,
    1.0,
    1.0,
    1.0,
    1.0,
    1 / 4,
    1 / 4,
    1 / 4,
    1 / 4,
    1 / 4,
    1 / 4,
    1 / 4,
    1 / 4,
)


def marginalize_d3q19_to_d2q9(populations: torch.Tensor) -> torch.Tensor:
    """Sum D3Q19 populations with identical in-plane lattice velocities."""
    if (
        not isinstance(populations, torch.Tensor)
        or populations.ndim != 4
        or populations.shape[0] != 19
    ):
        raise ValueError("populations must have shape (19,nz,ny,nx)")
    return torch.stack([
        populations[list(group)].sum(dim=0) for group in PLANAR_D3Q19_GROUPS
    ])


def lift_d2q9_to_d3q19(populations: torch.Tensor) -> torch.Tensor:
    """Lift D2Q9 marginals using conditional D3Q19 equilibrium weights.

    The lift is conservative by construction: every D2Q9 population is split
    only among D3Q19 directions with the same ``(cx,cy)``.  Consequently
    density and in-plane momentum are unchanged and the added z momentum is
    exactly zero.
    """
    if (
        not isinstance(populations, torch.Tensor)
        or populations.ndim != 4
        or populations.shape[0] != 9
    ):
        raise ValueError("populations must have shape (9,nz,ny,nx)")
    direction_map = torch.tensor(
        _D3_TO_D2, dtype=torch.long, device=populations.device,
    )
    ratios = torch.tensor(
        _D3_LIFT_RATIOS,
        device=populations.device,
        dtype=populations.dtype,
    )
    return populations[direction_map] * ratios[:, None, None, None]


def collide_planar_cumulant_d3q19(
    populations: torch.Tensor,
    tau: float,
    *,
    omega_b: float = 1.0,
    omega_3: float = 1.0,
    omega_4: float = 1.0,
) -> torch.Tensor:
    """Apply exact D2Q9 cumulant collision and lift back to D3Q19.

    The z planes are treated as a batch of cell-local D2Q9 collisions.  This
    mode is intended for extruded, z-periodic validation cases; it is not a
    three-dimensional collision model and must never be used for 3-D flow.
    """
    if not math.isfinite(tau) or tau <= 0.5:
        raise ValueError("tau must be finite and greater than 0.5")
    marginal = marginalize_d3q19_to_d2q9(populations)
    _, nz, ny, nx = marginal.shape
    batched = marginal.reshape(9, nz * ny, nx)
    collided = collide_cumulant_d2q9(
        batched,
        tau,
        omega_b=omega_b,
        omega_3=omega_3,
        omega_4=omega_4,
    ).reshape(9, nz, ny, nx)
    return lift_d2q9_to_d3q19(collided)


def maximum_planar_plane_spread(populations: torch.Tensor) -> float:
    """Return maximum D2Q9-marginal difference from the mean z plane."""
    marginal = marginalize_d3q19_to_d2q9(populations)
    spread = (marginal - marginal.mean(dim=1, keepdim=True)).abs().max()
    return float(spread.item())


__all__ = [
    "PLANAR_D3Q19_GROUPS",
    "collide_planar_cumulant_d3q19",
    "lift_d2q9_to_d3q19",
    "marginalize_d3q19_to_d2q9",
    "maximum_planar_plane_spread",
]
