"""Per-leaf Bouzidi BFL q-field for the octree boundary shell.

The q-field is evaluated directly on the *leaf* coordinates (world-space leaf
centres, which are known exactly from the Morton lattice), rather than being
re-sampled from the block's grid — this is what makes the shell truly
body-fitted.  The ray-sphere intersection follows the same maths as
``tensorlbm.bfl_common.compute_q_sphere_common``: for a fluid leaf centre
``x`` and direction ``d`` whose neighbour point ``x + c_d * dx`` is inside
the sphere, ``q`` is the first intersection parameter of the ray
``x + s * c_d * dx`` with the sphere, clamped to ``(0, 1]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tensorlbm.octree_boundary.geometry import OctreeGrid


def _lattice_params(lattice: str) -> tuple[int, torch.Tensor, torch.Tensor]:
    """``(Q, C, OPPOSITE)`` for D3Q19 or D3Q27."""
    if lattice == "D3Q27":
        from tensorlbm.d3q27 import OPPOSITE, C

        return 27, C, OPPOSITE
    if lattice != "D3Q19":
        raise ValueError(f"unsupported lattice {lattice!r} (D3Q19 or D3Q27)")
    from tensorlbm.d3q19 import OPPOSITE, C

    return 19, C, OPPOSITE


def compute_q_sphere_at_points(
    centers: torch.Tensor,
    dx: torch.Tensor | float,
    center: tuple[float, float, float],
    radius: float,
    *,
    device: torch.device = torch.device("cpu"),
    lattice: str = "D3Q19",
) -> tuple[torch.Tensor, torch.Tensor]:
    """BFL q-field of arbitrary points against the analytic sphere.

    Args:
        centers: ``(n, 3)`` float leaf centres in world units.
        dx: per-leaf lattice spacing (broadcastable to ``(n,)``).
        center: sphere centre ``(cx, cy, cz)`` in world units.
        radius: sphere radius in world units.

    Returns:
        ``(fluid_boundary_mask, q_field)`` each ``(Q, n)`` — mask True where
        the neighbour point ``x + c_d * dx`` lies inside the sphere, ``q``
        the first-intersection parameter in ``(0, 1]`` (0.5 / False where
        there is no wall along that direction, matching the flat default of
        ``compute_q_sphere_common``).
    """
    Q, C, _ = _lattice_params(lattice)
    c = C.to(device)
    centers = centers.to(device=device, dtype=torch.float64)
    n = centers.shape[0]
    dxv = torch.as_tensor(dx, dtype=torch.float64, device=device).reshape(-1, 1)
    if dxv.shape[0] == 1 and n > 1:
        dxv = dxv.expand(n, 1)

    cx, cy, cz = (float(v) for v in center)
    cs = torch.tensor([cx, cy, cz], dtype=torch.float64, device=device)  # (x,y,z)
    # NOTE: leaf centres are stored in (x, y, z) order (see geometry
    # ``_level1_leaves``); the sphere centre must be stacked in the same
    # axis order or the distances are silently permuted for non-isotropic
    # centres.

    mask = torch.zeros((Q, n), dtype=torch.bool, device=device)
    q_field = torch.full((Q, n), 0.5, dtype=torch.float32, device=device)

    r2 = radius * radius
    d_self = ((centers - cs) ** 2).sum(dim=1)
    self_fluid = d_self > r2

    for d in range(1, Q):
        c_d = c[d].to(device=device, dtype=torch.float64)
        if bool((c_d == 0).all()):
            continue
        v = c_d * dxv  # (n, 3) neighbour offset
        nb = centers + v
        nb_solid = ((nb - cs) ** 2).sum(dim=1) <= r2
        boundary = self_fluid & nb_solid
        if not bool(boundary.any()):
            continue
        a = (v**2).sum(dim=1)
        b = 2.0 * (v * (centers - cs)).sum(dim=1)
        cst = d_self - r2
        disc = b * b - 4.0 * a * cst
        safe_disc = torch.where(
            boundary & (disc >= 0.0),
            disc,
            torch.zeros_like(disc),
        )
        s = (-b - torch.sqrt(safe_disc)) / (2.0 * a)
        q = torch.where(boundary, s.clamp(1e-6, 1.0), torch.full_like(s, 0.5))
        mask[d] = boundary
        q_field[d] = q.to(torch.float32)

    return mask, q_field


def compute_leaf_q_field(
    grid: OctreeGrid,
    center: tuple[float, float, float],
    radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill ``grid.q_field`` / ``grid.bfl_mask`` on the leaf coordinates.

    The per-leaf spacing is ``2^-level`` (L1 cell = 1).  The rest direction
    keeps the default ``q = 0.5`` with mask False.
    """
    lattice = grid.meta.get("lattice", "D3Q19")
    if lattice == "D3Q27":
        from tensorlbm.d3q27 import OPPOSITE  # noqa: F401  (import contract parity)
    else:
        from tensorlbm.d3q19 import OPPOSITE  # noqa: F401  (import contract parity)

    level = grid.leaf_level
    dx = 2.0 ** (-level.to(torch.float64))  # (n,)
    mask, q = compute_q_sphere_at_points(
        grid.leaf_center,
        dx,
        center,
        radius,
        device=grid.leaf_center.device,
        lattice=lattice,
    )
    grid.bfl_mask = mask.contiguous()
    grid.q_field = q.contiguous()
    return grid.bfl_mask, grid.q_field
