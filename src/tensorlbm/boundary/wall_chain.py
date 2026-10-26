"""Reusable per-level BFL wall-chain construction for refined grids.

Campaign convention (owner directive 2026-10-11, common-module build): the
wall-model chain composition must not be copy-pasted into runners.  This
module factors the pieces that every refined backend (uniform grid, static
blocks, octree leaves) needs around :func:`tensorlbm.wall_model.bfl_wall_function_3d`:

* :func:`bfl_halfway_links_3d` — Q19 boundary-link mask and halfway
  ``q = 0.5`` field from a boolean solid (previously duplicated as private
  helpers in the uniform and AMR runners);
* :func:`suboff_surface_mesh` — surface-normal source dispatch
  (``gradient`` / ``suboff`` / ``hybrid`` analytical-blend, previously a
  private helper in two runners);
* :func:`build_level_wall_chain` — one call producing everything a refined
  level's wall model needs (near-wall mask, normals, BFL links, wall
  viscosity) at that level's own resolution and coordinate system;
* :func:`count_boundary_links_outside_box` — the geometric-vacuousness
  counter used to justify leaving the wall model off coarser levels whose
  hull lies strictly inside the child refinement region.

The existing uniform runner keeps its private helpers untouched (A/B
baseline reproduction); new code composes through this module instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tensorlbm.d3q19 import C
from tensorlbm.drag_pressure import SurfaceMesh, get_near_wall_3d

__all__ = [
    "LevelWallChain",
    "bfl_halfway_links_3d",
    "build_level_wall_chain",
    "count_boundary_links_outside_box",
    "suboff_surface_mesh",
]


@dataclass(frozen=True)
class LevelWallChain:
    """Per-level wall-model inputs on one refined grid.

    ``bfl_mask``/``bfl_q`` follow the (Q, nz, ny, nx) convention of
    ``bfl_wall_function_3d``; ``normals`` is the ``(nx, ny, nz)`` surface
    normal triple on near-wall cells.
    """

    near: torch.Tensor
    normals: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    bfl_mask: torch.Tensor
    bfl_q: torch.Tensor
    wall_nu: float


def bfl_halfway_links_3d(solid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BFL boundary-link mask and halfway q-field (q = 0.5) for Q19.

    A link exists where a fluid cell borders a solid cell along a lattice
    direction; the wall crossing is placed halfway along the link, matching
    the uniform formal-tier convention for direct A/B fidelity.
    """
    masks = torch.zeros((19, *solid.shape), dtype=torch.bool, device=solid.device)
    q_field = torch.full(masks.shape, 0.5, dtype=torch.float32, device=solid.device)
    for direction in range(1, 19):
        cx, cy, cz = (int(value) for value in C[direction].tolist())
        neighbour = torch.roll(solid, shifts=(-cz, -cy, -cx), dims=(0, 1, 2))
        masks[direction] = ~solid & neighbour
    return masks, q_field


def suboff_surface_mesh(
    solid: torch.Tensor,
    near: torch.Tensor,
    cx: float,
    cy: float,
    cz: float,
    length: float,
    mesh_source: str = "hybrid",
) -> SurfaceMesh:
    """Surface normals on an arbitrary (sub)grid, in its own coordinates.

    ``gradient`` uses voxel-gradient normals everywhere; ``suboff`` the
    analytical hull normals; ``hybrid`` blends them (analytical on the hull
    band and caps, gradient on appendage near cells).  The geometry is
    expressed in this grid's own coordinate system, so refined levels just
    pass their mapped center and cell-scaled length.
    """
    if mesh_source not in {"suboff", "gradient", "hybrid"}:
        raise ValueError(f"unsupported mesh_source: {mesh_source}")
    from tensorlbm.suboff_cad import SuboffConfig, suboff_radius_profile

    max_radius = SuboffConfig().r_over_l * length
    if mesh_source == "gradient":
        return SurfaceMesh.from_gradient(solid, near)
    mesh = SurfaceMesh.from_suboff(solid, near, cx, cy, cz, length, max_radius)
    if mesh_source != "hybrid":
        return mesh

    nz, ny, nx = solid.shape
    device = solid.device
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    xi = (xx - (cx - length / 2.0)) / length
    r_cell = torch.sqrt((yy - cy) ** 2 + (zz - cz) ** 2)
    r_hull = torch.as_tensor(
        suboff_radius_profile(xi.cpu().numpy()) * max_radius,
        device=device,
        dtype=torch.float32,
    )
    near_f = near.to(device).bool()
    in_axial = (xi >= 0.0) & (xi <= 1.0)
    hull_band = near_f & in_axial & ((r_cell - r_hull).abs() <= 1.5)
    caps = near_f & ~in_axial
    use_analytic = hull_band | caps

    grad = SurfaceMesh.from_gradient(solid, near)
    nx_n = torch.where(use_analytic, mesh.nx_n, grad.nx_n)
    ny_n = torch.where(use_analytic, mesh.ny_n, grad.ny_n)
    nz_n = torch.where(use_analytic, mesh.nz_n, grad.nz_n)
    return SurfaceMesh(near, nx_n, ny_n, nz_n)


def build_level_wall_chain(
    solid: torch.Tensor,
    wall_nu: float,
    cx: float,
    cy: float,
    cz: float,
    length: float,
    mesh_source: str = "hybrid",
) -> LevelWallChain:
    """Build the full wall-model input set for one refined level.

    Args:
        solid: boolean solid mask on this level's grid (ghost padding is
            the caller's choice; the chain follows whatever grid is given).
        wall_nu: wall-law lattice viscosity at the physical Reynolds number
            for this level (``u * L_level / Re``).
        cx, cy, cz: hull center in this grid's coordinate system.
        length: hull length in this grid's cells.
        mesh_source: ``gradient`` / ``suboff`` / ``hybrid``.
    """
    near = get_near_wall_3d(solid).to(solid.device)
    mesh = suboff_surface_mesh(solid, near, cx, cy, cz, length, mesh_source)
    bfl_mask, bfl_q = bfl_halfway_links_3d(solid)
    return LevelWallChain(
        near=near,
        normals=(mesh.nx_n, mesh.ny_n, mesh.nz_n),
        bfl_mask=bfl_mask,
        bfl_q=bfl_q,
        wall_nu=float(wall_nu),
    )


def count_boundary_links_outside_box(
    solid: torch.Tensor,
    owned_interior: torch.Tensor,
) -> int:
    """Fluid-to-solid boundary links whose fluid cell is NOT box-interior.

    A coarser level whose hull lies strictly inside the child refinement
    region has zero such links: every cell its wall model could touch is
    overwritten by the child restriction, so wall treatment there is
    geometrically vacuous.  Counts without allocating a (Q,nz,ny,nx) field.
    """
    count = 0
    for direction in range(1, 19):
        cx, cy, cz = (int(value) for value in C[direction].tolist())
        neighbour = torch.roll(solid, shifts=(-cz, -cy, -cx), dims=(0, 1, 2))
        count += int((~solid & neighbour & ~owned_interior).sum().item())
    return count
