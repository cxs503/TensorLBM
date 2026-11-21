"""Sparse List AOS (Array of Structures) LBM for D3Q19.

This module implements a lattice Boltzmann solver using a sparse list of
fluid nodes with an Array of Structures (AOS) layout for populations.

Layout
------
* ``nodes``: ``(K,)`` int64, dense indices of fluid nodes.
* ``neighbour``: ``(K, 19)`` int64, compressed indices of neighbours.
* ``f``: ``(K, 19)`` float64, populations per node.

This contrasts with the SOA (Structure of Arrays) layout in
:mod:`tensorlbm.sparse_sublist_neighbor_list` where ``f`` is ``(19, K)``.
AOS is more cache-friendly for collision (all populations of a node are
contiguous) but requires careful handling for streaming.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .d3q19 import C, OPPOSITE, W, CS2

SOLID = -1


@dataclass(frozen=True)
class SparseListAOS:
    """Compacted fluid-node map + neighbour list + sublist (AOS layout).

    Attributes
    ----------
    nodes : torch.Tensor, shape ``(K,)``, int64
        ``nodes[k] = z*ny*nx + y*nx + x`` -- the dense (row-major) index of
        fluid node ``k``.
    neighbour : torch.Tensor, shape ``(K, 19)``, int64
        ``neighbour[k, s]`` = compressed index of the neighbour of fluid node
        ``k`` in direction ``s``, or :data:`SOLID` (``-1``).
    index : torch.Tensor, shape ``(nz, ny, nx)``, int64
        dense -> compressed map, ``-1`` on solid sites.
    sublist : tuple[int, int]
        ``(ist, ind)`` -- the contiguous sub-range of the main list holding
        the fluid nodes.
    nz, ny, nx : int
        dense lattice shape the table was built for.
    """

    nodes: torch.Tensor
    neighbour: torch.Tensor
    index: torch.Tensor
    sublist: tuple[int, int]
    nz: int
    ny: int
    nx: int

    # ------------------------------------------------------------------ basic
    @property
    def n_fluid(self) -> int:
        return int(self.nodes.numel())

    @property
    def n_total(self) -> int:
        return int(self.nz) * int(self.ny) * int(self.nx)

    @property
    def porosity(self) -> float:
        return self.n_fluid / self.n_total

    def to(self, device: torch.device | str) -> "SparseListAOS":
        return SparseListAOS(
            nodes=self.nodes.to(device),
            neighbour=self.neighbour.to(device),
            index=self.index.to(device),
            sublist=self.sublist,
            nz=self.nz,
            ny=self.ny,
            nx=self.nx,
        )

    # ------------------------------------------------------------- footprints
    def state_bytes(self, dtype: torch.dtype = torch.float64, table_int: int = 8) -> dict:
        """Storage footprint of the compressed state vs the dense state."""
        fbytes = torch.tensor([], dtype=dtype).element_size()
        k = self.n_fluid
        n = self.n_total
        compressed_pdf = 19 * k * fbytes
        compressed_table = 19 * k * table_int + k * table_int
        dense_pdf = 19 * n * fbytes
        dense_mask = n  # bool
        return {
            "compressed_pdf_bytes": compressed_pdf,
            "compressed_table_bytes": compressed_table,
            "compressed_total_bytes": compressed_pdf + compressed_table,
            "dense_pdf_bytes": dense_pdf,
            "dense_mask_bytes": dense_mask,
            "dense_total_bytes": dense_pdf + dense_mask,
            "pdf_only_ratio": dense_pdf / compressed_pdf,
            "total_ratio": (dense_pdf + dense_mask) / (compressed_pdf + compressed_table),
        }


def build_sparse_list_aos(
    solid: torch.Tensor,
    *,
    wrap: bool = True,
) -> SparseListAOS:
    """Compact the fluid sites of ``solid`` and build the neighbour list.

    Parameters
    ----------
    solid : torch.Tensor, shape ``(nz, ny, nx)``, bool
        ``True`` on solid (matrix) sites.
    wrap : bool
        Periodic wrap of the neighbour lookup in all lattice directions.

    Returns
    -------
    SparseListAOS
    """
    if solid.dtype != torch.bool:
        raise TypeError("solid must be a bool mask (True = solid)")
    if solid.dim() != 3:
        raise ValueError("solid must be 3-D (nz, ny, nx)")
    nz, ny, nx = int(solid.shape[0]), int(solid.shape[1]), int(solid.shape[2])
    fluid = ~solid
    n_fluid = int(fluid.sum())

    index = torch.full((nz, ny, nx), SOLID, dtype=torch.int64, device=solid.device)
    index[fluid] = torch.arange(n_fluid, dtype=torch.int64, device=solid.device)

    zs, ys, xs = fluid.nonzero(as_tuple=True)  # row-major
    nodes = zs * ny * nx + ys * nx + xs

    c = C.to(solid.device)
    neighbour = torch.full((n_fluid, 19), SOLID, dtype=torch.int64, device=solid.device)
    for slot, d in enumerate(range(19)):
        cz, cy, cx = int(c[d, 0]), int(c[d, 1]), int(c[d, 2])
        zz = zs + cz
        yy = ys + cy
        xx = xs + cx
        if wrap:
            zz = zz % nz
            yy = yy % ny
            xx = xx % nx
            neighbour[:, slot] = index[zz, yy, xx]
        else:
            inb = (zz >= 0) & (zz < nz) & (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
            neighbour[inb, slot] = index[zz[inb], yy[inb], xx[inb]]

    sublist = (0, n_fluid - 1)
    return SparseListAOS(
        nodes=nodes,
        neighbour=neighbour,
        index=index,
        sublist=sublist,
        nz=nz,
        ny=ny,
        nx=nx,
    )


def aos_collide(
    f: torch.Tensor,
    tau: float,
    force_x: float = 0.0,
    force_y: float = 0.0,
    force_z: float = 0.0,
) -> torch.Tensor:
    """BGK collision for AOS layout (K, 19).

    Parameters
    ----------
    f : torch.Tensor, shape ``(K, 19)``
        Populations of the ``K`` fluid nodes.
    tau : float
        Relaxation time.
    force_x, force_y, force_z : float
        Body force components.

    Returns
    -------
    f_eq : torch.Tensor, shape ``(K, 19)``
        Post-collision populations.
    """
    if f.dim() != 2 or int(f.shape[1]) != 19:
        raise ValueError("f must be (K, 19) with the D3Q19 AOS layout")

    w = W.to(f.device)
    c = C.to(f.device)
    opp = OPPOSITE.to(f.device)

    # Compute density
    rho = f.sum(dim=1, keepdim=True)

    # Compute velocity
    u = torch.zeros((f.shape[0], 3), dtype=f.dtype, device=f.device)
    for d in range(19):
        u[:, 0] += f[:, d] * c[d, 0]
        u[:, 1] += f[:, d] * c[d, 1]
        u[:, 2] += f[:, d] * c[d, 2]
    u[:, 0] /= rho.squeeze()
    u[:, 1] /= rho.squeeze()
    u[:, 2] /= rho.squeeze()

    # Guo forcing (simplified, same as porous module)
    # F = (Fx, Fy, Fz)
    F = torch.tensor([force_x, force_y, force_z], dtype=f.dtype, device=f.device).unsqueeze(0)
    u_star = u + 0.5 * F / rho.squeeze().unsqueeze(1)

    # Collision
    f_eq = torch.zeros_like(f)
    for d in range(19):
        cu = (c[d, 0] * u[:, 0] + c[d, 1] * u[:, 1] + c[d, 2] * u[:, 2])
        u2 = u[:, 0]**2 + u[:, 1]**2 + u[:, 2]**2
        feq = w[d] * rho.squeeze() * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2)
        f_eq[:, d] = f[:, d] - (1.0 - 1.0 / tau) * (f[:, d] - feq) + (1.0 - 1.0 / (2.0 * tau)) * w[d] * (3.0 * cu + 9.0 * (u[:, 0] * c[d, 0] + u[:, 1] * c[d, 1] + u[:, 2] * c[d, 2]))

    return f_eq


def aos_stream(
    f: torch.Tensor,
    table: SparseListAOS,
) -> torch.Tensor:
    """Pull streaming for AOS layout.

    f_new[k, dir] = f_old[table.neighbour[k, opposite[dir]], dir]

    Parameters
    ----------
    f : torch.Tensor, shape ``(K, 19)``
        Populations of the ``K`` fluid nodes.
    table : SparseListAOS
        The sparse list table.

    Returns
    -------
    f_new : torch.Tensor, shape ``(K, 19)``
        Post-streaming populations.
    """
    if f.dim() != 2 or int(f.shape[1]) != 19:
        raise ValueError("f must be (K, 19) with the D3Q19 AOS layout")

    f_new = torch.zeros_like(f)
    ist, ind = table.sublist
    neighbour = table.neighbour

    # Pull streaming: for each node k and direction dir,
    # f_new[k, dir] = f[neighbour[k, opposite[dir]], dir]
    # We need to handle solid neighbours (neighbour == -1)
    # For solid neighbours, we use bounce-back: f_new[k, dir] = f[k, opposite[dir]]
    
    # Vectorized implementation
    opp_indices = OPPOSITE.to(f.device)
    
    # Get neighbour indices for opposite directions
    # neighbour[k, opp[dir]] gives the source node for direction dir
    src_nodes = neighbour[:, opp_indices]
    
    # Mask for valid neighbours (not solid)
    valid_mask = src_nodes >= 0
    
    # Initialize with bounce-back for solid neighbours
    # f_new[k, dir] = f[k, opposite[dir]]
    f_new[:, :] = f[:, opp_indices]
    
    # Update for valid fluid neighbours
    # f_new[k, dir] = f[src_nodes[k], dir]
    # We need to gather from f using src_nodes
    # f[src_nodes, :] gives (K, 19, 19) because src_nodes is (K, 19)
    # We want f[src_nodes[k, d], d] for each k, d
    # This is equivalent to torch.gather(f, 0, src_nodes.unsqueeze(0).expand(19, -1, -1).permute(1, 2, 0))
    # But a simpler way is to loop over directions
    for d in range(19):
        src = src_nodes[:, d]
        f_new[valid_mask, d] = f[src[valid_mask], d]
    
    return f_new


def aos_step(
    f: torch.Tensor,
    table: SparseListAOS,
    tau: float,
    force_x: float = 0.0,
    force_y: float = 0.0,
    force_z: float = 0.0,
) -> torch.Tensor:
    """One AOS step: collide, then stream.

    Parameters
    ----------
    f : torch.Tensor, shape ``(K, 19)``
        Populations of the ``K`` fluid nodes.
    table : SparseListAOS
        The sparse list table.
    tau : float
        Relaxation time.
    force_x, force_y, force_z : float
        Body force components.

    Returns
    -------
    f_new : torch.Tensor, shape ``(K, 19)``
        Post-step populations.
    """
    f_collided = aos_collide(f, tau, force_x, force_y, force_z)
    f_new = aos_stream(f_collided, table)
    return f_new
