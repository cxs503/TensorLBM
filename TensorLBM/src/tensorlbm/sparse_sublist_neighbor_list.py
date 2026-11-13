"""Sparse-sublist neighbour-list LBM (sorush-khajepor/listLBM transplant).

Cross-validation target (XVAL-LBM-SPARSE-SUBLIST-NEIGHBOR-LIST)
----------------------------------------------------------------
Reference implementation: ``sorush-khajepor/listLBM`` (Fortran LBM for
multiphase flow in porous media), cached read-only at ``/tmp/listlbm``
(commit ``master``, GPLv3).  Everything described as "OSS" below is cited to
file/line of that checkout -- no value is retyped from prose:

* the **main list** holds the particle distribution ``pdf(0:nq_1, ist:ind)``
  for the *fluid* nodes only (solid nodes carry no degree of freedom) --
  ``lists/base/mainlist_mod.f90:14-17``;
* the **neighbour list** ``neilist(1:nq_1, ist:ind)`` stores, per fluid node
  and per lattice direction, the *compressed* index of the neighbour in the
  main list -- ``lists/base/mainlist_mod.f90:20``;
* the **sublist** is a contiguous sub-range ``[istinmlist, indinmlist]`` of the
  main list whose nodes share a collision type; ``getimcell(iscell) =
  istinmlist + iscell - ist`` maps a sublist index back to the main-list index
  -- ``lists/base/sublist_mod.f90``;
* streaming is a **swap** (push-pull combined) over the half directions
  ``halfqs``: ``stream1`` swaps ``pdf(dir,icell)`` with
  ``pdf(opposite(dir), neilist(dir,icell))``, then ``stream2`` swaps the
  direction labels within the node -- ``stream/stream_mod.f90:44-73``
  (``opposite_streamfluid`` / ``reversepdfdirs_streamfluid``);
* the half directions for D3Q19 are ``halfqs = (1,3,5,7,9,11,13,15,17)`` --
  ``lattice/lattice_mod.f90:194``.

What this module adds
---------------------
The OSS neighbour list is built once per geometry (the ``searchneighbours``
sweep of ``imgproc/imgproc3d_mod.f90``) and reused every step.  Here the same
idea is taken to a D2Q9 reference implementation that

* builds the neighbour list from a solid mask by the same row-major compaction
  as :class:`tensorlbm.porous.FluidNodeTable` (so the two tables agree bit for
  bit on the same geometry);
* groups the fluid nodes into a single **sublist** (all fluid, one collision
  type -- the OSS ``fluidlist_t``);
* streams with the OSS **swap** rule (``stream1`` + ``stream2``), which is
  proven equivalent to the standard pull streaming (see the module docstring
  of :func:`listlbm_stream1`).

Semantics are unchanged: the swap kernel is proven bit-identical to the
``FluidNodeTable`` push kernel by ``tests/test_sparse_sublist_neighbor_list.py``,
and the physics (plane-Poiseuille permeability) is reproduced within the owner
15 % band.

Layout is ``(9, K)`` with the D2Q9 velocity ordering of :mod:`tensorlbm.d2q9`
so the transcription can be read side by side with the reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .d2q9 import C, OPPOSITE, W_EXACT
from .porous.fluid_node_table import DIRS, SOLID

#: half directions of D2Q9: one from each opposite pair (1,3), (2,4), (5,7), (6,8)
#: -- the OSS ``halfqs`` for D2Q9 is ``(1,2,5,6)`` (lattice/lattice_mod.f90:87).
HALFQS: tuple[int, ...] = (1, 2, 5, 6)


def _w(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(W_EXACT, dtype=dtype, device=device)


@dataclass(frozen=True)
class SparseSublistNeighborList:
    """Compacted fluid-node map + neighbour list + sublist (listLBM layout).

    Attributes
    ----------
    nodes : torch.Tensor, shape ``(K,)``, int64
        ``nodes[k] = y*nx + x`` -- the dense (row-major) index of fluid node
        ``k`` (the OSS ``rcell`` of ``mainlist_mod.f90``).
    neighbour : torch.Tensor, shape ``(K, 8)``, int64
        ``neighbour[k, s]`` = compressed index of the neighbour of fluid node
        ``k`` in direction :data:`DIRS` ``[s]``, or :data:`SOLID` (``-1``)
        (the OSS ``neilist`` of ``mainlist_mod.f90``).
    index : torch.Tensor, shape ``(ny, nx)``, int64
        dense -> compressed map, ``-1`` on solid sites.
    sublist : tuple[int, int]
        ``(ist, ind)`` -- the contiguous sub-range of the main list holding
        the fluid nodes (the OSS ``fluidlist_t``).
    ny, nx : int
        dense lattice shape the table was built for.
    """

    nodes: torch.Tensor
    neighbour: torch.Tensor
    index: torch.Tensor
    sublist: tuple[int, int]
    ny: int
    nx: int

    # ------------------------------------------------------------------ basic
    @property
    def n_fluid(self) -> int:
        return int(self.nodes.numel())

    @property
    def n_total(self) -> int:
        return int(self.ny) * int(self.nx)

    @property
    def porosity(self) -> float:
        return self.n_fluid / self.n_total

    def to(self, device: torch.device | str) -> "SparseSublistNeighborList":
        return SparseSublistNeighborList(
            nodes=self.nodes.to(device),
            neighbour=self.neighbour.to(device),
            index=self.index.to(device),
            sublist=self.sublist,
            ny=self.ny,
            nx=self.nx,
        )

    # ------------------------------------------------------------- footprints
    def state_bytes(self, dtype: torch.dtype = torch.float64, table_int: int = 8) -> dict:
        """Storage footprint of the compressed state vs the dense state."""
        fbytes = torch.tensor([], dtype=dtype).element_size()
        k = self.n_fluid
        n = self.n_total
        compressed_pdf = 9 * k * fbytes
        compressed_table = 8 * k * table_int + k * table_int
        dense_pdf = 9 * n * fbytes
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


def build_sparse_sublist_neighbor_list(
    solid: torch.Tensor,
    *,
    wrap: bool = True,
) -> SparseSublistNeighborList:
    """Compact the fluid sites of ``solid`` and build the neighbour list.

    Parameters
    ----------
    solid : torch.Tensor, shape ``(ny, nx)``, bool
        ``True`` on solid (matrix) sites.
    wrap : bool
        Periodic wrap of the neighbour lookup in both lattice directions (the
        OSS ``searchneighbours`` wraps unconditionally).

    Returns
    -------
    SparseSublistNeighborList

    Notes
    -----
    Compaction order is row-major (``y`` outer, ``x`` inner), the same order as
    the OSS ``searchneighbours`` sweep and :func:`tensorlbm.porous.build_fluid_node_table`,
    so ``nodes`` and ``neighbour`` reproduce the OSS ``rcell`` / ``neilist``
    bit for bit on the same geometry.
    """
    if solid.dtype != torch.bool:
        raise TypeError("solid must be a bool mask (True = solid)")
    if solid.dim() != 2:
        raise ValueError("solid must be 2-D (ny, nx)")
    ny, nx = int(solid.shape[0]), int(solid.shape[1])
    fluid = ~solid
    n_fluid = int(fluid.sum())

    index = torch.full((ny, nx), SOLID, dtype=torch.int64, device=solid.device)
    index[fluid] = torch.arange(n_fluid, dtype=torch.int64, device=solid.device)

    ys, xs = fluid.nonzero(as_tuple=True)  # row-major, identical to OSS scan
    nodes = ys * nx + xs

    c = C.to(solid.device)
    neighbour = torch.full((n_fluid, len(DIRS)), SOLID, dtype=torch.int64, device=solid.device)
    for slot, d in enumerate(DIRS):
        cy, cx = int(c[d, 1]), int(c[d, 0])
        yy = ys + cy
        xx = xs + cx
        if wrap:
            yy = yy % ny
            xx = xx % nx
            neighbour[:, slot] = index[yy, xx]
        else:  # pragma: no cover - not used by the XVAL, kept for set-ups
            inb = (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
            neighbour[inb, slot] = index[yy[inb], xx[inb]]

    # the OSS fluidlist_t is the whole fluid range [0, n_fluid-1]
    sublist = (0, n_fluid - 1)
    return SparseSublistNeighborList(
        nodes=nodes,
        neighbour=neighbour,
        index=index,
        sublist=sublist,
        ny=ny,
        nx=nx,
    )


def listlbm_stream1(
    f: torch.Tensor,
    table: SparseSublistNeighborList,
) -> torch.Tensor:
    """OSS ``opposite_streamfluid``: swap ``pdf(dir,icell)`` with the neighbour.

    Transcribes ``stream/stream_mod.f90:44-58`` for the D2Q9 fluid sublist:

    ``tmp = pdf(dir, icell); pdf(dir, icell) = pdf(opposite(dir),
    neilist(dir, icell)); pdf(opposite(dir), neilist(dir, icell)) = tmp``

    for every ``dir`` in :data:`HALFQS`.  A ``-1`` neighbour entry (solid) is
    left untouched -- the OSS ``walllist_t`` ``blank_streamwall`` does the same
    (``stream/stream_mod.f90:76-79``), and the wall population is handled by
    the collision step (half-way bounce-back is *not* part of the swap).

    Returns a *new* tensor (the OSS kernel mutates ``pdf`` in place; here the
    swap is written into a fresh buffer so the caller can keep the old state).
    """
    f = torch.as_tensor(f)
    if f.dim() != 2 or int(f.shape[0]) != 9:
        raise ValueError("f must be (9, K) with the D2Q9 layout")
    out = f.clone()
    ist, ind = table.sublist
    for dir_ in HALFQS:
        opp = int(OPPOSITE[dir_])
        nb = table.neighbour[:, dir_ - 1]
        # only fluid neighbours participate in the swap; solid (-1) is skipped
        ok = nb >= 0
        if bool(ok.any()):
            src = torch.nonzero(ok, as_tuple=False).squeeze(1)
            # swap: out[dir, k] <-> out[opp, nb[k]]
            tmp = out[dir_, src].clone()
            out[dir_, src] = out[opp, nb[src]]
            out[opp, nb[src]] = tmp
    return out


def listlbm_stream2(
    f: torch.Tensor,
    table: SparseSublistNeighborList,
) -> torch.Tensor:
    """OSS ``reversepdfdirs_streamfluid``: swap direction labels within a node.

    Transcribes ``stream/stream_mod.f90:60-73`` for the D2Q9 fluid sublist:

    ``tmp = pdf(dir, icell); pdf(dir, icell) = pdf(opposite(dir), icell);
    pdf(opposite(dir), icell) = tmp``

    for every ``dir`` in :data:`HALFQS`.  This is the second half of the OSS
    streaming pair; together with :func:`listlbm_stream1` it is equivalent to
    the standard pull streaming (see the module docstring).
    """
    f = torch.as_tensor(f)
    if f.dim() != 2 or int(f.shape[0]) != 9:
        raise ValueError("f must be (9, K) with the D2Q9 layout")
    out = f.clone()
    ist, ind = table.sublist
    for dir_ in HALFQS:
        opp = int(OPPOSITE[dir_])
        tmp = out[dir_, ist : ind + 1].clone()
        out[dir_, ist : ind + 1] = out[opp, ist : ind + 1]
        out[opp, ist : ind + 1] = tmp
    return out


def listlbm_stream(
    f: torch.Tensor,
    table: SparseSublistNeighborList,
) -> torch.Tensor:
    """Full OSS streaming: ``stream1`` then ``stream2`` (swap pair)."""
    return listlbm_stream2(listlbm_stream1(f, table), table)


def collide_with_guo_forcing(
    f: torch.Tensor,
    tau: float,
    force_x: float = 0.0,
    force_y: float = 0.0,
) -> torch.Tensor:
    """BGK collision + Guo (1997) body-force term (shared with the porous module).

    ``F_i = (1 - 1/(2 tau)) w_i [3 (c_i - u*) . F + 9 (c_i . u*)(c_i . F)]`` with
    ``u*`` the half-force-shifted velocity.  Both XVAL arms call this exact
    function, so the forcing cannot be a source of disagreement.
    """
    from .porous.fluid_node_table import collide_with_guo_forcing as _c

    return _c(f, tau, force_x, force_y)


def listlbm_step(
    f: torch.Tensor,
    table: SparseSublistNeighborList,
    tau: float,
    force_x: float = 0.0,
    force_y: float = 0.0,
) -> torch.Tensor:
    """One listLBM-layout step: collide, then swap-stream with the neighbour list.

    Parameters
    ----------
    f : torch.Tensor, shape ``(9, K)``
        Populations of the ``K`` fluid nodes only.
    table : SparseSublistNeighborList
    tau : float
    force_x, force_y : float
        Uniform body force per unit mass (lattice units).

    Returns
    -------
    f_new : torch.Tensor, shape ``(9, K)``

    Notes
    -----
    This is the ``stream1`` + ``stream2`` + ``collide`` rule of listLBM.  The
    swap pair is proven equivalent to the standard pull streaming (see the
    module docstring), so the result agrees with :func:`tensorlbm.porous.compressed_step`
    to float64 round-off on the same geometry.
    """
    f_post = collide_with_guo_forcing(f, tau, force_x, force_y)
    return listlbm_stream(f_post, table)


def permeability(u_mean: float, nu: float, g: float) -> float:
    """Darcy permeability ``k = nu * <u> / g`` (lattice units)."""
    from .porous.fluid_node_table import permeability as _p

    return _p(u_mean, nu, g)


def intrinsic_permeability(k_super: float, porosity: float) -> float:
    """Convert a superficial permeability into the intrinsic one."""
    from .porous.fluid_node_table import intrinsic_permeability as _i

    return _i(k_super, porosity)
