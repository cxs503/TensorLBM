"""List-based sparse LBM with a static gather index array (waLBerla / lbmpy style).

Why this module exists
----------------------
:mod:`tensorlbm.porous.fluid_node_table` implements the *neighbour-table* flavour
of porous-media LBM compaction (openLBMPM): a ``(K, 8)`` table with a ``-1``
solid sentinel, branchy push streaming, bounce-back handled by an explicit
second write pass.

Mature list-based implementations instead build **one static index array** over
the fluid-cell list and fold *everything* -- fluid neighbours, half-way
bounce-back and periodic wrap -- into a single integer per (cell, direction).
The per-step kernel then becomes a branch-free gather.  That is the mechanism
transplanted here; the provenance of the *mechanism* is quoted below (cited, not
copied -- the reference implementation is the open-source lbmpy code generator
that waLBerla's ``python/lbmpy_walberla/sparse.py`` wraps).

Mechanism provenance (OSS pointers; see
``TensorLBM/xval/XVAL-LBM-LIST-BASED-SPARSE-LBM-PLUS-PERMEABILITY/oss_ref.py``)
------------------------------------------------------------------------------
``lbmpy/sparse/mapping.py`` ``SparseLbMapper.create_index_array``
    for every fluid cell and every non-rest direction stores
    ``pdf_index(cell_index, direction_index) = cell_index + direction_index * n_cells``
    of the *upstream* cell (gather form, "pull").  Two branches are folded into
    that single integer::

        if flag[upstream] & (fluid | boundary):  -> pdf_index(upstream_cell, direction)
        elif flag[upstream] & no_slip:           -> pdf_index(own_cell, inverse(direction))
        elif upstream is outside the domain:     -> periodic wrap of the coordinate

    i.e. half-way bounce-back is expressed as "read the opposite direction slot of
    my own cell", so the streaming kernel needs no solid mask at all.
``lbmpy/sparse/update_rule_sparse.py`` ``create_lb_update_rule_sparse``
    substitutes the pdf symbols of a normal collision rule by
    ``src.absolute_access((idx(i - 1),), ())`` (``kernel_type='stream_pull_collide'``
    -> stream and collide in one kernel, two alternating pdf buffers).
``lbmpy/sparse/update_rule_sparse.py`` ``create_symbolic_list(layout='SoA')``
    the pdfs live in a structure-of-arrays list ``(n_cells, Q)`` with stride
    ``(1, n_cells)``, i.e. a flat ``Q*N`` buffer addressed as ``v*N + cell``.

Layout used here
----------------
* pdf buffer ``f`` has shape ``(Q, N)`` (SoA, ``Q`` = 9 for D2Q9), the flat
  address of (direction ``d``, cell ``k``) is ``d * N + k`` -- identical to the
  OSS ``pdf_index``,
* ``SparseIndexTable.index`` has shape ``(N, Q-1)`` (rest direction omitted, as
  in the OSS index array) and holds those flat addresses,
* one step = gather (stream, with bounce-back already in the table) + BGK
  collision with Guo forcing (:func:`tensorlbm.porous.fluid_node_table.collide_with_guo_forcing`),
  which is the OSS ``stream_pull_collide`` kernel structure,
* the ``(Q, N)`` post-collision state is what is stored between steps, so the
  sparse state carries no solid sites and no ghost layers.

Extension w.r.t. the OSS code generator: periodic wrap is supported for both
axes.  ``SparseLbMapper.create_index_array`` in lbmpy 2.0 raises ``IndexError``
on a periodic border (verified: it forms ``uint32(cell - direction)`` before the
border test, see ``oss_ref.py:probe_periodic_support``), so this module keeps
the semantics but implements the wrap directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..d2q9 import C, OPPOSITE
from .fluid_node_table import collide_with_guo_forcing, macroscopic_with_forcing

Q = 9  #: D2Q9


@dataclass(frozen=True)
class SparseIndexTable:
    """Static gather table of the list-based sparse layout.

    Attributes
    ----------
    index : torch.Tensor, shape ``(N, 8)``, int64
        ``index[k, d-1]`` = flat SoA address ``d*N + k_up`` of the population that
        streams into (direction ``d``, fluid cell ``k``).  For a solid or
        out-of-domain upstream neighbour it is ``OPPOSITE[d]*N + k`` (half-way
        bounce-back folded into the table).
    cell : torch.Tensor, shape ``(N,)``, int64
        ``cell[k] = y*nx + x`` -- dense row-major index of fluid cell ``k``.
    lookup : torch.Tensor, shape ``(ny*nx,)``, int64
        ``lookup[y*nx + x]`` = fluid list index, ``-1`` on solid sites.
    ny, nx : int
    periodic : tuple[bool, bool]
        periodic wrap on ``(y, x)``.

    Bounce-back bookkeeping: ``index`` covers *every* (cell, direction) pair, so
    the streaming kernel never needs the solid mask -- this is the property that
    makes the list-based kernel branch free.
    """

    index: torch.Tensor
    cell: torch.Tensor
    lookup: torch.Tensor
    ny: int
    nx: int
    periodic: tuple[bool, bool]

    # ------------------------------------------------------------------ basic
    @property
    def n_fluid(self) -> int:
        return int(self.cell.numel())

    @property
    def n_total(self) -> int:
        return int(self.ny) * int(self.nx)

    @property
    def porosity(self) -> float:
        return self.n_fluid / self.n_total

    def to(self, device: torch.device | str) -> "SparseIndexTable":
        return SparseIndexTable(
            index=self.index.to(device),
            cell=self.cell.to(device),
            lookup=self.lookup.to(device),
            ny=self.ny,
            nx=self.nx,
            periodic=self.periodic,
        )

    # ------------------------------------------------------------- footprint
    def state_bytes(self, dtype: torch.dtype = torch.float64, index_int: int = 4) -> dict:
        """Storage footprint of the sparse list state vs the dense lattice state."""
        fbytes = torch.tensor([], dtype=dtype).element_size()
        k = self.n_fluid
        n = self.n_total
        sparse_pdf = Q * k * fbytes
        sparse_index = 8 * k * index_int + k * index_int
        dense_pdf = Q * n * fbytes
        dense_extra = n  # bool solid mask
        return {
            "sparse_pdf_bytes": sparse_pdf,
            "sparse_index_bytes": sparse_index,
            "sparse_total_bytes": sparse_pdf + sparse_index,
            "dense_pdf_bytes": dense_pdf,
            "dense_total_bytes": dense_pdf + dense_extra,
            "pdf_only_ratio": dense_pdf / sparse_pdf,
            "total_ratio": (dense_pdf + dense_extra) / (sparse_pdf + sparse_index),
        }


def build_index_table(
    solid: torch.Tensor,
    periodic: tuple[bool, bool] = (True, True),
) -> SparseIndexTable:
    """Build the static gather index array for a 2-D solid mask (lbmpy semantics).

    Parameters
    ----------
    solid : torch.Tensor, shape ``(ny, nx)``, bool
        ``True`` on the solid matrix.
    periodic : tuple[bool, bool]
        ``(y, x)`` periodicity.  A non-periodic border is treated exactly like a
        solid site (half-way bounce-back), which is the OSS ``no_slip`` branch.

    Returns
    -------
    SparseIndexTable

    Notes
    -----
    The build is ``O(8 N)`` and done once; the per-step cost is one gather plus
    one collision.  Folding bounce-back into the table is what the OSS index
    array does; it also means the *physics* of a wall is decided here and not in
    the kernel -- a wall is a half-way (mid-link) bounce-back, so no interpolation
    or wall distance enters, and a solid cell adjacent to the domain border is
    identical to a domain border.
    """
    if not isinstance(solid, torch.Tensor):
        solid = torch.as_tensor(solid)
    if solid.dim() != 2:
        raise ValueError("solid mask must be 2-D (ny, nx)")
    if solid.dtype != torch.bool:
        solid = solid.to(torch.bool)
    ny, nx = int(solid.shape[0]), int(solid.shape[1])
    fluid = ~solid
    cell = torch.nonzero(fluid.reshape(-1), as_tuple=False).squeeze(1)  # row-major
    n = int(cell.numel())
    if n == 0:
        raise ValueError("no fluid cells")

    kid = torch.arange(n, dtype=torch.int64)  # list index of every fluid cell
    lookup = torch.full((ny * nx,), -1, dtype=torch.int64)
    lookup[cell] = kid

    ys = torch.div(cell, nx, rounding_mode="floor")
    xs = cell - ys * nx
    idx = torch.empty((n, Q - 1), dtype=torch.int64)
    c = C.to(torch.int64)
    py, px = bool(periodic[0]), bool(periodic[1])
    for d in range(1, Q):
        cx, cy = int(c[d, 0]), int(c[d, 1])
        # gather source: the upstream neighbour (pull form, velocity labelling)
        uy = ys - cy
        ux = xs - cx
        if py:
            uy = torch.remainder(uy, ny)
        if px:
            ux = torch.remainder(ux, nx)
        inside = (uy >= 0) & (uy < ny) & (ux >= 0) & (ux < nx)
        flat = torch.where(inside, uy * nx + ux, torch.zeros_like(uy))
        nb = lookup[flat]
        is_fluid_up = inside & (nb >= 0)
        slot = torch.where(
            is_fluid_up,
            nb + d * n,                       # stream from the upstream fluid cell
            kid + int(OPPOSITE[d]) * n,       # half-way bounce-back (own cell, opposite slot)
        )
        idx[:, d - 1] = slot
    return SparseIndexTable(
        index=idx,
        cell=cell,
        lookup=lookup,
        ny=ny,
        nx=nx,
        periodic=(py, px),
    )


def gather(f: torch.Tensor, table: SparseIndexTable) -> torch.Tensor:
    """Streaming gather: ``g[d][k] = f.flat[index[k, d-1]]`` (bounce-back included)."""
    n = table.n_fluid
    if f.shape != (Q, n):
        raise ValueError(f"f must have shape (9, {n}), got {tuple(f.shape)}")
    g = f.reshape(-1)[table.index.reshape(-1)].reshape(n, Q - 1)
    out = torch.empty_like(f)
    out[0] = f[0]
    out[1:] = g.transpose(0, 1)
    return out


def sparse_step(
    f: torch.Tensor,
    table: SparseIndexTable,
    tau: float,
    force_x: float = 0.0,
    force_y: float = 0.0,
) -> torch.Tensor:
    """One list-based step: gather-stream + BGK collision with Guo forcing.

    ``f`` is the *post-collision* state of the previous step (``(9, N)`` SoA), the
    returned tensor is the post-collision state of this step -- the OSS
    ``stream_pull_collide`` structure with two alternating buffers.
    """
    g = gather(f, table)
    return collide_with_guo_forcing(g, tau, force_x, force_y)


def macroscopic_sparse(
    f: torch.Tensor,
    table: SparseIndexTable,
    force_x: float = 0.0,
    force_y: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``rho, ux, uy`` of the *pre-collision* state on the fluid-cell list."""
    return macroscopic_with_forcing(gather(f, table), force_x, force_y)


def equilibrium_sparse(
    table: SparseIndexTable,
    rho: float = 1.0,
    ux: float = 0.0,
    uy: float = 0.0,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Uniform-equilibrium start state on the fluid-cell list, shape ``(9, N)``."""
    f = torch.zeros((Q, table.n_fluid), dtype=dtype)
    from ..d2q9 import W_EXACT

    w = torch.tensor(W_EXACT, dtype=dtype)
    cu = 3.0 * (float(ux) * C[:, 0] + float(uy) * C[:, 1]).to(dtype)
    f[:] = (w * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * (ux * ux + uy * uy))).reshape(Q, 1)
    return f


def to_dense(f: torch.Tensor, table: SparseIndexTable, fill=float("nan")) -> torch.Tensor:
    """Scatter a ``(Q, N)`` list state onto the dense ``(Q, ny, nx)`` lattice (solid = ``fill``)."""
    out = torch.full((Q, table.ny * table.nx), fill, dtype=f.dtype)
    out[:, table.cell] = f
    return out.reshape(Q, table.ny, table.nx)


def dense_to_sparse(f: torch.Tensor, table: SparseIndexTable) -> torch.Tensor:
    """Compress a dense ``(Q, ny, nx)`` field onto the fluid-cell list."""
    n_fluid = table.n_fluid
    flat = f.reshape(Q, table.ny * table.nx)[:, table.cell]
    if flat.shape != (Q, n_fluid):
        raise ValueError("unexpected shape")
    return flat.contiguous()