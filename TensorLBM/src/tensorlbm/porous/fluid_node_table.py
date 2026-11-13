"""Compressed fluid-node neighbour table (porous-media LBM data layout).

Why this module exists
----------------------
In a porous medium only the fluid nodes carry degrees of freedom, yet the plain
lattice-Boltzmann layout keeps a 9-population PCR field on *every* lattice
site, including the solid matrix.  Mature porous-media implementations therefore
compact the unknowns onto the fluid nodes and keep a stencil lookup table that
maps each fluid node and each lattice direction to the *compressed* index of the
neighbouring fluid node (or to a sentinel that means "wall").

The layout transplanted here is the one used by
``PorousMediaSimulation/openLBMPM`` (see the provenance block below).  This
module supplies

* :class:`FluidNodeTable` -- the compaction map ``(ny, nx) -> [0, K)`` and the
  flat ``(K, 8)`` neighbour array with ``-1`` as the solid sentinel,
* :func:`compressed_step` -- one BGK + Guo-forcing + push-streaming step on the
  compressed layout, with in-place half-way bounce-back for solid neighbours,
* :func:`dense_step` -- the *same* physics on the full lattice (pull streaming,
  roll-based, as in :mod:`tensorlbm.d2q9`), used as the cross-validation
  control so that a representation change can be checked to be physics-neutral,
* small helpers (``permeability``, ``macroscopic_with_forcing``).

Provenance of the transplanted mechanism (openLBMPM, clone commit 3d84189)
--------------------------------------------------------------------------
``ShanChen2D/ShanChenD2Q9.py:589`` ``optimizeFluidArray()``
    builds ``fluidNodes[k] = i*nx + j`` (``i`` = y, ``j`` = x, row-major) and
    ``tmpIndicesDomain[i, j] = k`` for fluid, ``-1`` for solid.
``ShanChen2D/ShanChenD2Q9.py:605``
    ``neighboringNodes = np.zeros(fluidNodes.size * 8, dtype=np.int64)``.
``ShanChen2D/OptimizedD2Q9GPU.py:22`` ``fillNeighboringNodes()``
    stencil order ``E, N, W, S, NE, NW, SW, SE``; the neighbour index is looked
    up in the *compaction* map and both lattice directions wrap periodically
    (``tmpF = j+1 if j < nx-1 else 0``, ``tmpU = i+1 if i < ny-1 else 0``).
``ShanChen2D/OptimizedD2Q9GPU.py:452`` ``calStreaming1GPU()``
    push/scatter streaming: a population leaving fluid node ``k`` along ``d``
    lands in slot ``opposite(d)`` of the neighbour; when the table entry is
    ``-1`` it is reflected *in place* (half-way bounce-back),
    e.g. ``f_new[k, W] = f[k, E]``.

The OSS code is CUDA (``numba.cuda``); this environment has no GPU, so the
transplant reproduces the *semantics* (table build + streaming rules) exactly
and runs it on ``torch`` CPU in float64.

Conventions
-----------
* lattice directions follow :mod:`tensorlbm.d2q9`:
  ``0=(0,0) 1=E 2=N 3=W 4=S 5=NE 6=NW 7=SW 8=SE``,
* all quantities in lattice units, ``cs2 = 1/3``,
* the solid mask is ``True`` on the matrix (non-fluid) sites,
* the kernels are dtype-agnostic in principle but are meant to run in
  ``float64`` (``torch.set_default_dtype(torch.float64)``): in ``float32`` the
  push (compressed) and pull (dense) summation orders differ by ~1e-6 per step
  and the comparison tolerance of the XVAL (1e-8) is not reachable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..d2q9 import C, OPPOSITE, W_EXACT

#: streaming stencil order of the neighbour table: E, N, W, S, NE, NW, SW, SE
#: (identical to ``openLBMPM``'s ``fillNeighboringNodes`` order, slot 0..7)
DIRS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)

SOLID = -1  #: sentinel for "this neighbour is solid" (openLBMPM uses -1 too)


def _w(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(W_EXACT, dtype=dtype, device=device)


@dataclass(frozen=True)
class FluidNodeTable:
    """Compacted fluid-node map + flat neighbour table.

    Attributes
    ----------
    nodes : torch.Tensor, shape ``(K,)``, int64
        ``nodes[k] = y*nx + x`` -- the dense (row-major) index of fluid node
        ``k``, i.e. the inverse of ``index``.
    neighbour : torch.Tensor, shape ``(K, 8)``, int64
        ``neighbour[k, s]`` = compressed index of the neighbour of fluid node
        ``k`` in direction :data:`DIRS` ``[s]``, or :data:`SOLID` (``-1``).
    index : torch.Tensor, shape ``(ny, nx)``, int64
        dense -> compressed map, ``-1`` on solid sites.
    ny, nx : int
        dense lattice shape the table was built for.
    """

    nodes: torch.Tensor
    neighbour: torch.Tensor
    index: torch.Tensor
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

    def to(self, device: torch.device | str) -> "FluidNodeTable":
        return FluidNodeTable(
            nodes=self.nodes.to(device),
            neighbour=self.neighbour.to(device),
            index=self.index.to(device),
            ny=self.ny,
            nx=self.nx,
        )

    # ------------------------------------------------------------- footprints
    def state_bytes(self, dtype: torch.dtype = torch.float64, table_int: int = 8) -> dict:
        """Storage footprint of the compressed state vs the dense state.

        ``table_int`` is the integer width used for the neighbour table and the
        compaction map (8 = int64 as in openLBMPM, 4 = int32).
        """
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


def build_fluid_node_table(
    solid: torch.Tensor,
    *,
    wrap: bool = True,
) -> FluidNodeTable:
    """Compact the fluid sites of ``solid`` and build the neighbour stencil.

    Parameters
    ----------
    solid : torch.Tensor, shape ``(ny, nx)``, bool
        ``True`` on solid (matrix) sites.
    wrap : bool
        Periodic wrap of the neighbour lookup in both lattice directions.  The
        OSS implementation wraps unconditionally at table-build time; ``False``
        is kept for bounded (wall-bounded) set-ups and marks out-of-domain
        neighbours as solid.

    Returns
    -------
    FluidNodeTable

    Notes
    -----
    Compaction order is row-major (``y`` outer, ``x`` inner), the same order as
    ``openLBMPM``'s ``optimizeFluidArray`` double loop, so ``nodes`` reproduces
    ``fluidNodes`` bit for bit.
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
            inside = (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
            vals = torch.full_like(ys, SOLID)
            vals[inside] = index[yy[inside], xx[inside]]
            neighbour[:, slot] = vals
    return FluidNodeTable(nodes=nodes, neighbour=neighbour, index=index, ny=ny, nx=nx)


# --------------------------------------------------------------------- physics
def macroscopic_with_forcing(
    f: torch.Tensor,
    force_x: float = 0.0,
    force_y: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``rho`` and Guo-shifted velocity ``u* = (sum f c + F/2)/rho``.

    Works on any trailing shape (dense ``(9, ny, nx)`` and compressed ``(9, K)``).
    """
    c = C.to(f.device).to(f.dtype)
    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    ux = ((f * c[:, 0].view((-1,) + (1,) * (f.dim() - 1))).sum(dim=0) + 0.5 * force_x) / rho_safe
    uy = ((f * c[:, 1].view((-1,) + (1,) * (f.dim() - 1))).sum(dim=0) + 0.5 * force_y) / rho_safe
    return rho, ux, uy


def _equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    c = C.to(rho.device).to(rho.dtype)
    w = _w(rho.dtype, rho.device)
    shape = (-1,) + (1,) * rho.dim()
    u_sq = ux * ux + uy * uy
    cu = c[:, 0].view(shape) * ux + c[:, 1].view(shape) * uy
    return w.view(shape) * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(0))


def collide_with_guo_forcing(
    f: torch.Tensor,
    tau: float,
    force_x: float = 0.0,
    force_y: float = 0.0,
) -> torch.Tensor:
    """BGK collision + Guo (1997) body-force term, valid for any trailing shape.

    ``F_i = (1 - 1/(2 tau)) w_i [3 (c_i - u*) . F + 9 (c_i . u*)(c_i . F)]`` with
    ``u*`` the half-force-shifted velocity.  Both XVAL arms call this exact
    function, so the forcing cannot be a source of disagreement.
    """
    rho, ux, uy = macroscopic_with_forcing(f, force_x, force_y)
    feq = _equilibrium(rho, ux, uy)
    f_post = f - (f - feq) / tau
    if force_x != 0.0 or force_y != 0.0:
        c = C.to(f.device).to(f.dtype)
        w = _w(f.dtype, f.device)
        shape = (-1,) + (1,) * (f.dim() - 1)
        cxf = c[:, 0].view(shape)
        cyf = c[:, 1].view(shape)
        cu = cxf * ux + cyf * uy
        cF = cxf * force_x + cyf * force_y
        uF = ux * force_x + uy * force_y
        src = (1.0 - 0.5 / tau) * w.view(shape) * (3.0 * (cF - uF) + 9.0 * cu * cF)
        f_post = f_post + src
    return f_post


def compressed_step(
    f: torch.Tensor,
    table: FluidNodeTable,
    tau: float,
    force_x: float = 0.0,
    force_y: float = 0.0,
) -> torch.Tensor:
    """One compressed-layout step: collide, then push-stream with the table.

    Parameters
    ----------
    f : torch.Tensor, shape ``(9, K)``
        Populations of the ``K`` fluid nodes only.
    table : FluidNodeTable
    tau : float
    force_x, force_y : float
        Uniform body force per unit mass (lattice units).

    Returns
    -------
    f_new : torch.Tensor, shape ``(9, K)``

    Notes
    -----
    This is the ``calStreaming1GPU`` rule of openLBMPM: a population pushed
    along ``d`` lands in the neighbour's slot for the *same* velocity direction
    ``d``, and a ``-1`` table entry is reflected in place as ``opposite(d)``
    (half-way bounce-back).  Every (node, direction) entry is written by exactly
    one source, so the scatter is conflict free (a "push" without atomics).

    Label convention (transplantation choice, see the module docstring): the OSS
    kernel writes ``f_new[opposite(d)][neighbour] = f[d][node]``, i.e. it labels
    the populations by the side they *arrived from*; TensorLBM labels them by
    their velocity direction (:mod:`tensorlbm.d2q9`), so the same physical rule
    reads ``f_new[d][neighbour] = f[d][node]`` here.  Storage order of the
    neighbour table (E, N, W, S, NE, NW, SW, SE) is unchanged.
    """
    f_post = collide_with_guo_forcing(f, tau, force_x, force_y)
    f_new = torch.empty_like(f_post)
    f_new[0] = f_post[0]
    for slot, d in enumerate(DIRS):
        nb = table.neighbour[:, slot]
        ok = nb >= 0
        if bool(ok.any()):
            src = torch.nonzero(ok, as_tuple=False).squeeze(1)
            f_new[d][nb[src]] = f_post[d][src]
        wall = torch.nonzero(~ok, as_tuple=False).squeeze(1)
        if wall.numel():
            f_new[int(OPPOSITE[d])][wall] = f_post[d][wall]
    return f_new


def dense_stream(
    f_post: torch.Tensor,
    solid: torch.Tensor,
) -> torch.Tensor:
    """Pull streaming on the full lattice + solid-mask half-way bounce-back.

    Dense counterpart of :func:`compressed_step`'s streaming: periodic pull
    (``torch.roll``, the scheme of :func:`tensorlbm.d2q9.stream`) followed by the
    in-place reflection ``f_new[opp(d)] = f_post[d]`` at every node whose
    ``d``-neighbour is solid.  Values on solid nodes are never read by a fluid
    node after the reflection, so they are left as garbage on purpose (that is
    precisely the memory the compressed layout saves).
    """
    c = C.to(f_post.device)
    f_new = torch.empty_like(f_post)
    ny, nx = int(solid.shape[0]), int(solid.shape[1])
    if int(f_post.shape[1]) != ny or int(f_post.shape[2]) != nx:
        raise ValueError("f_post trailing shape must equal the solid mask shape")
    f_new[0] = f_post[0]
    for d in range(1, 9):
        cx, cy = int(c[d, 0]), int(c[d, 1])
        # pull: out[y, x] = in[y - cy, x - cx]  (velocity-direction labelling;
        # tensorlbm.d2q9.stream uses the mirrored source-side labelling, see the
        # module docstring and test_porous_fluid_node_table.py)
        f_new[d] = torch.roll(f_post[d], shifts=(cy, cx), dims=(0, 1))
    for d in range(1, 9):
        cx, cy = int(c[d, 0]), int(c[d, 1])
        dest_solid = torch.roll(solid, shifts=(-cy, -cx), dims=(0, 1))
        f_new[int(OPPOSITE[d])][dest_solid] = f_post[d][dest_solid]
    return f_new


def dense_step(
    f: torch.Tensor,
    solid: torch.Tensor,
    tau: float,
    force_x: float = 0.0,
    force_y: float = 0.0,
) -> torch.Tensor:
    """One dense-lattice step sharing collision/forcing with :func:`compressed_step`."""
    f_post = collide_with_guo_forcing(f, tau, force_x, force_y)
    return dense_stream(f_post, solid)


def permeability(u_mean: float, nu: float, g: float) -> float:
    """Darcy permeability ``k = nu * <u> / g`` (lattice units).

    Two conventions have to be kept apart (mix them and the answer is wrong by
    a factor of the porosity):

    * **superficial / Darcy** permeability -- ``<u>`` averaged over *all* lattice
      sites (solid sites carry ``u = 0``).  This is the permeability a Darcy
      solver consumes and the one the XVAL reports as ``k_super``.
    * **intrinsic** permeability -- ``<u>`` averaged over the fluid sites only:
      ``k_intrinsic = k_super / porosity``.  The classic parallel-plate value
      ``H^2/12`` is an *intrinsic* permeability of the channel cross-section.

    For the periodic slab channel used by the XVAL, ``k_super = phi * H^2/12``
    with ``phi = H/ny`` is the exact analytic reference.
    """
    return nu * u_mean / g


def intrinsic_permeability(k_super: float, porosity: float) -> float:
    """Convert a superficial permeability into the intrinsic one."""
    if not 0.0 < porosity <= 1.0:
        raise ValueError("porosity must be in (0, 1]")
    return k_super / porosity