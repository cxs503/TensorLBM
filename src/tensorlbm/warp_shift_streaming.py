"""Shared-memory warp-shift streaming (transplanted from openLBMPM).

Why this module exists
----------------------
The plain D2Q9 streaming step in :mod:`tensorlbm.d2q9` style codes is a *pull*
over the whole array (``torch.roll`` / ``np.roll``), i.e. one temporary copy of
the full ``(9, ny, nx)`` field per step.  Mature GPU porous-media codes instead
stream *inside a CUDA block*: the populations of one lattice row are staged in
shared memory along the x direction, shifted by one lane ("warp shift"), and
only the cross-tile and cross-row populations touch global memory.

This module supplies

* :func:`pull_stream` -- the in-repo reference streaming (periodic ``torch.roll``
  pull), used as the *control* arm of the cross-validation,
* :func:`oss_literal_stream` -- a line-by-line transcription of the upstream
  CUDA kernel, including its *boundary behaviour* (see PROVENANCE),
* :func:`warp_shift_stream` -- the vectorised form of the same rule, with an
  explicit switch for the two wrappings (x inside a tile, and the domain edges)
  so that the semantic equivalence to the OSS kernel can be checked exactly,
* :func:`bgk_stream_step` -- BGK collision + one warp-shift streaming step.

PROVENANCE (upstream clone commit ``3d841891daafe7b6e9f65e51dd9bf20f2b0e1792``)
------------------------------------------------------------------------------
``ShanChen2D/AccelerateGPU2D.py:1260`` ``@cuda.jit('void(int64, int64, int64,
float64[:, :], float64[:, :])')`` ``def calStreamingGPU(nx, ny, numberThreads,
fluidDistrOld, fluidDistrNew)`` -- the mechanism transplanted here:

* ``:1262-1263`` ``tx = cuda.threadIdx.x; bx = cuda.blockIdx.x; by = cuda.blockIdx.y``
  and ``:1264-1266`` ``xStart = tx + bx*numberThreads; yStart = by;
  indicesK = nx*yStart + xStart`` -- **one block == one lattice row**, a block
  covers a tile of ``numberThreads`` consecutive columns.
* ``:1268-1276`` nine shared arrays ``cuda.shared.array(shape=(32,), dtype=float64)``
  (``distrFOut0`` .. ``distrFOut8``) -- the row tile is staged in shared memory;
  the tile width is the CUDA warp size (32), i.e. the shift is a warp shift.
* ``:1279-1281`` the rest/vertical populations are staged at the *same* lane
  (``distrFOut0[tx]``, ``distrFOut2[tx]``, ``distrFOut4[tx]``).
* ``:1282-1305`` push (scatter) shift by one lane inside the tile:
  ``distrFOut1[tx+1] = fluidDistrOld[1, indicesK]`` (E),
  ``distrFOut3[tx-1] = ...`` (W), ``[5]->tx+1`` (NE), ``[6]->tx-1`` (NW),
  ``[7]->tx-1`` (SW), ``[8]->tx+1`` (SE); the edge lanes use the **modular
  in-tile wrap** (``tx == 0``: ``distrFOut1[tx+1]`` and
  ``distrFOut3[numberThreads-1]``; ``tx == numberThreads-1``:
  ``distrFOut1[0]`` and ``distrFOut3[tx-1]``), so a population leaving the tile
  is *stashed* at the opposite edge lane of the same tile.
* ``:1306`` ``cuda.syncthreads()`` -- the staging buffer is read after the shift.
* ``:1308-1330`` write-out.  ``fluidDistrNew[0|1|3, indicesK] = distrFOut*[tx]``
  (x-shifted, same row); ``fluidDistrNew[2|5|6, nx*(yStart+1)+xStart]`` and
  ``fluidDistrNew[4|7|8, nx*(yStart-1)+xStart]`` -- the vertical half of the
  stencil is delivered **directly into the neighbouring row of the global
  array**, guarded by ``if (by < ny - 1)`` / ``if (by > 0)``.  The periodic
  y-wrap lines are present in the source but commented out (``:1318-1321``,
  ``:1332-1336``), so the kernel itself is *not* periodic in y.
* ``:2733`` ``exchangeDistrGlobalGPU(nx, ny, numberThreads, fluidDistr)`` -- the
  cross-tile repair: the east-going populations are copied
  ``fluidDistr[1|5|8, kTargetE] = fluidDistr[1|5|8, kStartE]`` in **descending**
  tile order (``:2747-2760``) and the west-going ones
  ``fluidDistr[3|6|7, kTargetW] = fluidDistr[3|6|7, kStartW]`` in **ascending**
  tile order (``:2739-2746``), reading the stashed edge lane written by the
  shared kernel.  The two opposite traversal directions are what makes the
  in-place global repair race-free (no extra buffer).

Consequences that are *not* in the upstream kernel and matter here
------------------------------------------------------------------
1. the domain edge columns only carry the in-tile stash, i.e. the kernel is
   interior-consistent only; upstream fixes the edges afterwards with its
   boundary-condition kernels.  :func:`warp_shift_stream` therefore exposes
   ``x_periodic``/``y_periodic``: with both ``True`` the domain edges wrap
   physically, which is the only mode that can be compared with a periodic
   reference flow.
2. with ``x_periodic=False, y_periodic=False`` the vectorised form is expected
   to be **bit-exactly** equal to :func:`oss_literal_stream`, i.e. the fast form
   is a faithful transcription and not a re-derivation.

Conventions
-----------
* field layout ``f[9, ny, nx]``, direction order of :mod:`tensorlbm.d2q9`
  (``0=rest, 1=E, 2=N, 3=W, 4=S, 5=NE, 6=NW, 7=SW, 8=SE``) -- the same order as
  ``openLBMPM``'s ``microVelocity`` and of the neighbour table transplanted in
  :mod:`tensorlbm.porous` (see ``DIRS = (1, 2, 3, 4, 5, 6, 7, 8)``),
* lattice units, ``cs2 = 1/3``, ``nu = (tau - 0.5) / 3``,
* ``nx`` must be a multiple of the tile width (upstream also assumes
  ``nx % numberThreads == 0`` when it computes ``nbx = int(nx/numberThreads)+1``).
"""

from __future__ import annotations

import torch

from .d2q9 import W_EXACT, equilibrium, macroscopic

#: upstream tile width == CUDA warp size (``cuda.shared.array(shape=(32,))``)
WARP_TILE: int = 32

#: directions whose x component is +1 / -1 (they are lane-shifted inside a tile)
EAST: tuple[int, ...] = (1, 5, 8)
WEST: tuple[int, ...] = (3, 6, 7)
#: directions whose y component is +1 / -1 (they are written into a neighbour row)
NORTH: tuple[int, ...] = (2, 5, 6)
SOUTH: tuple[int, ...] = (4, 7, 8)


def _check(f: torch.Tensor, tile: int) -> tuple[int, int, int]:
    if f.ndim != 3 or f.shape[0] != 9:
        raise ValueError(f"expected f[9, ny, nx], got {tuple(f.shape)}")
    ny, nx = int(f.shape[1]), int(f.shape[2])
    if tile < 1:
        raise ValueError(f"tile must be >= 1, got {tile}")
    if nx % tile != 0:
        raise ValueError(
            f"nx={nx} must be a multiple of tile={tile} "
            "(upstream assumes nx % numberThreads == 0)"
        )
    return 9, ny, nx


def pull_stream(f: torch.Tensor) -> torch.Tensor:
    """Reference streaming: whole-array periodic pull (``torch.roll``).

    ``f_new[d, y, x] = f[d, y - cy_d, x - cx_d]`` with periodic wrap on both
    axes.  This is the control arm: it defines the streaming operator that the
    warp-shift form is compared against.
    """
    out = torch.empty_like(f)
    c = ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1))
    out[0] = f[0]
    for d, (cx, cy) in zip(range(1, 9), c):
        out[d] = torch.roll(f[d], shifts=(cy, cx), dims=(0, 1))
    return out


def oss_literal_stream(f: torch.Tensor, tile: int = WARP_TILE) -> torch.Tensor:
    """Line-by-line transcription of ``calStreamingGPU`` + ``exchangeDistrGlobalGPU``.

    Deliberately *not* vectorised: the loops over tiles and lanes follow the
    upstream thread/block indices, the staging arrays are separate buffers,
    the write-out is guarded exactly as in the CUDA source, and the cross-tile
    repair runs in the upstream traversal order so that the in-place global
    swap is reproduced.  Used to prove that :func:`warp_shift_stream` implements
    the same rule (K1 of the XVAL) instead of an idealised re-derivation.
    """
    _, ny, nx = _check(f, tile)
    new = torch.zeros_like(f)
    nbx = nx // tile

    # ---- calStreamingGPU: one block per row, one tile of columns per block ----
    for by in range(ny):
        for bx in range(nbx):
            # shared staging buffers (upstream: nine (32,) float64 arrays)
            stage = [torch.zeros(tile, dtype=f.dtype) for _ in range(9)]
            old_row = f[:, by, :]
            # rest / vertical populations stay at the same lane (tx)
            for d in (0, 2, 4):
                for tx in range(tile):
                    stage[d][tx] = old_row[d, bx * tile + tx]
            # push shift by one lane, with the modular in-tile wrap
            for tx in range(tile):
                x = bx * tile + tx
                if tx == 0:
                    stage[1][tx + 1] = old_row[1, x]
                    stage[3][tile - 1] = old_row[3, x]
                    stage[5][tx + 1] = old_row[5, x]
                    stage[6][tile - 1] = old_row[6, x]
                    stage[7][tile - 1] = old_row[7, x]
                    stage[8][tx + 1] = old_row[8, x]
                if tx == tile - 1:
                    stage[1][0] = old_row[1, x]
                    stage[3][tx - 1] = old_row[3, x]
                    stage[5][0] = old_row[5, x]
                    stage[6][tx - 1] = old_row[6, x]
                    stage[7][tx - 1] = old_row[7, x]
                    stage[8][0] = old_row[8, x]
                if 0 < tx < tile - 1:
                    stage[1][tx + 1] = old_row[1, x]
                    stage[3][tx - 1] = old_row[3, x]
                    stage[5][tx + 1] = old_row[5, x]
                    stage[6][tx - 1] = old_row[6, x]
                    stage[7][tx - 1] = old_row[7, x]
                    stage[8][tx + 1] = old_row[8, x]
            # write-out (after the upstream syncthreads)
            for tx in range(tile):
                x = bx * tile + tx
                new[0, by, x] = stage[0][tx]
                new[1, by, x] = stage[1][tx]
                new[3, by, x] = stage[3][tx]
                if by < ny - 1:
                    new[2, by + 1, x] = stage[2][tx]
                    new[5, by + 1, x] = stage[5][tx]
                    new[6, by + 1, x] = stage[6][tx]
                if by > 0:
                    new[4, by - 1, x] = stage[4][tx]
                    new[7, by - 1, x] = stage[7][tx]
                    new[8, by - 1, x] = stage[8][tx]

    # ---- exchangeDistrGlobalGPU: cross-tile repair, in the upstream order ----
    for i in range(nbx - 1):  # west-going, ascending tile order
        x_start_w = i * tile + 2 * tile - 1  # last column of tile i+1
        x_target_w = x_start_w - tile  # last column of tile i
        if x_start_w < nx and x_target_w < nx:
            for d in WEST:
                new[d, :, x_target_w] = new[d, :, x_start_w]
    for j in range(nbx - 1):  # east-going, descending tile order
        i = (nbx - 2) - j
        x_start_e = i * tile
        x_target_e = x_start_e + tile
        if x_start_e < nx and x_target_e < nx:
            for d in EAST:
                new[d, :, x_target_e] = new[d, :, x_start_e]
    return new


def warp_shift_stream(
    f: torch.Tensor,
    tile: int = WARP_TILE,
    x_periodic: bool = False,
    y_periodic: bool = False,
) -> torch.Tensor:
    """Vectorised shared-memory warp-shift streaming.

    Same rule as :func:`oss_literal_stream` (in-tile modular lane shift +
    cross-tile repair + direct neighbour-row delivery for the vertical half of
    the stencil), written as block-wise ``roll`` instead of per-lane loops.

    ``x_periodic`` / ``y_periodic`` close the two domain edges that the upstream
    kernel leaves to its boundary-condition stage; with both ``True`` the
    operator is the periodic streaming operator of :func:`pull_stream`.
    """
    _, ny, nx = _check(f, tile)
    nbx = nx // tile
    out = torch.zeros_like(f)
    out[0] = f[0]

    # --- x half of the stencil: stage each row tile, shift by one lane ---
    staged = f.clone()  # directions 2/4 keep their column: roll by 0
    blk_shape = (ny, nbx, tile)
    for d in EAST:
        blk = f[d].reshape(blk_shape)
        shifted = torch.roll(blk, shifts=1, dims=2)  # in-tile modular warp shift
        shifted[:, 1:, 0] = blk[:, :-1, tile - 1]  # cross-tile repair (asc? desc)
        if x_periodic:
            shifted[:, 0, 0] = blk[:, -1, tile - 1]  # physical x wrap
        staged[d] = shifted.reshape(ny, nx)
    for d in WEST:
        blk = f[d].reshape(blk_shape)
        shifted = torch.roll(blk, shifts=-1, dims=2)
        shifted[:, :-1, tile - 1] = blk[:, 1:, 0]
        if x_periodic:
            shifted[:, -1, tile - 1] = blk[:, 0, 0]
        staged[d] = shifted.reshape(ny, nx)

    # --- y half: deliver the shifted row tile into the neighbour row ---
    for d in NORTH:
        out[d, 1:, :] = staged[d, :-1, :]
        if y_periodic:
            out[d, 0, :] = staged[d, -1, :]
    for d in SOUTH:
        out[d, :-1, :] = staged[d, 1:, :]
        if y_periodic:
            out[d, -1, :] = staged[d, 0, :]
    for d in (0, 1, 3):
        out[d] = staged[d]
    return out


def bgk_stream_step(f: torch.Tensor, tau: float, tile: int = WARP_TILE) -> torch.Tensor:
    """BGK collision + one periodic warp-shift streaming step (atol-friendly)."""
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    post = f + (feq - f) / tau
    return warp_shift_stream(post, tile=tile, x_periodic=True, y_periodic=True)


def kinetic_energy(f: torch.Tensor) -> torch.Tensor:
    """Total kinetic energy ``0.5 * sum(rho * |u|^2)`` (Taylor-Green diagnostic)."""
    rho, ux, uy = macroscopic(f)
    return 0.5 * torch.sum(rho * (ux * ux + uy * uy))


def taylor_green_field(
    nx: int,
    ny: int | None = None,
    u0: float = 0.04,
    rho0: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """D2Q9 Taylor-Green vortex initial condition (single-mode, periodic domain)."""
    ny = nx if ny is None else ny
    kx, ky = 2.0 * torch.pi / nx, 2.0 * torch.pi / ny
    y, x = torch.meshgrid(
        torch.arange(ny, dtype=dtype),
        torch.arange(nx, dtype=dtype),
        indexing="ij",
    )
    ux = u0 * torch.sin(kx * x) * torch.cos(ky * y)
    uy = -u0 * torch.cos(kx * x) * torch.sin(ky * y)
    rho = rho0 - 0.75 * u0 * u0 * (torch.cos(2 * kx * x) + torch.cos(2 * ky * y))
    return equilibrium(rho, ux, uy)


def weights(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Exact D2Q9 weights (float64 literals, see :func:`tensorlbm.d2q9._w_for`)."""
    return torch.tensor(W_EXACT, dtype=dtype)