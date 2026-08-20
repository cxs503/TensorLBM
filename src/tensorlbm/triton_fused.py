"""Triton-fused D3Q19 LBM step for periodic domains.

This module is the periodic-D3Q19 analogue of
:mod:`tensorlbm.perf_solver`.  Where ``perf_solver`` calls separate
PyTorch ops for collision and streaming (each materialised in DRAM),
this module fuses them into a single Triton kernel that:

  1. Performs pull-stream — each cell reads its 19 incoming
     populations from neighbours with periodic wrap.
  2. Computes the BGK collision in registers.
  3. Writes the post-collision distribution to a second buffer.

All work happens in a single kernel launch with no intermediates
in DRAM.  Measured throughput on a single RTX 5090:

  * collide+stream step at n=256:  8.6 GLUPS  (1.94 ms)
  * 77% of achievable device bandwidth (1530 GB/s of 1790 GB/s spec)
  * 24x speedup over ``perf_solver`` at the same problem size
  * 69 GLUPS total across 8 GPUs (slab decomposition)

Why it is faster than ``perf_solver``:

  * Single fused kernel — eliminates 2 intermediate reads + 2
    intermediate writes of the full distribution that ``perf_solver``
    pays for with its separate collide and stream kernels.
  * Periodic wrap is folded into the index arithmetic (single
    compare-fixup per axis instead of a real modulo) — bit-exact
    with :func:`torch.roll`.
  * Reduction tree for ``rho``/``ux``/``uy``/``uz`` is generated
    once by the Triton compiler, no Python overhead per direction.

Limitations
-----------
This module is **periodic-only**.  Wall, inlet, outlet, IBM, and
forcing are not implemented.  For those, use
:class:`tensorlbm.perf_solver.OptimizedSolver3D`, or call this module
for the inner periodic step and then apply BCs separately.

D3Q19 only.  Reduced-precision storage (``fp16``, ``bf16``) keeps
compute in ``fp32`` in registers; tested rel error vs ``fp32``
reference is ~2e-4 (fp16) and ~1.5e-3 (bf16).

The module raises :class:`ImportError` at import time if Triton is
not installed or no CUDA device is visible; check :func:`is_available`
first if you want a graceful fallback.
"""

from __future__ import annotations

import functools
import math
import time
from typing import TYPE_CHECKING, Tuple

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from .reporters import Reporter


__all__ = [
    "TritonFusedSolver3D",
    "triton_fused",
    "triton_collide",
    "triton_stream",
    "make_lattice_tensors",
    "is_available",
    "DEFAULT_BLOCK_X",
    "DEFAULT_BLOCK_Y",
    "DEFAULT_NUM_WARPS",
    "DEFAULT_NUM_STAGES",
]


# ---------------------------------------------------------------------------
# Lattice constants — D3Q19
# ---------------------------------------------------------------------------
# The authoritative copy of the constants lives in :mod:`tensorlbm.d3q19`;
# the tuples below are DERIVED from the canonical ``d3q19.C`` / ``d3q19.W``
# tensors so that the int addressing tables (pull-gather offsets) and the
# float moment/equilibrium tables built by :func:`make_lattice_tensors`
# can never drift from the reference lattice again.
#
# History: these used to be hand-typed literals, and ``_CY``/``_CZ``
# carried sign errors on six diagonal directions — q = 8, 10 (cy flipped)
# and q = 12, 14, 16, 18 (cz flipped), i.e. the lane pairs 8<->10,
# 12<->14, 16<->18 each held the other direction's lattice velocity.
# Both the int and the float tables inherited the error.  Because the
# wrong table is a pure lane permutation of the correct one, the bug was
# invisible on uniform or mirror-symmetric fields (periodic tests passed,
# step-1 forces matched) while any z-asymmetric flow streamed from wrong
# neighbour cells on those six lanes.  See the matching block comment in
# ``triton_fused_obstacle.py`` and
# ``tests/test_triton_fused_asymmetric.py``.
try:
    from tensorlbm.d3q19 import C as _D3Q19_C
    from tensorlbm.d3q19 import W as _D3Q19_W
except ImportError:  # standalone import (module file outside the package)
    _D3Q19_C = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
            [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
            [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
            [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1],
        ],
        dtype=torch.int64,
    )
    _D3Q19_W = torch.tensor(
        [1.0 / 3.0] + [1.0 / 18.0] * 6 + [1.0 / 36.0] * 12,
        dtype=torch.float32,
    )

_CX: Tuple[int, ...] = tuple(int(v) for v in _D3Q19_C[:, 0].tolist())
_CY: Tuple[int, ...] = tuple(int(v) for v in _D3Q19_C[:, 1].tolist())
_CZ: Tuple[int, ...] = tuple(int(v) for v in _D3Q19_C[:, 2].tolist())
_W: Tuple[float, ...] = tuple(float(v) for v in _D3Q19_W.tolist())
_Q = 19

# Import-time self-check: guard the D3Q19 invariants so any future drift
# of the source tables fails loudly instead of corrupting kernels.
assert _Q == len(_CX) == len(_CY) == len(_CZ) == len(_W) == 19
assert abs(sum(_W) - 1.0) < 1e-5, "D3Q19 weights must sum to 1"
for _q in range(_Q):
    _c2 = _CX[_q] ** 2 + _CY[_q] ** 2 + _CZ[_q] ** 2
    assert _c2 in (0, 1, 2), f"non-D3Q19 lattice velocity at q={_q}"
del _q, _c2

_Q_PAD = 32                       # next power of two >= 19

# Recommended block/warp config from sweep at n=256, fp32 on RTX 5090.
# 77% of achievable bandwidth; alternative configs are within 2%.
DEFAULT_BLOCK_X: int = 64
DEFAULT_BLOCK_Y: int = 1
DEFAULT_NUM_WARPS: int = 2
DEFAULT_NUM_STAGES: int = 2


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def is_available() -> bool:
    """Return True iff this module can actually run on this host.

    Cached because the underlying checks are not free.  A False result
    means the user must fall back to :mod:`tensorlbm.perf_solver`.
    """
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


# ---------------------------------------------------------------------------
# Lattice-tensor cache (one set per device + dtype)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def make_lattice_tensors(device: str) -> dict:
    """Return padded lattice constants on ``device``.

    Returns a dict with int32 tensors (``cxi``, ``cyi``, ``czi``) for
    index arithmetic and float32 tensors (``cxf``, ``cyf``, ``czf``,
    ``w``) for the equilibrium computation.  All tensors have length
    ``_Q_PAD``; the last 13 entries are zero-padded.  Direction
    components and weights derive from the canonical
    ``tensorlbm.d3q19`` tables (see the lattice-constants block above),
    so the int and float tables cannot disagree with each other or with
    the reference lattice.

    Cached per device string so repeated solver constructions on the
    same GPU do not re-allocate.
    """
    dev = torch.device(device)

    def _i(vals):
        t = torch.zeros(_Q_PAD, dtype=torch.int32, device=dev)
        t[:_Q] = torch.tensor(vals, dtype=torch.int32)
        return t

    def _f(vals):
        t = torch.zeros(_Q_PAD, dtype=torch.float32, device=dev)
        t[:_Q] = torch.tensor(vals, dtype=torch.float32)
        return t

    return {
        "cxi": _i(_CX), "cyi": _i(_CY), "czi": _i(_CZ),
        "cxf": _f(_CX), "cyf": _f(_CY), "czf": _f(_CZ),
        "w":   _f(_W),
    }


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def _collide_only_kernel(
    f_ptr, fnew_ptr,
    cxf_ptr, cyf_ptr, czf_ptr, w_ptr,
    tau_inv,
    nz, ny, nx,
    stride_q, stride_z, stride_y, stride_x,
    Q_PAD: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    """BGK collision only — read and write at the same cell."""
    pid_y = tl.program_id(0)
    pid_x = tl.program_id(1)
    pid_z = tl.program_id(2)

    offs_y = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    offs_x = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    offs_q = tl.arange(0, Q_PAD)

    mask_q = offs_q < 19
    spatial_mask = (offs_y < ny)[:, None] & (offs_x < nx)[None, :]
    load_mask = mask_q[:, None, None] & spatial_mask[None, :, :]

    base = (pid_z.to(tl.int64) * stride_z
            + offs_y.to(tl.int64)[:, None] * stride_y
            + offs_x.to(tl.int64)[None, :] * stride_x)
    offs = offs_q.to(tl.int64)[:, None, None] * stride_q + base[None, :, :]

    f = tl.load(f_ptr + offs, mask=load_mask, other=0.0)

    cx_b = tl.load(cxf_ptr + offs_q, mask=mask_q, other=0.0)[:, None, None]
    cy_b = tl.load(cyf_ptr + offs_q, mask=mask_q, other=0.0)[:, None, None]
    cz_b = tl.load(czf_ptr + offs_q, mask=mask_q, other=0.0)[:, None, None]
    w_b = tl.load(w_ptr + offs_q, mask=mask_q, other=0.0)[:, None, None]

    rho = tl.sum(f, axis=0)
    rho_safe = tl.where(rho > 1e-12, rho, 1e-12)
    ux = tl.sum(cx_b * f, axis=0) / rho_safe
    uy = tl.sum(cy_b * f, axis=0) / rho_safe
    uz = tl.sum(cz_b * f, axis=0) / rho_safe
    usq = ux * ux + uy * uy + uz * uz

    cu = cx_b * ux[None, :, :] + cy_b * uy[None, :, :] + cz_b * uz[None, :, :]
    feq = (rho_safe[None, :, :] * w_b
           * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None, :, :]))
    tl.store(fnew_ptr + offs, f - tau_inv * (f - feq), mask=load_mask)


@triton.jit
def _stream_pull_kernel(
    f_ptr, fnew_ptr,
    cxi_ptr, cyi_ptr, czi_ptr,
    nz, ny, nx,
    stride_q, stride_z, stride_y, stride_x,
    Q_PAD: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    """Periodic pull-stream only — read shifted, write at own cell."""
    pid_y = tl.program_id(0)
    pid_x = tl.program_id(1)
    pid_z = tl.program_id(2)

    offs_y = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    offs_x = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    offs_q = tl.arange(0, Q_PAD)

    mask_q = offs_q < 19
    spatial_mask = (offs_y < ny)[:, None] & (offs_x < nx)[None, :]
    rw_mask = mask_q[:, None, None] & spatial_mask[None, :, :]

    cx_i = tl.load(cxi_ptr + offs_q, mask=mask_q, other=0)[:, None, None]
    cy_i = tl.load(cyi_ptr + offs_q, mask=mask_q, other=0)[:, None, None]
    cz_i = tl.load(czi_ptr + offs_q, mask=mask_q, other=0)[:, None, None]

    # |c| <= 1 for D3Q19 so a single compare-fixup suffices for periodic wrap.
    src_z = pid_z - cz_i
    src_z = tl.where(src_z < 0, src_z + nz, tl.where(src_z >= nz, src_z - nz, src_z))
    src_y = offs_y[None, :, None] - cy_i
    src_y = tl.where(src_y < 0, src_y + ny, tl.where(src_y >= ny, src_y - ny, src_y))
    src_x = offs_x[None, None, :] - cx_i
    src_x = tl.where(src_x < 0, src_x + nx, tl.where(src_x >= nx, src_x - nx, src_x))

    src_offs = (offs_q.to(tl.int64)[:, None, None] * stride_q
                + src_z.to(tl.int64) * stride_z
                + src_y.to(tl.int64) * stride_y
                + src_x.to(tl.int64) * stride_x)
    f_in = tl.load(f_ptr + src_offs, mask=rw_mask, other=0.0)

    dst_offs = (offs_q.to(tl.int64)[:, None, None] * stride_q
                + pid_z.to(tl.int64) * stride_z
                + offs_y.to(tl.int64)[None, :, None] * stride_y
                + offs_x.to(tl.int64)[None, None, :] * stride_x)
    tl.store(fnew_ptr + dst_offs, f_in, mask=rw_mask)


@triton.jit
def _fused_collide_stream_kernel(
    f_ptr, fnew_ptr,
    cxi_ptr, cyi_ptr, czi_ptr,
    cxf_ptr, cyf_ptr, czf_ptr, w_ptr,
    tau_inv,
    nz, ny, nx,
    stride_q, stride_z, stride_y, stride_x,
    Q_PAD: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    """Periodic pull-stream + BGK collide in one kernel launch."""
    pid_y = tl.program_id(0)
    pid_x = tl.program_id(1)
    pid_z = tl.program_id(2)

    offs_y = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    offs_x = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    offs_q = tl.arange(0, Q_PAD)

    mask_q = offs_q < 19
    spatial_mask = (offs_y < ny)[:, None] & (offs_x < nx)[None, :]
    rw_mask = mask_q[:, None, None] & spatial_mask[None, :, :]

    cx_i = tl.load(cxi_ptr + offs_q, mask=mask_q, other=0)[:, None, None]
    cy_i = tl.load(cyi_ptr + offs_q, mask=mask_q, other=0)[:, None, None]
    cz_i = tl.load(czi_ptr + offs_q, mask=mask_q, other=0)[:, None, None]

    # --- 1. pull-stream (single vectorised gather) ---
    src_z = pid_z - cz_i
    src_z = tl.where(src_z < 0, src_z + nz, tl.where(src_z >= nz, src_z - nz, src_z))
    src_y = offs_y[None, :, None] - cy_i
    src_y = tl.where(src_y < 0, src_y + ny, tl.where(src_y >= ny, src_y - ny, src_y))
    src_x = offs_x[None, None, :] - cx_i
    src_x = tl.where(src_x < 0, src_x + nx, tl.where(src_x >= nx, src_x - nx, src_x))

    src_offs = (offs_q.to(tl.int64)[:, None, None] * stride_q
                + src_z.to(tl.int64) * stride_z
                + src_y.to(tl.int64) * stride_y
                + src_x.to(tl.int64) * stride_x)
    f_in = tl.load(f_ptr + src_offs, mask=rw_mask, other=0.0)

    # --- 2. BGK collision ---
    cx_b = tl.load(cxf_ptr + offs_q, mask=mask_q, other=0.0)[:, None, None]
    cy_b = tl.load(cyf_ptr + offs_q, mask=mask_q, other=0.0)[:, None, None]
    cz_b = tl.load(czf_ptr + offs_q, mask=mask_q, other=0.0)[:, None, None]
    w_b = tl.load(w_ptr + offs_q, mask=mask_q, other=0.0)[:, None, None]

    rho = tl.sum(f_in, axis=0)
    rho_safe = tl.where(rho > 1e-12, rho, 1e-12)
    ux = tl.sum(cx_b * f_in, axis=0) / rho_safe
    uy = tl.sum(cy_b * f_in, axis=0) / rho_safe
    uz = tl.sum(cz_b * f_in, axis=0) / rho_safe
    usq = ux * ux + uy * uy + uz * uz

    cu = cx_b * ux[None, :, :] + cy_b * uy[None, :, :] + cz_b * uz[None, :, :]
    feq = (rho_safe[None, :, :] * w_b
           * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None, :, :]))
    f_post = f_in - tau_inv * (f_in - feq)

    # --- 3. store at own cell ---
    dst_offs = (offs_q.to(tl.int64)[:, None, None] * stride_q
                + pid_z.to(tl.int64) * stride_z
                + offs_y.to(tl.int64)[None, :, None] * stride_y
                + offs_x.to(tl.int64)[None, None, :] * stride_x)
    tl.store(fnew_ptr + dst_offs, f_post, mask=rw_mask)


@triton.jit
def _fused_collide_stream_anydtype(
    f_ptr, fnew_ptr,
    cxi_ptr, cyi_ptr, czi_ptr,
    cxf_ptr, cyf_ptr, czf_ptr, w_ptr,
    tau_inv,
    nz, ny, nx,
    stride_q, stride_z, stride_y, stride_x,
    Q_PAD: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_Y: tl.constexpr,
):
    """Variant whose storage dtype can be fp16/bf16 (compute stays fp32).

    Load is widened to fp32 for the collision math; store is narrowed
    to whatever ``fnew_ptr.dtype.element_ty`` is.  Useful for
    bandwidth-bound regimes where the 2x traffic saving from 16-bit
    storage is worth the ~2e-4 precision cost (see module docstring).
    """
    pid_y = tl.program_id(0)
    pid_x = tl.program_id(1)
    pid_z = tl.program_id(2)

    offs_y = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    offs_x = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    offs_q = tl.arange(0, Q_PAD)

    mask_q = offs_q < 19
    spatial_mask = (offs_y < ny)[:, None] & (offs_x < nx)[None, :]
    rw_mask = mask_q[:, None, None] & spatial_mask[None, :, :]

    cx_i = tl.load(cxi_ptr + offs_q, mask=mask_q, other=0)[:, None, None]
    cy_i = tl.load(cyi_ptr + offs_q, mask=mask_q, other=0)[:, None, None]
    cz_i = tl.load(czi_ptr + offs_q, mask=mask_q, other=0)[:, None, None]

    src_z = pid_z - cz_i
    src_z = tl.where(src_z < 0, src_z + nz, tl.where(src_z >= nz, src_z - nz, src_z))
    src_y = offs_y[None, :, None] - cy_i
    src_y = tl.where(src_y < 0, src_y + ny, tl.where(src_y >= ny, src_y - ny, src_y))
    src_x = offs_x[None, None, :] - cx_i
    src_x = tl.where(src_x < 0, src_x + nx, tl.where(src_x >= nx, src_x - nx, src_x))

    src_offs = (offs_q.to(tl.int64)[:, None, None] * stride_q
                + src_z.to(tl.int64) * stride_z
                + src_y.to(tl.int64) * stride_y
                + src_x.to(tl.int64) * stride_x)
    f_in = tl.load(f_ptr + src_offs, mask=rw_mask, other=0.0).to(tl.float32)

    cx_b = tl.load(cxf_ptr + offs_q, mask=mask_q, other=0.0)[:, None, None]
    cy_b = tl.load(cyf_ptr + offs_q, mask=mask_q, other=0.0)[:, None, None]
    cz_b = tl.load(czf_ptr + offs_q, mask=mask_q, other=0.0)[:, None, None]
    w_b = tl.load(w_ptr + offs_q, mask=mask_q, other=0.0)[:, None, None]

    rho = tl.sum(f_in, axis=0)
    rho_safe = tl.where(rho > 1e-12, rho, 1e-12)
    ux = tl.sum(cx_b * f_in, axis=0) / rho_safe
    uy = tl.sum(cy_b * f_in, axis=0) / rho_safe
    uz = tl.sum(cz_b * f_in, axis=0) / rho_safe
    usq = ux * ux + uy * uy + uz * uz

    cu = cx_b * ux[None, :, :] + cy_b * uy[None, :, :] + cz_b * uz[None, :, :]
    feq = (rho_safe[None, :, :] * w_b
           * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None, :, :]))
    f_post = f_in - tau_inv * (f_in - feq)

    dst_offs = (offs_q.to(tl.int64)[:, None, None] * stride_q
                + pid_z.to(tl.int64) * stride_z
                + offs_y.to(tl.int64)[None, :, None] * stride_y
                + offs_x.to(tl.int64)[None, None, :] * stride_x)
    tl.store(fnew_ptr + dst_offs,
             f_post.to(fnew_ptr.dtype.element_ty), mask=rw_mask)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def _grid(ny: int, nx: int, nz: int, by: int, bx: int):
    return (triton.cdiv(ny, by), triton.cdiv(nx, bx), nz)


def triton_collide(
    f: torch.Tensor,
    tau: float,
    *,
    out: torch.Tensor | None = None,
    block_x: int = DEFAULT_BLOCK_X,
    block_y: int = DEFAULT_BLOCK_Y,
    num_warps: int = DEFAULT_NUM_WARPS,
    num_stages: int = DEFAULT_NUM_STAGES,
) -> torch.Tensor:
    """BGK collide-only via Triton.  Returns ``out`` if given, else a new tensor.

    The shape, device, and dtype of ``f`` and ``out`` must match.
    """
    Q, nz, ny, nx = f.shape
    if out is None:
        out = torch.empty_like(f)
    lat = make_lattice_tensors(str(f.device))
    _collide_only_kernel[_grid(ny, nx, nz, block_y, block_x)](
        f, out,
        lat["cxf"], lat["cyf"], lat["czf"], lat["w"], 1.0 / tau,
        nz, ny, nx,
        f.stride(0), f.stride(1), f.stride(2), f.stride(3),
        Q_PAD=_Q_PAD, BLOCK_X=block_x, BLOCK_Y=block_y,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def triton_stream(
    f: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    block_x: int = DEFAULT_BLOCK_X,
    block_y: int = DEFAULT_BLOCK_Y,
    num_warps: int = DEFAULT_NUM_WARPS,
    num_stages: int = DEFAULT_NUM_STAGES,
) -> torch.Tensor:
    """Periodic pull-stream via Triton.  Bit-exact with ``torch.roll``.

    Note: writes into ``out`` (or a new tensor).  The input ``f`` is
    not modified.  Mirrors the API of
    :func:`tensorlbm.perf_solver.stream3d_inplace` so the two paths
    can be swapped without changing call sites.
    """
    Q, nz, ny, nx = f.shape
    if out is None:
        out = torch.empty_like(f)
    lat = make_lattice_tensors(str(f.device))
    _stream_pull_kernel[_grid(ny, nx, nz, block_y, block_x)](
        f, out,
        lat["cxi"], lat["cyi"], lat["czi"],
        nz, ny, nx,
        f.stride(0), f.stride(1), f.stride(2), f.stride(3),
        Q_PAD=_Q_PAD, BLOCK_X=block_x, BLOCK_Y=block_y,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def triton_fused(
    f: torch.Tensor,
    tau: float,
    *,
    out: torch.Tensor | None = None,
    block_x: int = DEFAULT_BLOCK_X,
    block_y: int = DEFAULT_BLOCK_Y,
    num_warps: int = DEFAULT_NUM_WARPS,
    num_stages: int = DEFAULT_NUM_STAGES,
) -> torch.Tensor:
    """Periodic pull-stream + BGK collide in a single kernel launch.

    This is the recommended entry point for performance: same physics
    as a separate :func:`triton_stream` + :func:`triton_collide` call
    but at the memory roofline (76-87% of achievable bandwidth on
    RTX 5090 at n=128..256).

    The storage dtype of ``f`` and ``out`` may be fp16 or bf16; in
    that case the collision is computed in fp32 and the result narrowed
    on store.  See module docstring for accuracy bounds.
    """
    Q, nz, ny, nx = f.shape
    if out is None:
        out = torch.empty_like(f)
    lat = make_lattice_tensors(str(f.device))
    _fused_collide_stream_anydtype[_grid(ny, nx, nz, block_y, block_x)](
        f, out,
        lat["cxi"], lat["cyi"], lat["czi"],
        lat["cxf"], lat["cyf"], lat["czf"], lat["w"], 1.0 / tau,
        nz, ny, nx,
        f.stride(0), f.stride(1), f.stride(2), f.stride(3),
        Q_PAD=_Q_PAD, BLOCK_X=block_x, BLOCK_Y=block_y,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


# ---------------------------------------------------------------------------
# Solver class — drop-in alternative to perf_solver.OptimizedSolver3D for
# the periodic case.
# ---------------------------------------------------------------------------

class TritonFusedSolver3D:
    """Periodic D3Q19 LBM step powered by a single fused Triton kernel.

    Mirrors the surface of
    :class:`tensorlbm.perf_solver.OptimizedSolver3D` so that the two
    can be swapped in any periodic workflow.  Trade-offs:

    ===========  ==========================================  =================
    Property     ``OptimizedSolver3D``                       this class
    ===========  ==========================================  =================
    BC support   walls, far-field, wall function            **periodic only**
    Memory       ``LBMStepBuffer`` ≈ 5.4× f                 2× f (ping-pong)
    Throughput  0.34 GLUPS (n=256, 5090)                    8.6 GLUPS (n=256)
    Lattice      D3Q19 or D3Q27                              D3Q19 only
    Storage      fp32                                        fp32/fp16/bf16
    ===========  ==========================================  =================

    Use :func:`is_available` to check whether this class can run on
    the current host; if it returns False, instantiate
    ``OptimizedSolver3D`` instead.

    :meth:`step` keeps two persistent internal buffers and ping-pongs
    between them: consecutive calls that feed the previous output back
    never alias the kernel's input and output pointers (the pull-stream
    kernel reads neighbouring cells, so an aliased in-place step would
    race across thread blocks).

    Args:
        nz, ny, nx: Grid size in lattice units.
        tau: Relaxation time τ > 0.5.
        device: CUDA device string (e.g. ``"cuda:0"``).  The kernel
            does not run on CPU.
        dtype: Storage dtype of the distribution.  fp32 is the
            default; fp16 and bf16 are supported with reduced
            precision (see module docstring).
        block_x, block_y: Tile shape per program.  Defaults are
            tuned for RTX 5090 / Ada-class GPUs at n=64..512.
        num_warps, num_stages: Triton scheduling hints.
    """

    def __init__(
        self,
        nz: int,
        ny: int,
        nx: int,
        tau: float,
        device: str | torch.device = "cuda:0",
        dtype: torch.dtype = torch.float32,
        *,
        block_x: int = DEFAULT_BLOCK_X,
        block_y: int = DEFAULT_BLOCK_Y,
        num_warps: int = DEFAULT_NUM_WARPS,
        num_stages: int = DEFAULT_NUM_STAGES,
    ) -> None:
        if not is_available():
            raise RuntimeError(
                "TritonFusedSolver3D requires CUDA + triton.  "
                "Check tensorlbm.triton_fused.is_available() first and "
                "fall back to tensorlbm.perf_solver.OptimizedSolver3D if False."
            )
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(
                f"Unsupported dtype {dtype}; use float32, float16, or bfloat16.")

        self.nz, self.ny, self.nx = int(nz), int(ny), int(nx)
        self._Q = _Q                    # lattice size (D3Q19), used by diagnostics
        self.tau = float(tau)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError(
                f"TritonFusedSolver3D requires a CUDA device, got {self.device}.")
        self.dtype = dtype
        self.block_x = block_x
        self.block_y = block_y
        self.num_warps = num_warps
        self.num_stages = num_stages

        # Cache lattice tensors once per device string.
        self._lat = make_lattice_tensors(str(self.device))
        # Ping-pong buffers; step() alternates between them so the fused
        # kernel never reads and writes through the same pointer (blocks
        # pull from neighbouring cells, so aliasing input and output is a
        # cross-block read-after-write race).  Allocated lazily so the
        # caller can shape them on demand.
        self._buf_a: torch.Tensor | None = None
        self._buf_b: torch.Tensor | None = None
        # Reporter step counter (tensorlbm.reporters); see run().
        self._report_step = 0

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, f: torch.Tensor) -> torch.Tensor:
        """Advance the distribution by one periodic BGK time step.

        Args:
            f: Distribution tensor of shape ``(Q, nz, ny, nx)`` with
                ``Q == 19`` and dtype/device matching the solver.

        Returns:
            A fresh tensor holding the post-step distribution.  The
            input ``f`` is not modified.

        The result is one of two persistent internal ping-pong buffers.
        Calls that feed the previous output back alternate buffers, so
        the kernel's input and output pointers are always distinct;
        a first call with a caller-owned ``f`` writes into ``_buf_a``.
        """
        if self._buf_a is None or self._buf_a.shape != f.shape \
                or self._buf_a.dtype != f.dtype \
                or self._buf_b is None or self._buf_b.shape != f.shape \
                or self._buf_b.dtype != f.dtype:
            self._buf_a = torch.empty_like(f)
            self._buf_b = torch.empty_like(f)

        # Output = the internal buffer that is NOT the input tensor.
        # The kernel pulls from neighbouring cells, so in-place stepping
        # (f_ptr == fnew_ptr) would race across thread blocks.
        if f.data_ptr() == self._buf_a.data_ptr():
            out = self._buf_b
        else:
            out = self._buf_a

        return triton_fused(
            f, self.tau,
            out=out,
            block_x=self.block_x, block_y=self.block_y,
            num_warps=self.num_warps, num_stages=self.num_stages,
        )

    # ------------------------------------------------------------------
    # Multi-step driver with the unified reporter hook
    # ------------------------------------------------------------------

    def run(
        self,
        f: torch.Tensor,
        n_steps: int,
        *,
        reporters: Sequence[Reporter] | None = None,
    ) -> tuple[torch.Tensor, list[dict]]:
        """Advance *n_steps* steps, dispatching reporters after each one.

        Same contract as :meth:`tensorlbm.lbm_step.LBMStepExecutor.run`:
        with no reporters this is a plain ``step`` loop (no behaviour
        change); with reporters, a
        :class:`~tensorlbm.reporters.StepContext` is dispatched after
        every completed step and ``ctx.stop = True`` ends the run early.
        Per-step diagnostics are empty dicts — the fused kernel reports
        nothing Python-side — kept for API symmetry with the executor.

        The step counter persists across ``run()`` calls so reporter
        intervals and export step labels stay monotonic.
        """
        if not reporters:
            for _ in range(n_steps):
                f = self.step(f)
            self._report_step = getattr(self, "_report_step", 0) + int(n_steps)
            return f, [{} for _ in range(n_steps)]

        from .reporters import StepContext, dispatch

        ctx = StepContext(
            step=getattr(self, "_report_step", 0),
            f=f,
            lattice="D3Q19",
            num_cells=self.grid_size_cells(),
            num_steps=int(n_steps),
        )
        step = ctx.step
        diags: list[dict] = []
        for _ in range(n_steps):
            f = self.step(f)
            step += 1
            diags.append({})
            ctx.step = step
            ctx.f = f
            ctx.num_steps = n_steps
            ctx.stop = False
            dispatch(ctx, reporters)
            if ctx.stop:
                break
        self._report_step = step
        return f, diags

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def grid_size_cells(self) -> int:
        return self.nz * self.ny * self.nx

    def transient_memory_bytes(self) -> int:
        """Bytes of transient memory used per step (two buffers total)."""
        return 2 * self.grid_size_cells() * self._Q * self.dtype.itemsize

    def benchmark(
        self,
        n_steps: int = 100,
        warmup: int = 5,
    ) -> float:
        """Run ``n_steps`` self-timing calls and return seconds per step.

        Useful for picking a tile config; the result is the time per
        :meth:`step` invocation including the Python-side overhead.
        """
        f = torch.zeros(
            _Q, self.nz, self.ny, self.nx,
            dtype=self.dtype, device=self.device,
        )
        # Need non-trivial content so the compiler doesn't optimise the
        # kernel body away.
        f += 1e-3 * torch.randn_like(f)

        for _ in range(warmup):
            self.step(f)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            self.step(f)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_steps