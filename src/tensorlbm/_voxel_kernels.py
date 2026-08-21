"""GPU kernels for STL voxelisation (module-private; hard-requires Triton).

Import this module lazily from CPU-only environments — see
:func:`tensorlbm.geometry_voxel.voxelize_stl`, which falls back to the
pure-torch reference path when this import fails.

Two kernels implement textbook ray-triangle geometry (no external code
was consulted — the licence terms of the usual GPU-LBM references forbid
copying, so everything below is written from first principles):

* **Solid mask (ray parity).**  Every grid cell casts a ray along +x from
  its centre.  A cell is inside the closed surface iff the number of
  triangle crossings strictly ahead of it is odd.  Per program a
  ``(BLOCK, TRI_CHUNK)`` tile of ray x triangle tests is evaluated with
  projected edge functions in the (y, z) plane; triangles whose plane
  contains the ray direction (zero x-component of the face normal) are
  masked out because the projected triangle is degenerate.
* **Link q (Möller–Trumbore).**  For one D3Q19 lattice direction per
  launch, each fluid boundary link is intersected with every triangle,
  using the *unnormalised* lattice velocity as ray direction so the
  intersection parameter is directly the Bouzidi fraction ``q in (0, 1]``
  (the convention of
  :func:`tensorlbm.bfl_d3q19.compute_q_cylinder_d3q19`).

All arithmetic is fp32 (the lattice storage precision).  The caller is
responsible for culling triangles that cannot contribute and for the
sub-cell ray-origin perturbation documented in
:mod:`tensorlbm.geometry_voxel`.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = [
    "link_q_min_t",
    "parity_solid_mask",
    "DEFAULT_BLOCK",
    "DEFAULT_TRI_CHUNK",
    "DEFAULT_NUM_WARPS",
]

# (cells-per-program, triangles-per-chunk) tile shapes.  Chosen so a
# (BLOCK, TRI_CHUNK) fp32 tile stays comfortably within register budget.
DEFAULT_BLOCK: int = 256
DEFAULT_TRI_CHUNK: int = 32
DEFAULT_NUM_WARPS: int = 4

# Barycentric / parameter tolerances (fp32, coordinates of order 1e2).
# Instantiated as tl.constexpr so the kernels can read them as globals.
_BARY_EPS = tl.constexpr(1.0e-6)
_T_MIN = tl.constexpr(1.0e-6)
_T_MAX_PAD = tl.constexpr(1.0e-6)
_PARALLEL_EPS = tl.constexpr(1.0e-12)


@triton.jit
def _solid_parity_kernel(
    tri_ptr,  # (T, 9) fp32, contiguous: v0x v0y v0z v1x ... v2z
    out_ptr,  # (nz, ny, nx) int8 scratch, 1 = solid
    n_tri,
    ny,
    nx,
    x0,
    y0,
    z0,
    dx,
    dy,
    dz,
    stride_z,
    stride_y,
    TRI_CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Accumulate +x-ray crossing parity for a tile of rays per program.

    One program covers ``BLOCK`` rays of a single z-layer; triangles are
    streamed in ``TRI_CHUNK`` chunks (cached in shared memory by the
    compiler's load pipeline).
    """
    pid_b = tl.program_id(0)
    pid_z = tl.program_id(1)

    offs = pid_b * BLOCK + tl.arange(0, BLOCK)
    ray_ok = offs < ny * nx
    j = offs // nx
    i = offs - j * nx

    ox = x0 + (i.to(tl.float32) + 0.5) * dx
    oy = y0 + (j.to(tl.float32) + 0.5) * dy
    oz = z0 + (pid_z.to(tl.float32) + 0.5) * dz

    count = tl.zeros((BLOCK,), dtype=tl.int32)

    for t0 in range(0, n_tri, TRI_CHUNK):
        tidx = t0 + tl.arange(0, TRI_CHUNK)
        tmask = tidx < n_tri
        base = tidx * 9
        v0x = tl.load(tri_ptr + base + 0, mask=tmask, other=0.0)
        v0y = tl.load(tri_ptr + base + 1, mask=tmask, other=0.0)
        v0z = tl.load(tri_ptr + base + 2, mask=tmask, other=0.0)
        v1x = tl.load(tri_ptr + base + 3, mask=tmask, other=0.0)
        v1y = tl.load(tri_ptr + base + 4, mask=tmask, other=0.0)
        v1z = tl.load(tri_ptr + base + 5, mask=tmask, other=0.0)
        v2x = tl.load(tri_ptr + base + 6, mask=tmask, other=0.0)
        v2y = tl.load(tri_ptr + base + 7, mask=tmask, other=0.0)
        v2z = tl.load(tri_ptr + base + 8, mask=tmask, other=0.0)

        # Edge functions of the (y, z)-projected triangle at (oy, oz).
        py0 = oy[:, None] - v0y[None, :]
        pz0 = oz - v0z[None, :]
        d0 = (v1y - v0y)[None, :] * pz0 - (v1z - v0z)[None, :] * py0
        py1 = oy[:, None] - v1y[None, :]
        pz1 = oz - v1z[None, :]
        d1 = (v2y - v1y)[None, :] * pz1 - (v2z - v1z)[None, :] * py1
        py2 = oy[:, None] - v2y[None, :]
        pz2 = oz - v2z[None, :]
        d2 = (v0y - v2y)[None, :] * pz2 - (v0z - v2z)[None, :] * py2

        has_neg = (d0 < 0.0) | (d1 < 0.0) | (d2 < 0.0)
        has_pos = (d0 > 0.0) | (d1 > 0.0) | (d2 > 0.0)
        inside = ~(has_neg & has_pos)

        nx_n = (v1y - v0y) * (v2z - v0z) - (v1z - v0z) * (v2y - v0y)
        ny_n = (v1z - v0z) * (v2x - v0x) - (v1x - v0x) * (v2z - v0z)
        nz_n = (v1x - v0x) * (v2y - v0y) - (v1y - v0y) * (v2x - v0x)
        ok = (tl.abs(nx_n) > _PARALLEL_EPS) & tmask
        safe_nx = tl.where(ok, nx_n, 1.0)

        # Plane equation n . (P - v0) = 0 with P = (t, oy, oz).
        x_isect = v0x[None, :] - (ny_n[None, :] * py0 + nz_n[None, :] * pz0) / safe_nx[None, :]

        hit = inside & ok[None, :] & (x_isect > ox[:, None])
        count += tl.sum(hit.to(tl.int32), axis=1)

    solid = (count % 2) == 1
    out_offs = pid_z.to(tl.int64) * stride_z + j.to(tl.int64) * stride_y + i.to(tl.int64)
    tl.store(out_ptr + out_offs, solid.to(tl.int8), mask=ray_ok)


def parity_solid_mask(
    tri9: torch.Tensor,
    shape: tuple[int, int, int],
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    *,
    block: int = DEFAULT_BLOCK,
    tri_chunk: int = DEFAULT_TRI_CHUNK,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> torch.Tensor:
    """Launch :func:`_solid_parity_kernel`; return a bool solid mask.

    Args:
        tri9: ``(T, 9)`` fp32 contiguous triangle table on a CUDA device
            (already culled and with the parity ray-origin perturbation
            applied to ``origin`` by the caller).
        shape: Grid shape ``(nz, ny, nx)``.
        origin: Physical lower corner ``(x0, y0, z0)`` of the grid.
        spacing: Cell sizes ``(dx, dy, dz)``.

    Returns:
        Bool tensor of ``shape``, True inside the closed surface.
    """
    nz, ny, nx = shape
    x0, y0, z0 = (float(v) for v in origin)
    dx, dy, dz = (float(v) for v in spacing)
    scratch = torch.empty((nz, ny, nx), dtype=torch.int8, device=tri9.device)
    grid = (triton.cdiv(ny * nx, block), nz)
    _solid_parity_kernel[grid](
        tri9,
        scratch,
        tri9.shape[0],
        ny,
        nx,
        x0,
        y0,
        z0,
        dx,
        dy,
        dz,
        scratch.stride(0),
        scratch.stride(1),
        TRI_CHUNK=tri_chunk,
        BLOCK=block,
        num_warps=num_warps,
    )
    return scratch != 0


@triton.jit
def _link_q_kernel(
    tri_ptr,
    q_ptr,  # (N,) fp32, filled with the min-t sentinel 2.0 when no hit
    jj_ptr,
    ii_ptr,
    kk_ptr,
    n_tri,
    n_cells,
    cxx,
    cyy,
    czz,
    x0,
    y0,
    z0,
    dx,
    dy,
    dz,
    TRI_CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Minimum Möller–Trumbore t over all triangles, per boundary link.

    The ray direction is the *physical* lattice velocity
    ``(dx*cx, dy*cy, dz*cz)`` passed via ``cxx, cyy, czz``, so ``t`` is
    the Bouzidi fraction of the link directly for any spacing.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    inb = offs < n_cells

    j = tl.load(jj_ptr + offs, mask=inb, other=0)
    i = tl.load(ii_ptr + offs, mask=inb, other=0)
    k = tl.load(kk_ptr + offs, mask=inb, other=0)

    ox = x0 + (i.to(tl.float32) + 0.5) * dx
    oy = y0 + (j.to(tl.float32) + 0.5) * dy
    oz = z0 + (k.to(tl.float32) + 0.5) * dz

    best = tl.full((BLOCK,), 2.0, dtype=tl.float32)

    for t0 in range(0, n_tri, TRI_CHUNK):
        tidx = t0 + tl.arange(0, TRI_CHUNK)
        tmask = tidx < n_tri
        base = tidx * 9
        v0x = tl.load(tri_ptr + base + 0, mask=tmask, other=0.0)
        v0y = tl.load(tri_ptr + base + 1, mask=tmask, other=0.0)
        v0z = tl.load(tri_ptr + base + 2, mask=tmask, other=0.0)
        v1x = tl.load(tri_ptr + base + 3, mask=tmask, other=0.0)
        v1y = tl.load(tri_ptr + base + 4, mask=tmask, other=0.0)
        v1z = tl.load(tri_ptr + base + 5, mask=tmask, other=0.0)
        v2x = tl.load(tri_ptr + base + 6, mask=tmask, other=0.0)
        v2y = tl.load(tri_ptr + base + 7, mask=tmask, other=0.0)
        v2z = tl.load(tri_ptr + base + 8, mask=tmask, other=0.0)

        e1x = v1x - v0x
        e1y = v1y - v0y
        e1z = v1z - v0z
        e2x = v2x - v0x
        e2y = v2y - v0y
        e2z = v2z - v0z

        # h = cross(D, e2) with the constant ray direction D = (cxx, cyy, czz).
        hx = cyy * e2z - czz * e2y
        hy = czz * e2x - cxx * e2z
        hz = cxx * e2y - cyy * e2x
        a = e1x * hx + e1y * hy + e1z * hz

        ok = (tl.abs(a) > _PARALLEL_EPS) & tmask
        safe_a = tl.where(ok, a, 1.0)
        f = tl.where(ok, 1.0 / safe_a, 0.0)

        sx = ox[:, None] - v0x[None, :]
        sy = oy[:, None] - v0y[None, :]
        sz = oz[:, None] - v0z[None, :]

        u = f[None, :] * (sx * hx[None, :] + sy * hy[None, :] + sz * hz[None, :])

        qx = sy * e1z[None, :] - sz * e1y[None, :]
        qy = sz * e1x[None, :] - sx * e1z[None, :]
        qz = sx * e1y[None, :] - sy * e1x[None, :]

        v = f[None, :] * (cxx * qx + cyy * qy + czz * qz)
        t = f[None, :] * (e2x[None, :] * qx + e2y[None, :] * qy + e2z[None, :] * qz)

        hit = (
            ok[None, :]
            & (u >= -_BARY_EPS)
            & (v >= -_BARY_EPS)
            & (u + v <= 1.0 + _BARY_EPS)
            & (t > _T_MIN)
            & (t <= 1.0 + _T_MAX_PAD)
        )
        best = tl.minimum(best, tl.min(tl.where(hit, t, 2.0), axis=1))

    tl.store(q_ptr + offs, best, mask=inb)


def link_q_min_t(
    tri9: torch.Tensor,
    cell_kji: torch.Tensor,
    direction: tuple[int, int, int],
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    *,
    block: int = 128,
    tri_chunk: int = 16,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> torch.Tensor:
    """Launch :func:`_link_q_kernel` for one lattice direction.

    Args:
        tri9: ``(T, 9)`` fp32 contiguous triangle table on a CUDA device.
        cell_kji: ``(N, 3)`` int tensor of boundary cell indices
            ``(k, j, i)`` (tensor-layout order, streamwise x last).
        direction: Integer lattice velocity ``(cx, cy, cz)``.
        origin: Physical lower corner ``(x0, y0, z0)``.
        spacing: Cell sizes ``(dx, dy, dz)``.

    Returns:
        ``(N,)`` fp32 tensor of minimum link fractions ``t in (0, 1]``,
        or the sentinel ``2.0`` where no triangle intersects the link.
    """
    cxx, cyy, czz = (int(v) for v in direction)
    x0, y0, z0 = (float(v) for v in origin)
    dx, dy, dz = (float(v) for v in spacing)
    # Physical lattice velocity: the link spans [x, x + (dx*cx, dy*cy, dz*cz)]
    # so the MT parameter t is the Bouzidi fraction for any spacing.
    dir_x = float(cxx) * dx
    dir_y = float(cyy) * dy
    dir_z = float(czz) * dz
    n_cells = cell_kji.shape[0]
    out = torch.empty((n_cells,), dtype=torch.float32, device=tri9.device)
    if n_cells == 0:
        return out
    idx = cell_kji.to(dtype=torch.int32).contiguous()
    grid = (triton.cdiv(n_cells, block),)
    _link_q_kernel[grid](
        tri9,
        out,
        idx[:, 1].contiguous(),
        idx[:, 2].contiguous(),
        idx[:, 0].contiguous(),
        tri9.shape[0],
        n_cells,
        dir_x,
        dir_y,
        dir_z,
        x0,
        y0,
        z0,
        dx,
        dy,
        dz,
        TRI_CHUNK=tri_chunk,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out
