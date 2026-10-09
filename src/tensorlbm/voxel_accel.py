"""Spatially accelerated STL voxelisation: uniform triangle-bin hash grid.

:mod:`tensorlbm.geometry_voxel` evaluates its ray-parity solid mask and
its Möller–Trumbore link q-field against **every** triangle of the mesh:
``O(rays x triangles)`` and ``O(links x triangles)``.  A 10^6-triangle
 hull makes that unusable (hours on CPU, minutes on a single GPU).  This
module adds a *uniform spatial hash grid* over triangle axis-aligned
bounding boxes and keeps only the per-ray / per-link candidates it stores,
while reproducing the reference results **bit-for-bit**.

Why a uniform grid (and not a BVH)
----------------------------------
The dominant query is an axis-aligned +x ray cast from every lattice
cell centre — for that pattern a 2-D grid over the ``(y, z)`` columns is
exact: a triangle can only be crossed by the ray of column ``(k, j)``
if the ray point lies inside the triangle's ``(y, z)`` bounding box, so
binning triangles by that AABB and querying one bin per ray loses
nothing.  The q-field links span at most one cell per axis, so their
candidates come from the ``<= 2x2x2`` block of a 3-D cell grid.  Both
structures are CSR arrays — fully vectorisable in torch (``repeat_interleave``
/ ragged ranges) and directly consumable by a Triton kernel, unlike the
pointer-chasing traversal a BVH would need.

Bitwise parity with the brute-force reference
---------------------------------------------
The candidate sets are conservative supersets of the hitting sets:

* the acceptance tests of the reference (edge-function ``inside`` sign
  test, Möller–Trumbore ``u/v/t`` tolerances) place every accepted pair
  within the triangle's AABB up to floating-point dust, and every bin
  query pads the AABBs by ``pad = 1e-5 * bbox_diagonal + 1e-6`` — five
  orders of magnitude above the fp64/fp32 rounding excursions;
* per-pair arithmetic is *copied verbatim* from the reference (same
  expressions, same operand order, same fp64 dtype), so each candidate's
  ``x_isect`` / Möller–Trumbore ``t`` is bit-identical whether computed
  alone or inside a brute-force tile;
* the reductions are exact: crossing **counts** are integers and the
  link **min-t** is an order-invariant minimum, so reducing over a
  superset of the hitting pairs yields the same value as reducing over
  all triangles (non-hits contribute the ``2.0`` sentinel / zero).

Hence ``torch.equal``-level equality of ``solid_mask``, boundary masks
and ``q_field`` with the brute-force reference (verified in
``tests/test_voxel_accel.py``), and the accelerated path is only ever
*more* inclusive: an over-filled bin can never change a parity count
because the pair test itself decides.

The opt-in Triton variants (``use_triton=True``) re-use the same bins
inside :mod:`tensorlbm._voxel_kernels`; their solid and boundary masks
match the brute-force Triton kernels bit-for-bit, while ``q`` agrees to
fp32 rounding dust (max 2.9e-6 measured over every 20k–1.3M-face x
34^3–128^3 combination, worst on the coarsest meshes; zero entries
beyond 1e-3, no hit/miss or sentinel flips) because the two kernel tile
shapes compile to slightly different FMA contraction — quantified in
``docs/benchmarks/voxel_accel_benchmark.md``.

Public API
----------

* :func:`build_column_bins` / :func:`build_cell_bins` — CSR candidate
  grids (2-D columns for the parity pass, 3-D cells for the q pass);
* :func:`solid_mask_parity_accelerated` / :func:`q_field_accelerated` —
  accelerated counterparts of the reference functions;
* :func:`voxelize_stl_accelerated` — accelerated mirror of
  ``voxelize_stl_reference`` (wired into
  :func:`tensorlbm.geometry_voxel.voxelize_stl` via ``accelerate=True``).

Benchmark numbers (10^6-triangle icosphere target) are collected in
``docs/benchmarks/voxel_accel_benchmark.md``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .d3q19 import C as _C19
from .geometry_voxel import (
    _REF_BARY_EPS,
    _REF_T_MAX_PAD,
    _REF_T_MIN,
    _as_triangle_tensor,
    _boundary_scaffold,
    _cull_triangles,
    _parity_origin,
    _shift_solid,
    _validate_grid,
)

__all__ = [
    "TriangleBins",
    "build_cell_bins",
    "build_column_bins",
    "q_field_accelerated",
    "solid_mask_parity_accelerated",
    "voxelize_stl_accelerated",
]

# AABB padding for bin insertion: relative to the mesh bounding-box
# diagonal (fp64 reference dust is ~1e-16 * diag, the fp32 Triton path's
# barycentric tolerance excursions are <= 1e-6 * edge length <= 1e-6 * diag;
# 1e-5 * diag + 1e-6 covers both with an order of magnitude to spare).
_PAD_REL: float = 1.0e-5
_PAD_ABS: float = 1.0e-6

# Ragged pair budget (pairs per tile) — bounds transient fp64 memory to
# roughly budget * ~25 arrays * 8 bytes.
_DEFAULT_PAIR_BUDGET: int = 2_000_000


def _mesh_pad(tri: torch.Tensor) -> float:
    """Conservative AABB padding (module docstring) for a triangle table."""
    if tri.shape[0] == 0:
        return _PAD_ABS
    vmin = tri.amin(dim=(0, 1))
    vmax = tri.amax(dim=(0, 1))
    diag = float((vmax - vmin).norm().item())
    return _PAD_REL * diag + _PAD_ABS


@dataclass(frozen=True)
class TriangleBins:
    """CSR uniform grid of per-bin candidate triangle lists.

    Attributes:
        offsets: ``(n_bins + 1,)`` int64 — candidates of bin ``b`` are
            ``entries[offsets[b] : offsets[b + 1]]``.
        entries: ``(nnz,)`` int64 triangle row indices, sorted by bin
            (stable, so deterministic for a given mesh).
        n_bins: Bin counts per grid axis (physical axis order as built).
        axes: Physical axis indices (0=x, 1=y, 2=z) of the binned axes.
        pad: AABB padding used (mesh units).
    """

    offsets: torch.Tensor
    entries: torch.Tensor
    n_bins: tuple[int, ...]
    axes: tuple[int, ...]
    pad: float

    @property
    def nnz(self) -> int:
        return int(self.entries.shape[0])

    @property
    def memory_bytes(self) -> int:
        return int(self.offsets.nbytes + self.entries.nbytes)

    def lengths(self) -> torch.Tensor:
        """Per-bin candidate counts, shape ``(prod(n_bins),)``."""
        return self.offsets[1:] - self.offsets[:-1]


def _build_bins(
    tri: torch.Tensor,
    axes: tuple[tuple[int, int, float, float], ...],
    pad: float,
) -> TriangleBins:
    """Build a CSR triangle-bin grid over the given physical axes.

    Args:
        tri: ``(T, 3, 3)`` fp64 triangle table (any device).
        axes: One tuple ``(axis, n_bins, origin, spacing)`` per binned
            axis; the first entry is the slowest bin dimension.
        pad: AABB padding in mesh units (see :data:`_PAD_REL`).

    Returns:
        :class:`TriangleBins` with ``prod(n_bins)`` bins.
    """
    device = tri.device
    n_tri = tri.shape[0]
    nbins_total = 1
    for _, n, _, _ in axes:
        nbins_total *= n
    if n_tri == 0:
        empty_off = torch.zeros(nbins_total + 1, dtype=torch.int64, device=device)
        empty_ent = torch.zeros(0, dtype=torch.int64, device=device)
        return TriangleBins(
            empty_off,
            empty_ent,
            tuple(n for _, n, _, _ in axes),
            tuple(a for a, _, _, _ in axes),
            pad,
        )
    vmin = tri.amin(dim=1)  # (T, 3)
    vmax = tri.amax(dim=1)

    los: list[torch.Tensor] = []
    spans: list[torch.Tensor] = []
    for ax, n, o, h in axes:
        lo = torch.div(vmin[:, ax] - o - pad, h, rounding_mode="floor").to(torch.int64)
        hi = torch.div(vmax[:, ax] - o + pad, h, rounding_mode="floor").to(torch.int64)
        # Clamp so that fully-outside triangles get an empty span (hi < lo)
        # while straddling ones reach the edge bins.  Over-inclusion is
        # harmless for parity (extra candidates simply fail the pair
        # test); under-inclusion is what the padding prevents.
        lo = lo.clamp(0, n - 1)
        hi = hi.clamp(-1, n - 1)
        los.append(lo)
        spans.append((hi - lo + 1).clamp(min=0))

    lens = spans[0]
    for s in spans[1:]:
        lens = lens * s
    total = int(lens.sum())
    if total == 0:
        empty_off = torch.zeros(nbins_total + 1, dtype=torch.int64, device=device)
        empty_ent = torch.zeros(0, dtype=torch.int64, device=device)
        return TriangleBins(
            empty_off,
            empty_ent,
            tuple(n for _, n, _, _ in axes),
            tuple(a for a, _, _, _ in axes),
            pad,
        )

    tri_of = torch.repeat_interleave(torch.arange(n_tri, device=device), lens)
    starts = torch.zeros_like(lens)
    starts[1:] = lens.cumsum(0)[:-1]
    w = torch.arange(total, device=device) - starts[tri_of]

    # Mixed-radix decomposition of w into per-axis bin digits.  For the
    # entries of item t the digit bases are the per-triangle spans, so
    # the radix stride of axis a is prod(spans of the faster axes, t);
    # `stride` holds exactly that product when the loop reaches axis a.
    # The linear bin id uses the *global* axis extents instead.
    strides_axis: list[int] = []
    trailing = 1
    for a in range(len(axes) - 1, -1, -1):
        strides_axis.append(trailing)
        trailing *= axes[a][1]
    strides_axis.reverse()  # strides_axis[a] = prod(n_bins of axes after a)

    bin_id = torch.zeros(total, dtype=torch.int64, device=device)
    stride = torch.ones_like(lens)
    for a in range(len(axes) - 1, -1, -1):
        digit = torch.div(w, stride[tri_of], rounding_mode="floor") % spans[a][tri_of]
        bin_id += (los[a][tri_of] + digit) * strides_axis[a]
        stride = stride * spans[a]

    counts = torch.bincount(bin_id, minlength=nbins_total)
    offsets = F.pad(counts.cumsum(0), (1, 0))
    order = torch.argsort(bin_id, stable=True)
    entries = tri_of[order]
    return TriangleBins(
        offsets, entries, tuple(n for _, n, _, _ in axes), tuple(a for a, _, _, _ in axes), pad
    )


def build_column_bins(
    triangles,
    shape,
    *,
    origin=None,
    spacing=None,
    pad: float | None = None,
) -> TriangleBins:
    """Bin triangles by their ``(y, z)`` AABB for +x ray-parity queries.

    Args:
        triangles: ``(T, 3, 3)`` triangle table (array-like or tensor).
        shape: Grid shape ``(nz, ny, nx)``.
        origin: Physical lower corner ``(x0, y0, z0)`` (default zero).
        spacing: Cell sizes ``(dx, dy, dz)`` (default one).
        pad: AABB padding (default: ``1e-5 * diag + 1e-6``).

    Returns:
        :class:`TriangleBins` with ``ny * nz`` bins; bin
        ``(j, k)`` (linear id ``j * nz + k``) holds every triangle whose
        padded ``(y, z)`` AABB overlaps column cell
        ``[y0 + j dy, y0 + (j+1) dy) x [z0 + k dz, z0 + (k+1) dz)``.
    """
    dims, org, spc = _validate_grid(shape, origin, spacing)
    _, ny, nz = dims
    tri = _as_triangle_tensor(triangles).to(dtype=torch.float64)
    if pad is None:
        pad = _mesh_pad(tri)
    return _build_bins(
        tri,
        (
            (1, ny, float(org[1]), float(spc[1])),
            (2, nz, float(org[2]), float(spc[2])),
        ),
        pad,
    )


def build_cell_bins(
    triangles,
    shape,
    *,
    origin=None,
    spacing=None,
    pad: float | None = None,
) -> TriangleBins:
    """Bin triangles by their 3-D AABB into lattice cells.

    Used by the link q-field pass: a D3Q19 link spans at most one cell
    per axis, so its candidates live in the ``<= 2 x 2 x 2`` block of
    cells anchored at the link's fluid node.

    Args:
        triangles: ``(T, 3, 3)`` triangle table (array-like or tensor).
        shape: Grid shape ``(nz, ny, nx)``.
        origin: Physical lower corner ``(x0, y0, z0)`` (default zero).
        spacing: Cell sizes ``(dx, dy, dz)`` (default one).
        pad: AABB padding (default: ``1e-5 * diag + 1e-6``).

    Returns:
        :class:`TriangleBins` with ``nx * ny * nz`` bins; linear bin id
        ``(i * ny + j) * nz + k``.
    """
    dims, org, spc = _validate_grid(shape, origin, spacing)
    nx, ny, nz = dims[2], dims[1], dims[0]
    tri = _as_triangle_tensor(triangles).to(dtype=torch.float64)
    if pad is None:
        pad = _mesh_pad(tri)
    return _build_bins(
        tri,
        (
            (0, nx, float(org[0]), float(spc[0])),
            (1, ny, float(org[1]), float(spc[1])),
            (2, nz, float(org[2]), float(spc[2])),
        ),
        pad,
    )


def _ragged_pairs(lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Explode per-item lengths into ``(item_of_pair, inner_offset)``."""
    device = lens.device
    total = int(lens.sum())
    item = torch.repeat_interleave(torch.arange(lens.shape[0], device=device), lens)
    if total == 0:
        return item, torch.zeros(0, dtype=torch.int64, device=device)
    starts = torch.zeros_like(lens)
    starts[1:] = lens.cumsum(0)[:-1]
    inner = torch.arange(total, device=device) - starts[item]
    return item, inner


def _link_candidate_csr(
    bins: TriangleBins,
    slot_ids: list[torch.Tensor],
    link_lens: torch.Tensor,
    *,
    pair_budget: int = _DEFAULT_PAIR_BUDGET,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compact per-link candidate lists into CSR arrays for Triton kernels.

    ``slot_ids`` holds, per link, the linear bin ids of its ``2 x 2 x 2``
    block (one tensor per slot); the output lists concatenate the slots'
    CSR segments per link.  Cross-slot duplicates remain — harmless for
    the order-invariant min reduction of the q pass.

    Returns:
        ``(starts, ends, entries)`` — ``(N,)`` int64 CSR bounds and the
        ``(nnz,)`` int64 triangle ids grouped by link.
    """
    device = link_lens.device
    n_links = link_lens.shape[0]
    starts = torch.zeros(n_links, dtype=torch.int64, device=device)
    starts[1:] = link_lens.cumsum(0)[:-1]
    ends = starts + link_lens
    total = int(link_lens.sum())
    entries = torch.zeros(0, dtype=torch.int64, device=device)
    if total == 0:
        return starts, ends, entries

    bin_len = bins.lengths()
    sid_mat = torch.stack(slot_ids, dim=1)  # (n_links, n_slots)
    lens_mat = bin_len[sid_mat]
    slot_prefix = torch.zeros_like(lens_mat)
    slot_prefix[:, 1:] = lens_mat.cumsum(1)[:, :-1]

    entries = torch.empty(total, dtype=torch.int64, device=device)
    for lo, hi in _pair_blocks(link_lens, pair_budget):
        blk_lens = lens_mat[lo:hi].reshape(-1)
        blk_of_pair, inner = _ragged_pairs(blk_lens)
        link_local = lo + torch.div(blk_of_pair, len(slot_ids), rounding_mode="floor")
        slot_local = blk_of_pair % len(slot_ids)
        pos = starts[link_local] + slot_prefix[link_local, slot_local] + inner
        sid = sid_mat[link_local, slot_local]
        entries[pos] = bins.entries[bins.offsets[sid] + inner]
    return starts, ends, entries


def _pair_blocks(lens: torch.Tensor, budget: int):
    """Yield ``(lo, hi)`` item ranges whose pair totals stay near *budget*.

    Always yields at least one item per block, so a single item heavier
    than *budget* is still processed in its own block.
    """
    n = int(lens.shape[0])
    if n == 0:
        return
    cum = torch.zeros(n + 1, dtype=torch.int64, device=lens.device)
    cum[1:] = lens.to(torch.int64).cumsum(0)
    lo = 0
    while lo < n:
        target = int(cum[lo].item()) + int(budget)
        hi = int(torch.searchsorted(cum, target).item())
        hi = min(max(hi, lo + 1), n)
        yield lo, hi
        lo = hi


# ---------------------------------------------------------------------------
# 1. SOLID MASK (ray parity), accelerated
# ---------------------------------------------------------------------------


def _load_kernels():
    """Import the GPU kernel module or return None (Triton unavailable)."""
    try:
        from . import _voxel_kernels
    except ImportError:  # pragma: no cover - depends on the environment
        return None
    return _voxel_kernels


def _wants_triton(use_triton: bool | None, dev: torch.device) -> bool:
    """Resolve the ``use_triton`` default.

    The default (``None``) selects the pure-torch fp64 path on *every*
    device: it is bit-identical to the brute-force reference and, as
    measured (``docs/benchmarks/voxel_accel_benchmark.md``), also faster
    than the binned Triton kernels at lattice-scale grids, so the Triton
    variant is strictly opt-in.
    """
    return bool(use_triton)


def solid_mask_parity_accelerated(
    triangles,
    shape,
    *,
    origin=None,
    spacing=None,
    device: str | torch.device | None = None,
    pad: float | None = None,
    pair_budget: int = _DEFAULT_PAIR_BUDGET,
    use_triton: bool | None = None,
) -> torch.Tensor:
    """Bit-exact accelerated counterpart of ``solid_mask_parity_reference``.

    The rays of a column ``(k, j)`` all share one candidate list (the
    triangles binned into that column, :func:`build_column_bins`), and the
    per-candidate crossing position ``x_isect`` is independent of the ray
    index ``i`` — only the comparison ``x_isect > ox_i`` varies per ray.
    The pass therefore runs in two phases:

    1. per column-candidate pair, compute the reference's edge-function
       ``inside`` test and ``x_isect`` (verbatim fp64 arithmetic);
    2. per column, count how many kept candidates lie ahead of each ray
       (exact ``>`` comparisons against the identical ``ox`` grid) and
       take the parity.

    Both reductions are exact, so the returned mask satisfies
    ``torch.equal(mask, solid_mask_parity_reference(...))``.

    With ``use_triton=True`` on a CUDA device the same column bins feed
    the binned Triton kernel
    (:func:`tensorlbm._voxel_kernels.parity_solid_mask_binned`), whose
    fp32 arithmetic is identical per pair to the brute-force Triton
    kernel — the mask then matches *that* path bit-for-bit (the default
    remains the fp64 torch path, which matches the reference).

    Args:
        triangles: ``(T, 3, 3)`` triangle table (array-like or tensor).
        shape: Grid shape ``(nz, ny, nx)``.
        origin: Physical lower corner ``(x0, y0, z0)`` (default zero).
        spacing: Cell sizes ``(dx, dy, dz)`` (default one).
        device: Target device (default: the triangle tensor's device).
        pad: Bin AABB padding (default: ``1e-5 * diag + 1e-6``).
        pair_budget: Column-candidate pairs per tile (memory knob).
        use_triton: Force the binned Triton kernel (``True``) or the
            pure-torch fp64 path (``False``, the default).

    Returns:
        Bool tensor of ``shape``, True inside the closed surface.
    """
    dims, org, spc = _validate_grid(shape, origin, spacing)
    nz, ny, nx = dims
    tri_in = _as_triangle_tensor(triangles)
    dev = tri_in.device if device is None else torch.device(device)
    if _wants_triton(use_triton, dev):
        kernels = _load_kernels()
        if kernels is None or dev.type != "cuda" or not torch.cuda.is_available():
            msg = "use_triton=True requires CUDA and the GPU kernel module"
            raise RuntimeError(msg)
        tri32_tab = _cull_triangles(
            tri_in.to(device=dev, dtype=torch.float32),
            dims,
            org,
            spc,
            (float("inf"), 0.0, 0.0),
        )
        tri9 = tri32_tab.reshape(-1, 9).contiguous()
        if pad is None:
            pad = _mesh_pad(tri32_tab.to(torch.float64))
        bins = _build_bins(
            tri32_tab.to(torch.float64),
            (
                (1, ny, float(org[1]), float(spc[1])),
                (2, nz, float(org[2]), float(spc[2])),
            ),
            pad,
        )
        return kernels.parity_solid_mask_binned(
            tri9, bins.offsets, bins.entries, dims, _parity_origin(org, spc), spc
        )
    tri = _cull_triangles(
        tri_in.to(device=dev, dtype=torch.float64), dims, org, spc, (float("inf"), 0.0, 0.0)
    )
    if pad is None:
        pad = _mesh_pad(tri)
    bins = _build_bins(
        tri,
        (
            (1, ny, float(org[1]), float(spc[1])),
            (2, nz, float(org[2]), float(spc[2])),
        ),
        pad,
    )

    x0, y0, z0 = _parity_origin(org, spc)
    dx, dy, dz = spc
    # Built exactly like the reference so every comparison operand is
    # bit-identical.
    ox_line = x0 + (torch.arange(nx, device=dev, dtype=torch.float64) + 0.5) * dx
    oy_line = y0 + (torch.arange(ny, device=dev, dtype=torch.float64) + 0.5) * dy

    v0, v1, v2 = tri[:, 0, :], tri[:, 1, :], tri[:, 2, :]
    e1 = v1 - v0
    e2 = v2 - v0
    normal = torch.linalg.cross(e1, e2)
    nx_n, ny_n, nz_n = normal[:, 0], normal[:, 1], normal[:, 2]
    ok = nx_n.abs() > 1.0e-12
    safe_nx = torch.where(ok, nx_n, torch.ones_like(nx_n))
    inv_nx = torch.where(ok, 1.0 / safe_nx, torch.zeros_like(nx_n))
    a0y, a0z = v1[:, 1] - v0[:, 1], v1[:, 2] - v0[:, 2]
    a1y, a1z = v2[:, 1] - v1[:, 1], v2[:, 2] - v1[:, 2]
    a2y, a2z = v0[:, 1] - v2[:, 1], v0[:, 2] - v2[:, 2]

    n_cols = ny * nz
    counts = torch.zeros((n_cols, nx), dtype=torch.int64, device=dev)
    col_lens = bins.lengths()

    # Phase A budget governs fp64 pair tiles; phase B (pure comparisons)
    # gets a boolean budget ~ 16x larger.
    bool_budget = 16 * pair_budget
    for lo, hi in _pair_blocks(col_lens, pair_budget):
        lens_blk = col_lens[lo:hi]
        col_of_pair, inner = _ragged_pairs(lens_blk)
        cols = lo + col_of_pair
        entry_idx = bins.offsets[cols] + inner
        t_idx = bins.entries[entry_idx]

        oy = oy_line[torch.div(cols, nz, rounding_mode="floor")]
        oz = z0 + ((cols % nz).to(torch.float64) + 0.5) * dz
        py0 = oy - v0[t_idx, 1]
        pz0 = oz - v0[t_idx, 2]
        d0 = a0y[t_idx] * pz0 - a0z[t_idx] * py0
        py1 = oy - v1[t_idx, 1]
        pz1 = oz - v1[t_idx, 2]
        d1 = a1y[t_idx] * pz1 - a1z[t_idx] * py1
        py2 = oy - v2[t_idx, 1]
        pz2 = oz - v2[t_idx, 2]
        d2 = a2y[t_idx] * pz2 - a2z[t_idx] * py2
        has_neg = (d0 < 0.0) | (d1 < 0.0) | (d2 < 0.0)
        has_pos = (d0 > 0.0) | (d1 > 0.0) | (d2 > 0.0)
        inside = ~(has_neg & has_pos)
        x_isect = v0[t_idx, 0] - (ny_n[t_idx] * py0 + nz_n[t_idx] * pz0) * inv_nx[t_idx]
        keep = inside & ok[t_idx]
        if not bool(keep.any()):
            continue

        x_kept = x_isect[keep]
        col_kept = col_of_pair[keep]  # pairs are column-major, so still sorted
        kept_lens = torch.bincount(col_kept, minlength=hi - lo)
        cumlens = kept_lens.cumsum(0)
        cmax = max(int(kept_lens.amax().item()), 1)
        sub_n = max(1, bool_budget // max(cmax * nx, 1))
        for blo in range(0, hi - lo, sub_n):
            bhi = min(blo + sub_n, hi - lo)
            lens_b = kept_lens[blo:bhi]
            if not bool((lens_b > 0).any()):
                continue
            row, pos = _ragged_pairs(lens_b)
            flat = row * cmax + pos
            mat = torch.full(((bhi - blo) * cmax,), -float("inf"), dtype=torch.float64, device=dev)
            # Kept pairs of rows [blo, bhi) form the contiguous range
            # [cumlens[blo-1], cumlens[bhi-1]) of the sorted x_kept array.
            p_lo = int(cumlens[blo - 1].item()) if blo > 0 else 0
            p_hi = int(cumlens[bhi - 1].item())
            mat[flat] = x_kept[p_lo:p_hi]
            mat = mat.view(bhi - blo, cmax)
            ahead = mat[:, :, None] > ox_line[None, None, :]
            counts[lo + blo : lo + bhi] += ahead.sum(dim=1)

    solid = counts % 2 == 1  # (ny*nz, nx), col id = j * nz + k
    return solid.view(ny, nz, nx).permute(1, 0, 2).contiguous()


# ---------------------------------------------------------------------------
# 2. LINK Q-FIELD (Möller–Trumbore), accelerated
# ---------------------------------------------------------------------------


def _link_slot_ids(
    base_id: torch.Tensor,
    direction: tuple[int, int, int],
    ny: int,
    nz: int,
) -> tuple[list[tuple[int, int, int]], list[torch.Tensor]]:
    """Bin ids of a link's ``<= 2 x 2 x 2`` cell block.

    Linear bin id ``(i * ny + j) * nz + k`` shifts by ``ny*nz`` / ``nz``
    / ``1`` for ``+x`` / ``+y`` / ``+z``; per axis with lattice component
    ``s`` the covered cells are ``{i, i + s}`` (one cell when ``s == 0``).
    """
    cx, cy, cz = direction
    all_slots = [(sz, sy, sx) for sz in (0, 1) for sy in (0, 1) for sx in (0, 1)]
    slots = [
        s
        for s in all_slots
        if (s[2] == 0 or cx != 0) and (s[1] == 0 or cy != 0) and (s[0] == 0 or cz != 0)
    ]
    slot_ids = [base_id + sx * cx * ny * nz + sy * cy * nz + sz * cz for (sz, sy, sx) in slots]
    return slots, slot_ids


def _boundary_and_q_cuda_binned(
    kernels,
    tri9: torch.Tensor,
    bins: TriangleBins,
    solid_mask: torch.Tensor,
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Binned-Triton counterpart of ``geometry_voxel._boundary_and_q_cuda``.

    Post-processing (resolve tolerance, clamps, 0.5 fallback) is copied
    verbatim from the brute-force Triton path so the two agree
    bit-for-bit given conservative candidate lists.
    """
    dims = tuple(solid_mask.shape)
    nz, ny, nx = dims
    bin_len = bins.lengths()
    boundary, q_field, fluid = _boundary_scaffold(solid_mask)
    for d in range(1, 19):
        cx, cy, cz = (int(v) for v in _C19[d].tolist())
        bnd = fluid & _shift_solid(solid_mask, cz, cy, cx)
        boundary[d] = bnd
        if not bool(bnd.any()):
            continue
        cells = bnd.nonzero(as_tuple=False)
        base_id = (cells[:, 2].to(torch.int64) * ny + cells[:, 1].to(torch.int64)) * nz + cells[
            :, 0
        ].to(torch.int64)
        _, slot_ids = _link_slot_ids(base_id, (cx, cy, cz), ny, nz)
        link_lens = torch.zeros(cells.shape[0], dtype=torch.int64, device=bin_len.device)
        for sid in slot_ids:
            link_lens += bin_len[sid]
        starts, ends, entries = _link_candidate_csr(bins, slot_ids, link_lens)
        best = kernels.link_q_min_t_binned(
            tri9, starts, ends, entries, cells, (cx, cy, cz), origin, spacing
        )
        resolved = best <= 1.0 + 1.0e-6
        qd = torch.where(resolved, best.clamp(1.0e-6, 1.0), torch.full_like(best, 0.5))
        q_field[d][bnd] = qd
    return boundary, q_field


def q_field_accelerated(
    triangles,
    solid_mask: torch.Tensor,
    *,
    origin=None,
    spacing=None,
    pad: float | None = None,
    pair_budget: int = _DEFAULT_PAIR_BUDGET,
    use_triton: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bit-exact accelerated counterpart of ``q_field_reference``.

    A D3Q19 link spans at most one cell per axis, so its candidate
    triangles are exactly those binned (:func:`build_cell_bins`) into the
    ``<= 2 x 2 x 2`` block of cells between the fluid node and its solid
    neighbour.  Cross-bin duplicates are possible and harmless: the
    reduction is an order-invariant ``min``.  The Möller–Trumbore
    arithmetic per (link, triangle) pair is copied verbatim from the
    reference (fp64, same operand order), so
    ``torch.equal``-level parity holds for both returned tensors.

    With ``use_triton=True`` on a CUDA device the per-link candidate
    lists are compacted to CSR on the host side and evaluated by the
    binned Triton kernel
    (:func:`tensorlbm._voxel_kernels.link_q_min_t_binned`).  Its fp32
    arithmetic is identical per pair to the brute-force Triton kernel,
    but the two kernel tile shapes compile to slightly different FMA
    contraction, so ``q`` agrees with the brute-force Triton path to
    fp32 rounding dust (max 2.9e-6 measured across all benchmarked
    mesh/grid combinations, none beyond 1e-3, no hit/miss flips; the
    solid and boundary masks are bit-identical) — see
    ``docs/benchmarks/voxel_accel_benchmark.md``.

    Args:
        triangles: ``(T, 3, 3)`` triangle table (array-like or tensor).
        solid_mask: Bool tensor ``(nz, ny, nx)``.
        origin: Physical lower corner ``(x0, y0, z0)`` used for the mask.
        spacing: Cell sizes ``(dx, dy, dz)``.
        pad: Bin AABB padding (default: ``1e-5 * diag + 1e-6``).
        pair_budget: Link-candidate pairs per tile (memory knob).
        use_triton: Force the binned Triton kernel (``True``) or the
            pure-torch fp64 path (``False``, the default).

    Returns:
        ``(fluid_boundary_mask, q_field)`` — ``(19, nz, ny, nx)`` bool and
        float32, same conventions as ``q_field_reference``.
    """
    dims, org, spc = _validate_grid(solid_mask.shape, origin, spacing)
    nz, ny, nx = dims
    dev = solid_mask.device
    tri_in = _as_triangle_tensor(triangles)
    if _wants_triton(use_triton, dev):
        kernels = _load_kernels()
        if kernels is None or dev.type != "cuda" or not torch.cuda.is_available():
            msg = "use_triton=True requires CUDA and the GPU kernel module"
            raise RuntimeError(msg)
        tri32_tab = _cull_triangles(
            tri_in.to(device=dev, dtype=torch.float32),
            dims,
            org,
            spc,
            (1.0, 1.0, 1.0),
        )
        tri9 = tri32_tab.reshape(-1, 9).contiguous()
        if pad is None:
            pad = _mesh_pad(tri32_tab.to(torch.float64))
        bins32 = _build_bins(
            tri32_tab.to(torch.float64),
            (
                (0, nx, float(org[0]), float(spc[0])),
                (1, ny, float(org[1]), float(spc[1])),
                (2, nz, float(org[2]), float(spc[2])),
            ),
            pad,
        )
        return _boundary_and_q_cuda_binned(kernels, tri9, bins32, solid_mask, org, spc)
    tri = _cull_triangles(
        _as_triangle_tensor(triangles).to(device=dev, dtype=torch.float64),
        dims,
        org,
        spc,
        (1.0, 1.0, 1.0),
    )
    if pad is None:
        pad = _mesh_pad(tri)
    bins = _build_bins(
        tri,
        (
            (0, nx, float(org[0]), float(spc[0])),
            (1, ny, float(org[1]), float(spc[1])),
            (2, nz, float(org[2]), float(spc[2])),
        ),
        pad,
    )
    boundary, q_field, fluid = _boundary_scaffold(solid_mask)
    bin_len = bins.lengths()

    v0 = tri[:, 0, :]
    v1 = tri[:, 1, :]
    v2 = tri[:, 2, :]
    e1 = v1 - v0
    e2 = v2 - v0

    for d in range(1, 19):
        cx, cy, cz = (int(v) for v in _C19[d].tolist())
        bnd = fluid & _shift_solid(solid_mask, cz, cy, cx)
        boundary[d] = bnd
        if not bool(bnd.any()):
            continue
        cells = bnd.nonzero(as_tuple=False)
        n_links = cells.shape[0]
        base_id = (cells[:, 2].to(torch.int64) * ny + cells[:, 1].to(torch.int64)) * nz + cells[
            :, 0
        ].to(torch.int64)

        # Per-direction triangle-only quantities (identical expressions).
        direction = torch.tensor(
            (float(cx) * spc[0], float(cy) * spc[1], float(cz) * spc[2]),
            dtype=torch.float64,
            device=dev,
        )
        h = torch.linalg.cross(direction.expand_as(e2), e2)
        a = (e1 * h).sum(dim=1)
        ok = a.abs() > 1.0e-12
        f = torch.where(ok, 1.0 / torch.where(ok, a, torch.ones_like(a)), torch.zeros_like(a))

        slots, slot_ids = _link_slot_ids(base_id, (cx, cy, cz), ny, nz)
        link_lens = torch.zeros(n_links, dtype=torch.int64, device=dev)
        for sid in slot_ids:
            link_lens += bin_len[sid]

        best = torch.full((n_links,), 2.0, dtype=torch.float64, device=dev)
        for lo, hi in _pair_blocks(link_lens, pair_budget):
            # Level-1 blocks are laid out link-major: (link, slot).
            lens1 = torch.stack([bin_len[sid[lo:hi]] for sid in slot_ids], dim=1).reshape(-1)
            blk_of_pair, inner = _ragged_pairs(lens1)
            sid_flat = torch.stack([sid[lo:hi] for sid in slot_ids], dim=1).reshape(-1)
            entry_idx = bins.offsets[sid_flat[blk_of_pair]] + inner
            t_idx = bins.entries[entry_idx]
            link_of_pair = lo + torch.div(blk_of_pair, len(slots), rounding_mode="floor")

            ox = org[0] + (cells[link_of_pair, 2].to(torch.float64) + 0.5) * spc[0]
            oy = org[1] + (cells[link_of_pair, 1].to(torch.float64) + 0.5) * spc[1]
            oz = org[2] + (cells[link_of_pair, 0].to(torch.float64) + 0.5) * spc[2]

            sx = ox - v0[t_idx, 0]
            sy = oy - v0[t_idx, 1]
            sz = oz - v0[t_idx, 2]
            u = f[t_idx] * (sx * h[t_idx, 0] + sy * h[t_idx, 1] + sz * h[t_idx, 2])
            e1_g = e1[t_idx]
            qx = sy * e1_g[:, 2] - sz * e1_g[:, 1]
            qy = sz * e1_g[:, 0] - sx * e1_g[:, 2]
            qz = sx * e1_g[:, 1] - sy * e1_g[:, 0]
            v = f[t_idx] * (direction[0] * qx + direction[1] * qy + direction[2] * qz)
            t = f[t_idx] * (e2[t_idx, 0] * qx + e2[t_idx, 1] * qy + e2[t_idx, 2] * qz)
            hit = (
                ok[t_idx]
                & (u >= -_REF_BARY_EPS)
                & (v >= -_REF_BARY_EPS)
                & (u + v <= 1.0 + _REF_BARY_EPS)
                & (t > _REF_T_MIN)
                & (t <= 1.0 + _REF_T_MAX_PAD)
            )
            vals = torch.where(hit, t, 2.0)
            best.scatter_reduce_(0, link_of_pair, vals, reduce="amin", include_self=True)

        resolved = best <= 1.0 + _REF_T_MAX_PAD
        qd = torch.where(
            resolved,
            best.clamp(_REF_T_MIN, 1.0),
            torch.full_like(best, 0.5),
        )
        q_field[d][bnd] = qd.to(torch.float32)
    return boundary, q_field


# ---------------------------------------------------------------------------
# 3. PUBLIC API
# ---------------------------------------------------------------------------


def voxelize_stl_accelerated(
    path_or_mesh,
    shape,
    *,
    device: str | torch.device = "cpu",
    origin=None,
    spacing=None,
    check_watertight: bool = True,
    pad: float | None = None,
    pair_budget: int = _DEFAULT_PAIR_BUDGET,
    use_triton: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accelerated mirror of :func:`tensorlbm.geometry_voxel.voxelize_stl_reference`.

    Identical outputs (bit-for-bit, see the module docstring) at a cost
    that scales with the *surface* rather than with ``rays x triangles``.
    The binned Triton kernels of :mod:`tensorlbm._voxel_kernels` can be
    forced with ``use_triton=True`` on CUDA (solid/boundary masks
    bit-identical to the brute-force Triton kernels, ``q`` within fp32
    rounding dust, max 2.9e-6 measured); the pure-torch fp64 default is
    bit-identical to the reference.
    See :func:`tensorlbm.geometry_voxel.voxelize_stl` for the argument
    contract; ``pad`` / ``pair_budget`` are memory knobs of the hash grid.
    """
    from .geometry_voxel import _coerce_mesh, mesh_watertight_status

    dims, org, spc = _validate_grid(shape, origin, spacing)
    dev = torch.device(device)
    triangles = _coerce_mesh(path_or_mesh, dev)

    if check_watertight:
        status = mesh_watertight_status(triangles)
        if not status["watertight"]:
            warnings.warn(
                "STL mesh is not closed "
                f"({status['boundary_edges']} boundary edges, "
                f"{status['nonmanifold_edges']} non-manifold edges); "
                "ray-parity voxelisation assumes a closed surface — the "
                "solid mask may be wrong near openings",
                stacklevel=2,
            )

    solid = solid_mask_parity_accelerated(
        triangles,
        dims,
        origin=org,
        spacing=spc,
        device=dev,
        pad=pad,
        pair_budget=pair_budget,
        use_triton=use_triton,
    )
    boundary, q_field = q_field_accelerated(
        triangles,
        solid,
        origin=org,
        spacing=spc,
        pad=pad,
        pair_budget=pair_budget,
        use_triton=use_triton,
    )
    return solid, boundary, q_field
