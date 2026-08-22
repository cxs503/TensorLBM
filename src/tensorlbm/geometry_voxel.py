"""GPU STL voxelisation: triangle mesh -> solid mask + BFL q-field.

Turns an arbitrary closed triangle mesh (STL) into the three tensors the
D3Q19 solver stack consumes, following the conventions of
:func:`tensorlbm.bfl_d3q19.compute_q_cylinder_d3q19` and
:func:`tensorlbm.preprocess_geo.compute_q_generic_3d`:

* tensor layout ``(nz, ny, nx)`` with the streamwise x axis last;
* ``solid_mask`` — True inside the closed surface;
* ``fluid_boundary_mask`` — ``(19, nz, ny, nx)`` bool, True at
  ``[d, k, j, i]`` when **fluid** node ``(i, j, k)`` has a solid D3Q19
  neighbour in direction ``d`` (no periodic wrap at the domain edge);
* ``q_field`` — ``(19, nz, ny, nx)`` float32 Bouzidi fraction: the wall
  lies at ``x + q * c_d`` for ``q in (0, 1]``; entries without a resolved
  wall (and all non-boundary entries) hold ``0.5``, which in the BFL
  formula is *exactly* standard halfway bounce-back — the degrade path.

Pipeline (all stages are original implementations written from textbook
ray-triangle geometry; no third-party source was consulted):

1. A minimal STL parser (:func:`read_stl_triangles`) reads binary or
   ASCII files with no dependency beyond NumPy and returns a
   ``(T, 3, 3)`` float32 tensor.
2. :func:`mesh_watertight_status` welds vertices and counts boundary /
   non-manifold edges; open meshes trigger a :class:`UserWarning`
   (ray parity silently assumes a closed surface).
3. The solid mask comes from ray parity: every cell casts a ray along +x
   and is inside iff an odd number of triangle crossings lie ahead.  On
   CUDA this runs as a Triton kernel (cells parallel, triangle chunks
   streamed per program, :mod:`tensorlbm._voxel_kernels`); otherwise a
   pure-torch float64 reference implementation runs on any device.
4. The q-field intersects every boundary link with the mesh
   (Möller–Trumbore along the spacing-scaled lattice velocity, taking
   the minimum fraction in ``(0, 1]``).

For large meshes the steps above cost ``O(rays x triangles)`` /
``O(links x triangles)``; passing ``accelerate=True`` to
:func:`voxelize_stl` routes them through the uniform spatial hash grid
of :mod:`tensorlbm.voxel_accel`, which returns bit-identical tensors at
a cost that scales with the mesh *surface* (target: 10^6 triangles).

Robustness notes
----------------
* Asymmetric sub-cell perturbation of the ray origin (``+1.3e-4`` cells
  in y, ``+3.7e-4`` in z, mirroring :mod:`tensorlbm.preprocess_geo`)
  breaks exact alignments of ray and triangle edges which would
  double-count shared edges and punch holes in the mask.
* The GPU kernel computes in fp32 (lattice precision); the reference
  path computes in fp64.  They agree to ~1e-5 in q and to parity
  disagreements only within fp rounding of a cell centre.
* Triangles are axis-bbox-culled before the kernels; for parity, culling
  never drops triangles on the +x side (a crossing beyond the domain
  still flips the parity of the cells behind it).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .d3q19 import C as _C19

__all__ = [
    "read_stl_triangles",
    "mesh_watertight_status",
    "solid_mask_parity_reference",
    "q_field_reference",
    "voxelize_stl",
    "voxelize_stl_reference",
]

# Asymmetric ray-origin perturbation (fraction of a cell) — see module docstring.
_PARITY_EPS_Y: float = 1.3e-4
_PARITY_EPS_Z: float = 3.7e-4

# Möller–Trumbore / clamping tolerances for the reference (fp64) path.
_REF_BARY_EPS: float = 1.0e-9
_REF_T_MIN: float = 1.0e-9
_REF_T_MAX_PAD: float = 1.0e-9


# ---------------------------------------------------------------------------
# 1. STL READER (binary + ASCII, no external dependency)
# ---------------------------------------------------------------------------


def _parse_binary_stl(data: bytes, n_tri: int) -> np.ndarray:
    """Parse a binary STL payload into a ``(T, 3, 3)`` float32 array."""
    dt = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("v0", "<f4", (3,)),
            ("v1", "<f4", (3,)),
            ("v2", "<f4", (3,)),
            ("attr", "<u2"),
        ]
    )
    records = np.frombuffer(data, dtype=dt, count=n_tri, offset=84)
    return np.stack([records["v0"], records["v1"], records["v2"]], axis=1)


def _parse_ascii_stl(text: str) -> np.ndarray:
    """Parse an ASCII STL payload into a ``(T, 3, 3)`` float32 array."""
    triangles: list[list[tuple[float, float, float]]] = []
    verts: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("vertex"):
            parts = stripped.split()
            if len(parts) >= 4:
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif stripped.startswith("endfacet"):
            if len(verts) == 3:
                triangles.append(verts)
            verts = []
    if triangles:
        return np.asarray(triangles, dtype=np.float32)
    return np.zeros((0, 3, 3), dtype=np.float32)


def read_stl_triangles(path, *, device: str | torch.device | None = None) -> torch.Tensor:
    """Read an STL file (binary or ASCII) as a triangle table.

    Binary format is detected by the exact file-size match
    ``84 + 50 * n_tri`` (with a small-remainder fallback for padded
    files); everything else is parsed as ASCII.

    Args:
        path: Path to the STL file.
        device: Optional target device for the returned tensor
            (default: CPU).

    Returns:
        ``(T, 3, 3)`` float32 tensor of triangle vertices
        ``[t, v, axis]`` with axis order ``(x, y, z)``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file contains no triangles.
    """
    path = Path(path)
    if not path.exists():
        msg = f"STL file not found: {path}"
        raise FileNotFoundError(msg)
    data = path.read_bytes()
    triangles: np.ndarray
    if len(data) < 84:
        triangles = _parse_ascii_stl(data.decode("utf-8", errors="replace"))
    else:
        n_candidate = int(np.frombuffer(data[80:84], dtype=np.uint32)[0])
        n_from_size = (len(data) - 84) // 50
        remainder = (len(data) - 84) % 50
        is_binary = n_candidate > 0 and (
            len(data) == 84 + 50 * n_candidate or (n_from_size > 0 and remainder < 4)
        )
        if is_binary:
            triangles = _parse_binary_stl(data[: 84 + 50 * n_from_size], n_from_size)
        else:
            triangles = _parse_ascii_stl(data.decode("utf-8", errors="replace"))
    if triangles.size == 0:
        triangles = _parse_ascii_stl(data.decode("utf-8", errors="replace"))
    if triangles.shape[0] == 0:
        msg = f"No triangles found in STL file: {path}"
        raise ValueError(msg)
    tensor = torch.from_numpy(np.ascontiguousarray(triangles))
    if device is not None:
        tensor = tensor.to(device)
    return tensor


# ---------------------------------------------------------------------------
# 2. MESH VALIDATION
# ---------------------------------------------------------------------------


def _as_triangle_tensor(triangles) -> torch.Tensor:
    """Coerce an array-like triangle table to a validated ``(T, 3, 3)`` tensor."""
    if isinstance(triangles, torch.Tensor):
        tensor = triangles
    else:
        tensor = torch.as_tensor(np.asarray(triangles))
    if tensor.ndim != 3 or tensor.shape[1:] != (3, 3):
        msg = f"triangles must have shape (T, 3, 3), got {tuple(tensor.shape)}"
        raise ValueError(msg)
    return tensor


def mesh_watertight_status(triangles, *, weld_tol: float = 1.0e-6) -> dict[str, int | bool]:
    """Check a triangle mesh for open / non-manifold edges.

    Vertices are welded by rounding to a ``weld_tol`` grid (STL repeats
    shared vertices bit-identically in practice), every triangle
    contributes three edges, and an edge shared by exactly two triangles
    is required for a closed manifold surface.  Ray-parity voxelisation
    only needs a *closed* surface; orientation is irrelevant.

    Args:
        triangles: ``(T, 3, 3)`` triangle table (array-like or tensor).
        weld_tol: Absolute vertex-welding tolerance in mesh units.

    Returns:
        Dict with keys ``watertight`` (bool), ``triangles``,
        ``boundary_edges`` (shared by one triangle), ``nonmanifold_edges``
        (shared by more than two) and ``degenerate_faces``.
    """
    tri = _as_triangle_tensor(triangles).to(dtype=torch.float64)
    flat = tri.reshape(-1, 3)
    quantized = torch.round(flat / weld_tol).to(torch.int64)
    _, inverse = torch.unique(quantized, dim=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)

    repeated = (
        (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
    )
    area2 = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).norm(dim=1)
    diag = float((flat.max(dim=0).values - flat.min(dim=0).values).norm().item())
    degenerate = repeated | (area2 <= 1.0e-10 * max(diag * diag, 1.0))

    manifold_faces = faces[~repeated]
    edges = torch.cat(
        [manifold_faces[:, [0, 1]], manifold_faces[:, [1, 2]], manifold_faces[:, [2, 0]]]
    )
    edges = torch.sort(edges, dim=1).values
    _, counts = torch.unique(edges, dim=0, return_counts=True)
    boundary_edges = int((counts == 1).sum().item())
    nonmanifold_edges = int((counts > 2).sum().item())
    return {
        "watertight": bool(tri.shape[0] > 0 and boundary_edges == 0 and nonmanifold_edges == 0),
        "triangles": int(tri.shape[0]),
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "degenerate_faces": int(degenerate.sum().item()),
    }


# ---------------------------------------------------------------------------
# 3. SHARED GRID UTILITIES
# ---------------------------------------------------------------------------


def _validate_grid(
    shape, origin, spacing
) -> tuple[tuple[int, int, int], tuple[float, float, float], tuple[float, float, float]]:
    dims = tuple(int(v) for v in shape)
    if len(dims) != 3 or any(v <= 0 for v in dims):
        msg = f"shape must be three positive ints (nz, ny, nx), got {shape!r}"
        raise ValueError(msg)
    org = (0.0, 0.0, 0.0) if origin is None else tuple(float(v) for v in origin)
    spc = (1.0, 1.0, 1.0) if spacing is None else tuple(float(v) for v in spacing)
    if len(org) != 3 or len(spc) != 3 or any(v <= 0.0 for v in spc):
        msg = f"origin/spacing must be three floats, spacing positive; got {origin!r}, {spacing!r}"
        raise ValueError(msg)
    return dims, org, spc  # type: ignore[return-value]


def _parity_origin(
    origin: tuple[float, float, float], spacing: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Apply the asymmetric ray-origin perturbation (module docstring)."""
    x0, y0, z0 = origin
    dx, dy, dz = spacing
    return (x0, y0 + _PARITY_EPS_Y * dy, z0 + _PARITY_EPS_Z * dz)


def _shift_solid(mask: torch.Tensor, sz: int, sy: int, sx: int) -> torch.Tensor:
    """Gather ``mask[k+sz, j+sy, i+sx]``; False outside the domain.

    Unlike :func:`torch.roll` there is no periodic wrap, so domain-edge
    fluid cells are never marked boundary by ghosts.  Valid for lattice
    shifts in ``{-1, 0, 1}``.
    """
    nz, ny, nx = mask.shape
    padded = F.pad(mask[None, None], (1, 1, 1, 1, 1, 1), value=False)[0, 0]
    return padded[
        1 + sz : 1 + sz + nz,
        1 + sy : 1 + sy + ny,
        1 + sx : 1 + sx + nx,
    ]


def _cull_triangles(
    triangles: torch.Tensor,
    shape: tuple[int, int, int],
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    margin: tuple[float, float, float],
) -> torch.Tensor:
    """Drop triangles whose axis bbox cannot interact with the grid.

    ``margin`` extends the grid bbox per axis (in cells).  For the parity
    pass use ``margin_x = (inf_negative_side)`` semantics by passing a
    large positive ``margin[0]`` upper extension — triangles ahead of the
    +x ray direction must be *kept* (their crossings still flip parity),
    so the +x margin is effectively infinite; the caller encodes this by
    passing ``math.inf`` where needed.
    """
    nz, ny, nx = shape
    x0, y0, z0 = origin
    dx, dy, dz = spacing
    mx, my, mz = margin
    vmin = triangles.min(dim=1).values
    vmax = triangles.max(dim=1).values
    keep = (
        (vmin[:, 0] <= x0 + nx * dx + mx)
        & (vmax[:, 0] >= x0 - mx)
        & (vmin[:, 1] <= y0 + ny * dy + my)
        & (vmax[:, 1] >= y0 - my)
        & (vmin[:, 2] <= z0 + nz * dz + mz)
        & (vmax[:, 2] >= z0 - mz)
    )
    return triangles[keep]


def _boundary_scaffold(
    solid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate ``(19, nz, ny, nx)`` boundary/q buffers and the fluid mask."""
    nz, ny, nx = solid_mask.shape
    device = solid_mask.device
    boundary = torch.zeros((19, nz, ny, nx), dtype=torch.bool, device=device)
    q_field = torch.full((19, nz, ny, nx), 0.5, dtype=torch.float32, device=device)
    return boundary, q_field, ~solid_mask


# ---------------------------------------------------------------------------
# 4. PURE-TORCH REFERENCE IMPLEMENTATIONS (fp64, any device)
# ---------------------------------------------------------------------------


def solid_mask_parity_reference(
    triangles,
    shape,
    *,
    origin=None,
    spacing=None,
    device: str | torch.device | None = None,
    ray_chunk: int = 16384,
    tri_chunk: int = 64,
) -> torch.Tensor:
    """Reference +x-ray-parity solid mask in pure torch (float64).

    Mirrors the Triton kernel in :mod:`tensorlbm._voxel_kernels` with the
    same acceptance rules and ray-origin perturbation, so the two can be
    cross-validated.  Slow (cells x triangles work) but exact; use it for
    CPU environments and correctness checks.

    Args:
        triangles: ``(T, 3, 3)`` triangle table (array-like or tensor).
        shape: Grid shape ``(nz, ny, nx)``.
        origin: Physical lower corner ``(x0, y0, z0)`` (default zero).
        spacing: Cell sizes ``(dx, dy, dz)`` (default one).
        device: Target device (default: the triangle tensor's device).
        ray_chunk: Rays per tile (memory knob).
        tri_chunk: Triangles per tile (memory knob).

    Returns:
        Bool tensor of ``shape``, True inside the closed surface.
    """
    dims, org, spc = _validate_grid(shape, origin, spacing)
    nz, ny, nx = dims
    tri = _as_triangle_tensor(triangles)
    dev = tri.device if device is None else torch.device(device)
    tri = _cull_triangles(
        tri.to(device=dev, dtype=torch.float64), dims, org, spc, (float("inf"), 0.0, 0.0)
    )

    x0, y0, z0 = _parity_origin(org, spc)
    dx, dy, dz = spc
    ox_line = x0 + (torch.arange(nx, device=dev, dtype=torch.float64) + 0.5) * dx
    oy_line = y0 + (torch.arange(ny, device=dev, dtype=torch.float64) + 0.5) * dy

    v0, v1, v2 = tri[:, 0, :], tri[:, 1, :], tri[:, 2, :]
    e1 = v1 - v0
    e2 = v2 - v0
    normal = torch.linalg.cross(e1, e2)
    nx_n, ny_n_full, nz_n_full = normal[:, 0], normal[:, 1], normal[:, 2]
    ok = nx_n.abs() > 1.0e-12
    safe_nx = torch.where(ok, nx_n, torch.ones_like(nx_n))
    inv_nx = torch.where(ok, 1.0 / safe_nx, torch.zeros_like(nx_n))

    n_tri = tri.shape[0]
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=dev)
    n_rays = ny * nx
    for k in range(nz):
        oz = z0 + (k + 0.5) * dz
        count = torch.zeros((n_rays,), dtype=torch.int64, device=dev)
        for t0 in range(0, n_tri, tri_chunk):
            sl = slice(t0, min(t0 + tri_chunk, n_tri))
            v0c, v1c, v2c = v0[sl], v1[sl], v2[sl]
            a0y, a0z = v1c[:, 1] - v0c[:, 1], v1c[:, 2] - v0c[:, 2]
            a1y, a1z = v2c[:, 1] - v1c[:, 1], v2c[:, 2] - v1c[:, 2]
            a2y, a2z = v0c[:, 1] - v2c[:, 1], v0c[:, 2] - v2c[:, 2]
            v0xc, v0yc, v0zc = v0c[:, 0], v0c[:, 1], v0c[:, 2]
            v1yc, v1zc = v1c[:, 1], v1c[:, 2]
            v2yc, v2zc = v2c[:, 1], v2c[:, 2]
            ny_c, nz_c, okc, inv_nxc = ny_n_full[sl], nz_n_full[sl], ok[sl], inv_nx[sl]
            for r0 in range(0, n_rays, ray_chunk):
                ridx = torch.arange(r0, min(r0 + ray_chunk, n_rays), device=dev)
                jdx = ridx // nx
                oy = oy_line[jdx][:, None]
                ox = ox_line[ridx - jdx * nx][:, None]
                py0 = oy - v0yc[None, :]
                pz0 = oz - v0zc[None, :]
                d0 = a0y[None, :] * pz0 - a0z[None, :] * py0
                py1 = oy - v1yc[None, :]
                pz1 = oz - v1zc[None, :]
                d1 = a1y[None, :] * pz1 - a1z[None, :] * py1
                py2 = oy - v2yc[None, :]
                pz2 = oz - v2zc[None, :]
                d2 = a2y[None, :] * pz2 - a2z[None, :] * py2
                has_neg = (d0 < 0.0) | (d1 < 0.0) | (d2 < 0.0)
                has_pos = (d0 > 0.0) | (d1 > 0.0) | (d2 > 0.0)
                inside = ~(has_neg & has_pos)
                x_isect = (
                    v0xc[None, :] - (ny_c[None, :] * py0 + nz_c[None, :] * pz0) * inv_nxc[None, :]
                )
                hit = inside & okc[None, :] & (x_isect > ox)
                count[ridx] += hit.sum(dim=1).to(torch.int64)
        solid[k] = (count % 2 == 1).reshape(ny, nx)
    return solid


def q_field_reference(
    triangles,
    solid_mask: torch.Tensor,
    *,
    origin=None,
    spacing=None,
    tri_chunk: int = 64,
    cell_chunk: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference BFL boundary mask and q-field in pure torch (float64).

    For every fluid node with a solid D3Q19 neighbour in direction
    ``d``, the link ``[x, x + c_d]`` is intersected with all triangles
    (Möller–Trumbore along the spacing-scaled lattice velocity) and the
    minimum fraction in ``(0, 1]`` becomes ``q_field[d]``.  Links without
    any intersection keep ``q = 0.5`` — exactly standard halfway
    bounce-back, the degrade path required by the BFL kernel.

    Args:
        triangles: ``(T, 3, 3)`` triangle table (array-like or tensor).
        solid_mask: Bool tensor ``(nz, ny, nx)``.
        origin: Physical lower corner ``(x0, y0, z0)`` used for the mask.
        spacing: Cell sizes ``(dx, dy, dz)``.
        tri_chunk: Triangles per tile (memory knob).
        cell_chunk: Boundary cells per tile (memory knob).

    Returns:
        ``(fluid_boundary_mask, q_field)`` with shapes
        ``(19, nz, ny, nx)`` (bool) and ``(19, nz, ny, nx)`` (float32).
    """
    dims, org, spc = _validate_grid(solid_mask.shape, origin, spacing)
    dev = solid_mask.device
    tri = _cull_triangles(
        _as_triangle_tensor(triangles).to(device=dev, dtype=torch.float64),
        dims,
        org,
        spc,
        (1.0, 1.0, 1.0),
    )
    boundary, q_field, fluid = _boundary_scaffold(solid_mask)

    n_tri = tri.shape[0]
    for d in range(1, 19):
        cx, cy, cz = (int(v) for v in _C19[d].tolist())
        bnd = fluid & _shift_solid(solid_mask, cz, cy, cx)
        boundary[d] = bnd
        if not bool(bnd.any()):
            continue
        cells = bnd.nonzero(as_tuple=False)
        best = torch.full((cells.shape[0],), 2.0, dtype=torch.float64, device=dev)
        for c0 in range(0, cells.shape[0], cell_chunk):
            csl = slice(c0, min(c0 + cell_chunk, cells.shape[0]))
            kdx, jdx, idx = cells[csl, 0], cells[csl, 1], cells[csl, 2]
            ox = org[0] + (idx.to(torch.float64) + 0.5) * spc[0]
            oy = org[1] + (jdx.to(torch.float64) + 0.5) * spc[1]
            oz = org[2] + (kdx.to(torch.float64) + 0.5) * spc[2]
            best_c = torch.full((csl.stop - csl.start,), 2.0, dtype=torch.float64, device=dev)
            for t0 in range(0, n_tri, tri_chunk):
                tsl = slice(t0, min(t0 + tri_chunk, n_tri))
                v0, v1, v2 = tri[tsl, 0, :], tri[tsl, 1, :], tri[tsl, 2, :]
                e1, e2 = v1 - v0, v2 - v0
                # Physical lattice velocity: the link spans
                # [x, x + (dx*cx, dy*cy, dz*cz)] so the MT parameter t is
                # directly the Bouzidi fraction for any spacing.
                direction = torch.tensor(
                    (float(cx) * spc[0], float(cy) * spc[1], float(cz) * spc[2]),
                    dtype=torch.float64,
                    device=dev,
                )
                h = torch.linalg.cross(direction.expand_as(e2), e2)
                a = (e1 * h).sum(dim=1)
                ok = a.abs() > 1.0e-12
                f = torch.where(
                    ok, 1.0 / torch.where(ok, a, torch.ones_like(a)), torch.zeros_like(a)
                )
                sx = ox[:, None] - v0[None, :, 0]
                sy = oy[:, None] - v0[None, :, 1]
                sz = oz[:, None] - v0[None, :, 2]
                u = f[None, :] * (sx * h[None, :, 0] + sy * h[None, :, 1] + sz * h[None, :, 2])
                qx = sy * e1[None, :, 2] - sz * e1[None, :, 1]
                qy = sz * e1[None, :, 0] - sx * e1[None, :, 2]
                qz = sx * e1[None, :, 1] - sy * e1[None, :, 0]
                v = f[None, :] * (direction[0] * qx + direction[1] * qy + direction[2] * qz)
                t = f[None, :] * (e2[None, :, 0] * qx + e2[None, :, 1] * qy + e2[None, :, 2] * qz)
                hit = (
                    ok[None, :]
                    & (u >= -_REF_BARY_EPS)
                    & (v >= -_REF_BARY_EPS)
                    & (u + v <= 1.0 + _REF_BARY_EPS)
                    & (t > _REF_T_MIN)
                    & (t <= 1.0 + _REF_T_MAX_PAD)
                )
                best_c = torch.minimum(best_c, torch.where(hit, t, 2.0).min(dim=1).values)
            best[csl] = best_c
        resolved = best <= 1.0 + _REF_T_MAX_PAD
        qd = torch.where(
            resolved,
            best.clamp(_REF_T_MIN, 1.0),
            torch.full_like(best, 0.5),
        )
        q_field[d][bnd] = qd.to(torch.float32)
    return boundary, q_field


# ---------------------------------------------------------------------------
# 5. PUBLIC API
# ---------------------------------------------------------------------------


def _load_kernels():
    """Import the GPU kernel module or return None (Triton unavailable)."""
    try:
        from . import _voxel_kernels
    except ImportError:  # pragma: no cover - depends on the environment
        return None
    return _voxel_kernels


def _coerce_mesh(path_or_mesh, device: torch.device) -> torch.Tensor:
    """Accept an STL path, a (T, 3, 3) table, or a (vertices, faces) pair."""
    if isinstance(path_or_mesh, (str, Path)):
        return read_stl_triangles(path_or_mesh, device=device)
    if isinstance(path_or_mesh, (tuple, list)) and len(path_or_mesh) == 2:
        vertices, faces = path_or_mesh
        vertices_t = torch.as_tensor(np.asarray(vertices)).to(device)
        faces_t = torch.as_tensor(np.asarray(faces)).to(device=device, dtype=torch.long)
        return vertices_t[faces_t].to(dtype=torch.float32)
    tensor = _as_triangle_tensor(path_or_mesh)
    return tensor.to(device=device, dtype=torch.float32)


def _boundary_and_q_cuda(
    kernels,
    triangles: torch.Tensor,
    solid_mask: torch.Tensor,
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton path: per-direction Möller–Trumbore min-q on boundary links."""
    boundary, q_field, fluid = _boundary_scaffold(solid_mask)
    tri_q = (
        _cull_triangles(triangles, tuple(solid_mask.shape), origin, spacing, (1.0, 1.0, 1.0))
        .reshape(-1, 9)
        .contiguous()
    )
    for d in range(1, 19):
        cx, cy, cz = (int(v) for v in _C19[d].tolist())
        bnd = fluid & _shift_solid(solid_mask, cz, cy, cx)
        boundary[d] = bnd
        if not bool(bnd.any()):
            continue
        cells = bnd.nonzero(as_tuple=False)
        best = kernels.link_q_min_t(tri_q, cells, (cx, cy, cz), origin, spacing)
        resolved = best <= 1.0 + 1.0e-6
        qd = torch.where(resolved, best.clamp(1.0e-6, 1.0), torch.full_like(best, 0.5))
        q_field[d][bnd] = qd
    return boundary, q_field


def voxelize_stl(
    path_or_mesh,
    shape,
    *,
    device: str | torch.device = "cuda",
    origin=None,
    spacing=None,
    check_watertight: bool = True,
    use_triton: bool | None = None,
    accelerate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Voxelize an STL triangle mesh into LBM solid mask + BFL q-field.

    On CUDA devices a Triton kernel evaluates the ray-parity test with
    cells in parallel and triangle chunks streamed per program; everywhere
    else (or when Triton is missing) the pure-torch float64 reference
    runs instead.  Both paths return the same conventions.

    Args:
        path_or_mesh: STL file path, a ``(T, 3, 3)`` triangle table
            (tensor or array-like), or a ``(vertices, faces)`` pair.
        shape: Grid shape ``(nz, ny, nx)`` — streamwise x is the **last**
            axis, matching the ``(Q, nz, ny, nx)`` population layout.
        device: Target device (default ``"cuda"``).
        origin: Physical lower corner ``(x0, y0, z0)`` (default zero).
        spacing: Cell sizes ``(dx, dy, dz)`` (default one).
        check_watertight: Warn when the mesh has open/non-manifold edges
            (ray parity assumes a closed surface).
        use_triton: Force the GPU kernel path (``True``) or the reference
            path (``False``); ``None`` auto-selects.
        accelerate: Route through the uniform spatial-hash-grid
            implementation of :mod:`tensorlbm.voxel_accel` (pure torch,
            any device, bit-identical to the reference path; overrides
            the Triton kernels).  Default ``False`` keeps the historical
            brute-force behaviour unchanged — see
            ``docs/benchmarks/voxel_accel_benchmark.md`` for when to
            turn this on (anything from ~10^5 triangles up).

    Returns:
        ``(solid_mask, fluid_boundary_mask, q_field)``:

        * ``solid_mask`` — bool ``(nz, ny, nx)``, True inside the mesh;
        * ``fluid_boundary_mask`` — bool ``(19, nz, ny, nx)``, True at
          fluid nodes with a solid neighbour in direction ``d``;
        * ``q_field`` — float32 ``(19, nz, ny, nx)``, Bouzidi fraction in
          ``(0, 1]`` at boundary entries, ``0.5`` elsewhere (halfway
          bounce-back, the degrade path for unresolved links).

    Raises:
        FileNotFoundError: STL path does not exist.
        ValueError: Empty/invalid STL or invalid shape/origin/spacing.
        RuntimeError: ``use_triton=True`` but Triton/CUDA unavailable.
    """
    dims, org, spc = _validate_grid(shape, origin, spacing)
    dev = torch.device(device)
    if accelerate:
        from . import voxel_accel

        return voxel_accel.voxelize_stl_accelerated(
            path_or_mesh,
            shape,
            device=dev,
            origin=origin,
            spacing=spacing,
            check_watertight=check_watertight,
        )
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

    kernels = _load_kernels()
    wants_gpu = (dev.type == "cuda") if use_triton is None else bool(use_triton)
    if wants_gpu and (kernels is None or not torch.cuda.is_available()):
        if use_triton:
            msg = "use_triton=True requires CUDA and the GPU kernel module"
            raise RuntimeError(msg)
        warnings.warn(
            "GPU kernels unavailable — falling back to the pure-torch reference voxelisation path",
            stacklevel=2,
        )
        wants_gpu = False

    if wants_gpu:
        parity_tri = _cull_triangles(triangles, dims, org, spc, (float("inf"), 0.0, 0.0))
        solid = kernels.parity_solid_mask(
            parity_tri.reshape(-1, 9).contiguous(),
            dims,
            _parity_origin(org, spc),
            spc,
        )
        boundary, q_field = _boundary_and_q_cuda(kernels, triangles, solid, org, spc)
    else:
        solid = solid_mask_parity_reference(triangles, dims, origin=org, spacing=spc, device=dev)
        boundary, q_field = q_field_reference(triangles, solid, origin=org, spacing=spc)
    return solid, boundary, q_field


def voxelize_stl_reference(
    path_or_mesh,
    shape,
    *,
    device: str | torch.device = "cpu",
    origin=None,
    spacing=None,
    check_watertight: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-torch reference of :func:`voxelize_stl` (float64 geometry).

    Intended for cross-validating the GPU kernels and for CPU-only
    environments; see :func:`voxelize_stl` for the argument contract.
    """
    return voxelize_stl(
        path_or_mesh,
        shape,
        device=device,
        origin=origin,
        spacing=spacing,
        check_watertight=check_watertight,
        use_triton=False,
    )
