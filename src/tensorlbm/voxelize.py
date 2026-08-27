"""Voxelise arbitrary STL geometry onto the B4 canonical grid.

numpy + stdlib only (no trimesh / scipy / torch): this module is the
CPU front-end of the drag-surrogate geometry stack.  It closes the
roadmap Phase-2 gap "general voxelisation (STL -> SDF)": CAD software
exports an STL, :func:`load_stl` reads it, :func:`place_on_grid` puts it
in the B4 canonical frame and :func:`mask_from_stl` rasterises the
boolean occupancy mask that the SDF encoder (``geom_encoder``, PR #235)
consumes.

Conventions
-----------
* Triangle tables are ``(T, 3, 3)`` float64 arrays indexed
  ``[triangle, vertex, axis]`` with mesh axes in ``(x, y, z)`` order.
* Grids follow the B4 / SUBOFF convention: a mask of shape
  ``(nz, ny, nx)`` has the streamwise mesh axis **x** on the last
  (fastest) array axis, matching the lattice layout ``(Q, nz, ny, nx)``.
* :func:`mask_from_stl` samples *cell centres*: the sample point of
  voxel ``(iz, iy, ix)`` is ``origin + (ix + 0.5, iy + 0.5, iz + 0.5) *
  spacing`` given as mesh ``(x, y, z)``.  A mesh whose faces sit on
  integer coordinates therefore rasterises to an exact half-open voxel
  box (min faces included, max faces excluded).
* :func:`place_on_grid` returns triangles in *voxel-index coordinates*:
  ``origin == (0, 0, 0)`` and ``spacing == 1.0`` so the placement can be
  fed straight back into :func:`mask_from_stl`.

Robustness notes (ray parity)
-----------------------------
One ray is cast per voxel column along a mesh axis (default x, the
streamwise axis) and a voxel is inside iff the number of triangle
crossings strictly ahead of it is odd.  The naive crossing rule
(barycentric ``u >= 0, v >= 0, u + v <= 1``) double-counts a crossing
that lands exactly on an edge shared by two triangles, which flips the
parity of every voxel behind it and punches a one-column hole in the
mask -- the standard failure for grid-aligned meshes.  With
``robust=True`` (default) :func:`mask_from_stl` applies both standard
fixes:

1. a deterministic sub-cell asymmetric perturbation of the ray origin in
   the two transverse coordinates (``+1.3e-4`` cells on the first
   transverse axis, ``-3.7e-4`` on the second, mirroring
   :mod:`tensorlbm.geometry_voxel` and :mod:`tensorlbm.preprocess_geo`),
   which moves rays off mesh edges and vertices exactly; and
2. strict barycentric bounds (``u > 0, v > 0, u + v < 1``) so that any
   tie that survives the perturbation is dropped from *both* triangles
   sharing the edge rather than counted twice.

The residual failure mode (a ray lying exactly on a mesh edge after the
perturbation, possible only for adversarially constructed coordinates)
drops a single crossing, so at most one column of the mask is affected.
A crossing that lands exactly on a voxel sample (ray-axis face aligned
with the sample grid) resolves by floating-point rounding of the
intersection parameter -- deterministic per mesh, but effectively
arbitrary; place such faces between samples or accept a one-cell shift.
``robust=False`` keeps the naive inclusive rule without perturbation and
exists only to demonstrate the failure mode; do not use it in production.

Open meshes
-----------
Ray parity is well-defined for any mesh, but the interior of a mesh that
is not closed (:func:`is_watertight` false) leaks: some columns cross the
surface an odd number of times "in the wrong place" and the mask gains or
loses whole wedges.  Check :func:`is_watertight` first; the module never
requires it.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "Placement",
    "StlMesh",
    "is_watertight",
    "load_stl",
    "mask_from_stl",
    "place_on_grid",
    "sdf_from_mask",
]

# Deterministic sub-cell ray-origin perturbation (fraction of one cell).
_EPS_T1: float = 1.3e-4
_EPS_T2: float = -3.7e-4

# (rays x triangles) elements per vectorised tile; bounds peak memory to
# a few tens of MB regardless of mesh or grid size.
_TILE_BUDGET: int = 1_000_000

# Below this many cells sdf_from_mask defaults to the exact brute-force
# distance; above it the 3-4-5 chamfer runs instead (documented error
# bound in sdf_from_mask).
_EXACT_DEFAULT_LIMIT: int = 32**3

_BIN_RECORD = np.dtype([("normal", "<f4", (3,)), ("verts", "<f4", (3, 3)), ("attr", "<u2")])


# ---------------------------------------------------------------------------
# 1. STL LOADER
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StlMesh:
    """Triangles and per-facet normals read from an STL file.

    Attributes:
        vertices: ``(T, 3, 3)`` float64 triangle table ``[t, v, axis]``,
            mesh axes in ``(x, y, z)`` order.  Empty solids give
            ``T == 0``.
        normals: ``(T, 3)`` float64 facet normals, or ``None`` when the
            file carries no usable normals (non-finite entries or every
            normal zero, as written by several exporters).
    """

    vertices: np.ndarray
    normals: np.ndarray | None


def _parse_float(token: bytes, path: Path, offset: int, line: bytes) -> float:
    try:
        return float(token.decode("ascii", errors="replace"))
    except ValueError as err:
        msg = f"{path}: malformed number {token!r} at byte offset {offset}: {line!r}"
        raise ValueError(msg) from err


def _parse_ascii_stl(data: bytes, path: Path) -> StlMesh:
    """Parse ASCII STL bytes, tracking byte offsets for error messages.

    Tolerates CRLF line endings, trailing whitespace, ``solid`` /
    ``endsolid`` names, blank lines and empty solids.
    """
    verts: list[list[tuple[float, float, float]]] = []
    normals: list[tuple[float, float, float]] = []
    facet_off = -1
    facet_normal: tuple[float, float, float] | None = None
    facet_verts: list[tuple[float, float, float]] = []
    offset = 0
    for raw in data.split(b"\n"):
        line_off = offset
        offset += len(raw) + 1
        tokens = raw.strip().split()
        if not tokens:
            continue
        kw = tokens[0].lower()
        if kw == b"facet":
            facet_off = line_off
            facet_verts = []
            facet_normal = None
            if len(tokens) >= 5 and tokens[1].lower() == b"normal":
                facet_normal = (
                    _parse_float(tokens[2], path, line_off, raw),
                    _parse_float(tokens[3], path, line_off, raw),
                    _parse_float(tokens[4], path, line_off, raw),
                )
        elif kw == b"vertex":
            if len(tokens) < 4:
                msg = f"{path}: vertex with < 3 coordinates at byte offset {line_off}"
                raise ValueError(msg)
            facet_verts.append(
                (
                    _parse_float(tokens[1], path, line_off, raw),
                    _parse_float(tokens[2], path, line_off, raw),
                    _parse_float(tokens[3], path, line_off, raw),
                )
            )
        elif kw == b"endfacet":
            if len(facet_verts) == 3:
                verts.append(facet_verts)
                normals.append(facet_normal or (0.0, 0.0, 0.0))
            elif len(facet_verts) != 0:
                msg = (
                    f"{path}: facet at byte offset {facet_off} closed with "
                    f"{len(facet_verts)} of 3 vertices at byte offset {line_off}"
                )
                raise ValueError(msg)
            facet_off = -1
            facet_verts = []
            facet_normal = None
    if facet_off >= 0:
        msg = (
            f"{path}: unterminated facet at byte offset {facet_off} "
            f"(file ends at byte offset {len(data)})"
        )
        raise ValueError(msg)
    vertices = np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)
    return StlMesh(vertices=vertices, normals=_validated_normals(np.asarray(normals)))


def _validated_normals(normals: np.ndarray) -> np.ndarray | None:
    """Return usable normals, or ``None`` when the file has none."""
    if normals.size == 0 or not np.isfinite(normals).all():
        return None
    if not np.any(normals):
        return None
    return np.ascontiguousarray(normals, dtype=np.float64)


def _parse_binary_stl(data: bytes, n_tri: int, path: Path) -> StlMesh:
    expected = 84 + 50 * n_tri
    if len(data) < expected:
        n_full = max((len(data) - 84) // 50, 0)
        cut = 84 + 50 * n_full  # end of the last complete record
        msg = (
            f"{path}: binary STL truncated at byte offset {cut}: header declares "
            f"{n_tri} triangles ({expected} bytes) but the file holds {len(data)}"
        )
        raise ValueError(msg)
    records = np.frombuffer(data, dtype=_BIN_RECORD, count=n_tri, offset=84)
    vertices = np.ascontiguousarray(records["verts"], dtype=np.float64)
    normals = np.ascontiguousarray(records["normal"], dtype=np.float64)
    return StlMesh(vertices=vertices, normals=_validated_normals(normals))


def load_stl(path: str | Path) -> StlMesh:
    """Read a binary or ASCII STL file into an :class:`StlMesh`.

    The format is auto-detected: a file whose size exactly matches
    ``84 + 50 * n_tri`` with ``n_tri`` read from the 80-byte header is
    binary (this wins even when the header starts with ``solid``, a
    common binary-exporter quirk); otherwise a file whose first token is
    ``solid`` is parsed as ASCII.  CRLF ASCII, trailing whitespace,
    ``endsolid`` names and empty solids are tolerated.  Truncated files
    are rejected with a :class:`ValueError` that names the byte offset.

    Args:
        path: STL file path.

    Returns:
        Triangle table ``(T, 3, 3)`` float64 (axes ``(x, y, z)``) plus
        facet normals when the file carries usable ones.

    Raises:
        ValueError: If the file is empty, truncated or malformed.
    """
    p = Path(path)
    data = p.read_bytes()
    if not data:
        msg = f"{p}: empty STL file (byte offset 0)"
        raise ValueError(msg)
    n_declared = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else None
    if n_declared and len(data) == 84 + 50 * n_declared:
        return _parse_binary_stl(data, n_declared, p)
    if data.lstrip(b" \t\r\n").startswith(b"solid"):
        mesh = _parse_ascii_stl(data, p)
        if mesh.vertices.shape[0] > 0 or not n_declared:
            return mesh
        # "solid"-headed file that is really a truncated binary STL.
    if n_declared is None:
        msg = (
            f"{p}: not an ASCII STL and too short for a binary STL "
            f"({len(data)} bytes < 84, byte offset 0)"
        )
        raise ValueError(msg)
    return _parse_binary_stl(data, n_declared, p)


# ---------------------------------------------------------------------------
# 2. MESH VALIDATION
# ---------------------------------------------------------------------------


def _as_tris(tris: StlMesh | np.ndarray) -> np.ndarray:
    """Coerce an :class:`StlMesh` or array to a validated ``(T, 3, 3)`` table."""
    arr = tris.vertices if isinstance(tris, StlMesh) else np.asarray(tris)
    if arr.ndim != 3 or arr.shape[1:] != (3, 3):
        msg = f"triangles must have shape (T, 3, 3), got {np.shape(arr)}"
        raise ValueError(msg)
    return np.ascontiguousarray(arr, dtype=np.float64)


def is_watertight(tris: StlMesh | np.ndarray, *, weld_tol: float = 1.0e-6) -> bool:
    """Check that a triangle mesh is closed and orientation-consistent.

    Vertices are welded by rounding to a ``weld_tol`` grid (STL repeats
    shared vertices bit-identically, so exact welding is the norm).  The
    mesh is watertight iff it has no degenerate faces, no directed edge
    ``(a, b)`` appears twice, and every directed edge appears exactly
    once with its opposite ``(b, a)`` -- i.e. a closed, edge-manifold,
    consistently oriented surface.

    Notes:
        Ray-parity voxelisation does not require this (see module
        docstring), but interiors of open meshes are unreliable: report,
        do not require.

    Args:
        tris: ``(T, 3, 3)`` triangle table or :class:`StlMesh`.
        weld_tol: Absolute vertex-welding tolerance in mesh units.

    Returns:
        True when the welded mesh is closed and consistently oriented.
    """
    tri = _as_tris(tris)
    if tri.shape[0] == 0:
        return False
    quantized = np.round(tri.reshape(-1, 3) / weld_tol).astype(np.int64)
    _, inverse = np.unique(quantized, axis=0, return_inverse=True)
    faces = np.reshape(inverse, -1).reshape(-1, 3)  # numpy>=2 gives (N, 1)
    repeated = (
        (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
    )
    if repeated.any():
        return False
    n = int(faces.max()) + 1
    fwd = np.concatenate(
        [
            faces[:, 0] * n + faces[:, 1],
            faces[:, 1] * n + faces[:, 2],
            faces[:, 2] * n + faces[:, 0],
        ]
    )
    if np.unique(fwd).size != fwd.size:
        return False  # duplicated directed edge: non-manifold or torn winding
    rev = np.concatenate(
        [
            faces[:, 1] * n + faces[:, 0],
            faces[:, 2] * n + faces[:, 1],
            faces[:, 0] * n + faces[:, 2],
        ]
    )
    return bool(np.array_equal(np.sort(fwd), np.sort(rev)))


# ---------------------------------------------------------------------------
# 3. RAY-PARITY VOXELISER
# ---------------------------------------------------------------------------


def _grid_geometry(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    dims = tuple(int(v) for v in shape)
    if len(dims) != 3 or any(v <= 0 for v in dims):
        msg = f"shape must be three positive ints (nz, ny, nx), got {shape!r}"
        raise ValueError(msg)
    return dims


def mask_from_stl(
    tris: StlMesh | np.ndarray,
    shape: tuple[int, int, int],
    *,
    origin: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    spacing: float | tuple[float, float, float] | np.ndarray = 1.0,
    axis: int = 0,
    robust: bool = True,
    column_chunk: int | None = None,
) -> np.ndarray:
    """Rasterise a triangle mesh to a boolean solid mask by ray parity.

    One ray is cast per voxel column along mesh axis ``axis`` (default 0
    = x, the streamwise axis) using vectorised Moller-Trumbore
    intersection over column batches; a voxel is inside iff an odd number
    of crossings lies strictly ahead of it.  See the module docstring for
    the ``robust`` tie-breaking rules.

    Args:
        tris: ``(T, 3, 3)`` triangle table or :class:`StlMesh` (mesh axes
            ``(x, y, z)``).  An empty table yields an all-False mask.
        shape: Grid shape ``(nz, ny, nx)`` (B4 convention: mesh x on the
            last array axis).
        origin: Mesh ``(x, y, z)`` coordinates of the centre of voxel
            ``[0, 0, 0]``.
        spacing: Isotropic cell size (scalar) or per-axis ``(dx, dy, dz)``,
            all positive.
        axis: Mesh axis the ray travels along (0=x streamwise, 1=y, 2=z).
            The output layout is independent of this choice.
        robust: Use the deterministic perturbation + strict-bounds
            crossing rule (default).  ``False`` keeps the naive inclusive
            rule, which double-counts shared-edge hits on grid-aligned
            meshes -- kept only to demonstrate the failure mode.
        column_chunk: Number of voxel columns per vectorised tile
            (memory knob; default sized for ~1e6 ray-triangle pairs).

    Returns:
        Bool array of ``shape`` in ``(nz, ny, nx)`` layout, True inside
        the closed surface.
    """
    dims = _grid_geometry(shape)
    nz, ny, nx = dims
    counts = (nx, ny, nz)
    if axis not in (0, 1, 2):
        msg = f"axis must be 0 (x), 1 (y) or 2 (z), got {axis!r}"
        raise ValueError(msg)
    org = np.asarray(origin, dtype=np.float64).reshape(-1)
    spc_arr = np.asarray(spacing, dtype=np.float64).reshape(-1)
    if org.size != 3 or spc_arr.size not in (1, 3):
        msg = f"origin needs 3 entries and spacing 1 or 3; got {origin!r}, {spacing!r}"
        raise ValueError(msg)
    spc = np.full(3, float(spc_arr[0])) if spc_arr.size == 1 else spc_arr
    if not np.all(spc > 0.0) or not np.isfinite(org).all() or not np.isfinite(spc).all():
        msg = f"origin/spacing must be finite, spacing positive; got {origin!r}, {spacing!r}"
        raise ValueError(msg)
    tri = _as_tris(tris)
    out = np.zeros((nz, ny, nx), dtype=bool)
    t1, t2 = (axis + 1) % 3, (axis + 2) % 3
    n_axis, n_t1, n_t2 = counts[axis], counts[t1], counts[t2]

    # Cell-centre sample coordinates along each mesh axis.
    samples = [org[k] + (np.arange(counts[k]) + 0.5) * spc[k] for k in range(3)]

    # Cull triangles that cannot intersect any ray (transverse overlap only;
    # crossings beyond the domain still flip parity, so no axis culling).
    if tri.shape[0]:
        vmin, vmax = tri.min(axis=1), tri.max(axis=1)
        keep = np.ones(tri.shape[0], dtype=bool)
        for k in (t1, t2):
            keep &= (vmin[:, k] <= samples[k][-1]) & (vmax[:, k] >= samples[k][0])
        tri = tri[keep]
    if tri.shape[0] == 0:
        return out

    # Pre-computed Moller-Trumbore terms (ray direction = unit axis vector).
    direction = np.zeros(3)
    direction[axis] = 1.0
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    pvec = np.cross(direction, e2)  # (T, 3)
    det = np.einsum("tk,tk->t", e1, pvec)
    scale = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
    parallel = np.abs(det) <= 1.0e-12 * np.maximum(scale, 1.0e-300)
    inv_det = np.zeros_like(det)
    inv_det[~parallel] = 1.0 / det[~parallel]
    v0 = tri[:, 0]

    edges = (np.arange(n_axis) + 0.5) * spc[axis]  # crossing thresholds in t
    n_cols = n_t1 * n_t2
    # whole rows of the (t1, t2) column grid per chunk so each chunk is a
    # rectangular block that can be scattered with plain slices
    chunk = column_chunk or max(n_t2, (_TILE_BUDGET // max(tri.shape[0], 1) // n_t2) * n_t2)
    # mesh axis -> array axis of (nz, ny, nx): x->2, y->1, z->0
    array_axis = {0: 2, 1: 1, 2: 0}
    mesh_pos = {t1: 0, t2: 1, axis: 2}  # position of each mesh axis in the block
    transpose_order = tuple(mesh_pos[2 - a] for a in range(3))
    for c0 in range(0, n_cols, chunk):
        c1 = min(c0 + chunk, n_cols)
        idx = np.arange(c0, c1)
        i1 = idx // n_t2
        i2 = idx % n_t2
        rays = np.empty((idx.size, 3))
        rays[:, axis] = org[axis]
        rays[:, t1] = samples[t1][i1]
        rays[:, t2] = samples[t2][i2]
        if robust:
            rays[:, t1] += _EPS_T1 * spc[t1]
            rays[:, t2] += _EPS_T2 * spc[t2]

        tvec = rays[:, None, :] - v0[None, :, :]  # (R, T, 3)
        u = np.einsum("rtk,tk->rt", tvec, pvec) * inv_det
        qvec = np.cross(tvec, e1[None, :, :])  # (R, T, 3)
        v = qvec[:, :, axis] * inv_det
        t = np.einsum("rtk,tk->rt", qvec, e2) * inv_det
        if robust:
            inside = (u > 0.0) & (v > 0.0) & (u + v < 1.0)
        else:
            inside = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
        hit = inside & ~parallel & (t > 0.0)

        rows = np.nonzero(hit.ravel())[0]
        parity = np.zeros((idx.size, n_axis), dtype=bool)
        if rows.size:
            total = hit.sum(axis=1)
            t_hit = t.ravel()[rows]
            b = np.searchsorted(edges, t_hit, side="right")
            in_grid = b < n_axis
            row_idx = rows // hit.shape[1]  # chunk-local column index
            key = row_idx[in_grid] * n_axis + b[in_grid]
            counts_ahead = np.bincount(key, minlength=idx.size * n_axis).reshape(idx.size, n_axis)
            passed = total[:, None] - np.cumsum(counts_ahead, axis=1)
            parity = (passed & 1).astype(bool)
        block = np.transpose(
            parity.reshape(-1, n_t2, n_axis), transpose_order
        )  # array order (z, y, x)
        sl = [slice(None)] * 3
        sl[array_axis[t1]] = slice(int(i1.min()), int(i1.max()) + 1)
        out[tuple(sl)] = block
    return out


# ---------------------------------------------------------------------------
# 4. CANONICAL-FRAME PLACEMENT
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Placement:
    """Result of :func:`place_on_grid`.

    Attributes:
        tris: Transformed triangle table in voxel-index coordinates:
            feeding ``mask_from_stl(tris, shape, origin=origin,
            spacing=spacing)`` reproduces the placed mask.
        origin: Mesh coordinates of the centre of voxel ``[0, 0, 0]``
            (always ``(0, 0, 0)`` for the canonical placement).
        spacing: Isotropic cell size (always ``1.0``).
        scale: Uniform mesh -> voxel scale factor that was applied.
        streamwise_extent: Streamwise (x) extent of the placed mesh in
            voxels, ``streamwise_frac * nx`` by construction.
    """

    tris: np.ndarray
    origin: np.ndarray
    spacing: float
    scale: float
    streamwise_extent: float


def place_on_grid(
    tris: StlMesh | np.ndarray,
    shape: tuple[int, int, int],
    *,
    scale: float | None = None,
    center_frac: tuple[float, float, float] = (0.35, 0.5, 0.5),
    streamwise_frac: float = 0.6,
) -> Placement:
    """Place a mesh in the B4 canonical frame (uniform scale, no rotation).

    The mesh bbox is scaled uniformly (aspect preserved) so the
    streamwise (x) extent covers ``streamwise_frac * nx`` voxels, then
    translated so the bbox centre lands on ``center_frac * (nx, ny, nz)``
    in mesh ``(x, y, z)`` order.  With the defaults this reproduces the
    SUBOFF convention -- hull centred at ``cx = 0.35 * nx`` with length
    ``0.6 * nx``, so the nose sits at ``0.05 * nx`` upstream -- assuming
    the STL was authored with streamwise = mesh x and the nose toward
    -x.  Transverse extents are *not* stretched to fill ny/nz; meshes
    with extreme aspect may overflow the grid (caller's responsibility).

    Args:
        tris: ``(T, 3, 3)`` triangle table or :class:`StlMesh`.
        shape: Grid shape ``(nz, ny, nx)``.
        scale: Explicit uniform scale factor; when ``None`` it is derived
            from ``streamwise_frac``.
        center_frac: Bbox-centre target as a fraction of ``(nx, ny, nz)``
            in mesh ``(x, y, z)`` order.
        streamwise_frac: Target streamwise extent as a fraction of nx
            (ignored when ``scale`` is given).

    Returns:
        The :class:`Placement`; triangles are in voxel-index coordinates
        (``origin=(0,0,0)``, ``spacing=1.0``).

    Raises:
        ValueError: If the mesh has zero streamwise extent, or any
            argument is out of range.
    """
    dims = _grid_geometry(shape)
    nz, ny, nx = dims
    tri = _as_tris(tris)
    frac = np.asarray(center_frac, dtype=np.float64).reshape(3)
    if not (np.isfinite(frac).all() and np.all(frac > 0.0) and np.all(frac < 1.0)):
        msg = f"center_frac must be three fractions in (0, 1), got {center_frac!r}"
        raise ValueError(msg)
    if tri.shape[0] == 0:
        msg = "cannot place an empty mesh"
        raise ValueError(msg)
    lo = tri.reshape(-1, 3).min(axis=0)
    hi = tri.reshape(-1, 3).max(axis=0)
    extent = hi - lo
    if extent[0] <= 0.0:
        msg = f"mesh has zero streamwise (x) extent; bbox = {lo} .. {hi}"
        raise ValueError(msg)
    if scale is None:
        if not 0.0 < streamwise_frac <= 1.0:
            msg = f"streamwise_frac must be in (0, 1], got {streamwise_frac!r}"
            raise ValueError(msg)
        scale = streamwise_frac * nx / extent[0]
    elif scale <= 0.0 or not np.isfinite(scale):
        msg = f"scale must be a positive finite float, got {scale!r}"
        raise ValueError(msg)
    target = frac * np.asarray([nx, ny, nz], dtype=np.float64)
    placed = (tri - 0.5 * (lo + hi)) * scale + target
    return Placement(
        tris=np.ascontiguousarray(placed, dtype=np.float64),
        origin=np.zeros(3),
        spacing=1.0,
        scale=float(scale),
        streamwise_extent=float(extent[0] * scale),
    )


# ---------------------------------------------------------------------------
# 5. SDF SEAM (see docs/voxelize_stl_20260824.md for the integration plan)
# ---------------------------------------------------------------------------


def _boundary_cells(mask: np.ndarray) -> np.ndarray:
    """Solid cells with at least one fluid 6-neighbour, as ``(B, 3)`` coords."""
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    flips = np.zeros(mask.shape, dtype=bool)
    for d in range(3):
        for step in (-1, 1):
            src = np.roll(padded, step, axis=d)[tuple(slice(1, s + 1) for s in mask.shape)]
            flips |= src != mask
    return np.argwhere(mask & flips)


def _brute_force_distance(
    shape: tuple[int, int, int], boundary: np.ndarray, spacing: float
) -> np.ndarray:
    """Exact Euclidean distance of every cell to the boundary set."""
    flat = np.empty(np.prod(shape), dtype=np.float64)
    b = boundary.astype(np.float64)
    rows_per_tile = max(1, 4_000_000 // max(b.shape[0], 1))
    for r0 in range(0, flat.size, rows_per_tile):
        r1 = min(r0 + rows_per_tile, flat.size)
        coords = np.stack(np.unravel_index(np.arange(r0, r1), shape), axis=1).astype(np.float64)
        d2 = (coords[:, 0, None] - b[None, :, 0]) ** 2
        for k in (1, 2):
            d2 += (coords[:, k, None] - b[None, :, k]) ** 2
        flat[r0:r1] = np.sqrt(d2.min(axis=1))
    return flat.reshape(shape) * spacing


_CHAMFER_WEIGHTS: tuple[tuple[int, int, int, int], ...] = tuple(
    (dz, dy, dx, {1: 3, 2: 4, 3: 5}[abs(dz) + abs(dy) + abs(dx)])
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if (dz, dy, dx) != (0, 0, 0)
    and (dz < 0 or (dz == 0 and dy < 0) or (dz == 0 and dy == 0 and dx < 0))
)


def _chamfer_distance(
    mask_shape: tuple[int, int, int], boundary: np.ndarray, spacing: float
) -> np.ndarray:
    """Two-pass 3-4-5 chamfer distance transform (no scipy).

    Each pass sweeps the (z, y) rows in scan order, relaxing the 12
    cross-row neighbourhood offsets and then propagating exactly along
    the row with the face weight (a vectorised 1-D min-plus scan, so
    multi-step straight paths accumulate within a single pass).  Weights
    3/4/5 (face/edge/corner neighbours) approximate the Euclidean
    distance with the classic Borgefors-type bound for the 3x3x3 chamfer
    mask: worst case about 11% relative error on shallow diagonals
    (displacement (3,1,1) costs (5+3+3)/3 = 3.67 against sqrt(11) =
    3.32), typically a few percent on voxelised smooth surfaces;
    verified against the brute-force distance in
    ``tests/test_voxelize.py``.
    """
    d = np.full(mask_shape, np.inf)
    d[boundary[:, 0], boundary[:, 1], boundary[:, 2]] = 0.0
    nz, ny, nx = mask_shape
    # cross-row offsets only (dz, dy) != (0, 0); same-row propagation is
    # handled exactly by the in-row scan
    cross = tuple(wt for wt in _CHAMFER_WEIGHTS if (wt[0], wt[1]) != (0, 0))
    face = 3  # face weight in 3-4-5 units
    ramp = np.arange(nx, dtype=np.float64)

    def pass_over(
        offsets: tuple[tuple[int, int, int, int], ...], k_range: range, j_range: range
    ) -> None:
        for k in k_range:
            for j in j_range:
                cur = d[k, j].copy()
                for dz, dy, dx, w in offsets:
                    kk, jj = k + dz, j + dy
                    if kk < 0 or kk >= nz or jj < 0 or jj >= ny:
                        continue
                    src = d[kk, jj]
                    if dx == 0:
                        cur = np.minimum(cur, src + w)
                    elif dx < 0:
                        cur[1:] = np.minimum(cur[1:], src[:-1] + w)
                    else:
                        cur[:-1] = np.minimum(cur[:-1], src[1:] + w)
                # exact in-row propagation with the face weight
                fwd = np.minimum.accumulate(cur - face * ramp) + face * ramp
                bwd = np.minimum.accumulate((cur + face * ramp)[::-1])[::-1] - face * ramp
                d[k, j] = np.minimum(fwd, bwd)

    pass_over(cross, range(nz), range(ny))
    pass_over(
        tuple((-dz, -dy, -dx, w) for dz, dy, dx, w in cross),
        range(nz - 1, -1, -1),
        range(ny - 1, -1, -1),
    )
    return d * (spacing / 3.0)


def sdf_from_mask(
    mask: np.ndarray,
    *,
    spacing: float = 1.0,
    exact: bool | None = None,
) -> np.ndarray:
    """Signed distance field of a solid mask (negative inside, positive out).

    Thin seam for the SDF encoder input contract -- deliberately *not*
    the boundary-restricted exact EDT that lives in ``geom_encoder``
    (PR #235): duplicating that implementation here would create a merge
    conflict.  Post-#235-merge, callers should prefer
    ``geom_encoder`` for the encoder input and keep this function for
    standalone / diagnostic use.

    Two self-consistent back-ends:

    * exact brute-force distance to the set of boundary cells (default
      for grids up to 32^3 cells, or whenever ``exact=True``; practical
      up to ~64^3);
    * a two-pass 3-4-5 chamfer with <= ~11% worst-case relative error
      on shallow diagonals (a few percent typical) for larger grids;
      the bound is checked against the brute-force distance in the
      test suite.

    The boundary set is the solid cells with at least one fluid
    6-neighbour (the domain border counts as fluid).

    Args:
        mask: Bool array ``(nz, ny, nx)``.
        spacing: Isotropic cell size.
        exact: Force the brute-force (``True``) or chamfer (``False``)
            back-end; ``None`` picks exact for small grids.

    Returns:
        Float64 array of ``mask.shape``; negative inside the solid.

    Raises:
        ValueError: If the mask has no boundary (uniformly True/False)
            or is not a 3-D array.
    """
    arr = np.asarray(mask)
    if arr.ndim != 3:
        msg = f"mask must be 3-D (nz, ny, nx), got shape {arr.shape}"
        raise ValueError(msg)
    if not spacing > 0.0 or not np.isfinite(spacing):
        msg = f"spacing must be a positive finite float, got {spacing!r}"
        raise ValueError(msg)
    solid = arr.astype(bool)
    boundary = _boundary_cells(solid)
    if boundary.shape[0] == 0:
        msg = "mask has no boundary cells (all True or all False); SDF undefined"
        raise ValueError(msg)
    use_exact = (solid.size <= _EXACT_DEFAULT_LIMIT) if exact is None else exact
    if use_exact:
        dist = _brute_force_distance(solid.shape, boundary, spacing)
    else:
        dist = _chamfer_distance(solid.shape, boundary, spacing)
    return np.where(solid, -dist, dist)
