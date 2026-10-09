"""Boxel STL ingestion for the drag-surrogate geometry path.

The L2 arbitrary-geometry line consumes hulls as ``(nz, ny, nx)`` boolean
solid masks: the corpus path builds them from SUBOFF CAD parameters
(:func:`tensorlbm.suboff_cad.build_suboff_mask`) and the 2026-09-04 e2e
campaign proved the CAD -> mask -> SDF chain regenerates the stored SDFs
bit-exactly for 121/122 corpus designs.  An external user, however, brings
a hull as an **STL mesh**.  This module closes that seam with two legs:

1. **Export** — :func:`mask_to_stl` writes a *boxel* (voxel-solid) mesh:
   one quad, split into two triangles, per **exposed** voxel face (solid
   on one side, fluid or domain border on the other).  The result is
   closed and edge-manifold by construction (watertight, consistently
   outward-oriented) and, fed back through :func:`stl_to_mask` on the
   same grid, reproduces the source mask **bit-exactly** — the
   round-trip guarantee a tessellated (smooth-CAD) mesh cannot give.
2. **Ingest** — :func:`stl_to_mask` re-voxelises any closed STL (binary
   or ASCII, bytes or file path) and :func:`stl_to_sdf` chains it into
   the *corpus* SDF path (:func:`tensorlbm.ai.geom_encoder.sdf_volume`
   — exact EDT, +-8-voxel clip to [-1, 1], stride-2 mean pool), so an
   STL-sourced hull meets the two-stage surrogate exactly like a
   CAD-sourced one.

This module deliberately reuses :mod:`tensorlbm.voxelize` (loader,
watertightness check, vectorised Moller-Trumbore ray-parity voxeliser)
and :mod:`tensorlbm.ai.geom_encoder` (the SDF chain) instead of
reimplementing them; only the boxel exporter and the bytes-level STL
entry point are new.

Conventions (shared with :mod:`tensorlbm.voxelize`):

- Masks are ``(nz, ny, nx)`` bool arrays, mesh **x on the last array
  axis**; mesh axes are ``(x, y, z)``.
- :func:`mask_to_stl` emits voxel ``(iz, iy, ix)`` as the unit box
  ``[ix, ix+1] x [iy, iy+1] x [iz, iz+1]`` in mesh coordinates
  (integer node grid).  :func:`stl_to_mask` with the default
  ``origin=(0, 0, 0)`` and ``spacing=1`` samples voxel centres at
  ``(ix + 0.5, iy + 0.5, iz + 0.5)`` — exactly the box centres — so the
  two are inverse operations on the same grid.
- The domain border counts as fluid: a mask that is solid to the grid
  edge still gets its outer boundary faces emitted.

Watertightness requirement (hard): ray parity is well-defined for any
mesh, but the interior of a mesh that is not closed *leaks* — some
columns cross the surface an odd number of times in the wrong place, and
the mask gains or loses whole wedges / column streaks.  :func:`stl_to_mask`
therefore runs :func:`tensorlbm.voxelize.is_watertight` by default and
raises on failure; pass ``require_watertight=False`` only for diagnosis
(the returned mask is then unreliable in exactly this streak sense).

Known degenerate case: a mask whose solid voxels touch only along edges
or corners (e.g. a checkerboard) has a *pinched* boundary — four faces
share an edge — which is closed but not edge-manifold, so
``is_watertight`` rejects it even though :func:`mask_to_stl` emitted a
valid closed surface and ray parity still round-trips it exactly
(``require_watertight=False``).  Face-connected solids, the physical
case for hull masks, are unaffected.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .voxelize import (
    StlMesh,
    _parse_ascii_stl,
    _parse_binary_stl,
    is_watertight,
    load_stl,
    mask_from_stl,
)

__all__ = [
    "mask_to_stl",
    "stl_to_mask",
    "stl_to_sdf",
    "write_mask_stl",
]

#: Binary STL record layout (50 bytes: normal + 3 vertices + attribute).
_BIN_RECORD = np.dtype([("normal", "<f4", (3,)), ("verts", "<f4", (3, 3)), ("attr", "<u2")])

#: Transverse axes ``(u, v)`` per mesh axis ``a`` with ``e_u x e_v = e_a``:
#: x = y x z, y = z x x, z = x x y.  A quad traversed A -> A+e_u ->
#: A+e_u+e_v -> A+e_v then split (A, B, D) / (B, C, D) faces outward
#: along +e_a; swapping (u, v) flips the outward sense.
_UV: dict[int, tuple[int, int]] = {0: (1, 2), 1: (2, 0), 2: (0, 1)}

#: Placeholder path name used in byte-buffer parse error messages.
_BYTES_NAME = "<stl bytes>"


# ---------------------------------------------------------------------------
# 1. BOXEL EXPORTER (mask -> STL)
# ---------------------------------------------------------------------------


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    """Validate and coerce ``mask`` to a 3-D boolean ``(nz, ny, nx)`` array."""
    arr = np.asarray(mask)
    if arr.dtype != np.bool_:
        msg = f"mask must be a boolean (nz, ny, nx) array, got dtype {arr.dtype}"
        raise TypeError(msg)
    if arr.ndim != 3:
        msg = f"mask must be a boolean (nz, ny, nx) array, got shape {arr.shape}"
        raise ValueError(msg)
    return arr


def _mask_to_triangles(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exposed-face triangles and outward normals of a boxel mask.

    Returns ``(triangles, normals)`` with shapes ``(2 * F, 3, 3)`` and
    ``(2 * F, 3)`` — two triangles per exposed quad face ``F``, mesh axes
    ``(x, y, z)`` on the last vertex axis, voxel boxes on the integer
    node grid.  Interior solid-solid faces are never emitted: they would
    cancel in ray parity anyway and tear the manifold.
    """
    arr = _as_bool_mask(mask)
    padded = np.pad(arr, 1, mode="constant", constant_values=False)
    tri_blocks: list[np.ndarray] = []
    nrm_blocks: list[np.ndarray] = []
    # Array axes are (z, y, x); mesh axes are (x, y, z) — the neighbour
    # test walks the ARRAY axis, the quad construction below works in MESH
    # axes, so map one to the other (mesh x = array 2, y = 1, z = 0).
    for axis_arr in range(3):
        axis = 2 - axis_arr
        for sign in (1, -1):
            sl = [slice(1, -1)] * 3
            sl[axis_arr] = slice(2, None) if sign > 0 else slice(0, -2)
            exposed = arr & ~padded[tuple(sl)]
            faces = np.nonzero(exposed)  # array axes (z, y, x)
            n_faces = faces[0].size
            if n_faces == 0:
                continue
            # Voxel indices in MESH axis order (x, y, z).
            vox = np.stack([faces[2], faces[1], faces[0]], axis=1).astype(np.float64)
            u, v = _UV[axis] if sign > 0 else (_UV[axis][1], _UV[axis][0])
            base = vox.copy()
            base[:, axis] += 1.0 if sign > 0 else 0.0
            eu = np.zeros(3)
            eu[u] = 1.0
            ev = np.zeros(3)
            ev[v] = 1.0
            a = base
            b = base + eu
            c = base + eu + ev
            d = base + ev
            tri_blocks.append(
                np.stack([np.stack([a, b, d], axis=1), np.stack([b, c, d], axis=1)]).reshape(
                    -1, 3, 3
                )
            )
            normal = np.zeros((2 * n_faces, 3))
            normal[:, axis] = float(sign)
            nrm_blocks.append(normal)
    if not tri_blocks:
        return np.zeros((0, 3, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(tri_blocks), np.concatenate(nrm_blocks)


def _to_ascii_stl(triangles: np.ndarray, normals: np.ndarray) -> bytes:
    """Serialise a triangle table as an ASCII STL payload."""
    lines = ["solid tensorlbm_boxel"]
    for (v0, v1, v2), n in zip(triangles, normals, strict=True):
        lines.append(f"  facet normal {n[0]:.9e} {n[1]:.9e} {n[2]:.9e}")
        lines.append("    outer loop")
        for vert in (v0, v1, v2):
            lines.append(f"      vertex {vert[0]:.9e} {vert[1]:.9e} {vert[2]:.9e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid tensorlbm_boxel")
    return ("\n".join(lines) + "\n").encode("ascii")


def mask_to_stl(mask: np.ndarray, *, binary: bool = True) -> bytes:
    """Write a boolean solid mask as a watertight boxel STL.

    One quad — two triangles, consistently outward-oriented — is emitted
    per exposed voxel face (solid vs fluid or domain border); interior
    solid-solid faces are omitted, so the mesh is closed and
    edge-manifold by construction and inverts bit-exactly through
    :func:`stl_to_mask` on the same grid with default ``origin`` /
    ``spacing``.

    Args:
        mask: Boolean ``(nz, ny, nx)`` solid mask (mesh x on the last
            axis).  Voxel ``(iz, iy, ix)`` is written as the unit box on
            the integer node grid ``[ix, ix+1] x [iy, iy+1] x
            [iz, iz+1]``.
        binary: Emit binary STL (default, compact).  ``False`` emits
            ASCII STL with full float precision.

    Returns:
        The STL payload as ``bytes``.

    Raises:
        TypeError: If ``mask`` is not boolean.
        ValueError: If ``mask`` is not 3-D or has no solid voxels (an
            empty STL cannot round-trip; treat it as an upstream error).
    """
    arr = _as_bool_mask(mask)
    if not arr.any():
        msg = "mask has no solid voxels; refusing to write an empty (0-triangle) STL"
        raise ValueError(msg)
    triangles, normals = _mask_to_triangles(arr)
    if binary:
        header = b"tensorlbm boxel mask_to_stl".ljust(80, b"\0")
        records = np.zeros(triangles.shape[0], dtype=_BIN_RECORD)
        records["normal"] = normals.astype("<f4")
        records["verts"] = triangles.astype("<f4")
        return header + struct.pack("<I", triangles.shape[0]) + records.tobytes()
    return _to_ascii_stl(triangles, normals)


def write_mask_stl(path: str | Path, mask: np.ndarray, *, binary: bool = True) -> Path:
    """Write :func:`mask_to_stl` output to a file.

    Args:
        path: Destination STL file path (parents are not created).
        mask: Boolean ``(nz, ny, nx)`` solid mask.
        binary: Binary (default) or ASCII STL.

    Returns:
        The resolved path, for chaining.
    """
    p = Path(path)
    p.write_bytes(mask_to_stl(mask, binary=binary))
    return p


# ---------------------------------------------------------------------------
# 2. STL -> MASK (ingestion)
# ---------------------------------------------------------------------------


def _mesh_from_bytes(data: bytes | bytearray) -> StlMesh:
    """Parse an in-memory STL payload (binary or ASCII, auto-detected).

    Detection mirrors :func:`tensorlbm.voxelize.load_stl` and reuses its
    parsers: a size matching ``84 + 50 * n_tri`` wins even when the
    80-byte header starts with ``solid`` (a common binary-exporter
    quirk); otherwise a ``solid``-headed buffer is parsed as ASCII.
    Truncated payloads raise :class:`ValueError` naming the byte offset.
    """
    buf = bytes(data)
    name = Path(_BYTES_NAME)
    if not buf:
        msg = f"{name}: empty STL buffer (byte offset 0)"
        raise ValueError(msg)
    n_declared = struct.unpack_from("<I", buf, 80)[0] if len(buf) >= 84 else None
    if n_declared and len(buf) == 84 + 50 * n_declared:
        return _parse_binary_stl(buf, n_declared, name)
    if buf.lstrip(b" \t\r\n").startswith(b"solid"):
        mesh = _parse_ascii_stl(buf, name)
        if mesh.vertices.shape[0] > 0 or not n_declared:
            return mesh
        # "solid"-headed buffer that is really a truncated binary STL.
    if n_declared is None:
        msg = (
            f"{name}: not an ASCII STL and too short for a binary STL "
            f"({len(buf)} bytes < 84, byte offset 0)"
        )
        raise ValueError(msg)
    return _parse_binary_stl(buf, n_declared, name)


def _to_mesh(source: StlMesh | np.ndarray | bytes | bytearray | str | Path) -> StlMesh:
    """Coerce an STL source (path / bytes / mesh / triangle table) to StlMesh."""
    if isinstance(source, StlMesh):
        return source
    if isinstance(source, (bytes, bytearray)):
        return _mesh_from_bytes(source)
    if isinstance(source, (str, Path)):
        return load_stl(source)
    return StlMesh(vertices=np.ascontiguousarray(source, dtype=np.float64), normals=None)


def stl_to_mask(
    source: StlMesh | np.ndarray | bytes | bytearray | str | Path,
    shape: tuple[int, int, int],
    *,
    origin: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    spacing: float | tuple[float, float, float] | np.ndarray = 1.0,
    axis: int = 0,
    robust: bool = True,
    require_watertight: bool = True,
    column_chunk: int | None = None,
) -> np.ndarray:
    """Re-voxelise an STL mesh into a boolean solid mask.

    The mesh is rasterised by vectorised Moller-Trumbore ray parity along
    mesh axis ``axis`` (:func:`tensorlbm.voxelize.mask_from_stl`); a voxel
    is solid iff an odd number of surface crossings lies strictly ahead
    of its centre.  With :func:`mask_to_stl` output, default
    ``origin`` / ``spacing`` and the same grid, this is the **exact**
    inverse (pinned by ``tests/test_geometry_stl.py``).

    Watertightness: the mesh must be closed and orientation-consistent
    (checked via :func:`tensorlbm.voxelize.is_watertight` unless
    ``require_watertight=False``).  An open mesh leaks parity — whole
    wedges or full voxel columns flip solid from the breach to the grid
    edge — and no voxeliser can repair that; fix the mesh upstream.

    Args:
        source: STL payload as a file path, raw ``bytes`` (binary or
            ASCII), an :class:`~tensorlbm.voxelize.StlMesh`, or a
            ``(T, 3, 3)`` triangle table (mesh axes ``(x, y, z)``).
        shape: Grid shape ``(nz, ny, nx)`` (mesh x on the last axis).
        origin: Mesh coordinates of the centre of voxel ``[0, 0, 0]``.
            The default ``(0, 0, 0)`` pairs with :func:`mask_to_stl`'s
            integer node boxes (centres at half-integers).
        spacing: Isotropic cell size or per-axis ``(dx, dy, dz)``.
        axis: Mesh axis the ray travels along (0=x streamwise default).
        robust: Deterministic sub-cell ray perturbation + strict
            barycentric bounds (default; see :mod:`tensorlbm.voxelize`).
        require_watertight: Raise on non-closed / non-manifold meshes
            (default).  ``False`` returns the (unreliable) parity mask
            for diagnosis only.
        column_chunk: Memory knob forwarded to
            :func:`tensorlbm.voxelize.mask_from_stl`.

    Returns:
        Boolean array of ``shape``, ``True`` inside the closed surface.

    Raises:
        ValueError: If the mesh is not watertight (or empty) and
            ``require_watertight`` is set, or the payload is malformed.
    """
    mesh = _to_mesh(source)
    if require_watertight and not is_watertight(mesh):
        n_tri = int(mesh.vertices.shape[0])
        msg = (
            "STL mesh is not watertight "
            f"({n_tri} triangles; closed + orientation-consistent required). "
            "Ray parity would leak: whole voxel columns flip solid/fluid "
            "across the breach. Repair the mesh, or pass "
            "require_watertight=False to inspect the (unreliable) parity mask."
        )
        raise ValueError(msg)
    return mask_from_stl(
        mesh,
        shape,
        origin=origin,
        spacing=spacing,
        axis=axis,
        robust=robust,
        column_chunk=column_chunk,
    )


# ---------------------------------------------------------------------------
# 3. STL -> SDF (the surrogate input contract)
# ---------------------------------------------------------------------------


def stl_to_sdf(
    source: StlMesh | np.ndarray | bytes | bytearray | str | Path,
    shape: tuple[int, int, int],
    *,
    origin: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    spacing: float | tuple[float, float, float] | np.ndarray = 1.0,
    axis: int = 0,
    robust: bool = True,
    require_watertight: bool = True,
    clip: float | None = None,
    pool: int | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Ingest an STL hull straight to the surrogate SDF volume.

    Convenience chain ``stl_to_mask`` -> :func:`tensorlbm.ai.geom_encoder.sdf_volume`
    (exact boundary-restricted EDT, ``phi < 0`` inside, clip to
    +-``SDF_CLIP_VOXELS`` scaled to ``[-1, 1]``, stride-``SDF_POOL_STRIDE``
    mean pool) — the *same* clip/pool chain the CAD corpus path uses, so
    STL-sourced and CAD-sourced hulls meet the two-stage surrogate under
    an identical input contract.  The SDF seam is imported, not
    reimplemented.

    Args:
        source: STL payload (path / bytes / :class:`StlMesh` / triangle
            table), as accepted by :func:`stl_to_mask`.
        shape: Grid shape ``(nz, ny, nx)`` of the raw mask (production
            ``(64, 64, 128)`` pools to ``(32, 32, 64)``).
        origin: Mesh coordinates of the centre of voxel ``[0, 0, 0]``.
        spacing: Isotropic cell size or per-axis ``(dx, dy, dz)``.
        axis: Mesh axis the ray travels along (default 0 = x).
        robust: Forwarded to :func:`stl_to_mask`.
        require_watertight: Forwarded to :func:`stl_to_mask`.
        clip: Signed-distance clip in voxels (default
            ``geom_encoder.SDF_CLIP_VOXELS`` = 8).
        pool: Mean-pool stride (default ``geom_encoder.SDF_POOL_STRIDE``
            = 2).
        device: Torch device for the EDT computation (``"cpu"`` or
            ``"cuda"``; both are deterministic per device).

    Returns:
        ``float32`` array ``(D', H', W')`` in ``[-1, 1]`` — the pooled
        SDF volume the SDF encoder consumes.

    Raises:
        ValueError: As :func:`stl_to_mask` (malformed payload,
            non-watertight mesh).
    """
    import torch

    from .ai import geom_encoder

    mask = stl_to_mask(
        source,
        shape,
        origin=origin,
        spacing=spacing,
        axis=axis,
        robust=robust,
        require_watertight=require_watertight,
    )
    volume = geom_encoder.sdf_volume(
        torch.as_tensor(mask, device=device),
        clip=geom_encoder.SDF_CLIP_VOXELS if clip is None else clip,
        pool=geom_encoder.SDF_POOL_STRIDE if pool is None else pool,
    )
    return volume[0, 0].detach().cpu().numpy()
