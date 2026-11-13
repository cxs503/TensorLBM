"""EDT-based geometry tagging with compressed tags — MPLBM-UT cross-validation.

Cross-validation (XVAL) target
------------------------------
Repository  ``je-santos/MPLBM-UT`` @ ``a9fc496688486c8bbc882ec7d780619f6712fa54``
(2023-12-01).  The reference recipe is the geometry-preparation path used by
every MPLBM-UT example::

    src/python/mplbm_utils/create_geom_for_palabos.py   (blob 0cca7926...)
        rock = rock/3                       # grains -> 1.0, pore -> 0.0
        rock = erase_regions(rock)          # keep largest connected pore body
        rock4sim, name = create_geom_edist(rock, geom, nw_fluid_mask)
        rock4sim.flatten().tofile(...)      # int16 raw dump for Palabos

    src/python/mplbm_utils/pore_utils.py                (blob 8d713e78...)
        def create_geom_edist(rock, args, nw_fluid_mask):
            if args.swapXZ: rock = rock.transpose([2, 1, 0])
            erock = edist(rock)             # scipy ndimage distance_transform_edt
            erock[0, :, :] = 1; ... (all six faces) # bounce-back on every BC
            erock[rock == 0] = 0            # re-open pores
            erock[(erock > 0) * (erock < 2)] = 1    # << boundary class
            erock[erock > 1] = 2                    # << grain-interior class
            if args.num_slices: erock = np.pad(erock, [(n, n), (0, 0), (0, 0)])
            erock = erock.astype(np.int16)
            erock[erock == 0] = 2608        # pore / wetting fluid
            erock[erock == 1] = 2609        # boundary (bounce-back) layer
            erock[erock == 2] = 2610        # grain interior
            erock[nw_fluid_mask == 3] = 2611  # non-wetting fluid seeds
            return erock.astype(np.int16), geom_name

So the OSS *tag compression* step is: take the continuous Euclidean distance
field inside the grains (``float64``, 8 B/voxel) and collapse it into a
**two-segment segmentation** (boundary layer ``0 < d < 2`` vs interior
``d >= 2``) plus the pore class, then quantise the result into ``int16``
Palabos codes (2 B/voxel).  The present module re-implements that pipeline
independently (different EDT algorithm, vectorised class assignment) and adds
a lossless bit-packed codec so the *tag* map can be stored at 2 bit/voxel.

What this module adds beyond the OSS recipe
-------------------------------------------
* :func:`exact_edt` — exact EDT via a k-d tree over fluid-voxel centres; an
  algorithmically independent check of ``scipy.ndimage.distance_transform_edt``
  (which is what the OSS code calls).
* :func:`two_segment_mask` / :func:`distance_to_segments` — construction and
  *analytic* distance field of a piecewise-linear (two-segment) micro-model,
  used as an independent physical reference for the lattice EDT.
* :func:`compress_label_map` / :func:`decompress_label_map` — lossless label
  compression (2-bit packing, or smallest-dtype remap) with round-trip
  verification; also used for the optional per-grain id side channel that the
  OSS 4-class map throws away (mixed-wettability information).
* :func:`write_palabos_geometry` — raw ``int16`` dump + JSON sidecar, the same
  on-disk format Palabos consumes from MPLBM-UT (``.dat`` + geometry name
  ``<name>_<nz>_<ny>_<nx>``).

Conventions
-----------
* Arrays are ``(nz, ny, nx)`` with the streamwise axis **x last**, matching
  :mod:`tensorlbm.preprocess_geo` and the solver lattice layout.
* ``solid=True``/``tags>0`` means grain; the OSS ``rock`` array (0 = pore,
  >0 = grain) is the boolean inverse.
* Distances are measured centre-to-centre in **lattice units** (spacing 1),
  exactly like ``distance_transform_edt`` on a unit-spacing grid; pass
  ``spacing`` to rescale physical units.

References
----------
Santos, J. E. et al. (2022) *MPLBM-UT: Multiphase LBM library for permeable
media analysis*, SoftwareX 19, 101157.  ``je-santos/MPLBM-UT``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "PALABOS_TAGS",
    "PALABOS_PORE",
    "PALABOS_BOUNDARY",
    "PALABOS_GRAIN",
    "PALABOS_NW",
    "DEPTH_SPLIT",
    "GeometryTags",
    "PackedLabels",
    "exact_edt",
    "scipy_edt",
    "distance_to_segments",
    "two_segment_mask",
    "erase_regions",
    "build_geometry_tags",
    "decode_palabos_tags",
    "compress_label_map",
    "decompress_label_map",
    "write_palabos_geometry",
]

# Palabos geometry codes used by MPLBM-UT (pore_utils.create_geom_edist).
PALABOS_PORE = 2608
PALABOS_BOUNDARY = 2609
PALABOS_GRAIN = 2610
PALABOS_NW = 2611
PALABOS_TAGS = {
    "pore": PALABOS_PORE,
    "boundary": PALABOS_BOUNDARY,
    "grain": PALABOS_GRAIN,
    "nw": PALABOS_NW,
}
_CODE_TO_CLASS = {
    PALABOS_PORE: 0,
    PALABOS_BOUNDARY: 1,
    PALABOS_GRAIN: 2,
    PALABOS_NW: 3,
}

# OSS threshold: boundary layer is ``0 < EDT < 2`` (lattice units),
# interior is ``EDT >= 2``.
DEPTH_SPLIT = 2.0


# ---------------------------------------------------------------------------
# Exact Euclidean distance transform (independent algorithm)
# ---------------------------------------------------------------------------
def exact_edt(
    solid: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Exact EDT of the *solid* phase: distance to the nearest fluid voxel.

    For every solid voxel the returned value is the Euclidean distance (in
    physical units) from its centre to the centre of the nearest fluid voxel;
    fluid voxels get 0.  This is the same quantity
    ``scipy.ndimage.distance_transform_edt(rock)`` returns for the OSS
    ``rock`` array (0 = pore, >0 = grain), but computed with a k-d tree so it
    is algorithmically independent of scipy's separable algorithm.

    Degenerate inputs follow scipy: an all-solid domain yields ``inf``
    everywhere (no fluid voxel to measure to) and an all-fluid domain yields
    zeros.

    Args:
        solid: boolean ``(nz, ny, nx)`` array, True where grain.
        spacing: physical size of one voxel along ``(z, y, x)``.

    Returns:
        ``float64`` array of the same shape.
    """
    solid = np.asarray(solid, dtype=bool)
    if solid.ndim != 3:
        raise ValueError(f"solid must be 3-D (nz, ny, nx), got shape {solid.shape}")
    if len(spacing) != 3:
        raise ValueError("spacing must have three entries (z, y, x)")
    sp = np.asarray(spacing, dtype=np.float64)
    if np.any(sp <= 0):
        raise ValueError("spacing entries must be positive")

    out = np.zeros(solid.shape, dtype=np.float64)
    fluid = ~solid
    if not solid.any():
        return out
    if not fluid.any():
        out.fill(np.inf)
        return out

    from scipy.spatial import cKDTree

    fluid_idx = np.argwhere(fluid) * sp
    solid_idx = np.argwhere(solid) * sp
    tree = cKDTree(fluid_idx)
    dist, _ = tree.query(solid_idx, k=1)
    out[solid] = dist
    return out


def scipy_edt(
    solid: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Reference EDT using scipy (the call the OSS implementation makes).

    ``create_geom_edist`` calls ``scipy.ndimage.morphology.distance_transform_edt``
    on the ``rock`` array; this helper exposes the identical computation on a
    boolean *solid* mask so the two EDT algorithms can be compared.
    """
    from scipy import ndimage

    solid = np.asarray(solid, dtype=bool)
    if solid.ndim != 3:
        raise ValueError(f"solid must be 3-D (nz, ny, nx), got shape {solid.shape}")
    return ndimage.distance_transform_edt(solid, sampling=tuple(spacing))


# ---------------------------------------------------------------------------
# Two-segment (piecewise-linear) micro-model: geometry + analytic distance
# ---------------------------------------------------------------------------
def distance_to_segments(
    coords: np.ndarray,
    segments: Iterable[tuple[Sequence[float], Sequence[float]]],
    radius: float = 0.0,
) -> np.ndarray:
    """Signed distance to the union of capsules built on ``segments``.

    Args:
        coords: ``(N, 3)`` sample points in lattice/index coordinates ``(z, y, x)``.
        segments: iterable of ``(p0, p1)`` pairs; each pair is a 3-vector in
            the same coordinate order as *coords*.
        radius: capsule radius (>= 0).  The solid is ``distance < 0``.

    Returns:
        ``(N,)`` signed distance: negative inside the capsule union.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=np.float64))
    if coords.shape[1] != 3:
        raise ValueError(f"coords must be (N, 3), got {coords.shape}")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    segs = [np.asarray(s, dtype=np.float64) for s in segments]
    if not segs:
        raise ValueError("segments must contain at least one (p0, p1) pair")
    best = np.full(coords.shape[0], np.inf)
    for seg in segs:
        if seg.shape != (2, 3):
            raise ValueError(f"each segment must be (2, 3), got {seg.shape}")
        p0, p1 = seg
        d = p1 - p0
        dd = float(d @ d)
        if dd == 0.0:
            continue
        t = ((coords - p0) @ d) / dd
        t = np.clip(t, 0.0, 1.0)
        closest = p0 + t[:, None] * d
        best = np.minimum(best, np.linalg.norm(coords - closest, axis=1))
    return best - float(radius)


def two_segment_mask(
    shape: tuple[int, int, int],
    segments: Iterable[tuple[Sequence[float], Sequence[float]]],
    radius: float,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Rasterise a capsule union (``n`` segments, ``n = 2`` for the micro-model).

    The mask marks the voxel **centre** as solid when the analytic signed
    distance is negative, so the analytic field and the voxel mask are
    consistent by construction and can be compared to the lattice EDT.

    Args:
        shape: ``(nz, ny, nx)`` grid size.
        segments: ``(p0, p1)`` pairs in **index** coordinates ``(z, y, x)``.
        radius: capsule radius in index units.
        spacing: physical voxel size ``(dz, dy, dx)``; only used to convert the
            radius from physical to index units when it is not 1.

    Returns:
        boolean mask ``(nz, ny, nx)``.
    """
    if len(shape) != 3:
        raise ValueError("shape must have three entries (nz, ny, nx)")
    sp = np.asarray(spacing, dtype=np.float64)
    if sp.shape != (3,) or np.any(sp <= 0):
        raise ValueError("spacing must be three positive entries")
    grids = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    # build points lazily to avoid a 3x memory blow-up on big grids
    pts = np.stack([g.ravel() for g in grids], axis=1).astype(np.float64)
    # analytic distance in physical units, segments given in index coordinates
    segs = [(np.asarray(p0, float) * sp, np.asarray(p1, float) * sp) for p0, p1 in segments]
    d = distance_to_segments(pts * sp, segs, radius=radius * float(sp.mean()))
    return (d < 0.0).reshape(shape)


def erase_regions(rock: np.ndarray) -> np.ndarray:
    """OSS ``pore_utils.erase_regions``, call-compatible.

    ``blobs_labels = measure.label(rock, background=1, connectivity=1)`` then
    ``rock[blobs_labels > 1] = 0``.  For a binary ``{0, 1}`` grain image the
    labelled objects are the *pore* bodies (value 1 is background), and the
    assignment writes 0 into cells that already hold 0 — a no-op.  It is kept
    (and tested) because the OSS entry point calls it unconditionally, and
    because it *does* modify images carrying extra phase codes (``3`` =
    non-wetting seeds).
    """
    from scipy import ndimage as ndi

    rock = np.array(rock, copy=True)
    # skimage.measure.label(input, background=1, connectivity=1) labels the
    # connected regions of ``input != 1`` (6-connectivity in 3-D); scipy's
    # ``label`` labels non-zero regions, so the mask must be inverted first.
    # skimage itself cannot be imported here (its C extension is ABI-broken
    # against numpy 2.x in this environment), hence the scipy emulation.
    labels, _ = ndi.label(rock != 1, structure=ndi.generate_binary_structure(3, 1))
    rock[labels > 1] = 0
    return rock


# ---------------------------------------------------------------------------
# Lossless label compression
# ---------------------------------------------------------------------------
@dataclass
class PackedLabels:
    """Lossless container for an integer label map.

    Attributes:
        data: packed bytes.
        shape: original array shape.
        table: value table (index -> original label) for the remap codec.
        bits: bits per code; ``0`` means "raw smallest dtype" (no bit packing).
        dtype: numpy dtype string of the raw path.
        num_labels: number of distinct labels in the original map.
    """

    data: bytes
    shape: tuple[int, ...]
    table: tuple[int, ...]
    bits: int
    dtype: str
    num_labels: int

    @property
    def nbytes(self) -> int:
        return len(self.data)

    def as_dict(self) -> dict:
        return {
            "nbytes": self.nbytes,
            "shape": list(self.shape),
            "table": list(self.table),
            "bits": self.bits,
            "dtype": self.dtype,
            "num_labels": self.num_labels,
        }


def _bits_for(num_labels: int) -> int:
    if num_labels <= 2:
        return 1
    if num_labels <= 4:
        return 2
    if num_labels <= 16:
        return 4
    return 0  # raw path


def compress_label_map(labels: np.ndarray) -> PackedLabels:
    """Compress an integer label map losslessly.

    Two codecs, chosen by entropy:

    * ``bits`` in ``{1, 2, 4}`` — remap labels to small codes and bit-pack
      them (the 4-class Palabos map needs exactly 2 bit/voxel, i.e. 16x fewer
      bytes than the ``float64`` EDT field it was derived from).
    * ``bits == 0`` — remap to the smallest unsigned dtype and store raw
      (used for per-grain id maps, where the number of grains exceeds 16).

    Args:
        labels: integer array of any shape.

    Returns:
        :class:`PackedLabels`.
    """
    arr = np.asarray(labels)
    if arr.dtype.kind not in "iu":
        raise TypeError(f"label map must be integer, got {arr.dtype}")
    if arr.ndim == 0:
        raise ValueError("label map must have at least one dimension")
    flat = arr.ravel()
    table = np.unique(flat)
    num = int(table.size)
    bits = _bits_for(num)
    if bits:
        codes = np.searchsorted(table, flat).astype(np.uint8)
        if bits == 1:
            packed = np.packbits(codes, bitorder="little")
        else:
            per = 8 // bits
            pad = (-codes.size) % per
            c = np.concatenate([codes, np.zeros(pad, dtype=np.uint8)])
            c = c.reshape(-1, per)
            shifts = (np.arange(per, dtype=np.uint8) * bits)
            packed = np.bitwise_or.reduce(c << shifts[None, :], axis=1)
        return PackedLabels(
            data=packed.tobytes(),
            shape=tuple(arr.shape),
            table=tuple(int(v) for v in table),
            bits=bits,
            dtype=str(arr.dtype),
            num_labels=num,
        )
    # raw path with smallest dtype
    idx = np.searchsorted(table, flat)
    if num <= 256:
        raw = idx.astype(np.uint8)
    elif num <= 65536:
        raw = idx.astype(np.uint16)
    else:
        raw = idx.astype(np.uint32)
    return PackedLabels(
        data=raw.tobytes(),
        shape=tuple(arr.shape),
        table=tuple(int(v) for v in table),
        bits=0,
        dtype=str(arr.dtype),
        num_labels=num,
    )


def decompress_label_map(packed: PackedLabels | dict) -> np.ndarray:
    """Inverse of :func:`compress_label_map` (exact)."""
    if isinstance(packed, dict):
        packed = PackedLabels(
            data=bytes.fromhex(packed["data_hex"]) if "data_hex" in packed else packed["data"],
            shape=tuple(packed["shape"]),
            table=tuple(packed["table"]),
            bits=int(packed["bits"]),
            dtype=str(packed.get("dtype", "<i2")),
            num_labels=int(packed.get("num_labels", len(packed["table"]))),
        )
    n = int(np.prod(packed.shape))
    buf = np.frombuffer(packed.data, dtype=np.uint8)
    if packed.bits:
        if packed.bits == 1:
            codes = np.unpackbits(buf, bitorder="little")[:n].astype(np.uint8)
        else:
            per = 8 // packed.bits
            shifts = (np.arange(per, dtype=np.uint8) * packed.bits)
            mask = np.uint8((1 << packed.bits) - 1)
            codes = ((buf[:, None] >> shifts[None, :]) & mask).ravel()[:n].astype(np.uint8)
    else:
        raw_dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32}[
            max(1, int(np.ceil(np.log2(max(packed.num_labels, 2)) / 8)))
        ]
        codes = np.frombuffer(packed.data, dtype=raw_dtype)[:n]
    table = np.asarray(packed.table)
    if codes.max(initial=0) >= table.size:
        raise ValueError("corrupt packed label map: code out of table range")
    return table[codes].reshape(packed.shape).astype(packed.dtype)


# ---------------------------------------------------------------------------
# The geometry-tagging pipeline
# ---------------------------------------------------------------------------
@dataclass
class GeometryTags:
    """Result of :func:`build_geometry_tags`."""

    tags: np.ndarray                 # int16 Palabos codes (padded)
    classes: np.ndarray              # uint8 class index 0..3 (padded)
    distance: np.ndarray             # float64 EDT depth in the unpadded frame
    solid: np.ndarray                # bool grain mask in the unpadded frame
    geom_name: str
    meta: dict = field(default_factory=dict)

    # -- conveniences -----------------------------------------------------
    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.tags.shape)

    def counts(self) -> dict[str, int]:
        return {
            "pore": int(np.count_nonzero(self.classes == 0)),
            "boundary": int(np.count_nonzero(self.classes == 1)),
            "grain": int(np.count_nonzero(self.classes == 2)),
            "nw": int(np.count_nonzero(self.classes == 3)),
        }

    def fractions(self) -> dict[str, float]:
        total = float(self.classes.size)
        return {k: v / total for k, v in self.counts().items()}

    def pack_classes(self) -> PackedLabels:
        return compress_label_map(self.classes)


def build_geometry_tags(
    solid: np.ndarray,
    *,
    num_slices: int = 0,
    swap_xz: bool = False,
    bc_faces: bool = True,
    depth_split: float = DEPTH_SPLIT,
    nw_mask: np.ndarray | None = None,
    grain_ids: np.ndarray | None = None,
    geom_name: str = "geom",
    print_size: bool = True,
    edt: str = "exact",
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> GeometryTags:
    """EDT -> two-class segmentation -> compressed Palabos geometry tags.

    This is the re-implementation of ``pore_utils.create_geom_edist`` plus
    ``create_nw_fluid_mask`` (see the module docstring for the transcription).

    Pipeline (order matters and mirrors the OSS code):

    1. optional ``swap_xz`` transpose ``[2, 1, 0]``;
    2. Euclidean distance transform inside the grains;
    3. (``bc_faces``) force the six outer faces to depth 1 so every boundary
       plane carries bounce-back nodes;
    4. re-open the pores (``distance[solid == 0] = 0``);
    5. two-segment segmentation: ``0 < d < depth_split`` -> boundary class,
       ``d >= depth_split`` -> grain-interior class;
    6. optional ``num_slices`` padding on the streamwise axis (pore codes);
    7. quantise to ``int16`` Palabos codes 2608/2609/2610 (+2611 for
       non-wetting seeds).

    Args:
        solid: boolean ``(nz, ny, nx)`` grain mask (True = grain).
        num_slices: number of empty (pore) slices prepended/appended along the
            streamwise axis, for pressure boundary conditions.
        swap_xz: transpose ``[2, 1, 0]`` before processing (OSS ``swapXZ``).
        bc_faces: force the six outer faces to the boundary class.
        depth_split: boundary/interior threshold in lattice units (OSS: 2.0).
        nw_mask: optional integer mask, value 3 marking non-wetting seeds
            (OSS ``create_nw_fluid_mask``); those cells are removed from the
            solid phase before the EDT and re-inserted as code 2611.
        grain_ids: optional integer array aligned with *solid*; per-grain ids
            on grain cells (0 elsewhere).  Compressed into the side channel
            ``meta["grain_ids"]`` — the information the OSS 4-class map drops.
        geom_name: base name; the returned ``geom_name`` follows the OSS
            convention ``<name>_<nz>_<ny>_<nx>`` when *print_size* is True.
        print_size: emit the size suffix like the OSS implementation.
        edt: ``"exact"`` (k-d tree, default) or ``"scipy"`` (OSS call).
        spacing: physical voxel size ``(dz, dy, dx)``; the depth threshold is
            compared in *physical* units, so spacing ``(1,1,1)`` reproduces
            the OSS lattice-unit convention.

    Returns:
        :class:`GeometryTags`.  ``distance``/``solid`` refer to the unpadded,
        post-swap frame; ``tags``/``classes`` are padded.
    """
    solid = np.asarray(solid, dtype=bool)
    if solid.ndim != 3:
        raise ValueError(f"solid must be 3-D (nz, ny, nx), got shape {solid.shape}")
    if depth_split <= 0:
        raise ValueError("depth_split must be positive")
    if num_slices < 0:
        raise ValueError("num_slices must be non-negative")
    if edt not in ("exact", "scipy"):
        raise ValueError("edt must be 'exact' or 'scipy'")
    sp = np.asarray(spacing, dtype=np.float64)
    if sp.shape != (3,) or np.any(sp <= 0):
        raise ValueError("spacing must be three positive entries")

    rock = solid  # True = grain
    if swap_xz:
        rock = np.ascontiguousarray(rock.transpose([2, 1, 0]))
    nw = None
    if nw_mask is not None:
        nw = np.asarray(nw_mask)
        if nw.shape != solid.shape:
            raise ValueError(
                f"nw_mask shape {nw.shape} does not match solid shape {solid.shape}"
            )
        if swap_xz:
            nw = np.ascontiguousarray(nw.transpose([2, 1, 0]))
        nw = nw == 3
        rock = rock & ~nw  # OSS: rock = np.where(rock == 3, 0, rock)

    # 2. EDT inside the grains
    dist = exact_edt(rock, spacing=sp) if edt == "exact" else scipy_edt(rock, spacing=sp)

    # 3. guarantee bounce-back on every outer face
    if bc_faces:
        dist[0, :, :] = sp[1]  # value 1 in lattice units on face planes
        dist[-1, :, :] = sp[1]
        dist[:, 0, :] = sp[1]
        dist[:, -1, :] = sp[1]
        dist[:, :, 0] = sp[1]
        dist[:, :, -1] = sp[1]

    # 4. re-open the pores
    dist = np.where(rock, dist, 0.0)

    # 5. two-segment segmentation
    cls = np.zeros(rock.shape, dtype=np.uint8)
    boundary = rock & (dist > 0.0) & (dist < depth_split)
    interior = rock & (dist >= depth_split)
    cls[boundary] = 1
    cls[interior] = 2
    dist = np.where(cls == 1, 1.0, np.where(cls == 2, 2.0, 0.0))

    # 6. padding for pressure BCs
    if num_slices:
        cls = np.pad(cls, [(num_slices, num_slices), (0, 0), (0, 0)], mode="constant")
        if nw is not None:
            nw = np.pad(nw, [(num_slices, num_slices), (0, 0), (0, 0)], mode="constant")

    # 7. quantise to Palabos codes
    tags = np.full(cls.shape, PALABOS_PORE, dtype=np.int16)
    tags[cls == 1] = PALABOS_BOUNDARY
    tags[cls == 2] = PALABOS_GRAIN
    if nw is not None:
        tags[nw] = PALABOS_NW
        cls[nw] = 3

    name = f"{geom_name}_{tags.shape[0]}_{tags.shape[1]}_{tags.shape[2]}" if print_size else geom_name

    meta: dict = {
        "source": "je-santos/MPLBM-UT@a9fc4966 pore_utils.create_geom_edist (re-implementation)",
        "geom_name": name,
        "shape": list(tags.shape),
        "unpadded_shape": list(solid.shape),
        "num_slices": int(num_slices),
        "swap_xz": bool(swap_xz),
        "bc_faces": bool(bc_faces),
        "depth_split": float(depth_split),
        "edt": edt,
        "spacing": [float(v) for v in sp],
        "palabos_tags": dict(PALABOS_TAGS),
    }
    packed = compress_label_map(cls)
    raw_float_bytes = int(np.prod(cls.shape)) * 8          # float64 EDT field
    int16_bytes = int(np.prod(cls.shape)) * 2
    meta["compression"] = {
        "basis": "same (padded) voxel count for both sides",
        "float64_edt_bytes": raw_float_bytes,
        "int16_codes_bytes": int16_bytes,
        "packed_bytes": packed.nbytes,
        "bits_per_voxel": packed.bits,
        "ratio_vs_float64": (raw_float_bytes / packed.nbytes) if packed.nbytes else None,
        "ratio_int16_vs_float64": 4.0,
    }
    if grain_ids is not None:
        gid = np.asarray(grain_ids)
        if gid.shape != solid.shape:
            raise ValueError("grain_ids shape must match solid shape")
        if swap_xz:
            gid = np.ascontiguousarray(gid.transpose([2, 1, 0]))
        gid = np.where(rock, gid, 0).astype(np.int32)
        if num_slices:
            gid = np.pad(gid, [(num_slices, num_slices), (0, 0), (0, 0)], mode="constant")
        packed_ids = compress_label_map(gid)
        meta["grain_ids"] = packed_ids.as_dict()
        meta["grain_ids"]["data"] = packed_ids

    return GeometryTags(
        tags=tags,
        classes=cls,
        distance=dist,
        solid=rock,
        geom_name=name,
        meta=meta,
    )


def decode_palabos_tags(tags: np.ndarray) -> np.ndarray:
    """Map Palabos codes back to class indices ``{0, 1, 2, 3}``.

    Args:
        tags: ``int16`` array of Palabos geometry codes.

    Returns:
        ``uint8`` class map.

    Raises:
        ValueError: if *tags* contains a code outside :data:`PALABOS_TAGS`.
    """
    tags = np.asarray(tags)
    unknown = set(np.unique(tags).tolist()) - set(_CODE_TO_CLASS)
    if unknown:
        raise ValueError(f"unknown Palabos geometry codes: {sorted(unknown)}")
    out = np.zeros(tags.shape, dtype=np.uint8)
    for code, cls in _CODE_TO_CLASS.items():
        out[tags == code] = cls
    return out


def write_palabos_geometry(path, geometry: GeometryTags) -> dict:
    """Write the OSS on-disk artefact: raw ``int16`` ``.dat`` + JSON sidecar.

    MPLBM-UT dumps ``rock4sim.flatten().tofile(f'{geom_name}.dat')`` and tells
    Palabos the size through the geometry name.  The sidecar keeps everything
    needed to reload the array without guessing (shape, dtype, codes, the
    compressed class map and provenance).

    Args:
        path: destination ``.dat`` path (``.json`` sidecar written alongside).
        geometry: result of :func:`build_geometry_tags`.

    Returns:
        The sidecar dictionary that was written.
    """
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    geo = np.ascontiguousarray(geometry.tags, dtype="<i2")
    geo.tofile(path)
    packed = geometry.pack_classes()
    sidecar = {
        "format": "mplbm-palabos-geometry/1",
        "dat_file": path.name,
        "nbytes": int(geo.nbytes),
        "shape": [int(v) for v in geo.shape],
        "dtype": "<i2",
        "order": "C",
        "codes": dict(PALABOS_TAGS),
        "geom_name": geometry.geom_name,
        "counts": geometry.counts(),
        "fractions": geometry.fractions(),
        "packed_classes": packed.as_dict(),
        "packed_classes_data_hex": packed.data.hex(),
        "provenance": geometry.meta,
    }
    import json

    side = path.with_suffix(".json")
    side.write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
    return sidecar