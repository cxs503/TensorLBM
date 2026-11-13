"""EDT two-segment geometry tagging + tag compression for voxel rock images.

Implements the geometry-preparation step that turns a binary/labelled rock
image into the *simulation* geometry consumed by a bounce-back LBM solver,
following the practice established by ``je-santos/MPLBM-UT``
(``src/python/mplbm_utils/pore_utils.py::create_geom_edist``).  The step has
two coupled halves, and both matter:

1. **Two-segment tagging** — a Euclidean distance transform (EDT) of the
   solid phase is thresholded so that solid voxels *near* the pore space
   form the **boundary segment** (one-to-few voxels thick; these cells get
   the bounce-back/wetting dynamics, i.e. they must exist for the no-slip
   condition to have a node) while solid voxels deeper inside the grain form
   the **grain segment** (no dynamics at all — pure zero-compute occupancy).
   Splitting the solid phase in two is what makes the interior free: only a
   shell of the solid costs anything per timestep.

2. **Tag compression** — the continuous EDT field (``float64``, one value
   per voxel) is discarded and replaced by a **4-symbol int16 palette**
   (``2608`` pore / ``2609`` boundary / ``2610`` grain / ``2611`` back-filled
   non-wetting phase).  The solver then dispatches dynamics from the symbol
   alone.  Beyond the 4x dtype saving (``float64`` -> ``int16``) the palette
   collapses ``log2(#distinct distances)`` bits of entropy per voxel to
   ~1.5 bits, so the on-disk geometry becomes strongly run-length
   compressible.

Discretisation conventions (pinned by tests, taken from the reference):

- The distance is the **exact Euclidean** distance (``scipy.ndimage.
  distance_transform_edt`` with unit sampling) from a solid voxel to the
  nearest **pore** (``rock == 0``) voxel.  ``rock != 0`` is solid, so a
  non-wetting label (e.g. ``3``) counts as solid for the EDT — the reference
  removes it beforehand via ``create_nw_fluid_mask`` and back-fills it after
  compression.
- All six **domain faces are forced to distance 1** before pores are
  re-opened, so every boundary condition has a solid node to push against
  even when the image itself ends in pore space.
- Pores are re-opened afterwards (``distance = 0`` where ``rock == 0``), so
  the face forcing never eats pore voxels.
- The segment split uses the reference threshold **``2.0``**: wall segment =
  ``0 < d < 2``, grain segment = ``d >= 2``.  With unit sampling a voxel
  sharing only an edge (d=sqrt2) or a corner (d=sqrt3) with pore space is
  therefore *wall*, not grain — the shell is thicker than the 6-neighbour
  shell.  ``shell_threshold`` is exposed so the sensitivity of this choice
  can be measured (it is not a tuning knob; the default reproduces the
  reference).
- Optional ``num_slices`` padding ``[(n, n), (0, 0), (0, 0)]`` with pores
  provides the inlet/outlet layers that a pressure-gradient drive needs.

The public API is :func:`edt_two_segment_tag` (tagging),
:func:`decompress_tags` / :func:`rle_encode` / :func:`rle_decode`
(compression round-trip) and :func:`tag_stats` / :func:`compression_report`
(diagnostics used by the cross-validation harness).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

__all__ = [
    "GRAIN_TAG",
    "NW_TAG",
    "PORE_TAG",
    "WALL_TAG",
    "TaggedGeometry",
    "compress_tags",
    "compression_report",
    "decompress_tags",
    "edt_two_segment_tag",
    "rle_decode",
    "rle_encode",
    "tag_stats",
]

#: Compressed tag palette (MPLBM-UT / Palabos geometry symbols).
PORE_TAG = 2608  # pore / wetting fluid
WALL_TAG = 2609  # boundary segment: bounce-back + wetting dynamics
GRAIN_TAG = 2610  # grain interior: no dynamics
NW_TAG = 2611  # back-filled non-wetting phase

_PALETTE = (PORE_TAG, WALL_TAG, GRAIN_TAG, NW_TAG)


@dataclass
class TaggedGeometry:
    """Result of :func:`edt_two_segment_tag`."""

    geom: np.ndarray  # int16 tag array, shape (nx, ny, nz)
    name: str
    edt: np.ndarray  # float64 distance field *before* compression (pre-pad)
    shell_threshold: float = 2.0
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.geom.shape)

    def counts(self) -> dict[str, int]:
        return tag_stats(self.geom)


def edt_two_segment_tag(
    rock: np.ndarray,
    *,
    swap_xz: bool = False,
    num_slices: int = 0,
    name: str = "geom",
    print_size: bool = False,
    nw_fluid_mask: np.ndarray | None = None,
    shell_threshold: float = 2.0,
    bc_faces: bool = True,
) -> TaggedGeometry:
    """Tag a rock image as ``{pore, boundary, grain}`` via an exact EDT.

    Parameters
    ----------
    rock:
        Integer/boolean array with ``rock == 0`` marking pore voxels and any
        non-zero value marking solid (grain/non-wetting) voxels.
        Shape is ``(Nx, Ny, Nz)`` — index 0 is the flow axis.
    swap_xz:
        Transpose ``(Nx, Ny, Nz) -> (Nz, Ny, Nx)`` before processing.  Must
        match the orientation convention used when the geometry was written.
    num_slices:
        Number of pore slices padded on both ends of axis 0 (inlet/outlet
        layers).
    name, print_size:
        Base geometry name; with ``print_size`` the padded shape is suffixed
        (``f"{name}_{Nx}_{Ny}_{Nz}"``) exactly as the reference does, because
        the name is what the solver input file uses to find the ``.dat``.
    nw_fluid_mask:
        Optional array in final (padded, post-swap) orientation whose
        ``== 3`` entries are re-stamped as :data:`NW_TAG`.
    shell_threshold:
        Distance cut between boundary and grain segment.  ``2.0`` reproduces
        the reference convention.
    bc_faces:
        Force the six domain faces of the distance field to 1 (reference
        behaviour; disable only to measure its effect).

    Returns
    -------
    TaggedGeometry
    """
    rock = np.asarray(rock)
    if rock.ndim != 3:
        raise ValueError(f"rock must be 3D, got shape {rock.shape}")
    if shell_threshold <= 0:
        raise ValueError(f"shell_threshold must be > 0, got {shell_threshold}")
    if int(num_slices) < 0:
        raise ValueError(f"num_slices must be >= 0, got {num_slices}")

    if swap_xz:
        rock = rock.transpose([2, 1, 0])

    solid = rock != 0

    # Exact Euclidean distance (unit sampling) to the nearest pore voxel.
    edt = ndimage.distance_transform_edt(solid)

    if bc_faces:
        edt[0, :, :] = 1.0
        edt[:, 0, :] = 1.0
        edt[:, :, 0] = 1.0
        edt[-1, :, :] = 1.0
        edt[:, -1, :] = 1.0
        edt[:, :, -1] = 1.0

    # Re-open pores so the face forcing never removes fluid.
    edt[rock == 0] = 0.0

    # Two-segment split: shell = 0 < d < threshold, grain = d >= threshold.
    wall = (edt > 0.0) & (edt < shell_threshold)
    grain = edt >= shell_threshold
    labels = np.zeros(rock.shape, dtype=np.int8)
    labels[wall] = 1
    labels[grain] = 2

    if num_slices:
        n = int(num_slices)
        labels = np.pad(labels, [(n, n), (0, 0), (0, 0)], "constant", constant_values=0)

    geom_name = name
    if print_size:
        geom_name = f"{name}_{labels.shape[0]}_{labels.shape[1]}_{labels.shape[2]}"

    geom = compress_tags(labels)

    if nw_fluid_mask is not None:
        nw = np.asarray(nw_fluid_mask)
        if nw.shape != geom.shape:
            raise ValueError(
                f"nw_fluid_mask shape {nw.shape} != geometry shape {geom.shape} "
                "(expected final padded/post-swap orientation)"
            )
        geom[nw == 3] = NW_TAG

    counts = tag_stats(geom)
    meta = {
        "swap_xz": bool(swap_xz),
        "num_slices": int(num_slices),
        "bc_faces": bool(bc_faces),
        "pore_voxels_input": int((~solid).sum()),
        "solid_voxels_input": int(solid.sum()),
        "counts": counts,
    }
    return TaggedGeometry(
        geom=geom,
        name=geom_name,
        edt=edt,
        shell_threshold=float(shell_threshold),
        meta=meta,
    )


def compress_tags(labels: np.ndarray) -> np.ndarray:
    """Map compact labels ``0/1/2/3`` to the int16 tag palette."""
    lab = np.asarray(labels)
    if lab.size and (lab.min() < 0 or lab.max() > 3):
        raise ValueError(f"labels must lie in 0..3, got [{lab.min()}, {lab.max()}]")
    out = np.zeros(lab.shape, dtype=np.int16)
    for compact, tag in enumerate(_PALETTE):
        out[lab == compact] = tag
    return out


def decompress_tags(geom: np.ndarray) -> np.ndarray:
    """Map the int16 tag palette back to compact labels ``0/1/2/3``."""
    g = np.asarray(geom)
    out = np.zeros(g.shape, dtype=np.uint8)
    for compact, tag in enumerate(_PALETTE):
        out[g == tag] = compact
    unknown = ~np.isin(g, _PALETTE)
    if unknown.any():
        raise ValueError(f"geometry contains {int(unknown.sum())} non-palette tags")
    return out


def tag_stats(geom: np.ndarray) -> dict[str, int]:
    """Population of each tag plus derived segment diagnostics."""
    g = np.asarray(geom)
    counts = {str(tag): int((g == tag).sum()) for tag in _PALETTE}
    solid = counts[str(WALL_TAG)] + counts[str(GRAIN_TAG)]
    return {
        **counts,
        "total": int(g.size),
        "solid": solid,
        "wall_fraction_of_solid": (
            counts[str(WALL_TAG)] / solid if solid else 0.0
        ),
    }


def rle_encode(geom: np.ndarray) -> np.ndarray:
    """Run-length encode a tag array (order-preserving, lossless).

    Returns an ``(n_runs, 2)`` int64 array of ``(tag, run_length)`` pairs.
    """
    flat = np.asarray(geom).reshape(-1)
    if flat.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], change))
    lengths = np.diff(np.concatenate((starts, [flat.size])))
    return np.stack([flat[starts].astype(np.int64), lengths.astype(np.int64)], axis=1)


def rle_decode(rle: np.ndarray, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Inverse of :func:`rle_encode`."""
    rle = np.asarray(rle)
    if rle.size == 0:
        return np.zeros(shape or (0,), dtype=np.int16)
    flat = np.repeat(rle[:, 0].astype(np.int16), rle[:, 1].astype(np.int64))
    return flat.reshape(shape) if shape is not None else flat


def _entropy_bits(values: np.ndarray) -> float:
    """Shannon entropy (bits per voxel) of the voxel symbol distribution."""
    _, counts = np.unique(values.reshape(-1), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def compression_report(tagged: TaggedGeometry) -> dict[str, float | int | str]:
    """Quantify what the tag compression actually buys.

    Compares the on-disk cost of

    * the raw continuous EDT field (``float64``; what the segmentation
      *derives from* and what a naive pipeline would persist),
    * the compressed int16 palette geometry,
    * the run-length encoded palette geometry (lossless, verified by
      decoding back and comparing bit-for-bit).
    """
    geom = tagged.geom
    n = geom.size
    rle = rle_encode(geom)
    roundtrip = rle_decode(rle, geom.shape)
    if not np.array_equal(roundtrip, geom):
        raise AssertionError("RLE round-trip is not lossless")

    def _nbytes_entropy(bits: float) -> int:
        return int(np.ceil(bits * n / 8.0))

    return {
        "voxels": int(n),
        "raw_edt_bytes_float64": int(8 * n),
        "raw_edt_bytes_float32": int(4 * n),
        "palette_bytes_int16": int(2 * n),
        "palette_bytes_int16_rle": int(2 * rle.shape[0] * 2),  # pairs of int16
        "palette_runs": int(rle.shape[0]),
        "entropy_bits_per_voxel": _entropy_bits(decompress_tags(geom)),
        "entropy_only_bytes": _nbytes_entropy(_entropy_bits(decompress_tags(geom))),
        "ratio_float64_to_int16": 8 * n / (2 * n),
        "ratio_float64_to_rle": (8 * n) / max(1, 4 * rle.shape[0]),
        "rle_roundtrip_exact": True,
        "distinct_tags": int(np.unique(geom).size),
    }