"""Tests for :mod:`tensorlbm.geom_edt_tag` (EDT two-segment geometry tagging).

Two independent oracles are used for the segmentation:

* a **brute-force distance oracle** (explicit minimum over all pore voxels on a
  small grid) — checks that the two-segment split really is the exact-EDT
  partition and not an approximate/structure-element shell; and
* the **upstream reference** ``xval/.../oss_reference.py`` (verbatim
  ``je-santos/MPLBM-UT`` ``create_geom_edist``) — checks bit-for-bit tag parity
  on the knobs the reference exposes (padding, swap, non-wetting back-fill).

The remaining tests pin the conventions that the physics depends on: complete
link coverage of the boundary segment (no leaking diagonal links), tag palette
round-trip, lossless run-length compression, and input validation.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from tensorlbm.geom_edt_tag import (
    GRAIN_TAG,
    NW_TAG,
    PORE_TAG,
    WALL_TAG,
    compress_tags,
    compression_report,
    decompress_tags,
    edt_two_segment_tag,
    rle_decode,
    rle_encode,
    tag_stats,
)

REPO = Path(__file__).resolve().parents[1]
REF_DIR = REPO / "xval" / "edt_two_segment_geometry_tag_compression"


# --------------------------------------------------------------------------
# geometry fixtures
# --------------------------------------------------------------------------
def sphere_mask(n: int = 16, radius: float = 4.0) -> np.ndarray:
    idx = np.arange(n, dtype=np.float64)
    d = idx - (n - 1) / 2.0
    dist2 = d[:, None, None] ** 2 + d[None, :, None] ** 2 + d[None, None, :] ** 2
    return (dist2 <= radius ** 2).astype(np.int8)


def random_mask(n: int = 12, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((n, n, n)) < 0.35).astype(np.int8)


def brute_force_tags(rock: np.ndarray, shell_threshold: float = 2.0) -> np.ndarray:
    """Exact EDT partition by explicit minimum distance to every pore voxel."""
    solid = rock != 0
    pore_idx = np.argwhere(~solid)
    assert pore_idx.size, "oracle needs at least one pore voxel"
    solid_idx = np.argwhere(solid)
    out = np.zeros(rock.shape, dtype=np.int8)
    for iz, iy, ix in solid_idx:
        d = np.sqrt(((pore_idx - np.array([iz, iy, ix])) ** 2).sum(axis=1).min())
        out[iz, iy, ix] = 1 if d < shell_threshold else 2
    # face forcing is part of the convention: the distance field, not the tag,
    # is what the face rule touches, and pores are re-opened afterwards, so the
    # oracle needs the faces applied *before* the split.
    for axis in range(3):
        for side in (0, -1):
            sl = [slice(None)] * 3
            sl[axis] = side
            face = solid[tuple(sl)]
            out[tuple(sl)][face] = 1
    return out


# --------------------------------------------------------------------------
# 1. exact-EDT semantics
# --------------------------------------------------------------------------
def test_two_segment_split_matches_bruteforce_edt():
    rock = random_mask(12, seed=3)
    tagged = edt_two_segment_tag(rock, name="t")
    ours = decompress_tags(tagged.geom)
    oracle = brute_force_tags(rock)
    assert np.array_equal(ours, oracle)


def test_sphere_shell_is_thinner_than_one_voxel_ring_but_covers_diagonals():
    rock = sphere_mask()
    tagged = edt_two_segment_tag(rock, name="t")
    wall = tagged.geom == WALL_TAG
    grain = tagged.geom == GRAIN_TAG
    # every solid voxel that shares a face/edge/corner with a pore is wall
    solid = rock != 0
    pore = ~solid
    grown = np.zeros_like(pore)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == dy == dx == 0:
                    continue
                grown |= np.roll(np.roll(np.roll(pore, dz, 0), dy, 1), dx, 2)
    assert not (solid & grown & grain).any()
    # d = sqrt(2) and sqrt(3) voxels sit in the wall segment (threshold 2.0)
    assert (tagged.edt[wall] > 1.0).any()


def test_leaky_threshold_below_sqrt2_leaves_grain_adjacent_to_pores():
    """Threshold 1.0 has an empty wall segment -> diagonal links would leak."""
    rock = sphere_mask()
    leaky = edt_two_segment_tag(rock, name="t", shell_threshold=1.0)
    assert (leaky.geom == WALL_TAG).sum() == 0
    assert (leaky.geom == GRAIN_TAG).sum() == (rock != 0).sum()


def test_pore_volume_is_conserved_and_faces_reopen():
    rock = random_mask(10, seed=5)
    tagged = edt_two_segment_tag(rock, name="t")
    assert int((tagged.geom == PORE_TAG).sum()) == int((rock == 0).sum())
    solid = rock != 0
    # a pore voxel on a face must stay pore despite the face forcing
    face_pores = solid[0, :, :] == False  # noqa: E712 - explicit
    assert (tagged.geom[0, :, :][face_pores] == PORE_TAG).all()


def test_face_forcing_gives_every_face_solid_voxel_a_wall_tag():
    rock = np.zeros((8, 8, 8), dtype=np.int8)
    rock[3:6, 3:6, 3:6] = 1
    tagged = edt_two_segment_tag(rock, name="t")
    assert (tagged.geom[0, :, :][rock[0, :, :] != 0] == WALL_TAG).all() if rock[0].any() else True
    # with a slab touching the face, the face row of the slab is wall
    slab = np.zeros((8, 8, 8), dtype=np.int8)
    slab[4:, :, :] = 1
    t2 = edt_two_segment_tag(slab, name="s")
    assert (t2.geom[4, :, :] == WALL_TAG).all()


# --------------------------------------------------------------------------
# 2. compression
# --------------------------------------------------------------------------
def test_palette_roundtrip_and_unknown_tag_rejected():
    labels = np.array([0, 1, 2, 3], dtype=np.int8).reshape(1, 1, 4)
    geom = compress_tags(labels)
    assert geom.dtype == np.int16
    assert geom.reshape(-1).tolist() == [PORE_TAG, WALL_TAG, GRAIN_TAG, NW_TAG]
    assert np.array_equal(decompress_tags(geom), labels)
    with pytest.raises(ValueError):
        decompress_tags(np.array([9999], dtype=np.int16))
    with pytest.raises(ValueError):
        compress_tags(np.array([7], dtype=np.int8))


def test_rle_is_lossless_and_compression_report_is_consistent():
    rock = sphere_mask()
    tagged = edt_two_segment_tag(rock, name="t")
    rle = rle_encode(tagged.geom)
    assert np.array_equal(rle_decode(rle, tagged.geom.shape), tagged.geom)
    rep = compression_report(tagged)
    n = tagged.geom.size
    assert rep["voxels"] == n
    assert rep["ratio_float64_to_int16"] == pytest.approx(4.0)
    assert rep["palette_runs"] == rle.shape[0]
    assert 0.0 < rep["entropy_bits_per_voxel"] < 2.0  # 4-symbol palette
    assert rep["rle_roundtrip_exact"] is True


def test_tag_stats_split_the_solid_into_two_segments():
    rock = sphere_mask(24, radius=6.0)
    stats = tag_stats(edt_two_segment_tag(rock, name="t").geom)
    assert stats["solid"] == int((rock != 0).sum())
    assert 0 < stats["wall_fraction_of_solid"] < 1


# --------------------------------------------------------------------------
# 3. shapes / naming / validation
# --------------------------------------------------------------------------
def test_padding_swap_and_name_conventions():
    rock = random_mask(10, seed=11)
    plain = edt_two_segment_tag(rock, name="g")
    assert plain.name == "g"
    padded = edt_two_segment_tag(rock, name="g", num_slices=2, print_size=True)
    assert padded.geom.shape == (14, 10, 10)
    assert padded.name == "g_14_10_10"
    assert (padded.geom[:2] == PORE_TAG).all() and (padded.geom[-2:] == PORE_TAG).all()
    swapped = edt_two_segment_tag(rock, name="g", swap_xz=True)
    assert swapped.geom.shape == (10, 10, 10)
    assert np.array_equal(swapped.geom, plain.geom.transpose(2, 1, 0))


def test_nw_backfill_and_validation():
    rock = random_mask(10, seed=2)
    nw = np.zeros_like(rock)
    nw[:, :3, :] = 3
    tagged = edt_two_segment_tag(rock, name="g", nw_fluid_mask=nw)
    assert (tagged.geom[nw == 3] == NW_TAG).all()
    with pytest.raises(ValueError):
        edt_two_segment_tag(rock, name="g", nw_fluid_mask=np.zeros((3, 3, 3), dtype=np.int8))
    with pytest.raises(ValueError):
        edt_two_segment_tag(rock, name="g", shell_threshold=0.0)
    with pytest.raises(ValueError):
        edt_two_segment_tag(rock[0], name="g")


# --------------------------------------------------------------------------
# 4. parity with the upstream OSS reference (verbatim MPLBM-UT function)
# --------------------------------------------------------------------------
def _load_reference():
    if not (REF_DIR / "oss_reference.py").exists():
        return None
    if str(REF_DIR) not in sys.path:
        sys.path.insert(0, str(REF_DIR))
    import oss_reference

    return oss_reference


@pytest.mark.parametrize("geometry", ["sphere", "random", "slab"])
@pytest.mark.parametrize("variant", ["plain", "pad", "swap", "nw"])
def test_tag_parity_with_oss_reference(geometry, variant):
    ref_mod = _load_reference()
    if ref_mod is None:
        pytest.skip("OSS reference snapshot not present")
    if geometry == "sphere":
        rock = sphere_mask(14, radius=4.0)
    elif geometry == "random":
        rock = random_mask(12, seed=13)
    else:
        rock = np.zeros((10, 10, 10), dtype=np.int8)
        rock[5:, :, :] = 1

    kwargs = {}
    nw = None
    if variant == "pad":
        kwargs = {"num_slices": 2, "print_size": True}
    elif variant == "swap":
        kwargs = {"swap_xz": True}
    elif variant == "nw":
        nw = np.zeros_like(rock)
        nw[:2, :, :] = 3
        kwargs = {"nw_fluid_mask": nw}

    ours = edt_two_segment_tag(rock, name="g", **kwargs)
    ref_geom, ref_name = ref_mod.reference_tag_geometry(
        rock,
        swap_xz=kwargs.get("swap_xz", False),
        num_slices=kwargs.get("num_slices", 0),
        name="g",
        print_size=kwargs.get("print_size", False),
        nw_fluid_mask=nw,
    )
    assert ours.geom.shape == ref_geom.shape
    assert ours.name == ref_name
    assert np.array_equal(ours.geom, ref_geom)


def test_reference_reproduces_known_palette_constants():
    """The four palette constants are the upstream ones (2608-2611)."""
    assert (PORE_TAG, WALL_TAG, GRAIN_TAG, NW_TAG) == (2608, 2609, 2610, 2611)
    rock = sphere_mask(12, radius=3.0)
    geom = edt_two_segment_tag(rock, name="t").geom
    assert set(np.unique(geom).tolist()) <= {PORE_TAG, WALL_TAG, GRAIN_TAG}


def test_threshold_sensitivity_is_declared_not_silent():
    """Moving the shell threshold away from 2.0 changes the tag split."""
    rock = random_mask(16, seed=21)
    base = edt_two_segment_tag(rock, name="t").geom
    tighter = edt_two_segment_tag(rock, name="t", shell_threshold=math.sqrt(2.0)).geom
    assert not np.array_equal(base, tighter)
    assert (base == WALL_TAG).sum() > (tighter == WALL_TAG).sum()