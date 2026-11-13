"""Tests for EDT two-segment geometry tagging and tag compression.

Anchor: XVAL-LBM-EDT-TWO-SEGMENT-GEOMETRY-TAG-COMPRESSION against
``je-santos/MPLBM-UT`` @ ``a9fc4966`` (``mplbm_utils/pore_utils.py``).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from tensorlbm.edt_geometry_tags import (
    PALABOS_BOUNDARY,
    PALABOS_GRAIN,
    PALABOS_NW,
    PALABOS_PORE,
    PackedLabels,
    build_geometry_tags,
    compress_label_map,
    decode_palabos_tags,
    decompress_label_map,
    distance_to_segments,
    erase_regions,
    exact_edt,
    scipy_edt,
    two_segment_mask,
    write_palabos_geometry,
)
from tensorlbm.multiphase3d import wall_adhesion_force_3d
from tensorlbm.preprocess_geo import palabos_geometry_tags


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def slab_mask(nz=8, ny=16, nx=12, t=3):
    m = np.zeros((nz, ny, nx), dtype=bool)
    m[:, :t, :] = True
    m[:, -t:, :] = True
    return m


def random_mask(shape=(12, 14, 10), seed=0, p=0.35):
    rng = np.random.default_rng(seed)
    return rng.random(shape) < p


# ---------------------------------------------------------------------------
# EDT correctness
# ---------------------------------------------------------------------------
def test_exact_edt_matches_scipy():
    for seed in range(4):
        m = random_mask(seed=seed)
        a = exact_edt(m)
        b = scipy_edt(m)
        assert np.allclose(a, b, atol=1e-12), f"seed {seed}: max diff {np.abs(a - b).max()}"


def test_exact_edt_anisotropic_spacing():
    m = np.zeros((5, 5, 5), dtype=bool)
    m[2, 2, 2] = True
    d = exact_edt(m, spacing=(2.0, 1.0, 1.0))
    # nearest fluid centre is one x-neighbour away -> spacing 1.0 in x
    assert d[2, 2, 2] == pytest.approx(1.0)
    m2 = np.ones((5, 5, 5), dtype=bool)
    m2[:, :, 1:] = False                      # only the x = 0 plane is solid
    d2 = exact_edt(m2, spacing=(1.0, 1.0, 2.0))
    assert d2[2, 2, 0] == pytest.approx(2.0)  # spacing along x is 2.0


def test_exact_edt_degenerate_domains():
    assert np.all(exact_edt(np.zeros((4, 4, 4), dtype=bool)) == 0.0)
    assert np.all(np.isinf(exact_edt(np.ones((4, 4, 4), dtype=bool))))


def test_exact_edt_rejects_bad_input():
    with pytest.raises(ValueError):
        exact_edt(np.zeros((4, 4), dtype=bool))
    with pytest.raises(ValueError):
        exact_edt(np.zeros((4, 4, 4), dtype=bool), spacing=(1.0, 1.0))
    with pytest.raises(ValueError):
        exact_edt(np.zeros((4, 4, 4), dtype=bool), spacing=(1.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# two-segment analytic geometry
# ---------------------------------------------------------------------------
def test_distance_to_segments_closed_form():
    segs = [((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))]
    pts = np.array([
        [5.0, 3.0, 0.0],    # perpendicular offset -> 3
        [0.0, 0.0, -4.0],   # beyond start cap    -> 4
        [14.0, 0.0, 0.0],   # beyond far cap      -> 4
    ])
    d = distance_to_segments(pts, segs)
    assert np.allclose(d, [3.0, 4.0, 4.0])
    assert np.allclose(distance_to_segments(pts, segs, radius=1.0), [2.0, 3.0, 3.0])


def test_two_segment_mask_is_union_of_capsules():
    segs = [((5.0, 5.0, 2.0), (5.0, 5.0, 8.0)),
            ((5.0, 5.0, 8.0), (5.0, 10.0, 8.0))]
    m = two_segment_mask((11, 13, 13), segs, radius=1.5)
    assert m.dtype == bool and m.shape == (11, 13, 13)
    assert m[5, 5, 5]          # on the first segment axis
    assert m[5, 9, 8]          # on the second segment axis
    assert not m[0, 0, 0]


def test_two_segment_lattice_edt_tracks_analytic_depth():
    segs = [((10.0, 10.0, 4.0), (10.0, 10.0, 20.0)),
            ((10.0, 10.0, 20.0), (10.0, 22.0, 20.0))]
    m = two_segment_mask((21, 25, 25), segs, radius=5.0)
    phi = distance_to_segments(np.argwhere(np.ones_like(m)).reshape(-1, 3), segs, radius=5.0)
    phi = phi.reshape(m.shape)
    d = exact_edt(m)
    deep = m & (phi <= -4.0)
    assert deep.sum() > 20
    rel = np.abs(d[deep] - (-phi[deep])) / (-phi[deep])
    assert rel.mean() < 0.15, f"mean relative depth error {rel.mean():.3f}"


# ---------------------------------------------------------------------------
# tag pipeline
# ---------------------------------------------------------------------------
def test_tag_codes_and_two_class_split():
    m = slab_mask(t=4)
    geo = build_geometry_tags(m, geom_name="slab")
    assert geo.tags.dtype == np.int16
    assert set(np.unique(geo.tags)).issubset({PALABOS_PORE, PALABOS_BOUNDARY, PALABOS_GRAIN})
    # a 4-cell-thick plate: d == 1 at both faces (boundary, OSS rule 0 < d < 2)
    # and d == 2 for the two middle cells (interior, OSS rule d >= 2)
    col = geo.tags[4, 0:4, 6]
    assert col.tolist() == [PALABOS_BOUNDARY, PALABOS_GRAIN, PALABOS_GRAIN, PALABOS_BOUNDARY]
    # pore codes elsewhere
    assert geo.tags[4, 8, 6] == PALABOS_PORE
    assert geo.counts()["pore"] > 0
    assert geo.counts()["boundary"] % 2 == 0


def test_bc_faces_forced_to_boundary():
    m = np.zeros((6, 6, 6), dtype=bool)          # all fluid: no grain at all
    geo = build_geometry_tags(m, bc_faces=True)
    # with no grain the OSS code still writes 1 into the six faces; the pore
    # reopening then zeroes them, so the class map must stay all pore.
    assert np.all(geo.classes == 0)
    m2 = np.ones((6, 6, 6), dtype=bool)
    geo2 = build_geometry_tags(m2, bc_faces=True)
    assert np.all(geo2.tags[0] == PALABOS_BOUNDARY)


def test_padding_and_swap_xz():
    m = slab_mask(nz=6, ny=8, nx=10, t=2)
    geo = build_geometry_tags(m, num_slices=2)
    assert geo.tags.shape == (10, 8, 10)
    assert np.all(geo.tags[:2] == PALABOS_PORE) and np.all(geo.tags[-2:] == PALABOS_PORE)
    geo_sw = build_geometry_tags(m, swap_xz=True)
    assert geo_sw.tags.shape == (10, 8, 6)
    assert geo_sw.counts() == build_geometry_tags(m).counts()


def test_nw_seed_codes():
    m = slab_mask(t=2)
    nw = np.zeros_like(m, dtype=np.int8)
    nw[:, 6:9, :] = 3
    geo = build_geometry_tags(m, nw_mask=nw)
    assert geo.counts()["nw"] == int(np.count_nonzero(nw == 3))
    assert np.all(geo.tags[nw == 3] == PALABOS_NW)


def test_unknown_palabos_code_rejected():
    with pytest.raises(ValueError):
        decode_palabos_tags(np.array([[[2608, 9999]]], dtype=np.int16))


def test_decode_roundtrip():
    geo = build_geometry_tags(slab_mask())
    back = decode_palabos_tags(geo.tags)
    assert np.array_equal(back, geo.classes)


# ---------------------------------------------------------------------------
# compression
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_labels", [1, 2, 3, 4, 7, 17, 300])
def test_compress_roundtrip(n_labels):
    rng = np.random.default_rng(n_labels)
    labels = rng.integers(0, n_labels, size=(7, 9, 5)).astype(np.int32)
    packed = compress_label_map(labels)
    assert isinstance(packed, PackedLabels)
    back = decompress_label_map(packed)
    assert np.array_equal(back, labels)
    assert packed.nbytes <= labels.size * 4


def test_compression_bits_and_ratio():
    geo = build_geometry_tags(slab_mask(nz=10, ny=20, nx=16, t=4))
    packed = geo.pack_classes()
    assert packed.bits == 2                       # 4 classes -> 2 bit/voxel
    n = geo.classes.size
    assert packed.nbytes == pytest.approx(np.ceil(2 * n / 8), abs=1)
    # 8 B (float64 EDT) -> 0.25 B: 32x fewer bytes
    assert geo.meta["compression"]["ratio_vs_float64"] >= 32.0
    assert np.array_equal(decompress_label_map(packed), geo.classes)


def test_corrupt_packed_map_raises():
    packed = compress_label_map(np.zeros((2, 2, 2), dtype=np.uint8))
    bad = PackedLabels(data=b"\x03" * 4, shape=packed.shape, table=(0,), bits=2,
                       dtype="uint8", num_labels=1)
    with pytest.raises(ValueError):
        decompress_label_map(bad)


def test_grain_id_side_channel_roundtrip():
    m = np.zeros((6, 10, 8), dtype=bool)
    m[:, :3, :] = True
    m[:, 6:, :] = True
    gid = np.zeros_like(m, dtype=np.int32)
    gid[:, :3, :] = 11
    gid[:, 6:, :] = 42
    geo = build_geometry_tags(m, grain_ids=gid)
    packed = geo.meta["grain_ids"]["data"]
    back = decompress_label_map(packed)
    assert np.array_equal(back, np.where(m, gid, 0))


def test_deterministic_and_idempotent():
    m = random_mask()
    a = build_geometry_tags(m)
    b = build_geometry_tags(m)
    assert np.array_equal(a.tags, b.tags)
    assert a.tags.tobytes() == b.tags.tobytes()


def test_erase_regions_is_noop_on_binary_masks():
    m = np.zeros((8, 10, 10), dtype=np.float64)
    m[:, :3, :] = 1.0
    m[:, 6:, :] = 1.0                              # two disconnected grains
    out = erase_regions(m)
    # OSS semantics: label regions of (rock != 1) -> the whole pore space is one
    # component, so nothing is erased (the grains must survive).
    assert np.array_equal(out, m)


# ---------------------------------------------------------------------------
# on-disk artefact
# ---------------------------------------------------------------------------
def test_write_palabos_geometry(tmp_path):
    geo = build_geometry_tags(slab_mask(), num_slices=1, geom_name="slab")
    dat = tmp_path / "sub" / f"{geo.geom_name}.dat"
    side = write_palabos_geometry(dat, geo)
    assert dat.exists()
    raw = np.fromfile(dat, dtype="<i2")
    assert raw.size == geo.tags.size
    assert np.array_equal(raw.reshape(geo.tags.shape), geo.tags)
    loaded = json.loads(dat.with_suffix(".json").read_text())
    assert loaded["shape"] == list(geo.tags.shape)
    assert loaded["codes"]["pore"] == PALABOS_PORE
    assert loaded["format"] == "mplbm-palabos-geometry/1"
    assert side["counts"] == geo.counts()


# ---------------------------------------------------------------------------
# solver-facing integration
# ---------------------------------------------------------------------------
def test_preprocess_geo_adapter_returns_torch():
    m = torch.zeros(8, 12, 16, dtype=torch.bool)
    m[:, :3, :] = True
    tags, depth, meta = palabos_geometry_tags(m, geom_name="torch_case")
    assert isinstance(tags, torch.Tensor) and tags.dtype == torch.int16
    assert isinstance(depth, torch.Tensor) and depth.dtype == torch.float32
    assert tags.shape == depth.shape == (8, 12, 16)
    assert meta["counts"]["grain"] + meta["counts"]["boundary"] > 0


def test_wall_adhesion_operator_flat_wall():
    solid = torch.zeros(4, 5, 6, dtype=torch.bool)
    solid[0] = True                                # wall at z = 0
    psi = torch.ones(4, 5, 6)
    fz, fy, fx = wall_adhesion_force_3d(psi, solid, g_wall=-5.0)
    # closed form: Fz = g_wall * psi * sum_{c_z=-1} w = g_wall / 6
    assert float(fz[1, 2, 2]) == pytest.approx(-5.0 / 6.0, rel=1e-6)
    assert float(fy[1, 2, 2]) == pytest.approx(0.0, abs=1e-7)
    assert float(fx[1, 2, 2]) == pytest.approx(0.0, abs=1e-7)
    # no solid neighbour within one link -> exactly zero
    assert torch.all(fz[2:] == 0)
    # far from the wall the force is zero even at the transverse edges
    assert float(fz[3, 0, 0]) == 0.0


def test_wall_adhesion_per_grain_and_linearity():
    solid = torch.zeros(4, 6, 6, dtype=torch.bool)
    solid[0, :3] = True                            # grain 7 half
    solid[0, 3:] = True                            # grain 9 half
    gid = torch.zeros(4, 6, 6, dtype=torch.int32)
    gid[0, :3] = 7
    gid[0, 3:] = 9
    psi = torch.ones(4, 6, 6)
    fz, _, _ = wall_adhesion_force_3d(psi, solid, g_wall=-5.0, grain_ids=gid,
                                     g_wall_by_grain={7: -2.0, 9: -8.0})
    assert float(fz[1, 1, 2]) == pytest.approx(-2.0 / 6.0, rel=1e-6)
    assert float(fz[1, 4, 2]) == pytest.approx(-8.0 / 6.0, rel=1e-6)
    # linearity in psi (uniform G_wall reference: g_wall / 6)
    fz2, _, _ = wall_adhesion_force_3d(2.0 * psi, solid, g_wall=-5.0)
    assert float(fz2[1, 1, 2]) == pytest.approx(2.0 * (-5.0 / 6.0), rel=1e-5)


def test_wall_adhesion_shape_mismatch():
    with pytest.raises(ValueError):
        wall_adhesion_force_3d(torch.ones(3, 3, 3), torch.zeros(2, 2, 2, dtype=torch.bool))


def test_integration_chain_mask_tags_grain_ids_force():
    """Full chain: two-segment geometry -> compressed tags -> per-grain wall force."""
    segs = [((6.0, 4.0, 6.0), (6.0, 20.0, 6.0)), ((6.0, 20.0, 6.0), (6.0, 20.0, 18.0))]
    m = two_segment_mask((13, 24, 24), segs, radius=3.0)
    gid = np.zeros(m.shape, dtype=np.int32)
    gid[: m.shape[0] // 2] = 1
    gid[m.shape[0] // 2:] = 2
    gid = np.where(m, gid, 0)
    tags, depth, meta = palabos_geometry_tags(m, grain_ids=gid, geom_name="chain")
    assert tags.dtype == torch.int16
    assert meta["counts"]["boundary"] > 0 and meta["counts"]["grain"] > 0
    back_dtype = meta["grain_ids"]["dtype"]
    assert back_dtype.startswith("int32")
    solid = torch.as_tensor(m)
    psi = torch.ones(solid.shape)
    fz, fy, fx = wall_adhesion_force_3d(psi, solid, grain_ids=torch.as_tensor(gid),
                                        g_wall=-5.0, g_wall_by_grain={1: -5.0, 2: -1.0})
    # force only on the boundary shell: interior solid nodes carry no force
    interior = torch.as_tensor(np.asarray(depth) > 1.5)
    assert torch.all(fz[interior] == 0)
    # mirror symmetry in y for the vertical leg
    assert float(fz[6, 20, 10]) == pytest.approx(float(fz[6, 20, 14]), rel=1e-5)


def test_wall_adhesion_zero_outside_boundary_shell():
    m = slab_mask(nz=6, ny=10, nx=10, t=3)
    tags, depth, _ = palabos_geometry_tags(m)
    psi = torch.ones(tuple(m.shape))
    fz, fy, fx = wall_adhesion_force_3d(psi, torch.as_tensor(m), g_wall=-5.0)
    fluid_zero = (tags == PALABOS_PORE) & (depth == 0)
    core = torch.as_tensor(np.zeros(m.shape, dtype=bool))
    # nodes at least 2 cells away from the plates carry no wall force
    core[:, 5, :] = True
    assert torch.all(fx[core] == 0) and torch.all(fy[core] == 0) and torch.all(fz[core] == 0)
    assert fluid_zero.any()                       # sanity: mask not empty