"""Boxel STL ingestion round trips: mask -> STL -> mask -> SDF.

Pins the contracts of :mod:`tensorlbm.geometry_stl`:

1. boxel export is watertight, outward-oriented and volume-exact
   (divergence-theorem volume == solid voxel count, 2 triangles per
   exposed quad face);
2. mask -> STL -> mask is **bit-exact** on the same grid for both STL
   payloads (binary and ASCII) — the guarantee a tessellated smooth-CAD
   mesh cannot give;
3. the STL -> SDF chain reproduces the CAD-leg ``geom_encoder.sdf_volume``
   output bit-exactly (same clip/pool chain as the corpus);
4. general (non-boxel) meshes: hand-built cube and cube-with-cavity
   triangle tables rasterise to the exact expected masks;
5. non-watertight meshes are rejected by default and demonstrably leak
   parity (column streaks) when the guard is disabled;
6. corpus-real shapes: SUBOFF CAD masks round trip exactly, including
   one full PRODUCTION_GRID (64x64x128) case.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tensorlbm.ai.drag_cond import PRODUCTION_GRID, SuboffGrid
from tensorlbm.ai.geom_encoder import SDF_CLIP_VOXELS, SDF_POOL_STRIDE, sdf_volume
from tensorlbm.geometry_stl import mask_to_stl, stl_to_mask, stl_to_sdf, write_mask_stl
from tensorlbm.suboff_cad import SuboffConfig, build_suboff_mask
from tensorlbm.voxelize import is_watertight, load_stl

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def box_mask(
    shape: tuple[int, int, int],
    lo: tuple[int, int, int],
    hi: tuple[int, int, int],
) -> np.ndarray:
    """Solid box ``[lo, hi)`` (z, y, x half-open) in a ``shape`` mask."""
    m = np.zeros(shape, dtype=bool)
    z0, y0, x0 = lo
    m[z0 : hi[0], y0 : hi[1], x0 : hi[2]] = True
    return m


def axis_box(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> np.ndarray:
    """Closed axis-aligned box as 12 hand-written outward-oriented triangles.

    Independent of :func:`tensorlbm.geometry_stl.mask_to_stl` — the
    general-mesh leg is pinned against a mesh authored here, not against
    the module under test.
    """
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    quads = [
        ((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)),  # -x
        ((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)),  # +x
        ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)),  # -y
        ((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)),  # +y
        ((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)),  # -z
        ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)),  # +z
    ]
    tris: list[list[tuple[float, float, float]]] = []
    for a, b, c, d in quads:
        tris.extend([[a, b, c], [a, c, d]])
    return np.asarray(tris, dtype=np.float64)


def open_box(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> np.ndarray:
    """Axis box with the +z cap removed (10 triangles, NOT watertight)."""
    full = axis_box(lo, hi)
    keep = np.ones(full.shape[0], dtype=bool)
    keep[-2:] = False  # last quads entry is +z
    return full[keep]


def signed_volume(tris: np.ndarray) -> float:
    """Divergence-theorem volume of a closed triangle mesh."""
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(np.einsum("ij,ij->", v0, np.cross(v1, v2)).sum() / 6.0)


def exposed_face_count(mask: np.ndarray) -> int:
    """Independent count of exposed voxel faces (border counts as fluid)."""
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    total = 0
    for axis in range(3):
        for sl in (
            slice(2, None),
            slice(0, -2),
        ):
            selector = [slice(1, -1)] * 3
            selector[axis] = sl
            total += int((mask & ~padded[tuple(selector)]).sum())
    return total


def cad_mask(
    hull: str = "full",
    sail: float = 1.0,
    fin: float = 1.0,
    mults: dict[str, float] | None = None,
    grid: SuboffGrid | None = None,
) -> np.ndarray:
    """Corpus-real SUBOFF mask through the CAD path (the e2e pattern)."""
    g = grid if grid is not None else PRODUCTION_GRID
    cfg = SuboffConfig(sail_scale=sail, fin_scale=fin, **(mults or {}))
    m, _stats = build_suboff_mask(
        hull_type=hull,
        nx=g.nx,
        ny=g.ny,
        nz=g.nz,
        cx=g.cx,
        cy=g.cy,
        cz=g.cz,
        length=g.length,
        config=cfg,
        device="cpu",
    )
    return np.asarray(m, dtype=bool)


SMALL_MASKS: list[str] = ["single", "box", "notched", "all_solid", "checker"]

#: Masks whose solid voxels touch only along edges/corners (checkerboard)
#: have a PINCHED boundary: four faces share an edge, so the closed
#: surface is not edge-manifold and :func:`is_watertight` rejects it.
#: Ray parity still round-trips them exactly — the flag is the documented
#: diagnostic escape hatch for such degenerate (unphysical-for-hulls)
#: masks.
PINCHED_MASKS: set[str] = {"checker"}


def make_small_mask(name: str) -> np.ndarray:
    shape = (6, 7, 8)
    if name == "single":
        m = np.zeros(shape, dtype=bool)
        m[3, 4, 5] = True
        return m
    if name == "box":
        return box_mask(shape, (1, 2, 3), (4, 6, 7))
    if name == "notched":
        m = box_mask(shape, (0, 0, 0), (5, 6, 7))
        m[0, 0, 0] = False  # corner notch
        return m
    if name == "all_solid":
        return np.ones(shape, dtype=bool)
    iz, iy, ix = np.indices(shape)
    return (iz + iy + ix) % 2 == 0  # checkerboard: every face exposed


# ---------------------------------------------------------------------------
# 1-2. Boxel export contract + bit-exact round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("binary", [True, False], ids=["binary", "ascii"])
@pytest.mark.parametrize("name", SMALL_MASKS)
def test_boxel_roundtrip_bit_exact(name: str, binary: bool) -> None:
    mask = make_small_mask(name)
    back = stl_to_mask(
        mask_to_stl(mask, binary=binary), mask.shape, require_watertight=name not in PINCHED_MASKS
    )
    assert np.array_equal(back, mask)


def test_mask_to_stl_watertight_volume_facecount(tmp_path: Path) -> None:
    for name in SMALL_MASKS:
        mask = make_small_mask(name)
        path = write_mask_stl(tmp_path / f"{name}.stl", mask)
        mesh = load_stl(path)
        if name not in PINCHED_MASKS:
            assert is_watertight(mesh), name
        # 2 triangles per exposed quad face, no interior faces
        assert mesh.vertices.shape[0] == 2 * exposed_face_count(mask), name
        # closed outward-oriented shell encloses exactly the solid voxels
        assert signed_volume(mesh.vertices) == pytest.approx(float(mask.sum())), name


def test_write_mask_stl_file_roundtrip(tmp_path: Path) -> None:
    mask = box_mask((5, 6, 7), (1, 1, 1), (4, 5, 6))
    mesh = load_stl(write_mask_stl(tmp_path / "f.stl", mask))
    assert np.array_equal(stl_to_mask(mesh, mask.shape), mask)


def test_empty_and_nonbool_masks_rejected() -> None:
    with pytest.raises(ValueError, match="no solid voxels"):
        mask_to_stl(np.zeros((3, 3, 3), dtype=bool))
    with pytest.raises(TypeError, match="boolean"):
        mask_to_stl(np.ones((3, 3, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="nz, ny, nx"):
        mask_to_stl(np.ones(3, dtype=bool))


# ---------------------------------------------------------------------------
# 3. General (non-boxel) meshes
# ---------------------------------------------------------------------------


def test_general_mesh_cube() -> None:
    shape = (4, 4, 4)
    tris = axis_box((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
    assert is_watertight(tris)
    expected = np.zeros(shape, dtype=bool)
    expected[0:2, 0:2, 0:2] = True
    assert np.array_equal(stl_to_mask(tris, shape), expected)


def test_general_mesh_cube_with_cavity() -> None:
    # 3x3x3 cube with an interior 1-voxel cavity (reversed inner shell):
    # mesh = outer box + inner box with flipped winding.
    shape = (5, 5, 5)
    outer = axis_box((0.0, 0.0, 0.0), (3.0, 3.0, 3.0))
    inner = axis_box((1.0, 1.0, 1.0), (2.0, 2.0, 2.0))[:, ::-1]
    tris = np.concatenate([outer, inner])
    assert is_watertight(tris)
    expected = np.zeros(shape, dtype=bool)
    expected[0:3, 0:3, 0:3] = True
    expected[1, 1, 1] = False
    assert np.array_equal(stl_to_mask(tris, shape), expected)


def test_non_watertight_raises_and_leaks() -> None:
    shape = (4, 4, 4)
    # The breach is the missing +z cap; the box sits one cell off the
    # ray-origin plane so the remaining -z cap is strictly ahead (t > 0)
    # of the z-directed rays.
    tris = open_box((0.0, 0.0, 1.0), (2.0, 2.0, 3.0))
    assert not is_watertight(tris)
    with pytest.raises(ValueError, match="not watertight"):
        stl_to_mask(tris, shape)
    # guard off + ray along z (the breach axis): the unpaired entry
    # crossing flips every voxel column behind it — solid below the box,
    # true interior lost — the documented parity-leak failure mode.
    closed = stl_to_mask(axis_box((0.0, 0.0, 1.0), (2.0, 2.0, 3.0)), shape, axis=2)
    leaked = stl_to_mask(tris, shape, axis=2, require_watertight=False)
    assert not np.array_equal(leaked, closed)
    assert leaked[0, 0:2, 0:2].all()  # phantom solid below the open box
    assert not leaked[1:3, 0:2, 0:2].any()  # true interior lost
    assert not leaked[:, 2:, 2:].any()  # outside the footprint stays fluid


# ---------------------------------------------------------------------------
# 4. SDF chain agreement (STL leg == CAD leg, bit-exact)
# ---------------------------------------------------------------------------


def test_stl_to_sdf_bit_exact_vs_cad_chain() -> None:
    grid = SuboffGrid.from_resolution(64)  # (32, 32, 64) raw mask
    mask = cad_mask("full", grid=grid)
    sdf_cad = sdf_volume(torch.from_numpy(mask))[0, 0].numpy()
    sdf_stl = stl_to_sdf(mask_to_stl(mask), mask.shape)
    assert sdf_stl.shape == (grid.nz // 2, grid.ny // 2, grid.nx // 2)
    assert sdf_stl.dtype == np.float32
    assert np.array_equal(sdf_stl, sdf_cad)
    # defaults must be the corpus constants, not copies of them
    assert SDF_CLIP_VOXELS == 8.0 and SDF_POOL_STRIDE == 2


def test_stl_to_sdf_shape_and_bounds() -> None:
    mask = box_mask((8, 8, 8), (2, 2, 2), (6, 6, 6))
    vol = stl_to_sdf(mask_to_stl(mask), (8, 8, 8), clip=2.0, pool=2)
    assert vol.shape == (4, 4, 4)
    assert float(np.abs(vol).max()) <= 1.0


# ---------------------------------------------------------------------------
# 5. Corpus-real round trips (incl. one PRODUCTION_GRID case)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hull", "mults"),
    [
        ("bare_hull", {}),
        ("with_sail", {"l_over_d_mult": 1.25}),
        ("full", {"nose_len_mult": 1.4}),
    ],
)
def test_cad_mask_roundtrip_small_grid(hull: str, mults: dict[str, float]) -> None:
    grid = SuboffGrid.from_resolution(64)
    mask = cad_mask(hull, mults=mults, grid=grid)
    assert np.array_equal(stl_to_mask(mask_to_stl(mask), mask.shape), mask)


def test_production_grid_roundtrip_bit_exact() -> None:
    mask = cad_mask("full", mults={"l_over_d_mult": 1.3}, grid=PRODUCTION_GRID)
    assert mask.shape == (64, 64, 128)
    stl = mask_to_stl(mask)
    back = stl_to_mask(stl, mask.shape)
    assert np.array_equal(back, mask)
    # and the full SDF chain agrees bit-exactly with the CAD leg
    sdf_cad = sdf_volume(torch.from_numpy(mask))[0, 0].numpy()
    assert np.array_equal(stl_to_sdf(stl, mask.shape), sdf_cad)
