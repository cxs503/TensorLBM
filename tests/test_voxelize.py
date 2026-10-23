"""B4-P2a general voxelisation: STL -> mask / SDF on the canonical grid.

All fixtures are generated in-test (no binary blobs): an axis-aligned
box, a subdivided icosphere and rotations thereof pin the exact
contracts of :mod:`tensorlbm.voxelize` -- layout, cell-centre sampling,
half-open boundary handling, the robust ray-parity tie-break and the
loader quirks of real CAD exports.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from tensorlbm.voxelize import (
    Placement,
    StlMesh,
    is_watertight,
    load_stl,
    mask_from_stl,
    place_on_grid,
    sdf_from_mask,
)

# ---------------------------------------------------------------------------
# Mesh fixtures (in-test generators, no dependencies)
# ---------------------------------------------------------------------------


def axis_box(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> np.ndarray:
    """Closed axis-aligned box as 12 consistently outward-oriented triangles.

    Quad diagonals run from the (y0, z0) corner to the (y1, z1) corner on
    the x faces, so with half-integer bounds the diagonals lie exactly on
    ray lines -- the degenerate shared-edge configuration of the
    robustness tests below.
    """
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    quads = [
        # (outward axis, sign, four corners in CCW order seen from outside)
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


def icosphere(subdiv: int) -> np.ndarray:
    """Unit icosphere: 20-face icosahedron + midpoint subdivision."""
    t = (1.0 + 5.0**0.5) / 2.0
    verts = [
        (-1, t, 0),
        (1, t, 0),
        (-1, -t, 0),
        (1, -t, 0),
        (0, -1, t),
        (0, 1, t),
        (0, -1, -t),
        (0, 1, -t),
        (t, 0, -1),
        (t, 0, 1),
        (-t, 0, -1),
        (-t, 0, 1),
    ]
    verts = [v / np.linalg.norm(v) for v in verts]
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    vlist: list[list[list[float]]] = [list(map(float, v)) for v in verts]
    for _ in range(subdiv):
        cache: dict[tuple[int, int], int] = {}
        new_faces: list[tuple[int, int, int]] = []

        def midpoint(i: int, j: int) -> int:
            key = (i, j) if i < j else (j, i)
            if key not in cache:
                m = (np.asarray(vlist[i]) + np.asarray(vlist[j])) / 2.0
                vlist.append(list(m / np.linalg.norm(m)))
                cache[key] = len(vlist) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])
        faces = new_faces
    pts = np.asarray(vlist, dtype=np.float64)
    return pts[np.asarray(faces, dtype=np.int64)]


def mesh_volume(tris: np.ndarray) -> float:
    """Exact enclosed volume via the divergence theorem."""
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(abs(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0))


def rotate(tris: np.ndarray, rx_deg: float, ry_deg: float) -> np.ndarray:
    """Rotate a mesh about x then y, preserving its centre of geometry."""
    ax, ay = np.radians(rx_deg), np.radians(ry_deg)
    mx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    my = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rot = my @ mx
    centre = tris.reshape(-1, 3).mean(axis=0)
    return (tris - centre) @ rot.T + centre


def write_binary_stl(path: Path, tris: np.ndarray, header: bytes = b"bin") -> None:
    with path.open("wb") as fh:
        fh.write(header.ljust(80, b"\0"))
        fh.write(struct.pack("<I", tris.shape[0]))
        for tri in tris:
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            n = n / np.linalg.norm(n)
            fh.write(struct.pack("<3f", *n))
            for v in tri:
                fh.write(struct.pack("<3f", *v))
            fh.write(struct.pack("<H", 0))


def write_ascii_stl(path: Path, tris: np.ndarray, name: str = "box") -> None:
    lines = [f"solid {name}"]
    for tri in tris:
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        n = n / np.linalg.norm(n)
        lines.append(f"facet normal {n[0]:.9e} {n[1]:.9e} {n[2]:.9e}")
        lines.append("outer loop")
        for v in tri:
            lines.append(f"vertex {v[0]:.9e} {v[1]:.9e} {v[2]:.9e} ")
        lines.append("endloop")
        lines.append("endfacet")
    lines.append(f"endsolid {name}")
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("ascii"))


BOX = ((2.0, 3.0, 4.0), (6.0, 7.0, 8.0))
TRIS_BOX = axis_box(*BOX)


def expected_box_mask(
    shape: tuple[int, int, int],
    lo: tuple[float, float, float],
    hi: tuple[float, float, float],
) -> np.ndarray:
    """Half-open cell-centre occupancy of an axis box."""
    nz, ny, nx = shape
    ix = np.arange(nx) + 0.5
    iy = np.arange(ny) + 0.5
    iz = np.arange(nz) + 0.5
    inside = (
        (ix >= lo[0]) & (ix < hi[0]),
        (iy >= lo[1]) & (iy < hi[1]),
        (iz >= lo[2]) & (iz < hi[2]),
    )
    return inside[2][:, None, None] & inside[1][None, :, None] & inside[0][None, None, :]


# ---------------------------------------------------------------------------
# STL loader
# ---------------------------------------------------------------------------


def test_load_stl_binary_vs_ascii_equivalence(tmp_path: Path) -> None:
    write_binary_stl(tmp_path / "b.stl", TRIS_BOX)
    write_ascii_stl(tmp_path / "a.stl", TRIS_BOX)
    b = load_stl(tmp_path / "b.stl")
    a = load_stl(tmp_path / "a.stl")
    assert b.vertices.shape == (12, 3, 3)
    # quarter-integer coordinates are fp32-exact, so both loaders agree bitwise
    assert np.array_equal(a.vertices, b.vertices)
    assert a.normals is not None and b.normals is not None
    assert np.allclose(a.normals, b.normals, atol=1e-7)
    for mesh in (a, b):
        mask = mask_from_stl(mesh, (16, 16, 16))
        assert np.array_equal(mask, expected_box_mask((16, 16, 16), *BOX))


def test_load_stl_binary_with_solid_header(tmp_path: Path) -> None:
    # binary size match must win over the "solid" header prefix
    write_binary_stl(tmp_path / "quirk.stl", TRIS_BOX, header=b"solid exported by CAD")
    mesh = load_stl(tmp_path / "quirk.stl")
    assert np.array_equal(mesh.vertices, TRIS_BOX)


def test_load_stl_ascii_quirks(tmp_path: Path) -> None:
    # empty solid + a real solid, CRLF, trailing spaces, endsolid names
    data = (
        b"solid empty\r\nendsolid empty\r\n"
        b"solid real\r\nfacet normal 0 0 0\r\nouter loop\r\n"
        b"vertex 0 0 0\r\nvertex 1 0 0\r\nvertex 0 1 0\r\n"
        b"endloop\r\nendfacet\r\nendsolid real\r\n"
    )
    (tmp_path / "q.stl").write_bytes(data)
    mesh = load_stl(tmp_path / "q.stl")
    assert mesh.vertices.shape == (1, 3, 3)
    assert mesh.normals is None  # all-zero normals read as absent

    (tmp_path / "empty.stl").write_bytes(b"solid x\r\nendsolid x\r\n")
    empty = load_stl(tmp_path / "empty.stl")
    assert empty.vertices.shape == (0, 3, 3)


def test_load_stl_truncated_binary(tmp_path: Path) -> None:
    write_binary_stl(tmp_path / "t.stl", TRIS_BOX)
    data = (tmp_path / "t.stl").read_bytes()
    (tmp_path / "t.stl").write_bytes(data[:-17])
    with pytest.raises(ValueError, match=r"byte offset 634"):
        load_stl(tmp_path / "t.stl")


def test_load_stl_truncated_ascii(tmp_path: Path) -> None:
    write_ascii_stl(tmp_path / "t.stl", TRIS_BOX)
    data = (tmp_path / "t.stl").read_bytes()
    cut = data.index(b"endloop") - 5
    (tmp_path / "t.stl").write_bytes(data[:cut])
    with pytest.raises(ValueError, match=r"byte offset \d+"):
        load_stl(tmp_path / "t.stl")


def test_load_stl_short_binary(tmp_path: Path) -> None:
    (tmp_path / "s.stl").write_bytes(b"\0" * 50)
    with pytest.raises(ValueError, match="too short"):
        load_stl(tmp_path / "s.stl")
    (tmp_path / "e.stl").write_bytes(b"")
    with pytest.raises(ValueError, match="empty STL file"):
        load_stl(tmp_path / "e.stl")


# ---------------------------------------------------------------------------
# Watertight check
# ---------------------------------------------------------------------------


def test_is_watertight_box_and_icosphere() -> None:
    assert is_watertight(TRIS_BOX)
    assert is_watertight(icosphere(0))
    assert is_watertight(icosphere(2))


def test_is_watertight_open_box() -> None:
    open_box = np.delete(TRIS_BOX, [10, 11], axis=0)  # drop one whole face
    assert not is_watertight(open_box)
    flipped = TRIS_BOX.copy()
    flipped[0] = flipped[0][::-1]  # single face wound backwards
    assert not is_watertight(flipped)


# ---------------------------------------------------------------------------
# Ray-parity voxeliser
# ---------------------------------------------------------------------------


def test_mask_axis_box_exact_at_integer_placement() -> None:
    shape = (16, 16, 16)
    mask = mask_from_stl(TRIS_BOX, shape)
    assert mask.dtype == np.bool_
    assert mask.shape == shape
    assert np.array_equal(mask, expected_box_mask(shape, *BOX))
    # every ray axis must give the identical mask (closed surface)
    for axis in (1, 2):
        assert np.array_equal(mask_from_stl(TRIS_BOX, shape, axis=axis), mask)
    # non-default origin/spacing map mesh coords to indices the same way
    shifted = mask_from_stl(TRIS_BOX + 5.0, shape, origin=(5.0, 5.0, 5.0))
    assert np.array_equal(shifted, mask)
    scaled = mask_from_stl(TRIS_BOX * 2.0, shape, spacing=2.0)
    assert np.array_equal(scaled, mask)


def test_mask_robust_against_degenerate_shared_edges() -> None:
    # half-integer bounds put face diagonals and vertices exactly on ray
    # lines: the configuration that breaks naive inclusive parity
    # the transverse faces (y, z) sit exactly on ray lines; the ray-axis
    # (x) faces sit between samples so their crossing side is unambiguous
    lo, hi = (2.7, 3.5, 3.5), (6.7, 7.5, 7.5)
    shape = (12, 12, 12)
    tris = axis_box(lo, hi)
    robust = mask_from_stl(tris, shape)
    # Exact contract of the deterministic perturbation (module epsilons
    # +1.3e-4 cells on t1=y, -3.7e-4 on t2=z):
    #   x (ray axis, strict t>0): half-open [2.7, 6.7) -> ix in {3..6}
    #   y: ray at iy+0.5+1.3e-4 inside [3.5, 7.5)      -> iy in {3..6}
    #   z: ray at iz+0.5-3.7e-4 inside [3.5, 7.5)      -> iz in {4..7}
    ix = np.arange(12)
    expected = (
        ((ix + 0.5 >= 2.7) & (ix + 0.5 < 6.7))[None, None, :]
        & ((ix + 0.5 + 1.3e-4 >= 3.5) & (ix + 0.5 + 1.3e-4 < 7.5))[None, :, None]
        & ((ix + 0.5 - 3.7e-4 >= 3.5) & (ix + 0.5 - 3.7e-4 < 7.5))[:, None, None]
    )
    assert np.array_equal(robust, expected)
    # column (iz=4, iy=4) through the diagonal midpoint: filled by robust
    assert robust[4, 4, :].sum() == 4
    # naive inclusive parity corrupts the diagonal/vertex columns: the
    # exact pattern is fp-rounding noise, but every column whose ray
    # lies on the shared diagonal is wrong somewhere
    naive = mask_from_stl(tris, shape, robust=False)
    assert not np.array_equal(naive, robust)
    bad_cols = {(int(a), int(b)) for a, b in np.argwhere((naive != robust).any(axis=2))}
    assert len(bad_cols) >= 8
    for iz, iy in ((4, 4), (5, 5), (6, 6)):
        assert (iz, iy) in bad_cols
    # column through the shared vertex (iz=3, iy=3): the -z perturbation
    # resolves the ray to the outside, so the whole column is excluded
    assert robust[3, 3, :].sum() == 0
    # column (iz=7, iy=7): the +y perturbation resolves it outside
    assert robust[7, 7, :].sum() == 0


def test_mask_icosphere_volume_convergence() -> None:
    shape = (64, 64, 64)
    tris = icosphere(2)
    placement = place_on_grid(tris, shape)
    mask = mask_from_stl(placement.tris, shape)
    v_vox = float(mask.sum())
    v_exact = mesh_volume(tris) * placement.scale**3
    assert abs(v_vox - v_exact) / v_exact < 0.05
    assert is_watertight(tris)


def test_mask_rotated_box_volume() -> None:
    shape = (64, 64, 64)
    side = 10.0
    tris = axis_box((0.0, 0.0, 0.0), (side, side, side))
    rotated = rotate(tris, 30.0, 45.0)
    assert is_watertight(rotated)
    placement = place_on_grid(rotated, shape)
    mask = mask_from_stl(placement.tris, shape)
    expected = side**3 * placement.scale**3
    assert abs(float(mask.sum()) - expected) / expected < 0.05


def test_mask_empty_mesh_is_all_false() -> None:
    mask = mask_from_stl(np.zeros((0, 3, 3)), (4, 5, 6))
    assert mask.shape == (4, 5, 6) and not mask.any()


def test_mask_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="shape"):
        mask_from_stl(TRIS_BOX, (0, 4, 4))
    with pytest.raises(ValueError, match="axis"):
        mask_from_stl(TRIS_BOX, (4, 4, 4), axis=3)
    with pytest.raises(ValueError, match="spacing"):
        mask_from_stl(TRIS_BOX, (4, 4, 4), spacing=0.0)
    with pytest.raises(ValueError, match="shape"):
        mask_from_stl(np.zeros((2, 3)), (4, 4, 4))


# ---------------------------------------------------------------------------
# Canonical-frame placement
# ---------------------------------------------------------------------------


def test_place_on_grid_contract() -> None:
    shape = (64, 64, 64)
    placement = place_on_grid(TRIS_BOX, shape)
    assert isinstance(placement, Placement)
    assert placement.origin.shape == (3,) and np.array_equal(placement.origin, np.zeros(3))
    assert placement.spacing == 1.0
    # streamwise extent hits streamwise_frac * nx
    assert placement.streamwise_extent == pytest.approx(0.6 * 64)
    flat = placement.tris.reshape(-1, 3)
    centre = 0.5 * (flat.min(axis=0) + flat.max(axis=0))
    assert np.allclose(centre, [0.35 * 64, 0.5 * 64, 0.5 * 64])
    # uniform scale preserves aspect: y/z extents scale by the same factor
    extent = flat.max(axis=0) - flat.min(axis=0)
    src = TRIS_BOX.reshape(-1, 3)
    src_extent = src.max(axis=0) - src.min(axis=0)
    assert np.allclose(extent, src_extent * placement.scale)
    # nose (upstream -x end) lands at 0.05 * nx per the SUBOFF convention
    assert flat[:, 0].min() == pytest.approx(0.05 * 64)
    # round trip straight into the voxeliser
    mask = mask_from_stl(placement.tris, shape, origin=placement.origin, spacing=placement.spacing)
    assert mask.any()
    assert mask.sum() == pytest.approx(4 * 4 * 4 * placement.scale**3, rel=0.05)


def test_place_on_grid_explicit_scale_and_validation() -> None:
    shape = (16, 16, 16)
    placement = place_on_grid(TRIS_BOX, shape, scale=0.5)
    extent = placement.tris.reshape(-1, 3)
    assert (extent.max(axis=0) - extent.min(axis=0))[0] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="streamwise"):
        place_on_grid(icosphere(0)[:1], shape, streamwise_frac=2.0)
    flat = TRIS_BOX.copy().reshape(-1, 3)
    flat[:, 0] = flat[:, 0].mean()  # zero streamwise extent
    with pytest.raises(ValueError, match="zero streamwise"):
        place_on_grid(flat.reshape(-1, 3, 3), shape)
    with pytest.raises(ValueError, match="center_frac"):
        place_on_grid(TRIS_BOX, shape, center_frac=(1.5, 0.5, 0.5))


# ---------------------------------------------------------------------------
# SDF seam
# ---------------------------------------------------------------------------


def test_sdf_sign_and_boundary() -> None:
    shape = (32, 32, 32)
    tris = icosphere(1)
    placement = place_on_grid(tris, shape, streamwise_frac=0.4)
    mask = mask_from_stl(placement.tris, shape)
    sdf = sdf_from_mask(mask, spacing=1.0)  # 32^3 -> exact back-end by default
    assert sdf.shape == mask.shape
    assert (sdf[mask] <= 0).all() and (sdf[~mask] >= 0).all()
    assert sdf[mask].min() < 0.0
    assert np.isfinite(sdf).all()
    assert sdf.min() == pytest.approx(-placement.streamwise_extent / 2, abs=2.0)


def test_sdf_chamfer_matches_bruteforce() -> None:
    shape = (32, 32, 32)
    tris = icosphere(1)
    placement = place_on_grid(tris, shape, streamwise_frac=0.4)
    mask = mask_from_stl(placement.tris, shape)
    exact = sdf_from_mask(mask, exact=True)
    chamfer = sdf_from_mask(mask, exact=False)
    rel = np.abs(chamfer - exact) / np.maximum(np.abs(exact), 1.0)
    assert float(rel.max()) < 0.12  # worst case ~11%: shallow diagonals like (3,1,1)
    assert float(np.median(rel)) < 0.05  # typical error a few percent
    # deep inside the solid both back-ends agree to a few percent
    deep = exact < -4.0
    assert deep.any()
    assert float(np.abs(chamfer[deep] - exact[deep]).max()) < 0.5


def test_sdf_rejects_degenerate_masks() -> None:
    with pytest.raises(ValueError, match="no boundary"):
        sdf_from_mask(np.zeros((4, 4, 4), dtype=bool))
    with pytest.raises(ValueError, match="3-D"):
        sdf_from_mask(np.ones((4, 4), dtype=bool))
    with pytest.raises(ValueError, match="spacing"):
        sdf_from_mask(np.ones((4, 4, 4), dtype=bool), spacing=-1.0)


def test_stlmesh_accepted_by_every_entrypoint() -> None:
    mesh = StlMesh(vertices=TRIS_BOX.copy(), normals=None)
    assert is_watertight(mesh)  # StlMesh accepted wherever raw tables are
    shape = (16, 16, 16)
    assert np.array_equal(mask_from_stl(mesh, shape), mask_from_stl(TRIS_BOX, shape))
    placement = place_on_grid(mesh, shape, scale=0.5)
    assert placement.tris.shape == TRIS_BOX.shape
