"""Spatial hash-grid acceleration of STL voxelisation: parity + behaviour.

Validates :mod:`tensorlbm.voxel_accel` (the ``accelerate=True`` path of
:func:`tensorlbm.geometry_voxel.voxelize_stl`) against the brute-force
reference:

* the pure-torch fp64 accelerated path must reproduce the reference
  **bit-for-bit** on the same device (integer crossing counts and an
  order-invariant min-t make the reductions exact, and the per-pair
  arithmetic is copied verbatim);
* the opt-in binned Triton kernels must reproduce the brute-force Triton
  kernels' solid/boundary masks bit-for-bit, with ``q`` within fp32
  rounding dust (FMA contraction differs between kernel tile shapes;
  max 2.9e-6 measured over all benchmarked mesh/grid combinations, zero
  entries beyond 1e-3 — quantified in
  ``docs/benchmarks/voxel_accel_benchmark.md``);
* ``accelerate=False`` (the default) must leave behaviour bit-identical
  to the pre-acceleration code path;
* the synthetic icosphere generator must be deterministic (bit-identical
  STL bytes across calls) with the exact ``20 * 4**n`` face count.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tensorlbm import voxel_accel
from tensorlbm.geometry_voxel import (
    mesh_watertight_status,
    q_field_reference,
    read_stl_triangles,
    solid_mask_parity_reference,
    voxelize_stl,
    voxelize_stl_reference,
)
from tensorlbm.stl_geometry import make_icosphere_stl, write_stl

# fp32 rounding dust between kernel tile shapes: max 2.9e-6 measured
# across every benchmarked mesh (2e4-1.3e6 faces) x grid (34^3-128^3)
# combination, worst on the coarsest meshes.  Anything a missed bin
# candidate would produce (wrong min-t) is O(0.1), seven orders above.
_Q_TOL = 4.0e-6


@pytest.fixture(scope="module")
def gpu_device():
    """CUDA device string, or skip when the GPU kernel module is absent."""
    if not torch.cuda.is_available():
        pytest.skip("GPU voxelisation tests require CUDA")
    try:
        from tensorlbm import _voxel_kernels  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"GPU kernel module unavailable: {exc}")
    return "cuda:0"


def _icosphere(tmp_path: Path, subdivisions: int, center, radius, scale=(1.0, 1.0, 1.0)) -> Path:
    verts, faces = make_icosphere_stl(center, radius, subdivisions=subdivisions, scale=scale)
    path = tmp_path / f"ico_s{subdivisions}.stl"
    write_stl(path, verts, faces, binary=True)
    return path


def _reference(tri: torch.Tensor, shape, **kw):
    solid = solid_mask_parity_reference(tri, shape, **kw)
    bnd, q = q_field_reference(tri, solid, **kw)
    return solid, bnd, q


def _accelerated(tri: torch.Tensor, shape, **kw):
    solid = voxel_accel.solid_mask_parity_accelerated(tri, shape, **kw)
    bnd, q = voxel_accel.q_field_accelerated(tri, solid, **kw)
    return solid, bnd, q


def _assert_bitwise(ref, acc) -> None:
    for name, r, a in zip(("solid", "boundary", "q"), ref, acc):
        assert torch.equal(r, a), f"{name} differs from the brute-force reference"


# ---------------------------------------------------------------------------
# 1. SYNTHETIC MESH GENERATOR
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subdivisions", [0, 1, 2, 3])
def test_icosphere_face_count_formula(subdivisions: int) -> None:
    verts, faces = make_icosphere_stl((0.0, 0.0, 0.0), 1.0, subdivisions=subdivisions)
    assert faces.shape[0] == 20 * 4**subdivisions
    # Euler's formula for subdivided icosahedra.
    assert verts.shape[0] == 10 * 4**subdivisions + 2


def test_icosphere_generator_is_deterministic(tmp_path: Path) -> None:
    v1, f1 = make_icosphere_stl((3.0, 4.0, 5.0), 7.5, subdivisions=3, scale=(1.0, 0.8, 1.7))
    v2, f2 = make_icosphere_stl((3.0, 4.0, 5.0), 7.5, subdivisions=3, scale=(1.0, 0.8, 1.7))
    assert np.array_equal(v1, v2)
    assert np.array_equal(f1, f2)
    p1, p2 = tmp_path / "a.stl", tmp_path / "b.stl"
    write_stl(p1, v1, f1, binary=True)
    write_stl(p2, v2, f2, binary=True)
    assert p1.read_bytes() == p2.read_bytes()


def test_icosphere_is_watertight() -> None:
    verts, faces = make_icosphere_stl((1.0, 1.0, 1.0), 2.0, subdivisions=3)
    tri = torch.from_numpy(verts[faces].astype(np.float32))
    status = mesh_watertight_status(tri)
    assert status["watertight"]
    assert status["boundary_edges"] == 0
    assert status["nonmanifold_edges"] == 0


# ---------------------------------------------------------------------------
# 2. BITWISE PARITY vs THE BRUTE-FORCE REFERENCE (fp64, same device)
# ---------------------------------------------------------------------------


def test_parity_bitwise_small_mesh(tmp_path: Path) -> None:
    tri = read_stl_triangles(_icosphere(tmp_path, 3, (17.3, 16.4, 16.8), 9.5))
    shape = (34, 34, 34)
    ref = _reference(tri, shape)
    _assert_bitwise(ref, _accelerated(tri, shape))
    assert int(ref[0].sum()) > 0  # sphere inside the grid


def test_parity_bitwise_ellipsoid_with_origin_and_spacing(tmp_path: Path) -> None:
    tri = read_stl_triangles(
        _icosphere(tmp_path, 4, (24.0, 20.0, 22.0), 13.0, scale=(1.0, 0.8, 1.7))
    )
    shape = (30, 36, 44)
    kw = {"origin": (-3.0, 2.5, 0.0), "spacing": (0.5, 0.7, 1.3)}
    _assert_bitwise(_reference(tri, shape, **kw), _accelerated(tri, shape, **kw))


def test_parity_bitwise_coarse_mesh_multi_bin_triangles(tmp_path: Path) -> None:
    # 80 large triangles whose AABBs span dozens of bins each.
    tri = read_stl_triangles(_icosphere(tmp_path, 1, (17.0, 17.0, 17.0), 12.0))
    shape = (34, 34, 34)
    _assert_bitwise(_reference(tri, shape), _accelerated(tri, shape))


def test_parity_bitwise_mesh_fully_outside_grid(tmp_path: Path) -> None:
    tri = read_stl_triangles(_icosphere(tmp_path, 2, (100.0, 100.0, 100.0), 10.0))
    shape = (24, 24, 24)
    ref = _reference(tri, shape)
    assert not bool(ref[0].any())
    _assert_bitwise(ref, _accelerated(tri, shape))


def test_parity_bitwise_mesh_straddling_grid_edge(tmp_path: Path) -> None:
    tri = read_stl_triangles(_icosphere(tmp_path, 4, (34.0, 17.0, 17.0), 15.0))
    shape = (34, 34, 34)
    _assert_bitwise(_reference(tri, shape), _accelerated(tri, shape))


# ---------------------------------------------------------------------------
# 3. PUBLIC API FLAG SEMANTICS
# ---------------------------------------------------------------------------


def test_accelerate_flag_matches_reference_bitwise(tmp_path: Path) -> None:
    path = _icosphere(tmp_path, 3, (17.3, 16.4, 16.8), 9.5)
    shape = (34, 34, 34)
    ref = voxelize_stl_reference(path, shape, device="cpu", check_watertight=False)
    acc = voxelize_stl(path, shape, device="cpu", check_watertight=False, accelerate=True)
    _assert_bitwise(ref, acc)


def test_accelerate_default_off_is_bit_unchanged(tmp_path: Path) -> None:
    # The default (accelerate=False) must be bit-identical both to the
    # explicit flag and to the historical reference path.
    path = _icosphere(tmp_path, 3, (17.3, 16.4, 16.8), 9.5)
    shape = (30, 30, 30)
    s0 = voxelize_stl(path, shape, device="cpu", check_watertight=False)
    s1 = voxelize_stl(path, shape, device="cpu", check_watertight=False, accelerate=False)
    s2 = voxelize_stl_reference(path, shape, device="cpu", check_watertight=False)
    _assert_bitwise(s0, s1)
    _assert_bitwise(s0, s2)


def test_accelerated_open_mesh_still_warns(tmp_path: Path) -> None:
    verts, faces = make_icosphere_stl((17.0, 17.0, 17.0), 12.0, subdivisions=2)
    path = tmp_path / "open.stl"
    write_stl(path, verts, faces[:-2], binary=True)  # drop two facets
    with pytest.warns(UserWarning, match="not closed"):
        voxel_accel.voxelize_stl_accelerated(path, (34, 34, 34), device="cpu")


# ---------------------------------------------------------------------------
# 4. BIN STRUCTURE SANITY
# ---------------------------------------------------------------------------


def test_triangle_bins_csr_invariants(tmp_path: Path) -> None:
    tri = read_stl_triangles(_icosphere(tmp_path, 3, (17.0, 17.0, 17.0), 12.0))
    shape = (34, 34, 34)
    n_tri = tri.shape[0]
    for bins in (
        voxel_accel.build_column_bins(tri, shape),
        voxel_accel.build_cell_bins(tri, shape),
    ):
        assert int(bins.offsets[0]) == 0 and int(bins.offsets[-1]) == bins.nnz
        assert bool((bins.lengths() >= 0).all())
        assert bins.entries.dtype == torch.int64
        assert int(bins.entries.max()) < n_tri and int(bins.entries.min()) >= 0
        assert bins.nnz >= n_tri  # every AABB lands in >= 1 bin
        assert bins.memory_bytes > 0


# ---------------------------------------------------------------------------
# 5. GPU: BINNED TRITON KERNELS vs BRUTE-FORCE TRITON KERNELS
# ---------------------------------------------------------------------------


def test_gpu_binned_triton_matches_brute_triton(tmp_path: Path, gpu_device) -> None:
    path = _icosphere(tmp_path, 4, (17.3, 16.4, 16.8), 9.5)
    tri = read_stl_triangles(path, device=gpu_device)
    shape = (34, 34, 34)
    s1, b1, q1 = voxelize_stl(tri, shape, device=gpu_device, check_watertight=False)
    s2, b2, q2 = voxel_accel.voxelize_stl_accelerated(
        tri, shape, device=gpu_device, check_watertight=False, use_triton=True
    )
    torch.cuda.synchronize()
    assert torch.equal(s1, s2)
    assert torch.equal(b1, b2)
    # q within fp32 FMA-contraction dust; the resolved/unresolved pattern
    # (q == 0.5 sentinel) must not flip anywhere.
    assert torch.equal(q1[b1] == 0.5, q2[b2] == 0.5)
    assert float((q1[b1] - q2[b2]).abs().max()) <= _Q_TOL


def test_gpu_torch_path_matches_gpu_reference_bitwise(tmp_path: Path, gpu_device) -> None:
    # Default (use_triton=False): fp64 binned torch vs fp64 brute torch,
    # both on the GPU — same-device arithmetic is verbatim-identical.
    path = _icosphere(tmp_path, 4, (17.3, 16.4, 16.8), 9.5)
    tri = read_stl_triangles(path, device=gpu_device)
    shape = (34, 34, 34)
    ref = voxelize_stl(tri, shape, device=gpu_device, use_triton=False, check_watertight=False)
    acc = voxel_accel.voxelize_stl_accelerated(
        tri, shape, device=gpu_device, check_watertight=False, use_triton=False
    )
    torch.cuda.synchronize()
    _assert_bitwise(ref, acc)


def test_gpu_torch_q_close_to_cpu_reference(tmp_path: Path, gpu_device) -> None:
    # Cross-device fp64 sums may round differently (quantified: <= 2e-6);
    # masks stay bit-identical, q agrees to fp32 ulps.
    path = _icosphere(tmp_path, 4, (24.0, 20.0, 22.0), 13.0, scale=(1.0, 0.8, 1.7))
    shape = (30, 36, 44)
    kw = {"origin": (-3.0, 2.5, 0.0), "spacing": (0.5, 0.7, 1.3)}
    s0, b0, q0 = voxelize_stl_reference(path, shape, device="cpu", check_watertight=False, **kw)
    tri = read_stl_triangles(path, device=gpu_device)
    s1, b1, q1 = voxel_accel.voxelize_stl_accelerated(
        tri, shape, device=gpu_device, check_watertight=False, use_triton=False, **kw
    )
    torch.cuda.synchronize()
    assert torch.equal(s0, s1.cpu())
    assert torch.equal(b0, b1.cpu())
    assert torch.equal(q0 == 0.5, q1.cpu() == 0.5)
    assert float((q0 - q1.cpu()).abs().max()) <= _Q_TOL
