"""STL voxelisation: mesh -> solid mask + BFL q-field.

Validates ``tensorlbm.geometry_voxel`` against the analytic conventions
of the repository (compute_q_cylinder_d3q19 / compute_q_generic_3d):
tensor layout ``(nz, ny, nx)`` with streamwise x last,
``fluid_boundary_mask`` marking fluid nodes with a solid D3Q19 neighbour,
and q as the fractional link distance in ``(0, 1]`` with the 0.5
halfway-bounce-back fallback.

The GPU kernel path is exercised only where CUDA is present (the kernel
module imports lazily); everything else runs the pure-torch reference so
the suite is green on CPU-only CI.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorlbm.d3q19 import C as C19
from tensorlbm.geometry_voxel import (
    mesh_watertight_status,
    read_stl_triangles,
    voxelize_stl,
    voxelize_stl_reference,
)
from tensorlbm.stl_geometry import make_cylinder_stl, make_sphere_stl, write_stl

_SPHERE_SHAPE = (34, 34, 34)
_SPHERE_CENTER = (17.3, 16.4, 16.8)
_SPHERE_RADIUS = 9.5


def _write_sphere(
    tmp_path: Path,
    center=_SPHERE_CENTER,
    radius=_SPHERE_RADIUS,
    n_lat=64,
    n_lon=96,
    binary=True,
) -> Path:
    verts, faces = make_sphere_stl(center, radius, n_lat=n_lat, n_lon=n_lon)
    path = tmp_path / ("sphere_bin.stl" if binary else "sphere_ascii.stl")
    write_stl(path, verts, faces, binary=binary)
    return path


def _analytic_solid_sphere(shape, center, radius) -> torch.Tensor:
    nz, ny, nx = shape
    k, j, i = torch.meshgrid(torch.arange(nz), torch.arange(ny), torch.arange(nx), indexing="ij")
    x = i + 0.5
    y = j + 0.5
    z = k + 0.5
    cx, cy, cz = center
    return (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= radius**2


def _analytic_solid_cylinder_z(shape, center, radius, length) -> torch.Tensor:
    nz, ny, nx = shape
    k, j, i = torch.meshgrid(torch.arange(nz), torch.arange(ny), torch.arange(nx), indexing="ij")
    cx, cy, cz = center
    radial = (i + 0.5 - cx) ** 2 + (j + 0.5 - cy) ** 2
    axial = ((k + 0.5) - cz).abs() <= length / 2.0
    return (radial <= radius**2) & axial


def _shift(mask: torch.Tensor, sz: int, sy: int, sx: int) -> torch.Tensor:
    nz, ny, nx = mask.shape
    padded = torch.nn.functional.pad(mask[None, None], (1, 1, 1, 1, 1, 1))[0, 0]
    return padded[1 + sz : 1 + sz + nz, 1 + sy : 1 + sy + ny, 1 + sx : 1 + sx + nx]


def _boundary_cell_set(solid: torch.Tensor) -> torch.Tensor:
    fluid = ~solid
    neighbour = torch.zeros_like(solid)
    for sz, sy, sx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        neighbour |= _shift(solid, sz, sy, sx)
    return fluid & neighbour


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


# ---------------------------------------------------------------------------
# 1. Analytic geometry cross-checks (reference path)
# ---------------------------------------------------------------------------


def test_sphere_solid_mask_matches_analytic(tmp_path) -> None:
    path = _write_sphere(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # watertight mesh must not warn
        solid, boundary, q = voxelize_stl(path, _SPHERE_SHAPE, device="cpu")

    assert solid.shape == _SPHERE_SHAPE
    assert solid.dtype == torch.bool
    assert boundary.shape == (19, *_SPHERE_SHAPE)
    assert boundary.dtype == torch.bool
    assert q.shape == (19, *_SPHERE_SHAPE)
    assert q.dtype == torch.float32

    analytic = _analytic_solid_sphere(_SPHERE_SHAPE, _SPHERE_CENTER, _SPHERE_RADIUS)
    agreement = (solid == analytic).float().mean().item()
    assert agreement > 0.99, f"interior agreement {agreement:.4f}"

    set_a = _boundary_cell_set(analytic)
    set_b = _boundary_cell_set(solid)
    union = (set_a | set_b).sum().item()
    jaccard = (set_a & set_b).sum().item() / union
    assert jaccard > 0.95, f"boundary Jaccard {jaccard:.4f}"


def test_sphere_q_field_conventions(tmp_path) -> None:
    solid, boundary, q = voxelize_stl_reference(
        _write_sphere(tmp_path), _SPHERE_SHAPE, device="cpu"
    )
    n_links = int(boundary.sum().item())
    assert n_links > 0
    q_vals = q[boundary]
    assert (q_vals > 0.0).all()
    assert (q_vals <= 1.0).all()
    # Resolved links must deviate from the 0.5 fallback somewhere.
    assert ((q_vals - 0.5).abs() > 1e-3).float().mean().item() > 0.1

    # fluid_boundary_mask semantics: fluid node whose +c_d neighbour is solid.
    fluid = ~solid
    for d in (1, 3, 7, 11, 18):
        cx, cy, cz = (int(v) for v in C19[d].tolist())
        expected = fluid & _shift(solid, cz, cy, cx)
        assert torch.equal(boundary[d], expected)


def test_cylinder_solid_mask_matches_analytic(tmp_path) -> None:
    shape = (26, 24, 22)
    center = (10.6, 12.4, 13.3)
    radius, length = 6.0, 16.0
    verts, faces = make_cylinder_stl(center, radius, length, n_circ=96, axis="z")
    path = tmp_path / "cylinder.stl"
    write_stl(path, verts, faces)

    solid, boundary, q = voxelize_stl(path, shape, device="cpu")
    analytic = _analytic_solid_cylinder_z(shape, center, radius, length)

    agreement = (solid == analytic).float().mean().item()
    assert agreement > 0.99, f"interior agreement {agreement:.4f}"

    set_a = _boundary_cell_set(analytic)
    set_b = _boundary_cell_set(solid)
    jaccard = (set_a & set_b).sum().item() / (set_a | set_b).sum().item()
    assert jaccard > 0.95, f"boundary Jaccard {jaccard:.4f}"

    n_links = int(boundary.sum().item())
    assert n_links > 0
    assert (q[boundary] > 0.0).all() and (q[boundary] <= 1.0).all()


def test_translation_and_scale_invariance(tmp_path) -> None:
    shape = (26, 25, 24)
    center = (11.3, 12.1, 12.7)
    radius = 7.0
    verts, faces = make_sphere_stl(center, radius, n_lat=32, n_lon=56)
    base_path = tmp_path / "base.stl"
    write_stl(base_path, verts, faces)
    solid0, _, q0 = voxelize_stl(base_path, shape, device="cpu")

    # Pure translation: mesh and grid origin shift together.
    shift = (2.0, -3.0, 1.5)
    moved = verts + np.asarray(shift, dtype=np.float32)
    moved_path = tmp_path / "moved.stl"
    write_stl(moved_path, moved, faces)
    solid1, _, q1 = voxelize_stl(
        moved_path, shape, device="cpu", origin=shift, spacing=(1.0, 1.0, 1.0)
    )
    assert torch.equal(solid0, solid1)
    # fp32 mesh coordinates shift by ~1 ulp under the translation.
    assert torch.allclose(q0, q1, atol=1e-5)

    # Pure scaling by 2 (exact in fp32): mesh, origin and spacing scale.
    scaled_path = tmp_path / "scaled.stl"
    write_stl(scaled_path, verts * 2.0, faces)
    solid2, _, q2 = voxelize_stl(
        scaled_path, shape, device="cpu", origin=(0.0, 0.0, 0.0), spacing=(2.0, 2.0, 2.0)
    )
    assert torch.equal(solid0, solid2)
    assert torch.allclose(q0, q2, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. STL reader
# ---------------------------------------------------------------------------


def test_ascii_and_binary_stl_equivalence(tmp_path) -> None:
    bin_path = _write_sphere(tmp_path, n_lat=24, n_lon=40, binary=True)
    ascii_path = _write_sphere(tmp_path, n_lat=24, n_lon=40, binary=False)

    tri_bin = read_stl_triangles(bin_path)
    tri_ascii = read_stl_triangles(ascii_path)
    assert tri_bin.dtype == torch.float32
    assert tri_bin.ndim == 3 and tri_bin.shape[1:] == (3, 3)
    assert tri_ascii.shape == tri_bin.shape
    # ASCII writer uses %.6f — round-trip agrees to rounding granularity.
    assert torch.allclose(tri_ascii, tri_bin, atol=5e-6, rtol=0.0)

    solid_b, bnd_b, q_b = voxelize_stl(bin_path, (26, 26, 26), device="cpu")
    solid_a, bnd_a, q_a = voxelize_stl(ascii_path, (26, 26, 26), device="cpu")
    assert torch.equal(solid_b, solid_a)
    assert torch.equal(bnd_b, bnd_a)
    assert torch.allclose(q_b, q_a, atol=1e-4)


def test_malformed_inputs(tmp_path) -> None:
    empty = tmp_path / "empty.stl"
    empty.write_text("solid empty\nendsolid empty\n")
    with pytest.raises(ValueError, match="No triangles"):
        read_stl_triangles(empty)

    garbage = tmp_path / "garbage.stl"
    garbage.write_bytes(b"not an stl file at all" * 11)
    with pytest.raises(ValueError, match="No triangles"):
        read_stl_triangles(garbage)

    with pytest.raises(FileNotFoundError):
        read_stl_triangles(tmp_path / "missing.stl")

    with pytest.raises(ValueError, match="positive"):
        voxelize_stl(np.zeros((1, 3, 3), dtype=np.float32), (0, 4, 4), device="cpu")


# ---------------------------------------------------------------------------
# 3. Open-mesh behaviour
# ---------------------------------------------------------------------------


def test_open_mesh_warns_and_still_returns(tmp_path) -> None:
    verts, faces = make_sphere_stl((10.3, 10.6, 10.9), 6.0, n_lat=20, n_lon=32)
    status_closed = mesh_watertight_status(torch.from_numpy(verts[faces]))
    assert status_closed["watertight"]

    open_path = tmp_path / "open.stl"
    write_stl(open_path, verts, faces[:-8])
    status_open = mesh_watertight_status(torch.from_numpy(verts[faces[:-8]]))
    assert not status_open["watertight"]
    assert status_open["boundary_edges"] > 0

    with pytest.warns(UserWarning, match="not closed"):
        solid, boundary, q = voxelize_stl(open_path, (22, 22, 22), device="cpu")
    # Documented assumption: parity still produces full-shape tensors.
    assert solid.shape == (22, 22, 22)
    assert boundary.shape == (19, 22, 22, 22)


# ---------------------------------------------------------------------------
# 4. GPU kernel path vs CPU reference
# ---------------------------------------------------------------------------


def test_gpu_kernels_match_cpu_reference(tmp_path, gpu_device) -> None:
    shape = (30, 30, 30)
    center = (15.4, 16.2, 17.3)
    verts, faces = make_sphere_stl(center, 9.0, n_lat=36, n_lon=60)
    path = tmp_path / "gpu_sphere.stl"
    write_stl(path, verts, faces)

    solid_c, bnd_c, q_c = voxelize_stl_reference(path, shape, device="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solid_g, bnd_g, q_g = voxelize_stl(path, shape, device=gpu_device)

    # Solid mask: parity must agree up to fp32 rounding of cell centres.
    mismatches = (solid_g.cpu() != solid_c).sum().item()
    assert mismatches <= max(2, int(1e-3 * solid_c.numel())), f"{mismatches} mask mismatches"

    # Boundary scaffold identical by construction.
    assert torch.equal(bnd_g.cpu(), bnd_c)

    # q field: same conventions within fp32 tolerance; a link may flip
    # between resolved and fallback only within the same tolerance band.
    diff = (q_g.cpu() - q_c).abs()
    bad = (diff > 1e-3) & bnd_c
    assert int(bad.sum().item()) <= int(2e-3 * bnd_c.sum().item())
    tol_mask = bnd_c & ~bad
    if bool(tol_mask.any()):
        assert float(diff[tol_mask].max().item()) <= 2e-5


def test_gpu_sphere_matches_analytic(tmp_path, gpu_device) -> None:
    shape = (30, 30, 30)
    center = (15.4, 16.2, 17.3)
    verts, faces = make_sphere_stl(center, 9.0, n_lat=36, n_lon=60)
    path = tmp_path / "gpu_analytic.stl"
    write_stl(path, verts, faces)

    solid, boundary, q = voxelize_stl(path, shape, device=gpu_device)
    analytic = _analytic_solid_sphere(shape, center, 9.0)
    agreement = (solid.cpu() == analytic).float().mean().item()
    assert agreement > 0.99, f"interior agreement {agreement:.4f}"

    set_a = _boundary_cell_set(analytic)
    set_b = _boundary_cell_set(solid.cpu())
    jaccard = (set_a & set_b).sum().item() / (set_a | set_b).sum().item()
    assert jaccard > 0.95, f"boundary Jaccard {jaccard:.4f}"

    q_vals = q[boundary].cpu()
    assert (q_vals > 0.0).all() and (q_vals <= 1.0).all()
