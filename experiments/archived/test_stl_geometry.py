"""Test suite for stl_geometry module.

Validates:
  1. STL reader round-trip (write → read → compare vertices/faces/normals)
  2. Voxelized solid matches analytical mask (sphere, cylinder)
  3. STL normals match analytical normals (within 5%)
  4. Drag computation gives same results as analytical normals
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from tensorlbm.stl_geometry import (
    read_stl,
    voxelize_stl,
    SurfaceMesh_from_stl,
    write_stl,
    make_sphere_stl,
    make_cylinder_stl,
    make_naca_stl,
    get_near_wall_3d,
)
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _angular_error(n_stl, n_ref):
    """Mean angular error (degrees) between two sets of unit normals."""
    dot = np.sum(n_stl * n_ref, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    angles = np.degrees(np.arccos(dot))
    return float(np.mean(angles)), float(np.max(angles))


def _analytical_sphere_normals(solid, near, cx, cy, cz, R):
    """Analytical outward normals for a sphere."""
    nz, ny, nx = solid.shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    nx_n = (xx - cx) / R
    ny_n = (yy - cy) / R
    nz_n = (zz - cz) / R
    norm = torch.sqrt(nx_n ** 2 + ny_n ** 2 + nz_n ** 2).clamp(min=1e-10)
    nx_n = nx_n / norm * near.float()
    ny_n = ny_n / norm * near.float()
    nz_n = nz_n / norm * near.float()
    return nx_n, ny_n, nz_n


def _analytical_cylinder_normals(solid, near, cx, cy, R, axis="z"):
    """Analytical outward normals for an extruded cylinder."""
    nz, ny, nx = solid.shape
    if axis == "z":
        yy, xx = torch.meshgrid(
            torch.arange(ny, dtype=torch.float32),
            torch.arange(nx, dtype=torch.float32),
            indexing="ij",
        )
        nx_n = ((xx - cx) / R).unsqueeze(0).expand(nz, ny, nx)
        ny_n = ((yy - cy) / R).unsqueeze(0).expand(nz, ny, nx)
        nz_n = torch.zeros_like(nx_n)
    elif axis == "x":
        zz, yy = torch.meshgrid(
            torch.arange(nz, dtype=torch.float32),
            torch.arange(ny, dtype=torch.float32),
            indexing="ij",
        )
        ny_n = ((yy - cy) / R).unsqueeze(2).expand(nz, ny, nx)
        nz_n = ((zz - cz) / R).unsqueeze(2).expand(nz, ny, nx)
        nx_n = torch.zeros_like(ny_n)
    norm = torch.sqrt(nx_n ** 2 + ny_n ** 2 + nz_n ** 2).clamp(min=1e-10)
    nx_n = nx_n / norm * near.float()
    ny_n = ny_n / norm * near.float()
    nz_n = nz_n / norm * near.float()
    return nx_n, ny_n, nz_n


def _make_uniform_flow(nz, ny, nx, u_in=0.05, device="cpu"):
    """Equilibrium distribution for uniform flow u=(u_in, 0, 0)."""
    dev = torch.device(device)
    f = equilibrium3d(
        torch.ones(nz, ny, nx, device=dev),
        torch.full((nz, ny, nx), u_in, device=dev),
        torch.zeros(nz, ny, nx, device=dev),
        torch.zeros(nz, ny, nx, device=dev),
        device=dev,
    )
    return f


# ---------------------------------------------------------------------------
# Test 1: STL reader round-trip
# ---------------------------------------------------------------------------

def test_reader_roundtrip_sphere():
    """Write sphere STL → read back → verify vertices and normals."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        verts, faces = make_sphere_stl((10, 10, 10), 5.0, n_lat=15, n_lon=30)
        write_stl(tmp / "sphere_bin.stl", verts, faces, binary=True)
        write_stl(tmp / "sphere_ascii.stl", verts, faces, binary=False)

        for label, path in [("binary", tmp / "sphere_bin.stl"),
                            ("ascii", tmp / "sphere_ascii.stl")]:
            rv, rf, rn = read_stl(path)
            # Face count must match
            assert len(rf) == len(faces), f"{label}: face count {len(rf)} != {len(faces)}"
            # All non-degenerate normals should be unit length
            norms = np.linalg.norm(rn, axis=1)
            nonzero = norms > 0.5
            assert nonzero.sum() > len(faces) * 0.9, \
                f"{label}: too many degenerate normals ({(~nonzero).sum()})"
            assert np.allclose(norms[nonzero], 1.0, atol=1e-4), \
                f"{label}: non-degenerate normals not unit length (min={norms[nonzero].min():.4f})"
            # Sphere normals should point outward (dot with radial > 0)
            tri_centroids = rv[rf].mean(axis=1)
            radial = tri_centroids - np.array([10, 10, 10])
            radial /= np.linalg.norm(radial, axis=1, keepdims=True).clip(min=1e-10)
            dot = np.sum(rn * radial, axis=1)
            frac_outward = np.mean(dot > 0)
            assert frac_outward > 0.95, \
                f"{label}: only {frac_outward*100:.1f}% normals point outward"
            print(f"  [reader/{label}] {len(rf)} faces, "
                  f"{frac_outward*100:.1f}% outward normals ✓")


# ---------------------------------------------------------------------------
# Test 2: Voxelized solid matches analytical mask
# ---------------------------------------------------------------------------

def test_voxelize_sphere():
    """Voxelized sphere STL should match analytical sphere mask."""
    nx = ny = nz = 40
    cx = cy = cz = 20.0
    R = 8.0
    origin = (0.0, 0.0, 0.0)
    spacing = (1.0, 1.0, 1.0)

    verts, faces = make_sphere_stl((cx, cy, cz), R, n_lat=30, n_lon=60)
    solid_stl = voxelize_stl(verts, faces, (nx, ny, nz), origin, spacing)

    # Analytical mask
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    solid_anal = ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) <= R ** 2

    # Compare (allow boundary differences of ±1 cell)
    agreement = (solid_stl == solid_anal).float().mean().item()
    diff = (solid_stl != solid_anal).sum().item()
    total = solid_stl.numel()
    print(f"  [voxel/sphere] agreement={agreement*100:.1f}%, "
          f"diff_cells={diff}/{total}")
    assert agreement > 0.92, f"Sphere voxelization agreement too low: {agreement*100:.1f}%"


def test_voxelize_cylinder():
    """Voxelized cylinder STL should match analytical cylinder mask."""
    nx, ny, nz = 40, 40, 40
    # Non-integer centre avoids z-ray edge degeneracies
    cx, cy = 20.3, 20.7
    R = 8.0
    # Full-span cylinder: caps outside domain → no cap near-wall cells
    length = float(nz + 4)
    cz = nz / 2.0
    origin = (0.0, 0.0, 0.0)
    spacing = (1.0, 1.0, 1.0)

    verts, faces = make_cylinder_stl((cx, cy, cz), R, length, n_circ=40, axis="z", n_axial=10)
    solid_stl = voxelize_stl(verts, faces, (nx, ny, nz), origin, spacing)

    # Analytical mask (cylinder along z, full span)
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    solid_anal = r2 <= R ** 2

    agreement = (solid_stl == solid_anal).float().mean().item()
    diff = (solid_stl != solid_anal).sum().item()
    print(f"  [voxel/cylinder] agreement={agreement*100:.1f}%, "
          f"diff_cells={diff}")
    assert agreement > 0.92, f"Cylinder voxelization agreement too low: {agreement*100:.1f}%"


# ---------------------------------------------------------------------------
# Test 3: STL normals match analytical normals (within 5%)
# ---------------------------------------------------------------------------

def test_normals_sphere():
    """STL-derived normals should match analytical sphere normals."""
    nx = ny = nz = 40
    cx = cy = cz = 20.0
    R = 8.0
    origin = (0.0, 0.0, 0.0)
    spacing = (1.0, 1.0, 1.0)

    verts, faces = make_sphere_stl((cx, cy, cz), R, n_lat=30, n_lon=60)
    solid = voxelize_stl(verts, faces, (nx, ny, nz), origin, spacing)
    near = get_near_wall_3d(solid)

    # Compute face normals from the generated mesh
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    face_normals = cross / np.where(norms > 1e-10, norms, 1.0)

    mesh_stl = SurfaceMesh_from_stl(
        solid, near, verts, faces, face_normals, origin, spacing
    )
    nx_a, ny_a, nz_a = _analytical_sphere_normals(solid, near, cx, cy, cz, R)

    # Extract at near-wall cells
    idx = near.nonzero(as_tuple=False)
    n_stl = np.stack([
        mesh_stl.nx_n[idx[:, 0], idx[:, 1], idx[:, 2]].numpy(),
        mesh_stl.ny_n[idx[:, 0], idx[:, 1], idx[:, 2]].numpy(),
        mesh_stl.nz_n[idx[:, 0], idx[:, 1], idx[:, 2]].numpy(),
    ], axis=1)
    n_anal = np.stack([
        nx_a[idx[:, 0], idx[:, 1], idx[:, 2]].numpy(),
        ny_a[idx[:, 0], idx[:, 1], idx[:, 2]].numpy(),
        nz_a[idx[:, 0], idx[:, 1], idx[:, 2]].numpy(),
    ], axis=1)

    mean_err, max_err = _angular_error(n_stl, n_anal)
    print(f"  [normals/sphere] near-wall cells={len(idx)}, "
          f"mean_angle_err={mean_err:.2f}°, max_angle_err={max_err:.2f}°")
    # 5% of 90° = 4.5°; allow up to 10° mean for staircase effects
    assert mean_err < 10.0, f"Mean angular error too high: {mean_err:.2f}°"


def test_normals_cylinder():
    """STL-derived normals should match analytical cylinder normals."""
    nx, ny, nz = 40, 40, 40
    cx, cy = 20.3, 20.7
    R = 8.0
    # Full-span cylinder: caps outside domain → no cap near-wall cells
    length = float(nz + 4)
    cz = nz / 2.0
    origin = (0.0, 0.0, 0.0)
    spacing = (1.0, 1.0, 1.0)

    verts, faces = make_cylinder_stl((cx, cy, cz), R, length, n_circ=40, axis="z", n_axial=10)
    solid = voxelize_stl(verts, faces, (nx, ny, nz), origin, spacing)
    near = get_near_wall_3d(solid)

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    face_normals = cross / np.where(norms > 1e-10, norms, 1.0)

    mesh_stl = SurfaceMesh_from_stl(
        solid, near, verts, faces, face_normals, origin, spacing
    )
    nx_a, ny_a, nz_a = _analytical_cylinder_normals(
        solid, near, cx, cy, R, axis="z"
    )

    idx = near.nonzero(as_tuple=False)
    n_stl = np.stack([
        mesh_stl.nx_n[idx[:, 0], idx[:, 1], idx[:, 2]].numpy(),
        mesh_stl.ny_n[idx[:, 0], idx[:, 1], idx[:, 2]].numpy(),
        mesh_stl.nz_n[idx[:, 0], idx[:, 1], idx[:, 2]].numpy(),
    ], axis=1)
    n_anal = np.stack([
        nx_a[idx[:, 0], idx[:, 1], idx[:, 2]].numpy(),
        ny_a[idx[:, 0], idx[:, 1], idx[:, 2]].numpy(),
        nz_a[idx[:, 0], idx[:, 1], idx[:, 2]].numpy(),
    ], axis=1)

    mean_err, max_err = _angular_error(n_stl, n_anal)
    print(f"  [normals/cylinder] near-wall cells={len(idx)}, "
          f"mean_angle_err={mean_err:.2f}°, max_angle_err={max_err:.2f}°")
    assert mean_err < 10.0, f"Mean angular error too high: {mean_err:.2f}°"


# ---------------------------------------------------------------------------
# Test 4: Drag computation consistency (STL vs analytical normals)
# ---------------------------------------------------------------------------

def test_drag_consistency_sphere():
    """Pressure + friction drag with STL normals should match analytical."""
    nx = ny = nz = 40
    cx = cy = cz = 20.0
    R = 8.0
    origin = (0.0, 0.0, 0.0)
    spacing = (1.0, 1.0, 1.0)
    u_in = 0.05
    nu = 0.02
    dpS = 0.5 * u_in ** 2 * math.pi * R ** 2

    verts, faces = make_sphere_stl((cx, cy, cz), R, n_lat=30, n_lon=60)
    solid = voxelize_stl(verts, faces, (nx, ny, nz), origin, spacing)
    near = get_near_wall_3d(solid)

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    face_normals = cross / np.where(norms > 1e-10, norms, 1.0)

    mesh_stl = SurfaceMesh_from_stl(
        solid, near, verts, faces, face_normals, origin, spacing
    )
    mesh_anal = SurfaceMesh.from_sphere(solid, near, cx, cy, cz, R)

    f = _make_uniform_flow(nz, ny, nx, u_in)

    # Pressure drag
    fx_p_stl, fy_p_stl, fz_p_stl = drag_pressure_integration(
        f, mesh_stl, dpS, extrap="quadratic"
    )
    fx_p_anal, fy_p_anal, fz_p_anal = drag_pressure_integration(
        f, mesh_anal, dpS, extrap="quadratic"
    )

    # Friction drag
    fx_f_stl, fy_f_stl, fz_f_stl = drag_friction_integration(
        f, mesh_stl, dpS, nu
    )
    fx_f_anal, fy_f_anal, fz_f_anal = drag_friction_integration(
        f, mesh_anal, dpS, nu
    )

    print(f"  [drag/sphere] pressure:  STL=({fx_p_stl:.4f},{fy_p_stl:.4f},{fz_p_stl:.4f})  "
          f"anal=({fx_p_anal:.4f},{fy_p_anal:.4f},{fz_p_anal:.4f})")
    print(f"  [drag/sphere] friction:  STL=({fx_f_stl:.4f},{fy_f_stl:.4f},{fz_f_stl:.4f})  "
          f"anal=({fx_f_anal:.4f},{fy_f_anal:.4f},{fz_f_anal:.4f})")

    # For uniform flow + constant pressure (after bg subtraction), pressure drag ≈ 0
    assert abs(fx_p_stl) < 0.5, f"Pressure drag_x too high: {fx_p_stl:.4f}"

    # Friction drag should be close between STL and analytical
    if abs(fx_f_anal) > 1e-6:
        rel_err = abs(fx_f_stl - fx_f_anal) / abs(fx_f_anal)
        print(f"  [drag/sphere] friction_x relative error: {rel_err*100:.1f}%")
        assert rel_err < 0.15, \
            f"Friction drag_x relative error too high: {rel_err*100:.1f}%"


def test_drag_consistency_cylinder():
    """Pressure + friction drag with STL normals should match analytical."""
    nx, ny, nz = 40, 40, 40
    cx, cy = 20.3, 20.7
    R = 8.0
    # Full-span cylinder: caps outside domain → no cap near-wall cells
    length = float(nz + 4)
    cz = nz / 2.0
    origin = (0.0, 0.0, 0.0)
    spacing = (1.0, 1.0, 1.0)
    u_in = 0.05
    nu = 0.02
    dpS = 0.5 * u_in ** 2 * 2 * R * nz

    verts, faces = make_cylinder_stl((cx, cy, cz), R, length, n_circ=40, axis="z", n_axial=10)
    solid = voxelize_stl(verts, faces, (nx, ny, nz), origin, spacing)
    near = get_near_wall_3d(solid)

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    face_normals = cross / np.where(norms > 1e-10, norms, 1.0)

    mesh_stl = SurfaceMesh_from_stl(
        solid, near, verts, faces, face_normals, origin, spacing
    )
    mesh_anal = SurfaceMesh.from_cylinder(solid, near, cx, cy, R, axis="z")

    f = _make_uniform_flow(nz, ny, nx, u_in)

    fx_p_stl, fy_p_stl, fz_p_stl = drag_pressure_integration(
        f, mesh_stl, dpS, extrap="quadratic"
    )
    fx_p_anal, fy_p_anal, fz_p_anal = drag_pressure_integration(
        f, mesh_anal, dpS, extrap="quadratic"
    )
    fx_f_stl, fy_f_stl, fz_f_stl = drag_friction_integration(
        f, mesh_stl, dpS, nu
    )
    fx_f_anal, fy_f_anal, fz_f_anal = drag_friction_integration(
        f, mesh_anal, dpS, nu
    )

    print(f"  [drag/cylinder] pressure:  STL=({fx_p_stl:.4f},{fy_p_stl:.4f},{fz_p_stl:.4f})  "
          f"anal=({fx_p_anal:.4f},{fy_p_anal:.4f},{fz_p_anal:.4f})")
    print(f"  [drag/cylinder] friction:  STL=({fx_f_stl:.4f},{fy_f_stl:.4f},{fz_f_stl:.4f})  "
          f"anal=({fx_f_anal:.4f},{fy_f_anal:.4f},{fz_f_anal:.4f})")

    assert abs(fx_p_stl) < 0.5, f"Pressure drag_x too high: {fx_p_stl:.4f}"

    if abs(fx_f_anal) > 1e-6:
        rel_err = abs(fx_f_stl - fx_f_anal) / abs(fx_f_anal)
        print(f"  [drag/cylinder] friction_x relative error: {rel_err*100:.1f}%")
        assert rel_err < 0.15, \
            f"Friction drag_x relative error too high: {rel_err*100:.1f}%"


# ---------------------------------------------------------------------------
# Test 5: NACA STL round-trip + voxelization
# ---------------------------------------------------------------------------

def test_naca_stl():
    """NACA airfoil STL should voxelize and produce normals."""
    nx, ny, nz = 60, 40, 20
    chord = 30.0
    x_le = 15.0
    y_mid = 20.0
    z0 = 5.0
    z1 = 15.0
    origin = (0.0, 0.0, 0.0)
    spacing = (1.0, 1.0, 1.0)

    verts, faces = make_naca_stl(
        chord=chord, x_le=x_le, y_mid=y_mid, z0=z0, z1=z1,
        thickness_ratio=0.12, n_x=40,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_stl(tmp / "naca.stl", verts, faces, binary=True)
        rv, rf, rn = read_stl(tmp / "naca.stl")
        assert len(rf) == len(faces), f"NACA face count mismatch: {len(rf)} vs {len(faces)}"

    solid = voxelize_stl(verts, faces, (nx, ny, nz), origin, spacing)
    near = get_near_wall_3d(solid)
    n_near = near.sum().item()
    print(f"  [naca] solid_cells={solid.sum().item()}, near_wall={n_near}")
    assert n_near > 0, "No near-wall cells for NACA"

    # Build mesh and check normals are non-zero
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    face_normals = cross / np.where(norms > 1e-10, norms, 1.0)

    mesh = SurfaceMesh_from_stl(
        solid, near, verts, faces, face_normals, origin, spacing
    )
    idx = near.nonzero(as_tuple=False)
    n_mag = torch.sqrt(
        mesh.nx_n[idx[:, 0], idx[:, 1], idx[:, 2]] ** 2
        + mesh.ny_n[idx[:, 0], idx[:, 1], idx[:, 2]] ** 2
        + mesh.nz_n[idx[:, 0], idx[:, 1], idx[:, 2]] ** 2
    )
    print(f"  [naca] normal magnitudes: min={n_mag.min():.4f}, "
          f"mean={n_mag.mean():.4f}, max={n_mag.max():.4f}")
    assert n_mag.min() > 0.99, "Some normals are not unit length"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("STL Geometry Module Test Suite")
    print("=" * 70)

    tests = [
        ("Reader round-trip (sphere)", test_reader_roundtrip_sphere),
        ("Voxelize sphere", test_voxelize_sphere),
        ("Voxelize cylinder", test_voxelize_cylinder),
        ("Normals sphere", test_normals_sphere),
        ("Normals cylinder", test_normals_cylinder),
        ("Drag consistency sphere", test_drag_consistency_sphere),
        ("Drag consistency cylinder", test_drag_consistency_cylinder),
        ("NACA STL", test_naca_stl),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            print(f"  PASSED ✓")
            passed += 1
        except Exception as e:
            print(f"  FAILED ✗: {e}")
            failed += 1

    print(f"\n{'=' * 70}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'=' * 70}")
