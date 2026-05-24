"""Tests for preprocess_geo.py and unit_converter.py."""
from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from tensorlbm.preprocess_geo import (
    compute_q_generic_3d,
    poly_to_mask_2d,
    random_porosity_mask_2d,
    random_porosity_mask_3d,
    voxelize_stl_3d,
)
from tensorlbm.unit_converter import LBMUnitConverter

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_binary_stl(path: Path, triangles: np.ndarray) -> None:
    """Write a minimal binary STL file for the given triangles (N, 3, 3)."""
    n_tri = triangles.shape[0]
    with open(path, "wb") as fh:
        fh.write(b"\x00" * 80)  # header
        fh.write(struct.pack("<I", n_tri))
        for tri in triangles:
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0]).astype(np.float32)
            fh.write(normal.tobytes())
            for v in tri:
                fh.write(v.astype(np.float32).tobytes())
            fh.write(b"\x00\x00")  # attribute


def _unit_cube_triangles() -> np.ndarray:
    """Return the 12 triangles forming a unit cube [0,1]^3."""
    # 6 faces × 2 triangles
    faces = [
        # -z face (z=0)
        [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
        [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
        # +z face (z=1)
        [[0, 0, 1], [1, 1, 1], [1, 0, 1]],
        [[0, 0, 1], [0, 1, 1], [1, 1, 1]],
        # -y face (y=0)
        [[0, 0, 0], [0, 0, 1], [1, 0, 1]],
        [[0, 0, 0], [1, 0, 1], [1, 0, 0]],
        # +y face (y=1)
        [[0, 1, 0], [1, 1, 1], [0, 1, 1]],
        [[0, 1, 0], [1, 1, 0], [1, 1, 1]],
        # -x face (x=0)
        [[0, 0, 0], [0, 1, 0], [0, 1, 1]],
        [[0, 0, 0], [0, 1, 1], [0, 0, 1]],
        # +x face (x=1)
        [[1, 0, 0], [1, 1, 1], [1, 1, 0]],
        [[1, 0, 0], [1, 0, 1], [1, 1, 1]],
    ]
    return np.array(faces, dtype=np.float32)


# ---------------------------------------------------------------------------
# poly_to_mask_2d
# ---------------------------------------------------------------------------


class TestPolyToMask2D:
    def test_square_inside(self) -> None:
        """Cells inside a square polygon should be solid."""
        # 4×4 grid; square [1, 3]×[1, 3] in lattice coords
        verts = [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)]
        mask = poly_to_mask_2d(verts, ny=4, nx=4, device=torch.device("cpu"))
        assert mask.dtype == torch.bool
        assert mask.shape == (4, 4)
        # cell centres at (0.5,0.5),(1.5,0.5),...
        # cells (ix=1,iy=1), (ix=2,iy=1), (ix=1,iy=2), (ix=2,iy=2) should be inside
        assert mask[1, 1].item()
        assert mask[1, 2].item()
        assert mask[2, 1].item()
        assert mask[2, 2].item()

    def test_outside_cells_not_solid(self) -> None:
        """Corner cells should be outside a square that doesn't cover them."""
        verts = [(1.5, 1.5), (2.5, 1.5), (2.5, 2.5), (1.5, 2.5)]
        mask = poly_to_mask_2d(verts, ny=4, nx=4, device=torch.device("cpu"))
        assert not mask[0, 0].item()
        assert not mask[3, 3].item()

    def test_full_domain_polygon(self) -> None:
        """A polygon covering the entire grid should mark all cells solid."""
        ny, nx = 5, 6
        # Slightly larger than the full grid
        verts = [(-0.1, -0.1), (nx + 0.1, -0.1), (nx + 0.1, ny + 0.1), (-0.1, ny + 0.1)]
        mask = poly_to_mask_2d(verts, ny=ny, nx=nx, device=torch.device("cpu"))
        assert mask.all()

    def test_output_dtype_bool(self) -> None:
        verts = [(0.5, 0.5), (3.5, 0.5), (3.5, 3.5), (0.5, 3.5)]
        mask = poly_to_mask_2d(verts, ny=5, nx=5, device=torch.device("cpu"))
        assert mask.dtype == torch.bool

    def test_triangle_polygon(self) -> None:
        """A triangle polygon produces a solid mask with at least one True cell."""
        verts = [(2.0, 0.5), (0.5, 3.5), (3.5, 3.5)]
        mask = poly_to_mask_2d(verts, ny=5, nx=5, device=torch.device("cpu"))
        assert mask.any()


# ---------------------------------------------------------------------------
# voxelize_stl_3d
# ---------------------------------------------------------------------------


class TestVoxelizeSTL3D:
    def test_unit_cube_has_solid_cells(self, tmp_path: Path) -> None:
        """A unit-cube STL should produce solid cells in the interior."""
        stl_path = tmp_path / "cube.stl"
        _write_binary_stl(stl_path, _unit_cube_triangles())
        mask = voxelize_stl_3d(stl_path, nx=10, ny=10, nz=10, device=torch.device("cpu"))
        assert mask.shape == (10, 10, 10)
        assert mask.dtype == torch.bool
        assert mask.any(), "No solid cells found for unit-cube STL"

    def test_unit_cube_solid_fraction(self, tmp_path: Path) -> None:
        """Solid fraction should be roughly in (0, 1) for a cube inside the domain."""
        stl_path = tmp_path / "cube.stl"
        _write_binary_stl(stl_path, _unit_cube_triangles())
        mask = voxelize_stl_3d(stl_path, nx=12, ny=12, nz=12, device=torch.device("cpu"))
        frac = float(mask.float().mean().item())
        # With 5 % padding, cube occupies (1/1.1)^3 ≈ 75 % of the grid
        assert 0.1 < frac < 1.0

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            voxelize_stl_3d("/nonexistent/file.stl", nx=4, ny=4, nz=4, device=torch.device("cpu"))

    def test_output_shape(self, tmp_path: Path) -> None:
        stl_path = tmp_path / "cube.stl"
        _write_binary_stl(stl_path, _unit_cube_triangles())
        for nx, ny, nz in [(8, 8, 8), (12, 6, 4)]:
            mask = voxelize_stl_3d(stl_path, nx=nx, ny=ny, nz=nz, device=torch.device("cpu"))
            assert mask.shape == (nz, ny, nx)

    def test_ascii_stl(self, tmp_path: Path) -> None:
        """ASCII STL files should also be parsed and voxelised correctly."""
        tris = _unit_cube_triangles()
        lines = ["solid cube"]
        for tri in tris:
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            lines.append(f"  facet normal {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}")
            lines.append("    outer loop")
            for v in tri:
                lines.append(f"      vertex {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}")
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append("endsolid cube")

        stl_path = tmp_path / "cube_ascii.stl"
        stl_path.write_text("\n".join(lines))
        mask = voxelize_stl_3d(stl_path, nx=8, ny=8, nz=8, device=torch.device("cpu"))
        assert mask.any()


# ---------------------------------------------------------------------------
# random_porosity_mask_2d
# ---------------------------------------------------------------------------


class TestRandomPorosityMask2D:
    def test_output_shape(self) -> None:
        mask = random_porosity_mask_2d(ny=16, nx=20, porosity=0.5, device=torch.device("cpu"))
        assert mask.shape == (16, 20)
        assert mask.dtype == torch.bool

    def test_porosity_approx(self) -> None:
        """Fluid fraction should be close to the requested porosity for large grids."""
        porosity = 0.6
        mask = random_porosity_mask_2d(
            ny=200, nx=200, porosity=porosity, device=torch.device("cpu")
        )
        fluid_frac = float((~mask).float().mean().item())
        assert abs(fluid_frac - porosity) < 0.05, f"fluid_frac={fluid_frac:.3f}"

    def test_solid_fraction_approx(self) -> None:
        """Solid fraction should be close to 1 - porosity."""
        porosity = 0.3
        mask = random_porosity_mask_2d(
            ny=200, nx=200, porosity=porosity, device=torch.device("cpu")
        )
        solid_frac = float(mask.float().mean().item())
        assert abs(solid_frac - (1.0 - porosity)) < 0.05

    def test_reproducibility(self) -> None:
        """Same seed should produce identical masks."""
        dev = torch.device("cpu")
        m1 = random_porosity_mask_2d(ny=20, nx=20, porosity=0.5, device=dev, seed=42)
        m2 = random_porosity_mask_2d(ny=20, nx=20, porosity=0.5, device=dev, seed=42)
        assert (m1 == m2).all()

    def test_different_seeds_differ(self) -> None:
        dev = torch.device("cpu")
        m1 = random_porosity_mask_2d(ny=20, nx=20, porosity=0.5, device=dev, seed=1)
        m2 = random_porosity_mask_2d(ny=20, nx=20, porosity=0.5, device=dev, seed=2)
        assert not (m1 == m2).all()

    def test_with_smoothing(self) -> None:
        """Gaussian smoothing (sigma > 0) should produce a valid solid mask."""
        mask = random_porosity_mask_2d(
            ny=30, nx=30, porosity=0.5, device=torch.device("cpu"), sigma=2.0
        )
        assert mask.shape == (30, 30)
        assert mask.any()
        assert not mask.all()

    def test_invalid_porosity(self) -> None:
        with pytest.raises(ValueError):
            random_porosity_mask_2d(ny=10, nx=10, porosity=0.0, device=torch.device("cpu"))
        with pytest.raises(ValueError):
            random_porosity_mask_2d(ny=10, nx=10, porosity=1.0, device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# random_porosity_mask_3d
# ---------------------------------------------------------------------------


class TestRandomPorosityMask3D:
    def test_output_shape(self) -> None:
        mask = random_porosity_mask_3d(
            nz=8, ny=10, nx=12, porosity=0.4, device=torch.device("cpu")
        )
        assert mask.shape == (8, 10, 12)
        assert mask.dtype == torch.bool

    def test_porosity_approx(self) -> None:
        porosity = 0.55
        mask = random_porosity_mask_3d(
            nz=50, ny=50, nx=50, porosity=porosity, device=torch.device("cpu")
        )
        fluid_frac = float((~mask).float().mean().item())
        assert abs(fluid_frac - porosity) < 0.05

    def test_reproducibility(self) -> None:
        m1 = random_porosity_mask_3d(
            nz=10, ny=10, nx=10, porosity=0.5, device=torch.device("cpu"), seed=7
        )
        m2 = random_porosity_mask_3d(
            nz=10, ny=10, nx=10, porosity=0.5, device=torch.device("cpu"), seed=7
        )
        assert (m1 == m2).all()

    def test_with_smoothing(self) -> None:
        mask = random_porosity_mask_3d(
            nz=15, ny=15, nx=15, porosity=0.5, device=torch.device("cpu"), sigma=1.5
        )
        assert mask.shape == (15, 15, 15)
        assert mask.any()
        assert not mask.all()


# ---------------------------------------------------------------------------
# compute_q_generic_3d
# ---------------------------------------------------------------------------


class TestComputeQGeneric3D:
    def test_output_shapes(self) -> None:
        nz, ny, nx = 8, 10, 12
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        mask[4, 5, 6] = True
        fb, q = compute_q_generic_3d(mask, device=torch.device("cpu"))
        assert fb.shape == (19, nz, ny, nx)
        assert q.shape == (19, nz, ny, nx)
        assert fb.dtype == torch.bool
        assert q.dtype == torch.float32

    def test_q_values_are_half(self) -> None:
        """q should be 0.5 everywhere (standard halfway bounce-back)."""
        nz, ny, nx = 8, 8, 8
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        mask[4, 4, 4] = True
        _, q = compute_q_generic_3d(mask, device=torch.device("cpu"))
        assert torch.allclose(q, torch.full_like(q, 0.5))

    def test_fluid_boundary_mask_detects_neighbours(self) -> None:
        """Fluid nodes adjacent to solid should be flagged in fluid_boundary_mask."""
        nz, ny, nx = 8, 8, 8
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        mask[4, 4, 4] = True  # single solid cell in interior
        fb, _ = compute_q_generic_3d(mask, device=torch.device("cpu"))
        # There must be at least some fluid boundary nodes around the solid
        assert fb.any()

    def test_empty_mask_gives_no_boundary(self) -> None:
        nz, ny, nx = 8, 8, 8
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        fb, _ = compute_q_generic_3d(mask, device=torch.device("cpu"))
        assert not fb.any()

    def test_full_mask_gives_no_fluid_boundary(self) -> None:
        """All-solid domain has no fluid nodes → no fluid_boundary entries."""
        nz, ny, nx = 4, 4, 4
        mask = torch.ones((nz, ny, nx), dtype=torch.bool)
        fb, _ = compute_q_generic_3d(mask, device=torch.device("cpu"))
        assert not fb.any()

    def test_sphere_mask_has_boundary(self) -> None:
        """A sphere solid mask should produce boundary nodes on its surface."""
        nz, ny, nx = 16, 16, 16
        zz, yy, xx = torch.meshgrid(
            torch.arange(nz), torch.arange(ny), torch.arange(nx), indexing="ij"
        )
        mask = (xx - 8.0) ** 2 + (yy - 8.0) ** 2 + (zz - 8.0) ** 2 <= 4.0 ** 2
        fb, q = compute_q_generic_3d(mask.bool(), device=torch.device("cpu"))
        assert fb.any()
        assert torch.isfinite(q).all()


# ---------------------------------------------------------------------------
# LBMUnitConverter
# ---------------------------------------------------------------------------


class TestLBMUnitConverter:
    def test_basic_construction(self) -> None:
        uc = LBMUnitConverter(re=100.0, l_phys=1.0, u_phys=1.0, nu_phys=0.01, nx=100)
        assert uc.tau > 0.5
        assert uc.ma > 0.0
        assert uc.dx == pytest.approx(0.01)

    def test_re_consistency_check(self) -> None:
        """Inconsistent Re should trigger a warning."""
        with pytest.warns(UserWarning, match="differs from"):
            LBMUnitConverter(re=200.0, l_phys=1.0, u_phys=1.0, nu_phys=0.01, nx=100)

    def test_mach_warning(self) -> None:
        """High u_lb should trigger a Mach number warning."""
        with pytest.warns(UserWarning, match="Mach number"):
            LBMUnitConverter(
                re=100.0, l_phys=1.0, u_phys=1.0, nu_phys=0.01, nx=100,
                u_lb=0.35,  # Ma ≈ 0.61 > 0.1
            )

    def test_phys_to_lb_round_trip(self) -> None:
        uc = LBMUnitConverter(re=100.0, l_phys=1.0, u_phys=1.0, nu_phys=0.01, nx=100)
        v = 0.5
        assert uc.lb_to_phys(uc.phys_to_lb(v)) == pytest.approx(v, rel=1e-6)

    def test_phys_to_lb_inlet_velocity(self) -> None:
        """phys_to_lb(u_phys) should return u_lb by definition."""
        uc = LBMUnitConverter(re=100.0, l_phys=1.0, u_phys=1.0, nu_phys=0.01, nx=100)
        assert uc.phys_to_lb(uc.u_phys) == pytest.approx(uc.u_lb, rel=1e-6)

    def test_nu_lb_from_re(self) -> None:
        """nu_lb = u_lb * nx / Re must hold."""
        uc = LBMUnitConverter(re=200.0, l_phys=1.0, u_phys=2.0, nu_phys=0.01, nx=200)
        assert uc.nu_lb == pytest.approx(uc.u_lb * uc.nx / uc.re, rel=1e-6)

    def test_tau_relation(self) -> None:
        """tau = 0.5 + nu_lb / cs^2 = 0.5 + 3 * nu_lb."""
        uc = LBMUnitConverter(re=100.0, l_phys=1.0, u_phys=1.0, nu_phys=0.01, nx=100)
        expected_tau = 0.5 + 3.0 * uc.nu_lb
        assert uc.tau == pytest.approx(expected_tau, rel=1e-6)

    def test_invalid_re(self) -> None:
        with pytest.raises(ValueError):
            LBMUnitConverter(re=-1.0, l_phys=1.0, u_phys=1.0, nu_phys=0.01, nx=100)

    def test_invalid_nx(self) -> None:
        with pytest.raises(ValueError):
            LBMUnitConverter(re=100.0, l_phys=1.0, u_phys=1.0, nu_phys=0.01, nx=0)

    def test_invalid_u_lb(self) -> None:
        with pytest.raises(ValueError):
            LBMUnitConverter(
                re=100.0, l_phys=1.0, u_phys=1.0, nu_phys=0.01, nx=100, u_lb=1.0
            )

    def test_time_conversion_round_trip(self) -> None:
        uc = LBMUnitConverter(re=100.0, l_phys=1.0, u_phys=1.0, nu_phys=0.01, nx=100)
        n = 500
        assert uc.phys_time_to_steps(uc.steps_to_phys_time(n)) == n

    def test_repr_contains_key_info(self) -> None:
        uc = LBMUnitConverter(re=100.0, l_phys=1.0, u_phys=1.0, nu_phys=0.01, nx=100)
        r = repr(uc)
        assert "Re=100.0" in r
        assert "Ma" in r

    def test_summary_returns_string(self) -> None:
        uc = LBMUnitConverter(re=100.0, l_phys=1.0, u_phys=1.0, nu_phys=0.01, nx=100)
        s = uc.summary()
        assert "tau" in s
        assert "Ma" in s

    def test_exports_from_tensorlbm(self) -> None:
        """LBMUnitConverter must be importable directly from tensorlbm."""
        from tensorlbm import LBMUnitConverter as UC  # noqa: PLC0415

        uc = UC(re=100.0, l_phys=1.0, u_phys=1.0, nu_phys=0.01, nx=100)
        assert uc.tau > 0.5
