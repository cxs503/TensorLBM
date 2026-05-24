"""Tests for the 3D porous-media module."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from tensorlbm.porous_media3d import (
    PorousDrainageConfig3D,
    make_random_sphere_medium,
    make_tube_array_medium_3d,
    run_porous_drainage_3d,
)


class TestMakeRandomSphereMedium:
    def test_returns_correct_shape(self) -> None:
        nz, ny, nx = 20, 16, 16
        solid = make_random_sphere_medium(nz, ny, nx, n_spheres=3, r_min=1.5, r_max=3.0, seed=0)
        assert solid.shape == (nz, ny, nx)
        assert solid.dtype == torch.bool

    def test_has_face_walls(self) -> None:
        nz, ny, nx = 20, 16, 16
        solid = make_random_sphere_medium(nz, ny, nx, n_spheres=3, r_min=1.5, r_max=3.0)
        assert solid[0].all(), "z=0 face should be solid"
        assert solid[-1].all(), "z=-1 face should be solid"
        assert solid[:, 0, :].all(), "y=0 face should be solid"
        assert solid[:, -1, :].all(), "y=-1 face should be solid"
        assert solid[:, :, 0].all(), "x=0 face should be solid"
        assert solid[:, :, -1].all(), "x=-1 face should be solid"

    def test_has_fluid_nodes(self) -> None:
        nz, ny, nx = 20, 16, 16
        solid = make_random_sphere_medium(nz, ny, nx, n_spheres=1, r_min=1.0, r_max=2.0, seed=0)
        # Should still have fluid nodes inside
        assert (~solid).any()

    def test_reproducible_with_same_seed(self) -> None:
        nz, ny, nx = 20, 16, 16
        s1 = make_random_sphere_medium(nz, ny, nx, n_spheres=4, r_min=1.5, r_max=3.0, seed=42)
        s2 = make_random_sphere_medium(nz, ny, nx, n_spheres=4, r_min=1.5, r_max=3.0, seed=42)
        assert torch.equal(s1, s2)

    def test_different_seeds_differ(self) -> None:
        nz, ny, nx = 20, 16, 16
        s1 = make_random_sphere_medium(nz, ny, nx, n_spheres=4, r_min=1.5, r_max=3.0, seed=1)
        s2 = make_random_sphere_medium(nz, ny, nx, n_spheres=4, r_min=1.5, r_max=3.0, seed=2)
        assert not torch.equal(s1, s2)


class TestMakeTubeArrayMedium3D:
    def test_returns_correct_shape(self) -> None:
        nz, ny, nx = 20, 16, 16
        solid = make_tube_array_medium_3d(nz, ny, nx, n_tubes_y=2, n_tubes_x=2, tube_width=4)
        assert solid.shape == (nz, ny, nx)
        assert solid.dtype == torch.bool

    def test_has_face_walls(self) -> None:
        nz, ny, nx = 20, 16, 16
        solid = make_tube_array_medium_3d(nz, ny, nx, n_tubes_y=1, n_tubes_x=1, tube_width=4)
        assert solid[0].all()
        assert solid[-1].all()

    def test_has_fluid_nodes_in_interior(self) -> None:
        nz, ny, nx = 20, 16, 16
        solid = make_tube_array_medium_3d(nz, ny, nx, n_tubes_y=1, n_tubes_x=1, tube_width=6)
        # Interior z-slices should have some fluid nodes
        interior_solid = solid[1:-1]
        assert (~interior_solid).any()


class TestPorousDrainageConfig3D:
    def test_validate_ok(self) -> None:
        cfg = PorousDrainageConfig3D(nz=20, ny=16, nx=16, n_steps=10, output_interval=5)
        cfg.validate()  # should not raise

    def test_validate_bad_dims(self) -> None:
        import pytest
        cfg = PorousDrainageConfig3D(nz=4, ny=4, nx=4)
        with pytest.raises(ValueError, match="nz, ny, nx"):
            cfg.validate()

    def test_validate_bad_tau(self) -> None:
        import pytest
        cfg = PorousDrainageConfig3D(tau_water=0.4)
        with pytest.raises(ValueError, match="tau"):
            cfg.validate()

    def test_validate_bad_medium(self) -> None:
        import pytest
        cfg = PorousDrainageConfig3D(medium="invalid")
        with pytest.raises(ValueError, match="medium"):
            cfg.validate()

    def test_resolved_run_name_custom(self) -> None:
        cfg = PorousDrainageConfig3D(run_name="my_run")
        assert cfg.resolved_run_name() == "my_run"


class TestRunPorousDrainage3D:
    def test_smoke_random_spheres(self) -> None:
        """Short smoke test with random_spheres medium."""
        import math as _math
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = PorousDrainageConfig3D(
                nz=16,
                ny=12,
                nx=12,
                medium="random_spheres",
                n_spheres=2,
                r_min=1.5,
                r_max=2.5,
                G_12=0.7,
                tau_water=1.2,
                tau_gas=1.2,
                n_steps=20,
                output_interval=10,
                output_root=Path(tmpdir),
                run_name="smoke_spheres",
                overwrite=True,
            )
            result = run_porous_drainage_3d(cfg)
            assert "porosity" in result
            assert "saturation_series" in result
            series = result["saturation_series"]
            assert isinstance(series, list)
            assert len(series) >= 1
            for entry in series:
                val = float(entry["gas_saturation"])
                assert not _math.isnan(val), "Gas saturation should not be NaN"

    def test_smoke_tube_array(self) -> None:
        """Short smoke test with tube_array medium."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = PorousDrainageConfig3D(
                nz=16,
                ny=12,
                nx=12,
                medium="tube_array",
                n_tubes_y=1,
                n_tubes_x=1,
                tube_width=4,
                G_12=0.7,
                tau_water=1.2,
                tau_gas=1.2,
                n_steps=20,
                output_interval=10,
                output_root=Path(tmpdir),
                run_name="smoke_tubes",
                overwrite=True,
            )
            result = run_porous_drainage_3d(cfg)
            assert "porosity" in result
            assert result["porosity"] > 0.0

    def test_output_files_created(self) -> None:
        """Check that output files are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = PorousDrainageConfig3D(
                nz=16,
                ny=10,
                nx=10,
                medium="random_spheres",
                n_spheres=1,
                r_min=1.0,
                r_max=2.0,
                G_12=0.7,
                tau_water=1.2,
                tau_gas=1.2,
                n_steps=10,
                output_interval=5,
                output_root=Path(tmpdir),
                run_name="smoke_files",
                overwrite=True,
            )
            run_porous_drainage_3d(cfg)
            run_dir = Path(tmpdir) / "porous_drainage_3d" / "smoke_files"
            assert (run_dir / "run_metadata.json").exists()
            assert (run_dir / "saturation.csv").exists()
