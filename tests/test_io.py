"""Tests for io.py: save_vtk and save_hdf5."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tensorlbm import save_vtk


class TestSaveVtk2D:
    def test_creates_file(self, tmp_path: Path) -> None:
        ux = torch.rand((8, 10))
        uy = torch.rand((8, 10))
        path = save_vtk(tmp_path / "out.vtk", ux, uy)
        assert path.exists()

    def test_returns_path(self, tmp_path: Path) -> None:
        ux = torch.rand((8, 10))
        uy = torch.rand((8, 10))
        path = save_vtk(tmp_path / "out.vtk", ux, uy)
        assert isinstance(path, Path)

    def test_header_content(self, tmp_path: Path) -> None:
        ux = torch.rand((8, 10))
        uy = torch.rand((8, 10))
        path = save_vtk(tmp_path / "out.vtk", ux, uy)
        content = path.read_text(encoding="ascii")
        assert "vtk DataFile Version" in content
        assert "DATASET STRUCTURED_POINTS" in content
        assert "VECTORS velocity float" in content

    def test_dimensions_line(self, tmp_path: Path) -> None:
        ux = torch.rand((6, 12))
        uy = torch.rand((6, 12))
        path = save_vtk(tmp_path / "out.vtk", ux, uy)
        content = path.read_text(encoding="ascii")
        assert "DIMENSIONS 12 6 1" in content

    def test_with_rho(self, tmp_path: Path) -> None:
        ux = torch.rand((8, 10))
        uy = torch.rand((8, 10))
        rho = torch.ones((8, 10))
        path = save_vtk(tmp_path / "out.vtk", ux, uy, rho=rho)
        content = path.read_text(encoding="ascii")
        assert "SCALARS density float 1" in content

    def test_with_vorticity(self, tmp_path: Path) -> None:
        ux = torch.rand((8, 10))
        uy = torch.rand((8, 10))
        vort = torch.rand((8, 10))
        path = save_vtk(tmp_path / "out.vtk", ux, uy, vorticity=vort)
        content = path.read_text(encoding="ascii")
        assert "SCALARS vorticity float 1" in content

    def test_point_count(self, tmp_path: Path) -> None:
        ny, nx = 5, 7
        ux = torch.rand((ny, nx))
        uy = torch.rand((ny, nx))
        path = save_vtk(tmp_path / "out.vtk", ux, uy)
        content = path.read_text(encoding="ascii")
        assert f"POINT_DATA {ny * nx}" in content


class TestSaveVtk3D:
    def test_creates_file_3d(self, tmp_path: Path) -> None:
        ux = torch.rand((4, 6, 8))
        uy = torch.rand((4, 6, 8))
        uz = torch.rand((4, 6, 8))
        path = save_vtk(tmp_path / "out3d.vtk", ux, uy, uz=uz)
        assert path.exists()

    def test_dimensions_line_3d(self, tmp_path: Path) -> None:
        nz, ny, nx = 3, 5, 7
        ux = torch.rand((nz, ny, nx))
        uy = torch.rand((nz, ny, nx))
        uz = torch.rand((nz, ny, nx))
        path = save_vtk(tmp_path / "out3d.vtk", ux, uy, uz=uz)
        content = path.read_text(encoding="ascii")
        assert f"DIMENSIONS {nx} {ny} {nz}" in content

    def test_vectors_written_3d(self, tmp_path: Path) -> None:
        ux = torch.rand((4, 6, 8))
        uy = torch.rand((4, 6, 8))
        uz = torch.rand((4, 6, 8))
        path = save_vtk(tmp_path / "out3d.vtk", ux, uy, uz=uz)
        content = path.read_text(encoding="ascii")
        assert "VECTORS velocity float" in content

    def test_3d_with_rho(self, tmp_path: Path) -> None:
        ux = torch.rand((4, 6, 8))
        uy = torch.rand((4, 6, 8))
        uz = torch.rand((4, 6, 8))
        rho = torch.ones((4, 6, 8))
        path = save_vtk(tmp_path / "out3d.vtk", ux, uy, uz=uz, rho=rho)
        content = path.read_text(encoding="ascii")
        assert "SCALARS density float 1" in content


class TestSaveHdf5:
    pytest.importorskip("h5py")

    def test_creates_file(self, tmp_path: Path) -> None:
        try:
            import h5py  # noqa: F401
        except ImportError:
            pytest.skip("h5py not installed")
        from tensorlbm import save_hdf5

        ux = torch.rand((8, 10))
        uy = torch.rand((8, 10))
        path = save_hdf5(tmp_path / "out.h5", step=0, ux=ux, uy=uy)
        assert path.exists()

    def test_group_named_by_step(self, tmp_path: Path) -> None:
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not installed")
        from tensorlbm import save_hdf5

        ux = torch.rand((8, 10))
        uy = torch.rand((8, 10))
        path = save_hdf5(tmp_path / "out.h5", step=42, ux=ux, uy=uy)
        with h5py.File(path, "r") as fh:
            assert "step_000042" in fh

    def test_datasets_present(self, tmp_path: Path) -> None:
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not installed")
        from tensorlbm import save_hdf5

        ux = torch.rand((8, 10))
        uy = torch.rand((8, 10))
        path = save_hdf5(tmp_path / "out.h5", step=1, ux=ux, uy=uy)
        with h5py.File(path, "r") as fh:
            grp = fh["step_000001"]
            assert "ux" in grp
            assert "uy" in grp

    def test_optional_uz_and_rho(self, tmp_path: Path) -> None:
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not installed")
        from tensorlbm import save_hdf5

        ux = torch.rand((4, 6, 8))
        uy = torch.rand((4, 6, 8))
        uz = torch.rand((4, 6, 8))
        rho = torch.ones((4, 6, 8))
        path = save_hdf5(tmp_path / "out.h5", step=5, ux=ux, uy=uy, uz=uz, rho=rho)
        with h5py.File(path, "r") as fh:
            grp = fh["step_000005"]
            assert "uz" in grp
            assert "rho" in grp

    def test_step_attribute(self, tmp_path: Path) -> None:
        try:
            import h5py
        except ImportError:
            pytest.skip("h5py not installed")
        from tensorlbm import save_hdf5

        ux = torch.rand((8, 10))
        uy = torch.rand((8, 10))
        path = save_hdf5(tmp_path / "out.h5", step=7, ux=ux, uy=uy)
        with h5py.File(path, "r") as fh:
            assert fh["step_000007"].attrs["step"] == 7

    def test_overwrite_existing_group(self, tmp_path: Path) -> None:
        try:
            import h5py  # noqa: F401
        except ImportError:
            pytest.skip("h5py not installed")
        from tensorlbm import save_hdf5

        ux = torch.rand((8, 10))
        uy = torch.rand((8, 10))
        path = tmp_path / "out.h5"
        save_hdf5(path, step=1, ux=ux, uy=uy)
        # Write again — should not raise
        save_hdf5(path, step=1, ux=ux * 2.0, uy=uy)

    def test_import_error_without_h5py(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys
        h5py_backup = sys.modules.pop("h5py", None)
        monkeypatch.setitem(sys.modules, "h5py", None)  # type: ignore[arg-type]
        try:
            from tensorlbm.io import save_hdf5 as _save_hdf5
            with pytest.raises(ImportError, match="h5py"):
                _save_hdf5(tmp_path / "x.h5", step=0, ux=torch.zeros(2, 2), uy=torch.zeros(2, 2))
        finally:
            if h5py_backup is not None:
                sys.modules["h5py"] = h5py_backup
            else:
                sys.modules.pop("h5py", None)
