"""Tests for utils.py: device helpers, run dirs, and reproducibility metadata."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from tensorlbm import (
    DiagnosticPoint,
    configure_cpu_threads,
    get_reproducibility_metadata,
    prepare_run_dir,
    resolve_device,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestResolveDevice:
    def test_cpu_returns_cpu_device(self) -> None:
        device = resolve_device("cpu")
        assert device == torch.device("cpu")

    def test_cuda_unavailable_raises(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("CUDA is available; this test is for CPU-only environments")
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device("cuda")

    def test_mps_unavailable_raises(self) -> None:
        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if mps_ok:
            pytest.skip("MPS is available; this test is for non-MPS environments")
        with pytest.raises(RuntimeError, match="MPS"):
            resolve_device("mps")

    def test_unknown_device_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported device"):
            resolve_device("tpu")

    def test_returns_torch_device_type(self) -> None:
        device = resolve_device("cpu")
        assert isinstance(device, torch.device)


class TestConfigureCpuThreads:
    def test_cpu_with_none_keeps_current_value(self) -> None:
        current = torch.get_num_threads()
        assert configure_cpu_threads("cpu", None) == current

    def test_non_cpu_device_is_noop(self) -> None:
        current = torch.get_num_threads()
        assert configure_cpu_threads("sdaa:0", 2) == current

    def test_invalid_thread_count_raises(self) -> None:
        with pytest.raises(ValueError, match="num_threads"):
            configure_cpu_threads("cpu", 0)


class TestPrepareRunDir:
    def test_creates_directory(self, tmp_path: Path) -> None:
        run_dir = prepare_run_dir(tmp_path, "my_sim", "run01", overwrite=False)
        assert run_dir.exists()

    def test_returns_correct_path(self, tmp_path: Path) -> None:
        run_dir = prepare_run_dir(tmp_path, "my_sim", "run01", overwrite=False)
        assert run_dir == tmp_path / "my_sim" / "run01"

    def test_nested_subdir_created(self, tmp_path: Path) -> None:
        run_dir = prepare_run_dir(tmp_path, "nested/subdir", "run01", overwrite=False)
        assert run_dir.exists()

    def test_raises_if_dir_exists_and_no_overwrite(self, tmp_path: Path) -> None:
        prepare_run_dir(tmp_path, "sim", "run01", overwrite=False)
        with pytest.raises(FileExistsError):
            prepare_run_dir(tmp_path, "sim", "run01", overwrite=False)

    def test_overwrite_removes_existing_dir(self, tmp_path: Path) -> None:
        run_dir = prepare_run_dir(tmp_path, "sim", "run01", overwrite=False)
        # Put a sentinel file inside
        (run_dir / "sentinel.txt").write_text("hello")
        # Now overwrite
        run_dir2 = prepare_run_dir(tmp_path, "sim", "run01", overwrite=True)
        assert run_dir2.exists()
        assert not (run_dir2 / "sentinel.txt").exists()


class TestGetReproducibilityMetadata:
    def test_returns_dict(self) -> None:
        meta = get_reproducibility_metadata()
        assert isinstance(meta, dict)

    def test_contains_python_version(self) -> None:
        meta = get_reproducibility_metadata()
        assert "python_version" in meta
        assert isinstance(meta["python_version"], str)

    def test_contains_package_versions(self) -> None:
        meta = get_reproducibility_metadata()
        assert "package_versions" in meta
        pkg = meta["package_versions"]
        assert isinstance(pkg, dict)
        assert "torch" in pkg

    def test_git_commit_key_present(self) -> None:
        meta = get_reproducibility_metadata()
        assert "git_commit" in meta
        # Value may be None (no git repo) or a string
        assert meta["git_commit"] is None or isinstance(meta["git_commit"], str)


class TestDiagnosticPoint:
    def test_creation(self) -> None:
        dp = DiagnosticPoint(step=10, mass=120.0, mass_drift=0.001, max_speed=0.05, mean_rho=1.0)
        assert dp.step == 10
        assert dp.mass == 120.0
        assert dp.mass_drift == pytest.approx(0.001)
        assert dp.max_speed == pytest.approx(0.05)
        assert dp.mean_rho == pytest.approx(1.0)

    def test_immutable(self) -> None:
        dp = DiagnosticPoint(step=1, mass=10.0, mass_drift=0.0, max_speed=0.01, mean_rho=1.0)
        with pytest.raises((AttributeError, TypeError)):
            dp.step = 99  # type: ignore[misc]
