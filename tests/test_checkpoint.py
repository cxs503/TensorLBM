"""Tests for checkpoint.py: save_checkpoint and load_checkpoint."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from tensorlbm import load_checkpoint, save_checkpoint

if TYPE_CHECKING:
    from pathlib import Path


class TestSaveCheckpoint:
    def test_returns_run_dir(self, tmp_path: Path) -> None:
        f = torch.ones((9, 4, 6))
        result = save_checkpoint(f, step=10, run_dir=tmp_path)
        assert result == tmp_path

    def test_creates_tensor_file(self, tmp_path: Path) -> None:
        f = torch.ones((9, 4, 6))
        save_checkpoint(f, step=5, run_dir=tmp_path)
        assert (tmp_path / "checkpoint_f.pt").exists()

    def test_creates_meta_file(self, tmp_path: Path) -> None:
        f = torch.ones((9, 4, 6))
        save_checkpoint(f, step=5, run_dir=tmp_path)
        assert (tmp_path / "checkpoint_meta.json").exists()

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "subdir" / "nested"
        f = torch.ones((9, 4, 6))
        save_checkpoint(f, step=1, run_dir=new_dir)
        assert new_dir.exists()

    def test_extra_metadata_stored(self, tmp_path: Path) -> None:
        import json

        f = torch.ones((9, 4, 6))
        save_checkpoint(f, step=7, run_dir=tmp_path, extra={"re": 100.0, "label": "test"})
        meta = json.loads((tmp_path / "checkpoint_meta.json").read_text(encoding="utf-8"))
        assert meta["re"] == 100.0
        assert meta["label"] == "test"
        assert meta["step"] == 7

    def test_step_written_to_meta(self, tmp_path: Path) -> None:
        import json

        f = torch.ones((9, 4, 6))
        save_checkpoint(f, step=42, run_dir=tmp_path)
        meta = json.loads((tmp_path / "checkpoint_meta.json").read_text(encoding="utf-8"))
        assert meta["step"] == 42


class TestLoadCheckpoint:
    def _save(self, tmp_path: Path, f: torch.Tensor, step: int, extra: dict | None = None) -> None:
        save_checkpoint(f, step=step, run_dir=tmp_path, extra=extra)

    def test_roundtrip_tensor(self, tmp_path: Path) -> None:
        f_orig = torch.rand((9, 4, 6))
        self._save(tmp_path, f_orig, step=3)
        f_loaded, step, meta = load_checkpoint(tmp_path)
        assert torch.allclose(f_loaded, f_orig, atol=1e-6)

    def test_roundtrip_step(self, tmp_path: Path) -> None:
        f = torch.ones((9, 4, 6))
        self._save(tmp_path, f, step=99)
        _, step, _ = load_checkpoint(tmp_path)
        assert step == 99

    def test_roundtrip_meta_contains_step(self, tmp_path: Path) -> None:
        f = torch.ones((9, 4, 6))
        self._save(tmp_path, f, step=12)
        _, _, meta = load_checkpoint(tmp_path)
        assert meta["step"] == 12

    def test_roundtrip_extra_metadata(self, tmp_path: Path) -> None:
        f = torch.ones((9, 4, 6))
        self._save(tmp_path, f, step=1, extra={"nu": 0.01})
        _, _, meta = load_checkpoint(tmp_path)
        assert meta["nu"] == 0.01

    def test_missing_tensor_raises(self, tmp_path: Path) -> None:
        f = torch.ones((9, 4, 6))
        self._save(tmp_path, f, step=1)
        (tmp_path / "checkpoint_f.pt").unlink()
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path)

    def test_missing_meta_raises(self, tmp_path: Path) -> None:
        f = torch.ones((9, 4, 6))
        self._save(tmp_path, f, step=1)
        (tmp_path / "checkpoint_meta.json").unlink()
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path)

    def test_missing_step_metadata_raises(self, tmp_path: Path) -> None:
        import json

        f = torch.ones((9, 4, 6))
        self._save(tmp_path, f, step=1)
        (tmp_path / "checkpoint_meta.json").write_text(
            json.dumps({"label": "corrupt"}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="missing 'step' key"):
            load_checkpoint(tmp_path)

    def test_non_integer_step_metadata_raises(self, tmp_path: Path) -> None:
        import json

        f = torch.ones((9, 4, 6))
        self._save(tmp_path, f, step=1)
        (tmp_path / "checkpoint_meta.json").write_text(
            json.dumps({"step": "1"}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="'step' must be an integer"):
            load_checkpoint(tmp_path)

    def test_3d_tensor_roundtrip(self, tmp_path: Path) -> None:
        f_orig = torch.rand((19, 4, 6, 8))
        self._save(tmp_path, f_orig, step=5)
        f_loaded, step, _ = load_checkpoint(tmp_path)
        assert f_loaded.shape == f_orig.shape
        assert torch.allclose(f_loaded, f_orig, atol=1e-6)

    def test_loaded_on_cpu(self, tmp_path: Path) -> None:
        f = torch.ones((9, 4, 6))
        self._save(tmp_path, f, step=1)
        f_loaded, _, _ = load_checkpoint(tmp_path, device=torch.device("cpu"))
        assert f_loaded.device.type == "cpu"
