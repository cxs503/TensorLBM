"""Focused tests for the SUBOFF AI inference and error endpoints."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch


class _EchoEncoder:
    def eval(self) -> _EchoEncoder:
        return self

    def __call__(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        assert x.shape[2] == pos.shape[1]
        return x


class _EchoDecoder:
    def eval(self) -> _EchoDecoder:
        return self

    def __call__(self, z: torch.Tensor, pos_q: torch.Tensor, pos_kv: torch.Tensor) -> torch.Tensor:
        assert z.shape[2] == pos_q.shape[1] == pos_kv.shape[1]
        return z[:, 0]


def test_suboff_predict_uses_snapshot_coords_dir(monkeypatch: MonkeyPatch) -> None:
    from backend.routers import ai_suboff  # type: ignore[import-not-found]

    snapshot_dir = "/tmp/suboff_demo"
    seen_dirs: list[str | None] = []
    data = torch.arange(400, dtype=torch.float32).reshape(1, 100, 4)

    def fake_get_coords(data_dir: str | None, n_points: int) -> torch.Tensor:
        seen_dirs.append(data_dir)
        return torch.zeros((n_points, 3), dtype=torch.float32)

    monkeypatch.setattr(
        ai_suboff,
        "_get_model",
        lambda: (_EchoEncoder(), _EchoDecoder(), torch.device("cpu"), None),
    )
    monkeypatch.setattr(ai_suboff, "_get_coords", fake_get_coords)
    monkeypatch.setattr(
        ai_suboff,
        "_load_npy_snapshots",
        lambda data_dir: data if data_dir == snapshot_dir else None,
    )
    monkeypatch.setattr(ai_suboff.os.path, "isdir", lambda data_dir: data_dir == snapshot_dir)

    response = ai_suboff.suboff_predict(ai_suboff.SuboffPredictRequest(n_points=100))

    assert response["status"] == "ok"
    assert seen_dirs == [snapshot_dir]


def test_suboff_error_uses_available_coordinate_count(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    from backend.routers import ai_suboff  # type: ignore[import-not-found]

    data = torch.arange(800, dtype=torch.float32).reshape(2, 100, 4)

    monkeypatch.setattr(ai_suboff, "_load_npy_snapshots", lambda _data_dir: data)
    monkeypatch.setattr(ai_suboff, "_get_coords", lambda _data_dir, _n_points: torch.zeros((3, 3)))
    monkeypatch.setattr(ai_suboff, "build_model", lambda _device: (_EchoEncoder(), _EchoDecoder()))
    monkeypatch.setattr(ai_suboff, "CKPT_DIR", tmp_path)

    response = ai_suboff.suboff_error_analysis(
        ai_suboff.SuboffErrorRequest(data_dir=str(tmp_path), n_points=100, device="cpu")
    )

    assert response["status"] == "ok"
    assert response["n_points"] == 3
    assert response["summary"]["vx"]["rel_l2_mean"] == 0.0
