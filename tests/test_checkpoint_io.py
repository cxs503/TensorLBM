from __future__ import annotations

from pathlib import Path

import torch

from tensorlbm.checkpoint_io import atomic_torch_save


def test_atomic_torch_save_replaces_complete_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "state.ckpt"
    atomic_torch_save({"value": torch.tensor([1, 2])}, path)
    atomic_torch_save({"value": torch.tensor([3, 4])}, path)
    loaded = torch.load(path, weights_only=True)
    assert torch.equal(loaded["value"], torch.tensor([3, 4]))
    assert not list(tmp_path.glob(".*.tmp-*"))
