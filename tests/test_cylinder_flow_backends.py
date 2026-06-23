from __future__ import annotations

import json
from pathlib import Path

import tensorlbm.backends as B
from tensorlbm import CylinderFlowConfig, run_cylinder_flow
from tensorlbm.backends import torch_backend
from tensorlbm import cylinder_flow as cylinder_flow_mod


def test_run_cylinder_flow_explicit_backend_switches_temporarily(monkeypatch, tmp_path: Path) -> None:
    seen: list[str] = []

    def fake_run(*args, **kwargs):
        seen.append(B.get_backend())
        return tmp_path

    monkeypatch.setattr(cylinder_flow_mod, "_run_cylinder_flow_backend", fake_run)
    B.set_backend("torch")
    cfg = CylinderFlowConfig(
        nx=32,
        ny=16,
        radius=3.0,
        re=60.0,
        n_steps=4,
        output_interval=2,
        output_root=tmp_path,
        overwrite=True,
        backend="mindspore",
    )
    result = run_cylinder_flow(cfg)
    assert result == tmp_path
    assert seen == ["mindspore"]
    assert B.get_backend() == "torch"


def test_run_cylinder_flow_non_torch_backend_smoke(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cylinder_flow_mod, "get_ops", lambda: torch_backend)
    cfg = CylinderFlowConfig(
        nx=32,
        ny=16,
        radius=3.0,
        re=60.0,
        n_steps=6,
        output_interval=3,
        output_root=tmp_path,
        overwrite=True,
        backend="paddle",
    )
    run_dir = run_cylinder_flow(cfg)
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["runtime"]["backend"] == "paddle"
    assert (run_dir / "forces.csv").exists()
    assert (run_dir / "checkpoint_f.pt").exists()

