from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch


def _load_cylinder_flow_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "cylinder_flow.py"
    spec = importlib.util.spec_from_file_location("cylinder_flow", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_args_overrides_values():
    mod = _load_cylinder_flow_module()
    cfg = mod.parse_args(
        [
            "--nx",
            "128",
            "--ny",
            "64",
            "--steps",
            "25",
            "--output-interval",
            "10",
            "--log-interval",
            "5",
            "--radius",
            "8",
            "--cx",
            "20",
            "--cy",
            "15",
            "--tau",
            "0.7",
            "--output-root",
            "outputs_test",
            "--run-name",
            "my-run",
        ]
    )
    assert cfg.nx == 128
    assert cfg.ny == 64
    assert cfg.n_steps == 25
    assert cfg.output_interval == 10
    assert cfg.log_interval == 5
    assert cfg.radius == 8
    assert cfg.cx == 20
    assert cfg.cy == 15
    assert cfg.tau == 0.7
    assert cfg.output_root == "outputs_test"
    assert cfg.run_name == "my-run"


def test_make_run_dir_and_metadata_written(tmp_path: Path):
    mod = _load_cylinder_flow_module()
    cfg = mod.parse_args(["--output-root", str(tmp_path), "--run-name", "testrun"])
    run_dir = mod.make_run_dir(cfg)
    assert run_dir.exists()
    assert run_dir.name == "testrun"

    metadata_path = mod.save_run_metadata(
        run_dir=run_dir,
        config=cfg,
        tau=0.62,
        cx=10.0,
        cy=11.0,
        device=torch.device("cpu"),
    )
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert data["config"]["run_name"] == "testrun"
    assert data["derived"]["tau"] == 0.62
    assert data["runtime"]["device"] == "cpu"
