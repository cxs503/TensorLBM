from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from tensorlbm import CylinderFlowConfig


def _load_cylinder_flow_module() -> object:
    module_path = Path(__file__).resolve().parents[1] / "examples" / "cylinder_flow.py"
    spec = importlib.util.spec_from_file_location("cylinder_flow", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_parser_defaults() -> None:
    mod = _load_cylinder_flow_module()
    parser = mod.build_parser()  # type: ignore[attr-defined]
    args = parser.parse_args([])
    assert args.nx == 320
    assert args.ny == 100
    assert args.re == 100.0
    assert args.n_steps == 1200
    assert args.output_interval == 200
    assert args.device == "cpu"
    assert args.num_threads is None


def test_build_parser_custom_values() -> None:
    mod = _load_cylinder_flow_module()
    parser = mod.build_parser()  # type: ignore[attr-defined]
    args = parser.parse_args([
        "--nx", "128",
        "--ny", "64",
        "--n-steps", "25",
        "--output-interval", "10",
        "--radius", "8",
        "--re", "120",
        "--run-name", "my-run",
        "--output-root", "outputs_test",
        "--num-threads", "2",
    ])
    assert args.nx == 128
    assert args.ny == 64
    assert args.n_steps == 25
    assert args.output_interval == 10
    assert args.radius == 8.0
    assert args.re == 120.0
    assert args.run_name == "my-run"
    assert args.output_root == "outputs_test"
    assert args.num_threads == 2


def test_cylinder_flow_config_from_defaults(tmp_path: Path) -> None:
    cfg = CylinderFlowConfig(
        nx=64,
        ny=32,
        n_steps=10,
        output_root=tmp_path,
        run_name="test-run",
    )
    assert cfg.tau > 0.5
    assert cfg.resolved_run_name() == "test-run"
    assert cfg.output_root == tmp_path
