"""R1 application-boundary tests for the existing D3Q19 free-surface runner."""
from __future__ import annotations

import ast
import json
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import pytest

from tensorlbm.dam_break_3d import DamBreak3DConfig


def _fs_config(tmp_path: Path, **overrides: object) -> DamBreak3DConfig:
    values: dict[str, object] = {
        "nx": 32,
        "ny": 16,
        "nz": 16,
        "dam_width": 8,
        "fill_height": 8,
        "model": "fs",
        "n_steps": 2,
        "output_interval": 2,
        "output_root": tmp_path,
        "run_name": "application",
        "overwrite": True,
        "hydrostatic_init": False,
    }
    values.update(overrides)
    return DamBreak3DConfig(**values)


def _write_metadata(run_dir: Path, *, gate_passed: bool = True) -> Path:
    run_dir.mkdir()
    (run_dir / "run_metadata.json").write_text(json.dumps({
        "config": {"model": "fs"},
        "free_surface_quality_gate": {
            "passed": gate_passed,
            "diagnostic": "tracked-mass accounting only; not a physical/PV closure claim",
        },
        "free_surface_topology_events": [{"step": 1}],
    }), encoding="utf-8")
    return run_dir


def test_scenario_only_accepts_existing_free_surface_config_and_result_is_immutable(tmp_path):
    from tensorlbm.free_surface_application import FreeSurfaceScenario

    scenario = FreeSurfaceScenario("dam-break-r1", _fs_config(tmp_path))
    assert scenario.config.model == "fs"
    assert scenario.metadata["lattice"] == "D3Q19"
    assert scenario.metadata["formulation"] == "Körner"
    assert scenario.metadata["validation_scope"] != "physical_accuracy"
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        scenario.scenario_id = "other"  # type: ignore[misc]
    with pytest.raises(ValueError, match="scenario_id"):
        FreeSurfaceScenario("", _fs_config(tmp_path))
    with pytest.raises(ValueError, match="model"):
        FreeSurfaceScenario("wrong-model", _fs_config(tmp_path, model="cg"))


def test_normal_runner_metadata_is_deeply_immutable_and_accuracy_is_withheld(tmp_path, monkeypatch):
    import tensorlbm.free_surface_application as app

    run_dir = _write_metadata(tmp_path / "normal")
    calls: list[DamBreak3DConfig] = []
    monkeypatch.setattr(app, "run_dam_break_3d", lambda config: calls.append(config) or run_dir)

    config = _fs_config(tmp_path)
    result = app.run_free_surface_scenario(app.FreeSurfaceScenario("normal", config))

    assert calls == [config]
    assert result.state == "COMPLETED"
    assert result.mass_gate_status == "PASS"
    assert result.validation_status == "WITHHELD"
    assert result.run_metadata["free_surface_topology_events"][0]["step"] == 1
    assert result.evidence["formulation"] == "D3Q19 Körner"
    assert result.evidence["dynamic_topology_physical_accuracy"] == "WITHHELD"
    assert result.evidence["physical_reference"] == "WITHHELD"
    assert result.evidence["D3Q27_phase_field_equivalence"] is False
    assert result.evidence["color_gradient_equivalence"] is False
    assert result.evidence["physical_accuracy_claim"] is False
    assert is_dataclass(result)
    with pytest.raises(TypeError):
        result.run_metadata["config"]["model"] = "cg"
    with pytest.raises(TypeError):
        result.run_metadata["free_surface_topology_events"][0]["step"] = 2


def test_metadata_wrapping_artifact_is_explicitly_cpu_only(tmp_path):
    from tensorlbm.free_surface_application import write_metadata_wrapping_observation

    artifact = tmp_path / "metadata-observation.json"
    write_metadata_wrapping_observation(artifact)
    observation = json.loads(artifact.read_text(encoding="utf-8"))

    assert observation["scope"] == "CPU metadata wrapping observation"
    assert observation["does_not_represent"] == [
        "solver performance", "GPU performance", "SDAA performance", "physical performance",
    ]


def test_real_small_config_delegates_to_existing_runner_once(tmp_path, monkeypatch):
    import tensorlbm.free_surface_application as app

    real_runner = app.run_dam_break_3d
    calls = 0

    def counted_runner(config: DamBreak3DConfig):
        nonlocal calls
        calls += 1
        return real_runner(config)

    monkeypatch.setattr(app, "run_dam_break_3d", counted_runner)
    result = app.run_free_surface_scenario(app.FreeSurfaceScenario("small-real", _fs_config(tmp_path)))

    assert calls == 1
    assert result.state == "COMPLETED"
    assert result.mass_gate_status == "PASS"
    assert result.validation_status == "WITHHELD"


def test_existing_strict_runner_failure_maps_to_failed_result(tmp_path, monkeypatch):
    import tensorlbm.free_surface_application as app

    def fail_closed(_: DamBreak3DConfig) -> Path:
        raise RuntimeError("free-surface quality gate fail-closed at step 1: tracked mass")

    monkeypatch.setattr(app, "run_dam_break_3d", fail_closed)
    result = app.run_free_surface_scenario(app.FreeSurfaceScenario("strict-failure", _fs_config(tmp_path)))

    assert result.state == "FAILED"
    assert result.mass_gate_status == "FAIL"
    assert result.validation_status == "FAIL"
    assert "quality gate fail-closed" in result.evidence["failure_reason"]


def test_runner_call_boundary_has_no_loop_step_or_backend_compile():
    import inspect
    import tensorlbm.free_surface_application as app

    source = inspect.getsource(app)
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "run_free_surface_scenario")
    assert not any(isinstance(node, (ast.For, ast.While, ast.AsyncFor)) for node in ast.walk(function))
    assert "free_surface_step" not in source
    assert "TorchBackend" not in source
    assert "compile(" not in source