"""Prevent a passing number from silently changing method or evidence scope."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VM = ROOT / "benchmarks" / "validation_matrix"
sys.path.insert(0, str(VM))
import runner  # noqa: E402
from evidence_admission import admit_evidence  # noqa: E402


def test_shallow_water_and_correlations_are_not_lbm_validation_passes():
    for cid, method in (
        ("dam_break_ritter_ci", "shallow_water_fvm"),
        ("ship_resistance_suboff", "lbm"),
        ("sphere_re100", "lbm"),
    ):
        case = runner.find_case(runner.load_matrix(), cid)
        result = runner.evaluate_case(case)
        assert result["evidence_admission"]["method"] == method
        assert not result["lbm_validation_pass"]
        assert not result["evidence_admission"]["engineering_validated"]
    # Keep the real, useful finite-volume numeric result rather than erase it.
    result = runner.evaluate_case(runner.find_case(runner.load_matrix(), "dam_break_ritter_ci"))
    assert result["machine_pass"]


def test_unknown_origin_is_withheld_even_if_numeric_metrics_pass(tmp_path):
    case = copy.deepcopy(runner.find_case(runner.load_matrix(), "cavity_re100"))
    case.pop("evidence")
    result = runner.evaluate_case(case)
    assert result["machine_pass"]
    assert not result["lbm_validation_pass"]


@pytest.mark.parametrize(
    "marker",
    [
        {"synthetic": True},
        {"metadata": {"evidence_type": "synthetic_fixture"}},
        {"provenance": {"usable_for_physical_validation": False}},
    ],
)
def test_synthetic_marker_overrides_an_otherwise_passing_artifact(tmp_path, marker):
    case = copy.deepcopy(runner.find_case(runner.load_matrix(), "cavity_re100"))
    payload = json.loads((ROOT / case["measure"]["artifact"]).read_text())
    payload.update(marker)
    artifact = tmp_path / "fixture.json"
    artifact.write_text(json.dumps(payload))
    case["measure"]["artifact"] = str(artifact)
    result = runner.evaluate_case(case)
    assert result["status"] == "failed"
    assert not result["machine_pass"]
    assert not result["lbm_validation_pass"]


def test_synthetic_generator_cannot_emit_validation_evidence(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "synthetic_generator", ROOT / "generate_synthetic_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = module.main(output_dir=tmp_path)
    assert (tmp_path / "synthetic_plot_fixture.json").exists()
    assert not (tmp_path / "verification_results.json").exists()
    assert "acceptance_summary" not in payload
    assert payload["metadata"]["solver_steps_executed"] == 0
    case = {
        "evidence": {
            "origin": "recorded_solver_run",
            "method": "lbm",
            "claim_scope": "benchmark_validation",
        }
    }
    assert not admit_evidence(case, payload)["lbm_validation_eligible"]


def test_verification_figure_loader_rejects_synthetic_fixture(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "channel_figures", ROOT / "generate_channel_figures.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "fake.json"
    path.write_text(json.dumps({"synthetic": True, "results": {}}))
    with pytest.raises(ValueError, match="Synthetic/demo"):
        module.load_verification_results(path)
