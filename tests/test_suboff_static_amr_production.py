from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

EXAMPLES = Path(__file__).parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
MODULE_PATH = EXAMPLES / "suboff_static_amr_resistance.py"
SPEC = importlib.util.spec_from_file_location(
    "suboff_static_amr_resistance_production", MODULE_PATH,
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def _arguments(tmp_path: Path, *, steps: int, resume: bool = False):
    checkpoint = tmp_path / "amr.ckpt"
    output = tmp_path / f"amr-{steps}.json"
    values = [
        "--device", "cpu", "--hull-type", "bare_hull",
        "--nx", "80", "--ny", "40", "--nz", "40",
        "--hull-length", "24", "--center-x-fraction", "0.35",
        "--wall-margin", "4", "--wake-cells", "8", "--cv-margin", "3",
        "--aux-cv-margins", "2,4", "--surface-force-interval", "1",
        "--steps", str(steps), "--warmup-steps", "2",
        "--report-interval", "2", "--average-window", "2",
        "--ramp-steps", "2", "--resolved-reynolds", "2000",
        "--collision-model", "cumulant_smagorinsky",
        "--wall-law", "musker", "--stress-exchange-distance", "1",
        "--wall-diagnostic-interval", "1", "--sponge-width", "3",
        "--checkpoint", str(checkpoint), "--checkpoint-interval", "2",
        "--output", str(output),
    ]
    if resume:
        values.append("--resume")
    return module.parser().parse_args(values), checkpoint


def test_static_amr_checkpoint_resumes_complete_evidence_ledger(
    tmp_path: Path,
) -> None:
    first_args, checkpoint = _arguments(tmp_path, steps=4)
    first = module.run(first_args)
    resumed_args, _ = _arguments(tmp_path, steps=6, resume=True)
    resumed = module.run(resumed_args)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert first["schema"] == "tensorlbm-suboff-static-amr-v6"
    assert resumed["configuration"]["resumed_from_step"] == 4
    assert resumed["result"]["finite"] is True
    assert resumed["acceptance"]["physical_validation"] is False
    assert resumed["acceptance"]["numerical_quality_admitted"] is False
    assert resumed["acceptance"]["duration_target_met"] is False
    assert resumed["geometry"]["surface_area_weighting"][
        "calibrated_area"
    ] == pytest.approx(resumed["geometry"]["wetted_area_lu2"], rel=1e-6)
    assert state["schema"] == "tensorlbm-suboff-static-amr-checkpoint-v6"
    assert state["step"] == 6
    assert len(state["force_history"]) == 4
    assert len(state["wall_y_plus_mean_history"]) == 4
    assert state["maximum_reflux_population_residual"] < 1e-6
    assert state["maximum_reflux_requested_correction"] >= 0.0
    assert state["maximum_reflux_applied_correction"] >= 0.0
    assert state["maximum_reflux_limited_directions"] == 0
    assert state["maximum_raw_kinetic_mismatch"] >= 0.0
    assert len(state["paired_primary_cv_samples"]) == 8
    assert set(state["auxiliary_cv_samples"]) == {2, 4}
    assert len(state["surface_pressure_samples"]) == 4
    assert len(state["paired_bfl_total_samples"]) == 4
    assert len(state["numerical_momentum_source_samples"]) == 8
    assert len(state["corrected_cv_samples"]) == 8
    assert resumed["result"]["nested_control_volume_invariance"][
        "auxiliary_count"
    ] == 2
    assert resumed["result"][
        "source_corrected_cv_vs_bfl_difference_pct"
    ] < 1.0
    assert resumed["configuration"]["wall_viscosity_basis"] == "physical_reynolds"
    assert resumed["configuration"]["wall_model_reynolds"] == pytest.approx(
        resumed["configuration"]["physical_reynolds"],
    )
    assert resumed["configuration"]["wall_nu_fine"] < 0.01 * (
        2.0 * (resumed["configuration"]["tau_coarse"] - 0.5) / 3.0
    )


def test_underresolved_aff8_records_component_and_area_evidence(
    tmp_path: Path,
) -> None:
    args, _ = _arguments(tmp_path, steps=4)
    args.hull_type = "full"
    result = module.run(args)
    geometry = result["geometry"]
    resolution = geometry["geometry_resolution"]

    assert geometry["wetted_area_scope"] == "bare_hull_analytical_approximation"
    assert geometry["force_integration_area_scope"] == "full"
    assert geometry["force_integration_calibrated_area_lu2"] > geometry[
        "bare_hull_wetted_area_lu2"
    ]
    assert resolution["hull_type"] == "full"
    assert resolution["sail_only_cells"] >= 0
    assert resolution["fin_only_cells"] >= 0
    assert resolution["absolute_reference_resolved"] is False
    assert result["acceptance"][
        "absolute_reference_geometry_target_met"
    ] is False
