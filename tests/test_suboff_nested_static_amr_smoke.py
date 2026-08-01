from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

EXAMPLES = Path(__file__).parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
MODULE_PATH = EXAMPLES / "suboff_nested_static_amr_smoke.py"
SPEC = importlib.util.spec_from_file_location("suboff_nested_static_amr_smoke", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _args(
    tmp_path: Path,
    *,
    steps: int = 2,
    preflight: bool = False,
    resume: bool = False,
    hull_type: str = "bare_hull",
):
    values = [
        "--device", "cpu",
        "--hull-type", hull_type,
        "--nx", "80", "--ny", "40", "--nz", "40",
        "--hull-length", "24", "--center-x-fraction", "0.35",
        "--outer-wall-margin", "4", "--outer-wake-cells", "8",
        "--inner-wall-margin", "3", "--inner-wake-cells", "0",
        "--cv-margin", "2", "--steps", str(steps), "--ramp-steps", "0",
        "--aux-cv-margins", "1,3", "--surface-force-interval", "1",
        "--warmup-steps", "0", "--statistics-window-steps", str(steps),
        "--report-interval", "1", "--wall-diagnostic-interval", "1",
        "--resolved-reynolds", "2000", "--sponge-width", "3",
        "--memory-bytes-per-cell", "742",
        "--output", str(tmp_path / "nested-smoke.json"),
        "--checkpoint", str(tmp_path / "nested-smoke.ckpt"),
        "--checkpoint-interval", "1",
    ]
    if preflight:
        values.append("--preflight-only")
    if resume:
        values.append("--resume")
    return MODULE.parser().parse_args(values)


def test_nested_suboff_smoke_closes_force_and_both_interfaces(tmp_path: Path) -> None:
    result = MODULE.run(_args(tmp_path))

    assert result["status"] == "integration_smoke_pass"
    assert result["physical_validation"] is False
    assert result["geometry"]["geometry_owner_level"] == 2
    assert result["geometry"]["force_owner_level"] == 2
    assert result["result"]["maximum_source_corrected_observer_difference_pct"] < 0.1
    assert max(result["result"]["maximum_reflux_residual_by_interface"]) < 1.0e-6
    assert result["acceptance"]["resistance_accuracy_assessed"] is False
    assert result["acceptance"]["fully_activated_steps_assessed"] == 2
    assert result["acceptance"]["single_grid_candidate"] is False
    assert result["result"]["statistics"]["statistics_window_steps_resolved"] == 2
    assert result["result"]["statistics"]["auxiliary_cv_difference_pct"] is not None
    assert result["result"]["statistics"]["surface_observer_difference_pct"] is not None
    assert result["result"]["statistics"]["wall_exchange"]["mean_distance_cells"] > 0.0


def test_nested_suboff_preflight_does_not_claim_physics(tmp_path: Path) -> None:
    result = MODULE.run(_args(tmp_path, preflight=True))

    assert result["status"] == "preflight_only"
    assert result["physical_validation"] is False
    assert result["planning"]["total_allocated_cells"] > 0
    assert result["planning"]["memory_estimate_bytes_per_cell"] == 742.0


def test_nested_suboff_checkpoint_restores_all_levels(tmp_path: Path) -> None:
    first = MODULE.run(_args(tmp_path, steps=1))
    resumed = MODULE.run(_args(tmp_path, steps=2, resume=True))

    assert first["result"]["steps"][0]["step"] == 1
    assert [record["step"] for record in resumed["result"]["steps"]] == [1, 2]
    assert resumed["configuration"]["resumed_from_step"] == 1
    assert resumed["status"] == "integration_smoke_pass"


def test_nested_aff8_smoke_records_appendage_resolution(tmp_path: Path) -> None:
    result = MODULE.run(_args(tmp_path, steps=1, hull_type="full"))

    resolution = result["geometry"]["resolution"]
    assert resolution["hull_type"] == "full"
    assert result["geometry"]["appendage_halfway_links"] > 0
    assert resolution["sail_only_cells"] > 0
    assert resolution["fin_only_cells"] > 0
    assert result["physical_validation"] is False


def test_bare_hull_can_resume_exact_legacy_v2_signature(tmp_path: Path) -> None:
    MODULE.run(_args(tmp_path, steps=1))
    checkpoint = tmp_path / "nested-smoke.ckpt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state["configuration"]["schema_version"] = 2
    state["configuration"].pop("hull_type")
    state["schema"] = "tensorlbm-suboff-nested-amr-smoke-checkpoint-v2"
    torch.save(state, checkpoint)

    resumed = MODULE.run(_args(tmp_path, steps=2, resume=True))

    assert resumed["configuration"]["resumed_legacy_v2_checkpoint"] is True
    assert resumed["configuration"]["resumed_from_step"] == 1
