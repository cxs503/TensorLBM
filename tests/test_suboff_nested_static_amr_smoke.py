from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
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
    regularize_restriction: bool = False,
    ghost_interpolation: str = "injection",
    enforce_transfer_positivity: bool = False,
    disable_wall_stress: bool = False,
    collision_model: str = "cumulant_smagorinsky",
    omega_bulk: float = 1.0,
    interface_filter_width: int = 0,
    interface_filter_strength: float = 0.0,
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
        "--ghost-interpolation", ghost_interpolation,
        "--collision-model", collision_model,
        "--kbc-max-iterations", "4",
        "--omega-bulk", str(omega_bulk),
        "--interface-filter-width", str(interface_filter_width),
        "--interface-filter-strength", str(interface_filter_strength),
        "--output", str(tmp_path / "nested-smoke.json"),
        "--checkpoint", str(tmp_path / "nested-smoke.ckpt"),
        "--checkpoint-interval", "1",
    ]
    if preflight:
        values.append("--preflight-only")
    if resume:
        values.append("--resume")
    if regularize_restriction:
        values.append("--regularize-restriction")
    if enforce_transfer_positivity:
        values.append("--enforce-transfer-positivity")
    if disable_wall_stress:
        values.append("--disable-wall-stress")
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
    assert result["result"]["statistics"]["target_reynolds_convective_times"] > 0
    assert result["result"]["statistics"]["fully_physical_convective_times"] > 0
    assert result["result"]["statistics"]["auxiliary_cv_difference_pct"] is not None
    assert result["result"]["statistics"]["surface_observer_difference_pct"] is not None
    assert result["result"]["statistics"]["wall_exchange"]["mean_distance_cells"] > 0.0


def test_nested_suboff_preflight_does_not_claim_physics(tmp_path: Path) -> None:
    result = MODULE.run(_args(tmp_path, preflight=True))

    assert result["status"] == "preflight_only"
    assert result["physical_validation"] is False
    assert result["planning"]["total_allocated_cells"] > 0
    assert result["planning"]["memory_estimate_bytes_per_cell"] == 742.0
    assert result["planning"]["wall_buffer_finest_cells"] == 6


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


def test_nested_smoke_can_regularize_both_restriction_interfaces(
    tmp_path: Path,
) -> None:
    result = MODULE.run(_args(tmp_path, steps=1, regularize_restriction=True))

    assert result["configuration"]["regularize_restriction"] is True
    assert result["result"]["finite"] is True
    assert max(result["result"]["maximum_reflux_residual_by_interface"]) < 1e-6


def test_nested_smoke_can_use_cell_centered_trilinear_ghosts(tmp_path: Path) -> None:
    result = MODULE.run(_args(tmp_path, steps=1, ghost_interpolation="trilinear"))

    assert result["configuration"]["ghost_interpolation"] == "trilinear"
    assert result["result"]["finite"] is True


def test_nested_smoke_records_transfer_positivity_diagnostics(tmp_path: Path) -> None:
    result = MODULE.run(_args(
        tmp_path,
        steps=1,
        enforce_transfer_positivity=True,
    ))

    assert result["configuration"]["enforce_transfer_positivity"] is True
    assert len(result["result"]["minimum_transfer_alpha_by_interface"]) == 2


def test_health_cadence_records_both_interface_ledgers(tmp_path: Path) -> None:
    args = _args(tmp_path, steps=1, enforce_transfer_positivity=True)
    args.health_interval = 1
    result = MODULE.run(args)

    health = result["result"]["population_health"][0]
    assert health["collision_resolved_reynolds"] == 2000.0
    assert len(health["collision_tau_by_level"]) == 3
    assert health["target_reynolds_reached"] is True
    assert health["maximum_collision_limited_fraction"] == 0.0
    assert health["maximum_wall_sample_rejected_fraction"] == 0.0
    assert [record["finite"] for record in health["levels"]] == [True, True, True]
    assert len(health["interfaces"]) == 2
    assert health["finest_peak_speed_context"] is not None
    assert "bfl_link_count" in health["finest_peak_speed_context"]
    assert all(
        "restriction_minimum_alpha" in record
        for record in health["interfaces"]
    )
    assert result["acceptance"]["population_health_target_met"] is True
    assert result["result"]["maximum_observed_speed"] < 0.3


def test_wall_stress_can_be_disabled_only_as_nonphysical_diagnostic(
    tmp_path: Path,
) -> None:
    result = MODULE.run(_args(tmp_path, steps=1, disable_wall_stress=True))

    assert result["configuration"]["disable_wall_stress"] is True
    assert result["acceptance"]["single_grid_candidate"] is False


def test_nested_smoke_dispatches_entropic_kbc_collision(tmp_path: Path) -> None:
    result = MODULE.run(_args(
        tmp_path,
        steps=1,
        collision_model="entropic_kbc",
    ))

    assert result["configuration"]["collision_model"] == "entropic_kbc"
    assert result["result"]["finite"] is True


def test_nested_smoke_records_independent_bulk_relaxation(tmp_path: Path) -> None:
    result = MODULE.run(_args(tmp_path, steps=1, omega_bulk=0.5))

    assert result["configuration"]["omega_bulk"] == 0.5
    assert result["result"]["finite"] is True


def test_nested_smoke_can_filter_both_physical_interface_shells(
    tmp_path: Path,
) -> None:
    result = MODULE.run(_args(
        tmp_path,
        steps=1,
        interface_filter_width=1,
        interface_filter_strength=0.2,
    ))

    assert result["configuration"]["interface_filter_width"] == 1
    assert result["configuration"]["interface_filter_strength"] == 0.2
    assert result["result"]["finite"] is True


def test_nested_smoke_uses_smooth_resolved_viscosity_continuation(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, steps=2)
    args.resolved_reynolds_start = 500.0
    args.resolved_reynolds = 2000.0
    args.viscosity_ramp_start_step = 0
    args.viscosity_ramp_end_step = 2
    result = MODULE.run(args)

    records = result["result"]["steps"]
    assert records[0]["collision_resolved_reynolds"] == pytest.approx(800.0)
    assert records[1]["collision_resolved_reynolds"] == 2000.0
    assert result["result"]["statistics"]["statistics_window_steps_resolved"] == 1
    assert result["result"]["statistics"]["target_reynolds_steps_available"] == 1
    assert result["acceptance"]["target_reynolds_steps_assessed"] == 1
    assert result["acceptance"]["target_reynolds_reached"] is True
    assert result["result"]["statistics"]["target_reynolds_convective_times"] == pytest.approx(
        0.06 / 24.0,
    )
    assert result["acceptance"]["target_reynolds_duration_target_met"] is False
    assert result["configuration"]["initial_tau_by_level"][0] > (
        result["configuration"]["tau_by_level"][0]
    )


def test_bare_hull_can_resume_exact_legacy_v2_signature(tmp_path: Path) -> None:
    MODULE.run(_args(tmp_path, steps=1))
    checkpoint = tmp_path / "nested-smoke.ckpt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state["configuration"]["schema_version"] = 2
    state["configuration"].pop("hull_type")
    state["configuration"].pop("regularize_restriction")
    state["configuration"].pop("ghost_interpolation")
    state["configuration"].pop("enforce_transfer_positivity")
    state["configuration"].pop("interface_filter_width")
    state["configuration"].pop("interface_filter_strength")
    state["configuration"].pop("wall_stress_enabled")
    state["configuration"].pop("collision_model")
    state["configuration"].pop("kbc_max_iterations")
    state["configuration"].pop("omega_bulk")
    state["configuration"].pop("resolved_reynolds_start")
    state["configuration"].pop("viscosity_ramp_start_step")
    state["configuration"].pop("viscosity_ramp_end_step")
    state["schema"] = "tensorlbm-suboff-nested-amr-smoke-checkpoint-v2"
    torch.save(state, checkpoint)

    resumed = MODULE.run(_args(tmp_path, steps=2, resume=True))

    assert resumed["configuration"]["resumed_legacy_v2_checkpoint"] is True
    assert resumed["configuration"]["resumed_from_step"] == 1


def test_baseline_can_resume_v3_checkpoint_before_transfer_options(
    tmp_path: Path,
) -> None:
    MODULE.run(_args(tmp_path, steps=1))
    checkpoint = tmp_path / "nested-smoke.ckpt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state["configuration"].pop("regularize_restriction")
    state["configuration"].pop("ghost_interpolation")
    state["configuration"].pop("enforce_transfer_positivity")
    state["configuration"].pop("interface_filter_width")
    state["configuration"].pop("interface_filter_strength")
    state["configuration"].pop("wall_stress_enabled")
    state["configuration"].pop("collision_model")
    state["configuration"].pop("kbc_max_iterations")
    state["configuration"].pop("omega_bulk")
    state["configuration"].pop("resolved_reynolds_start")
    state["configuration"].pop("viscosity_ramp_start_step")
    state["configuration"].pop("viscosity_ramp_end_step")
    torch.save(state, checkpoint)

    resumed = MODULE.run(_args(tmp_path, steps=2, resume=True))

    assert resumed["configuration"]["resumed_legacy_v3_checkpoint"] is True
    assert resumed["configuration"]["resumed_from_step"] == 1
