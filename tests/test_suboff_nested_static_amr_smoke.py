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
    regularize_prolongation: bool = False,
    ghost_interpolation: str = "injection",
    reflux_correction_stencil: str = "exterior_cells",
    enforce_transfer_positivity: bool = False,
    disable_wall_stress: bool = False,
    collision_model: str = "cumulant_smagorinsky",
    omega_bulk: float = 1.0,
    wale_cw: float = 0.5,
    vreman_cv: float = 0.025,
    interface_filter_width: int = 0,
    interface_filter_strength: float = 0.0,
    sponge_inlet: bool = False,
    deep_wall_margin: int = 0,
    deep_wake_cells: int = 0,
):
    values = [
        "--device", "cpu",
        "--hull-type", hull_type,
        "--nx", "80", "--ny", "40", "--nz", "40",
        "--hull-length", "24", "--center-x-fraction", "0.35",
        "--outer-wall-margin", "4", "--outer-wake-cells", "8",
        "--inner-wall-margin", "3", "--inner-wake-cells", "0",
        "--deep-wall-margin", str(deep_wall_margin),
        "--deep-wake-cells", str(deep_wake_cells),
        "--cv-margin", "2", "--steps", str(steps), "--ramp-steps", "0",
        "--aux-cv-margins", "1,3", "--surface-force-interval", "1",
        "--warmup-steps", "0", "--statistics-window-steps", str(steps),
        "--report-interval", "1", "--wall-diagnostic-interval", "1",
        "--resolved-reynolds", "2000", "--sponge-width", "3",
        "--memory-bytes-per-cell", "742",
        "--ghost-interpolation", ghost_interpolation,
        "--reflux-correction-stencil", reflux_correction_stencil,
        "--collision-model", collision_model,
        "--kbc-max-iterations", "4",
        "--omega-bulk", str(omega_bulk),
        "--wale-cw", str(wale_cw),
        "--vreman-cv", str(vreman_cv),
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
    if regularize_prolongation:
        values.append("--regularize-prolongation")
    if enforce_transfer_positivity:
        values.append("--enforce-transfer-positivity")
    if disable_wall_stress:
        values.append("--disable-wall-stress")
    if sponge_inlet:
        values.append("--sponge-inlet")
    return MODULE.parser().parse_args(values)


def test_nested_smoke_can_include_upstream_sponge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = MODULE.build_sponge_sigma_3d
    observed: dict[str, tuple[str, ...]] = {}

    def capture_faces(*args, **kwargs):
        observed["faces"] = kwargs["faces"]
        return original(*args, **kwargs)

    monkeypatch.setattr(MODULE, "build_sponge_sigma_3d", capture_faces)
    result = MODULE.run(_args(tmp_path, steps=1, sponge_inlet=True))

    assert observed["faces"] == ("x-", "x+", "y-", "y+", "z-", "z+")
    assert result["configuration"]["sponge_inlet"] is True


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
    assert result["planning"]["cuda_persistent_allocated_gib_by_device"] == {}
    statistics = result["result"]["statistics"]
    assert statistics["mean_bfl_pressure_n"] is not None
    assert statistics["mean_wall_shear_n"] is not None
    assert statistics["bfl_pressure_fraction"] + statistics[
        "wall_shear_fraction"
    ] == pytest.approx(1.0)


def test_nested_suboff_preflight_does_not_claim_physics(tmp_path: Path) -> None:
    result = MODULE.run(_args(tmp_path, preflight=True))

    assert result["status"] == "preflight_only"
    assert result["physical_validation"] is False
    assert result["planning"]["total_allocated_cells"] > 0
    assert result["planning"]["memory_estimate_bytes_per_cell"] == 742.0
    assert result["planning"]["wall_buffer_finest_cells"] == 6
    clearance = result["planning"]["control_volume_interface_clearance"]
    assert clearance["all_flux_stencils_outside_filter"] is True
    assert len(clearance["volumes"]) == 3
    resolution = result["planning"]["geometry_resolution"]
    assert resolution["hull_type"] == "bare_hull"
    assert resolution["diameter_cells"] == pytest.approx(96.0 / 8.57)


def test_nested_aff8_preflight_measures_appendage_resolution(
    tmp_path: Path,
) -> None:
    result = MODULE.run(_args(
        tmp_path, steps=1, preflight=True, hull_type="full",
    ))

    resolution = result["planning"]["geometry_resolution"]
    assert resolution["hull_type"] == "full"
    assert resolution["appendage_boundary_links"] > 0
    assert resolution["appendage_halfway_links"] == 0
    assert resolution["appendage_link_scheme"] == (
        "continuous_parametric_bisection_v1"
    )
    assert resolution["sail_only_cells"] > 0
    assert resolution["fin_only_cells"] > 0


def test_four_level_preflight_and_runtime_use_deepest_geometry(tmp_path: Path) -> None:
    preflight = MODULE.run(_args(
        tmp_path, steps=1, preflight=True, deep_wall_margin=4,
    ))

    assert preflight["planning"]["refinement_depth"] == 3
    assert preflight["planning"]["level_count"] == 4
    assert preflight["planning"]["force_samples_per_root_step"] == 8
    assert len(preflight["planning"]["allocated_cells_by_level"]) == 4
    assert len(preflight["planning"]["fine_physical_shapes_by_level"]) == 3
    assert preflight["planning"]["stress_exchange_distance_cells"] == 1.0
    assert preflight["planning"]["estimated_exchange_y_plus"] > 0.0
    bounds = preflight["planning"]["estimated_bfl_exchange_y_plus_bounds"]
    assert bounds["maximum_exchange_y_plus_estimate"] >= (
        preflight["planning"]["estimated_exchange_y_plus"]
    )
    assert sum(preflight["planning"]["allocated_cells_by_level"]) == (
        preflight["planning"]["total_allocated_cells"]
    )

    result = MODULE.run(_args(
        tmp_path, steps=1, deep_wall_margin=4,
    ))
    checkpoint = torch.load(
        tmp_path / "nested-smoke.ckpt", map_location="cpu", weights_only=True,
    )
    assert result["geometry"]["geometry_owner_level"] == 3
    assert result["geometry"]["force_owner_level"] == 3
    assert result["configuration"]["force_samples_per_root_step"] == 8
    assert result["result"]["force_sample_aggregation"] == {
        "refinement_depth": 3,
        "expected_samples": 8,
        "observed_samples": 8,
        "uniform_sample_count_met": True,
    }
    assert len(result["result"]["maximum_reflux_residual_by_interface"]) == 3
    assert len(checkpoint["level_populations"]) == 4
    assert len(checkpoint["level_solid_masks"]) == 4
    assert all(mask is None for mask in checkpoint["level_solid_masks"][:3])
    assert bool(checkpoint["level_solid_masks"][3].any())


def test_health_record_exposes_recursive_wall_exchange_samples(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, steps=1, deep_wall_margin=4)
    args.health_interval = 1

    result = MODULE.run(args)

    health = result["result"]["population_health"][0]
    wall = health["wall_exchange"]
    assert wall["force_samples_observed"] == 8
    assert wall["force_samples_expected"] == 8
    assert wall["diagnostic_samples"] == 8
    assert wall["mean_distance_cells"] > 0.0
    assert wall["minimum_y_plus"] > 0.0
    assert wall["minimum_y_plus"] <= wall["mean_y_plus"]
    assert wall["mean_y_plus"] <= wall["maximum_y_plus"]


def test_nested_preflight_rejects_cv_flux_stencil_inside_interface_filter(
    tmp_path: Path,
) -> None:
    args = _args(
        tmp_path,
        preflight=True,
        interface_filter_width=3,
        interface_filter_strength=0.2,
    )

    with pytest.raises(ValueError, match="streaming flux stencil requires"):
        MODULE.run(args)


def test_nested_suboff_checkpoint_restores_all_levels(tmp_path: Path) -> None:
    first = MODULE.run(_args(tmp_path, steps=1))
    checkpoint_state = torch.load(
        tmp_path / "nested-smoke.ckpt", map_location="cpu", weights_only=True,
    )
    resumed = MODULE.run(_args(tmp_path, steps=2, resume=True))

    assert first["result"]["steps"][0]["step"] == 1
    assert [record["step"] for record in resumed["result"]["steps"]] == [1, 2]
    assert resumed["configuration"]["resumed_from_step"] == 1
    assert resumed["status"] == "integration_smoke_pass"
    assert len(checkpoint_state["level_solid_masks"]) == 3
    assert checkpoint_state["level_solid_masks"][0] is None
    assert checkpoint_state["level_solid_masks"][1] is None
    assert bool(checkpoint_state["level_solid_masks"][2].any())


def test_checkpoint_before_collision_chunk_option_can_resume(
    tmp_path: Path,
) -> None:
    MODULE.run(_args(
        tmp_path, steps=1, collision_model="natural_kbc",
    ))
    checkpoint = tmp_path / "nested-smoke.ckpt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state["configuration"].pop("collision_chunk_cells")
    torch.save(state, checkpoint)

    resume_args = _args(
        tmp_path, steps=2, resume=True, collision_model="natural_kbc",
    )
    resume_args.collision_chunk_cells = 512
    resumed = MODULE.run(resume_args)

    assert resumed["configuration"][
        "resumed_pre_collision_chunk_checkpoint"
    ] is True
    assert resumed["configuration"]["resumed_from_step"] == 1
    assert resumed["configuration"]["collision_chunk_cells"] == 512


def test_checkpoint_without_mass_conservative_wall_source_is_rejected(
    tmp_path: Path,
) -> None:
    MODULE.run(_args(tmp_path, steps=1))
    path = tmp_path / "nested-smoke.ckpt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    state["configuration"].pop("wall_traction_source_scheme")
    torch.save(state, path)

    with pytest.raises(ValueError, match="configuration does not match"):
        MODULE.run(_args(tmp_path, steps=2, resume=True))


def test_nested_aff8_smoke_records_appendage_resolution(tmp_path: Path) -> None:
    result = MODULE.run(_args(tmp_path, steps=1, hull_type="full"))

    resolution = result["geometry"]["resolution"]
    assert resolution["hull_type"] == "full"
    assert result["geometry"]["appendage_boundary_links"] > 0
    assert result["geometry"]["appendage_halfway_links"] == 0
    assert result["geometry"]["appendage_link_intersection"][
        "target_links"
    ] == result["geometry"]["appendage_boundary_links"]
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


def test_nested_smoke_can_regularize_both_prolongation_interfaces(
    tmp_path: Path,
) -> None:
    result = MODULE.run(_args(tmp_path, steps=1, regularize_prolongation=True))

    assert result["configuration"]["regularize_prolongation"] is True
    assert result["result"]["finite"] is True


def test_nested_smoke_can_use_cell_centered_trilinear_ghosts(tmp_path: Path) -> None:
    result = MODULE.run(_args(tmp_path, steps=1, ghost_interpolation="trilinear"))

    assert result["configuration"]["ghost_interpolation"] == "trilinear"
    assert result["result"]["finite"] is True


def test_nested_smoke_can_use_crossing_link_reflux(tmp_path: Path) -> None:
    result = MODULE.run(_args(
        tmp_path,
        steps=1,
        reflux_correction_stencil="crossing_links",
    ))

    assert result["configuration"]["reflux_correction_stencil"] == "crossing_links"
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
    assert health["wall_normal_activation"] == 1.0
    assert health["wall_shear_activation"] == 1.0
    assert health["target_reynolds_reached"] is True
    assert health["maximum_collision_limited_fraction"] == 0.0
    assert health["maximum_wall_sample_rejected_fraction"] == 0.0
    assert [record["finite"] for record in health["levels"]] == [True, True, True]
    assert len(health["interfaces"]) == 2
    assert all(
        record["raw_mass_mismatch"] >= 0.0
        and record["raw_momentum_mismatch_norm"] >= 0.0
        for record in health["interfaces"]
    )
    assert health["finest_peak_speed_context"] is not None
    assert "bfl_link_count" in health["finest_peak_speed_context"]
    assert all(
        "restriction_minimum_alpha" in record
        for record in health["interfaces"]
    )
    assert all(
        "prolongation_minimum_alpha" in record
        for record in health["interfaces"]
    )
    assert all(
        "maximum_applied_correction_fraction" in record
        for record in health["interfaces"]
    )
    assert result["acceptance"]["population_health_target_met"] is True
    assert result["result"]["maximum_observed_speed"] < 0.3
    assert result["result"]["minimum_observed_population"] > 1.0e-8
    assert len(result["result"]["maximum_raw_mass_mismatch_by_interface"]) == 2
    assert len(
        result["result"]["maximum_raw_momentum_mismatch_by_interface"],
    ) == 2


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
    assert result["acceptance"]["collision_viscosity_target_met"] is False
    assert result["acceptance"]["single_grid_candidate"] is False


def test_nested_smoke_dispatches_natural_kbc_as_diagnostic(tmp_path: Path) -> None:
    result = MODULE.run(_args(
        tmp_path,
        steps=1,
        collision_model="natural_kbc",
    ))

    assert result["configuration"]["collision_model"] == "natural_kbc"
    assert result["acceptance"]["collision_viscosity_target_met"] is False
    assert result["acceptance"]["single_grid_candidate"] is False
    assert result["result"]["finite"] is True


def test_nested_smoke_runs_natural_kbc_with_bounded_collision_memory(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, steps=1, collision_model="natural_kbc")
    args.collision_chunk_cells = 512

    result = MODULE.run(args)

    assert result["configuration"]["collision_chunk_cells"] == 512
    assert result["planning"]["collision_chunk_cells"] == 512
    assert result["planning"]["wall_force_direction_chunk"] == 4
    assert result["planning"]["low_memory_wall_macroscopic"] is False
    assert result["result"]["finite"] is True


@pytest.mark.parametrize(
    ("collision_model", "coefficient_key", "coefficient"),
    (
        ("cumulant_wale", "wale_cw", 0.5),
        ("cumulant_vreman", "vreman_cv", 0.025),
    ),
)
def test_nested_smoke_dispatches_gradient_sgs_as_diagnostic(
    tmp_path: Path,
    collision_model: str,
    coefficient_key: str,
    coefficient: float,
) -> None:
    result = MODULE.run(_args(
        tmp_path,
        steps=1,
        collision_model=collision_model,
        **{coefficient_key: coefficient},
    ))

    assert result["configuration"]["collision_model"] == collision_model
    assert result["configuration"][coefficient_key] == coefficient
    assert result["acceptance"]["collision_viscosity_target_met"] is True
    assert result["acceptance"]["single_grid_candidate"] is False
    assert result["result"]["finite"] is True


def test_nested_smoke_records_independent_bulk_relaxation(tmp_path: Path) -> None:
    result = MODULE.run(_args(tmp_path, steps=1, omega_bulk=0.5))

    assert result["configuration"]["omega_bulk"] == 0.5
    assert result["result"]["finite"] is True


def test_nested_single_grid_gate_requires_flat_plate_wall_scaling(
    tmp_path: Path,
) -> None:
    unscaled = MODULE.run(_args(tmp_path, steps=1))
    scaled_args = _args(tmp_path, steps=1)
    scaled_args.stress_exchange_distance = (3.0 / 256.0) * (4.0 * 24.0)
    scaled_args.checkpoint = tmp_path / "scaled.ckpt"
    scaled_args.output = tmp_path / "scaled.json"
    scaled = MODULE.run(scaled_args)

    assert unscaled["acceptance"]["wall_exchange_scaling_target_met"] is False
    assert scaled["acceptance"]["wall_exchange_scaling_target_met"] is True
    assert scaled["configuration"][
        "stress_exchange_distance_over_finest_length"
    ] == pytest.approx(3.0 / 256.0)


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


def test_nested_smoke_records_independent_wall_activation_ramps(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, steps=2)
    args.ramp_steps = 4
    args.wall_normal_ramp_steps = 1
    args.wall_shear_ramp_steps = 4
    args.health_interval = 1

    result = MODULE.run(args)

    health = result["result"]["population_health"][-1]
    assert health["wall_normal_activation"] == 1.0
    assert health["wall_shear_activation"] == pytest.approx(0.5)
    assert result["configuration"]["resolved_wall_normal_ramp_steps"] == 1
    assert result["configuration"]["resolved_wall_shear_ramp_steps"] == 4
    assert result["result"]["steps"][-1]["wall_fully_activated"] is False


def test_nested_health_population_floor_fails_during_run(tmp_path: Path) -> None:
    args = _args(tmp_path, steps=1)
    args.health_interval = 1
    args.minimum_health_population = 0.1

    with pytest.raises(FloatingPointError, match="population-health floor"):
        MODULE.run(args)


def test_nested_smoke_rejects_unbounded_sgs_coefficient(tmp_path: Path) -> None:
    args = _args(tmp_path, steps=1)
    args.cs_smag = 0.31

    with pytest.raises(ValueError, match="cs_smag"):
        MODULE.run(args)


def test_nested_health_reflux_correction_gate_fails_during_run(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, steps=1)
    args.health_interval = 1
    args.maximum_reflux_applied_correction_fraction = 1.0e-20

    with pytest.raises(FloatingPointError, match="reflux-correction gate"):
        MODULE.run(args)


def test_bare_hull_can_resume_exact_legacy_v2_signature(tmp_path: Path) -> None:
    MODULE.run(_args(tmp_path, steps=1))
    checkpoint = tmp_path / "nested-smoke.ckpt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state["configuration"]["schema_version"] = 2
    state["configuration"].pop("hull_type")
    state["configuration"].pop("regularize_restriction")
    state["configuration"].pop("regularize_prolongation")
    state["configuration"].pop("reflux_correction_stencil")
    state["configuration"].pop("ghost_interpolation")
    state["configuration"].pop("enforce_transfer_positivity")
    state["configuration"].pop("interface_filter_width")
    state["configuration"].pop("interface_filter_strength")
    state["configuration"].pop("wall_stress_enabled")
    state["configuration"].pop("collision_model")
    state["configuration"].pop("wale_cw")
    state["configuration"].pop("vreman_cv")
    state["configuration"].pop("kbc_max_iterations")
    state["configuration"].pop("omega_bulk")
    state["configuration"].pop("resolved_reynolds_start")
    state["configuration"].pop("viscosity_ramp_start_step")
    state["configuration"].pop("viscosity_ramp_end_step")
    state["configuration"].pop("wall_normal_ramp_steps")
    state["configuration"].pop("wall_shear_ramp_steps")
    state["configuration"].pop("minimum_health_population")
    state["configuration"].pop("maximum_positivity_limited_fraction")
    state["configuration"].pop("maximum_reflux_applied_correction_fraction")
    state["configuration"].pop("sponge_inlet")
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
    state["configuration"].pop("regularize_prolongation")
    state["configuration"].pop("reflux_correction_stencil")
    state["configuration"].pop("ghost_interpolation")
    state["configuration"].pop("enforce_transfer_positivity")
    state["configuration"].pop("interface_filter_width")
    state["configuration"].pop("interface_filter_strength")
    state["configuration"].pop("wall_stress_enabled")
    state["configuration"].pop("collision_model")
    state["configuration"].pop("wale_cw")
    state["configuration"].pop("vreman_cv")
    state["configuration"].pop("kbc_max_iterations")
    state["configuration"].pop("omega_bulk")
    state["configuration"].pop("resolved_reynolds_start")
    state["configuration"].pop("viscosity_ramp_start_step")
    state["configuration"].pop("viscosity_ramp_end_step")
    state["configuration"].pop("wall_normal_ramp_steps")
    state["configuration"].pop("wall_shear_ramp_steps")
    state["configuration"].pop("minimum_health_population")
    state["configuration"].pop("maximum_positivity_limited_fraction")
    state["configuration"].pop("maximum_reflux_applied_correction_fraction")
    state["configuration"].pop("sponge_inlet")
    torch.save(state, checkpoint)

    resumed = MODULE.run(_args(tmp_path, steps=2, resume=True))

    assert resumed["configuration"]["resumed_legacy_v3_checkpoint"] is True
    assert resumed["configuration"]["resumed_from_step"] == 1


def test_natural_kbc_can_resume_checkpoint_before_gradient_sgs_options(
    tmp_path: Path,
) -> None:
    MODULE.run(_args(
        tmp_path, steps=1, collision_model="natural_kbc",
    ))
    checkpoint = tmp_path / "nested-smoke.ckpt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state["configuration"].pop("wale_cw")
    state["configuration"].pop("vreman_cv")
    state["configuration"].pop("sponge_inlet")
    torch.save(state, checkpoint)

    resumed = MODULE.run(_args(
        tmp_path, steps=2, resume=True, collision_model="natural_kbc",
    ))

    assert resumed["configuration"]["resumed_pre_gradient_sgs_checkpoint"] is True
    assert resumed["configuration"]["resumed_from_step"] == 1


def test_current_checkpoint_can_resume_before_inlet_sponge_option(
    tmp_path: Path,
) -> None:
    MODULE.run(_args(tmp_path, steps=1))
    checkpoint = tmp_path / "nested-smoke.ckpt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state["configuration"].pop("sponge_inlet")
    torch.save(state, checkpoint)

    resumed = MODULE.run(_args(tmp_path, steps=2, resume=True))

    assert resumed["configuration"]["resumed_pre_inlet_sponge_checkpoint"] is True
    assert resumed["configuration"]["resumed_from_step"] == 1
