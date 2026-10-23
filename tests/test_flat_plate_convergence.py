from __future__ import annotations

from copy import deepcopy

from tensorlbm.flat_plate_convergence import assess_flat_plate_convergence


def _record(length: int) -> dict[str, object]:
    reference = 0.0046875
    return {
        "schema": "tensorlbm-flat-plate-wall-model-v4",
        "configuration": {
            "shape_zyx": [3, length // 2, 2 * length],
            "plate_length": length,
            "plate_start_fraction": 0.2,
            "reynolds": 1e6,
            "resolved_reynolds": 2e4,
            "lattice_speed": 0.06,
            "wall_law": "musker",
            "stress_exchange_distance": 3.0 * length / 256.0,
            "steps": 125.0 * length,
            "warmup_steps": 62.5 * length,
            "ramp_steps": 8.0 * length,
            "statistics_window_steps_resolved": 62.5 * length,
            "report_interval": 4.0 * length,
            "wall_diagnostic_interval": length / 4.0,
            "sponge_width": 3 * length // 32,
            "sponge_strength": 0.2,
            "cv_margin": 3 * length // 128,
            "smagorinsky_cs": 0.05,
            "positivity_limiter": True,
            "link_force_frame": "laboratory_after_wall_activation",
            "wall_traction_source_scheme": ("mass_conservative_post_collision_guo_v2"),
        },
        "result": {
            "friction_coefficient": reference + 0.5 / length**2,
            "ittc_1957_reference": reference,
        },
        "acceptance": {"admitted": True},
    }


def test_equivalent_three_grid_sequence_is_admitted() -> None:
    result = assess_flat_plate_convergence(
        [
            _record(256),
            _record(384),
            _record(512),
        ]
    )
    assert result["configuration_identity"]["admitted"] is True
    assert result["configuration_identity"]["numerical_length_ratio_invariant"] is True
    assert result["spatial_convergence"]["observed_order"] > 1.99
    assert result["admitted"] is True


def test_legacy_schema_or_changed_exchange_ratio_fails_provenance() -> None:
    records = [_record(256), _record(384), _record(512)]
    records[0]["schema"] = "tensorlbm-flat-plate-wall-model-v2"
    changed = deepcopy(records)
    changed[1]["configuration"]["stress_exchange_distance"] = 3.0
    changed_margin = deepcopy(records)
    changed_margin[1]["configuration"]["cv_margin"] = 8
    changed_time = deepcopy(records)
    changed_time[1]["configuration"]["statistics_window_steps_resolved"] += 1
    assert assess_flat_plate_convergence(records)["admitted"] is False
    assert assess_flat_plate_convergence(changed)["admitted"] is False
    assert assess_flat_plate_convergence(changed_margin)["admitted"] is False
    assert (
        assess_flat_plate_convergence(changed_time)["configuration_identity"][
            "time_ratio_invariant"
        ]
        is False
    )


def test_pre_correction_wall_source_rejects_sequence() -> None:
    records = [_record(length) for length in (256.0, 384.0, 512.0)]
    records[1]["configuration"].pop("wall_traction_source_scheme")

    result = assess_flat_plate_convergence(records)

    assert result["configuration_identity"]["required_fields_present"] is False
    assert result["admitted"] is False


def test_flat_plate_convergence_assessor_is_public() -> None:
    import tensorlbm

    assert tensorlbm.assess_flat_plate_convergence is assess_flat_plate_convergence
