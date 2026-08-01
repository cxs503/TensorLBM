from __future__ import annotations

from copy import deepcopy

from tensorlbm.flat_plate_convergence import assess_flat_plate_convergence


def _record(length: int) -> dict[str, object]:
    reference = 0.0046875
    return {
        "schema": "tensorlbm-flat-plate-wall-model-v3",
        "configuration": {
            "shape_zyx": [3, length // 2, 2 * length],
            "plate_length": length,
            "plate_start_fraction": 0.2,
            "reynolds": 1e6,
            "resolved_reynolds": 2e4,
            "lattice_speed": 0.06,
            "wall_law": "musker",
            "stress_exchange_distance": 3.0 * length / 256.0,
            "ramp_steps": 2000,
            "sponge_width": 3 * length // 32,
            "sponge_strength": 0.2,
            "cv_margin": 3 * length // 128,
            "smagorinsky_cs": 0.05,
            "positivity_limiter": True,
        },
        "result": {
            "friction_coefficient": reference + 0.5 / length**2,
            "ittc_1957_reference": reference,
        },
        "acceptance": {"admitted": True},
    }


def test_equivalent_three_grid_sequence_is_admitted() -> None:
    result = assess_flat_plate_convergence([
        _record(256), _record(384), _record(512),
    ])
    assert result["configuration_identity"]["admitted"] is True
    assert result["configuration_identity"][
        "numerical_length_ratio_invariant"
    ] is True
    assert result["spatial_convergence"]["observed_order"] > 1.99
    assert result["admitted"] is True


def test_legacy_schema_or_changed_exchange_ratio_fails_provenance() -> None:
    records = [_record(256), _record(384), _record(512)]
    records[0]["schema"] = "tensorlbm-flat-plate-wall-model-v2"
    changed = deepcopy(records)
    changed[1]["configuration"]["stress_exchange_distance"] = 3.0
    changed_margin = deepcopy(records)
    changed_margin[1]["configuration"]["cv_margin"] = 8
    assert assess_flat_plate_convergence(records)["admitted"] is False
    assert assess_flat_plate_convergence(changed)["admitted"] is False
    assert assess_flat_plate_convergence(changed_margin)["admitted"] is False


def test_flat_plate_convergence_assessor_is_public() -> None:
    import tensorlbm

    assert tensorlbm.assess_flat_plate_convergence is assess_flat_plate_convergence
