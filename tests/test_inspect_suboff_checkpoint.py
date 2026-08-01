from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inspect_suboff_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("inspect_suboff_checkpoint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _pairs(values: list[tuple[int, float]]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def test_audit_groups_fine_substeps_and_reports_observer_closure() -> None:
    state = {
        "schema": "tensorlbm.suboff-static-amr.v8",
        "step": 12,
        "configuration": {
            "hull_type": "bare_hull",
            "speed_knots": 5.92,
            "hull_length": 90.0,
            "rho_water": 998.2,
            "lattice_speed": 0.06,
            "warmup_steps": 8,
            "nu_water": 1.004e-6,
            "stress_exchange_distance": 2.109375,
        },
        "force_history": torch.tensor([10.0, 12.0], dtype=torch.float64),
        "bfl_total_history": torch.tensor([10.1, 12.1], dtype=torch.float64),
        "pressure_history": torch.tensor([4.0, 6.0], dtype=torch.float64),
        "wall_shear_history": torch.tensor([6.1, 6.1], dtype=torch.float64),
        "wall_y_plus_min_history": torch.tensor([30.0, 32.0]),
        "wall_y_plus_mean_history": torch.tensor([50.0, 54.0]),
        "wall_y_plus_max_history": torch.tensor([80.0, 90.0]),
        "wall_rejected_fraction_history": torch.tensor([0.0, 0.001]),
        "paired_primary_cv_samples": _pairs([
            (10, 10.0), (10, 12.0), (12, 14.0), (12, 16.0),
        ]),
        "paired_bfl_total_samples": _pairs([(10, 11.1), (12, 15.1)]),
        "numerical_momentum_source_samples": _pairs([
            (10, 0.05), (10, 0.15), (12, 0.05), (12, 0.15),
        ]),
        "corrected_cv_samples": _pairs([
            (10, 10.05), (10, 12.15), (12, 14.05), (12, 16.15),
        ]),
        "surface_total_samples": _pairs([(10, 12.0), (12, 16.0)]),
        "auxiliary_cv_samples": {
            4: _pairs([
                (10, 10.5), (10, 11.5), (12, 14.5), (12, 15.5),
            ]),
        },
    }

    report = MODULE.audit_checkpoint(state)

    assert report["sampled_coarse_steps"] == 2
    assert report["case"]["hull_type"] == "bare_hull"
    assert report["force_decomposition"]["mean_modeled_wall_shear_n"] == pytest.approx(6.1)
    assert report["wall_model_applicability"]["mean_observed_y_plus"] == pytest.approx(52.0)
    assert report["wall_model_applicability"]["ittc_exchange_y_plus_prior"] > 5000.0
    assert report["partial_physical_result"]["force_stationarity"][
        "sample_count"
    ] == 2
    closure = report["observer_closure"]
    assert closure["history_cv_vs_bfl"]["mean_residual_n"] == pytest.approx(0.1)
    assert closure["sampled_cv_vs_bfl"]["mean_difference_pct"] == pytest.approx(
        0.1 / 13.0 * 100.0,
    )
    assert closure["source_corrected_cv_vs_bfl"]["mean_difference_pct"] == pytest.approx(0.0)
    assert closure["auxiliary_control_volumes"]["4"]["mean_difference_pct"] == pytest.approx(0.0)


def test_grouping_accepts_one_or_more_samples_per_step() -> None:
    grouped = MODULE._group_sample_means(_pairs([
        (1, 2.0), (2, 3.0), (2, 5.0), (2, 7.0), (4, 11.0),
    ]))

    assert grouped.tolist() == [[1.0, 2.0], [2.0, 5.0], [4.0, 11.0]]
