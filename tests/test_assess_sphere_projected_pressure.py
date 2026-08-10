from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[1]
PATH = ROOT / "scripts" / "assess_sphere_projected_pressure.py"
SPEC = importlib.util.spec_from_file_location("assess_sphere_projected", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _artifacts(tmp_path: Path, *, offset: float = 0.0) -> tuple[Path, Path]:
    configuration = {
        "shape_zyx": [32, 32, 48],
        "radius": 4.0,
        "reynolds": 100.0,
        "lattice_speed": 0.04,
        "collision_model": "natural_kbc_d3q19",
        "far_field_mode": "non_equilibrium_extrapolation",
        "steps": 80,
        "statistics_window_steps_resolved": 80,
        "projected_pressure_interval": 1,
        "projected_pressure_reconstruction": "linear",
    }
    result = {
        "schema": "tensorlbm-sphere-bfl-control-volume-v3",
        "configuration": configuration,
        "acceptance": {"admitted": True},
    }
    result_path = tmp_path / "sphere.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    samples = [
        {
            "step": step,
            "pressure_force_x": 1.0 + offset,
            "paired_control_volume_force_x": 1.0,
            "diagnostics": {
                "requested_links": 100,
                "usable_links": 100,
                "fallback_cells": 0,
            },
        }
        for step in range(1, 81)
    ]
    checkpoint_path = tmp_path / "sphere.ckpt"
    torch.save({
        "step": 80,
        "configuration": configuration,
        "projected_pressure_samples": samples,
    }, checkpoint_path)
    return result_path, checkpoint_path


def test_paired_projected_pressure_candidate_passes_registered_gates(
    tmp_path: Path,
) -> None:
    result_path, checkpoint_path = _artifacts(tmp_path, offset=0.01)

    report = MODULE.assess(result_path, checkpoint_path)

    assert report["status"] == "single_grid_candidate_convergence_required"
    assert report["acceptance"]["single_grid_candidate"] is True
    assert report["acceptance"]["grid_convergence_assessed"] is False
    assert report["observations"]["mean_difference_pct"] == pytest.approx(1.0)


def test_large_projected_pressure_bias_is_rejected(tmp_path: Path) -> None:
    result_path, checkpoint_path = _artifacts(tmp_path, offset=0.1)

    report = MODULE.assess(result_path, checkpoint_path)

    assert report["status"] == "rejected"
    assert report["acceptance"]["mean_pair_target_met"] is False
    assert report["acceptance"]["used_for_production_force"] is False
