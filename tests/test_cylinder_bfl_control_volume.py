from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tensorlbm.cylinder_bfl_control_volume import (
    CYLINDER_RE100_CD_REFERENCE,
    CYLINDER_RE100_ST_REFERENCE,
    CylinderBFLControlVolumeConfig,
    estimate_strouhal_from_lift,
    run_cylinder_bfl_control_volume,
)


def test_cylinder_reference_and_tau() -> None:
    cfg = CylinderBFLControlVolumeConfig(radius=10, reynolds=100, lattice_speed=0.05)
    assert CYLINDER_RE100_CD_REFERENCE == 1.33
    assert CYLINDER_RE100_ST_REFERENCE == 0.164
    assert math.isclose(cfg.tau, 0.53)


def test_reference_domain_requires_unconfined_clearance() -> None:
    compact = CylinderBFLControlVolumeConfig(radius=12, nx=320, ny=200)
    unconfined = CylinderBFLControlVolumeConfig(radius=12, nx=480, ny=480)

    assert compact.domain_reference_adequate is False
    assert unconfined.domain_clearance_diameters == {
        "upstream_center_distance": 6.0,
        "downstream_center_distance": 14.0,
        "lateral_center_distance": 10.0,
    }
    assert unconfined.domain_reference_adequate is True


def test_short_periodic_cylinder_composition_is_finite() -> None:
    cfg = CylinderBFLControlVolumeConfig(
        nx=56, ny=40, nz=3, radius=4, center_x_fraction=0.35,
        reynolds=20, lattice_speed=0.04, steps=4, warmup_steps=2,
        ramp_steps=2, sponge_width=3, cv_margin=2, device="cpu",
    )
    artifact = run_cylinder_bfl_control_volume(cfg)
    result = artifact["result"]
    assert result["finite"] is True
    assert math.isfinite(result["cd_control_volume"])
    assert math.isfinite(result["cd_bfl_link"])
    assert result["drag_stationarity"]["sufficiently_sampled"] is False
    assert artifact["schema"] == "tensorlbm-cylinder-bfl-control-volume-v4"
    assert artifact["configuration"]["link_force_frame"] == (
        "laboratory_after_wall_activation"
    )
    assert artifact["acceptance"]["domain_reference_target_met"] is False


def test_short_natural_kbc_cylinder_composition_is_finite() -> None:
    cfg = CylinderBFLControlVolumeConfig(
        nx=56, ny=40, nz=3, radius=4, center_x_fraction=0.35,
        reynolds=20, lattice_speed=0.04, steps=4, warmup_steps=2,
        ramp_steps=2, sponge_width=3, cv_margin=2, device="cpu",
        collision_model="natural_kbc_d3q19",
        collision_chunk_cells=56 * 40,
    )

    artifact = run_cylinder_bfl_control_volume(cfg)

    assert artifact["result"]["finite"] is True
    assert artifact["configuration"]["collision_model"] == "natural_kbc_d3q19"
    assert artifact["result"]["collision_execution"]["collision_calls"] == 12


def test_short_planar_cumulant_cylinder_is_finite_and_extruded() -> None:
    cfg = CylinderBFLControlVolumeConfig(
        nx=56, ny=40, nz=3, radius=4, center_x_fraction=0.35,
        reynolds=20, lattice_speed=0.04, steps=4, warmup_steps=2,
        ramp_steps=2, sponge_width=3, cv_margin=2, device="cpu",
        collision_model="planar_cumulant_d2q9",
    )
    artifact = run_cylinder_bfl_control_volume(cfg)
    assert artifact["result"]["finite"] is True
    assert artifact["acceptance"]["planar_extrusion_target_met"] is True
    assert artifact["result"]["collision_execution"]["collision_calls"] == 4


def test_compiled_cylinder_collision_requires_natural_kbc() -> None:
    with pytest.raises(ValueError, match="natural_kbc"):
        CylinderBFLControlVolumeConfig(compile_natural_kbc=True).validate()


def test_negative_cylinder_collision_chunk_is_rejected() -> None:
    with pytest.raises(ValueError, match="collision_chunk_cells"):
        CylinderBFLControlVolumeConfig(collision_chunk_cells=-1).validate()


def test_unknown_cylinder_collision_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="collision_model"):
        CylinderBFLControlVolumeConfig(collision_model="magic").validate()


def test_strouhal_estimator_recovers_synthetic_lift_frequency() -> None:
    speed, diameter, target = 0.08, 20.0, 0.16
    frequency = target * speed / diameter
    lift = [math.sin(2.0 * math.pi * frequency * step) for step in range(20000)]
    estimated, cycles = estimate_strouhal_from_lift(
        lift, lattice_speed=speed, diameter=diameter,
    )
    assert estimated == pytest.approx(target, rel=0.02)
    assert cycles == pytest.approx(12.8, rel=0.02)


def test_cylinder_cli_help() -> None:
    source_root = Path(__file__).parents[1] / "src"
    completed = subprocess.run(
        [sys.executable, "examples/cylinder_bfl_cv_validate.py", "--help"],
        check=True, capture_output=True, text=True,
        env=os.environ | {"PYTHONPATH": str(source_root)},
    )
    assert "--far-field-mode" in completed.stdout
    assert "--collision-chunk-cells" in completed.stdout
    assert "--compile-natural-kbc" in completed.stdout


def test_cylinder_checkpoint_can_resume(tmp_path) -> None:
    checkpoint = tmp_path / "cylinder.ckpt"
    common = {
        "nx": 48, "ny": 36, "nz": 3, "radius": 4,
        "center_x_fraction": 0.35, "reynolds": 20,
        "lattice_speed": 0.04, "warmup_steps": 2,
        "ramp_steps": 2, "sponge_width": 3, "cv_margin": 2,
        "report_interval": 0, "checkpoint_interval": 2,
        "checkpoint_path": str(checkpoint), "device": "cpu",
    }
    run_cylinder_bfl_control_volume(CylinderBFLControlVolumeConfig(
        **common, steps=4,
    ))
    resumed = run_cylinder_bfl_control_volume(CylinderBFLControlVolumeConfig(
        **common, steps=6, resume=True,
    ))
    assert checkpoint.exists()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["schema"] == "tensorlbm-cylinder-checkpoint-v4"
    assert resumed["configuration"]["resumed_from_step"] == 4
    assert resumed["result"]["finite"] is True
    assert resumed["configuration"]["statistics_window_steps_requested"] == 0
    with pytest.raises(ValueError, match="configuration"):
        run_cylinder_bfl_control_volume(CylinderBFLControlVolumeConfig(
            **(common | {"sponge_width": 4}), steps=6, resume=True,
        ))
