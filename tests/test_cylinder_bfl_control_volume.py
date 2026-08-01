from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

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


def test_short_periodic_cylinder_composition_is_finite() -> None:
    cfg = CylinderBFLControlVolumeConfig(
        nx=56, ny=40, nz=3, radius=4, center_x_fraction=0.35,
        reynolds=20, lattice_speed=0.04, steps=4, warmup_steps=2,
        ramp_steps=2, sponge_width=3, cv_margin=2, device="cpu",
    )
    result = run_cylinder_bfl_control_volume(cfg)["result"]
    assert result["finite"] is True
    assert math.isfinite(result["cd_control_volume"])
    assert math.isfinite(result["cd_bfl_link"])
    assert result["drag_stationarity"]["sufficiently_sampled"] is False


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
    completed = subprocess.run(
        [sys.executable, "examples/cylinder_bfl_cv_validate.py", "--help"],
        check=True, capture_output=True, text=True,
    )
    assert "--far-field-mode" in completed.stdout


def test_cylinder_checkpoint_can_resume(tmp_path: Path) -> None:
    checkpoint = tmp_path / "cylinder.ckpt"
    common = dict(
        nx=48, ny=36, nz=3, radius=4, center_x_fraction=0.35,
        reynolds=20, lattice_speed=0.04, warmup_steps=2,
        ramp_steps=2, sponge_width=3, cv_margin=2,
        report_interval=0, checkpoint_interval=2,
        checkpoint_path=str(checkpoint), device="cpu",
    )
    run_cylinder_bfl_control_volume(CylinderBFLControlVolumeConfig(
        **common, steps=4,
    ))
    resumed = run_cylinder_bfl_control_volume(CylinderBFLControlVolumeConfig(
        **common, steps=6, resume=True,
    ))
    assert checkpoint.exists()
    assert resumed["configuration"]["resumed_from_step"] == 4
    assert resumed["result"]["finite"] is True
