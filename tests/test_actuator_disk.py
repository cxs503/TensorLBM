"""Tests for actuator disk module."""
from __future__ import annotations

import torch
from tensorlbm.actuator_disk import (
    ActuatorDiskConfig,
    apply_actuator_disk,
    run_actuator_disk_benchmark,
)
from pathlib import Path


def test_actuator_disk_config() -> None:
    cfg = ActuatorDiskConfig(diameter=48.0, rpm_lu=0.0012)
    assert cfg.radius == 24.0
    assert cfg.hub_radius > 3.0
    assert cfg.disk_volume > 1000


def test_apply_actuator_disk() -> None:
    ux = torch.ones(40, 60, 80) * 0.04
    uy = torch.zeros(40, 60, 80)
    uz = torch.zeros(40, 60, 80)
    fx, fy, fz = apply_actuator_disk(ux, uy, uz, 24, 20, 20, 48.0, 0.18, 0.001)
    # Thrust is negative (pushes fluid backward)
    assert fx.max() <= 0.0
    assert fx.abs().gt(0).sum() > 0  # at least some cells
    # Swirl should be non-zero
    assert fy.abs().gt(0).sum() > 0 or fz.abs().gt(0).sum() > 0


def test_actuator_disk_benchmark_runs() -> None:
    """Minimal smoke test — one speed, few steps."""
    cfg = ActuatorDiskConfig(
        diameter=32.0, rpm_lu=0.002,
        inflow_velocities=(0.04,),
        nx=80, ny=40, nz=40,
        tau=0.58, smagorinsky_cs=0.0,
        n_steps=100, warmup_steps=30,
        device="cpu",
        output_root=Path("/tmp/ad_test"),
        overwrite=True,
    )
    result = run_actuator_disk_benchmark(cfg)
    assert "results" in result
    assert len(result["results"]) == 1
    r = result["results"][0]
    assert r["kt_measured"] is not None
    assert r["j"] > 0.1


__all__ = [
    "test_actuator_disk_config",
    "test_apply_actuator_disk",
    "test_actuator_disk_benchmark_runs",
]
