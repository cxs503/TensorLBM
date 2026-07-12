"""Focused contracts for propeller raw ME persistence and KP505 wording."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from tensorlbm import propeller_benchmark
from tensorlbm.propeller_benchmark import PropellerBenchmarkConfig, _run_single_speed


_BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "bench_propeller", Path(__file__).parents[1] / "benchmarks" / "bench_propeller.py",
)
assert _BENCHMARK_SPEC and _BENCHMARK_SPEC.loader
_BENCHMARK_MODULE = importlib.util.module_from_spec(_BENCHMARK_SPEC)
_BENCHMARK_SPEC.loader.exec_module(_BENCHMARK_MODULE)


def test_single_speed_persists_each_post_warmup_me_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PropellerBenchmarkConfig(
        inflow_velocities=(0.005,), rpm=0.5, nx=40, ny=20, nz=20,
        warmup_steps=1, n_revolutions=1,
    )
    mask = torch.zeros((20, 20, 40), dtype=torch.bool)
    monkeypatch.setattr(propeller_benchmark, "resolve_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(propeller_benchmark, "build_propeller_mask", lambda **_: mask)
    monkeypatch.setattr(propeller_benchmark, "make_channel_wall_mask_3d", lambda *args, **kwargs: mask)
    monkeypatch.setattr(
        propeller_benchmark, "rotating_wall_velocity_3d",
        lambda *args, **kwargs: (torch.zeros_like(mask, dtype=torch.float32),) * 3,
    )
    monkeypatch.setattr(propeller_benchmark, "equilibrium3d", lambda *args, **kwargs: torch.zeros((19, 20, 20, 40)))
    monkeypatch.setattr(propeller_benchmark, "collide_smagorinsky_mrt3d", lambda f, **kwargs: f)
    monkeypatch.setattr(propeller_benchmark, "stream3d", lambda f: f)
    monkeypatch.setattr(propeller_benchmark, "apply_zou_he_channel_boundaries_3d", lambda f, **kwargs: f)
    monkeypatch.setattr(propeller_benchmark, "moving_wall_bounce_back_3d", lambda f, *args: f)
    values = iter((1.0, 10.0, 2.0, 20.0, 3.0, 30.0))
    monkeypatch.setattr(propeller_benchmark, "compute_obstacle_forces_3d", lambda *args: (torch.tensor(next(values)), torch.tensor(0.0), torch.tensor(0.0)))
    monkeypatch.setattr(propeller_benchmark, "compute_obstacle_moments_3d", lambda *args: (torch.tensor(next(values)), torch.tensor(0.0), torch.tensor(0.0)))

    result = _run_single_speed(config=config, u_in=0.005)

    samples = result["me_samples"]
    assert samples == [
        {"step": 2, "fx_me_lu": 2.0, "mx_me_lu": 20.0},
        {"step": 3, "fx_me_lu": 3.0, "mx_me_lu": 30.0},
    ]
    assert isinstance(samples, list)
    assert result["sampling_steps"] == len(samples) == 2


def test_kp505_rows_are_context_not_a_validation_verdict() -> None:
    report = _BENCHMARK_MODULE._summarize_kp505_context(
        [{"u_in": 0.005, "kt": 0.3, "kq": 0.04}], rpm=0.001, diameter=10.0,
    )

    assert report["claim_status"] == "context_only_not_validation"
    assert "pass" not in report
    assert "ok" not in report["matches"][0]
