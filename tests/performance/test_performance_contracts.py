from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from tensorlbm.performance.contracts import BenchmarkSpec, PerformanceArtifact, artifact_to_dict


def test_benchmark_spec_is_frozen_and_validates_inputs():
    spec = BenchmarkSpec(shape=(3, 4, 5), warmup_steps=1, measured_steps=2, tau=0.6)

    assert spec.device == "cpu"
    assert spec.dtype_name == "float32"
    with pytest.raises(FrozenInstanceError):
        spec.tau = 0.7  # type: ignore[misc]

    for shape in ((2, 3, 3), (3, 3), (3, 3, 3, 3)):
        with pytest.raises(ValueError):
            BenchmarkSpec(shape=shape)
    for kwargs in (
        {"warmup_steps": -1},
        {"measured_steps": 0},
        {"tau": 0.5},
        {"dtype_name": "float64"},
        {"device": "cuda"},
    ):
        with pytest.raises(ValueError):
            BenchmarkSpec(**kwargs)


def test_performance_artifact_validates_and_serializes():
    artifact = PerformanceArtifact(
        schema_version="tensorlbm.performance.r1",
        git_sha="a" * 40,
        backend="torch",
        lattice="D3Q19",
        collision="MRT",
        device="cpu",
        dtype_name="float32",
        grid=(3, 4, 5),
        warmup_steps=1,
        measured_steps=3,
        median_step_seconds=0.2,
        p95_step_seconds=0.3,
        mlups=0.0003,
        peak_memory_bytes=None,
    )

    assert artifact_to_dict(artifact) == {
        "schema_version": "tensorlbm.performance.r1",
        "git_sha": "a" * 40,
        "backend": "torch",
        "lattice": "D3Q19",
        "collision": "MRT",
        "device": "cpu",
        "dtype_name": "float32",
        "grid": (3, 4, 5),
        "warmup_steps": 1,
        "measured_steps": 3,
        "median_step_seconds": 0.2,
        "p95_step_seconds": 0.3,
        "mlups": 0.0003,
        "peak_memory_bytes": None,
    }
    with pytest.raises(FrozenInstanceError):
        artifact.mlups = 1.0  # type: ignore[misc]
    for change in (
        {"backend": "numpy"},
        {"schema_version": "not-r1"},
        {"git_sha": "not-a-sha"},
        {"warmup_steps": True},
        {"measured_steps": True},
        {"median_step_seconds": float("nan")},
        {"median_step_seconds": True},
        {"p95_step_seconds": 0.0},
        {"p95_step_seconds": 0.1},
        {"mlups": True},
        {"mlups": float("inf")},
        {"peak_memory_bytes": True},
        {"peak_memory_bytes": -1},
    ):
        with pytest.raises(ValueError):
            replace(artifact, **change)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"shape": (True, 3, 3)},
        {"warmup_steps": True},
        {"measured_steps": True},
        {"tau": True},
    ),
)
def test_benchmark_spec_rejects_boolean_numeric_values(kwargs):
    with pytest.raises(ValueError):
        BenchmarkSpec(**kwargs)
