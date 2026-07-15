"""Immutable, validated records for CPU D3Q19 MRT performance baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
import re
from typing import Any

_SCHEMA_VERSION = "tensorlbm.performance.r1"
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


def _is_int(value: object) -> bool:
    """Accept integers while rejecting bool, which is an ``int`` subclass."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_float(value: object) -> bool:
    """Accept numeric scalar values while rejecting bool and non-finite values."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


@dataclass(frozen=True)
class BenchmarkSpec:
    """Inputs for a direct PyTorch D3Q19 MRT CPU benchmark."""

    shape: tuple[int, int, int] = (16, 16, 16)
    warmup_steps: int = 3
    measured_steps: int = 10
    tau: float = 0.6
    dtype_name: str = "float32"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(not _is_int(size) or size < 3 for size in self.shape):
            raise ValueError("shape must contain exactly three integer dimensions, each >= 3")
        if not _is_int(self.warmup_steps) or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer")
        if not _is_int(self.measured_steps) or self.measured_steps < 1:
            raise ValueError("measured_steps must be a positive integer")
        if not _is_finite_float(self.tau) or self.tau <= 0.5:
            raise ValueError("tau must be finite and > 0.5")
        if self.dtype_name != "float32":
            raise ValueError("R1 supports only dtype_name='float32'")
        if self.device != "cpu":
            raise ValueError("R1 supports only device='cpu'")


@dataclass(frozen=True)
class PerformanceArtifact:
    """Validated result of one direct PyTorch D3Q19 MRT measurement."""

    schema_version: str
    git_sha: str
    backend: str
    lattice: str
    collision: str
    device: str
    dtype_name: str
    grid: tuple[int, int, int]
    warmup_steps: int
    measured_steps: int
    median_step_seconds: float
    p95_step_seconds: float
    mlups: float
    peak_memory_bytes: int | None

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION!r}")
        if not isinstance(self.git_sha, str) or not _GIT_SHA_PATTERN.fullmatch(self.git_sha):
            raise ValueError("git_sha must be a 40-character lowercase hexadecimal SHA")
        if self.backend != "torch" or self.lattice != "D3Q19" or self.collision != "MRT":
            raise ValueError("artifact must describe the torch D3Q19 MRT baseline")
        BenchmarkSpec(
            shape=self.grid,
            warmup_steps=self.warmup_steps,
            measured_steps=self.measured_steps,
            tau=0.6,
            dtype_name=self.dtype_name,
            device=self.device,
        )
        for name, value in (
            ("median_step_seconds", self.median_step_seconds),
            ("p95_step_seconds", self.p95_step_seconds),
            ("mlups", self.mlups),
        ):
            if not _is_finite_float(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if self.p95_step_seconds < self.median_step_seconds:
            raise ValueError("p95_step_seconds must be >= median_step_seconds")
        if self.peak_memory_bytes is not None and (not _is_int(self.peak_memory_bytes) or self.peak_memory_bytes < 0):
            raise ValueError("peak_memory_bytes must be None or a non-negative integer")


def artifact_to_dict(artifact: PerformanceArtifact) -> dict[str, Any]:
    """Convert a completed artifact for caller-owned serialization or output."""
    return asdict(artifact)
