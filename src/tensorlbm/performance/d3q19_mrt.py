"""Direct PyTorch D3Q19 MRT CPU hot-path performance baseline."""

from __future__ import annotations

import ast
import inspect
from statistics import median
from time import perf_counter
from typing import Callable

import torch

from ..d3q19 import equilibrium3d
from ..solver3d import collide_mrt3d, stream3d
from .contracts import BenchmarkSpec, PerformanceArtifact

_SCHEMA_VERSION = "tensorlbm.performance.r1"


def build_d3q19_mrt_initial_state(spec: BenchmarkSpec) -> torch.Tensor:
    """Create the deterministic float32 equilibrium-like D3Q19 state at startup."""
    rho = torch.ones(spec.shape, dtype=torch.float32, device=spec.device)
    zero = torch.zeros_like(rho)
    return equilibrium3d(rho, zero, zero, zero)


def run_d3q19_mrt_benchmark(
    spec: BenchmarkSpec,
    git_sha: str,
    clock: Callable[[], float] = perf_counter,
) -> PerformanceArtifact:
    """Measure direct MRT-collision plus streaming steps; MLUPS counts lattice cells."""
    if not git_sha:
        raise ValueError("git_sha must be non-empty")
    f = build_d3q19_mrt_initial_state(spec)
    for _ in range(spec.warmup_steps):
        f = collide_mrt3d(f, spec.tau)
        f = stream3d(f)

    step_seconds: list[float] = []
    for _ in range(spec.measured_steps):
        start = clock()
        f = collide_mrt3d(f, spec.tau)
        f = stream3d(f)
        step_seconds.append(clock() - start)

    median_step_seconds = float(median(step_seconds))
    ordered = sorted(step_seconds)
    p95_position = (len(ordered) - 1) * 0.95
    lower = int(p95_position)
    upper = min(lower + 1, len(ordered) - 1)
    p95_step_seconds = float(
        ordered[lower] + (ordered[upper] - ordered[lower]) * (p95_position - lower)
    )
    cells = spec.shape[0] * spec.shape[1] * spec.shape[2]
    return PerformanceArtifact(
        schema_version=_SCHEMA_VERSION,
        git_sha=git_sha,
        backend="torch",
        lattice="D3Q19",
        collision="MRT",
        device=spec.device,
        dtype_name=spec.dtype_name,
        grid=spec.shape,
        warmup_steps=spec.warmup_steps,
        measured_steps=spec.measured_steps,
        median_step_seconds=median_step_seconds,
        p95_step_seconds=p95_step_seconds,
        mlups=cells / median_step_seconds / 1_000_000.0,
        peak_memory_bytes=None,
    )


def assert_hot_path_uses_direct_torch_kernels() -> None:
    """Fail if the measured loop stops being a direct MRT plus streaming hot path."""
    source = inspect.getsource(run_d3q19_mrt_benchmark)
    function = ast.parse(source).body[0]
    if not isinstance(function, ast.FunctionDef):
        raise AssertionError("benchmark source did not parse as a function")
    loops = [node for node in ast.walk(function) if isinstance(node, ast.For)]
    if len(loops) != 2:
        raise AssertionError("benchmark must contain distinct warmup and measured loops")
    measured_loop = loops[-1]
    direct_calls = [
        node.func.id
        for node in ast.walk(measured_loop)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    if direct_calls.count("collide_mrt3d") != 1 or direct_calls.count("stream3d") != 1:
        raise AssertionError("measured loop must call collide_mrt3d and stream3d exactly once")
    forbidden = {"json", "logging", "agent", "registry", "ModelComposition", "backend"}
    for node in ast.walk(measured_loop):
        if isinstance(node, ast.Name) and node.id in forbidden:
            raise AssertionError(f"forbidden hot-path reference: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in forbidden:
            raise AssertionError(f"forbidden hot-path reference: {node.attr}")
