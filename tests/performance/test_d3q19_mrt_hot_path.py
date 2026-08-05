from __future__ import annotations

import ast
import inspect
import math

import pytest
import torch

import tensorlbm.performance.d3q19_mrt as benchmark_module
from tensorlbm.performance.contracts import BenchmarkSpec
from tensorlbm.performance.d3q19_mrt import (
    assert_hot_path_uses_direct_torch_kernels,
    build_d3q19_mrt_initial_state,
    run_d3q19_mrt_benchmark,
)


def test_initial_state_is_deterministic_d3q19_float32_cpu():
    spec = BenchmarkSpec(shape=(3, 4, 5))

    first = build_d3q19_mrt_initial_state(spec)
    second = build_d3q19_mrt_initial_state(spec)

    assert first.shape == (19, 3, 4, 5)
    assert first.dtype is torch.float32
    assert first.device.type == "cpu"
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_fake_clock_excludes_warmup_and_calculates_deterministic_metrics(monkeypatch):
    calls = []
    times = iter((10.0, 10.2, 20.0, 20.1, 30.0, 30.4))
    spec = BenchmarkSpec(shape=(3, 3, 3), warmup_steps=2, measured_steps=3, tau=0.6)

    def identity_collide(f, tau):
        calls.append(("collide", tau))
        return f

    def identity_stream(f):
        calls.append(("stream", None))
        return f

    monkeypatch.setattr(benchmark_module, "collide_mrt3d", identity_collide)
    monkeypatch.setattr(benchmark_module, "stream3d", identity_stream)

    artifact = run_d3q19_mrt_benchmark(spec, git_sha="d" * 40, clock=lambda: next(times))

    assert len(calls) == 2 * (spec.warmup_steps + spec.measured_steps)
    assert artifact.median_step_seconds == pytest.approx(0.2)
    assert artifact.p95_step_seconds == pytest.approx(0.38)
    assert artifact.mlups == pytest.approx(27 / 0.2 / 1_000_000)


def test_hot_path_ast_contract_uses_only_direct_kernel_calls():
    assert_hot_path_uses_direct_torch_kernels()

    function = next(
        node
        for node in ast.parse(inspect.getsource(benchmark_module)).body
        if isinstance(node, ast.FunctionDef) and node.name == "run_d3q19_mrt_benchmark"
    )
    measured_loop = next(
        node for node in ast.walk(function) if isinstance(node, ast.For) and node.target.id == "_"
    )
    calls = [node.func.id for node in ast.walk(measured_loop) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert calls.count("collide_mrt3d") == 1
    assert calls.count("stream3d") == 1


def test_real_small_cpu_smoke_returns_finite_d3q19_mrt_artifact():
    artifact = run_d3q19_mrt_benchmark(
        BenchmarkSpec(shape=(3, 3, 3), warmup_steps=1, measured_steps=2, tau=0.6),
        git_sha="e" * 40,
    )

    assert artifact.backend == "torch"
    assert artifact.lattice == "D3Q19"
    assert artifact.collision == "MRT"
    assert artifact.device == "cpu"
    assert artifact.peak_memory_bytes is None
    assert all(math.isfinite(value) and value > 0 for value in (
        artifact.median_step_seconds,
        artifact.p95_step_seconds,
        artifact.mlups,
    ))
