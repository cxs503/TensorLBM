from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.solver3d import collide_mrt3d, stream3d
from tensorlbm.models.contracts import ModelComposition
from tensorlbm.models.torch_execution import (
    PlanPerformanceSample,
    TorchD3Q19MRTPlan,
    compile_torch_d3q19_mrt_plan,
    measure_plan_overhead,
)


def _composition(**overrides: object) -> ModelComposition:
    values: dict[str, object] = {
        "lattice": "D3Q19",
        "collision": "MRT",
        "turbulence": None,
        "forcing": (),
        "boundaries": (),
        "physics_modules": {"single_phase": "incompressible"},
    }
    values.update(overrides)
    return ModelComposition(**values)  # type: ignore[arg-type]


def _state() -> torch.Tensor:
    rho = torch.ones((3, 3, 3), dtype=torch.float32)
    zero = torch.zeros_like(rho)
    return equilibrium3d(rho, zero, zero, zero)


def test_compile_accepts_only_torch_d3q19_mrt_incompressible_single_phase():
    plan = compile_torch_d3q19_mrt_plan(_composition(), tau=0.6)

    assert isinstance(plan, TorchD3Q19MRTPlan)
    assert plan.collision_kernel is collide_mrt3d
    assert plan.stream_kernel is stream3d
    assert plan.tau == 0.6


@pytest.mark.parametrize(
    ("composition", "tau"),
    [
        (_composition(lattice="D3Q27"), 0.6),
        (_composition(collision="BGK"), 0.6),
        (_composition(physics_modules={"single_phase": "compressible"}), 0.6),
        (_composition(physics_modules={}), 0.6),
        (_composition(), True),
        (_composition(), 0.5),
    ],
)
def test_compile_rejects_incompatible_composition_or_tau(composition: ModelComposition, tau: object):
    with pytest.raises(ValueError):
        compile_torch_d3q19_mrt_plan(composition, tau=tau)  # type: ignore[arg-type]


def test_plan_step_is_bitwise_equal_to_direct_mrt_then_stream():
    initial = _state()
    plan = compile_torch_d3q19_mrt_plan(_composition(), tau=0.6)

    direct = stream3d(collide_mrt3d(initial.clone(), 0.6))
    actual = plan.step(initial.clone())

    assert torch.equal(actual, direct)


def test_step_ast_contains_only_the_two_prebound_kernel_calls():
    function = ast.parse(textwrap.dedent(inspect.getsource(TorchD3Q19MRTPlan.step))).body[0]
    assert isinstance(function, ast.FunctionDef)
    assert len(function.body) == 1
    assert isinstance(function.body[0], ast.Return)
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    assert len(calls) == 2
    assert [call.func.attr for call in calls if isinstance(call.func, ast.Attribute)] == ["stream_kernel", "collision_kernel"]
    names = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
    assert "composition" not in names
    assert not {"registry", "backend", "json", "logging", "Agent"} & names


def test_execution_module_has_no_framework_generic_backend_imports():
    module = ast.parse(inspect.getsource(__import__("tensorlbm.models.torch_execution", fromlist=["*"])))
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(module)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not {"paddle", "mindspore", "backend", "backends"} & imports


def test_measure_plan_overhead_reports_ratio_without_performance_verdict():
    initial = _state()
    plan = compile_torch_d3q19_mrt_plan(_composition(), tau=0.6)
    sample = measure_plan_overhead(
        plan,
        initial,
        lambda f: stream3d(collide_mrt3d(f, 0.6)),
        warmup_steps=1,
        measured_steps=3,
    )

    assert isinstance(sample, PlanPerformanceSample)
    assert sample.direct_median_seconds > 0.0
    assert sample.plan_median_seconds > 0.0
    assert sample.ratio == sample.plan_median_seconds / sample.direct_median_seconds
    assert not hasattr(sample, "passes")


def test_measure_plan_overhead_rejects_boolean_or_invalid_counts():
    plan = compile_torch_d3q19_mrt_plan(_composition(), tau=0.6)
    with pytest.raises(ValueError):
        measure_plan_overhead(plan, _state(), lambda f: f, warmup_steps=True, measured_steps=1)
    with pytest.raises(ValueError):
        measure_plan_overhead(plan, _state(), lambda f: f, warmup_steps=0, measured_steps=0)
