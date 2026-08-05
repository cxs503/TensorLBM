"""Tests for the R1 PyTorch-only cold-path backend facade."""
from __future__ import annotations

import ast
import inspect
from unittest.mock import patch

import pytest
import torch

from tensorlbm.backends.contracts import DeviceSpec
from tensorlbm.backends.torch_backend import TorchBackend, build_torch_lattice_constants
from tensorlbm.core.lattice import D3Q19, D3Q27
from tensorlbm.d3q19 import C, OPPOSITE, W, equilibrium3d
from tensorlbm.models.contracts import ModelComposition
from tensorlbm.models.torch_execution import compile_torch_d3q19_mrt_plan
from tensorlbm.solver3d import collide_mrt3d, stream3d


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


def test_torch_backend_capabilities_are_cpu_float32_r1_only() -> None:
    capabilities = TorchBackend().capabilities
    assert capabilities.support.value == "supported"
    assert capabilities.supported_devices == ("cpu",)
    assert capabilities.supported_dtypes == ("float32",)
    assert "D3Q19 MRT" in capabilities.notes


@pytest.mark.parametrize("spec", [DeviceSpec("cuda", "float32"), DeviceSpec("sdaa", "float32"), DeviceSpec("cpu", "float64")])
def test_validate_device_rejects_unverified_device_or_dtype(spec: DeviceSpec) -> None:
    with pytest.raises(ValueError, match="R1 supports only"):
        TorchBackend().validate_device(spec)


def test_compile_rejects_unsupported_device_dtype_and_composition() -> None:
    backend = TorchBackend()
    with pytest.raises(ValueError):
        backend.compile_d3q19_mrt(_composition(), 0.6, DeviceSpec("cuda", "float32"))
    with pytest.raises(ValueError):
        backend.compile_d3q19_mrt(_composition(), 0.6, DeviceSpec("cpu", "float64"))
    with pytest.raises(ValueError):
        backend.compile_d3q19_mrt(_composition(lattice="D3Q27"), 0.6, DeviceSpec("cpu", "float32"))


def test_compile_calls_existing_compiler_once_and_is_direct_path_equal() -> None:
    backend = TorchBackend()
    with patch("tensorlbm.backends.torch_backend.compile_torch_d3q19_mrt_plan", wraps=compile_torch_d3q19_mrt_plan) as compiler:
        plan = backend.compile_d3q19_mrt(_composition(), 0.6, DeviceSpec("cpu", "float32"))
    assert compiler.call_count == 1
    initial = _state()
    assert torch.equal(plan.step(initial.clone()), stream3d(collide_mrt3d(initial.clone(), 0.6)))


def test_backend_has_no_step_or_hot_path_dispatch() -> None:
    assert "step" not in TorchBackend.__dict__
    module = ast.parse(inspect.getsource(__import__("tensorlbm.backends.torch_backend", fromlist=["*"])))
    class_node = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "TorchBackend")
    method_names = {node.name for node in class_node.body if isinstance(node, ast.FunctionDef)}
    assert method_names == {"capabilities", "validate_device", "compile_d3q19_mrt"}
    names = {node.id for node in ast.walk(class_node) if isinstance(node, ast.Name)}
    assert names.isdisjoint({"get_backend", "get_ops", "set_backend", "stream3d", "collide_mrt3d"})


def test_lattice_adapter_is_cold_path_and_matches_existing_d3q19_constants() -> None:
    constants = build_torch_lattice_constants(D3Q19, DeviceSpec("cpu", "float32"))
    assert constants["directions"].device.type == "cpu"
    assert constants["directions"].dtype is torch.int64
    assert constants["weights"].dtype is torch.float32
    assert constants["opposite"].dtype is torch.int64
    assert torch.equal(constants["directions"], C)
    assert torch.equal(constants["weights"], W)
    assert torch.equal(constants["opposite"], OPPOSITE)


@pytest.mark.parametrize("descriptor, spec", [(D3Q27, DeviceSpec("cpu", "float32")), (D3Q19, DeviceSpec("cuda", "float32")), (D3Q19, DeviceSpec("cpu", "float64"))])
def test_lattice_adapter_rejects_everything_outside_r1(descriptor: object, spec: DeviceSpec) -> None:
    with pytest.raises(ValueError):
        build_torch_lattice_constants(descriptor, spec)  # type: ignore[arg-type]
