"""Contracts for the R1 cold-path backend boundary."""
from __future__ import annotations

import ast
import inspect

import pytest

from tensorlbm.backends.contracts import (
    BackendCapabilities,
    BackendId,
    BackendSupport,
    DeviceSpec,
)


def test_contract_module_is_framework_free_by_ast() -> None:
    module = ast.parse(inspect.getsource(__import__("tensorlbm.backends.contracts", fromlist=["*"])))
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(module)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".")[0]
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imported_roots.isdisjoint({"torch", "paddle", "mindspore"})


@pytest.mark.parametrize("value", ["", True, 1, None])
def test_device_spec_rejects_empty_or_non_string_values(value: object) -> None:
    with pytest.raises(ValueError):
        DeviceSpec(device=value, dtype_name="float32")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DeviceSpec(device="cpu", dtype_name=value)  # type: ignore[arg-type]


def test_paddle_and_mindspore_are_explicitly_not_supported() -> None:
    for backend_id in (BackendId.PADDLE, BackendId.MINDSPORE):
        capabilities = BackendCapabilities(
            backend_id=backend_id,
            support=BackendSupport.NOT_SUPPORTED,
            supported_devices=(),
            supported_dtypes=(),
            notes="not implemented in R1",
        )
        assert capabilities.support is BackendSupport.NOT_SUPPORTED


@pytest.mark.parametrize("backend_id", [BackendId.PADDLE, BackendId.MINDSPORE])
def test_paddle_and_mindspore_cannot_claim_support(backend_id: BackendId) -> None:
    with pytest.raises(ValueError, match="only Torch"):
        BackendCapabilities(
            backend_id=backend_id,
            support=BackendSupport.SUPPORTED,
            supported_devices=("cpu",),
            supported_dtypes=("float32",),
            notes="invalid",
        )


def test_torch_r1_capability_is_supported() -> None:
    capabilities = BackendCapabilities(
        backend_id=BackendId.TORCH,
        support=BackendSupport.SUPPORTED,
        supported_devices=("cpu",),
        supported_dtypes=("float32",),
        notes="R1 direct D3Q19 MRT plan",
    )
    assert capabilities.backend_id is BackendId.TORCH
    assert capabilities.support is BackendSupport.SUPPORTED
