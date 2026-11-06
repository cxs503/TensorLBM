"""Mirror-contract tests across the three backend families.

:mod:`tensorlbm.backends` documents a hard invariant that no test enforced:
*"New functions added to one backend must be added to all three."*  Paddle and
MindSpore are not installed in this environment, so instead of re-testing their
numerics we test the surface that makes them drop-in interchangeable:

* identical public op surface (names + positional/keyword shape + defaults);
* each module declares the ``BACKEND_NAME`` used by ``get_ops`` callers;
* every op that touches the framework raises a *diagnosable* ``ImportError``
  when the framework is absent (negative path, forced through ``sys.modules``
  so it also holds on hosts where the framework *is* installed).
"""

from __future__ import annotations

import inspect
import math
import sys
from types import ModuleType

import numpy
import pytest

from tensorlbm.backends import (
    mindspore_backend,
    paddle_backend,
    torch_5op_step,
    torch_backend,
    triton_backend,
)

FAMILIES: tuple[tuple[str, ModuleType], ...] = (
    ("paddle", paddle_backend),
    ("mindspore", mindspore_backend),
)

#: Only the R1 cold-path facade helpers are torch-exclusive by design.
TORCH_ONLY = frozenset({"build_torch_lattice_constants"})
#: MindSpore needs a hand-written gradient step (``value_and_grad``) instead of
#: the autograd-backed ``train_step`` of the other two families.
MINDSPORE_ONLY = frozenset({"ms_train_step"})


def _public_functions(module: ModuleType) -> dict[str, object]:
    return {
        name: func
        for name, func in vars(module).items()
        if not name.startswith("_")
        and inspect.isfunction(func)
        and func.__module__ == module.__name__
    }


def _call_shape(func: object) -> list[tuple[str, object, object]]:
    """Parameter name/kind/default triple per parameter (annotations ignored)."""
    signature = inspect.signature(func)  # type: ignore[arg-type]
    return [(param.name, param.kind, param.default) for param in signature.parameters.values()]


def test_torch_defines_the_reference_op_surface() -> None:
    torch_ops = _public_functions(torch_backend)
    assert len(torch_ops) >= 60
    for required in ("zeros", "tensor", "build_flow_transformer", "train_step", "to_numpy"):
        assert required in torch_ops


@pytest.mark.parametrize(("family", "module"), FAMILIES)
def test_family_module_mirrors_the_torch_op_surface(family: str, module: ModuleType) -> None:
    torch_names = set(_public_functions(torch_backend))
    family_names = set(_public_functions(module))
    missing = (torch_names - TORCH_ONLY) - family_names
    assert missing == set(), f"{family} backend is missing torch ops: {sorted(missing)}"
    extra = family_names - (torch_names - TORCH_ONLY)
    expected_extra = MINDSPORE_ONLY if family == "mindspore" else set()
    assert extra == set(expected_extra), f"{family} backend has undocumented ops: {sorted(extra)}"


@pytest.mark.parametrize(("family", "module"), FAMILIES)
def test_family_call_signatures_match_torch(family: str, module: ModuleType) -> None:
    torch_ops = _public_functions(torch_backend)
    family_ops = _public_functions(module)
    mismatches = {
        name: (_call_shape(torch_ops[name]), _call_shape(family_ops[name]))
        for name in sorted(set(torch_ops) - TORCH_ONLY)
        if _call_shape(torch_ops[name]) != _call_shape(family_ops[name])
    }
    assert mismatches == {}, f"{family} signature drift: {mismatches}"


@pytest.mark.parametrize(("family", "module"), FAMILIES)
def test_family_declares_backend_name(family: str, module: ModuleType) -> None:
    assert module.BACKEND_NAME == family


def test_torch_backend_declares_backend_name() -> None:
    assert torch_backend.BACKEND_NAME == "torch"


@pytest.mark.parametrize(
    ("framework", "module", "hint"),
    [
        ("paddle", paddle_backend, "pip install paddlepaddle"),
        ("mindspore", mindspore_backend, "pip install mindspore"),
    ],
)
@pytest.mark.parametrize(
    "entry_point", ["zeros", "tensor", "from_numpy", "rand", "build_flow_transformer"]
)
def test_missing_framework_raises_actionable_importerror(
    monkeypatch: pytest.MonkeyPatch,
    framework: str,
    module: ModuleType,
    hint: str,
    entry_point: str,
) -> None:
    """Negative path: framework absent -> ImportError naming the install command."""
    monkeypatch.setitem(sys.modules, framework, None)
    func = getattr(module, entry_point)
    args: tuple[object, ...]
    if entry_point == "tensor":
        args = ([[1.0, 2.0]],)
    elif entry_point == "from_numpy":
        args = (numpy.zeros((2, 2)),)
    elif entry_point == "build_flow_transformer":
        args = (4, 8, 2, 1, 16, 0.0, 4)
    else:  # zeros / rand take a shape
        args = ([2, 2],)
    with pytest.raises(ImportError) as excinfo:
        func(*args)
    assert hint in str(excinfo.value)


def test_pure_math_helpers_do_not_require_the_framework(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pi``/``_non-``tensor helpers must stay import-and-call cheap."""
    monkeypatch.setitem(sys.modules, "paddle", None)
    monkeypatch.setitem(sys.modules, "mindspore", None)
    assert paddle_backend.pi() == math.pi
    assert mindspore_backend.pi() == math.pi
    with pytest.raises(ImportError, match="pip install paddlepaddle"):
        paddle_backend.float32_dtype()
    with pytest.raises(ImportError, match="pip install mindspore"):
        mindspore_backend.float32_dtype()


def test_step_impl_modules_declare_their_public_entry_points() -> None:
    """The step-impl layer (torch_5op / triton) must expose one documented API."""
    assert torch_5op_step.__all__ == ["step_torch_5op"]
    assert set(triton_backend.__all__) == {"TritonStepState", "triton_suboff_step", "is_available"}
    for name in torch_5op_step.__all__:
        assert callable(getattr(torch_5op_step, name))
    for name in triton_backend.__all__:
        assert getattr(triton_backend, name) is not None
