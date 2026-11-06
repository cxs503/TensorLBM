"""Dispatch-layer tests for :mod:`tensorlbm.backends`.

``tests/backends/test_backend_contracts.py`` covers the framework-free R1
contracts and ``tests/backends/test_torch_backend.py`` the R1 cold-path
facade, but the registry itself (``set_backend`` / ``get_ops`` /
``using_backend`` and the import-time ``TENSORLBM_BACKEND`` default) had no
coverage under ``tests/backends/``.  These tests fill that gap and exercise
the Paddle/MindSpore branches of ``get_ops`` *without* requiring either
framework to be installed (both backend modules import lazily).
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

import tensorlbm.backends as B

KNOWN_BACKENDS = ("torch", "paddle", "mindspore")


@pytest.fixture(autouse=True)
def _restore_active_backend() -> None:
    """Leave the process-wide backend exactly as we found it."""
    previous = B.get_backend()
    yield
    B.set_backend(previous)


# ---------------------------------------------------------------------------
# set_backend / get_backend (positive + negative)
# ---------------------------------------------------------------------------


def test_set_backend_accepts_every_known_backend_name() -> None:
    for name in KNOWN_BACKENDS:
        B.set_backend(name)
        assert B.get_backend() == name


@pytest.mark.parametrize("name", ["tensorflow", "", "TORCH", " torch", "triton"])
def test_set_backend_rejects_unknown_names_and_lists_valid_choices(name: str) -> None:
    previous = B.get_backend()
    with pytest.raises(ValueError) as excinfo:
        B.set_backend(name)
    message = str(excinfo.value)
    assert "Unknown backend" in message
    assert repr(name) in message
    # The error text must enumerate the accepted choices so users can recover.
    assert "['mindspore', 'paddle', 'torch']" in message
    # A rejected switch must not mutate global state.
    assert B.get_backend() == previous


def test_set_backend_rejects_non_string_values() -> None:
    for value in (None, 3, True):  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown backend"):
            B.set_backend(value)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# get_ops: module selection + cache (positive)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", KNOWN_BACKENDS)
def test_get_ops_returns_matching_family_module(name: str) -> None:
    B.set_backend(name)
    ops = B.get_ops()
    assert ops.__name__ == f"tensorlbm.backends.{name}_backend"
    assert ops.BACKEND_NAME == name


def test_get_ops_caches_one_distinct_module_per_backend() -> None:
    resolved = {}
    for name in KNOWN_BACKENDS:
        B.set_backend(name)
        resolved[name] = B.get_ops()
        assert B.get_ops() is resolved[name], "repeated get_ops must hit the cache"
    assert len({id(module) for module in resolved.values()}) == len(KNOWN_BACKENDS)


def test_paddle_and_mindspore_modules_import_without_the_framework() -> None:
    """The family modules must be importable on a torch-only host.

    Only the *call* may raise (see ``test_backend_family_parity.py``); if the
    module body required Paddle/MindSpore, ``get_ops`` would explode at switch
    time and ``using_backend`` could never restore the previous backend.
    """
    for name in ("paddle", "mindspore"):
        B.set_backend(name)
        ops = B.get_ops()
        assert callable(ops.zeros)
        assert callable(ops.train_step)


# ---------------------------------------------------------------------------
# get_ops: unregistered backend (negative)
# ---------------------------------------------------------------------------


def test_get_ops_rejects_active_backend_without_module(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``set_backend`` validates, but the import-time env var does not, so a
    # bogus ``TENSORLBM_BACKEND`` can leave an unresolvable name in place.
    monkeypatch.setattr(B, "_current_backend", "tensorflow")
    with pytest.raises(ValueError, match="No ops module for backend 'tensorflow'"):
        B.get_ops()


# ---------------------------------------------------------------------------
# using_backend context manager
# ---------------------------------------------------------------------------


def test_using_backend_switches_and_restores() -> None:
    B.set_backend("torch")
    with B.using_backend("mindspore") as ops:
        assert B.get_backend() == "mindspore"
        assert ops.BACKEND_NAME == "mindspore"
    assert B.get_backend() == "torch"


def test_using_backend_with_none_keeps_current_backend() -> None:
    B.set_backend("paddle")
    with B.using_backend(None) as ops:
        assert B.get_backend() == "paddle"
        assert ops is B.get_ops()
    assert B.get_backend() == "paddle"


def test_using_backend_restores_previous_backend_after_body_raises() -> None:
    B.set_backend("torch")
    with pytest.raises(RuntimeError, match="boom"):
        with B.using_backend("paddle"):
            assert B.get_backend() == "paddle"
            raise RuntimeError("boom")
    assert B.get_backend() == "torch"


def test_using_backend_propagates_invalid_name_without_state_change() -> None:
    B.set_backend("torch")
    with pytest.raises(ValueError, match="Unknown backend"):
        with B.using_backend("tensorflow"):
            pytest.fail("invalid backend must not enter the context body")
    assert B.get_backend() == "torch"


# ---------------------------------------------------------------------------
# Import-time TENSORLBM_BACKEND default (subprocess: needs a fresh interpreter)
# ---------------------------------------------------------------------------

_PROBE = """
import tensorlbm.backends as B
print("selected=" + B.get_backend())
try:
    B.get_ops()
except ValueError as exc:
    print("get_ops_error=" + str(exc))
"""


def _probe(env_extra: dict[str, str]) -> str:
    env = {**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}
    env.pop("TENSORLBM_BACKEND", None)
    env.update(env_extra)
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    return completed.stdout.strip()


def test_default_backend_is_torch_without_env_var() -> None:
    assert _probe({}).splitlines()[0] == "selected=torch"


def test_env_var_selects_backend_at_import_time() -> None:
    assert _probe({"TENSORLBM_BACKEND": "paddle"}).splitlines()[0] == "selected=paddle"


def test_invalid_env_var_is_accepted_at_import_but_fails_at_get_ops() -> None:
    lines = _probe({"TENSORLBM_BACKEND": "tensorflow"}).splitlines()
    assert lines[0] == "selected=tensorflow"
    assert lines[1] == "get_ops_error=No ops module for backend 'tensorflow'"
