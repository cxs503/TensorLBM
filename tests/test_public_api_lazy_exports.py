"""Invariants for the lazy (:pep:`562`) public API in ``tensorlbm/__init__.py``.

The package exports 786 names drawn from ~150 submodules.  They are resolved
on demand by ``__getattr__`` instead of being imported eagerly, which is only
safe while three things hold, each locked by a test below:

1. every ``__all__`` entry actually resolves, to the object the eager import
   block would have bound (``_LAZY_ATTRS`` agrees with the ``TYPE_CHECKING``
   statements that mypy and IDEs read);
2. a bare ``import tensorlbm`` stays lazy — no accidental eager edge creeps
   back in via a module-level import;
3. the escape hatches behave: unknown names raise ``AttributeError``,
   submodules stay reachable as attributes, and ``dir()`` stays complete
   without importing anything.

Running the whole surface also smoke-tests that all ~150 submodules import
cleanly, which the eager ``__init__`` used to give for free.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import tensorlbm

_INIT = Path(tensorlbm.__file__)


def _type_checking_imports() -> dict[str, tuple[str, str]]:
    """``{exported name: (submodule, attribute)}`` per the TYPE_CHECKING block."""
    tree = ast.parse(_INIT.read_text())
    block = next(
        node
        for node in tree.body
        if isinstance(node, ast.If) and getattr(node.test, "id", "") == "TYPE_CHECKING"
    )
    return {
        alias.asname or alias.name: (stmt.module or "", alias.name)
        for stmt in block.body
        if isinstance(stmt, ast.ImportFrom)
        for alias in stmt.names
    }


def test_lazy_table_matches_type_checking_block() -> None:
    """The runtime table and the static-analysis block must not drift apart.

    mypy, IDEs and ``__getattr__`` would otherwise disagree about where a name
    lives — the failure mode is a symbol that type-checks but raises at
    runtime (or vice versa), which no other test would catch.
    """
    assert tensorlbm._LAZY_ATTRS == _type_checking_imports()


def test_all_is_fully_resolvable() -> None:
    """Every advertised export resolves, and nothing advertised is missing."""
    unresolved = [name for name in tensorlbm.__all__ if not hasattr(tensorlbm, name)]
    assert unresolved == []

    covered = set(tensorlbm._LAZY_ATTRS) | {"__version__"}
    assert set(tensorlbm.__all__) - covered == set()


def test_exports_resolve_to_their_declared_submodule() -> None:
    """``tensorlbm.X`` is the object ``from .sub import X`` would have bound."""
    mismatched: list[tuple[str, str, str]] = []
    for name, (submodule, attr) in tensorlbm._LAZY_ATTRS.items():
        module = __import__(f"tensorlbm.{submodule}", fromlist=[attr])
        if getattr(tensorlbm, name) is not getattr(module, attr):
            mismatched.append((name, submodule, attr))
    assert mismatched == []


def test_unknown_attribute_raises_attribute_error() -> None:
    """``__getattr__`` must not turn typos into ``ImportError`` or hangs."""
    with pytest.raises(AttributeError, match="no attribute 'definitely_not_exported'"):
        tensorlbm.definitely_not_exported


def test_submodules_remain_reachable_as_attributes() -> None:
    """Attribute access to submodules survives, including transitive ones.

    Under the eager ``__init__`` these were bound as a side effect of
    ``from .x import y`` (``d2q9``) or of some other module importing them
    (``lattice``, ``core``).  Both kinds must keep working.
    """
    for name in ("d2q9", "d3q19", "solver", "lattice", "core"):
        assert getattr(tensorlbm, name).__name__ == f"tensorlbm.{name}"


def test_dir_is_complete_and_does_not_import() -> None:
    """``dir()`` lists the full surface while staying side-effect free."""
    listed = set(dir(tensorlbm))
    assert set(tensorlbm.__all__) <= listed
    assert set(tensorlbm._LAZY_ATTRS) <= listed

    probe = "import sys, tensorlbm; dir(tensorlbm); print(len([m for m in sys.modules if m.startswith('tensorlbm')]))"
    loaded = int(
        subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        ).stdout
    )
    assert loaded <= 4, f"dir() imported submodules: {loaded} loaded"


def test_bare_import_stays_lazy() -> None:
    """A bare import must not drag the library in.

    This is the whole point of the lazy layer, and the easiest thing to
    regress: one module-level ``from .heavy import thing`` added to
    ``__init__.py`` silently restores the ~190-submodule eager import.
    """
    probe = (
        "import sys, tensorlbm; print(len([m for m in sys.modules if m.startswith('tensorlbm')]))"
    )
    loaded = int(
        subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        ).stdout
    )
    assert loaded <= 4, f"import tensorlbm eagerly loaded {loaded} submodules"


@pytest.mark.slow
def test_every_export_imports_cleanly() -> None:
    """Touch the whole surface: all ~150 submodules must import without error.

    The eager ``__init__`` gave this coverage implicitly; with lazy imports a
    broken submodule would otherwise stay invisible until a user hit it.
    """
    failures: list[tuple[str, str]] = []
    for name in tensorlbm.__all__:
        try:
            getattr(tensorlbm, name)
        except Exception as exc:  # noqa: BLE001 - reporting every failure at once
            failures.append((name, f"{type(exc).__name__}: {exc}"))
    assert failures == []
