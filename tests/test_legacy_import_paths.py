"""Real-runtime compatibility contracts for the boundary/turbulence extraction."""

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src" / "tensorlbm"
RELOCATIONS = [
    (path.stem, f"{package}.{path.stem}")
    for package in ("boundary", "turbulence")
    for path in sorted((ROOT / package).glob("*.py"))
    if path.stem not in ("__init__", package)
]


@pytest.mark.parametrize(("old", "new"), RELOCATIONS)
def test_legacy_module_is_canonical_module(old, new):
    # Fresh interpreters expose import-order bugs hidden by a warmed sys.modules.
    for order in ((old, new), (new, old)):
        script = f"""
import importlib
import tensorlbm
first = importlib.import_module('tensorlbm.{order[0]}')
second = importlib.import_module('tensorlbm.{order[1]}')
assert first is second
assert getattr(tensorlbm, '{old}') is first
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_turbulence_symbols_and_private_helpers():
    package = importlib.import_module("tensorlbm.turbulence")
    implementation = importlib.import_module("tensorlbm.turbulence.turbulence")
    assert package.__all__ == implementation.__all__
    for name in implementation.__all__ + ["_nu_t_to_tau_eff", "_get_d3q19_mrt_matrices"]:
        assert getattr(package, name) is getattr(implementation, name)
    # Exact syntax used by existing consumers, including star imports.
    namespace = {}
    exec("from tensorlbm.turbulence import *", namespace)
    assert namespace["collide_smagorinsky_bgk"] is implementation.collide_smagorinsky_bgk


def test_turbulence_remains_lazy():
    script = "import sys; import tensorlbm.turbulence; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True)


def test_alias_inventory_covers_all_moved_modules():
    for old, new in RELOCATIONS:
        assert (ROOT / f"{old}.py").is_file(), new
    # Keep the lazy package's static star-import list synchronized.
    tree = ast.parse((ROOT / "turbulence" / "__init__.py").read_text())
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "__getattr__" for node in tree.body
    )
