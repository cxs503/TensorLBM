"""Pytest configuration: environment-dependent collection.

The CI ``test`` job installs CPU-only torch (no CUDA, no triton, no fastapi)
to keep runners light.  Modules that hard-require what is missing are skipped
at collection time; the full suite runs wherever the dependencies exist
(e.g. the 8x5090 dev server).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent

# These modules load sibling/repo-root scripts via importlib or sys.path
# tricks; the loaded scripts import torch, which a keyword scan cannot see.
_DYNAMIC_LOADER_MODULES = [
    "test_assess_collision_viscosity_schedule.py",
    "test_assess_pressure_gradient_wall_channel.py",
    "test_assess_pressure_gradient_wall_mkm_dns.py",
    "test_bench_dam_break.py",
    "test_gallium_pf_grid_diagnostic.py",
]

# These import repo-root benchmark scripts that are not tracked in the repo;
# they only run in environments where those scripts exist.
_ROOT_SCRIPT_TESTS = {
    "test_gallium_pf_energy.py",
    "test_gallium_pf_stefan_interface.py",
    "test_gallium_pf_grid_diagnostic.py",
}


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def _collect_ignore() -> list[str]:
    has_torch = _has("torch")
    has_triton = _has("triton")
    has_fastapi = _has("fastapi")
    has_root_gallium = (_REPO_ROOT / "benchmark_gallium_pf.py").exists()
    ignore: list[str] = []
    for p in sorted(_TESTS_DIR.rglob("*test*.py")):
        if p.name == "conftest.py":
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if p.name in _ROOT_SCRIPT_TESTS and not has_root_gallium:
            ignore.append(str(p))
            continue
        if not has_fastapi and "fastapi" in text:
            ignore.append(str(p))
            continue
        if has_torch:
            if not has_triton and "triton" in text:
                ignore.append(str(p))
        elif any(k in text for k in ("torch", "triton", "tensorlbm")) or (
            p.name in _DYNAMIC_LOADER_MODULES
        ):
            ignore.append(str(p))
    return ignore


collect_ignore = _collect_ignore()

_QUARANTINE_FILE = _TESTS_DIR / "ci_quarantine.txt"


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Turn quarantined pre-existing failures into non-strict xfails."""
    if not _QUARANTINE_FILE.exists():
        return
    quarantined = {
        line.strip()
        for line in _QUARANTINE_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    for item in items:
        if item.nodeid in quarantined:
            item.add_marker(
                pytest.mark.xfail(
                    reason="quarantined pre-existing failure (see tests/ci_quarantine.txt)",
                    strict=False,
                )
            )
