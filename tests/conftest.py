"""Pytest configuration: environment-dependent collection.

The CI ``test`` job installs only ``pip install -e ".[dev]"``: torch from
PyPI plus the dev extras, with no GPU device.  fastapi and the rest of the
platform-app stack stay out of that set, so modules that hard-require what
is missing are skipped at collection time; the full suite runs wherever the
dependencies exist (e.g. the 8x5090 dev server).

Two facts from the 2026-09-05 CI-coverage audit
(``runs/ci_audit_20260905``): stock PyPI torch (>=2.14) depends on triton
on Linux, so CI *does* have triton and the triton branch below is inert
there — it only fires for CPU-index torch builds; and ``collect_ignore``
entries are invisible in pytest output, which is how PR #272's catalog
test went unrun for its whole life.  Ignored files are therefore reported
via ``pytest_terminal_summary`` so the gap stays visible in every log.
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
            _IGNORE_REASONS[str(p)] = "untracked repo-root benchmark_gallium_pf.py"
            ignore.append(str(p))
            continue
        if not has_fastapi and "fastapi" in text:
            _IGNORE_REASONS[str(p)] = "fastapi absent"
            ignore.append(str(p))
            continue
        if has_torch:
            if not has_triton and "triton" in text:
                _IGNORE_REASONS[str(p)] = "triton absent"
                ignore.append(str(p))
        elif any(k in text for k in ("torch", "triton", "tensorlbm")) or (
            p.name in _DYNAMIC_LOADER_MODULES
        ):
            _IGNORE_REASONS[str(p)] = "torch absent"
            ignore.append(str(p))
    return ignore


# Populated as a side effect of _collect_ignore() at import time, then read
# by pytest_terminal_summary() below.
_IGNORE_REASONS: dict[str, str] = {}

collect_ignore = _collect_ignore()


def pytest_terminal_summary(terminalreporter, exitstatus):  # noqa: ARG001
    """Report collection-ignored files so silent CI coverage gaps stay visible.

    ``collect_ignore`` entries never appear in pytest's skip counts: a file
    dropped there simply vanishes from the run, which is how PR #272's
    data-catalog test went unexecuted by CI for its entire life.  Echo the
    ignore list (grouped by reason) in the terminal summary, which is
    printed even under ``pytest -q`` as CI runs it.
    """
    terminalreporter.section("collection-ignore")
    if not _IGNORE_REASONS:
        terminalreporter.write_line("none (every test file under tests/ is collectable here)")
        return
    for reason in sorted(set(_IGNORE_REASONS.values())):
        names = sorted(Path(p).name for p, r in _IGNORE_REASONS.items() if r == reason)
        terminalreporter.write_line(f"({reason}): {len(names)} files: {', '.join(names)}")


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
