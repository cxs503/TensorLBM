"""Regression tests for the case-pool driver ``benchmarks/case_pool/run_pool.py``.

Both defects below were found while auditing m21672 (CASE-BACKSTEP-FVM-RE100,
LBM arm), whose L3 claim rests on ``run_pool.py`` discovering/executing the
case.  They are repo-level (they hit every pool case, not just that one):

1. the documented invocation ``PYTHONPATH=src python3 benchmarks/case_pool/
   run_pool.py --case <name>`` failed with ``ModuleNotFoundError: case_pool``
   because ``benchmarks/`` (the package root) was never put on ``sys.path``;
2. the driver always called ``mod.main(extra)``, so the three cases whose
   ``main()`` takes no argv (couette_2d, karman_re100, mcmp_laplace) raised
   ``TypeError: main() takes 0 positional arguments but 1 was given`` -- which
   made ``--all`` unusable.

The tests exercise the driver module directly (no case is actually stepped), so
they cost milliseconds and never touch ``artifacts/``.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys

from pathlib import Path



POOL = Path(__file__).resolve().parents[1] / "benchmarks" / "case_pool"
DRIVER = POOL / "run_pool.py"


def _load_driver():
    """Import the driver script as a module without a package context."""
    spec = importlib.util.spec_from_file_location("case_pool_driver_ut", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_driver_puts_case_pool_root_on_sys_path():
    driver = _load_driver()
    assert str(driver.POOL_ROOT) == str(POOL.parent)
    assert str(POOL.parent) in sys.path
    # the documented invocation therefore resolves the package
    mod = importlib.import_module("case_pool.backward_step_armaly.run_case")
    assert callable(mod.main)


def test_discover_lists_backward_step_case():
    driver = _load_driver()
    assert "backward_step_armaly" in driver.discover()


def test_run_case_forwards_argv_when_supported():
    driver = _load_driver()
    seen = []

    class _Mod:
        @staticmethod
        def main(argv=None):
            seen.append(argv)
            return 0

    driver.importlib.import_module = lambda name: _Mod
    assert driver.run_case("anycase", ["--smoke"]) == 0
    assert seen == [["--smoke"]]


def test_run_case_tolerates_zero_arg_main():
    driver = _load_driver()
    seen = []

    class _Mod:
        @staticmethod
        def main():
            seen.append("called")
            return 7

    driver.importlib.import_module = lambda name: _Mod
    # must not raise TypeError and must propagate the case's exit code
    assert driver.run_case("anycase", ["--smoke"]) == 7
    assert seen == ["called"]


def test_run_case_isolates_sys_argv_for_zero_arg_main():
    """A sys.argv-parsing case must not see the driver's own flags."""
    driver = _load_driver()
    seen = []
    outer = list(sys.argv)

    class _Mod:
        @staticmethod
        def main():
            seen.append(list(sys.argv))
            return 0

    driver.importlib.import_module = lambda name: _Mod
    sys.argv = ["run_pool.py", "--case", "anycase", "--smoke"]
    try:
        assert driver.run_case("anycase", ["--smoke"]) == 0
    finally:
        restored = sys.argv == ["run_pool.py", "--case", "anycase", "--smoke"]
        sys.argv = outer
    assert restored, "driver leaked its argv rewrite"
    assert seen == [["anycase.run_case"]], seen


def test_main_accepts_argv_signature_detection():
    driver = _load_driver()
    assert driver._main_accepts_argv(lambda argv=None: 0) is True
    assert driver._main_accepts_argv(lambda *a, **k: 0) is True
    assert driver._main_accepts_argv(lambda: 0) is False


def test_in_process_zero_arg_case_runs_through_driver():
    """End-to-end: a no-argv case package is importable and runnable."""
    driver = _load_driver()
    mods = []
    for k in ("couette_2d", "karman_re100", "mcmp_laplace"):
        m = importlib.import_module(f"case_pool.{k}.run_case")
        mods.append((k, m))
        assert driver._main_accepts_argv(m.main) is False
    assert len(mods) == 3