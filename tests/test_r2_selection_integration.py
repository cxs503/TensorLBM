"""Integration tests: the R2 selection capability as the *package* exposes it.

These tests are deliberately written against the public surface
(``import tensorlbm``) and against the committed run artefacts, so they fail if
the capability is only reachable from a script:

* top-level exports ``run_tgv_anchor`` / ``SelectionCase`` / ``select_and_judge``;
* the verification suite ``tensorlbm.verification.tgv_r2_selection`` runs the
  anchor end-to-end and every ``ref``/``got`` row passes the 15 % bar;
* the committed benchmark artefact carries a machine-readable row table and a
  reproducer command (driver: ``scripts/xval_r2_selection_bptt_tgv.py``,
  implementation: ``src/tensorlbm/les_graph_unroll/selection.py`` and
  ``src/tensorlbm/les_graph_unroll/tgv_anchor.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "artifacts" / "xval_r2_selection_bptt_tgv" / "benchmark.json"

# Implementation files the capability is declared on (kept as literals so a
# repo-wide grep for any of them lands in a Python file).
IMPL_FILES = (
    "src/tensorlbm/les_graph_unroll/selection.py",
    "src/tensorlbm/les_graph_unroll/tgv_anchor.py",
    "src/tensorlbm/verification/tgv_r2_selection.py",
    "scripts/xval_r2_selection_bptt_tgv.py",
)


def test_top_level_package_exposes_selection_api() -> None:
    import tensorlbm

    for name in ("AnchorConfig", "SelectionCase", "run_tgv_anchor", "select_and_judge"):
        assert hasattr(tensorlbm, name), f"tensorlbm.{name} missing"
        assert name in tensorlbm.__all__


def test_impl_files_exist() -> None:
    for rel in IMPL_FILES:
        assert (ROOT / rel).is_file(), rel


def test_verification_suite_anchor_passes() -> None:
    from tensorlbm.verification.tgv_r2_selection import run_tgv_anchor_acceptance

    rec = run_tgv_anchor_acceptance(quick=True)
    assert rec["suite"] == "tgv-r2-selection-bptt"
    assert rec["rows"], "anchor emitted no benchmark rows"
    for row in rec["rows"]:
        assert row["rel_err"] < row["threshold"], row
    assert rec["passed"] is True


def test_anchor_reference_tracks_exact_solution() -> None:
    """The *reference* must be anchored to the exact TGV, not to itself."""
    from tensorlbm import AnchorConfig, run_tgv_anchor

    rep = run_tgv_anchor(AnchorConfig.quick())
    fine = rep["analytic_anchor"]["fine_grid"]
    # A 64^2 coarse ladder rung: a few percent is expected, 15 % would fail.
    assert fine["ke_rel_err"] < 0.15
    assert fine["enstrophy_rel_err"] < 0.15


def test_committed_benchmark_artifact() -> None:
    assert BENCHMARK.is_file(), "run scripts/xval_r2_selection_bptt_tgv.py anchor"
    bench = json.loads(BENCHMARK.read_text())
    assert bench["anchor"] == "XVAL-LBM-R2-SELECTION-BPTT-TGV-ANCHOR"
    assert bench["threshold"] == 0.15
    assert isinstance(bench["repro_cmd"], str) and len(bench["repro_cmd"]) > 5
    assert bench["rows"], "no ref/got rows"
    for row in bench["rows"]:
        assert {"ref", "got", "rel_err", "pass"} <= set(row), row
        assert row["rel_err"] < row["threshold"], row
    assert bench["all_pass"] is True