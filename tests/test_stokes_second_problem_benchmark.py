"""B25 stokes_second_problem benchmark: harness regression guards.

Guards the three skeleton defects fixed on 2026-11-03 that made the case unable to
ever emit a verdict file, plus a physics sanity bound at a cheap grid (H=16).
Runtime ~10 s on CPU (18x8 lattice, ~4.5k steps) -- no GPU, no long sweep.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

RUN_PY = Path(__file__).resolve().parents[1] / "benchmarks/pending/stokes_second_problem/run.py"

pytestmark = pytest.mark.skipif(not RUN_PY.exists(), reason="B25 run.py not present")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("b25_stokes_second", RUN_PY)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def case(mod):
    # H=16 => delta=4 cells, T~1005 steps; cycles (3,4) => ~4.5k steps
    return mod.run_case(16, 0.65, 0.05, (3, 4), nx=8, compile_mode="eager")


def test_result_is_json_serializable(case, mod):
    """Defect 1: period_convergence returned numpy.bool_ -> json.dumps TypeError."""
    pc = mod.period_convergence(case, case["U0"])
    assert isinstance(pc["steady"], bool) and type(pc["steady"]) is bool
    json.dumps({"case": {k: case[k] for k in case if k != "per_step"}, "pc": pc})


def test_period_convergence_has_rows_with_cycles(mod, case):
    """Defect 2: scan printout indexed pc['H'] -> KeyError (no 'H' key by contract)."""
    pc = mod.period_convergence(case, case["U0"])
    assert "H" not in pc  # callers must pair with the case dict
    # rows carry round(phi, 6), not the exact float, by construction
    assert {r["phase"] for r in pc["rows"]} == {
        round(0.0, 6),
        round(math.pi / 2.0, 6),
        round(math.pi, 6),
    }
    for row in pc["rows"]:
        assert row["t_late"] > row["t_early"]


def test_compare_grids_similarity_collapse_is_finite(mod, case):
    """Defect 3: descending y_profile fed to np.interp -> garbage cross-grid diff.

    Same case compared against itself must collapse to ~0 once the abscissa is
    sorted, and the eta=y/delta block must be present and finite.
    """
    conv = mod.compare_grids(case, case, case["U0"])
    assert conv["converged"] is True and type(conv["converged"]) is bool
    for row in conv["rows"]:
        assert "similarity_collapse_max_diff_over_U0_pct" in row
        assert math.isfinite(row["similarity_collapse_max_diff_over_U0_pct"])
        # self-comparison: identical profiles -> exactly zero difference
        assert row["similarity_collapse_max_diff_over_U0_pct"] == pytest.approx(0.0, abs=1e-9)
        assert row["max_profile_diff_over_U0_pct"] == pytest.approx(0.0, abs=1e-9)
        # (H-0.5)/delta = (16-0.5)/4
        assert row["eta_max_compared"] == pytest.approx(3.875, abs=1e-9)
    json.dumps(conv)


def test_stokes_layer_physics_at_coarse_grid(case):
    """Coarse-grid sanity: finite, mass conserving, Stokes layer in the right place."""
    assert case["finite"] is True
    assert abs(case["mass_drift_pct"]) < 1.0
    late = max(s["cycle_k"] for s in case["per_step"])
    for s in case["per_step"]:
        # amplitude decays away from the oscillating wall (never grows inward)
        prof = s["u_profile"]  # wall-first (descending y)
        assert abs(prof[0]) < abs(prof[-1])
        if s["cycle_k"] == late:
            # Bounds are deliberately loose: delta = 4 cells is grossly
            # under-resolved for a Stokes layer (measured 15.8% at phi=pi/2 here,
            # vs 2.16% at delta = 10 cells and 1.01% at delta = 20 cells in the
            # full campaign).  This is a smoke guard, not an accuracy claim.
            assert s["max_rel_err_pct"] < 20.0, s
            assert s["max_abs_err_over_U0_pct"] < 10.0, s


REPORT_PY = Path(__file__).resolve().parents[1] / "scripts/stokes_second_report.py"
COMMITTED_DOC = (
    Path(__file__).resolve().parents[1] / "docs/stokes_second_problem_verification_20261103.md"
)


def _report_module():
    spec = importlib.util.spec_from_file_location("b25_report", REPORT_PY)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


def test_order_bullet_follows_data_not_stale_text():
    """Defect 4 (2026-11-06): the ⑤ order bullet was hand-edited to claim an H=160
    diagnostic result, so re-running the generator silently reverted the claim.
    The bullet must be derived from the measured per-grid errors."""
    m = _report_module()
    with_diag = m.order_gate_bullet({40: 2.16, 80: 1.012, 160: 1.452})
    assert "1.452%" in with_diag and "1.012%" in with_diag
    assert "H=160" in with_diag
    without = m.order_gate_bullet({40: 2.16, 80: 1.012})
    assert "诊断档未跑" in without
    assert "1.452" not in without and "本轮补跑" not in without


def test_report_regeneration_reproduces_committed_doc(tmp_path):
    """The generator is the only source of the doc: regenerating it in an isolated
    cwd (case data symlinked) must reproduce the committed file byte for byte."""
    if not REPORT_PY.exists() or not COMMITTED_DOC.exists():
        pytest.skip("report generator or committed doc absent")
    (tmp_path / "benchmarks/pending").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "benchmarks/pending/stokes_second_problem").symlink_to(
        RUN_PY.parent, target_is_directory=True
    )
    proc = subprocess.run(
        [sys.executable, str(REPORT_PY)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    generated = (tmp_path / "docs/stokes_second_problem_verification_20261103.md").read_text()
    assert generated == COMMITTED_DOC.read_text()
