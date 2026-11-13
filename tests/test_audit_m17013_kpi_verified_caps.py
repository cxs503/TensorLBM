"""Board-audit invariants for the m17013 claim (anchor KPI-LBM-VERIFIED-CAPS).

This test file is the executable form of ``docs/audit_m17013/AUDIT_m17013.md``.
It does not re-audit TensorLBM physics; it pins the three facts the audit rests
on, so that any future change that would invalidate the verdict breaks a test
instead of silently rotting:

1. the shipped driver really does reproduce its shipped ``result.json``
   (the deliverable is not hand-edited);
2. the driver exercises no solver path (so the promotion to "verified
   capability" cannot be read as an L2/L3 integration);
3. the claimed free-surface physics is *not* covered — mutation A (break
   ``elevation``) is still reported as PASS.  When this test starts failing the
   driver has improved: re-audit and update the verdict instead of deleting it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CASE = Path("benchmarks/verified/nwt_stokes2")
DRIVER = CASE / "run.py"
SHIPPED = CASE / "result.json"

SOLVER_MODULES = ("tensorlbm.nwt.tank", "tensorlbm.nwt.wavemaker", "tensorlbm.solver",
                  "tensorlbm.d2q9", "tensorlbm.vof_free_surface")
SURFACE_CALLS = (".elevation(", ".potential(", ".kinematics(", ".surface_bc_residual(")


def _isolated_run(patch: str | None = None) -> dict:
    """Copy src/ + the driver to a temp dir, optionally mutate, run, read result."""
    tmp = Path(tempfile.mkdtemp(prefix="audit_m17013_"))
    shutil.copytree(REPO / "src", tmp / "src")
    (tmp / CASE).mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / DRIVER, tmp / DRIVER)
    if patch:
        f = tmp / "src/tensorlbm/nwt/stokes2.py"
        txt = f.read_text()
        assert patch in txt, "mutation target vanished — update the audit"
        f.write_text(txt.replace(patch, patch + "\n        return 12345.0  # mutation", 1))
    proc = subprocess.run([sys.executable, str(tmp / DRIVER)], cwd=tmp,
                          capture_output=True, text=True)
    out = tmp / SHIPPED
    return {"code": proc.returncode, "stdout": proc.stdout + proc.stderr,
            "result": json.loads(out.read_text()) if out.exists() else {}}


@pytest.fixture(scope="module")
def fresh():
    return _isolated_run()


def test_driver_reproduces_shipped_result(fresh):
    """The delivered result.json must be reproducible from the delivered code."""
    assert fresh["code"] == 0, fresh["stdout"]
    assert fresh["result"]["verdict"] == "PASS"
    assert fresh["result"]["gates_passed"] == fresh["result"]["gates_total"] == 12
    assert fresh["result"] == json.loads((REPO / SHIPPED).read_text()), \
        "shipped result.json is not what run.py produces"


def test_gate_count_is_call_count_not_capability_count(fresh):
    """12 gate calls collapse to 9 distinct gate names (dispersion repeated 4x)."""
    names = [c["gate"] for c in fresh["result"]["checks"]]
    assert len(names) == 12
    assert len(set(names)) == 9


def test_driver_exercises_no_solver_path():
    """L2/L3 cannot be claimed from this artefact: no solver import, no physics call site."""
    src = (REPO / DRIVER).read_text()
    imported = [m for m in SOLVER_MODULES if m in src.split('"""', 2)[-1]]
    assert imported == [], f"driver now touches solver modules {imported}: re-audit"
    assert [s for s in SURFACE_CALLS if s in src] == [], \
        "driver now calls free-surface physics: the coverage gap may be closed, re-audit"


def test_mutation_a_documents_coverage_gap():
    """Breaking the free surface (elevation -> 12345) is NOT detected by the gates."""
    mutated = _isolated_run("    def elevation(self, x, t: float):")
    assert mutated["result"]["verdict"] == "PASS", \
        "coverage gap closed — promotion may now be justified, re-audit m17013"
    assert mutated["result"]["gates_passed"] == mutated["result"]["gates_total"]


def test_control_mutation_is_detected():
    """Positive control: a covered quantity (a2) *is* refereed by the driver."""
    mutated = _isolated_run("    def second_harmonic_amplitude(self) -> float:")
    assert mutated["result"]["verdict"] == "FAIL"
    assert mutated["result"]["gates_passed"] < mutated["result"]["gates_total"]