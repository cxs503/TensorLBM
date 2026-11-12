"""Independent control-arm check for CASE-LBM-RB-ONSET-RA1708.

``benchmarks/case_pool/rayleigh_benard_onset/independent_check_control.py``
re-derives the continuum marginal Rayleigh number by time marching the
single-mode linearised NS-Boussinesq equations (RK4 + second-order FD) and
bisecting on the sign of ``sigma(Ra)`` -- no eigen-solve, no shared code with
``tensorlbm.rb_continuum_stability`` (Route C of ``run_case.py``).

Two layers:

* **logic** (seconds): the free-slip wall model has the closed-form answer
  ``Ra_c = 27 pi^4/4 = 657.511``; the operator must report ``sigma < 0`` below it
  and ``sigma > 0`` above it.  This is what licenses using the same operator for
  the rigid-wall arm.
* **artifact**: the committed run JSON
  (``runs/case_rayleigh_benard_onset/independent_control.json``) is re-graded
  against the registered criteria; skipped when the artifact is absent.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmarks" / "case_pool" / "rayleigh_benard_onset" / "independent_check_control.py"
RUN_JSONS = [REPO / "benchmarks" / "case_pool" / "rayleigh_benard_onset" / "artifacts" / "independent_control.json",
             REPO / "runs" / "case_rayleigh_benard_onset" / "independent_control.json"]
RUN_JSON = next((p for p in RUN_JSONS if p.exists()), RUN_JSONS[0])


def _load():
    spec = importlib.util.spec_from_file_location("rb_indep_ctrl", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


def test_free_slip_wall_model_reproduces_the_relaxation_threshold(mod):
    """Closed-form calibration: sigma changes sign at Ra_c = 27 pi^4/4 (free-slip)."""
    ra_c = mod.RA_FREEFREE
    below = mod.growth_rate(0.80 * ra_c, mod.A_FREEFREE, ny=40, walls="free",
                            steps=8000, dt=2e-4)
    above = mod.growth_rate(1.20 * ra_c, mod.A_FREEFREE, ny=40, walls="free",
                            steps=8000, dt=2e-4)
    assert below["sigma"] < 0.0 < above["sigma"]
    assert below["r2"] > 0.99 and above["r2"] > 0.99


def test_rigid_walls_are_stabilising_relative_to_free_slip(mod):
    """Rigid walls must raise the threshold: at Ra=900 free-slip is unstable while
    rigid walls are still stable (Ra_c: 657.5 free-slip vs 1707.8 rigid)."""
    ra = 900.0
    free = mod.growth_rate(ra, mod.A_FREEFREE, ny=40, walls="free", steps=8000, dt=2e-4)
    rigid = mod.growth_rate(ra, mod.A_RIGID, ny=40, walls="rigid", steps=8000, dt=2e-4)
    assert free["sigma"] > 0.0 > rigid["sigma"]


def test_bisection_reports_an_invalid_bracket_instead_of_guessing(mod):
    """A bracket that does not straddle sigma=0 must be reported, not silently used."""
    out = mod.bisect(mod.A_RIGID, "rigid", 100.0, 200.0, ny=16, steps=500, dt=2e-4)
    assert out["valid"] is False
    assert "reason" in out


def test_independent_control_artifact_meets_the_registered_criteria():
    if not RUN_JSON.exists():
        pytest.skip("no independent-control artifact committed yet")
    data = json.loads(RUN_JSON.read_text())
    ff = data["calibration_free_slip"]
    rr = data["rigid_rigid"]
    # calibration is the licence to use the rigid number at all
    assert ff["valid"] is True
    assert abs(ff["err_pct"]) < 1.0, ff["err_pct"]
    # the rigid arm must land within 1.5 % of Chandrasekhar's 1707.7597
    assert rr["valid"] is True
    assert abs(rr["err_pct"]) < 1.5, rr["err_pct"]
    # the bracket really straddles zero and every fitted sigma is a real fit
    assert rr["sigma_bracket"][0] < 0.0 < rr["sigma_bracket"][1]
    assert all(h["r2"] > 0.99 for h in rr["history"])
    # a claimed grid sensitivity must be reported, not assumed
    assert len(data["grid_sensitivity"]) == 3
    assert data["all_checks_pass"] is True
