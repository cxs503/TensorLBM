"""Invariant + regression tests for XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY.

Three things are asserted here:

1. the *reference* (generic 4x4 two-layer Stokes solver) satisfies its own
   boundary/continuity/stress invariants and agrees with the library's existing
   closed form ``tensorlbm.porous_media._two_phase_poiseuille_analytical`` at
   that function's legacy convention — i.e. the verifier is itself verified;
2. the two-phase-only model passes the owner band on a small arm that is run
   inside the test (so the headline claim cannot rot into a stale json);
3. the report tables are rendered from ``results.json`` (zero hand-typed
   numbers): re-rendering reproduces the committed ``REPORT.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
XVAL_DIR = REPO_ROOT / "xval" / "XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(XVAL_DIR))

import run_xval  # noqa: E402
from two_phase_only import (  # noqa: E402
    analytic_profile,
    mix_viscosity,
    phase_indicator,
    run_two_phase_only,
    two_layer_analytic,
)
from tensorlbm.porous_media import _two_phase_poiseuille_analytical  # noqa: E402


# ---------------------------------------------------------------------------
# 1. The reference
# ---------------------------------------------------------------------------


def test_analytic_invariants_machine_precision():
    """Walls, velocity continuity and stress continuity are closed exactly."""
    ny = 64
    sol = two_layer_analytic(ny, 0.5 * (ny - 1), 4e-6, 0.5, 0.125)
    res = sol.residuals()
    for key, val in res.items():
        assert val < 1e-12, f"{key} residual {val:.3e} is not machine precision"


def test_analytic_equals_single_phase_parabola_at_equal_viscosity():
    """nu_lo == nu_hi must collapse to u = G/(2 nu) y (H - y)."""
    ny = 48
    G, nu = 4e-6, 0.5
    sol = two_layer_analytic(ny, 0.5 * (ny - 1), G, nu, nu)
    y = np.arange(ny, dtype=np.float64)
    ref = G / (2.0 * nu) * y * (ny - 1 - y)
    assert np.max(np.abs(analytic_profile(sol, ny) - ref)) < 1e-15


def test_analytic_matches_library_closed_form():
    """Cross-check of the library analytic (legacy convention: walls at j=0/ny-1).

    This is a verification-of-the-verifier check: the generic linear-system
    solver in this study must reproduce the library's hand-derived profile, so
    the study neither bypasses nor silently "fixes" the library.
    """
    ny, G = 64, 4e-6
    nu_lo, nu_hi = 0.5, 0.125
    half = ny // 2
    legacy = np.asarray(_two_phase_poiseuille_analytical(ny, half, G, nu_lo, nu_hi, nu_lo / nu_hi))
    ours = analytic_profile(two_layer_analytic(ny, float(half), G, nu_lo, nu_hi), ny)
    assert np.max(np.abs(legacy - ours)) < 1e-12


def test_viscosity_mixing_rules_bracket_the_layer_values():
    """All three rules give the layer values at C=0/1 and are ordered at C=1/2."""
    C = torch.tensor([[1.0], [0.5], [0.0]])
    lo, hi = 0.5, 0.125
    for rule in ("linear", "geometric", "harmonic"):
        nu = mix_viscosity(C, lo, hi, rule)
        assert abs(float(nu[0, 0]) - lo) < 1e-12
        assert abs(float(nu[2, 0]) - hi) < 1e-12
    half_vals = {r: float(mix_viscosity(C, lo, hi, r)[1, 0]) for r in ("linear", "geometric", "harmonic")}
    assert half_vals["harmonic"] < half_vals["geometric"] < half_vals["linear"]


def test_phase_indicator_is_monotone_and_centred():
    C = phase_indicator(64, 4, 31.5, 2.0)
    prof = C.mean(dim=1).numpy()
    assert np.all(np.diff(prof) <= 1e-6), "C must decrease with y"
    assert abs(float(prof[0]) - 1.0) < 1e-3 and abs(float(prof[-1])) < 1e-3


# ---------------------------------------------------------------------------
# 2. The model claim, re-run (small arm)
# ---------------------------------------------------------------------------


def test_two_phase_only_reproduces_layered_poiseuille_in_band():
    """Owner band (15%) on a small arm executed inside the test."""
    rec = run_two_phase_only(
        48, nx=4, G=4e-6, nu_lo=0.5, nu_hi=0.25, width=2.0, rule="harmonic",
        max_steps=20000, stop_tol=1e-6, check_every=500,
    )
    assert rec["converged"], "the small arm must reach its steady state inside 20000 steps"
    assert rec["l2_rel"] < 0.15, rec["l2_rel"]
    assert rec["max_rel_peak"] < 0.15, rec["max_rel_peak"]
    assert rec["Q_lo_rel"] < 0.15 and rec["Q_hi_rel"] < 0.15
    # The frozen-interface assumption is only valid if the normal velocity vanishes.
    assert rec["uy_spurious_rel"] < 1e-2
    # Mass is conserved by the bounce-back walls up to the float32 accumulation
    # level (the predeclared K6 threshold of 1e-6 is *not* met and is reported as
    # a failed check; this is only a sanity bound, not a restatement of K6).
    assert rec["mass_drift_rel"] < 1e-3


# ---------------------------------------------------------------------------
# 3. Report provenance: numbers cannot be hand-typed
# ---------------------------------------------------------------------------


RESULTS = XVAL_DIR / "results.json"
REPORT = XVAL_DIR / "REPORT.md"


@pytest.mark.skipif(not RESULTS.exists(), reason="results.json not generated yet")
def test_report_is_rendered_from_results_json():
    results = json.loads(RESULTS.read_text())
    rendered = run_xval.render_report(results)
    assert REPORT.exists(), "REPORT.md missing"
    assert rendered.strip() == REPORT.read_text().strip(), (
        "REPORT.md is not the render of results.json -- numbers were edited by hand"
    )


@pytest.mark.skipif(not RESULTS.exists(), reason="results.json not generated yet")
def test_results_json_shape_and_thresholds():
    results = json.loads(RESULTS.read_text())
    assert results["anchor"] == "XVAL-LBM-NO-MCMP-TWO-PHASE-ONLY"
    assert len(results["runs"]) >= 15, "the predeclared matrix must be run in full"
    for rec in results["runs"]:
        assert np.isfinite(rec["l2_rel"]), rec["arm"]
        assert rec["steps"] > 0
        for key, val in rec["analytic_checks"].items():
            assert val < 1e-10, f"analytic invariant {key} = {val}"
    ev = results["verdict"]["checks"]
    # Every matched-density two-phase arm must be inside the owner band.
    for rec in results["runs"]:
        if rec["arm"]["kind"] == "two_phase" and rec["arm"].get("rho_hi", 1.0) == 1.0:
            assert rec["l2_rel"] < 0.15, (rec["arm"], rec["l2_rel"])
    assert ev["K1_analytic_selfcheck"]["pass"]
    assert ev["K6_mass_drift_rel"]["pass"]