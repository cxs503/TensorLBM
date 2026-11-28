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


def test_legacy_round1_import_aliases_packaged_authority():
    import two_phase_only

    from tensorlbm.verification import two_phase_only as canonical
    from tensorlbm.verification.two_phase_only_evolving import _two_layer_analytic

    assert two_phase_only is canonical
    assert canonical is _two_layer_analytic()
    assert canonical.two_layer_analytic is two_layer_analytic
    assert canonical.run_two_phase_only is run_two_phase_only


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
    # K6 is an honest, documented FAIL in round 1 (fluid-only mass drift above the
    # predeclared 1e-6).  What is asserted here is the *contract*, not a pass:
    # the predeclared threshold was never relaxed and the recorded verdict is the
    # plain comparison of the measured value against it (nothing hard-coded).
    assert ev["K6_mass_drift_rel"]["threshold"] == 1e-06, "K6 threshold must not be relaxed"
    assert ev["K6_mass_drift_rel"]["pass"] == (
        ev["K6_mass_drift_rel"]["value"] <= 1e-06
    ), "K6 verdict must be the honest comparison of value vs threshold"


# ---------------------------------------------------------------------------
# 5. Mass budget (M21325): the K6 failure must be explainable, not assumed
# ---------------------------------------------------------------------------


def _mass_arm(**kw):
    args = dict(ny=32, nx=4, G=4e-6, nu_lo=0.5, nu_hi=0.25, width=2.0,
                rule="harmonic", max_steps=2000, stop_tol=1e-9, check_every=200)
    args.update(kw)
    return run_two_phase_only(**args)


def test_operators_conserve_mass_individually():
    """Collision, streaming, bounce-back and the Guo source each conserve mass."""
    import torch

    from tensorlbm.boundary.boundaries import bounce_back_cells
    from tensorlbm.d2q9 import equilibrium, macroscopic
    from tensorlbm.multiphase import guo_force_term
    from tensorlbm.solver import stream

    torch.manual_seed(0)
    ny, nx = 24, 3
    rho = 1.0 + 0.01 * torch.rand(ny, nx)
    ux = 1e-4 * torch.rand(ny, nx)
    uy = 1e-4 * torch.rand(ny, nx)
    f = equilibrium(rho, ux, uy).clone()
    wall = torch.zeros((ny, nx), dtype=torch.bool)
    wall[0, :] = True
    wall[-1, :] = True
    tau = torch.full((ny, nx), 0.7)
    fx = torch.full((ny, nx), 1e-6)
    m = lambda t: float(t.sum())  # noqa: E731
    m0 = m(f)

    assert abs(m(stream(f)) - m0) / m0 < 1e-12, "streaming must conserve mass"
    assert abs(m(bounce_back_cells(f.clone(), wall)) - m0) / m0 < 1e-12, (
        "full-way bounce-back must conserve mass on a closed domain"
    )
    r, u1, u2 = macroscopic(f)
    feq = equilibrium(r, u1, u2)
    assert abs(m(f - (f - feq) / tau) - m0) / m0 < 1e-6, "BGK collision must conserve mass"
    u1s = u1 + 0.5 * fx / torch.clamp(r, min=1e-12)
    src = guo_force_term(r, u1s, u2, fx, torch.zeros_like(fx), tau)
    assert abs(m(src)) / m0 < 1e-6, "the Guo source must have (numerically) zero mass moment"


def test_mass_budget_identifies_round_off_not_a_leak():
    """float32 drift is round-off: whole-domain budget reconciles and float64 collapses it.

    * every arm records fluid-only drift (K6), wall drift, whole-domain drift and
      the redistribution share;
    * the wall buffer rows are *not* the sink (share << 1);
    * the same arm in float64 conserves mass to ~round-off, so the operator chain
      of the two-phase-only model is conservative and K6 measured precision.
    """
    rec32 = _mass_arm(dtype=__import__("torch").float32)
    rec64 = _mass_arm(dtype=__import__("torch").float64)
    for rec in (rec32, rec64):
        for key in ("mass_drift_rel", "mass_wall_drift_rel", "mass_total_drift_rel",
                    "mass_redistribution_share", "mass_dtype", "mass_history"):
            assert key in rec, key
        assert rec["mass_total_drift_rel"] >= 0.0
        assert rec["mass_redistribution_share"] is not None
    assert rec32["mass_dtype"] == "torch.float32" and rec64["mass_dtype"] == "torch.float64"
    # the wall buffer is not where the fluid-only deficit went
    assert rec32["mass_redistribution_share"] < 0.1, rec32["mass_redistribution_share"]
    # float64 conservation is at round-off level
    assert rec64["mass_total_drift_rel"] < 1e-9, rec64["mass_total_drift_rel"]
    assert rec64["mass_total_drift_rel"] < rec32["mass_total_drift_rel"] / 100.0
    # and the audit changes nothing physical
    assert abs(rec64["l2_rel"] - rec32["l2_rel"]) < 1e-3
    # the recorded history is monotone in step and consistent with the final number
    hist = rec32["mass_history"]
    assert hist and [h["step"] for h in hist] == sorted(h["step"] for h in hist)
    assert abs(hist[-1]["fluid_rel"] - rec32["mass_drift_rel"]) < 1e-9


def test_posthoc_addendum_does_not_relax_the_predeclared_contract():
    add = json.loads((XVAL_DIR / "posthoc_addendum.json").read_text())
    pre = json.loads((XVAL_DIR / "predeclared.json").read_text())
    assert add["status"] == "post-hoc"
    # the predeclared K6 threshold is untouched by the addendum
    assert pre["predeclared_thresholds"]["K6_mass_drift_rel"] == 1e-06
    assert add["thresholds"]["D1_mass_drift_float64"] == 1e-06
    assert add["thresholds"]["D2_not_parked_in_walls"] < 1.0
