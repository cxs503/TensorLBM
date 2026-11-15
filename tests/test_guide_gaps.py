"""Acceptance gate for the three closed guide gaps (m21096).

Covers ``dam_break_sc``, ``forchheimer`` and ``kelvin_helmholtz``: every case is
required to deliver (a) a real solver, (b) a machine-comparable reference and
(c) at least two independent assertions -- an analytic comparison plus a
degeneration / convergence guard.

Run with::

    python3 -m pytest tests/test_guide_gaps.py -q

The cases are computed once per session (``--quick`` settings) and compared with
the committed artifacts, so a stale artifact fails here instead of silently
disagreeing with the report.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GAP_DIR = REPO / "examples" / "guide_gaps"
ART_DIR = REPO / "artifacts" / "guide_gaps"
DOC = REPO / "docs" / "guide_gaps" / "REPORT.md"

if str(GAP_DIR) not in sys.path:
    sys.path.insert(0, str(GAP_DIR))

_CACHE: dict[str, dict] = {}


def _payload(case: str) -> dict:
    """Compute (once) the ``--quick`` payload of one guide-gap case."""
    if case not in _CACHE:
        mod = importlib.import_module(case)
        _CACHE[case] = mod.run(quick=True)
    return _CACHE[case]


def _check(payload: dict, name: str) -> dict:
    for row in payload["checks"]:
        if row["name"] == name:
            return row
    raise KeyError(f"check {name!r} not found in {payload['case']}")


def _artifact(case: str) -> dict:
    return json.loads((ART_DIR / f"{case}.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# dam_break_sc
# --------------------------------------------------------------------------- #
def test_dam_break_analytic_front_and_shock_strength():
    """Front position and post-shock depth vs the closed-form Stoker solution."""
    p = _payload("dam_break_sc")
    for name in ("front_position_theta0", "shock_strength_theta0",
                 "front_position_oblique", "shock_strength_oblique",
                 "rarefaction_head_theta0"):
        row = _check(p, name)
        assert row["pass"], (name, row["rel_err"])
        assert row["rel_err"] < 0.03, (name, row["rel_err"])
    # the headline quantity is a real number, not a placeholder
    assert 0.5 < p["computed"] < 1.5
    assert abs(p["computed"] - p["reference"]) / p["reference"] < 0.03


def test_dam_break_grid_convergence():
    """Refining the mesh must reduce both the front and the profile error."""
    p = _payload("dam_break_sc")
    gc = p["grid_convergence"]
    assert gc["rel_err_front"][1] < gc["rel_err_front"][0]
    assert gc["rel_l1_profile"][1] < gc["rel_l1_profile"][0]
    assert gc["oblique_rel_err_front"][1] < gc["oblique_rel_err_front"][0]
    assert gc["oblique_rel_err_h2"][1] < gc["oblique_rel_err_h2"][0]


def test_dam_break_obliquity_artefact_and_symmetry():
    """Aligned front -> zero spurious transverse velocity; obliquity -> non-zero.

    Also the exact theta <-> 90-theta symmetry of the square isotropic problem,
    which is what makes the oblique numbers trustworthy.
    """
    p = _payload("dam_break_sc")
    aligned = _check(p, "spurious_transverse_aligned")
    assert aligned["computed"] < 1e-12, aligned
    oblique = _check(p, "spurious_transverse_oblique")
    assert oblique["computed"] > 1e-3, "oblique front should generate u_t != 0"
    assert oblique["pass"]
    sym = _check(p, "obliquity_symmetry")
    assert sym["rel_err"] < 5e-3, sym


def test_dam_break_conservation_and_artifact():
    """Discrete mass budget closes; the committed artifact matches this run."""
    p = _payload("dam_break_sc")
    assert _check(p, "mass_budget_residual")["computed"] < 1e-12
    art = _artifact("dam_break_sc")
    assert art["case"] == "dam_break_sc"
    assert art["evidence_grade"] == "REAL"
    assert abs(art["computed"] - p["computed"]) <= 1e-12 + 1e-9 * abs(p["computed"])
    assert art["reference_source"].startswith("Stoker (1957)")
    assert art["checks"] and all(c["pass"] for c in art["checks"])


# --------------------------------------------------------------------------- #
# forchheimer
# --------------------------------------------------------------------------- #
def test_forchheimer_curve_fit_coefficients():
    """Fitted quadratic coefficient and permeability interpolation vs analysis."""
    p = _payload("forchheimer")
    for name in ("quadratic_coefficient", "permeability_from_intercept",
                 "bulk_velocity_agreement"):
        row = _check(p, name)
        assert row["pass"], (name, row["rel_err"], row)
        assert row["rel_err"] < 0.03
    curve = p["bulk_curve"]
    assert curve["fit_r2"] > 0.999999


def test_forchheimer_darcy_degeneration():
    """Switching the inertial term off must collapse onto Darcy / Brinkman."""
    p = _payload("forchheimer")
    for name in ("darcy_velocity_exact", "darcy_degeneration_slope",
                 "darcy_degeneration_K", "darcy_degeneration_closed_form"):
        row = _check(p, name)
        assert row["pass"], (name, row["rel_err"], row)
    d = p["darcy"]
    assert d["fit_B_free_slip"] == pytest.approx(0.0, abs=1e-6 * d["fit_A_free_slip"])
    # the wall layer lowers the effective permeability -- documented, not hidden
    assert 0.5 < d["wall_bias_on_bulk_fit"] < 1.0


def test_forchheimer_ergun_inversion_and_convergence():
    """K/beta inversion against the Ergun correlation + 2nd-order convergence."""
    p = _payload("forchheimer")
    assert _check(p, "permeability_inversion_ergun")["rel_err"] < 0.03
    assert _check(p, "forchheimer_coefficient_inversion_ergun")["rel_err"] < 0.03
    assert _check(p, "channel_Q_error_finest")["rel_err"] < 0.03
    gc = p["grid_convergence"]
    errs = gc["max_rel_err"]
    assert errs == sorted(errs, reverse=True), errs
    assert all(1.7 < o < 2.4 for o in gc["observed_order"]), gc["observed_order"]
    assert all(gc["columns_identical"])


# --------------------------------------------------------------------------- #
# kelvin_helmholtz
# --------------------------------------------------------------------------- #
def test_kh_dispersion_growth_rates():
    """>= 3 wavenumbers: numerical growth rate vs the inviscid dispersion relation."""
    p = _payload("kelvin_helmholtz")
    rows = [c for c in p["checks"] if c["name"].startswith("growth_rate_k")]
    assert len(rows) >= 3
    for row in rows:
        assert row["pass"], row
        assert row["rel_err"] < 0.03
    assert _check(p, "phase_speed")["rel_err"] < 0.03
    assert p["dispersion"]["k_cutoff"] > 0


def test_kh_spectral_convergence():
    """Error must fall (spectrally) with the number of Chebyshev points."""
    p = _payload("kelvin_helmholtz")
    gc = p["grid_convergence"]
    errs = gc["rel_err"]
    assert all(b < a for a, b in zip(errs, errs[1:])), errs
    assert errs[-1] < errs[0] * 1e-4, errs
    assert all(o > 2.0 for o in gc["observed_order"]), gc["observed_order"]
    assert _check(p, "spectral_convergence_shortfall")["pass"]


def test_kh_degenerate_branches():
    """sigma=0 branch, equal-density limit, surface-tension cutoff, k_max."""
    p = _payload("kelvin_helmholtz")
    for name in ("sigma_zero_branch", "equal_density_limit",
                 "surface_tension_cutoff", "most_unstable_wavenumber",
                 "far_field_truncation"):
        row = _check(p, name)
        assert row["pass"], (name, row["rel_err"], row)
    d = p["dispersion"]
    assert d["omega_imag_numeric"][0] < d["omega_imag_numeric"][-1]
    assert abs(d["k_most_unstable_numeric"] / d["k_most_unstable_analytic"] - 1) < 0.03


# --------------------------------------------------------------------------- #
# pack-level gates
# --------------------------------------------------------------------------- #
def test_artifact_schema_and_summary_verdicts():
    """Every artifact carries the mandated schema; the summary verdicts agree."""
    import make_summary

    for case in ("dam_break_sc", "forchheimer", "kelvin_helmholtz"):
        art = _artifact(case)
        for key in ("case", "quantity", "computed", "reference", "rel_err",
                    "evidence_grade"):
            assert key in art, (case, key)
        assert art["evidence_grade"] in ("REAL", "WITHHELD")
        if art["evidence_grade"] == "WITHHELD":
            assert art.get("withheld_reason")
        assert art["checks"], case
        assert all(c["pass"] for c in art["checks"]), case

    summary = make_summary.build_summary(write=False, run_missing=False)
    assert summary["n_cases"] == 3
    for case, row in summary["cases"].items():
        assert isinstance(row["lt_15pct"], bool) and isinstance(row["lt_3pct"], bool)
        assert row["lt_15pct"] and row["lt_3pct"], (case, row["rel_err"])
        assert isinstance(row["reference_source"], str) and len(row["reference_source"]) > 20
        assert row["evidence_grade"] in ("REAL", "WITHHELD")
        assert row["n_checks_failed"] == 0
    assert summary["all_lt_15pct"] and summary["all_lt_3pct"]
    assert summary["grades_acceptable"]


def test_examples_cli_quick_exit_zero(tmp_path):
    """Each example must run standalone as a script and write its artifact."""
    for case in ("kelvin_helmholtz", "forchheimer", "dam_break_sc"):
        proc = subprocess.run(
            [sys.executable, str(GAP_DIR / f"{case}.py"), "--quick"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert proc.returncode == 0, (case, proc.stdout[-2000:], proc.stderr[-2000:])
        art = _artifact(case)
        assert art["case"] == case
        assert art["evidence_grade"] in ("REAL", "WITHHELD")
        assert art["command"].endswith("--quick")


def test_report_present_with_figures():
    """REPORT.md has the six mandated sections and at least one generated png."""
    assert DOC.exists(), DOC
    text = DOC.read_text(encoding="utf-8")
    for marker in ("## 1. 案例介绍", "## 2. 怎么做的", "## 3. 计算结果",
                   "## 4. 注意点", "## 5. 误差判定", "## 6. 未完成"):
        assert marker in text, marker
    assert "是否 < 3 %" in text
    figs = sorted((ART_DIR / "figures").glob("*.png"))
    assert len(figs) >= 1
    assert figs[0].stat().st_size > 5000
    assert "figures/" in text