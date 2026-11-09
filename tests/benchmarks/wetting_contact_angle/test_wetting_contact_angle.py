"""Instrument + gate tests for the static contact-angle benchmark.

Three layers, in increasing cost:

1. **estimator self-validation** on analytic circular segments -- proves the
   angle measurement is unbiased before it is used to judge the wall force;
2. **library conventions** -- pins the supplement-vs-physical convention of
   :mod:`tensorlbm.contact_line.theta_fit` and the Young inversion/limits, so a
   future change in either cannot silently invalidate the benchmark;
3. **artifact gate** -- the sweep's JSON (``results/contact_angle_sweep.json``)
   is checked for internal consistency, for the recorded verdicts against the
   pre-registered criterion, and for zero hand-transcription (the generated
   markdown tables are re-rendered from the JSON and compared back).

Layer 3 is skipped when the artifact is absent (producing it takes minutes of
relaxation): run ``bash run_all.sh`` + ``assemble.py`` + ``make_report.py``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import contact_angle as ca  # noqa: E402
import make_report  # noqa: E402
import wetting_bench as wb  # noqa: E402
from tensorlbm.contact_line.theta_fit import (  # noqa: E402
    circle_fit,
    contact_angle_from_circle,
    interface_points,
)

ARTIFACT = HERE / "results" / "contact_angle_sweep.json"
REPORT = HERE / "results" / "REPORT.md"
RHO_L, RHO_V = 1.957, 0.1596
SMOKE = wb.Grid(name="smoke", nx=64, ny=48, radius=10.0, steps=600, samples=3)


# ---------------------------------------------------------------------------
# 1. estimator self-validation on analytic caps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta_true", [60.0, 90.0, 110.0, 130.0])
def test_estimators_recover_analytic_cap(theta_true):
    """All three estimators must reproduce the analytic segment angle.

    Tolerance: 2 deg for the area/height inversion (it integrates the diffuse
    profile, whose finite tanh width biases the apex by ~0.2 cell), 1.5 deg for
    the circle fits (exact for a clean cap).
    """
    rho, solid, y_wall = ca.synthetic_cap_density(
        theta_deg=theta_true, radius=13.0, nx=96, ny=64, rho_l=RHO_L, rho_v=RHO_V)
    m = ca.measure_contact_angle(rho, rho_l=RHO_L, rho_v=RHO_V, y_wall=y_wall,
                                solid=solid, radius=13.0)
    assert abs(m["theta_cap"] - theta_true) < 2.0, m
    assert abs(m["theta_circle"] - theta_true) < 1.5, m
    assert abs(m["theta_band"] - theta_true) < 2.5, m


def test_library_contour_estimator_returns_supplement():
    """The library estimators report ``180 - theta_physical`` (documented trap).

    A 60 deg wetting segment must be read as ~120 deg by
    :func:`contact_angle_from_circle`; ``contact_angle`` converts it back.
    """
    rho, solid, y_wall = ca.synthetic_cap_density(
        theta_deg=60.0, radius=13.0, nx=96, ny=64, rho_l=RHO_L, rho_v=RHO_V)
    pts = interface_points(torch.as_tensor(rho, dtype=torch.float32),
                           level=0.5 * (RHO_L + RHO_V),
                           solid=torch.as_tensor(solid))
    raw, _ = contact_angle_from_circle(circle_fit(pts), y_wall)
    assert abs(raw - 120.0) < 1.5, raw
    assert ca.LIB_IS_SUPPLEMENT is True
    m = ca.measure_contact_angle(rho, rho_l=RHO_L, rho_v=RHO_V, y_wall=y_wall,
                                solid=solid, radius=13.0)
    assert abs(m["theta_circle"] - 60.0) < 1.5, m


def test_cap_inversion_flags_dewetted_wall():
    """A vapour-filled box has no drop: the estimator must return NaN, not a number."""
    rho = np.full((64, 96), RHO_V)
    solid = np.zeros_like(rho, dtype=bool)
    solid[:2] = True
    solid[-2:] = True
    out = ca.cap_inversion(rho, rho_l=RHO_L, rho_v=RHO_V, y_wall=1.5, solid=solid)
    assert math.isnan(out["theta_deg"])
    assert out["area"] < 1.0


# ---------------------------------------------------------------------------
# 2. Young relation / prescription law bookkeeping
# ---------------------------------------------------------------------------


def test_young_relation_roundtrip_and_limits():
    psi_l, psi_v = wb._psi_pair()
    for theta in (30.0, 70.0, 90.0, 100.0):
        fw = wb.wall_fraction_for_angle(theta, psi_l, psi_v)
        assert wb.theta_from_wall_fraction(fw, psi_l, psi_v) == pytest.approx(theta, abs=1e-6)
    # psi_w = psi_l  <->  complete wetting
    assert wb.theta_from_wall_fraction(1.0, psi_l, psi_v) == pytest.approx(0.0, abs=1e-6)
    # psi_w = 0 bounds the reachable angles of a non-negative wall pseudopotential
    theta_max = wb.reachable_angle_limit_deg(psi_l, psi_v)
    assert wb.theta_from_wall_fraction(0.0, psi_l, psi_v) == pytest.approx(theta_max, abs=1e-9)
    assert 90.0 < theta_max < 180.0
    # ... so an angle beyond it cannot be prescribed by the Young law at all
    with pytest.raises(ValueError):
        wb.wall_fraction_for_angle(110.0, psi_l, psi_v)


def test_calibration_inversion_is_honest():
    """Inversion must be piecewise-linear in the *measured* points and refuse extrapolation."""
    curve = {
        "theta_deg": [40.0, 60.0, 80.0],
        "wall_fraction": [0.8, 0.7, 0.6],
        "source_grid": ["coarse"] * 3,
        "source_theta_init": [40.0, 60.0, 80.0],
        "monotone_decreasing_fw": True,
    }
    assert wb.invert_calibration(curve, 50.0) == pytest.approx(0.75)
    assert wb.invert_calibration(curve, 60.0) == pytest.approx(0.70)
    with pytest.raises(ValueError):
        wb.invert_calibration(curve, 30.0)
    assert wb.calibration_curve(
        [{"theta_measured": 100.0, "wall_fraction": 0.5, "grid": "coarse", "theta_eq": 90.0,
          "converged": True},
         {"theta_measured": 80.0, "wall_fraction": 0.7, "grid": "coarse", "theta_eq": 90.0,
          "converged": True}]
    )["monotone_decreasing_fw"] is True


# ---------------------------------------------------------------------------
# 3. a short relaxation (fast smoke, no artifact needed)
# ---------------------------------------------------------------------------


def test_short_relaxation_conserves_mass_and_is_finite():
    rec = wb.run_case(90.0, SMOKE, threads=2, keep_points=False)
    assert rec["status"] in {"PASS", "FAIL", "UNCONVERGED", "NO_INTERFACE"}
    assert math.isfinite(rec["theta_measured"])
    assert abs(rec["mass_drift"]) / rec["mass_start"] < 0.01
    assert math.isfinite(rec["spurious_max"])


# ---------------------------------------------------------------------------
# 4. artifact gate
# ---------------------------------------------------------------------------


needs_artifact = pytest.mark.skipif(
    not ARTIFACT.exists(), reason="results/contact_angle_sweep.json missing (run run_all.sh)")


@pytest.fixture(scope="module")
def payload():
    return json.loads(ARTIFACT.read_text())


@needs_artifact
def test_artifact_status_matches_criterion(payload):
    """Every case's status must follow the pre-registered rule (no verdict laundering)."""
    tol = payload["meta"]["accept_deg"]
    for stage in ("stage_a_young", "stage_b_calibration", "stage_b_verify"):
        for c in payload[stage]["cases"]:
            if c["status"] in ("OUT_OF_RANGE", "NO_INTERFACE"):
                assert c.get("theta_eq") is not None
                continue
            if c["status"] == "UNCONVERGED":
                assert not c["converged"]
                continue
            assert math.isfinite(c["theta_measured"])
            ok = abs(c["theta_measured"] - c["theta_eq"]) < tol
            assert (c["status"] == "PASS") == ok, c
            assert c["delta"] == pytest.approx(c["theta_measured"] - c["theta_eq"], abs=1e-9)


@needs_artifact
def test_young_law_is_falsified(payload):
    """Law A must be recorded as failed -- a pass here would invalidate the finding."""
    v = payload["verdicts"]["young"]
    assert v["status"] == "FAIL", v
    assert v["max_abs_delta"] >= payload["meta"]["accept_deg"]
    measured = [c for c in payload["stage_a_young"]["cases"] if c.get("theta_measured")]
    assert measured, "no Young-law case produced a measured angle"
    # the outlier is systematic (same sign), i.e. a mis-calibrated knob, not noise
    assert all(c["delta"] > 0 for c in measured), [c["delta"] for c in measured]


@needs_artifact
def test_calibrated_prescription_verdict_is_recorded_honestly(payload):
    """Law B: either it passes, or the FAIL is carried with its root-cause evidence.

    An honest FAIL must not be overwritten by relaxing the tolerance or dropping
    cases, so a failing verdict is only accepted together with the quantitative
    root-cause blocks (grid shift, metastability split) and the documented
    limitations in the report.
    """
    v = payload["verdicts"]["calibrated"]
    tol = payload["meta"]["accept_deg"]
    assert v["n_cases"] >= 3, v
    if v["status"] == "PASS":
        assert v["max_abs_delta"] < tol, v
        return
    assert v["max_abs_delta"] >= tol, v
    assert payload.get("grid_sensitivity"), "grid-sensitivity block missing"
    assert payload["stage_c_metastability"]["split_deg"] is not None, "no metastability probe"
    if REPORT.exists():
        text = REPORT.read_text()
        for needle in ("honest FAIL", "Root cause", "Grid sensitivity", "Metastability"):
            assert needle.lower() in text.lower(), f"report lacks '{needle}'"

    # the deviation must be systematic (same sign across cases), not scatter
    deltas = [c["delta"] for c in payload["stage_b_verify"]["cases"] if c.get("delta") is not None]
    assert deltas and (all(d > 0 for d in deltas) or all(d < 0 for d in deltas)), deltas


@needs_artifact
def test_grid_sensitivity_is_recorded(payload):
    """The same prescribed angle must be present on >= 2 grid levels in stage A/B."""
    grids = {c["grid"] for c in payload["stage_a_young"]["cases"]}
    assert len(grids) >= 2, grids
    pairs = {}
    for c in payload["stage_a_young"]["cases"]:
        pairs.setdefault(c["theta_eq"], set()).add(c["grid"])
    assert any(len(v) >= 2 for v in pairs.values()), pairs


@needs_artifact
def test_report_tables_are_generated_not_typed():
    """Zero hand-copy: re-render every table from the JSON and find it in REPORT.md."""
    assert REPORT.exists(), "results/REPORT.md missing (run make_report.py)"
    payload = json.loads(ARTIFACT.read_text())
    text = REPORT.read_text()
    tables = make_report.generated_tables(payload)
    for line in tables.splitlines():
        if line.startswith("| ") and not line.startswith("|---") and "θ" not in line:
            assert line in text, f"table row not found verbatim in REPORT.md: {line}"
    for stage in ("stage_a_young", "stage_b_calibration", "stage_b_verify"):
        rendered = make_report.table_cases(payload[stage]["cases"])
        for line in rendered.splitlines()[2:]:
            assert line in text, f"{stage} row missing from REPORT.md: {line}"