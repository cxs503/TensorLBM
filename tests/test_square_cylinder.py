"""Unit + end-to-end tests for the square-cylinder benchmark support module.

Scope
-----
These tests assert *invariants* rather than merely executing code:

1. geometry — the half-way bounce-back mask must realise an effective body of
   exactly ``D`` lattice units and the requested 1/20 blockage;
2. force normalisation — ``Cd = Fx / (0.5 rho U^2 D)`` including the error
   paths (mismatched input lengths, non-positive dynamic pressure);
3. Strouhal estimation — a synthetic lift signal of known frequency must be
   recovered to within the FFT bin/parabolic-refinement accuracy, and the
   *seed* frequency must not be reported when the physical response differs
   (the "seeding only triggers, it does not lock" invariant);
4. reference grading — withheld references are never silently scored, tolerance
   boundaries are inclusive, and a missing exact-Re rung is flagged;
5. end to end — a short, coarse case runs through the real solver chain and
   writes a self-describing result record.

The end-to-end case is deliberately tiny (D=6, 400 steps).  It is a wiring
test, not an accuracy test: no tolerance on Cd/St is asserted there because a
6-node body cannot resolve the wake.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.square_cylinder import (  # noqa: E402
    assess,
    assess_case,
    blockage_ratio,
    drag_statistics,
    force_coefficients,
    load_reference,
    rect_mask_2d,
    reference_band,
    strouhal_fft,
    strouhal_from_series,
    strouhal_zero_crossings,
)

CASE_DIR = REPO / "benchmarks" / "verified" / "square_cylinder_re200"


# ---------------------------------------------------------------- geometry
@pytest.mark.parametrize("D,height", [(8, 20), (24, 20), (32, 20), (16, 40)])
def test_mask_realises_effective_body_and_blockage(D: int, height: int) -> None:
    """Solid nodes = D x D, blocked rows / ny = 1/height (exact mask geometry)."""
    nx, ny = 30 * D, height * D
    mask = rect_mask_2d(nx, ny, 8.0 * D, ny / 2.0, D)
    assert int(mask.sum()) == D * D, "half-way BB body must be exactly D x D nodes"
    assert blockage_ratio(mask) == pytest.approx(1.0 / height, abs=1e-12)
    # body is centred within one node in the cross-stream direction
    rows = torch.nonzero(mask.any(dim=1)).flatten()
    assert abs(float(rows.float().mean()) - ny / 2.0) <= 1.0


def test_mask_rejects_out_of_domain() -> None:
    with pytest.raises(ValueError):
        rect_mask_2d(64, 64, 2.0, 32.0, 16)  # runs out of the inlet edge
    with pytest.raises(ValueError):
        rect_mask_2d(64, 64, 32.0, 32.0, 0)  # degenerate side


# ------------------------------------------------------------------ forces
def test_force_normalisation_and_error_paths() -> None:
    u, D, rho = 0.05, 24, 1.0
    q = 0.5 * rho * u * u * D
    cd, cl = force_coefficients([q, 2 * q], [0.0, -q], u, D, rho)
    assert cd == pytest.approx([1.0, 2.0])
    assert cl == pytest.approx([0.0, -1.0])
    with pytest.raises(ValueError):
        force_coefficients([1.0, 2.0], [1.0], u, D)
    with pytest.raises(ValueError):
        force_coefficients([1.0], [1.0], 0.0, D)


def test_drag_statistics_uses_window() -> None:
    cd = np.concatenate([np.full(100, 99.0), np.full(100, 1.4)])
    stats = drag_statistics(cd, start=100)
    assert stats["mean"] == pytest.approx(1.4)
    assert stats["n_samples"] == 100
    with pytest.raises(ValueError):
        drag_statistics(cd, start=1000)


# --------------------------------------------------------------- strouhal
def _synthetic_lift(st_true: float, u: float, D: int, n: int, noise: float = 0.0, seed: int = 0):
    dt = 1.0
    f = st_true * u / D
    t = np.arange(n) * dt
    rng = np.random.default_rng(seed)
    sig = np.sin(2 * math.pi * f * t)
    if noise:
        sig = sig + noise * rng.standard_normal(n)
    return sig


@pytest.mark.parametrize("st_true", [0.147, 0.16, 0.169, 0.18])
def test_strouhal_recovered_from_synthetic_lift(st_true: float) -> None:
    u, D = 0.05, 24
    sig = _synthetic_lift(st_true, u, D, n=40000)
    out = strouhal_from_series(sig, u, D)
    assert out["st"] == pytest.approx(st_true, rel=0.01)
    assert out["crossing"]["st"] == pytest.approx(st_true, rel=0.01)
    assert out["fft"]["energy_frac"] > 0.9


def test_strouhal_robust_to_noise() -> None:
    u, D, st_true = 0.05, 24, 0.165
    sig = _synthetic_lift(st_true, u, D, n=40000, noise=0.05, seed=7)
    out = strouhal_from_series(sig, u, D)
    assert out["st"] == pytest.approx(st_true, rel=0.02)


def test_seed_frequency_does_not_lock_the_measurement() -> None:
    """The reported St must follow the signal, not the seeding frequency."""
    u, D, st_seed, st_physical = 0.05, 24, 0.16, 0.172
    sig = _synthetic_lift(st_physical, u, D, n=40000)
    out = strouhal_from_series(sig, u, D)
    assert out["st"] != pytest.approx(st_seed, rel=0.02)
    assert out["st"] == pytest.approx(st_physical, rel=0.01)


def test_strouhal_rejects_short_series() -> None:
    with pytest.raises(ValueError):
        strouhal_fft([0.0, 1.0, 0.0], 0.05, 24)
    with pytest.raises(ValueError):
        strouhal_zero_crossings([0.0, 1.0, 0.0], 0.05, 24)


# --------------------------------------------------------------- reference
def test_reference_file_loads_and_is_graded() -> None:
    ref = load_reference(CASE_DIR / "reference.json")
    assert ref["case"] == "CASE-LBM-SQUARE-CYL-NORBERG"
    grades = {r["grade"] for r in ref["rows"]}
    assert "VERIFIED_TABLE" in grades and "WITHHELD" in grades
    assert ref["anchor_parameters"] == {"re": 200, "blockage": 0.05, "incidence_deg": 0}


def test_withheld_reference_is_never_scored() -> None:
    ref = load_reference(CASE_DIR / "reference.json")
    rows = assess_case({"cd": 1.40, "st": 0.168}, ref, 200.0)
    assert all(r["status"] == "WITHHELD" for r in rows)
    assert all(r["err_pct"] is None for r in rows)


def test_tolerance_boundaries_and_fallback_rung() -> None:
    row = {"re": 150, "metric": "cd", "ref": 1.41, "tol_pct": 10.0, "grade": "VERIFIED_TABLE"}
    # tolerance is inclusive: just inside +10% passes, just outside fails
    assert assess(row["ref"] * 1.0999, row)["status"] == "PASS"
    assert assess(np.nextafter(row["ref"] * 1.10, 0.0), row)["status"] == "PASS"
    assert assess(row["ref"] * 1.1001, row)["status"] == "FAIL"
    assert assess(row["ref"] * 0.8999, row)["status"] == "FAIL"
    assert assess(None, row)["status"] == "NO_DATA"

    ref = load_reference(CASE_DIR / "reference.json")
    interpolated = reference_band(ref["rows"], 175.0, "cd")
    assert interpolated["exact"] is False and interpolated["re"] == 150
    scored = assess(1.36, interpolated)
    assert scored["status"] == "PASS" and "trend-indicative" in scored["note"]
    exact = reference_band(ref["rows"], 150.0, "cd")
    assert exact["exact"] is True


def test_withheld_row_must_explain_itself(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"rows": [{"re": 200, "metric": "cd", "ref": None}]}))
    with pytest.raises(KeyError):
        load_reference(bad)


# --------------------------------------------------------------- end to end
@pytest.mark.slow
def test_end_to_end_writes_self_describing_record(tmp_path: Path) -> None:
    out = tmp_path / "case"
    cmd = [
        sys.executable,
        str(CASE_DIR / "run.py"),
        "--D", "6",
        "--re", "150",
        "--steps", "400",
        "--threads", "2",
        "--out", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rec = json.loads((out / "case_D6.json").read_text())
    assert rec["finite"] is True
    assert rec["cd"]["mean"] > 0.0
    assert math.isfinite(rec["st"]["value"])
    assert rec["geometry"]["blockage_measured"] == pytest.approx(0.05, abs=1e-12)
    assert rec["re"] == 150.0
    assert (out / "series_D6.npz").exists()
    assert len(rec["scoring"]) == 2
    summary = json.loads((out / "result.json").read_text())
    assert summary["case"] == "CASE-LBM-SQUARE-CYL-NORBERG"
    assert "run.py" in summary["repro_cmd"]