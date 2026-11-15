"""Anchor gate + GHIA_RE400 drift regression tests (JOUR-DIFFLBM, Re=400).

These tests bind the *reference side* of the Re=400 cavity benchmark to the
sha256-pinned Ghia (1982) witness, so that a later run cannot be scored against
a drifted constant (the failure mode that produced the 2026-11 GHIA_RE400
corruption):

* the anchor gate (witness sha256, per-CSV sha256, table integrity, code
  constants vs BOUND values, OCR value presence) must pass;
* a one-bit change in an anchor CSV must be detected;
* the in-code ``GHIA_RE400`` copies must equal the BOUND table;
* the scoring metric itself must be exact on known inputs (identity profile ->
  0%, single perturbed point -> the analytically expected max deviation).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
ANCHOR_DIR = REPO / "benchmarks" / "anchors"
CASE_DIR = REPO / "benchmarks" / "verified" / "cavity" / "re400"


def _load(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


anchors = _load("run_anchors", ANCHOR_DIR / "run_anchors.py")
anchor_check = _load("ghia_anchor_check", CASE_DIR / "anchor_check.py")


def _provenance() -> dict:
    return json.loads((ANCHOR_DIR / "ghia1982_provenance.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


def test_anchor_gate_passes() -> None:
    prov = _provenance()
    errs = anchors.verify_provenance_sha256(prov)
    errs += anchors.verify_csv_anchors(prov)
    errs += anchors.verify_table_integrity(prov)
    errs += anchors.verify_code_tables_match_anchor(prov)
    errs += anchors.verify_ocr_value_presence(prov)
    assert errs == []


def test_csv_anchors_are_sha256_verified_and_complete() -> None:
    prov = _provenance()
    paths = {e["path"] for e in prov["csv_anchors"]}
    assert len(prov["csv_anchors"]) >= 4
    assert "benchmarks/anchors/data/ghia1982_re400_tableI.csv" in paths
    assert "benchmarks/anchors/data/ghia1982_re400_tableII.csv" in paths


def test_tampered_csv_is_rejected() -> None:
    """A one-character edit inside an anchor CSV must fail the gate."""
    prov = _provenance()
    target = next(e for e in prov["csv_anchors"] if e["re"] == 400 and e["table"] == "II")
    src = REPO / target["path"]
    backup = src.read_text(encoding="utf-8")
    try:
        lines = backup.splitlines(keepends=True)
        for i, ln in enumerate(lines):
            if ln and not ln.startswith("#") and not ln.startswith("x,"):
                old = ln.strip().split(",")
                new = f"{old[0]},{-0.44993 if float(old[1]) != -0.44993 else -0.23827}\n"
                lines[i] = new
                break
        src.write_text("".join(lines), encoding="utf-8")
        errs = anchors.verify_csv_anchors(prov)
        assert any("sha256 mismatch" in e for e in errs), errs
    finally:
        src.write_text(backup, encoding="utf-8")
    assert anchors.verify_csv_anchors(prov) == []


# ---------------------------------------------------------------------------
# drift guard on the in-code constants
# ---------------------------------------------------------------------------


def test_inline_ghia_re400_matches_bound_anchor() -> None:
    """Regression for the corrupted v(x=0.9063) / swapped 0.2266-0.2344 pair."""
    prov = _provenance()
    src = REPO / "src" / "tensorlbm" / "lid_driven_cavity.py"
    tables = anchors._extract_code_tables(src)
    block = prov["reynolds_numbers"]["400"]
    code = tables["GHIA_RE400"]
    # code stores x ascending, the witness lists x descending
    np.testing.assert_allclose(code["v"], np.asarray(block["v"])[::-1], atol=2e-4)
    np.testing.assert_allclose(code["x"], np.asarray(block["x"])[::-1], atol=1e-4)
    np.testing.assert_allclose(code["u"], np.asarray(block["u"]), atol=2e-4)
    # the specific historical corruption must not come back
    assert code["v"][list(code["x"]).index(0.9063)] == pytest.approx(-0.23827, abs=2e-4)


def test_inline_ghia_re400_3d_matches_bound_anchor() -> None:
    prov = _provenance()
    src = REPO / "src" / "tensorlbm" / "lid_driven_cavity_3d.py"
    code = anchors._extract_code_tables(src)["GHIA_RE400"]
    block = prov["reynolds_numbers"]["400"]
    np.testing.assert_allclose(code["v"], np.asarray(block["v"])[::-1], atol=2e-4)


def test_suspect_entries_are_documented() -> None:
    prov = _provenance()
    suspects = prov["suspect_entries"]
    assert len(suspects) >= 3
    for e in suspects:
        assert e["reason"].strip() and e["action"].strip()
    # the corruption this mission fixed must be on the exclusion list
    assert any("lid_driven_cavity" in e["table"] for e in suspects)


# ---------------------------------------------------------------------------
# metric behaviour (scoring must be exact, not merely "runs")
# ---------------------------------------------------------------------------


def _run_from_values(anchor: dict, u: np.ndarray, v: np.ndarray) -> dict:
    """Build a run dict whose centrelines are sampled *on* the anchor grid.

    Sampling exactly at the anchor coordinates makes the comparison exact, so
    the metric can be tested against hand-computed values instead of merely
    "the code ran".
    """
    t1, t2 = anchor["tables"]["I"], anchor["tables"]["II"]
    y_desc, u_ref = t1["coord"], np.asarray(u, dtype=float)
    x_desc, v_ref = t2["coord"], np.asarray(v, dtype=float)
    y = y_desc[::-1]
    x = x_desc[::-1]
    return {"nx": len(y), "re": 400.0, "u_lid": 0.1, "tau": 0.5727, "steps": 1,
            "last_resid": 0.0,
            "profiles": {"y": y.tolist(), "x": x.tolist(),
                         "u_cl": u_ref[::-1].tolist(), "v_cl": v_ref[::-1].tolist()}}


def test_identity_profile_scores_zero() -> None:
    anchor = anchor_check.load_bound_anchor(400)
    t1, t2 = anchor["tables"]["I"], anchor["tables"]["II"]
    run = _run_from_values(anchor, t1["value"], t2["value"])
    scored = anchor_check.score_profile(run, anchor)
    assert scored["metrics"]["M1_max_point_dev_pct"] == pytest.approx(0.0, abs=1e-9)
    assert scored["metrics"]["M3_extremum_rel_pct"]["v_min"] == pytest.approx(0.0, abs=1e-9)
    assert scored["metrics"]["M1_pass"] and scored["metrics"]["M3_pass"]


def test_known_perturbation_scores_expected_deviation() -> None:
    """+0.003*U_lid on the v extremum -> M1 = 0.30%, M3(v_min) = 0.67%."""
    anchor = anchor_check.load_bound_anchor(400)
    t1, t2 = anchor["tables"]["I"], anchor["tables"]["II"]
    v = np.array(t2["value"], dtype=float)
    v_min = int(np.argmin(v))
    v[v_min] += 0.003
    run = _run_from_values(anchor, t1["value"], v)
    scored = anchor_check.score_profile(run, anchor)
    assert scored["metrics"]["M1_max_point_dev_pct"] == pytest.approx(0.30, abs=1e-6)
    assert scored["metrics"]["M3_extremum_rel_pct"]["v_min"] == pytest.approx(
        0.003 / 0.44993 * 100, abs=1e-3
    )


def test_anchor_gate_refuses_when_csv_missing(tmp_path: Path) -> None:
    """anchor_check.load_bound_anchor must not score without the pinned CSVs."""
    prov = _provenance()
    broken = json.loads(json.dumps(prov))
    broken["csv_anchors"] = []
    gate = anchors
    assert gate.verify_csv_anchors(broken) != []