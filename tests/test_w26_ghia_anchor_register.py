"""W26 anchor-register acceptance tests.

Checks the deliverable papers/artifacts/w26/d3q19-bgk-analytic-fourcase/anchor_register.json:

  * every criterion in the register is PASS and the numbers behind the verdicts are
    consistent with the stored evidence;
  * the four analytic cases, recomputed on the fly, stay at or below 1e-12 while the
    negative controls stay far above it (a residual test that cannot fail is not a test);
  * the registered Ghia tables satisfy the incompressible boundary conditions and the
    1/128 line-grid lattice, and -- when the source PDF is available -- re-extracting
    from the PDF reproduces the registered numbers bit for bit.

The PDF is copyrighted and is not part of the repository, so the re-extraction test
skips unless GHIA_PDF points at a local copy whose sha256 matches the register.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(REPO, "papers", "artifacts", "w26", "d3q19-bgk-analytic-fourcase",
                   "anchor_register.json")
SCRIPT_DIR = os.path.join(REPO, "scripts", "w26_ghia")
sys.path.insert(0, SCRIPT_DIR)


@pytest.fixture(scope="module")
def register():
    if not os.path.exists(ART):
        pytest.skip("anchor register not built")
    with open(ART, encoding="utf-8") as fh:
        return json.load(fh)


def _profiles(register, table, re_key):
    rows = register["tables"][table]["rows"]
    coord = np.array([r["coord"] for r in rows], float)
    val = np.array([r["profile"][re_key]["value"] for r in rows], float)
    return coord, val


def test_all_criteria_pass(register):
    verdicts = {c["id"]: c["verdict"] for c in register["criteria"]}
    assert verdicts == {"C1": "PASS", "C2": "PASS", "C3": "PASS", "C4": "PASS"}, verdicts


def test_three_re_tables_registered(register):
    assert set(register["tables"]) == {"I", "II"}
    for table in ("I", "II"):
        rows = register["tables"][table]["rows"]
        assert len(rows) == 17
        for r in rows:
            for re_key in ("100.0", "400.0", "1000.0"):
                assert r["profile"][re_key]["value"] is not None
    assert register["anchor_availability"]["status"] == "AVAILABLE"


def test_every_value_has_both_witnesses(register):
    for table in ("I", "II"):
        for r in register["tables"][table]["rows"]:
            for re_key in ("100.0", "400.0", "1000.0"):
                v = r["profile"][re_key]
                assert v["ocr_token"], "missing verbatim OCR token"
                assert v["sign_from_ink"] in (-1.0, 1.0), "sign must come from the ink"
                assert v["ink_width_pt"] is not None


def test_coordinates_on_128_lattice(register):
    for table in ("I", "II"):
        coord, _ = _profiles(register, table, "100.0")
        n = np.round(coord * 128)
        assert np.max(np.abs(coord - n / 128.0)) <= 5e-5 + 1e-12


def test_incompressible_boundary_conditions(register):
    for re_key in ("100.0", "400.0", "1000.0"):
        y, u = _profiles(register, "I", re_key)
        x, v = _profiles(register, "II", re_key)
        assert abs(u[np.argmax(y)] - 1.0) < 1e-9, "lid must move with u/U_lid = 1"
        assert abs(u[np.argmin(y)]) < 1e-9, "bottom wall at rest"
        assert abs(v[0]) < 1e-9 and abs(v[-1]) < 1e-9, "side walls: v = 0"


def test_analytic_residuals_and_controls():
    import analytic_fourcase as AF

    c1 = AF.case1_poiseuille()
    c2 = AF.case2_couette()
    c3 = AF.case3_taylor_green()
    assert c1["residual_continuum"]["residual_rel"] <= 1e-12
    assert c1["residual_discrete_laplacian"]["residual_abs"] <= 1e-12
    assert c1["control_wrong_amplitude_1pct"]["residual_rel"] > 1e-2
    assert abs(c1["convergence"]["fitted_slope_p_asymptotic"] - 2.0) <= 1e-2
    assert c2["residual_continuum"]["residual_rel"] <= 1e-12
    assert c2["control_parabolic_profile"]["residual_rel"] > 1e-1
    assert c2["wall_shear"]["rel_error"] <= 1e-12
    for key in ("divergence", "momentum_x", "momentum_y"):
        assert c3[key]["residual_rel"] <= 1e-12
    assert c3["control_pressure_3pct"]["momentum_x_rel"] > 1e-2


def test_analytic_case4_cavity_limit():
    import analytic_fourcase as AF

    c4 = AF.case4_cavity_limits(dps=30)
    assert c4["worst_residual"] <= max(1e-12, 10 * c4["fd_noise_floor"])
    assert c4["control_wrong_exponent"]["max_abs_high_precision"] > 1.0


def test_secondary_transcription_defect_registered(register):
    d = register["in_repo_secondary_table_defect"]
    assert d["verdict"] == "DEFECT_REGISTERED"
    assert max(d["worst_abs_difference_by_profile"].values()) > 0.2


def _pdf_path(register):
    path = os.environ.get("GHIA_PDF", "/tmp/ghia1982.pdf")
    if not os.path.exists(path):
        pytest.skip("source PDF not available (copyrighted, not in the repo)")
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != register["anchor_availability"]["pdf"]["sha256"]:
        pytest.skip("PDF sha256 differs from the registered one")
    return path


def test_regex_extraction_reproduces_register(register):
    path = _pdf_path(register)
    import extract_ghia_tables as EX

    cells = EX.extract(path)
    for table in ("I", "II"):
        mine = cells["tables"][table]["rows"]
        theirs = register["tables"][table]["rows"]
        assert len(mine) == len(theirs) == 17
        for a, b in zip(mine, theirs):
            assert abs(a["coord"] - b["coord"]) < 1e-12
            for re_key in ("100.0", "400.0", "1000.0"):
                assert a["values"][re_key]["value"] == b["profile"][re_key]["value"]