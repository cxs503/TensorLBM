"""Guards for the Schäfer–Turek 2D-1 cylinder case (CASE-CYLINDER-DRAG-RE20).

Three things are locked here:

1. the *reference anchor* — the Cd used for validation must be the pinned
   in-repo value (read from ``benchmarks/verified/cylinder/re20_st/result.json``,
   sha256-pinned in ``refs.json``), never a number typed from memory;
2. the *case geometry / dimensionless set* — a silent change of D, blockage or
   Re would invalidate every historical comparison;
3. the *pipeline actually runs and reports* — a coarse run must produce finite
   forces, a normalised Cd, a plateau diagnostic and the pass/fail table, and
   the case-pool entry must write its artifacts.

The quantitative validation itself (D=20/40, Cd within 5 %) is a compute job,
not a unit test; these tests only forbid the failure modes that made previous
deliveries unverifiable (dead code, drifting anchors, entries that cannot run).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import tensorlbm
from tensorlbm.cylinder_st2d1 import (
    GEOMETRY,
    REFERENCE,
    case_geometry,
    compare_to_reference,
    run_cylinder_st2d1,
)

REPO = Path(__file__).resolve().parents[1]
ANCHOR_FILE = REPO / "benchmarks/verified/cylinder/re20_st/result.json"
POOL = REPO / "benchmarks/case_pool/cylinder_drag_re20"


def test_geometry_matches_schafer_turek_setup():
    """D=40 must discretise 2.2 x 0.41 with D=0.1 at (0.2, 0.2) exactly."""
    geo = case_geometry(40)
    assert (geo["nx"], geo["ny"]) == (880, 164)
    assert (geo["cx"], geo["cy"]) == (80, 80)
    assert geo["radius"] == pytest.approx(20.0)
    assert geo["dx"] == pytest.approx(0.1 / 40)
    assert geo["blockage"] == pytest.approx(0.1 / 0.41)
    assert GEOMETRY["re"] == pytest.approx(GEOMETRY["u_mean"] * GEOMETRY["diameter"] / GEOMETRY["nu"])
    with pytest.raises(ValueError):
        case_geometry(3)


def test_reference_anchor_is_pinned_not_transcribed():
    """refs.json, the module constant and the in-repo anchor must all agree."""
    refs = json.loads((POOL / "refs.json").read_text())
    assert refs["anchor"] == "CASE-CYLINDER-DRAG-RE20"
    assert refs["ref"]["cd"] == REFERENCE["cd"]
    assert refs["ref"]["cl"] == REFERENCE["cl"]
    assert refs["tolerance_pct"] == REFERENCE["tolerance_pct"] == 5.0
    # sha256 pin: an edit to the anchor file must be re-acknowledged here
    digest = hashlib.sha256(ANCHOR_FILE.read_bytes()).hexdigest()
    assert digest == refs["provenance"]["anchor_file_sha256"]
    anchor = json.loads(ANCHOR_FILE.read_text())
    assert anchor["ref_cd"] == REFERENCE["cd"]
    assert anchor["ref_cl"] == REFERENCE["cl"]
    assert "Schäfer" in refs["case"] or "Schafer" in refs["case"]
    # provenance honesty: the print original is declared unreachable
    assert "NOT reachable offline" in refs["provenance"]["print_original"]


def test_short_run_is_finite_and_normalised():
    """A coarse run must produce finite, normalised forces and full diagnostics."""
    res = run_cylinder_st2d1(
        D_cells=8,
        steps=200,
        u_lb=0.05,
        avg_window=50,
        resid_interval=100,
        progress=False,
        keep_fields=True,
    )
    assert res["n_steps"] == 200
    assert res["lattice_units"]["re"] == pytest.approx(20.0)
    assert res["lattice_units"]["u_mean"] == pytest.approx(0.05)
    assert res["lattice_units"]["tau"] == pytest.approx(0.5 + 3 * 0.05 * 8 / 20)
    assert res["lattice_units"]["re"] == pytest.approx(20.0)
    assert res["geometry"]["nx"] == 176 and res["geometry"]["ny"] == 33
    assert res["n_solid_cells"] > 0 and res["n_surface_cells"] > 0
    assert res["cd"] == res["cd"] and abs(res["cd"]) > 0.1  # finite, physical order
    assert res["err_cd_pct"] == (res["cd"] - REFERENCE["cd"]) / REFERENCE["cd"] * 100
    assert isinstance(res["stationary"], bool)
    assert len(res["series"]["cd"]) == len(res["series"]["step"]) == 200 // 10
    fields = res["fields"]
    assert fields["rho"].shape == (33, 176)
    assert fields["solid"].sum() == res["n_solid_cells"]
    assert fields["vorticity"].shape == (33, 176)


def test_mach_guard_rejects_coarse_grid_with_fixed_tau():
    """tau fixed at 0.8 makes u_mean = Re*nu/D grow as 1/D: D=8 is Ma=0.43."""
    with pytest.raises(ValueError, match="lattice Mach"):
        run_cylinder_st2d1(D_cells=8, steps=5, progress=False, keep_fields=False)


def test_compare_to_reference_semantics():
    """The gate must reject drifting windows and out-of-tolerance drag."""
    good = {"case": "x", "geometry": {"D_cells": 40}, "cd": REFERENCE["cd"], "cl": 0.0, "stationary": True}
    v = compare_to_reference(good, 5.0)
    assert v["passed"] and v["rows"][0]["pass"] and v["rows"][1]["gating"] is False

    drifting = dict(good, stationary=False)
    assert compare_to_reference(drifting, 5.0)["passed"] is False

    off = dict(good, cd=REFERENCE["cd"] * 1.06)
    v = compare_to_reference(off, 5.0)
    assert v["passed"] is False and v["rows"][0]["pass"] is False
    assert v["rows"][0]["err_pct"] == pytest.approx(6.0)


def test_case_pool_entry_runs_and_writes_artifacts(tmp_path):
    """The entry must execute end-to-end and emit the documented artifact set."""
    out = tmp_path / "run"
    proc = subprocess.run(
        [
            sys.executable,
            str(POOL / "run_case.py"),
            "--grids",
            "8",
            "--u-lb",
            "0.05",
            "--steps",
            "120",
            "--avg-window",
            "40",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(REPO),
    )
    assert proc.returncode == 1, proc.stderr[-3000:]  # not converged → honest FAIL, not a crash
    summary = json.loads((out / "summary.json").read_text())
    assert summary["anchor"] == "CASE-CYLINDER-DRAG-RE20"
    assert summary["tolerance_pct"] == 5.0
    assert summary["passed"] is False
    row = summary["rows"][0]
    assert row["D_cells"] == 8 and row["stationary"] is False
    assert (out / "results_D8.json").exists()
    assert (out / "series_D8.csv").read_text().startswith("step,cd,cl")
    assert (out / "fields_D8.npz").exists()
    assert summary["repro_cmd"].startswith("PYTHONPATH=src python benchmarks/case_pool/cylinder_drag_re20")


def test_public_api_surface_exposes_case():
    """The library package must re-export the case (platform/main-solver entry)."""
    assert tensorlbm.run_cylinder_st2d1 is run_cylinder_st2d1
    assert tensorlbm.CYLINDER_ST2D1_REFERENCE is REFERENCE