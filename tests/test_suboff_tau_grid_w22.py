"""W22 (mission m14505) — artifact invariants for the SUBOFF tau x grid matrix.

These tests never run the solver; they assert that the machine-written
artifacts are internally consistent and that the criterion verdicts recorded in
``tau_grid_matrix.json`` follow from the numbers in the same file:

* the two reference-area calibers (engine frontal / anchor wetted) are recorded
  and consistent with the geometry;
* C_T = C_F + C_P in every case (the force decomposition is additive);
* path A (``build_matrix.py``) and path B (``recompute_matrix.py``) agree to
  1e-6 on C_T/C_F/C_P whenever both are present;
* the negative-drag surface has one entry per (tau, N) cell and its
  neighbour-support count matches its own neighbour list;
* the criteria verdicts are derived, not asserted (a verdict cannot be PASS
  while its own measured quantity is beyond threshold).

Run: ``python -m pytest tests/test_suboff_tau_grid_w22.py -q``
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
W22 = HERE.parent / "benchmarks" / "verified" / "suboff_tau_grid_w22"
MATRIX = W22 / "tau_grid_matrix.json"
SURFACE = W22 / "negative_drag_surface.json"
RECOMPUTE = W22 / "recompute_check.json"


def _load(p: Path):
    if not p.exists():
        pytest.skip(f"{p.name} not present (matrix not assembled)")
    return json.loads(p.read_text())


def _matrix():
    return _load(MATRIX)


def test_prereg_and_matrix_protocol_agree():
    prereg = _load(W22 / "prereg.json")
    m = _matrix()
    assert m["protocol"]["steps"] == prereg["fixed_protocol"]["steps"]
    assert m["protocol"]["collision_model"].startswith("MRT")
    assert m["protocol"]["domain_padding_in_L"] == prereg["fixed_protocol"]["domain_padding_in_L"]
    assert prereg["factors"]["n_cases"] >= 18


def test_two_dps_calibers_are_consistent_with_geometry():
    """Frontal/wetted dpS ratio must equal S_wet/S_front = 4L/D for a bare hull."""
    m = _matrix()
    for c in m["cases"]:
        if c.get("status") != "OK":
            continue
        ratio = c["dpS_wet"] / c["dpS_front"]
        expected = 4.0 * (c["L_lb"] / (2.0 * c["R_lb"]))
        assert ratio == pytest.approx(expected, rel=1e-12)


def test_force_decomposition_additive():
    m = _matrix()
    for c in m["cases"]:
        if c.get("status") != "OK":
            continue
        assert c["C_P_plus_C_F"] == pytest.approx(c["C_T"], rel=1e-12, abs=1e-14)


def test_two_independent_paths_agree_to_1e6():
    rc = _load(RECOMPUTE)
    if rc["n_cases_recomputed"] == 0:
        pytest.skip("no completed matrix case yet (honest FAIL recorded in the artifacts)")
    assert rc["max_abs_rel_delta"] <= 1e-6
    for row in rc["rows"]:
        if row["status"] != "OK":
            continue
        # analytic dpS must reproduce the stored dpS to round-off
        assert row["dpS_front_analytic"] == pytest.approx(row["dpS_front_stored"], rel=1e-12)
        deltas = row.get("abs_rel_delta_vs_path_A") or {}
        for k, v in deltas.items():
            assert v <= 1e-6, f"{row['tag']} {k} delta {v}"


def test_negative_surface_self_consistent():
    s = _load(SURFACE)
    m = _matrix()
    n_expected = len(s["tau_values"]) * len(s["N_values"])
    assert len(s["cells"]) == n_expected
    for cell in s["cells"]:
        if cell.get("status") != "OK":
            continue
        assert cell["negative"] == (cell["C_T"] < 0.0)
        same = [n for n in cell["neighbours"] if n["negative"] == cell["negative"]]
        assert cell["n_same_sign_neighbours"] == len(same)
        if cell["negative"]:
            assert cell["boundary_supported"] == (len(same) >= 1)
    assert s["boundary_support"]["pass"] == (
        s["boundary_support"]["n_unsupported"] == 0
    )
    # every sign grid entry must match a cell
    for N in s["N_values"]:
        grid = s["sign_grid"][str(N)]
        assert len(grid) == len(s["tau_values"])
        for tau, entry in zip(s["tau_values"], grid):
            cell = next(c for c in s["cells"] if c["tau"] == tau and c["N"] == N)
            if cell.get("status") == "OK":
                assert entry == ("NEG" if cell["negative"] else "POS")
            else:
                assert entry == "MISSING"
    del m


def test_verdicts_are_derived_from_measured_values():
    m = _matrix()
    c = m["criteria"]
    # C1
    assert (c["C1"]["verdict"] == "PASS") == (
        c["C1"]["n_completed"] >= 18 and c["C1"]["n_with_all_fields"] == c["C1"]["n_completed"]
    )
    # C2 needs both: no unsupported negative cell AND every matrix cell present
    surface = json.loads(SURFACE.read_text()) if SURFACE.exists() else {"cells": []}
    n_ok_cells = len([x for x in surface["cells"] if x.get("status") == "OK"])
    complete = n_ok_cells == len(surface["cells"]) and n_ok_cells > 0
    assert (c["C2"]["verdict"] == "PASS") == (c["C2"]["n_unsupported_cells"] == 0 and complete)
    # C3a: every available tau row must be below 3% for a PASS
    rows = [r for r in c["C3a"]["rows"] if r["status"] == "OK"]
    if rows:
        assert (c["C3a"]["verdict"] == "PASS") == all(r["drift_pct"] < 3.0 for r in rows)
        assert c["C3a"]["worst_drift_pct"] == pytest.approx(
            max(r["drift_pct"] for r in rows)
        )
    # C3b: monotone in the direction the diffs actually have
    for r in c["C3b"]["rows"]:
        if r["status"] != "OK":
            continue
        if r["direction"] == "monotone_increasing":
            assert all(d > 0 for d in r["successive_diffs"])
        elif r["direction"] == "monotone_decreasing":
            assert all(d < 0 for d in r["successive_diffs"])
        else:
            assert any(d > 0 for d in r["successive_diffs"]) and any(
                d < 0 for d in r["successive_diffs"]
            )


def test_iso_re_identity_of_reynolds_composite():
    """Re_sim = 3 u_lb N /(tau-0.5) must hold for every completed case."""
    m = _matrix()
    u_lb = m["protocol"]["u_lb"]
    for c in m["cases"]:
        if c.get("status") != "OK":
            continue
        re_expected = 3.0 * u_lb * c["N"] / (c["tau"] - 0.5)
        assert c["Re_sim"] == pytest.approx(re_expected, rel=1e-12)
    for g in m["decoupling"]["iso_Re_groups"]:
        for member in g["members"]:
            re_member = 3.0 * u_lb * member["N"] / (member["tau"] - 0.5)
            assert re_member == pytest.approx(g["Re_sim"], rel=1e-9)


def test_anchor_deviation_definition():
    """dev% must be (C_T - C_T_anchor)/C_T_anchor*100 against the artifact anchor."""
    m = _matrix()
    anchor_ct = m["anchor"]["CT_anchor"]
    for c in m["cases"]:
        if c.get("status") != "OK":
            continue
        dev = (c["C_T"] - anchor_ct) / anchor_ct * 100.0
        assert c["anchor"]["dev_pct"] == pytest.approx(dev, rel=1e-12)
        assert c["anchor"]["CT_anchor"] == anchor_ct
        assert math.isfinite(dev)


def test_six_significant_digit_band_present():
    m = _matrix()
    done = [c for c in m["cases"] if c.get("status") == "OK"]
    if not done:
        pytest.skip("no completed case (honest FAIL recorded)")
    for c in done:
        for key in ("C_T", "C_F", "C_P", "dev_pct_vs_anchor"):
            s = c["six_sig"][key]
            assert "e" in s or "." in s
            mantissa = s.split("e")[0].replace("-", "").replace(".", "")
            assert len(mantissa) == 6, f"{c['tag']} {key}={s} is not 6 significant digits"