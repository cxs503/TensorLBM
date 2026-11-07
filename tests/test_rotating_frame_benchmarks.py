"""Acceptance gates for the rotating-frame verification anchors.

The three anchors (Ekman spiral, Taylor column, rotating-tank source-sink) are
*verified* benchmarks: their records under ``benchmarks/verified/rotating_*/``
are the evidence, and this test makes that evidence machine-checkable.

Rules enforced here:

* every record whose own ``verdict`` is ``True`` must satisfy the published
  thresholds (no self-declared pass that the numbers do not support);
* each benchmark must contribute at least one passing record;
* the physics pre-conditions of the benchmark are re-checked from the record
  itself (Rossby number, Ekman number, resolved Ekman thickness, convergence);
* the Taylor-column non-rotating control must *not* show a stagnant column, so
  that the rotational anchor cannot be passing for the wrong reason;
* records superseded by a later attempt live in ``records/attempt1/`` and are
  ignored here (they are kept for the audit trail, honest FAIL included).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

RECORDS = Path(__file__).resolve().parents[1] / "benchmarks" / "verified"


def _records(case: str) -> list[dict]:
    d = RECORDS / case / "records"
    out = []
    for p in sorted(d.glob("*.json")):
        rec = json.loads(p.read_text())
        rec["_path"] = str(p)
        out.append(rec)
    return out


def _passing(case: str) -> list[dict]:
    recs = _records(case)
    assert recs, f"no records in {case}"
    ok = [r for r in recs if r.get("verdict") is True]
    assert ok, f"{case}: no record with verdict == True"
    return ok


# --------------------------------------------------------------------------- #
# Ekman layer                                                                 #
# --------------------------------------------------------------------------- #
def test_ekman_record_meets_thresholds():
    for rec in _passing("rotating_ekman"):
        chk = rec["checks"]
        assert chk["delta_err"] <= 0.05, (rec["_path"], chk["delta_err"])
        assert chk["profile_max_err"] <= 0.03, (rec["_path"], chk["profile_max_err"])
        # the analytic Ekman thickness must actually be resolved by the grid
        cfg = rec["config"]
        assert cfg["delta_nodes"] >= 8.0
        assert cfg["Reynolds_delta"] < 10.0
        # five probe heights, as required by the campaign criterion
        assert len(rec["probes"]) >= 5
        # the flow must be steady and essentially incompressible in the layer
        assert rec["convergence"]["converged"] is True
        assert rec["checks"]["density_variation"] < 1e-3
        # every probe error individually inside the 3 % band
        for probe in rec["probes"]:
            assert probe["err_max"] <= 0.03, (rec["_path"], probe)


def test_ekman_fit_recovers_declared_thickness():
    """``delta_E`` and ``U_g`` must be recovered, not assumed."""
    for rec in _passing("rotating_ekman"):
        fit = rec["fit"]
        assert fit["delta_ana"] == pytest.approx(rec["config"]["delta_nodes"], rel=1e-9)
        assert fit["U_g_fit"] == pytest.approx(rec["config"]["U_g"], rel=0.05)


# --------------------------------------------------------------------------- #
# Taylor column                                                               #
# --------------------------------------------------------------------------- #
def test_taylor_column_record_meets_thresholds():
    for rec in _passing("rotating_taylor_column"):
        cfg, col, chk = rec["config"], rec["column"], rec["checks"]
        assert cfg["rotate"] is True
        # Ro = U/(2 Omega a) < 0.1 and E_L = nu/(Omega a^2) << 1 are the
        # pre-conditions of the Proudman-Taylor theorem
        assert cfg["Ro"] < 0.1, cfg["Ro"]
        e_l = cfg["nu_lattice"] / (cfg["Omega_lattice"] * cfg["radius"] ** 2)
        assert e_l < 0.05, e_l
        # stagnation inside the column
        assert col["w_max_rel"] <= 0.05, (rec["_path"], col["w_max_rel"])
        # the column must reach the far cap: extent error < 10 %
        assert col["length_rel_err"] <= 0.10, (rec["_path"], col["length_rel_err"])
        assert col["length_available"] == pytest.approx(
            cfg["shape"][0] - 2 - cfg["z_body"][1]
        )
        # columnarity: horizontal velocity in the interior column is z-uniform
        assert col["columnarity_dev"] <= 0.10, col["columnarity_dev"]
        assert chk["free_stream_speed_rel"] > 0.3  # the drive actually drives
        assert rec["convergence"]["converged"] is True


def test_taylor_column_negative_control_is_not_stagnant():
    """Without rotation the same geometry must *not* show a stagnant column."""
    ctrl = [r for r in _records("rotating_taylor_column") if r["config"]["rotate"] is False]
    if not ctrl:
        pytest.skip("no non-rotating control record present")
    for rec in ctrl:
        assert rec["column"]["w_max_rel"] > 0.05, rec["_path"]
        assert rec["verdict"] is False


# --------------------------------------------------------------------------- #
# Rotating-tank source-sink                                                   #
# --------------------------------------------------------------------------- #
def test_source_sink_record_meets_thresholds():
    for rec in _passing("rotating_source_sink"):
        rp, chk, cfg = rec["radial_profile"], rec["checks"], rec["config"]
        # v_r ~ r^-1 within 0.05 in the exponent
        assert abs(rp["exponent"] + 1.0) <= 0.05, (rec["_path"], rp["exponent"])
        assert rp["r2"] > 0.99
        # Taylor-Proudman interior: v_r independent of z
        assert chk["axial_uniformity_max"] <= 0.05, (rec["_path"], chk["axial_uniformity_max"])
        # the measured slab must be clear of the cap Ekman layers
        assert chk["cap_exclude_cells"] >= 3.0 * math.sqrt(
            cfg["nu_lattice"] / cfg["Omega_lattice"]
        ) - 0.5
        # mass injected must be removed
        assert abs(chk["mass_in"] - chk["mass_out"]) <= 1e-3 * max(chk["mass_in"], 1.0)
        # rotation genuinely dominates: swirl stays a small correction
        assert chk["max_relative_swirl"] < 0.5
        assert rec["convergence"]["converged"] is True


def test_rotating_records_carry_provenance():
    for case in ("rotating_ekman", "rotating_taylor_column", "rotating_source_sink"):
        for rec in _records(case):
            prov = rec["provenance"]
            assert prov["git_sha"] and len(prov["git_sha"]) >= 7
            assert prov["torch"] and prov["python"]
            assert prov["timestamp"]
